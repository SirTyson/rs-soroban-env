//! Explicit-token-id helpers used by performance-sensitive callers (e.g. the
//! native Soroswap pair fast path) that need to perform a SAC `transfer`
//! between two contract addresses without paying for an entire nested
//! `Frame::StellarAssetContract` (frame push/pop, auth-frame snapshot,
//! diagnostics, dispatch overhead). These helpers reproduce the storage,
//! event, and TTL side effects of the generic SAC `transfer` path under an
//! explicit token `ContractId`, but only for the exact shape the caller
//! validates up-front. Any uncertain shape returns `None` so the caller can
//! fall back to the regular nested-call path.

use crate::{
    builtin_contracts::stellar_asset_contract::{
        balance::{
            extend_contract_balance_ttl_pub as extend_contract_balance_ttl,
            read_contract_balance_entry_for_token, write_contract_balance_entry_for_token,
            BalanceFetched,
        },
        storage_types::{INSTANCE_EXTEND_AMOUNT, INSTANCE_TTL_THRESHOLD},
    },
    host::metered_clone::MeteredClone,
    xdr::{ContractId, ScErrorCode, ScErrorType, ScString, ScVal},
    AddressObject, EnvBase, Host, HostError, Symbol, TryFromVal, TryIntoVal,
    VecObject,
};

/// Outcome of attempting the fused direct-SAC-transfer-and-balance fast path.
pub(crate) enum DirectTransferOutcome {
    /// Fused path applied; returned value is the `from` contract's post-transfer
    /// balance for this token. Caller should treat this as the authoritative
    /// post-transfer balance, equivalent to what a subsequent SAC `balance`
    /// call on `from` would return.
    Applied(i128),
    /// The caller-provided shape is not supported by the fused path. Caller
    /// must fall back to the regular nested SAC `transfer` + `balance` calls.
    Fallback,
}

/// Attempts the fused contract→contract SAC `transfer` followed by reading the
/// updated `from` balance, all without pushing a `Frame::StellarAssetContract`.
///
/// Preconditions enforced internally; any failed precondition returns
/// `Fallback` so the caller invokes the regular nested SAC path instead:
///
/// - `amount > 0` (zero / negative amounts go through the standard path so
///   their semantics — no-op or `NegativeAmountError` — are produced by the
///   generic SAC `transfer`).
/// - The token contract instance must exist and have
///   `ContractExecutable::StellarAsset`.
/// - Both `from` and `to` addresses must be `ScAddress::Contract` and distinct
///   (so the SAC event is always a plain `transfer`, never a self-transfer or
///   mint/burn — a contract address can never be an asset issuer).
/// - Both balance ledger entries must already exist and be `authorized`.
/// - `from` balance must be `>= amount`.
///
/// On success, the fused path reproduces the observable side effects of the
/// generic SAC `transfer`: extends the SAC instance and code TTLs, mutates
/// both `Balance(from)` and `Balance(to)` persistent contract data entries,
/// extends their TTLs, and emits the SAC `transfer` event under the explicit
/// token contract id with topics `[Symbol("transfer"), from, to, name]` and
/// `i128` amount data.
///
/// Auth is intentionally skipped: this helper is only used by callers that
/// have already established `from == current_contract`, in which case
/// `from.require_auth()` would succeed via the direct-invoker rule without
/// consuming any tracker entry (and the AuthorizationManager state is
/// unchanged).
pub(crate) fn try_direct_contract_to_contract_transfer(
    e: &Host,
    token_id: &ContractId,
    from_id: &ContractId,
    to_id: &ContractId,
    from_addr: AddressObject,
    to_addr: AddressObject,
    amount: i128,
) -> Result<DirectTransferOutcome, HostError> {
    if amount <= 0 {
        return Ok(DirectTransferOutcome::Fallback);
    }
    if from_id == to_id {
        return Ok(DirectTransferOutcome::Fallback);
    }

    // Load the token's instance ledger entry. Extract the asset name (used as
    // the 4th topic of the SAC transfer event) and confirm the executable is
    // StellarAsset. If anything looks unusual (missing METADATA, malformed
    // name, non-SAC executable), bail to fallback.
    let instance_key = e.contract_instance_ledger_key(token_id)?;
    let Some(name) = e.peek_stellar_asset_metadata_name_from_instance(&instance_key)? else {
        return Ok(DirectTransferOutcome::Fallback);
    };

    // Read both balance entries. Both must exist and be authorized; the from
    // entry must have sufficient amount. Otherwise bail.
    let Some(from_fetched) = read_contract_balance_entry_for_token(e, token_id, from_id)? else {
        return Ok(DirectTransferOutcome::Fallback);
    };
    if !from_fetched.balance.authorized || from_fetched.balance.amount < amount {
        return Ok(DirectTransferOutcome::Fallback);
    }
    let Some(to_fetched) = read_contract_balance_entry_for_token(e, token_id, to_id)? else {
        return Ok(DirectTransferOutcome::Fallback);
    };
    if !to_fetched.balance.authorized {
        return Ok(DirectTransferOutcome::Fallback);
    }

    // All preconditions met: from here onward we mirror SAC `transfer`'s
    // side effects in order.

    // 1. Extend SAC instance + code TTL (matches
    //    `extend_current_contract_instance_and_code_ttl` in SAC `transfer`).
    e.extend_contract_instance_ttl_from_contract_id(
        instance_key.clone(),
        INSTANCE_TTL_THRESHOLD,
        INSTANCE_EXTEND_AMOUNT,
    )?;
    e.extend_contract_code_ttl_from_contract_id(
        instance_key,
        INSTANCE_TTL_THRESHOLD,
        INSTANCE_EXTEND_AMOUNT,
    )?;

    // 2. Mutate balances and write back (matches `spend_balance` +
    //    `receive_balance`).
    let BalanceFetched {
        balance: mut from_balance,
        key: from_key,
        live_until_ledger: from_live_until,
    } = from_fetched;
    let BalanceFetched {
        balance: mut to_balance,
        key: to_key,
        live_until_ledger: to_live_until,
    } = to_fetched;

    let new_from_amount = from_balance.amount.checked_sub(amount).ok_or_else(|| {
        e.err(
            ScErrorType::Value,
            ScErrorCode::ArithDomain,
            "SAC direct-transfer underflow on from balance",
            &[],
        )
    })?;
    let new_to_amount = to_balance.amount.checked_add(amount).ok_or_else(|| {
        e.err(
            ScErrorType::Value,
            ScErrorCode::ArithDomain,
            "SAC direct-transfer overflow on to balance",
            &[],
        )
    })?;
    from_balance.amount = new_from_amount;
    to_balance.amount = new_to_amount;

    write_contract_balance_entry_for_token(
        e,
        token_id,
        from_id,
        &from_key,
        from_live_until,
        &from_balance,
    )?;
    write_contract_balance_entry_for_token(
        e,
        token_id,
        to_id,
        &to_key,
        to_live_until,
        &to_balance,
    )?;

    // 3. Extend both balance TTLs (matches the end-of-write_contract_balance
    //    extend_contract_balance_ttl call).
    extend_contract_balance_ttl(e, from_key)?;
    extend_contract_balance_ttl(e, to_key)?;

    // 4. Emit transfer event under the explicit token contract id.
    emit_sac_transfer_event_for_token(e, token_id, from_addr, to_addr, &name, amount)?;

    Ok(DirectTransferOutcome::Applied(new_from_amount))
}

fn emit_sac_transfer_event_for_token(
    e: &Host,
    token_id: &ContractId,
    from_addr: AddressObject,
    to_addr: AddressObject,
    name: &ScString,
    amount: i128,
) -> Result<(), HostError> {
    // Build topics: [Symbol("transfer"), from, to, name].
    let transfer_sym = Symbol::try_from_val(e, &"transfer")?;
    let name_val = e.to_host_val(&ScVal::String(name.metered_clone(e)?))?;
    let topics: VecObject = e.vec_new_from_slice(&[
        transfer_sym.to_val(),
        from_addr.to_val(),
        to_addr.to_val(),
        name_val,
    ])?;
    let data = amount.try_into_val(e)?;
    e.record_contract_event_for_contract_id(token_id, topics, data)?;
    Ok(())
}
