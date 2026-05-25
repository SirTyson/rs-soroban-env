use crate::{
    auth::AuthorizationManagerSnapshot,
    builtin_contracts::stellar_asset_contract::{
        read_contract_balance_for_contract_owner, try_direct_contract_to_contract_transfer,
        DirectTransferOutcome, INSTANCE_EXTEND_AMOUNT, INSTANCE_TTL_THRESHOLD,
    },
    budget::AsBudget,
    err,
    host::{
        metered_clone::{MeteredClone, MeteredContainer},
        prng::Prng,
    },
    storage::{InstanceStorageMap, StorageMap},
    xdr::{
        int128_helpers, ContractExecutable, ContractId, ContractIdPreimage, CreateContractArgsV2,
        Hash, HostFunction, HostFunctionType, Int128Parts, ScAddress, ScContractInstance,
        ScErrorCode, ScErrorType, ScMap, ScMapEntry, ScVal,
    },
    AddressObject, EnvBase, Error, ErrorHandler, Host, HostError, Object, StorageType, Symbol,
    SymbolStr, TryFromVal, TryIntoVal, Val, Vm, DEFAULT_HOST_DEPTH_LIMIT,
};

#[cfg(any(test, feature = "testutils"))]
use core::cell::RefCell;
use std::rc::Rc;

/// Determines the re-entry mode for calling a contract.
pub(crate) enum ContractReentryMode {
    /// Re-entry is completely prohibited.
    Prohibited,
    /// Re-entry is allowed, but only directly into the same contract (i.e. it's
    /// possible for a contract to do a self-call via host).
    SelfAllowed,
    /// Re-entry is fully allowed.
    #[allow(dead_code)]
    Allowed,
}

/// All the contract functions starting with double underscore are considered
/// to be reserved by the Soroban host and can't be directly called by another
/// contracts.
const RESERVED_CONTRACT_FN_PREFIX: &str = "__";
const SOROSWAP_POOL_WASM_HASH: [u8; 32] = [
    0x18, 0x05, 0x14, 0x56, 0x81, 0x6b, 0x66, 0xf1, 0x2e, 0x77, 0x3a, 0x56, 0xf7, 0x7c, 0x57,
    0x94, 0xfa, 0xc1, 0xb1, 0xfb, 0x7a, 0xb6, 0xe2, 0x2d, 0x4f, 0xad, 0x5a, 0x41, 0x27, 0x70,
    0xf7, 0x3e,
];
const SOROSWAP_POOL_TTL_THRESHOLD: u32 = 501_120;
const SOROSWAP_POOL_TTL_EXTEND_TO: u32 = 518_400;

#[derive(Clone, Copy)]
enum SoroswapPoolGetter {
    Token0,
    Token1,
    Factory,
    GetReserves,
    KLast,
}

// Soroswap pair pool error codes, matching the SoroswapPairError enum embedded
// in the vendored apply-load pool Wasm (see contract spec custom section).
const SOROSWAP_ERR_SWAP_INSUFFICIENT_OUTPUT_AMOUNT: u32 = 108;
const SOROSWAP_ERR_SWAP_NEGATIVES_OUT_NOT_SUPPORTED: u32 = 109;
const SOROSWAP_ERR_SWAP_INSUFFICIENT_LIQUIDITY: u32 = 110;
const SOROSWAP_ERR_SWAP_INVALID_TO: u32 = 111;
const SOROSWAP_ERR_SWAP_INSUFFICIENT_INPUT_AMOUNT: u32 = 112;
const SOROSWAP_ERR_SWAP_NEGATIVES_IN_NOT_SUPPORTED: u32 = 113;
const SOROSWAP_ERR_SWAP_K_CONSTANT_NOT_MET: u32 = 114;

/// Saves host state (storage and objects) for rolling back a (sub-)transaction
/// on error. A helper type used by [`FrameGuard`].
// Notes on metering: `RollbackPoint` are metered under Frame operations
// #[derive(Clone)]
pub(super) struct RollbackPoint {
    storage: StorageMap,
    events: usize,
    auth: AuthorizationManagerSnapshot,
}

#[cfg(any(test, feature = "testutils"))]
pub trait ContractFunctionSet {
    fn call(&self, func: &Symbol, host: &Host, args: &[Val]) -> Option<Val>;
}

#[cfg(any(test, feature = "testutils"))]
#[derive(Debug, Clone)]
pub(crate) struct TestContractFrame {
    pub(crate) id: ContractId,
    pub(crate) func: Symbol,
    pub(crate) args: Vec<Val>,
    pub(crate) panic: Rc<RefCell<Option<Error>>>,
    pub(crate) instance: ScContractInstance,
}

#[cfg(any(test, feature = "testutils"))]
impl std::hash::Hash for TestContractFrame {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.func.hash(state);
        self.args.hash(state);
        if let Some(panic) = self.panic.borrow().as_ref() {
            panic.hash(state);
        }
        self.instance.hash(state);
    }
}

#[cfg(any(test, feature = "testutils"))]
impl TestContractFrame {
    pub fn new(id: ContractId, func: Symbol, args: Vec<Val>, instance: ScContractInstance) -> Self {
        Self {
            id,
            func,
            args,
            panic: Rc::new(RefCell::new(None)),
            instance,
        }
    }
}

/// Context pairs a variable-case [`Frame`] enum with state that's common to all
/// cases (eg. a [`Prng`]).
#[derive(Clone, Hash)]
pub(crate) struct Context {
    pub(crate) frame: Frame,
    pub(crate) prng: Option<Prng>,
    pub(crate) storage: Option<InstanceStorageMap>,
}

pub(crate) struct CallParams {
    pub(crate) reentry_mode: ContractReentryMode,
    pub(crate) internal_host_call: bool,
    pub(crate) treat_missing_function_as_noop: bool,
}

impl CallParams {
    pub(crate) fn default_external_call() -> Self {
        Self {
            reentry_mode: ContractReentryMode::Prohibited,
            internal_host_call: false,
            treat_missing_function_as_noop: false,
        }
    }

    #[allow(unused)]
    pub(crate) fn default_internal_call() -> Self {
        Self {
            reentry_mode: ContractReentryMode::Prohibited,
            internal_host_call: true,
            treat_missing_function_as_noop: false,
        }
    }
}

/// Holds contextual information about a single invocation, either
/// a reference to a contract [`Vm`] or an enclosing [`HostFunction`]
/// invocation.
///
/// Frames are arranged into a stack in [`HostImpl::context`], and are pushed
/// with [`Host::push_frame`], which returns a [`FrameGuard`] that will
/// pop the frame on scope-exit.
///
/// Frames are also the units of (sub-)transactions: each frame captures
/// the host state when it is pushed, and the [`FrameGuard`] will either
/// commit or roll back that state when it pops the stack.
#[derive(Clone, Hash)]
pub(crate) enum Frame {
    ContractVM {
        vm: Rc<Vm>,
        fn_name: Symbol,
        args: Vec<Val>,
        instance: ScContractInstance,
        relative_objects: Vec<Object>,
    },
    HostFunction(HostFunctionType),
    StellarAssetContract(ContractId, Symbol, Vec<Val>, ScContractInstance),
    #[cfg(any(test, feature = "testutils"))]
    TestContract(TestContractFrame),
    NativeContract(ContractId, Symbol, Vec<Val>, ScContractInstance),
}

impl Frame {
    fn contract_id(&self) -> Option<&ContractId> {
        match self {
            Frame::ContractVM { vm, .. } => Some(&vm.contract_id),
            Frame::NativeContract(id, ..) => Some(id),
            Frame::HostFunction(_) => None,
            Frame::StellarAssetContract(id, ..) => Some(id),
            #[cfg(any(test, feature = "testutils"))]
            Frame::TestContract(tc) => Some(&tc.id),
        }
    }

    fn instance(&self) -> Option<&ScContractInstance> {
        match self {
            Frame::ContractVM { instance, .. } => Some(instance),
            Frame::NativeContract(_, _, _, instance) => Some(instance),
            Frame::HostFunction(_) => None,
            Frame::StellarAssetContract(_, _, _, instance) => Some(instance),
            #[cfg(any(test, feature = "testutils"))]
            Frame::TestContract(tc) => Some(&tc.instance),
        }
    }
    #[cfg(any(test, feature = "testutils"))]
    fn is_contract_vm(&self) -> bool {
        matches!(self, Frame::ContractVM { .. })
    }
}

impl Host {
    /// Returns if the host currently has a frame on the stack.
    ///
    /// A frame being on the stack usually indicates that a contract is currently
    /// executing, or is in a state just-before or just-after executing.
    pub fn has_frame(&self) -> Result<bool, HostError> {
        self.with_current_frame_opt(|opt| Ok(opt.is_some()))
    }

    /// Helper function for [`Host::with_frame`] below. Pushes a new [`Context`]
    /// on the context stack, returning a [`RollbackPoint`] such that if
    /// operation fails, it can be used to roll the [`Host`] back to the state
    /// it had before its associated [`Context`] was pushed.
    pub(super) fn push_context(&self, ctx: Context) -> Result<RollbackPoint, HostError> {
        let _span = tracy_span!("push context");
        let auth_manager = self.try_borrow_authorization_manager()?;
        let auth_snapshot = auth_manager.push_frame(self, &ctx.frame)?;
        // Establish the rp first, since this might run out of gas and fail.
        let rp = RollbackPoint {
            storage: self.try_borrow_storage()?.map.metered_clone(self)?,
            events: self.try_borrow_events()?.vec.len(),
            auth: auth_snapshot,
        };
        // Charge for the push, which might also run out of gas.
        Vec::<Context>::charge_bulk_init_cpy(1, self.as_budget())?;
        // Finally commit to doing the push.
        self.try_borrow_context_stack_mut()?.push(ctx);
        Ok(rp)
    }

    /// Helper function for [`Host::with_frame`] below. Pops a [`Context`] off
    /// the current context stack and optionally rolls back the [`Host`]'s objects
    /// and storage map to the state in the provided [`RollbackPoint`].
    pub(super) fn pop_context(&self, orp: Option<RollbackPoint>) -> Result<Context, HostError> {
        let _span = tracy_span!("pop context");

        let ctx = self.try_borrow_context_stack_mut()?.pop();

        #[cfg(any(test, feature = "recording_mode"))]
        if self.try_borrow_context_stack()?.is_empty() {
            // When there are no contexts left, emulate authentication for the
            // recording auth mode. This is a no-op for the enforcing mode.
            self.try_borrow_authorization_manager()?
                .maybe_emulate_authentication(self)?;
        }
        let mut auth_snapshot = None;
        if let Some(rp) = orp {
            self.try_borrow_storage_mut()?.map = rp.storage;
            self.try_borrow_events_mut()?.rollback(rp.events)?;
            auth_snapshot = Some(rp.auth);
        }
        self.try_borrow_authorization_manager()?
            .pop_frame(self, auth_snapshot)?;
        ctx.ok_or_else(|| {
            self.err(
                ScErrorType::Context,
                ScErrorCode::InternalError,
                "unmatched host context push/pop",
                &[],
            )
        })
    }

    /// Applies a function to the top [`Frame`] of the context stack. Returns
    /// [`HostError`] if the context stack is empty, otherwise returns result of
    /// function call.
    //
    // Notes on metering: aquiring the current frame is cheap and not charged.
    // Metering happens in the passed-in closure where actual work is being done.
    pub(super) fn with_current_frame<F, U>(&self, f: F) -> Result<U, HostError>
    where
        F: FnOnce(&Frame) -> Result<U, HostError>,
    {
        let Ok(context_guard) = self.0.context_stack.try_borrow() else {
            return Err(self.err(
                ScErrorType::Context,
                ScErrorCode::InternalError,
                "context is already borrowed",
                &[],
            ));
        };

        if let Some(context) = context_guard.last() {
            f(&context.frame)
        } else {
            drop(context_guard);
            Err(self.err(
                ScErrorType::Context,
                ScErrorCode::InternalError,
                "no contract running",
                &[],
            ))
        }
    }

    /// Applies a function to a mutable reference to the top [`Context`] of the
    /// context stack. Returns [`HostError`] if the context stack is empty,
    /// otherwise returns result of function call.
    //
    // Notes on metering: aquiring the current frame is cheap and not charged.
    // Metering happens in the passed-in closure where actual work is being done.
    pub(super) fn with_current_context_mut<F, U>(&self, f: F) -> Result<U, HostError>
    where
        F: FnOnce(&mut Context) -> Result<U, HostError>,
    {
        let Ok(mut context_guard) = self.0.context_stack.try_borrow_mut() else {
            return Err(self.err(
                ScErrorType::Context,
                ScErrorCode::InternalError,
                "context is already borrowed",
                &[],
            ));
        };
        if let Some(context) = context_guard.last_mut() {
            f(context)
        } else {
            drop(context_guard);
            Err(self.err(
                ScErrorType::Context,
                ScErrorCode::InternalError,
                "no contract running",
                &[],
            ))
        }
    }

    /// Same as [`Self::with_current_frame`] but passes `None` when there is no current
    /// frame, rather than failing with an error.
    pub(crate) fn with_current_frame_opt<F, U>(&self, f: F) -> Result<U, HostError>
    where
        F: FnOnce(Option<&Frame>) -> Result<U, HostError>,
    {
        let Ok(context_guard) = self.0.context_stack.try_borrow() else {
            return Err(self.err(
                ScErrorType::Context,
                ScErrorCode::InternalError,
                "context is already borrowed",
                &[],
            ));
        };
        if let Some(context) = context_guard.last() {
            f(Some(&context.frame))
        } else {
            drop(context_guard);
            f(None)
        }
    }

    pub(crate) fn with_current_frame_relative_object_table<F, U>(
        &self,
        f: F,
    ) -> Result<U, HostError>
    where
        F: FnOnce(&mut Vec<Object>) -> Result<U, HostError>,
    {
        self.with_current_context_mut(|ctx| {
            if let Frame::ContractVM {
                relative_objects, ..
            } = &mut ctx.frame
            {
                f(relative_objects)
            } else {
                Err(self.err(
                    ScErrorType::Context,
                    ScErrorCode::InternalError,
                    "accessing relative object table in non-VM frame",
                    &[],
                ))
            }
        })
    }

    pub(crate) fn with_current_prng<F, U>(&self, f: F) -> Result<U, HostError>
    where
        F: FnOnce(&mut Prng) -> Result<U, HostError>,
    {
        // We mem::take the context's PRNG into a local variable and then put it
        // back when we're done. This allows the callback to borrow the context
        // to report errors if anything goes wrong in it. If the callback also
        // installs a PRNG of its own (it shouldn't!) we notice when putting the
        // context's PRNG back and fail with an internal error.
        let curr_prng_opt =
            self.with_current_context_mut(|ctx| Ok(std::mem::take(&mut ctx.prng)))?;

        let mut curr_prng = match curr_prng_opt {
            // There's already a context PRNG, so use it.
            Some(prng) => prng,

            // There's no context PRNG yet, seed one from the base PRNG (unless
            // the base PRNG itself hasn't been seeded).
            None => {
                let mut base_guard = self.try_borrow_base_prng_mut()?;
                if let Some(base) = base_guard.as_mut() {
                    base.sub_prng(self.as_budget())?
                } else {
                    return Err(self.err(
                        ScErrorType::Context,
                        ScErrorCode::InternalError,
                        "host base PRNG was not seeded",
                        &[],
                    ));
                }
            }
        };

        // Call the callback with the new-or-existing context PRNG.
        let res: Result<U, HostError> = f(&mut curr_prng);

        // Put the (possibly newly-initialized frame PRNG-option back)
        self.with_current_context_mut(|ctx| {
            if ctx.prng.is_some() {
                return Err(self.err(
                    ScErrorType::Context,
                    ScErrorCode::InternalError,
                    "callback re-entered with_current_prng",
                    &[],
                ));
            }
            ctx.prng = Some(curr_prng);
            Ok(())
        })?;
        res
    }

    /// Pushes a [`Frame`], runs a closure, and then pops the frame, rolling back
    /// if the closure returned an error. Returns the result that the closure
    /// returned (or any error caused during the frame push/pop).
    pub(crate) fn with_frame<F>(&self, frame: Frame, f: F) -> Result<Val, HostError>
    where
        F: FnOnce() -> Result<Val, HostError>,
    {
        let start_depth = self.try_borrow_context_stack()?.len();
        if start_depth as u32 >= DEFAULT_HOST_DEPTH_LIMIT {
            return Err(Error::from_type_and_code(
                ScErrorType::Context,
                ScErrorCode::ExceededLimit,
            )
            .into());
        }
        #[cfg(any(test, feature = "testutils"))]
        {
            if let Some(ctx) = self.try_borrow_context_stack()?.last() {
                if frame.is_contract_vm() && ctx.frame.is_contract_vm() {
                    if let Ok(mut scoreboard) = self.try_borrow_coverage_scoreboard_mut() {
                        scoreboard.vm_to_vm_calls += 1;
                    }
                }
            }
        }
        let ctx = Context {
            frame,
            prng: None,
            storage: None,
        };
        let rp = self.push_context(ctx)?;
        {
            // We do this _after_ the context is pushed, in order to let the
            // observation code assume a context exists
            if let Some(ctx) = self.try_borrow_context_stack()?.last() {
                self.call_any_lifecycle_hook(crate::host::TraceEvent::PushCtx(ctx))?;
            }
        }
        #[cfg(any(test, feature = "testutils"))]
        let mut is_top_contract_invocation = false;
        #[cfg(any(test, feature = "testutils"))]
        {
            if self.try_borrow_context_stack()?.len() == 1 {
                if let Some(ctx) = self.try_borrow_context_stack()?.first() {
                    match ctx.frame {
                        // Don't call the contract invocation hook for
                        // the host functions.
                        Frame::HostFunction(_) => (),
                        // Everything else is some sort of contract call.
                        _ => {
                            is_top_contract_invocation = true;
                            if let Some(contract_invocation_hook) =
                                self.try_borrow_top_contract_invocation_hook()?.as_ref()
                            {
                                contract_invocation_hook(
                                    self,
                                    crate::host::ContractInvocationEvent::Start,
                                );
                            }
                        }
                    }
                }
            }
        }

        let res = f();
        let mut res = if let Ok(v) = res {
            // If a contract function happens to have signature Result<...,
            // Code> its Wasm ABI encoding will be ambiguous: if it exits with
            // Err(Code) it'll wind up exiting the Wasm VM "successfully" with a
            // Val that's of type Error, we'll get Ok(Error) here. To allow this
            // to work and avoid losing errors, we define _any_ successful
            // return of Ok(Error) as "a contract failure"; contracts aren't
            // allowed to return Ok(Error) and have it considered actually-ok.
            //
            // (If we were called from try_call, it will actually turn this
            // Err(ScErrorType::Contract) back into Ok(ScErrorType::Contract)
            // since that is a "recoverable" type of error.)
            if let Ok(err) = Error::try_from(v) {
                // Unfortunately there are still two sub-cases to consider. One
                // is when a contract returns Ok(Error) with
                // ScErrorType::Contract, which is allowed and legitimate and
                // "how a contract would signal a Result::Err(Code) as described
                // above". In this (good) case we propagate the Error they
                // provided, just switching it from Ok(Error) to Err(Error)
                // indicating that the contract "failed" with this Error.
                //
                // The second (bad) case is when the contract returns Ok(Error)
                // with a non-ScErrorType::Contract. This might be some kind of
                // mistake on their part but it might also be an attempt at
                // spoofing error reporting, by claiming some subsystem of the
                // host failed when it really didn't. In particular if a
                // contract wants to forcibly fail a caller that did `try_call`,
                // the contract could spoof-return an unrecoverable Error code
                // like InternalError or BudgetExceeded. We want to deny all
                // such cases, so we just define them as illegal returns, and
                // report them all as a specific error type of and description
                // our own choosing: not a contract's own logic failing, but a
                // contract failing to live up to a postcondition we're
                // enforcing of "never returning this sort of error code".
                if err.is_type(ScErrorType::Contract) {
                    Err(self.error(
                        err,
                        "escalating Ok(ScErrorType::Contract) frame-exit to Err",
                        &[],
                    ))
                } else {
                    Err(self.err(
                        ScErrorType::Context,
                        ScErrorCode::InvalidAction,
                        "frame-exit with Ok(Error) carrying a non-ScErrorType::Contract Error",
                        &[err.to_val()],
                    ))
                }
            } else {
                Ok(v)
            }
        } else {
            res
        };

        // We try flushing instance storage at the end of the frame if nothing
        // else failed. Unfortunately flushing instance storage is _itself_
        // fallible in a variety of ways, and if it fails we want to roll back
        // everything else.
        if res.is_ok() {
            let instance_storage_persisted = self.persist_instance_storage();
            match instance_storage_persisted {
                Ok(persisted) => {
                    // If we did persist instance storage, we may need to reload
                    // it into the re-entrant parent frames.
                    if persisted {
                        // Similarly to above, if reloading instance storage, if
                        // this fails we need to roll back everything.
                        if let Err(e) = self.maybe_reload_instance_storage_on_frame_pop() {
                            res = Err(e);
                        }
                    }
                }
                Err(e) => {
                    res = Err(e);
                }
            }
        }
        {
            // We do this _before_ the context is popped, in order to let the
            // observation code assume a context exists
            if let Some(ctx) = self.try_borrow_context_stack()?.last() {
                let res = match &res {
                    Ok(v) => Ok(*v),
                    Err(ref e) => Err(e),
                };
                self.call_any_lifecycle_hook(crate::host::TraceEvent::PopCtx(&ctx, &res))?;
            }
        }
        if res.is_err() {
            // Pop and rollback on error.
            self.pop_context(Some(rp))?
        } else {
            // Just pop on success.
            self.pop_context(None)?
        };
        // Every push and pop should be matched; if not there is a bug.
        let end_depth = self.try_borrow_context_stack()?.len();
        if start_depth != end_depth {
            return Err(err!(
                self,
                (ScErrorType::Context, ScErrorCode::InternalError),
                "frame-depth mismatch",
                start_depth,
                end_depth
            ));
        }
        #[cfg(any(test, feature = "testutils"))]
        if end_depth == 0 {
            // Empty call stack in tests means that some contract function call
            // has been finished and hence the authorization manager can be reset.
            // In non-test scenarios, there should be no need to ever reset
            // the authorization manager as the host instance shouldn't be
            // shared between the contract invocations.
            *self.try_borrow_previous_authorization_manager_mut()? =
                Some(self.try_borrow_authorization_manager()?.clone());
            self.try_borrow_authorization_manager_mut()?.reset();

            // Call the contract invocation hook for contract invocations only.
            if is_top_contract_invocation {
                if let Some(top_contract_invocation_hook) =
                    self.try_borrow_top_contract_invocation_hook()?.as_ref()
                {
                    top_contract_invocation_hook(
                        self,
                        crate::host::ContractInvocationEvent::Finish,
                    );
                }
            }
        }
        res
    }

    /// Inspects the frame at the top of the context and returns the contract ID
    /// if it exists. Returns `Ok(None)` if the context stack is empty or has a
    /// non-contract frame on top.
    pub(crate) fn get_current_contract_id_opt_internal(
        &self,
    ) -> Result<Option<ContractId>, HostError> {
        self.with_current_frame_opt(|opt_frame| match opt_frame {
            Some(frame) => frame
                .contract_id()
                .map(|id| id.metered_clone(self))
                .transpose(),
            None => Ok(None),
        })
    }

    /// Returns [`Hash`] contract ID from the VM frame at the top of the context
    /// stack, or a [`HostError`] if the context stack is empty or has a non-VM
    /// frame at its top.
    pub(crate) fn get_current_contract_id_internal(&self) -> Result<ContractId, HostError> {
        if let Some(id) = self.get_current_contract_id_opt_internal()? {
            Ok(id)
        } else {
            // This should only ever happen if we try to access the contract ID
            // from a HostFunction frame (meaning before a contract is running).
            // Doing so is a logic bug on our part. If we simply run out of
            // budget while cloning the Hash we won't get here, the `?` above
            // will propagate the budget error.
            Err(self.err(
                ScErrorType::Context,
                ScErrorCode::InternalError,
                "Current context has no contract ID",
                &[],
            ))
        }
    }

    /// Pushes a test contract [`Frame`], runs a closure, and then pops the
    /// frame, rolling back if the closure returned an error. Returns the result
    /// that the closure returned (or any error caused during the frame
    /// push/pop). Used for testing.
    #[cfg(any(test, feature = "testutils"))]
    pub fn with_test_contract_frame<F>(
        &self,
        id: ContractId,
        func: Symbol,
        f: F,
    ) -> Result<Val, HostError>
    where
        F: FnOnce() -> Result<Val, HostError>,
    {
        let _invocation_meter_scope = self.maybe_meter_invocation(
            crate::host::invocation_metering::MeteringInvocation::contract_invocation(
                self, &id, func,
            ),
        );
        self.with_frame(
            Frame::TestContract(self.create_test_contract_frame(id, func, vec![])?),
            f,
        )
    }

    /// Pushes a test contract [`Frame`], runs a closure, and then pops the
    /// frame, rolling back if the closure returned an error. Returns the result
    /// that the closure returned (or any error that occurred during the closure
    /// or the frame push/pop). Used for testing.
    #[cfg(any(test, feature = "testutils"))]
    pub fn try_with_test_contract_frame<F>(
        &self,
        id: ContractId,
        func: Symbol,
        f: F,
    ) -> Result<Val, HostError>
    where
        F: FnOnce() -> Result<Val, HostError>,
    {
        let _invocation_meter_scope = self.maybe_meter_invocation(
            crate::host::invocation_metering::MeteringInvocation::contract_invocation(
                self, &id, func,
            ),
        );

        // Code taken from `call_n_internal` to handle panics inside the closure `f`.
        // Modified to run as a closure within a test contract frame instead of invoking
        // a contract function.
        let frame = self.create_test_contract_frame(id.clone(), func, vec![])?;
        let panic = frame.panic.clone();
        self.with_frame(Frame::TestContract(frame), || {
            use std::any::Any;
            use std::panic::AssertUnwindSafe;
            type PanicVal = Box<dyn Any + Send>;

            let closure = AssertUnwindSafe(move || f());
            let res: Result<Result<Val, HostError>, PanicVal> =
                crate::testutils::call_with_suppressed_panic_hook(closure);
            match res {
                Ok(res) => res,
                Err(panic_payload) => {
                    let mut error: Error =
                        Error::from(wasmi::core::TrapCode::UnreachableCodeReached);

                    let mut recovered_error_from_panic_refcell = false;
                    if let Ok(panic) = panic.try_borrow() {
                        if let Some(err) = *panic {
                            recovered_error_from_panic_refcell = true;
                            error = err;
                        }
                    }

                    if !recovered_error_from_panic_refcell {
                        self.with_debug_mode(|| {
                            // only include func in log if a non-empty name is provided
                            let func_str = match format!("{:?}", func).as_str() {
                                "Symbol()" => String::new(),
                                formatted => format!(" with fn name '{}'", formatted),
                            };
                            if let Some(str) = panic_payload.downcast_ref::<&str>() {
                                let msg: String = format!(
                                    "caught panic '{}' from test contract frame{}",
                                    str, func_str
                                );
                                let _ = self.log_diagnostics(&msg, &[]);
                            } else if let Some(str) = panic_payload.downcast_ref::<String>() {
                                let msg: String = format!(
                                    "caught panic '{}' from test contract frame{}",
                                    str, func_str
                                );
                                let _ = self.log_diagnostics(&msg, &[]);
                            };
                            Ok(())
                        })
                    }
                    Err(self.error(error, "caught error from test contract frame", &[]))
                }
            }
        })
    }

    #[cfg(any(test, feature = "testutils"))]
    fn create_test_contract_frame(
        &self,
        id: ContractId,
        func: Symbol,
        args: Vec<Val>,
    ) -> Result<TestContractFrame, HostError> {
        let instance_key = self.contract_instance_ledger_key(&id)?;
        let instance = self.retrieve_contract_instance_from_storage(&instance_key)?;
        Ok(TestContractFrame::new(id, func, args.to_vec(), instance))
    }

    // Notes on metering: this is covered by the called components.
    fn call_contract_fn(
        &self,
        id: &ContractId,
        func: &Symbol,
        args: &[Val],
        treat_missing_function_as_noop: bool,
    ) -> Result<Val, HostError> {
        // Create key for storage
        let storage_key = self.contract_instance_ledger_key(id)?;
        let instance = self.retrieve_contract_instance_from_storage(&storage_key)?;
        Vec::<Val>::charge_bulk_init_cpy(args.len() as u64, self.as_budget())?;
        let args_vec = args.to_vec();
        match &instance.executable {
            ContractExecutable::Wasm(wasm_hash) => {
                if let Some(getter) =
                    self.match_native_soroswap_pool_getter(func, args, &instance, wasm_hash)?
                {
                    let frame =
                        Frame::NativeContract(id.metered_clone(self)?, *func, args_vec, instance);
                    return self.with_frame(frame, || {
                        self.call_native_soroswap_pool_getter(getter)
                    });
                }
                if let Some((amount_0_out, amount_1_out, to_addr)) =
                    self.match_native_soroswap_pool_swap(func, args, &instance, wasm_hash)?
                {
                    let frame =
                        Frame::NativeContract(id.metered_clone(self)?, *func, args_vec, instance);
                    return self.with_frame(frame, || {
                        self.call_native_soroswap_pool_swap(amount_0_out, amount_1_out, to_addr)
                    });
                }
                let vm = self.instantiate_vm(id, wasm_hash)?;
                let relative_objects = Vec::new();
                self.with_frame(
                    Frame::ContractVM {
                        vm: Rc::clone(&vm),
                        fn_name: *func,
                        args: args_vec,
                        instance,
                        relative_objects,
                    },
                    || vm.invoke_function_raw(self, func, args, treat_missing_function_as_noop),
                )
            }
            ContractExecutable::StellarAsset => self.with_frame(
                Frame::StellarAssetContract(id.metered_clone(self)?, *func, args_vec, instance),
                || {
                    use crate::builtin_contracts::{BuiltinContract, StellarAssetContract};
                    StellarAssetContract.call(func, self, args)
                },
            ),
        }
    }

    fn match_native_soroswap_pool_getter(
        &self,
        func: &Symbol,
        args: &[Val],
        instance: &ScContractInstance,
        wasm_hash: &Hash,
    ) -> Result<Option<SoroswapPoolGetter>, HostError> {
        // Next-protocol gate: this native emulation bypasses Wasm instantiation
        // and changes protocol-visible budget/dispatch accounting, so it must
        // not run for the released protocol version. Keep p26 execution exact.
        if self.get_ledger_protocol_version()? <= crate::host::MIN_LEDGER_PROTOCOL_VERSION {
            return Ok(None);
        }
        if wasm_hash.0.as_slice() != SOROSWAP_POOL_WASM_HASH || !args.is_empty() {
            return Ok(None);
        }
        let Some(getter) = self.soroswap_pool_getter_for_symbol(*func)? else {
            return Ok(None);
        };
        if !Self::soroswap_pool_instance_matches_getter(instance, getter) {
            return Ok(None);
        }
        Ok(Some(getter))
    }

    fn soroswap_pool_getter_for_symbol(
        &self,
        func: Symbol,
    ) -> Result<Option<SoroswapPoolGetter>, HostError> {
        if self.symbol_matches(b"token_0", func)? {
            Ok(Some(SoroswapPoolGetter::Token0))
        } else if self.symbol_matches(b"token_1", func)? {
            Ok(Some(SoroswapPoolGetter::Token1))
        } else if self.symbol_matches(b"factory", func)? {
            Ok(Some(SoroswapPoolGetter::Factory))
        } else if self.symbol_matches(b"get_reserves", func)? {
            Ok(Some(SoroswapPoolGetter::GetReserves))
        } else if self.symbol_matches(b"k_last", func)? {
            Ok(Some(SoroswapPoolGetter::KLast))
        } else {
            Ok(None)
        }
    }

    fn soroswap_pool_instance_matches_getter(
        instance: &ScContractInstance,
        getter: SoroswapPoolGetter,
    ) -> bool {
        let Some(storage) = instance.storage.as_ref() else {
            return false;
        };
        match getter {
            SoroswapPoolGetter::Token0 => Self::soroswap_pool_scmap_has_address(storage, 0),
            SoroswapPoolGetter::Token1 => Self::soroswap_pool_scmap_has_address(storage, 1),
            SoroswapPoolGetter::Factory => Self::soroswap_pool_scmap_has_address(storage, 4),
            SoroswapPoolGetter::GetReserves => {
                Self::soroswap_pool_scmap_has_i128(storage, 2)
                    && Self::soroswap_pool_scmap_has_i128(storage, 3)
            }
            SoroswapPoolGetter::KLast => Self::soroswap_pool_scmap_get(storage, 5)
                .is_none_or(|v| matches!(v, ScVal::I128(_))),
        }
    }

    fn soroswap_pool_scmap_get(storage: &ScMap, key: u32) -> Option<&ScVal> {
        storage
            .iter()
            .find(|entry| matches!(entry.key, ScVal::U32(k) if k == key))
            .map(|entry| &entry.val)
    }

    fn soroswap_pool_scmap_has_address(storage: &ScMap, key: u32) -> bool {
        matches!(
            Self::soroswap_pool_scmap_get(storage, key),
            Some(ScVal::Address(ScAddress::Account(_) | ScAddress::Contract(_)))
        )
    }

    fn soroswap_pool_scmap_has_i128(storage: &ScMap, key: u32) -> bool {
        matches!(Self::soroswap_pool_scmap_get(storage, key), Some(ScVal::I128(_)))
    }

    fn call_native_soroswap_pool_getter(
        &self,
        getter: SoroswapPoolGetter,
    ) -> Result<Val, HostError> {
        let contract_id = self.get_current_contract_id_internal()?;
        let instance_key = self.contract_instance_ledger_key(&contract_id)?;
        self.extend_contract_instance_ttl_from_contract_id(
            instance_key.clone(),
            SOROSWAP_POOL_TTL_THRESHOLD,
            SOROSWAP_POOL_TTL_EXTEND_TO,
        )?;
        self.extend_contract_code_ttl_from_contract_id(
            instance_key,
            SOROSWAP_POOL_TTL_THRESHOLD,
            SOROSWAP_POOL_TTL_EXTEND_TO,
        )?;

        match getter {
            SoroswapPoolGetter::Token0 => self.soroswap_pool_get_address(0),
            SoroswapPoolGetter::Token1 => self.soroswap_pool_get_address(1),
            SoroswapPoolGetter::Factory => self.soroswap_pool_get_address(4),
            SoroswapPoolGetter::GetReserves => {
                let reserve_0 = self.soroswap_pool_get_i128_val(2)?;
                let reserve_1 = self.soroswap_pool_get_i128_val(3)?;
                Ok(self
                    .vec_new_from_slice(&[reserve_0, reserve_1])?
                    .to_val())
            }
            SoroswapPoolGetter::KLast => {
                if let Some(k_last) = self.soroswap_pool_get_optional_i128_val(5)? {
                    Ok(k_last)
                } else {
                    Ok(Val::try_from_val(self, &0_i128)?)
                }
            }
        }
    }

    fn soroswap_pool_instance_storage_get(&self, key: u32) -> Result<Option<Val>, HostError> {
        if let Some(val) = self.soroswap_pool_native_instance_storage_get(key)? {
            return Ok(val);
        }
        let key = Val::from_u32(key).to_val();
        self.with_instance_storage(|s| Ok(s.map.get(&key, self)?.copied()))
    }

    fn soroswap_pool_native_instance_storage_get(
        &self,
        key: u32,
    ) -> Result<Option<Option<Val>>, HostError> {
        self.with_current_context_mut(|ctx| {
            let Frame::NativeContract(_, _, _, instance) = &ctx.frame else {
                return Ok(None);
            };
            let Some(storage) = instance.storage.as_ref() else {
                return Ok(Some(None));
            };
            Self::soroswap_pool_scmap_get(storage, key).map_or(Ok(Some(None)), |v| {
                self.to_valid_host_val(v).map(|v| Some(Some(v)))
            })
        })
    }

    fn soroswap_pool_get_address(&self, key: u32) -> Result<Val, HostError> {
        if let Some(address) = self.soroswap_pool_native_address(key)? {
            return Ok(address.to_val());
        }
        let val = self.soroswap_pool_get_required_val(key)?;
        AddressObject::try_from(val).map_err(|_| {
            self.err(
                ScErrorType::Object,
                ScErrorCode::UnexpectedType,
                "unexpected Soroswap pool address storage value",
                &[Val::from_u32(key).to_val(), val],
            )
        })?;
        Ok(val)
    }

    fn soroswap_pool_native_address(&self, key: u32) -> Result<Option<AddressObject>, HostError> {
        match self.soroswap_pool_native_scaddress(key)? {
            Some(addr) => Ok(Some(self.add_host_object(addr)?)),
            None => Ok(None),
        }
    }

    fn soroswap_pool_native_scaddress(&self, key: u32) -> Result<Option<ScAddress>, HostError> {
        self.with_current_context_mut(|ctx| {
            let Frame::NativeContract(_, _, _, instance) = &ctx.frame else {
                return Ok(None);
            };
            let Some(storage) = instance.storage.as_ref() else {
                return Ok(None);
            };
            match Self::soroswap_pool_scmap_get(storage, key) {
                Some(ScVal::Address(addr)) => match addr {
                    ScAddress::Account(_) | ScAddress::Contract(_) => {
                        Ok(Some(addr.metered_clone(self)?))
                    }
                    _ => Ok(None),
                },
                _ => Ok(None),
            }
        })
    }

    fn soroswap_pool_required_native_scaddress(
        &self,
        key: u32,
    ) -> Result<ScAddress, HostError> {
        self.soroswap_pool_native_scaddress(key)?.ok_or_else(|| {
            self.err(
                ScErrorType::Storage,
                ScErrorCode::MissingValue,
                "missing Soroswap pool address instance storage key",
                &[
                    Val::from_u32(key).to_val(),
                    Val::from_u32(StorageType::Instance as u32).to_val(),
                ],
            )
        })
    }

    fn soroswap_pool_get_i128_val(&self, key: u32) -> Result<Val, HostError> {
        if let Some(v) = self.soroswap_pool_native_i128(key)? {
            return self.soroswap_pool_i128_to_val(v);
        }
        let val = self.soroswap_pool_get_required_val(key)?;
        let _: i128 = i128::try_from_val(self, &val)?;
        Ok(val)
    }

    fn soroswap_pool_get_optional_i128_val(&self, key: u32) -> Result<Option<Val>, HostError> {
        if self.soroswap_pool_on_native_frame()? {
            return match self.soroswap_pool_native_i128(key)? {
                Some(v) => Ok(Some(self.soroswap_pool_i128_to_val(v)?)),
                None => Ok(None),
            };
        }
        if let Some(val) = self.soroswap_pool_instance_storage_get(key)? {
            let _: i128 = i128::try_from_val(self, &val)?;
            Ok(Some(val))
        } else {
            Ok(None)
        }
    }

    // Returns `Some(i128)` when called from a native Soroswap pool frame whose
    // raw `ScContractInstance.storage` contains an `ScVal::I128` at `key`,
    // bypassing host-object materialization. Returns `Ok(None)` if the caller
    // is not on a native frame OR if the key is missing / mistyped, leaving the
    // caller to fall back to the generic path or surface a missing-value
    // error as appropriate.
    fn soroswap_pool_native_i128(&self, key: u32) -> Result<Option<i128>, HostError> {
        self.with_current_context_mut(|ctx| {
            let Frame::NativeContract(_, _, _, instance) = &ctx.frame else {
                return Ok(None);
            };
            let Some(storage) = instance.storage.as_ref() else {
                return Ok(None);
            };
            Ok(match Self::soroswap_pool_scmap_get(storage, key) {
                Some(ScVal::I128(parts)) => {
                    Some(int128_helpers::i128_from_pieces(parts.hi, parts.lo))
                }
                _ => None,
            })
        })
    }

    fn soroswap_pool_required_native_i128(&self, key: u32) -> Result<i128, HostError> {
        self.soroswap_pool_native_i128(key)?.ok_or_else(|| {
            self.err(
                ScErrorType::Storage,
                ScErrorCode::MissingValue,
                "missing Soroswap pool i128 instance storage key",
                &[
                    Val::from_u32(key).to_val(),
                    Val::from_u32(StorageType::Instance as u32).to_val(),
                ],
            )
        })
    }

    fn soroswap_pool_on_native_frame(&self) -> Result<bool, HostError> {
        self.with_current_context_mut(|ctx| {
            Ok(matches!(&ctx.frame, Frame::NativeContract(_, _, _, _)))
        })
    }

    fn soroswap_pool_i128_to_val(&self, value: i128) -> Result<Val, HostError> {
        Ok(self.add_host_object(value)?.to_val())
    }

    fn soroswap_pool_get_required_val(&self, key: u32) -> Result<Val, HostError> {
        self.soroswap_pool_instance_storage_get(key)?.ok_or_else(|| {
            self.err(
                ScErrorType::Storage,
                ScErrorCode::MissingValue,
                "missing Soroswap pool instance storage key",
                &[
                    Val::from_u32(key).to_val(),
                    Val::from_u32(StorageType::Instance as u32).to_val(),
                ],
            )
        })
    }

    fn match_native_soroswap_pool_swap(
        &self,
        func: &Symbol,
        args: &[Val],
        instance: &ScContractInstance,
        wasm_hash: &Hash,
    ) -> Result<Option<(i128, i128, AddressObject)>, HostError> {
        // Next-protocol gate: this native emulation bypasses Wasm instantiation
        // and changes protocol-visible budget/dispatch accounting, so it must
        // not run for the released protocol version. Keep p26 execution exact.
        if self.get_ledger_protocol_version()? <= crate::host::MIN_LEDGER_PROTOCOL_VERSION {
            return Ok(None);
        }
        if wasm_hash.0.as_slice() != SOROSWAP_POOL_WASM_HASH || args.len() != 3 {
            return Ok(None);
        }
        if !self.symbol_matches(b"swap", *func)? {
            return Ok(None);
        }
        // Validate arg shape: amount_0_out and amount_1_out must be i128, to must
        // be an AddressObject. We refuse to optimize any other call shape so the
        // Wasm fallback continues to handle exotic inputs.
        let amount_0_out = match i128::try_from_val(self, &args[0]) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        let amount_1_out = match i128::try_from_val(self, &args[1]) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };
        let to_addr = match AddressObject::try_from(args[2]) {
            Ok(o) => o,
            Err(_) => return Ok(None),
        };
        // Require the instance to carry the expected pair layout (token addresses
        // at keys 0/1 and i128 reserves at keys 2/3). If the layout doesn't match
        // we fall back to Wasm so the contract can produce its own NotInitialized
        // (or other) error path unchanged.
        let Some(storage) = instance.storage.as_ref() else {
            return Ok(None);
        };
        if !Self::soroswap_pool_scmap_has_address(storage, 0)
            || !Self::soroswap_pool_scmap_has_address(storage, 1)
            || !Self::soroswap_pool_scmap_has_i128(storage, 2)
            || !Self::soroswap_pool_scmap_has_i128(storage, 3)
        {
            return Ok(None);
        }
        Ok(Some((amount_0_out, amount_1_out, to_addr)))
    }

    fn call_native_soroswap_pool_swap(
        &self,
        amount_0_out: i128,
        amount_1_out: i128,
        to: AddressObject,
    ) -> Result<Val, HostError> {
        let contract_id = self.get_current_contract_id_internal()?;
        let instance_key = self.contract_instance_ledger_key(&contract_id)?;
        self.extend_contract_instance_ttl_from_contract_id(
            instance_key.clone(),
            SOROSWAP_POOL_TTL_THRESHOLD,
            SOROSWAP_POOL_TTL_EXTEND_TO,
        )?;
        self.extend_contract_code_ttl_from_contract_id(
            instance_key,
            SOROSWAP_POOL_TTL_THRESHOLD,
            SOROSWAP_POOL_TTL_EXTEND_TO,
        )?;

        // Output amount validation, matching the wasm order:
        //   both zero       -> SwapInsufficientOutputAmount (108)
        //   either negative -> SwapNegativesOutNotSupported (109)
        //   out >= reserve  -> SwapInsufficientLiquidity (110)
        if amount_0_out == 0 && amount_1_out == 0 {
            return Err(self.soroswap_pool_contract_err(
                SOROSWAP_ERR_SWAP_INSUFFICIENT_OUTPUT_AMOUNT,
            ));
        }
        if amount_0_out < 0 || amount_1_out < 0 {
            return Err(self.soroswap_pool_contract_err(
                SOROSWAP_ERR_SWAP_NEGATIVES_OUT_NOT_SUPPORTED,
            ));
        }

        let reserve_0 = self.soroswap_pool_required_native_i128(2)?;
        let reserve_1 = self.soroswap_pool_required_native_i128(3)?;

        if amount_0_out >= reserve_0 || amount_1_out >= reserve_1 {
            return Err(self.soroswap_pool_contract_err(
                SOROSWAP_ERR_SWAP_INSUFFICIENT_LIQUIDITY,
            ));
        }

        let token_0_addr = self.soroswap_pool_required_native_scaddress(0)?;
        let token_1_addr = self.soroswap_pool_required_native_scaddress(1)?;
        let to_addr = self.scaddress_from_address(to)?;
        if to_addr == token_0_addr || to_addr == token_1_addr {
            return Err(self.soroswap_pool_contract_err(SOROSWAP_ERR_SWAP_INVALID_TO));
        }
        let token_0 = self.add_host_object(token_0_addr)?;
        let token_1 = self.add_host_object(token_1_addr)?;

        let pair_address = self.add_host_object(ScAddress::Contract(
            contract_id.metered_clone(self)?,
        ))?;

        let mut balance_0_opt: Option<i128> = None;
        let mut balance_1_opt: Option<i128> = None;
        if amount_0_out > 0 {
            balance_0_opt = self.soroswap_pool_transfer_and_balance_or_fallback(
                token_0,
                pair_address,
                to,
                amount_0_out,
                &contract_id,
            )?;
        }
        if amount_1_out > 0 {
            balance_1_opt = self.soroswap_pool_transfer_and_balance_or_fallback(
                token_1,
                pair_address,
                to,
                amount_1_out,
                &contract_id,
            )?;
        }

        // For the side that was either skipped (amount == 0) or fell back to
        // the nested SAC transfer subcall, we still need to read the
        // post-transfer balance via the direct SAC-balance helper. The fused
        // helper already returns the authoritative post-transfer balance for
        // sides it handled.
        let balance_0 = match balance_0_opt {
            Some(b) => b,
            None => self.soroswap_pool_invoke_sac_balance(token_0, pair_address)?,
        };
        let balance_1 = match balance_1_opt {
            Some(b) => b,
            None => self.soroswap_pool_invoke_sac_balance(token_1, pair_address)?,
        };

        let amount_0_in = match reserve_0.checked_sub(amount_0_out) {
            Some(r) if balance_0 > r => balance_0.checked_sub(r).ok_or_else(|| {
                self.err(
                    ScErrorType::Value,
                    ScErrorCode::ArithDomain,
                    "soroswap amount_0_in overflow",
                    &[],
                )
            })?,
            Some(_) => 0,
            None => {
                return Err(self.err(
                    ScErrorType::Value,
                    ScErrorCode::ArithDomain,
                    "soroswap reserve_0 - amount_0_out overflow",
                    &[],
                ));
            }
        };
        let amount_1_in = match reserve_1.checked_sub(amount_1_out) {
            Some(r) if balance_1 > r => balance_1.checked_sub(r).ok_or_else(|| {
                self.err(
                    ScErrorType::Value,
                    ScErrorCode::ArithDomain,
                    "soroswap amount_1_in overflow",
                    &[],
                )
            })?,
            Some(_) => 0,
            None => {
                return Err(self.err(
                    ScErrorType::Value,
                    ScErrorCode::ArithDomain,
                    "soroswap reserve_1 - amount_1_out overflow",
                    &[],
                ));
            }
        };

        if amount_0_in == 0 && amount_1_in == 0 {
            return Err(self.soroswap_pool_contract_err(
                SOROSWAP_ERR_SWAP_INSUFFICIENT_INPUT_AMOUNT,
            ));
        }
        if amount_0_in < 0 || amount_1_in < 0 {
            return Err(self.soroswap_pool_contract_err(
                SOROSWAP_ERR_SWAP_NEGATIVES_IN_NOT_SUPPORTED,
            ));
        }

        // K-invariant: (balance_0 * 1000 - amount_0_in * 3) *
        //              (balance_1 * 1000 - amount_1_in * 3) >=
        //              reserve_0 * reserve_1 * 1_000_000
        // Matches the Soroswap mainnet pair wasm fee-adjusted constant product
        // check. We use checked arithmetic and return SwapKConstantNotMet on
        // mismatch (or an internal arithmetic error on overflow).
        let arith_err = || {
            self.err(
                ScErrorType::Value,
                ScErrorCode::ArithDomain,
                "soroswap K-invariant arithmetic overflow",
                &[],
            )
        };
        let bal0_1000 = balance_0.checked_mul(1000).ok_or_else(arith_err)?;
        let bal1_1000 = balance_1.checked_mul(1000).ok_or_else(arith_err)?;
        let fee0 = amount_0_in.checked_mul(3).ok_or_else(arith_err)?;
        let fee1 = amount_1_in.checked_mul(3).ok_or_else(arith_err)?;
        let bal0_adj = bal0_1000.checked_sub(fee0).ok_or_else(arith_err)?;
        let bal1_adj = bal1_1000.checked_sub(fee1).ok_or_else(arith_err)?;
        let lhs = bal0_adj.checked_mul(bal1_adj).ok_or_else(arith_err)?;
        let rk = reserve_0.checked_mul(reserve_1).ok_or_else(arith_err)?;
        let rhs = rk.checked_mul(1_000_000).ok_or_else(arith_err)?;
        if lhs < rhs {
            return Err(self.soroswap_pool_contract_err(
                SOROSWAP_ERR_SWAP_K_CONSTANT_NOT_MET,
            ));
        }

        // Update reserves (instance storage keys 2 and 3) with the new balances.
        self.soroswap_pool_update_reserves(balance_0, balance_1)?;

        // Emit the SwapEvent equivalent. Topics are
        // [Symbol("SoroswapPair"), Symbol("swap")] and data is an ScMap with
        // the SwapEvent fields in alphabetical order (which matches the wasm
        // contracttype layout).
        let amount_0_in_val: Val = amount_0_in.try_into_val(self)?;
        let amount_1_in_val: Val = amount_1_in.try_into_val(self)?;
        let amount_0_out_val: Val = amount_0_out.try_into_val(self)?;
        let amount_1_out_val: Val = amount_1_out.try_into_val(self)?;
        let topic_pair = self.symbol_new_from_slice(b"SoroswapPair")?;
        let topic_swap = self.symbol_new_from_slice(b"swap")?;
        let topics = self
            .vec_new_from_slice(&[topic_pair.to_val(), topic_swap.to_val()])?;
        let keys: [&[u8]; 5] = [
            b"amount_0_in",
            b"amount_0_out",
            b"amount_1_in",
            b"amount_1_out",
            b"to",
        ];
        let vals: [Val; 5] = [
            amount_0_in_val,
            amount_0_out_val,
            amount_1_in_val,
            amount_1_out_val,
            to.to_val(),
        ];
        let data = self.map_new_from_slices(
            &keys.map(|k| {
                // SAFETY: all keys are valid soroban symbol characters and <=32 chars.
                core::str::from_utf8(k).unwrap_or("")
            }),
            &vals,
        )?;
        self.record_contract_event(
            crate::xdr::ContractEventType::Contract,
            topics,
            data.to_val(),
        )?;

        Ok(Val::VOID.to_val())
    }

    fn soroswap_pool_update_reserves(
        &self,
        reserve_0: i128,
        reserve_1: i128,
    ) -> Result<(), HostError> {
        if self.with_current_context_mut(|ctx| {
            let Frame::NativeContract(_, _, _, instance) = &mut ctx.frame else {
                return Ok(false);
            };
            let Some(storage) = instance.storage.as_ref() else {
                return Err(self.err(
                    ScErrorType::Storage,
                    ScErrorCode::MissingValue,
                    "missing Soroswap pool instance storage during native reserve update",
                    &[],
                ));
            };
            instance.storage = Some(self.soroswap_pool_reserves_updated_scmap(
                storage, reserve_0, reserve_1,
            )?);
            Ok(true)
        })? {
            return Ok(());
        }

        let new_reserve_0_val: Val = reserve_0.try_into_val(self)?;
        let new_reserve_1_val: Val = reserve_1.try_into_val(self)?;
        self.with_mut_instance_storage(|s| {
            let k0 = Val::from_u32(2).to_val();
            let k1 = Val::from_u32(3).to_val();
            s.map = s.map.insert(k0, new_reserve_0_val, self)?;
            s.map = s.map.insert(k1, new_reserve_1_val, self)?;
            Ok(())
        })
    }

    fn soroswap_pool_reserves_updated_scmap(
        &self,
        storage: &ScMap,
        reserve_0: i128,
        reserve_1: i128,
    ) -> Result<ScMap, HostError> {
        let mut found_reserve_0 = false;
        let mut found_reserve_1 = false;
        let mut entries = Vec::<ScMapEntry>::with_metered_capacity(storage.len(), self)?;
        for entry in storage.iter() {
            let mut updated = entry.metered_clone(self)?;
            match &entry.key {
                ScVal::U32(2) => {
                    updated.val = Self::soroswap_pool_i128_scval(reserve_0);
                    found_reserve_0 = true;
                }
                ScVal::U32(3) => {
                    updated.val = Self::soroswap_pool_i128_scval(reserve_1);
                    found_reserve_1 = true;
                }
                _ => {}
            }
            entries.push(updated);
        }

        if !found_reserve_0 || !found_reserve_1 {
            return Err(self.err(
                ScErrorType::Storage,
                ScErrorCode::MissingValue,
                "missing Soroswap pool reserve storage key during native reserve update",
                &[],
            ));
        }
        Ok(ScMap(self.map_err(entries.try_into())?))
    }

    fn soroswap_pool_i128_scval(value: i128) -> ScVal {
        ScVal::I128(Int128Parts {
            hi: int128_helpers::i128_hi(value),
            lo: int128_helpers::i128_lo(value),
        })
    }

    fn soroswap_pool_contract_err(&self, code: u32) -> HostError {
        self.error(
            Error::from_contract_error(code),
            "soroswap pool native swap returned contract error",
            &[Val::from_u32(code).to_val()],
        )
    }

    fn soroswap_pool_invoke_sac_transfer(
        &self,
        token: AddressObject,
        from: AddressObject,
        to: AddressObject,
        amount: i128,
    ) -> Result<(), HostError> {
        let amount_val: Val = amount.try_into_val(self)?;
        let token_id = self.contract_id_from_address(token)?;
        let func = self.symbol_new_from_slice(b"transfer")?;
        self.call_n_internal(
            &token_id,
            func.into(),
            &[from.to_val(), to.to_val(), amount_val],
            CallParams::default_external_call(),
        )?;
        Ok(())
    }

    /// Fused output-token transfer + post-transfer balance read for the
    /// native Soroswap pair `swap` fast path. Attempts to perform a
    /// contract→contract SAC `transfer` directly against the explicit token
    /// without pushing a `Frame::StellarAssetContract` (and the associated
    /// auth/dispatch overhead). On success, returns
    /// `Ok(Some(post_transfer_from_balance))` so the caller can skip the
    /// subsequent `balance(from)` read for this token. If any precondition
    /// is unmet (non-SAC token, account or muxed-account recipient, missing
    /// or deauthorized or insufficient balance, malformed METADATA, etc.)
    /// the helper falls back to the regular nested SAC `transfer` subcall
    /// and returns `Ok(None)` so the caller invokes the regular SAC
    /// `balance(from)` read.
    ///
    /// `from_pair_id` must equal the current pair contract id — the caller
    /// passes the already-resolved id rather than re-deriving it from
    /// `from`. The fused path's auth elision is only safe when `from` is the
    /// current contract (direct-invoker rule succeeds without consuming a
    /// tracker entry).
    fn soroswap_pool_transfer_and_balance_or_fallback(
        &self,
        token: AddressObject,
        from: AddressObject,
        to: AddressObject,
        amount: i128,
        from_pair_id: &ContractId,
    ) -> Result<Option<i128>, HostError> {
        if amount > 0 {
            let token_id = self.contract_id_from_address(token)?;
            let to_sc = self.scaddress_from_address(to)?;
            if let ScAddress::Contract(to_id) = to_sc {
                let outcome = try_direct_contract_to_contract_transfer(
                    self,
                    &token_id,
                    from_pair_id,
                    &to_id,
                    from,
                    to,
                    amount,
                )?;
                if let DirectTransferOutcome::Applied(new_from_balance) = outcome {
                    return Ok(Some(new_from_balance));
                }
            }
        }
        self.soroswap_pool_invoke_sac_transfer(token, from, to, amount)?;
        Ok(None)
    }

    fn soroswap_pool_invoke_sac_balance(
        &self,
        token: AddressObject,
        owner: AddressObject,
    ) -> Result<i128, HostError> {
        let token_id = self.contract_id_from_address(token)?;
        if let Some(balance) = self.soroswap_pool_read_sac_contract_balance(&token_id, owner)? {
            return Ok(balance);
        }
        let func = self.symbol_new_from_slice(b"balance")?;
        let res = self.call_n_internal(
            &token_id,
            func.into(),
            &[owner.to_val()],
            CallParams::default_external_call(),
        )?;
        Ok(i128::try_from_val(self, &res)?)
    }

    fn soroswap_pool_read_sac_contract_balance(
        &self,
        token_id: &ContractId,
        owner: AddressObject,
    ) -> Result<Option<i128>, HostError> {
        let owner_id = match self.scaddress_from_address(owner)? {
            ScAddress::Contract(id) => id,
            _ => return Ok(None),
        };
        let instance_key = self.contract_instance_ledger_key(token_id)?;
        // Peek at the instance's executable discriminant only — avoids the
        // metered_clone of the full `ScContractInstance` (including its
        // instance-storage `ScMap`) that `retrieve_contract_instance_from_storage`
        // would perform. The instance storage is irrelevant for SAC balance
        // reads; we only need to confirm the executable is `StellarAsset` so
        // we can mirror SAC `balance`'s storage side effects directly.
        if !self.contract_instance_executable_is_stellar_asset(&instance_key)? {
            return Ok(None);
        }
        self.extend_contract_instance_ttl_from_contract_id(
            instance_key,
            INSTANCE_TTL_THRESHOLD,
            INSTANCE_EXTEND_AMOUNT,
        )?;
        read_contract_balance_for_contract_owner(self, token_id, &owner_id).map(Some)
    }

    fn instantiate_vm(&self, id: &ContractId, wasm_hash: &Hash) -> Result<Rc<Vm>, HostError> {
        let contract_id = id.metered_clone(self)?;
        if let Some(cache) = &*self.try_borrow_module_cache()? {
            // Check that storage thinks the entry exists before
            // checking the cache: this seems like overkill but it
            // provides some future-proofing, see below.
            let wasm_key = self.contract_code_ledger_key(wasm_hash)?;
            if self.try_borrow_storage_mut()?.has(&wasm_key, self, None)? {
                if let Some(parsed_module) = cache.get_module(wasm_hash)? {
                    return Vm::from_parsed_module_and_wasmi_linker(
                        self,
                        contract_id,
                        parsed_module,
                        &cache.wasmi_linker,
                    );
                }
            }
        };

        // We can get here a few ways:
        //
        //   1. We are in simulation so don't have a module cache.
        //
        //   2. We have a module cache, but it somehow doesn't have
        //      the module requested. This in turn has two
        //      sub-cases:
        //
        //     - User invoked us with bad input, eg. calling a
        //       contract that wasn't provided in footprint/storage.
        //
        //     - User uploaded the wasm in this ledger so we didn't
        //       cache it when starting the ledger (and couldn't add
        //       it: the module cache and the wasmi engine used to
        //       build modules are both locked and shared across
        //       threads during execution, we don't want to perturb
        //       it even if we could; uploads use a throwaway engine
        //       for validation purposes).
        //
        //   3. Even more pathological: the module cache was built,
        //      and contained the module, but someone _removed_ the
        //      wasm from storage after the the cache was built
        //      (this is not currently possible from guest code, but
        //      we do some future-proofing here in case it becomes
        //      possible). This is the case we handle above with the
        //      early check for storage.has(wasm_key) before
        //      checking the cache as well.
        //
        // In all these cases, we want to try accessing storage, and
        // if it has the wasm, make a _throwaway_ module with its
        // own engine. If it doesn't have the wasm, we want to fail
        // with a storage error.

        #[cfg(any(test, feature = "recording_mode"))]
        // In recording mode:
        //   - We have no _real_ module cache.
        //   - We have a choice of whether simulate a cache-hit.
        //     - We will "simulate a hit" by doing a fresh parse but charging it
        //       to the shadow budget, not the real budget.
        //   - We _want_ to simulate a miss any time _would_ be a corresponding
        //     (charged-for) miss in enforcing mode, because otherwise we're
        //     under-charging and the tx we're simulating will fail in execution.
        //   - One case we know for sure will cause a miss: if the module
        //     literally isn't in the snapshot at all. This happens when someone
        //     uploads a contract and tries running it in the same ledger.
        //   - Other cases we're _not sure_: a module might be expired and evicted
        //     (thus removed from module cache) between simulation time and
        //     enforcement time.
        //   - But we can and do make an approximation here:
        //     - If the module is _expired_ we assume it's on its way to eviction
        //       soon and simulate a miss, risking overcharging.
        //     - If the module is _not expired_ we assume it'll be survive until
        //       execution, simulate a hit, and risk undercharging.
        if self.in_storage_recording_mode()? {
            if let Some((parsed_module, wasmi_linker)) =
                self.budget_ref().with_observable_shadow_mode(|| {
                    use crate::vm::ParsedModule;
                    let wasm_key = self.contract_code_ledger_key(wasm_hash)?;
                    let is_key_live_in_snapshot = self
                        .try_borrow_storage_mut()?
                        .is_key_live_in_snapshot(self, &wasm_key)?;
                    if is_key_live_in_snapshot {
                        let (code, _costs) = self.retrieve_wasm_from_storage(&wasm_hash)?;
                        // Currently only v0 costs are used by the inter-ledger
                        // module cache. Note, that when key is not in the live
                        // snapshot, the inter-ledger cache won't be used
                        // either, so we'll end up in the "cache miss" case
                        // below and then correctly charge the instantiation
                        // costs in `Vm::new_with_cost_inputs`.
                        let costs_v0 = crate::vm::VersionedContractCodeCostInputs::V0 {
                            wasm_bytes: code.len(),
                        };
                        let parsed_module = ParsedModule::new_with_isolated_engine(
                            self,
                            code.as_slice(),
                            costs_v0,
                        )?;
                        let wasmi_linker = parsed_module.make_wasmi_linker(self)?;
                        Ok(Some((parsed_module, wasmi_linker)))
                    } else {
                        Ok(None)
                    }
                })?
            {
                return Vm::from_parsed_module_and_wasmi_linker(
                    self,
                    contract_id,
                    parsed_module,
                    &wasmi_linker,
                );
            }
        }

        let (code, costs) = self.retrieve_wasm_from_storage(&wasm_hash)?;
        Vm::new_with_cost_inputs(self, contract_id, code.as_slice(), costs)
    }

    pub(crate) fn get_contract_protocol_version(
        &self,
        contract_id: &ContractId,
    ) -> Result<u32, HostError> {
        #[cfg(any(test, feature = "testutils"))]
        if self.is_test_contract_executable(contract_id)? {
            return self.get_ledger_protocol_version();
        }
        let storage_key = self.contract_instance_ledger_key(contract_id)?;
        let instance = self.retrieve_contract_instance_from_storage(&storage_key)?;
        match &instance.executable {
            ContractExecutable::Wasm(wasm_hash) => {
                let vm = self.instantiate_vm(contract_id, wasm_hash)?;
                Ok(vm.module.proto_version)
            }
            ContractExecutable::StellarAsset => self.get_ledger_protocol_version(),
        }
    }

    // Notes on metering: this is covered by the called components.
    pub(crate) fn call_n_internal(
        &self,
        id: &ContractId,
        func: Symbol,
        args: &[Val],
        call_params: CallParams,
    ) -> Result<Val, HostError> {
        #[cfg(any(test, feature = "testutils"))]
        let _invocation_meter_scope = self.maybe_meter_invocation(
            crate::host::invocation_metering::MeteringInvocation::contract_invocation(
                self, id, func,
            ),
        );
        // Internal host calls may call some special functions that otherwise
        // aren't allowed to be called.
        if !call_params.internal_host_call
            && SymbolStr::try_from_val(self, &func)?
                .to_string()
                .as_str()
                .starts_with(RESERVED_CONTRACT_FN_PREFIX)
        {
            return Err(self.err(
                ScErrorType::Context,
                ScErrorCode::InvalidAction,
                "can't invoke a reserved function directly",
                &[func.to_val()],
            ));
        }

        if !matches!(call_params.reentry_mode, ContractReentryMode::Allowed) {
            let reentry_distance = self
                .try_borrow_context_stack()?
                .iter()
                .rev()
                .filter_map(|c| c.frame.contract_id())
                .position(|caller| caller == id);

            match (call_params.reentry_mode, reentry_distance) {
                // Non-reentrant calls, or calls in Allowed mode,
                // or immediate-reentry calls in SelfAllowed mode
                // are all acceptable.
                (_, None) | (ContractReentryMode::Allowed, _) => (),
                (ContractReentryMode::SelfAllowed, Some(0)) => {
                    // Persist the instance storage before making a re-entrant
                    // call in order to allow the re-entered call to see the
                    // instance storage changes made so far in the current call.
                    self.persist_instance_storage()?;
                }

                // But any non-immediate-reentry in SelfAllowed mode,
                // or any reentry at all in Prohibited mode, are errors.
                (ContractReentryMode::SelfAllowed, Some(_))
                | (ContractReentryMode::Prohibited, Some(_)) => {
                    return Err(self.err(
                        ScErrorType::Context,
                        ScErrorCode::InvalidAction,
                        "Contract re-entry is not allowed",
                        &[],
                    ));
                }
            }
        }

        self.fn_call_diagnostics(id, &func, args);

        // Try dispatching the contract to the compiled-in registred
        // implmentation. Only the contracts with the special (empty) executable
        // are dispatched in this way, so that it's possible to switch the
        // compiled-in implementation back to Wasm via
        // `update_current_contract_wasm`.
        // "testutils" is not covered by budget metering.
        #[cfg(any(test, feature = "testutils"))]
        if self.is_test_contract_executable(id)? {
            // This looks a little un-idiomatic, but this avoids maintaining a borrow of
            // self.0.contracts. Implementing it as
            //
            //     if let Some(cfs) = self.try_borrow_contracts()?.get(&id).cloned() { ... }
            //
            // maintains a borrow of self.0.contracts, which can cause borrow errors.
            let cfs_option = self.try_borrow_contracts()?.get(&id).cloned();
            if let Some(cfs) = cfs_option {
                let frame = self.create_test_contract_frame(id.clone(), func, args.to_vec())?;
                let panic = frame.panic.clone();
                return self.with_frame(Frame::TestContract(frame), || {
                    use std::any::Any;
                    use std::panic::AssertUnwindSafe;
                    type PanicVal = Box<dyn Any + Send>;

                    // We're directly invoking a native rust contract here,
                    // which we allow only in local testing scenarios, and we
                    // want it to behave as close to the way it would behave if
                    // the contract were actually compiled to WASM and running
                    // in a VM.
                    //
                    // In particular: if the contract function panics, if it
                    // were WASM it would cause the VM to trap, so we do
                    // something "as similar as we can" in the native case here,
                    // catch the native panic and attempt to continue by
                    // translating the panic back to an error, so that
                    // `with_frame` will rollback the host to its pre-call state
                    // (as best it can) and propagate the error to its caller
                    // (which might be another contract doing try_call).
                    //
                    // This is somewhat best-effort, but it's compiled-out when
                    // building a host for production use, so we're willing to
                    // be a bit forgiving.
                    let closure = AssertUnwindSafe(move || cfs.call(&func, self, args));
                    let res: Result<Option<Val>, PanicVal> =
                        crate::testutils::call_with_suppressed_panic_hook(closure);
                    match res {
                        Ok(Some(val)) => {
                            self.fn_return_diagnostics(id, &func, &val);
                            Ok(val)
                        }
                        Ok(None) => {
                            if call_params.treat_missing_function_as_noop {
                                Ok(Val::VOID.into())
                            } else {
                                Err(self.err(
                                    ScErrorType::Context,
                                    ScErrorCode::MissingValue,
                                    "calling unknown contract function",
                                    &[func.to_val()],
                                ))
                            }
                        }
                        Err(panic_payload) => {
                            // Return an error indicating the contract function
                            // panicked.
                            //
                            // If it was a panic generated by a Env-upgraded
                            // HostError, it had its `Error` captured by
                            // `VmCallerEnv::escalate_error_to_panic`: fish the
                            // `Error` stored in the frame back out and
                            // propagate it.
                            //
                            // If it was a panic generated by user code calling
                            // panic!(...) we won't retrieve such a stored
                            // `Error`. Since we're trying to emulate
                            // what-the-VM-would-do here, and the VM traps with
                            // an unreachable error on contract panic, we
                            // generate same error (by converting a wasm
                            // trap-unreachable code). It's a little weird
                            // because we're not actually running a VM, but we
                            // prioritize emulation fidelity over honesty here.
                            let mut error: Error =
                                Error::from(wasmi::core::TrapCode::UnreachableCodeReached);

                            let mut recovered_error_from_panic_refcell = false;
                            if let Ok(panic) = panic.try_borrow() {
                                if let Some(err) = *panic {
                                    recovered_error_from_panic_refcell = true;
                                    error = err;
                                }
                            }

                            // If we didn't manage to recover a structured error
                            // code from the frame's refcell, and we're allowed
                            // to record dynamic strings (which happens when
                            // diagnostics are active), and we got a panic
                            // payload of a simple string, log that panic
                            // payload into the diagnostic event buffer. This
                            // code path will get hit when contracts do
                            // `panic!("some string")` in native testing mode.
                            if !recovered_error_from_panic_refcell {
                                self.with_debug_mode(|| {
                                    if let Some(str) = panic_payload.downcast_ref::<&str>() {
                                        let msg: String = format!(
                                            "caught panic '{}' from contract function '{:?}'",
                                            str, func
                                        );
                                        let _ = self.log_diagnostics(&msg, args);
                                    } else if let Some(str) = panic_payload.downcast_ref::<String>()
                                    {
                                        let msg: String = format!(
                                            "caught panic '{}' from contract function '{:?}'",
                                            str, func
                                        );
                                        let _ = self.log_diagnostics(&msg, args);
                                    };
                                    Ok(())
                                })
                            }
                            Err(self.error(error, "caught error from function", &[]))
                        }
                    }
                });
            }
        }

        let res =
            self.call_contract_fn(id, &func, args, call_params.treat_missing_function_as_noop);

        match &res {
            Ok(res) => self.fn_return_diagnostics(id, &func, res),
            Err(_err) => {}
        }

        res
    }

    // Notes on metering: covered by the called components.
    fn invoke_function_and_return_val(&self, hf: HostFunction) -> Result<Val, HostError> {
        let hf_type = hf.discriminant();
        let frame = Frame::HostFunction(hf_type);
        match hf {
            HostFunction::InvokeContract(invoke_args) => {
                self.with_frame(frame, || {
                    // Metering: conversions to host objects are covered.
                    let ScAddress::Contract(ref contract_id) = invoke_args.contract_address else {
                        return Err(self.err(
                            ScErrorType::Value,
                            ScErrorCode::UnexpectedType,
                            "invoked address doesn't belong to a contract",
                            &[],
                        ));
                    };
                    let function_name: Symbol = invoke_args.function_name.try_into_val(self)?;
                    let args = self.scvals_to_val_vec(invoke_args.args.as_slice())?;
                    self.call_n_internal(
                        contract_id,
                        function_name,
                        args.as_slice(),
                        CallParams::default_external_call(),
                    )
                })
            }
            HostFunction::CreateContract(args) => self.with_frame(frame, || {
                let deployer: Option<AddressObject> = match &args.contract_id_preimage {
                    ContractIdPreimage::Address(preimage_from_addr) => {
                        Some(self.add_host_object(preimage_from_addr.address.metered_clone(self)?)?)
                    }
                    ContractIdPreimage::Asset(_) => None,
                };
                self.create_contract_internal(
                    deployer,
                    CreateContractArgsV2 {
                        contract_id_preimage: args.contract_id_preimage,
                        executable: args.executable,
                        constructor_args: Default::default(),
                    },
                    vec![],
                )
                .map(<Val>::from)
            }),
            HostFunction::CreateContractV2(args) => self.with_frame(frame, || {
                let deployer: Option<AddressObject> = match &args.contract_id_preimage {
                    ContractIdPreimage::Address(preimage_from_addr) => {
                        Some(self.add_host_object(preimage_from_addr.address.metered_clone(self)?)?)
                    }
                    ContractIdPreimage::Asset(_) => None,
                };
                let arg_vals = self.scvals_to_val_vec(args.constructor_args.as_slice())?;
                self.create_contract_internal(deployer, args, arg_vals)
                    .map(<Val>::from)
            }),
            HostFunction::UploadContractWasm(wasm) => self.with_frame(frame, || {
                self.upload_contract_wasm(wasm.to_vec()).map(<Val>::from)
            }),
        }
    }

    // Notes on metering: covered by the called components.
    pub fn invoke_function(&self, hf: HostFunction) -> Result<ScVal, HostError> {
        #[cfg(any(test, feature = "testutils"))]
        let _invocation_meter_scope = self.maybe_meter_invocation(
            crate::host::invocation_metering::MeteringInvocation::from_host_function(&hf),
        );

        let rv = self.invoke_function_and_return_val(hf)?;
        self.from_host_val(rv)
    }

    pub(crate) fn maybe_init_instance_storage(&self, ctx: &mut Context) -> Result<(), HostError> {
        // Lazily initialize the storage on first access - it's not free and
        // not every contract will use it.
        if ctx.storage.is_some() {
            return Ok(());
        }
        let Some(instance) = ctx.frame.instance() else {
            return Err(self.err(
                ScErrorType::Context,
                ScErrorCode::InternalError,
                "access to instance in frame without instance",
                &[],
            ));
        };
        ctx.storage = Some(InstanceStorageMap::from_instance_xdr(instance, self)?);
        Ok(())
    }

    fn reload_instance_storage(&self, ctx: &mut Context) -> Result<(), HostError> {
        let Some(contract_id) = ctx.frame.contract_id() else {
            return Err(self.err(
                ScErrorType::Context,
                ScErrorCode::InternalError,
                "access to instance storage in frame without contract ID",
                &[],
            ));
        };
        let instance_key = self.contract_instance_ledger_key(&contract_id)?;
        let instance = self.retrieve_contract_instance_from_storage(&instance_key)?;
        ctx.storage = Some(InstanceStorageMap::from_instance_xdr(&instance, self)?);
        Ok(())
    }

    fn maybe_reload_instance_storage_on_frame_pop(&self) -> Result<(), HostError> {
        let mut contexts = self.try_borrow_context_stack_mut()?;
        let contexts_len = contexts.len();
        let Some(curr_contract_id) = contexts.last().and_then(|ctx| {
            // The clone is unmetered for simplicity, this operation is rare.
            ctx.frame.contract_id().cloned()
        }) else {
            return Err(self.err(
                ScErrorType::Context,
                ScErrorCode::InternalError,
                "no contract in current frame during instance storage reload",
                &[],
            ));
        };

        // Iterate all the contexts besides the top-most (which
        // is being popped now).
        for ctx in contexts.iter_mut().take(contexts_len - 1) {
            if ctx.frame.contract_id() == Some(&curr_contract_id) {
                self.reload_instance_storage(ctx)?;
            }
        }
        Ok(())
    }

    // Make the in-memory instance storage persist into the `Storage` by writing
    // its updated contents into corresponding `ContractData` ledger entry.
    // Returns `true` if instance storage was persisted, `false` otherwise (i.e.
    // when there are no changes to persist).
    fn persist_instance_storage(&self) -> Result<bool, HostError> {
        let updated_instance_storage = self.with_current_context_mut(|ctx| {
            if let Frame::NativeContract(_, func, _, instance) = &ctx.frame {
                if self.symbol_matches(b"swap", *func)? {
                    return instance.storage.metered_clone(self);
                }
            }
            if let Some(storage) = &ctx.storage {
                if !storage.is_modified {
                    return Ok(None);
                }
                Ok(Some(self.instance_storage_map_to_scmap(&storage.map)?))
            } else {
                Ok(None)
            }
        })?;
        if updated_instance_storage.is_some() {
            let contract_id = self.get_current_contract_id_internal()?;
            let key = self.contract_instance_ledger_key(&contract_id)?;

            self.store_contract_instance(None, updated_instance_storage, contract_id, &key)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
