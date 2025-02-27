use crate::{
    budget::AsBudget,
    events::Events,
    xdr::{self, Hash, LedgerKey, ScAddress, ScError, ScErrorCode, ScErrorType},
    ConversionError, EnvBase, Error, Host, TryFromVal, U32Val, Val,
};
use backtrace::{Backtrace, BacktraceFrame};
use core::fmt::Debug;
use std::{
    cell::{Ref, RefCell, RefMut},
    ops::DerefMut,
    rc::Rc,
};

use super::metered_clone::MeteredClone;

#[derive(Clone)]
pub(crate) struct DebugInfo {
    pub(crate) events: Events,
    pub(crate) backtrace: Backtrace,
}

#[derive(Clone)]
pub struct HostError {
    pub error: Error,
    pub(crate) info: Option<Box<DebugInfo>>,
}

impl std::error::Error for HostError {}

impl Into<Error> for HostError {
    fn into(self) -> Error {
        self.error
    }
}

impl Debug for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // We do a little trimming here, skipping the first two frames (which
        // are always into, from, and one or more Host::err_foo calls) and all
        // the frames _after_ the short-backtrace marker that rust compiles-in.

        fn frame_name_matches(frame: &BacktraceFrame, pat: &str) -> bool {
            for sym in frame.symbols() {
                match sym.name() {
                    Some(sn) if format!("{:}", sn).contains(pat) => {
                        return true;
                    }
                    _ => (),
                }
            }
            false
        }

        fn frame_is_short_backtrace_start(frame: &BacktraceFrame) -> bool {
            frame_name_matches(frame, "__rust_begin_short_backtrace")
        }

        fn frame_is_initial_error_plumbing(frame: &BacktraceFrame) -> bool {
            frame_name_matches(frame, "::from")
                || frame_name_matches(frame, "::into")
                || frame_name_matches(frame, "host::err")
                || frame_name_matches(frame, "Host::err")
                || frame_name_matches(frame, "Host>::err")
                || frame_name_matches(frame, "::augment_err_result")
        }

        writeln!(f, "HostError: {:?}", self.error)?;
        if let Some(info) = &self.info {
            let mut bt = info.backtrace.clone();
            bt.resolve();
            let frames: Vec<BacktraceFrame> = bt
                .frames()
                .iter()
                .skip_while(|f| frame_is_initial_error_plumbing(f))
                .take_while(|f| !frame_is_short_backtrace_start(f))
                .cloned()
                .collect();
            let bt: Backtrace = frames.into();
            // TODO: maybe make this something users can adjust?
            const MAX_EVENTS: usize = 25;
            let mut wrote_heading = false;
            for (i, e) in info.events.0.iter().rev().take(MAX_EVENTS).enumerate() {
                if !wrote_heading {
                    writeln!(f)?;
                    writeln!(f, "Event log (newest first):")?;
                    wrote_heading = true;
                }
                writeln!(f, "   {}: {}", i, e)?;
            }
            if info.events.0.len() > MAX_EVENTS {
                writeln!(f, "   {}: ... elided ...", MAX_EVENTS)?;
            }
            writeln!(f)?;
            writeln!(f, "Backtrace (newest first):")?;
            writeln!(f, "{:?}", bt)
        } else {
            writeln!(f, "DebugInfo not available")
        }
    }
}

impl HostError {
    #[cfg(test)]
    pub fn result_matches_err<T, C>(res: Result<T, HostError>, code: C) -> bool
    where
        Error: From<C>,
    {
        match res {
            Ok(_) => false,
            Err(he) => {
                let error: Error = code.into();
                he.error == error
            }
        }
    }

    /// Identifies whether the error can be meaningfully recovered from.
    ///
    /// We consider errors that occur due to broken execution preconditions (
    /// such as incorrect footprint) non-recoverable.
    pub fn is_recoverable(&self) -> bool {
        // All internal errors that originate from the host can be considered
        // non-recoverable (they should only appear if there is some bug in the
        // host implementation or setup).
        if !self.error.is_type(ScErrorType::Contract)
            && self.error.is_code(ScErrorCode::InternalError)
        {
            return false;
        }
        // Exceeding the budget or storage limit is non-recoverable. Exceeding
        // storage 'limit' is basically accessing entries outside of the
        // supplied footprint.
        if self.error.is_code(ScErrorCode::ExceededLimit)
            && (self.error.is_type(ScErrorType::Storage) || self.error.is_type(ScErrorType::Budget))
        {
            return false;
        }

        true
    }
}

impl<T> From<T> for HostError
where
    Error: From<T>,
{
    fn from(error: T) -> Self {
        let error = error.into();
        Self { error, info: None }
    }
}

impl std::fmt::Display for HostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <HostError as Debug>::fmt(self, f)
    }
}

impl TryFrom<&HostError> for ScError {
    type Error = xdr::Error;
    fn try_from(err: &HostError) -> Result<Self, Self::Error> {
        err.error.try_into()
    }
}

impl From<HostError> for std::io::Error {
    fn from(e: HostError) -> Self {
        std::io::Error::new(std::io::ErrorKind::Other, e)
    }
}

pub(crate) trait TryBorrowOrErr<T> {
    fn try_borrow_or_err(&self) -> Result<Ref<'_, T>, Error>;
    fn try_borrow_mut_or_err(&self) -> Result<RefMut<'_, T>, Error>;
    fn try_borrow_or_err_with(&self, host: &Host, msg: &str) -> Result<Ref<'_, T>, HostError> {
        self.try_borrow_or_err()
            .map_err(|e| host.error(e, msg, &[]))
    }
    fn try_borrow_mut_or_err_with(
        &self,
        host: &Host,
        msg: &str,
    ) -> Result<RefMut<'_, T>, HostError> {
        self.try_borrow_mut_or_err()
            .map_err(|e| host.error(e, msg, &[]))
    }
}

impl<T> TryBorrowOrErr<T> for RefCell<T> {
    fn try_borrow_or_err(&self) -> Result<Ref<'_, T>, Error> {
        self.try_borrow().map_err(|_| {
            Error::from_type_and_code(ScErrorType::Context, ScErrorCode::InternalError)
        })
    }

    fn try_borrow_mut_or_err(&self) -> Result<RefMut<'_, T>, Error> {
        self.try_borrow_mut().map_err(|_| {
            Error::from_type_and_code(ScErrorType::Context, ScErrorCode::InternalError)
        })
    }
}

impl Host {
    /// Convenience function to construct an [Error] and pass to [Host::error].
    pub fn err(&self, type_: ScErrorType, code: ScErrorCode, msg: &str, args: &[Val]) -> HostError {
        let error = Error::from_type_and_code(type_, code);
        self.error(error, msg, args)
    }

    /// At minimum constructs and returns a [HostError] build from the provided
    /// [Error], and when running in [DiagnosticMode::Debug] additionally
    /// records a diagnostic event with the provided `msg` and `args` and then
    /// enriches the returned [Error] with [DebugInfo] in the form of a
    /// [Backtrace] and snapshot of the [Events] buffer.
    pub fn error(&self, error: Error, msg: &str, args: &[Val]) -> HostError {
        self.with_debug_mode(
            || {
                // We _try_ to take a mutable borrow of the events buffer refcell
                // while building up the event we're going to emit into the events
                // log, failing gracefully (just emitting a no-debug-info
                // `HostError` wrapping the supplied `Error`) if we cannot acquire
                // the refcell. This is to handle the "double fault" case where we
                // get an error _while performing_ any of the steps needed to record
                // an error as an event, below.
                if let Ok(mut events_refmut) = self.0.events.try_borrow_mut() {
                    self.record_err_diagnostics(events_refmut.deref_mut(), error, msg, args);
                }
                Ok(HostError {
                    error,
                    info: self.maybe_get_debug_info(),
                })
            },
            || error.into(),
        )
    }

    pub(crate) fn maybe_get_debug_info(&self) -> Option<Box<DebugInfo>> {
        self.with_debug_mode(
            || {
                if let Ok(events_ref) = self.0.events.try_borrow() {
                    let events = events_ref.externalize(self)?;
                    let backtrace = Backtrace::new_unresolved();
                    Ok(Some(Box::new(DebugInfo { backtrace, events })))
                } else {
                    Ok(None)
                }
            },
            || None,
        )
    }

    // Some common error patterns here.

    pub(crate) fn err_arith_overflow(&self) -> HostError {
        self.err(
            ScErrorType::Value,
            ScErrorCode::ArithDomain,
            "arithmetic overflow",
            &[],
        )
    }

    pub(crate) fn err_oob_linear_memory(&self) -> HostError {
        self.err(
            ScErrorType::WasmVm,
            ScErrorCode::IndexBounds,
            "out-of-bounds access to WASM linear memory",
            &[],
        )
    }

    pub(crate) fn err_oob_object_index(&self, index: Option<u32>) -> HostError {
        let type_ = ScErrorType::Object;
        let code = ScErrorCode::IndexBounds;
        let msg = "object index out of bounds";
        match index {
            None => self.err(type_, code, msg, &[]),
            Some(index) => self.err(type_, code, msg, &[U32Val::from(index).to_val()]),
        }
    }

    /// Given a result carrying some error type that can be converted to an
    /// [Error] and supports [core::fmt::Debug], calls [Host::error] with the
    /// error when there's an error, also passing the result of
    /// [core::fmt::Debug::fmt] when [Host::is_debug] is `true`. Returns a
    /// [Result] over [HostError].
    ///
    /// If you have an error type `T` you want to record as a detailed debug
    /// event and a less-detailed [Error] code embedded in a [HostError], add an
    /// `impl From<T> for Error` over in `soroban_env_common::error`, or in the
    /// module defining `T`, and call this where the error is generated.
    ///
    /// Note: we do _not_ want to `impl From<T> for HostError` for such types,
    /// as doing so will avoid routing them through the host in order to record
    /// their extended diagnostic information into the event log. This means you
    /// will wind up writing `host.map_err(...)?` a bunch in code that you used
    /// to be able to get away with just writing `...?`, there's no way around
    /// this if we want to record the diagnostic information.
    pub fn map_err<T, E>(&self, res: Result<T, E>) -> Result<T, HostError>
    where
        Error: From<E>,
        E: Debug,
    {
        res.map_err(|e| {
            if let Ok(true) = self.is_debug() {
                let msg = format!("{:?}", e);
                self.error(e.into(), &msg, &[])
            } else {
                self.error(e.into(), "", &[])
            }
        })
    }

    // Extracts the account id from the given ledger key as address object `Val`.
    // Returns Void for unsupported entries.
    // Useful as a helper for error reporting.
    pub(crate) fn account_address_from_key(&self, lk: &Rc<LedgerKey>) -> Result<Val, HostError> {
        let account_id = match lk.as_ref() {
            LedgerKey::Account(e) => &e.account_id,
            LedgerKey::Trustline(e) => &e.account_id,
            _ => {
                return Ok(Val::VOID.into());
            }
        };
        self.add_host_object(ScAddress::Account(
            account_id.metered_clone(self.as_budget())?,
        ))
        .map(|a| a.to_val())
    }

    pub(crate) fn decorate_account_footprint_error(
        &self,
        err: HostError,
        lk: &Rc<LedgerKey>,
        msg: &str,
    ) -> HostError {
        if err.error.is_type(ScErrorType::Storage) && err.error.is_code(ScErrorCode::ExceededLimit)
        {
            let account_address = self.account_address_from_key(lk);
            match account_address {
                Ok(account_address) => {
                    return self.err(
                        ScErrorType::Storage,
                        ScErrorCode::ExceededLimit,
                        msg,
                        &[account_address],
                    );
                }
                Err(e) => {
                    return e;
                }
            }
        }
        err
    }

    pub(crate) fn decorate_contract_data_storage_error(
        &self,
        err: HostError,
        key: Val,
    ) -> HostError {
        if !err.error.is_type(ScErrorType::Storage) {
            return err;
        }
        if err.error.is_code(ScErrorCode::ExceededLimit) {
            return self.err(
                ScErrorType::Storage,
                ScErrorCode::ExceededLimit,
                "trying to access contract storage key outside of the footprint",
                &[key],
            );
        }
        if err.error.is_code(ScErrorCode::MissingValue) {
            return self.err(
                ScErrorType::Storage,
                ScErrorCode::MissingValue,
                "trying to get non-existing value for contract storage key",
                &[key],
            );
        }
        err
    }

    pub(crate) fn decorate_contract_instance_storage_error(
        &self,
        err: HostError,
        contract_id: &Hash,
    ) -> HostError {
        if err.error.is_type(ScErrorType::Storage) && err.error.is_code(ScErrorCode::ExceededLimit)
        {
            return self.err(
                ScErrorType::Storage,
                ScErrorCode::ExceededLimit,
                "trying to access contract instance key outside of the footprint",
                // No need for metered clone here as we are on the unrecoverable
                // error path.
                &[self
                    .add_host_object(ScAddress::Contract(contract_id.clone()))
                    .map(|a| a.into())
                    .unwrap_or(Val::VOID.into())],
            );
        }
        err
    }

    pub(crate) fn decorate_contract_code_storage_error(
        &self,
        err: HostError,
        wasm_hash: &Hash,
    ) -> HostError {
        if err.error.is_type(ScErrorType::Storage) && err.error.is_code(ScErrorCode::ExceededLimit)
        {
            return self.err(
                ScErrorType::Storage,
                ScErrorCode::ExceededLimit,
                "trying to access contract code key outside of the footprint",
                // No need for metered clone here as we are on the unrecoverable
                // error path.
                &[self
                    .add_host_object(self.scbytes_from_hash(wasm_hash).unwrap_or_default())
                    .map(|a| a.into())
                    .unwrap_or(Val::VOID.into())],
            );
        }
        err
    }
}

pub(crate) trait DebugArg {
    fn debug_arg(host: &Host, arg: &Self) -> Val {
        // We similarly guard against double-faulting here by try-acquiring the event buffer,
        // which will fail if we're re-entering error reporting _while_ forming a debug argument.
        if let Ok(_guard) = host.0.events.try_borrow_mut() {
            host.with_debug_mode(
                || Self::debug_arg_maybe_expensive_or_fallible(host, arg),
                || {
                    Error::from_type_and_code(ScErrorType::Events, ScErrorCode::InternalError)
                        .into()
                },
            )
        } else {
            Error::from_type_and_code(ScErrorType::Events, ScErrorCode::InternalError).into()
        }
    }
    fn debug_arg_maybe_expensive_or_fallible(host: &Host, arg: &Self) -> Result<Val, HostError>;
}

impl<T> DebugArg for T
where
    Val: TryFromVal<Host, T>,
    HostError: From<<Val as TryFromVal<Host, T>>::Error>,
{
    fn debug_arg_maybe_expensive_or_fallible(host: &Host, arg: &Self) -> Result<Val, HostError> {
        Val::try_from_val(host, arg).map_err(|e| HostError::from(e))
    }
}

impl DebugArg for xdr::Hash {
    fn debug_arg_maybe_expensive_or_fallible(host: &Host, arg: &Self) -> Result<Val, HostError> {
        host.bytes_new_from_slice(arg.as_slice()).map(|b| b.into())
    }
}

impl DebugArg for str {
    fn debug_arg_maybe_expensive_or_fallible(host: &Host, arg: &Self) -> Result<Val, HostError> {
        host.string_new_from_slice(arg).map(|s| s.into())
    }
}

impl DebugArg for usize {
    fn debug_arg_maybe_expensive_or_fallible(_host: &Host, arg: &Self) -> Result<Val, HostError> {
        u32::try_from(*arg)
            .map(|x| U32Val::from(x).into())
            .map_err(|_| HostError::from(ConversionError))
    }
}

/// Helper for building multi-argument errors.
/// For example:
/// ```ignore
/// err!(host, error, "message", arg1, arg2);
/// ```
/// All arguments must be convertible to [Val] with [TryIntoVal]. This is
/// expected to be called from within a function that returns
/// `Result<_, HostError>`. If these requirements can't be fulfilled, use
/// the [Host::error] function directly.
#[macro_export]
macro_rules! err {
    ($host:expr, $error:expr, $msg:literal, $($args:expr),*) => {
        {
            if let Ok(true) = $host.is_debug() {
                $host.error($error.into(), $msg, &[$(<_ as $crate::host::error::DebugArg>::debug_arg($host, &$args)),*])
            } else {
                $host.error($error.into(), $msg, &[])
            }
        }
    };
}
