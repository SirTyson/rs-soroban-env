use crate::{
    budget::Budget,
    crypto::sha256_hash_from_bytes_raw,
    xdr::{ContractCostType, Limited, ReadXdr, ScBytes, ScErrorCode, ScErrorType, WriteXdr},
    BytesObject, Host, HostError, DEFAULT_XDR_RW_LIMITS,
};
use std::io::Write;

use super::ErrorHandler;

/// XDR encoder writer that defers `ValSer` budget charging by recording a
/// histogram of `(buf.len(), count)` pairs as the encoder emits chunks. The
/// caller in `metered_write_xdr` then performs a single batched charge after
/// serialization completes. This preserves the per-leaf `ValSer` cost totals
/// (each bucket is charged as `count * evaluate(1, Some(buf.len()))` for both
/// CPU and memory), and only changes the observable timing of the
/// budget-exceeded error from "mid-write" to "after the write completes into a
/// local Vec<u8>", which is non-observable to the caller because the buffer is
/// not exposed on the error path.
struct MeteredWrite<'a, W: Write> {
    histogram: Vec<(u64, u64)>,
    w: &'a mut W,
}

impl<W> Write for MeteredWrite<'_, W>
where
    W: Write,
{
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let len = buf.len() as u64;
        match self.histogram.iter_mut().find(|(l, _)| *l == len) {
            Some(entry) => entry.1 = entry.1.saturating_add(1),
            None => self.histogram.push((len, 1)),
        }
        self.w.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

impl Host {
    pub fn metered_hash_xdr(&self, obj: &impl WriteXdr) -> Result<[u8; 32], HostError> {
        let _span = tracy_span!("hash xdr");
        let mut buf = vec![];
        metered_write_xdr(self.budget_ref(), obj, &mut buf)?;
        sha256_hash_from_bytes_raw(&buf, self)
    }

    pub fn metered_from_xdr<T: ReadXdr>(&self, bytes: &[u8]) -> Result<T, HostError> {
        let _span = tracy_span!("read xdr");
        self.charge_budget(ContractCostType::ValDeser, Some(bytes.len() as u64))?;
        let mut limits = DEFAULT_XDR_RW_LIMITS;
        limits.len = bytes.len();
        self.map_err(T::from_xdr(bytes, limits))
    }

    pub(crate) fn metered_from_xdr_obj<T: ReadXdr>(
        &self,
        bytes: BytesObject,
    ) -> Result<T, HostError> {
        self.visit_obj(bytes, |hv: &ScBytes| self.metered_from_xdr(hv.as_slice()))
    }
}

pub fn metered_write_xdr(
    budget: &Budget,
    obj: &impl WriteXdr,
    w: &mut Vec<u8>,
) -> Result<(), HostError> {
    let _span = tracy_span!("write xdr");
    if budget.coalesced_host_metering()? {
        let start_len = w.len();
        let mut limited = Limited::new(w, DEFAULT_XDR_RW_LIMITS);
        let write_res = obj.write_xdr(&mut limited);
        let bytes_written = limited.inner.len().saturating_sub(start_len);
        if bytes_written != 0 {
            budget.charge(ContractCostType::ValSer, Some(bytes_written as u64))?;
        }
        return write_res.map_err(|_| (ScErrorType::Budget, ScErrorCode::ExceededLimit).into());
    }

    let mw = MeteredWrite {
        histogram: Vec::with_capacity(16),
        w,
    };
    let mut limited = Limited::new(mw, DEFAULT_XDR_RW_LIMITS);
    let write_res = obj.write_xdr(&mut limited);
    // Apply the deferred batched ValSer charge for every chunk that was
    // actually written, regardless of whether the encoder ultimately failed.
    // This preserves exact per-leaf cost totals for the bytes that were
    // emitted, and matches the behavior of the per-write path which also
    // charges for each chunk before returning any error.
    budget.charge_val_ser_batched(&limited.inner.histogram)?;
    // If the encoder itself failed (e.g. XDR length limit), surface that as a
    // budget error to match the prior behavior, since `Vec<u8>` cannot produce
    // a real IO error.
    write_res.map_err(|_| (ScErrorType::Budget, ScErrorCode::ExceededLimit).into())
}

// Host-less metered XDR decoding.
// Prefer using `metered_from_xdr` when host is available for better error
// reporting.
pub fn metered_from_xdr_with_budget<T: ReadXdr>(
    bytes: &[u8],
    budget: &Budget,
) -> Result<T, HostError> {
    let _span = tracy_span!("read xdr with budget");
    budget.charge(ContractCostType::ValDeser, Some(bytes.len() as u64))?;
    let mut limits = DEFAULT_XDR_RW_LIMITS;
    limits.len = bytes.len();
    T::from_xdr(bytes, limits).map_err(|e| e.into())
}
