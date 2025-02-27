#[cfg(any(test, feature = "testutils", feature = "recording_auth"))]
use crate::{budget::Budget, HostError};

#[cfg(any(test, feature = "testutils"))]
use crate::host::error::TryBorrowOrErr;

#[cfg(test)]
use crate::{budget::model::ScaledU64, xdr::ContractCostType};

#[cfg(test)]
impl Budget {
    pub fn reset_models(&self) -> Result<(), HostError> {
        self.mut_budget(|mut b| {
            b.cpu_insns.reset_models();
            b.mem_bytes.reset_models();
            Ok(())
        })
    }

    pub(crate) fn override_model_with_scaled_params(
        &self,
        ty: ContractCostType,
        const_cpu: u64,
        lin_cpu: ScaledU64,
        const_mem: u64,
        lin_mem: ScaledU64,
    ) -> Result<(), HostError> {
        let mut bgt = self.0.try_borrow_mut_or_err()?;

        let cpu_model = bgt.cpu_insns.get_cost_model_mut(ty);
        cpu_model.const_term = const_cpu;
        cpu_model.lin_term = lin_cpu;

        let mem_model = bgt.mem_bytes.get_cost_model_mut(ty);
        mem_model.const_term = const_mem;
        mem_model.lin_term = lin_mem;
        Ok(())
    }

    pub(crate) fn override_model_with_unscaled_params(
        &self,
        ty: ContractCostType,
        const_cpu: u64,
        lin_cpu: u64,
        const_mem: u64,
        lin_mem: u64,
    ) -> Result<(), HostError> {
        self.override_model_with_scaled_params(
            ty,
            const_cpu,
            ScaledU64::from_unscaled_u64(lin_cpu),
            const_mem,
            ScaledU64::from_unscaled_u64(lin_mem),
        )
    }

    pub(crate) fn track_wasm_mem_alloc(&self, delta: u64) -> Result<(), HostError> {
        let mut bgt = self.0.try_borrow_mut_or_err()?;
        bgt.tracker.wasm_memory = bgt.tracker.wasm_memory.saturating_add(delta);
        Ok(())
    }

    pub(crate) fn get_wasm_mem_alloc(&self) -> Result<u64, HostError> {
        Ok(self.0.try_borrow_or_err()?.tracker.wasm_memory)
    }
}

#[cfg(any(test, feature = "testutils"))]
impl Budget {
    pub fn reset_default(&self) -> Result<(), HostError> {
        *self.0.try_borrow_mut_or_err()? = super::BudgetImpl::default();
        Ok(())
    }

    pub fn reset_unlimited(&self) -> Result<(), HostError> {
        self.reset_unlimited_cpu()?;
        self.reset_unlimited_mem()?;
        Ok(())
    }

    pub fn reset_unlimited_cpu(&self) -> Result<(), HostError> {
        self.mut_budget(|mut b| {
            b.cpu_insns.reset(u64::MAX);
            Ok(())
        })?; // panic means multiple-mut-borrow bug
        self.reset_tracker()
    }

    pub fn reset_unlimited_mem(&self) -> Result<(), HostError> {
        self.mut_budget(|mut b| {
            b.mem_bytes.reset(u64::MAX);
            Ok(())
        })?;
        self.reset_tracker()
    }

    pub fn reset_tracker(&self) -> Result<(), HostError> {
        self.0.try_borrow_mut_or_err()?.tracker.reset();
        Ok(())
    }

    pub fn reset_limits(&self, cpu: u64, mem: u64) -> Result<(), HostError> {
        self.mut_budget(|mut b| {
            b.cpu_insns.reset(cpu);
            b.mem_bytes.reset(mem);
            Ok(())
        })?;
        self.reset_tracker()
    }

    /// Resets the `FuelConfig` we pass into Wasmi before running calibration.
    /// Wasmi instruction calibration requires running the same Wasmi insn
    /// a fixed number of times, record their actual cpu and mem consumption, then
    /// divide those numbers by the number of iterations, which is the fuel count.
    /// Fuel count is kept tracked on the Wasmi side, based on the `FuelConfig`
    /// of a specific fuel category. In order to get the correct, unscaled fuel
    /// count, we have to preset all the `FuelConfig` entries to 1.
    pub fn reset_fuel_config(&self) -> Result<(), HostError> {
        self.0.try_borrow_mut_or_err()?.fuel_config.reset();
        Ok(())
    }

    pub fn get_shadow_cpu_insns_consumed(&self) -> Result<u64, HostError> {
        Ok(self.0.try_borrow_or_err()?.cpu_insns.shadow_total_count)
    }

    pub fn get_shadow_mem_bytes_consumed(&self) -> Result<u64, HostError> {
        Ok(self.0.try_borrow_or_err()?.mem_bytes.shadow_total_count)
    }

    #[allow(unused)]
    pub fn shadow_cpu_limit_exceeded(&self) -> Result<bool, HostError> {
        let cpu = &self.0.try_borrow_or_err()?.cpu_insns;
        Ok(cpu.shadow_total_count > cpu.shadow_limit)
    }

    pub fn shadow_mem_limit_exceeded(&self) -> Result<bool, HostError> {
        let mem = &self.0.try_borrow_or_err()?.mem_bytes;
        Ok(mem.shadow_total_count > mem.shadow_limit)
    }
}

#[cfg(any(test, feature = "recording_auth"))]
impl Budget {
    /// Fallible version of `with_shadow_mode`, enabled only in testing and
    /// non-production scenarios. The non-fallible `with_shadow_mode` is the
    /// preferred method and should be used if at all possible.
    /// However, in testing and non-production workflows, sometimes we need the
    /// convenience of temporarily "turning off" the budget. This can happen for
    /// several reasons: we want the some test logic to not affect the production
    /// budget, or we want to maintain an accurate prediction of production budget
    /// during preflight. In the latter case, we want to exclude preflight-only
    /// logic from the budget. By routing metering to the shadow budget instead
    /// of turning the budget off completely, it offers some DOS-mitigation.
    pub(crate) fn with_shadow_mode_fallible<T, F>(&self, f: F) -> Result<T, HostError>
    where
        F: FnOnce() -> Result<T, HostError>,
    {
        let mut prev = false;
        let should_execute = self.mut_budget(|mut b| {
            prev = b.is_in_shadow_mode;
            b.is_in_shadow_mode = true;
            b.cpu_insns.check_budget_limit(true)?;
            b.mem_bytes.check_budget_limit(true)
        });

        let rt = match should_execute {
            Ok(_) => f(),
            Err(e) => Err(e),
        };

        self.mut_budget(|mut b| {
            b.is_in_shadow_mode = prev;
            Ok(())
        })?;

        rt
    }
}
