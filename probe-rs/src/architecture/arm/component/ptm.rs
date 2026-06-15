//! Driver for the Program Trace Macrocell (PTM), ARM DDI 0314H.
//!
//! The PTM is the instruction trace source on ARMv7-A/R processors (Cortex-A5/A7/A8/A9/A15,
//! Cortex-R4/R5). It implements the ETMv3 register interface, generating a compressed stream
//! of branch packets that can be used to reconstruct instruction-level execution history.
//!
//! On the RZ/A1L (Cortex-A9), the PTM is at:
//!   - Debug-APB address: 0x8003C000   (via AP1, "viewed from debugger")
//!   - AHB system address: 0xFC03C000  (via AP0)
//!
//! Typical usage with a circular ETF sink:
//! 1. Unlock the PTM (write magic value to LAR)
//! 2. Assert Programming bit (ETMCR.ProgBit) to enter programming mode
//! 3. Set the trace ID (ETMTRACEIDR, must be unique per trace source on the ATB bus)
//! 4. Program TraceEnable for unconditional tracing (`ETMTEEVR=0x6F`, `ETMTECR1=BIT(24)`)
//! 5. Program a periodic synchronization interval (`ETMSYNCFR=0x400`)
//! 6. Clear Programming bit to start tracing
//! 7. Check ETMSR.ProgBit == 0 to confirm trace is running

use crate::{
    Error,
    architecture::arm::{
        ArmDebugInterface, ArmError,
        component::{DebugComponentInterface, TraceEnabledFeatures, TraceMemoryConfig},
        memory::CoresightComponent,
    },
    memory_mapped_bitfield_register,
};

// ETMv3 register offsets (ARM DDI 0314H §3.2, word-addressed at 4-byte steps)
const REGISTER_OFFSET_ETMCCR: u32 = 0x004;  // Configuration Code Register (RO)
const REGISTER_OFFSET_ETMTRIGGER: u32 = 0x008; // Trigger Event Register
const REGISTER_OFFSET_ETMTSSCR: u32 = 0x018; // TraceEnable Start/Stop Control Register
const REGISTER_OFFSET_ETMTECR2: u32 = 0x01C; // TraceEnable Control Register 2
const REGISTER_OFFSET_ETMTEEVR: u32 = 0x020; // TraceEnable Event Register
const REGISTER_OFFSET_ETMTECR1: u32 = 0x024; // TraceEnable Control Register 1
const REGISTER_OFFSET_ETMEXTINSELR: u32 = 0x1EC; // External Input Select Register
const REGISTER_OFFSET_ETMSYNCFR: u32 = 0x1E0; // Synchronization Frequency Register
const REGISTER_OFFSET_ETMCCER: u32 = 0x1E8; // Configuration Code Extension Register
const REGISTER_OFFSET_ETMTSEVR: u32 = 0x1F8; // Timestamp Event Register
const REGISTER_OFFSET_ETMTRACEIDR: u32 = 0x200; // CoreSight Trace ID Register
const REGISTER_OFFSET_ETMLAR: u32 = 0xFB0;  // Lock Access Register (write 0xC5ACCE55 to unlock)


/// Magic value to unlock the PTM for programming (CoreSight standard).
const LAR_UNLOCK_KEY: u32 = 0xC5ACCE55;

/// ETMv3 always-true resource selector used by Linux's ETM3 driver and described as such in the TRM.
const ETM_HARD_WIRED_RESOURCE_A: u32 = 0x6F;
const ETM_EVENT_NOT_A: u32 = 1 << 14;
const ETM_DEFAULT_EVENT_VALUE: u32 = ETM_HARD_WIRED_RESOURCE_A | ETM_EVENT_NOT_A;
const ETMTSSCR_DISABLED: u32 = 0;
const ETMTECR2_DISABLED: u32 = 0;
const ETMTECR1_INCLUDE_EXCEPTIONS: u32 = 1 << 24;
// Note: there is no "trace enable" bit in ETMCR. Per the PTM-A9 TRM (DDI0401C, Table 2.3)
// bit[11] is Reserved/SBZP. Tracing is enabled solely by clearing ETMCR.ProgBit (bit 10)
// via `exit_programming_mode()`.
const ETMCR_TIMESTAMP_ENABLE: u32 = 1 << 28;
const ETMCR_RETURN_STACK_ENABLE: u32 = 1 << 29;
const ETMCCER_TIMESTAMP_SUPPORTED: u32 = 1 << 22;
const ETMCCER_RETURN_STACK_SUPPORTED: u32 = 1 << 23;

/// Linux uses the TRM-recommended default of 0x400 for periodic synchronization.
const DEFAULT_SYNC_INTERVAL: u32 = 0x400;

/// The Program Trace Macrocell (PTM).
///
/// Provides instruction-level execution trace for ARMv7-A/R cores.
pub struct ProgramTraceMacrocell<'a> {
    component: &'a CoresightComponent,
    interface: &'a mut dyn ArmDebugInterface,
}

impl<'a> ProgramTraceMacrocell<'a> {
    /// Construct a PTM driver from a discovered CoreSight component.
    pub fn new(interface: &'a mut dyn ArmDebugInterface, component: &'a CoresightComponent) -> Self {
        Self { component, interface }
    }

    /// Unlock the PTM for programming by writing the CoreSight lock access key.
    ///
    /// Must be called before any other configuration register writes.
    pub fn unlock(&mut self) -> Result<(), ArmError> {
        self.component.write_reg(self.interface, REGISTER_OFFSET_ETMLAR, LAR_UNLOCK_KEY)
    }

    /// Assert the Programming bit to enter programming mode.
    ///
    /// Must be set before changing trace configuration registers.
    /// Check `is_ready()` before clearing to confirm the PTM accepted the configuration.
    pub fn enter_programming_mode(&mut self) -> Result<(), ArmError> {
        let mut cr = EtmCr::load(self.component, self.interface)?;
        cr.set_prog_bit(true);
        cr.store(self.component, self.interface)
    }

    /// Clear the Programming bit to start trace capture.
    ///
    /// After this, call `is_ready()` to confirm trace is running.
    pub fn exit_programming_mode(&mut self) -> Result<(), ArmError> {
        let mut cr = EtmCr::load(self.component, self.interface)?;
        cr.set_prog_bit(false);
        cr.store(self.component, self.interface)
    }

    /// Program the PTM synchronization interval.
    ///
    /// Short ETF snapshots are hard to decode unless the stream contains a periodic A-Sync
    /// followed by I-Sync. A small interval trades bandwidth for a much higher chance that
    /// a bounded circular-buffer capture starts at a synchronization point.
    pub fn set_sync_interval(&mut self, interval: u32) -> Result<(), ArmError> {
        self.component.write_reg(
            self.interface,
            REGISTER_OFFSET_ETMSYNCFR,
            interval & 0x0fff,
        )
    }

    /// Program the TraceEnable logic for unconditional tracing.
    ///
    /// Also clears `ETMTRIGGER` and `ETMEXTINSELR` to prevent stale register state from
    /// emitting phantom `TRIGGER` or external-input packets into the trace stream.
    pub fn configure_trace_enable(&mut self) -> Result<(), ArmError> {
        // Disable the trigger event (set to a never-true resource encoding).
        // Without this, a stale ETMTRIGGER from a previous debug session could inject
        // spurious TRIGGER bytes that confuse the decoder.
        self.component
            .write_reg(self.interface, REGISTER_OFFSET_ETMTRIGGER, 0)?;
        self.component
            .write_reg(self.interface, REGISTER_OFFSET_ETMTSSCR, ETMTSSCR_DISABLED)?;
        self.component
            .write_reg(self.interface, REGISTER_OFFSET_ETMTECR2, ETMTECR2_DISABLED)?;
        self.component.write_reg(
            self.interface,
            REGISTER_OFFSET_ETMTEEVR,
            ETM_HARD_WIRED_RESOURCE_A,
        )?;
        self.component.write_reg(
            self.interface,
            REGISTER_OFFSET_ETMTECR1,
            ETMTECR1_INCLUDE_EXCEPTIONS,
        )?;
        // Disable external input selection (avoids phantom trace from external triggers).
        self.component
            .write_reg(self.interface, REGISTER_OFFSET_ETMEXTINSELR, 0)
    }

    /// Program the timestamp event register.
    pub fn configure_timestamp_event(&mut self) -> Result<(), ArmError> {
        self.component.write_reg(
            self.interface,
            REGISTER_OFFSET_ETMTSEVR,
            ETM_DEFAULT_EVENT_VALUE,
        )
    }

    /// Configure and start tracing.
    ///
    /// Sets the ATB trace ID, configures the TraceEnable logic, applies any supported richness
    /// options, and starts trace capture.
    /// The `trace_id` must be unique across all trace sources on the ATB bus (1–112 valid).
    ///
    /// Returns [`TraceEnabledFeatures`] indicating which optional features were actually activated.
    /// Features requested in `config` but not advertised by `ETMCCER` will have their
    /// corresponding field set to `false` in the returned struct.
    pub fn enable(&mut self, trace_id: u8, config: TraceMemoryConfig) -> Result<TraceEnabledFeatures, Error> {
        self.unlock()?;

        // Enter programming mode
        self.enter_programming_mode()?;

        // Set the ATB trace source ID (7-bit field, bits [6:0])
        self.component.write_reg(
            self.interface,
            REGISTER_OFFSET_ETMTRACEIDR,
            (trace_id & 0x7F) as u32,
        )?;

        self.set_sync_interval(DEFAULT_SYNC_INTERVAL)?;
        self.configure_trace_enable()?;

        // Configure main control register.
        // BranchOutput (bit[8]): when set, the PTM emits a branch address packet for every
        // executed branch (branch-broadcast mode), enabling offline trace reconstruction
        // without an ELF but at the cost of higher trace bandwidth.
        let mut cr = EtmCr::load(self.component, self.interface)?;
        cr.set_power_down(false);
        cr.set_branch_output(config.branch_broadcast);
        cr.set_cycle_accurate(false);
        cr.set_context_id_size(0);

        let capabilities = self.configuration_code_extension()?;
        let mut features = TraceEnabledFeatures::default();

        if config.timestamps && (capabilities & ETMCCER_TIMESTAMP_SUPPORTED != 0) {
            self.configure_timestamp_event()?;
            cr.0 |= ETMCR_TIMESTAMP_ENABLE;
            features.timestamps = true;
        }
        if config.return_stack && (capabilities & ETMCCER_RETURN_STACK_SUPPORTED != 0) {
            cr.0 |= ETMCR_RETURN_STACK_ENABLE;
            features.return_stack = true;
        }
        if config.branch_broadcast {
            // BranchOutput is always available — no CCER gate.
            features.branch_broadcast = true;
        }

        cr.store(self.component, self.interface)?;

        // Exit programming mode (clears ProgBit) — this is what starts trace capture.
        self.exit_programming_mode()?;

        Ok(features)
    }

    /// Disable trace capture (set power-down bit).
    pub fn disable(&mut self) -> Result<(), Error> {
        self.unlock()?;
        self.enter_programming_mode()?;
        let mut cr = EtmCr::load(self.component, self.interface)?;
        cr.set_power_down(true);
        cr.store(self.component, self.interface)?;
        Ok(())
    }

    /// Returns true if the PTM is idle (ProgBit in ETMSR is set = not yet running, in programming mode).
    ///
    /// After calling `exit_programming_mode()`, poll until this returns `false` to confirm
    /// that tracing has started.
    pub fn is_programming(&mut self) -> Result<bool, ArmError> {
        let sr = EtmSr::load(self.component, self.interface)?;
        Ok(sr.prog_bit())
    }

    /// Read the Configuration Code Register to identify PTM capabilities.
    pub fn configuration_code(&mut self) -> Result<u32, ArmError> {
        self.component.read_reg(self.interface, REGISTER_OFFSET_ETMCCR)
    }

    /// Read the Configuration Code Extension Register to identify optional PTM features.
    pub fn configuration_code_extension(&mut self) -> Result<u32, ArmError> {
        self.component.read_reg(self.interface, REGISTER_OFFSET_ETMCCER)
    }
}

memory_mapped_bitfield_register! {
    /// ETMv3 Main Control Register (ETMCR), ARM DDI 0314H §3.3.1
    pub struct EtmCr(u32);
    0x000, "ETMCR",
    impl From;

    /// PowerDown: 1 = PTM powered down (safe to read other registers), 0 = active.
    pub power_down, set_power_down: 0;
    /// ProgBit: set 1 to enter programming mode, clear to start trace
    pub prog_bit, set_prog_bit: 10;
    /// BranchOutputEnable: include branch addresses in trace stream
    pub branch_output, set_branch_output: 8;
    /// CycleAccurate: include cycle count packets (increases bandwidth significantly)
    pub cycle_accurate, set_cycle_accurate: 12;
    /// ContextIDSize[0:1]: number of context ID bytes (0 = disabled)
    pub u8, context_id_size, set_context_id_size: 15, 14;
}

impl DebugComponentInterface for EtmCr {}

memory_mapped_bitfield_register! {
    /// ETMv3 Status Register (ETMSR), ARM DDI 0314H §3.3.5
    pub struct EtmSr(u32);
    0x010, "ETMSR",
    impl From;

    /// UE (Untraced Execute): 1 = instructions executed that were not traced
    pub untraced_execute, _: 0;
    /// ProgBit: mirrors ETMCR.ProgBit, 1 = in programming mode (not tracing)
    pub prog_bit, _: 1;
    /// Overflow: 1 = trace FIFO overflowed, some trace data lost
    pub overflow, _: 2;
    /// Triggered: 1 = trigger event has been asserted
    pub triggered, _: 3;
}

impl DebugComponentInterface for EtmSr {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_enable_recipe_matches_trm_and_linux_etm3_defaults() {
        assert_eq!(ETMTSSCR_DISABLED, 0);
        assert_eq!(ETMTECR2_DISABLED, 0);
        assert_eq!(ETM_HARD_WIRED_RESOURCE_A, 0x6F);
        assert_eq!(ETMTECR1_INCLUDE_EXCEPTIONS, 1 << 24);
    }

    #[test]
    fn sync_interval_matches_recommended_default() {
        assert_eq!(DEFAULT_SYNC_INTERVAL, 0x400);
    }

    #[test]
    fn feature_bits_match_linux_etm3_definitions() {
        assert_eq!(ETMCR_TIMESTAMP_ENABLE, 1 << 28);
        assert_eq!(ETMCR_RETURN_STACK_ENABLE, 1 << 29);
        assert_eq!(ETMCCER_TIMESTAMP_SUPPORTED, 1 << 22);
        assert_eq!(ETMCCER_RETURN_STACK_SUPPORTED, 1 << 23);
        assert_eq!(ETM_DEFAULT_EVENT_VALUE, 0x406F);
    }

    #[test]
    fn branch_broadcast_uses_etmcr_bit8() {
        // ETMCR bit[8] is the BranchOutput (branch-broadcast) flag.
        // Verify the EtmCr bitfield accessor targets the correct bit position.
        let mut cr = EtmCr(0);
        assert!(!cr.branch_output());
        cr.set_branch_output(true);
        assert_eq!(cr.0 & (1 << 8), 1 << 8, "BranchOutput should set ETMCR bit[8]");
        cr.set_branch_output(false);
        assert_eq!(cr.0 & (1 << 8), 0, "BranchOutput clear should clear ETMCR bit[8]");
    }
}
