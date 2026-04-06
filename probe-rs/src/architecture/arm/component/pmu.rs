//! Performance Monitoring Unit (PMU) driver for ARM Cortex-A9.
//!
//! The PMU is a CoreSight component accessed via the APB debug bus.  Its base
//! address is discovered automatically through the ROM-table walk
//! (`PeripheralType::Pmu`).  All register accesses go through the standard
//! `component.read_reg` / `component.write_reg` helpers so the core does not
//! need to be halted during readout.
//!
//! **Register offsets** follow the IHI0029 "ARM Performance Monitors
//! Architecture" external debug register map (PMUv2).  Cross-reference:
//! ARM DDI 0388I (Cortex-A9 TRM r4p1) §11.3 "PMU External register summary".

use crate::{
    architecture::arm::{ArmDebugInterface, ArmError, memory::CoresightComponent},
};

// ── External debug register offsets (from PMU component base address) ────────
// IHI0029D Table 10-2 / DDI0388I §11.3
const PMEVCNTR_BASE: u32 = 0x000; // PMEVCNTRn = base + 4*n   (n = 0..5)
const PMCCNTR: u32 = 0x07C; // Cycle counter (32-bit)
const PMEVTYPER_BASE: u32 = 0x400; // PMEVTYPERn = base + 4*n  (n = 0..5)
const PMCNTENSET: u32 = 0xC00; // Counter enable set
const PMCNTENCLR: u32 = 0xC20; // Counter enable clear
const PMOVSR: u32 = 0xC80; // Overflow flag status (write 1 to clear)
#[allow(dead_code)]
const PMSELR: u32 = 0xD00; // Event counter selection (reserved for future use)
#[allow(dead_code)]
const PMXEVTYPER: u32 = 0xDA0; // Event type for selected counter (reserved for future use)
#[allow(dead_code)]
const PMXEVCNTR: u32 = 0xDC0; // Event count for selected counter (reserved for future use)
#[allow(dead_code)]
const PMUSERENR: u32 = 0xE00; // User-mode enable (EL0) (reserved for future use)
const PMCR: u32 = 0xE04; // PMU control register

// ── PMCR bit fields ───────────────────────────────────────────────────────────
const PMCR_E: u32 = 1 << 0; // Global enable
const PMCR_P: u32 = 1 << 1; // Reset all event counters to 0
const PMCR_C: u32 = 1 << 2; // Reset cycle counter to 0
#[allow(dead_code)]
const PMCR_D: u32 = 1 << 3; // Clock divider (0 = count every cycle)
const PMCR_N_SHIFT: u32 = 11; // N-field: number of event counters [15:11]
const PMCR_N_MASK: u32 = 0x1F;

// ── PMCNTENSET / PMCNTENCLR bit 31 = cycle counter ───────────────────────────
const PMCNTEN_CCNTR: u32 = 1 << 31;

/// Cortex-A9 PMU hardware event selectors.
///
/// Values are the 8-bit event identifiers written into PMEVTYPERn
/// (ARM DDI 0388I Table 11-23).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PmuEvent {
    /// Software-triggered increment via PMSWINC.
    SoftwareIncrement = 0x00,
    /// Instruction fetch that causes a refill in the L1 instruction cache.
    L1ICacheRefill = 0x01,
    /// Instruction TLB refill.
    ItlbRefill = 0x02,
    /// Data access that causes a refill in the L1 data cache.
    L1DCacheRefill = 0x03,
    /// Data or unified cache access.
    L1DCacheAccess = 0x04,
    /// Data TLB refill.
    DtlbRefill = 0x05,
    /// Data reads (including SWP, LDM, etc.).
    DataRead = 0x06,
    /// Data writes (including SWP, STM, etc.).
    DataWrite = 0x07,
    /// Instruction executed.
    InstructionExecuted = 0x08,
    /// Exception taken.
    ExceptionTaken = 0x09,
    /// Exception return executed.
    ExceptionReturn = 0x0A,
    /// Change to ContextID retired.
    ContextIdRetired = 0x0B,
    /// Software change of PC.
    SWChangePC = 0x0C,
    /// Immediate branch that is architecturally executed.
    ImmBranchExecuted = 0x0D,
    /// Procedure call executed.
    ProcedureCall = 0x0E,
    /// Unaligned load or store executed.
    UnalignedAccess = 0x0F,
    /// Branch mispredicted or not predicted.
    BranchMispredict = 0x10,
    /// Cycle counter (alias — normally the PMCCNTR is used directly).
    CycleCountAlias = 0x11,
    /// Predictable branches speculatively executed.
    BranchPredicted = 0x12,
    /// Data memory access.
    DataMemoryAccess = 0x13,
    /// L1 instruction cache access.
    L1ICacheAccess = 0x14,
    /// L1 data cache write-back.
    L1DCacheWriteback = 0x15,
    /// L2 data cache access.
    L2DCacheAccess = 0x16,
    /// L2 data cache refill.
    L2DCacheRefill = 0x17,
    /// L2 data cache write-back.
    L2DCacheWriteback = 0x18,
    /// Bus access.
    BusAccess = 0x19,
    /// Memory error (parity/ECC).
    MemoryError = 0x1A,
    /// Instruction speculatively executed.
    InstructionSpeculative = 0x1B,
    /// Bus cycle.
    BusCycle = 0x1D,
    /// Chain: even and odd event counter chained.
    Chain = 0x1E,
}

impl std::fmt::Display for PmuEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::SoftwareIncrement => "SW_INCR",
            Self::L1ICacheRefill => "L1I_CACHE_REFILL",
            Self::ItlbRefill => "L1I_TLB_REFILL",
            Self::L1DCacheRefill => "L1D_CACHE_REFILL",
            Self::L1DCacheAccess => "L1D_CACHE",
            Self::DtlbRefill => "L1D_TLB_REFILL",
            Self::DataRead => "LD_RETIRED",
            Self::DataWrite => "ST_RETIRED",
            Self::InstructionExecuted => "INST_RETIRED",
            Self::ExceptionTaken => "EXC_TAKEN",
            Self::ExceptionReturn => "EXC_RETURN",
            Self::ContextIdRetired => "CID_WRITE_RETIRED",
            Self::SWChangePC => "PC_WRITE_RETIRED",
            Self::ImmBranchExecuted => "BR_IMMED_RETIRED",
            Self::ProcedureCall => "BR_RETURN_RETIRED",
            Self::UnalignedAccess => "UNALIGNED_LDST_RETIRED",
            Self::BranchMispredict => "BR_MIS_PRED",
            Self::CycleCountAlias => "CPU_CYCLES",
            Self::BranchPredicted => "BR_PRED",
            Self::DataMemoryAccess => "MEM_ACCESS",
            Self::L1ICacheAccess => "L1I_CACHE",
            Self::L1DCacheWriteback => "L1D_CACHE_WB",
            Self::L2DCacheAccess => "L2D_CACHE",
            Self::L2DCacheRefill => "L2D_CACHE_REFILL",
            Self::L2DCacheWriteback => "L2D_CACHE_WB",
            Self::BusAccess => "BUS_ACCESS",
            Self::MemoryError => "MEMORY_ERROR",
            Self::InstructionSpeculative => "INST_SPEC",
            Self::BusCycle => "BUS_CYCLES",
            Self::Chain => "CHAIN",
        };
        write!(f, "{s}")
    }
}

/// A snapshot of PMU counter values taken at one instant.
#[derive(Debug, Clone)]
pub struct PmuSnapshot {
    /// Cycle counter value (PMCCNTR).
    pub cycles: u32,
    /// (Event selector, count) pairs for each configured event counter slot.
    pub events: Vec<(PmuEvent, u32)>,
}

/// Driver for the Cortex-A9 Performance Monitoring Unit.
pub struct PerformanceMonitoringUnit<'a> {
    component: &'a CoresightComponent,
    interface: &'a mut dyn ArmDebugInterface,
}

impl<'a> PerformanceMonitoringUnit<'a> {
    /// Attach to the PMU CoreSight component.
    pub fn new(
        interface: &'a mut dyn ArmDebugInterface,
        component: &'a CoresightComponent,
    ) -> Self {
        Self {
            component,
            interface,
        }
    }

    /// Read the PMCR.N field: number of event counters implemented.
    pub fn n_counters(&mut self) -> Result<u8, ArmError> {
        let pmcr = self.component.read_reg(self.interface, PMCR)?;
        Ok(((pmcr >> PMCR_N_SHIFT) & PMCR_N_MASK) as u8)
    }

    /// Reset and configure the PMU to count the given events.
    ///
    /// - Disables all counters.
    /// - Resets cycle counter and all event counters to zero.
    /// - Programs each available event counter slot with the requested event.
    /// - Enables the cycle counter and requested event counters.
    /// - Enables the PMU globally.
    ///
    /// Slots beyond the hardware N-counter limit are silently ignored.
    pub fn configure(&mut self, events: &[PmuEvent]) -> Result<(), ArmError> {
        // 1. Disable all counters while we configure them.
        self.component.write_reg(self.interface, PMCNTENCLR, 0xFFFF_FFFF)?;

        // 2. Reset cycle counter + event counters, disable clock divider.
        self.component.write_reg(
            self.interface,
            PMCR,
            PMCR_P | PMCR_C, // reset both, keep E=0 for now
        )?;

        // 3. Clear overflow flags.
        self.component.write_reg(self.interface, PMOVSR, 0xFFFF_FFFF)?;

        // 4. Program event types.
        let n = self.n_counters()? as usize;
        let n_to_configure = events.len().min(n);

        for (i, event) in events.iter().enumerate().take(n_to_configure) {
            self.component.write_reg(
                self.interface,
                PMEVTYPER_BASE + 4 * i as u32,
                *event as u32,
            )?;
        }

        // 5. Build enable mask: bit 31 = CCNTR, bits 0..n_to_configure-1 = event counters.
        let enable_mask = PMCNTEN_CCNTR | ((1u32 << n_to_configure) - 1);
        self.component.write_reg(self.interface, PMCNTENSET, enable_mask)?;

        // 6. Enable the PMU globally (PMCR.E = 1).
        self.component.write_reg(self.interface, PMCR, PMCR_E)?;

        Ok(())
    }

    /// Read a snapshot of the current counter values.
    ///
    /// `events` must match the slice passed to `configure` so the result can be
    /// labelled correctly.
    pub fn read_results(&mut self, events: &[PmuEvent]) -> Result<PmuSnapshot, ArmError> {
        let cycles = self.component.read_reg(self.interface, PMCCNTR)?;

        let n = self.n_counters()? as usize;
        let n_to_read = events.len().min(n);

        let mut event_counts = Vec::with_capacity(n_to_read);
        for (i, &event) in events.iter().enumerate().take(n_to_read) {
            let count = self.component.read_reg(
                self.interface,
                PMEVCNTR_BASE + 4 * i as u32,
            )?;
            event_counts.push((event, count));
        }

        Ok(PmuSnapshot {
            cycles,
            events: event_counts,
        })
    }

    /// Disable the PMU (PMCR.E = 0, all counters disabled).
    pub fn disable(&mut self) -> Result<(), ArmError> {
        self.component.write_reg(self.interface, PMCNTENCLR, 0xFFFF_FFFF)?;
        self.component.write_reg(self.interface, PMCR, 0)?;
        Ok(())
    }
}
