//! Debug sequences for Renesas RZ/A1L Group (R7S721010/020/030).
//!
//! RZA1L CoreSight topology (TRM R01UH0437EJ0700, Table 43.6, Debug-APB addresses):
//!
//!   PTM-A9  (0x8003C000)  ← instruction trace source
//!       │
//!   CPU Trace Funnel (0x80024000)  ← single ATB link, port 0
//!       │
//!   CPU-ETF (0x80021000)  ← 4KB on-chip trace buffer (TMC, circular or software mode)
//!       │
//!   CPU-TPIU (0x80023000) ← parallel trace port (optional, requires PCB trace pins)
//!
//! The default `trace_start()` opens all funnel ports, which is sufficient for basic use.
//! This device-specific override puts the ETF into **circular buffer mode** (overwrites
//! oldest data when full) rather than the probe-rs default software-poll stall mode,
//! so tracing continues uninterrupted even when the probe is not actively reading.
//!
//! After halting the CPU, call `session.read_trace_data()` to drain the ETF.

use std::sync::Arc;

use probe_rs_target::CoreType;

use crate::{
    MemoryInterface,
    architecture::arm::{
        ArmDebugInterface, ArmError,
        memory::ArmMemoryInterface,
        component::{TmcMode, TraceMemoryController, TraceFunnel, TraceSink},
        core::{
            armv7ar::{execute_instruction, set_instruction_input},
            armv7ar_debug_regs::{Dbgdrcr, Dbgdscr, Dbgdtrtx, Dbgvcr},
            instructions::aarch32::{build_mcr, build_mrc},
            registers::cortex_m::{PC, XPSR},
        },
        memory::romtable::{CoresightComponent, PeripheralType},
        sequences::{ArmDebugSequence, ArmDebugSequenceError},
    },
    core::memory_mapped_registers::MemoryMappedRegister,
};

/// Debug sequences for Renesas RZ/A1L, RZ/A1LU, and RZ/A1LC.
#[derive(Debug)]
pub struct RZA1L;

impl RZA1L {
    /// Create a debug sequence handle for any RZ/A1L group device.
    pub fn create() -> Arc<Self> {
        Arc::new(Self)
    }
}

impl ArmDebugSequence for RZA1L {
    /// Skip the CWRR (Cortex Warm Reset Request) for RZ/A1L.
    ///
    /// The default `armv7ar_reset_system` triggers DBGPRCR.CWRR, which resets the
    /// Cortex-A9 core.  On RZ/A1L the ROM bootloader then reloads previous firmware
    /// from QSPI flash before the debug halt can catch the reset vector.  That
    /// firmware's startup refills the PL310 L2 cache with its own code/data at
    /// 0x2001_0000+, defeating the L2 invalidation that follows in `reset_catch_clear`.
    ///
    /// This matches the J-Link approach: halt the CPU in place (no system reset).
    /// The PL310 retains the pre-halt cache state, which `reset_catch_clear` then
    /// invalidates cleanly — with the CPU halted, no background agent can refill the
    /// cache before firmware starts executing.
    ///
    /// Consequence: SYSCR3 may already be 0x0F (set by the previously-running firmware
    /// during its startup), so the SYSCR3 write in `reset_catch_clear` is a no-op but
    /// harmless.  SCTLR, VBAR, CPSR, and all banked registers are fully overwritten in
    /// `writeback_registers` before the new firmware's first instruction executes.
    fn reset_system(
        &self,
        interface: &mut dyn ArmMemoryInterface,
        _core_type: CoreType,
        debug_base: Option<u64>,
    ) -> Result<(), ArmError> {
        let debug_base = debug_base
            .ok_or_else(|| ArmError::from(ArmDebugSequenceError::DebugBaseNotSpecified))?;

        // Issue a halt request (DBGDRCR.HRQ=1).  If the CPU is already halted this is a
        // no-op; if it is running it will halt within a few cycles.
        let drcr_addr = Dbgdrcr::get_mmio_address_from_base(debug_base)?;
        let mut drcr = Dbgdrcr(0);
        drcr.set_hrq(true);
        interface.write_word_32(drcr_addr, drcr.into())?;

        Ok(())
    }

    /// After halting the CPU in place (no CWRR), `reset_catch_clear`:
    ///
    /// 1. Enables ITREN in DBGDSCR so DBGITR instructions are forwarded to the CPU.
    /// 2. Writes CPG.SYSCR3 = 0x0F via DBGITR `STRB`, enabling all SRAM write banks.
    /// 3. Disables and fully invalidates the PL310 L2 cache via DBGITR STR/LDR.
    /// 4. Clears the DBGVCR reset-vector-catch bit (harmless since no reset happened).
    fn reset_catch_clear(
        &self,
        core: &mut dyn ArmMemoryInterface,
        _core_type: CoreType,
        debug_base: Option<u64>,
    ) -> Result<(), ArmError> {
        let debug_base = debug_base
            .ok_or_else(|| ArmError::from(ArmDebugSequenceError::DebugBaseNotSpecified))?;

        // Step 1: enable ITREN so DBGITR instructions are forwarded to the CPU.
        let dscr_addr = Dbgdscr::get_mmio_address_from_base(debug_base)?;
        let mut dbgdscr = Dbgdscr(core.read_word_32(dscr_addr)?);
        dbgdscr.set_itren(true);
        core.write_word_32(dscr_addr, dbgdscr.into())?;

        // Step 2: write CPG.SYSCR3 = 0x0F via STRB so the CPU can write retention SRAM.
        // SYSCR3 is a byte register at 0xFCFE_0408; use STRB to avoid clobbering neighbours.
        //
        //   MRC p14, 0, r0, c0, c5, 0   ; r0 ← DTRRX (= SYSCR3 address)
        //   MRC p14, 0, r1, c0, c5, 0   ; r1 ← DTRRX (= 0x0F)
        //   STRB r1, [r0]               ; mem8[0xFCFE_0408] ← 0x0F
        const SYSCR3_ADDR: u32 = 0xFCFE_0408;
        const SYSCR3_VAL: u32 = 0x0F;
        // STRB r1, [r0] = STR byte, r1->[r0+0]: cond=AL(E) 0101 1100 0000 0001 0000...0000
        const STRB_R1_R0: u32 = 0xE5C0_1000;

        set_instruction_input(core, debug_base, SYSCR3_ADDR)?;
        execute_instruction(core, debug_base, build_mrc(14, 0, 0, 0, 5, 0))?; // r0 = SYSCR3 addr

        set_instruction_input(core, debug_base, SYSCR3_VAL)?;
        execute_instruction(core, debug_base, build_mrc(14, 0, 1, 0, 5, 0))?; // r1 = 0x0F

        execute_instruction(core, debug_base, STRB_R1_R0)?; // STRB r1, [r0]

        tracing::debug!("RZA1L reset_catch_clear: set SYSCR3=0x0F at {:#010x}", SYSCR3_ADDR);

        // Step 4a: Stop all DMAC channels before DCCISW and ELF write.
        //
        // The previously-running firmware's DMA may be writing to SRAM addresses in the
        // firmware range.  Stopping DMAC here (before the 6-second DCCISW) ensures no
        // DMA traffic corrupts the DCCISW writeback or the subsequent ELF write.
        {
            const DMAC_BASE: u64      = 0xE820_0000;
            const CHCTRL_OFFS: u64    = 0x28;
            const CHCTRL_CLREN_SWRST: u32 = 0x0000_000A;
            for ch in 0u64..8 {
                let addr = DMAC_BASE + ch * 64 + CHCTRL_OFFS;
                if let Err(e) = core.write_word_32(addr, CHCTRL_CLREN_SWRST) {
                    tracing::debug!("DMAC ch{ch} stop failed: {e:?}");
                }
            }
            const DMAC_HI_EXTRA: u64  = 0x200;
            // Channels 8–15 live at a +0x200 offset from the ch*64 address.
            // RZ/A1 TRM Rev.3.00 Table 16.1 (DMAC3): channels 0–7 at 0xE8200000,
            // channels 8–15 at 0xE8200200 — same stride, different base.
            for ch in 8u64..16 {
                let addr = DMAC_BASE + ch * 64 + DMAC_HI_EXTRA + CHCTRL_OFFS;
                if let Err(e) = core.write_word_32(addr, CHCTRL_CLREN_SWRST) {
                    tracing::debug!("DMAC ch{ch} stop failed: {e:?}");
                }
            }
        }

        // Step 4b: Clear ALL debug vector catches (DBGVCR = 0).
        //
        // reset_catch_clear previously did a read-modify-write clearing only DBGVCR.R.
        // Any other catch bits set by a prior debugging session (e.g. DBGVCR.su = UNDEF
        // catch, DBGVCR.ss = SVC catch) would remain, causing the firmware to halt at the
        // exception vector entry on every exception instead of executing the handler.
        // This manifests as PC stuck at 0x20010004 (UNDEF) / 0x20010008 (SVC) with the
        // firmware never progressing past the vector table.
        let vcr_addr = Dbgvcr::get_mmio_address_from_base(debug_base)?;
        core.write_word_32(vcr_addr, 0)?;

        // Step 4c: Clean-and-invalidate the L1 D-cache (DCCISW) before the ELF write.
        //
        // The previously-running firmware may have dirty write-back D-cache lines covering
        // the SRAM region 0x20010000-0x20016000.  If we later execute DCCISW after writing
        // the new firmware, the Cortex-A9 implementation writes dirty lines back to SRAM
        // BEFORE invalidating ("clean-then-invalidate" behavior even for the plain DCISW
        // opcode, per ARM Cortex-A9 TRM §3.7) — overwriting the new firmware.
        // (Note: probe-rs uses DCC fast mode, not MEM-AP, for bulk RAM writes.)
        //
        // The correct sequence is: clean+invalidate BEFORE the ELF write, while SRAM still
        // holds the old firmware's data (the writeback is harmless).  After this, D-cache
        // has no dirty lines, so the subsequent MEM-AP ELF write goes to SRAM uncorrupted.
        //
        // DCCISW = MCR p15, 0, Rt, c7, c14, 2  (Data Cache Clean and Invalidate by Set/Way)
        // Cortex-A9: 4-way, 256 sets.  Operand: way[31:30] | set[13:5].  1024 iterations.
        {
            let dccisw = build_mcr(15, 0, 0, 7, 14, 2);
            let dsb    = build_mcr(15, 0, 0, 7, 10, 4);
            for way in 0u32..4 {
                for set in 0u32..256 {
                    let operand = (way << 30) | (set << 5);
                    set_instruction_input(core, debug_base, operand)?;
                    execute_instruction(core, debug_base, build_mrc(14, 0, 0, 0, 5, 0))?; // r0 ← operand
                    execute_instruction(core, debug_base, dccisw)?;
                }
            }
            execute_instruction(core, debug_base, dsb)?;
        }

        // Step 4d: Disable MMU and L1 caches before the RAM download.
        //
        // probe-rs writes the ELF via the halted core's DCC data path. If the previous
        // firmware's SCTLR.M/C/I state is still live, those writes can be affected by
        // stale address translation and cacheability attributes. J-Link appears to avoid
        // this by forcing a simpler pre-download CPU state. Keep the cleanup here so
        // startup.rs can stay generic.
        {
            let dtrtx_addr = Dbgdtrtx::get_mmio_address_from_base(debug_base)?;
            let read_sctlr = build_mrc(15, 0, 0, 1, 0, 0);
            let write_sctlr = build_mcr(15, 0, 0, 1, 0, 0);
            let emit_r0 = build_mcr(14, 0, 0, 0, 5, 0);
            let dsb = build_mcr(15, 0, 0, 7, 10, 4);
            let isb = build_mcr(15, 0, 0, 7, 5, 4);
            let iciallu = build_mcr(15, 0, 0, 7, 5, 0);
            let bpiall = build_mcr(15, 0, 0, 7, 5, 6);

            execute_instruction(core, debug_base, read_sctlr)?;
            execute_instruction(core, debug_base, emit_r0)?;
            let mut sctlr = core.read_word_32(dtrtx_addr)?;
            tracing::debug!(
                "RZA1L reset_catch_clear: SCTLR before pre-download clamp = {:#010x}",
                sctlr
            );

            sctlr &= !((1 << 12) | (1 << 11) | (1 << 2) | (1 << 0));

            set_instruction_input(core, debug_base, sctlr)?;
            execute_instruction(core, debug_base, build_mrc(14, 0, 0, 0, 5, 0))?;
            execute_instruction(core, debug_base, write_sctlr)?;
            execute_instruction(core, debug_base, dsb)?;
            execute_instruction(core, debug_base, isb)?;
            execute_instruction(core, debug_base, iciallu)?;
            execute_instruction(core, debug_base, bpiall)?;
            execute_instruction(core, debug_base, dsb)?;
            execute_instruction(core, debug_base, isb)?;
        }

        // Step 4b already wrote DBGVCR = 0, which has R = 0.  No further
        // read-modify-write is needed to clear the reset vector catch bit.

        Ok(())
    }

    /// Prepare the RZ/A1L core to run a RAM-loaded image.
    ///
    /// The default implementation only writes CPSR and PC to the register cache, which is
    /// sufficient for most Cortex-A targets. RZ/A1L needs two extra steps:
    ///
    /// 1. **Disable the GIC** (Distributor + CPU Interface) via AHB-AP before releasing
    ///    the CPU. A warm reset (DBGPRCR.CWRR) resets the CPU core but NOT the GIC
    ///    peripheral. If the previously-running firmware left a FIQ source active in the GIC,
    ///    and the stale VBAR still points to old firmware vectors, the FIQ would fire
    ///    immediately on restart — before our startup can execute `cpsid aif` or set VBAR.
    ///    This is compounded by NMFI=1 on RZ/A1L: CPSR.F cannot be set to 1 by software,
    ///    so the interrupt mask in CPSR has no effect on FIQs.
    ///
    /// 2. Set CPSR and PC in the register cache exactly as the default does.
    ///
    /// Our startup code reinitialises the GIC from scratch, so it is safe to disable it
    /// here temporarily.
    fn prepare_running_on_ram(
        &self,
        session: &mut crate::Session,
        vector_table_addr: u64,
        core_id: usize,
    ) -> Result<(), crate::Error> {
        tracing::info!("RZA1L: preparing RAM image start");

        let mut core = session.core(core_id)?;

        // Set CPSR = SYS mode (0x1F), ARM32, A/I masked.
        // CPSR.F (FIQ mask, bit 6) cannot be set to 1 on RZ/A1L because NMFI=1
        // (Non-Maskable FIQ). The write is attempted anyway; if the hardware ignores
        // the F bit that is expected and handled.
        const CPSR_SYS_ARM: u32 = 0x0000_01DF;
        tracing::debug!(
            "RZA1L prepare_running_on_ram: caching CPSR={:#010x} PC={:#010x}",
            CPSR_SYS_ARM,
            vector_table_addr
        );
        core.write_core_reg(XPSR.id, CPSR_SYS_ARM)?;
        core.write_core_reg(PC.id, vector_table_addr)?;

        // Disable the GIC Distributor (GICD_CTLR = 0) so it forwards no interrupts.
        // Address: 0xE820_1000 + 0x000 = 0xE820_1000 (RZ/A1 TRM Table 43-2).
        tracing::debug!("RZA1L prepare_running_on_ram: disabling GIC Distributor (GICD_CTLR)");
        core.write_32(0xE820_1000, &[0u32])?;

        // Disable the GIC CPU Interface (GICC_CTLR = 0) so no FIQ/IRQ signal
        // reaches CPU0 even if the distributor has a pending interrupt.
        // Address: 0xE820_2000 + 0x000 = 0xE820_2000 (RZ/A1 TRM Table 43-3).
        tracing::debug!("RZA1L prepare_running_on_ram: disabling GIC CPU Interface (GICC_CTLR)");
        core.write_32(0xE820_2000, &[0u32])?;

        // Re-check the L2 cache state immediately before run().
        // The L2C invalidation in reset_catch_clear appears to be ineffective by the
        // time writeback_registers runs. Re-do the disable+invalidation here, as close
        // as possible to the actual CPU restart (MOV PC in writeback_registers).
        //
        // Note: DMAC was already stopped in reset_catch_clear (before DCCISW).
        {
            const L2C_CTRL_ADDR: u64  = 0x3FFF_F100; // REG1_CONTROL
            const L2C_INV_ADDR: u64   = 0x3FFF_F77C; // REG7_INV_WAY
            const L2C_ALL_WAYS: u32   = 0xFF;

            // Re-disable and re-invalidate the L2 cache.
            core.write_32(L2C_CTRL_ADDR, &[0u32])?;         // disable L2
            core.write_32(L2C_INV_ADDR, &[L2C_ALL_WAYS])?;  // start invalidation

            let inv_start = std::time::Instant::now();
            loop {
                let mut ways_buf = [0u32; 1];
                if core.read_32(L2C_INV_ADDR, &mut ways_buf).is_ok()
                    && ways_buf[0] == 0
                {
                    break;
                }
                if inv_start.elapsed() > std::time::Duration::from_millis(500) {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Clear RZA1L-specific stale CPU state before releasing the core.
    ///
    /// Called by `writeback_registers` immediately before `MOV PC, r0`.
    ///
    /// Performs work that is specific to the RZA1L scenario where the previously-running
    /// firmware was halted in place:
    ///
    /// 1. **Clears SCTLR bits** that the previous firmware may have left enabled:
    ///    - Bit 0 (M): MMU — the previous firmware's page table is invalid for new firmware.
    ///    - Bit 2 (C): D-cache — stale cache lines could corrupt new firmware data.
    ///    - Bit 11 (Z): branch predictor — stale predictions from old code.
    ///    - Bit 12 (I): I-cache — stale instruction lines from old code.
    ///    - Bit 30 (TE): Thumb exceptions — may have been set if previous firmware used Thumb handlers; new firmware uses ARM.
    ///
    /// 2. **Resets banked exception-mode registers** (FIQ/IRQ/SVC/ABT/UNDEF):
    ///    LR_<mode> = pc_value (safe return address), SPSR_<mode> = SYS ARM mode.
    ///    This prevents stale LR/SPSR from causing a Prefetch Abort when the
    ///    new firmware's exception handlers execute `subs pc, lr, #n`.
    fn pre_run_writeback(
        &self,
        interface: &mut dyn ArmMemoryInterface,
        debug_base: u64,
        pc_value: u32,
    ) -> Result<(), ArmError> {
        let load_r0  = build_mrc(14, 0, 0, 0, 5, 0);
        let read_sctlr  = build_mrc(15, 0, 0, 1, 0, 0);
        let write_sctlr = build_mcr(15, 0, 0, 1, 0, 0);
        let dsb_mcr  = build_mcr(15, 0, 0, 7, 10, 4);
        let isb_mcr  = build_mcr(15, 0, 0, 7, 5,  4);
        let iciallu  = build_mcr(15, 0, 0, 7, 5,  0);
        let bpiall   = build_mcr(15, 0, 0, 7, 5,  6);

        // Step 1: clear additional SCTLR bits the previous firmware may have left set.
        // MRC p15, 0, r0, c1, c0, 0  (read SCTLR → r0)
        execute_instruction(interface, debug_base, read_sctlr)?;
        // BIC r0, r0, #0x1000  (I-cache, bit 12)
        execute_instruction(interface, debug_base, 0xE3C0_0A01)?;
        // BIC r0, r0, #0x800   (Z / branch predictor, bit 11)
        // imm12 = {rot=0xC, imm8=0x08} → 0x08 ROR 24 = 0x800
        execute_instruction(interface, debug_base, 0xE3C0_0C08)?;
        // BIC r0, r0, #0x5     (D-cache bit 2, MMU bit 0)
        execute_instruction(interface, debug_base, 0xE3C0_0005)?;
        // BIC r0, r0, #0x40000000  (TE = Thumb-exception enable, bit 30)
        // The previous firmware may have used Thumb exception handlers (TE=1).
        // New firmware has an ARM vector table; TE must be clear before restart.
        // imm12 = {rot=1, imm8=0x01} → 0x01 ROR 2 = 0x4000_0000
        execute_instruction(interface, debug_base, 0xE3C0_0101)?;
        // MCR p15, 0, r0, c1, c0, 0  (write SCTLR ← r0)
        execute_instruction(interface, debug_base, write_sctlr)?;
        execute_instruction(interface, debug_base, dsb_mcr)?;
        execute_instruction(interface, debug_base, isb_mcr)?;
        execute_instruction(interface, debug_base, iciallu)?;
        execute_instruction(interface, debug_base, bpiall)?;

        // Step 2: reset banked exception-mode LR and SPSR registers.
        //
        // Banked LR/SPSR registers survive across a halt. If a new firmware
        // exception fires and returns via `subs pc, lr, #4`, a stale LR would branch
        // into an arbitrary address, causing a Prefetch Abort.
        //
        // Set LR_<mode> = pc_value (firmware start, a safe "return" address) and
        // SPSR_<mode> = SYS mode ARM (0x0000_019F: SYS|A|I, F=0 per NMFI) so any
        // exception return lands in SYS mode, ARM state, with no further surprises.
        //
        // ARM32 instruction encodings (cond=AL = 0xE):
        //   MSR CPSR_c, #imm8  → 0xE321_F0xx  (change processor mode)
        //   MOV LR, R0         → 0xE1A0_E000
        //   MSR SPSR_fsxc, R0  → 0xE16F_F000  (all four mask bits = fsxc)
        {
            const MSR_FIQ:  u32 = 0xE321_F0D1; // FIQ  mode (10001) + I+A masked
            const MSR_IRQ:  u32 = 0xE321_F0D2; // IRQ  mode (10010)
            const MSR_SVC:  u32 = 0xE321_F0D3; // SVC  mode (10011)
            const MSR_ABT:  u32 = 0xE321_F0D7; // ABT  mode (10111)
            const MSR_UNDEF: u32 = 0xE321_F0DB; // UNDEF mode (11011)
            const MSR_SYS:  u32 = 0xE321_F0DF; // SYS  mode (11111)
            const MOV_LR_R0: u32 = 0xE1A0_E000;
            // MSR SPSR_fsxc, R0: write all SPSR fields from r0
            const MSR_SPSR_R0: u32 = 0xE16F_F000;
            // SYS mode, ARM state, A=1, I=1, F=0 (F cannot be forced on NMFI hardware)
            const SAFE_SPSR: u32 = 0x0000_019F;

            for &mode_instr in &[MSR_FIQ, MSR_IRQ, MSR_SVC, MSR_ABT, MSR_UNDEF] {
                execute_instruction(interface, debug_base, mode_instr)?;
                // LR_<mode> = pc_value
                set_instruction_input(interface, debug_base, pc_value)?;
                execute_instruction(interface, debug_base, load_r0)?;   // r0 = pc_value
                execute_instruction(interface, debug_base, MOV_LR_R0)?; // LR = r0
                // SPSR_<mode> = SYS ARM mode
                set_instruction_input(interface, debug_base, SAFE_SPSR)?;
                execute_instruction(interface, debug_base, load_r0)?;   // r0 = SAFE_SPSR
                execute_instruction(interface, debug_base, MSR_SPSR_R0)?;
            }
            // Return to SYS mode before the caller issues MOV PC.
            execute_instruction(interface, debug_base, MSR_SYS)?;
        }

        Ok(())
    }

    /// Configure the RZ/A1L CoreSight trace infrastructure.
    ///
    /// For `TraceSink::TraceMemory`: puts the CPU-ETF into circular buffer mode so
    /// trace capture continues without stalling the CPU when the buffer fills.
    /// The PTM source is enabled by the generic `setup_tracing()` in `component/mod.rs`.
    ///
    /// The default `trace_start()` would open all funnel ports, which is also correct here —
    /// this override adds the ETF mode configuration on top.
    fn trace_start(
        &self,
        interface: &mut dyn ArmDebugInterface,
        components: &[CoresightComponent],
        sink: &TraceSink,
    ) -> Result<(), ArmError> {
        // Enable the CPU trace funnel (port 0 connects the PTM).
        for trace_funnel in components
            .iter()
            .filter_map(|c| c.find_component(PeripheralType::TraceFunnel))
        {
            let mut funnel = TraceFunnel::new(interface, trace_funnel);
            funnel.unlock()?;
            funnel.enable_port(0x01)?; // port 0 only — PTM is the only source
        }

        if let TraceSink::TraceMemory(_) = sink {
            // Override: put the ETF into circular buffer mode.
            // The generic setup_tracing() will also call enable_capture(), but it sets
            // Software mode (stall-on-full). We change it to Circular here first so the
            // subsequent enable_capture() call latches the correct mode.
            //
            // Circular mode: when the 4KB buffer fills, oldest trace data is overwritten.
            // This ensures the ETF always holds the most recent ~4KB of trace.
            if let Some(tmc_component) = components
                .iter()
                .find_map(|c| c.find_component(PeripheralType::Tmc))
            {
                let mut tmc = TraceMemoryController::new(interface, tmc_component);
                tmc.disable_capture()
                    .map_err(|e| ArmError::Other(format!("ETF disable_capture failed: {e}")))?;
                // set_mode can only be called while capture is disabled
                tmc.set_mode(TmcMode::Circular)
                    .map_err(|e| ArmError::Other(format!("ETF set_mode failed: {e}")))?;
                // Don't enable_capture here — setup_tracing() does that
            }
        }

        Ok(())
    }
}
