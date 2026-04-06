//! Types and functions for interacting with CoreSight Components

use std::time::{Duration, Instant};

mod dwt;
mod itm;
mod ptm;
pub mod ptm_decoder;
mod pmu;
mod scs;
mod swo;
mod tmc;
mod tpiu;
mod trace_funnel;

use crate::{
    Core, Error, MemoryInterface, MemoryMappedRegister,
    architecture::arm::{
        ArmDebugInterface, ArmError, SwoConfig, SwoMode,
        core::armv6m::Demcr,
        dp::DpAddress,
        memory::romtable::{CoresightComponent, PeripheralType, RomTableError},
    },
};

pub use self::itm::Itm;
pub use dwt::Dwt;
pub use pmu::{PerformanceMonitoringUnit, PmuEvent, PmuSnapshot};
pub use ptm::ProgramTraceMacrocell;
pub use scs::Scs;
pub use swo::Swo;
pub use tmc::{Mode as TmcMode, TraceMemoryController};
pub use tpiu::Tpiu;
pub use trace_funnel::TraceFunnel;

use super::memory::Component;

/// Specifies the data sink (destination) for trace data.
#[derive(Debug, Copy, Clone)]
pub enum TraceSink {
    /// Trace data should be sent to the SWO peripheral.
    ///
    /// # Note
    /// On some architectures, there is no distinction between SWO and TPIU.
    Swo(SwoConfig),

    /// Trace data should be sent to the TPIU peripheral.
    Tpiu(SwoConfig),

    /// Trace data should be sent to the embedded trace buffer for software-based trace collection.
    TraceMemory(TraceMemoryConfig),
}

/// PTM/ETF trace-memory configuration.
#[derive(Debug, Copy, Clone, Default)]
pub struct TraceMemoryConfig {
    /// Enable PTM timestamp packets when the trace source advertises support.
    pub timestamps: bool,
    /// Enable PTM return-stack packets when the trace source advertises support.
    pub return_stack: bool,
    /// Enable PTM branch-broadcast mode (ETMCR bit[8]).
    ///
    /// In branch-broadcast mode the PTM emits a branch address packet for every executed
    /// branch, including direct (compile-time-known) branches.  This allows offline
    /// execution reconstruction without an ELF, at the cost of higher trace bandwidth.
    pub branch_broadcast: bool,
}

/// Reports which optional PTM features were actually activated during trace setup.
///
/// Returned by [`setup_tracing`] so callers can emit clear diagnostics when a requested feature
/// was not advertised by the connected trace source.
#[derive(Debug, Copy, Clone, Default)]
pub struct TraceEnabledFeatures {
    /// True if timestamp packets will be emitted by the PTM.
    pub timestamps: bool,
    /// True if the PTM return-stack was enabled.
    pub return_stack: bool,
    /// True if branch-broadcast mode was enabled.
    pub branch_broadcast: bool,
}

/// An error when operating a core ROM table component occurred.
#[derive(thiserror::Error, Debug)]
pub enum ComponentError {
    /// Nordic chips do not support setting all TPIU clocks. Try choosing another clock speed.
    #[error("Nordic does not support TPIU CLK value of {0}")]
    NordicUnsupportedTPUICLKValue(u32),
}

/// A trait to be implemented on memory mapped register types for debug component interfaces.
pub trait DebugComponentInterface:
    MemoryMappedRegister<u32> + Clone + From<u32> + Into<u32> + Sized + std::fmt::Debug
{
    /// Loads the register value from the given debug component via the given core.
    fn load(
        component: &CoresightComponent,
        interface: &mut dyn ArmDebugInterface,
    ) -> Result<Self, ArmError> {
        Ok(Self::from(
            component.read_reg(interface, Self::ADDRESS_OFFSET as u32)?,
        ))
    }

    /// Loads the register value from the given component in given unit via the given core.
    fn load_unit(
        component: &CoresightComponent,
        interface: &mut dyn ArmDebugInterface,
        unit: usize,
    ) -> Result<Self, ArmError> {
        Ok(Self::from(component.read_reg(
            interface,
            Self::ADDRESS_OFFSET as u32 + 16 * unit as u32,
        )?))
    }

    /// Stores the register value to the given debug component via the given core.
    fn store(
        &self,
        component: &CoresightComponent,
        interface: &mut dyn ArmDebugInterface,
    ) -> Result<(), ArmError> {
        component.write_reg(interface, Self::ADDRESS_OFFSET as u32, self.clone().into())
    }

    /// Stores the register value to the given component in given unit via the given core.
    fn store_unit(
        &self,
        component: &CoresightComponent,
        interface: &mut dyn ArmDebugInterface,
        unit: usize,
    ) -> Result<(), ArmError> {
        component.write_reg(
            interface,
            Self::ADDRESS_OFFSET as u32 + 16 * unit as u32,
            self.clone().into(),
        )
    }
}

fn wait_for_tmc_ready(tmc: &mut TraceMemoryController<'_>, context: &str) -> Result<(), ArmError> {
    const TMC_READY_TIMEOUT: Duration = Duration::from_millis(500);

    let start = Instant::now();
    while !tmc.ready().map_err(|e| ArmError::Other(e.to_string()))? {
        if start.elapsed() >= TMC_READY_TIMEOUT {
            let empty = tmc.empty().map_err(|e| ArmError::Other(e.to_string()))?;
            let full = tmc.full().map_err(|e| ArmError::Other(e.to_string()))?;
            let triggered = tmc.triggered().map_err(|e| ArmError::Other(e.to_string()))?;
            let fill_level = tmc
                .fill_level()
                .map_err(|e| ArmError::Other(e.to_string()))?;

            return Err(ArmError::Other(format!(
                "Timed out waiting for ETF ready during {context} after {:?} (empty={empty}, full={full}, triggered={triggered}, fill_level={fill_level})",
                start.elapsed()
            )));
        }
    }

    Ok(())
}

fn tmc_words_available_before_stop(
    tmc: &mut TraceMemoryController<'_>,
) -> Result<usize, ArmError> {
    let fifo_size = tmc.fifo_size()? as usize;
    let fill_level = tmc.fill_level().map_err(|e| ArmError::Other(e.to_string()))? as usize;
    let read_pointer = tmc.read_pointer()? as usize;
    let write_pointer = tmc.write_pointer()? as usize;
    let full_buffer = tmc.full().map_err(|e| ArmError::Other(e.to_string()))?
        || tmc.triggered().map_err(|e| ArmError::Other(e.to_string()))?;

    let pointer_distance = if fifo_size == 0 {
        0
    } else if write_pointer >= read_pointer {
        write_pointer - read_pointer
    } else {
        fifo_size - ((read_pointer - write_pointer) % fifo_size)
    };

    let bytes_to_read = if full_buffer {
        fifo_size
    } else {
        fill_level.max(pointer_distance).min(fifo_size)
    };

    Ok(bytes_to_read / core::mem::size_of::<u32>())
}

/// Reads all the available ARM CoresightComponents of the currently attached target.
///
/// This will recursively parse the Romtable of the attached target
/// and create a list of all the contained components.
pub fn get_arm_components(
    interface: &mut dyn ArmDebugInterface,
    dp: DpAddress,
) -> Result<Vec<CoresightComponent>, ArmError> {
    let mut components = Vec::new();

    for ap_index in interface.access_ports(dp)? {
        let component = if let Ok(mut memory) = interface.memory_interface(&ap_index) {
            if let Ok(addr) = memory.base_address() {
                match addr {
                    0 => Err(Error::Other("AP has a base address of 0".to_string())),
                    debug_base_address => {
                        let component = Component::try_parse(&mut *memory, debug_base_address)?;
                        Ok(CoresightComponent::new(component, ap_index.clone()))
                    }
                }
            } else {
                // If the base address is not present then continue to the next entry.
                continue;
            }
        } else {
            // Return an error, only possible to get Component from MemoryAP
            Err(Error::Other(format!(
                "AP {:#x?} is not a MemoryAP, unable to get ARM component.",
                ap_index.clone()
            )))
        };

        match component {
            Ok(component) => {
                components.push(component);
            }
            Err(e) => {
                tracing::info!("Not counting AP {} because of: {}", ap_index.ap_v1()?, e);
            }
        }
    }

    Ok(components)
}

/// Goes through every component in the vector and tries to find the first component with the given type
pub fn find_component(
    components: &[CoresightComponent],
    peripheral_type: PeripheralType,
) -> Result<&CoresightComponent, ArmError> {
    let component = components
        .iter()
        .find_map(|component| component.find_component(peripheral_type))
        .ok_or_else(|| RomTableError::ComponentNotFound(peripheral_type))?;

    Ok(component)
}

/// Configure the Trace Port Interface Unit
///
/// # Note
/// This configures the TPIU in serial wire mode.
///
/// # Args
/// * `interface` - The interface with the probe.
/// * `component` - The TPIU CoreSight component found.
/// * `config` - The SWO pin configuration to use.
fn configure_tpiu(
    interface: &mut dyn ArmDebugInterface,
    component: &CoresightComponent,
    config: &SwoConfig,
) -> Result<(), Error> {
    let mut tpiu = Tpiu::new(interface, component);

    tpiu.set_port_size(1)?;
    let prescaler = (config.tpiu_clk() / config.baud()) - 1;
    tpiu.set_prescaler(prescaler)?;
    match config.mode() {
        SwoMode::Manchester => tpiu.set_pin_protocol(1)?,
        SwoMode::Uart => tpiu.set_pin_protocol(2)?,
    }

    // Formatter: TrigIn enabled, bypass optional
    if config.tpiu_continuous_formatting() {
        // Set EnFCont for continuous formatting even over SWO.
        tpiu.set_formatter(0x102)?;
    } else {
        // Clear EnFCont to only pass through raw ITM/DWT data.
        tpiu.set_formatter(0x100)?;
    }

    Ok(())
}

/// Sets up all the SWV components.
///
/// Expects to be given a list of all ROM table `components` as the second argument.
///
/// Returns [`TraceEnabledFeatures`] describing which optional PTM features were actually
/// activated.  For non-`TraceMemory` sinks the returned struct will always be all-false.
pub(crate) fn setup_tracing(
    interface: &mut dyn ArmDebugInterface,
    components: &[CoresightComponent],
    sink: &TraceSink,
) -> Result<TraceEnabledFeatures, Error> {
    let mut enabled = TraceEnabledFeatures::default();
    // Configure DWT/ITM when present. Cortex-A/R trace paths may provide PTM + funnel +
    // ETF/TPIU without M-profile DWT/ITM blocks.
    if let Ok(dwt_component) = find_component(components, PeripheralType::Dwt) {
        let mut dwt = Dwt::new(interface, dwt_component);
        dwt.enable()?;
        dwt.enable_exception_trace()?;
    }

    if let Ok(itm_component) = find_component(components, PeripheralType::Itm) {
        let mut itm = Itm::new(interface, itm_component);
        itm.unlock()?;
        itm.tx_enable()?;
    }

    // Configure the trace destination.
    match sink {
        TraceSink::Tpiu(config) => {
            configure_tpiu(
                interface,
                find_component(components, PeripheralType::Tpiu)?,
                config,
            )?;
        }

        TraceSink::Swo(config) => {
            if let Ok(peripheral) = find_component(components, PeripheralType::Swo) {
                let mut swo = Swo::new(interface, peripheral);
                swo.unlock()?;

                let prescaler = (config.tpiu_clk() / config.baud()) - 1;
                swo.set_prescaler(prescaler)?;

                match config.mode() {
                    SwoMode::Manchester => swo.set_pin_protocol(1)?,
                    SwoMode::Uart => swo.set_pin_protocol(2)?,
                }
            } else {
                // For Cortex-M4, the SWO and the TPIU are combined. If we don't find a SWO
                // peripheral, use the TPIU instead.
                configure_tpiu(
                    interface,
                    find_component(components, PeripheralType::Tpiu)?,
                    config,
                )?;
            }
        }

        TraceSink::TraceMemory(config) => {
            let mut tmc = TraceMemoryController::new(
                interface,
                find_component(components, PeripheralType::Tmc)?,
            );

            // Clear out the TMC FIFO and restart capture. The mode (Software or Circular) was
            // already set by trace_start() — either the default (Software) or a device-specific
            // override (e.g. Circular for RZA1L). Changing the mode here would override that.
            tmc.disable_capture()?;
            wait_for_tmc_ready(&mut tmc, "trace setup")?;
            tmc.enable_formatter()?;

            tmc.enable_capture()?;

            // Enable any PTM (Program Trace Macrocell) sources found in the ROM table.
            // PTMs are present on ARMv7-A/R cores (Cortex-A5/A7/A8/A9/A15, Cortex-R4/R5).
            // Trace IDs are assigned starting at 1; each PTM on the ATB bus needs a unique ID.
            for (idx, ptm_component) in components
                .iter()
                .filter_map(|c| c.find_component(PeripheralType::Ptm))
                .enumerate()
            {
                let trace_id = (idx + 1) as u8;
                let mut ptm = ProgramTraceMacrocell::new(interface, ptm_component);
                // OR-accumulate: a feature is reported as enabled if any PTM on this ATB bus
                // activates it (all PTMs are configured identically so this is typically all-or-nothing).
                let ptm_features = ptm.enable(trace_id, *config)?;
                enabled.timestamps |= ptm_features.timestamps;
                enabled.return_stack |= ptm_features.return_stack;
            }
        }
    }

    Ok(enabled)
}

/// Read trace data from internal trace memory
///
/// # Args
/// * `interface` - The interface with the debug probe.
/// * `components` - The CoreSight debug components identified in the system.
///
/// # Note
/// This function will read any available trace data in trace memory without blocking. At most,
/// this function will read as much data as can fit in the FIFO - if the FIFO continues to be
/// filled while trace data is being extracted, this function can be called again to return that
/// data.
///
/// # Returns
/// All data stored in trace memory, with an upper bound at the size of internal trace memory.
pub(crate) fn read_trace_memory(
    interface: &mut dyn ArmDebugInterface,
    components: &[CoresightComponent],
) -> Result<Vec<u8>, ArmError> {
    let mut tmc =
        TraceMemoryController::new(interface, find_component(components, PeripheralType::Tmc)?);

    let words_to_read = tmc_words_available_before_stop(&mut tmc)?;

    // Stop capture before draining. Without this, data could be written while being read,
    // causing corrupted frames — especially critical in Circular mode.
    tmc.disable_capture().map_err(|e| ArmError::Other(e.to_string()))?;
    wait_for_tmc_ready(&mut tmc, "trace memory drain")?;

    // Drain the ETF via the RRD register.
    //
    // In software FIFO mode the RRD register eventually returns the 0xFFFF_FFFF sentinel.
    // In circular-buffer ETF mode on RZ/A1L, relying on that sentinel can hang indefinitely,
    // so we bound the read count from the pre-stop fill level or full buffer size.
    let mut etf_trace: Vec<u8> = Vec::new();
    for _ in 0..words_to_read {
        match tmc.read()? {
            Some(data) => etf_trace.extend_from_slice(&data.to_le_bytes()),
            None => break,
        }
    }

    // The TMC formats data into frames, each 16 bytes, from multiple trace sources.
    // Extract only ITM data (ATID 13, set by Itm::tx_enable()).
    let mut id = 0.into();
    let mut itm_trace = Vec::new();

    for frame_buffer in etf_trace.chunks_exact(16) {
        let mut frame = tmc::Frame::new(frame_buffer, id);
        for (fid, data) in &mut frame {
            match fid.into() {
                // ITM ATID, see Itm::tx_enable()
                13u8 => itm_trace.push(data),
                0u8 => (),
                other => tracing::warn!("Unexpected trace source ATID {other}: {data:#04x}, ignoring"),
            }
        }
        id = frame.id();
    }

    Ok(itm_trace)
}

/// Read PTM instruction trace data from the ETF circular buffer.
///
/// Stops ETF capture, drains the buffer, demultiplexes frames filtering to the specified PTM
/// trace ID, then re-enables capture so tracing continues (the circular buffer will resume
/// overwriting oldest data).
///
/// # Args
/// * `interface` - The debug probe interface.
/// * `components` - CoreSight components from the ROM table.
/// * `trace_id` - The ATB trace source ID assigned to the PTM (set by [`ProgramTraceMacrocell::enable`]).
///   Typically `1` for the first (and only) PTM on a Cortex-A9.
pub(crate) fn read_ptm_trace_memory(
    interface: &mut dyn ArmDebugInterface,
    components: &[CoresightComponent],
    trace_id: u8,
) -> Result<Vec<u8>, ArmError> {
    let mut tmc =
        TraceMemoryController::new(interface, find_component(components, PeripheralType::Tmc)?);

    let words_to_read = tmc_words_available_before_stop(&mut tmc)?;

    // Stop capture; wait until ETF pipelines are fully drained.
    tmc.disable_capture().map_err(|e| ArmError::Other(e.to_string()))?;
    wait_for_tmc_ready(&mut tmc, "PTM trace drain")?;

    // Drain via RRD using a bounded word count.
    let mut raw: Vec<u8> = Vec::new();
    for _ in 0..words_to_read {
        match tmc.read()? {
            Some(word) => raw.extend_from_slice(&word.to_le_bytes()),
            None => break,
        }
    }

    // Re-enable capture so circular tracing resumes immediately.
    tmc.enable_capture().map_err(|e| ArmError::Other(e.to_string()))?;

    Ok(choose_best_ptm_bytes(&raw, trace_id))
}

fn choose_best_ptm_bytes(raw: &[u8], trace_id: u8) -> Vec<u8> {
    // In circular mode the oldest byte can land mid-frame. Try all formatter frame offsets and a
    // small set of initial IDs, then keep the candidate that yields the most decodable PTM data.
    let mut best_bytes = Vec::new();
    let mut best_score = 0usize;

    for offset in 0..16 {
        if raw.len() <= offset {
            continue;
        }

        for initial_id in [0u8, trace_id] {
            let ptm_bytes = extract_ptm_bytes(&raw[offset..], trace_id, initial_id);
            let score = ptm_decoder::Decoder::new(&ptm_bytes, 0).count();

            if score > best_score || (score == best_score && ptm_bytes.len() > best_bytes.len()) {
                best_score = score;
                best_bytes = ptm_bytes;
            }
        }
    }

    best_bytes
}

fn extract_ptm_bytes(raw: &[u8], trace_id: u8, initial_id: u8) -> Vec<u8> {
    let mut id = initial_id.into();
    let mut ptm_bytes = Vec::new();

    for chunk in raw.chunks_exact(16) {
        let mut frame = tmc::Frame::new(chunk, id);
        for (fid, data) in &mut frame {
            let atid: u8 = fid.into();
            if atid == trace_id {
                ptm_bytes.push(data);
            }
        }
        id = frame.id();
    }

    ptm_bytes
}

/// Configure the PMU to count the given events and enable it.
///
/// Call this while the core is halted.  After returning, resume the core and
/// let it run.  Later, halt the core again and call [`snapshot_pmu`] to read
/// the accumulated counts.
pub(crate) fn configure_pmu(
    interface: &mut dyn ArmDebugInterface,
    components: &[CoresightComponent],
    events: &[PmuEvent],
) -> Result<(), ArmError> {
    let mut pmu =
        PerformanceMonitoringUnit::new(interface, find_component(components, PeripheralType::Pmu)?);
    pmu.configure(events)
}

/// Read a snapshot of PMU counter values.
///
/// Call this while the core is halted (typically after `configure_pmu` + run).
pub(crate) fn snapshot_pmu(
    interface: &mut dyn ArmDebugInterface,
    components: &[CoresightComponent],
    events: &[PmuEvent],
) -> Result<PmuSnapshot, ArmError> {
    let mut pmu =
        PerformanceMonitoringUnit::new(interface, find_component(components, PeripheralType::Pmu)?);
    pmu.read_results(events)
}

/// Configures DWT trace unit `unit` to begin tracing `address`.
///
///
/// Expects to be given a list of all ROM table `components` as the second argument.
pub(crate) fn add_swv_data_trace(
    interface: &mut dyn ArmDebugInterface,
    components: &[CoresightComponent],
    unit: usize,
    address: u32,
) -> Result<(), ArmError> {
    let mut dwt = Dwt::new(interface, find_component(components, PeripheralType::Dwt)?);
    dwt.enable_data_trace(unit, address)
}

/// Configures DWT trace unit `unit` to stop tracing `address`.
///
///
/// Expects to be given a list of all ROM table `components` as the second argument.
pub fn remove_swv_data_trace(
    interface: &mut dyn ArmDebugInterface,
    components: &[CoresightComponent],
    unit: usize,
) -> Result<(), ArmError> {
    let mut dwt = Dwt::new(interface, find_component(components, PeripheralType::Dwt)?);
    dwt.disable_data_trace(unit)
}

/// Sets TRCENA in DEMCR to begin trace generation.
pub fn enable_tracing(core: &mut Core) -> Result<(), Error> {
    let mut demcr = Demcr(core.read_word_32(Demcr::get_mmio_address())?);
    demcr.set_dwtena(true);
    core.write_word_32(Demcr::get_mmio_address(), demcr.into())?;
    Ok(())
}

/// Disables TRCENA in DEMCR to disable trace generation.
pub fn disable_swv(core: &mut Core) -> Result<(), Error> {
    let mut demcr = Demcr(core.read_word_32(Demcr::get_mmio_address())?);
    demcr.set_dwtena(false);
    core.write_word_32(Demcr::get_mmio_address(), demcr.into())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_single_id_frame(trace_id: u8, payload: &[u8]) -> [u8; 16] {
        assert!(payload.len() <= 14);

        let mut frame = [0u8; 16];
        frame[0] = (trace_id << 1) | 1;
        if let Some(&first_byte) = payload.first() {
            frame[1] = first_byte;
        }

        for (idx, &byte) in payload.iter().enumerate().skip(1) {
            let frame_idx = idx + 1;
            if frame_idx >= 15 {
                break;
            }

            if frame_idx & 1 == 0 {
                frame[frame_idx] = byte & 0xFE;
                if byte & 1 != 0 {
                    frame[15] |= 1 << (frame_idx >> 1);
                }
            } else {
                frame[frame_idx] = byte;
            }
        }

        frame
    }

    #[test]
    fn circular_ptm_extraction_recovers_from_mid_frame_start() {
        let trace_id = 1;
        let mut ptm = vec![0x00; 5];
        ptm.extend([0x80, 0x08, 0x10, 0x00, 0x00, 0x00, 0xD3]);

        let frame = encode_single_id_frame(trace_id, &ptm);
        let mut raw = vec![0xAA, 0x55, 0xCC, 0x33, 0x77];
        raw.extend_from_slice(&frame);

        let extracted = choose_best_ptm_bytes(&raw, trace_id);
        assert!(extracted.starts_with(&ptm));

        let packets: Vec<_> = ptm_decoder::Decoder::new(&extracted, 0).collect();
        assert!(packets.iter().any(|packet| matches!(packet, ptm_decoder::PtmPacket::Sync)));
        assert!(packets.iter().any(|packet| matches!(
            packet,
            ptm_decoder::PtmPacket::ISync {
                pc: 0x10,
                cpsr_low8: 0xD3,
                context_id: 0,
            }
        )));
    }
}
