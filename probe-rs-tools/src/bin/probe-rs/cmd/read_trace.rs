//! `probe-rs read-trace`: drain PTM instruction trace from the ETF circular buffer.
//!
//! Configures the CoreSight trace infrastructure for ETF capture, waits for a specified
//! duration while the target runs, then stops and drains the ETF, writing raw PTM bytes
//! to a file or stdout.
//!
//! Typical usage:
//! ```
//! # dump raw bytes to a file for offline decoding with ptm2human / Trace Compass:
//! probe-rs read-trace --chip R7S721010 --duration-ms 2000 --output trace.bin
//!
//! # decode inline and print packets to stderr:
//! probe-rs read-trace --chip R7S721010 --duration-ms 2000 --decode
//!
//! # decode and symbolize branch/isync targets using the firmware ELF:
//! probe-rs read-trace --chip R7S721010 --duration-ms 2000 --decode \
//!     --elf target/armv7a-none-eabihf/debug/firmware
//!
//! # compact execution-flow view (only ISync + Branch targets, with symbols):
//! probe-rs read-trace --chip R7S721010 --duration-ms 2000 --flow \
//!     --elf target/armv7a-none-eabihf/debug/firmware
//!
//! # JSON Lines output for programmatic processing:
//! probe-rs read-trace --chip R7S721010 --duration-ms 2000 --decode \
//!     --elf target/armv7a-none-eabihf/debug/firmware \
//!     --output-format json 2>&1 | jq 'select(.type == "Branch")'
//! ```

use std::io::Write;
use std::path::PathBuf;
use std::thread::sleep;
use std::time::Duration;

use capstone::{Capstone, Endian, arch::arm::ArchMode as ArmMode, prelude::*};
use probe_rs::CoreStatus;
use probe_rs::architecture::arm::component::{TraceMemoryConfig, TraceSink, ptm_decoder};
use probe_rs::config::Registry;
use probe_rs::probe::list::Lister;

use crate::util::common_options::ProbeOptions;
use crate::CoreOptions;

use crate::util::samply;

/// Thin wrapper around `addr2line::Loader` for address → symbol/source lookups.
struct Symbols {
    loader: addr2line::Loader,
}

impl Symbols {
    fn load(path: &PathBuf) -> anyhow::Result<Self> {
        let loader = addr2line::Loader::new(path)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self { loader })
    }

    /// Resolve an address to a demangled function name, if available.
    fn function_name(&self, addr: u64) -> Option<String> {
        let mut frames = self.loader.find_frames(addr).ok()?;
        frames
            .next().ok().flatten()
            .and_then(|f| f.function)
            .and_then(|n| n.demangle().map(|s| s.into_owned()).ok())
            .or_else(|| self.loader.find_symbol(addr).map(|s| s.to_string()))
    }

    /// Resolve an address to `file:line`, if DWARF info is available.
    fn source_location(&self, addr: u64) -> Option<String> {
        let loc = self.loader.find_location(addr).ok()??;
        let file = loc.file?;
        let line = loc.line?;
        // Try to strip everything up to and including the last "src/" component so
        // that paths render as "src/tasks/audio.rs:42" regardless of workspace root.
        // If no "src/" component is present (e.g. external crates), use the file as-is.
        let display = if let Some(tail) = file.rsplit_once("src/") {
            format!("src/{}", tail.1)
        } else {
            file.to_owned()
        };
        Some(format!("{display}:{line}"))
    }
}

// ─── ELF image / flow reconstruction ─────────────────────────────────────────

/// A single loadable segment from an ELF, held in memory for instruction fetch.
struct ElfSegment {
    vaddr: u32,
    data: Vec<u8>,
}

/// In-memory image of the ELF's executable segments.
///
/// Used by [`FlowReconstructor`] to fetch instruction bytes at any program-counter
/// address without re-reading the file on every instruction decode.
struct ElfImage {
    segments: Vec<ElfSegment>,
}

impl ElfImage {
    fn load(path: &PathBuf) -> anyhow::Result<Self> {
        use object::{Object, ObjectSegment};
        let data = std::fs::read(path)?;
        let elf = object::File::parse(data.as_slice())
            .map_err(|e| anyhow::anyhow!("ELF parse: {e}"))?;
        let mut segments = Vec::new();
        for seg in elf.segments() {
            if let Ok(bytes) = seg.data() {
                if bytes.is_empty() { continue; }
                segments.push(ElfSegment {
                    vaddr: seg.address() as u32,
                    data: bytes.to_vec(),
                });
            }
        }
        Ok(Self { segments })
    }

    /// Read up to `len` bytes starting at `addr`, or `None` if not mapped.
    fn read_bytes(&self, addr: u32, len: usize) -> Option<&[u8]> {
        for seg in &self.segments {
            let start = seg.vaddr;
            let end   = start.wrapping_add(seg.data.len() as u32);
            if addr >= start && addr < end {
                let off = (addr - start) as usize;
                let avail = seg.data.len() - off;
                return Some(&seg.data[off .. off + avail.min(len)]);
            }
        }
        None
    }
}

/// A single decoded execution step produced by [`FlowReconstructor`].
#[derive(Debug)]
pub struct FlowEntry {
    /// Address of the instruction.
    pub address: u32,
    /// Capstone mnemonic + operands.
    pub text: String,
    /// True for the leading entry of a new function (ISync boundary).
    #[allow(dead_code)]
    pub is_sync: bool,
}

/// A single callstack snapshot captured from PTM instruction trace.
///
/// The callstack is ordered bottom-first (index 0 = root / oldest frame,
/// last index = currently executing PC).
#[derive(Debug, Clone)]
pub struct PtmCallstackSample {
    /// Stack frames as raw instruction addresses (not adjusted).
    pub callstack: Vec<u32>,
    /// Monotonic sample index, used as a proxy timestamp.
    pub sample_idx: u32,
}

/// Reconstructs the full instruction-level execution trace from PTM packets.
///
/// ## How it works
///
/// PTM emits three packet types that advance the PC:
///
/// - **ISync**: establishes the initial PC and ISA (T-bit in CPSR).
/// - **Branch**: jumps to a new address (indirect branch, function call, exception).
/// - **Atom (E/N)**: one outcome per conditional direct branch.  
///   `E` = taken → set PC to the branch target extracted from disassembly.  
///   `N` = not-taken → advance PC by instruction size.
///
/// The reconstructor disassembles one instruction at a time from the in-memory ELF
/// image, maintaining the current PC and Thumb/ARM ISA state across packets.
///
/// When `sample_interval > 0`, the reconstructor also maintains a shadow call stack
/// by tracking `BL`/`BLX` calls and returns (`BX LR`, `POP {..,pc}`).  A
/// [`PtmCallstackSample`] is emitted at every ISync boundary and every
/// `sample_interval` instructions.  Accumulated samples can be retrieved with
/// [`FlowReconstructor::take_ptm_samples`].
struct FlowReconstructor {
    cs_thumb: capstone::Capstone,
    cs_arm:   capstone::Capstone,
    pc:       Option<u32>,
    is_thumb: bool,
    /// Shadow call stack: each entry is the return address pushed by a BL/BLX.
    shadow_stack: Vec<u32>,
    /// Accumulated PTM callstack samples.
    ptm_samples: Vec<PtmCallstackSample>,
    /// Monotonic sample counter.
    sample_count: u32,
    /// Emit a sample every this many instructions (0 = ISync-only).
    sample_interval: u32,
    /// Instructions executed since last interval-based sample.
    insn_since_interval: u32,
}

impl FlowReconstructor {
    /// Create a FlowReconstructor that also emits PTM callstack samples.
    ///
    /// `sample_interval` — emit a sample every this many instructions in
    /// addition to the mandatory sample at every ISync boundary.  Pass `0`
    /// to only sample at ISync boundaries.
    fn new_with_sampling(sample_interval: u32) -> anyhow::Result<Self> {
        let cs_thumb = Capstone::new()
            .arm()
            .mode(ArmMode::Thumb)
            .endian(Endian::Little)
            .detail(true)
            .build()
            .map_err(|e| anyhow::anyhow!("capstone (Thumb): {e:?}"))?;
        let cs_arm = Capstone::new()
            .arm()
            .mode(ArmMode::Arm)
            .endian(Endian::Little)
            .detail(true)
            .build()
            .map_err(|e| anyhow::anyhow!("capstone (ARM): {e:?}"))?;
        Ok(Self {
            cs_thumb,
            cs_arm,
            pc: None,
            is_thumb: true,
            shadow_stack: Vec::new(),
            ptm_samples: Vec::new(),
            sample_count: 0,
            sample_interval,
            insn_since_interval: 0,
        })
    }

    /// Drain and return all accumulated PTM callstack samples.
    fn take_ptm_samples(&mut self) -> Vec<PtmCallstackSample> {
        std::mem::take(&mut self.ptm_samples)
    }

    /// Record a callstack snapshot using the current PC and shadow stack.
    fn emit_sample(&mut self) {
        let pc = match self.pc {
            Some(p) => p,
            None    => return, // no anchor yet — skip
        };
        // Build bottom-first: shadow_stack[0] is the deepest return addr.
        // We reverse it so index 0 = oldest caller, last = current PC.
        let mut callstack: Vec<u32> = self.shadow_stack.iter().rev().copied().collect();
        callstack.push(pc);
        self.ptm_samples.push(PtmCallstackSample {
            callstack,
            sample_idx: self.sample_count,
        });
        self.sample_count += 1;
        self.insn_since_interval = 0;
    }

    /// Consume a PTM packet, returning any [`FlowEntry`]s it produced.
    fn consume(&mut self, packet: &ptm_decoder::PtmPacket, image: &ElfImage) -> Vec<FlowEntry> {
        use ptm_decoder::PtmPacket as P;
        match packet {
            P::ISync { pc, .. } => {
                // ISync address bit 0 = T (Thumb state), matching the Branch address convention
                // (IHI0035B §5.6.2). The info byte (cpsr_low8) carries T at bit 4, not bit 5
                // (bit 5 is reason[0]).
                self.is_thumb = pc & 1 == 1;
                // ISync address has bit 0 set when Thumb.
                self.pc = Some(pc & !1);
                // ISync means we lost continuity (exception, coarse sync, etc.).
                // The shadow stack context is no longer valid — clear it and
                // emit a single-frame sample anchored at the new PC.
                self.shadow_stack.clear();
                self.insn_since_interval = 0;
                self.emit_sample();
                if let Some(entry) = self.disasm_one(image, true) {
                    vec![entry]
                } else {
                    vec![]
                }
            }
            P::Branch { address, .. } => {
                // Bit 0 of the branch address encodes the new ISA.
                self.is_thumb = (address & 1) == 1;
                self.pc = Some(address & !1);
                if let Some(entry) = self.disasm_one(image, false) {
                    vec![entry]
                } else {
                    vec![]
                }
            }
            P::Atom { executed, count } => {
                let mut entries = Vec::new();
                for bit in 0..*count {
                    let taken = (executed >> bit) & 1 == 1;
                    if let Some(entry) = self.step_atom(image, taken) {
                        entries.push(entry);
                    } else {
                        break; // lost sync — give up for this atom packet
                    }
                }
                entries
            }
            _ => vec![],
        }
    }

    /// Disassemble the instruction at the current PC and return a [`FlowEntry`].
    fn disasm_one(&mut self, image: &ElfImage, is_sync: bool) -> Option<FlowEntry> {
        let pc = self.pc?;
        let bytes = image.read_bytes(pc, 4)?;
        let cs = if self.is_thumb { &self.cs_thumb } else { &self.cs_arm };
        let insns = cs.disasm_count(bytes, pc as u64, 1).ok()?;
        let insn = insns.first()?;
        let text = format!(
            "{} {}",
            insn.mnemonic().unwrap_or("?"),
            insn.op_str().unwrap_or("")
        );
        Some(FlowEntry { address: pc, text, is_sync })
    }

    /// Advance the PC by one instruction, following the branch outcome.
    ///
    /// Also maintains the shadow call stack when `sample_interval` is set:
    /// - `BL` / `BLX` (calls) push the return address.
    /// - `BX LR` / `POP {…, pc}` / `LDMFD sp!, {…, pc}` (returns) pop.
    ///
    /// Returns the [`FlowEntry`] for the instruction *at* the current PC.
    fn step_atom(&mut self, image: &ElfImage, taken: bool) -> Option<FlowEntry> {
        let pc = self.pc?;
        let bytes = image.read_bytes(pc, 4)?;
        let cs = if self.is_thumb { &self.cs_thumb } else { &self.cs_arm };
        let insns = cs.disasm_count(bytes, pc as u64, 1).ok()?;
        let insn = insns.first()?;
        let insn_size = insn.len() as u32;
        // Collect owned strings before any mutable borrows (borrow-checker).
        let mnemonic = insn.mnemonic().unwrap_or("").to_owned();
        let op_str   = insn.op_str().unwrap_or("").to_owned();
        // Drop `insns` (and therefore the borrow of `cs`/`self`) before calling
        // emit_sample which needs `&mut self`.
        drop(insns);

        // ── Shadow call-stack maintenance ────────────────────────────────────
        let is_call   = mnemonic == "bl" || mnemonic == "blx";
        let is_return = (mnemonic == "bx"  && op_str.contains("lr"))
            || ((mnemonic == "pop" || mnemonic.starts_with("ldm"))
                && op_str.contains("pc"));

        let return_addr = pc.wrapping_add(insn_size);
        if is_call {
            self.shadow_stack.push(return_addr);
        } else if is_return {
            self.shadow_stack.pop();
        }

        // ── PC advancement ───────────────────────────────────────────────────
        if taken {
            if let Some(target) = parse_branch_target(&op_str) {
                self.pc = Some(target);
            } else {
                self.pc = None;
            }
        } else {
            self.pc = Some(return_addr);
        }

        // ── Interval-based sampling ──────────────────────────────────────────
        self.insn_since_interval += 1;
        if self.sample_interval > 0 && self.insn_since_interval >= self.sample_interval {
            self.emit_sample();
        }

        let text = format!("{mnemonic} {op_str}");
        Some(FlowEntry { address: pc, text, is_sync: false })
    }
}

/// Parse a branch target address from a capstone operand string.
///
/// Capstone renders direct branch targets as `#0xNNNN` or `#NNNN`.
fn parse_branch_target(op_str: &str) -> Option<u32> {
    let s = op_str.trim();
    // Strip leading `#` if present.
    let s = if let Some(rest) = s.strip_prefix('#') { rest } else { s };
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).ok()
    } else {
        // Some targets render as plain decimal.
        s.parse::<u32>().ok()
    }
}

// ─── PTM callstack → Firefox Profiler output ────────────────────────────────

/// Serialize accumulated PTM callstack samples into a gzipped Firefox Profiler
/// JSON file that can be opened directly with `samply load <file>`.
///
/// `samples`  — samples collected by [`FlowReconstructor`] (bottom-first).
/// `elf_path` — path to the firmware ELF, used for library metadata.
/// `profile_path` — output `.json.gz` file to create.
fn save_ptm_flamegraph(
    samples: &[PtmCallstackSample],
    elf_path: &std::path::Path,
    profile_path: &std::path::Path,
    clock_mhz: u32,
) -> anyhow::Result<()> {
    use fxprof_processed_profile as fxprofpp;

    let elf_bytes = std::fs::read(elf_path)?;
    let obj = object::File::parse(elf_bytes.as_slice())
        .map_err(|e| anyhow::anyhow!("ELF parse: {e}"))?;

    // Convert clock frequency to a per-instruction sampling interval.
    // The Firefox Profiler timeline is in wall-clock time, so we need to
    // know how long one instruction takes at the given clock speed.
    // 1 / clock_mhz MHz = (1_000_000 / clock_mhz) ns per instruction.
    let nanos_per_sample = if clock_mhz > 0 { 1_000_000 / clock_mhz as u64 } else { 2500 };
    let sampling_interval = std::time::Duration::from_nanos(nanos_per_sample);
    let start_time = std::time::SystemTime::now();

    let abs_path = elf_path
        .canonicalize()
        .unwrap_or_else(|_| elf_path.to_path_buf());
    let abs_str = abs_path.to_str().unwrap_or("").to_owned();
    let name = abs_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("firmware")
        .to_owned();

    let mut profile = fxprofpp::Profile::new(
        &name,
        start_time.into(),
        sampling_interval.into(),
    );
    let category = profile.add_category("ptm", fxprofpp::CategoryColor::Blue);

    let process = profile.add_process(
        "firmware",
        0,
        fxprofpp::Timestamp::from_nanos_since_reference(0),
    );

    // Library metadata lets samply resolve addresses to symbols.
    let debug_id = samply::debug_id_for_object(&obj);
    let start_avma = samply::relative_address_base(&obj);
    if let Some(did) = debug_id {
        let lib = profile.add_lib(fxprofpp::LibraryInfo {
            name: name.clone(),
            debug_name: name.clone(),
            path: abs_str.clone(),
            debug_path: abs_str.clone(),
            debug_id: did,
            code_id: None,
            arch: None,
            symbol_table: None,
        });
        profile.add_lib_mapping(process, lib, start_avma, u64::MAX, 0);
    }

    let thread = profile.add_thread(
        process,
        0,
        fxprofpp::Timestamp::from_nanos_since_reference(0),
        true,
    );

    for sample in samples {
        let t_ns = sample.sample_idx as u64 * sampling_interval.as_nanos() as u64;
        let frames = sample.callstack.iter().map(|&addr| {
            fxprofpp::FrameInfo {
                frame: fxprofpp::Frame::InstructionPointer(addr as u64),
                category_pair: category.into(),
                flags: fxprofpp::FrameFlags::empty(),
            }
        });
        let stack = profile.intern_stack_frames(thread, frames);
        profile.add_sample(
            thread,
            fxprofpp::Timestamp::from_nanos_since_reference(t_ns),
            stack,
            fxprofpp::CpuDelta::ZERO,
            1,
        );
    }

    let out_file = std::fs::File::create(profile_path)?;
    let writer   = std::io::BufWriter::new(out_file);
    let builder  = flate2::GzBuilder::new()
        .filename(profile_path.file_name().unwrap_or_default().as_encoded_bytes());
    let gz = builder.write(writer, flate2::Compression::new(2));
    let gz = std::io::BufWriter::new(gz);
    serde_json::to_writer(gz, &profile)?;
    eprintln!("Wrote PTM flamegraph to {}", profile_path.display());
    Ok(())
}

/// Output format for decoded or flow packet display.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default, PartialEq, Eq)]
enum OutputFormat {
    /// Human-readable text (default).
    #[default]
    Text,
    /// JSON Lines (JSONL): one JSON object per packet, suitable for programmatic processing.
    Json,
}

/// Serialize a [`PtmPacket`] as a JSON object, post-processing the serde output to:
///
/// - Flatten the adjacently-tagged `"data"` wrapper into the top-level object.
/// - Hex-format address and byte fields (`pc`, `address`, `context_id`, `cpsr_low8`, for
///   numeric readability).
/// - Replace `Atom`'s raw `executed`/`count` fields with a human-readable `outcomes` string.
/// - Inject `function` and optionally `source` fields from ELF symbol data.
///
/// Using `serde_json` for the final serialization means symbol names containing `"`, `\`, or
/// any Unicode are always correctly escaped.
fn packet_to_json(
    packet: &ptm_decoder::PtmPacket,
    syms: &Option<Symbols>,
    include_source: bool,
    last_ts: Option<u64>,
) -> String {
    use ptm_decoder::PtmPacket as P;
    use serde_json::{Map, Value};

    // Serialize via serde; result is {"type":"Branch","data":{...}} or {"type":"Sync"}.
    let mut obj: Map<String, Value> = match serde_json::to_value(packet) {
        Ok(Value::Object(m)) => m,
        _ => return String::from("{\"type\":\"SerializeError\"}"),
    };

    // Flatten the "data" wrapper into the top-level map.
    if let Some(Value::Object(data)) = obj.remove("data") {
        obj.extend(data);
    } else if let Some(v) = obj.remove("data") {
        // Newtype variants (ContextId, Timestamp, Unknown) produce a bare value under "data".
        // Assign a meaningful field name based on packet type.
        let field = match obj.get("type").and_then(|t| t.as_str()) {
            Some("ContextId") => "id",
            Some("Timestamp") => "value",
            Some("Unknown")   => "byte",
            _                 => "data",
        };
        obj.insert(field.to_string(), v);
    }

    // Hex-format selected numeric fields for readability.
    for field in ["pc", "address", "context_id"] {
        if let Some(Value::Number(n)) = obj.get(field).cloned() {
            if let Some(i) = n.as_u64() {
                obj.insert(field.to_string(), Value::String(format!("{i:#010x}")));
            }
        }
    }
    for field in ["cpsr_low8", "id", "byte"] {
        if let Some(Value::Number(n)) = obj.get(field).cloned() {
            if let Some(i) = n.as_u64() {
                obj.insert(field.to_string(), Value::String(format!("{i:#04x}")));
            }
        }
    }

    // Atom: replace raw executed/count with a human-readable outcomes string.
    if obj.get("type").and_then(|t| t.as_str()) == Some("Atom") {
        if let (Some(Value::Number(ex)), Some(Value::Number(cnt))) =
            (obj.remove("executed"), obj.get("count").cloned())
        {
            let executed = ex.as_u64().unwrap_or(0) as u8;
            let count    = cnt.as_u64().unwrap_or(0) as u8;
            let outcomes: String = (0..count)
                .map(|i| if (executed >> i) & 1 == 1 { 'E' } else { 'N' })
                .collect();
            obj.insert("outcomes".to_string(), Value::String(outcomes));
        }
    }

    // Inject symbol/source for address-bearing packets — serde_json handles escaping.
    if let Some(ref syms) = *syms {
        let addr = match packet {
            P::ISync { pc, .. }       => Some(*pc as u64),
            P::Branch { address, .. } => Some(*address as u64),
            _ => None,
        };
        if let Some(addr) = addr {
            if let Some(name) = syms.function_name(addr) {
                obj.insert("function".to_string(), Value::String(name));
            }
            if include_source {
                if let Some(loc) = syms.source_location(addr) {
                    obj.insert("source".to_string(), Value::String(loc));
                }
            }
        }
    }

    // Inject timestamp if present.
    if let Some(ts) = last_ts {
        obj.insert("ts".to_string(), Value::Number(ts.into()));
    }

    serde_json::to_string(&obj).unwrap_or_else(|_| String::from("{\"type\":\"SerializeError\"}"))
}

#[derive(clap::Parser)]
pub struct Cmd {
    #[clap(flatten)]
    shared: CoreOptions,

    #[clap(flatten)]
    common: ProbeOptions,

    /// How long to capture trace data (milliseconds).
    ///
    /// The target runs freely during this period. The ETF circular buffer holds the
    /// most recent ~4 KB of trace when the duration completes.
    #[clap(long, default_value = "1000")]
    duration_ms: u64,

    /// ATB trace source ID to extract from the ETF frames.
    ///
    /// Must match the ID assigned when PTM tracing was configured. The default
    /// value of 1 matches the first PTM on a Cortex-A9 (RZ/A1L).
    #[clap(long, default_value = "1")]
    trace_id: u8,

    /// Number of context ID bytes in PTM packets (0–4, matches ETMCR.CONTEXTIDSIZE).
    ///
    /// For bare-metal Cortex-A9 without an OS, this is typically 0.
    #[clap(long, default_value = "0")]
    context_id_bytes: u8,

    /// Decode the raw PTM bytes and print packets to stderr in human-readable form.
    ///
    /// When set, the raw bytes are also still written to stdout or --output.
    #[clap(long)]
    decode: bool,

    /// Request PTM timestamp packets when supported by the trace source.
    #[clap(long)]
    timestamps: bool,

    /// Request PTM return-stack packets when supported by the trace source.
    #[clap(long)]
    return_stack: bool,

    /// Enable PTM branch-broadcast mode (ETMCR bit[8]).
    ///
    /// In this mode the PTM emits a branch address packet for every executed branch,
    /// including direct (compile-time-known) branches.  Enables offline execution
    /// reconstruction without an ELF, at the cost of higher trace bandwidth.
    #[clap(long)]
    branch_broadcast: bool,

    /// Firmware ELF to use for address-to-symbol resolution in --decode output.
    ///
    /// When provided, BRANCH and ISYNC addresses are annotated with the demangled
    /// function name (and optionally source location with --source).
    #[clap(long)]
    elf: Option<PathBuf>,

    /// Also print source file and line number when --elf is provided.
    #[clap(long, requires = "elf")]
    source: bool,

    /// Output format for --decode and --flow output.
    #[clap(long, default_value = "text")]
    output_format: OutputFormat,

    /// Compact execution-flow view: only print ISync and Branch targets, one per line.
    ///
    /// Much more readable than --decode for understanding where the CPU went.
    /// Pairs well with --elf for symbol annotations. Can be combined with --decode.
    #[clap(long)]
    flow: bool,

    /// Full instruction-level execution flow reconstruction.
    ///
    /// Uses the firmware ELF (required alongside --elf) to disassemble the instruction
    /// stream and assign E/N Atom outcomes to actual addresses.  Emits one line per
    /// executed instruction; much more verbose than --flow but gives complete coverage.
    /// Implies --flow output style for branch targets.
    #[clap(long, requires = "elf")]
    full_flow: bool,

    /// Flat sampling profile: reconstruct the instruction stream (with --elf) or count
    /// Branch/ISync targets (without --elf), then print a table of the hottest functions
    /// sorted by sample count.
    ///
    /// With --elf, every executed instruction is tallied using full flow reconstruction,
    /// giving instruction-level coverage akin to a hardware PMU profile.
    /// Without --elf, only Branch and ISync target addresses are counted; useful as a
    /// quick indicator when capturing with --branch-broadcast.
    #[clap(long)]
    profile: bool,

    /// Number of top functions to display in --profile output (default 20).
    #[clap(long, default_value = "20")]
    top_n: usize,

    /// Continuously drain the ETF in a loop, printing output after each window.
    ///
    /// Each capture window is `--duration-ms` long.  The loop runs until Ctrl-C or a
    /// probe error.  In this mode raw bytes are not written (--output is ignored).
    ///
    /// Pairs well with --flow (streaming branch view), --profile (rolling hot-function
    /// table), or --full-flow (verbose but complete instruction stream).
    ///
    /// The FlowReconstructor state (current PC / ISA) is preserved across windows so
    /// that Atom outcomes at window boundaries decode correctly.
    #[clap(long)]
    continuous: bool,

    /// Write raw PTM bytes to this file instead of stdout.
    #[clap(long, short = 'o')]
    output: Option<PathBuf>,

    /// Reconstruct call stacks from PTM instruction trace and write a
    /// Firefox Profiler-compatible flamegraph to this file (`.json.gz`).
    ///
    /// Requires `--elf`.  Uses full instruction-level flow reconstruction to
    /// track BL/BLX call depth and emits a callstack sample at every PTM ISync
    /// boundary (approximately every 1 024 instructions by default, controlled
    /// by `ETMSYNCFR`).  The resulting file can be opened with:
    ///
    /// ```
    /// samply load <file>
    /// ```
    ///
    /// The profiling is non-invasive — the target is never halted.
    #[clap(long, requires = "elf")]
    samply_output: Option<PathBuf>,

    /// Emit an extra callstack sample every N instructions when `--samply-output`
    /// is active, in addition to the mandatory ISync-boundary samples.
    ///
    /// Lower values give denser coverage at the cost of more memory.  A value of
    /// `0` (the default) only samples at ISync boundaries.
    #[clap(long, default_value = "0")]
    samply_interval: u32,

    /// Nominal CPU clock frequency in MHz for `--samply-output` timestamp scaling.
    ///
    /// The Firefox Profiler timeline uses wall-clock time, so the profiler needs to
    /// know how many nanoseconds correspond to one PTM instruction (formula:
    /// `1_000_000 / clock_mhz` ns per instruction).  Use the
    /// core frequency your target is running at (e.g. 400 for the RZ/A1L Cortex-A9
    /// at its maximum speed, 200 for half-speed operation).
    #[clap(long, default_value = "400")]
    samply_clock_mhz: u32,
}

impl Cmd {
    pub fn run(self, registry: &mut Registry, lister: &Lister) -> anyhow::Result<()> {
        let (mut session, _probe_options) = self.common.simple_attach(registry, lister)?;

        // Configure the CoreSight trace infrastructure: PTM → Funnel → ETF (circular mode).
        // For RZ/A1L the RZA1L sequence sets the ETF to Circular; setup_tracing() then
        // enables the PTM and starts capture.
        let enabled = session.setup_tracing(
            self.shared.core,
            TraceSink::TraceMemory(TraceMemoryConfig {
                timestamps: self.timestamps,
                return_stack: self.return_stack,
                branch_broadcast: self.branch_broadcast,
            }),
        )?;

        // Warn immediately if the connected trace source doesn't support a requested feature.
        if self.timestamps && !enabled.timestamps {
            eprintln!(
                "Warning: --timestamps was requested but the PTM on this target does \
                 not advertise timestamp support (ETMCCER bit 22 clear). \
                 Timestamp packets will not appear in the capture."
            );
        }
        if self.return_stack && !enabled.return_stack {
            eprintln!(
                "Warning: --return-stack was requested but the PTM on this target does \
                 not advertise return-stack support (ETMCCER bit 23 clear). \
                 Return-stack packets will not appear in the capture."
            );
        }

        {
            let mut core = session.core(self.shared.core)?;
            if matches!(core.status()?, CoreStatus::Halted(_)) {
                core.run()?;
            }
        }

        // Pre-load symbols and ELF image once — expensive and unchanging across windows.
        let symbols: Option<Symbols> = self.elf.as_ref().and_then(|p| Symbols::load(p).ok());
        let need_elf_image = self.full_flow || self.profile || self.samply_output.is_some();
        let elf_image: Option<ElfImage> = if need_elf_image {
            self.elf.as_ref().and_then(|p| match ElfImage::load(p) {
                Ok(img) => Some(img),
                Err(e)  => { eprintln!("Warning: failed to load ELF image: {e}"); None }
            })
        } else {
            None
        };

        // FlowReconstructor persists across capture windows so that PC/ISA state at the
        // end of one window carries into the next — critical for correct Atom decoding.
        // When --samply-output is active we use the sampling-capable constructor.
        let need_recon = (self.full_flow || self.profile || self.samply_output.is_some())
            && elf_image.is_some();
        let mut flow_recon: Option<FlowReconstructor> = if need_recon {
            let interval = if self.samply_output.is_some() { self.samply_interval } else { 0 };
            match FlowReconstructor::new_with_sampling(interval) {
                Ok(r)  => Some(r),
                Err(e) => { eprintln!("Warning: failed to create disassembler: {e}"); None }
            }
        } else {
            None
        };

        // Open raw byte output once.  Suppressed in --continuous mode: writing an
        // unbounded concatenation of capture windows to a file is rarely useful.
        let mut raw_out: Option<Box<dyn Write>> = if !self.continuous {
            Some(match self.output {
                Some(ref path) => {
                    let file = std::fs::File::create(path)?;
                    eprintln!("Writing to {}", path.display());
                    Box::new(file)
                }
                None => Box::new(std::io::stdout()),
            })
        } else {
            None
        };

        'drain: loop {
            // Wait for the target to generate trace data.
            if self.duration_ms > 0 {
                if self.continuous {
                    eprintln!(
                        "Capturing {} ms window (Ctrl-C to stop)…",
                        self.duration_ms
                    );
                } else {
                    eprintln!(
                        "Capturing trace for {} ms (target running freely)…",
                        self.duration_ms
                    );
                }
                sleep(Duration::from_millis(self.duration_ms));
            }

            // Drain the ETF: stop capture, read all frames, filter to our PTM's trace ID,
            // then re-enable capture so tracing can resume (circular mode).
            let ptm_bytes = match session.read_ptm_trace_data(self.trace_id) {
                Ok(b)  => b,
                Err(e) => {
                    if self.continuous {
                        eprintln!("Warning: drain failed: {e}; stopping.");
                        break 'drain;
                    } else {
                        return Err(e.into());
                    }
                }
            };

            eprintln!(
                "Read {} raw PTM bytes (trace_id={}).",
                ptm_bytes.len(),
                self.trace_id
            );

            // Optionally decode and/or show the execution flow.
            if self.decode || self.flow || self.full_flow || self.profile {
                // Collect into a Vec so all display modes share the same packets.
                let packets: Vec<ptm_decoder::PtmPacket> =
                    ptm_decoder::Decoder::new(&ptm_bytes, self.context_id_bytes).collect();

                // Pre-compute the running timestamp value in effect at each packet position.
                // The PTM emits Timestamp packets at configurable intervals; we propagate the
                // most recent seen value forward so every packet can show its approximate time.
                let packet_timestamps: Vec<Option<u64>> = {
                    let mut current: Option<u64> = None;
                    packets
                        .iter()
                        .map(|p| {
                            if let ptm_decoder::PtmPacket::Timestamp(ts) = p {
                                current = Some(*ts);
                            }
                            current
                        })
                        .collect()
                };

                // Per-type counters for the summary table.
                let mut n_sync = 0usize;
                let mut n_isync = 0usize;
                let mut n_branch = 0usize;
                let mut n_atom = 0usize;
                let mut n_trigger = 0usize;
                let mut n_waypoint = 0usize;
                let mut n_exc_return = 0usize;
                let mut n_overflow = 0usize;
                let mut n_context_id = 0usize;
                let mut n_timestamp = 0usize;
                let mut n_unknown = 0usize;

                for packet in &packets {
                    match packet {
                        ptm_decoder::PtmPacket::Sync               => n_sync += 1,
                        ptm_decoder::PtmPacket::ISync { .. }       => n_isync += 1,
                        ptm_decoder::PtmPacket::Branch { .. }      => n_branch += 1,
                        ptm_decoder::PtmPacket::Atom { .. }        => n_atom += 1,
                        ptm_decoder::PtmPacket::Trigger            => n_trigger += 1,
                        ptm_decoder::PtmPacket::WaypointUpdate     => n_waypoint += 1,
                        ptm_decoder::PtmPacket::ExceptionReturn    => n_exc_return += 1,
                        ptm_decoder::PtmPacket::Overflow           => n_overflow += 1,
                        ptm_decoder::PtmPacket::ContextId(_)       => n_context_id += 1,
                        ptm_decoder::PtmPacket::Timestamp(_)       => n_timestamp += 1,
                        ptm_decoder::PtmPacket::Unknown(_)         => n_unknown += 1,
                    }
                }

                // Run flow reconstruction once per window; share entries across
                // --full-flow, --profile, and --samply-output to avoid running the disassembler twice.
                let flow_entries: Option<Vec<FlowEntry>> =
                    if self.full_flow || self.profile || self.samply_output.is_some() {
                        if let (Some(img), Some(recon)) =
                            (elf_image.as_ref(), flow_recon.as_mut())
                        {
                            let mut all = Vec::new();
                            for packet in &packets {
                                all.extend(recon.consume(packet, img));
                            }
                            Some(all)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                // Compact execution-flow view: only ISync and Branch targets, one per line.
                if self.flow {
                    for (i, packet) in packets.iter().enumerate() {
                        let (label, addr) = match packet {
                            ptm_decoder::PtmPacket::ISync { pc, .. }       => ("ISYNC ", *pc as u64),
                            ptm_decoder::PtmPacket::Branch { address, .. } => ("BRANCH", *address as u64),
                            _ => continue,
                        };
                        let last_ts = packet_timestamps[i];
                        match self.output_format {
                            OutputFormat::Text => {
                                eprint!("{label}  {addr:#010x}");
                                if let Some(ref syms) = symbols {
                                    if let Some(name) = syms.function_name(addr) {
                                        eprint!("  {name}");
                                        if self.source {
                                            if let Some(loc) = syms.source_location(addr) {
                                                eprint!("  ({loc})");
                                            }
                                        }
                                    }
                                }
                                if let Some(ts) = last_ts {
                                    eprint!("  [ts:{ts}]");
                                }
                                eprintln!();
                            }
                            OutputFormat::Json => {
                                eprintln!("{}", packet_to_json(packet, &symbols, self.source, last_ts));
                            }
                        }
                    }
                }

                // Full packet dump.
                if self.decode {
                    for (i, packet) in packets.iter().enumerate() {
                        let last_ts = packet_timestamps[i];
                        match self.output_format {
                            OutputFormat::Text => {
                                eprintln!("{packet}");
                                if let Some(ref syms) = symbols {
                                    let addr = match packet {
                                        ptm_decoder::PtmPacket::Branch { address, .. } => Some(*address as u64),
                                        ptm_decoder::PtmPacket::ISync { pc, .. }       => Some(*pc as u64),
                                        _ => None,
                                    };
                                    if let Some(addr) = addr {
                                        if let Some(name) = syms.function_name(addr) {
                                            eprint!("  \u{21b3} {name}");
                                            if self.source {
                                                if let Some(loc) = syms.source_location(addr) {
                                                    eprint!("  ({loc})");
                                                }
                                            }
                                            if let Some(ts) = last_ts {
                                                eprint!("  [ts:{ts}]");
                                            }
                                            eprintln!();
                                        }
                                    }
                                }
                            }
                            OutputFormat::Json => {
                                eprintln!("{}", packet_to_json(packet, &symbols, self.source, last_ts));
                            }
                        }
                    }
                }

                // Summary table — suppressed in JSON mode to avoid mixing formats.
                if self.output_format == OutputFormat::Text && (self.decode || self.flow) {
                    let total = n_sync + n_isync + n_branch + n_atom + n_trigger
                        + n_waypoint + n_exc_return + n_overflow + n_context_id
                        + n_timestamp + n_unknown;

                    eprintln!("---");
                    eprintln!("Packet summary ({total} total):");
                    eprintln!("  SYNC            {n_sync:>6}");
                    eprintln!("  ISYNC           {n_isync:>6}");
                    eprintln!("  BRANCH          {n_branch:>6}");
                    eprintln!("  ATOM            {n_atom:>6}");
                    eprintln!("  TRIGGER         {n_trigger:>6}");
                    eprintln!("  WAYPOINT        {n_waypoint:>6}");
                    eprintln!("  EXCEPTION_RETURN{n_exc_return:>6}");
                    eprintln!("  OVERFLOW        {n_overflow:>6}");
                    eprintln!("  CONTEXT_ID      {n_context_id:>6}");
                    eprintln!("  TIMESTAMP       {n_timestamp:>6}");
                    if n_unknown > 0 {
                        eprintln!("  UNKNOWN         {n_unknown:>6}  \u{2190} possible framing issue");
                    }
                    eprintln!("---");
                }

                // Full instruction-level execution flow reconstruction.
                if self.full_flow {
                    match flow_entries {
                        Some(ref entries) => {
                            let n_insn = entries.len();
                            for entry in entries {
                                match self.output_format {
                                    OutputFormat::Text => {
                                        eprint!("{:#010x}  {}", entry.address, entry.text);
                                        if let Some(ref syms) = symbols {
                                            if let Some(name) = syms.function_name(entry.address as u64) {
                                                eprint!("  \u{21b3} {name}");
                                                if self.source {
                                                    if let Some(loc) = syms.source_location(entry.address as u64) {
                                                        eprint!("  ({loc})");
                                                    }
                                                }
                                            }
                                        }
                                        eprintln!();
                                    }
                                    OutputFormat::Json => {
                                        let mut obj = serde_json::json!({
                                            "type": "Insn",
                                            "address": format!("{:#010x}", entry.address),
                                            "insn": entry.text,
                                        });
                                        if let Some(ref syms) = symbols {
                                            if let Some(name) = syms.function_name(entry.address as u64) {
                                                obj["function"] = serde_json::Value::String(name);
                                            }
                                            if self.source {
                                                if let Some(loc) = syms.source_location(entry.address as u64) {
                                                    obj["source"] = serde_json::Value::String(loc);
                                                }
                                            }
                                        }
                                        eprintln!("{}", serde_json::to_string(&obj).unwrap_or_default());
                                    }
                                }
                            }
                            if self.output_format == OutputFormat::Text {
                                eprintln!("--- full-flow: {n_insn} instructions reconstructed");
                            }
                        }
                        None => {
                            if elf_image.is_none() {
                                eprintln!("Warning: --full-flow requires --elf");
                            }
                        }
                    }
                }

                // Flat sampling profile.
                if self.profile {
                    let mut hit_counts: std::collections::HashMap<u32, u64> =
                        std::collections::HashMap::new();

                    if let Some(ref entries) = flow_entries {
                        // Full instruction-level profile from flow reconstruction.
                        for entry in entries {
                            *hit_counts.entry(entry.address).or_insert(0) += 1;
                        }
                    } else {
                        // Branch-only profile: count ISync and Branch target addresses.
                        for packet in &packets {
                            let addr = match packet {
                                ptm_decoder::PtmPacket::ISync { pc, .. }       => Some(pc & !1),
                                ptm_decoder::PtmPacket::Branch { address, .. } => Some(address & !1),
                                _ => None,
                            };
                            if let Some(a) = addr {
                                *hit_counts.entry(a).or_insert(0) += 1;
                            }
                        }
                    }

                    if hit_counts.is_empty() {
                        eprintln!("--- profile: (no samples collected)");
                    } else {
                        let total: u64 = hit_counts.values().sum();

                        // Group per-address hits by function name (or hex address as fallback).
                        let mut fn_hits: std::collections::HashMap<String, u64> =
                            std::collections::HashMap::new();
                        // Remember one representative address per function for source lookup.
                        let mut fn_addr: std::collections::HashMap<String, u32> =
                            std::collections::HashMap::new();

                        for (&addr, &hits) in &hit_counts {
                            let name = symbols
                                .as_ref()
                                .and_then(|s| s.function_name(addr as u64))
                                .unwrap_or_else(|| format!("{addr:#010x}"));
                            *fn_hits.entry(name.clone()).or_insert(0) += hits;
                            fn_addr.entry(name).or_insert(addr);
                        }

                        let mut sorted: Vec<(String, u64)> = fn_hits.into_iter().collect();
                        sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                        sorted.truncate(self.top_n);

                        eprintln!("--- profile ({total} total samples) ---");
                        eprintln!("{:>6}  {:>9}  {}", "% time", "samples", "function");
                        for (name, hits) in &sorted {
                            let pct = *hits as f64 / total as f64 * 100.0;
                            let src = if self.source {
                                let addr = fn_addr.get(name).copied().unwrap_or(0);
                                symbols
                                    .as_ref()
                                    .and_then(|s| s.source_location(addr as u64))
                                    .map(|s| format!("  ({s})"))
                                    .unwrap_or_default()
                            } else {
                                String::new()
                            };
                            eprintln!("{pct:>6.2}  {hits:>9}  {name}{src}");
                        }
                        eprintln!("---");
                    }
                }
            }

            // Write raw bytes to file or stdout (single-shot mode only).
            if let Some(ref mut out) = raw_out {
                out.write_all(&ptm_bytes)?;
                out.flush()?;
            }

            if !self.continuous {
                break 'drain;
            }
        }

        // Emit samply-compatible flamegraph if requested.
        if let Some(ref out_path) = self.samply_output {
            if let (Some(ref elf_path), Some(ref mut recon)) =
                (self.elf.as_ref(), flow_recon.as_mut())
            {
                let samples = recon.take_ptm_samples();
                eprintln!("PTM sampling: {} callstack samples collected.", samples.len());
                if samples.is_empty() {
                    eprintln!("Warning: no samples collected. \
                        Ensure --elf is correct and the capture window contained executed code.");
                } else {
                    save_ptm_flamegraph(&samples, elf_path, out_path, self.samply_clock_mhz)?;
                }
            } else if self.elf.is_none() {
                eprintln!("Error: --samply-output requires --elf.");
            }
        }

        Ok(())
    }
}
