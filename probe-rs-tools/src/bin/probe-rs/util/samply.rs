// This code is adapted from samply-object and samply-debugid in the
// [samply codebase](https://github.com/mstange/samply)
// Dual licensed under Apache-2.0 and MIT.
// Code not relevant to ELF files has been removed.
// TODO: replace with the samply-object crate once it is published on crates.io.

use debugid::DebugId;
use object::{FileFlags, Object, ObjectSection};
use uuid::Uuid;

/// Tries to obtain a [`DebugId`] for an object.
///
/// Uses the ELF build ID if available; otherwise hashes the first page of `.text`.
/// Returns `None` on failure.
pub(crate) fn debug_id_for_object<'data>(obj: &impl Object<'data>) -> Option<DebugId> {
    if let Ok(Some(build_id)) = obj.build_id() {
        return Some(DebugId::from_identifier(build_id, obj.is_little_endian()));
    }
    if let Some(section) = obj.section_by_name(".text") {
        let data_len = section.size().min(4096);
        if let Ok(Some(first_page_data)) = section.data_range(section.address(), data_len) {
            return Some(DebugId::from_text_first_page(
                first_page_data,
                obj.is_little_endian(),
            ));
        }
    }
    None
}

/// Returns the "relative address base" for `obj`.
///
/// Relative addresses subtract this base from an SVMA to produce a
/// [`LookupAddress::Relative`](https://docs.rs/samply-symbols/latest/samply_symbols/enum.LookupAddress.html#variant.Relative).
///
/// For ELF binaries this is the vmaddr of the first LOAD segment. For all other
/// formats (PE etc.) it falls back to `object::Object::relative_address_base()`.
pub(crate) fn relative_address_base<'data>(obj: &impl Object<'data>) -> u64 {
    use object::read::ObjectSegment;
    if let FileFlags::Elf { .. } = obj.flags() {
        if let Some(first_segment) = obj.segments().next() {
            return first_segment.address();
        }
    }
    obj.relative_address_base()
}

/// Extension methods for constructing a [`DebugId`] from raw bytes or a text section hash.
pub(crate) trait DebugIdExt {
    /// Constructs a `DebugId` from an arbitrary identifier byte slice (e.g. an ELF build ID).
    /// Interprets the first 16 bytes as a UUID whose fields are encoded in the file's
    /// endianness, matching the Breakpad / sentry/symbolic convention.
    fn from_identifier(identifier: &[u8], little_endian: bool) -> Self;

    /// Constructs a `DebugId` by XOR-folding the first 4096 bytes of `.text` into 16 bytes,
    /// then calling `from_identifier`.
    fn from_text_first_page(text_first_page: &[u8], little_endian: bool) -> Self;
}

impl DebugIdExt for DebugId {
    fn from_identifier(identifier: &[u8], little_endian: bool) -> Self {
        // Pad or truncate to exactly 16 bytes. ELF build IDs are typically 20 bytes
        // (SHA-1), so this is lossy — same behaviour as Breakpad / symbolic.
        let mut d = [0u8; 16];
        let shared_len = identifier.len().min(d.len());
        d[0..shared_len].copy_from_slice(&identifier[0..shared_len]);

        // Treat the 16 bytes as a UUID whose u32/u16/u16 fields are in the file's
        // endianness, then re-serialise as big-endian via `Uuid::from_fields`.
        let (d1, d2, d3) = if little_endian {
            (
                u32::from_le_bytes([d[0], d[1], d[2], d[3]]),
                u16::from_le_bytes([d[4], d[5]]),
                u16::from_le_bytes([d[6], d[7]]),
            )
        } else {
            (
                u32::from_be_bytes([d[0], d[1], d[2], d[3]]),
                u16::from_be_bytes([d[4], d[5]]),
                u16::from_be_bytes([d[6], d[7]]),
            )
        };
        let uuid = Uuid::from_fields(d1, d2, d3, d[8..16].try_into().unwrap());
        DebugId::from_uuid(uuid)
    }

    fn from_text_first_page(text_first_page: &[u8], little_endian: bool) -> Self {
        const UUID_SIZE: usize = 16;
        const PAGE_SIZE: usize = 4096;
        let mut hash = [0u8; UUID_SIZE];
        for (i, byte) in text_first_page.iter().cloned().take(PAGE_SIZE).enumerate() {
            hash[i % UUID_SIZE] ^= byte;
        }
        DebugId::from_identifier(&hash, little_endian)
    }
}
