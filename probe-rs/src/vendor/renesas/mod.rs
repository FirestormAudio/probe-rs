//! Renesas vendor support.

pub mod sequences;

use std::borrow::Cow;

use jep106::JEP106Code;
use probe_rs_target::{Chip, chip_detection::ChipDetectionMethod};

use crate::{
    Error,
    architecture::arm::{
        ArmChipInfo, ArmDebugInterface, FullyQualifiedApAddress,
        dp::{DpRegister as _, TARGETID},
    },
    config::{DebugSequence, Registry},
    vendor::Vendor,
};

use sequences::rza1l::RZA1L;

/// Renesas
#[derive(docsplay::Display)]
pub struct Renesas;

const JEP_RENESAS: JEP106Code = JEP106Code::new(0x4, 0x23);

impl Vendor for Renesas {
    fn try_create_debug_sequence(&self, chip: &Chip) -> Option<DebugSequence> {
        // Match on the chip family name prefix to assign the RZA1L sequence to all
        // R7S721010/020/030 variants.
        let name = chip.name.as_str();
        if name.starts_with("R7S721") {
            return Some(DebugSequence::Arm(RZA1L::create()));
        }
        None
    }

    fn try_detect_arm_chip(
        &self,
        registry: &Registry,
        interface: &mut dyn ArmDebugInterface,
        chip_info: ArmChipInfo,
    ) -> Result<Option<String>, Error> {
        // Renesas provides part number registers (PNRn) for most of the RA variants.  However
        // where the registers live depends on the actual chip itself, often in areas that other
        // variants consider "reserved, do not touch". There should be four registers for a total
        // of 16 bytes with the data being zero padded.
        //
        // To narrow down the location of the PNR registers the reference manuals define PIDR 0 and
        // PIDR 1 for the CoreSight™ ROM table, and this will provide the TARGETID value.
        // Typically: ((PIDR0 << 4) | (PIDR1 & 0x0F))
        //
        // For future variants: ensure that if the TARGETID is shared with another variant that
        // the PNR registers are at the same location. If there is a conflict, this logic needs to
        // be reworked.

        if chip_info.manufacturer != JEP_RENESAS {
            return Ok(None);
        }

        // FIXME: This is a bit shaky but good enough for now.
        let access_port = &FullyQualifiedApAddress::v1_with_default_dp(0);

        let target_id = TARGETID(
            interface
                .read_raw_dp_register(interface.current_debug_port().unwrap(), TARGETID::ADDRESS)?,
        );
        let target_pn = target_id.tpartno();

        let mut part_number = [0_u8; 16];

        for family in registry.families() {
            for info in family
                .chip_detection
                .iter()
                .filter_map(ChipDetectionMethod::as_renesas_pnr)
            {
                if target_pn != info.target_id {
                    continue;
                }

                interface
                    .memory_interface(access_port)?
                    .read_8(info.mcu_pn_base as _, &mut part_number)?;

                let Ok(part_number) = std::str::from_utf8(&part_number) else {
                    continue;
                };

                let part_number: Cow<str> = match info.reverse_string {
                    true => Cow::Owned(part_number.chars().rev().collect()),
                    false => Cow::Borrowed(part_number),
                };
                let part_number = part_number.trim();

                for variant in info.variants.iter() {
                    if part_number.starts_with(variant) {
                        tracing::info!("Variant match: {}", variant);
                        return Ok(Some(variant.clone()));
                    }
                }
            }
        }

        Ok(None)
    }
}
