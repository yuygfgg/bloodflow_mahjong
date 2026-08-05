//! Binding-facing engine constants.
//!
//! This module is the single source of truth for constants exported by the
//! Python and WebAssembly bindings. Binding crates must not redefine values.

/// One named integer exported through FFI surfaces.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngineConstant {
    pub name: &'static str,
    pub value: i64,
}

/// Expand every binding constant with the provided macro.
///
/// The callback receives items of the form:
/// `NAME = { value_expr };`
#[macro_export]
macro_rules! with_engine_constants {
    ($macro:ident) => {
        $macro! {
            ACTION_SPACE_SIZE = { $crate::ACTION_SPACE_SIZE as i64 };
            ACTION_EXCHANGE_TILE_OFFSET = { $crate::ACTION_EXCHANGE_TILE_OFFSET as i64 };
            ACTION_CHOOSE_MISSING_OFFSET = { $crate::ACTION_CHOOSE_MISSING_OFFSET as i64 };
            ACTION_DISCARD_OFFSET = { $crate::ACTION_DISCARD_OFFSET as i64 };
            ACTION_HU = { $crate::ACTION_HU as i64 };
            ACTION_PONG = { $crate::ACTION_PONG as i64 };
            ACTION_EXPOSED_KONG = { $crate::ACTION_EXPOSED_KONG as i64 };
            ACTION_CONCEALED_KONG_OFFSET = { $crate::ACTION_CONCEALED_KONG_OFFSET as i64 };
            ACTION_ADDED_KONG_OFFSET = { $crate::ACTION_ADDED_KONG_OFFSET as i64 };
            ACTION_PASS = { $crate::ACTION_PASS as i64 };
            LEGAL_ACTION_MASK_WORDS = { $crate::LEGAL_ACTION_MASK_WORDS as i64 };
            STEP_RECORD_WIDTH = { $crate::STEP_RECORD_WIDTH as i64 };
            EVENT_RECORD_WIDTH = { $crate::EVENT_RECORD_WIDTH as i64 };
            EVENT_HISTORY_CAPACITY = { $crate::EVENT_HISTORY_CAPACITY as i64 };
            PLAYER_UI_STATS_FIELDS = { $crate::PLAYER_UI_STATS_FIELDS as i64 };
            PLAYER_UI_STATS_WIDTH = { $crate::PLAYER_UI_STATS_WIDTH as i64 };
            WALL_SETTLEMENT_FIELDS = { $crate::WALL_SETTLEMENT_FIELDS as i64 };
            WALL_SETTLEMENT_META_WIDTH = { $crate::WALL_SETTLEMENT_META_WIDTH as i64 };
            WALL_SETTLEMENT_HANDS_WIDTH = { $crate::WALL_SETTLEMENT_HANDS_WIDTH as i64 };
            ENGINE_RULES_VERSION = { $crate::ENGINE_RULES_VERSION as i64 };
            SHANTEN_COMPLETE = { $crate::SHANTEN_COMPLETE as i64 };
            SHANTEN_MAX = { $crate::SHANTEN_MAX as i64 };
            SHANTEN_TERMINAL = { $crate::SHANTEN_TERMINAL as i64 };
            SIMPLE_RULE_ACTION_TERMINAL = { $crate::SIMPLE_RULE_ACTION_TERMINAL as i64 };
            RULE_EV_ACTION_TERMINAL = { $crate::RULE_EV_ACTION_TERMINAL as i64 };
            RULE_PLANNER_ACTION_TERMINAL = { $crate::RULE_PLANNER_ACTION_TERMINAL as i64 };
            TILE_OBSERVATION_WIDTH = { $crate::TILE_OBSERVATION_WIDTH as i64 };
            TILE_OBSERVATION_PLANES = { $crate::TILE_OBSERVATION_PLANES as i64 };
            MELD_OBSERVATION_WIDTH = { $crate::MELD_OBSERVATION_WIDTH as i64 };
            MELD_SLOTS = { $crate::MELD_SLOTS as i64 };
            MELD_FIELDS = { $crate::MELD_FIELDS as i64 };
            RIVER_OBSERVATION_WIDTH = { $crate::RIVER_OBSERVATION_WIDTH as i64 };
            RIVER_TILE_CAPACITY = { $crate::RIVER_TILE_CAPACITY as i64 };
            RIVER_FIELDS = { $crate::RIVER_FIELDS as i64 };
            META_OBSERVATION_WIDTH = { $crate::META_OBSERVATION_WIDTH as i64 };
            ORACLE_TILE_COUNT_PLANES = { $crate::ORACLE_TILE_COUNT_PLANES as i64 };
            PLAYER_COUNT = { $crate::PLAYER_COUNT as i64 };
            TILE_KIND_COUNT = { $crate::TILE_KIND_COUNT as i64 };
            PHASE_EXCHANGE = { $crate::Phase::Exchange.code() as i64 };
            PHASE_CHOOSE_MISSING = { $crate::Phase::ChooseMissing.code() as i64 };
            PHASE_TURN = { $crate::Phase::Turn.code() as i64 };
            PHASE_HU_RESPONSE = { $crate::Phase::HuResponse.code() as i64 };
            PHASE_MELD_RESPONSE = { $crate::Phase::MeldResponse.code() as i64 };
            PHASE_FINISHED = { $crate::Phase::Finished.code() as i64 };
            TERMINATION_WALL_EXHAUSTED = { $crate::TerminationReason::WallExhausted.code() as i64 };
            TERMINATION_THREE_PLAYERS_BANKRUPT = {
                $crate::TerminationReason::ThreePlayersBankrupt.code() as i64
            };
            EVENT_KIND_ACTION = { $crate::EventKind::Action.code() as i64 };
            EVENT_KIND_GAME_START = { $crate::EventKind::GameStart.code() as i64 };
            EVENT_KIND_TURN_START = { $crate::EventKind::TurnStart.code() as i64 };
            EVENT_KIND_DRAW = { $crate::EventKind::Draw.code() as i64 };
            EVENT_KIND_DISCARD = { $crate::EventKind::Discard.code() as i64 };
            EVENT_KIND_EXCHANGE_COMPLETE = { $crate::EventKind::ExchangeComplete.code() as i64 };
            EVENT_KIND_MISSING_REVEALED = { $crate::EventKind::MissingRevealed.code() as i64 };
            EVENT_KIND_MELD = { $crate::EventKind::Meld.code() as i64 };
            EVENT_KIND_HU = { $crate::EventKind::Hu.code() as i64 };
            EVENT_KIND_PAYMENT = { $crate::EventKind::Payment.code() as i64 };
            EVENT_KIND_GAME_END = { $crate::EventKind::GameEnd.code() as i64 };
            EVENT_KIND_SETTLEMENT_STAGE = { $crate::EventKind::SettlementStage.code() as i64 };
            SETTLEMENT_STAGE_FLOWER_PIG = { $crate::SettlementStage::FlowerPig.code() as i64 };
            SETTLEMENT_STAGE_DAJIAO = { $crate::SettlementStage::Dajiao.code() as i64 };
            EVENT_FLAG_REPLACEMENT_DRAW = { $crate::EVENT_FLAG_REPLACEMENT_DRAW as i64 };
            EVENT_FLAG_LAST_WALL_TILE = { $crate::EVENT_FLAG_LAST_WALL_TILE as i64 };
            EVENT_FLAG_AFTER_KONG = { $crate::EVENT_FLAG_AFTER_KONG as i64 };
            EVENT_FLAG_OPENING_DISCARD = { $crate::EVENT_FLAG_OPENING_DISCARD as i64 };
            EVENT_FLAG_SELF_DRAW = { $crate::EVENT_FLAG_SELF_DRAW as i64 };
            EVENT_FLAG_ROB_KONG = { $crate::EVENT_FLAG_ROB_KONG as i64 };
            EVENT_FLAG_HEAVENLY = { $crate::EVENT_FLAG_HEAVENLY as i64 };
            EVENT_FLAG_EARTHLY = { $crate::EVENT_FLAG_EARTHLY as i64 };
        }
    };
}

macro_rules! engine_constants_table {
    ($($name:ident = { $value:expr };)*) => {
        /// All binding-exported engine constants in stable declaration order.
        pub const ENGINE_CONSTANTS: &[EngineConstant] = &[
            $(
                EngineConstant {
                    name: stringify!($name),
                    value: $value,
                },
            )*
        ];
    };
}

with_engine_constants!(engine_constants_table);

#[cfg(test)]
mod tests {
    use super::ENGINE_CONSTANTS;
    use crate::{ACTION_PASS, ACTION_SPACE_SIZE, ENGINE_RULES_VERSION};

    #[test]
    fn table_contains_core_action_and_rules_markers() {
        let by_name: std::collections::BTreeMap<_, _> = ENGINE_CONSTANTS
            .iter()
            .map(|item| (item.name, item.value))
            .collect();
        assert_eq!(by_name["ACTION_SPACE_SIZE"], ACTION_SPACE_SIZE as i64);
        assert_eq!(by_name["ACTION_PASS"], ACTION_PASS as i64);
        assert_eq!(
            by_name["ENGINE_RULES_VERSION"],
            i64::from(ENGINE_RULES_VERSION)
        );
        assert_eq!(ENGINE_CONSTANTS.len(), by_name.len());
    }
}
