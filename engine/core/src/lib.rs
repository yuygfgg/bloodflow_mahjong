//! Deterministic, allocation-conscious Blood Flow Mahjong rules engine.
//!
//! [`Game`] is the authoritative environment state. Submit only actions from
//! [`Game::legal_actions`], then use [`StepOutcome::for_player`] before a
//! transition is delivered to a player policy.

#![forbid(unsafe_code)]

mod action;
mod game;
mod hand;
mod rng;
mod rules;
mod types;

pub use action::{
    ACTION_ADDED_KONG_OFFSET, ACTION_CHOOSE_MISSING_OFFSET, ACTION_CONCEALED_KONG_OFFSET,
    ACTION_DISCARD_OFFSET, ACTION_EXCHANGE_TILE_OFFSET, ACTION_EXPOSED_KONG, ACTION_HU,
    ACTION_PASS, ACTION_PONG, ACTION_SPACE_SIZE, Action, ActionId, ActionMask,
    LEGAL_ACTION_MASK_WORDS,
};
pub use game::{
    Batch, Decision, DiscardEvent, DrawEvent, DrawNotice, ENGINE_RULES_VERSION,
    EVENT_FLAG_AFTER_KONG, EVENT_FLAG_EARTHLY, EVENT_FLAG_HEAVENLY, EVENT_FLAG_LAST_WALL_TILE,
    EVENT_FLAG_OPENING_DISCARD, EVENT_FLAG_REPLACEMENT_DRAW, EVENT_FLAG_ROB_KONG,
    EVENT_FLAG_SELF_DRAW, EVENT_HISTORY_CAPACITY, EVENT_RECORD_WIDTH, EventKind, Game, GameError,
    LegalActions, MELD_OBSERVATION_WIDTH, META_OBSERVATION_WIDTH, ORACLE_TILE_COUNT_PLANES,
    ORACLE_TILE_COUNT_WIDTH, Phase, PlayerStepOutcome, RIVER_OBSERVATION_WIDTH, STEP_RECORD_WIDTH,
    StepOutcome, TILE_OBSERVATION_WIDTH, TerminationReason,
};
pub use hand::{
    MaxWaitEvaluation, Pattern, PatternSet, SHANTEN_COMPLETE, SHANTEN_MAX, SHANTEN_TERMINAL,
    ShantenAnalysis, WinEvaluation, WinFlags, analyze_shanten, evaluate_max_wait, evaluate_shanten,
    evaluate_win, is_winning,
};
pub use rules::ev::{
    RULE_EV_ACTION_TERMINAL, RuleEvConfig, RuleEvDefense, RuleEvSearchStats,
    reset_rule_ev_search_stats, rule_ev_search_stats,
};
#[cfg(feature = "rule-ev-analysis")]
pub use rules::ev::{
    RuleEvSearchAction, RuleEvSearchCandidateSet, RuleEvSearchGate, RuleEvSearchTrace,
    RuleEvSearchTraceConfig, RuleEvSearchTraceError,
};
pub use rules::fast::SIMPLE_RULE_ACTION_TERMINAL;
pub use rules::planner::{
    RulePlannerConfig, RulePlannerSearchStats, reset_rule_planner_search_stats,
    rule_planner_search_stats,
};
pub use types::{ExchangeDirection, Meld, MeldKind, Seat, Suit, Tile, WinSource};
