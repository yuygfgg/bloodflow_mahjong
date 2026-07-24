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
mod types;

pub use action::{
    ACTION_ADDED_KONG_OFFSET, ACTION_CHOOSE_MISSING_OFFSET, ACTION_CONCEALED_KONG_OFFSET,
    ACTION_DISCARD_OFFSET, ACTION_EXCHANGE_TILE_OFFSET, ACTION_EXPOSED_KONG, ACTION_HU,
    ACTION_PASS, ACTION_PONG, ACTION_SPACE_SIZE, Action, ActionId, ActionMask,
    LEGAL_ACTION_MASK_WORDS,
};
pub use game::{
    Batch, Decision, DiscardEvent, DrawEvent, DrawNotice, Game, GameError, LegalActions,
    MELD_OBSERVATION_WIDTH, META_OBSERVATION_WIDTH, Phase, PlayerStepOutcome,
    RIVER_OBSERVATION_WIDTH, STEP_RECORD_WIDTH, StepOutcome, TILE_OBSERVATION_WIDTH,
};
pub use hand::{
    MaxWaitEvaluation, Pattern, PatternSet, WinEvaluation, WinFlags, evaluate_max_wait,
    evaluate_win, is_winning,
};
pub use types::{ExchangeDirection, Meld, MeldKind, Seat, Suit, Tile, WinSource};
