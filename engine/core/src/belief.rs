//! Stable feature contract for learned hidden-world belief scoring.
//!
//! The scorer observes one public root state and independently assigns one
//! residual log weight to each legal hidden-world candidate. It never observes
//! the order of the remaining wall.

use thiserror::Error;

use crate::game::{
    EVENT_RECORD_WIDTH, Game, MELD_OBSERVATION_WIDTH, META_OBSERVATION_WIDTH,
    RIVER_OBSERVATION_WIDTH, TILE_OBSERVATION_WIDTH,
};
use crate::types::{PLAYER_COUNT, Seat, TILE_KIND_COUNT};

/// Increment when the meaning, order, shape, or scalar encoding of a belief
/// feature changes.
pub const BELIEF_FEATURE_SCHEMA_VERSION: u32 = 2;
/// Increment when the proposal sampler or hand-written posterior target changes.
pub const BELIEF_TARGET_VERSION: u32 = 2;
/// Number of independent proposal streams used by deployed paired search.
pub const BELIEF_PROPOSAL_STREAM_COUNT: usize = 2;
/// Number of newest visible events retained by the belief scorer.
pub const BELIEF_EVENT_HISTORY_LENGTH: usize = 192;
pub const BELIEF_EVENT_FEATURE_WIDTH: usize = BELIEF_EVENT_HISTORY_LENGTH * EVENT_RECORD_WIDTH;
pub const BELIEF_OPPONENT_COUNT: usize = PLAYER_COUNT - 1;

/// Public, viewer-scoped features shared by all candidates at one root.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BeliefPublicFeatures {
    pub tile_observation: [u8; TILE_OBSERVATION_WIDTH],
    pub melds: [u8; MELD_OBSERVATION_WIDTH],
    pub river: [u8; RIVER_OBSERVATION_WIDTH],
    pub meta: [i32; META_OBSERVATION_WIDTH],
    /// Flat `[BELIEF_EVENT_HISTORY_LENGTH, EVENT_RECORD_WIDTH]` storage.
    /// Unused trailing capacity contains `-1` as defined by the engine event
    /// schema. `event_len` identifies the valid prefix.
    pub events: [i32; BELIEF_EVENT_FEATURE_WIDTH],
    pub event_len: u16,
}

/// Hidden allocation for one proposal world.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BeliefCandidateFeatures {
    /// Complete concealed counts for relative seats 1, 2, and 3.
    pub opponent_concealed: [[u8; TILE_KIND_COUNT]; BELIEF_OPPONENT_COUNT],
    /// Unordered histogram of the live wall. Wall order is not represented.
    pub live_wall: [u8; TILE_KIND_COUNT],
}

/// One candidate and the current hand-written posterior offset.
#[cfg(any(feature = "belief-training", test))]
#[derive(Clone, Debug, PartialEq)]
pub struct BeliefRootCandidate {
    pub features: BeliefCandidateFeatures,
    pub handwritten_log_weight: f64,
}

/// Failure reported by a learned belief evaluator or by output validation.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum BeliefResidualError {
    #[error("belief residual batch length mismatch: {candidates} candidates, {outputs} outputs")]
    BatchLength { candidates: usize, outputs: usize },
    #[error("belief residual evaluation failed: {message}")]
    Evaluation { message: String },
    #[error("belief residual {index} is not finite")]
    NonFinite { index: usize },
    #[error("belief residual requires at least {minimum} search iterations, got {actual}")]
    SearchIterations { actual: u16, minimum: u16 },
}

impl BeliefResidualError {
    pub fn evaluation(message: impl Into<String>) -> Self {
        Self::Evaluation {
            message: message.into(),
        }
    }
}

/// Batch interface implemented by a model runtime outside the rules engine.
///
/// Each output must depend only on `public` and the candidate at the same
/// index. Reordering or splitting a candidate batch must only reorder or split
/// the corresponding outputs. This property preserves stable particle prefixes
/// when the planner search budget grows.
pub trait BeliefResidualEvaluator: Send + Sync {
    fn evaluate_residuals(
        &self,
        public: &BeliefPublicFeatures,
        candidates: &[BeliefCandidateFeatures],
        output: &mut [f32],
    ) -> Result<(), BeliefResidualError>;
}

impl Game {
    /// Extracts the fixed public belief feature schema for one viewer.
    pub(crate) fn belief_public_features(&self, viewer: Seat) -> BeliefPublicFeatures {
        let mut tile_observation = [0; TILE_OBSERVATION_WIDTH];
        let mut melds = [0; MELD_OBSERVATION_WIDTH];
        let mut river = [0; RIVER_OBSERVATION_WIDTH];
        let mut meta = [0; META_OBSERVATION_WIDTH];
        self.observation_into(
            viewer,
            &mut tile_observation,
            &mut melds,
            &mut river,
            &mut meta,
        )
        .expect("belief observation buffers use the engine schema widths");

        let mut events = [-1; BELIEF_EVENT_FEATURE_WIDTH];
        let event_len = self
            .events_into(viewer, &mut events)
            .expect("belief event buffer uses a supported fixed capacity");

        BeliefPublicFeatures {
            tile_observation,
            melds,
            river,
            meta,
            events,
            event_len: event_len
                .try_into()
                .expect("belief event history length fits in u16"),
        }
    }

    /// Extracts one hidden-world candidate without exposing live-wall order.
    pub(crate) fn belief_candidate_features(&self, viewer: Seat) -> BeliefCandidateFeatures {
        let opponent_concealed =
            core::array::from_fn(|index| *self.concealed(viewer.offset(index as u8 + 1)));
        let mut live_wall = [0; TILE_KIND_COUNT];
        for offset in 0..self.wall_remaining() {
            let tile = self
                .live_wall_tile(offset)
                .expect("offset below wall_remaining has one live tile");
            live_wall[tile.index()] += 1;
        }
        BeliefCandidateFeatures {
            opponent_concealed,
            live_wall,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_features_use_the_fixed_event_capacity() {
        let game = Game::new(17);
        let features = game.belief_public_features(Seat::EAST);

        assert_eq!(features.events.len(), 192 * EVENT_RECORD_WIDTH);
        assert_eq!(usize::from(features.event_len), game.event_count());
    }

    #[test]
    fn candidate_features_ignore_live_wall_order() {
        let game = Game::new(19);
        let shuffled = game.resample_live_wall(23);

        assert_eq!(
            game.belief_candidate_features(Seat::EAST),
            shuffled.belief_candidate_features(Seat::EAST),
        );
    }

    #[test]
    fn candidate_features_use_relative_opponent_order() {
        let game = Game::new(29);
        let viewer = Seat::EAST.offset(2);
        let features = game.belief_candidate_features(viewer);

        for relative in 1..PLAYER_COUNT {
            assert_eq!(
                features.opponent_concealed[relative - 1],
                *game.concealed(viewer.offset(relative as u8)),
            );
        }
        assert_eq!(
            features
                .live_wall
                .iter()
                .map(|&count| usize::from(count))
                .sum::<usize>(),
            game.wall_remaining(),
        );
    }
}
