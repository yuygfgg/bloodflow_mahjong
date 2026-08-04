//! ONNX-backed neural policy used by the `rule-nn` feature.
//!
//! The exported policy graph contains only the raw actor logits.  Legality is
//! an engine concern, so this module applies [`Game::legal_action_mask`] after
//! inference and never allows a model to submit an invalid action.

use std::io::Cursor;
use std::sync::Arc;

use thiserror::Error;
use tract_onnx::prelude::*;

use crate::{
    ActionId, ActionMask, EVENT_RECORD_WIDTH, Game, GameError, MELD_OBSERVATION_WIDTH,
    META_OBSERVATION_WIDTH, TILE_OBSERVATION_WIDTH,
};

/// Number of event records exported to the policy graph.
pub const RULE_NN_HISTORY: usize = 192;
/// Number of logits in the fixed engine action space.
pub const RULE_NN_LOGITS: usize = crate::ACTION_SPACE_SIZE;

const INPUT_COUNT: usize = 5;
const OUTPUT_COUNT: usize = 1;

/// Errors returned while loading or running an ONNX policy.
#[derive(Debug, Error)]
pub enum RuleNnError {
    #[error("failed to read ONNX model: {0}")]
    ModelLoad(String),
    #[error("failed to optimize ONNX model: {0}")]
    ModelOptimize(String),
    #[error("ONNX model schema is incompatible with rule-nn: {0}")]
    ModelSchema(String),
    #[error("failed to build rule-nn input tensor: {0}")]
    InputTensor(String),
    #[error("ONNX inference failed: {0}")]
    Inference(String),
    #[error("ONNX output schema is incompatible with rule-nn: {0}")]
    OutputSchema(String),
    #[error("no finite logit is available for a legal action")]
    NoFiniteLegalLogit,
    #[error("the engine returned a decision without a legal-action mask")]
    DecisionWithoutLegalActions,
    #[error("engine observation encoding failed: {0}")]
    Observation(#[from] GameError),
}

/// A thread-safe, immutable ONNX policy.
///
/// `tract` creates a fresh execution state for each call to `run`, so sharing
/// this model between tournament workers does not share mutable inference
/// state.  The model bytes are consumed during construction and are not
/// retained.
#[derive(Clone)]
pub struct RuleNn {
    model: Arc<TypedRunnableModel>,
}

impl RuleNn {
    /// Load an exported rule-nn graph from its ONNX bytes.
    pub fn from_onnx_bytes(bytes: &[u8]) -> Result<Self, RuleNnError> {
        if bytes.is_empty() {
            return Err(RuleNnError::ModelLoad("model is empty".to_owned()));
        }

        let model = tract_onnx::onnx()
            .model_for_read(&mut Cursor::new(bytes))
            .map_err(|error| RuleNnError::ModelLoad(error.to_string()))?
            .into_optimized()
            .map_err(|error| RuleNnError::ModelOptimize(error.to_string()))?;
        validate_model_schema(&model)?;
        let model = model
            .into_runnable()
            .map_err(|error| RuleNnError::ModelOptimize(error.to_string()))?;
        Ok(Self { model })
    }

    /// Evaluate the current actor's policy.
    ///
    /// A terminal game returns `Ok(None)`.  For an active game the result is
    /// always legal according to the engine mask, even when the model emits
    /// malformed or extreme logits.
    pub fn action(&self, game: &Game) -> Result<Option<ActionId>, RuleNnError> {
        let Some(decision) = game.decision() else {
            return Ok(None);
        };
        let legal = game
            .legal_action_mask()
            .ok_or(RuleNnError::DecisionWithoutLegalActions)?;

        let mut tile_obs = [0_u8; TILE_OBSERVATION_WIDTH];
        let mut melds = [0_u8; MELD_OBSERVATION_WIDTH];
        // The exported actor does not consume river planes, but the engine's
        // observation writer requires the complete fixed-width destination.
        let mut river = [0_u8; crate::RIVER_OBSERVATION_WIDTH];
        let mut meta = [0_i32; META_OBSERVATION_WIDTH];
        game.observation_into(
            decision.actor,
            &mut tile_obs,
            &mut melds,
            &mut river,
            &mut meta,
        )?;

        let mut events = [-1_i32; RULE_NN_HISTORY * EVENT_RECORD_WIDTH];
        let event_length = game.events_into(decision.actor, &mut events)?;
        debug_assert!(event_length <= RULE_NN_HISTORY);
        let event_length = event_length as i64;

        let inputs = tvec!(
            tensor(&[1, 10, 27], &tile_obs, "tile_obs")?,
            tensor(&[1, 4, 4, 3], &melds, "melds")?,
            tensor(&[1, 34], &meta, "meta")?,
            tensor(&[1, RULE_NN_HISTORY, EVENT_RECORD_WIDTH], &events, "events")?,
            tensor(&[1], &[event_length], "event_lengths")?,
        );
        let mut outputs = self
            .model
            .run(inputs)
            .map_err(|error| RuleNnError::Inference(error.to_string()))?;
        if outputs.len() != OUTPUT_COUNT {
            return Err(RuleNnError::OutputSchema(format!(
                "expected one output, got {}",
                outputs.len()
            )));
        }
        let output = outputs.remove(0);
        if output.datum_type() != DatumType::F32 || output.shape() != [1, RULE_NN_LOGITS] {
            return Err(RuleNnError::OutputSchema(format!(
                "expected f32 [1, {}], got {:?} {:?}",
                RULE_NN_LOGITS,
                output.datum_type(),
                output.shape()
            )));
        }
        let plain = output.as_plain().ok_or_else(|| {
            RuleNnError::OutputSchema("output is not plain tensor data".to_owned())
        })?;
        let logits = plain
            .as_slice::<f32>()
            .map_err(|error| RuleNnError::OutputSchema(error.to_string()))?;
        Ok(Some(select_legal_action(logits, legal)?))
    }
}

fn tensor<T: Datum + Copy>(shape: &[usize], data: &[T], name: &str) -> Result<TValue, RuleNnError> {
    Tensor::from_shape(shape, data)
        .map(IntoTValue::into_tvalue)
        .map_err(|error| RuleNnError::InputTensor(format!("{name}: {error}")))
}

fn select_legal_action(logits: &[f32], legal: ActionMask) -> Result<ActionId, RuleNnError> {
    if logits.len() != RULE_NN_LOGITS {
        return Err(RuleNnError::OutputSchema(format!(
            "expected {} logits, got {}",
            RULE_NN_LOGITS,
            logits.len()
        )));
    }

    // `ActionMask::iter` yields ascending action ids.  Replacing only on a
    // strict improvement preserves PyTorch argmax's lowest-index tie break.
    let mut best: Option<(ActionId, f32)> = None;
    for action in legal.iter() {
        let value = logits[action.index()];
        if !value.is_finite() {
            continue;
        }
        if best.is_none_or(|(_, best_value)| value > best_value) {
            best = Some((action, value));
        }
    }
    best.map(|(action, _)| action)
        .ok_or(RuleNnError::NoFiniteLegalLogit)
}

fn validate_model_schema(model: &TypedModel) -> Result<(), RuleNnError> {
    let inputs = model
        .input_outlets()
        .map_err(|error| RuleNnError::ModelSchema(error.to_string()))?;
    if inputs.len() != INPUT_COUNT {
        return Err(RuleNnError::ModelSchema(format!(
            "expected {INPUT_COUNT} inputs, got {}",
            inputs.len()
        )));
    }
    let expected = [
        ("tile_obs", DatumType::U8, vec![1, 10, 27]),
        ("melds", DatumType::U8, vec![1, 4, 4, 3]),
        ("meta", DatumType::I32, vec![1, 34]),
        (
            "events",
            DatumType::I32,
            vec![1, RULE_NN_HISTORY, EVENT_RECORD_WIDTH],
        ),
        ("event_lengths", DatumType::I64, vec![1]),
    ];
    for (index, (name, datum_type, shape)) in expected.into_iter().enumerate() {
        let fact = model
            .input_fact(index)
            .map_err(|error| RuleNnError::ModelSchema(error.to_string()))?;
        let actual_shape = fact
            .shape
            .as_concrete()
            .ok_or_else(|| RuleNnError::ModelSchema(format!("input {name} has symbolic shape")))?;
        if fact.datum_type != datum_type || actual_shape != shape.as_slice() {
            return Err(RuleNnError::ModelSchema(format!(
                "input {name}: expected {datum_type:?} {shape:?}, got {:?} {:?}",
                fact.datum_type, actual_shape
            )));
        }
    }

    let outputs = model
        .output_outlets()
        .map_err(|error| RuleNnError::ModelSchema(error.to_string()))?;
    if outputs.len() != OUTPUT_COUNT {
        return Err(RuleNnError::ModelSchema(format!(
            "expected one output, got {}",
            outputs.len()
        )));
    }
    let fact = model
        .output_fact(0)
        .map_err(|error| RuleNnError::ModelSchema(error.to_string()))?;
    let shape = fact
        .shape
        .as_concrete()
        .ok_or_else(|| RuleNnError::ModelSchema("output has symbolic shape".to_owned()))?;
    if fact.datum_type != DatumType::F32 || shape != [1, RULE_NN_LOGITS] {
        return Err(RuleNnError::ModelSchema(format!(
            "expected f32 [1, {}] output, got {:?} {:?}",
            RULE_NN_LOGITS, fact.datum_type, shape
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_policy_traits<T: Clone + Send + Sync>() {}

    #[test]
    fn policy_is_clone_send_and_sync() {
        assert_policy_traits::<RuleNn>();
    }

    #[test]
    fn selects_highest_finite_legal_logit_and_skips_illegal_values() {
        let mut legal = ActionMask::EMPTY;
        legal.insert(ActionId::new(4).unwrap());
        legal.insert(ActionId::new(9).unwrap());
        let mut logits = [f32::NEG_INFINITY; RULE_NN_LOGITS];
        logits[4] = 100.0;
        logits[9] = 101.0;
        logits[10] = f32::INFINITY;
        assert_eq!(select_legal_action(&logits, legal).unwrap().index(), 9);
    }

    #[test]
    fn ties_choose_lowest_action_id() {
        let mut legal = ActionMask::EMPTY;
        legal.insert(ActionId::new(7).unwrap());
        legal.insert(ActionId::new(3).unwrap());
        let mut logits = [0.0; RULE_NN_LOGITS];
        logits[3] = 2.0;
        logits[7] = 2.0;
        assert_eq!(select_legal_action(&logits, legal).unwrap().index(), 3);
    }

    #[test]
    fn rejects_when_all_legal_logits_are_non_finite() {
        let mut legal = ActionMask::EMPTY;
        legal.insert(ActionId::new(2).unwrap());
        let mut logits = [0.0; RULE_NN_LOGITS];
        logits[2] = f32::NAN;
        assert!(matches!(
            select_legal_action(&logits, legal),
            Err(RuleNnError::NoFiniteLegalLogit)
        ));
    }
}
