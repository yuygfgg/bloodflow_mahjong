//! Candle runtime for the observation-only learned belief residual.
//!
//! The crate owns model files and tensor preprocessing. The rules engine only
//! sees the small [`BeliefResidualEvaluator`] interface and never sees a
//! filesystem path or a Candle tensor.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use bloodflow_mahjong::{
    BELIEF_EVENT_HISTORY_LENGTH, BELIEF_FEATURE_SCHEMA_VERSION, BELIEF_PROPOSAL_STREAM_COUNT,
    BELIEF_TARGET_VERSION, BeliefCandidateFeatures, BeliefPublicFeatures, BeliefResidualError,
    BeliefResidualEvaluator, ENGINE_RULES_VERSION,
};
use candle_core::shape::ShapeWithOneHole;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear, linear_no_bias};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use thiserror::Error;

const D_MODEL: usize = 128;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = D_MODEL / NUM_HEADS;
const STATIC_LAYERS: usize = 2;
const HISTORY_LAYERS: usize = 3;
const FFN_DIM: usize = 384;
const WORLD_HIDDEN: usize = 512;
const INTERACTION_WIDTH: usize = 256;
const TILE_KIND_COUNT: usize = 27;
const PLAYER_COUNT: usize = 4;
const MELD_SLOTS: usize = 4;
const MELD_FIELDS: usize = 3;
const TILE_PLANES: usize = 10;
const META_WIDTH: usize = 34;
const EVENT_WIDTH: usize = 8;
const EVENT_KIND_COUNT: usize = 11;
const ROPE_THETA: f32 = 10_000.0;
const GOLDEN_TOLERANCE: f32 = 1e-4;
const MODEL_FINGERPRINT_LENGTH: usize = 12;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("model file error: {0}")]
    Io(#[from] std::io::Error),
    #[error("model manifest error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("invalid model artifact: {0}")]
    InvalidArtifact(String),
    #[error("golden vector mismatch for {name}: max absolute error {max_error} > {tolerance}")]
    GoldenMismatch {
        name: &'static str,
        max_error: f32,
        tolerance: f32,
    },
}

#[derive(Clone, Debug, Deserialize)]
struct ArtifactManifest {
    artifact_version: u32,
    model_kind: String,
    belief_schema_version: u32,
    belief_target_version: u32,
    engine_rules_version: u32,
    proposal_stream_count: usize,
    calibration_particle_count: usize,
    max_history: usize,
    candidate_world_planes: usize,
    tile_kind_count: usize,
    config: ModelConfig,
    beta: f32,
    model_sha256: String,
    golden_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelConfig {
    d_model: usize,
    num_heads: usize,
    static_layers: usize,
    history_layers: usize,
    ffn_dim: usize,
    world_hidden: usize,
    interaction_width: usize,
    max_history: usize,
    rope_theta: f32,
}

impl ArtifactManifest {
    fn validate(&self) -> Result<(), ModelError> {
        let expected = [
            (self.artifact_version == 1, "artifact_version"),
            (self.model_kind == "belief_residual", "model_kind"),
            (
                self.belief_schema_version == BELIEF_FEATURE_SCHEMA_VERSION,
                "belief_schema_version",
            ),
            (
                self.belief_target_version == BELIEF_TARGET_VERSION,
                "belief_target_version",
            ),
            (
                self.engine_rules_version == ENGINE_RULES_VERSION,
                "engine_rules_version",
            ),
            (
                self.proposal_stream_count == BELIEF_PROPOSAL_STREAM_COUNT,
                "proposal_stream_count",
            ),
            (
                self.calibration_particle_count >= 2,
                "calibration_particle_count",
            ),
            (
                self.max_history == BELIEF_EVENT_HISTORY_LENGTH,
                "max_history",
            ),
            (self.candidate_world_planes == 4, "candidate_world_planes"),
            (self.tile_kind_count == TILE_KIND_COUNT, "tile_kind_count"),
            (self.config.d_model == D_MODEL, "config.d_model"),
            (self.config.num_heads == NUM_HEADS, "config.num_heads"),
            (
                self.config.static_layers == STATIC_LAYERS,
                "config.static_layers",
            ),
            (
                self.config.history_layers == HISTORY_LAYERS,
                "config.history_layers",
            ),
            (self.config.ffn_dim == FFN_DIM, "config.ffn_dim"),
            (
                self.config.world_hidden == WORLD_HIDDEN,
                "config.world_hidden",
            ),
            (
                self.config.interaction_width == INTERACTION_WIDTH,
                "config.interaction_width",
            ),
            (
                self.config.max_history == BELIEF_EVENT_HISTORY_LENGTH,
                "config.max_history",
            ),
        ];
        if let Some((false, name)) = expected.into_iter().find(|(ok, _)| !ok) {
            return Err(ModelError::InvalidArtifact(format!("unsupported {name}")));
        }
        if !self.beta.is_finite() || !(0.0..=1.0).contains(&self.beta) {
            return Err(ModelError::InvalidArtifact(
                "beta must be finite and in 0..=1".into(),
            ));
        }
        if !self.config.rope_theta.is_finite()
            || (self.config.rope_theta - ROPE_THETA).abs() > f32::EPSILON
        {
            return Err(ModelError::InvalidArtifact(
                "unsupported config.rope_theta".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct RmsNorm {
    weight: Tensor,
}

impl RmsNorm {
    fn load(vb: VarBuilder, width: usize) -> candle_core::Result<Self> {
        Ok(Self {
            weight: vb.get(width, "weight")?,
        })
    }

    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        candle_nn::ops::rms_norm(input, &self.weight, 1e-6)
    }
}

#[derive(Clone, Debug)]
struct SwiGlu {
    up: Linear,
    down: Linear,
}

impl SwiGlu {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            up: linear_no_bias(D_MODEL, FFN_DIM * 2, vb.pp("up"))?,
            down: linear_no_bias(FFN_DIM, D_MODEL, vb.pp("down"))?,
        })
    }

    fn forward(&self, input: &Tensor) -> candle_core::Result<Tensor> {
        let chunks = self.up.forward(input)?.chunk(2, D::Minus1)?;
        self.down.forward(&(&chunks[0].silu()? * &chunks[1])?)
    }
}

#[derive(Clone, Debug)]
struct TransformerBlock {
    attention_norm: RmsNorm,
    qkv: Linear,
    output: Linear,
    ffn_norm: RmsNorm,
    ffn: SwiGlu,
}

impl TransformerBlock {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            attention_norm: RmsNorm::load(vb.pp("attention_norm"), D_MODEL)?,
            qkv: linear_no_bias(D_MODEL, D_MODEL * 3, vb.pp("attention").pp("qkv"))?,
            output: linear_no_bias(D_MODEL, D_MODEL, vb.pp("attention").pp("output"))?,
            ffn_norm: RmsNorm::load(vb.pp("ffn_norm"), D_MODEL)?,
            ffn: SwiGlu::load(vb.pp("ffn"))?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        mask: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> candle_core::Result<Tensor> {
        let normalized = self.attention_norm.forward(input)?;
        let qkv = self.qkv.forward(&normalized)?.chunk(3, D::Minus1)?;
        let (mut query, mut key, value) = (
            split_heads(&qkv[0])?,
            split_heads(&qkv[1])?,
            split_heads(&qkv[2])?,
        );
        if let Some((cos, sin)) = rope {
            query = apply_interleaved_rope(&query, cos, sin)?;
            key = apply_interleaved_rope(&key, cos, sin)?;
        }
        let scores = (query.matmul(&key.transpose(2, 3)?)? / (HEAD_DIM as f64).sqrt())?;
        let scores = scores.broadcast_add(mask)?;
        let attention = candle_nn::ops::softmax_last_dim(&scores)?;
        let attended = attention.matmul(&value)?;
        let attended = attended.transpose(1, 2)?.contiguous()?.reshape((
            input.dim(0)?,
            input.dim(1)?,
            D_MODEL,
        ))?;
        let hidden = (input + self.output.forward(&attended)?)?;
        let feed_forward = self.ffn.forward(&self.ffn_norm.forward(&hidden)?)?;
        hidden + feed_forward
    }
}

fn split_heads(input: &Tensor) -> candle_core::Result<Tensor> {
    let (batch, length, _) = input.dims3()?;
    input
        .reshape((batch, length, NUM_HEADS, HEAD_DIM))?
        .transpose(1, 2)
}

fn apply_interleaved_rope(
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> candle_core::Result<Tensor> {
    let (batch, heads, length, dim) = input.dims4()?;
    let half = dim / 2;
    let paired = input.reshape((batch, heads, length, half, 2))?;
    let even = paired.narrow(4, 0, 1)?.squeeze(4)?;
    let odd = paired.narrow(4, 1, 1)?.squeeze(4)?;
    let rotated_even = (even.broadcast_mul(cos)? - odd.broadcast_mul(sin)?)?;
    let rotated_odd = (even.broadcast_mul(sin)? + odd.broadcast_mul(cos)?)?;
    Tensor::stack(&[&rotated_even, &rotated_odd], 4)?.reshape((batch, heads, length, dim))
}

fn make_rope(device: &Device, length: usize) -> candle_core::Result<(Tensor, Tensor)> {
    let mut cos = Vec::with_capacity(length * HEAD_DIM / 2);
    let mut sin = Vec::with_capacity(length * HEAD_DIM / 2);
    for position in 0..length {
        for pair in 0..HEAD_DIM / 2 {
            let inverse = 1.0 / ROPE_THETA.powf((2 * pair) as f32 / HEAD_DIM as f32);
            let angle = position as f32 * inverse;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    Ok((
        Tensor::from_vec(cos, (1, 1, length, HEAD_DIM / 2), device)?,
        Tensor::from_vec(sin, (1, 1, length, HEAD_DIM / 2), device)?,
    ))
}

#[derive(Clone, Debug)]
struct StaticEncoder {
    count_embeddings: Vec<Embedding>,
    tile_embedding: Embedding,
    suit_embedding: Embedding,
    rank_embedding: Embedding,
    binary_embedding: Embedding,
    phase_embedding: Embedding,
    direction_embedding: Embedding,
    optional_tile_embedding: Embedding,
    optional_seat_embedding: Embedding,
    optional_suit_embedding: Embedding,
    reaction_embedding: Embedding,
    global_numeric: Linear,
    seat_embedding: Embedding,
    player_numeric: Linear,
    meld_kind_embedding: Embedding,
    meld_slot_embedding: Embedding,
    token_type_embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    output_norm: RmsNorm,
}

impl StaticEncoder {
    fn load(vb: VarBuilder) -> candle_core::Result<Self> {
        let count_embeddings = (0..TILE_PLANES)
            .map(|index| embedding(5, D_MODEL, vb.pp("count_embeddings").pp(index)))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let blocks = (0..STATIC_LAYERS)
            .map(|index| TransformerBlock::load(vb.pp("blocks").pp(index)))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            count_embeddings,
            tile_embedding: embedding(TILE_KIND_COUNT, D_MODEL, vb.pp("tile_embedding"))?,
            suit_embedding: embedding(3, D_MODEL, vb.pp("suit_embedding"))?,
            rank_embedding: embedding(9, D_MODEL, vb.pp("rank_embedding"))?,
            binary_embedding: embedding(2, D_MODEL, vb.pp("binary_embedding"))?,
            phase_embedding: embedding(6, D_MODEL, vb.pp("phase_embedding"))?,
            direction_embedding: embedding(4, D_MODEL, vb.pp("direction_embedding"))?,
            optional_tile_embedding: embedding(
                TILE_KIND_COUNT + 1,
                D_MODEL,
                vb.pp("optional_tile_embedding"),
            )?,
            optional_seat_embedding: embedding(
                PLAYER_COUNT + 1,
                D_MODEL,
                vb.pp("optional_seat_embedding"),
            )?,
            optional_suit_embedding: embedding(4, D_MODEL, vb.pp("optional_suit_embedding"))?,
            reaction_embedding: embedding(8, D_MODEL, vb.pp("reaction_embedding"))?,
            global_numeric: linear_no_bias(3, D_MODEL, vb.pp("global_numeric"))?,
            seat_embedding: embedding(PLAYER_COUNT, D_MODEL, vb.pp("seat_embedding"))?,
            player_numeric: linear_no_bias(3, D_MODEL, vb.pp("player_numeric"))?,
            meld_kind_embedding: embedding(5, D_MODEL, vb.pp("meld_kind_embedding"))?,
            meld_slot_embedding: embedding(MELD_SLOTS, D_MODEL, vb.pp("meld_slot_embedding"))?,
            token_type_embedding: embedding(4, D_MODEL, vb.pp("token_type_embedding"))?,
            blocks,
            output_norm: RmsNorm::load(vb.pp("output_norm"), D_MODEL)?,
        })
    }

    fn forward(
        &self,
        tile_obs: &[u8; TILE_PLANES * TILE_KIND_COUNT],
        melds: &[u8; PLAYER_COUNT * MELD_SLOTS * MELD_FIELDS],
        meta: &[i32; META_WIDTH],
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let global = self.global_token(meta, device)?.unsqueeze(1)?;
        let tiles = self.tile_tokens(tile_obs, meta, device)?;
        let players = self.player_tokens(meta, device)?;
        let (meld_tokens, padding) = self.meld_tokens(melds, device)?;
        let hidden = Tensor::cat(&[&global, &tiles, &players, &meld_tokens], 1)?;
        let mask = static_mask(&padding, device)?;
        let mut hidden = hidden;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &mask, None)?;
        }
        self.output_norm
            .forward(&hidden.narrow(1, 0, 1)?.squeeze(1)?)
    }

    fn global_token(
        &self,
        meta: &[i32; META_WIDTH],
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let phase = [clamp(meta[0], 0, 5) as u32];
        let direction = [clamp(meta[3], 0, 3) as u32];
        let draw = [optional(meta[5], TILE_KIND_COUNT) as u32];
        let response = [optional(meta[8], TILE_KIND_COUNT) as u32];
        let seat = [optional(meta[7], PLAYER_COUNT) as u32];
        let suit = [optional(meta[11], 3) as u32];
        let reaction = [clamp(meta[29], 0, 7) as u32];
        let binary_draw = [clamp(meta[6], 0, 1) as u32];
        let binary_response = [clamp(meta[28], 0, 1) as u32];
        let numeric = [
            meta[4] as f32 / 55.0,
            meta[9] as f32 / 108.0,
            meta[10] as f32 / 3.0,
        ];
        let mut token = self
            .phase_embedding
            .forward(&indices(&phase, (1,), device)?)?;
        token = (token
            + self
                .direction_embedding
                .forward(&indices(&direction, (1,), device)?)?)?;
        token = (token
            + self
                .optional_tile_embedding
                .forward(&indices(&draw, (1,), device)?)?)?;
        token = (token
            + self
                .optional_seat_embedding
                .forward(&indices(&seat, (1,), device)?)?)?;
        token = (token
            + self
                .optional_tile_embedding
                .forward(&indices(&response, (1,), device)?)?)?;
        token = (token
            + self
                .optional_suit_embedding
                .forward(&indices(&suit, (1,), device)?)?)?;
        token = (token
            + self
                .reaction_embedding
                .forward(&indices(&reaction, (1,), device)?)?)?;
        token = (token
            + self
                .binary_embedding
                .forward(&indices(&binary_draw, (1,), device)?)?)?;
        token = (token
            + self
                .binary_embedding
                .forward(&indices(&binary_response, (1,), device)?)?)?;
        token = (token
            + self
                .global_numeric
                .forward(&Tensor::from_slice(&numeric, (1, 3), device)?)?)?;
        let type_zero = self
            .token_type_embedding
            .forward(&indices(&[0], (1,), device)?)?;
        token + type_zero
    }

    fn tile_tokens(
        &self,
        tile_obs: &[u8; TILE_PLANES * TILE_KIND_COUNT],
        meta: &[i32; META_WIDTH],
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let tile_ids: Vec<u32> = (0..TILE_KIND_COUNT as u32).collect();
        let mut token =
            self.tile_embedding
                .forward(&indices(&tile_ids, (1, TILE_KIND_COUNT), device)?)?;
        let suit: Vec<u32> = (0..TILE_KIND_COUNT).map(|id| (id / 9) as u32).collect();
        let rank: Vec<u32> = (0..TILE_KIND_COUNT).map(|id| (id % 9) as u32).collect();
        token = (token
            + self
                .suit_embedding
                .forward(&indices(&suit, (1, TILE_KIND_COUNT), device)?)?)?;
        token = (token
            + self
                .rank_embedding
                .forward(&indices(&rank, (1, TILE_KIND_COUNT), device)?)?)?;
        for plane in 0..TILE_PLANES {
            let counts: Vec<u32> = (0..TILE_KIND_COUNT)
                .map(|tile| u32::from(tile_obs[plane * TILE_KIND_COUNT + tile].min(4)))
                .collect();
            token = (token
                + self.count_embeddings[plane].forward(&indices(
                    &counts,
                    (1, TILE_KIND_COUNT),
                    device,
                )?)?)?;
        }
        let draw: Vec<u32> = (0..TILE_KIND_COUNT)
            .map(|id| u32::from(meta[5] == id as i32))
            .collect();
        let response: Vec<u32> = (0..TILE_KIND_COUNT)
            .map(|id| u32::from(meta[8] == id as i32))
            .collect();
        let missing: Vec<u32> = (0..TILE_KIND_COUNT)
            .map(|id| u32::from(meta[16] == (id / 9) as i32))
            .collect();
        token = (token
            + self
                .binary_embedding
                .forward(&indices(&draw, (1, TILE_KIND_COUNT), device)?)?)?;
        token = (token
            + self.binary_embedding.forward(&indices(
                &response,
                (1, TILE_KIND_COUNT),
                device,
            )?)?)?;
        token = (token
            + self
                .binary_embedding
                .forward(&indices(&missing, (1, TILE_KIND_COUNT), device)?)?)?;
        let type_one = self
            .token_type_embedding
            .forward(&indices(&[1], (1,), device)?)?;
        token + type_one.broadcast_as((1, TILE_KIND_COUNT, D_MODEL))?
    }

    fn player_tokens(
        &self,
        meta: &[i32; META_WIDTH],
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let seats: Vec<u32> = (0..PLAYER_COUNT as u32).collect();
        let mut token =
            self.seat_embedding
                .forward(&indices(&seats, (1, PLAYER_COUNT), device)?)?;
        let dealer: Vec<u32> = (0..PLAYER_COUNT)
            .map(|seat| u32::from(meta[2] == seat as i32))
            .collect();
        token = (token
            + self
                .binary_embedding
                .forward(&indices(&dealer, (1, PLAYER_COUNT), device)?)?)?;
        let suits: Vec<u32> = (0..PLAYER_COUNT)
            .map(|seat| optional_clamped(meta[16 + seat], 3) as u32)
            .collect();
        token = (token
            + self.optional_suit_embedding.forward(&indices(
                &suits,
                (1, PLAYER_COUNT),
                device,
            )?)?)?;
        let binary: Vec<u32> = (0..PLAYER_COUNT)
            .map(|seat| clamp(meta[20 + seat], 0, 1) as u32)
            .collect();
        token = (token
            + self
                .binary_embedding
                .forward(&indices(&binary, (1, PLAYER_COUNT), device)?)?)?;
        let mut numeric = Vec::with_capacity(PLAYER_COUNT * 3);
        for seat in 0..PLAYER_COUNT {
            numeric.push(meta[12 + seat] as f32 / 10_000.0);
            numeric.push(meta[24 + seat] as f32 / 18.0);
            numeric.push((1.0 + meta[30 + seat].max(0) as f32).log2() / 8.0);
        }
        token = (token
            + self.player_numeric.forward(&Tensor::from_slice(
                &numeric,
                (1, PLAYER_COUNT, 3),
                device,
            )?)?)?;
        let type_two = self
            .token_type_embedding
            .forward(&indices(&[2], (1,), device)?)?;
        token + type_two.broadcast_as((1, PLAYER_COUNT, D_MODEL))?
    }

    fn meld_tokens(
        &self,
        melds: &[u8; PLAYER_COUNT * MELD_SLOTS * MELD_FIELDS],
        device: &Device,
    ) -> candle_core::Result<(Tensor, Vec<bool>)> {
        let mut tiles = Vec::with_capacity(PLAYER_COUNT * MELD_SLOTS);
        let mut kinds = Vec::with_capacity(PLAYER_COUNT * MELD_SLOTS);
        let mut sources = Vec::with_capacity(PLAYER_COUNT * MELD_SLOTS);
        let mut slots = Vec::with_capacity(PLAYER_COUNT * MELD_SLOTS);
        let mut owners = Vec::with_capacity(PLAYER_COUNT * MELD_SLOTS);
        let mut padding = Vec::with_capacity(PLAYER_COUNT * MELD_SLOTS);
        for owner in 0..PLAYER_COUNT {
            for slot in 0..MELD_SLOTS {
                let offset = (owner * MELD_SLOTS + slot) * MELD_FIELDS;
                let is_padding = melds[offset] == 255;
                padding.push(is_padding);
                tiles.push(if is_padding {
                    TILE_KIND_COUNT
                } else {
                    melds[offset].min(26) as usize
                } as u32);
                kinds.push(if is_padding {
                    4
                } else {
                    melds[offset + 1].min(3)
                } as u32);
                sources.push(if is_padding {
                    PLAYER_COUNT as u32
                } else {
                    u32::from(melds[offset + 2].min(3))
                });
                slots.push(slot as u32);
                owners.push(owner as u32);
            }
        }
        let shape = (1, PLAYER_COUNT * MELD_SLOTS);
        let mut token = self
            .optional_tile_embedding
            .forward(&indices(&tiles, shape, device)?)?;
        token = (token
            + self
                .meld_kind_embedding
                .forward(&indices(&kinds, shape, device)?)?)?;
        token = (token
            + self
                .seat_embedding
                .forward(&indices(&owners, shape, device)?)?)?;
        token = (token
            + self
                .optional_seat_embedding
                .forward(&indices(&sources, shape, device)?)?)?;
        token = (token
            + self
                .meld_slot_embedding
                .forward(&indices(&slots, shape, device)?)?)?;
        let type_three = self
            .token_type_embedding
            .forward(&indices(&[3], (1,), device)?)?;
        Ok((
            (token + type_three.broadcast_as((1, PLAYER_COUNT * MELD_SLOTS, D_MODEL))?)?,
            padding,
        ))
    }
}

fn static_mask(padding: &[bool], device: &Device) -> candle_core::Result<Tensor> {
    let length = 1 + TILE_KIND_COUNT + PLAYER_COUNT + PLAYER_COUNT * MELD_SLOTS;
    let mut values = vec![0.0_f32; length * length];
    for query in 0..length {
        for key in 0..length {
            let valid = key < length - padding.len() || !padding[key - (length - padding.len())];
            if !valid {
                values[query * length + key] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(values, (1, 1, length, length), device)
}

#[derive(Clone, Debug)]
struct HistoryEncoder {
    kind_embedding: Embedding,
    seat_embedding: Embedding,
    tile_embedding: Embedding,
    flags_embedding: Embedding,
    numeric: Linear,
    blocks: Vec<TransformerBlock>,
    output_norm: RmsNorm,
    empty_summary: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    causal_mask: Tensor,
}

impl HistoryEncoder {
    fn load(vb: VarBuilder, device: &Device) -> candle_core::Result<Self> {
        let blocks = (0..HISTORY_LAYERS)
            .map(|index| TransformerBlock::load(vb.pp("blocks").pp(index)))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let (rope_cos, rope_sin) = make_rope(device, BELIEF_EVENT_HISTORY_LENGTH)?;
        let causal_mask = causal_mask(device)?;
        Ok(Self {
            kind_embedding: embedding(EVENT_KIND_COUNT + 1, D_MODEL, vb.pp("kind_embedding"))?,
            seat_embedding: embedding(PLAYER_COUNT + 1, D_MODEL, vb.pp("seat_embedding"))?,
            tile_embedding: embedding(TILE_KIND_COUNT + 1, D_MODEL, vb.pp("tile_embedding"))?,
            flags_embedding: embedding(256, D_MODEL, vb.pp("flags_embedding"))?,
            numeric: linear_no_bias(2, D_MODEL, vb.pp("numeric"))?,
            blocks,
            output_norm: RmsNorm::load(vb.pp("output_norm"), D_MODEL)?,
            empty_summary: vb.get(D_MODEL, "empty_summary")?,
            rope_cos,
            rope_sin,
            causal_mask,
        })
    }

    fn forward(
        &self,
        events: &[i32; BELIEF_EVENT_HISTORY_LENGTH * EVENT_WIDTH],
        length: usize,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        validate_event_length(length).map_err(candle_core::Error::msg)?;
        let mut kinds = Vec::with_capacity(BELIEF_EVENT_HISTORY_LENGTH);
        let mut actors = Vec::with_capacity(BELIEF_EVENT_HISTORY_LENGTH);
        let mut targets = Vec::with_capacity(BELIEF_EVENT_HISTORY_LENGTH);
        let mut tiles = Vec::with_capacity(BELIEF_EVENT_HISTORY_LENGTH);
        let mut flags = Vec::with_capacity(BELIEF_EVENT_HISTORY_LENGTH);
        let mut numeric = Vec::with_capacity(BELIEF_EVENT_HISTORY_LENGTH * 2);
        let denominator = (1.0 + 40_000.0_f32).ln();
        for event in events.chunks_exact(EVENT_WIDTH) {
            kinds.push(clamp(event[0], 0, EVENT_KIND_COUNT as i32 - 1) as u32);
            actors.push(optional(event[1], PLAYER_COUNT) as u32);
            targets.push(optional(event[2], PLAYER_COUNT) as u32);
            tiles.push(optional(event[3], TILE_KIND_COUNT) as u32);
            flags.push(clamp(event[4], 0, 255) as u32);
            for value in &event[5..7] {
                let value = *value as f32;
                numeric.push(value.signum() * (1.0 + value.abs()).ln() / denominator);
            }
        }
        let shape = (1, BELIEF_EVENT_HISTORY_LENGTH);
        let mut hidden = self
            .kind_embedding
            .forward(&indices(&kinds, shape, device)?)?;
        hidden = (hidden
            + self
                .seat_embedding
                .forward(&indices(&actors, shape, device)?)?)?;
        hidden = (hidden
            + self
                .seat_embedding
                .forward(&indices(&targets, shape, device)?)?)?;
        hidden = (hidden
            + self
                .tile_embedding
                .forward(&indices(&tiles, shape, device)?)?)?;
        hidden = (hidden
            + self
                .flags_embedding
                .forward(&indices(&flags, shape, device)?)?)?;
        let numeric = Tensor::from_vec(numeric, (1, BELIEF_EVENT_HISTORY_LENGTH, 2), device)?;
        hidden = (hidden + self.numeric.forward(&numeric.clamp(-1.0, 1.0)?)?)?;
        for block in &self.blocks {
            hidden = block.forward(
                &hidden,
                &self.causal_mask,
                Some((&self.rope_cos, &self.rope_sin)),
            )?;
        }
        let hidden = self.output_norm.forward(&hidden)?;
        if length == 0 {
            return self.empty_summary.unsqueeze(0);
        }
        hidden.narrow(1, length - 1, 1)?.squeeze(1)
    }
}

fn causal_mask(device: &Device) -> candle_core::Result<Tensor> {
    let length = BELIEF_EVENT_HISTORY_LENGTH;
    let mut values = vec![f32::NEG_INFINITY; length * length];
    for query in 0..length {
        for key in 0..=query {
            values[query * length + key] = 0.0;
        }
    }
    Tensor::from_vec(values, (1, 1, length, length), device)
}

#[derive(Clone, Debug)]
struct BeliefNetwork {
    static_encoder: StaticEncoder,
    history_encoder: HistoryEncoder,
    public_norm: RmsNorm,
    public_projection: Linear,
    world_first: Linear,
    world_second: Linear,
    interaction_norm: RmsNorm,
    interaction_first: Linear,
    interaction_second: Linear,
}

impl BeliefNetwork {
    fn load(vb: VarBuilder, device: &Device) -> candle_core::Result<Self> {
        Ok(Self {
            static_encoder: StaticEncoder::load(vb.pp("static_encoder"))?,
            history_encoder: HistoryEncoder::load(vb.pp("history_encoder"), device)?,
            public_norm: RmsNorm::load(vb.pp("public_projection").pp(0), D_MODEL * 2)?,
            public_projection: linear(
                D_MODEL * 2,
                INTERACTION_WIDTH,
                vb.pp("public_projection").pp(1),
            )?,
            world_first: linear(
                4 * TILE_KIND_COUNT,
                WORLD_HIDDEN,
                vb.pp("world_encoder").pp(0),
            )?,
            world_second: linear(
                WORLD_HIDDEN,
                INTERACTION_WIDTH,
                vb.pp("world_encoder").pp(2),
            )?,
            interaction_norm: RmsNorm::load(vb.pp("residual_head").pp(0), INTERACTION_WIDTH * 3)?,
            interaction_first: linear(
                INTERACTION_WIDTH * 3,
                INTERACTION_WIDTH,
                vb.pp("residual_head").pp(1),
            )?,
            interaction_second: linear(INTERACTION_WIDTH, 1, vb.pp("residual_head").pp(3))?,
        })
    }

    fn encode_public(
        &self,
        public: &BeliefPublicFeatures,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        let static_summary = self.static_encoder.forward(
            &public.tile_observation,
            &public.melds,
            &public.meta,
            device,
        )?;
        let history_summary =
            self.history_encoder
                .forward(&public.events, usize::from(public.event_len), device)?;
        let combined = Tensor::cat(&[&static_summary, &history_summary], 1)?;
        self.public_projection
            .forward(&self.public_norm.forward(&combined)?)?
            .silu()
    }

    fn score(
        &self,
        public: &Tensor,
        candidates: &[BeliefCandidateFeatures],
        device: &Device,
    ) -> candle_core::Result<Vec<f32>> {
        let mut world_data = Vec::with_capacity(candidates.len() * 4 * TILE_KIND_COUNT);
        for candidate in candidates {
            for hand in &candidate.opponent_concealed {
                world_data.extend(hand.iter().map(|&count| f32::from(count) / 4.0));
            }
            world_data.extend(
                candidate
                    .live_wall
                    .iter()
                    .map(|&count| f32::from(count) / 4.0),
            );
        }
        let worlds = Tensor::from_vec(world_data, (candidates.len(), 4 * TILE_KIND_COUNT), device)?;
        let worlds = self.world_first.forward(&worlds)?.silu()?;
        let worlds = self.world_second.forward(&worlds)?.silu()?;
        let public = public.broadcast_as((candidates.len(), INTERACTION_WIDTH))?;
        let product = (&public * &worlds)?;
        let interaction = Tensor::cat(&[&public, &worlds, &product], 1)?;
        let hidden = self
            .interaction_first
            .forward(&self.interaction_norm.forward(&interaction)?)?
            .silu()?;
        let output = self.interaction_second.forward(&hidden)?.squeeze(1)?;
        output.to_vec1()
    }
}

/// A loaded CPU model. `beta` is read from the artifact manifest and scales
/// the learned residual before it enters the engine's hand-written log weight.
#[derive(Clone, Debug)]
pub struct CandleBeliefModel {
    network: BeliefNetwork,
    device: Device,
    beta: f32,
    calibration_particle_count: usize,
    artifact_dir: PathBuf,
    model_sha256: String,
    golden_sha256: String,
}

impl CandleBeliefModel {
    /// Loads the complete artifact and validates its golden vector. Only CPU
    /// inference is supported in the first version.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let artifact_dir = path.as_ref().to_path_buf();
        let manifest_bytes = fs::read(artifact_dir.join("manifest.json"))?;
        let manifest: ArtifactManifest = serde_json::from_slice(&manifest_bytes)?;
        manifest.validate()?;
        let model_path = artifact_dir.join("model.safetensors");
        let model_bytes = fs::read(&model_path)?;
        let model_hash = sha256_bytes(&model_bytes);
        if model_hash != manifest.model_sha256 {
            return Err(ModelError::InvalidArtifact(format!(
                "model SHA-256 mismatch: expected {}, got {}",
                manifest.model_sha256, model_hash
            )));
        }
        let golden_bytes = fs::read(artifact_dir.join("golden.safetensors"))?;
        let golden_hash = sha256_bytes(&golden_bytes);
        if golden_hash != manifest.golden_sha256 {
            return Err(ModelError::InvalidArtifact(format!(
                "golden SHA-256 mismatch: expected {}, got {golden_hash}",
                manifest.golden_sha256,
            )));
        }
        let device = Device::Cpu;
        let vb = VarBuilder::from_buffered_safetensors(model_bytes, DType::F32, &device)?;
        let network = BeliefNetwork::load(vb, &device)?;
        let model = Self {
            network,
            device,
            beta: manifest.beta,
            calibration_particle_count: manifest.calibration_particle_count,
            artifact_dir,
            model_sha256: model_hash,
            golden_sha256: manifest.golden_sha256,
        };
        model.verify_golden_bytes(&golden_bytes, GOLDEN_TOLERANCE)?;
        Ok(model)
    }

    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Returns the proposal count used by offline ESS/beta calibration.
    pub fn calibration_particle_count(&self) -> usize {
        self.calibration_particle_count
    }

    pub fn artifact_dir(&self) -> &Path {
        &self.artifact_dir
    }

    /// Returns the first 12 hexadecimal digits of the verified model digest.
    pub fn fingerprint(&self) -> &str {
        short_fingerprint(&self.model_sha256)
    }

    fn evaluate_raw(
        &self,
        public: &BeliefPublicFeatures,
        candidates: &[BeliefCandidateFeatures],
    ) -> Result<Vec<f32>, ModelError> {
        validate_event_length(usize::from(public.event_len))?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let public = self.network.encode_public(public, &self.device)?;
        let values = self.network.score(&public, candidates, &self.device)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::InvalidArtifact(
                "model returned a non-finite residual".into(),
            ));
        }
        Ok(values)
    }

    /// Compares both the public encoder output and residual head with the
    /// Python-exported golden vector. This is intentionally explicit so an
    /// artifact cannot enter a tournament without a reproducibility check.
    pub fn verify_golden(&self, tolerance: f32) -> Result<(), ModelError> {
        validate_tolerance(tolerance)?;
        let bytes = fs::read(self.artifact_dir.join("golden.safetensors"))?;
        let hash = sha256_bytes(&bytes);
        if hash != self.golden_sha256 {
            return Err(ModelError::InvalidArtifact(format!(
                "golden SHA-256 mismatch: expected {}, got {hash}",
                self.golden_sha256,
            )));
        }
        self.verify_golden_bytes(&bytes, tolerance)
    }

    fn verify_golden_bytes(&self, bytes: &[u8], tolerance: f32) -> Result<(), ModelError> {
        validate_tolerance(tolerance)?;
        let tensors = candle_core::safetensors::load_buffer(bytes, &self.device)?;
        let tile_obs = tensor_rows_u8::<{ TILE_PLANES * TILE_KIND_COUNT }>(
            &tensors,
            "tile_obs",
            &[2, TILE_PLANES, TILE_KIND_COUNT],
        )?;
        let melds = tensor_rows_u8::<{ PLAYER_COUNT * MELD_SLOTS * MELD_FIELDS }>(
            &tensors,
            "melds",
            &[2, PLAYER_COUNT, MELD_SLOTS, MELD_FIELDS],
        )?;
        let meta = tensor_rows_i32::<META_WIDTH>(&tensors, "meta", &[2, META_WIDTH])?;
        let events = tensor_rows_i32::<{ BELIEF_EVENT_HISTORY_LENGTH * EVENT_WIDTH }>(
            &tensors,
            "events",
            &[2, BELIEF_EVENT_HISTORY_LENGTH, EVENT_WIDTH],
        )?;
        let lengths_tensor = tensor(&tensors, "event_lengths")?;
        require_shape(lengths_tensor, "event_lengths", &[2])?;
        let lengths = lengths_tensor.to_vec1::<i32>()?;
        let worlds_tensor = tensor(&tensors, "candidate_worlds")?;
        let world_shape = worlds_tensor.dims4()?;
        if world_shape.0 != 2
            || world_shape.1 == 0
            || world_shape.2 != 4
            || world_shape.3 != TILE_KIND_COUNT
        {
            return Err(ModelError::InvalidArtifact(
                "golden candidate_worlds shape mismatch".into(),
            ));
        }
        let worlds = worlds_tensor
            .reshape((world_shape.0 * world_shape.1, 4 * TILE_KIND_COUNT))?
            .to_vec2::<u8>()?;
        let public_tensor = tensor(&tensors, "public")?;
        require_shape(public_tensor, "public", &[2, INTERACTION_WIDTH])?;
        let expected_public = public_tensor.to_vec2::<f32>()?;
        let residual_tensor = tensor(&tensors, "residuals")?;
        require_shape(residual_tensor, "residuals", &[2, world_shape.1])?;
        let expected_residuals = residual_tensor.to_vec2::<f32>()?;
        for row in 0..2 {
            let event_len: u16 = lengths[row].try_into().map_err(|_| {
                ModelError::InvalidArtifact("golden event length is outside u16".into())
            })?;
            validate_event_length(usize::from(event_len))?;
            let public = BeliefPublicFeatures {
                tile_observation: tile_obs[row],
                melds: melds[row],
                river: [0; 108 * 2],
                meta: meta[row],
                events: events[row],
                event_len,
            };
            let encoded_tensor = self.network.encode_public(&public, &self.device)?;
            let encoded = encoded_tensor.to_vec2::<f32>()?;
            let max_public = checked_max_abs(&encoded[0], &expected_public[row], "public")?;
            if max_public > tolerance {
                return Err(ModelError::GoldenMismatch {
                    name: "public",
                    max_error: max_public,
                    tolerance,
                });
            }
            let candidate_count = expected_residuals[row].len();
            let mut candidates = Vec::with_capacity(candidate_count);
            for candidate in 0..candidate_count {
                let flat = worlds
                    .get(row * candidate_count + candidate)
                    .ok_or_else(|| {
                        ModelError::InvalidArtifact(
                            "golden candidate_worlds row count mismatch".into(),
                        )
                    })?;
                let mut opponent_concealed = [[0_u8; TILE_KIND_COUNT]; 3];
                opponent_concealed[0].copy_from_slice(&flat[..TILE_KIND_COUNT]);
                opponent_concealed[1].copy_from_slice(&flat[TILE_KIND_COUNT..2 * TILE_KIND_COUNT]);
                opponent_concealed[2]
                    .copy_from_slice(&flat[2 * TILE_KIND_COUNT..3 * TILE_KIND_COUNT]);
                let mut live_wall = [0_u8; TILE_KIND_COUNT];
                live_wall.copy_from_slice(&flat[3 * TILE_KIND_COUNT..]);
                candidates.push(BeliefCandidateFeatures {
                    opponent_concealed,
                    live_wall,
                });
            }
            let values = self
                .network
                .score(&encoded_tensor, &candidates, &self.device)?;
            let max_residual = checked_max_abs(&values, &expected_residuals[row], "residuals")?;
            if max_residual > tolerance {
                return Err(ModelError::GoldenMismatch {
                    name: "residuals",
                    max_error: max_residual,
                    tolerance,
                });
            }
        }
        Ok(())
    }
}

impl BeliefResidualEvaluator for CandleBeliefModel {
    fn evaluate_residuals(
        &self,
        public: &BeliefPublicFeatures,
        candidates: &[BeliefCandidateFeatures],
        output: &mut [f32],
    ) -> Result<(), BeliefResidualError> {
        if candidates.len() != output.len() {
            return Err(BeliefResidualError::BatchLength {
                candidates: candidates.len(),
                outputs: output.len(),
            });
        }
        let values = self
            .evaluate_raw(public, candidates)
            .map_err(|error| BeliefResidualError::evaluation(error.to_string()))?;
        for (index, (destination, value)) in output.iter_mut().zip(values).enumerate() {
            let value = value * self.beta;
            if !value.is_finite() {
                return Err(BeliefResidualError::NonFinite { index });
            }
            *destination = value;
        }
        Ok(())
    }
}

fn clamp(value: i32, low: i32, high: i32) -> i32 {
    value.clamp(low, high)
}

fn optional(value: i32, missing: usize) -> usize {
    if value < 0 {
        missing
    } else {
        value.clamp(0, missing as i32 - 1) as usize
    }
}

fn optional_clamped(value: i32, missing: usize) -> usize {
    optional(value, missing)
}

fn indices<S: ShapeWithOneHole>(
    values: &[u32],
    shape: S,
    device: &Device,
) -> candle_core::Result<Tensor> {
    Tensor::from_vec(values.to_vec(), shape, device)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

fn short_fingerprint(sha256: &str) -> &str {
    sha256.get(..MODEL_FINGERPRINT_LENGTH).unwrap_or(sha256)
}

fn validate_tolerance(tolerance: f32) -> Result<(), ModelError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ModelError::InvalidArtifact(
            "golden tolerance must be finite and non-negative".into(),
        ));
    }
    Ok(())
}

fn validate_event_length(length: usize) -> Result<(), ModelError> {
    if length > BELIEF_EVENT_HISTORY_LENGTH {
        return Err(ModelError::InvalidArtifact(format!(
            "event length {length} exceeds belief history capacity {}",
            BELIEF_EVENT_HISTORY_LENGTH,
        )));
    }
    Ok(())
}

fn checked_max_abs(left: &[f32], right: &[f32], name: &'static str) -> Result<f32, ModelError> {
    if left.len() != right.len() {
        return Err(ModelError::InvalidArtifact(format!(
            "golden value length mismatch for {name}: {} vs {}",
            left.len(),
            right.len()
        )));
    }
    let mut maximum = 0.0_f32;
    for (&actual, &expected) in left.iter().zip(right) {
        if !actual.is_finite() || !expected.is_finite() {
            return Err(ModelError::InvalidArtifact(format!(
                "golden contains a non-finite value for {name}"
            )));
        }
        maximum = maximum.max((actual - expected).abs());
    }
    Ok(maximum)
}

fn tensor<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor, ModelError> {
    tensors
        .get(name)
        .ok_or_else(|| ModelError::InvalidArtifact(format!("golden missing {name}")))
}

fn require_shape(tensor: &Tensor, name: &str, expected: &[usize]) -> Result<(), ModelError> {
    if tensor.dims() != expected {
        return Err(ModelError::InvalidArtifact(format!(
            "golden shape mismatch for {name}: expected {expected:?}, got {:?}",
            tensor.dims()
        )));
    }
    Ok(())
}

fn tensor_rows_u8<const N: usize>(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<[u8; N]>, ModelError> {
    let tensor = tensor(tensors, name)?;
    require_shape(tensor, name, expected_shape)?;
    let rows = *expected_shape
        .first()
        .ok_or_else(|| ModelError::InvalidArtifact(format!("golden shape for {name} is empty")))?;
    if expected_shape[1..].iter().product::<usize>() != N {
        return Err(ModelError::InvalidArtifact(format!(
            "golden shape constant mismatch for {name}"
        )));
    }
    let values = tensor.flatten_all()?.to_vec1::<u8>()?;
    Ok(values
        .chunks_exact(N)
        .take(rows)
        .map(|chunk| {
            let mut row = [0_u8; N];
            row.copy_from_slice(chunk);
            row
        })
        .collect())
}

fn tensor_rows_i32<const N: usize>(
    tensors: &HashMap<String, Tensor>,
    name: &str,
    expected_shape: &[usize],
) -> Result<Vec<[i32; N]>, ModelError> {
    let tensor = tensor(tensors, name)?;
    require_shape(tensor, name, expected_shape)?;
    let rows = *expected_shape
        .first()
        .ok_or_else(|| ModelError::InvalidArtifact(format!("golden shape for {name} is empty")))?;
    if expected_shape[1..].iter().product::<usize>() != N {
        return Err(ModelError::InvalidArtifact(format!(
            "golden shape constant mismatch for {name}"
        )));
    }
    let values = tensor.flatten_all()?.to_vec1::<i32>()?;
    Ok(values
        .chunks_exact(N)
        .take(rows)
        .map(|chunk| {
            let mut row = [0_i32; N];
            row.copy_from_slice(chunk);
            row
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use serde_json::{Value, json};

    use super::*;

    static NEXT_TEST_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new(name: &str) -> Self {
            let sequence = NEXT_TEST_DIRECTORY.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "bloodflow-model-runtime-{name}-{}-{sequence}",
                std::process::id(),
            ));
            fs::create_dir(&path).expect("create test artifact directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn valid_manifest() -> ArtifactManifest {
        ArtifactManifest {
            artifact_version: 1,
            model_kind: "belief_residual".into(),
            belief_schema_version: BELIEF_FEATURE_SCHEMA_VERSION,
            belief_target_version: BELIEF_TARGET_VERSION,
            engine_rules_version: ENGINE_RULES_VERSION,
            proposal_stream_count: BELIEF_PROPOSAL_STREAM_COUNT,
            calibration_particle_count: 64,
            max_history: BELIEF_EVENT_HISTORY_LENGTH,
            candidate_world_planes: 4,
            tile_kind_count: TILE_KIND_COUNT,
            config: ModelConfig {
                d_model: D_MODEL,
                num_heads: NUM_HEADS,
                static_layers: STATIC_LAYERS,
                history_layers: HISTORY_LAYERS,
                ffn_dim: FFN_DIM,
                world_hidden: WORLD_HIDDEN,
                interaction_width: INTERACTION_WIDTH,
                max_history: BELIEF_EVENT_HISTORY_LENGTH,
                rope_theta: ROPE_THETA,
            },
            beta: 0.25,
            model_sha256: String::new(),
            golden_sha256: String::new(),
        }
    }

    fn manifest_json(model_sha256: &str, golden_sha256: &str) -> Value {
        json!({
            "artifact_version": 1,
            "model_kind": "belief_residual",
            "belief_schema_version": BELIEF_FEATURE_SCHEMA_VERSION,
            "belief_target_version": BELIEF_TARGET_VERSION,
            "engine_rules_version": ENGINE_RULES_VERSION,
            "proposal_stream_count": BELIEF_PROPOSAL_STREAM_COUNT,
            "calibration_particle_count": 64,
            "max_history": BELIEF_EVENT_HISTORY_LENGTH,
            "candidate_world_planes": 4,
            "tile_kind_count": TILE_KIND_COUNT,
            "config": {
                "d_model": D_MODEL,
                "num_heads": NUM_HEADS,
                "static_layers": STATIC_LAYERS,
                "history_layers": HISTORY_LAYERS,
                "ffn_dim": FFN_DIM,
                "world_hidden": WORLD_HIDDEN,
                "interaction_width": INTERACTION_WIDTH,
                "max_history": BELIEF_EVENT_HISTORY_LENGTH,
                "rope_theta": ROPE_THETA,
            },
            "beta": 0.25,
            "model_sha256": model_sha256,
            "golden_sha256": golden_sha256,
        })
    }

    fn write_manifest(directory: &Path, manifest: &Value) {
        fs::write(
            directory.join("manifest.json"),
            serde_json::to_vec(manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
    }

    fn assert_invalid_artifact(result: Result<(), ModelError>, expected: &str) {
        match result {
            Err(ModelError::InvalidArtifact(message)) => {
                assert!(
                    message.contains(expected),
                    "expected {expected:?} in {message:?}",
                );
            }
            other => panic!("expected invalid artifact containing {expected:?}, got {other:?}"),
        }
    }

    #[test]
    fn manifest_accepts_supported_contract() {
        valid_manifest().validate().expect("valid manifest");
    }

    #[test]
    fn manifest_rejects_wrong_engine_rules_version() {
        let mut manifest = valid_manifest();
        manifest.engine_rules_version = ENGINE_RULES_VERSION.wrapping_add(1);

        assert_invalid_artifact(manifest.validate(), "engine_rules_version");
    }

    #[test]
    fn manifest_rejects_wrong_belief_target_version() {
        let mut manifest = valid_manifest();
        manifest.belief_target_version = BELIEF_TARGET_VERSION.wrapping_add(1);

        assert_invalid_artifact(manifest.validate(), "belief_target_version");
    }

    #[test]
    fn manifest_rejects_wrong_proposal_stream_count() {
        let mut manifest = valid_manifest();
        manifest.proposal_stream_count = BELIEF_PROPOSAL_STREAM_COUNT - 1;

        assert_invalid_artifact(manifest.validate(), "proposal_stream_count");
    }

    #[test]
    fn manifest_requires_proposal_stream_count() {
        let mut manifest = manifest_json("model", "golden");
        manifest
            .as_object_mut()
            .expect("manifest object")
            .remove("proposal_stream_count");

        let error = serde_json::from_value::<ArtifactManifest>(manifest)
            .expect_err("missing proposal stream count must be rejected");
        assert!(
            error.to_string().contains("proposal_stream_count"),
            "{error}"
        );
    }

    #[test]
    fn manifest_requires_belief_target_version() {
        let mut manifest = manifest_json("model", "golden");
        manifest
            .as_object_mut()
            .expect("manifest object")
            .remove("belief_target_version");

        let error = serde_json::from_value::<ArtifactManifest>(manifest)
            .expect_err("missing belief target version must be rejected");
        assert!(
            error.to_string().contains("belief_target_version"),
            "{error}"
        );
    }

    #[test]
    fn fingerprint_is_a_short_model_digest() {
        let digest = sha256_bytes(b"model identity");

        assert_eq!(
            short_fingerprint(&digest),
            &digest[..MODEL_FINGERPRINT_LENGTH]
        );
        assert_eq!(short_fingerprint("short"), "short");
    }

    #[test]
    fn manifest_rejects_invalid_beta() {
        for beta in [-0.1, 1.1, f32::NAN, f32::INFINITY] {
            let mut manifest = valid_manifest();
            manifest.beta = beta;

            assert_invalid_artifact(manifest.validate(), "beta");
        }
    }

    #[test]
    fn manifest_rejects_invalid_rope_theta() {
        for rope_theta in [ROPE_THETA * 2.0, f32::NAN, f32::INFINITY] {
            let mut manifest = valid_manifest();
            manifest.config.rope_theta = rope_theta;

            assert_invalid_artifact(manifest.validate(), "config.rope_theta");
        }
    }

    #[test]
    fn tolerance_must_be_finite_and_non_negative() {
        validate_tolerance(0.0).expect("zero tolerance");
        validate_tolerance(GOLDEN_TOLERANCE).expect("positive tolerance");
        for tolerance in [-f32::EPSILON, f32::NAN, f32::INFINITY] {
            assert_invalid_artifact(validate_tolerance(tolerance), "tolerance");
        }
    }

    #[test]
    fn golden_tensor_shape_must_match_contract() {
        let tensor = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).expect("test tensor");

        assert_invalid_artifact(
            require_shape(&tensor, "public", &[2, 4]),
            "shape mismatch for public",
        );
    }

    #[test]
    fn golden_values_must_be_finite() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_invalid_artifact(
                checked_max_abs(&[0.0], &[value], "public").map(|_| ()),
                "non-finite value for public",
            );
        }
    }

    #[test]
    fn event_length_accepts_full_contract_range() {
        validate_event_length(0).expect("empty history");
        validate_event_length(BELIEF_EVENT_HISTORY_LENGTH).expect("full history");
        assert_invalid_artifact(
            validate_event_length(BELIEF_EVENT_HISTORY_LENGTH + 1),
            "exceeds belief history capacity",
        );
    }

    #[test]
    fn load_requires_golden_file_before_model_decode() {
        let directory = TestDirectory::new("missing-golden");
        let model_bytes = b"model bytes are not decoded before golden preflight";
        fs::write(directory.path().join("model.safetensors"), model_bytes)
            .expect("write model placeholder");
        write_manifest(
            directory.path(),
            &manifest_json(&sha256_bytes(model_bytes), &sha256_bytes(b"golden")),
        );

        match CandleBeliefModel::load(directory.path()) {
            Err(ModelError::Io(error)) => assert_eq!(error.kind(), std::io::ErrorKind::NotFound),
            other => panic!("expected missing golden file, got {other:?}"),
        }
    }

    #[test]
    fn load_rejects_wrong_model_sha_before_decode() {
        let directory = TestDirectory::new("wrong-model-sha");
        fs::write(directory.path().join("model.safetensors"), b"model")
            .expect("write model placeholder");
        write_manifest(
            directory.path(),
            &manifest_json(&sha256_bytes(b"different model"), &sha256_bytes(b"golden")),
        );

        match CandleBeliefModel::load(directory.path()) {
            Err(ModelError::InvalidArtifact(message)) => {
                assert!(message.contains("model SHA-256 mismatch"), "{message}");
            }
            other => panic!("expected model SHA mismatch, got {other:?}"),
        }
    }

    #[test]
    fn load_rejects_wrong_golden_sha_before_model_decode() {
        let directory = TestDirectory::new("wrong-golden-sha");
        let model_bytes = b"model";
        fs::write(directory.path().join("model.safetensors"), model_bytes)
            .expect("write model placeholder");
        fs::write(directory.path().join("golden.safetensors"), b"golden")
            .expect("write golden placeholder");
        write_manifest(
            directory.path(),
            &manifest_json(
                &sha256_bytes(model_bytes),
                &sha256_bytes(b"different golden"),
            ),
        );

        match CandleBeliefModel::load(directory.path()) {
            Err(ModelError::InvalidArtifact(message)) => {
                assert!(message.contains("golden SHA-256 mismatch"), "{message}");
            }
            other => panic!("expected golden SHA mismatch, got {other:?}"),
        }
    }
}
