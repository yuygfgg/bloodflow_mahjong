use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

use bloodflow_mahjong::{
    BELIEF_EVENT_HISTORY_LENGTH, BELIEF_FEATURE_SCHEMA_VERSION, BELIEF_PROPOSAL_STREAM_COUNT,
    BELIEF_TARGET_VERSION, BeliefCandidateFeatures, BeliefPublicFeatures, BeliefRootCandidate,
    ENGINE_RULES_VERSION, Game, Phase, RULE_PLANNER_MIN_BELIEF_RESIDUAL_SEARCH_ITERATIONS,
    RulePlannerConfig,
};
use clap::Parser;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use safetensors::tensor::{Dtype, TensorView};
use serde::Serialize;
use sha2::{Digest, Sha256};

const TILE_KIND_COUNT: usize = 27;
const META_OBSERVATION_WIDTH: usize = 34;
const CANDIDATE_WORLD_PLANES: usize = 4;
const MIN_CANDIDATE_COUNT: usize = RULE_PLANNER_MIN_BELIEF_RESIDUAL_SEARCH_ITERATIONS as usize;
const MAX_CANDIDATE_COUNT: usize = 256;
const TRUTH_STREAM: u8 = BELIEF_PROPOSAL_STREAM_COUNT as u8;
const PROPOSAL_STREAM_DOMAINS: [u64; BELIEF_PROPOSAL_STREAM_COUNT] =
    [0xa2f9_6c41_3b77_5d08, 0x49dc_1e8b_f607_a235];

#[derive(Debug, Parser)]
#[command(about)]
struct Args {
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 20_000)]
    roots: usize,
    #[arg(long, default_value_t = 64, value_parser = parse_candidate_count)]
    candidates: usize,
    #[arg(long, default_value_t = 2_048)]
    shard_roots: usize,
    #[arg(long, default_value_t = 20_260_731)]
    seed: u64,
    #[arg(long, default_value_t = 0.15, value_parser = parse_probability)]
    random_action_probability: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Policy {
    Fast,
    Ev,
    Planner,
    NoisyFast,
}

impl Policy {
    const ALL: [Self; 4] = [Self::Fast, Self::Ev, Self::Planner, Self::NoisyFast];

    const fn name(self) -> &'static str {
        match self {
            Self::Fast => "fast",
            Self::Ev => "ev",
            Self::Planner => "planner-fast",
            Self::NoisyFast => "noisy-fast",
        }
    }

    #[cfg(test)]
    const fn index(self) -> usize {
        match self {
            Self::Fast => 0,
            Self::Ev => 1,
            Self::Planner => 2,
            Self::NoisyFast => 3,
        }
    }
}

const GAMES_PER_BLOCK: u64 = Policy::ALL.len() as u64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Split {
    Train,
    Calibration,
    Development,
}

impl Split {
    const COUNT: usize = 3;

    const fn for_block(block: u64) -> Self {
        match block % 10 {
            0..=7 => Self::Train,
            8 => Self::Calibration,
            9 => Self::Development,
            _ => unreachable!(),
        }
    }

    const fn index(self) -> usize {
        match self {
            Self::Train => 0,
            Self::Calibration => 1,
            Self::Development => 2,
        }
    }

    const fn name(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Calibration => "calibration",
            Self::Development => "development",
        }
    }
}

#[derive(Debug, Serialize)]
struct Manifest {
    schema_version: u32,
    belief_target_version: u32,
    engine_rules_version: u32,
    proposal_stream_count: usize,
    candidate_count: usize,
    max_history: usize,
    root_seed: u64,
    random_action_probability: f64,
    policy_families: Vec<&'static str>,
    minimum_roots: usize,
    roots: usize,
    games: u64,
    audit: Audit,
    shards: Vec<ShardManifest>,
}

#[derive(Debug, Serialize)]
struct ShardManifest {
    path: String,
    split: &'static str,
    roots: usize,
    sha256: String,
}

#[derive(Debug, Default, Serialize)]
struct Audit {
    positive_nonfinite: usize,
    proposal_nonfinite: usize,
    proposal_streams_without_finite_weight: usize,
    proposal_collisions: usize,
    positive_collisions: usize,
}

#[derive(Default)]
struct ShardBuilder {
    tile_obs: Vec<u8>,
    melds: Vec<u8>,
    river: Vec<u8>,
    meta: Vec<i32>,
    events: Vec<i32>,
    event_lengths: Vec<u16>,
    candidate_worlds: Vec<u8>,
    handwritten_log_weights: Vec<f32>,
    positive_mask: Vec<u8>,
    proposal_streams: Vec<u8>,
    block_ids: Vec<u64>,
    root_ids: Vec<u64>,
    roots: usize,
}

impl ShardBuilder {
    fn push(
        &mut self,
        public: &BeliefPublicFeatures,
        candidates: &[BeliefRootCandidate],
        positive_mask: &[u8],
        proposal_streams: &[u8],
        block_id: u64,
        root_id: u64,
    ) {
        self.tile_obs.extend_from_slice(&public.tile_observation);
        self.melds.extend_from_slice(&public.melds);
        self.river.extend_from_slice(&public.river);
        self.meta.extend_from_slice(&public.meta);
        self.events.extend_from_slice(&public.events);
        self.event_lengths.push(public.event_len);
        for candidate in candidates {
            append_candidate(&mut self.candidate_worlds, &candidate.features);
            self.handwritten_log_weights
                .push(candidate.handwritten_log_weight as f32);
        }
        self.positive_mask.extend_from_slice(positive_mask);
        self.proposal_streams.extend_from_slice(proposal_streams);
        self.block_ids.push(block_id);
        self.root_ids.push(root_id);
        self.roots += 1;
    }

    fn clear(&mut self) {
        *self = Self::default();
    }
}

struct Collector {
    args: Args,
    output_dir: PathBuf,
    builders: [ShardBuilder; Split::COUNT],
    shard_indices: [usize; Split::COUNT],
    shards: Vec<ShardManifest>,
    audit: Audit,
    collected_roots: usize,
}

impl Collector {
    fn new(args: Args) -> io::Result<Self> {
        if args.roots == 0 || args.shard_roots == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "roots and shard-roots must be positive",
            ));
        }
        prepare_output_dir(&args.output_dir)?;
        Ok(Self {
            output_dir: args.output_dir.clone(),
            args,
            builders: std::array::from_fn(|_| ShardBuilder::default()),
            shard_indices: [0; Split::COUNT],
            shards: Vec::new(),
            audit: Audit::default(),
            collected_roots: 0,
        })
    }

    fn collect_root(
        &mut self,
        game: &Game,
        block_id: u64,
        root_id: u64,
        proposal_seed: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let public = game
            .belief_root_public_features()
            .ok_or("turn root has no public belief features")?;
        let truth = game
            .belief_root_candidate()
            .ok_or("turn root has no authoritative belief candidate")?;
        let expected_proposals = self.args.candidates;
        let mut proposals = Vec::with_capacity(expected_proposals * BELIEF_PROPOSAL_STREAM_COUNT);
        let mut proposal_streams =
            Vec::with_capacity(expected_proposals * BELIEF_PROPOSAL_STREAM_COUNT);
        for (stream, domain) in PROPOSAL_STREAM_DOMAINS.into_iter().enumerate() {
            let stream_seed = mix64(proposal_seed ^ domain);
            let stream_proposals = game
                .belief_root_proposals(stream_seed, expected_proposals)
                .ok_or("turn root cannot sample belief proposals")?;
            if stream_proposals.len() != expected_proposals {
                return Err(format!(
                    "root {root_id} stream {stream} produced {} proposals, expected {expected_proposals}",
                    stream_proposals.len()
                )
                .into());
            }
            self.audit.proposal_nonfinite += stream_proposals
                .iter()
                .filter(|candidate| !candidate.handwritten_log_weight.is_finite())
                .count();
            proposals.extend(stream_proposals);
            proposal_streams.extend(core::iter::repeat_n(stream as u8, expected_proposals));
        }
        let (proposals, positive_mask, proposal_streams) = candidate_group(
            proposals,
            proposal_streams,
            truth,
            mix64(proposal_seed ^ 0x2d91_46e3_5c78_a0bf),
        );

        self.audit.positive_nonfinite += proposals
            .iter()
            .zip(&positive_mask)
            .filter(|(candidate, positive)| {
                **positive != 0 && !candidate.handwritten_log_weight.is_finite()
            })
            .count();
        for stream in 0..BELIEF_PROPOSAL_STREAM_COUNT as u8 {
            self.audit.proposal_streams_without_finite_weight += usize::from(
                proposals
                    .iter()
                    .zip(&proposal_streams)
                    .filter(|(_, candidate_stream)| **candidate_stream == stream)
                    .all(|(candidate, _)| !candidate.handwritten_log_weight.is_finite()),
            );
        }

        let mut multiplicity = BTreeMap::<Vec<u8>, usize>::new();
        for candidate in &proposals {
            *multiplicity
                .entry(candidate_key(&candidate.features))
                .or_default() += 1;
        }
        self.audit.proposal_collisions +=
            multiplicity.values().map(|count| count - 1).sum::<usize>();
        self.audit.positive_collisions += positive_mask
            .iter()
            .map(|&value| usize::from(value))
            .sum::<usize>()
            - 1;

        let split = Split::for_block(block_id);
        let index = split.index();
        self.builders[index].push(
            &public,
            &proposals,
            &positive_mask,
            &proposal_streams,
            block_id,
            root_id,
        );
        self.collected_roots += 1;
        if self.builders[index].roots >= self.args.shard_roots {
            self.flush(split)?;
        }
        Ok(())
    }

    fn flush(&mut self, split: Split) -> Result<(), Box<dyn std::error::Error>> {
        let index = split.index();
        let builder = &self.builders[index];
        if builder.roots == 0 {
            return Ok(());
        }
        let shard_index = self.shard_indices[index];
        let filename = format!("{}-{shard_index:05}.safetensors", split.name());
        let path = self.output_dir.join(&filename);
        write_shard(
            &path,
            builder,
            self.args.candidates,
            BELIEF_PROPOSAL_STREAM_COUNT,
        )?;
        self.shards.push(ShardManifest {
            path: filename,
            split: split.name(),
            roots: builder.roots,
            sha256: sha256(&path)?,
        });
        self.shard_indices[index] += 1;
        self.builders[index].clear();
        Ok(())
    }

    fn finish(mut self, games: u64) -> Result<Manifest, Box<dyn std::error::Error>> {
        self.flush(Split::Train)?;
        self.flush(Split::Calibration)?;
        self.flush(Split::Development)?;
        let manifest = Manifest {
            schema_version: BELIEF_FEATURE_SCHEMA_VERSION,
            belief_target_version: BELIEF_TARGET_VERSION,
            engine_rules_version: ENGINE_RULES_VERSION,
            proposal_stream_count: BELIEF_PROPOSAL_STREAM_COUNT,
            candidate_count: self.args.candidates,
            max_history: BELIEF_EVENT_HISTORY_LENGTH,
            root_seed: self.args.seed,
            random_action_probability: self.args.random_action_probability,
            policy_families: Policy::ALL.into_iter().map(Policy::name).collect(),
            minimum_roots: self.args.roots,
            roots: self.collected_roots,
            games,
            audit: self.audit,
            shards: self.shards,
        };
        let path = self.output_dir.join("manifest.json");
        let temporary = self.output_dir.join("manifest.json.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&manifest)?)?;
        fs::rename(temporary, path)?;
        Ok(manifest)
    }
}

fn prepare_output_dir(path: &Path) -> io::Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
        return Ok(());
    }
    if !path.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("output path is not a directory: {}", path.display()),
        ));
    }
    if fs::read_dir(path)?.next().transpose()?.is_some() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("output directory is not empty: {}", path.display()),
        ));
    }
    Ok(())
}

fn parse_probability(value: &str) -> Result<f64, String> {
    let parsed: f64 = value.parse().map_err(|error| format!("{error}"))?;
    if (0.0..=1.0).contains(&parsed) {
        Ok(parsed)
    } else {
        Err("probability must be in 0..=1".into())
    }
}

fn parse_candidate_count(value: &str) -> Result<usize, String> {
    let parsed: usize = value.parse().map_err(|error| format!("{error}"))?;
    if (MIN_CANDIDATE_COUNT..=MAX_CANDIDATE_COUNT).contains(&parsed) {
        Ok(parsed)
    } else {
        Err(format!(
            "candidate count must be in {MIN_CANDIDATE_COUNT}..={MAX_CANDIDATE_COUNT}"
        ))
    }
}

fn append_candidate(output: &mut Vec<u8>, candidate: &BeliefCandidateFeatures) {
    for hand in &candidate.opponent_concealed {
        output.extend_from_slice(hand);
    }
    output.extend_from_slice(&candidate.live_wall);
}

fn candidate_group(
    mut proposals: Vec<BeliefRootCandidate>,
    mut proposal_streams: Vec<u8>,
    truth: BeliefRootCandidate,
    shuffle_seed: u64,
) -> (Vec<BeliefRootCandidate>, Vec<u8>, Vec<u8>) {
    let truth_features = truth.features.clone();
    proposals.push(truth);
    proposal_streams.push(TRUTH_STREAM);
    assert_eq!(proposals.len(), proposal_streams.len());
    let mut shuffle = ChaCha8Rng::seed_from_u64(shuffle_seed);
    let mut paired: Vec<_> = proposals.into_iter().zip(proposal_streams).collect();
    paired.shuffle(&mut shuffle);
    let (proposals, proposal_streams): (Vec<_>, Vec<_>) = paired.into_iter().unzip();
    let positive_mask = proposals
        .iter()
        .map(|candidate| u8::from(candidate.features == truth_features))
        .collect();
    (proposals, positive_mask, proposal_streams)
}

fn candidate_key(candidate: &BeliefCandidateFeatures) -> Vec<u8> {
    let mut key = Vec::with_capacity(CANDIDATE_WORLD_PLANES * TILE_KIND_COUNT);
    append_candidate(&mut key, candidate);
    key
}

fn policy_action(
    game: &Game,
    policy: Policy,
    random_probability: f64,
    random: &mut ChaCha8Rng,
) -> Option<bloodflow_mahjong::ActionId> {
    if matches!(policy, Policy::NoisyFast) && random.random::<f64>() < random_probability {
        let legal = game.legal_action_mask()?;
        let selected = random.random_range(0..legal.count_ones() as usize);
        return legal.iter().nth(selected);
    }
    match policy {
        Policy::Fast => game.simple_rule_action(),
        Policy::Ev => game.rule_ev_action(),
        Policy::Planner => {
            let config = RulePlannerConfig::FAST
                .with_draw_horizon(1)?
                .with_candidate_states(1)?;
            game.rule_planner_action_with_config(config)
        }
        Policy::NoisyFast => game.simple_rule_action(),
    }
}

fn policy_for_game_seat(game_index: u64, seat_index: usize) -> Policy {
    Policy::ALL[(seat_index + game_index as usize) % Policy::ALL.len()]
}

fn is_belief_training_root(phase: Phase, can_hu: bool, legal_action_count: u32) -> bool {
    phase == Phase::Turn && !can_hu && legal_action_count > 1
}

fn write_shard(
    path: &Path,
    builder: &ShardBuilder,
    candidates: usize,
    proposal_stream_count: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let roots = builder.roots;
    let group_count = candidates
        .checked_mul(proposal_stream_count)
        .and_then(|value| value.checked_add(1))
        .ok_or("candidate group count overflow")?;
    let meta = encode_i32(&builder.meta);
    let events = encode_i32(&builder.events);
    let event_lengths = encode_u16(&builder.event_lengths);
    let weights = encode_f32(&builder.handwritten_log_weights);
    let block_ids = encode_u64(&builder.block_ids);
    let root_ids = encode_u64(&builder.root_ids);
    let entries = [
        view(
            "tile_obs",
            Dtype::U8,
            vec![roots, 10, TILE_KIND_COUNT],
            &builder.tile_obs,
        )?,
        view("melds", Dtype::U8, vec![roots, 4, 4, 3], &builder.melds)?,
        view("river", Dtype::U8, vec![roots, 108, 2], &builder.river)?,
        view(
            "meta",
            Dtype::I32,
            vec![roots, META_OBSERVATION_WIDTH],
            &meta,
        )?,
        view(
            "events",
            Dtype::I32,
            vec![roots, BELIEF_EVENT_HISTORY_LENGTH, 8],
            &events,
        )?,
        view("event_lengths", Dtype::U16, vec![roots], &event_lengths)?,
        view(
            "candidate_worlds",
            Dtype::U8,
            vec![roots, group_count, CANDIDATE_WORLD_PLANES, TILE_KIND_COUNT],
            &builder.candidate_worlds,
        )?,
        view(
            "handwritten_log_weights",
            Dtype::F32,
            vec![roots, group_count],
            &weights,
        )?,
        view(
            "positive_mask",
            Dtype::U8,
            vec![roots, group_count],
            &builder.positive_mask,
        )?,
        view(
            "proposal_streams",
            Dtype::U8,
            vec![roots, group_count],
            &builder.proposal_streams,
        )?,
        view("block_ids", Dtype::U64, vec![roots], &block_ids)?,
        view("root_ids", Dtype::U64, vec![roots], &root_ids)?,
    ];
    safetensors::serialize_to_file(entries, Some(shard_metadata()), path)?;
    Ok(())
}

fn shard_metadata() -> HashMap<String, String> {
    HashMap::from([
        (
            "belief_schema_version".into(),
            BELIEF_FEATURE_SCHEMA_VERSION.to_string(),
        ),
        (
            "belief_target_version".into(),
            BELIEF_TARGET_VERSION.to_string(),
        ),
        (
            "engine_rules_version".into(),
            ENGINE_RULES_VERSION.to_string(),
        ),
        (
            "proposal_stream_count".into(),
            BELIEF_PROPOSAL_STREAM_COUNT.to_string(),
        ),
    ])
}

fn view<'a>(
    name: &'static str,
    dtype: Dtype,
    shape: Vec<usize>,
    data: &'a [u8],
) -> Result<(&'static str, TensorView<'a>), safetensors::SafeTensorError> {
    Ok((name, TensorView::new(dtype, shape, data)?))
}

fn encode_i32(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn encode_u16(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn encode_u64(values: &[u64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn sha256(path: &Path) -> io::Result<String> {
    let mut digest = Sha256::new();
    digest.update(fs::read(path)?);
    Ok(format!("{:x}", digest.finalize()))
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let target_roots = args.roots;
    let seed = args.seed;
    let started = Instant::now();
    let mut collector = Collector::new(args)?;
    let mut game_index = 0_u64;
    while collector.collected_roots < target_roots {
        let block_id = game_index / GAMES_PER_BLOCK;
        let block_end = game_index + GAMES_PER_BLOCK;
        while game_index < block_end {
            let game_seed = mix64(seed.wrapping_add(game_index));
            let mut game = Game::new(game_seed);
            let mut random = ChaCha8Rng::seed_from_u64(game_seed ^ 0x4c2e_9187_aa75_3d10);
            let mut decision_index = 0_u64;
            while game.phase() != Phase::Finished {
                let decision = game.decision().ok_or("active game has no decision")?;
                let legal = game
                    .legal_action_mask()
                    .ok_or("active game has no legal mask")?;
                let can_hu = game
                    .legal_actions()
                    .ok_or("active game has no legal actions")?
                    .can_hu;
                if is_belief_training_root(decision.phase, can_hu, legal.count_ones()) {
                    let root_id = (game_index << 32) | decision_index;
                    let proposal_seed = mix64(seed ^ root_id ^ 0xb816_aa91_c542_073d);
                    collector.collect_root(&game, block_id, root_id, proposal_seed)?;
                    if collector.collected_roots % 1_000 == 0 {
                        eprintln!(
                            "roots {} target >= {} games {} elapsed {:.1}s",
                            collector.collected_roots,
                            target_roots,
                            game_index + 1,
                            started.elapsed().as_secs_f64(),
                        );
                    }
                }
                let policy = policy_for_game_seat(game_index, decision.actor.index());
                let action = policy_action(
                    &game,
                    policy,
                    collector.args.random_action_probability,
                    &mut random,
                )
                .ok_or("policy returned no action in an active game")?;
                game.step_id(action)?;
                decision_index += 1;
            }
            game_index += 1;
        }
    }
    let manifest = collector.finish(game_index)?;
    println!(
        "RESULT roots {} games {} shards {} elapsed {:.1}s positive_nonfinite {} proposal_nonfinite {}",
        manifest.roots,
        manifest.games,
        manifest.shards.len(),
        started.elapsed().as_secs_f64(),
        manifest.audit.positive_nonfinite,
        manifest.audit.proposal_nonfinite,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, fs, io};

    use bloodflow_mahjong::{BeliefCandidateFeatures, BeliefRootCandidate};

    use super::{
        GAMES_PER_BLOCK, MAX_CANDIDATE_COUNT, MIN_CANDIDATE_COUNT, Policy, Split, TRUTH_STREAM,
        candidate_group, is_belief_training_root, parse_candidate_count, policy_for_game_seat,
        prepare_output_dir, shard_metadata,
    };
    use bloodflow_mahjong::{
        BELIEF_FEATURE_SCHEMA_VERSION, BELIEF_PROPOSAL_STREAM_COUNT, BELIEF_TARGET_VERSION,
        ENGINE_RULES_VERSION, Phase,
    };

    fn candidate(tile: usize) -> BeliefRootCandidate {
        let mut live_wall = [0; 27];
        live_wall[tile] = 1;
        BeliefRootCandidate {
            features: BeliefCandidateFeatures {
                opponent_concealed: [[0; 27]; 3],
                live_wall,
            },
            handwritten_log_weight: -(tile as f64),
        }
    }

    #[test]
    fn split_assignment_reserves_complete_blocks_for_calibration_and_development() {
        for cycle in 0..4 {
            let offset = cycle * 10;
            for block in offset..offset + 8 {
                assert_eq!(Split::for_block(block), Split::Train);
            }
            assert_eq!(Split::for_block(offset + 8), Split::Calibration);
            assert_eq!(Split::for_block(offset + 9), Split::Development);
        }
    }

    #[test]
    fn hu_turns_are_outside_the_belief_training_root_contract() {
        assert!(!is_belief_training_root(Phase::Turn, true, 2));
        assert!(!is_belief_training_root(Phase::Turn, false, 1));
        assert!(is_belief_training_root(Phase::Turn, false, 2));
        assert!(!is_belief_training_root(Phase::MeldResponse, false, 2));
    }

    #[test]
    fn candidate_count_matches_the_deployable_search_budget() {
        assert_eq!(
            parse_candidate_count(&MIN_CANDIDATE_COUNT.to_string()),
            Ok(MIN_CANDIDATE_COUNT),
        );
        assert!(parse_candidate_count(&(MIN_CANDIDATE_COUNT - 1).to_string()).is_err());
        assert_eq!(
            parse_candidate_count(&MAX_CANDIDATE_COUNT.to_string()),
            Ok(MAX_CANDIDATE_COUNT),
        );
        assert!(parse_candidate_count(&(MAX_CANDIDATE_COUNT + 1).to_string()).is_err());
    }

    #[test]
    fn four_game_blocks_balance_every_policy_at_every_seat() {
        for block in 0..5 {
            for seat in 0..Policy::ALL.len() {
                let mut counts = [0; Policy::ALL.len()];
                for game_offset in 0..GAMES_PER_BLOCK {
                    let game_index = block * GAMES_PER_BLOCK + game_offset;
                    counts[policy_for_game_seat(game_index, seat).index()] += 1;
                }
                assert_eq!(counts, [1; Policy::ALL.len()]);
            }
        }
        assert_eq!(GAMES_PER_BLOCK, Policy::ALL.len() as u64);
    }

    #[test]
    fn candidate_shuffle_is_root_local_and_marks_the_full_equivalence_class() {
        let proposals = vec![candidate(0), candidate(1), candidate(2)];
        let truth = candidate(0);
        let streams = vec![0, 0, 1];
        let first = candidate_group(proposals.clone(), streams.clone(), truth.clone(), 17);
        let repeated = candidate_group(proposals.clone(), streams.clone(), truth.clone(), 17);

        assert_eq!(first, repeated);
        assert_eq!(
            first
                .1
                .iter()
                .map(|&value| usize::from(value))
                .sum::<usize>(),
            2
        );
        for (candidate, &positive) in first.0.iter().zip(&first.1) {
            assert_eq!(positive != 0, candidate.features == truth.features);
        }
        assert!(first.2.contains(&TRUTH_STREAM));

        let target_positions: BTreeSet<_> = (0..16)
            .map(|seed| {
                let (_, mask, _) =
                    candidate_group(proposals.clone(), streams.clone(), truth.clone(), seed);
                mask.iter().position(|&value| value != 0).unwrap()
            })
            .collect();
        assert!(target_positions.len() > 1);
    }

    #[test]
    fn output_directory_must_be_absent_or_empty() -> io::Result<()> {
        let temporary = tempfile::tempdir()?;
        let output = temporary.path().join("dataset");

        prepare_output_dir(&output)?;
        prepare_output_dir(&output)?;
        fs::write(output.join("stale.safetensors"), b"stale")?;

        let error = prepare_output_dir(&output).expect_err("non-empty output must be rejected");
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert!(error.to_string().contains("not empty"));
        Ok(())
    }

    #[test]
    fn shard_metadata_records_every_semantic_contract_version() {
        let metadata = shard_metadata();

        assert_eq!(
            metadata.get("belief_schema_version"),
            Some(&BELIEF_FEATURE_SCHEMA_VERSION.to_string()),
        );
        assert_eq!(
            metadata.get("belief_target_version"),
            Some(&BELIEF_TARGET_VERSION.to_string()),
        );
        assert_eq!(
            metadata.get("engine_rules_version"),
            Some(&ENGINE_RULES_VERSION.to_string()),
        );
        assert_eq!(
            metadata.get("proposal_stream_count"),
            Some(&BELIEF_PROPOSAL_STREAM_COUNT.to_string()),
        );
    }
}
