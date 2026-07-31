//! Balanced tournament evaluation for deterministic rule policies.

use std::num::NonZeroUsize;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use bloodflow_mahjong::{
    Action, ActionId, Game, LegalActions, Phase, RuleEvConfig, RuleEvDefense,
    RulePlannerAnalysisOptions, RulePlannerConfig, RulePlannerContinuation,
    RulePlannerContinuationPolicy, RulePlannerContinuationProfile, RulePlannerRootBelief,
    RulePlannerSearchAnalysis, RulePlannerSearchOutcome, Seat, reset_rule_planner_search_stats,
    rule_planner_search_stats,
};
use clap::{Parser, ValueEnum, error::ErrorKind};
use rand::{Rng as _, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

const DEFAULT_BLOCKS: NonZeroUsize = NonZeroUsize::new(4_096).unwrap();
const DEFAULT_ROOT_SEED: u64 = 20_260_729;
const DEFAULT_BOOTSTRAP_SAMPLES: NonZeroUsize = NonZeroUsize::new(2_000).unwrap();
const MAX_ACTIONS_PER_GAME: usize = 4_096;
const ELO_SCALE: f64 = 400.0 / std::f64::consts::LN_10;
const PL_RIDGE: f64 = 1e-8;
const PL_ITERATIONS: usize = 32;
const PL_TOLERANCE: f64 = 1e-10;
const ELO_ZERO_TOLERANCE: f64 = 1e-8;
const BOOTSTRAP_DOMAIN: u64 = 0x3c6e_f372_fe94_f82b;

// Every two-versus-two assignment appears once. Within a seed block, each
// policy controls each seat three times.
const POLICY_A_SEAT_MASKS: [u8; 6] = [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100];

// Rank-order patterns with exactly two policy-A players. Bit zero is first
// place, bit one is second place, and so on.
const RANK_PATTERNS: [u8; 6] = [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100];

#[derive(Clone, Debug, Eq, Parser, PartialEq)]
#[command(
    name = "rule-tournament",
    about = "Run balanced two-policy tournament blocks."
)]
struct Config {
    #[arg(long, default_value_t = DEFAULT_BLOCKS)]
    blocks: NonZeroUsize,
    #[arg(long, default_value_t = DEFAULT_ROOT_SEED)]
    root_seed: u64,
    #[arg(long, default_value_t = DEFAULT_BOOTSTRAP_SAMPLES)]
    bootstrap_samples: NonZeroUsize,
    /// Maximum games evaluated concurrently. Nested search defaults to one.
    #[arg(long)]
    parallel_games: Option<NonZeroUsize>,
    #[arg(long, value_enum, default_value = "rule-ev")]
    policy_a: PolicyKind,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..=3))]
    a_lookahead_depth: u8,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..=2))]
    a_hand_changes: u8,
    #[arg(long, default_value_t = 27, value_parser = clap::value_parser!(u8).range(0..=32))]
    a_draw_horizon: u8,
    #[arg(long, default_value_t = 4_096, value_parser = clap::value_parser!(u32).range(1..=200_000))]
    a_candidate_states: u32,
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u16).range(0..=256))]
    a_belief_worlds: u16,
    #[arg(long, value_enum, default_value = "posterior")]
    a_root_belief: PlannerRootBelief,
    #[arg(long, value_enum, default_value = "current")]
    a_continuation: PlannerContinuation,
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u16).range(0..=256))]
    a_response_worlds: u16,
    #[arg(
        long,
        visible_alias = "a-search-budget",
        default_value_t = 0,
        value_parser = clap::value_parser!(u16).range(0..=4_096)
    )]
    a_search_iterations: u16,
    #[arg(long, value_enum, default_value = "heuristic")]
    a_defense: Defense,
    #[arg(long, value_enum, default_value = "rule-fast")]
    policy_b: PolicyKind,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..=3))]
    b_lookahead_depth: u8,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..=2))]
    b_hand_changes: u8,
    #[arg(long, default_value_t = 27, value_parser = clap::value_parser!(u8).range(0..=32))]
    b_draw_horizon: u8,
    #[arg(long, default_value_t = 4_096, value_parser = clap::value_parser!(u32).range(1..=200_000))]
    b_candidate_states: u32,
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u16).range(0..=256))]
    b_belief_worlds: u16,
    #[arg(long, value_enum, default_value = "posterior")]
    b_root_belief: PlannerRootBelief,
    #[arg(long, value_enum, default_value = "current")]
    b_continuation: PlannerContinuation,
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u16).range(0..=256))]
    b_response_worlds: u16,
    #[arg(
        long,
        visible_alias = "b-search-budget",
        default_value_t = 0,
        value_parser = clap::value_parser!(u16).range(0..=4_096)
    )]
    b_search_iterations: u16,
    #[arg(long, value_enum, default_value = "heuristic")]
    b_defense: Defense,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PolicyKind {
    #[value(name = "rule-fast")]
    Fast,
    #[value(name = "rule-ev")]
    Ev,
    #[value(name = "rule-planner")]
    Planner,
}

impl PolicyKind {
    const fn name(self) -> &'static str {
        match self {
            Self::Fast => "rule-fast",
            Self::Ev => "rule-ev",
            Self::Planner => "rule-planner",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Defense {
    None,
    Heuristic,
}

impl From<Defense> for RuleEvDefense {
    fn from(value: Defense) -> Self {
        match value {
            Defense::None => Self::None,
            Defense::Heuristic => Self::Heuristic,
        }
    }
}

impl Defense {
    const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Heuristic => "heuristic",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum PlannerRootBelief {
    #[default]
    Posterior,
    Uniform,
    OracleHidden,
}

impl PlannerRootBelief {
    const fn name_suffix(self) -> &'static str {
        match self {
            Self::Posterior => "",
            Self::Uniform => "_uniform",
            Self::OracleHidden => "_oracle_hidden",
        }
    }
}

impl From<PlannerRootBelief> for RulePlannerRootBelief {
    fn from(value: PlannerRootBelief) -> Self {
        match value {
            PlannerRootBelief::Posterior => Self::Posterior,
            PlannerRootBelief::Uniform => Self::Uniform,
            PlannerRootBelief::OracleHidden => Self::OracleHidden,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum PlannerContinuation {
    #[default]
    Current,
    OracleContinuation,
}

impl PlannerContinuation {
    const fn name_suffix(self) -> &'static str {
        match self {
            Self::Current => "",
            Self::OracleContinuation => "_oracle_continuation",
        }
    }
}

#[derive(Clone, Debug)]
enum Policy {
    Fast,
    Ev(RuleEvConfig),
    Planner {
        config: RulePlannerConfig,
        root_belief: PlannerRootBelief,
        continuation: PlannerContinuation,
    },
}

#[derive(Clone, Copy, Debug)]
struct PolicyDecision {
    action: ActionId,
    search: Option<RulePlannerSearchAnalysis>,
}

impl PolicyDecision {
    const fn without_search(action: ActionId) -> Self {
        Self {
            action,
            search: None,
        }
    }
}

impl Policy {
    fn continuation_policy(&self) -> RulePlannerContinuationPolicy {
        match self {
            Self::Fast => RulePlannerContinuationPolicy::Fast,
            Self::Ev(config) => RulePlannerContinuationPolicy::Ev(*config),
            Self::Planner { config, .. } => RulePlannerContinuationPolicy::PlannerBaseline(*config),
        }
    }

    fn decide(
        &self,
        game: &Game,
        continuation_profile: RulePlannerContinuationProfile,
    ) -> Option<PolicyDecision> {
        match self {
            Self::Fast => game
                .simple_rule_action()
                .map(PolicyDecision::without_search),
            Self::Ev(config) => game
                .rule_ev_action_with_config(*config)
                .map(PolicyDecision::without_search),
            Self::Planner {
                config,
                root_belief,
                continuation,
            } => {
                let continuation = match continuation {
                    PlannerContinuation::Current => RulePlannerContinuation::Current,
                    PlannerContinuation::OracleContinuation => {
                        RulePlannerContinuation::KnownPolicies(continuation_profile)
                    }
                };
                let options = RulePlannerAnalysisOptions::new((*root_belief).into())
                    .with_continuation(continuation);
                game.rule_planner_analysis_with_options(*config, options)
                    .map(|analysis| PolicyDecision {
                        action: analysis.action(),
                        search: analysis.search(),
                    })
            }
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            blocks: DEFAULT_BLOCKS,
            root_seed: DEFAULT_ROOT_SEED,
            bootstrap_samples: DEFAULT_BOOTSTRAP_SAMPLES,
            parallel_games: None,
            policy_a: PolicyKind::Ev,
            a_lookahead_depth: RuleEvConfig::STANDARD.search_depth(),
            a_hand_changes: RulePlannerConfig::STANDARD.hand_changes(),
            a_draw_horizon: RulePlannerConfig::STANDARD.draw_horizon(),
            a_candidate_states: RulePlannerConfig::STANDARD.candidate_states(),
            a_belief_worlds: RulePlannerConfig::STANDARD.belief_worlds(),
            a_root_belief: PlannerRootBelief::Posterior,
            a_continuation: PlannerContinuation::Current,
            a_response_worlds: RulePlannerConfig::STANDARD.response_worlds(),
            a_search_iterations: RulePlannerConfig::STANDARD.search_iterations(),
            a_defense: Defense::Heuristic,
            policy_b: PolicyKind::Fast,
            b_lookahead_depth: RuleEvConfig::STANDARD.search_depth(),
            b_hand_changes: RulePlannerConfig::STANDARD.hand_changes(),
            b_draw_horizon: RulePlannerConfig::STANDARD.draw_horizon(),
            b_candidate_states: RulePlannerConfig::STANDARD.candidate_states(),
            b_belief_worlds: RulePlannerConfig::STANDARD.belief_worlds(),
            b_root_belief: PlannerRootBelief::Posterior,
            b_continuation: PlannerContinuation::Current,
            b_response_worlds: RulePlannerConfig::STANDARD.response_worlds(),
            b_search_iterations: RulePlannerConfig::STANDARD.search_iterations(),
            b_defense: Defense::Heuristic,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GameResult {
    policy_a_seat_mask: u8,
    ranks: [u8; 4],
    score_deltas: [i64; 4],
    actions: u32,
    decisions: [DecisionStats; 2],
    searches: [PlannerSearchStats; 2],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct PlannerSearchStats {
    decisions: u64,
    proposals: u64,
    validation_rejections: u64,
    overrides: u64,
    rollouts: u64,
}

impl PlannerSearchStats {
    fn observe(&mut self, search: Option<RulePlannerSearchAnalysis>) {
        let Some(search) = search else {
            return;
        };
        self.decisions += 1;
        self.rollouts += search.rollouts();
        match search.outcome() {
            RulePlannerSearchOutcome::NoProposal => {}
            RulePlannerSearchOutcome::Rejected(_) => {
                self.proposals += 1;
                self.validation_rejections += 1;
            }
            RulePlannerSearchOutcome::Accepted(_) => {
                self.proposals += 1;
                self.overrides += 1;
            }
        }
    }

    fn merge(&mut self, other: Self) {
        self.decisions += other.decisions;
        self.proposals += other.proposals;
        self.validation_rejections += other.validation_rejections;
        self.overrides += other.overrides;
        self.rollouts += other.rollouts;
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DecisionStats {
    turns: u64,
    turn_hu_available: u64,
    turn_hu_taken: u64,
    concealed_kong_available: u64,
    concealed_kong_taken: u64,
    added_kong_available: u64,
    added_kong_taken: u64,
    hu_responses: u64,
    hu_response_taken: u64,
    meld_responses: u64,
    pong_available: u64,
    pong_taken: u64,
    exposed_kong_available: u64,
    exposed_kong_taken: u64,
    response_passes: u64,
}

impl DecisionStats {
    fn observe(&mut self, legal: &LegalActions, action: Action) {
        match legal.decision.phase {
            Phase::Turn => {
                self.turns += 1;
                self.turn_hu_available += u64::from(legal.can_hu);
                self.turn_hu_taken += u64::from(matches!(action, Action::Hu));
                self.concealed_kong_available += u64::from(legal.concealed_kong_mask != 0);
                self.concealed_kong_taken += u64::from(matches!(action, Action::ConcealedKong(_)));
                self.added_kong_available += u64::from(legal.added_kong_mask != 0);
                self.added_kong_taken += u64::from(matches!(action, Action::AddedKong(_)));
            }
            Phase::HuResponse => {
                self.hu_responses += 1;
                self.hu_response_taken += u64::from(matches!(action, Action::Hu));
                self.response_passes += u64::from(matches!(action, Action::Pass));
            }
            Phase::MeldResponse => {
                self.meld_responses += 1;
                self.pong_available += u64::from(legal.can_pong);
                self.pong_taken += u64::from(matches!(action, Action::Pong));
                self.exposed_kong_available += u64::from(legal.can_exposed_kong);
                self.exposed_kong_taken += u64::from(matches!(action, Action::ExposedKong));
                self.response_passes += u64::from(matches!(action, Action::Pass));
            }
            Phase::Exchange | Phase::ChooseMissing | Phase::Finished => {}
        }
    }

    fn merge(&mut self, other: Self) {
        self.turns += other.turns;
        self.turn_hu_available += other.turn_hu_available;
        self.turn_hu_taken += other.turn_hu_taken;
        self.concealed_kong_available += other.concealed_kong_available;
        self.concealed_kong_taken += other.concealed_kong_taken;
        self.added_kong_available += other.added_kong_available;
        self.added_kong_taken += other.added_kong_taken;
        self.hu_responses += other.hu_responses;
        self.hu_response_taken += other.hu_response_taken;
        self.meld_responses += other.meld_responses;
        self.pong_available += other.pong_available;
        self.pong_taken += other.pong_taken;
        self.exposed_kong_available += other.exposed_kong_available;
        self.exposed_kong_taken += other.exposed_kong_taken;
        self.response_passes += other.response_passes;
    }
}

#[derive(Clone, Debug)]
struct BlockResult {
    games: [GameResult; POLICY_A_SEAT_MASKS.len()],
    rank_pattern_counts: [u8; RANK_PATTERNS.len()],
}

#[derive(Clone, Copy, Debug)]
struct PolicySummary {
    seat_games: u64,
    mean_rank: f64,
    mean_score_delta: f64,
    first_rate: f64,
    last_rate: f64,
}

#[derive(Clone, Copy, Debug)]
struct TournamentSummary {
    elo_like_delta: f64,
    uncertainty: Option<TournamentUncertainty>,
    cross_policy_win_rate: f64,
    policy_a: PolicySummary,
    policy_b: PolicySummary,
}

#[derive(Clone, Copy, Debug)]
struct TournamentUncertainty {
    ci95: [f64; 2],
    stronger_probability: f64,
}

#[derive(Debug, Default)]
struct TournamentProgress {
    active_games: AtomicUsize,
    completed_games: AtomicUsize,
    actions: AtomicU64,
}

struct ActiveGame<'a> {
    progress: Option<&'a TournamentProgress>,
}

impl<'a> ActiveGame<'a> {
    fn new(progress: Option<&'a TournamentProgress>) -> Self {
        if let Some(progress) = progress {
            progress.active_games.fetch_add(1, Ordering::Relaxed);
        }
        Self { progress }
    }

    fn complete(mut self) {
        if let Some(progress) = self.progress.take() {
            progress.active_games.fetch_sub(1, Ordering::Relaxed);
            progress.completed_games.fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl Drop for ActiveGame<'_> {
    fn drop(&mut self) {
        if let Some(progress) = self.progress {
            progress.active_games.fetch_sub(1, Ordering::Relaxed);
        }
    }
}

struct MonitorCompletion<'a> {
    finished: &'a AtomicBool,
    monitor: std::thread::Thread,
}

impl<'a> MonitorCompletion<'a> {
    fn new(finished: &'a AtomicBool, monitor: std::thread::Thread) -> Self {
        Self { finished, monitor }
    }
}

impl Drop for MonitorCompletion<'_> {
    fn drop(&mut self) {
        self.finished.store(true, Ordering::Release);
        self.monitor.unpark();
    }
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn play_game(
    seed: u64,
    policy_a_seat_mask: u8,
    policy_a: &Policy,
    policy_b: &Policy,
) -> GameResult {
    play_game_with_progress(seed, policy_a_seat_mask, policy_a, policy_b, None)
}

fn continuation_profile(
    policy_a_seat_mask: u8,
    policy_a: &Policy,
    policy_b: &Policy,
) -> RulePlannerContinuationProfile {
    RulePlannerContinuationProfile::new(Seat::ALL.map(|seat| {
        if policy_a_seat_mask & (1 << seat.index()) != 0 {
            policy_a.continuation_policy()
        } else {
            policy_b.continuation_policy()
        }
    }))
}

fn play_game_with_progress(
    seed: u64,
    policy_a_seat_mask: u8,
    policy_a: &Policy,
    policy_b: &Policy,
    progress: Option<&TournamentProgress>,
) -> GameResult {
    debug_assert_eq!(policy_a_seat_mask.count_ones(), 2);
    let active_game = ActiveGame::new(progress);
    let mut game = Game::new(seed);
    let continuation_profile = continuation_profile(policy_a_seat_mask, policy_a, policy_b);
    let initial_scores = Seat::ALL.map(|seat| game.score(seat));
    let mut decisions = [DecisionStats::default(); 2];
    let mut searches = [PlannerSearchStats::default(); 2];

    for action_index in 0..MAX_ACTIONS_PER_GAME {
        let legal = game
            .legal_actions()
            .expect("a non-terminal game always has a decision");
        let policy_a_controls_actor = policy_a_seat_mask & (1 << legal.decision.actor.index()) != 0;
        let decision = if policy_a_controls_actor {
            policy_a.decide(&game, continuation_profile)
        } else {
            policy_b.decide(&game, continuation_profile)
        }
        .expect("a non-terminal rule policy always returns an action");
        let policy_index = if policy_a_controls_actor { 0 } else { 1 };
        decisions[policy_index].observe(&legal, decision.action.action());
        searches[policy_index].observe(decision.search);
        let outcome = game.step_id(decision.action).unwrap_or_else(|error| {
            panic!(
                "rule policy selected an illegal action: seed={seed}, mask={policy_a_seat_mask:#06b}, error={error}"
            )
        });
        if let Some(progress) = progress {
            progress.actions.fetch_add(1, Ordering::Relaxed);
        }
        if !outcome.terminal {
            continue;
        }

        let mut ranks = [0_u8; 4];
        for (index, seat) in game.rankings().into_iter().enumerate() {
            ranks[seat.index()] = (index + 1) as u8;
        }
        let score_deltas = Seat::ALL.map(|seat| game.score(seat) - initial_scores[seat.index()]);
        let result = GameResult {
            policy_a_seat_mask,
            ranks,
            score_deltas,
            actions: (action_index + 1) as u32,
            decisions,
            searches,
        };
        active_game.complete();
        return result;
    }

    panic!("game exceeded the action limit: seed={seed}, mask={policy_a_seat_mask:#06b}");
}

fn rank_pattern(game: &GameResult) -> u8 {
    let mut pattern = 0_u8;
    for seat in 0..4 {
        if game.policy_a_seat_mask & (1 << seat) != 0 {
            pattern |= 1 << (game.ranks[seat] - 1);
        }
    }
    pattern
}

fn pattern_index(pattern: u8) -> usize {
    RANK_PATTERNS
        .iter()
        .position(|&candidate| candidate == pattern)
        .expect("two-versus-two results have a known rank pattern")
}

fn summarize_block(games: [GameResult; POLICY_A_SEAT_MASKS.len()]) -> BlockResult {
    let mut rank_pattern_counts = [0_u8; RANK_PATTERNS.len()];
    for game in &games {
        rank_pattern_counts[pattern_index(rank_pattern(game))] += 1;
    }
    BlockResult {
        games,
        rank_pattern_counts,
    }
}

fn policy_a_probability(rating: f64, remaining_a: u32, remaining_b: u32) -> f64 {
    match (remaining_a, remaining_b) {
        (0, _) => 0.0,
        (_, 0) => 1.0,
        _ => {
            let log_odds = rating + f64::from(remaining_a).ln() - f64::from(remaining_b).ln();
            if log_odds >= 0.0 {
                1.0 / (1.0 + (-log_odds).exp())
            } else {
                let odds = log_odds.exp();
                odds / (1.0 + odds)
            }
        }
    }
}

fn fit_pl_rating(pattern_counts: &[u64; RANK_PATTERNS.len()]) -> f64 {
    let mut rating = 0.0_f64;
    for _ in 0..PL_ITERATIONS {
        let mut gradient = -PL_RIDGE * rating;
        let mut hessian = -PL_RIDGE;
        for (&pattern, &count) in RANK_PATTERNS.iter().zip(pattern_counts) {
            if count == 0 {
                continue;
            }
            let weight = count as f64;
            for position in 0..3 {
                let remaining = pattern >> position;
                let remaining_a = remaining.count_ones();
                let remaining_b = 4 - position as u32 - remaining_a;
                let probability = policy_a_probability(rating, remaining_a, remaining_b);
                let winner_is_a = f64::from((remaining & 1) != 0);
                gradient += weight * (winner_is_a - probability);
                hessian -= weight * probability * (1.0 - probability);
            }
        }
        let step = gradient / hessian;
        rating -= step;
        if step.abs() < PL_TOLERANCE {
            break;
        }
    }
    rating
}

fn aggregate_patterns(blocks: &[BlockResult]) -> [u64; RANK_PATTERNS.len()] {
    let mut totals = [0_u64; RANK_PATTERNS.len()];
    for block in blocks {
        for (total, &count) in totals.iter_mut().zip(&block.rank_pattern_counts) {
            *total += u64::from(count);
        }
    }
    totals
}

fn bootstrap_ratings(blocks: &[BlockResult], samples: usize, seed: u64) -> Vec<f64> {
    (0..samples)
        .into_par_iter()
        .map(|sample| {
            let mut random = ChaCha8Rng::seed_from_u64(mix64(seed.wrapping_add(sample as u64)));
            let mut counts = [0_u64; RANK_PATTERNS.len()];
            for _ in 0..blocks.len() {
                let selected = &blocks[random.random_range(0..blocks.len())];
                for (total, &count) in counts.iter_mut().zip(&selected.rank_pattern_counts) {
                    *total += u64::from(count);
                }
            }
            fit_pl_rating(&counts) * ELO_SCALE
        })
        .collect()
}

fn quantile_sorted(values: &[f64], probability: f64) -> f64 {
    assert!(!values.is_empty());
    assert!((0.0..=1.0).contains(&probability));
    let index = probability * (values.len() - 1) as f64;
    let low = index.floor() as usize;
    let high = index.ceil() as usize;
    let fraction = index - low as f64;
    values[low] * (1.0 - fraction) + values[high] * fraction
}

fn positive_probability(values: &[f64]) -> f64 {
    let positive_mass: f64 = values
        .iter()
        .map(|&value| {
            if value > ELO_ZERO_TOLERANCE {
                1.0
            } else if value < -ELO_ZERO_TOLERANCE {
                0.0
            } else {
                0.5
            }
        })
        .sum();
    positive_mass / values.len() as f64
}

fn summarize_policy(blocks: &[BlockResult], policy_a: bool) -> PolicySummary {
    let mut seat_games = 0_u64;
    let mut rank_sum = 0_u64;
    let mut score_sum = 0_i128;
    let mut firsts = 0_u64;
    let mut lasts = 0_u64;
    for game in blocks.iter().flat_map(|block| &block.games) {
        for seat in 0..4 {
            if (game.policy_a_seat_mask & (1 << seat) != 0) != policy_a {
                continue;
            }
            let rank = game.ranks[seat];
            seat_games += 1;
            rank_sum += u64::from(rank);
            score_sum += i128::from(game.score_deltas[seat]);
            firsts += u64::from(rank == 1);
            lasts += u64::from(rank == 4);
        }
    }
    let denominator = seat_games as f64;
    PolicySummary {
        seat_games,
        mean_rank: rank_sum as f64 / denominator,
        mean_score_delta: score_sum as f64 / denominator,
        first_rate: firsts as f64 / denominator,
        last_rate: lasts as f64 / denominator,
    }
}

fn cross_policy_win_rate(blocks: &[BlockResult]) -> f64 {
    let mut wins = 0_u64;
    let mut comparisons = 0_u64;
    for game in blocks.iter().flat_map(|block| &block.games) {
        for policy_a_seat in 0..4 {
            if game.policy_a_seat_mask & (1 << policy_a_seat) == 0 {
                continue;
            }
            for policy_b_seat in 0..4 {
                if game.policy_a_seat_mask & (1 << policy_b_seat) != 0 {
                    continue;
                }
                wins += u64::from(game.ranks[policy_a_seat] < game.ranks[policy_b_seat]);
                comparisons += 1;
            }
        }
    }
    wins as f64 / comparisons as f64
}

fn summarize_tournament(
    blocks: &[BlockResult],
    bootstrap_samples: usize,
    bootstrap_seed: u64,
) -> TournamentSummary {
    let rating = fit_pl_rating(&aggregate_patterns(blocks)) * ELO_SCALE;
    let uncertainty = (blocks.len() >= 2).then(|| {
        let mut bootstrap = bootstrap_ratings(blocks, bootstrap_samples, bootstrap_seed);
        let stronger_probability = positive_probability(&bootstrap);
        bootstrap.sort_by(f64::total_cmp);
        TournamentUncertainty {
            ci95: [
                quantile_sorted(&bootstrap, 0.025),
                quantile_sorted(&bootstrap, 0.975),
            ],
            stronger_probability,
        }
    });
    TournamentSummary {
        elo_like_delta: rating,
        uncertainty,
        cross_policy_win_rate: cross_policy_win_rate(blocks),
        policy_a: summarize_policy(blocks, true),
        policy_b: summarize_policy(blocks, false),
    }
}

fn print_policy(name: &str, summary: PolicySummary) {
    println!(
        "{name:<9} seat-games {:>8}  mean-rank {:.5}  mean-score {:+.2}  first {:.4}  last {:.4}",
        summary.seat_games,
        summary.mean_rank,
        summary.mean_score_delta,
        summary.first_rate,
        summary.last_rate,
    );
}

fn aggregate_decisions(blocks: &[BlockResult]) -> [DecisionStats; 2] {
    let mut totals = [DecisionStats::default(); 2];
    for game in blocks.iter().flat_map(|block| &block.games) {
        totals[0].merge(game.decisions[0]);
        totals[1].merge(game.decisions[1]);
    }
    totals
}

fn aggregate_searches(blocks: &[BlockResult]) -> [PlannerSearchStats; 2] {
    let mut totals = [PlannerSearchStats::default(); 2];
    for game in blocks.iter().flat_map(|block| &block.games) {
        totals[0].merge(game.searches[0]);
        totals[1].merge(game.searches[1]);
    }
    totals
}

fn print_decisions(name: &str, stats: DecisionStats) {
    println!(
        "Decisions {name}  turns {}  Hu turn {}/{} response {}/{}  kong concealed {}/{} added {}/{} exposed {}/{}  pong {}/{}  response-pass {}",
        stats.turns,
        stats.turn_hu_taken,
        stats.turn_hu_available,
        stats.hu_response_taken,
        stats.hu_responses,
        stats.concealed_kong_taken,
        stats.concealed_kong_available,
        stats.added_kong_taken,
        stats.added_kong_available,
        stats.exposed_kong_taken,
        stats.exposed_kong_available,
        stats.pong_taken,
        stats.pong_available,
        stats.response_passes,
    );
}

fn print_searches(name: &str, stats: PlannerSearchStats) {
    println!(
        "Planner search {name}  decisions {}  proposals {}  rejected {}  overrides {} ({:.2}%)  rollouts {}",
        stats.decisions,
        stats.proposals,
        stats.validation_rejections,
        stats.overrides,
        if stats.decisions == 0 {
            0.0
        } else {
            100.0 * stats.overrides as f64 / stats.decisions as f64
        },
        stats.rollouts,
    );
}

#[derive(Clone, Copy, Debug)]
struct PolicySettings {
    argument_prefix: &'static str,
    kind: PolicyKind,
    lookahead_depth: u8,
    hand_changes: u8,
    draw_horizon: u8,
    candidate_states: u32,
    belief_worlds: u16,
    root_belief: PlannerRootBelief,
    continuation: PlannerContinuation,
    response_worlds: u16,
    search_iterations: u16,
    defense: Defense,
}

fn invalid_policy_value(
    settings: PolicySettings,
    argument: &str,
    value: impl std::fmt::Display,
    expected: impl std::fmt::Display,
) -> clap::Error {
    clap::Error::raw(
        ErrorKind::InvalidValue,
        format!(
            "invalid value '{value}' for '--{}-{argument}' with '{}': expected {expected}",
            settings.argument_prefix,
            settings.kind.name(),
        ),
    )
}

fn build_policy(settings: PolicySettings) -> Result<(Policy, String), clap::Error> {
    match settings.kind {
        PolicyKind::Fast => Ok((Policy::Fast, "rule_fast".into())),
        PolicyKind::Ev => {
            let policy = RuleEvConfig::with_search_depth(settings.lookahead_depth)
                .ok_or_else(|| {
                    invalid_policy_value(
                        settings,
                        "lookahead-depth",
                        settings.lookahead_depth,
                        "an integer from 0 through 3",
                    )
                })?
                .with_defense(settings.defense.into());
            Ok((
                Policy::Ev(policy),
                format!(
                    "rule_ev_d{}_{}",
                    settings.lookahead_depth,
                    settings.defense.name(),
                ),
            ))
        }
        PolicyKind::Planner => {
            let policy = RulePlannerConfig::STANDARD
                .with_hand_changes(settings.hand_changes)
                .ok_or_else(|| {
                    invalid_policy_value(
                        settings,
                        "hand-changes",
                        settings.hand_changes,
                        "an integer from 0 through 2",
                    )
                })?
                .with_draw_horizon(settings.draw_horizon)
                .ok_or_else(|| {
                    invalid_policy_value(
                        settings,
                        "draw-horizon",
                        settings.draw_horizon,
                        "an integer from 0 through 32",
                    )
                })?
                .with_candidate_states(settings.candidate_states)
                .ok_or_else(|| {
                    invalid_policy_value(
                        settings,
                        "candidate-states",
                        settings.candidate_states,
                        "an integer from 1 through 200000",
                    )
                })?
                .with_belief_worlds(settings.belief_worlds)
                .ok_or_else(|| {
                    invalid_policy_value(
                        settings,
                        "belief-worlds",
                        settings.belief_worlds,
                        "an integer from 0 through 256",
                    )
                })?
                .with_response_worlds(settings.response_worlds)
                .ok_or_else(|| {
                    invalid_policy_value(
                        settings,
                        "response-worlds",
                        settings.response_worlds,
                        "an integer from 0 through 256",
                    )
                })?
                .with_search_iterations(settings.search_iterations)
                .ok_or_else(|| {
                    invalid_policy_value(
                        settings,
                        "search-iterations",
                        settings.search_iterations,
                        "an integer from 0 through 4096",
                    )
                })?;
            Ok((
                Policy::Planner {
                    config: policy,
                    root_belief: settings.root_belief,
                    continuation: settings.continuation,
                },
                format!(
                    "rule_planner_h{}_d{}_c{}_b{}_r{}_i{}{}{}",
                    settings.hand_changes,
                    settings.draw_horizon,
                    settings.candidate_states,
                    settings.belief_worlds,
                    settings.response_worlds,
                    settings.search_iterations,
                    settings.root_belief.name_suffix(),
                    settings.continuation.name_suffix(),
                ),
            ))
        }
    }
}

fn run(config: Config) -> Result<(), clap::Error> {
    let block_count = config.blocks.get();
    let bootstrap_samples = config.bootstrap_samples.get();
    let games = block_count
        .checked_mul(POLICY_A_SEAT_MASKS.len())
        .expect("game count overflowed usize");
    let nested_search = (config.policy_a == PolicyKind::Planner
        && (config.a_search_iterations != 0
            || config.a_belief_worlds != 0
            || config.a_response_worlds != 0))
        || (config.policy_b == PolicyKind::Planner
            && (config.b_search_iterations != 0
                || config.b_belief_worlds != 0
                || config.b_response_worlds != 0));
    let parallel_games = config
        .parallel_games
        .map_or_else(
            || {
                if nested_search {
                    1
                } else {
                    rayon::current_num_threads()
                }
            },
            NonZeroUsize::get,
        )
        .min(games);
    let settings_a = PolicySettings {
        argument_prefix: "a",
        kind: config.policy_a,
        lookahead_depth: config.a_lookahead_depth,
        hand_changes: config.a_hand_changes,
        draw_horizon: config.a_draw_horizon,
        candidate_states: config.a_candidate_states,
        belief_worlds: config.a_belief_worlds,
        root_belief: config.a_root_belief,
        continuation: config.a_continuation,
        response_worlds: config.a_response_worlds,
        search_iterations: config.a_search_iterations,
        defense: config.a_defense,
    };
    let settings_b = PolicySettings {
        argument_prefix: "b",
        kind: config.policy_b,
        lookahead_depth: config.b_lookahead_depth,
        hand_changes: config.b_hand_changes,
        draw_horizon: config.b_draw_horizon,
        candidate_states: config.b_candidate_states,
        belief_worlds: config.b_belief_worlds,
        root_belief: config.b_root_belief,
        continuation: config.b_continuation,
        response_worlds: config.b_response_worlds,
        search_iterations: config.b_search_iterations,
        defense: config.b_defense,
    };
    let (policy_a, policy_a_name) = build_policy(settings_a)?;
    let (policy_b, policy_b_name) = build_policy(settings_b)?;
    println!(
        "Rule tournament  policy-a {}  policy-b {}  blocks {}  games {}  root-seed {}  bootstrap {}  rayon-threads {}  parallel-games {}",
        policy_a_name,
        policy_b_name,
        block_count,
        games,
        config.root_seed,
        bootstrap_samples,
        rayon::current_num_threads(),
        parallel_games,
    );

    // Initialize shared hand-analysis tables without serially evaluating an
    // expensive policy before the tournament's parallel game schedule starts.
    let fast = Policy::Fast;
    play_game(
        mix64(config.root_seed ^ BOOTSTRAP_DOMAIN),
        POLICY_A_SEAT_MASKS[0],
        &fast,
        &fast,
    );
    reset_rule_planner_search_stats();

    let started = Instant::now();
    let progress = TournamentProgress::default();
    let finished = AtomicBool::new(false);
    let game_results: Vec<_> = std::thread::scope(|scope| {
        let monitor = scope.spawn(|| {
            loop {
                std::thread::park_timeout(Duration::from_secs(5));
                let done = progress.completed_games.load(Ordering::Relaxed);
                let active = progress.active_games.load(Ordering::Relaxed);
                let actions = progress.actions.load(Ordering::Relaxed);
                let elapsed = started.elapsed().as_secs_f64();
                let eta = (done != 0 && done < games)
                    .then(|| elapsed * (games - done) as f64 / done as f64);
                eprintln!(
                    "Progress {done}/{games} ({:.1}%)  active {active}  actions {actions} ({:.1}/s)  elapsed {}  ETA {}",
                    100.0 * done as f64 / games as f64,
                    actions as f64 / elapsed.max(f64::EPSILON),
                    format_duration(elapsed),
                    eta.map_or_else(|| "--:--".into(), format_duration),
                );
                if finished.load(Ordering::Acquire) {
                    break;
                }
            }
        });
        let monitor_completion = MonitorCompletion::new(&finished, monitor.thread().clone());
        let next_game = AtomicUsize::new(0);
        let results: Vec<OnceLock<GameResult>> = (0..games).map(|_| OnceLock::new()).collect();
        (0..parallel_games).into_par_iter().for_each(|_| {
            loop {
                let game_index = next_game.fetch_add(1, Ordering::Relaxed);
                if game_index >= games {
                    return;
                }
                let block = game_index / POLICY_A_SEAT_MASKS.len();
                let assignment = game_index % POLICY_A_SEAT_MASKS.len();
                let result = play_game_with_progress(
                    mix64(config.root_seed.wrapping_add(block as u64)),
                    POLICY_A_SEAT_MASKS[assignment],
                    &policy_a,
                    &policy_b,
                    Some(&progress),
                );
                assert!(
                    results[game_index].set(result).is_ok(),
                    "a tournament game is evaluated once"
                );
            }
        });
        let completed = results
            .into_iter()
            .map(|result| {
                result
                    .into_inner()
                    .expect("every tournament game completed")
            })
            .collect();
        drop(monitor_completion);
        completed
    });
    let blocks: Vec<_> = game_results
        .chunks_exact(POLICY_A_SEAT_MASKS.len())
        .map(|games| {
            summarize_block(
                games
                    .try_into()
                    .expect("the flat game schedule contains complete blocks"),
            )
        })
        .collect();
    let play_elapsed = started.elapsed().as_secs_f64();
    let action_count: u64 = blocks
        .iter()
        .flat_map(|block| &block.games)
        .map(|game| u64::from(game.actions))
        .sum();
    let planner_stats = rule_planner_search_stats();

    let statistics_started = Instant::now();
    let summary = summarize_tournament(
        &blocks,
        bootstrap_samples,
        mix64(config.root_seed ^ BOOTSTRAP_DOMAIN),
    );
    let statistics_elapsed = statistics_started.elapsed().as_secs_f64();

    if let Some(uncertainty) = summary.uncertainty {
        println!(
            "{} Elo-like delta vs {} {:+.2}  CI95 [{:+.2}, {:+.2}]  P(stronger) {:.4}  cross-policy-win {:.4}",
            policy_a_name,
            policy_b_name,
            summary.elo_like_delta,
            uncertainty.ci95[0],
            uncertainty.ci95[1],
            uncertainty.stronger_probability,
            summary.cross_policy_win_rate,
        );
    } else {
        println!(
            "{} Elo-like delta vs {} {:+.2}  uncertainty unavailable (need at least 2 blocks)  cross-policy-win {:.4}",
            policy_a_name, policy_b_name, summary.elo_like_delta, summary.cross_policy_win_rate,
        );
    }
    print_policy(&policy_a_name, summary.policy_a);
    print_policy(&policy_b_name, summary.policy_b);
    let decision_stats = aggregate_decisions(&blocks);
    print_decisions(&policy_a_name, decision_stats[0]);
    print_decisions(&policy_b_name, decision_stats[1]);
    let search_stats = aggregate_searches(&blocks);
    print_searches(&policy_a_name, search_stats[0]);
    print_searches(&policy_b_name, search_stats[1]);
    println!(
        "Throughput  play {:.3}s  statistics {:.3}s  games/s {:.3}  actions/s {:.1}  actions {}",
        play_elapsed,
        statistics_elapsed,
        games as f64 / play_elapsed,
        action_count as f64 / play_elapsed,
        action_count,
    );
    println!(
        "Planner  turns {}  overrides {} ({:.2}%)  mean-hazard {:.3} points/candidate  won {:.3}  unwon {:.3}",
        planner_stats.planned_turns,
        planner_stats.turn_overrides,
        if planner_stats.planned_turns == 0 {
            0.0
        } else {
            100.0 * planner_stats.turn_overrides as f64 / planner_stats.planned_turns as f64
        },
        if planner_stats.hazard_candidates == 0 {
            0.0
        } else {
            planner_stats.hazard_loss_millipoints as f64
                / 1_000.0
                / planner_stats.hazard_candidates as f64
        },
        if planner_stats.hazard_candidates == 0 {
            0.0
        } else {
            planner_stats.hazard_won_loss_millipoints as f64
                / 1_000.0
                / planner_stats.hazard_candidates as f64
        },
        if planner_stats.hazard_candidates == 0 {
            0.0
        } else {
            planner_stats
                .hazard_loss_millipoints
                .saturating_sub(planner_stats.hazard_won_loss_millipoints) as f64
                / 1_000.0
                / planner_stats.hazard_candidates as f64
        },
    );
    if let Some(uncertainty) = summary.uncertainty {
        println!(
            "RESULT {}-vs-{} Elo {:+.2} [{:+.2},{:+.2}] P {:+.4}",
            policy_a_name,
            policy_b_name,
            summary.elo_like_delta,
            uncertainty.ci95[0],
            uncertainty.ci95[1],
            uncertainty.stronger_probability,
        );
    } else {
        println!(
            "RESULT {}-vs-{} Elo {:+.2} uncertainty-unavailable",
            policy_a_name, policy_b_name, summary.elo_like_delta,
        );
    }
    Ok(())
}

fn format_duration(seconds: f64) -> String {
    let total = seconds.max(0.0).round() as u64;
    let hours = total / 3_600;
    let minutes = total / 60 % 60;
    let seconds = total % 60;
    if hours == 0 {
        format!("{minutes:02}:{seconds:02}")
    } else {
        format!("{hours}:{minutes:02}:{seconds:02}")
    }
}

fn main() {
    if let Err(error) = run(Config::parse()) {
        error.exit();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_is_balanced() {
        let mut appearances = [0_u8; 4];
        for &mask in &POLICY_A_SEAT_MASKS {
            assert_eq!(mask.count_ones(), 2);
            for (seat, count) in appearances.iter_mut().enumerate() {
                *count += u8::from(mask & (1 << seat) != 0);
            }
        }
        assert_eq!(appearances, [3; 4]);
        for (left_index, &left) in POLICY_A_SEAT_MASKS.iter().enumerate() {
            for &right in POLICY_A_SEAT_MASKS.iter().skip(left_index + 1) {
                assert_ne!(left, right);
            }
        }
    }

    #[test]
    fn mix64_matches_splitmix64_reference() {
        assert_eq!(mix64(0), 0xe220_a839_7b1d_cdaf);
        assert_eq!(mix64(1), 0x910a_2dec_8902_5cc1);
    }

    #[test]
    fn pl_rating_respects_policy_label_symmetry() {
        let mut dominant = [0_u64; RANK_PATTERNS.len()];
        dominant[pattern_index(0b0011)] = 128;
        let rating = fit_pl_rating(&dominant);
        assert!(rating > 0.0);

        let mut reversed = [0_u64; RANK_PATTERNS.len()];
        reversed[pattern_index(0b1100)] = 128;
        let reversed_rating = fit_pl_rating(&reversed);
        assert!(
            (rating + reversed_rating).abs() < 1e-7,
            "forward={rating}, reversed={reversed_rating}"
        );
    }

    #[test]
    fn pl_rating_is_zero_for_balanced_rank_patterns() {
        let counts = [64_u64; RANK_PATTERNS.len()];
        assert!(fit_pl_rating(&counts).abs() < 1e-12);
    }

    #[test]
    fn quantile_uses_linear_interpolation() {
        let values = [0.0, 10.0, 20.0, 30.0, 40.0];
        assert_eq!(quantile_sorted(&values, 0.25), 10.0);
        assert_eq!(quantile_sorted(&values, 0.125), 5.0);
    }

    #[test]
    fn stronger_probability_splits_zero_mass_evenly() {
        assert_eq!(positive_probability(&[0.0]), 0.5);
        assert_eq!(positive_probability(&[-1.0, 0.0, 1.0]), 0.5);
    }

    #[test]
    fn one_block_does_not_claim_bootstrap_uncertainty() {
        let block = BlockResult {
            games: POLICY_A_SEAT_MASKS.map(|policy_a_seat_mask| GameResult {
                policy_a_seat_mask,
                ranks: [1, 2, 3, 4],
                score_deltas: [0; 4],
                actions: 0,
                decisions: [DecisionStats::default(); 2],
                searches: [PlannerSearchStats::default(); 2],
            }),
            rank_pattern_counts: [1; RANK_PATTERNS.len()],
        };

        assert!(summarize_tournament(&[block], 16, 7).uncertainty.is_none());
    }

    #[test]
    fn planner_search_stats_remain_separate_by_policy_side() {
        let mut games = POLICY_A_SEAT_MASKS.map(|policy_a_seat_mask| GameResult {
            policy_a_seat_mask,
            ranks: [1, 2, 3, 4],
            score_deltas: [0; 4],
            actions: 0,
            decisions: [DecisionStats::default(); 2],
            searches: [PlannerSearchStats::default(); 2],
        });
        games[0].searches = [
            PlannerSearchStats {
                decisions: 3,
                proposals: 2,
                validation_rejections: 1,
                overrides: 1,
                rollouts: 30,
            },
            PlannerSearchStats {
                decisions: 5,
                proposals: 4,
                validation_rejections: 3,
                overrides: 1,
                rollouts: 50,
            },
        ];
        let block = BlockResult {
            games,
            rank_pattern_counts: [1; RANK_PATTERNS.len()],
        };

        assert_eq!(aggregate_searches(&[block]), games[0].searches);
    }

    #[test]
    fn arguments_have_defaults_and_named_overrides() {
        assert_eq!(
            Config::try_parse_from(["rule-tournament"]).unwrap(),
            Config::default()
        );
        assert_eq!(
            Config::try_parse_from([
                "rule-tournament",
                "--blocks",
                "12",
                "--root-seed",
                "34",
                "--bootstrap-samples",
                "56",
                "--policy-a",
                "rule-planner",
                "--a-hand-changes",
                "2",
                "--a-draw-horizon",
                "19",
                "--a-candidate-states",
                "2048",
                "--a-belief-worlds",
                "8",
                "--a-root-belief",
                "oracle-hidden",
                "--a-continuation",
                "oracle-continuation",
                "--a-search-iterations",
                "16",
                "--policy-b",
                "rule-ev",
                "--b-lookahead-depth",
                "0",
                "--b-defense",
                "none",
            ])
            .unwrap(),
            Config {
                blocks: NonZeroUsize::new(12).unwrap(),
                root_seed: 34,
                bootstrap_samples: NonZeroUsize::new(56).unwrap(),
                parallel_games: None,
                policy_a: PolicyKind::Planner,
                a_lookahead_depth: 1,
                a_hand_changes: 2,
                a_draw_horizon: 19,
                a_candidate_states: 2_048,
                a_belief_worlds: 8,
                a_root_belief: PlannerRootBelief::OracleHidden,
                a_continuation: PlannerContinuation::OracleContinuation,
                a_response_worlds: 0,
                a_search_iterations: 16,
                a_defense: Defense::Heuristic,
                policy_b: PolicyKind::Ev,
                b_lookahead_depth: 0,
                b_hand_changes: 1,
                b_draw_horizon: 27,
                b_candidate_states: 4_096,
                b_belief_worlds: 0,
                b_root_belief: PlannerRootBelief::Posterior,
                b_continuation: PlannerContinuation::Current,
                b_response_worlds: 0,
                b_search_iterations: 0,
                b_defense: Defense::None,
            }
        );
    }

    #[test]
    fn clap_rejects_out_of_range_values() {
        assert!(Config::try_parse_from(["rule-tournament", "--a-lookahead-depth", "4"]).is_err());
        assert!(Config::try_parse_from(["rule-tournament", "--a-hand-changes", "3"]).is_err());
        assert!(Config::try_parse_from(["rule-tournament", "--b-draw-horizon", "33"]).is_err());
        assert!(Config::try_parse_from(["rule-tournament", "--a-candidate-states", "0"]).is_err());
        assert!(Config::try_parse_from(["rule-tournament", "--a-belief-worlds", "257"]).is_err());
        assert!(Config::try_parse_from(["rule-tournament", "--a-response-worlds", "257"]).is_err());
        assert!(
            Config::try_parse_from(["rule-tournament", "--b-search-iterations", "4097"]).is_err()
        );
        assert!(Config::try_parse_from(["rule-tournament", "--blocks", "0"]).is_err());
    }

    #[test]
    fn rule_ev_ignores_planner_only_parameters() {
        let settings = PolicySettings {
            argument_prefix: "a",
            kind: PolicyKind::Ev,
            lookahead_depth: RuleEvConfig::STANDARD.search_depth(),
            hand_changes: RulePlannerConfig::STANDARD.hand_changes(),
            draw_horizon: RulePlannerConfig::STANDARD.draw_horizon(),
            candidate_states: RulePlannerConfig::STANDARD.candidate_states(),
            belief_worlds: RulePlannerConfig::STANDARD.belief_worlds(),
            root_belief: PlannerRootBelief::OracleHidden,
            continuation: PlannerContinuation::OracleContinuation,
            response_worlds: RulePlannerConfig::STANDARD.response_worlds(),
            search_iterations: 256,
            defense: Defense::Heuristic,
        };
        let with_budget = build_policy(settings).unwrap();
        let without_budget = build_policy(PolicySettings {
            search_iterations: 0,
            root_belief: PlannerRootBelief::Posterior,
            continuation: PlannerContinuation::Current,
            ..settings
        })
        .unwrap();

        assert_eq!(with_budget.1, without_budget.1);
        match (with_budget.0, without_budget.0) {
            (Policy::Ev(left), Policy::Ev(right)) => assert_eq!(left, right),
            _ => panic!("both policies must be rule-ev"),
        }
    }

    #[test]
    fn monitor_completion_unparks_during_unwind() {
        let finished = AtomicBool::new(false);
        let unwind = std::panic::catch_unwind(|| {
            std::thread::scope(|scope| {
                let monitor = scope.spawn(|| {
                    while !finished.load(Ordering::Acquire) {
                        std::thread::park();
                    }
                });
                let _completion = MonitorCompletion::new(&finished, monitor.thread().clone());
                panic!("simulate a worker panic");
            });
        });

        assert!(unwind.is_err());
        assert!(finished.load(Ordering::Acquire));
    }

    #[test]
    fn continuation_profile_follows_every_balanced_seat_mask() {
        let policy_a = Policy::Ev(RuleEvConfig::FAST);
        let policy_b = Policy::Planner {
            config: RulePlannerConfig::FAST,
            root_belief: PlannerRootBelief::OracleHidden,
            continuation: PlannerContinuation::OracleContinuation,
        };

        for mask in POLICY_A_SEAT_MASKS {
            let profile = continuation_profile(mask, &policy_a, &policy_b);
            for seat in Seat::ALL {
                let expected = if mask & (1 << seat.index()) != 0 {
                    RulePlannerContinuationPolicy::Ev(RuleEvConfig::FAST)
                } else {
                    RulePlannerContinuationPolicy::PlannerBaseline(RulePlannerConfig::FAST)
                };
                assert_eq!(profile.for_seat(seat), expected);
            }
        }
    }
}
