//! Balanced tournament evaluation for deterministic rule policies.

use std::num::NonZeroUsize;
use std::time::Instant;

use bloodflow_mahjong::{
    Game, RuleEvConfig, RuleEvDefense, RuleEvSearchGate, Seat, reset_rule_ev_search_stats,
    rule_ev_search_stats,
};
use clap::{Parser, ValueEnum};
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
const BOOTSTRAP_DOMAIN: u64 = 0x3c6e_f372_fe94_f82b;

// Every two-versus-two assignment appears once. Within a seed block, each
// policy controls each seat three times.
const RULE_EV_SEAT_MASKS: [u8; 6] = [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100];

// Rank-order patterns with exactly two rule_ev players. Bit zero is first
// place, bit one is second place, and so on.
const RANK_PATTERNS: [u8; 6] = [0b0011, 0b0101, 0b1001, 0b0110, 0b1010, 0b1100];

#[derive(Clone, Debug, Eq, Parser, PartialEq)]
#[command(
    name = "rule-tournament",
    about = "Run balanced rule-EV versus rule policy tournament blocks."
)]
struct Config {
    #[arg(long, default_value_t = DEFAULT_BLOCKS)]
    blocks: NonZeroUsize,
    #[arg(long, default_value_t = DEFAULT_ROOT_SEED)]
    root_seed: u64,
    #[arg(long, default_value_t = DEFAULT_BOOTSTRAP_SAMPLES)]
    bootstrap_samples: NonZeroUsize,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..=3))]
    candidate_depth: u8,
    #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u16).range(0..=256))]
    candidate_worlds: u16,
    #[arg(long, value_enum, default_value = "heuristic")]
    candidate_defense: Defense,
    #[arg(long, value_enum, default_value = "world")]
    candidate_search_gate: SearchGate,
    #[arg(long, value_enum, default_value = "rule-fast")]
    opponent: Opponent,
    #[arg(long, value_enum, default_value = "heuristic")]
    opponent_defense: Defense,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum SearchGate {
    World,
    ScenarioStrict,
    ScenarioRelaxed,
}

impl From<SearchGate> for RuleEvSearchGate {
    fn from(value: SearchGate) -> Self {
        match value {
            SearchGate::World => Self::WorldClustered,
            SearchGate::ScenarioStrict => Self::ScenarioStrict,
            SearchGate::ScenarioRelaxed => Self::ScenarioRelaxed,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Opponent {
    RuleFast,
    #[value(name = "rule-ev-0")]
    RuleEv0,
    #[value(name = "rule-ev-1")]
    RuleEv1,
    #[value(name = "rule-ev-2")]
    RuleEv2,
    #[value(name = "rule-ev-3")]
    RuleEv3,
}

impl Opponent {
    fn name(self) -> String {
        match self {
            Self::RuleFast => "rule_fast".into(),
            Self::RuleEv0 => "rule_ev_d0".into(),
            Self::RuleEv1 => "rule_ev_d1".into(),
            Self::RuleEv2 => "rule_ev_d2".into(),
            Self::RuleEv3 => "rule_ev_d3".into(),
        }
    }

    fn search_depth(self) -> Option<u8> {
        match self {
            Self::RuleFast => None,
            Self::RuleEv0 => Some(0),
            Self::RuleEv1 => Some(1),
            Self::RuleEv2 => Some(2),
            Self::RuleEv3 => Some(3),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpponentPolicy {
    RuleFast,
    RuleEv(RuleEvConfig),
}

impl Default for Config {
    fn default() -> Self {
        Self {
            blocks: DEFAULT_BLOCKS,
            root_seed: DEFAULT_ROOT_SEED,
            bootstrap_samples: DEFAULT_BOOTSTRAP_SAMPLES,
            candidate_depth: RuleEvConfig::STANDARD.search_depth(),
            candidate_worlds: 0,
            candidate_defense: Defense::Heuristic,
            candidate_search_gate: SearchGate::World,
            opponent: Opponent::RuleFast,
            opponent_defense: Defense::Heuristic,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GameResult {
    rule_ev_seat_mask: u8,
    ranks: [u8; 4],
    score_deltas: [i64; 4],
    actions: u32,
}

#[derive(Clone, Debug)]
struct BlockResult {
    games: [GameResult; RULE_EV_SEAT_MASKS.len()],
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
    elo_like_ci95: [f64; 2],
    stronger_probability: f64,
    cross_policy_win_rate: f64,
    rule_ev: PolicySummary,
    rule_fast: PolicySummary,
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn play_game(
    seed: u64,
    rule_ev_seat_mask: u8,
    candidate: RuleEvConfig,
    opponent: OpponentPolicy,
) -> GameResult {
    debug_assert_eq!(rule_ev_seat_mask.count_ones(), 2);
    let mut game = Game::new(seed);
    let initial_scores = Seat::ALL.map(|seat| game.score(seat));

    for action_index in 0..MAX_ACTIONS_PER_GAME {
        let decision = game
            .decision()
            .expect("a non-terminal game always has a decision");
        let rule_ev_controls_actor = rule_ev_seat_mask & (1 << decision.actor.index()) != 0;
        let action = if rule_ev_controls_actor {
            game.rule_ev_action_with_config(candidate)
        } else {
            match opponent {
                OpponentPolicy::RuleFast => game.simple_rule_action(),
                OpponentPolicy::RuleEv(config) => game.rule_ev_action_with_config(config),
            }
        }
        .expect("a non-terminal rule policy always returns an action");
        let outcome = game.step_id(action).unwrap_or_else(|error| {
            panic!(
                "rule policy selected an illegal action: seed={seed}, mask={rule_ev_seat_mask:#06b}, error={error}"
            )
        });
        if !outcome.terminal {
            continue;
        }

        let mut ranks = [0_u8; 4];
        for (index, seat) in game.rankings().into_iter().enumerate() {
            ranks[seat.index()] = (index + 1) as u8;
        }
        let score_deltas = Seat::ALL.map(|seat| game.score(seat) - initial_scores[seat.index()]);
        return GameResult {
            rule_ev_seat_mask,
            ranks,
            score_deltas,
            actions: (action_index + 1) as u32,
        };
    }

    panic!("game exceeded the action limit: seed={seed}, mask={rule_ev_seat_mask:#06b}");
}

fn rank_pattern(game: &GameResult) -> u8 {
    let mut pattern = 0_u8;
    for seat in 0..4 {
        if game.rule_ev_seat_mask & (1 << seat) != 0 {
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

fn play_block(seed: u64, candidate: RuleEvConfig, opponent: OpponentPolicy) -> BlockResult {
    let games = RULE_EV_SEAT_MASKS.map(|mask| play_game(seed, mask, candidate, opponent));
    let mut rank_pattern_counts = [0_u8; RANK_PATTERNS.len()];
    for game in &games {
        rank_pattern_counts[pattern_index(rank_pattern(game))] += 1;
    }
    BlockResult {
        games,
        rank_pattern_counts,
    }
}

fn rule_ev_probability(rating: f64, remaining_ev: u32, remaining_fast: u32) -> f64 {
    match (remaining_ev, remaining_fast) {
        (0, _) => 0.0,
        (_, 0) => 1.0,
        _ => {
            let log_odds = rating + f64::from(remaining_ev).ln() - f64::from(remaining_fast).ln();
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
                let remaining_ev = remaining.count_ones();
                let remaining_fast = 4 - position as u32 - remaining_ev;
                let probability = rule_ev_probability(rating, remaining_ev, remaining_fast);
                let winner_is_ev = f64::from((remaining & 1) != 0);
                gradient += weight * (winner_is_ev - probability);
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

fn summarize_policy(blocks: &[BlockResult], rule_ev: bool) -> PolicySummary {
    let mut seat_games = 0_u64;
    let mut rank_sum = 0_u64;
    let mut score_sum = 0_i128;
    let mut firsts = 0_u64;
    let mut lasts = 0_u64;
    for game in blocks.iter().flat_map(|block| &block.games) {
        for seat in 0..4 {
            if (game.rule_ev_seat_mask & (1 << seat) != 0) != rule_ev {
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
        for rule_ev_seat in 0..4 {
            if game.rule_ev_seat_mask & (1 << rule_ev_seat) == 0 {
                continue;
            }
            for rule_fast_seat in 0..4 {
                if game.rule_ev_seat_mask & (1 << rule_fast_seat) != 0 {
                    continue;
                }
                wins += u64::from(game.ranks[rule_ev_seat] < game.ranks[rule_fast_seat]);
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
    let mut bootstrap = bootstrap_ratings(blocks, bootstrap_samples, bootstrap_seed);
    let stronger_probability =
        bootstrap.iter().filter(|&&sample| sample > 0.0).count() as f64 / bootstrap.len() as f64;
    bootstrap.sort_by(f64::total_cmp);
    TournamentSummary {
        elo_like_delta: rating,
        elo_like_ci95: [
            quantile_sorted(&bootstrap, 0.025),
            quantile_sorted(&bootstrap, 0.975),
        ],
        stronger_probability,
        cross_policy_win_rate: cross_policy_win_rate(blocks),
        rule_ev: summarize_policy(blocks, true),
        rule_fast: summarize_policy(blocks, false),
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

fn run(config: Config) {
    let block_count = config.blocks.get();
    let bootstrap_samples = config.bootstrap_samples.get();
    let games = block_count
        .checked_mul(RULE_EV_SEAT_MASKS.len())
        .expect("game count overflowed usize");
    let candidate_defense = RuleEvDefense::from(config.candidate_defense);
    let candidate_search_gate = RuleEvSearchGate::from(config.candidate_search_gate);
    let candidate = RuleEvConfig::with_search_depth(config.candidate_depth)
        .expect("argument parsing validates candidate depth")
        .with_search_worlds(config.candidate_worlds)
        .expect("argument parsing validates candidate worlds")
        .with_defense(candidate_defense)
        .with_search_gate(candidate_search_gate);
    let defense_name = config.candidate_defense.name();
    let gate_name = match config.candidate_search_gate {
        SearchGate::World => "world",
        SearchGate::ScenarioStrict => "scenario-strict",
        SearchGate::ScenarioRelaxed => "scenario-relaxed",
    };
    let candidate_name = format!(
        "rule_ev_d{}_w{}_{}_{}",
        config.candidate_depth, config.candidate_worlds, defense_name, gate_name,
    );
    let opponent_name = match config.opponent.search_depth() {
        None => config.opponent.name(),
        Some(search_depth) => format!(
            "rule_ev_d{}_{}",
            search_depth,
            config.opponent_defense.name(),
        ),
    };
    let opponent = match config.opponent.search_depth() {
        None => OpponentPolicy::RuleFast,
        Some(search_depth) => OpponentPolicy::RuleEv(
            RuleEvConfig::with_search_depth(search_depth)
                .expect("the CLI only exposes valid opponent depths")
                .with_defense(config.opponent_defense.into()),
        ),
    };
    println!(
        "Rule tournament  candidate {}  opponent {}  blocks {}  games {}  root-seed {}  bootstrap {}  rayon-threads {}",
        candidate_name,
        opponent_name,
        block_count,
        games,
        config.root_seed,
        bootstrap_samples,
        rayon::current_num_threads(),
    );

    // Initialize lazily built hand-analysis tables outside the measurement.
    let _ = play_block(
        mix64(config.root_seed ^ BOOTSTRAP_DOMAIN),
        candidate,
        opponent,
    );
    reset_rule_ev_search_stats();

    let started = Instant::now();
    let blocks: Vec<_> = (0..block_count)
        .into_par_iter()
        .map(|block| {
            play_block(
                mix64(config.root_seed.wrapping_add(block as u64)),
                candidate,
                opponent,
            )
        })
        .collect();
    let play_elapsed = started.elapsed().as_secs_f64();
    let action_count: u64 = blocks
        .iter()
        .flat_map(|block| &block.games)
        .map(|game| u64::from(game.actions))
        .sum();
    let search_stats = rule_ev_search_stats();

    let statistics_started = Instant::now();
    let summary = summarize_tournament(
        &blocks,
        bootstrap_samples,
        mix64(config.root_seed ^ BOOTSTRAP_DOMAIN),
    );
    let statistics_elapsed = statistics_started.elapsed().as_secs_f64();

    println!(
        "{} Elo-like delta vs {} {:+.2}  CI95 [{:+.2}, {:+.2}]  P(stronger) {:.4}  cross-policy-win {:.4}",
        candidate_name,
        opponent_name,
        summary.elo_like_delta,
        summary.elo_like_ci95[0],
        summary.elo_like_ci95[1],
        summary.stronger_probability,
        summary.cross_policy_win_rate,
    );
    print_policy(&candidate_name, summary.rule_ev);
    print_policy(&opponent_name, summary.rule_fast);
    println!(
        "Throughput  play {:.3}s  statistics {:.3}s  games/s {:.0}  actions/s {:.0}  actions {}",
        play_elapsed,
        statistics_elapsed,
        games as f64 / play_elapsed,
        action_count as f64 / play_elapsed,
        action_count,
    );
    println!(
        "Search  decisions {}  overrides {} ({:.2}%)  rollouts {}",
        search_stats.decisions,
        search_stats.overrides,
        if search_stats.decisions == 0 {
            0.0
        } else {
            100.0 * search_stats.overrides as f64 / search_stats.decisions as f64
        },
        search_stats.rollouts,
    );
    println!(
        "RESULT {}-vs-{} Elo {:+.2} [{:+.2},{:+.2}] P {:+.4}",
        candidate_name,
        opponent_name,
        summary.elo_like_delta,
        summary.elo_like_ci95[0],
        summary.elo_like_ci95[1],
        summary.stronger_probability,
    );
}

fn main() {
    run(Config::parse());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_is_balanced() {
        let mut appearances = [0_u8; 4];
        for &mask in &RULE_EV_SEAT_MASKS {
            assert_eq!(mask.count_ones(), 2);
            for (seat, count) in appearances.iter_mut().enumerate() {
                *count += u8::from(mask & (1 << seat) != 0);
            }
        }
        assert_eq!(appearances, [3; 4]);
        for (left_index, &left) in RULE_EV_SEAT_MASKS.iter().enumerate() {
            for &right in RULE_EV_SEAT_MASKS.iter().skip(left_index + 1) {
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
                "--candidate-depth",
                "2",
                "--candidate-worlds",
                "8",
                "--candidate-defense",
                "none",
                "--candidate-search-gate",
                "scenario-strict",
                "--opponent",
                "rule-ev-0",
                "--opponent-defense",
                "heuristic",
            ])
            .unwrap(),
            Config {
                blocks: NonZeroUsize::new(12).unwrap(),
                root_seed: 34,
                bootstrap_samples: NonZeroUsize::new(56).unwrap(),
                candidate_depth: 2,
                candidate_worlds: 8,
                candidate_defense: Defense::None,
                candidate_search_gate: SearchGate::ScenarioStrict,
                opponent: Opponent::RuleEv0,
                opponent_defense: Defense::Heuristic,
            }
        );
    }

    #[test]
    fn clap_rejects_out_of_range_values() {
        assert!(Config::try_parse_from(["rule-tournament", "--candidate-depth", "4"]).is_err());
        assert!(Config::try_parse_from(["rule-tournament", "--candidate-worlds", "257"]).is_err());
        assert!(Config::try_parse_from(["rule-tournament", "--blocks", "0"]).is_err());
    }
}
