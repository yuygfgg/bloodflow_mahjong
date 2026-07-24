use std::time::Instant;

use bloodflow_mahjong::{ActionId, ActionMask, Batch, StepOutcome};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::current_num_threads;

const DEFAULT_ENVIRONMENTS: usize = 1_024;
const DEFAULT_ACTIONS: usize = 4 * 1_024 * 1_024;
const WARMUP_ROUNDS: usize = 64;
const GAME_SEED: u64 = 0x6a09_e667_f3bc_c909;
const POLICY_SEED: u64 = 0xbb67_ae85_84ca_a73b;
const RESET_SEED: u64 = 0x3c6e_f372_fe94_f82b;
const SEED_STEP: u64 = 0x9e37_79b9_7f4a_7c15;

fn choose_action(mask: ActionMask, rng: &mut ChaCha8Rng) -> ActionId {
    let ordinal = rng.random_range(0..mask.count_ones()) as usize;
    mask.iter()
        .nth(ordinal)
        .expect("a live environment always has a legal action")
}

fn fill_actions(
    legal: &[Option<ActionMask>],
    actions: &mut [ActionId],
    policy_rngs: &mut [ChaCha8Rng],
) {
    for ((legal, action), rng) in legal
        .iter()
        .copied()
        .zip(actions.iter_mut())
        .zip(policy_rngs.iter_mut())
    {
        *action = choose_action(
            legal.expect("active batch slots always have a legal action"),
            rng,
        );
    }
}

fn run_rounds(
    batch: &mut Batch,
    rounds: usize,
    legal: &mut [Option<ActionMask>],
    actions: &mut [ActionId],
    outcomes: &mut [StepOutcome],
    policy_rngs: &mut [ChaCha8Rng],
    next_game: &mut u64,
) -> u64 {
    let mut completed_games = 0;
    for _ in 0..rounds {
        batch
            .legal_action_masks_into(legal)
            .expect("benchmark buffers match the batch size");
        fill_actions(legal, actions, policy_rngs);
        batch
            .step_ids(actions, outcomes)
            .expect("the benchmark policy only submits legal actions");

        for (index, outcome) in outcomes.iter().enumerate() {
            if outcome.terminal {
                let seed = RESET_SEED.wrapping_add(next_game.wrapping_mul(SEED_STEP));
                *next_game = next_game.wrapping_add(1);
                batch
                    .reset_at(index, seed)
                    .expect("terminal slot index is in the batch");
                completed_games += 1;
            }
        }
    }
    completed_games
}

fn positive_arg(index: usize, name: &str) -> Option<usize> {
    let value = std::env::args().nth(index)?;
    let parsed = value
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("{name} must be a positive integer, got {value:?}"));
    assert!(parsed > 0, "{name} must be positive");
    Some(parsed)
}

fn main() {
    let environments = positive_arg(1, "environment count").unwrap_or(DEFAULT_ENVIRONMENTS);
    let default_rounds = DEFAULT_ACTIONS.div_ceil(environments);
    let rounds = positive_arg(2, "round count").unwrap_or(default_rounds);

    let mut batch = Batch::new(environments, GAME_SEED);
    let mut legal = vec![None; environments];
    let mut actions = vec![ActionId::PASS; environments];
    let mut outcomes = vec![StepOutcome::default(); environments];
    let mut policy_rngs: Vec<_> = (0..environments)
        .map(|index| {
            ChaCha8Rng::seed_from_u64(
                POLICY_SEED.wrapping_add((index as u64).wrapping_mul(SEED_STEP)),
            )
        })
        .collect();
    let mut next_game = environments as u64;

    run_rounds(
        &mut batch,
        WARMUP_ROUNDS,
        &mut legal,
        &mut actions,
        &mut outcomes,
        &mut policy_rngs,
        &mut next_game,
    );
    batch.reset_all(GAME_SEED);

    let started = Instant::now();
    let completed_games = run_rounds(
        &mut batch,
        rounds,
        &mut legal,
        &mut actions,
        &mut outcomes,
        &mut policy_rngs,
        &mut next_game,
    );
    let elapsed = started.elapsed();
    let seconds = elapsed.as_secs_f64();
    let action_count = environments as u64 * rounds as u64;

    println!("environments:  {environments}");
    println!("rayon threads: {}", current_num_threads());
    println!("actions:       {action_count}");
    println!("games:         {completed_games}");
    println!("elapsed:       {seconds:.3} s");
    println!("actions/s:     {:.0}", action_count as f64 / seconds);
    println!("games/s:       {:.0}", completed_games as f64 / seconds);
}
