use std::hint::black_box;
use std::time::Instant;

use bloodflow_mahjong::{
    Batch, LEGAL_ACTION_MASK_WORDS, MELD_OBSERVATION_WIDTH, META_OBSERVATION_WIDTH,
    RIVER_OBSERVATION_WIDTH, STEP_RECORD_WIDTH, TILE_OBSERVATION_WIDTH,
};
use rayon::current_num_threads;

const DEFAULT_ENVIRONMENTS: usize = 1_024;
const DEFAULT_ACTIONS: usize = 4 * 1_024 * 1_024;
const WARMUP_ROUNDS: usize = 64;
const GAME_SEED: u64 = 0x6a09_e667_f3bc_c909;
const RESET_SEED: u64 = 0x3c6e_f372_fe94_f82b;
const SEED_STEP: u64 = 0x9e37_79b9_7f4a_7c15;

struct Buffers {
    actions: Vec<u8>,
    records: Vec<i64>,
    masks: Vec<u64>,
    tile_obs: Vec<u8>,
    melds: Vec<u8>,
    river: Vec<u8>,
    meta: Vec<i32>,
}

impl Buffers {
    fn new(environments: usize) -> Self {
        Self {
            actions: vec![0; environments],
            records: vec![0; environments * STEP_RECORD_WIDTH],
            masks: vec![0; environments * LEGAL_ACTION_MASK_WORDS],
            tile_obs: vec![0; environments * TILE_OBSERVATION_WIDTH],
            melds: vec![0; environments * MELD_OBSERVATION_WIDTH],
            river: vec![0; environments * RIVER_OBSERVATION_WIDTH],
            meta: vec![0; environments * META_OBSERVATION_WIDTH],
        }
    }

    fn fill_first_legal_actions(&mut self) {
        for (words, action) in self
            .masks
            .chunks_exact(LEGAL_ACTION_MASK_WORDS)
            .zip(&mut self.actions)
        {
            let index = if words[0] != 0 {
                words[0].trailing_zeros()
            } else {
                assert_ne!(words[1], 0, "active environment has no legal action");
                64 + words[1].trailing_zeros()
            };
            *action = index as u8;
        }
    }

    fn observe_and_mask(&mut self, batch: &Batch) {
        batch
            .observations_into(
                &mut self.tile_obs,
                &mut self.melds,
                &mut self.river,
                &mut self.meta,
            )
            .expect("benchmark observation buffers match the batch size");
        batch
            .legal_action_mask_words_into(&mut self.masks)
            .expect("benchmark mask buffer matches the batch size");

        // The Python caller can inspect every output after every call. Keep the
        // same writes observable under release LTO even though this benchmark's
        // policy only consumes the masks.
        black_box(&self.tile_obs);
        black_box(&self.melds);
        black_box(&self.river);
        black_box(&self.meta);
    }
}

fn run_rounds(batch: &mut Batch, buffers: &mut Buffers, rounds: usize, next_game: &mut u64) -> u64 {
    let mut completed_games = 0;
    for _ in 0..rounds {
        buffers.fill_first_legal_actions();
        batch
            .step_indices_into(&buffers.actions, &mut buffers.records)
            .expect("the benchmark policy only submits legal actions");
        buffers.observe_and_mask(batch);

        let mut reset_any = false;
        for (index, record) in buffers.records.chunks_exact(STEP_RECORD_WIDTH).enumerate() {
            if record[11] == 0 {
                continue;
            }
            let seed = RESET_SEED.wrapping_add(next_game.wrapping_mul(SEED_STEP));
            *next_game = next_game.wrapping_add(1);
            batch
                .reset_at(index, seed)
                .expect("terminal slot index is in the batch");
            completed_games += 1;
            reset_any = true;
        }
        if reset_any {
            buffers.observe_and_mask(batch);
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
    let mut buffers = Buffers::new(environments);
    let mut next_game = environments as u64;
    buffers.observe_and_mask(&batch);
    run_rounds(&mut batch, &mut buffers, WARMUP_ROUNDS, &mut next_game);

    batch.reset_all(GAME_SEED);
    next_game = environments as u64;
    buffers.observe_and_mask(&batch);

    let started = Instant::now();
    let completed_games = run_rounds(&mut batch, &mut buffers, rounds, &mut next_game);
    let elapsed = started.elapsed();
    let seconds = elapsed.as_secs_f64();
    let action_count = environments as u64 * rounds as u64;

    println!("environments:  {environments}");
    println!("rayon threads: {}", current_num_threads());
    println!("actions:       {action_count}");
    println!("games:         {completed_games}");
    println!("elapsed:       {seconds:.3} s");
    println!("actions/s:     {:.0}", action_count as f64 / seconds);
}
