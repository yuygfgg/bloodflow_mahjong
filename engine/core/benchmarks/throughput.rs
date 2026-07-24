use std::time::Instant;

use bloodflow_mahjong::{Action, Game, LegalActions, Phase, Suit, Tile};
use rand::{Rng as _, SeedableRng};
use rand_chacha::ChaCha8Rng;

const DEFAULT_GAMES: usize = 10_000;
const MAX_ACTIONS_PER_GAME: usize = 1_024;

#[derive(Default)]
struct Counts {
    actions: u64,
    draws: u64,
    discards: u64,
}

fn tile_from_mask(mut mask: u32, rng: &mut ChaCha8Rng) -> Tile {
    let mut ordinal = rng.random_range(0..mask.count_ones() as usize);
    loop {
        let index = mask.trailing_zeros() as u8;
        if ordinal == 0 {
            return Tile::new(index).expect("legal masks contain valid tile indices");
        }
        ordinal -= 1;
        mask &= mask - 1;
    }
}

fn choose_action(legal: LegalActions, rng: &mut ChaCha8Rng) -> Action {
    match legal.decision.phase {
        Phase::Exchange => Action::SelectExchangeTile(tile_from_mask(legal.exchange_mask, rng)),
        Phase::ChooseMissing => {
            Action::ChooseMissing(Suit::ALL[rng.random_range(0..Suit::ALL.len())])
        }
        Phase::HuResponse => {
            if legal.can_hu && rng.random_range(0..2) == 1 {
                Action::Hu
            } else {
                Action::Pass
            }
        }
        Phase::MeldResponse => {
            let mut actions = [Action::Pass; 3];
            let mut len = 1;
            if legal.can_pong {
                actions[len] = Action::Pong;
                len += 1;
            }
            if legal.can_exposed_kong {
                actions[len] = Action::ExposedKong;
                len += 1;
            }
            actions[rng.random_range(0..len)]
        }
        Phase::Turn => {
            let mut categories = 1;
            categories += usize::from(legal.can_hu);
            categories += usize::from(legal.concealed_kong_mask != 0);
            categories += usize::from(legal.added_kong_mask != 0);
            let mut selected = rng.random_range(0..categories);

            if legal.can_hu {
                if selected == 0 {
                    return Action::Hu;
                }
                selected -= 1;
            }
            if legal.concealed_kong_mask != 0 {
                if selected == 0 {
                    return Action::ConcealedKong(tile_from_mask(legal.concealed_kong_mask, rng));
                }
                selected -= 1;
            }
            if legal.added_kong_mask != 0 && selected == 0 {
                return Action::AddedKong(tile_from_mask(legal.added_kong_mask, rng));
            }
            Action::Discard(tile_from_mask(legal.discard_mask, rng))
        }
        Phase::Finished => unreachable!("finished games have no legal actions"),
    }
}

fn play_game(seed: u64) -> Counts {
    let mut game = Game::new(seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xd1b5_4a32_d192_ed03);
    let mut counts = Counts::default();

    for _ in 0..MAX_ACTIONS_PER_GAME {
        let Some(legal) = game.legal_actions() else {
            assert_eq!(game.phase(), Phase::Finished);
            return counts;
        };
        let action = choose_action(legal, &mut rng);
        let outcome = game
            .step(action)
            .unwrap_or_else(|error| panic!("legal-action policy produced {action:?}: {error:?}"));
        counts.actions += 1;
        counts.draws += u64::from(outcome.draw.is_some());
        counts.discards += u64::from(outcome.discard.is_some());
        if outcome.terminal {
            return counts;
        }
        assert!(
            outcome.next.is_some(),
            "nonterminal step has no next decision"
        );
    }

    panic!("game {seed} exceeded {MAX_ACTIONS_PER_GAME} actions");
}

fn game_count() -> usize {
    let Some(value) = std::env::args().nth(1) else {
        return DEFAULT_GAMES;
    };
    let count = value
        .parse::<usize>()
        .unwrap_or_else(|_| panic!("game count must be a positive integer, got {value:?}"));
    assert!(count > 0, "game count must be positive");
    count
}

fn main() {
    let games = game_count();

    // Build the lazily initialized hand table before starting the measurement.
    let _ = play_game(0x243f_6a88_85a3_08d3);

    let started = Instant::now();
    let mut totals = Counts::default();
    for index in 0..games {
        let seed = 0x6a09_e667_f3bc_c909_u64
            .wrapping_add((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let counts = play_game(seed);
        totals.actions += counts.actions;
        totals.draws += counts.draws;
        totals.discards += counts.discards;
    }
    let elapsed = started.elapsed();
    let seconds = elapsed.as_secs_f64();

    println!("games:       {games}");
    println!("actions:     {}", totals.actions);
    println!("draw events: {}", totals.draws);
    println!("discards:    {}", totals.discards);
    println!("elapsed:     {seconds:.3} s");
    println!("games/s:     {:.0}", games as f64 / seconds);
    println!("actions/s:   {:.0}", totals.actions as f64 / seconds);
}
