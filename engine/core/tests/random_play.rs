use bloodflow_mahjong::{
    ACTION_SPACE_SIZE, Action, ActionId, Game, LegalActions, Phase, Seat, Suit, Tile,
};
use rand::{Rng as _, SeedableRng};
use rand_chacha::ChaCha8Rng;

const GAME_COUNT: u64 = 2_000;
const MAX_ACTIONS_PER_GAME: usize = 1_024;
const SCORE_TOTAL: i64 = 40_000;

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
        Phase::Exchange => {
            assert_ne!(legal.exchange_mask, 0, "exchange must expose a legal tile");
            Action::SelectExchangeTile(tile_from_mask(legal.exchange_mask, rng))
        }
        Phase::ChooseMissing => {
            Action::ChooseMissing(Suit::ALL[rng.random_range(0..Suit::ALL.len())])
        }
        Phase::HuResponse => {
            let mut actions = [Action::Pass; 2];
            let mut len = 0;
            if legal.can_pass {
                actions[len] = Action::Pass;
                len += 1;
            }
            if legal.can_hu {
                actions[len] = Action::Hu;
                len += 1;
            }
            actions[rng.random_range(0..len)]
        }
        Phase::MeldResponse => {
            let mut actions = [Action::Pass; 3];
            let mut len = 0;
            if legal.can_pass {
                actions[len] = Action::Pass;
                len += 1;
            }
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

            assert_ne!(legal.discard_mask, 0, "a turn must have a legal discard");
            Action::Discard(tile_from_mask(legal.discard_mask, rng))
        }
        Phase::Finished => panic!("a finished game must not expose legal actions"),
    }
}

fn scores(game: &Game) -> [i64; 4] {
    core::array::from_fn(|index| game.score(Seat::ALL[index]))
}

fn assert_score_invariants(game: &Game, seed: u64, action_count: usize) {
    let current = scores(game);
    assert!(
        current.iter().all(|score| *score >= 0),
        "seed {seed}, action {action_count}: negative score {current:?}"
    );
    assert_eq!(
        current.iter().sum::<i64>(),
        SCORE_TOTAL,
        "seed {seed}, action {action_count}: score total changed: {current:?}"
    );
}

#[test]
fn seeded_random_legal_play_reaches_terminal_and_preserves_invariants() {
    for game_index in 0..GAME_COUNT {
        let seed =
            0x6a09_e667_f3bc_c909_u64.wrapping_add(game_index.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut game = Game::new(seed);
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xd1b5_4a32_d192_ed03);
        assert_score_invariants(&game, seed, 0);

        let mut terminal = false;
        for action_count in 1..=MAX_ACTIONS_PER_GAME {
            assert_ne!(
                game.phase(),
                Phase::Finished,
                "seed {seed}: terminal phase retained a decision"
            );
            let decision = game
                .decision()
                .unwrap_or_else(|| panic!("seed {seed}, action {action_count}: no decision"));
            let legal = game.legal_actions().unwrap_or_else(|| {
                panic!("seed {seed}, action {action_count}: no legal-action description")
            });
            assert_eq!(legal.decision, decision);

            let mask = game.legal_action_mask().unwrap_or_else(|| {
                panic!("seed {seed}, action {action_count}: no legal-action mask")
            });
            for index in 0..ACTION_SPACE_SIZE {
                let id = ActionId::new(index).expect("action-space indices are valid");
                assert_eq!(
                    mask.contains(id),
                    game.is_legal_action(id.action()),
                    "seed {seed}, action {action_count}, decision {decision:?}, action id {index}"
                );
            }

            let action = choose_action(legal, &mut rng);
            let before_scores = scores(&game);
            let outcome = game.step(action).unwrap_or_else(|error| {
                panic!(
                    "seed {seed}, action {action_count}, decision {decision:?}, action {action:?}: {error:?}"
                )
            });

            assert_eq!(outcome.next, game.decision());
            assert_eq!(outcome.terminal, game.phase() == Phase::Finished);
            assert_eq!(outcome.terminal, outcome.next.is_none());
            assert_eq!(outcome.score_delta.iter().sum::<i64>(), 0);
            let after_scores = scores(&game);
            assert_eq!(
                outcome.score_delta,
                core::array::from_fn(|index| after_scores[index] - before_scores[index])
            );

            if let Action::Discard(tile) = action {
                let discard = outcome.discard.unwrap_or_else(|| {
                    panic!("seed {seed}, action {action_count}: discard event was not exposed")
                });
                assert_eq!(discard.player, decision.actor);
                assert_eq!(discard.tile, tile);
            } else {
                assert!(outcome.discard.is_none());
            }

            if let Some(draw) = outcome.draw {
                assert!(game.concealed(draw.player)[draw.tile.index()] > 0);
                let next = outcome.next.unwrap_or_else(|| {
                    panic!("seed {seed}, action {action_count}: draw had no following decision")
                });
                assert_eq!(next.actor, draw.player);
                assert_eq!(next.phase, Phase::Turn);
            }

            assert_score_invariants(&game, seed, action_count);
            if outcome.terminal {
                terminal = true;
                break;
            }

            assert!(
                game.decision().is_some() && game.legal_actions().is_some(),
                "seed {seed}, action {action_count}: nonterminal game cannot continue"
            );
        }

        assert!(
            terminal,
            "seed {seed}: did not terminate within {MAX_ACTIONS_PER_GAME} actions"
        );
        assert_eq!(game.phase(), Phase::Finished);
        assert!(game.decision().is_none());
        assert!(game.legal_actions().is_none());
    }
}
