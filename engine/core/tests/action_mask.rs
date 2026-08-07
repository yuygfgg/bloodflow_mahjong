use bloodflow_mahjong::{
    ACTION_SPACE_SIZE, Action, ActionId, ActionMask, Batch, Decision, DrawEvent, ExchangeDirection,
    Game, GameError, LegalActions, Meld, Phase, Seat, StepOutcome, Suit, Tile,
};

const TILE_KIND_COUNT: usize = 27;
const EXCHANGE_START: usize = 0;
const MISSING_START: usize = 27;
const DISCARD_START: usize = 30;
const HU: usize = 57;
const PONG: usize = 58;
const EXPOSED_KONG: usize = 59;
const CONCEALED_KONG_START: usize = 60;
const ADDED_KONG_START: usize = 87;
const PASS: usize = 114;

fn tile(index: usize) -> Tile {
    Tile::new(index as u8).expect("test tile index is valid")
}

fn action_id(index: usize) -> ActionId {
    ActionId::new(index).expect("test action index is valid")
}

fn tile_mask(counts: &[u8; TILE_KIND_COUNT]) -> u32 {
    counts
        .iter()
        .enumerate()
        .fold(0_u32, |mask, (index, &count)| {
            mask | (u32::from(count > 0) << index)
        })
}

fn initial_exchange_mask(counts: &[u8; TILE_KIND_COUNT]) -> u32 {
    let mut mask = 0_u32;
    for suit in Suit::ALL {
        let start = suit as usize * 9;
        if counts[start..start + 9].iter().copied().sum::<u8>() >= 3 {
            mask |= tile_mask(counts) & suit.mask();
        }
    }
    mask
}

fn remaining_suit_mask(
    counts: &[u8; TILE_KIND_COUNT],
    selected: &[u8; TILE_KIND_COUNT],
    suit: Suit,
) -> u32 {
    let start = suit as usize * 9;
    (start..start + 9).fold(0_u32, |mask, index| {
        mask | (u32::from(counts[index] > selected[index]) << index)
    })
}

fn mask_indices(mask: ActionMask) -> Vec<usize> {
    mask.iter().map(ActionId::index).collect()
}

fn expected_mask_indices(legal: LegalActions) -> Vec<usize> {
    let mut expected = Vec::new();
    for index in 0..TILE_KIND_COUNT {
        if legal.exchange_mask & (1 << index) != 0 {
            expected.push(EXCHANGE_START + index);
        }
    }
    if legal.can_choose_missing {
        expected.extend(MISSING_START..MISSING_START + 3);
    }
    for index in 0..TILE_KIND_COUNT {
        if legal.discard_mask & (1 << index) != 0 {
            expected.push(DISCARD_START + index);
        }
    }
    if legal.can_hu {
        expected.push(HU);
    }
    if legal.can_pong {
        expected.push(PONG);
    }
    if legal.can_exposed_kong {
        expected.push(EXPOSED_KONG);
    }
    for index in 0..TILE_KIND_COUNT {
        if legal.concealed_kong_mask & (1 << index) != 0 {
            expected.push(CONCEALED_KONG_START + index);
        }
    }
    for index in 0..TILE_KIND_COUNT {
        if legal.added_kong_mask & (1 << index) != 0 {
            expected.push(ADDED_KONG_START + index);
        }
    }
    if legal.can_pass {
        expected.push(PASS);
    }
    expected.sort_unstable();
    expected
}

fn assert_mask_matches_legal(game: &Game) {
    let legal = game
        .legal_actions()
        .expect("a nonterminal game exposes typed legal actions");
    let mask = game
        .legal_action_mask()
        .expect("a nonterminal game exposes a policy mask");
    let expected = expected_mask_indices(legal);
    assert_eq!(mask_indices(mask), expected);
    assert_eq!(mask.count_ones() as usize, expected.len());
    assert_eq!(
        mask.to_dense(),
        core::array::from_fn(|index| u8::from(expected.binary_search(&index).is_ok()))
    );
}

fn complete_exchange(game: &mut Game) {
    for seat in Seat::ALL {
        for _ in 0..3 {
            let decision = game.decision().expect("exchange has a decision");
            assert_eq!(decision.actor, seat);
            assert_eq!(decision.phase, Phase::Exchange);
            let selected = game
                .legal_action_mask()
                .expect("exchange has legal tiles")
                .iter()
                .next()
                .expect("exchange mask is nonempty");
            game.step_id(selected)
                .expect("masked exchange action is legal");
        }
    }
}

fn complete_opening(game: &mut Game) {
    complete_exchange(game);
    for (seat, suit) in Seat::ALL.into_iter().zip(Suit::ALL.into_iter().cycle()) {
        assert_eq!(
            game.decision(),
            Some(Decision {
                actor: seat,
                phase: Phase::ChooseMissing,
            })
        );
        game.step_id(ActionId::choose_missing(suit))
            .expect("missing-suit action is legal");
    }
}

#[derive(Debug, Eq, PartialEq)]
struct PublicSnapshot {
    phase: Phase,
    decision: Option<Decision>,
    wall_remaining: usize,
    current_draw: Option<DrawEvent>,
    concealed: [[u8; TILE_KIND_COUNT]; 4],
    locked: [[u8; TILE_KIND_COUNT]; 4],
    exchange: [[u8; TILE_KIND_COUNT]; 4],
    missing: [Option<Suit>; 4],
    scores: [i64; 4],
    melds: [[Option<Meld>; 4]; 4],
    has_won: [bool; 4],
    discards: Vec<(Seat, Tile)>,
    legal_mask: Option<ActionMask>,
}

fn snapshot(game: &Game) -> PublicSnapshot {
    PublicSnapshot {
        phase: game.phase(),
        decision: game.decision(),
        wall_remaining: game.wall_remaining(),
        current_draw: game.current_draw(),
        concealed: Seat::ALL.map(|seat| *game.concealed(seat)),
        locked: Seat::ALL.map(|seat| *game.locked(seat)),
        exchange: Seat::ALL.map(|seat| *game.exchange_selection(seat)),
        missing: Seat::ALL.map(|seat| game.missing_suit(seat)),
        scores: Seat::ALL.map(|seat| game.score(seat)),
        melds: Seat::ALL.map(|seat| core::array::from_fn(|index| game.meld(seat, index))),
        has_won: Seat::ALL.map(|seat| game.has_won(seat)),
        discards: game.discards().collect(),
        legal_mask: game.legal_action_mask(),
    }
}

#[test]
fn fixed_action_ids_have_stable_layout_and_round_trip() {
    assert_eq!(ACTION_SPACE_SIZE, 115);
    assert_eq!(size_of::<ActionId>(), 1);
    assert_eq!(size_of::<ActionMask>(), 16);
    assert!(ActionId::new(ACTION_SPACE_SIZE).is_none());

    for index in 0..ACTION_SPACE_SIZE {
        let id = action_id(index);
        assert_eq!(id.index(), index);
        assert_eq!(id.action().id(), id);
        assert_eq!(ActionId::from(Action::from(id)), id);
    }

    for index in 0..TILE_KIND_COUNT {
        let tile = tile(index);
        assert_eq!(ActionId::select_exchange_tile(tile).index(), index);
        assert_eq!(ActionId::discard(tile).index(), DISCARD_START + index);
        assert_eq!(
            ActionId::concealed_kong(tile).index(),
            CONCEALED_KONG_START + index
        );
        assert_eq!(ActionId::added_kong(tile).index(), ADDED_KONG_START + index);
    }
    for suit in Suit::ALL {
        assert_eq!(
            ActionId::choose_missing(suit).index(),
            MISSING_START + suit as usize
        );
    }
    assert_eq!(ActionId::HU.index(), HU);
    assert_eq!(ActionId::PONG.index(), PONG);
    assert_eq!(ActionId::EXPOSED_KONG.index(), EXPOSED_KONG);
    assert_eq!(ActionId::PASS.index(), PASS);
    assert!(ActionMask::EMPTY.is_empty());
    assert_eq!(ActionMask::EMPTY.count_ones(), 0);
    assert!(ActionMask::EMPTY.iter().next().is_none());
}

#[test]
fn exchange_is_twelve_single_tile_decisions_with_suit_locked_per_player() {
    let mut game = Game::new_with_direction(0x051e_c7ed, ExchangeDirection::Across);
    let original = Seat::ALL.map(|seat| *game.concealed(seat));
    let mut selected = [[0_u8; TILE_KIND_COUNT]; 4];

    for seat in Seat::ALL {
        let mut chosen_suit = None;
        for pick in 0..3 {
            assert_eq!(
                game.decision(),
                Some(Decision {
                    actor: seat,
                    phase: Phase::Exchange,
                })
            );
            assert_mask_matches_legal(&game);

            let expected_tile_mask = match chosen_suit {
                None => initial_exchange_mask(&original[seat.index()]),
                Some(suit) => {
                    remaining_suit_mask(&original[seat.index()], &selected[seat.index()], suit)
                }
            };
            let legal = game.legal_actions().expect("exchange is nonterminal");
            assert_eq!(legal.exchange_mask, expected_tile_mask);
            assert_eq!(
                mask_indices(game.legal_action_mask().expect("exchange has a mask")),
                (0..TILE_KIND_COUNT)
                    .filter(|&index| expected_tile_mask & (1 << index) != 0)
                    .collect::<Vec<_>>()
            );

            let index = expected_tile_mask.trailing_zeros() as usize;
            let selected_tile = tile(index);
            chosen_suit.get_or_insert(selected_tile.suit());
            selected[seat.index()][index] += 1;

            let outcome = game
                .step_id(ActionId::select_exchange_tile(selected_tile))
                .expect("a tile exposed by the mask must be legal");
            assert_eq!(outcome.draw, None);
            assert_eq!(outcome.discard, None);
            assert_eq!(outcome.score_delta, [0; 4]);
            assert!(!outcome.terminal);
            assert_eq!(game.exchange_selection(seat), &selected[seat.index()]);

            let is_last_selection = seat == Seat::ALL[3] && pick == 2;
            if !is_last_selection {
                assert_eq!(
                    Seat::ALL.map(|current| *game.concealed(current)),
                    original,
                    "tiles move only after all four players selected three"
                );
            }
        }
    }

    assert_eq!(
        game.decision(),
        Some(Decision {
            actor: Seat::EAST,
            phase: Phase::ChooseMissing,
        })
    );

    let mut expected = original;
    for sender in Seat::ALL {
        let receiver = sender.offset(game.exchange_direction().offset());
        for index in 0..TILE_KIND_COUNT {
            expected[sender.index()][index] -= selected[sender.index()][index];
            expected[receiver.index()][index] += selected[sender.index()][index];
        }
    }
    assert_eq!(
        Seat::ALL.map(|seat| *game.concealed(seat)),
        expected,
        "the twelve selections are exchanged in one atomic move"
    );
}

#[test]
fn missing_suit_is_one_three_way_decision_and_turn_has_no_pass() {
    let mut game = Game::new(0x00de_c1de);
    complete_exchange(&mut game);
    let choices = [Suit::Characters, Suit::Bamboo, Suit::Dots, Suit::Characters];

    for (index, seat) in Seat::ALL.into_iter().enumerate() {
        assert_eq!(
            game.decision(),
            Some(Decision {
                actor: seat,
                phase: Phase::ChooseMissing,
            })
        );
        assert_mask_matches_legal(&game);
        let mask = game.legal_action_mask().expect("missing suit has a mask");
        assert_eq!(
            mask_indices(mask),
            vec![MISSING_START, MISSING_START + 1, MISSING_START + 2]
        );
        assert!(!mask.contains(ActionId::PASS));

        game.step_id(ActionId::choose_missing(choices[index]))
            .expect("all three missing suits are legal");
    }

    assert_eq!(game.phase(), Phase::Turn);
    assert_eq!(
        game.decision().expect("dealer acts first").actor,
        game.dealer()
    );
    assert_eq!(
        Seat::ALL.map(|seat| game.missing_suit(seat)),
        choices.map(Some)
    );
    assert_mask_matches_legal(&game);
    let mask = game.legal_action_mask().expect("turn has a mask");
    assert!(!mask.contains(ActionId::PASS));
    assert!(
        game.legal_actions()
            .expect("turn is nonterminal")
            .discard_mask
            != 0
    );

    for id in mask.iter() {
        let mut by_id = game.clone();
        let mut by_typed_action = game.clone();
        let id_outcome = by_id.step_id(id).expect("masked id is legal");
        let typed_outcome = by_typed_action
            .step(id.action())
            .expect("decoded masked action is legal");
        assert_eq!(id_outcome, typed_outcome);
        assert_eq!(snapshot(&by_id), snapshot(&by_typed_action));
    }
}

#[test]
fn masks_match_typed_legality_through_complete_seeded_games() {
    for seed in 0..32_u64 {
        let mut game = Game::new(seed.wrapping_mul(0x9e37_79b9).wrapping_add(17));
        let mut terminal = false;

        for step in 0..512 {
            assert_mask_matches_legal(&game);
            let legal = game.legal_actions().expect("game is not terminal");
            let mask = game.legal_action_mask().expect("game is not terminal");
            assert!(!mask.is_empty(), "seed {seed}, step {step}");
            for index in 0..ACTION_SPACE_SIZE {
                let id = action_id(index);
                assert_eq!(
                    mask.contains(id),
                    game.is_legal_action(id.action()),
                    "seed {seed}, step {step}, action {index}"
                );
            }

            let preferred = match legal.decision.phase {
                Phase::Exchange | Phase::ChooseMissing => mask.iter().next(),
                Phase::Turn => mask
                    .contains(ActionId::HU)
                    .then_some(ActionId::HU)
                    .or_else(|| {
                        (ADDED_KONG_START..ADDED_KONG_START + TILE_KIND_COUNT)
                            .map(action_id)
                            .find(|&id| mask.contains(id))
                    })
                    .or_else(|| {
                        (CONCEALED_KONG_START..CONCEALED_KONG_START + TILE_KIND_COUNT)
                            .map(action_id)
                            .find(|&id| mask.contains(id))
                    })
                    .or_else(|| mask.iter().next()),
                Phase::HuResponse | Phase::MeldResponse => [
                    ActionId::HU,
                    ActionId::EXPOSED_KONG,
                    ActionId::PONG,
                    ActionId::PASS,
                ]
                .into_iter()
                .find(|&id| mask.contains(id)),
                Phase::Finished => None,
            }
            .expect("a live state has at least one policy action");

            let mut typed = game.clone();
            let expected = typed
                .step(preferred.action())
                .expect("mask decodes to a typed legal action");
            let actual = game
                .step_id(preferred)
                .expect("mask action id must be accepted");
            assert_eq!(actual, expected, "seed {seed}, step {step}");
            assert_eq!(
                snapshot(&game),
                snapshot(&typed),
                "seed {seed}, step {step}"
            );

            if actual.terminal {
                terminal = true;
                break;
            }
        }

        assert!(terminal, "seed {seed} did not terminate");
        assert_eq!(game.phase(), Phase::Finished);
        assert!(game.legal_action_mask().is_none());
    }
}

#[test]
fn masked_out_ids_are_rejected_without_mutating_state() {
    let mut game = Game::new(91);
    let before = snapshot(&game);
    assert_eq!(
        game.step_id(ActionId::choose_missing(Suit::Characters)),
        Err(GameError::InvalidAction)
    );
    assert_eq!(snapshot(&game), before);

    let first = game
        .legal_action_mask()
        .expect("exchange has a mask")
        .iter()
        .next()
        .expect("exchange has a tile");
    game.step_id(first).expect("first exchange tile is legal");
    let chosen_suit = first.action();
    let Action::SelectExchangeTile(chosen_tile) = chosen_suit else {
        panic!("exchange range decodes to an exchange action");
    };
    let wrong_suit_tile = (0..TILE_KIND_COUNT)
        .map(tile)
        .find(|candidate| candidate.suit() != chosen_tile.suit())
        .expect("there are three suits");
    let before = snapshot(&game);
    assert_eq!(
        game.step_id(ActionId::select_exchange_tile(wrong_suit_tile)),
        Err(GameError::InvalidExchange)
    );
    assert_eq!(snapshot(&game), before);
}

#[test]
fn batch_masks_and_steps_equal_individual_games_and_reject_atomically() {
    const BATCH_SIZE: usize = 64;
    let mut batch = Batch::new(BATCH_SIZE, 0xabc0_1234);
    let mut masks = vec![None; BATCH_SIZE];
    batch
        .legal_action_masks_into(&mut masks)
        .expect("matching batch output length is valid");
    for (game, mask) in batch.games().iter().zip(masks.iter().copied()) {
        assert_eq!(mask, game.legal_action_mask());
    }

    let ids: Vec<_> = masks
        .iter()
        .map(|mask| {
            mask.expect("new game is live")
                .iter()
                .next()
                .expect("new game has an exchange action")
        })
        .collect();
    let mut expected_batch = batch.clone();
    let expected_outcomes: Vec<_> = expected_batch
        .games_mut()
        .iter_mut()
        .zip(ids.iter().copied())
        .map(|(game, id)| game.step_id(id).expect("individual masked action is legal"))
        .collect();
    let mut actual_outcomes = vec![StepOutcome::default(); BATCH_SIZE];
    batch
        .step_ids(&ids, &mut actual_outcomes)
        .expect("batch accepts the same masked actions");
    assert_eq!(actual_outcomes, expected_outcomes);
    for (actual, expected) in batch.games().iter().zip(expected_batch.games()) {
        assert_eq!(snapshot(actual), snapshot(expected));
    }

    assert_eq!(
        batch.legal_action_masks_into(&mut masks[..BATCH_SIZE - 1]),
        Err(GameError::BatchLength)
    );
    assert_eq!(
        batch.step_ids(&ids[..BATCH_SIZE - 1], &mut actual_outcomes),
        Err(GameError::BatchLength)
    );

    let mut atomic = Batch::new(2, 7);
    let before = atomic.games().iter().map(snapshot).collect::<Vec<_>>();
    let valid = atomic.games()[0]
        .legal_action_mask()
        .expect("new game is live")
        .iter()
        .next()
        .expect("new game has an exchange action");
    let invalid = ActionId::choose_missing(Suit::Characters);
    let mut outcomes = [StepOutcome::default(); 2];
    assert_eq!(
        atomic.step_ids(&[valid, invalid], &mut outcomes),
        Err(GameError::InvalidAction)
    );
    assert_eq!(
        atomic.games().iter().map(snapshot).collect::<Vec<_>>(),
        before,
        "batch validation must finish before any environment advances"
    );
}

#[test]
fn opening_helper_reaches_a_normal_turn() {
    let mut game = Game::new(3);
    complete_opening(&mut game);
    assert_eq!(game.phase(), Phase::Turn);
    assert_mask_matches_legal(&game);
}
