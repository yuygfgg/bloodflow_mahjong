use bloodflow_mahjong::{
    ActionId, Batch, EVENT_HISTORY_CAPACITY, EVENT_RECORD_WIDTH, EventKind, Game, Phase, Seat, Suit,
};

fn first_legal(game: &Game) -> ActionId {
    game.legal_action_mask()
        .expect("live game has a mask")
        .iter()
        .next()
        .expect("live game has a legal action")
}

fn complete_opening(game: &mut Game) {
    for _ in 0..12 {
        assert_eq!(game.phase(), Phase::Exchange);
        let action = first_legal(game);
        game.step_id(action).expect("exchange action is legal");
    }
    for suit in Suit::ALL.into_iter().cycle().take(4) {
        assert_eq!(game.phase(), Phase::ChooseMissing);
        game.step_id(ActionId::choose_missing(suit))
            .expect("missing-suit action is legal");
    }
    assert_eq!(game.phase(), Phase::Turn);
}

fn records(buffer: &[i32], length: usize) -> impl Iterator<Item = &[i32]> {
    buffer[..length * EVENT_RECORD_WIDTH].chunks_exact(EVENT_RECORD_WIDTH)
}

#[test]
fn event_history_contains_public_setup_and_private_actions() {
    let mut game = Game::new_with_direction(42, bloodflow_mahjong::ExchangeDirection::Across);
    assert_eq!(game.event_count(), 1);
    let mut history = vec![0_i32; 16 * EVENT_RECORD_WIDTH];
    let length = game
        .events_into(Seat::EAST, &mut history)
        .expect("history buffer is valid");
    assert_eq!(length, 1);
    let start = &history[..EVENT_RECORD_WIDTH];
    assert_eq!(start[0], i32::from(EventKind::GameStart.code()));
    assert_eq!(start[1], 0);
    assert_eq!(start[4], 2); // Across exchange direction.

    for _ in 0..12 {
        let viewer = game.decision().expect("exchange decision").actor;
        game.step_id(first_legal(&game))
            .expect("exchange action is legal");
        let mut delta = [0_i32; 4 * EVENT_RECORD_WIDTH];
        let length = game
            .step_events_into(viewer, &mut delta)
            .expect("step event buffer is valid");
        assert_eq!(length, game.step_event_count());
        assert_eq!(delta[0], i32::from(EventKind::Action.code()));
    }

    let mut history = vec![0_i32; EVENT_HISTORY_CAPACITY * EVENT_RECORD_WIDTH];
    let length = game
        .events_into(Seat::EAST, &mut history)
        .expect("history buffer is valid");
    assert!(
        records(&history, length)
            .any(|record| record[0] == i32::from(EventKind::ExchangeComplete.code()))
    );

    for suit in Suit::ALL.into_iter().cycle().take(4) {
        let viewer = game.decision().expect("missing-suit decision").actor;
        game.step_id(ActionId::choose_missing(suit))
            .expect("missing-suit action is legal");
        let mut delta = [0_i32; 8 * EVENT_RECORD_WIDTH];
        let length = game
            .step_events_into(viewer, &mut delta)
            .expect("step event buffer is valid");
        assert_eq!(length, game.step_event_count());
    }
    let length = game
        .events_into(Seat::EAST, &mut history)
        .expect("history buffer is valid");
    assert_eq!(game.phase(), Phase::Turn);
    assert!(
        records(&history, length)
            .any(|record| record[0] == i32::from(EventKind::MissingRevealed.code()))
    );
    assert!(
        records(&history, length).any(|record| record[0] == i32::from(EventKind::TurnStart.code()))
    );
}

#[test]
fn event_draw_tiles_are_visible_only_to_the_drawer() {
    for seed in 0..128_u64 {
        let mut game = Game::new(seed);
        complete_opening(&mut game);
        let action = first_legal(&game);
        if action.index() < 30 || action.index() > 56 {
            continue;
        }
        let outcome = game.step_id(action).expect("discard action is legal");
        let Some(draw) = outcome.draw else { continue };
        let mut drawer_buffer = [0_i32; 16 * EVENT_RECORD_WIDTH];
        let drawer_len = game
            .step_events_into(draw.player, &mut drawer_buffer)
            .expect("drawer event buffer is valid");
        let drawer_draw = records(&drawer_buffer, drawer_len)
            .find(|record| record[0] == i32::from(EventKind::Draw.code()))
            .expect("draw event is in the step delta");
        assert_eq!(drawer_draw[3], i32::from(draw.tile.as_u8()));

        let other = draw.player.next();
        let mut other_buffer = [0_i32; 16 * EVENT_RECORD_WIDTH];
        let other_len = game
            .step_events_into(other, &mut other_buffer)
            .expect("other event buffer is valid");
        let other_draw = records(&other_buffer, other_len)
            .find(|record| record[0] == i32::from(EventKind::Draw.code()))
            .expect("public draw event is visible to other players");
        assert_eq!(other_draw[3], -1);
        return;
    }
    panic!("the deterministic seed range did not produce a normal draw");
}

#[test]
fn batch_step_event_delta_matches_each_game() {
    const SIZE: usize = 64;
    const CAPACITY: usize = 16;
    let mut batch = Batch::new(SIZE, 19);
    let mut actions = vec![0_u8; SIZE];
    for (game, action) in batch.games().iter().zip(actions.iter_mut()) {
        *action = first_legal(game).index() as u8;
    }
    let mut records = vec![0_i64; SIZE * 12];
    batch
        .step_indices_into(&actions, &mut records)
        .expect("batch actions are legal");
    let mut events = vec![0_i32; SIZE * CAPACITY * EVENT_RECORD_WIDTH];
    let mut lengths = vec![0_u16; SIZE];
    batch
        .step_events_into(CAPACITY, &mut events, &mut lengths)
        .expect("batch event output is valid");

    for (index, game) in batch.games().iter().enumerate() {
        let mut expected = vec![0_i32; CAPACITY * EVENT_RECORD_WIDTH];
        let viewer = game
            .decision()
            .map_or(game.dealer(), |decision| decision.actor);
        let expected_len = game
            .step_events_into(viewer, &mut expected)
            .expect("single event output is valid");
        let start = index * CAPACITY * EVENT_RECORD_WIDTH;
        assert_eq!(usize::from(lengths[index]), expected_len);
        assert_eq!(
            &events[start..start + CAPACITY * EVENT_RECORD_WIDTH],
            expected.as_slice()
        );
    }
}

#[test]
fn batch_masked_history_writes_only_matching_viewers() {
    const SIZE: usize = 128;
    const CAPACITY: usize = 8;
    let batch = Batch::new(SIZE, 31);
    let mut history_seat_masks = vec![0_u8; SIZE];
    for (index, game) in batch.games().iter().enumerate() {
        let viewer = game
            .decision()
            .map_or(game.dealer(), |decision| decision.actor);
        let seat = if index % 2 == 0 {
            viewer.as_u8()
        } else {
            (viewer.as_u8() + 1) % 4
        };
        history_seat_masks[index] = 1 << seat;
    }
    let sentinel = -777_i32;
    let mut events = vec![sentinel; SIZE * CAPACITY * EVENT_RECORD_WIDTH];
    let mut lengths = vec![u16::MAX; SIZE];
    batch
        .events_masked_into(&history_seat_masks, CAPACITY, &mut events, &mut lengths)
        .expect("masked event output is valid");

    for index in 0..SIZE {
        let row = &events
            [index * CAPACITY * EVENT_RECORD_WIDTH..(index + 1) * CAPACITY * EVENT_RECORD_WIDTH];
        if index % 2 == 0 {
            assert_eq!(lengths[index], 1);
            assert_eq!(row[0], i32::from(EventKind::GameStart.code()));
        } else {
            assert_eq!(lengths[index], 0);
            assert!(row.iter().all(|&value| value == sentinel));
        }
    }

    assert!(matches!(
        batch.events_masked_into(
            &history_seat_masks[..SIZE - 1],
            CAPACITY,
            &mut events,
            &mut lengths,
        ),
        Err(bloodflow_mahjong::GameError::BatchLength)
    ));
}
