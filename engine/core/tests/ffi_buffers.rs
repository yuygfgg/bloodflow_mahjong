use bloodflow_mahjong::{
    ActionId, Batch, Game, GameError, LEGAL_ACTION_MASK_WORDS, MELD_OBSERVATION_WIDTH,
    META_OBSERVATION_WIDTH, Phase, RIVER_OBSERVATION_WIDTH, STEP_RECORD_WIDTH, Seat, StepOutcome,
    TILE_OBSERVATION_WIDTH,
};

const BATCH_SIZE: usize = 64;
const TILE_KIND_COUNT: usize = 27;

fn first_actions(batch: &Batch, output: &mut [u8]) {
    for (game, action) in batch.games().iter().zip(output) {
        *action = game
            .legal_action_mask()
            .expect("test game is not terminal")
            .iter()
            .next()
            .expect("a live decision has a legal action")
            .index() as u8;
    }
}

fn expected_record(outcome: StepOutcome) -> [i64; STEP_RECORD_WIDTH] {
    let mut record = [0_i64; STEP_RECORD_WIDTH];
    match outcome.draw {
        Some(draw) => {
            record[0] = i64::from(draw.player.as_u8());
            record[1] = i64::from(draw.tile.as_u8());
            record[2] = i64::from(u8::from(draw.replacement));
        }
        None => {
            record[0] = -1;
            record[1] = -1;
        }
    }
    match outcome.discard {
        Some(discard) => {
            record[3] = i64::from(discard.player.as_u8());
            record[4] = i64::from(discard.tile.as_u8());
        }
        None => {
            record[3] = -1;
            record[4] = -1;
        }
    }
    record[5..9].copy_from_slice(&outcome.score_delta);
    match outcome.next {
        Some(next) => {
            record[9] = i64::from(next.actor.as_u8());
            record[10] = i64::from(next.phase.code());
        }
        None => {
            record[9] = -1;
            record[10] = -1;
        }
    }
    record[11] = i64::from(u8::from(outcome.terminal));
    record
}

fn relative_seat(viewer: Seat, seat: Seat) -> usize {
    (seat.index() + 4 - viewer.index()) % 4
}

fn assert_observation_matches_public_state(
    game: &Game,
    tile_obs: &[u8],
    melds: &[u8],
    river: &[u8],
    meta: &[i32],
) {
    let decision = game.decision();
    let viewer = decision.map_or(game.dealer(), |decision| decision.actor);
    assert_eq!(&tile_obs[..TILE_KIND_COUNT], game.concealed(viewer));
    assert_eq!(
        &tile_obs[TILE_KIND_COUNT..2 * TILE_KIND_COUNT],
        game.exchange_selection(viewer)
    );

    let discards = game.discards().collect::<Vec<_>>();
    for relative in 0..4 {
        let seat = viewer.offset(relative as u8);
        let locked_start = (2 + relative) * TILE_KIND_COUNT;
        let expected_wins = game.public_win_tiles(seat);
        let expected_tiles = if seat == viewer {
            game.locked(seat)
        } else {
            &expected_wins
        };
        assert_eq!(
            &tile_obs[locked_start..locked_start + TILE_KIND_COUNT],
            expected_tiles
        );
        for tile_index in 0..TILE_KIND_COUNT {
            let expected = discards
                .iter()
                .filter(|&&(owner, tile)| owner == seat && tile.index() == tile_index)
                .count() as u8;
            assert_eq!(
                tile_obs[(6 + relative) * TILE_KIND_COUNT + tile_index],
                expected
            );
        }

        for meld_index in 0..4 {
            let offset = (relative * 4 + meld_index) * 3;
            match game.meld(seat, meld_index) {
                Some(meld) => {
                    assert_eq!(melds[offset], meld.tile.as_u8());
                    assert_eq!(melds[offset + 1], meld.kind.code());
                    assert_eq!(
                        melds[offset + 2] as usize,
                        relative_seat(viewer, meld.source)
                    );
                }
                None => assert_eq!(&melds[offset..offset + 3], &[u8::MAX; 3]),
            }
        }
    }

    for index in 0..108 {
        let offset = index * 2;
        if let Some(&(owner, tile)) = discards.get(index) {
            assert_eq!(river[offset], tile.as_u8());
            assert_eq!(river[offset + 1] as usize, relative_seat(viewer, owner));
        } else {
            assert_eq!(&river[offset..offset + 2], &[u8::MAX; 2]);
        }
    }

    assert_eq!(meta[0], i32::from(game.phase().code()));
    assert_eq!(
        meta[1],
        decision.map_or(-1, |decision| i32::from(decision.actor.as_u8()))
    );
    assert_eq!(meta[2] as usize, relative_seat(viewer, game.dealer()));
    assert_eq!(meta[3], i32::from(game.exchange_direction() as u8));
    assert_eq!(meta[4], game.wall_remaining() as i32);
    let draw = game.current_draw();
    assert_eq!(
        meta[5],
        draw.map_or(-1, |draw| i32::from(draw.tile.as_u8()))
    );
    assert_eq!(
        meta[6],
        draw.map_or(0, |draw| i32::from(u8::from(draw.replacement)))
    );
    assert_eq!(meta[9], discards.len() as i32);

    if game.phase() == Phase::Exchange {
        let selection = game.exchange_selection(viewer);
        let selected: i32 = selection.iter().map(|&count| i32::from(count)).sum();
        let suit = selection
            .iter()
            .position(|&count| count != 0)
            .map_or(-1, |index| (index / 9) as i32);
        assert_eq!(meta[10], selected);
        assert_eq!(meta[11], suit);
    } else {
        assert_eq!(meta[10], 0);
        assert_eq!(meta[11], -1);
    }

    if matches!(game.phase(), Phase::HuResponse | Phase::MeldResponse) {
        let &(source, tile) = discards
            .last()
            .expect("early test response follows a discard");
        assert_eq!(meta[7] as usize, relative_seat(viewer, source));
        assert_eq!(meta[8], i32::from(tile.as_u8()));
    } else {
        assert_eq!(meta[7], -1);
        assert_eq!(meta[8], -1);
    }

    for relative in 0..4 {
        let seat = viewer.offset(relative as u8);
        assert_eq!(meta[12 + relative], game.score(seat) as i32);
        assert_eq!(
            meta[16 + relative],
            game.missing_suit(seat)
                .map_or(-1, |suit| i32::from(suit as u8))
        );
        assert_eq!(meta[20 + relative], i32::from(u8::from(game.has_won(seat))));
        assert_eq!(
            meta[24 + relative],
            game.concealed(seat)
                .iter()
                .map(|&count| i32::from(count))
                .sum::<i32>()
        );
        assert_eq!(meta[30 + relative], game.max_win_multiplier(seat) as i32);
    }
    assert_eq!(
        meta[28],
        i32::from(u8::from(game.phase() == Phase::Finished))
    );
    let opening_discard =
        game.phase() == Phase::HuResponse && discards.len() == 1 && discards[0].0 == game.dealer();
    assert_eq!(meta[29], i32::from(u8::from(opening_discard)) << 2);
}

#[test]
fn ffi_buffers_match_typed_batch_across_states() {
    assert_eq!(LEGAL_ACTION_MASK_WORDS, 2);
    assert_eq!(STEP_RECORD_WIDTH, 12);
    assert_eq!(Phase::Exchange.code(), 0);
    assert_eq!(Phase::ChooseMissing.code(), 1);
    assert_eq!(Phase::Turn.code(), 2);
    assert_eq!(Phase::HuResponse.code(), 3);
    assert_eq!(Phase::MeldResponse.code(), 4);
    assert_eq!(Phase::Finished.code(), 5);

    let mut buffered = Batch::new(BATCH_SIZE, 0xfeed_5eed);
    let mut typed = buffered.clone();
    let mut typed_masks = vec![None; BATCH_SIZE];
    let mut mask_words = vec![u64::MAX; BATCH_SIZE * LEGAL_ACTION_MASK_WORDS];
    let mut raw_actions = vec![0_u8; BATCH_SIZE];
    let mut typed_actions = vec![ActionId::HU; BATCH_SIZE];
    let mut typed_outcomes = vec![StepOutcome::default(); BATCH_SIZE];
    let mut records = vec![i64::MIN; BATCH_SIZE * STEP_RECORD_WIDTH];

    for _ in 0..48 {
        typed
            .legal_action_masks_into(&mut typed_masks)
            .expect("typed mask output has the right length");
        buffered
            .legal_action_mask_words_into(&mut mask_words)
            .expect("packed mask output has the right length");
        for (mask, words) in typed_masks
            .iter()
            .zip(mask_words.chunks_exact(LEGAL_ACTION_MASK_WORDS))
        {
            assert_eq!(mask.expect("test game is live").words(), words);
        }

        first_actions(&buffered, &mut raw_actions);
        for (typed_action, &raw_action) in typed_actions.iter_mut().zip(&raw_actions) {
            *typed_action = ActionId::new(raw_action as usize).expect("mask ids are valid");
        }
        typed
            .step_ids(&typed_actions, &mut typed_outcomes)
            .expect("typed actions selected from masks are legal");
        buffered
            .step_indices_into(&raw_actions, &mut records)
            .expect("raw actions selected from masks are legal");

        for (outcome, record) in typed_outcomes
            .iter()
            .zip(records.chunks_exact(STEP_RECORD_WIDTH))
        {
            assert_eq!(expected_record(*outcome).as_slice(), record);
        }
        assert_eq!(
            format!("{:?}", typed.games()),
            format!("{:?}", buffered.games())
        );
    }
}

#[test]
fn observation_buffers_match_public_state_in_parallel() {
    assert_eq!(TILE_OBSERVATION_WIDTH, 270);
    assert_eq!(MELD_OBSERVATION_WIDTH, 48);
    assert_eq!(RIVER_OBSERVATION_WIDTH, 216);
    assert_eq!(META_OBSERVATION_WIDTH, 34);

    let mut batch = Batch::new(BATCH_SIZE, 0x0b5e_7e57);
    let mut actions = vec![0_u8; BATCH_SIZE];
    let mut records = vec![0_i64; BATCH_SIZE * STEP_RECORD_WIDTH];
    for _ in 0..20 {
        first_actions(&batch, &mut actions);
        batch
            .step_indices_into(&actions, &mut records)
            .expect("masked actions are legal");
    }

    let mut tile_obs = vec![u8::MAX; BATCH_SIZE * TILE_OBSERVATION_WIDTH];
    let mut melds = vec![0_u8; BATCH_SIZE * MELD_OBSERVATION_WIDTH];
    let mut river = vec![0_u8; BATCH_SIZE * RIVER_OBSERVATION_WIDTH];
    let mut meta = vec![i32::MIN; BATCH_SIZE * META_OBSERVATION_WIDTH];
    batch
        .observations_into(&mut tile_obs, &mut melds, &mut river, &mut meta)
        .expect("observation buffers have the right lengths");

    for index in 0..BATCH_SIZE {
        assert_observation_matches_public_state(
            &batch.games()[index],
            &tile_obs[index * TILE_OBSERVATION_WIDTH..(index + 1) * TILE_OBSERVATION_WIDTH],
            &melds[index * MELD_OBSERVATION_WIDTH..(index + 1) * MELD_OBSERVATION_WIDTH],
            &river[index * RIVER_OBSERVATION_WIDTH..(index + 1) * RIVER_OBSERVATION_WIDTH],
            &meta[index * META_OBSERVATION_WIDTH..(index + 1) * META_OBSERVATION_WIDTH],
        );
    }
}

#[test]
fn ffi_records_and_terminal_masks_use_documented_sentinels() {
    let mut batch = Batch::new(1, 0x0517_11e1);
    let mut action = [0_u8; 1];
    let mut record = [i64::MIN; STEP_RECORD_WIDTH];

    first_actions(&batch, &mut action);
    batch
        .step_indices_into(&action, &mut record)
        .expect("initial exchange action is legal");
    assert_eq!(&record[0..2], &[-1, -1]);
    assert_eq!(record[2], 0);
    assert_eq!(&record[3..5], &[-1, -1]);
    assert_eq!(&record[5..9], &[0, 0, 0, 0]);
    assert_eq!(record[11], 0);

    let mut steps = 1;
    while record[11] == 0 {
        first_actions(&batch, &mut action);
        batch
            .step_indices_into(&action, &mut record)
            .expect("masked action is legal");
        steps += 1;
        assert!(steps < 1_024, "deterministic game should terminate");
    }

    assert_eq!(batch.games()[0].phase(), Phase::Finished);
    assert_eq!(record[9], -1);
    assert_eq!(record[10], -1);
    assert_eq!(record[11], 1);
    let mut terminal_mask = [u64::MAX; LEGAL_ACTION_MASK_WORDS];
    batch
        .legal_action_mask_words_into(&mut terminal_mask)
        .expect("terminal mask output has the right length");
    assert_eq!(terminal_mask, [0; LEGAL_ACTION_MASK_WORDS]);

    let mut tile_obs = [u8::MAX; TILE_OBSERVATION_WIDTH];
    let mut melds = [0_u8; MELD_OBSERVATION_WIDTH];
    let mut river = [0_u8; RIVER_OBSERVATION_WIDTH];
    let mut meta = [i32::MIN; META_OBSERVATION_WIDTH];
    batch
        .observations_into(&mut tile_obs, &mut melds, &mut river, &mut meta)
        .expect("terminal observation lengths are valid");
    assert_observation_matches_public_state(&batch.games()[0], &tile_obs, &melds, &river, &meta);
    assert_eq!(meta[1], -1);
    assert_eq!(meta[2], 0);
    assert_eq!(meta[28], 1);
}

#[test]
fn raw_batch_rejects_bad_lengths_ids_and_actions_atomically() {
    let mut batch = Batch::new(BATCH_SIZE, 0xbad1_dea5);
    let mut actions = vec![0_u8; BATCH_SIZE];
    first_actions(&batch, &mut actions);
    let mut records = vec![777_i64; BATCH_SIZE * STEP_RECORD_WIDTH];
    let original_records = records.clone();
    let original_games = format!("{:?}", batch.games());

    let mut tile_obs = vec![11_u8; BATCH_SIZE * TILE_OBSERVATION_WIDTH];
    let original_tile_obs = tile_obs.clone();
    let mut melds = vec![22_u8; BATCH_SIZE * MELD_OBSERVATION_WIDTH];
    let mut river = vec![33_u8; BATCH_SIZE * RIVER_OBSERVATION_WIDTH];
    let mut meta = vec![44_i32; BATCH_SIZE * META_OBSERVATION_WIDTH - 1];
    assert_eq!(
        batch.observations_into(&mut tile_obs, &mut melds, &mut river, &mut meta),
        Err(GameError::BatchLength)
    );
    assert_eq!(tile_obs, original_tile_obs);
    assert!(melds.iter().all(|&value| value == 22));
    assert!(river.iter().all(|&value| value == 33));
    assert!(meta.iter().all(|&value| value == 44));

    let mut short_masks = vec![u64::MAX; BATCH_SIZE * LEGAL_ACTION_MASK_WORDS - 1];
    assert_eq!(
        batch.legal_action_mask_words_into(&mut short_masks),
        Err(GameError::BatchLength)
    );
    assert!(short_masks.iter().all(|&word| word == u64::MAX));
    assert_eq!(
        batch.step_indices_into(&actions[..BATCH_SIZE - 1], &mut records),
        Err(GameError::BatchLength)
    );
    assert_eq!(records, original_records);
    assert_eq!(
        batch.step_indices_into(&actions, &mut records[..BATCH_SIZE * STEP_RECORD_WIDTH - 1]),
        Err(GameError::BatchLength)
    );
    assert_eq!(records, original_records);

    actions[17] = u8::MAX;
    assert_eq!(
        batch.step_indices_into(&actions, &mut records),
        Err(GameError::InvalidAction)
    );
    assert_eq!(format!("{:?}", batch.games()), original_games);
    assert_eq!(records, original_records);

    first_actions(&batch, &mut actions);
    actions[31] = ActionId::PASS.index() as u8;
    assert_eq!(
        batch.step_indices_into(&actions, &mut records),
        Err(GameError::InvalidAction)
    );
    assert_eq!(format!("{:?}", batch.games()), original_games);
    assert_eq!(records, original_records);
}
