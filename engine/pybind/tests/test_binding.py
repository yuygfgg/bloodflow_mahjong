import numpy as np
import pytest

import bloodflow_mahjong as bm


def first_legal_action(words: np.ndarray) -> int:
    for word_index, value in enumerate(words):
        bits = int(value)
        if bits:
            return word_index * 64 + (bits & -bits).bit_length() - 1
    raise AssertionError("non-terminal environment has no legal action")


def buffers(batch_size: int) -> tuple[np.ndarray, ...]:
    return (
        np.empty((batch_size, bm.STEP_RECORD_WIDTH), dtype=np.int64),
        np.empty((batch_size, bm.LEGAL_ACTION_MASK_WORDS), dtype=np.uint64),
        np.empty(
            (batch_size, bm.TILE_OBSERVATION_PLANES, bm.TILE_KIND_COUNT),
            dtype=np.uint8,
        ),
        np.empty(
            (batch_size, bm.PLAYER_COUNT, bm.MELD_SLOTS, bm.MELD_FIELDS),
            dtype=np.uint8,
        ),
        np.empty((batch_size, bm.RIVER_TILE_CAPACITY, bm.RIVER_FIELDS), dtype=np.uint8),
        np.empty((batch_size, bm.META_OBSERVATION_WIDTH), dtype=np.int32),
    )


def test_game_step_and_state_buffers() -> None:
    assert (
        bm.ACTION_EXCHANGE_TILE_OFFSET,
        bm.ACTION_CHOOSE_MISSING_OFFSET,
        bm.ACTION_DISCARD_OFFSET,
        bm.ACTION_HU,
        bm.ACTION_PONG,
        bm.ACTION_EXPOSED_KONG,
        bm.ACTION_CONCEALED_KONG_OFFSET,
        bm.ACTION_ADDED_KONG_OFFSET,
        bm.ACTION_PASS,
    ) == (0, 27, 30, 57, 58, 59, 60, 87, 114)
    game = bm.Game(seed=42)
    assert game.phase == bm.PHASE_EXCHANGE
    assert game.decision == (0, bm.PHASE_EXCHANGE)

    hand = np.empty(bm.TILE_KIND_COUNT, dtype=np.uint8)
    game.concealed_into(0, hand)
    assert int(hand.sum()) == 14

    action = first_legal_action(np.asarray(game.legal_action_mask, dtype=np.uint64))
    record = game.step_id(action)
    assert len(record) == bm.STEP_RECORD_WIDTH


def test_game_step_into_rejects_unaligned_output_before_mutating() -> None:
    game = bm.Game(seed=5)
    before = game.legal_action_mask
    action = first_legal_action(np.asarray(before, dtype=np.uint64))
    storage = np.empty(bm.STEP_RECORD_WIDTH * 8 + 1, dtype=np.uint8)
    unaligned = storage[1:].view(np.int64)
    assert unaligned.flags.c_contiguous and not unaligned.flags.aligned

    with pytest.raises(ValueError, match="aligned"):
        game.step_into(action, unaligned)
    assert game.legal_action_mask == before


def test_batch_uses_fixed_numpy_buffers() -> None:
    batch = bm.Batch(128, seed=7)
    masks = np.empty((len(batch), bm.LEGAL_ACTION_MASK_WORDS), dtype=np.uint64)
    actions = np.empty(len(batch), dtype=np.uint8)
    records = np.empty((len(batch), bm.STEP_RECORD_WIDTH), dtype=np.int64)

    batch.legal_action_masks_into(masks)
    for index, words in enumerate(masks):
        actions[index] = first_legal_action(words)
    batch.step_into(actions, records)

    assert np.all(records[:, 11] == 0)
    assert np.all(records[:, 9] >= 0)


def test_observe_and_combined_step_fill_caller_buffers() -> None:
    size = 16
    batch = bm.Batch(size, seed=13)
    records, masks, tile_obs, melds, river, meta = buffers(size)
    actions = np.empty(size, dtype=np.uint8)

    batch.observe_into(tile_obs, melds, river, meta)
    assert np.all(tile_obs[:, 0].sum(axis=1) == 14)
    assert np.all(meta[:, 0] == bm.PHASE_EXCHANGE)

    batch.legal_action_masks_into(masks)
    for index, words in enumerate(masks):
        actions[index] = first_legal_action(words)
    batch.step_and_observe_into(actions, records, masks, tile_obs, melds, river, meta)
    assert np.all(meta[:, 0] == bm.PHASE_EXCHANGE)
    assert np.all(records[:, 9] == 0)


def test_combined_step_rejects_overlapping_output_views() -> None:
    size = 2
    batch = bm.Batch(size, seed=3)
    records, masks, tile_obs, _, river, meta = buffers(size)
    actions = np.empty(size, dtype=np.uint8)
    batch.legal_action_masks_into(masks)
    for index, words in enumerate(masks):
        actions[index] = first_legal_action(words)
    before = masks.copy()

    shared = np.empty(size * bm.TILE_OBSERVATION_WIDTH, dtype=np.uint8)
    overlapping_tile_obs = shared.reshape(
        size, bm.TILE_OBSERVATION_PLANES, bm.TILE_KIND_COUNT
    )
    overlapping_melds = shared[: size * bm.MELD_OBSERVATION_WIDTH].reshape(
        size, bm.PLAYER_COUNT, bm.MELD_SLOTS, bm.MELD_FIELDS
    )
    with pytest.raises(ValueError, match="overlaps"):
        batch.step_and_observe_into(
            actions,
            records,
            masks,
            overlapping_tile_obs,
            overlapping_melds,
            river,
            meta,
        )
    after = np.empty_like(masks)
    batch.legal_action_masks_into(after)
    np.testing.assert_array_equal(after, before)


def test_reset_many_validates_all_indices_before_mutating() -> None:
    batch = bm.Batch(4, seed=19)
    masks = np.empty((4, bm.LEGAL_ACTION_MASK_WORDS), dtype=np.uint64)
    actions = np.empty(4, dtype=np.uint8)
    records = np.empty((4, bm.STEP_RECORD_WIDTH), dtype=np.int64)
    batch.legal_action_masks_into(masks)
    for index, words in enumerate(masks):
        actions[index] = first_legal_action(words)
    batch.step_into(actions, records)
    before = np.empty_like(masks)
    batch.legal_action_masks_into(before)

    with pytest.raises(ValueError, match="out of range"):
        batch.reset_many(
            np.asarray([0, 99], dtype=np.uint32),
            np.asarray([100, 101], dtype=np.uint64),
        )
    batch.legal_action_masks_into(masks)
    np.testing.assert_array_equal(masks, before)

    batch.reset_many(
        np.asarray([0, 2], dtype=np.uint32),
        np.asarray([100, 101], dtype=np.uint64),
    )


def test_reset_many_rejects_shape_dtype_and_layout() -> None:
    batch = bm.Batch(4)
    with pytest.raises(ValueError, match="shape"):
        batch.reset_many(
            np.asarray([0, 1], dtype=np.uint32), np.asarray([1], dtype=np.uint64)
        )
    with pytest.raises(TypeError):
        batch.reset_many(
            np.asarray([0, 1], dtype=np.int64), np.asarray([1, 2], dtype=np.uint64)
        )
    with pytest.raises(ValueError, match="C-contiguous"):
        batch.reset_many(
            np.asarray([0, 9, 1, 9], dtype=np.uint32)[::2],
            np.asarray([1, 2], dtype=np.uint64),
        )


def test_batch_rejects_wrong_shape_dtype_and_layout() -> None:
    batch = bm.Batch(8)
    with pytest.raises(ValueError, match="shape"):
        batch.legal_action_masks_into(np.empty((8, 3), dtype=np.uint64))
    with pytest.raises(TypeError):
        batch.legal_action_masks_into(np.empty((8, 2), dtype=np.int64))

    masks = np.empty((8, 4), dtype=np.uint64)[:, ::2]
    assert not masks.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        batch.legal_action_masks_into(masks)

    actions = np.zeros(16, dtype=np.uint8)[::2]
    records = np.empty((8, bm.STEP_RECORD_WIDTH), dtype=np.int64)
    with pytest.raises(ValueError, match="C-contiguous"):
        batch.step_into(actions, records)


def test_batch_rejects_illegal_actions_atomically() -> None:
    batch = bm.Batch(4, seed=9)
    before = np.empty((4, 2), dtype=np.uint64)
    after = np.empty_like(before)
    records = np.empty((4, bm.STEP_RECORD_WIDTH), dtype=np.int64)
    actions = np.full(4, bm.ACTION_SPACE_SIZE - 1, dtype=np.uint8)

    batch.legal_action_masks_into(before)
    with pytest.raises(ValueError, match="legal"):
        batch.step_into(actions, records)
    batch.legal_action_masks_into(after)
    np.testing.assert_array_equal(after, before)


def test_game_event_history_and_step_delta_are_zero_copy_buffers() -> None:
    game = bm.Game(seed=42)
    history = np.empty((8, bm.EVENT_RECORD_WIDTH), dtype=np.int32)
    assert game.event_count == 1
    length = game.events_into(0, history)
    assert length == 1
    assert history[0, 0] == bm.EventKind.GAME_START

    action = first_legal_action(np.asarray(game.legal_action_mask, dtype=np.uint64))
    game.step_id(action)
    delta = np.empty((4, bm.EVENT_RECORD_WIDTH), dtype=np.int32)
    delta_length = game.step_events_into(0, delta)
    assert delta_length == 1
    assert delta[0, 0] == bm.EventKind.ACTION
    assert delta[0, 5] == action
    assert delta[0, 6] == bm.PHASE_EXCHANGE


def test_batch_event_history_and_combined_step_delta() -> None:
    size = 16
    capacity = 16
    batch = bm.Batch(size, seed=77)
    history = np.empty((size, capacity, bm.EVENT_RECORD_WIDTH), dtype=np.int32)
    lengths = np.empty(size, dtype=np.uint16)
    batch.events_into(history, lengths)
    assert np.all(lengths == 1)
    assert np.all(history[:, 0, 0] == bm.EventKind.GAME_START)

    records, masks, tile_obs, melds, river, meta = buffers(size)
    actions = np.empty(size, dtype=np.uint8)
    batch.legal_action_masks_into(masks)
    for index, words in enumerate(masks):
        actions[index] = first_legal_action(words)
    events = np.empty((size, capacity, bm.EVENT_RECORD_WIDTH), dtype=np.int32)
    event_lengths = np.empty(size, dtype=np.uint16)
    batch.step_and_observe_events_into(
        actions,
        records,
        masks,
        tile_obs,
        melds,
        river,
        meta,
        events,
        event_lengths,
    )
    assert np.all(event_lengths >= 1)
    assert np.all(events[:, 0, 0] == bm.EventKind.ACTION)


def test_event_buffers_validate_shape_dtype_and_capacity() -> None:
    game = bm.Game(seed=3)
    with pytest.raises(ValueError, match="shape"):
        game.events_into(0, np.empty((4, 7), dtype=np.int32))
    with pytest.raises(TypeError):
        game.events_into(0, np.empty((4, bm.EVENT_RECORD_WIDTH), dtype=np.int64))

    batch = bm.Batch(2, seed=4)
    with pytest.raises(ValueError, match="capacity"):
        batch.events_into(
            np.empty(
                (2, bm.EVENT_HISTORY_CAPACITY + 1, bm.EVENT_RECORD_WIDTH),
                dtype=np.int32,
            ),
            np.empty(2, dtype=np.uint16),
        )
