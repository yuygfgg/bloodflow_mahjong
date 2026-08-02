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
    rule_action = game.simple_rule_action()
    assert rule_action is not None
    rule_word = rule_action // 64
    rule_bit = rule_action % 64
    assert int(game.legal_action_mask[rule_word]) & (1 << rule_bit)

    hand = np.empty(bm.TILE_KIND_COUNT, dtype=np.uint8)
    game.concealed_into(0, hand)
    assert int(hand.sum()) == 14
    shanten, improving_tiles = game.hand_analysis(0)
    assert bm.SHANTEN_COMPLETE <= shanten <= bm.SHANTEN_MAX
    assert improving_tiles >> bm.TILE_KIND_COUNT == 0

    with pytest.raises(ValueError, match="seat"):
        game.hand_analysis(4)

    action = first_legal_action(np.asarray(game.legal_action_mask, dtype=np.uint64))
    record = game.step_id(action)
    assert len(record) == bm.STEP_RECORD_WIDTH


def test_rule_configs_validate_budgets_and_defaults() -> None:
    ev = bm.RuleEvConfig()
    assert (ev.search_depth, ev.defense) == (1, True)
    assert (bm.RuleEvConfig.fast().search_depth, bm.RuleEvConfig.fast().defense) == (
        0,
        True,
    )
    assert "search_depth=1" in repr(ev)
    with pytest.raises(ValueError, match="search_depth"):
        bm.RuleEvConfig(search_depth=4)
    with pytest.raises(AttributeError):
        ev.search_depth = 0

    planner = bm.RulePlannerConfig()
    assert (
        planner.hand_changes,
        planner.draw_horizon,
        planner.candidate_states,
        planner.belief_worlds,
        planner.response_worlds,
        planner.search_iterations,
    ) == (0, 1, 1, 64, 0, 64)
    assert "candidate_states=1" in repr(planner)

    invalid_values = (
        ("hand_changes", 3),
        ("draw_horizon", 33),
        ("candidate_states", 0),
        ("belief_worlds", 257),
        ("response_worlds", 257),
        ("search_iterations", 4097),
    )
    for name, value in invalid_values:
        with pytest.raises(ValueError, match=name):
            bm.RulePlannerConfig(**{name: value})


def test_game_rule_ev_and_planner_actions_are_legal() -> None:
    game = bm.Game(seed=43)
    minimal_planner = bm.RulePlannerConfig(
        hand_changes=0,
        draw_horizon=0,
        candidate_states=1,
    )
    for action in (game.rule_ev_action(), game.rule_planner_action()):
        assert action is not None
        word = action // 64
        bit = action % 64
        assert int(game.legal_action_mask[word]) & (1 << bit)

    while game.phase != bm.PHASE_TURN:
        action = game.simple_rule_action()
        assert action is not None
        game.step_id(action)

    actions = (
        game.rule_ev_action(bm.RuleEvConfig.fast()),
        game.rule_planner_action(minimal_planner),
    )

    for action in actions:
        assert action is not None
        word = action // 64
        bit = action % 64
        assert int(game.legal_action_mask[word]) & (1 << bit)

    with pytest.raises(TypeError):
        game.rule_ev_action(bm.RulePlannerConfig())
    with pytest.raises(TypeError):
        game.rule_planner_action(bm.RuleEvConfig.fast())


def test_batch_rule_ev_and_planner_write_fixed_buffers() -> None:
    size = 8
    batch = bm.Batch(size, seed=47)
    masks = np.empty((size, bm.LEGAL_ACTION_MASK_WORDS), dtype=np.uint64)
    ev_actions = np.empty(size, dtype=np.uint8)
    planner_actions = np.empty(size, dtype=np.uint8)
    minimal_planner = bm.RulePlannerConfig(
        hand_changes=0,
        draw_horizon=0,
        candidate_states=1,
    )
    records = np.empty((size, bm.STEP_RECORD_WIDTH), dtype=np.int64)
    _, _, tile_obs, melds, river, meta = buffers(size)
    for _ in range(32):
        batch.observe_into(tile_obs, melds, river, meta)
        if np.all(meta[:, 0] == bm.PHASE_TURN):
            break
        batch.simple_rule_actions_into(ev_actions)
        batch.step_into(ev_actions, records)
    else:
        raise AssertionError("batch did not reach the first turn")

    batch.legal_action_masks_into(masks)
    batch.rule_ev_actions_into(ev_actions, bm.RuleEvConfig.fast())
    batch.rule_planner_actions_into(planner_actions, minimal_planner)

    for actions in (ev_actions, planner_actions):
        for action, words in zip(actions, masks, strict=True):
            assert int(words[int(action) // 64]) & (1 << (int(action) % 64))

    enabled = (np.arange(size) % 2 == 0).astype(np.uint8)
    sentinel = np.uint8(0xA5)
    ev_masked = np.full(size, sentinel, dtype=np.uint8)
    planner_masked = np.full(size, sentinel, dtype=np.uint8)
    batch.rule_ev_actions_masked_into(
        enabled, ev_masked, bm.RuleEvConfig.fast()
    )
    batch.rule_planner_actions_masked_into(
        enabled, planner_masked, minimal_planner
    )
    np.testing.assert_array_equal(ev_masked[enabled == 1], ev_actions[enabled == 1])
    np.testing.assert_array_equal(
        planner_masked[enabled == 1], planner_actions[enabled == 1]
    )
    assert np.all(ev_masked[enabled == 0] == sentinel)
    assert np.all(planner_masked[enabled == 0] == sentinel)

    with pytest.raises(ValueError, match="shape"):
        batch.rule_ev_actions_into(ev_actions[:-1])
    invalid = enabled.copy()
    invalid[1] = 2
    with pytest.raises(ValueError, match="action"):
        batch.rule_planner_actions_masked_into(invalid, planner_masked)

    assert bm.RULE_EV_ACTION_TERMINAL == np.iinfo(np.uint8).max
    assert bm.RULE_PLANNER_ACTION_TERMINAL == np.iinfo(np.uint8).max


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
    batch.simple_rule_actions_into(actions)
    for action, words in zip(actions, masks, strict=True):
        assert int(words[int(action) // 64]) & (1 << (int(action) % 64))

    shanten = np.empty(len(batch), dtype=np.int8)
    improving_tiles = np.empty(len(batch), dtype=np.uint32)
    batch.hand_analysis_into(shanten, improving_tiles)
    assert np.all(shanten >= bm.SHANTEN_COMPLETE)
    assert np.all(shanten <= bm.SHANTEN_MAX)
    assert np.all(improving_tiles >> bm.TILE_KIND_COUNT == 0)

    for index, words in enumerate(masks):
        actions[index] = first_legal_action(words)
    batch.step_into(actions, records)

    assert np.all(records[:, 11] == 0)
    assert np.all(records[:, 9] >= 0)


def test_rule_and_hand_analysis_buffers_validate_shape_and_dtype() -> None:
    batch = bm.Batch(8)
    with pytest.raises(ValueError, match="shape"):
        batch.simple_rule_actions_into(np.empty(7, dtype=np.uint8))
    with pytest.raises(TypeError):
        batch.simple_rule_actions_into(np.empty(8, dtype=np.int8))

    enabled = np.ones(8, dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        batch.simple_rule_actions_masked_into(enabled[:-1], np.empty(8, dtype=np.uint8))
    with pytest.raises(TypeError):
        batch.simple_rule_actions_masked_into(
            enabled.astype(np.int8), np.empty(8, dtype=np.uint8)
        )

    shanten = np.empty(8, dtype=np.int8)
    improving_tiles = np.empty(8, dtype=np.uint32)
    with pytest.raises(ValueError, match="shape"):
        batch.hand_analysis_into(shanten[:-1], improving_tiles)
    with pytest.raises(TypeError):
        batch.hand_analysis_into(np.empty(8, dtype=np.uint8), improving_tiles)


def test_indexed_hand_analysis_matches_full_batch() -> None:
    batch = bm.Batch(128, seed=31)
    full_shanten = np.empty(128, dtype=np.int8)
    full_improving = np.empty(128, dtype=np.uint32)
    batch.hand_analysis_into(full_shanten, full_improving)
    indices = np.asarray([0, 7, 63, 127], dtype=np.uint32)
    shanten = np.empty(len(indices), dtype=np.int8)
    improving = np.empty(len(indices), dtype=np.uint32)

    batch.hand_analysis_indices_into(indices, shanten, improving)

    np.testing.assert_array_equal(shanten, full_shanten[indices])
    np.testing.assert_array_equal(improving, full_improving[indices])


def test_masked_rule_actions_touch_only_enabled_rows() -> None:
    batch = bm.Batch(128, seed=37)
    expected = np.empty(128, dtype=np.uint8)
    batch.simple_rule_actions_into(expected)
    enabled = (np.arange(128) % 3 == 0).astype(np.uint8)
    actions = np.full(128, 0xA5, dtype=np.uint8)

    batch.simple_rule_actions_masked_into(enabled, actions)

    np.testing.assert_array_equal(actions[enabled == 1], expected[enabled == 1])
    assert np.all(actions[enabled == 0] == 0xA5)
    before = actions.copy()
    enabled[1] = 2
    with pytest.raises(ValueError, match="action"):
        batch.simple_rule_actions_masked_into(enabled, actions)
    np.testing.assert_array_equal(actions, before)


def test_observe_and_combined_step_fill_caller_buffers() -> None:
    size = 16
    batch = bm.Batch(size, seed=13)
    records, masks, tile_obs, melds, river, meta = buffers(size)
    actions = np.empty(size, dtype=np.uint8)
    history_seat_masks = np.full(size, 0x0F, dtype=np.uint8)
    events = np.empty((size, 16, bm.EVENT_RECORD_WIDTH), dtype=np.int32)
    event_lengths = np.empty(size, dtype=np.uint16)

    batch.observe_into(tile_obs, melds, river, meta)
    assert np.all(tile_obs[:, 0].sum(axis=1) == 14)
    assert np.all(meta[:, 0] == bm.PHASE_EXCHANGE)

    batch.legal_action_masks_into(masks)
    for index, words in enumerate(masks):
        actions[index] = first_legal_action(words)
    batch.step_and_observe_history_into(
        actions,
        history_seat_masks,
        records,
        masks,
        tile_obs,
        melds,
        river,
        meta,
        events,
        event_lengths,
    )
    assert np.all(meta[:, 0] == bm.PHASE_EXCHANGE)
    assert np.all(records[:, 9] == 0)
    assert np.all(event_lengths >= 2)


def test_combined_step_rejects_overlapping_output_views() -> None:
    size = 2
    batch = bm.Batch(size, seed=3)
    records, masks, tile_obs, _, river, meta = buffers(size)
    actions = np.empty(size, dtype=np.uint8)
    history_seat_masks = np.full(size, 0x0F, dtype=np.uint8)
    events = np.empty((size, 16, bm.EVENT_RECORD_WIDTH), dtype=np.int32)
    event_lengths = np.empty(size, dtype=np.uint16)
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
        batch.step_and_observe_history_into(
            actions,
            history_seat_masks,
            records,
            masks,
            overlapping_tile_obs,
            overlapping_melds,
            river,
            meta,
            events,
            event_lengths,
        )
    after = np.empty_like(masks)
    batch.legal_action_masks_into(after)
    np.testing.assert_array_equal(after, before)


def test_history_selection_and_selective_reset_touch_only_requested_rows() -> None:
    size = 16
    batch = bm.Batch(size, seed=17)
    records, masks, tile_obs, melds, river, meta = buffers(size)
    actions = np.empty(size, dtype=np.uint8)
    events = np.empty((size, 16, bm.EVENT_RECORD_WIDTH), dtype=np.int32)
    event_lengths = np.empty(size, dtype=np.uint16)
    history_seat_masks = np.zeros(size, dtype=np.uint8)
    history_seat_masks[1::2] = 0x0F
    batch.legal_action_masks_into(masks)
    for index, words in enumerate(masks):
        actions[index] = first_legal_action(words)

    batch.step_and_observe_history_into(
        actions,
        history_seat_masks,
        records,
        masks,
        tile_obs,
        melds,
        river,
        meta,
        events,
        event_lengths,
    )
    assert np.all(event_lengths[::2] == 0)
    assert np.all(event_lengths[1::2] >= 2)

    before_tile_obs = tile_obs.copy()
    before_meta = meta.copy()
    reset_flags = np.zeros(size, dtype=np.uint8)
    reset_flags[[1, 3]] = 1
    seeds = np.arange(size, dtype=np.uint64) + 100
    batch.reset_and_observe_history_into(
        reset_flags,
        seeds,
        history_seat_masks,
        masks,
        tile_obs,
        melds,
        river,
        meta,
        events,
        event_lengths,
    )
    untouched = np.asarray([row for row in range(size) if row not in (1, 3)])
    np.testing.assert_array_equal(tile_obs[untouched], before_tile_obs[untouched])
    np.testing.assert_array_equal(meta[untouched], before_meta[untouched])
    assert np.all(meta[[1, 3], 0] == bm.PHASE_EXCHANGE)
    assert np.all(event_lengths[[1, 3]] == 1)


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


def test_clone_indices_repeats_states_and_is_independent() -> None:
    batch = bm.Batch(4, seed=23)
    clone = batch.clone_indices(np.asarray([2, 2, 0], dtype=np.uint32))
    assert len(clone) == 3

    original_masks = np.empty((4, bm.LEGAL_ACTION_MASK_WORDS), dtype=np.uint64)
    clone_masks = np.empty((3, bm.LEGAL_ACTION_MASK_WORDS), dtype=np.uint64)
    batch.legal_action_masks_into(original_masks)
    clone.legal_action_masks_into(clone_masks)
    np.testing.assert_array_equal(clone_masks[0], original_masks[2])
    np.testing.assert_array_equal(clone_masks[1], original_masks[2])
    np.testing.assert_array_equal(clone_masks[2], original_masks[0])

    clone_actions = np.asarray(
        [first_legal_action(words) for words in clone_masks], dtype=np.uint8
    )
    clone_records = np.empty((3, bm.STEP_RECORD_WIDTH), dtype=np.int64)
    clone.step_into(clone_actions, clone_records)
    unchanged_masks = np.empty_like(original_masks)
    batch.legal_action_masks_into(unchanged_masks)
    np.testing.assert_array_equal(unchanged_masks, original_masks)

    with pytest.raises(ValueError, match="out of range"):
        batch.clone_indices(np.asarray([0, 4], dtype=np.uint32))


def test_remove_indices_swap_returns_the_survivor_order() -> None:
    batch = bm.Batch(4, seed=23)
    masks = np.empty((4, bm.LEGAL_ACTION_MASK_WORDS), dtype=np.uint64)
    batch.legal_action_masks_into(masks)

    order = batch.remove_indices_swap(np.asarray([0, 2], dtype=np.uint32))
    assert order == [3, 1]
    retained = np.empty((2, bm.LEGAL_ACTION_MASK_WORDS), dtype=np.uint64)
    batch.legal_action_masks_into(retained)
    np.testing.assert_array_equal(retained, masks[order])

    with pytest.raises(ValueError, match="strictly increasing"):
        batch.remove_indices_swap(np.asarray([1, 0], dtype=np.uint32))
    with pytest.raises(ValueError, match="strictly increasing"):
        batch.remove_indices_swap(np.asarray([0, 0], dtype=np.uint32))
    with pytest.raises(ValueError, match="out of range"):
        batch.remove_indices_swap(np.asarray([2], dtype=np.uint32))
    assert len(batch) == 2


def test_information_set_resampling_preserves_viewer_observation() -> None:
    game = bm.Game(seed=913)
    while game.phase in (bm.PHASE_EXCHANGE, bm.PHASE_CHOOSE_MISSING):
        action = game.simple_rule_action()
        assert action is not None
        game.step_id(action)
    assert game.decision is not None
    viewer = game.decision[0]

    tile_obs = np.empty((bm.TILE_OBSERVATION_PLANES, 27), dtype=np.uint8)
    melds = np.empty((4, bm.MELD_SLOTS, bm.MELD_FIELDS), dtype=np.uint8)
    river = np.empty((bm.RIVER_TILE_CAPACITY, bm.RIVER_FIELDS), dtype=np.uint8)
    meta = np.empty(bm.META_OBSERVATION_WIDTH, dtype=np.int32)
    game.observe_into(viewer, tile_obs, melds, river, meta)

    sampled = game.resample_information_set(71)
    sampled_tile_obs = np.empty_like(tile_obs)
    sampled_melds = np.empty_like(melds)
    sampled_river = np.empty_like(river)
    sampled_meta = np.empty_like(meta)
    sampled.observe_into(
        viewer,
        sampled_tile_obs,
        sampled_melds,
        sampled_river,
        sampled_meta,
    )
    np.testing.assert_array_equal(sampled_tile_obs, tile_obs)
    np.testing.assert_array_equal(sampled_melds, melds)
    np.testing.assert_array_equal(sampled_river, river)
    np.testing.assert_array_equal(sampled_meta, meta)

    oracle = np.empty((bm.ORACLE_TILE_COUNT_PLANES, 27), dtype=np.uint8)
    sampled_oracle = np.empty_like(oracle)
    game.oracle_tile_counts_into(oracle)
    sampled.oracle_tile_counts_into(sampled_oracle)
    np.testing.assert_array_equal(
        oracle[:4].sum(axis=0) + oracle[8],
        sampled_oracle[:4].sum(axis=0) + sampled_oracle[8],
    )
    assert not np.array_equal(sampled_oracle, oracle)


def test_explicit_viewer_observation_masks_another_players_draw() -> None:
    game = bm.Game(seed=127)
    for _ in range(256):
        if game.current_draw is not None:
            break
        action = game.simple_rule_action()
        assert action is not None
        game.step_id(action)
    assert game.current_draw is not None
    drawer = game.current_draw[0]
    other = (drawer + 1) % 4
    drawer_tile = np.empty((bm.TILE_OBSERVATION_PLANES, 27), dtype=np.uint8)
    other_tile = np.empty_like(drawer_tile)
    drawer_melds = np.empty((4, bm.MELD_SLOTS, bm.MELD_FIELDS), dtype=np.uint8)
    other_melds = np.empty_like(drawer_melds)
    drawer_river = np.empty((bm.RIVER_TILE_CAPACITY, bm.RIVER_FIELDS), dtype=np.uint8)
    other_river = np.empty_like(drawer_river)
    drawer_meta = np.empty(bm.META_OBSERVATION_WIDTH, dtype=np.int32)
    other_meta = np.empty_like(drawer_meta)

    game.observe_into(drawer, drawer_tile, drawer_melds, drawer_river, drawer_meta)
    game.observe_into(other, other_tile, other_melds, other_river, other_meta)

    assert drawer_meta[5] == game.current_draw[1]
    assert other_meta[5] == -1
    assert drawer_meta[1] == other_meta[1] == drawer


def test_batch_information_set_sampling_and_oracle_buffers() -> None:
    batch = bm.Batch(4, seed=23)
    _, original_masks, original_tile, original_melds, original_river, original_meta = (
        buffers(4)
    )
    batch.legal_action_masks_into(original_masks)
    batch.observe_into(original_tile, original_melds, original_river, original_meta)
    indices = np.asarray([2, 2, 0], dtype=np.uint32)
    seeds = np.asarray([101, 101, 103], dtype=np.uint64)

    sampled = batch.resample_information_sets(indices, seeds)
    _, sampled_masks, sampled_tile, sampled_melds, sampled_river, sampled_meta = buffers(
        len(indices)
    )
    sampled.legal_action_masks_into(sampled_masks)
    sampled.observe_into(sampled_tile, sampled_melds, sampled_river, sampled_meta)
    np.testing.assert_array_equal(sampled_masks, original_masks[indices])
    np.testing.assert_array_equal(sampled_tile, original_tile[indices])
    np.testing.assert_array_equal(sampled_melds, original_melds[indices])
    np.testing.assert_array_equal(sampled_river, original_river[indices])
    np.testing.assert_array_equal(sampled_meta, original_meta[indices])

    oracle = np.empty((len(indices), bm.ORACLE_TILE_COUNT_PLANES, 27), dtype=np.uint8)
    sampled.oracle_tile_counts_into(oracle)
    np.testing.assert_array_equal(oracle[0], oracle[1])
    assert np.all(oracle[:, 4:8] <= oracle[:, :4])

    with pytest.raises(ValueError, match="shape"):
        batch.resample_information_sets(indices, seeds[:2])
    with pytest.raises(ValueError, match="out of range"):
        batch.resample_information_sets(
            np.asarray([4], dtype=np.uint32), np.asarray([1], dtype=np.uint64)
        )
    with pytest.raises(ValueError, match="shape"):
        sampled.oracle_tile_counts_into(np.empty((3, 8, 27), dtype=np.uint8))


def test_batch_live_wall_resampling_preserves_current_oracle() -> None:
    batch = bm.Batch(4, seed=37)
    actions = np.empty(4, dtype=np.uint8)
    records = np.empty((4, bm.STEP_RECORD_WIDTH), dtype=np.int64)
    for _ in range(20):
        batch.simple_rule_actions_into(actions)
        batch.step_into(actions, records)

    indices = np.asarray([2, 2, 0], dtype=np.uint32)
    seeds = np.asarray([101, 101, 103], dtype=np.uint64)
    sampled = batch.resample_live_walls(indices, seeds)
    original = np.empty((4, bm.ORACLE_TILE_COUNT_PLANES, 27), dtype=np.uint8)
    oracle = np.empty((3, bm.ORACLE_TILE_COUNT_PLANES, 27), dtype=np.uint8)
    batch.oracle_tile_counts_into(original)
    sampled.oracle_tile_counts_into(oracle)
    np.testing.assert_array_equal(oracle, original[indices])
    np.testing.assert_array_equal(oracle[0], oracle[1])

    with pytest.raises(ValueError, match="shape"):
        batch.resample_live_walls(indices, seeds[:2])
    with pytest.raises(ValueError, match="out of range"):
        batch.resample_live_walls(
            np.asarray([4], dtype=np.uint32), np.asarray([1], dtype=np.uint64)
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


def test_batch_masked_step_leaves_disabled_rows_unchanged() -> None:
    batch = bm.Batch(8, seed=211)
    actions = np.empty(8, dtype=np.uint8)
    batch.simple_rule_actions_into(actions)
    enabled = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    records = np.empty((8, bm.STEP_RECORD_WIDTH), dtype=np.int64)
    before = np.empty((8, bm.ORACLE_TILE_COUNT_PLANES, 27), dtype=np.uint8)
    after = np.empty_like(before)
    batch.oracle_tile_counts_into(before)

    batch.step_masked_into(enabled, actions, records)

    batch.oracle_tile_counts_into(after)
    np.testing.assert_array_equal(after[enabled == 0], before[enabled == 0])
    assert np.all(records[enabled == 0] == -1)
    assert np.all(records[enabled == 1, 9] >= 0)

    with pytest.raises(ValueError, match="shape"):
        batch.step_masked_into(enabled[:7], actions, records)
    invalid = enabled.copy()
    invalid[0] = 2
    with pytest.raises(ValueError, match="action"):
        batch.step_masked_into(invalid, actions, records)


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


def test_batch_event_history_and_combined_step_history() -> None:
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
    history_seat_masks = np.full(size, 0x0F, dtype=np.uint8)
    batch.step_and_observe_history_into(
        actions,
        history_seat_masks,
        records,
        masks,
        tile_obs,
        melds,
        river,
        meta,
        events,
        event_lengths,
    )
    assert np.all(event_lengths >= 2)
    assert np.all(events[:, 0, 0] == bm.EventKind.GAME_START)
    assert np.all(events[np.arange(size), event_lengths - 1, 0] == bm.EventKind.ACTION)


def test_batch_masked_event_history_uses_current_viewer_bits() -> None:
    size = 128
    capacity = 8
    batch = bm.Batch(size, seed=83)
    _, _, tile_obs, melds, river, meta = buffers(size)
    batch.observe_into(tile_obs, melds, river, meta)
    viewers = meta[:, 1].astype(np.uint8)
    history_seat_masks = np.left_shift(np.uint8(1), viewers)
    history_seat_masks[1::2] = np.left_shift(
        np.uint8(1), (viewers[1::2] + 1) % 4
    )
    sentinel = np.int32(-777)
    events = np.full(
        (size, capacity, bm.EVENT_RECORD_WIDTH), sentinel, dtype=np.int32
    )
    lengths = np.full(size, np.iinfo(np.uint16).max, dtype=np.uint16)

    batch.events_masked_into(history_seat_masks, events, lengths)

    assert np.all(lengths[::2] == 1)
    assert np.all(events[::2, 0, 0] == bm.EventKind.GAME_START)
    assert np.all(lengths[1::2] == 0)
    assert np.all(events[1::2] == sentinel)

    with pytest.raises(ValueError, match="shape"):
        batch.events_masked_into(history_seat_masks[:-1], events, lengths)
    with pytest.raises(TypeError):
        batch.events_masked_into(history_seat_masks.astype(np.int8), events, lengths)


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
