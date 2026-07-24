#!/usr/bin/env python3
import argparse
import time

import numpy as np

import bloodflow_mahjong as bm

WARMUP_ITERATIONS = 64
GAME_SEED = 1
RESET_SEED = np.uint64(0x3C6E_F372_FE94_F82B)
SEED_STEP = np.uint64(0x9E37_79B9_7F4A_7C15)


def choose_first_legal(masks: np.ndarray, actions: np.ndarray) -> None:
    low = masks[:, 0]
    low_nonzero = low != 0
    active_low = low[low_nonzero]
    low_bits = active_low ^ (active_low & (active_low - np.uint64(1)))
    actions[low_nonzero] = np.log2(low_bits).astype(np.uint8)

    high = masks[~low_nonzero, 1]
    if high.size:
        if np.any(high == 0):
            raise RuntimeError("active environment has no legal action")
        high_bits = high ^ (high & (high - np.uint64(1)))
        actions[~low_nonzero] = 64 + np.log2(high_bits).astype(np.uint8)


def run_iterations(
    batch: bm.Batch,
    iterations: int,
    actions: np.ndarray,
    records: np.ndarray,
    masks: np.ndarray,
    tile_obs: np.ndarray,
    melds: np.ndarray,
    river: np.ndarray,
    meta: np.ndarray,
    terminal: np.ndarray,
    all_indices: np.ndarray,
    reset_indices: np.ndarray,
    reset_ordinals: np.ndarray,
    reset_seeds: np.ndarray,
    next_game: int,
) -> tuple[int, int]:
    completed_games = 0
    for _ in range(iterations):
        choose_first_legal(masks, actions)
        batch.step_and_observe_into(
            actions, records, masks, tile_obs, melds, river, meta
        )

        np.equal(records[:, 11], 1, out=terminal)
        reset_count = int(np.count_nonzero(terminal))
        if reset_count == 0:
            continue

        np.compress(terminal, all_indices, out=reset_indices[:reset_count])
        np.add(reset_ordinals[:reset_count], next_game, out=reset_seeds[:reset_count])
        np.multiply(reset_seeds[:reset_count], SEED_STEP, out=reset_seeds[:reset_count])
        np.add(reset_seeds[:reset_count], RESET_SEED, out=reset_seeds[:reset_count])
        batch.reset_many(reset_indices[:reset_count], reset_seeds[:reset_count])
        batch.observe_into(tile_obs, melds, river, meta)
        batch.legal_action_masks_into(masks)
        next_game += reset_count
        completed_games += reset_count

    return next_game, completed_games


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=4096)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.iterations <= 0:
        parser.error("--batch-size and --iterations must be positive")

    batch = bm.Batch(args.batch_size, seed=1)
    masks = np.empty((args.batch_size, bm.LEGAL_ACTION_MASK_WORDS), dtype=np.uint64)
    actions = np.empty(args.batch_size, dtype=np.uint8)
    records = np.empty((args.batch_size, bm.STEP_RECORD_WIDTH), dtype=np.int64)
    tile_obs = np.empty(
        (args.batch_size, bm.TILE_OBSERVATION_PLANES, bm.TILE_KIND_COUNT),
        dtype=np.uint8,
    )
    melds = np.empty(
        (args.batch_size, bm.PLAYER_COUNT, bm.MELD_SLOTS, bm.MELD_FIELDS),
        dtype=np.uint8,
    )
    river = np.empty(
        (args.batch_size, bm.RIVER_TILE_CAPACITY, bm.RIVER_FIELDS), dtype=np.uint8
    )
    meta = np.empty((args.batch_size, bm.META_OBSERVATION_WIDTH), dtype=np.int32)
    terminal = np.empty(args.batch_size, dtype=np.bool_)
    all_indices = np.arange(args.batch_size, dtype=np.uint32)
    reset_indices = np.empty(args.batch_size, dtype=np.uint32)
    reset_ordinals = np.arange(args.batch_size, dtype=np.uint64)
    reset_seeds = np.empty(args.batch_size, dtype=np.uint64)

    batch.observe_into(tile_obs, melds, river, meta)
    batch.legal_action_masks_into(masks)
    next_game, _ = run_iterations(
        batch,
        WARMUP_ITERATIONS,
        actions,
        records,
        masks,
        tile_obs,
        melds,
        river,
        meta,
        terminal,
        all_indices,
        reset_indices,
        reset_ordinals,
        reset_seeds,
        args.batch_size,
    )

    batch.reset_all(GAME_SEED)
    batch.observe_into(tile_obs, melds, river, meta)
    batch.legal_action_masks_into(masks)

    started = time.perf_counter()
    _, completed_games = run_iterations(
        batch,
        args.iterations,
        actions,
        records,
        masks,
        tile_obs,
        melds,
        river,
        meta,
        terminal,
        all_indices,
        reset_indices,
        reset_ordinals,
        reset_seeds,
        args.batch_size,
    )

    elapsed = time.perf_counter() - started
    steps = args.batch_size * args.iterations
    print(f"batch size:    {args.batch_size:,}")
    print(f"actions:       {steps:,}")
    print(f"games:         {completed_games:,}")
    print(f"elapsed:       {elapsed:.3f} s")
    print(f"actions/s:     {steps / elapsed:,.0f}")


if __name__ == "__main__":
    main()
