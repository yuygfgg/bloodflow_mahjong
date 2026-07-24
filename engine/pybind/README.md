# Python bindings

This crate exposes the Rust rules engine as the `bloodflow_mahjong` Python
extension. Build and install it into the active virtual environment with:

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
```

The fixed policy action space has 115 entries:

| IDs | Action |
| --- | --- |
| `0..26` | select one exchange tile |
| `27..29` | choose missing suit |
| `30..56` | discard tile |
| `57`, `58`, `59` | hu, pong, exposed kong |
| `60..86` | concealed kong by tile |
| `87..113` | added kong by tile |
| `114` | pass in a response window |

The module exports named `ACTION_*` constants for every singleton action and
range offset. Batch methods operate directly on caller-owned, C-contiguous
NumPy buffers:

```python
import numpy as np
import bloodflow_mahjong as bm

batch = bm.Batch(1024, seed=7)
masks = np.empty((len(batch), bm.LEGAL_ACTION_MASK_WORDS), dtype=np.uint64)
actions = np.empty(len(batch), dtype=np.uint8)
records = np.empty((len(batch), bm.STEP_RECORD_WIDTH), dtype=np.int64)
tile_obs = np.empty((len(batch), bm.TILE_OBSERVATION_PLANES, 27), dtype=np.uint8)
melds = np.empty((len(batch), 4, bm.MELD_SLOTS, bm.MELD_FIELDS), dtype=np.uint8)
river = np.empty((len(batch), bm.RIVER_TILE_CAPACITY, bm.RIVER_FIELDS), dtype=np.uint8)
meta = np.empty((len(batch), bm.META_OBSERVATION_WIDTH), dtype=np.int32)

batch.legal_action_masks_into(masks)
# Fill actions from the two packed mask words, then step all environments.
batch.step_and_observe_into(actions, records, masks, tile_obs, melds, river, meta)
```

`records` has 12 columns:

| Column | Value |
| --- | --- |
| 0 | draw player, or `-1` |
| 1 | drawn tile, or `-1` |
| 2 | replacement draw flag |
| 3 | discard player, or `-1` |
| 4 | discarded tile, or `-1` |
| 5..8 | score delta for seats 0..3 |
| 9 | next actor, or `-1` |
| 10 | next phase, or `-1` |
| 11 | terminal flag |

## Event stream

The extension exposes a fixed-width event stream for recurrent policies,
replay, and diagnostics. One record is eight `int32` fields:

```text
[kind, actor_relative, target_relative, tile, flags, value, aux, reserved]
```

`actor_relative` and `target_relative` use the same seat rotation as the
observation. `-1` means that a field is not applicable. Event kinds are
available as the `EventKind(IntEnum)` class and bit flags as
`EventFlag(IntFlag)`. The values of both enums are the stable integer codes
stored in the arrays.

| Kind | Fields |
| --- | --- |
| `ACTION` | `value=action_id`, `aux=phase`; visible only to the acting player |
| `GAME_START` | `actor=dealer`, `flags=exchange_direction` |
| `TURN_START` | initial dealer turn marker; `aux=1` |
| `DRAW` | `actor=drawer`, `tile` is hidden from other players, `flags` contains replacement/last-wall, `value=wall_remaining` |
| `DISCARD` | `actor`, `tile`, `flags` contains after-kong/opening-discard |
| `EXCHANGE_COMPLETE` | `flags=exchange_direction` |
| `MISSING_REVEALED` | `actor=player`, `value=missing_suit` |
| `MELD` | `actor`, `target=source` when applicable, `tile`, `flags=MeldKind` |
| `HU` | `actor=winner`, `target=source` for discard/rob-kong, `tile`, `flags`, `value=multiplier`, `aux=PatternSet.bits()` |
| `PAYMENT` | `actor=payer`, `target=payee`, `value=actual_amount` |
| `GAME_END` | `flags` marks an empty wall when applicable |

`Game.events_into(viewer, output)` copies the newest retained history into a
caller-owned `int32[capacity, EVENT_RECORD_WIDTH]` buffer and returns its
length. `Game.step_events_into` copies only events emitted by the most recent
step. For a `Batch`, `events_into` and `step_events_into` write
`int32[batch, capacity, EVENT_RECORD_WIDTH]` plus `uint16[batch]` lengths.
The batch viewer is the current decision actor, or the dealer after terminal.

The Rust side retains a 512-record ring per environment; `Game.event_dropped`
and `Batch.event_dropped_into` report overwritten records. Use the step-delta API in the training loop so a
large history is not copied on every action. `Batch.step_and_observe_events_into`
combines step, transition, observation, legal mask, and step-event delta into
one GIL-free call. Buffers are caller-owned, C-contiguous, aligned, and never
copied through Python objects.

Array dtypes and shapes are part of the API. Batch calls reject non-contiguous
views rather than copying them. The GIL is released while reset, mask, and step
work runs in Rust; the Rust batch implementation uses Rayon for sufficiently
large batches.

Observations are rotated to the current actor: relative seat zero is the policy
that acts next, followed by seats 1, 2, and 3 in turn order. A terminal game has
no actor and uses the dealer as relative seat zero. `step_and_observe_into`
writes the state and legal mask after applying the submitted actions.

`tile_obs` has shape `[batch, 10, 27]` and contains tile counts:

| Plane | Value |
| --- | --- |
| 0 | relative seat 0 concealed hand |
| 1 | relative seat 0's tiles selected during exchange |
| 2..5 | locked tiles for relative seats 0..3 |
| 6..9 | discarded-tile histograms for relative seats 0..3 |

`melds` has shape `[batch, 4, 4, 3]`. Its axes are relative player, meld slot,
and `[tile, kind, relative source]`; kind is `0` pong, `1` exposed kong, `2`
added kong, or `3` concealed kong. `river` has shape `[batch, 108, 2]` and each
chronological entry is `[tile, relative owner]`. Empty meld and river slots use
`255` in every field.

`meta` has shape `[batch, META_OBSERVATION_WIDTH]`:

| Index | Value |
| --- | --- |
| 0 | phase (`PHASE_*`) |
| 1 | absolute actor, or `-1` |
| 2 | relative dealer |
| 3 | exchange direction (`1` left, `2` across, `3` right) |
| 4 | wall tiles remaining |
| 5 | current actor's drawn tile, or `-1` |
| 6 | replacement-draw flag |
| 7 | pending discard/kong source as a relative seat, or `-1` |
| 8 | pending response tile, or `-1` |
| 9 | chronological river length |
| 10 | current actor's selected exchange-tile count |
| 11 | current actor's exchange suit, or `-1` |
| 12..15 | scores for relative seats 0..3 |
| 16..19 | missing suits for relative seats 0..3, or `-1` |
| 20..23 | has-won flags for relative seats 0..3 |
| 24..27 | concealed tile counts for relative seats 0..3 |
| 28 | terminal flag |
| 29 | response flags: bit 0 rob-kong, bit 1 after-kong discard, bit 2 opening discard |
| 30..33 | maximum completed-win multiplier for relative seats 0..3 |

Only the current actor's concealed hand and pending exchange choices are
included. Other concealed hands and their pending choices remain hidden;
locked tiles, melds, and the river are public. Use the exported width constants
when allocating buffers so schema extensions are caught by shape validation.

The transition record is authoritative engine data. In particular, its drawn
tile is not filtered for a previous viewer. Treat the actor-relative observation
as policy input and the record as environment/control data.
