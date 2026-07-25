# Transformer training components

The first model is deliberately narrow: a 48-token bidirectional encoder for
the current actor-visible state, a 192-event causal history encoder with
standard RoPE on Q/K, and fully connected actor/distributional-critic heads. It
does not depend on a ResNet or behavior cloning.

```python
import torch
from training import BloodFlowTransformer, unpack_action_masks

model = BloodFlowTransformer().cuda().to(torch.bfloat16)
legal = torch.from_numpy(unpack_action_masks(mask_words)).cuda()
output = model(tile_obs, melds, meta, events, event_lengths, legal)
actions = torch.distributions.Categorical(logits=output.logits).sample()
```

`HistoryEncoder.forward_cached` and `BloodFlowTransformer.forward_cached`
append viewer-scoped events to per-layer K/V tensors. The rollout collector
uses grouped `(environment, absolute-seat)` caches for both learner and frozen
opponent inference, and falls back to a full rebuild after reset or a sliding
window change. Exact cache-shape groups smaller than 32 rows are merged into a
single padded full forward instead of launching many small GPU kernels. A cache
is valid only for one game, one fixed viewer, and one fixed model version; all
rollout K/V tensors are released before PPO starts.

History positions use standard interleaved RoPE with `theta=10_000`. Full
forwards rotate positions `0..L-1`; cached appends rotate the absolute suffix
starting at the cached length. The cached/full equivalence test covers this
position offset path.

Run the unit and engine integration tests with:

```bash
python -m pytest training/tests
```

Run an end-to-end CPU smoke test before committing GPU time:

```bash
python -m training.train \
  --smoke \
  --device cpu \
  --output-dir /tmp/bloodflow-smoke
```

Start the default 24-hour run on one GPU with:

```bash
python -m training.train \
  --device cuda \
  --hours 24 \
  --total-transitions 200000000 \
  --output-dir runs/transformer
```

`metrics.jsonl` records the learning rate, curriculum stage, the four policy
family assignment counts across the three non-learner seats, active snapshot,
PPO statistics, rollout/PPO/evaluation wall times, cache hit/full/group counts,
and fixed-rule evaluation.
`latest.pt` contains the learner, optimizer, RNG state, and retained frozen
opponent snapshots, and can be resumed with `--resume runs/transformer/latest.pt`.

The three non-learner seats are sampled independently.  Opponent stages are
gated by repeated rule-anchor evaluations rather than transition percentages:
the learner stays in `bootstrap` below 70% first-place rate, enters `mixed`
between 70% and 80%, and enters the rule-anchored `league` stage at 80%.  The
league stage still retains rule opponents; it does not silently become pure
self-play.  See `TRAINING.md` for exact policy probabilities and snapshot
timing.

On the local RTX 5080, the smoke benchmark is:

```bash
python -m training.benchmarks.transformer --device cuda
python -m training.benchmarks.train_step --device cuda
```

The engine's `Batch.events_into` output should be allocated with capacity
`192` (or sliced to the newest 192 records) before passing it to the model.
