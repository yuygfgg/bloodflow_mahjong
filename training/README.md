# Transformer training components

The first model is deliberately narrow: a 48-token bidirectional encoder for
the current actor-visible state, a 192-event causal history encoder, and fully
connected actor/distributional-critic heads. It does not depend on a ResNet or
behavior cloning.

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
window change. A cache is valid only for one game, one fixed viewer, and one
fixed model version.

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
  --total-transitions 100000000 \
  --output-dir runs/transformer
```

`metrics.jsonl` records the learning rate, curriculum stage, the four policy
family assignment counts across the three non-learner seats, active snapshot,
PPO statistics, and fixed-rule evaluation.
`latest.pt` contains the learner, optimizer, RNG state, and retained frozen
opponent snapshots, and can be resumed with `--resume runs/transformer/latest.pt`.

The three non-learner seats are sampled independently.  The default transition
schedule is `bootstrap` before 10%, `mixed` from 10% through 35%, `league` from
35% through 75%, and pure Transformer `self-play` after 75%; see `TRAINING.md`
for exact policy probabilities and snapshot timing.

On the local RTX 5080, the smoke benchmark is:

```bash
python -m training.benchmarks.transformer --device cuda
python -m training.benchmarks.train_step --device cuda
```

The engine's `Batch.events_into` output should be allocated with capacity
`192` (or sliced to the newest 192 records) before passing it to the model.
