# Training

## Verify the pipeline

Build the current Python engine binding before running the training tests:

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
python -m pytest training/tests
```

## Rules compatibility

Engine rules version 10 uses training input schema
`viewer-kong-forbidden-v3`. Version 7, 8, and 9 checkpoints and supervised
Actors are not compatible with this rule set. In schema v3, tile plane 0 is the
viewer's concealed tile count. Planes 2 through 5 expose each seat's locked
winning tiles; the viewer's own plane also includes its stable winning base.
Plane 10 exposes the viewer's persistent Kong-forbidden tile kinds. The engine
keeps the stable winning base, historical winning references, and active tiles
separate. Historical references cannot be consumed by later hand actions.
Version 10 exposes the complete Hu/Pong/exposed-Kong/Pass choice set whenever a
Hu candidate exists; Hu has global priority and one discard can produce
multiple Hu results. Start a new supervised run and a new PPO run instead of
resuming or forking an older checkpoint.

Exported ONNX files must declare metadata
`engine_rules_version=10`. The Rust loader rejects a missing, malformed, or
different version before it accepts the model. After training, export the new
policy and replace `model/latest.onnx` before using `rule-nn` for play or
evaluation.

Run both training stages with small CPU settings:

```bash
python -m training.supervised \
  --smoke \
  --device cpu \
  --output-dir /tmp/bloodflow-sl-smoke

python -m training.train \
  --smoke \
  --device cpu \
  --init-actor /tmp/bloodflow-sl-smoke/actor.pt \
  --output-dir /tmp/bloodflow-ppo-smoke
```

`--smoke` is the only built-in finite run. A normal PPO run has no time or
update limit.

## Supervised warm start

The supervised stage labels each current-player state with deterministic
`RuleEvConfig.standard()` actions. Rule-EV controls all four seats. Uniform
random legal behavior is used for 15% of actions to expand the visited state
distribution, but Rule-EV still supplies every label. Forced decisions with
one legal action are not stored.

```bash
python -m training.supervised \
  --device cuda \
  --seed 7 \
  --labels 10000000 \
  --output-dir runs/rule-ev-sl-seed7
```

`actor.pt` contains the shared observation encoder and policy head. It also
records the model configuration, engine rules version, and training input
schema. PPO initializes the value and auxiliary heads and starts with a new
optimizer.

## PPO objective

PPO uses this reward by default:

```text
score_delta / 10,000 + terminal_rank_utility
```

Rank utilities for first through fourth place are `+1`, `+1/3`, `-1/3`, and
`-1`. Configure the two terms with `--score-reward-weight` and
`--rank-reward-weight`.

`--kl-control monitor` measures the full legal-action distribution after each
epoch and does not change optimization. `off` disables this measurement.
`rollback` is an explicit experiment that restores the model and optimizer
when an epoch exceeds `--target-kl`.

Learning controls do not use wall-clock time. Rule-EV evaluation plateaus
control the learning rate. Observed policy entropy controls the entropy
coefficient. The self-play level controls the auxiliary loss scale. Elapsed
time remains available only as throughput telemetry.

## Run lifetime and persistence

Training runs until the user presses `Ctrl+C`. The first `Ctrl+C` sets a stop
request. The process completes the active rollout and PPO update, atomically
saves `latest.pt`, appends an `interrupted` record to `metrics.jsonl`, and
exits without a final evaluation. The checkpoint never contains a partial PPO
update.

The following intervals count completed PPO updates:

- `--eval-every N` runs the internal Rule-EV gate every `N` updates. The
  default is 10.
- `--checkpoint-every N` replaces `latest.pt` every `N` updates. The default
  is 10.
- `--snapshot-every N` writes `snapshot_u<update>.pt` every `N` updates. This
  archival snapshot interval is disabled by default.

These intervals trigger work only. They do not change learning rates,
entropy, curriculum decisions, or the stopping condition.

## Rule-EV gate

The internal gate always evaluates the learner against three standard Rule-EV
players. No analysis opponent can replace this gate.

Each independent seed panel contains four games. The learner occupies each
seat once. Statistics first average the four seats in a panel, then calculate
standard errors across panels. Periodic gate evaluations use non-overlapping
seed panels. The controller stores the next seed and the complete rolling
window in `latest.pt`.

With the default 95% confidence multiplier, one evaluation passes only when
both conditions are true:

```text
score lower confidence bound > +75
rank  upper confidence bound < 2.45
```

The default rolling window contains three independent evaluations. Promotion
also requires all of these conditions:

- The latest evaluation passes its individual confidence bounds.
- At least two of the three evaluations pass individually.
- The pooled panel statistics from all three evaluations pass the same
  confidence bounds.

The controller therefore does not promote from one high-variance result. A
promotion clears the evidence window before evidence for the next level is
collected.

Demotion uses separate hysteresis thresholds. The latest and pooled results
must show `score upper confidence bound < -75` or
`rank lower confidence bound > 2.55`, and at least two window entries must
fail individually.

## Opponent curriculum

Self-play is enabled by default, but a fresh run starts with no frozen-policy
opponents. Each non-learner seat is sampled independently with these default
probabilities:

| Level | Rule-Fast | Rule-EV | Frozen policy |
| ---: | ---: | ---: | ---: |
| 0 | 33.33% | 66.67% | 0% |
| 1 | 28.33% | 56.67% | 15% |
| 2 | 23.33% | 46.67% | 30% |
| 3 | 18.33% | 36.67% | 45% |

The frozen fraction increases by 15 percentage points after each stable gate
promotion and is capped at 45%. The remaining rule-policy fraction always
keeps the Rule-Fast to Rule-EV ratio at 1:2. Use `--no-self-play` to keep the
run at level 0.

A promotion creates a frozen opponent snapshot from the current learner. At
level 3, a new stable champion refreshes the opponent snapshot without
increasing the frozen fraction. Demotion reduces exposure but retains league
history. Opponent snapshots are metric-driven; `--snapshot-every` does not
control them.

## Human analysis

`--analysis-opponent` selects an additional deterministic report against
`rule-fast`, `rule-ev`, or `rule-nn`. Rule-Fast and Rule-NN results are
human-facing diagnostics only. They never control the
gate, learning rate, entropy coefficient, self-play level, or snapshots.

Rule-EV is the default analysis opponent and reuses the gate result when both
evaluations use the same game count. A fresh run therefore does not require an
ONNX model. Rule-NN requires `--analysis-nn-model`. The process loads the ONNX
model once and records its path and SHA-256 digest with analysis metrics.

Use `--analysis-games` to reduce the cost of a human-facing evaluation. Use
`--analysis-every` to run it after every `N` gate evaluations. Both gate and
analysis game counts must be positive multiples of four. Rule-NN can be
substantially slower than Rule-EV.

## Start a fresh PPO run

Start from a supervised Actor. The command runs until `Ctrl+C`:

```bash
python -m training.train \
  --device cuda \
  --seed 7 \
  --init-actor runs/rule-ev-sl-seed7/actor.pt \
  --output-dir runs/ppo-metric-seed7 \
  --eval-every 10 \
  --checkpoint-every 10 \
  --snapshot-every 100
```

Omit `--init-actor` to start from a random model. The default opponent mix is
still Rule-Fast and Rule-EV, so cold-start training does not depend on
Rule-NN.

## Resume a run

Resume the complete model, optimizer, RNG, rollout collector, opponent pool,
training controller, gate evidence, and next gate seed:

```bash
python -m training.train \
  --device cuda \
  --resume runs/ppo-metric-seed7/latest.pt
```

The output directory is inferred from the checkpoint. Static PPO settings
must match the checkpoint. Reporting options, including the analysis
opponent, can change between sessions.

Use `--fork CHECKPOINT --output-dir NEW_DIRECTORY` to copy complete PPO state
into a new run. A fork can change the opponent curriculum and learning-rate
schedule. An explicit `--learning-rate` starts a new plateau schedule: it sets
the current optimizer rate and clears the old best-metric and plateau evidence.
Changing the minimum rate, decay, or patience in a fork also requires an
explicit `--learning-rate`. The fork preserves model, optimizer moments, RNG,
entropy control, opponent state, and self-play gate evidence.

Use `--stop-after-updates N` for a finite experiment. The count starts at zero
for the current process. The run saves its final checkpoint and evaluation
after it completes exactly `N` updates.

## Add Rule-NN analysis

This example runs the independent Rule-EV gate every 10 updates and reports
256 games against the fixed ONNX policy every second gate evaluation:

```bash
python -m training.train \
  --device cuda \
  --resume runs/ppo-metric-seed7/latest.pt \
  --eval-every 10 \
  --analysis-opponent rule-nn \
  --analysis-nn-model model/latest.onnx \
  --analysis-games 256 \
  --analysis-every 2
```

Replace `rule-nn` with `rule-fast` to select another human-only anchor. Do not
provide `--analysis-nn-model` unless the selected
opponent is `rule-nn`.

## Export ONNX

Export a complete PPO checkpoint to the fixed `rule-nn` contract:

```bash
python -m training.export_onnx \
  runs/ppo-metric-seed7/latest.pt \
  model/latest.onnx
```

The graph accepts `tile_obs`, `melds`, `meta`, 192 observer-view event records,
and the event length. It emits raw Actor logits with shape `[1, 115]`. The
engine applies the current legal-action mask after inference.

By default, the exporter validates the graph and compares one deterministic
input with PyTorch. It reports maximum and mean absolute error and requires an
identical argmax. Use `--no-check` or `--no-parity` only for isolated exporter
diagnosis.

Run one game through the Rust loader:

```bash
cargo run --release --manifest-path engine/Cargo.toml \
  -p bloodflow-mahjong \
  --features rule-nn \
  --example rule_nn_smoke -- \
  model/latest.onnx 7
```

See [`../engine/tools/rule-tournament/README.md`](../engine/tools/rule-tournament/README.md)
for balanced rule-policy tournaments.

## Cross-play checkpoints

Use the deterministic arena to compare PPO checkpoints. Each matrix cell runs
the row policy against three copies of the column policy with balanced focal
seats:

```bash
python -m training.arena \
  --device cuda \
  --games 1024 \
  --output runs/ppo-metric-seed7/crossplay.json \
  u100=runs/ppo-metric-seed7/snapshot_u100.pt \
  u200=runs/ppo-metric-seed7/snapshot_u200.pt
```

The arena averages tied placements and estimates standard errors from the
four-seat seed panels. Compare cross-play with the independent Rule-EV gate;
do not infer general strength from one matchup alone.

## Outputs

- `config.json` records the static PPO and model configuration, input schema,
  engine rules version, and command arguments.
- `metrics.jsonl` records complete structured updates, gate evidence,
  curriculum decisions, and optional analysis results.
- `latest.pt` stores the complete resumable state and is replaced atomically.
- `snapshot_u<update>.pt` is an optional archival checkpoint created by
  `--snapshot-every`.

The terminal prints one compact line after each supervised batch or PPO
update. `ppo_elapsed_seconds` is reporting data only and never controls the
training algorithm.
