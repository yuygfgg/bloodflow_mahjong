# Neural training baseline

This directory restores the compact Transformer and online PPO pipeline from
commit `5735b01`. It does not restore the later IQL/AWR, oracle, Monte Carlo,
search-distillation, or policy-validation stack.

The Actor reads only the current player's observation. The static encoder uses
48 bidirectional tokens. The causal history encoder reads at most 192
viewer-scoped events. The policy has 115 masked actions. PPO also trains a
distributional value head and short-lived shanten and improving-tile auxiliary
heads.

## Verification

Build the current Python binding before running the Python tests:

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
python -m pytest training/tests
```

Run both stages with small CPU settings:

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

## Rule-EV supervised start

The supervised stage labels every current-actor state with deterministic
`RuleEvConfig.standard()`. Rule-EV controls all four seats. A 15% uniformly
random legal behavior action broadens the visited state distribution, but the
label remains the Rule-EV action. Forced decisions with one legal action are
not stored as labels.

Use one million labels for the first causal pilot:

```bash
python -m training.supervised \
  --device cuda \
  --labels 1000000 \
  --output-dir runs/rule-ev-sl-pilot
```

The full historical SL budget was ten million labels. The restored defaults
retain its measured batch settings: 4,096 environments, 65,536 labels per
batch, one epoch, a 4,096 minibatch, a 512 microbatch, and a `3e-4` learning
rate. The previous checkpoint predates the current public-observation schema
and Actor checkpoint format, so it is not a valid initialization source even
when individual tensor shapes happen to match.

The output `actor.pt` contains only the shared observation encoders and policy
head. It records the model config, engine rules version, and training input
schema. Loading rejects any mismatch. The value and auxiliary heads start
fresh for PPO, and PPO does not inherit SL optimizer moments.

## PPO

PPO uses a hybrid reward by default:

```text
score_delta / 10,000 + terminal_rank_utility
```

Terminal rank utility is `+1`, `+1/3`, `-1/3`, or `-1` for ranks one through
four. Tied players use the same strict-score comparison as evaluation. Score
and rank weights are recorded in the checkpoint and can be set with
`--score-reward-weight` and `--rank-reward-weight`.

Each PPO epoch randomly assigns all rollout states to minibatches. History
length sorting occurs only inside each minibatch to reduce padding. It cannot
move early-game states into the first minibatches globally.

The default `--kl-control monitor` measures the full legal-action distribution
for a fixed random monitor sample after each epoch and does not change
optimization. `--kl-control off` removes that measurement.
`--kl-control rollback` is an optional separate experiment: it restores the
model and optimizer when an epoch exceeds `--target-kl`.

Start a two-hour pilot from the supervised Actor:

```bash
python -m training.train \
  --device cuda \
  --hours 2 \
  --init-actor runs/rule-ev-sl-pilot/actor.pt \
  --output-dir runs/ppo-rule-ev-pilot
```

PPO has one stopping budget: accumulated run wall time from `--hours`.
Transitions and updates are throughput metrics only. Learning-rate, entropy,
and auxiliary schedules always use a fixed 24-hour horizon, so stopping a pilot
at two hours and resuming it does not restart or rescale annealing. A resumed
run restores the accumulated time and collector seed, then continues toward
the new hour target:

```bash
python -m training.train \
  --device cuda \
  --hours 24 \
  --resume runs/ppo-rule-ev-pilot/latest.pt
```

Each resume appends an explicit `resume` record to `metrics.jsonl`. The record
identifies the checkpoint state and the new run target. Metrics after the prior
checkpoint can remain in the file after an interrupted process; treat the
resume record as the start of a new metric generation.

The first baseline keeps self-play disabled. Each non-learner seat independently
uses Rule-Fast with probability `1/3` or Rule-EV standard with probability
`2/3`.

Use `--fork` to start self-play from a complete PPO checkpoint without changing
the source run. A fork restores the model, value head, optimizer, RNG state,
collector state, and accumulated 24-hour schedule. It permits changes only to
the opponent curriculum. The following command continues the 16-hour policy to
the cumulative 24-hour target:

```bash
python -m training.train \
  --device cuda \
  --hours 24 \
  --fork runs/ppo-rule-ev-2h-seed7/snapshot_16h.pt \
  --output-dir runs/ppo-self-play-25-from16h-seed7 \
  --self-play \
  --self-play-fraction 0.25 \
  --historical-snapshot-probability 0.5 \
  --opponent-refresh-updates 200 \
  --checkpoint-every 50
```

After the Rule-EV gate passes, each non-learner seat uses a frozen policy with
probability `25%`, Rule-EV with probability `50%`, or Rule-Fast with probability
`25%`. The pool creates a snapshot every 200 PPO updates and retains four,
including the fork anchor.
Each rollout uses the newest snapshot or a retained historical snapshot with
equal probability. Periodic evaluation continues to use three Rule-EV players
as a fixed external anchor.

Start the corrected hybrid pilot from the completed Rule-EV supervised Actor.
Do not resume the failed score-only PPO run:

```bash
python -m training.train \
  --device cuda \
  --seed 114514 \
  --hours 0.25 \
  --score-reward-weight 1 \
  --rank-reward-weight 1 \
  --kl-control monitor \
  --init-actor runs/rule-ev-sl-10m-seed114514/actor.pt \
  --output-dir runs/ppo-hybrid-rule-ev-pilot-seed114514
```

After the pilot confirms that the SL baseline is not immediately lost, run the
comparable ten-hour job from the same SL Actor:

```bash
python -m training.train \
  --device cuda \
  --seed 114514 \
  --hours 10 \
  --score-reward-weight 1 \
  --rank-reward-weight 1 \
  --kl-control monitor \
  --init-actor runs/rule-ev-sl-10m-seed114514/actor.pt \
  --output-dir runs/ppo-hybrid-rule-ev-10h-seed114514
```

Run the full baseline from a new output directory:

```bash
python -m training.supervised \
  --device cuda \
  --seed 7 \
  --labels 10000000 \
  --output-dir runs/rule-ev-sl-10m-seed7

python -m training.train \
  --device cuda \
  --seed 7 \
  --hours 24 \
  --init-actor runs/rule-ev-sl-10m-seed7/actor.pt \
  --output-dir runs/ppo-rule-ev-24h-seed7
```

Every evaluation uses a fixed panel of exact `(seed, focal seat)` pairs. Each
seed appears once in all four seats, and the other three players use Rule-EV
standard. Finished rows are not refilled, so short games cannot be counted more
often. Periodic evaluations reuse one fixed panel for trend measurement; the
final evaluation uses a fresh panel.

To test whether PPO snapshots improve against each other, run the deterministic
cross-play arena. Each matrix cell evaluates the row snapshot against three
copies of the column snapshot, with every focal seat represented equally:

```bash
python -m training.arena \
  --device cuda \
  --games 1024 \
  --output runs/ppo-rule-ev-2h-seed7/crossplay.json \
  2h=runs/ppo-rule-ev-2h-seed7/snapshot_2h.pt \
  6h=runs/ppo-rule-ev-2h-seed7/snapshot_6h.pt \
  12h=runs/ppo-rule-ev-2h-seed7/snapshot_12h.pt \
  16h=runs/ppo-rule-ev-2h-seed7/snapshot_16h.pt
```

The arena splits tied placements across their occupied ranks. It computes
standard errors from independent seeds after averaging each four-seat panel.
The diagonal is rank 2.5 and score zero. A later snapshot should have lower
rank and positive score against an earlier snapshot if it learned a generally
stronger policy. Compare the matrix with the separate Rule-EV evaluations
before drawing conclusions from one matchup.

`latest.pt` stores the complete PPO model and optimizer, RNG states, opponent
state, collector seed, accumulated PPO time, engine rules version, input
schema, and exact PPO config. Snapshots are also written at 2, 6, 12, and 24
accumulated PPO hours.

The terminal prints one compact progress summary per SL batch or PPO update.
`metrics.jsonl` retains every complete structured metric record.
