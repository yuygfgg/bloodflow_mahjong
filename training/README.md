# 训练

当前推荐主线是 CUDA champion/candidate policy generation。它从 SL 训练得到的 U61 Actor 开始。每个 attempt 冻结 champion，采集 4 批九类各 128 个状态，使用独立的 32-world selection、64-world validation 和 32-world audit。validation 使用 paired mean-rank advantage、lower confidence bound 和 BH-FDR=0.05。通过验证的行形成 mirror CE target；未通过行的梯度权重为零。audit 只续局 candidate 与 champion greedy action 不同的状态，但未翻转状态仍以零差值进入完整统计。

一个 generation 内执行 4 个 Nesterov inner steps。速度只在该 generation 内保留。每一步都投影到相对 champion 的累计 `KL <= 1e-4`。最终 candidate 在 fixed-rule 和 historical-opponent 阵容各跑 65536 局配对 arena。只有 pooled dRank 显著改善且所有名次、分差和 audit 安全护栏通过时，candidate 才会晋升。拒绝不会修改 champion、自博弈课程或 opponent snapshots，但会推进历史对手轮换游标。

旧的 expected-Q、split CE、AdamW、SGD 和跨 update Nesterov 路径仍保留用于复现实验。U61 主线使用 `live_wall`；当前 information-set 重采样不是历史条件后验，因此主线配置会拒绝该组合。

详细算法、统计边界和参数见 [TRAINING.md](../TRAINING.md)。

## 准备

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
```

训练只支持 CUDA。梯度更新和策略推理都使用 eager mode 和 BF16 autocast；推理保留 pinned-memory 双缓冲和按 history width 分桶，不回退 CPU。生产路径不使用 TorchScript trace 或 `torch.compile`：trace 在真实 source 状态上产生过 greedy action 翻转，Inductor 的 BF16 autocast 融合也会产生错误 logits。Inductor 最小复现见 [`benchmarks/TORCH_COMPILE_ISSUE.md`](benchmarks/TORCH_COMPILE_ISSUE.md)。SL 输入必须是 Actor-only checkpoint，顶层严格包含 `model_config` 和 `model`。

## 正式训练

```bash
python -u -m training.train \
  --resume runs/policy-iteration-u61-champion-lcb-eager-fast-v1/latest.pt
```

训练无限运行，直到按 `Ctrl+C`。终端会显示每个 source、world replicate、inner step、累计 KL 和 arena 的进度。每个完整 attempt 打印 `PROMOTE` 或 `REJECT`、当前 champion、arena dRank、audit、KL 和耗时。

恢复：

```bash
python -u -m training.train \
  --resume runs/policy-iteration-u61-champion-lcb-eager-fast-v1/latest.pt
```

该 eager-v4 fast run 从 U61 champion fork，保留 Actor、self-play curriculum、opponent pool 和 root seed。评测使用 4096 个并行环境；这会改变 BF16 forward 的分块，不能与旧 512-env run 要求逐局相同。trace-v3 run 不再用于训练，它生成的 reference panel 也不会复用。v6 `latest.pt` 分别记录下一 attempt 和当前 champion iteration。中断时，未完成 attempt 的 A/B/C world shards 保留在 `pending/`；恢复会确定性补齐。恢复不接受 seed 或配置覆盖。v3、v4 和 v5 checkpoint 仍可读取和 fork。

## History Tournament

```bash
python -m training.history_tournament \
  --checkpoint runs/policy-iteration-v3/latest.pt \
  --output-dir runs/history-tournament-v1
```

该独立评测读取当前 Actor、历史池快照和冻结 SL，再加入 `rule_fast`、`rule_safe`。它枚举全部四人组合和座位排列，输出平均名次、分差、两两交手，以及以 SL 为零点的 Plackett-Luce Elo-like 相对强度和分层 block-bootstrap 区间。该 Elo 只描述这个固定对手池中的相对排序，不能当作跨规则或跨版本的绝对段位；`games.npz` 保留原始结果供后续复算，`agents.pt` 固化本次模型锚点，不受训练历史池后续轮换影响。

不同 run 的当前 Actor 可以直接加入同一场锦标赛；用 `--agents` 限定四人池可集中统计功效做 checkpoint 对比（必须保留 `sl` 作为 Elo 零点）：

```bash
python -m training.history_tournament \
  --checkpoint runs/policy-iteration-u56-fast-anchor-v1/latest.pt \
  --extra-checkpoint original_u063=runs/policy-iteration-v3/latest.pt \
  --agents sl u066 original_u063 rule_fast \
  --output-dir runs/history-tournament-fast-anchor-vs-original-v1 \
  --rounds-per-combination 192 \
  --bootstrap-samples 1000
```

## Batch Size Sweep

正式 sweep 比较每类 `64/128/256/512` 个状态，即总 batch `576/1152/2304/4608`。Sweep 使用独立的 1152 状态 KL calibration：

```bash
python -m training.batch_sweep \
  --output-dir runs/batch-sweep-v3 \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt
```

正式比较建议使用三个独立 seed：

```bash
python -m training.batch_sweep \
  --output-dir runs/batch-sweep-v3-multiseed \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt \
  --seeds 20260727 20260728 20260729
```

同一命令可恢复；各 seed 独立缓存，最终在根目录生成 pooled `aggregate.json` 和 seed 间波动统计。

训练和 sweep 的 target rollout 都默认以 64-query group 执行：Rust 引擎在组内使用 Rayon 多核，Actor 以最多 128 行的 CUDA batch 推理。单 GPU 不并发多个 seed 进程，避免复制模型、争用显存和 CUDA context；多 seed 仍按顺序运行并分别恢复。

各 batch 使用同一最大嵌套训练 corpus、独立 calibration 和 heldout corpus，以及同一 16384 局固定规则面板。输出包括方向一致性、有效样本量、heldout policy value、paired `dRank`、score、耗时和吞吐。

同一命令可恢复共享 corpus、每个 batch 的方向、candidate 原始评测面板和已完成候选。快速验证控制流：

```bash
python -m training.batch_sweep \
  --output-dir /tmp/batch-sweep-smoke \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt \
  --smoke
```

Smoke 仍要求 CUDA，只证明管线能完整执行，不证明策略提升。

## Optimizer 对照

QPC sweep 完成后，可以在不重跑 source/target rollout 的前提下比较 raw SGD 和现有 AdamW。命令只读输入目录，结果写入独立目录；默认跑输入里的全部 QPC 和全部 seed：

```bash
python -u -m training.optimizer_sweep \
  --batch-sweep-dir runs/batch-sweep-u56-qpc-v1 \
  --output-dir runs/optimizer-sweep-u56-sgd-v1 \
  --sgd-learning-rate 0.1 \
  --seeds 20260728 20260729
```

`--seeds` 允许跳过尚未完成的输入 seed；不指定时仍严格要求配置中的全部 seed 完成。输出会保留 SGD 原始方向、同 KL calibration、heldout value、与 AdamW 的同 seed paired 对局和 pooled aggregate。Nesterov 不在这个单 checkpoint sweep 中单独计入：冷启动的第一步与 SGD 共线，真正的差异需要跨 update 保存 velocity。

要隔离 AdamW 是否被“必须走满 KL”放大过头，复用同一输入只测 QPC 256 的原始方向 `0.5x` 和 `1.0x`。`1.0x` 就是 KL 只作上限时会提交的 policy；原 sweep 中已经缓存的约 `1.8x-2.4x` 强制标定 policy 直接作为对照，不会重跑：

```bash
python -u -m training.kl_scale_sweep \
  --batch-sweep-dir runs/batch-sweep-u56-qpc-v1 \
  --output-dir runs/kl-scale-sweep-u56-adamw-v1 \
  --qpc 256 \
  --scales 0.5 1.0 \
  --seeds 20260728 20260729
```

该实验不采集 source，也不生成新 Q target；只增加四个 16384 局固定规则 panel。`aggregate.json` 分别报告各 scale 相对 U56、原强制 KL AdamW、raw `1.0x` 的 pooled paired `dRank`，并保留 calibration KL、greedy flip 和 heldout value。

若缩小 AdamW 步长仍不能改善 reference，可进一步复用 AdamW/SGD 方向和两套 heldout Q，做跨 seed 泛化诊断：

```bash
python -u -m training.direction_generalization \
  --batch-sweep-dir runs/batch-sweep-u56-qpc-v1 \
  --optimizer-sweep-dir runs/optimizer-sweep-u56-sgd-v1 \
  --output-dir runs/direction-generalization-u56-v2 \
  --qpc 256 \
  --seeds 20260728 20260729 \
  --target-kl 0.0001
```

该命令不运行环境，只做 CUDA 前向。四个方向先在两 seed calibration 合并的共同 probe 上归一到相同小 KL，然后输出 `direction seed x heldout seed` 的 soft/greedy Q 矩阵，以及九类决策的跨 seed policy-delta cosine、flip、动作冲突率、reference 策略饱和度和 Q 最优动作分歧。

若主要问题是小 Q batch 同时更新全部表示层导致过拟合，可在不生成新 target 的情况下比较更新子空间：

```bash
python -u -m training.update_subspace_sweep \
  --batch-sweep-dir runs/batch-sweep-u56-qpc-v1 \
  --output-dir runs/update-subspace-sweep-u56-v1 \
  --qpc 256 \
  --seeds 20260728 20260729 \
  --scopes full last_blocks actor \
  --optimizers adamw sgd \
  --target-kl 0.0001
```

每个配置分别使用两个独立 seed 和两者合并的 4608-state pooled batch。输出比较 train-Q、两套 heldout-Q、跨 seed policy cosine，以及 pooled direction 在两套 heldout 上的最差值；只执行已有 Q 上的梯度和模型前向。

从 U56 fork 做长跑对照时，下面三条分支共享同一个 source iteration、self-play pool、seed，并都关闭已经单独检验过的 fast anchor；每条命令都应使用独立空目录：

```bash
python -u -m training.fork_policy_iteration \
  --source-checkpoint runs/policy-iteration-v3/latest.pt \
  --iteration 56 \
  --output-dir runs/optimizer-u56-adamw-v1 \
  --no-anchor-rule-fast \
  --direction-optimizer adamw \
  --direction-learning-rate 1e-5

python -u -m training.fork_policy_iteration \
  --source-checkpoint runs/policy-iteration-v3/latest.pt \
  --iteration 56 \
  --output-dir runs/optimizer-u56-sgd-v1 \
  --no-anchor-rule-fast \
  --direction-optimizer sgd \
  --direction-learning-rate 0.1

python -u -m training.fork_policy_iteration \
  --source-checkpoint runs/policy-iteration-v3/latest.pt \
  --iteration 56 \
  --output-dir runs/optimizer-u56-nesterov-v1 \
  --no-anchor-rule-fast \
  --direction-optimizer nesterov \
  --direction-learning-rate 0.1 \
  --direction-momentum 0.9
```

每个 fork 创建完成后分别恢复：

```bash
python -u -m training.train --resume runs/optimizer-u56-nesterov-v1/latest.pt
```

Nesterov 保存的是每次 KL 校准后真正提交的参数位移，在 `theta + 0.9 * velocity` 处计算下一轮梯度；中断恢复会一并恢复该 state。`momentum` 选项保留作 Polyak 对照。

## 输出

正式训练目录：

- `latest.pt`：唯一权威恢复点；
- `actor.pt`：Actor-only 部署权重；
- `config.json`：不可变 run 配置和输入身份；
- `metrics.jsonl`：每轮 source、target、optimizer、calibration 和 evaluation 指标；
- `reference_panel.npz`：冻结 SL 的固定规则面板；
- `pending/`：仅用于恢复未完成目标生成。

Sweep 目录额外包含 `summary.json`、`actor-qpc*.pt`、`shared/` corpus、`directions/`、`evaluation/`、训练/heldout target shard 和 reference panel。

## 验证

```bash
python -m py_compile training/*.py training/tests/*.py
python -m pytest training/tests -q
python -m training.benchmarks.transformer --device cuda
```

单元测试可以检查数据和算法不变量；实际训练、性能判断和端到端 smoke 必须在 CUDA 上完成。
