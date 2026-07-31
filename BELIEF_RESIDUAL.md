# Belief residual 操作手册

本文说明如何生成 belief residual 数据、训练模型、校验模型包，并在 `rule-planner` 中使用模型。架构和长期实验方案见 [`NEURAL_PLANNER.md`](NEURAL_PLANNER.md)。

## 范围

当前链路只实现根节点隐藏世界后验残差：

```text
Rust belief-dataset
  -> safetensors 数据分片
  -> PyTorch belief residual 训练
  -> safetensors 模型包和 golden 样本
  -> Candle CPU 校验与推理
  -> rule-planner 根节点信息集搜索
```

模型为每个候选隐藏世界输出一个 residual log weight。Rust 仍负责游戏规则、合法动作、手写后验、配对根搜索、独立验证和最终动作。模型包不是完整策略，不能单独运行一局。

固定输入契约如下：

- 公开事件历史为 `[192, 8]`。`event_lengths` 标记有效前缀，padding 值为 `-1`。
- 每个候选隐藏世界有 4 个 `27` 维平面。前三个平面是相对座位 1、2、3 的暗手计数；第四个平面是无序剩余牌直方图。
- 候选世界不包含未来摸牌顺序。
- 一个训练根状态包含两条各有 `K` 个 proposal 的独立粒子流，以及 1 个 truth。候选总数为 `2K+1`。

## 环境

Rust workspace 需要 Rust 1.85 或更高版本。Python 训练器需要 Python 3.10 或更高版本、PyTorch、NumPy 和 `safetensors`。

从仓库根目录构建当前链路：

```bash
cargo build --manifest-path engine/Cargo.toml --release \
  -p bloodflow-mahjong-belief-dataset \
  -p bloodflow-mahjong-model-runtime \
  -p bloodflow-mahjong-rule-tournament

python -c 'import numpy, safetensors, torch; print(torch.__version__, numpy.__version__, safetensors.__version__)'
```

`--device cuda` 在 CUDA 不可用时会停止。CPU 训练可显式使用 `--device cpu`。Candle 运行时当前只使用 CPU。

## 生成数据

以下命令生成至少 20,000 个根状态：

```bash
RAYON_NUM_THREADS=32 cargo run --manifest-path engine/Cargo.toml --release \
  -p bloodflow-mahjong-belief-dataset -- \
  --output-dir runs/belief-residual-data-v2 \
  --roots 20000 \
  --candidates 64 \
  --shard-roots 2048 \
  --seed 20260801 \
  --random-action-probability 0.15
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `--roots` | 请求的最小根状态数 |
| `--candidates` | 每条 proposal 流的粒子数 `K`，可部署范围为 `9..=256` |
| `--shard-roots` | 每个完整分片的最大根状态数 |
| `--seed` | 数据生成根 seed |
| `--random-action-probability` | 随机化数据来源策略的动作概率 |

`--output-dir` 必须不存在或为空。生成器不会把新分片写入已有数据目录。

生成器以 4 局为一个平衡 block。生成器会完成已经开始的 block，所以实际根状态数可以略高于 `--roots`。train、calibration 和 development 按完整 block 以 8:1:1 分配；同一 block 不会跨 split。

每个分片包含以下主要张量：

| 张量 | shape | dtype |
| --- | --- | --- |
| `tile_obs` | `[N, 10, 27]` | `u8` |
| `melds` | `[N, 4, 4, 3]` | `u8` |
| `river` | `[N, 108, 2]` | `u8` |
| `meta` | `[N, 34]` | `i32` |
| `events` | `[N, 192, 8]` | `i32` |
| `event_lengths` | `[N]` | `u16` |
| `candidate_worlds` | `[N, 2K+1, 4, 27]` | `u8` |
| `handwritten_log_weights` | `[N, 2K+1]` | `f32` |
| `positive_mask` | `[N, 2K+1]` | `u8` |
| `proposal_streams` | `[N, 2K+1]` | `u8` |
| `block_ids`、`root_ids` | `[N]` | `u64` |

`river` 保留在数据 schema 中。当前网络不把它作为独立输入；公开弃牌信息由状态特征和事件历史表达。

## 审计数据

先检查 manifest 的版本、split 和审计计数：

```bash
DATA=runs/belief-residual-data-v2

jq '{
  schema_version,
  belief_target_version,
  engine_rules_version,
  proposal_stream_count,
  candidate_count,
  max_history,
  minimum_roots,
  roots,
  games,
  split_roots: (.shards | group_by(.split) |
    map({split: .[0].split, roots: (map(.roots) | add), shards: length})),
  audit
}' "$DATA/manifest.json"
```

当前版本必须满足：

- `schema_version=2`；
- `belief_target_version=2`；
- `engine_rules_version=3`；
- `proposal_stream_count=2`；
- `max_history=192`；
- `candidate_count=K`；
- `audit.positive_nonfinite=0`。

再校验所有分片的 SHA-256：

```bash
(
  cd "$DATA"
  jq -r '.shards[] | "\(.sha256)  \(.path)"' manifest.json | sha256sum --check
)
```

`proposal_nonfinite` 和 `proposal_streams_without_finite_weight` 可以非零。它们表示 proposal 没有手写 posterior 支持。部署路径把该粒子流的 learned ESS 记为 0，并原子回退两条流。`proposal_collisions` 和 `positive_collisions` 也必须报告；重复 truth 会作为完整 positive 等价类参与训练。

训练器会再次校验分片 SHA-256、tensor shape、dtype、positive 等价类和 split block 不相交。

## 训练模型

以下命令使用当前正式 pilot 配置：

```bash
python -m learning.belief.train \
  --dataset runs/belief-residual-data-v2 \
  --output runs/belief-residual-model-v2 \
  --device cuda \
  --seed 20260801 \
  --epochs 6 \
  --batch-size 64 \
  --learning-rate 3e-4 \
  --weight-decay 0.01 \
  --gradient-clip 1.0 \
  --variance-weight 1e-3 \
  --warmup-fraction 0.02
```

`--output` 必须不存在或为空。训练器使用 AdamW 和分组 contrastive density-ratio loss。手写 log weight 是固定 offset，网络只学习 residual。

数据用途严格分离：

- train 更新模型参数；
- calibration 在 `0, 0.25, 0.5, 0.75, 1.0` 中选择 residual scale `beta`；
- development 在模型参数和 `beta` 固定后只评估一次。

calibration 和 development 使用 FP32。每个 `beta` 在两条 proposal 流上分别计算 ESS。任一流 ESS 小于 8 时，两条流都回退到手写 posterior；训练器使用回退后的 deployment NLL 选择 `beta`。

输出目录包含：

| 文件 | 用途 |
| --- | --- |
| `model.safetensors` | 模型参数 |
| `golden.safetensors` | PyTorch 参考输入、公开编码和 residual 输出 |
| `manifest.json` | 模型配置、版本、`beta`、粒子数、SHA-256、审计和指标 |

`model.safetensors` 不是 Python checkpoint。模型运行时不需要 Python。

## 校验模型包

先校验模型包内两个文件的摘要：

```bash
MODEL=runs/belief-residual-model-v2

(
  cd "$MODEL"
  jq -r '"\(.model_sha256)  model.safetensors\n\(.golden_sha256)  golden.safetensors"' \
    manifest.json | sha256sum --check
)
```

再用 Candle 复现 PyTorch golden 输出：

```bash
cargo run --manifest-path engine/Cargo.toml --release \
  -p bloodflow-mahjong-model-runtime \
  --bin belief-model-check -- \
  --model runs/belief-residual-model-v2 \
  --tolerance 1e-4
```

运行时严格检查 artifact 类型、schema、target、游戏规则版本、双 proposal 流、校准粒子数、网络 shape、有限 `beta`、两个文件的 SHA-256、golden tensor shape 和 CPU 数值误差。任何检查失败都会停止加载。运行时不会静默换用其他模型。

## 在 planner 中使用模型

`rule-tournament` 是当前完整的模型使用入口。以下命令比较带 residual 的 planner 和完全相同的手写 posterior planner：

```bash
RAYON_NUM_THREADS=32 cargo run --manifest-path engine/Cargo.toml --release \
  -p bloodflow-mahjong-rule-tournament -- \
  --blocks 32 \
  --bootstrap-samples 10000 \
  --root-seed 20260803 \
  --parallel-games 2 \
  --policy-a rule-planner \
  --a-hand-changes 0 \
  --a-draw-horizon 1 \
  --a-candidate-states 1 \
  --a-belief-worlds 64 \
  --a-response-worlds 0 \
  --a-search-iterations 64 \
  --a-root-belief posterior \
  --a-continuation current \
  --a-belief-model runs/belief-residual-model-v2 \
  --policy-b rule-planner \
  --b-hand-changes 0 \
  --b-draw-horizon 1 \
  --b-candidate-states 1 \
  --b-belief-worlds 64 \
  --b-response-worlds 0 \
  --b-search-iterations 64 \
  --b-root-belief posterior \
  --b-continuation current
```

带模型的一侧必须满足以下契约：

- 策略必须是 `rule-planner`；
- `root-belief` 必须是 `posterior`；
- `continuation` 必须是 `current`；
- `search-iterations` 必须至少为 9；
- `search-iterations` 必须严格等于模型包的 `calibration_particle_count`。

当前 v2 模型使用 `K=64`，所以两侧都使用 `search-iterations=64`。`belief-worlds` 和 `response-worlds` 是 planner 的其他预算，不表示模型校准粒子数。

模型在每个可搜索根状态批量计算两条粒子流的 residual。selection 和 validation 使用独立 seed domain。任一 learned 流的 ESS 小于 8 时，两条流原子回退到手写 posterior。输出中的 `Belief residual` 行报告决策数和 ESS fallback 数。

一个 tournament block 包含 6 局平衡座位分配。置信区间按完整 block bootstrap 计算。smoke test 只能检查执行链路，不能判断强度。正式比较必须换独立 root seed，并保留完整命令和输出。

## 当前 v2 pilot

以下数字只描述模型指纹 `6fb91996e699` 的首轮 pilot。它们不是最终强度结论。

| 项目 | 实测值 |
| --- | --- |
| 数据 | 20,079 roots，664 games，`K=64`，129 candidates/root |
| split | train 16,251；calibration 1,914；development 1,914 |
| 数据大小 | 413 MiB |
| 数据生成 | 1,050.6 秒，不含首次 Cargo 编译 |
| 模型参数 | 1,578,113 |
| 训练 | 6 epochs，97,506 root exposures，约 1,524 optimizer steps |
| PyTorch 计时 | 15.71 秒，不含初始分片 SHA 审计 |
| 最终 `beta` | 1.0 |
| calibration deployment NLL | 2.24688 -> 2.22900 |
| development deployment NLL | 2.36638 -> 2.35034 |
| development atomic fallback | 32.60% -> 33.54% |
| 8-block smoke | `+0.00 Elo-like`，平均分 `+22.92` |
| 32-block 独立 seed | `-1.42 Elo-like`，CI95 `[-4.25, 0.00]`，平均分 `+18.23` |

离线 NLL 有一致改善，平均分也有弱正向变化，但名次没有改善。当前结果只能说明模型学到了有限的后验信号；它没有证明整局强度提升。扩展数据量、对手分布或模型之前，必须继续使用独立 seed 和同配置 baseline 区分后验质量与名次目标。

## 故障排查

- 输出目录非空：使用新的目录。不要混合不同 schema 或 seed 的分片。
- 分片 SHA、shape 或版本失败：保留 manifest，确认数据和代码来自同一版本。
- CUDA 不可用：使用 `--device cpu` 做功能检查，或在 CUDA 主机上训练。
- tournament 拒绝 `search-iterations`：读取模型 manifest 的 `calibration_particle_count`，并让模型侧和 baseline 侧使用相同值。
- Candle golden 失败：停止实验。不要修改容差来掩盖系统性误差。
- ESS fallback 比例异常：检查零支持流比例、`beta` 和粒子预算。不要只报告成功使用 residual 的根状态。

每次正式实验至少保存以下信息：Git commit、完整命令、硬件、线程数、数据 manifest、模型 manifest、模型指纹、seed、blocks、bootstrap 样本数、planner 参数和完整 stdout/stderr。

## 验证

从仓库根目录执行：

```bash
cargo fmt --manifest-path engine/Cargo.toml --all -- --check
cargo test --manifest-path engine/Cargo.toml --workspace --all-targets
cargo clippy --manifest-path engine/Cargo.toml --workspace --all-targets -- -D warnings
python -m pytest learning/belief/tests
```
