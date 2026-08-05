# 训练

## 验证

运行 Python 测试前，先构建当前的 Python 绑定：

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
python -m pytest training/tests
```

用较小的 CPU 配置运行两个阶段：

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

## ONNX 导出

将完整的 PPO checkpoint 导出为固定的 `rule-nn` 契约：

```bash
python -m training.export_onnx \
  runs/ppo-self-play-25-from16h-seed7/latest.pt \
  model/latest.onnx
```

导出器将策略移至 CPU，并写出形状为 `[1, 115]` 的原始 Actor logits。计算图接受 `tile_obs`、`melds`、`meta`、192 条观察者视角事件和事件长度。引擎在推理后应用合法动作 mask。

默认情况下，该命令检查 ONNX 计算图，并将一个确定性输入与 PyTorch 对比：报告最大和平均绝对误差，并要求 argmax 一致。checkpoint 模型必须支持至少 192 条历史事件。仅在隔离排查导出器问题时使用 `--no-check` 或 `--no-parity`。

从仓库根目录通过 Rust 加载器运行一局：

```bash
cargo run --release --manifest-path engine/Cargo.toml \
  -p bloodflow-mahjong \
  --features rule-nn \
  --example rule_nn_smoke -- \
  model/latest.onnx 7
```

`rule-nn` 的平衡测评见 [`../engine/tools/rule-tournament/README.md`](../engine/tools/rule-tournament/README.md)。

## Rule-EV 监督学习起点

监督阶段用确定性 `RuleEvConfig.standard()` 为每个当前玩家状态打标签。Rule-EV 控制全部四个座位。15% 的均匀随机合法行为动作用于扩展访问到的状态分布，但标签仍为 Rule-EV 动作。只有一个合法动作的强制决策不存储为标签。

```bash
python -m training.supervised \
  --device cuda \
  --labels 10000000 \
  --output-dir runs/rule-ev-sl-pilot
```

输出 `actor.pt` 只包含共享的 observation 编码器和策略头。它记录模型配置、引擎规则版本和训练输入 schema，加载时任何不匹配都会被拒绝。PPO 的价值头和辅助头从零开始，且 PPO 不继承监督学习的优化器状态。

## PPO

PPO 默认使用混合奖励：

```text
score_delta / 10,000 + terminal_rank_utility
```

终局名次效用为第 1 至第 4 名分别取 `+1`、`+1/3`、`-1/3`、`-1`。平分玩家与测评使用相同的严格分数比较。分数和名次权重记录在 checkpoint 中，可用 `--score-reward-weight` 和 `--rank-reward-weight` 设置。

每个 PPO epoch 将所有 rollout 状态随机分配到 minibatch。

默认的 `--kl-control monitor` 在每个 epoch 后对固定的随机监控样本测量完整合法动作分布，不改变优化过程。`--kl-control off` 移除该测量。`--kl-control rollback` 是可选的分开实验：当某个 epoch 超过 `--target-kl` 时恢复模型和优化器。

从监督学习 Actor 开始两小时试点：

```bash
python -m training.train \
  --device cuda \
  --hours 2 \
  --init-actor runs/rule-ev-sl-pilot/actor.pt \
  --output-dir runs/ppo-rule-ev-pilot
```

PPO 只有一个停止预算：`--hours` 决定的累计运行墙钟时间。transition 和 update 只是吞吐指标。学习率、熵和辅助调度始终使用固定的 24 小时视界，因此在两小时处停止试点再恢复，不会重启或重新缩放退火。恢复的 run 会还原累计时间和收集器种子，然后继续朝新的小时目标运行：

```bash
python -m training.train \
  --device cuda \
  --hours 24 \
  --resume runs/ppo-rule-ev-pilot/latest.pt
```

每次恢复都会向 `metrics.jsonl` 追加一条显式的 `resume` 记录，标识 checkpoint 状态和新的 run 目标。进程中断后，先前 checkpoint 之后的指标可能仍留在文件中；请把 resume 记录视为新一个指标世代的开头。

第一个基线保持关闭 self-play。每个非学习座位独立地以 `1/3` 概率使用 Rule-Fast、`2/3` 概率使用标准 Rule-EV。

用 `--fork` 从完整 PPO checkpoint 开启 self-play，而不修改源 run。fork 会还原模型、价值头、优化器、RNG 状态、收集器状态和累计的 24 小时调度，只允许修改对手课程。以下命令将 16 小时策略继续到累计 24 小时目标：

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

Rule-EV 门槛通过后，每个非学习座位以 `25%` 概率使用冻结策略、`50%` 概率使用 Rule-EV、`25%` 概率使用 Rule-Fast。对手池每 200 次 PPO 更新创建一个快照，并保留四个，包括 fork 锚点。每次 rollout 以相等概率使用最新快照或保留的历史快照。定期测评仍使用三名 Rule-EV 玩家作为固定的外部锚点。

从已完成的 Rule-EV 监督 Actor 启动修正后的混合试点。不要恢复失败的纯分数 PPO run：

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

试点确认监督基线没有立即丢失后，从同一个监督 Actor 运行可比的十小时任务：

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

从新的输出目录运行完整基线：

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

每次测评使用固定的精确 `(seed, focal seat)` 组合面板。每个种子在四个座位各出现一次，其余三名玩家使用标准 Rule-EV。已完成的排位不会被重新填充，因此短局不会被重复计数。定期测评复用同一个固定面板以测量趋势；最终测评使用新面板。

要检验 PPO 快照之间是否互相变强，运行确定性交叉对战竞技场。矩阵每个单元格用行快照对阵列快照的三份拷贝，每个焦点座位均等出现：

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

竞技场把并列名次分摊到其占用的排位。它对每个四座位面板取平均后，根据独立种子计算标准误。对角线为第 2.5 名、分数为 0。若某个快照学到了总体上更强的策略，较晚的快照相对较早快照应名次更低、分数为正。下结论前请将矩阵与独立的 Rule-EV 测评对比，不要只看单个对阵。

`latest.pt` 存储完整的 PPO 模型和优化器、RNG 状态、对手状态、收集器种子、累计 PPO 时间、引擎规则版本、输入 schema 和精确的 PPO 配置。在累计 PPO 时间的 2、6、12 和 24 小时处还会写入快照。

终端在每个监督批次或 PPO update 后打印一行紧凑的进度摘要。`metrics.jsonl` 保留每条完整的结构化指标记录。
