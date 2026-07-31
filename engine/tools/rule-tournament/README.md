# 策略锦标赛

`rule-tournament` 对任意两种内置策略执行平衡的二对二测评。工具只比较策略，不改变游戏规则。

支持的 CLI 策略标识符为：

- `rule-fast`
- `rule-ev`
- `rule-planner`

## 平衡设计

一个 block 使用一个牌局 seed，并运行两种策略各占两个座位的全部 6 种分配。一个 block 内：

- 每种策略共获得 12 个 seat-game；
- 每种策略在每个绝对座位出现 3 次；
- 六局共享初始牌序，用于降低座位和发牌噪声。

同一 block 内的六局不是独立样本。置信区间必须按完整 block bootstrap，不能把 6 局拆成独立样本。

## 快速验证

在 `engine/` 目录执行：

```bash
cargo run --release -p bloodflow-mahjong-rule-tournament -- \
  --blocks 1 \
  --bootstrap-samples 100 \
  --policy-a rule-ev \
  --policy-b rule-fast
```

一个 block 无法计算有效的不确定性。该命令只检查 CLI、牌局执行和结果输出。

## 全局参数

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--blocks` | `4096` | 独立 seed block 数；总局数为 `blocks x 6` |
| `--root-seed` | `20260729` | 派生牌局和 bootstrap seed 的根 seed |
| `--bootstrap-samples` | `2000` | block bootstrap 重采样次数 |
| `--parallel-games` | 自动 | 同时执行的最大局数 |
| `--policy-a` | `rule-ev` | A 侧策略 |
| `--policy-b` | `rule-fast` | B 侧策略 |

默认 `4096` blocks 会运行 24576 局。planner 搜索下可能需要很长时间。先用较小 blocks 测量吞吐，再决定正式样本量。

## 策略参数

A 侧参数使用 `--a-*`，B 侧使用 `--b-*`。两侧参数完全对称。

| 参数后缀 | `rule-fast` | `rule-ev` | `rule-planner` |
| --- | --- | --- | --- |
| `lookahead-depth` | 忽略 | 确定性前瞻深度 `0..3` | 忽略 |
| `hand-changes` | 忽略 | 忽略 | 候选图允许的有效换牌次数 `0..2` |
| `draw-horizon` | 忽略 | 忽略 | 候选图摸牌视野 `0..32` |
| `candidate-states` | 忽略 | 忽略 | 候选图状态上限 `1..200000` |
| `belief-worlds` | 忽略 | 忽略 | 信念粒子数 `0..256` |
| `response-worlds` | 忽略 | 忽略 | 响应分析世界数 `0..256` |
| `search-iterations` | 忽略 | 忽略 | 配对 rollout iteration 数 `0..4096` |
| `defense` | 忽略 | `none` 或 `heuristic` | 忽略 |

CLI 为两侧提供统一参数集。与所选策略无关的参数不会进入动作配置。`search-iterations` 只影响 `rule-planner`；`rule-fast` 和 `rule-ev` 会忽略该参数。

只有 `rule-planner` 的 `belief-worlds`、`response-worlds` 和 `search-iterations` 会影响内层搜索的自动并发判断。需要固定调度时，显式设置 `--parallel-games`。

查看当前二进制的完整默认值：

```bash
cargo run --release -p bloodflow-mahjong-rule-tournament -- --help
```

## Planner 对 EV 示例

以下命令运行 32 blocks，共 192 局：

```bash
cargo run --release -p bloodflow-mahjong-rule-tournament -- \
  --blocks 32 \
  --bootstrap-samples 10000 \
  --root-seed 20265001 \
  --parallel-games 4 \
  --policy-a rule-planner \
  --a-hand-changes 0 \
  --a-draw-horizon 1 \
  --a-candidate-states 1 \
  --a-belief-worlds 64 \
  --a-response-worlds 0 \
  --a-search-iterations 64 \
  --policy-b rule-ev \
  --b-lookahead-depth 1 \
  --b-defense heuristic
```

该命令用于统计验证，不是快速测试。`--parallel-games` 同时增加局级并行和内层搜索竞争。过高的值可能降低吞吐。正式运行前应在目标机器上比较 `1`、`2` 和 `4`。

## 输出解释

主要结果包含：

- `Elo-like delta`：基于四人名次的二标签 Plackett-Luce 拟合，再转换到 Elo 刻度；正值表示 A 侧更强；
- `CI95`：按 seed block bootstrap 的 95% 区间；
- `P(stronger)`：bootstrap 样本中 A 侧 delta 大于零的比例；数值为零的样本按 `0.5` 计入；
- `cross-policy-win`：同局内 A 座位名次优于 B 座位的两两比例；
- `mean-rank`、`mean-score`、`first` 和 `last`：每种策略的 seat-game 汇总；`mean-score` 是相对每局初始分数的平均变化，不是终局绝对分数；
- `Decisions`：胡、碰、杠和过的选择计数；
- `Planner search`、`Planner validation` 和 `Planner`：搜索决策、验证拒绝、改动和 rollout 计数；
- `Throughput`：牌局、动作和统计耗时。

`Elo-like delta` 不是跨规则、跨阵容或跨版本通用的外部等级分。它只描述本次二对二实验。只有在新 seed 上复现且置信区间稳定时，才能把正点估计解释为可靠提升。

## 进度和并行

工具每 5 秒向 stderr 输出完成局数、活跃局数、动作吞吐、耗时和 ETA。启用内层搜索时，工具倾向使用保守的局级并发；无内层搜索时，默认使用 Rayon 线程数。最终选择会打印在启动行中。

固定 root seed、二进制版本和策略配置后，游戏结果不依赖任务完成顺序。动态 work queue 只改变调度，不改变 game index、seed 或座位分配。
