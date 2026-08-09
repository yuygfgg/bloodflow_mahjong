# 策略锦标赛

`rule-tournament` 对任意两种受支持策略执行平衡的二对二测评。工具只比较策略，不改变游戏规则。

支持的 CLI 策略标识符为：

- `rule-fast`
- `rule-ev`
- `rule-nn`

## 平衡设计

一个 block 使用一个牌局 seed，并运行两种策略各占两个座位的全部 6 种分配。一个 block 内：

- 每种策略共获得 12 个 seat-game（一个策略在一个座位的整局成绩）；
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

默认 `4096` blocks 会运行 24576 局。ONNX 推理可能需要较长时间。先用较小 blocks 测量吞吐，再决定正式样本量。

## 策略参数

A 侧参数使用 `--a-*`，B 侧使用 `--b-*`。两侧参数完全对称。

| 参数后缀 | `rule-fast` | `rule-ev` | `rule-nn` |
| --- | --- | --- | --- |
| `lookahead-depth` | 忽略 | 确定性前瞻深度 `0..3` | 忽略 |
| `defense` | 忽略 | `none` 或 `heuristic` | 忽略 |
| `nn-model` | 忽略 | 忽略 | 必填 ONNX 文件路径 |

CLI 为两侧提供统一参数集。与所选策略无关的参数不会进入动作配置。`nn-model` 只影响 `rule-nn`，并且选择 `rule-nn` 时必须提供。

查看当前二进制的完整默认值：

```bash
cargo run --release -p bloodflow-mahjong-rule-tournament -- --help
```

## NN 对 Rule-Fast 示例

`rule-nn` 在启动时加载一次 ONNX 模型。模型经过 schema 检查和图优化后，由所有 tournament worker 共享。必须先训练并导出包含 `engine_rules_version=10` metadata 的模型。仓库当前附带的旧 `model/latest.onnx` 不兼容，不能直接用于该命令。

以下命令应在 `engine/` 目录执行：

```bash
cargo run --release -p bloodflow-mahjong-rule-tournament -- \
  --blocks 32 \
  --bootstrap-samples 2000 \
  --root-seed 20260804 \
  --policy-a rule-nn \
  --a-nn-model /path/to/rules-v10.onnx \
  --policy-b rule-fast
```

当 B 侧使用 `rule-nn` 时，改用 `--b-nn-model`。策略名称包含模型文件名；例如 `rules-v10.onnx` 在输出中显示为 `rule_nn_rules_v10`。模型的输入、输出和合法动作过滤契约见 [`../../core/README.md`](../../core/README.md) 的 `rule-nn` 一节。

## 输出示例

`--blocks 1 --bootstrap-samples 100`，`rule-ev` 对 `rule-fast`输出：

```text
Rule tournament  policy-a rule_ev_d1_heuristic  policy-b rule_fast  blocks 1  games 6  root-seed 20260729  bootstrap 100  rayon-threads 12  parallel-games 6
Progress 6/6 (100.0%)  active 0  actions 665 (2835.3/s)  elapsed 00:00  ETA --:--
rule_ev_d1_heuristic Elo-like delta vs rule_fast +64.47  uncertainty unavailable (need at least 2 blocks)  cross-policy-win 0.5833
rule_ev_d1_heuristic seat-games       12  mean-rank 2.33333  mean-score +1516.67  first 0.2500  last 0.1667
rule_fast seat-games       12  mean-rank 2.66667  mean-score -1516.67  first 0.2500  last 0.3333
Decisions rule_ev_d1_heuristic  turns 172  Hu turn 16/16 response 86/86  kong concealed 0/0 added 2/9 exposed 0/0  pong 7/15  response-pass 8
Decisions rule_fast  turns 190  Hu turn 21/21 response 86/86  kong concealed 0/0 added 6/6 exposed 1/1  pong 19/20  response-pass 0
Throughput  play 0.235s  statistics 0.000s  games/s 25.576  actions/s 2834.7  actions 665
RESULT rule_ev_d1_heuristic-vs-rule_fast Elo +64.47 uncertainty-unavailable
```

## 输出解释

主要结果包含：

- `Elo-like delta`：基于四人名次的二标签 Plackett-Luce 拟合，再转换到 Elo 刻度；正值表示 A 侧更强；
- `CI95`：按 seed block bootstrap 的 95% 区间；
- `P(stronger)`：bootstrap 样本中 A 侧 delta 大于零的比例；数值为零的样本按 `0.5` 计入；
- `cross-policy-win`：同局内 A 座位名次优于 B 座位的两两比例；
- `mean-rank`、`mean-score`、`first` 和 `last`：每种策略的 seat-game 汇总；`mean-score` 是相对每局初始分数的平均变化，不是终局绝对分数；
- `Decisions`：胡、碰、杠和过的选择计数；
- `Throughput`：牌局、动作和统计耗时。

`Elo-like delta` 不是跨规则、跨阵容或跨版本通用的外部等级分。它只描述本次二对二实验。只有在新 seed 上复现且置信区间稳定时，才能把正点估计解释为可靠提升。

## 进度和并行

工具每 5 秒向 stderr 输出完成局数、活跃局数、动作吞吐、耗时和 ETA。最终选择会打印在启动行中。

固定 root seed、二进制版本和策略配置后，游戏结果不依赖任务完成顺序。动态 work queue 只改变调度，不改变 game index、seed 或座位分配。
