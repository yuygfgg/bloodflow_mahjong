# 无牌谱 IQL/AWR 训练方案

本文对应 `training/` 当前实现。主链路使用完整合成轨迹上的离线到在线策略迭代：

```text
冻结 SL / 规则 / 当前策略 / 历史策略生成完整对局
    -> 紧凑轨迹和严格确定性重放
    -> Monte-Carlo return-to-go
    -> 独立 Double-Q + Expectile V
    -> 通过校准门控后做 AWR Actor 提取
    -> 立即生成新轨迹并继续训练
```

训练只支持 CUDA，默认无限运行，直到用户按 `Ctrl+C`。没有自动停止、强制解冻或按 update 数强行推进 Actor 的兜底。

## 信息边界和奖励

- Actor、Q1、Q2 和 V 只能读取当前行动者可见的手牌、公开状态和 viewer-scoped 事件历史。
- Critic 与 Actor 完全不共享参数，Critic 梯度不能覆盖 SL Actor 已学到的表征。
- 每一步的目标是该座位从当前动作到终局的真实累计分差，按 `10_000` 归一化。
- 不加入排名奖励、胡牌 bonus、手工 shaping 或 reward clipping。
- 四个座位和九类决策都进入 replay，不能只训练中后盘弃牌。
- 非法动作在 Actor 和 Q 网络内都被 legal mask 排除。

对于轨迹中第 `t` 个动作和座位 `i`：

```text
G_t(i) = sum_{k=t..terminal} score_delta_k(i) / 10_000
```

开局的正确期望通常接近零，因此 Critic 是否学会不能只看全局 value loss。验证会按早、中、晚局和九类动作分别报告误差、相关性、常数基线改进和 Double-Q disagreement。

## Actor 和 SL 引用

默认 Actor 是约 8.84M 参数的双塔 Transformer：

- `d_model=256`、8 个 attention heads；
- 静态状态塔 3 层双向 self-attention；
- 历史塔 5 层 causal self-attention，最多 192 条事件并使用 RoPE；
- FFN 宽度 1024，策略头输出 115 个动作 logits。

新训练器不执行 SL。它直接读取当前 Actor-only 格式的：

```text
runs/counterfactual-larger/sl_reference.pt
```

该策略同时作为：

1. 当前 Actor 的初始化；
2. 永久冻结的参考策略；
3. 合成数据行为策略和对手池成员；
4. AWR 更新中的 KL 锚点。

`--sl-checkpoint` 必须是只含 `model_config` 与 `model` 的 Actor-only 文件。训练目录不会生成或覆盖该参考策略。

## 完整轨迹数据

### 采集

每局从初始 seed 开始完整运行到引擎终局，一局同时记录四个座位的所有决策。默认使用 512 个并行环境，先采集 8192 局永久 anchor 数据，之后每轮再采集 1024 局在线数据。

CUDA 推理会把同策略行数 pad 到少量固定 batch bucket，并把有效历史长度 pad 到分桶宽度；重复行只用于计算填充，结果会在写回前截断，不改变动作分布。训练包还默认启用 PyTorch expandable allocator segments。这两项用于避免长时间混合策略采集时为大量不同 attention shape 保留碎片化显存；在 RTX 5080 16GB 上，完整默认启动阶段已通过压力测试。

训练更新使用单批后台 replay 预取：CUDA 训练 batch N 时，独立 CPU 线程从 seed/action 轨迹重建 batch N+1。重建只为实际抽中的 `(trajectory, step)` 生成 observation、history 和 legal mask；此前未抽中的前缀只推进引擎，不生成会被丢弃的大数组。一个 optimizer batch 整体只搬到 CUDA 一次，microbatch 在设备上切 view。训练器刚刚由同一引擎生成的完整轨迹不再立即重复跑第二遍；外部轨迹入库仍执行严格确定性重放，持久化 shard 在加载时仍检查格式、规则版本和 CRC。

### CUDA 吞吐和稳定性

训练对当前 Actor、采集策略、冻结 SL、历史 Actor、Q1/Q2/V 和 Oracle 全部使用 CUDA eager，不调用 `torch.compile`。本机的 `max-autotune` 曾触发 NVIDIA GSP `KernelChannelGroupApi alloc RPC` 失败；普通动态 compile 也在确定性复现中于历史长度 11 生成大量 NaN/Inf logits，而相同 checkpoint、seed 和 512 个环境的 eager 采集完整通过。因此所有编译路径已从代码中删除。

启动和恢复不再有图编译或 autotune 等待。终端会明确打印 `CUDA eager mode; torch.compile disabled`。

本机 RTX 5080 的真实 replay/CUDA 基准如下；这些结果用于选择默认值，不以占满显存为目标：

| 项目 | 实测 |
| --- | ---: |
| eager 采集，`envs=512` | 约 9,800 states/s |
| 采集，`envs=1024` | 约 8,195 states/s |
| Critic，旧串行 replay + CUDA | 约 0.67 s/update |
| Critic，按需 replay + 后台预取 | 约 0.45 s/update |

因此默认保留 `envs=512` 和 `microbatch-size=256`：microbatch 384 没有吞吐收益，512 会使 Actor OOM；即使监控中仍有空闲显存，也不要仅为提高显存占用而放大它们。

对手从离线阶段开始就是混合分布：

| 成员 | 默认权重 |
| --- | ---: |
| `rule_fast` | 15% |
| `rule_safe` | 15% |
| 冻结 SL | 20% |
| 当前 Actor | 30% |
| 历史冻结 Actor | 20% |

历史池为空时，其 20% 权重暂时回到当前 Actor。每 50 次可信 Actor 更新冻结一个快照，最多保留 `max_history=16` 个；对应的 CUDA 策略也只保留这些有效快照，快照淘汰时同步释放其模型。历史权重在最近 4 个和更早快照之间分层分配。规则和冻结 SL 始终保留，训练不会退化为纯当前策略自博弈。每局保证一个轮换座位使用当前 Actor，避免当前策略在采集分布中消失。

行为探索按决策类别设置，而不是统一做 15% 合法动作均匀随机：

- 换牌和弃牌使用温度及 top-k，并保留少量全合法集探索；
- 定缺探索更低；
- 副露响应只保留很小探索；
- 胡牌响应仍有非零探索，但默认仅 `0.1%`。

轨迹保存每个实际动作的 behavior probability、采样温度、策略来源和版本。规则动作也经过同一合法采样层，因此记录的概率与真实执行分布一致。

### 紧凑格式和重放

轨迹不持久化每一步的大 observation。版本化格式保存：

- 初始 seed、换牌方向和引擎规则版本；
- 每步动作、绝对座位、阶段、九类决策类别；
- 策略来源、策略版本、动作概率和温度；
- 四家终局分数、名次和终止原因；
- 格式版本与 CRC。

动作和类别使用紧凑整数编码，每步固定记录约 15 字节。数据按 shard 落盘。采样时从 seed 和动作序列严格重建 observation、事件历史、legal mask、Oracle tile counts 和四座位 return-to-go。以下任一不一致都会直接报错：

- 动作在对应状态非法；
- actor、phase 或决策类别不同；
- 提前终局或轨迹未到终局；
- 终局分数、名次或终止原因不同；
- 数据版本、规则版本或 CRC 不匹配。

train/validation 按完整轨迹划分，默认 10% 对局进入 validation，同一局的状态不会泄漏到两边。

### Replay 平衡

Anchor 轨迹永久保留；在线 replay 是最多 2,000,000 transitions 的滑动窗口。抽样同时按来源和九类决策设置硬保底，并对重复状态及过度集中的策略版本降采样。默认每个 batch 至少包含：

- 冻结 SL 12%；
- `rule_fast` 8%；
- `rule_safe` 8%；
- 当前策略 10%；
- 已存在时的历史策略 5%；
- 每个决策类别至少 1%。

若 replay 缺少必要来源或决策类别，训练会明确失败，不会静默用失衡 batch 继续。

## 独立 Q1/Q2/V

Q1、Q2、V 都有自己的可见状态编码器，不共享 Actor 或彼此的参数。默认每个 Critic 使用较小的 `d_model=128` Transformer；Q 网络一次输出全部 115 个动作值，非法动作被 mask。

实际行为动作使用完整轨迹 return-to-go 做 Double-Q Huber 回归：

```text
L_Q = Huber(Q1(s, a), G_t) + Huber(Q2(s, a), G_t)
```

离线数据对两个 Q 分别加入只覆盖合法动作的 CQL：

```text
L_CQL = logsumexp(Q(s, legal_actions)) - Q(s, behavior_action)
```

CQL 不按 update 数机械归零。其系数随 replay 中当前策略数据覆盖率下降，并始终保留最小比例。

V 使用保守 Double-Q 行为动作值做 expectile regression，默认 `tau=0.7`：

```text
q = min(Q1(s, a), Q2(s, a))
residual = q - V(s)
L_V = |tau - 1[residual < 0]| * residual^2
```

新 run 先执行默认 500 个 Critic warmup steps。之后每轮继续 64 个 Critic steps；Actor 是否更新由验证质量决定，而不是只看累计步数。

普通 Critic 更新始终使用完整轨迹 replay：Q1、Q2 做绝对 return-to-go 回归，V 做 expectile 回归，CQL 只约束合法动作。C 的 MC 更新是额外的一步，只更新 Q1/Q2，不替换普通 Q 目标，也不更新 V；因此普通 Critic 仍负责绝对分数校准。

## Critic 门控和 AWR

所有实验先检查 partial Critic。Actor 只有同时满足以下基础条件才可能更新：

- Critic 至少训练 500 steps；
- 中盘和晚盘 Q 都优于对应目标均值的常数基线；
- 中盘和晚盘 Q 与真实 return-to-go 有最低正相关；
- Q1/Q2 平均 disagreement 不超过阈值；
- validation 中存在对应阶段样本。

若基础门控不通过，Actor 保持冻结，但系统仍继续训练 Critic 并采集新完整轨迹。B/C 还必须让各自教师独立通过验证并连续稳定 3 次。没有“最多等 20 次”、按 update 数强制解冻或 smoke 强行放行的逻辑。

B 的 Oracle 必须独立满足早、中、晚盘 Q 校准，整体和早盘 Q MAE 都至少比 partial 低 2%，Q disagreement、中晚盘 V correlation 和 expectile balance 也必须达标。开局 V 接近零本来就可能正确，因此不拿早盘 V correlation 强行门控。Oracle 未通过时仍独立训练，但既不能给 Actor 计算 advantage，也不能向 partial Critic 蒸馏。

C 在 partial 基础门控通过后就开始查询 MC，即使 Actor 仍冻结。训练与验证 MC 目标分别积累；只有目标数量、可靠验证状态组数、可靠动作 pair 数、这些 pair 上的 accuracy 和平均动作 regret 都达标并连续稳定 3 次，才允许 Actor 使用已经受 MC 校验的 partial Critic。MC teacher 不再重复门控 absolute-Q MAE、常数基线 improvement 或 correlation：普通 `critic_ready` 已负责绝对尺度，MC 数据只训练可靠动作差值，不直接作为 Actor 分类标签。

通过门控后，Actor 每轮默认执行 8 个 AWR steps。只学习 replay 中真实执行过的合法动作：

```text
A(s, a) = min(Q1(s, a), Q2(s, a)) - V(s)
w = min(exp(A / beta), w_max)
L_actor = -w * log pi(a | s) + lambda_ref * KL(pi || frozen_SL)
```

默认 `beta=0.10`、`w_max=20`、`lambda_ref=0.05`。MC teacher 行不直接充当 Actor 分类标签。日志会报告相对 SL 的 KL、优势、权重、有效样本量及各动作类别分布。

每次可信 Actor 提取后立即增加策略版本、同步采集器、生成新轨迹，并在到达快照间隔时将 Actor 加入历史对手池。

## 三组递进实验

通过 `--experiment` 选择实验，三者应按等 GPU 时间依次比较，不能把附加项全部堆叠后再猜测收益来源。

### A：Partial-observation IQL/AWR

```bash
python -m training.train \
  --experiment a \
  --output-dir runs/iql-awr-a-v3
```

这是默认基线，只使用部署时可见信息训练 Q1/Q2/V 和 Actor。先确认完整轨迹 Critic 确实能在中晚盘及少数动作类别上超过常数基线。

### B：Oracle Critic

```bash
python -m training.train \
  --experiment b \
  --output-dir runs/iql-awr-b-v3
```

Oracle Q1/Q2/V 只在训练期额外读取四家暗手与剩余牌墙的计数表示。只有 Oracle 连续通过独立教师门控后，Partial Critic 才蒸馏验证覆盖到的 logged-action Q，AWR 才能使用 Oracle advantage；未执行合法动作不做全动作蒸馏，因为这些输出没有 return 标签验证。Actor 的输入和部署图没有 Oracle 特征，Oracle 参数也不与 Actor 共享。

该实验用于判断主要瓶颈是否来自部分可观测下的 Critic 方差，不是允许 Actor 偷看暗牌。

### C：选择性信息集 Monte Carlo

```bash
python -m training.train \
  --experiment c \
  --output-dir runs/iql-awr-c-v3
```

只有 Critic 门控通过后，C 才查询高不确定状态。候选来自当前 Actor、冻结 SL、规则动作和保守 Q；查询优先级综合 Q1/Q2 disagreement、Actor/SL 分歧、接近零优势和中晚盘风险。

每个查询默认最多 3 个候选动作、32 个信息集世界和 1 次 continuation。引擎固定行动者可见信息，重新分配其他玩家暗手与牌墙；同一组候选共享 world seed，以候选间逐 world 的回报差做 paired confidence interval。只有 `abs(mean_difference) - confidence_half_width >= 0.02` 且 half-width 不超过上限的动作边才算可靠。没有可靠边的动作会被裁掉；剩余完整 query 带持久化 query ID 和对称可靠边图整组接收，容量淘汰也按整组执行，不能混用不同查询的 world。响应窗口当前不做重采样查询，因为保留 pending legality 需要固定暗手。

MC 目标只作为 Critic 回归数据，训练和 validation 查询分开，不能直接生成硬 argmax Actor 标签。训练 MC 可以继续从在线状态积累；validation MC 每 2 个 iteration 从永久 anchor validation 轨迹查询，并排除已经存在的状态。validation corpus 只有同时达到 512 个 targets、128 个含可靠边的 query groups 和 128 条可靠 pair 后才将 `validation_frozen` 置为 true；只满足 target 数量不会停止补充查询。冻结后在线状态不再进入 validation teacher corpus，validation 目标也不参与 train MC Q 更新。validation 只在可靠边上报告 pairwise accuracy、top-action accuracy 和 regret；在这些动作级指标可信前 Actor 始终冻结。

对一个已接收 query 的可靠边 `(j,k)`，令 paired-world MC 目标差为 `d_y = y_j-y_k`，Q 差为 `d_q = q_j-q_k`：

```text
L_difference = equal-query mean Huber(d_q, d_y)
w_jk = min(abs(d_y) / 0.10, 1)
L_ranking = equal-query mean w_jk * 0.10 * softplus(-sign(d_y) * d_q / 0.10)
L_MC = 1.0 * L_difference + 0.25 * L_ranking
```

Q1 和 Q2 分别计算后取平均；每个 query 权重相等，可靠边较多的 query 不会支配 batch。`absolute_loss` 仅作为诊断指标，不进入 MC 优化目标；普通轨迹 Critic 负责把 Q 的绝对尺度校准到真实分差。

## 评测和 best 选择

默认每 5 轮在固定 seed panel 和独立 fresh seed panel 上各做一次确定性评测。每个 panel 都包含：

- `rules`：当前 Actor 对交替的 `rule_fast` / `rule_safe`；
- `sl`：当前 Actor 对三家冻结 SL；
- `mixed`：当前 Actor 对完整混合池；
- `history`：当前 Actor 对最近历史策略，无历史时回退冻结 SL。

每组报告平均名次、首位率、末位率、平均分差、分差标准差以及跨 seed 波动。只有固定 seeds 和 fresh seeds 的规则评测都优于当前 best，才写入 `best.pt`；训练本身不会因此停止。

同时应重点查看：

- 中晚盘 Q MAE、相关性、校准误差和常数基线改进；
- 各决策类别的 Q 误差及 Q1/Q2 disagreement；
- Actor 相对冻结 SL 的 KL、AWR 权重和有效样本量；
- replay 来源、九类决策覆盖和采集吞吐；
- B 的 Oracle/partial 差距或 C 的 MC 方差、置信区间与 GPU 成本。
- C 的 `mc_critic.mc_critic_loss`、`mc_absolute_loss`、`mc_centered_loss`、`mc_pairwise_loss`、`mc_train_pairwise_accuracy`、`mc_train_groups` 和 `mc_train_pairs`；
- C 的 `mc.accepted_queries`、`accepted_targets`、`terminal_rollouts`、`rollout_states`、`train_targets`、`train_targets_after_trim`、`validation_targets`、`validation_reliable_targets`、`validation_reliable_groups`、`validation_reliable_pairs`、`validation_frozen`、`mean_variance`、`mean_confidence_half_width`；
- C 的 train/validation pairwise accuracy，以及 `mc.validation_metrics.action_ranking` 下的可靠 `group_count`、`pair_count`、`all_pair_count`、`top_action_accuracy`、`mean_regret`、`maximum_regret` 和 `mean_action_gap`。MC Q MAE 仍可观察，但不参与 teacher gate。

## 运行、恢复和输出

安装 release Python 扩展后启动默认实验：

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
python -m training.train \
  --output-dir runs/iql-awr-v3 \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt
```

训练会一直运行。按一次 `Ctrl+C` 后，当前循环退出并原子写入 `latest.pt`。

恢复时只需提供完整 IQL/AWR checkpoint：

```bash
python -m training.train --resume runs/iql-awr-v3/latest.pt
```

C 实验恢复示例：

```bash
python -m training.train --resume runs/iql-awr-c-v3/latest.pt
```

恢复会读取 checkpoint 中的实验、模型与训练配置、Actor、冻结 SL、Q1/Q2/V、可选 Oracle、全部优化器、教师连续就绪计数、策略池、采样器 RNG、replay cursor 和随机状态；replay shard/manifest 必须与 checkpoint 一致。恢复不会重新 SL，也不能改成另一实验类型。当前 checkpoint 格式为 v3；v1/v2 IQL/AWR checkpoint 会明确拒绝，不做默认字段填充或迁移。

每个 run 目录包含：

- `latest.pt`：最近完整 checkpoint；
- `best.pt`：固定和 fresh seed 都提升时写出的 Actor-only 部署权重，不用于恢复训练；
- `metrics.jsonl`：完整结构化训练记录；
- `dashboard.html`：由 metrics 生成的本地摘要；
- `config.json`：实验、模型、训练配置和评测 seeds；
- `replay/manifest.json` 与 `replay/*.bfsh`：轨迹清单和紧凑 shard；
- `snapshots/*.pt`：历史 Actor-only 策略。

终端默认只打印 baseline、Critic、iteration、门控、KL、名次和分差摘要；`--verbose-console` 额外打印完整 JSON。

先做一次 CUDA 端到端检查：

```bash
python -m training.train \
  --smoke \
  --output-dir /tmp/bloodflow-iql-smoke \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt
```

`--smoke` 会缩小环境、模型、replay 和评测规模，只运行一个 iteration 后保存退出。它通过显式的小样本配置和宽松阈值覆盖 Actor、Oracle 与 MC 路径，不会在控制流里强行覆盖失败门控。smoke checkpoint 不应用于长训。正式训练绝不回退 CPU；显存不足时优先降低 `--microbatch-size`，其次再调整 batch 或并行环境数。

## 验证

```bash
cargo test --manifest-path engine/Cargo.toml --workspace --all-targets
cargo clippy --manifest-path engine/Cargo.toml --workspace --all-targets -- -D warnings
python -m pytest engine/pybind/tests training/tests
```

基准脚本：

```bash
python -m training.benchmarks.transformer --device cuda
python -m training.benchmarks.train_step --device cuda
```
