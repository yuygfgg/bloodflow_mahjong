# 血流麻将 AI 训练设计

本文定义无真人牌谱条件下的首版训练方案和冷启动规则策略。首个理想实验只用规则 AI 直接启动 Transformer；ResNet 不进入这条训练链路，等 Transformer 训练闭环稳定后再作为吞吐和样本效率对照。目标是在单张 RTX 5080 16GB、单次约 24 小时的预算内，建立可重复评测和持续迭代的训练闭环。

## 原则与边界

- 只使用当前行动者可见的策略观测。其他玩家暗手和牌墙只允许进入训练期 Critic 或辅助标签，不能进入 Actor。
- Actor 从第一步就只优化实际分差；向听、有效牌等规则指标只作为很小、很短的结构辅助，不作为胡牌奖励。
- 每局只训练一个座位，另外三个绝对座位分别从四种对手策略族中采样。learner 座位在牌局间轮换，保证四个座位都被训练。
- 训练过程用固定种子、四座轮换的 1v3 对局评测候选，不以训练 loss 或单次自博弈胜率判断强度。
- 首版只实现 Transformer 的 rollout、PPO 和评测器。ResNet 的输入和动作接口保留为未来对照，不为了它提前加入一套手工牌形特征。

## 共享输入编码

引擎观测是行动者视角，静态状态包含 `tile_obs[10,27]`、`melds[4,4,3]` 和 `meta[34]`；时间轴使用最近最多 192 条 `events[192,8]`。`river[108,2]` 保留给规则检查和消融，不再同时送入首版模型，避免和事件历史重复编码。模型侧按以下语义编码：

- 27 个牌种特征：自己的暗手和换牌选择、四家锁牌、四家弃牌计数、公开副露汇总、当前摸牌和响应牌标记。
- 最多 192 个历史事件：事件类型、相对行动者/目标、牌种、flags、数值字段和时间位置。
- 16 个副露槽：牌种、类型、相对玩家和来源玩家。
- 4 个玩家特征：相对庄家、当前分数、缺门、是否已经胡牌、暗手张数和历史最大胡牌倍率。
- 全局特征：阶段、换牌方向、墙剩余、换牌进度、补牌和响应 flags。

计数 `0..4`、花色、阶段、座位和副露类型使用 categorical embedding；分数除以 `10_000`，墙剩余除以 `55`，最大倍率使用 `log2(1+x)` 后归一化。`meta[1]` 的绝对行动座位不输入模型。

花色置换是合理的后续增强，但不能只在 PPO minibatch 中改观测：那会让 rollout 保存的 old log-prob 与增强后的状态不匹配。首版实现暂不做增强；若加入，必须在 rollout 推理前为每局固定采样一种置换，并把观测、事件、合法动作、执行动作和辅助标签一起映射。座位不做镜像增强，因为行牌方向和响应次序不是任意置换对称。

## 首版模型：静态编码器 + GPT 历史编码器

首版模型明确拆成两条路径：一个小型双向 Transformer 编码“当下”，一个 GPT 式因果 Transformer 编码 viewer-scoped 事件时间轴。两边各自产生一个 192 维 embedding，拼成 384 维后由全连接 Actor 和分布式 Critic 输出。这里不加入 ResNet 或额外手工牌形通道。

| 参数 | 配置 |
|---|---|
| `d_model` | 192 |
| static blocks | 2，双向 full attention |
| history blocks | 4，causal attention |
| attention heads | 6 |
| FFN | 768，SwiGLU |
| history position encoding | standard RoPE on Q/K，`theta=10,000` |
| normalization | Pre-RMSNorm |
| dropout | 0 |
| 计算精度 | BF16，FP32 optimizer state |

静态序列是 `[GLOBAL] + 27 TILE + 4 PLAYER + 16 MELD`，共 48 个 token。双向 attention 可以完整比较三门牌、公开副露、锁牌、当前分数和缺门状态；`GLOBAL` 输出作为 static embedding。

历史序列最多 192 个 event token，按时间顺序使用 causal mask，最后一个有效 token 的输出作为 history embedding。历史塔不再把可学习绝对位置向量加到 event embedding，而是在每层 attention 的 Q/K 上使用标准交错 RoPE（`theta=10,000`）。full forward 使用 `0..L-1` 位置；KV cache 追加时使用 `past_length..past_length+delta-1`，因此 cached/full 输出保持一致。rollout 为 learner 和冻结 Transformer 各维护一套 `(环境,绝对座位)` cache；viewer 变化、reset 或 192-token 滑动窗口变化时自动 full rebuild。少于 32 行的碎组并入一次 padded full forward，避免 cache 反而制造大量小 kernel；rollout 结束后释放全部 K/V，再进入 PPO。

拼接后的 384 维表示经过两层 MLP 输出 115 维 Actor logits，再应用 Legal Action Mask。Critic 输出固定 `[-4,4]` support 上的 categorical value distribution，并由其期望得到标量 value；Actor advantage 始终对应未经裁剪的真实分差。

预计参数量约 300 万到 400 万。这个拆分只包含一个明确的结构先验：当下状态允许双向比较，历史只能看过去。先验证规则 AI 直接冷启动能否学到分数策略，再决定是否需要更复杂的融合。

## 后置对照：牌种 ResNet + 牌河塔

牌种分支把 27 种牌 reshape 为 `[3,9]`，卷积只沿同花色的 rank 轴进行：

| 参数 | 配置 |
|---|---|
| channels | 192 |
| residual blocks | 20 |
| convolution | 每块两个 `(1,3)` convolution |
| normalization | GroupNorm 16 groups |
| activation | SiLU |
| river tower | 128 channels，4 个 dilation `1,2,4,8` temporal blocks |
| fusion | 512 维全局上下文回注到 27 个牌位置 |
| 计算精度 | BF16，FP32 optimizer state |

牌种分支不做 pooling。三个花色共享卷积权重，牌种特征与全局上下文融合后使用结构化策略头。该模型只在 Transformer 基线跑通后作为吞吐对照，不参与首版冷启动，不把额外的人类牌形先验混入首个实验。

预计参数量约 600 万到 800 万。它可能更快学会向听和普通胡，但不能据此推断后期总分上限；比较时按相同环境步数和 wall-clock 同时报告，不以最初两小时的胡牌率选型。

## Actor、Critic 与辅助任务

Actor 和首版 Critic 都只使用当前行动者可见的静态状态 embedding 与 viewer-scoped 历史 embedding。Critic 输出 `[-4,4]` support 上的 categorical 分布，并以期望值作为 PPO value；不对 Actor 暴露其他玩家暗手或牌墙。训练期 oracle Critic（四家暗手和剩余牌墙）作为后续消融，不是首版实现依赖。

首版实际训练两个低权重辅助头：

- 首次胡牌前的常规结构向听数，权重 `0.01..0.03`，前 10% 到 15% transitions 退火到零；
- 首次胡牌前的 27 维有效牌标签，接在静态结构分支，不反向主导历史上下文；

以下只保留为后续候选，不在当前 pipeline 中假装已经实现：

- 当前可胡牌型的 `log2(multiplier)` 分类和未来胡牌概率；
- 分数回报分布或 quantile value，而不是只预测“有没有胡”；
- 三家暗手计数的 belief prediction；
- 未来若干个同座位决策内的胡牌、点炮、杠收益、终局结算和分差。

辅助标签不参与推理。向听辅助头只在目标有效时计算 loss，不把终局哨兵或胡后常规 `-1` 当作标签。Actor 不增加胡牌次数或倍率 bonus；真实 score delta 已经表达低价值多胡和高价值少胡的权衡。

## 向听 API 与训练用法

引擎提供常规结构向听 evaluator，同时考虑标准四组一对、血流规则下龙七对的“四张算两对”、公开副露和定缺约束：

- Rust `analyze_shanten(counts, melds, missing_suit)` 返回 `ShantenAnalysis { shanten, improving_tiles }`；只需要数值时使用 `evaluate_shanten`。
- Rust `Game::hand_analysis(seat)` 分析指定座位；`Batch::hand_analysis_into` 批量分析全部当前行动者，`Batch::hand_analysis_indices_into` 只分析所选行。
- Python `Game.hand_analysis(seat) -> (shanten, improving_mask)`；训练使用 `Batch.hand_analysis_indices_into(uint32[N], int8[N], uint32[N])`，不会为约四分之一的 learner 行计算整个 batch。
- `shanten=-1` 表示已经包含胡牌结构，`0` 表示听牌，普通结果上限为 `8`。批量终局槽使用 `SHANTEN_TERMINAL=127`。
- `improving_tiles` 的低 27 bit 对应牌种；置位表示加入该牌会严格降低结构向听，不代表牌墙中一定还有该牌。

这是“当前完整持牌中是否已有一个成牌子结构”的常规向听，不是血流玩家胡牌后“距下一次可结算胡牌还有几步”。胡后锁定的旧结构仍在持牌中，结果通常持续为 `-1`。首版训练中，向听势函数和有效牌辅助 loss 必须在 `has_won` 后关闭；下一胡距离需要另做 lock-aware evaluator，不能复用本接口。

定缺牌不能组成面子或对子。势函数还应把当前缺门张数作为单独、较小的惩罚项；不要把它伪装成精确向听的一部分。有效牌剩余数由训练器用手牌和公开牌估算，隐藏牌墙不能进入 Actor 特征。

## 逐玩家 transition 与奖励

同一个玩家的一条 transition 从该玩家提交动作开始，到它下一次获得决策结束。收集器必须按绝对座位维护 pending transition：

1. 玩家 `p` 行动时保存观测、动作、log-prob 和 value。
2. 后续每个引擎 step 都把 `record[5+p] / 10_000` 累加到该 transition。
3. 玩家 `p` 再次行动时，用新的行动者视角观测结束上一条 transition。
4. 终局关闭四家的全部 pending transition。

这保证弃牌后经过多人响应才发生的点炮损失仍归因给原弃牌者。不能把一个 step 的分差简单交给执行该 step 的行动者。

基础奖励为真实分差：

```text
r_score = score_delta / 10_000
```

首版不额外加入排名奖励；如果部署目标最终按名次结算，再单独做小权重的后期 rank fine-tune。已经逐步累计分差后不能再次加入最终总分。

冷启动阶段曾考虑以下势函数：

```text
r = r_score + beta * (potential(next_state) - potential(state))
```

首版实现默认令 `beta=0`，即从第一个 transition 起只使用真实分差。只有冷启动实测出现长时间零收益时才开启该消融；首次胡牌前的 `potential` 只能由常规向听数、公开估计有效牌剩余数和缺门张数组成，胡牌后必须关闭。不要用 reward clipping 或 log transform 抹平 1 倍和高倍率实际分差；Critic 使用分布式 value 处理长尾。

## 冷启动规则对手

对手池有四种真正接入 rollout 的策略族，不是四个同时上桌的额外玩家：

| 名称 | 行为 | 用途 |
|---|---|---|
| `random_hu` | 有胡必胡，否则均匀随机合法动作 | 保持探索和最低强度锚点 |
| `rule_fast` | 引擎确定性 R2：弱门换牌/定缺，弃牌最小化精确向听并最大化公开估计有效张；积极碰杠 | 快速成牌冷启动 |
| `rule_safe` | 以 `rule_fast` 为结构默认，拒绝可过的碰/直杠；有明显更安全的公开熟张时才覆盖原弃牌 | 制造不同攻守轨迹 |
| `frozen_transformer` | 当前 learner 的冻结快照或保留的历史快照，只读取行动者观测 | 逐步转入自博弈 |

`rule_fast` 只能读取当前行动者暗手、自己的换牌选择、合法动作及公开锁牌、副露和弃牌。它不评估番型收益，也不读取对手暗手或牌墙，因此是冷启动基线，不是专家教师。Python 可用 `Game.simple_rule_action()` 和 `Batch.simple_rule_actions_into(uint8[B])` 调用；批量终局槽写 `SIMPLE_RULE_ACTION_TERMINAL=255`。

首版不做规则动作模仿：随机初始化的 Transformer 直接对阵上述在线对手，规则策略不产生监督 loss。只有实测出现数值不稳定、几乎不胡或 entropy 崩溃时，才单独实验最多 1 epoch 的在线动作预热；进入 PPO 前移除 imitation loss。ResNet 不参与首版链路。

## PPO 与对手联赛

首版 PPO 配置：

| 参数 | 初始值 |
|---|---|
| parallel environments | 2048 |
| learner batch | 65,536 个完整逐玩家 transitions |
| PPO epochs | 2 |
| minibatch | 4096，microbatch 512 累积；按历史长度分桶并裁剪右侧 padding |
| rollout KV cache 分组阈值 | 1024 行；小于阈值走一次合并的 full forward |
| learning rate | `2e-4` cosine decay 到 `3e-5` |
| gamma | 1.0 |
| GAE lambda | 0.95 |
| policy clip | 0.15 |
| value coefficient | 0.5 |
| max gradient norm | 0.5 |
| target KL | 0.015 |
| 规则混入门槛 | 首位率 `>= 0.70`，连续 3 次评测 |
| league 门槛 | 首位率 `>= 0.80`，连续 3 次评测 |
| entropy | `H/log(max(legal_count,2))`，系数 `0.01` 退火到 `0.002` |

每局是 `1 learner + 3 opponent`。另外三个绝对座位独立采样策略类型，因此同桌可以同时出现 `rule_fast`、`rule_safe` 和冻结 Transformer；不会强行让三个 opponent 使用同一种策略。为保持 GPU 批量推理效率，每个 rollout 只激活一个 Transformer 快照，所有抽中 `frozen_transformer` 的座位共享该版本。PPO policy loss 不按高倍率结果重采样。

对手 curriculum 不再按 learner transitions 的百分比切换，而由固定规则锚点评测门控。规则评测使用模型作为一方、另外三家使用 `simple_rule_actions_into`，以首位率作为“赢过三家规则对手”的定义。为避免 256 局噪声误触发，必须连续 3 次评测通过门槛；评测状态也会写入 checkpoint。

| 阶段 | 默认范围 | `random_hu` | `rule_fast` | `rule_safe` | `frozen_transformer` |
|---|---:|---:|---:|---:|---:|
| bootstrap | 规则首位率 `< 70%`，或尚未连续通过 3 次 | 30% | 50% | 20% | 0% |
| mixed | 规则首位率 `70%..<80%`，连续通过 70% 门槛 | 10% | 35% | 20% | 35% |
| league | 规则首位率 `>= 80%`，连续通过 80% 门槛 | 5% | 15% | 15% | 65% |

上表是“每个非 learner 座位”的抽样概率，不是四家座位的占比。达到 70% 只代表开始引入 Transformer 对手，不代表已经可以纯自博弈；达到 80% 后仍保留 35% 的规则对手作为锚点。当前版本不根据规则胜率自动切换到纯 Transformer self-play，避免模型在尚未证明能稳定击败规则策略时脱离外部锚点。日志中的 `opponent_assignments` 统计的是三个 opponent seat 的策略抽样次数。

首次连续通过 70% 门槛时创建第一个冻结快照；之后默认每 10 次 PPO 更新刷新一次。最多保留最近 4 个版本。league 阶段每次刷新时有 25% 概率激活历史快照，否则用最新快照。这里的“历史快照”是训练轨迹样本，不冒充经过显著性检验的 champion。

当前 pipeline 默认每 10 次更新对固定 `rule_fast` 锚点做四座轮换评测，并记录平均分差、分差标准差、首位率和平均名次；`2h/6h/12h/24h` 自动保存完整 checkpoint。训练结束后再对这些候选做更大规模的配对重复种子评测，只有平均效用 95% bootstrap 置信区间下界大于零时才标为 champion。当前训练循环不会根据小样本点估计自动晋级并改变训练分布。

### 阈值与 cache 的预算依据

- 2026-07-24 在本机 RTX 5080 的实测为 65,536 transitions/update 用 25.60 秒：rollout 4.80 秒，PPO 20.80 秒，总吞吐约 2,560 transitions/s。默认 200M 约 3,052 次 update，纯训练约 21.7 小时；加周期评测和 checkpoint 后对应约 22 到 24 小时。进入 mixed 或 league 的时间取决于规则胜率，不再由 wall-clock 或 transition 百分比保证。
- 一条 192-token、4 层、6 头、`d_model=192` 的 BF16 K/V cache 是 `4 * 2 * 6 * 192 * 32 * 2 = 589,824` bytes，约 0.56 MiB/viewer。`history_seat_masks` 只为 learner/冻结 Transformer 的下一决策写历史；只有达到 `history_cache_min_batch=1024` 的大组才走 cache，小碎组改走合并 full forward。所有 rollout cache 在 PPO 前释放，因此不会和 microbatch 512 的训练激活峰值叠加。
- 本机 full history 基准约 22.4k state/s，cached history 约 105.7k state/s；但端到端 rollout 还受 Python 分组、KV 拼接和 GPU launch 数量影响。RTX 5080、真实 `65,536` transitions 的 bootstrap rollout 为阈值 32/1024 分别约 4.90/3.77 秒；league 为约 10.47/8.18 秒。达到 192 条并发生环形窗口截断时自动 full rebuild，不能为了 cache 使用错误的旧前缀。

### 训练速度诊断与优化

此前一次约 5 小时的日志中，单个 update 平均约 25.6 秒，其中 rollout 约 11 秒、PPO 约 20.7 秒；因此瓶颈是 Python 侧重复的 Transformer forward 和过小的 GPU microbatch，而不是 Rayon 引擎 step。现有实现做了三项直接优化：

- PPO 使用适合 16GB 卡的 microbatch 512，并提供 `--microbatch` 覆盖值；先确认显存余量后再尝试 768 或 1024。
- rollout 数据不再整批常驻 GPU；每个 microbatch 只传输所需状态，避免 65,536 条历史记录额外占用数百 MB 显存。
- PPO 每个 epoch 按历史长度分桶，每个 microbatch 只把事件序列裁到该批次的最大有效长度。rollout 的 full/cache-miss forward 也做同样裁剪，保留 192 的上限和绝对 RoPE 位置。KV cache 只对达到 1024 行的同形状组启用；小组的动态 cache bookkeeping 比省下的 attention 更贵，因此合并为 full forward。
- RoPE 的 `sin/cos` 表按 `max_history` 预计算并在各层共享，避免每次 forward 重复三角函数计算。
- `training` 包在导入模型前默认设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，让变长 KV cache 使用可扩展 segment；不在每个 update 调用 `torch.cuda.empty_cache()`，避免强制每轮重新申请大块显存。
- 辅助向听标签退火后不再调用引擎分析；修复了此前 `collect_auxiliary=False` 时没有写入 pending transition、第二个 update 永久等待的逻辑错误。

这些改动只减少 padding、分析和 kernel-launch 开销，不改变合法动作、奖励、GAE 或 PPO loss。GPU 机器上应比较日志的 `rollout_seconds`、`ppo_seconds` 和 `states_per_second`；如果提高 microbatch 后出现 OOM，退回 `--microbatch 512`。

## 24 小时单次预算

先运行 CPU smoke，再直接启动默认训练：

```bash
python -m training.train --smoke --device cpu --output-dir /tmp/bloodflow-smoke
python -m training.train --device cuda --hours 24 \
  --total-transitions 200000000 --output-dir runs/transformer
```

- 前 1 小时：规则对手和 Transformer rollout smoke test，检查动作分布、KV cache 一致性和奖励归因。
- 中间阶段：先持续规则 bootstrap，达到规则首位率门槛后逐步加入冻结 Transformer 对手。
- 后段：保留规则锚点的 league 训练、固定快照、四座轮换评测和最终候选选择。

首版重训从随机初始化开始，不恢复旧的 self-play checkpoint。先用短 smoke test 检查实现；只有规则首位率连续达到 70% 才开始混入 Transformer 对手，达到 80% 后仍保留规则锚点。保存 `2h/6h/12h/24h` 快照并比较平均分、每次胡牌收入、高倍率尾部、点炮损失和终局结算。后续若实现 ResNet，再按相同环境步数与 wall-clock 做独立对照。

事件缓存必须绑定固定 viewer。若只训练一个 learner 座位，可以在该座位再次决策时追加其事件；若四个座位共享一个模型，则每个座位维护独立 cache，不能把相对座位已经旋转过的历史混用。

## 评测与“超过 90% 人类”

离线评测以四局为一组：同一种子中 challenger 分别坐四个座位，其余三家为同一 champion。至少报告：

- 平均分差及 bootstrap 95% CI；
- 平均名次、首位率和末位率；
- 胡牌、点炮、碰、杠、查花猪和查大叫频率；
- 不同换牌方向、庄闲和牌局长度分桶结果；
- 相对每个历史 champion 和规则锚点的结果。

“超过 90% 人类选手”指相同规则、相同时间限制下的 rating percentile，不能由自博弈胜率或行为预测准确率推出。最终声明需要授权真人对局或同规则天梯数据；离线联赛只负责选择和回归检测。

## 类似项目中可迁移的结论

- [Suphx](https://arxiv.org/abs/2003.13590) 展示了麻将中的 oracle guiding、全局奖励预测和自博弈价值，但其真人牌谱监督预热与日麻规则不能直接搬用。本方案只借鉴训练期特权 Critic 和最终按对局收益评测。
- [Mortal](https://github.com/Equim-chan/Mortal) 是开源日麻深度强化学习系统，说明规则引擎、合法动作约束和在线 RL 的工程闭环可以独立于闭源牌谱标签建立；动作和计分语义仍需以本血流引擎为准。
- [Kanachan](https://github.com/Cryolite/kanachan) 同时保留无序状态 token、牌河序列和 Transformer，并提供专门向听实现，支持这里的结构化 Transformer 输入设计。它明确依赖大量雀魂牌谱，所以这里只参考表示与测试方法，不采用其监督 curriculum。

这些项目都不是血流规则下的现成教师。能迁移的是信息边界、序列表示、规则验证和训练稳定化方法，不是模型权重、动作标签或人类强度结论。
