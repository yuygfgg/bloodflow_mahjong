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
| normalization | Pre-RMSNorm |
| dropout | 0 |
| 计算精度 | BF16，FP32 optimizer state |

静态序列是 `[GLOBAL] + 27 TILE + 4 PLAYER + 16 MELD`，共 48 个 token。双向 attention 可以完整比较三门牌、公开副露、锁牌、当前分数和缺门状态；`GLOBAL` 输出作为 static embedding。

历史序列最多 192 个 event token，按时间顺序使用 causal mask，最后一个有效 token 的输出作为 history embedding。每层实现了 K/V cache，并有 cached/full 一致性测试。rollout 为 learner 和冻结 Transformer 各维护一套 `(环境,绝对座位)` cache，按精确的 `(past_length, delta_length)` 分组后追加事件；viewer 变化、reset、事件窗口截断或历史前缀不一致时自动 full rebuild。这样保留 cache 的主要收益，同时不强行 padding 不同长度的因果前缀。

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
- Rust `Game::hand_analysis(seat)` 分析指定座位；`Batch::hand_analysis_into` 批量分析当前行动者。
- Python `Game.hand_analysis(seat) -> (shanten, improving_mask)`；`Batch.hand_analysis_into(int8[B], uint32[B])` 直接写调用方 NumPy 缓冲区。
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
| minibatch | 4096，按显存使用 microbatch 256 或 1024 累积 |
| learning rate | `2e-4` cosine decay 到 `3e-5` |
| gamma | 1.0 |
| GAE lambda | 0.95 |
| policy clip | 0.15 |
| value coefficient | 0.5 |
| max gradient norm | 0.5 |
| target KL | 0.015 |
| entropy | `H/log(max(legal_count,2))`，系数 `0.01` 退火到 `0.002` |

每局是 `1 learner + 3 opponent`。另外三个绝对座位独立采样策略类型，因此同桌可以同时出现 `rule_fast`、`rule_safe` 和冻结 Transformer；不会强行让三个 opponent 使用同一种策略。为保持 GPU 批量推理效率，每个 rollout 只激活一个 Transformer 快照，所有抽中 `frozen_transformer` 的座位共享该版本。PPO policy loss 不按高倍率结果重采样。

默认 `--total-transitions 100000000` 时按 learner transitions 分三段，切换发生在下一次 rollout 开始前：

| 阶段 | 默认范围 | `random_hu` | `rule_fast` | `rule_safe` | `frozen_transformer` |
|---|---:|---:|---:|---:|---:|
| bootstrap | `0..<10M`，即 `0..<10%` | 30% | 50% | 20% | 0% |
| mixed | `10M..<35M`，即 `10%..<35%` | 10% | 35% | 20% | 35% |
| league | `35M..<75M`，即 `35%..<75%` | 5% | 15% | 15% | 65% |
| self-play | `75M+`，即 `75%+` | 0% | 0% | 0% | 100% |

上表是“每个非 learner 座位”的抽样概率，不是四家座位的占比。league 阶段一桌的 frozen seat 数量服从 `Binomial(3, 0.65)`：平均 `1.95` 个；恰好 0/1/2/3 个 frozen 的概率约为 `4.3%/23.9%/44.4%/27.5%`。最后的 self-play 阶段才是严格的 `1 learner + 3 frozen Transformer`，规则 AI 只用于独立评测和回归，不再进入训练 rollout。日志中的 `opponent_assignments` 统计的是三个 opponent seat 的策略抽样次数。

达到 10M transitions 时创建第一个冻结快照；之后默认每 10 次 PPO 更新刷新一次。最多保留最近 4 个版本。mixed、league 和 self-play 阶段每次刷新时有 25% 概率激活历史快照，否则用最新快照。这里的“历史快照”是训练轨迹样本，不冒充经过显著性检验的 champion。

当前 pipeline 默认每 10 次更新对固定 `rule_fast` 锚点做四座轮换评测，并记录平均分差、分差标准差、首位率和平均名次；`2h/6h/12h/24h` 自动保存完整 checkpoint。训练结束后再对这些候选做更大规模的配对重复种子评测，只有平均效用 95% bootstrap 置信区间下界大于零时才标为 champion。当前训练循环不会根据小样本点估计自动晋级并改变训练分布。

### 阈值与 cache 的预算依据

- 默认 100M learner transitions、65,536 transitions/update，共约 1,526 次 update。BF16 基准的单个 256-state PPO step 为 0.074 秒，按两次 PPO epoch 折算，纯反向约 38 秒/update、约 16 小时；加上引擎 rollout、快照和评测，实际约 20 到 24 小时。因此 10% 规则冷启动约占 2 小时，35% 约在 7 到 9 小时进入 league，75% 后保留约四分之一预算做纯 Transformer self-play；50% 会把切换再推迟约 3 到 4 小时，没有足够收益。
- 一条 192-token、4 层、6 头、`d_model=192` 的 BF16 K/V cache 是 `4 * 2 * 6 * 192 * 32 * 2 = 589,824` bytes，约 0.56 MiB/viewer。2048 桌中 learner 约 1.1 GiB，三个冻结对手的上界约 3.4 GiB，加上分组拼接的临时峰值约 5 到 6 GiB；和训练 step 的约 2.9 GiB 峰值合计仍适合 16 GiB 卡。
- 本机 full history 基准约 22.4k state/s，cached history 约 105.7k state/s。实际 rollout 还包含静态编码、Python 分组和 Rust step，不能直接承诺 4.7 倍端到端提升；按 viewer/past-length 分组仍能显著减少历史塔重复计算。达到 192 条并发生环形窗口截断时自动 full rebuild，不能为了 cache 使用错误的旧前缀。

## 24 小时单次预算

先运行 CPU smoke，再直接启动默认训练：

```bash
python -m training.train --smoke --device cpu --output-dir /tmp/bloodflow-smoke
python -m training.train --device cuda --hours 24 \
  --total-transitions 100000000 --output-dir runs/transformer
```

- 前 1 小时：规则对手和 Transformer rollout smoke test，检查动作分布、KV cache 一致性和奖励归因。
- 中间约 20 小时：PPO 联赛训练，按 learner transitions 而不是 epoch 计进度。
- 最后约 3 小时：固定快照、四座轮换评测、故障余量和最终候选选择。

首版 24 小时只运行 Transformer。先用短 smoke test 检查实现，不用前两小时胡牌率淘汰模型；保存 `2h/6h/12h/24h` 快照并比较平均分、每次胡牌收入、高倍率尾部、点炮损失和终局结算。后续若实现 ResNet，再按相同环境步数与 wall-clock 做独立对照。

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
