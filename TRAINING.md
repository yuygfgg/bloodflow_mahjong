# 血流麻将 AI 训练设计

本文定义无真人牌谱条件下的首版训练方案、两个候选模型和冷启动规则策略。目标是在单张 RTX 5080 16GB、单次约 24 小时的预算内，建立可重复评测和持续迭代的训练闭环。

## 原则与边界

- 只使用当前行动者可见的策略观测。其他玩家暗手和牌墙只允许进入训练期 Critic 或辅助标签，不能进入 Actor。
- 训练目标以实际分差为主。向听、有效牌等规则指标只用于早期势函数和辅助任务，并在训练前 20% 内退火。
- 每局只训练一个座位，另外三个座位使用冻结对手。四个座位在不同牌局中均匀轮换。
- 候选模型通过固定种子、四座轮换的 1v3 对局晋级，不以训练 loss 或单次自博弈胜率选择模型。
- Transformer 和 ResNet 使用相同的输入语义、动作头、Critic、rollout 和评测器，确保消融结果可比较。

## 共享输入编码

引擎观测是行动者视角，包含 `tile_obs[10,27]`、`melds[4,4,3]`、`river[108,2]` 和 `meta[34]`。模型侧按以下语义编码：

- 27 个牌种特征：自己的暗手和换牌选择、四家锁牌、四家弃牌计数、公开副露汇总、当前摸牌和响应牌标记。
- 108 个牌河事件：牌种、相对持有者、绝对位置和距当前的相对位置。
- 16 个副露槽：牌种、类型、相对玩家和来源玩家。
- 4 个玩家特征：相对庄家、当前分数、缺门、是否已经胡牌、暗手张数和历史最大胡牌倍率。
- 全局特征：阶段、换牌方向、墙剩余、换牌进度、补牌和响应 flags。

计数 `0..4`、花色、阶段、座位和副露类型使用 categorical embedding；分数除以 `10_000`，墙剩余除以 `55`，最大倍率使用 `log2(1+x)` 后归一化。`meta[1]` 的绝对行动座位不输入模型。

训练时随机应用全部 6 种花色置换，并同步置换牌特征、牌河、副露、缺门以及四段 27 维牌动作。座位不做镜像增强，因为行牌方向和响应次序不是任意置换对称。

## 模型 A：结构化 Transformer

首版使用双向 Transformer Encoder，不使用 recurrent hidden state：

| 参数 | 配置 |
|---|---|
| `d_model` | 192 |
| blocks | 6 |
| attention heads | 6 |
| FFN | 768，SwiGLU |
| normalization | Pre-RMSNorm |
| dropout | 0 |
| 计算精度 | BF16，FP32 optimizer state |

Token 顺序为 `[GLOBAL] + 27 TILE + 108 RIVER + 4 PLAYER + 16 MELD`，最大长度 156。使用 padding mask，不使用 causal mask；所有输入事件在当前决策前已经可见。

27 个 `TILE` 输出分别经过共享的四分类线性层，形成换牌、弃牌、暗杠和加杠四组牌 logits。`GLOBAL` 输出定缺 3 项以及胡、碰、直杠、过 4 项。按固定动作布局拼成 115 维 logits 后应用 Legal Action Mask。

预计参数量约 300 万到 400 万。本机简化的 136 token BF16 eager 基准约为 4.3 万到 4.9 万 state/s。若注意力成为主要瓶颈，备选优化是保留 `[GLOBAL] + 27 TILE` 为 query，把牌河、玩家和副露作为 memory，改用 query self-attention 加 cross-attention。

## 模型 B：牌种 ResNet + 牌河塔

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

牌种分支不做 pooling。三个花色共享卷积权重，牌种特征与全局上下文融合后使用和模型 A 相同的结构化策略头。牌河塔对 108 个有序弃牌做 embedding 和时间卷积，attention pooling 后进入全局上下文。

预计参数量约 600 万到 800 万。本机 718 万参数的代表性牌种 ResNet BF16 eager 基准约为 17 万 state/s。该模型提供更强的牌型归纳偏置和更高吞吐，作为 Transformer 的必要基线长期保留。

## Actor、Critic 与辅助任务

Actor 严格使用 viewer-scoped 观测。Critic 共享公开特征 trunk，并允许拼接训练专用 oracle embedding：

- 四家暗手的 `[4,27]` 计数；
- 剩余牌墙的 27 维牌种直方图，不需要牌墙顺序。

oracle 特征只进入 value head，Actor 导出图中不包含该输入。首版 value head 预测当前行动者从当前状态到终局的归一化分数变化。

建议联合训练以下辅助头，单项 loss 权重从 `0.05..0.2` 搜索：

- 首次胡牌前的常规结构向听数；
- 首次胡牌前的 27 维有效牌标签；
- 最大可胡倍率；
- 三家暗手计数的 belief prediction；
- 未来若干个同座位决策内的胡牌、点炮和分差。

辅助标签不参与推理。向听辅助头只在目标有效时计算 loss，不把终局哨兵或胡后常规 `-1` 当作标签。

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

终局可额外加入零和排名奖励 `[0.15, 0.05, -0.05, -0.15]`，平分时取并列名次奖励的平均值。已经逐步累计分差后不能再次加入最终总分。

冷启动阶段允许加入势函数：

```text
r = r_score + beta * (potential(next_state) - potential(state))
```

首次胡牌前，`potential` 由常规向听数、公开估计有效牌剩余数、缺门张数和最大潜在倍率组成并裁剪到 `[-1,1]`；胡牌后去掉向听和有效牌项。`beta` 从 `0.03..0.05` 开始，在前 20% learner transitions 内线性退火到零。

## 冷启动规则对手

规则对手不是专家标签，而是用来打破四个随机共享策略的对称性，并提供永久评测锚点：

- `R0`：均匀随机合法动作，但有胡必胡。
- `R1`：换弱门、定缺最弱门、保留对子和搭子的结构策略；有胡必胡，杠优先，基础版本允许积极碰牌。
- `R2`：弃牌先最小化精确常规向听，再最大化公开估计的有效牌剩余张数，以局部牌形和公开暴露度破同分；下一步再加入保守碰杠和局面风险。

当前实现是确定性的早期 `R2`：它只能读取当前行动者暗手、自己的换牌选择、合法动作及公开锁牌、副露和弃牌。它已经使用精确常规向听与有效牌 mask，但仍然有胡必胡、见杠就杠、可碰就碰，不评估番型收益、点炮风险或胡后下一胡距离，因此只是冷启动对手，不应描述为强 AI。Python 可用 `Game.simple_rule_action()` 和 `Batch.simple_rule_actions_into(uint8[B])` 调用；批量终局槽写 `SIMPLE_RULE_ACTION_TERMINAL=255`。

默认不做长时间规则模仿：直接让 learner 对阵 `R0/R1/R2` 可以产生在线冷启动数据，也不会把规则策略变成模型上限。若随机初始化在 30 分钟内仍出现极高非法数值、几乎不胡或 entropy 崩溃，再用规则动作做最多 1 epoch、100 万到 200 万在线状态的短预热；进入 PPO 前完全移除 imitation loss。

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

每局随机指定一个 learner 座位，其他三家使用同一个冻结对手版本，减少推理分组和同局非平稳性。

冷启动分三段：

1. `0..15M` learner transitions：`20% R0 + 60% R1/R2 + 20% 早期快照`，使用势函数和较高 entropy。
2. `15M..50M`：`30% R0/R1 + 70% 冻结 champion/历史快照`，势函数退火到零。
3. `50M+`：`50% 当前 champion + 35% 最近 8 个 champion + 15% R1`，只优化真实分差和终局排名。

候选模型每 30 到 60 分钟评测一次。只有在配对重复种子的平均效用 95% bootstrap 置信区间下界大于零时才晋级 champion；证据不足时继续训练，不因点估计领先而晋级。

## 24 小时单次预算

- 前 1 小时：规则对手和辅助任务的在线 warm-up，检查动作分布和奖励归因。
- 中间约 20 小时：PPO 联赛训练，按 learner transitions 而不是 epoch 计进度。
- 最后约 3 小时：固定快照、四座轮换评测、故障余量和最终候选选择。

Transformer 与 ResNet 首次比较各运行约 2 小时的相同种子短实验，使用每小时固定评测效用提升作为指标。最终 24 小时运行只选择短实验中 wall-clock 效率更好的配置；另一个模型继续作为消融基线。

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
