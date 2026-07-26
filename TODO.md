# Synthetic Offline-to-Online RL 实现清单

本清单已于 2026-07-26 完成落地。`[x]` 表示代码、指标、门控或实验入口已经实现并通过测试；“优于常数基线”“不劣于 SL”和 A/B/C 等算力比较仍是正式长训时的验收条件，不代表 smoke test 已经证明模型强度提升。

原来的单隐藏状态、单次终局反事实分支不能稳定区分相近策略动作。当前训练主线已经改为合成完整轨迹上的离线到在线策略迭代：先学习可校准的 `Q/V`，再用 AWR 从数据中提取 Actor，持续生成新轨迹并重复训练，直到用户按 `Ctrl+C`。

## 已确认的问题与方案判断

当前失败的核心不是麻将规则模拟不准确，而是反事实标签的统计含义不可靠：`Batch.clone_indices()` 复制的是完全相同的真实暗手和牌墙，每个候选动作又只续着一次。候选回报差会被对手暗手、后续摸牌顺序和单次续着波动主导，而当前 Actor 与冻结 SL 的实际动作差异很小，因此训练主要在追逐隐藏状态噪声。

扩大相同隐藏状态上的分支数量不能解决这个问题。它只能更精确地估计“已知完整隐藏世界时哪个动作更好”，而部署 Actor 真正需要的是：

```text
E[terminal score delta | actor-visible observation, action]
```

因此当前实现采用以下优先级：

1. 普通随机化完整轨迹是批量训练数据的主来源。
2. Oracle Critic 是较低成本的训练期降方差实验。
3. 信息集蒙特卡洛是选择性高质量教师，不是全量 replay 生成器。

合成数据的“质量”首先由策略与对手多样性、九类决策覆盖、合理探索、behavior probability、replay 平衡和确定性重放保证，而不是简单增加每个状态的 rollout 数。

## 主数据引擎的统计依据

普通完整轨迹中的单个 return-to-go 仍然有较高方差，但大量不同隐藏状态上的样本共同训练后，partial-observation Critic 可以逼近当前可见信息下的期望回报。完整对局还可以同时为四个座位、早中晚局和所有动作类别产生样本，单位终局 rollout 的利用率远高于逐状态动作搜索。

主链固定为：

```text
rule / SL / current / historical policy mixture
    -> stochastic full trajectories for all four seats
    -> Monte-Carlo return-to-go labels
    -> Double-Q + Expectile V
    -> AWR Actor extraction
    -> new trajectory generation
    -> repeat
```

这里的 `Monte-Carlo return-to-go` 指完整真实轨迹的终局采样回报，不等于为每个状态枚举动作并搜索多个隐藏世界。

## 目标与约束

- [x] 复用 `runs/counterfactual-larger/sl_reference.pt`，不重新执行 SL。
- [x] 保留冻结 SL Actor 作为行为策略、参考策略和退化保护；训练 Actor 从相同权重初始化。
- [x] Actor 在推理时只能读取行动者可见信息，不能读取暗手或牌墙。
- [x] 训练覆盖四个座位和全部九类决策，不能只优化中后盘弃牌。
- [x] 真实累计分差是主奖励，使用 `score_delta / 10_000`，不添加手工胡牌或排名奖励。
- [x] 所有采集和训练只支持 CUDA，不实现 CPU fallback。
- [x] 默认无限循环训练和采集；只在用户按 `Ctrl+C` 时保存并退出，不设自动停止门槛。
- [x] 新实现落地后删除当前 counterfactual trainer、旧 checkpoint 兼容和不再使用的配置，不保留双训练路径。

## 1. 紧凑轨迹数据

- [x] 定义带版本号的轨迹格式，至少保存：
  - 初始随机种子和引擎规则版本；
  - 每步动作、行动座位和决策类别；
  - behavior policy 的版本、动作概率和采样温度；
  - 四家终局分数、名次和终止原因。
- [x] 动作使用 `uint8` 等紧凑表示，不为每一步持久化完整 observation。
- [x] 使用种子和动作序列确定性重放，重建 observation、legal mask 和每个座位的 return-to-go。
- [x] 重放时检查动作合法性、终局分数和原轨迹一致；任何不一致直接报错，不能静默丢弃。
- [x] 按轨迹而不是单状态切分 train/validation，避免同一局泄漏到两边。
- [x] Replay 记录数据来源：`sl`、`rule_fast`、`rule_safe`、`current`、`frozen_policy`、`mc_teacher`。

## 2. 合成数据生成器

- [x] 批量生成完整对局，并记录四个座位的所有决策，使一局数据可训练四个视角。
- [x] 对手池永久包含 `rule_fast`、`rule_safe`、冻结 SL、当前策略和若干历史冻结策略。
- [x] 不再等待 55% 首位率才启用 self-play；从离线数据阶段就混合策略来源。
- [x] 在线阶段保留固定比例的规则和冻结 SL 对手，不能收敛成纯当前策略自博弈。
- [x] 探索按动作类别设置，禁止统一的 15% 合法动作均匀随机：
  - 换牌和弃牌使用温度或 top-k 采样；
  - 副露响应使用更低探索率；
  - 胡牌响应只保留极低但非零的探索。
- [x] 对过度重复的状态、动作和策略版本做降采样，同时保持九类决策的最低覆盖量。
- [x] 保存采集策略的真实动作概率，供离线诊断和可能的 importance weighting 使用。

## 3. 独立 Q1/Q2/V

- [x] 保持现有约 8.84M Actor 架构不变，新增参数量更小且完全独立的 `Q1`、`Q2`、`V` 网络。
- [x] Critic 不共享 Actor 编码器，避免 value 梯度破坏已经学到的 SL 表征。
- [x] `Q(s, a)` 对同一状态下所有合法动作批量输出；非法动作必须被 mask。
- [x] 主 Q 目标使用从动作时刻到终局的实际累计分差：

```text
G_t = terminal score delta from action t / 10_000
L_Q = Huber(Q1(s, a), G_t) + Huber(Q2(s, a), G_t)
```

- [x] `V` 使用 clipped Double-Q 中较保守的估计做 expectile regression：

```text
q = min(Q1(s, a), Q2(s, a))
L_V = |tau - 1[q - V < 0]| * (q - V)^2
```

- [x] 离线阶段只在合法动作集合上加入 CQL，抑制数据外动作的虚高 Q：

```text
L_CQL = logsumexp(Q(s, legal_actions)) - Q(s, behavior_action)
```

- [x] 在线 replay 中逐步衰减 CQL，但不要按 update 数机械归零；依据当前策略动作在 replay 中的覆盖度调整。
- [x] Critic 训练按早、中、晚局分别报告 loss、MAE、相关性和校准误差，避免开局大量接近零的目标掩盖中后盘质量。
- [x] 增加按动作类别的 Q 误差与 Double-Q disagreement，确认胡牌/副露等少数类没有被总体指标掩盖。

## 4. AWR Actor 提取

- [x] 使用保守优势更新 Actor：

```text
A(s, a) = min(Q1(s, a), Q2(s, a)) - V(s)
w = clip(exp(A / beta), max=w_max)
L_actor = -w * log pi(a | s) + lambda_ref * KL(pi || frozen_SL)
```

- [x] Actor 只学习 replay 中真实执行过的合法动作，不把未采样动作硬标为负样本。
- [x] 对 advantage、权重和有效样本量分动作类别记录分布，防止少量极端 Q 值控制更新。
- [x] 每轮 Actor 更新后评测相对冻结 SL 的 KL、固定种子平均名次和平均分差。
- [x] 若 Q disagreement 或离线校准未达标，只继续训练 Critic 和采集数据，不进行不可信的 Actor 更新。
- [x] Actor 更新后立即生成新轨迹，形成 `collect -> critic -> actor -> evaluate` 的持续循环。

## 5. Replay 与策略池

- [x] 初始离线 replay 永久保留一部分规则和冻结 SL 数据，作为覆盖锚点。
- [x] 在线 replay 使用滑动窗口，但按来源和决策类别设保底比例，不能仅保留最新 self-play。
- [x] 定期冻结 Actor 快照加入对手池；按最近、历史和参考策略分层采样，避免只对当前版本过拟合。
- [x] 对策略版本、对手组合和座位轮换做固定种子评测，区分真实提升与对手分布漂移。
- [x] checkpoint 保存 Actor、Q1/Q2/V、所有优化器、策略池清单、replay 游标、随机状态和数据格式版本。

## 6. 选择性蒙特卡洛教师

蒙特卡洛不是批量 replay 的默认来源。它作为显式选择的实验 C 存在，只有普通完整轨迹已经证明 Critic 能学习、但关键状态仍有高不确定性时才应启动。

一次典型查询的预算约为：

```text
3 candidate actions * 8-16 hidden worlds * 1-2 continuations
= 24-96 terminal rollouts per queried state
```

相比一条能为四个座位产生许多训练状态的普通完整轨迹，这至少贵一个数量级。不能在所有 replay 状态上默认执行。

引擎现已提供 `resample_information_set` 和批量 `resample_information_sets`：固定行动者手牌、公开副露、牌河、事件历史和已知牌，将剩余未知牌在对手暗手与牌墙之间重新分配，并通过测试验证采样前后的行动者 observation 完全一致。

- [x] 从高 Q disagreement、Actor/SL/规则动作分歧、优势接近零或中后盘高分差风险状态中选择查询。
- [x] 从行动者的信息集重新采样多个可能的对手暗手和剩余牌墙，不能反复续着同一个真实隐藏状态。
- [x] 每个候选动作在相同的一组隐藏世界上评估，并尽量使用成对的共同随机数续着，降低动作间比较方差。
- [x] 每个 `(state, action)` 保存均值、方差、样本数和置信区间；候选共享 hidden worlds，并用逐 world 回报差的 paired confidence interval 判断可靠边。
- [x] 只有 `abs(mean_difference) - confidence_half_width >= 0.02` 且 half-width 达标的动作边进入 teacher；裁掉没有可靠边的候选，持久化对称可靠边图。
- [x] 已接收的完整 MC query group 只额外更新 Q1/Q2：在可靠边上使用 action-difference Huber 加按目标差距加权的 pairwise softplus，不直接生成硬 argmax 标签；普通完整轨迹的绝对 Q、V、CQL 更新继续保留。
- [x] MC 的 absolute loss 只作为诊断，不进入 MC 优化目标；普通轨迹 Critic 负责绝对 return-to-go 校准。
- [x] train MC 与 validation MC 完全分离；validation 只查询永久 anchor validation 轨迹，仅在达到 512 targets、128 reliable groups 和 128 reliable pairs 后冻结，后续在线状态不再进入 validation teacher corpus，也不参与 train MC Q 更新。
- [x] 先测量每单位 GPU 秒带来的 Q 误差下降和策略提升，再决定是否扩大查询比例。
- [x] 只有当普通轨迹 Critic 已能学习、AWR 基线趋于饱和且剩余误差集中在高不确定状态时，才提高 MC 查询预算。

## 7. 可选 Oracle Critic

- [x] 在大规模 MC 之前试验训练期专用的 oracle Q/V：允许读取全局暗手和牌墙，但其输出不进入 Actor 推理输入。
- [x] 比较 oracle teacher 蒸馏、普通 partial-observation Critic 和选择性 MC 的成本与校准效果。
- [x] 明确阻断 oracle 特征进入 Actor；增加测试验证部署图只依赖行动者可见信息。
- [x] Oracle 产生的 advantage 只能通过跨大量隐藏状态的期望更新 Actor，不能把任何隐藏特征写入部署模型或 observation。

## 8. 验证顺序与验收指标

以下条目的验证流程、指标和保护条件均已实现。涉及模型强度的阈值由正式训练数据决定：Critic 未超过中晚盘常数基线时 Actor 会保持冻结；候选策略未同时改善固定与新增种子 panel 时不会覆盖 `best.pt`。

- [x] 先验证轨迹确定性重放和 return-to-go，使用手工小局测试每个座位的分差符号与总和。
- [x] 冻结 Actor，只训练 Q1/Q2/V；确认 validation Q 指标在中晚局和各动作类别上都优于常数基线。
- [x] 在固定离线数据上做一次 AWR，要求固定规则评测不劣于 SL，且策略变化与预测优势方向一致。
- [x] 开启在线循环，至少同时报告：
  - 固定规则对手平均名次、首位率、末位率和平均分差；
  - 对冻结 SL、历史策略和混合对手池的 head-to-head；
  - Q/V 校准、Double-Q disagreement、AWR 有效样本量；
  - 九类决策覆盖和各来源 replay 占比。
- [x] 将“普通轨迹 + IQL/AWR”作为基线，与“增加 oracle teacher”和“增加选择性 MC”做等 GPU 时间对比。
- [x] 只有在固定种子和新增种子评测都提升时才保存为 `best.pt`；训练本身仍持续到 `Ctrl+C`。

等 GPU 时间实验按以下顺序进行，不能同时堆叠后再猜测收益来源：

1. `A`：普通完整轨迹 + partial-observation Double-Q/IQL/AWR。
2. `B`：在 `A` 上增加 Oracle Critic 或蒸馏。
3. `C`：在 `A` 或胜出的 `B` 上增加选择性信息集 MC。

每组同时比较固定规则评测、混合策略池评测、Critic 校准误差、Q disagreement、每 GPU 秒有效训练状态数，以及相对 SL 的真实提升。MC 只有在等算力下带来更好的校准或策略成绩时才进入默认方案。

## 9. 文档与清理

- [x] 新训练器稳定后重写 `TRAINING.md` 和 `training/README.md`，删除反事实策略迭代说明。
- [x] 更新终端摘要和 dashboard，突出评测名次、Q 校准、disagreement、Actor KL、replay 构成和 GPU 吞吐。
- [x] 删除旧 counterfactual 模型、训练、测试和 checkpoint 兼容代码；不提供旧实验恢复路径。

## 10. A/B 实验后教师门控修复

实验 B 的前 50 个 Actor update 暴露出一个控制流错误：Actor 是否解冻只检查 partial Critic，但 B 随后无条件使用尚未证明优于 partial 的 Oracle advantage；同一个未验证 Oracle 还会从第一个 Critic step 起向 partial 蒸馏。C 原流程也会在积累 MC 教师数据之前更新 Actor。

- [x] Oracle 验证同时报告 Q1/Q2/V、早中晚盘、动作类别、disagreement、expectile balance，以及相对 partial 的整体和早盘 MAE gain。
- [x] B 只有在 Oracle 自身绝对校准达标、整体和早盘 MAE 至少相对 partial 改善 2%，并连续通过 3 次验证后才更新 Actor。
- [x] Oracle 未就绪时只独立训练，禁止向 partial Critic 蒸馏；任一次验证失败立即冻结 Actor、清零 streak，并在下一轮关闭蒸馏。
- [x] Oracle 蒸馏只覆盖 validation 同口径的 logged action，不把仅受 CQL 约束的未执行动作伪装成已验证教师标签。
- [x] C 在 Actor 冻结期间就开始积累独立 train/validation MC 目标，并在 Actor update 之前完成验证。
- [x] MC validation 按完整可靠边图采样，明确报告 train/validation accuracy、reliable/all pair 数、reliable group 数、top-action accuracy 和 regret；数量与质量门槛连续通过 3 次后才解冻 Actor。
- [x] MC query 持久化独立 group ID；候选按整组接收和淘汰，禁止把缺失候选或不同 paired-world 查询拼成排序证据。
- [x] validation 同时达到 targets、reliable groups 和 reliable pairs 三项门槛后才设置 `validation_frozen`；固定 anchor validation corpus，避免在线策略状态改变验证分布。
- [x] ordinary Q1/Q2/V/CQL 更新保留；MC pass 只对完整 query 的可靠边做 Q1/Q2 difference/pairwise 更新，并记录对应 loss、train/validation accuracy、reliable/all pair 和 reliable group 数量。
- [x] 删除与 difference-only MC objective 不一致的 absolute-Q improvement/correlation teacher gate；MC Q MAE 只作诊断，普通 Critic gate 负责绝对尺度。
- [x] checkpoint 和 iteration metrics 保存 `mc_critic.*`、`mc.*` 与 `mc.validation_metrics.action_ranking.*`，包括 validation freeze 状态、置信区间和 regret。
- [x] checkpoint 保存教师 readiness streak，格式升为 v3；明确拒绝 v1/v2，不做旧配置字段填充或迁移。
- [x] `latest.pt` 保留可恢复的完整状态；`best.pt` 改为 Actor-only，避免它与持续变化的 replay manifest 假装可恢复。
- [x] 删除 smoke 强行放行门控的分支，只用显式 smoke 配置覆盖各训练路径。
