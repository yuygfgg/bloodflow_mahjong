# 神经网络增强 `rule-planner` 的实施方案

本文定义后续神经网络工作的主线。该主线保留 Rust `rule-planner` 的信息集搜索，用神经网络逐步替换可学习的评分组件。该主线不恢复旧的 Python 端到端策略梯度训练。

本文统一使用以下术语：

- **信息集**：行动者根据自己的手牌、公开状态和公开事件历史能够区分的状态。
- **隐藏世界**：与当前信息集一致的一组对手暗手和未摸牌。
- **proposal**：在合法信息集中生成隐藏世界的基础分布。
- **posterior**：在 proposal 上加入公开动作历史后得到的后验分布。
- **selection particles**：用于提出根动作改动的隐藏世界。
- **validation particles**：用于独立验证根动作改动的隐藏世界。
- **teacher**：使用高计算预算的冻结 planner。
- **student**：只读取合法可见信息的神经网络策略。

## 决策

后续系统采用以下分工：

```text
Rust 游戏引擎
  -> 构造合法信息集 proposal
  -> 计算或调用 posterior scorer
  -> 在共享粒子上配对评估全部根动作
  -> 用独立粒子验证候选动作
  -> 输出动作和结构化训练记录

Python/PyTorch
  -> 读取 Rust 生成的数据
  -> 训练 policy、belief 和 value
  -> 执行离线评测
  -> 导出带版本清单的模型

Rust 模型运行时
  -> 批量执行模型
  -> 将模型作为 planner 的可选评分组件
```

Rust 始终负责游戏规则、合法动作、随机种子、信息边界和最终搜索决策。Python 不复制游戏状态机，也不自行重建合法牌局。

## 目标

该方案必须达到以下目标：

1. 神经网络必须改善现有 `rule-planner`，而不是创建一个与搜索无关的新端到端策略。
2. 最终策略必须在全新 seed 上直接对 `rule-fast` 达到至少 `+45 Elo-like`。
3. 正式验收以点估计至少 `+55 Elo-like`，且 95% 置信区间下界高于 `+45` 为目标。
4. 增加搜索预算不得产生可复现的显著退步。至少一个预算增量必须产生置信区间下界高于零的收益。
5. 策略必须在混合对手和未参与训练的策略快照上保持收益，不能只利用 `rule-fast` 的固定缺陷。
6. 模型必须只读取部署时可见的信息。任何隐藏信息只能用于 belief 粒子评分的候选世界输入或离线诊断。
7. 每个学习组件必须能够单独关闭。关闭全部学习组件时，行为必须回到当前 Rust planner。

## 非目标

第一阶段不执行以下工作：

- 不恢复旧 `live_wall` 单暗手目标生成。
- 不从终局回报直接执行 policy gradient。
- 不用 KL 投影、signed SGD 或跨数据批次的 Nesterov 动量约束策略更新。
- 不让神经网络判断动作是否合法。
- 不让 Python 成为权威模拟器。
- 不用神经网络替换根动作的独立粒子验证。
- 不要求第一版模型缩短思考时间。单步最多约 10 秒仍可接受。
- 不同时启用多个未经独立验证的模型组件。

## 旧训练失败机制

旧 Python 主线固定来源牌局中的四家暗手，只重排尚未摸出的牌墙。它实际在一个隐藏世界上训练；其他实验若对合法隐藏世界均匀采样，也没有利用公开历史改变世界权重。前者把单一暗手当成目标，后者把不符合历史的世界当成同等可能。两种目标都不是行动者在信息集内应当优化的后验期望。该方法可以降低同一暗手下的未来牌墙方差，但不能积分对手暗手的不确定性。一个 query 的动作优势仍然依赖来源牌局中偶然出现的对手暗手。

公开弃牌、碰杠和胡牌历史会改变不同暗手的相对概率。旧训练没有按这些事件计算公开历史后验，也没有用独立粒子流验证动作改动。因此，增加每个 query 的牌墙样本数只能更精确地估计错误的条件目标。该操作不能消除隐藏暗手造成的偏差，也不能保证增加预算会提升强度。

优化器和 KL 约束只能改变参数如何跟随目标。它们不能修复目标本身的信息集偏差。QPC sweep、SGD/AdamW sweep 和 KL scale sweep没有得到稳定收益，与这一判断一致。

当前 `rule-planner` 已经验证了更可靠的结构：

- 所有根动作在同一组隐藏世界上配对比较；
- 隐藏世界按公开动作历史加后验权重；
- selection 和 validation 使用不同的粒子流；
- 根动作只有通过独立验证后才能覆盖 baseline；
- 增加预算时，稳定的粒子编号保留已经计算的样本，并加入新样本。

神经网络必须进入该结构。神经网络不能绕过该结构。

## 信息边界

### Policy、rollout 和 root prior

这三个用途共享一个 observation-only policy。输入只包含当前行动座位可见的内容：

- 自己的手牌；
- 四家的公开副露、弃牌、定缺、分数和和牌状态；
- 当前阶段、剩余牌数和合法动作 mask；
- 观察者视角的公开事件历史。

模型输出 115 维 action logits。Rust 在模型输出后应用合法动作 mask，并按固定 action ID 规则处理并列值。模型不得读取对手暗手、真实牌墙顺序、对手策略标识符或 tournament 侧别。

rollout 中的每个座位必须使用该座位自己的 observation。模拟器不能用根行动者的 observation 替其他座位决策，也不能把完整隐藏世界直接编码为 policy 输入。

### Belief scorer

belief scorer 评估一个候选隐藏世界与公开历史的一致程度。输入包含：

- 根行动者的公开 observation 和事件历史；
- 候选世界中三个对手的暗手计数；
- 候选世界对应的公开约束。

belief scorer 不得读取未摸牌的顺序。公开历史对尚未发生的牌墙排列没有额外信息。让 scorer 读取牌墙顺序会引入不可泛化的噪声。

belief scorer 输出一个未归一化的 log density ratio。Rust 将该输出与当前手写 history likelihood 组合，再在同一根状态的粒子之间执行 log-sum-exp 归一化。

第一版 scorer 只学习当前手写模型的残差：

```text
log_weight(world) = handwritten_log_weight(world) + neural_residual(world)
```

训练损失把 `handwritten_log_weight` 作为固定 offset。该设计允许模型修正现有后验，而不会重复计算相同的 likelihood。

### Leaf value

leaf value 接收焦点座位的合法 observation。模型输出：

- 最终第 1 至第 4 名的概率；
- 焦点座位的预期终局分数。

Rust 使用当前牌局的守恒总分，把两个输出转换为与 terminal rollout 相同的 rank-first、score-second utility。第一版 leaf value 不读取其他座位暗手。

## 学习组件

### Policy head

policy head 学习高预算 teacher 的搜索改进结果。训练目标为合法动作上的交叉熵：

```text
L_policy = -sum_a pi_teacher(a | observation) * log pi_student(a | observation)
```

`pi_teacher` 来自后验加权、配对根搜索。`pi_teacher` 不是来源牌局真实动作，也不是单个隐藏世界中的最优动作。

teacher 对 posterior particle 执行带权 bootstrap。每次 bootstrap 计算不同 rollout policy 下的最坏期望收益，并记录最优动作。各动作的获胜频率构成 `pi_teacher`。如果独立 validation particles 不接受搜索改动，teacher target 回到当前 baseline 动作。

### Rollout policy

同一个 policy head 首先作为第三种 rollout policy 接入 planner。现有手写 rollout policy 保留。候选动作必须同时满足以下条件：

- 相对 baseline 改善每一种 rollout policy 的均值；
- pooled paired lower confidence bound 大于零；
- validation particles 的有效样本量满足门槛。

该阶段不改变无搜索 baseline。它先验证模型是否能够提供新的、有效的 continuation model。

### Root prior

policy logits 在下一阶段作为根动作 prior。prior 只控制动作评估顺序和额外预算分配。每个合法根动作必须先获得相同的最小配对样本数。prior 不能把合法动作永久剪枝。

增加预算时，planner 必须保留原粒子、原动作评估和原随机流。新增预算只增加粒子或 rollout。该前缀稳定性是计算量收益实验的前提。

### Belief head

belief head 使用 contrastive density-ratio estimation。对每个训练根状态：

1. 正样本是生成该公开历史的真实对手暗手分配。
2. 负样本来自 Rust 信息集 proposal。
3. 所有样本共享相同的根 observation 和公开历史。
4. InfoNCE logits 使用手写 posterior log weight 作为固定 offset，再加 neural residual。

真实隐藏分配只进入 belief 数据集。policy、value 和 root prior 数据集不得包含该字段。

训练对手池定义了被学习的混合 posterior。对手策略标识符只用于分层统计，不能作为模型特征。这样 scorer 必须从公开行为本身推断暗手倾向。

### Value head

value head 使用完整 Rust 续局的终局名次和分数监督。第一版 value 只执行以下用途：

- 为 rollout task 排序；
- 在 selection 阶段筛除明显较差的候选；
- 作为离线误差诊断。

独立 validation 阶段继续运行到终局。只有 leaf value 在全新对局上通过单独消融后，才可以让它替代 validation 的部分终局续局。

## Oracle 消融

训练模型前必须执行 oracle 消融。该实验确定后验误差和 continuation model 误差各自还有多少可利用空间。

| 实验 | 仅改变的内容 | 回答的问题 |
| --- | --- | --- |
| 当前 baseline | 手写 posterior、当前 rollout policy | 当前可部署强度 |
| Uniform belief | 所有合法粒子等权 | 手写 posterior 是否实际提供收益 |
| Oracle hidden allocation | 使用该对局的真实对手暗手，并独立重采样未来牌墙 | 更准确的暗手信息能否改善根决策 |
| Oracle continuation | rollout 使用生成对局的已知冻结策略 | continuation mismatch 是否限制搜索 |
| Joint oracle | 同时使用真实暗手和已知 continuation | 固定搜索结构的诊断上限 |

Oracle 实验可以读取评测器保存的隐藏信息和策略身份。Oracle 结果不能成为部署结果，也不能进入最终 Elo 报告。

每项消融使用相同的平衡 seed blocks 和座位排列。比较使用 paired block bootstrap。只有 95% 置信区间下界高于零，才把该组件视为有可学习空间。

如果 Joint oracle 相对当前 planner 没有稳定收益，则同一搜索结构中的神经网络不太可能达到目标。此时应先修改 planner 的状态或动作抽象，不应扩大模型和数据量。

## 数据生成

### Rust 数据生成器

新增独立 Rust 工具负责数据生成。该工具直接调用 core crate，不通过 Python 重放游戏。core crate 只提供结构化搜索诊断接口，不依赖 Arrow、PyTorch 或模型文件格式。

数据工具使用 Apache Parquet 分片。每个数据目录包含一个 JSON manifest。manifest 至少记录：

- engine rules version 和 Git commit；
- observation schema version；
- teacher 和 rollout policy 的模型摘要；
- 全部 planner 配置；
- 对手池及其采样权重；
- 根 seed domain、selection seed domain 和 validation seed domain；
- 分片文件的 SHA-256；
- train、development 和 final split 的 seed block 范围。

不允许用日志文本反向解析训练标签。

### Policy 记录

每条 policy 记录包含：

- 可见 observation、事件历史和合法动作 mask；
- 牌局 seed、动作前缀、座位、阶段和决策序号；
- baseline 动作和 teacher target；
- 每个合法动作在各 rollout policy 下的 posterior-weighted mean、standard error 和 paired gain；
- selection 与 validation 的原始粒子数、有效粒子数和 ESS；
- teacher 配置和数据来源策略族。

高预算 teacher 必须对每个合法根动作记录相对 baseline 的 posterior-weighted paired advantage。teacher 结果还必须包含最终选择、是否通过 validation、拒绝原因和每个 rollout policy 的比较结果。数据生成器不得只写入 teacher 选择的动作；否则 student 会把 teacher 的访问分布误认为状态分布。

同一完整牌局产生的所有根状态必须进入同一个数据 split。同一个平衡 6 局 seed block 也必须进入同一个 split。该规则防止相同暗手、座位旋转或公共轨迹同时出现在训练集和评测集。

### Belief 记录

belief 数据使用单独的 Parquet schema 和目录。每条记录包含：

- policy 记录中的公开输入；
- 真实对手暗手分配；
- proposal 生成的负样本暗手分配；
- 每个候选的手写 log weight；
- proposal seed 和候选编号。

belief 文件不得被 policy 或 value dataloader 打开。训练代码必须根据 manifest 中的 dataset kind 拒绝错误 schema。

### 数据来源

根状态必须来自以下混合对手池：

- `rule-fast`；
- 确定性 `rule-ev`；
- 当前 `rule-planner` 的低、中、高预算变体；
- student 的多个冻结历史快照；
- 具有 full support 的随机化策略变体。

四个座位和所有主要阶段必须平衡。训练集可以包含 `rule-fast`，但不能让它占据多数根状态。development panel 必须包含训练中未出现的 student 快照、planner 预算或随机化策略。

## DAgger 闭环

训练采用 DAgger，而不是只蒸馏 teacher 自己访问的状态。

1. 第 0 轮从当前 Rust 策略池收集根状态。
2. 高预算 teacher 为这些根状态生成 `pi_teacher`。
3. Python 使用 masked cross-entropy 训练 observation-only student。
4. 下一轮让冻结 student 实际控制轮换座位，并收集 student 访问的状态。
5. teacher 只负责给这些状态打标签，不替 student 修正对局轨迹。
6. 新数据与固定比例的历史数据合并，防止策略忘记较早阶段。
7. 每轮只接受一个冻结 student。未通过 development arena 的 student 不进入后续对手池。

数据合并比例、teacher 预算和训练 epoch 数必须在本轮开始前写入 manifest。development 结果不能反向修改已完成轮次的目标或数据 split。

初始 student 从新随机权重训练。旧 SL 模型只作为对照。任何旧权重 warm start 都必须作为独立消融，不能成为主线的隐式依赖。

## 分阶段实验矩阵

每个阶段只改变一个可部署组件。每个阶段都保留前一阶段模型、配置和原始结果，以便执行 paired rollback。阶段按以下顺序接入：第三种 rollout policy、无搜索 baseline、belief、leaf value、root prior。root prior 最后接入，因为它会改变计算分配；在此之前必须先验证模型本身和后验、续局评分各自的价值。

| 阶段 | 单一变量与明确产物 | 接受条件 | 停止条件 |
| --- | --- | --- | --- |
| 0：冻结基线 | 三档预算的 `rule-planner` commit、配置、吞吐、动作摘要、平衡 seed blocks | 相同 commit、配置和 seed 完全复现动作摘要和 tournament 原始结果 | 无法精确复现时，停止后续实验并修复随机性或记录缺口 |
| 1：Oracle 和数据接口 | 五组 oracle 结果；结构化 root evaluation；Parquet schema、manifest、checksum、exact replay test | 至少一个 oracle 相对 baseline 的 95% CI 下界高于零；Python observation 与 Rust replay 字节一致 | Joint oracle 无稳定收益时，停止模型训练，先修改 planner 状态或动作抽象 |
| 2：DAgger policy | 高预算 teacher advantage 数据；至少两轮 observation-only DAgger；ONNX policy artifact | held-out opponent panel 上相对无搜索 baseline 不显著退步，并达到预注册的 teacher regret 门槛 | 两轮后无改善或出现信息边界违规时，停止并检查 teacher 数据和 observation schema |
| 3：第三种 rollout policy | 冻结 student 作为新增 rollout policy；原手写 rollout 保留 | 相对原 planner 的 Elo 95% CI 下界高于零；override rate、validation reject rate 和 ESS 无异常 | CI 下界不高于零，或收益只出现于单一对手时，移除该 rollout policy |
| 4：无搜索 baseline | student 单独替换无搜索 baseline 的 tournament 和 mixed panel 报告 | 不弱于原 baseline，且对 held-out mixed panel 无显著退步 | 模型仅对 `rule-fast` 有收益时，不进入后续组件阶段 |
| 5：Belief residual | density-ratio artifact；posterior calibration、ESS 与 tournament 报告 | 独立粒子上提高真实隐藏分配排序，且 tournament 95% CI 下界高于零 | 低 ESS fallback 成为主要路径，或 calibration 恶化时，停止接入 |
| 6：Leaf value | rank/score artifact；相对 terminal rollout 的排序和核时报告 | 相同核时提高 Elo，或相同决策质量减少核时；validation 仍使用 terminal rollout | 仅离线误差降低时，不接入 planner |
| 7：Root prior | policy prior 的动作排序和额外预算分配；最小配对预算保持不变 | 提高单位核时收益，并且中预算搜索稳定强于模型 baseline | 任一合法动作因 prior 未获最小样本，或预算扩展退步时，撤回 prior |

## 独立粒子验证和 ESS

selection 与 validation 必须使用不同 seed domain。两个粒子流不能共享隐藏世界、future wall 或 bootstrap 随机数。

同一粒子流内，全部根动作必须共享相同的隐藏世界和 rollout policy。收益估计必须使用动作间 paired difference。不同动作不能各自采样一组世界。

归一化权重为 `w_i` 时，有效样本量定义为：

```text
ESS = 1 / sum_i(w_i * w_i)
```

每次搜索必须记录：

- requested particles；
- valid particles；
- ESS；
- maximum normalized weight；
- weight entropy；
- selection 和 validation 的 paired mean、standard error 和 LCB。

learned belief 可能使权重集中。Rust 先按稳定粒子编号增加粒子，直到 ESS 达到预注册门槛或预算耗尽。如果预算耗尽后 ESS 仍不足，planner 不接受 neural override，并回到当前手写 posterior 或 baseline。正式结果必须报告 fallback 比例，不能把 fallback 隐藏为模型成功。

ESS 门槛、最大粒子数和 belief temperature 只能在 development split 上确定。final tournament 期间不得修改这些参数。

## 模型结构和导出

第一版模型使用无状态 encoder。输入宽度固定，只有 batch 维动态。event history 使用固定容量和显式 length，不使用 Python 对象或可变控制流。

policy 和 value 可以共享 observation encoder。belief 使用独立的 particle encoder，并可共享 public-history encoder。第一阶段保持两个独立 artifact，避免 belief 输入进入 policy graph。

Python 导出 ONNX opset 18。每个 artifact 目录包含：

- `model.onnx`；
- `manifest.json`；
- 固定输入和输出的 golden vectors；
- 文件 SHA-256。

manifest 必须记录：

- model kind；
- observation schema version；
- engine rules version；
- action space size；
- 输入名称、dtype 和 shape；
- 输出语义；
- 训练数据摘要；
- PyTorch 和 ONNX exporter 版本。

Rust 不加载 pickle 或 Python checkpoint。Rust 运行时在加载模型时验证全部版本、shape 和摘要。版本不匹配必须返回明确错误。正式评测不能静默回退到手写策略。

模型后端放在独立 crate。core crate 只定义批量 evaluator trait 和纯 Rust feature schema。这样 ONNX runtime 依赖不会进入权威游戏状态机。

## Rust 批量推理

神经 rollout 不能在每个 Rayon task 中单独执行一次模型。搜索调度器按 simulation step 批量推进 active rollouts：

1. 收集当前需要策略动作的全部模拟状态。
2. 按 model kind 和输入 shape 组成连续 batch。
3. 执行一次 policy inference。
4. 由 Rust 应用合法 mask，并推进全部模拟状态。
5. 重复以上步骤，直到 rollout 结束或到达 leaf。

belief scorer 在根状态一次性批量评估全部 particles。leaf value 一次性批量评估同一搜索层的全部 leaves。

batch 大小只改变调度效率，不能改变 logits、随机种子或动作 tie-break。1 线程和多线程执行必须产生相同的动作摘要。golden vectors 必须覆盖以下内容：

- 最短和最长 event history；
- 每种决策 phase；
- 含非连续合法 action ID 的 mask；
- 最大 batch；
- policy、belief 和 value 的非有限输出检查。

## 离线指标

离线指标用于发现错误，不能替代整局 tournament。

Policy 至少报告：

- masked cross-entropy；
- teacher top-1 agreement；
- teacher advantage-weighted regret；
- 按 phase、座位和对手族分层的指标。

Belief 至少报告：

- held-out InfoNCE loss；
- 真实隐藏分配在候选集合中的 rank；
- ESS、maximum weight 和 entropy 分布；
- 对下一次公开弃牌、碰杠或胡牌事件的 posterior predictive likelihood。

Value 至少报告：

- rank negative log-likelihood 和 Brier score；
- score mean absolute error；
- 按剩余牌数和焦点座位是否已胡分层的 residual；
- 相对完整 terminal rollout 的动作排序错误率。

## 防止对 `rule-fast` 过拟合

实验使用三层数据和评测边界：

1. **Train**：允许包含 `rule-fast`，用于训练模型。
2. **Development**：使用全新 seed、未见 student 快照和未见 planner 预算，用于组件选择。
3. **Final**：预先冻结的全新平衡 seed blocks，只用于正式验收。

每次 development arena 同时报告：

- 对 `rule-fast`；
- 对确定性 `rule-ev`；
- 对当前 planner；
- 对 planner 不同预算变体；
- 对多个 student 历史快照；
- 混合四人阵容。

模型选择不能只读取对 `rule-fast` 的单项 Elo。若模型只对 `rule-fast` 提升，但在 held-out mixed panel 上显著退步，该模型不得进入下一 DAgger 轮。

final seed blocks 一旦产生结果，本轮模型、搜索预算和超参数即冻结。不得根据 final 结果调参后重复使用相同 blocks。

## 正式验收

正式验收使用 `rule-tournament` 的平衡二对二协议。所有比较必须固定代码 commit、模型摘要、planner 配置、线程数规则和 bootstrap 方法。

最终报告至少包含以下比较：

- neural planner 对 `rule-fast`；
- neural planner 对当前无模型 planner；
- neural planner 对确定性 `rule-ev`；
- neural planner 的低、中、高三个计算预算；
- neural planner 在 mixed opponent panel 中的相对强度。

主目标通过条件为：

```text
neural planner vs rule_fast
point estimate >= +55 Elo-like
95% CI lower bound > +45 Elo-like
```

预算扩展通过条件为：

- 高预算相对中预算没有显著退步；
- 中预算相对低预算没有显著退步；
- 至少一个相邻预算比较的 95% 置信区间下界高于零；
- 增加预算时 ESS 和完成 rollout 数按预期增加；
- 提升不能只来自更多 fallback 或更少 search override。

在达到预注册局数前不得根据中途 Elo 停止失败实验。若采用 sequential stopping，必须在实验前固定边界，并把规则写入 manifest。

## 实施顺序

按以下顺序提交代码：

1. 冻结当前 planner 基线和三档预算。
2. 为 planner 增加结构化 root evaluation，不改变动作。
3. 实现 oracle evaluator 和 paired tournament。
4. 新增 Rust 数据生成工具、Parquet schema 和 replay tests。
5. 实现 Python policy trainer、DAgger driver 和 ONNX export。
6. 新增独立 Rust 模型运行时和 batch evaluator trait。
7. 将 policy 接为第三种 rollout policy。
8. 独立测试模型替换 planner 无搜索 baseline 的效果。
9. 实现 belief residual 数据和训练。
10. 实现 leaf value。
11. 最后接入 root prior 和预算分配。

每一步都必须保持无模型 planner 可构建、可测试、可测评。任何组件没有得到独立正收益时，后续阶段不得把该组件当作默认依赖。

## 明确废弃的路径

本方案不复用旧 `live_wall` 单暗手 PG/CE 路径。旧代码和文档可以保留用于复现实验，但它们不定义新模型的数据或优化目标。

新方案中的交叉熵是对高预算信息集搜索 target 的监督蒸馏。它与“在一个固定对手暗手上重排牌墙，再把动作收益直接压入 Actor”的旧 CE 不同。新方案也不使用终局 REINFORCE、单步 signed SGD 或强制 KL 归一化来补偿目标噪声。

若后续需要在线微调，在线样本仍必须通过 posterior particles、paired root actions 和独立 validation。任何绕过这些约束的实验都只能作为独立研究分支，不能进入 planner 主线。
