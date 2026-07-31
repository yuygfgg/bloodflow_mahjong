# 神经网络增强 `rule-planner` 的实施方案

本文定义后续神经网络工作的主线。该主线保留 Rust `rule-planner` 的信息集搜索，用神经网络逐步替换可学习的评分组件。

本文统一使用以下术语：

- **信息集**：行动者根据自己的手牌、公开状态和公开事件历史能够区分的状态。
- **隐藏世界**：与当前信息集一致的一组对手暗手和未摸牌。
- **proposal**：在合法信息集中生成隐藏世界的基础分布。
- **posterior**：在 proposal 上加入公开动作历史后得到的后验分布。
- **selection particles**：用于提出根动作改动的隐藏世界。
- **validation particles**：用于独立验证根动作改动的隐藏世界。
- **teacher**：使用高计算预算的冻结 planner。
- **student**：只读取合法可见信息的神经网络策略。
- **artifact**：可部署模型目录。目录包含模型权重、版本清单和 golden 校验样本。
- **残差**：在现有手写后验对数权重上增加的神经网络输出。

本文使用以下固定译法。命令行和代码中的英文名称保持不变：

| 英文名称 | 本文译法 | 含义 |
| --- | --- | --- |
| root | 根节点 | 当前行动前、需要比较候选动作的牌局状态 |
| hidden world | 隐藏世界 | 与当前信息集一致的对手暗手和剩余牌组成 |
| posterior | 后验 | 加入公开动作历史后的隐藏世界权重 |
| selection | 选择粒子流 | 用于提出根动作改动的粒子流 |
| validation | 验证粒子流 | 用于独立确认根动作改动的粒子流 |
| ESS | 有效样本量 | 衡量归一化粒子权重是否集中 |
| artifact | 模型包 | 包含权重、清单和 golden 校验样本的目录 |
| golden | 参考向量 | 用于检查 Python 与 Rust 数值一致性的固定输入输出 |

## 当前实现边界

截至 2026-07-31，代码只实现 learned belief residual 的首版链路。policy、rollout、leaf value、root prior 和 DAgger 仍属于长期方案，不应被描述为已完成。

- 事件历史容量固定为 `192` 条。每条事件使用 Rust 定义的 8 个字段；`event_len` 标记有效前缀，padding 不表示真实事件。
- 训练入口是 `learning.belief.train`。训练使用 PyTorch 和分组的 contrastive density-ratio objective。默认优化器是 AdamW。训练输出不是 Python checkpoint，而是 `safetensors` 权重和固定 golden 样本。
- `engine/model-runtime` 使用 Candle 在 CPU 上执行同一网络。首版不使用 ONNX、TorchScript、Python callback 或 GPU Rust 推理。
- 模型只修正 planner 根节点的隐藏世界后验。对候选世界的手写 log weight 加上 `beta * neural_residual`，然后由 Rust 在粒子之间重新归一化。模型不决定合法动作，不替换 rollout continuation，不改变 root action prior，也不替换独立验证。
- 选择粒子流和验证粒子流先分别计算带残差的权重和 ESS。任一粒子流的 ESS 小于 `8` 时，两条粒子流都原子地回退到手写 posterior；不能让一条流使用神经权重而另一条流使用手写权重。锦标赛会报告 residual decision 和 ESS fallback 计数。
- 模型包加载必须通过 `manifest.json`、模型 SHA-256、golden SHA-256、参考向量 shape/finite 检查和 Candle golden 数值校验。校验失败时加载失败，不静默回退。

首版链路已经完成端到端冒烟，但 pilot 数据量和对局数不足以证明 Elo 提升。任何强度结论必须来自独立 seed 的正式锦标赛。

## 长期架构决策

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

当前代码只启用下文架构中的 belief residual 分支。其余分支仍按分阶段实验矩阵推进。

## 目标

该方案必须达到以下目标：

1. 神经网络必须作为可关闭的评分组件改善现有 `rule-planner`。
2. 最终策略必须在全新 seed 上直接对 `rule-fast` 达到至少 `+45 Elo-like`。
3. 正式验收以点估计至少 `+55 Elo-like`，且 95% 置信区间下界高于 `+45` 为目标。
4. 增加搜索预算不得产生可复现的显著退步。至少一个预算增量必须产生置信区间下界高于零的收益。
5. 策略必须在混合对手和未参与训练的策略快照上保持收益，不能只利用 `rule-fast` 的固定缺陷。
6. 模型必须只读取部署时可见的信息。任何隐藏信息只能用于 belief 粒子评分的候选世界输入或离线诊断。
7. 每个学习组件必须能够单独关闭。关闭全部学习组件时，行为必须回到当前 Rust planner。

## 非目标

第一阶段不执行以下工作：

- 不从终局回报直接执行 policy gradient。
- 不让神经网络判断动作是否合法。
- 不让 Python 成为权威模拟器。
- 不用神经网络替换根动作的独立粒子验证。
- 不要求第一版模型缩短思考时间。单步最多约 10 秒仍可接受。
- 不同时启用多个未经独立验证的模型组件。

## 信息集目标

训练目标必须对当前信息集内的隐藏世界后验求期望。固定一个隐藏世界会把来源牌局中的偶然暗手写入动作优势；对所有合法隐藏世界均匀加权则会忽略公开历史。两种目标都会产生不可泛化的条件偏差。

公开弃牌、碰杠、杠和胡牌历史会改变不同暗手的相对概率。模型必须按这些事件修正 proposal，并在独立粒子流上验证根动作改动。增加同一错误目标的采样预算只能降低估计方差，不能修复后验偏差。

优化算法只决定模型如何拟合目标。优化算法不能替代正确的信息集、后验和独立验证结构。

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

首版 observation 的事件历史长度固定为 `192`。Rust 始终传递固定形状；短历史用 padding 填充，并单独传递 `event_len`。训练和 Candle 推理必须使用相同的容量、字段顺序和 padding 语义。

belief scorer 不得读取未摸牌的顺序。公开历史对尚未发生的牌墙排列没有额外信息。让 scorer 读取牌墙顺序会引入不可泛化的噪声。

belief scorer 输出一个未归一化的 log density ratio。Rust 将该输出与当前手写 history likelihood 组合，再在同一根状态的粒子之间执行 log-sum-exp 归一化。

首版 scorer 只学习当前手写模型的残差：

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

锦标赛通过 planner 参数 `root-belief` 提供 belief 消融。`posterior` 使用当前公开历史后验；`uniform` 对合法隐藏世界等权；`oracle-hidden` 固定真实暗手和剩余牌组成，但独立重排未来牌墙。`oracle-hidden` 测量完美当前暗手信息的价值，不是公开历史可学习信息的上界。

参数 `continuation` 提供 continuation 消融。`current` 使用生产路径中的两个代理模型。`oracle-continuation` 根据本局 seat mask 注入已知冻结策略。Fast 和 EV 保留原配置。planner 使用关闭 paired root search 的 baseline 配置，避免递归搜索，并保持一次策略改进的固定 continuation。每个模拟座位只使用自己的 observation。该模式知道策略身份，但不会把一个座位的暗手交给另一个座位的策略。

Current 使用 `posterior + current`。Oracle continuation 使用 `posterior + oracle-continuation`。Joint oracle 使用 `oracle-hidden + oracle-continuation`。这些模式共享相同的 selection、validation、终局 utility 和独立粒子流。

### 首轮 Oracle 诊断（2026-07-31）

belief 消融使用 `h0 d1 c1 b64 r0 i64`。每项比较运行 8 个平衡 block，共 48 局和每侧 96 个 seat-game。置信区间使用 10,000 次 paired block bootstrap。局级并行固定为 2；并行度 4 的吞吐更低。

下表只比较手写 `posterior`、`uniform` 和读取真实暗手的 `oracle-hidden`。它不使用 learned artifact，因此不能作为 learned belief residual 的 Elo 结果。

| 比较 | root seed | Elo-like delta | 95% CI | P(stronger) |
| --- | ---: | ---: | ---: | ---: |
| `oracle-hidden` 对 `posterior` | 20260731 | +61.54 | [+24.23, +110.07] | 1.0000 |
| `oracle-hidden` 对 `posterior` | 20260801 | +53.96 | [+7.56, +112.60] | 0.9917 |
| `posterior` 对 `uniform` | 20260731 | +6.56 | [+0.00, +15.13] | 0.9493 |

第二组 Oracle 复现中，`oracle-hidden` 接受 65/656 次根搜索改动，`posterior` 接受 29/694 次。Oracle 同时增加 proposal 数和 validation 通过率。该结果说明搜索结构对更准确的隐藏牌后验有可利用空间；它不证明公开历史模型能够恢复该差距。

`posterior` 相对 `uniform` 只有小幅收益。`posterior` 接受 36/622 次改动，`uniform` 接受 47/632 次。当前公开历史模型主要过滤高方差改动，但该结果仍需独立 seed 复现。

`oracle-hidden` 知道真实暗手。该结果不证明公开历史能够恢复全部信息，也不证明神经 belief scorer 已经有效。下一步必须训练只读取公开历史的 density-ratio residual，并在独立粒子和独立 seed 上测量它能恢复多少 Oracle 差距。

continuation oracle 已实现，但精确 planner baseline 续局不适合直接做整局 Elo。即使使用 `h0 d1 c1 b0 r0 i8`，单 block 探针在 95 秒后仍未完成首个昂贵外层搜索决策。`b64 i64` 会更慢几个数量级。后续 continuation 实验必须先提供批量 student 推理或等价缓存；不能继续缩小搜索预算并把失真的结果解释为 continuation 上界。

每项消融使用相同的平衡 seed blocks 和座位排列。比较使用 paired block bootstrap。只有 95% 置信区间下界高于零，才把该组件视为有可学习空间。

如果 Joint oracle 相对当前 planner 没有稳定收益，则同一搜索结构中的神经网络不太可能达到目标。此时应先修改 planner 的状态或动作抽象，不应扩大模型和数据量。

## 数据生成

### Rust 数据生成器

`engine/tools/belief-dataset` 直接调用 core crate，不通过 Python 重放游戏。该工具只生成首版 belief residual 数据。它不生成 policy、value 或 teacher advantage 标签。

工具输出 `safetensors` 分片和 `manifest.json`。manifest 记录 belief schema version、engine rules version、候选数、`max_history=192`、根 seed、随机动作概率、数据来源策略族、请求的最小根状态数、实际根状态数、牌局数、数据审计结果和每个分片的 SHA-256。每个 block 包含 4 局。4 种策略在每个座位各出现一次。split 以完整 block 为单位分配；同一 block 不会跨 train、calibration 和 development。当前实现按 8:1:1 划分这三个 split，尚未生成 final split。

生成器和训练器只接受不存在或为空的输出目录。它们拒绝复用包含其他文件的目录。训练器在构造数据集时校验每个分片的 SHA-256；每个 epoch 不重复计算摘要。

每个根状态包含 1 个权威隐藏世界和两条独立的 Rust proposal 流。每条流有 `K` 个候选，最终候选组大小为 `2K+1`。候选顺序随机打乱，`proposal_streams` 保存每个候选属于 selection、validation 或 truth 的标记。训练标签允许多个与权威隐藏世界相同的候选，`positive_mask` 必须标记全部且仅标记这些匹配项。`positive_collisions` 统计除权威样本外、与权威隐藏世界完全相同的额外候选数；它不等于所有 proposal collision 的数量。Rust 生成器和 Python 训练审计使用相同定义。数据工具同时保存每个候选的手写 log weight、公开输入、`block_id` 和 `root_id`。

不允许用日志文本反向解析训练标签。后续 policy 和 value 数据也必须使用结构化分片和显式 schema。

### 计划中的 Policy 记录

每条 policy 记录包含：

- 可见 observation、事件历史和合法动作 mask；
- 牌局 seed、动作前缀、座位、阶段和决策序号；
- baseline 动作和 teacher target；
- 每个合法动作在各 rollout policy 下的 posterior-weighted mean、standard error 和 paired gain；
- selection 与 validation 的原始粒子数、有效粒子数和 ESS；
- teacher 配置和数据来源策略族。

高预算 teacher 必须对每个合法根动作记录相对 baseline 的 posterior-weighted paired advantage。teacher 结果还必须包含最终选择、是否通过 validation、拒绝原因和每个 rollout policy 的比较结果。数据生成器不得只写入 teacher 选择的动作；否则 student 会把 teacher 的访问分布误认为状态分布。

同一牌局产生的所有已记录根状态必须进入同一个数据 split。同一个平衡 4 局策略分配 block 也必须进入同一个 split。该规则防止同一公共轨迹跨 split，并保持每个 split 的策略和座位分配平衡。4 局使用独立 game seed，不复用暗手或牌山。

### 已实现的 Belief 记录

belief 数据使用独立的 `safetensors` schema 和目录。每个分片包含以下张量：

- `tile_obs`、`melds`、`river` 和 `meta`；
- 固定形状为 `[roots, 192, 8]` 的 `events`，以及 `event_lengths`；
- 固定形状为 `[roots, 2K+1, 4, 27]` 的 `candidate_worlds`；
- `handwritten_log_weights`、`positive_mask` 和 `proposal_streams`；
- `block_ids` 和 `root_ids`。

候选世界的 4 个平面依次表示 3 家对手暗手计数和剩余牌组成。它不包含未来摸牌顺序。belief 文件不得被未来的 policy 或 value dataloader 打开。

### 数据来源

首版生成器轮换以下四种数据来源策略：

- `rule-fast`；
- 确定性 `rule-ev`；
- 低预算 `rule-planner`；
- 带 full-support 随机动作的 `rule-fast`。

默认随机动作概率是 `0.15`。该对手池只用于验证首版链路。它不是最终训练分布。

长期数据应来自以下混合对手池：

- `rule-fast`；
- 确定性 `rule-ev`；
- 当前 `rule-planner` 的低、中、高预算变体；
- student 的多个冻结历史快照；
- 具有 full support 的随机化策略变体。

四个座位和所有主要阶段必须平衡。训练集可以包含 `rule-fast`，但不能让它占据多数根状态。development panel 必须包含训练中未出现的 student 快照、planner 预算或随机化策略。

### 复现实验

数据生成、训练、模型校验和锦标赛命令统一维护在 [`BELIEF_RESIDUAL.md`](BELIEF_RESIDUAL.md)。本文件只定义架构、信息边界和验收标准，避免同一操作流程出现多个版本。

## DAgger 闭环

后续 policy 阶段计划采用 DAgger，使 teacher 不只标注自身访问的状态。

1. 第 0 轮从当前 Rust 策略池收集根状态。
2. 高预算 teacher 为这些根状态生成 `pi_teacher`。
3. Python 使用 masked cross-entropy 训练 observation-only student。
4. 下一轮让冻结 student 实际控制轮换座位，并收集 student 访问的状态。
5. teacher 只负责给这些状态打标签，不替 student 修正对局轨迹。
6. 新数据与固定比例的历史数据合并，防止策略忘记较早阶段。
7. 每轮只接受一个冻结 student。未通过 development arena 的 student 不进入后续对手池。

数据合并比例、teacher 预算和训练 epoch 数必须在本轮开始前写入 manifest。development 结果不能反向修改已完成轮次的目标或数据 split。

初始 student 从新随机权重训练。任何已有权重 warm start 都必须作为独立消融，不能成为主线的隐式依赖。

## 分阶段实验矩阵

每个阶段只改变一个可部署组件。每个阶段都保留前一阶段模型、配置和原始结果，以便执行 paired rollback。阶段按以下顺序接入：第三种 rollout policy、无搜索 baseline、belief、leaf value、root prior。root prior 最后接入，因为它会改变计算分配；在此之前必须先验证模型本身和后验、续局评分各自的价值。

| 阶段 | 单一变量与明确产物 | 接受条件 | 停止条件 |
| --- | --- | --- | --- |
| 0：冻结基线 | 三档预算的 `rule-planner` commit、配置、吞吐、动作摘要、平衡 seed blocks | 相同 commit、配置和 seed 完全复现动作摘要和 tournament 原始结果 | 无法精确复现时，停止后续实验并修复随机性或记录缺口 |
| 1：Oracle 和数据接口 | 五组 oracle 结果；结构化 root evaluation；`safetensors` schema、manifest、checksum、exact replay test | 至少一个 oracle 相对 baseline 的 95% CI 下界高于零；Python observation 与 Rust replay 字节一致 | Joint oracle 无稳定收益时，停止模型训练，先修改 planner 状态或动作抽象 |
| 2：DAgger policy | 高预算 teacher advantage 数据；至少两轮 observation-only DAgger；可由 Rust 加载的 policy artifact | held-out opponent panel 上相对无搜索 baseline 不显著退步，并达到预注册的 teacher regret 门槛 | 两轮后无改善或出现信息边界违规时，停止并检查 teacher 数据和 observation schema |
| 3：第三种 rollout policy | 冻结 student 作为新增 rollout policy；原手写 rollout 保留 | 相对原 planner 的 Elo 95% CI 下界高于零；override rate、validation reject rate 和 ESS 无异常 | CI 下界不高于零，或收益只出现于单一对手时，移除该 rollout policy |
| 4：无搜索 baseline | student 单独替换无搜索 baseline 的 tournament 和 mixed panel 报告 | 不弱于原 baseline，且对 held-out mixed panel 无显著退步 | 模型仅对 `rule-fast` 有收益时，不进入后续组件阶段 |
| 5：Belief residual | PyTorch 训练的 `safetensors` artifact；Candle golden；posterior calibration、ESS 与 tournament 报告 | 独立粒子上提高真实隐藏分配排序；独立 seed tournament 的 95% CI 下界高于零；fallback 率不异常 | 低 ESS fallback 成为主要路径，或 calibration 恶化时，停止接入 |
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

首版 learned belief 不动态增加粒子。Rust 对 selection 和 validation 各自计算带残差的归一化权重。没有任何有限手写权重的 learned 粒子流，其 ESS 定义为 `0`。只要任一粒子流的 ESS 小于 `8`，两条粒子流就原子地回退到 manual planner，避免 selection 和 validation 使用不同的目标。

learned ESS 只决定 residual 是否生效。原子回退后，manual planner 继续使用已有的 posterior 行为。如果 manual posterior 对一条合法粒子流仍然没有支持，manual planner 保留既有的均匀应急策略。该应急策略不改变 learned ESS，不计作 residual 成功，也不能把零支持报告为 learned 高 ESS。正式结果必须报告 residual decision 数和 ESS fallback 比例。

后续版本可以研究按稳定粒子编号扩展预算，但必须先预注册 ESS 门槛、最大粒子数和 belief temperature。final tournament 期间不得修改这些参数。

## 模型结构和导出

第一版模型使用无状态 encoder。输入宽度固定，只有 batch 维动态。事件历史固定为 `192`，并使用显式 `event_len`。模型不使用 Python 对象或可变控制流。

policy 和 value 可以共享 observation encoder。belief 使用独立的 particle encoder，并可共享 public-history encoder。第一阶段保持两个独立 artifact，避免 belief 输入进入 policy graph。

### 首版 belief artifact

Python/PyTorch 导出 `safetensors` 权重。每个 artifact 目录包含：

- `model.safetensors`；
- `golden.safetensors`；
- `manifest.json`；
- 模型和 golden 文件的 SHA-256。

manifest 必须记录：

- `artifact_version` 和 `model_kind=belief_residual`；
- `belief_schema_version`、`engine_rules_version` 和 `max_history=192`；
- 候选世界平面数、牌种数和完整网络配置；
- `beta`、`model_sha256` 和 `golden_sha256`；
- 训练数据路径、训练 seed、calibration 指标、development 指标和训练审计。

Rust 不加载 pickle 或 Python checkpoint。`CandleBeliefModel::load` 在构造运行时前验证 manifest 的版本、规则版本、网络 shape、有限 beta 和 RoPE 配置，再验证模型 SHA-256、golden SHA-256、golden tensor shape、有限值和 CPU 输出误差。`belief-model-check` 使用同一套检查。版本或校验不匹配必须返回明确错误；正式评测不能静默回退到手写策略。

模型后端放在独立 crate。core crate 只定义批量 evaluator trait 和纯 Rust feature schema。这样 Candle 依赖不会进入权威游戏状态机。首版只提供 CPU 推理；未来 policy/value artifact 可以复用该边界，但不能假定已经支持。

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

实验使用四层数据和评测边界：

1. **Train**：允许包含 `rule-fast`，用于训练模型。
2. **Calibration**：使用完整且不重叠的策略平衡 block，用于选择 `beta` 和其他预注册的部署参数。
3. **Development**：使用全新 seed、未见 student 快照和未见 planner 预算。训练器只在 artifact 固定后汇报该 split；结果不得反馈到同一个 artifact 的权重或 `beta`。
4. **Final**：预先冻结的全新平衡 seed blocks，只用于正式验收。

每次 development arena 在 artifact 固定后同时报告：

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

长期实施顺序如下。Oracle、belief 数据生成、belief residual 训练和 Candle runtime 已有首版实现；policy、value、DAgger 和 root prior 仍未实现。

1. 冻结当前 planner 基线和三档预算。
2. 为 planner 增加结构化 root evaluation，不改变动作。
3. 实现 oracle evaluator 和 paired tournament。
4. 新增 Rust 数据生成工具、`safetensors` schema 和 replay tests。
5. 实现 Python policy trainer、DAgger driver 和可由 Rust 加载的 artifact。
6. 新增独立 Rust Candle 模型运行时和 batch evaluator trait。
7. 将 policy 接为第三种 rollout policy。
8. 独立测试模型替换 planner 无搜索 baseline 的效果。
9. 扩展 belief residual 数据、训练和独立 seed 验收。
10. 实现 leaf value。
11. 最后接入 root prior 和预算分配。

每一步都必须保持无模型 planner 可构建、可测试、可测评。任何组件没有得到独立正收益时，后续阶段不得把该组件当作默认依赖。

## 在线微调约束

交叉熵目标用于监督蒸馏高预算信息集搜索。目标必须包含隐藏世界后验、配对根动作收益和独立验证结果。

若后续需要在线微调，在线样本仍必须通过 posterior particles、paired root actions 和独立 validation。任何绕过这些约束的实验都只能作为独立研究分支，不能进入 planner 主线。
