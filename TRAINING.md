# 保守策略迭代训练

`python -m training.train` 是当前唯一的正式训练入口。它从已有的 SL Actor 开始，在每个策略版本上重新采集独立状态、估计所有合法动作的配对反事实回报，然后只执行一次受 KL 约束的全批量更新。

```text
冻结当前 Actor，并从历史池轮换一个独立 opponent snapshot
  -> 按 curriculum 对规则 / 历史 opponent 采集全新完整对局
  -> 九类决策各选 256 个独立隐藏牌局
  -> 每个状态在 16 个配对 live-wall 续局中评估全部合法动作
  -> 累积完整 2304 状态的梯度并执行一次 AdamW.step()
  -> 用独立状态把更新方向缩放到 visitation-weighted KL = 0.001
  -> 在固定 16384 局规则面板上与 SL 做配对评测
  -> 原子提交 checkpoint，进入下一个策略版本
```

训练只支持 CUDA，使用 eager mode 和 BF16 autocast，不调用 `torch.compile`。正式运行没有最大 update 数，也没有自动停止条件，会一直运行到用户按 `Ctrl+C`。

## 为什么采用这条路径

麻将单局终局回报的方差很大，同一个可见状态还对应许多不同的对手暗手和牌墙。小批量在线更新容易把隐藏状态偶然性当成动作优势。当前流程针对这个问题施加四个约束：

- 一个训练状态至多来自一个完整隐藏牌局，避免把同一局的多个相关决策伪装成独立样本；
- 九类决策都有硬覆盖，每轮默认 `9 x 256 = 2304` 个状态；
- 同一状态的全部合法动作共享相同的 16 个未来牌墙排列，直接降低动作差值方差；
- 完整批量只产生一个优化器方向，再在独立校准集上缩放到很小的 KL，不对同一批数据做多轮拟合。

这里固定的是每轮状态数量，不是状态内容。每个策略版本都会重新生成 source games、训练状态、世界 seed 和校准状态。完整 update 提交后，其训练目标缓存会删除。因此长期训练不会反复记忆一个 2304 状态的数据集；主要风险是有限批量的方向噪声、规则对手分布偏差和模拟回报的系统偏差。

## 对手和可见信息

每局只有一个轮换座位由当前 Actor 控制。训练开始时另外三个座位由 `rule_fast` 和 `rule_safe` 交错控制。固定规则评测确认 Actor 相对冻结 SL 的 paired `dRank` 95% 区间上界小于 `0`，且 paired score delta 的区间上界不小于 `0`、即没有显著分差伤害后，下一轮开始加入 `self_play` 对手。首档比例为 `0.10`；此后相对 SL 的平均 `dRank` 每累计改善 `0.01` 再提高一档：`-0.01 -> 0.20`、`-0.02 -> 0.30`，依此类推，最高为 `2/3`。比例由累计能力档位决定，同一档不会逐轮重复增加；已经启用的比例保持单调，不因一次评测波动自动回退。Actor 的座位在四家之间均匀轮换，且始终至少保留一个规则对手。

自博弈对手不是本轮 learner 的副本。历史池在新 run 时以冻结 SL 初始化；每 `8` 个完整提交追加刚提交的 Actor 快照，只保留最近 `4` 个不同 digest 的快照。每一轮只从池中轮换选取一个与本轮 learner digest 不同的 snapshot，所有标为 `self_play` 的席位都使用它；该 snapshot 在本轮 source、校准 source 和反事实续局中保持不变。这样 learner 每轮仍冻结为目标生成的 reference，但对手只按较长周期更新，并在多个历史策略间轮换。snapshot 权重、digest、轮换计数和最后刷新 iteration 都保存于 `latest.pt`；target cache 指纹也包含所选 opponent digest。

训练 source、校准 source 和反事实续局使用完全相同的阵容比例和同一个历史 opponent snapshot；candidate 只有在提交后才有资格在后续池刷新时进入历史池。固定规则 reference panel、candidate 评测和 batch-size sweep 始终显式使用 `self_play_fraction=0`，所以 curriculum 不会污染能力门控或 batch 比较。

Actor 始终只读取当前行动者可见的手牌、公开状态和事件历史。目标生成不会猜测或重采样对手当前暗手：每个 query 保留来源牌局的四家当前手牌和公开状态，只独立重洗尚未摸出的 live wall。对手暗手的不确定性由 batch 中大量独立来源牌局做经验积分，而不是由一个冷启动 belief 模型提供。后续续局中，焦点座位调用本轮冻结 learner，标记为 self-play 的座位调用本轮选中的历史 snapshot，其余座位继续使用该局原有规则类型。

采集动作采用合法动作 mask 后的确定性 argmax。规则动作同样经过合法性检查。任何非有限 Actor logits、非法动作、轨迹重放不一致或引擎规则版本不匹配都会立即失败，不会静默跳过样本。

## 独立状态和九类决策

训练只选择当前 Actor 实际控制、且至少有两个合法动作的状态。选择器按稀缺类别优先分配完整对局，并保证一局最多贡献一个状态。默认每类 256 个：

1. 换三张第一张；
2. 换三张第二张；
3. 换三张第三张；
4. 定缺；
5. 早盘回合；
6. 中盘回合；
7. 晚盘回合；
8. 胡牌响应；
9. 碰杠响应。

虽然训练 batch 对九类做等额采样，优化目标并非九类等权。系统会从本轮完整 source trajectories 估计当前 Actor 的自然访问频率，并将同一组 visitation weights 用于训练损失和 KL 校准。这样既保证稀有类别有样本，又保持目标接近真实决策分布。

## 配对 Live-Wall 目标

每个选中状态本身来自一个独立的完整隐藏牌局。系统保持该时刻的四家手牌不变，只将尚未摸出的 live wall 独立重洗 `worlds=16` 次。所有合法候选动作共享同一组未来牌墙排列，然后各自续局到终局。目标同时记录名次效用和归一化分差；Actor 更新使用合法动作上的中心化名次效用：

```text
Q_rank(s, a) = mean rank utility over paired worlds
Q_centered(s, a) = Q_rank(s, a) - mean_legal_actions Q_rank(s, .)
```

同一来源暗手、同一未来牌墙上的候选动作配对比较，消除了它们的共同噪声；跨 2304 个独立来源牌局的全批量梯度再平均隐藏暗手差异。这个估计没有声称单个 query 已经积分完整信息集。`world_chunk` 只控制一次送入续局器的分支量，不改变统计目标。目标按 query shard 原子保存，缓存带 Actor 摘要、iteration、root seed、完整轨迹签名和 world count 指纹。

## 单步 Actor 更新

每轮先冻结当前 Actor 作为 reference，再复制出 candidate。完整 batch 的单步方向只在合法动作策略分布上最大化预期中心化名次效用：

```text
L_row = -E_pi[Q_centered(s, a)]
```

所有 microbatch 的梯度按 visitation row weight 累积；整批只 clip 一次梯度并执行一次 `AdamW.step()`。在这个 step 之前 candidate 与 reference 完全相同，因此把 reference KL 放进方向损失只会产生恒为零的梯度和一次多余前向；真正的 trust region 统一由后续独立 KL 校准实现。`microbatch_size` 只控制显存，不改变 batch、损失权重或优化器 step 数。

AdamW 得到的参数差只被当作一个方向。系统在完全独立的校准状态上做二分缩放，使 visitation-weighted reverse KL 接近但不超过默认 `0.001`。BF16 前向会让很小的校准集出现离散 KL 台阶，因此允许最多 5% 的保守 undershoot，并记录实际 `relative_shortfall`；无法进入这个区间、KL 非单调或方向为零都会直接报错。正式默认的 576 个校准状态通常比 smoke 的小面板平滑得多。

环境实际执行 greedy argmax，而 KL 衡量 softmax 分布；接近并列的动作可能在很小 KL 下发生离散翻转。系统因此同时报告 calibration 上 visitation-weighted `greedy_flip_rate`，但不使用未经验证的固定阈值阻止更新。固定规则整局配对评测仍是判断真实行为变化的主要指标。

## 固定规则评测

新 run 首先让冻结 SL 在固定的 16384 个 seeds 上对规则对手完成 reference panel。之后每个 candidate 使用完全相同的 seeds、焦点座位和规则阵容进行评测。评测报告以下指标，其中 paired `dRank` 的累计改善和置信区间驱动训练 curriculum，paired score delta 作为非劣化护栏：

- Actor 的平均名次、相对初始分的平均分差、首位率和末位率；
- 相对 SL 的 paired `dRank` 和 score delta；
- 以完整对局为单位的 95% bootstrap 区间。

`dRank < 0` 表示 Actor 名次优于 SL，score delta `> 0` 表示 Actor 分数优于 SL。固定面板用于降低迭代曲线噪声，并不是泛化证明；长期结果仍应在全新 seeds 和其他对手分布上复验。

checkpoint 额外保存当前 self-play 比例、最近一次固定规则首位率、首次启用的 iteration 和历史 opponent pool；恢复时不会重新计算或随机改变当前 iteration 的阵容。现有 v3 checkpoint 可直接恢复：当前 pending iteration 保持旧的同模型语义和 target 指纹，第一次完整提交后原子升级为带历史池的 v4 checkpoint；后续 iteration 使用历史 opponent。

## 启动与恢复

准备 release 版引擎扩展：

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
```

启动正式训练：

```bash
python -m training.train \
  --output-dir runs/policy-iteration-v3 \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt
```

`--sl-checkpoint` 必须是 Actor-only 文件，顶层严格包含 `model_config` 和 `model`。训练器不会重新执行 SL，也不会修改这份文件。省略参数时使用上面两个路径对应的默认值。

恢复：

```bash
python -m training.train \
  --resume runs/policy-iteration-v3/latest.pt
```

恢复时配置、root seed、SL 路径及其 SHA-256、模型结构、引擎规则版本和策略执行版本都来自 checkpoint，并进行严格校验。不能在恢复命令中覆盖训练参数或 seed。生产 v3 checkpoint 是历史池 v4 的唯一兼容输入；更早格式没有迁移路径。本轮 batching 优化改变了极少数 BF16 近并列动作的执行顺序，因此必须使用新的空输出目录，不能复用优化前的 reference/target 缓存。

`latest.pt` 是唯一权威的恢复点。它只在评测完成后原子替换，因此只代表完整提交的 iteration。若在目标生成期间中断，对应 `pending/iteration-*` 分片会保留；恢复后确定性重建同一批 query，并只补齐缺失分片。完整 iteration 提交后该 pending 目录会删除。

## 输出目录

一个正式 run 包含：

- `latest.pt`：可恢复的完整状态，包括 Actor、历史 opponent pool、配置、root seed、下一 iteration 和最后指标；
- `actor.pt`：Actor-only 部署权重；
- `config.json`：不可变的 run 身份、训练配置、SL 摘要和引擎规则版本；
- `metrics.jsonl`：每个已提交 iteration 的 source、目标、优化、校准和评测指标；
- `reference_panel.npz`：冻结 SL 的固定规则评测结果；
- `pending/iteration-*/targets/`：仅在 iteration 未完成时存在的可恢复目标分片。

正式默认的 Actor 梯度 batch 是 `9 x 256 = 2304`；独立 KL 校准集是 `9 x 64 = 576`。后者只测量策略距离，不产生训练目标或梯度，不能与训练 batch 混为一谈。

终端进度会显示当前阶段、完成量、速率、elapsed 和 ETA。每轮提交后打印一行 `dRank`、置信区间、score、KL 和总耗时摘要。

## 主要参数

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--source-games` | 4096 | 每轮训练状态的全新来源对局 |
| `--calibration-source-games` | 4096 | 每轮独立 KL 校准来源对局 |
| `--envs` | 512 | CUDA 采集并行环境数 |
| `--queries-per-category` | 256 | 每类训练状态数，总状态数为其 9 倍 |
| `--calibration-queries-per-category` | 64 | 每类独立校准状态数 |
| `--worlds` | 16 | 每个训练状态的配对未来牌墙数 |
| `--world-chunk` | 64 | 每组续局中每个 query 的 world 上限 |
| `--target-shard-size` | 64 | 每个可恢复目标 shard 的 query 数 |
| `--target-query-batch-size` | 64 | 合并到同一个引擎续局 batch 的 query 数 |
| `--rollout-inference-batch-size` | 128 | 续局 Actor 单次 CUDA 前向的最大行数 |
| `--direction-learning-rate` | `1e-5` | 产生单步参数方向的学习率 |
| `--microbatch-size` | 64 | 梯度累积块大小 |
| `--inference-batch-size` | 128 | KL 校准前向块大小 |
| `--target-kl` | `0.001` | 每个策略版本的目标 KL |
| `--evaluation-games` | 16384 | 固定规则配对评测局数 |
| `--evaluation-envs` | 512 | 评测并行环境数 |
| `--bootstrap-samples` | 10000 | 配对区间的 bootstrap 次数 |
| `--self-play-start-first-rate` | 0.55 | 仅为现有 v3 checkpoint 身份兼容保留，新门控不读取此值 |
| `--self-play-increment` | 0.10 | paired `dRank` 每改善 `0.01` 对应的 self-play 档位增量 |
| `--maximum-self-play-fraction` | 2/3 | self-play 对手席位上限，至少保留一个规则对手 |

新 run 可以覆盖这些参数；resume 不接受覆盖。不要仅为了提高显存占用而放大 `world_chunk`、microbatch 或 envs。三者控制的是不同阶段的瞬时工作集，应以端到端 iteration 时间和稳定性选择。

目标生成在单个 CUDA 进程内把 64 个 query 的全部 `(action, world)` 分支合并。这样 Rust `Batch` 会跨 CPU 核执行重放、观测与规则动作，Actor 则按最多 128 行连续送入 GPU；不会为同一张 GPU 启动多个会复制模型和 CUDA context 的 seed 进程。组内长续局会持续更新 step、active branches 和吞吐心跳。RTX 5080 的真实续局对照中，128 行推理块比 512 行快约 18%；更大的块虽占用更多显存，但该模型的长历史注意力吞吐反而下降。

## Batch Size Sweep

`training.batch_sweep` 用同一个最大训练 corpus 构造嵌套 batch，比较每类 `64/128/256/512`，即总计 `576/1152/2304/4608` 个状态。Sweep 的独立 KL calibration 固定为每类 128、总计 1152 个状态。所有候选共享：

- 最大训练 corpus 的分类别前缀；
- 独立 calibration corpus；
- 带 64 个配对世界的独立 heldout Q corpus；
- 同一个 16384 局固定规则 seed panel。

因此 batch 之间的差异主要来自状态数，而不是不同采样运气。每个候选仍然只执行一个 AdamW step，并校准到相同 KL。输出比较与最大 batch 的方向 cosine、有效样本量、heldout policy value、paired `dRank`、score、耗时和吞吐。

运行完整 sweep：

```bash
python -m training.batch_sweep \
  --output-dir runs/batch-sweep-v3 \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt
```

正式比较建议直接使用三个独立 seed；每个 seed 会在 `seed-<N>/` 下独立缓存，全部完成后根目录写入 pooled `aggregate.json`：

```bash
python -m training.batch_sweep \
  --output-dir runs/batch-sweep-v3-multiseed \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt \
  --seeds 20260727 20260728 20260729
```

中断后重复完全相同的命令即可恢复；已完成的 seed 不会重新采集。若最大 QPC 的稀缺类别配额不足，可在命令中增加 `--source-games 16384`。

同一命令可恢复。sweep 会分别原子缓存共享 train/calibration/heldout corpus、每个 QPC 的单步参数方向，以及每个 candidate 的原始固定规则评测面板；中断后只补缺失阶段。`summary.json` 已完整时会在加载模型和采集数据前直接返回。目录配置或 SL 文件发生变化时会拒绝混用结果。

CUDA smoke：

```bash
python -m training.batch_sweep \
  --output-dir /tmp/batch-sweep-smoke \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt \
  --smoke
```

Smoke 只验证端到端控制流、缓存、单步更新、校准、评测和输出文件，不用于判断策略强度。

## 如何选择 batch

不要只看单次固定面板的均值。优先顺序是：

1. heldout rank value 为正，且没有明显以 score value 为代价；
2. 较小 batch 的方向与最大 batch 高度一致；
3. 配对规则评测的 `dRank` 和 score 区间不显示伤害；
4. 增大 batch 带来的稳定性收益足以覆盖更低的策略版本更新频率。

三 seed sweep 表明 576 和 1152 状态仍会出现反向 seed；2304 是第一档三个 seed 的名次方向全部改善、且独立 heldout 均值转正的规模。4608 的 pooled 单步证据最强，但相对 2304 的直接差异不显著，目标生成成本接近两倍。因此正式训练默认采用 2304 状态，4608 保留为质量优先实验配置。

## 验证

CPU 单元测试只验证算法不变量和数据管线；正式训练及端到端 smoke 必须使用 CUDA：

```bash
python -m py_compile training/*.py training/tests/*.py
python -m pytest training/tests -q
python -m training.benchmarks.transformer --device cuda
```
