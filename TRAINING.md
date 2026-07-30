# 保守策略迭代训练

> 文档状态：本文保留 Python 训练实验的完整说明。当前未发现已知过时内容，但该训练路径已不再作为 Rust 主线维护，后续可能与代码产生偏差。旧文中的“规则对手”均指固定策略，不指游戏规则。

`python -m training.train` 是正式训练入口。推荐主线从 U61 champion 启动 `rank_lcb_mirror_ce` generation。旧的 `expected_q`、split CE 和单步 AdamW/SGD 路径仍可复现实验，但不再用于继续主线。

```text
冻结 champion，并从历史池选择一个 opponent snapshot
  -> 独立采集 4 批状态，每批九类各 128 个
  -> A=32 个 live-wall worlds 只选择候选动作
  -> B=64 个独立 worlds 估计 paired mean-rank advantage
  -> 对全部状态执行 BH-FDR=0.05，并构造 LCB mirror CE target
  -> 在同一 generation 内执行 4 个 Nesterov inner steps
  -> 每一步都投影到相对 champion 的累计 KL <= 0.0001
  -> C=32 个未参与 target 的 worlds 审计 greedy action 发生翻转的状态
  -> 在 fixed-rule 和 historical-opponent 各 65536 局 arena 比较 candidate/champion
  -> 仅在 pooled dRank 显著改善且所有安全护栏通过时晋升
```

训练只支持 CUDA。梯度更新和策略推理都使用 eager mode 和 BF16 autocast。推理保留 pinned-memory 双缓冲和按 history width 分桶，不使用 TorchScript trace 或 `torch.compile`。TorchScript trace 在真实 source 状态上产生过 greedy action 翻转；本机 PyTorch 2.13.0 + CUDA 13.0 的 Inductor BF16 autocast 融合也会产生错误 logits。Inductor 的最小复现和日志见 [`training/benchmarks/TORCH_COMPILE_ISSUE.md`](training/benchmarks/TORCH_COMPILE_ISSUE.md)。训练没有最大 attempt 数，会一直运行到用户按 `Ctrl+C`。被拒绝的 attempt 会保存指标并推进编号，但不会修改 champion、自博弈课程或 opponent snapshots；历史对手的轮换游标仍会前进。

## 为什么采用这条路径

麻将单局终局回报的方差很大。小批量在线更新容易把隐藏状态偶然性当成动作优势。当前 generation 针对这个问题施加以下约束：

- 一个训练状态至多来自一个完整隐藏牌局，避免把同一局的多个相关决策伪装成独立样本；
- 九类决策都有硬覆盖，每批 `9 x 128 = 1152` 个状态；
- 动作选择、统计验证和最终 audit 使用互不重叠的 A/B/C worlds；
- B worlds 使用 mean-rank paired advantage，并对全部状态统一控制 FDR；
- 未通过验证的状态行权重为零，不会在 Nesterov lookahead 上形成隐式旧策略梯度；
- Nesterov 速度只在一个固定 champion generation 内保留；
- KL 是相对 champion 的累计上限，不会被 4 个 inner steps 逐步穿透；
- candidate 必须通过独立 arena 才能成为下一 champion。

这里固定的是每个 generation 的采样协议，不是状态内容。每个 attempt 都重新生成 source games、训练状态、world seeds 和校准状态。完整 attempt 提交后，其目标缓存会删除。

## 对手和可见信息

每局只有一个轮换座位由当前 Actor 控制。训练开始时另外三个座位由 `rule_fast` 和 `rule_safe` 交错控制。固定规则评测确认 Actor 相对冻结 SL 的 paired `dRank` 95% 区间上界小于 `0`，且 paired score delta 的区间上界不小于 `0`、即没有显著分差伤害后，下一轮开始加入 `self_play` 对手。首档比例为 `0.10`；此后相对 SL 的平均 `dRank` 每累计改善 `0.01` 再提高一档：`-0.01 -> 0.20`、`-0.02 -> 0.30`，依此类推，最高为 `2/3`。比例由累计能力档位决定，同一档不会逐轮重复增加；已经启用的比例保持单调，不因一次评测波动自动回退。Actor 的座位在四家之间均匀轮换，且始终至少保留一个规则对手。

自博弈对手不是本轮 learner 的副本。历史池在新 run 时以冻结 SL 初始化；每 `8` 个完整提交追加刚提交的 Actor 快照，只保留最近 `4` 个不同 digest 的快照。每一轮只从池中轮换选取一个与本轮 learner digest 不同的 snapshot，所有标为 `self_play` 的席位都使用它；该 snapshot 在本轮 source、校准 source 和反事实续局中保持不变。这样 learner 每轮仍冻结为目标生成的 reference，但对手只按较长周期更新，并在多个历史策略间轮换。snapshot 权重、digest、轮换计数和最后刷新 iteration 都保存于 `latest.pt`；target cache 指纹也包含所选 opponent digest。

训练 source、校准 source 和反事实续局使用完全相同的阵容比例和同一个历史 opponent snapshot；candidate 只有在提交后才有资格在后续池刷新时进入历史池。固定规则 reference panel、candidate 评测和 batch-size sweep 始终显式使用 `self_play_fraction=0`，所以 curriculum 不会污染能力门控或 batch 比较。

在 `rank_lcb_mirror_ce` 主线中，“完成 attempt”和“晋升 champion”是两个事件。双 arena 拒绝 candidate 时，self-play fraction 和 opponent snapshots 保持不变，但轮换游标继续前进。只有晋升后的 Actor 才能推进 curriculum 或进入历史池。

`--anchor-rule-fast` 是用于检验强规则对手权重的实验开关。启用后，每局三个对手席位先固定一个 `rule_fast`，另外两个初始化为 `rule_safe`，self-play 只替换后两个席位。这样在 self-play fraction 为 `0.50` 时，期望阵容严格为 `rule_fast 1.0 / self-play 1.5 / rule_safe 0.5` 个席位，即对手席位占比约 `33% / 50% / 17%`；默认关闭时完全保留原有随机替换语义。

要从历史池里的完整 snapshot 做受控分支，而不是把该 Actor 当作全新 SL 重置 curriculum，可使用 fork 工具。例如从 U56 构造 fast-anchor 分支：

```bash
python -m training.fork_policy_iteration \
  --source-checkpoint runs/policy-iteration-v3/latest.pt \
  --iteration 56 \
  --output-dir runs/policy-iteration-u56-fast-anchor-v1 \
  --anchor-rule-fast

python -m training.train \
  --resume runs/policy-iteration-u56-fast-anchor-v1/latest.pt
```

fork 会恢复指定 iteration 提交后的 Actor、self-play fraction、activation iteration、当时的历史 opponent pool、rotation、root seed 和下一 iteration 编号；新分支用该 Actor 建立自己的固定 reference panel，并在 `fork.json` 中记录来源 checkpoint SHA-256。源 run 和其中未完成的 pending shard 都不会被修改。

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

## Rank-LCB Mirror Target

每个 generation batch 使用同一组 source states 和三组独立 world seeds。A worlds 只负责从 reference greedy action 之外选择一个候选动作。B worlds 只负责估计该候选动作相对 reference action 的 paired mean-rank utility advantage、标准误、单侧 p-value 和 95% lower confidence bound。训练器对本批全部 p-value 执行 Benjamini-Hochberg FDR 控制。

通过验证的行使用以下 mirror target：

```text
pi_target(a|s) proportional to pi_champion(a|s) * exp(LCB(a) / 0.05)
```

未通过验证的行设置 `row_confidence=0`。损失不会重新归一化剩余有效质量，因此统计证据不足会直接减小梯度，而不是放大少数标签。C worlds 不参与动作选择、验证或梯度。训练完成后，C worlds 只续局 candidate 与 champion greedy action 不同的状态；动作相同状态的 paired advantage 数学上恒为零，仍进入完整状态数的均值、标准误、分类统计和 flip rate。

当前主线使用 `live_wall`。该模式固定来源牌局中的四家暗手，只重排尚未摸出的牌墙，因此每个状态仍有多个未知未来。现有 `information_set` 实现会均匀重洗对手暗手，但不会按历史动作似然形成后验；在该后验问题修复前，主线配置会拒绝 information-set sampling。

## 配对 Live-Wall 续局

每个选中状态本身来自一个独立的完整隐藏牌局。系统保持该时刻的四家手牌不变，只将尚未摸出的 live wall 独立重洗 `worlds=16` 次。所有合法候选动作共享同一组未来牌墙排列，然后各自续局到终局。目标同时记录名次效用和归一化分差；Actor 更新使用合法动作上的中心化名次效用：

```text
Q_rank(s, a) = mean rank utility over paired worlds
Q_centered(s, a) = Q_rank(s, a) - mean_legal_actions Q_rank(s, .)
```

同一来源暗手、同一未来牌墙上的候选动作配对比较，消除了它们的共同噪声；跨 2304 个独立来源牌局的全批量梯度再平均隐藏暗手差异。这个估计没有声称单个 query 已经积分完整信息集。`world_chunk` 只控制一次送入续局器的分支量，不改变统计目标。目标按 query shard 原子保存，缓存带 Actor 摘要、iteration、root seed、完整轨迹签名和 world count 指纹。

## Legacy 单步 Actor 更新

每轮先冻结当前 Actor 作为 reference，再复制出 candidate。完整 batch 的单步方向只在合法动作策略分布上最大化预期中心化名次效用：

```text
L_row = -E_pi[Q_centered(s, a)]
```

所有 microbatch 的梯度按 visitation row weight 累积；整批只 clip 一次梯度并执行一次 optimizer step。默认 fresh AdamW 的 candidate 与 reference 在这个 step 之前完全相同，因此把 reference KL 放进方向损失只会产生恒为零的梯度和一次多余前向；真正的 trust region 统一由后续独立 KL 校准实现。`microbatch_size` 只控制显存，不改变 batch、损失权重或优化器 step 数。

优化器得到的参数差只被当作一个方向。系统在完全独立的校准状态上做二分缩放，使 visitation-weighted reverse KL 接近但不超过默认 `0.001`。BF16 前向会让很小的校准集出现离散 KL 台阶，因此允许最多 5% 的保守 undershoot，并记录实际 `relative_shortfall`；无法进入这个区间、KL 非单调或方向为零都会直接报错。正式默认的 576 个校准状态通常比 smoke 的小面板平滑得多。

环境实际执行 greedy argmax，而 KL 衡量 softmax 分布；接近并列的动作可能在很小 KL 下发生离散翻转。系统因此同时报告 calibration 上 visitation-weighted `greedy_flip_rate`，但不使用未经验证的固定阈值阻止更新。固定规则整局配对评测仍是判断真实行为变化的主要指标。

## Generation Nesterov 和累计 KL

主线 generation 在固定 champion 定义的同一个改进问题上执行 4 个 Nesterov inner steps。第一个 step 是 cold-start SGD；后续 step 在 `theta + 0.9 * velocity` 处计算新梯度。每个 step 完成后，训练器沿 `champion -> raw candidate` 投影到同一个 `KL <= 0.0001` 球，并用实际投影后的位移更新 generation-local velocity。generation 结束后 velocity 清空，不能跨越下一 attempt 的新状态和新 target。

日志报告每个 inner step 的 raw gradient、momentum、proposal、累计 KL、投影 scale 和实际位移。最终 calibration probe 还报告九类决策各自的 reverse KL、greedy flip rate，以及 exchange、定缺、弃牌、胡、碰、杠、过之间的动作迁移矩阵。

## 固定规则评测

新 run 首先让冻结 SL 在固定的 16384 个 seeds 上对规则对手完成 reference panel。之后每个 candidate 使用完全相同的 seeds、焦点座位和规则阵容进行评测。评测报告以下指标，其中 paired `dRank` 的累计改善和置信区间驱动训练 curriculum，paired score delta 作为非劣化护栏：

- Actor 的平均名次、相对初始分的平均分差、首位率和末位率；
- 相对 SL 的 paired `dRank` 和 score delta；
- 以完整对局为单位的 95% bootstrap 区间。

`dRank < 0` 表示 Actor 名次优于 SL，score delta `> 0` 表示 Actor 分数优于 SL。固定面板用于降低迭代曲线噪声，并不是泛化证明；长期结果仍应在全新 seeds 和其他对手分布上复验。

主线不会用这块固定 SL 曲线直接提交 candidate。晋升使用两组全新 seeds：65536 局 fixed-rule lineup，以及 65536 局当前 historical-opponent mix。两组都对 candidate/champion 使用相同 seeds、焦点座次和 lineup 随机流。只有 pooled paired dRank 的 95% 区间上界小于 0，且任一单独 lineup 均无显著名次或分差伤害、C-world audit 也无显著伤害时，candidate 才会晋升。

v6 checkpoint 同时保存 `next_iteration` 和 `champion_iteration`。前者是下一 attempt，后者是当前 Actor 的实际版本。v3、v4 和 v5 checkpoint 可读取并迁移；v5 中的历史 Nesterov velocity 不会带入新的 rank-LCB generation。

## 启动与恢复

准备 release 版引擎扩展：

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
```

U61 主线分支由 `training.fork_policy_iteration` 创建。策略执行版本 4 只使用 eager Actor。仓库当前已经生成 `runs/policy-iteration-u61-champion-lcb-eager-v1/latest.pt`。启动或恢复长跑：

```bash
python -u -m training.train \
  --resume runs/policy-iteration-u61-champion-lcb-eager-v1/latest.pt
```

该 checkpoint 从原始 U61 checkpoint 无损 fork。它保留 Actor、self-play curriculum、opponent pool、root seed 和训练配置，固定 `champion=61`、`next_attempt=62`、Nesterov learning rate `1e-4`、momentum `0.9`、gradient clipping off、累计 KL cap `1e-4`，并关闭 fast anchor。trace-v3 分支生成的 reference panel 不会复用。

通用恢复格式：

```bash
python -u -m training.train --resume <run>/latest.pt
```

恢复时配置、root seed、reference 路径及其 SHA-256、模型结构、引擎规则版本和策略执行版本都来自 checkpoint，并进行严格校验。恢复命令不能覆盖训练参数或 seed。

`latest.pt` 是唯一权威恢复点。它只在 arena 完成后原子替换，因此只代表完整 attempt。若在目标生成期间中断，对应 `pending/iteration-*` 分片会保留；恢复后确定性重建同一批 query，并只补齐缺失分片。完整 attempt 提交后该 pending 目录会删除。

## 输出目录

一个正式 run 包含：

- `latest.pt`：可恢复的完整状态，包括 champion Actor、历史 opponent pool、配置、下一 attempt、champion iteration 和最后指标；
- `actor.pt`：Actor-only 部署权重；
- `config.json`：不可变的 run 身份、训练配置、SL 摘要和引擎规则版本；
- `metrics.jsonl`：每个已完成 attempt 的 target、inner step、audit、arena 和晋升决定；
- `reference_panel.npz`：冻结 SL 的固定规则评测结果；
- `pending/iteration-*/generation-*/`：仅在 attempt 未完成时存在的 A/B/C world 分片。

U61 主线每批是 `9 x 128 = 1152` 个状态，一个 generation 共 4 批；独立 KL 校准集是 `9 x 64 = 576` 个状态。校准集只测量策略距离，不产生 target 或梯度。

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
| `--validation-worlds` | 64 | rank-LCB B 组验证 worlds |
| `--audit-worlds` | 32 | 不参与 target 的 C 组 audit worlds |
| `--generation-batches` | 4 | 固定 champion 内的独立 target batch 和 Nesterov step 数 |
| `--target-fdr` | 0.05 | B 组 p-value 的 BH-FDR 上限 |
| `--mirror-temperature` | 0.05 | LCB mirror target 温度 |
| `--mirror-prior-floor` | `1e-6` | legal action 的 champion prior 下限 |
| `--world-chunk` | 64 | 每组续局中每个 query 的 world 上限 |
| `--target-shard-size` | 64 | 每个可恢复目标 shard 的 query 数 |
| `--target-query-batch-size` | 64 | 合并到同一个引擎续局 batch 的 query 数 |
| `--rollout-inference-batch-size` | 128 | 续局 Actor 单次 CUDA 前向的最大行数 |
| `--direction-learning-rate` | `1e-5` | 产生单步参数方向的学习率 |
| `--microbatch-size` | 64 | 梯度累积块大小 |
| `--inference-batch-size` | 128 | KL 校准前向块大小 |
| `--target-kl` | `0.001` | 每个策略版本的目标 KL |
| `--evaluation-games` | 16384 | 固定规则配对评测局数 |
| `--arena-games` | 65536 | 每种 candidate/champion 晋升阵容的局数 |
| `--evaluation-envs` | 4096 | 评测并行环境数；BF16 分块与旧 512-env run 不同 |
| `--bootstrap-samples` | 10000 | 配对区间的 bootstrap 次数 |
| `--self-play-start-first-rate` | 0.55 | 仅为现有 v3 checkpoint 身份兼容保留，新门控不读取此值 |
| `--self-play-increment` | 0.10 | paired `dRank` 每改善 `0.01` 对应的 self-play 档位增量 |
| `--maximum-self-play-fraction` | 2/3 | self-play 对手席位上限，至少保留一个规则对手 |
| `--anchor-rule-fast` | false | 每局固定保留一个 `rule_fast`，self-play 只替换另外两个规则席位 |

新 run 可以覆盖这些参数；resume 不接受覆盖。不要仅为了提高显存占用而放大 `world_chunk`、microbatch 或 envs。三者控制的是不同阶段的瞬时工作集，应以端到端 iteration 时间和稳定性选择。

目标生成在单个 CUDA 进程内把 64 个 query 的全部 `(action, world)` 分支合并。这样 Rust `Batch` 会跨 CPU 核执行重放、观测与规则动作，Actor 则按最多 128 行连续送入 GPU；不会为同一张 GPU 启动多个会复制模型和 CUDA context 的 seed 进程。组内长续局会持续更新 step、active branches 和吞吐心跳。RTX 5080 的真实续局对照中，128 行推理块比 512 行快约 18%；更大的块虽占用更多显存，但该模型的长历史注意力吞吐反而下降。

## Batch Size Sweep

`training.batch_sweep` 用同一个最大训练 corpus 构造嵌套 batch，比较每类 `64/128/256/512`，即总计 `576/1152/2304/4608` 个状态。Sweep 的独立 KL calibration 固定为每类 128、总计 1152 个状态。所有候选共享：

- 最大训练 corpus 的分类别前缀；
- 独立 calibration corpus；
- 带 64 个配对世界的独立 heldout Q corpus；
- 同一个 16384 局固定规则 seed panel。

因此 batch 之间的差异主要来自状态数，而不是不同采样运气。每个候选仍然只执行一个 fresh AdamW step，并校准到相同 KL。输出比较与最大 batch 的方向 cosine、有效样本量、heldout policy value、paired `dRank`、score、耗时和吞吐。

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

## 优化器实验

当前 AdamW 每轮重建且只走一步，参数更新在实际数值上接近 signSGD；KL 校准再把方向缩放到同一个 trust region。为先隔离“方向”问题，可复用已完成的 QPC sweep 语料做 raw SGD 对照：

```bash
python -u -m training.optimizer_sweep \
  --batch-sweep-dir runs/batch-sweep-u56-qpc-v1 \
  --output-dir runs/optimizer-sweep-u56-sgd-v1 \
  --sgd-learning-rate 0.1 \
  --seeds 20260728 20260729
```

该命令不会写入输入 sweep 的 shard；未被 `--seeds` 选中的未完成 seed 会跳过，选中的 seed 若未完成则直接报错。不指定 `--seeds` 时仍要求配置中的全部 seed 完成。SGD 的 nominal learning rate 只需让方向能被 `maximum_scale` bracket，最终步长由独立 KL calibration 决定。

当前 U56 数据里 AdamW 原始 candidate KL 只有约 `0.00019-0.00032`，原流程把参数方向放大约 `1.8x-2.4x` 才提交。下面的固定 scale 实验直接检验“KL 只作 cap”而不重新生成 Q target：

```bash
python -u -m training.kl_scale_sweep \
  --batch-sweep-dir runs/batch-sweep-u56-qpc-v1 \
  --output-dir runs/kl-scale-sweep-u56-adamw-v1 \
  --qpc 256 \
  --scales 0.5 1.0 \
  --seeds 20260728 20260729
```

其中 `scale=1.0` 是 raw AdamW，也正是 cap 语义在原始 KL 低于上限时的提交点；`scale=0.5` 检查 raw step 是否仍过大。输入中原有的强制 KL policy 和 panel 作为 calibrated baseline 直接复用。输出的 `aggregate.json` 同时给出每个 scale 相对 reference、calibrated baseline 和 raw baseline 的 pooled 配对统计。这个 sweep 只新增四个固定 panel（两 scale、两 seed），不会采集 source 或执行 counterfactual rollout。

若 scale sweep 只把 candidate 拉回 reference 而没有产生正收益，下一步应区分“方向只拟合自己的 Q 样本”和“Q surrogate 本身不能预测 greedy 整局收益”。下面的实验加载同一 checkpoint 的两 seed AdamW/SGD cached direction，把四个方向统一到共同 probe 上 `KL=0.0001`，再交叉评价两套 heldout Q：

```bash
python -u -m training.direction_generalization \
  --batch-sweep-dir runs/batch-sweep-u56-qpc-v1 \
  --optimizer-sweep-dir runs/optimizer-sweep-u56-sgd-v1 \
  --output-dir runs/direction-generalization-u56-v2 \
  --qpc 256 \
  --seeds 20260728 20260729 \
  --target-kl 0.0001
```

输出 `result.json` 包含完整的 `direction seed x heldout seed` soft-policy `delta-pi * Q` 和 greedy-argmax Q 矩阵、参数空间 cosine、共同 probe 上的 policy-space cosine，以及九类的概率变化、flip、动作分歧、reference softmax 饱和度和 Q 最优动作分歧。该实验没有 source collection、counterfactual rollout 或整局 panel，只需模型前向。

若跨 seed 方向不稳定来自“2304 状态更新 880 万参数”的高自由度拟合，可复用相同 target 比较三种更新子空间：全模型、最后一层 static/history block 加 policy head、仅 policy head：

```bash
python -u -m training.update_subspace_sweep \
  --batch-sweep-dir runs/batch-sweep-u56-qpc-v1 \
  --output-dir runs/update-subspace-sweep-u56-v1 \
  --qpc 256 \
  --seeds 20260728 20260729 \
  --scopes full last_blocks actor \
  --optimizers adamw sgd \
  --target-kl 0.0001
```

每个 optimizer/scope 同时产生两个单 seed candidate 和一个合并两批独立 Q 的 pooled candidate，并统一到共同 calibration KL。`result.json` 报告 train/heldout 泛化、跨 seed policy cosine，以及 pooled candidate 在两套 heldout 上的均值和最差值；无新 rollout 或整局评测。

Nesterov 不能从一个 U56 checkpoint 的单步冷启动得到有效结论，因为第一步与 SGD 共线。长跑分支使用 `fork_policy_iteration` 保存跨 update 的 committed displacement：

```bash
python -u -m training.fork_policy_iteration \
  --source-checkpoint runs/policy-iteration-v3/latest.pt \
  --iteration 56 \
  --output-dir runs/optimizer-u56-nesterov-v1 \
  --no-anchor-rule-fast \
  --direction-optimizer nesterov \
  --direction-learning-rate 0.1 \
  --direction-momentum 0.9

python -u -m training.train --resume runs/optimizer-u56-nesterov-v1/latest.pt
```

应从同一 source iteration 再开 AdamW 和 raw SGD control，比较相同 iteration 的固定 panel 和 heldout；`--direction-optimizer momentum` 可作为 Polyak 对照。Nesterov 在 `theta + momentum * velocity` 的 lookahead 点算梯度，完成 KL 校准后才把实际提交位移写入 checkpoint，因此中断不会污染 velocity。

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
