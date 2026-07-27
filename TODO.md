# 保守策略迭代状态

旧训练实验已经退出正式代码路径。本文件只跟踪当前大独立状态、单步 KL 约束策略迭代，以及用于确定可靠批量的 nested batch sweep。实现完成不等于已经证明长期策略持续提升；统计结论必须由独立 CUDA 实验给出。

## 已完成实现

- [x] 将 `python -m training.train` 收敛为唯一正式训练入口，CUDA only，默认无限运行到 `Ctrl+C`。
- [x] 直接加载 Actor-only SL checkpoint，不重新执行 SL，不提供旧 checkpoint 迁移或配置兼容。
- [x] 每个策略版本冻结当前 Actor，并按相对 SL 的 paired `dRank` 置信门控和累计能力档位逐步加入历史池 self-play，以 score 非劣化为护栏，同时保留至少一个 fast/safe 规则对手；对手快照每 8 次提交刷新、保留最近 4 个并轮换。
- [x] 覆盖全部九类决策；三 seed sweep 后默认每类选择 256 个状态，并强制一局最多贡献一个训练状态。
- [x] 每轮从 source trajectories 估计自然访问频率，统一用于训练 row weights 和独立 KL 校准。
- [x] 每个状态保留来源暗手并重洗 16 个未来 live wall；全部合法动作共享 paired futures，隐藏手差异由独立状态 batch 平均。
- [x] 目标生成按 shard 原子缓存，带 Actor、iteration、seed 和配置指纹，可恢复未完成 iteration。
- [x] 将 64 个 query 的 action/world 分支合并为单个 ragged rollout，启用 Rayon 多核与分块 Actor CUDA 前向，并提供组内进度心跳。
- [x] 完整 batch 使用 microbatch 梯度累积，只 clip 一次并严格执行一次 `AdamW.step()`。
- [x] Actor 目标使用中心化名次效用与 current-reference reverse KL，不读取隐藏信息。
- [x] 使用全新独立状态校准参数方向，使 visitation-weighted KL 达到 `0.001`。
- [x] 使用固定 16384 局规则 seeds 对 SL 和每个 candidate 做精确配对评测，并报告 bootstrap 区间。
- [x] `latest.pt` 作为原子 commit point；完整提交后才推进策略版本，并删除已完成目标缓存。
- [x] checkpoint 严格校验格式、模型配置、SL SHA-256 和引擎规则版本。
- [x] 提供 TTY 原地进度及普通日志进度，覆盖采集、选样、目标生成、Actor、KL、评测和提交阶段。
- [x] 修复 Actor 采集使用未 mask logits 选择非法动作的问题。
- [x] 修复续局历史缓冲区未使用模型 `max_history` 的问题。
- [x] 删除已废弃训练模块、实验入口及其专用测试，不保留双训练路径。
- [x] 建立聚焦测试，覆盖九类映射、独立 quota、嵌套 corpus、访问频率权重、缓存 roundtrip、单 optimizer step、固定 seed 配对评测、checkpoint 和进度 ETA。
- [x] 正式默认采用三 seed sweep 选出的每类 256、总计 2304 个状态，并将 Actor rollout 前向块设为实测更快的 128 行。
- [x] 完成采集与续局热路径审计：只生成模型座位历史和规则座位动作，重叠 CPU 规则计算与 GPU 前向，使用持久 pinned staging、异步 H2D、NumPy 位解包及 Rust swap-remove 缩批。
- [x] 将策略执行版本纳入 checkpoint、reference panel、target 和 sweep 身份；batching 语义变化后严格拒绝旧缓存。

## Batch Sweep

- [x] 提供 `training.batch_sweep`，默认比较 QPC `64/128/256/512`，对应 `576/1152/2304/4608` 状态。
- [x] 所有候选共享同一最大 nested training corpus，避免 batch 之间混入 source sampling 差异。
- [x] calibration、heldout Q 和固定规则评测使用彼此独立且对所有候选共享的 seed 域。
- [x] 每个候选校准到相同 KL，并报告相对最大 batch 的方向 cosine、ESS、heldout value、paired `dRank`、score、耗时和吞吐。
- [x] 共享 corpus、target shard、逐 batch 方向、原始评测 panel 与 `summary.json` 分层恢复；完整重复命令零重算。
- [x] 提供有进度、吞吐、elapsed 和 ETA 的 CUDA smoke。

## 已有证据

- [x] 576 状态的独立对照结果未排除零，因此不能作为默认可靠 batch。
- [x] 两个独立 1152 状态候选在全新 16384 局规则面板上得到正向结果：`dRank -0.0073 [-0.0124,-0.0021]` 和 `-0.0037 [-0.0073,-0.0002]`。
- [x] 固定的是每轮 2304 的数量，不是状态数据；每个提交后的策略版本都会重新采集 source、query、world 和 calibration corpus。
- [x] 三 seed nested sweep 显示 2304 是首个三个 seed 名次均改善的规模；4608 相对 2304 的直接优势不显著且接近两倍 target 成本，因此正式默认采用 2304。

## 待运行的证据任务

以下项目是正式实验，不是缺失的训练代码：

- [ ] 用选定 batch 连续训练多个策略版本，观察 heldout value、KL、paired `dRank` 和 score 是否保持同向，而非只看单轮均值。
- [ ] 对候选 checkpoint 使用全新 seeds 和不同对手分布复验，量化对固定规则面板的分布过拟合。
- [ ] 只有固定 quota 的方差仍明显限制训练时，再评估按访问概率、梯度方差和单状态成本分配类别预算；必须保留每类覆盖下限和正确权重。

## 验收命令

```bash
python -m py_compile training/*.py training/tests/*.py
python -m pytest training/tests -q
python -m training.batch_sweep \
  --output-dir /tmp/batch-sweep-smoke \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt \
  --smoke
```

前两项验证静态和单元测试；最后一项必须在 CUDA 上执行，验证真实引擎、目标分片、单步更新、KL 校准、评测和恢复输出。
