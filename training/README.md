# 训练

当前只有一条正式训练路径：CUDA conservative policy iteration。它直接加载已有 SL Actor，每个策略版本重新采集全新独立状态，对所有合法动作生成配对 live-wall 续局回报，累积一个完整大 batch 的梯度，只执行一次 AdamW step，再把更新缩放到固定 KL。

默认一轮使用九类决策各 256 个状态，共 2304 个互不共享隐藏牌局的状态；每个状态使用 16 个配对世界。2304 是固定批量，不是固定数据集。完整轮次提交后训练目标会删除，下一轮从新策略和新 seeds 重新采集。

另有九类各 64 个、共 576 个独立状态只用于 KL 校准；它们不参与 Actor 梯度。生产更新没有把训练 batch 降到 576。

详细算法、统计边界和参数见 [TRAINING.md](../TRAINING.md)。

## 准备

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
```

训练只支持 CUDA eager mode，不回退 CPU，也不使用 `torch.compile`。SL 输入必须是 Actor-only checkpoint，顶层严格包含 `model_config` 和 `model`。

## 正式训练

```bash
python -m training.train \
  --output-dir runs/policy-iteration-v3 \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt
```

训练无限运行，直到按 `Ctrl+C`。终端会持续显示阶段进度、吞吐、elapsed 和 ETA；每个完整 iteration 提交后打印固定规则配对评测摘要及当前/下一轮 self-play 比例。固定规则评测确认相对 SL 的 paired `dRank` 显著改善且 score 没有显著伤害后启用 10% self-play；累计 `dRank` 每改善 `0.01` 提高一档，最多占两个对手席位。self-play 对手来自历史池，而非当前 learner：池以 SL 初始化，每 8 次提交加入刚完成的 Actor，保留最近 4 个快照并逐轮轮换。固定评测本身始终保持规则对手。

恢复：

```bash
python -m training.train \
  --resume runs/policy-iteration-v3/latest.pt
```

`latest.pt` 只包含完整提交的 iteration 和历史 opponent pool。中断时，未完成 iteration 的目标分片保留在 `pending/`；恢复会确定性重建同一批状态并补齐分片。恢复不接受 seed 或配置覆盖；生产 v3 checkpoint 可直接恢复，并先以原有同模型语义补齐当前 pending iteration，首次提交后升级为历史池 v4。早于 v3 的旧格式仍不迁移。当前执行版本必须使用新的空输出目录，不能混用 batching 优化前的缓存。

## Batch Size Sweep

正式 sweep 比较每类 `64/128/256/512` 个状态，即总 batch `576/1152/2304/4608`。Sweep 使用独立的 1152 状态 KL calibration：

```bash
python -m training.batch_sweep \
  --output-dir runs/batch-sweep-v3 \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt
```

正式比较建议使用三个独立 seed：

```bash
python -m training.batch_sweep \
  --output-dir runs/batch-sweep-v3-multiseed \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt \
  --seeds 20260727 20260728 20260729
```

同一命令可恢复；各 seed 独立缓存，最终在根目录生成 pooled `aggregate.json` 和 seed 间波动统计。

训练和 sweep 的 target rollout 都默认以 64-query group 执行：Rust 引擎在组内使用 Rayon 多核，Actor 以最多 128 行的 CUDA batch 推理。单 GPU 不并发多个 seed 进程，避免复制模型、争用显存和 CUDA context；多 seed 仍按顺序运行并分别恢复。

各 batch 使用同一最大嵌套训练 corpus、独立 calibration 和 heldout corpus，以及同一 16384 局固定规则面板。输出包括方向一致性、有效样本量、heldout policy value、paired `dRank`、score、耗时和吞吐。

同一命令可恢复共享 corpus、每个 batch 的方向、candidate 原始评测面板和已完成候选。快速验证控制流：

```bash
python -m training.batch_sweep \
  --output-dir /tmp/batch-sweep-smoke \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt \
  --smoke
```

Smoke 仍要求 CUDA，只证明管线能完整执行，不证明策略提升。

## 输出

正式训练目录：

- `latest.pt`：唯一权威恢复点；
- `actor.pt`：Actor-only 部署权重；
- `config.json`：不可变 run 配置和输入身份；
- `metrics.jsonl`：每轮 source、target、optimizer、calibration 和 evaluation 指标；
- `reference_panel.npz`：冻结 SL 的固定规则面板；
- `pending/`：仅用于恢复未完成目标生成。

Sweep 目录额外包含 `summary.json`、`actor-qpc*.pt`、`shared/` corpus、`directions/`、`evaluation/`、训练/heldout target shard 和 reference panel。

## 验证

```bash
python -m py_compile training/*.py training/tests/*.py
python -m pytest training/tests -q
python -m training.benchmarks.transformer --device cuda
```

单元测试可以检查数据和算法不变量；实际训练、性能判断和端到端 smoke 必须在 CUDA 上完成。
