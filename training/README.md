# 训练

当前训练器使用合成完整对局上的离线到在线 IQL/AWR。Actor 从已有冻结 SL 策略初始化；独立的 Q1/Q2/V 学习四个座位的终局分差 return-to-go，通过 Critic 校准门控后才更新 Actor。

数据循环为：

```text
规则 / 冻结 SL / 当前 / 历史策略完整对局
  -> 紧凑轨迹 replay
  -> Double-Q + Expectile V
  -> AWR + frozen-SL KL
  -> 新轨迹和历史快照
```

Actor 约 8.84M 参数，只读取行动者可见状态。Q1、Q2、V 使用各自独立的小型 Transformer，不与 Actor 或彼此共享编码器。奖励只使用真实累计分差 `score_delta / 10_000`。训练覆盖换三张三步、定缺、早中晚回合、胡响应和副露响应全部九类决策。

实验 C 的信息集 MC 是普通 Critic 的选择性辅助：只有 partial Critic 基础门控通过后才查询高不确定状态。候选共享 paired worlds，只有回报差置信区间排除零且保守差距至少 0.02 的动作边才进入 teacher。每个 query 按完整可靠边图接收或淘汰；train 和 validation 使用不同 corpus，validation 只来自永久 anchor 轨迹，并在同时达到 512 targets、128 reliable groups、128 reliable pairs 后冻结。MC 只额外更新 Q1/Q2 的可靠动作差值，普通完整轨迹仍训练绝对 Q、V、CQL，Actor 不把 MC 目标当作硬分类标签。

完整的统计依据、损失、replay 规则、Critic 门控和 A/B/C 实验说明见 [TRAINING.md](../TRAINING.md)。

## 准备

安装 release Python 扩展：

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
```

训练只支持 CUDA，不会回退 CPU。默认直接复用：

```text
runs/counterfactual-larger/sl_reference.pt
```

该文件必须是 Actor-only 格式，顶层只包含 `model_config` 与 `model`。训练入口直接用它初始化 Actor，并保留一份冻结参考策略。

## 启动

默认基线 A：

```bash
python -m training.train \
  --experiment a \
  --output-dir runs/iql-awr-a-v3 \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt
```

Oracle Critic 实验 B：

```bash
python -m training.train \
  --experiment b \
  --output-dir runs/iql-awr-b-v3 \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt
```

选择性信息集 Monte Carlo 实验 C：

```bash
python -m training.train \
  --experiment c \
  --output-dir runs/iql-awr-c-v3 \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt
```

三组实验应使用相同 GPU 时间分别比较。A 是默认方案；B 和 C 是用于定位 Critic 方差来源的递进实验，不应一开始就混成一组。

正式运行会无限循环，直到用户按 `Ctrl+C`，随后保存 `latest.pt`。

## CUDA Smoke Test

```bash
python -m training.train \
  --smoke \
  --output-dir /tmp/bloodflow-iql-smoke \
  --sl-checkpoint runs/counterfactual-larger/sl_reference.pt
```

Smoke 模式缩小环境、Critic、replay、MC 和评测规模，在一个 iteration 后保存退出。它用显式的小样本配置和宽松门槛覆盖 Actor、Oracle 与 MC 路径，不会在控制流里伪造 teacher ready。不要恢复 smoke checkpoint 做长训。Smoke 仍要求 CUDA。

## 恢复

```bash
python -m training.train --resume runs/iql-awr-a-v3/latest.pt
```

恢复时从 checkpoint 读取实验和全部配置，并校验同目录 `replay/manifest.json` 与 shard。它恢复 Actor、冻结参考策略、Q1/Q2/V、可选 Oracle、优化器、教师连续就绪计数、策略池、采样 RNG、replay cursor、采集 seed 和全局随机状态，不重新执行 SL。当前只接受 v3 checkpoint；v1/v2 不兼容，不做配置默认填充或迁移。

恢复时不要同时指定另一份 `--sl-checkpoint`，也不能把 A/B/C checkpoint 改成其他实验类型。

C 实验恢复：

```bash
python -m training.train --resume runs/iql-awr-c-v3/latest.pt
```

C 的关键诊断字段位于 `mc_critic.*`、`mc.*` 和 `mc.validation_metrics.action_ranking.*`：分别查看 train/validation pairwise accuracy、reliable/all pair 数、reliable group 数、`train_targets_after_trim` 和 `validation_frozen`。MC Q MAE 与 `absolute_loss` 仅用于观察，不参与 MC teacher gate 或 MC 优化目标。

## 常用参数

```text
--envs                    并行采集环境数
--anchor-games            新 run 的永久 anchor 对局数
--games-per-iteration     每轮新增完整对局数
--critic-batch-size       Critic batch
--actor-batch-size        AWR batch
--microbatch-size         显存控制优先调整项
--initial-critic-steps    Actor 首次门控前的 Critic warmup
--critic-steps            每轮 Critic steps
--actor-steps             门控通过后的每轮 AWR steps
--eval-every              评测间隔
--eval-games              每个 seed、每种对手 panel 的对局数
--checkpoint-every        latest checkpoint 间隔
--verbose-console         终端额外输出完整 JSON
```

默认 `--envs` 为 512。策略来源行数和历史宽度会自动 pad 到稳定的推理 bucket，并启用 expandable allocator segments，避免长时间采集中的 CUDA 显存碎片。显存仍不足时，训练更新优先降低 `--microbatch-size`，采集或评测前向则降低 `--envs`。不要用 CPU smoke 或 CPU 小实验替代 CUDA 验证。

## CUDA 吞吐配置

当前 Actor、采集策略、冻结 SL、历史 Actor、Q1/Q2/V 和 Oracle 全部使用 CUDA eager，不调用 `torch.compile`。本机的 `max-autotune` 曾触发 NVIDIA GSP 通道分配失败；普通动态 compile 也能在固定 seed 上复现非有限 logits，而 eager 对照完整通过。因此所有 compile 路径已从代码删除，启动时会明确打印 eager 状态。

RTX 5080 eager 实测中，`envs=512` 约为 9,800 states/s，`envs=1024` 约为 8,195 states/s。默认 `microbatch-size=256` 最快且不会 OOM，384 没有收益，512 会使 Actor OOM。因此不要为了填满显存而增大 `envs` 或 microbatch。

Critic 和 Actor 更新会在 CUDA 训练当前 batch 时后台重建下一批 replay。只有实际采样的步骤生成 observation/history/legal，完整 optimizer batch 只执行一次 H2D，随后在 CUDA 上切 microbatch view。RTX 5080 上默认 Critic 从原先约 `0.67 s/update` 降到约 `0.45 s/update`；64-step block 约从 43 秒降到 29 秒。collector 当场生成的轨迹不会再由同一引擎重复验证，外部轨迹仍在入库时严格重放，加载仍校验版本与 CRC。

历史策略最多保留 `max_history=16` 个 CUDA 模型；快照被淘汰时，其模型引用会同步释放。

## 输出

每个 run 目录包含：

- `latest.pt`：最近完整 checkpoint；
- `best.pt`：固定和 fresh seed 评测都提升时的 Actor-only 部署权重；
- `metrics.jsonl`：完整训练、Critic、Actor、replay、MC 和评测记录；
- `dashboard.html`：本地摘要页面；
- `config.json`：模型、训练配置和评测 seeds；
- `replay/manifest.json`、`replay/*.bfsh`：紧凑轨迹 replay；
- `snapshots/*.pt`：历史 Actor-only 对手。

终端默认只保留关键摘要。恢复或复盘应以 `metrics.jsonl` 和 checkpoint 为准，不从终端文本解析状态。

已有日志可以重新生成 dashboard：

```bash
python -m training.dashboard runs/iql-awr-a/metrics.jsonl
```

## 测试和基准

```bash
python -m pytest training/tests
python -m training.benchmarks.transformer --device cuda
python -m training.benchmarks.train_step --device cuda
```
