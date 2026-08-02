# Python 绑定

本 crate 将 Rust 引擎封装为 `bloodflow_mahjong` Python 扩展，面向模拟、批量环境、重放和数组调用者。游戏规则、合法动作和最终搜索决策都由 Rust 引擎执行，绑定只是接口层。

## 安装

需要 Python 3.10 或更高版本、NumPy 和 Maturin。在仓库根目录执行：

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
python -m pytest engine/pybind/tests
```

PyO3 使用 `abi3-py310`。模块公开 `Game`、`Batch`、事件枚举、常量，以及 `rule-fast`、`rule-ev` 和 `rule-planner` 三种内置策略。后两种策略使用不可变配置对象 `RuleEvConfig` 和 `RulePlannerConfig`。

## 快速开始

```python
import bloodflow_mahjong as bm

game = bm.Game(seed=42)
while (action := game.simple_rule_action()) is not None:
    game.step_id(action)
print(game.scores())
print(game.rankings())
```

`simple_rule_action` 在终局返回 `None`。`step_id` 返回该步的 step record（见下文）。要按观察者视角查看单步结果，用 `observe_into` 和 `events_into`。

`rule_ev_action` 和 `rule_planner_action` 接受可选配置。省略配置时使用默认预算：

```python
ev_config = bm.RuleEvConfig(search_depth=1, defense=True)
planner_config = bm.RulePlannerConfig()

ev_action = game.rule_ev_action(ev_config)
planner_action = game.rule_planner_action(planner_config)
```

`RuleEvConfig` 提供 `fast()` 和 `standard()`。`RulePlannerConfig` 只有一个默认配置：`hand_changes=0`、`draw_horizon=1`、`candidate_states=1`、`belief_worlds=64`、`response_worlds=0`、`search_iterations=64`。构造函数会拒绝超出 core 约束的预算。

## 固定动作空间

策略动作空间固定为 115 维：

| ID | 动作 |
| --- | --- |
| `0..26` | 选择一张换出牌 |
| `27..29` | 选择缺门 |
| `30..56` | 弃牌 |
| `57` | 胡 |
| `58` | 碰 |
| `59` | 直杠 |
| `60..86` | 按牌种选择暗杠 |
| `87..113` | 按牌种选择碰杠 |
| `114` | 在响应窗口中过 |

模块为单动作和每个区间起点公开 `ACTION_*` 常量。合法动作 mask 使用两个 `uint64`，按 little-endian bit 顺序解释。

## `Game` 和 `Batch`

`Game` 提供单局接口：

- reset、阶段、当前决策和合法动作 mask；
- 三种内置策略的单局动作接口和 `step_id`；
- observation、事件和全知 tile count 写入；
- 四家分数、缺门、暗手、锁牌、面子、弃牌、排名和终局原因；
- 信息集重采样。

`Batch` 提供对应的批量 `*_into` 接口，以及：

- 按索引重置、克隆和 swap-remove；
- 批量信息集或 live-wall 重采样；
- 三种内置策略的普通和 masked `*_actions_into` 接口；
- masked step 和融合 step；
- 向听分析；
- 只为指定绝对座位写入事件历史。

数组 dtype、shape、C-contiguous、对齐和内存不重叠是 API 合约。无效数组在状态推进前返回错误。Batch 执行 reset、mask、策略和 step 时释放 GIL，并在 batch 足够大时使用 Rayon。单局 `rule_ev_action` 和 `rule_planner_action` 也会释放 GIL。

## Step record

`records` 的最后一维长度为 12：

| 列 | 内容 |
| ---: | --- |
| `0` | 摸牌玩家，缺失时为 `-1` |
| `1` | 摸到的牌，缺失时为 `-1` |
| `2` | 是否为杠后补牌 |
| `3` | 弃牌玩家，缺失时为 `-1` |
| `4` | 弃牌，缺失时为 `-1` |
| `5..8` | 绝对座位 `0..3` 的分数变化 |
| `9` | 下一行动者，终局时为 `-1` |
| `10` | 下一阶段，终局时为 `-1` |
| `11` | 终局标记 |

Step record 是权威环境数据，不会按观察者隐藏摸牌。策略输入必须使用观察者视角 observation，不能直接把 record 当作可见信息。

## 向听分析

`Game.hand_analysis(seat)` 返回 `(shanten, improving_tiles)`。`improving_tiles` 是 27 bit 牌种 mask。

`Batch.hand_analysis_into` 写入当前行动者的 `int8[batch]` 向听数和 `uint32[batch]` 有效牌 mask。`Batch.hand_analysis_indices_into` 只分析指定 `uint32` 行。

向听数表示还差几次有效进张才能听牌，正常范围从 `SHANTEN_COMPLETE`（`-1`，结构已完成）到 `SHANTEN_MAX`（`8`）。终局 batch 行使用 `SHANTEN_TERMINAL`（`127`）和空 mask。

向听分析覆盖普通四面子一对、该游戏规则中的七对、公开面子和缺门。玩家和牌后，旧结构仍保留在扩展持牌中，因此 `-1` 不代表距离下一次血流和牌只有一步。

## 事件流

一个事件记录包含 8 个 `int32`：

```text
[kind, actor_relative, target_relative, tile, flags, value, aux, reserved]
```

`actor_relative` 和 `target_relative` 使用 observation 的相对座位。`-1` 表示字段不适用。`EventKind(IntEnum)` 提供稳定的事件代码。

| Kind | 字段语义 |
| --- | --- |
| `ACTION` | `value=action_id`，`aux=phase`；只对行动者可见 |
| `GAME_START` | `actor=dealer`，`flags=exchange_direction` |
| `TURN_START` | 庄家首次回合标记，`aux=1` |
| `DRAW` | `actor=drawer`；其他观察者看到 `tile=-1`；flags 可含补牌和最后一张；`value=wall_remaining` |
| `DISCARD` | `actor`、`tile`；flags 可含杠后、开局首弃、摸切和碰后弃牌 |
| `EXCHANGE_COMPLETE` | `flags=exchange_direction` |
| `MISSING_REVEALED` | `actor=player`，`value=missing_suit` |
| `MELD` | `actor`、可选 `target=source`、`tile`，`flags=MeldKind` |
| `HU` | `actor=winner`、可选 `target=source`、`tile`、事件 flags、`value=multiplier`、`aux=PatternSet.bits()` |
| `PAYMENT` | `actor=payer`、`target=payee`、`value=actual_amount` |
| `GAME_END` | 牌墙耗尽时设置最后一张相关 flag |

事件 flag 按位组合，须结合 `kind` 解释。bit 4 和 bit 5 在 `DISCARD` 与 `HU` 之间复用：

| 事件 | bit 4 | bit 5 |
| --- | --- | --- |
| `DISCARD` | 摸切 | 碰后立即弃牌 |
| `HU` | 自摸 | 抢杠胡 |

`EventFlag` 只为 bit 4 和 bit 5 提供了 `SELF_DRAW` 和 `ROB_KONG` 两个名字，语义对应 `HU` 事件。解析 `DISCARD` 时，这两个 bit 表示摸切和碰后弃牌，请直接检查 `1 << 4` 和 `1 << 5`，不要套用这两个名字。其余 flag 为 `REPLACEMENT_DRAW`、`LAST_WALL_TILE`、`AFTER_KONG`、`OPENING_DISCARD`、`HEAVENLY` 和 `EARTHLY`。

`Game.events_into(viewer, output)` 写入最新保留历史并返回长度。`step_events_into` 只写最近一次 step 产生的事件。Batch 版本使用 `int32[batch, capacity, 8]` 和 `uint16[batch]` lengths。

Rust 每局保留 512 条环形记录。`event_dropped` 报告被覆盖的记录数。融合历史接口要求 capacity 在 `1..=512`，并只在下一行动者的绝对座位 bit 被 `history_seat_masks` 选中时写入历史；其他行的 length 为零。

## Observation

Observation 以指定 viewer 为相对座位 `0`，其下家依次为 `1`、`2`、`3`。Batch 默认使用当前行动者；终局没有行动者时使用庄家。

### `tile_obs`

`Game.observe_into` 的 shape 为 `[10, 27]`，Batch 接口的 shape 为 `[batch, 10, 27]`：

| Plane | 内容 |
| ---: | --- |
| `0` | 相对座位 0 的暗手 |
| `1` | 相对座位 0 已选择的换牌 |
| `2` | 相对座位 `0` 的完整锁牌 |
| `3..5` | 相对座位 `1..3` 公开的和牌张，不包含隐藏牌型 |
| `6..9` | 相对座位 `0..3` 的弃牌计数 |

### `melds`

`Game.observe_into` 的 shape 为 `[4, 4, 3]`，Batch 接口的 shape 为 `[batch, 4, 4, 3]`。最后一维为 `[tile, kind, source_relative]`。`kind` 的值为：`0` 碰、`1` 直杠、`2` 碰杠、`3` 暗杠。空槽全部填 `255`。

### `river`

`Game.observe_into` 的 shape 为 `[108, 2]`，Batch 接口的 shape 为 `[batch, 108, 2]`。每项为 `[tile, owner_relative]`，按时间顺序排列。空槽全部填 `255`。

### `meta`

`Game.observe_into` 的 shape 为 `[META_OBSERVATION_WIDTH]`，Batch 接口的 shape 为 `[batch, META_OBSERVATION_WIDTH]`：

| 索引 | 内容 |
| ---: | --- |
| `0` | 阶段 `PHASE_*` |
| `1` | 绝对行动者；终局为 `-1` |
| `2` | 相对庄家 |
| `3` | 换牌方向：左 `1`、对家 `2`、右 `3` |
| `4` | 剩余牌墙数 |
| `5` | 当前 viewer 的摸牌；不可见或不存在时为 `-1` |
| `6` | 杠后补牌标记 |
| `7` | 待响应来源的相对座位；不存在时为 `-1` |
| `8` | 待响应牌；不存在时为 `-1` |
| `9` | 时间顺序牌河长度 |
| `10` | 当前行动者已选择的换牌数 |
| `11` | 当前行动者的换牌花色；不存在时为 `-1` |
| `12..15` | 相对座位 `0..3` 的分数 |
| `16..19` | 相对座位 `0..3` 的缺门；未公开时为 `-1` |
| `20..23` | 相对座位 `0..3` 是否已经和牌 |
| `24..27` | 相对座位 `0..3` 的暗手张数 |
| `28` | 终局标记 |
| `29` | 响应 flags：bit 0 抢杠、bit 1 杠后弃牌、bit 2 开局首弃 |
| `30..33` | 相对座位 `0..3` 的历史最高和牌牌型倍率 |

分配数组时必须使用模块公开的 width 常量。shape 校验会在 schema 变化时立即失败。

## 批量环境

所有高吞吐方法直接写入调用者分配的 C-contiguous NumPy 数组。下面的融合 step 一次完成动作执行、下一步 mask 和 observation 写入，以及下一行动者视角的事件历史：

```python
import numpy as np
import bloodflow_mahjong as bm

batch = bm.Batch(1024, seed=7)
masks = np.empty((len(batch), bm.LEGAL_ACTION_MASK_WORDS), dtype=np.uint64)
actions = np.empty(len(batch), dtype=np.uint8)
records = np.empty((len(batch), bm.STEP_RECORD_WIDTH), dtype=np.int64)
tile_obs = np.empty((len(batch), bm.TILE_OBSERVATION_PLANES, 27), dtype=np.uint8)
melds = np.empty((len(batch), 4, bm.MELD_SLOTS, bm.MELD_FIELDS), dtype=np.uint8)
river = np.empty((len(batch), bm.RIVER_TILE_CAPACITY, bm.RIVER_FIELDS), dtype=np.uint8)
meta = np.empty((len(batch), bm.META_OBSERVATION_WIDTH), dtype=np.int32)
events = np.empty((len(batch), 192, bm.EVENT_RECORD_WIDTH), dtype=np.int32)
event_lengths = np.empty(len(batch), dtype=np.uint16)

# Each bit selects one absolute seat controlled by an external policy.
history_seat_masks = np.full(len(batch), 0b0001, dtype=np.uint8)

batch.legal_action_masks_into(masks)
batch.simple_rule_actions_into(actions)
batch.step_and_observe_history_into(
    actions,
    history_seat_masks,
    records,
    masks,
    tile_obs,
    melds,
    river,
    meta,
    events,
    event_lengths,
)
```

融合 step 写入动作执行后的 observation、下一步 legal mask 和下一行动者视角的事件历史。各数组的 shape 与单局接口相同，只是多了 batch 维。

## 重放和信息集采样

`ENGINE_RULES_VERSION` 标识游戏规则执行和初始化语义。紧凑轨迹至少应保存该版本、seed 和动作序列。版本不同时应拒绝重放，不应猜测迁移。

`Game.resample_information_set(seed)` 返回一个与当前观测一致的隐藏世界采样（determinization），并保持当前行动者的 observation 和 legal mask 不变。普通回合会共同重洗对手未锁暗牌和 live wall。换牌阶段会固定已经选择牌的玩家。响应阶段会固定四家暗手，因为待响应集合本身依赖暗手。

`Batch.resample_information_sets(indices, seeds)` 合并按索引克隆和重采样。重复 index 配合成对 seed，可以让多个候选动作共享同一批隐藏世界。

`resample_live_walls` 只重排尚未摸出的 live wall。它保留四家手牌和全部公开状态，不能替代完整信息集采样。

## 全知 tile count

`oracle_tile_counts_into` 写入 `uint8[..., 9, 27]`：四家绝对座位暗手、四家对应锁牌，以及一个无顺序 live-wall 直方图。

该数组包含完美信息，只能用于模拟、诊断或显式的全知 Critic。它不能进入部署策略输入，也不会改变普通 observation。
