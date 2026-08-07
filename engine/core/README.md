# Core Rust Crate

`bloodflow-mahjong` 为血流麻将引擎核心库：`Game` 执行游戏规则，`Batch` 批量执行独立牌局，使用内置策略从合法动作中选择一个动作。

游戏规则见 [`../../GAME_RULES.md`](../../GAME_RULES.md)，确定性约定和信息边界见 [`../../IMPLEMENTATION.md`](../../IMPLEMENTATION.md)。

当前 `ENGINE_RULES_VERSION` 为 `10`。首次和牌锁定完整和牌基础；历史和牌张单独公开，胡后普通回合只能摸切，但仍可暗杠或碰杠。在一个弃牌响应窗口中，每名候选玩家只提交一次动作；有任何胡候选时，合法动作可以同时包含胡、碰、直杠和过，胡会覆盖所有面子选择。没有胡候选时才使用 `MeldResponse`。版本 7、8 和 9 的轨迹与训练模型与当前规则不兼容。

## 单局循环

```rust
use bloodflow_mahjong::{Game, GameError};

fn play(seed: u64) -> Result<Game, GameError> {
    let mut game = Game::new(seed);

    while let Some(action) = game.simple_rule_action() {
        game.step_id(action)?;
    }

    Ok(game)
}
```

`Game::legal_actions()` 返回结构化动作，`Game::legal_action_mask()` 返回固定 115 维动作 mask。`step()` 接受 `Action`，`step_id()` 接受 `ActionId`。终局后这些查询返回 `None`，继续 step 返回 `GameError::Finished`。

非法动作不得修改状态。需要重放时，保存 `ENGINE_RULES_VERSION`、初始 seed 和完整动作 ID 序列。

## 手牌分析

公开函数包括：

- `is_winning`：判断指定持牌是否包含合法和牌结构；
- `evaluate_win`：选择倍率最高的和牌拆分；
- `analyze_shanten`：返回结构向听数和有效牌 mask；
- `evaluate_max_wait`：返回查大叫所需的最大牌型与根倍率，不计算状态/事件番。

向听数表示还差几次有效进张才能听牌的结构指标；`SHANTEN_COMPLETE`（`-1`）表示当前结构已经完成。引擎分析玩家状态时会先排除历次和牌张。结果始终表示当前普通手牌距离下一次和牌的结构距离。

## 信息边界

`Game` 是全知模拟状态。以下数据不应直接交给部署策略：

- 其他玩家暗手；
- 牌墙顺序；
- 未公开的换牌选择；
- 未经过观察者过滤的摸牌 transition。

调用 `StepOutcome::for_player(viewer)` 过滤单步 transition。调用 `Game::observation_into` 和事件接口生成观察者视角数组。

`resample_information_set(seed)` 从当前行动者的信息集中采样。`resample_live_wall(seed)` 只重排未来牌墙，不改变任何玩家手牌。两者目标不同，不能互换。

## 策略 API

core 提供两种不依赖模型文件的规则策略：`rule-fast` 是轻量基准，`rule-ev` 是中等预算。feature `rule-nn` 额外提供 ONNX 神经网络策略。

### `rule-fast`

```rust
let action = game.simple_rule_action();
```

该策略使用固定启发式、向听数和有效牌。`Batch::simple_rule_actions_into` 提供批量接口。Python 绑定也公开该策略。

### `rule-ev`

```rust
use bloodflow_mahjong::{RuleEvConfig, RuleEvDefense};

let config = RuleEvConfig::with_search_depth(1)
    .expect("depth is in 0..=3")
    .with_defense(RuleEvDefense::Heuristic);
let action = game.rule_ev_action_with_config(config);
```

`search_depth` 控制确定性手牌前瞻：枚举公开有效牌和后续弃牌，不采样隐藏牌，也不读取权威牌墙。`RuleEvConfig::STANDARD` 使用 depth 1 和启发式防守。

`Batch::rule_ev_actions_into` 和 `Batch::rule_ev_actions_with_config_into` 提供批量接口。

### `rule-nn`

`RuleNn` 通过 `tract-onnx` 加载 Actor 图。core 默认不启用该依赖；调用者必须启用 feature `rule-nn`。调用者必须提供规则版本 10 重新训练的模型；仓库中的旧 [`../../model/latest.onnx`](../../model/latest.onnx) 不能用于当前引擎。模型应在进程启动时加载一次，并在后续决策中复用。

```rust
use std::error::Error;

use bloodflow_mahjong::{Game, RuleNn};

fn main() -> Result<(), Box<dyn Error>> {
    let bytes = std::fs::read("/path/to/rules-v10.onnx")?;
    let policy = RuleNn::from_onnx_bytes(&bytes)?;
    let game = Game::new(7);
    let action = policy.action(&game)?;
    println!("{action:?}");
    Ok(())
}
```

模型使用固定 batch size 1 的接口。模型 metadata 必须包含 `engine_rules_version=10`；缺少 metadata 或版本不匹配时，加载立即失败：

| 名称 | dtype | shape |
| --- | --- | --- |
| `tile_obs` | `uint8` | `[1, 11, 27]` |
| `melds` | `uint8` | `[1, 4, 4, 3]` |
| `meta` | `int32` | `[1, 34]` |
| `events` | `int32` | `[1, 192, 8]` |
| `event_lengths` | `int64` | `[1]` |
| `logits` | `float32` | `[1, 115]` |

ONNX 图只输出未屏蔽的 Actor logits。`RuleNn::action` 从当前行动者视角生成 observation 和事件历史，再使用引擎的 legal mask 选择最高有限 logit。模型不能提交非法动作。相同 logit 使用最小动作 ID。`Batch::rule_nn_actions_into` 和 masked 版本复用一个不可变模型，并通过 Rayon 并行执行每个固定 batch-size 1 的图。

## Batch

`Batch` 的高吞吐接口由调用者分配输出缓冲区。常用能力包括：

- `legal_action_mask_words_into`；
- `step_ids`、masked step 和 indexed step；
- `observations_into` 和事件写入；
- 融合 step、observation 与完整事件历史；
- 按索引克隆和信息集重采样。

方法会检查长度。Python 绑定还检查 dtype、shape、C-contiguous、对齐和重叠 view。

## 构建

```bash
cargo test --manifest-path ../Cargo.toml -p bloodflow-mahjong --all-targets
cargo test --manifest-path ../Cargo.toml -p bloodflow-mahjong \
  --all-targets --features rule-nn

cargo run --release --manifest-path ../Cargo.toml \
  -p bloodflow-mahjong \
  --features rule-nn \
  --example rule_nn_smoke -- \
  /path/to/rules-v10.onnx
```
