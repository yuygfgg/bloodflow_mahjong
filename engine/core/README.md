# Core Rust Crate

`bloodflow-mahjong` 为血流麻将引擎核心库：`Game` 执行游戏规则，`Batch` 批量执行独立牌局，使用内置策略从合法动作中选择一个动作。

游戏规则见 [`../../GAME_RULES.md`](../../GAME_RULES.md)，确定性约定和信息边界见 [`../../IMPLEMENTATION.md`](../../IMPLEMENTATION.md)。

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

向听数表示还差几次有效进张才能听牌的结构指标；`SHANTEN_COMPLETE`（`-1`）表示当前结构已经完成。玩家和牌后，旧结构仍保留在扩展持牌中，因此 `-1` 不代表距离下一次血流和牌只有一步。

## 信息边界

`Game` 是全知模拟状态。以下数据不应直接交给部署策略：

- 其他玩家暗手；
- 牌墙顺序；
- 未公开的换牌选择；
- 未经过观察者过滤的摸牌 transition。

调用 `StepOutcome::for_player(viewer)` 过滤单步 transition。调用 `Game::observation_into` 和事件接口生成观察者视角数组。

`resample_information_set(seed)` 从当前行动者的信息集中采样。`resample_live_wall(seed)` 只重排未来牌墙，不改变任何玩家手牌。两者目标不同，不能互换。

## 策略 API

三种策略按计算预算递进：`rule-fast` 是轻量基准，`rule-ev` 是中等预算，`rule-planner` 的预算最高。

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

`Batch::rule_ev_actions_into` 提供批量接口。隐藏世界搜索只由 `rule-planner` 实现。

### `rule-planner`

```rust
use bloodflow_mahjong::RulePlannerConfig;

let config = RulePlannerConfig::STANDARD
    .with_hand_changes(0)
    .expect("hand changes are in 0..=2")
    .with_draw_horizon(1)
    .expect("draw horizon is in 0..=32")
    .with_candidate_states(1)
    .expect("candidate states are in 1..=200000")
    .with_belief_worlds(64)
    .expect("belief worlds are in 0..=256")
    .with_response_worlds(0)
    .expect("response worlds are in 0..=256")
    .with_search_iterations(64)
    .expect("iterations are in 0..=4096");
let action = game.rule_planner_action_with_config(config);
```

planner 先评估手牌候选图和公开状态价值，再决定动作。`belief_worlds` 控制危险度和 rollout 使用的信息集粒子数。`search_iterations` 非零时，策略对候选动作做配对 rollout——固定其余三座位的行动不变，只改进当前动作——并用独立粒子流验证改动。

planner 当前没有 `Batch` 或 Python 动作接口。局级并行由调用者或 `rule-tournament` 管理。

#### 分析接口

feature `planner-analysis` 额外公开 `RulePlannerRootBelief`、`RulePlannerAnalysisOptions`、冻结 continuation profile 和两组 `Game::rule_planner_analysis_*` 接口。`with_config` 接口保留当前生产 continuation；`with_options` 接口可以独立选择根 belief 和 continuation model。结构化结果包含 baseline、proposal、validation 结果和真实执行的 rollout 计数。

`RulePlannerContinuationProfile` 为四个座位分别指定 `Fast`、`Ev` 或 `PlannerBaseline`。profile 是封闭策略集合，不接受任意回调；`PlannerBaseline` 会关闭 paired root search，但保留其余 planner 配置。这样保证一次根策略改进具有固定 continuation，并防止 rollout 内递归搜索。每个 continuation 策略只读取当前行动者的普通 observation 输入。

`OracleHidden` 读取权威隐藏状态，`KnownPolicies` 读取评测器提供的策略身份。两者只能用于诊断，不能用于部署或正式策略成绩。

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
  --all-targets --features planner-analysis
```
