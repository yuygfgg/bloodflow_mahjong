# Blood Flow Mahjong Engine

高性能、确定性的血流麻将规则引擎。规则来源为 [`rules.md`](rules.md)，Rust workspace 位于 [`engine`](engine)，分为纯 Rust [`core`](engine/core) 和 Python [`pybind`](engine/pybind)。

## 当前能力

- 108 张三门数牌、换三张、定缺
- 摸打、碰、直杠、碰杠、暗杠、抢杠胡和一炮多响
- 胡后锁牌并继续行牌，锁定牌可在后续胡牌和碰杠中复用；和牌后不能暗杠
- 全部结构牌型、独立事件番、即时杠分和自摸/点炮结算
- 查大叫、查花猪、最低 0 分和三家 0 分提前终局
- 115 维固定 Legal Action Mask、固定种子重放、Rayon 批量环境和无分配的 step 输出
- 固定宽度 viewer-scoped 事件流，支持完整历史环形缓冲和每步事件 delta

## Rust API

```rust
use bloodflow_mahjong::{ActionId, Game, GameError};

fn run() -> Result<(), GameError> {
    let mut game = Game::new(42);
    while let Some(mask) = game.legal_action_mask() {
        let action: ActionId = policy_action(&game, mask);
        let actor = game.decision().expect("active game has a decision").actor;
        let outcome = game.step_id(action)?;

        // 原始 outcome 属于全知环境；交给某个玩家前过滤暗摸牌面。
        let player_outcome = outcome.for_player(actor);
        consume_transition(player_outcome);
    }
    Ok(())
}
```

`ActionMask` 用两个 `u64` 表示 115 个固定动作，可通过 `words()` 零分配读取，也可通过 `to_dense()` 得到训练常用的 0/1 数组。换三张按每名玩家连续三次选一张实现，后两次 mask 只开放首张同花色且仍持有的牌；定缺保持一次三选一。`Batch::legal_action_masks_into` 和 `Batch::step_ids` 可直接用于并行环境。

正常轮次不存在空过：引擎强制摸牌，玩家随后必须弃牌、胡牌或杠。`Action::Pass` 只在其他玩家出牌或抢杠响应窗口合法。`StepOutcome` 明确返回本步摸牌、弃牌、分数变化和下一决策。

## Python API

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
```

训练热路径使用 `Batch.step_and_observe_history_into`：一次 Rayon 遍历完成动作、transition、下一状态观测、合法动作 mask 和 viewer-scoped 完整历史。`history_seat_masks[B]` 按绝对座位选择哪些下一行动者需要历史，因此 bootstrap 阶段通常只写 learner 的约四分之一行。终局由 `Batch.reset_and_observe_history_into` 只刷新实际 reset 的行，向听辅助标签由 `Batch.hand_analysis_indices_into` 只计算 learner 行；这些调用都释放 GIL 且不创建 Python 对象列表。

观测由当前行动者视角编码，不暴露其他玩家暗手、墙序以及尚未统一公开的换牌/定缺选择。事件记录为八个 `int32` 字段，摸牌牌面只对摸牌者可见，响应动作不向其他玩家泄露。完整数组布局见 [`engine/pybind/README.md`](engine/pybind/README.md)。

## 验证

```bash
cargo test --manifest-path engine/Cargo.toml --workspace --all-targets
cargo clippy --manifest-path engine/Cargo.toml --workspace --all-targets -- -D warnings
cargo run --manifest-path engine/Cargo.toml --release -p bloodflow-mahjong --bin throughput-benchmark -- 50000
cargo run --manifest-path engine/Cargo.toml --release -p bloodflow-mahjong --bin batch-throughput-benchmark -- 1024 4096
cargo run --manifest-path engine/Cargo.toml --release -p bloodflow-mahjong --bin ffi-throughput-benchmark -- 1024 4096
python -m pytest engine/pybind/tests
python engine/pybind/benchmarks/throughput.py --batch-size 1024 --iterations 4096
```

实现约定和验收矩阵见 [`IMPLEMENTATION.md`](IMPLEMENTATION.md)。

无真人牌谱条件下的模型候选、冷启动、自博弈 PPO 和评测方案见 [`TRAINING.md`](TRAINING.md)。

首版 Transformer 模型、静态状态编码器、事件 GPT 历史塔和 KV cache 位于 [`training`](training)，实现说明和 smoke test 见 [`training/README.md`](training/README.md)。
