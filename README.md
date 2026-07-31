# 血流麻将引擎

本仓库包含确定性的四人血流麻将 Rust 引擎、三种内置策略、平衡策略锦标赛工具，以及可选的 Python 绑定。

本文档统一使用以下术语：

- **游戏规则**：牌局流程、合法动作、牌型和计分。
- **策略**：根据可见状态选择动作的实现。
- **引擎**：执行游戏规则、维护状态并结算分数的程序。

CLI 沿用 `rule-fast`、`rule-ev` 和 `rule-planner` 作为策略标识符。标识符中的 `rule` 不表示存在三套不同的游戏规则。

## 当前范围

当前维护重点是 [`engine/`](engine/) 中的 Rust workspace。系统支持：

- 108 张三门数牌、换三张和定缺；
- 胡后继续行牌，以及和牌结构锁定；
- 碰、直杠、碰杠、暗杠、抢杠胡和一炮多响；
- 全部结构牌型、事件番和即时计分；
- 牌墙耗尽后的查花猪和查大叫；
- 固定种子、115 维动作空间、观察者视角事件和批量环境；
- 三种 Rust 策略，以及任意两种策略之间的平衡测评。

Python 扩展仍可用于模拟和批量数组接口。`training/` 下的神经网络训练代码是独立的实验性 Python 子系统，不属于当前 Rust 主线。相关文档为复现实验而保留，未来可能与当前代码产生偏差。

## 仓库结构

| 路径 | 职责 |
| --- | --- |
| [`GAME_RULES.md`](GAME_RULES.md) | 唯一的游戏规则和计分规范 |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | 确定性约定、状态机和信息边界 |
| [`engine/core`](engine/core/) | 权威游戏状态、计分、分析、批量环境和内置策略 |
| [`engine/pybind`](engine/pybind/) | PyO3 和 NumPy 兼容接口 |
| [`engine/tools/rule-tournament`](engine/tools/rule-tournament/) | 任意两种内置策略的平衡测评工具 |
| [`NEURAL_PLANNER.md`](NEURAL_PLANNER.md) | 神经网络增强 `rule-planner` 的实施方案 |
| [`TRAINING.md`](TRAINING.md) | 保留的 Python 训练实验文档 |

Rust workspace 包含三个 package：

| Package | Crate 或二进制 | 用途 |
| --- | --- | --- |
| `bloodflow-mahjong` | `bloodflow_mahjong` | Rust 库和诊断 benchmark |
| `bloodflow-mahjong-pybind` | `bloodflow_mahjong` | Python 扩展模块 |
| `bloodflow-mahjong-rule-tournament` | `rule-tournament` | 策略锦标赛 CLI |

## 快速开始

需要 Rust 1.85 或更高版本。

```bash
cargo test --manifest-path engine/Cargo.toml --workspace --all-targets

cargo run --manifest-path engine/Cargo.toml --release \
  -p bloodflow-mahjong-rule-tournament -- \
  --blocks 1 \
  --bootstrap-samples 100 \
  --policy-a rule-ev \
  --policy-b rule-fast
```

一个锦标赛 block 包含 6 局。上述命令只验证功能，样本量不足以判断策略强弱。大规模测评前，请先阅读 [`engine/tools/rule-tournament/README.md`](engine/tools/rule-tournament/README.md)。

## Rust 示例

```rust
use bloodflow_mahjong::{Game, GameError};

fn main() -> Result<(), GameError> {
    let mut game = Game::new(42);

    while let Some(action) = game.simple_rule_action() {
        let viewer = game
            .decision()
            .expect("an active game has a decision")
            .actor;
        let outcome = game.step_id(action)?;
        let _visible_outcome = outcome.for_player(viewer);
    }

    Ok(())
}
```

`Game` 是权威模拟状态。它为测试和重放提供全知接口。部署策略必须只读取观察者视角的 observation 和过滤后的 transition。具体边界见 [`IMPLEMENTATION.md`](IMPLEMENTATION.md)。

## 内置策略

| CLI 标识符 | 设计 | 公开接口 |
| --- | --- | --- |
| `rule-fast` | 低成本、确定性的基准策略 | Rust `Game` 和 `Batch`；Python `Game` 和 `Batch` |
| `rule-ev` | 手牌价值、防守启发式和确定性有限前瞻 | Rust `Game` 和 `Batch` |
| `rule-planner` | 手牌图、公开状态价值、信念采样和配对 rollout 改进 | Rust `Game` |

三种策略在锦标赛中地位相同。任意一侧都可以选择任意策略。增加计算预算不等于策略必然更强；预算比较必须使用独立 seed block 和置信区间。

## Python 绑定

Python 绑定需要 Python 3.10 或更高版本、NumPy 和 Maturin。

```bash
maturin develop --release --manifest-path engine/pybind/Cargo.toml
python -m pytest engine/pybind/tests
```

绑定公开 `Game`、`Batch`、压缩合法动作 mask、observation、事件、信息集重采样和 `rule-fast` 策略。绑定不公开 `rule-ev` 或 `rule-planner`。数组格式见 [`engine/pybind/README.md`](engine/pybind/README.md)。

## 验证

从仓库根目录执行：

```bash
cargo fmt --manifest-path engine/Cargo.toml --all -- --check
cargo test --manifest-path engine/Cargo.toml --workspace --all-targets
cargo clippy --manifest-path engine/Cargo.toml --workspace --all-targets -- -D warnings
cargo doc --manifest-path engine/Cargo.toml \
  -p bloodflow-mahjong \
  -p bloodflow-mahjong-rule-tournament \
  --no-deps
```

PyBind cdylib 与 core crate 使用同一个 Rust lib 名称，不能写入同一个 rustdoc 输出目录。因此，rustdoc 命令只生成 core 和 tournament 文档；Python 接口以 [`engine/pybind/README.md`](engine/pybind/README.md) 和 `.pyi` 为准。

仓库中的 benchmark 是独立诊断程序，不是 Cargo benchmark harness。完整命令见 [`engine/README.md`](engine/README.md)。
