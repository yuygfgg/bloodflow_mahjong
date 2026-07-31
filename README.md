# 血流麻将引擎

四人血流麻将 Rust 引擎，附带三种智能体作为虚拟玩家策略、策略锦标赛工具和 Python 绑定。

## 功能

- 108 张三门数牌（万/条/筒），换三张、定缺；
- 碰、直杠、碰杠、暗杠、抢杠胡和一炮多响；
- 胡后继续行牌，和牌结构锁定，可重复胡；
- 全部结构牌型、状态/事件番与即时计分；
- 牌墙耗尽后查花猪、查大叫，分数下限为 0；
- 115 维固定动作空间、观察者视角事件和批量环境；
- 三种内置策略和任意两两平衡测评。

完整玩法与计分规范见 [`GAME_RULES.md`](GAME_RULES.md)。


## 文档

| 文档 | 内容 |
 --- | --- |
| [`GAME_RULES.md`](GAME_RULES.md) | 玩法和计分规范 |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | 状态机、step 语义、115 维动作空间、信息边界、计分执行 |
| [`engine/README.md`](engine/README.md) | workspace 布局、构建与验证命令、诊断 benchmark |
| [`engine/core/README.md`](engine/core/README.md) | `Game`/`Batch`/策略/手牌分析的 API 与配置示例 |
| [`engine/pybind/README.md`](engine/pybind/README.md) | 安装、数组规格、事件与 observation 格式 |
| [`engine/tools/rule-tournament/README.md`](engine/tools/rule-tournament/README.md) | 统计方法、全部 CLI 参数和输出解释 |

## 仓库结构

| 路径 | 职责 |
| --- | --- |
| [`engine/core`](engine/core/) | 游戏状态、计分、分析、批量环境和内置策略 |
| [`engine/pybind`](engine/pybind/) | PyO3 和 NumPy 兼容接口 |
| [`engine/tools/rule-tournament`](engine/tools/rule-tournament/) | 任意两种内置策略的平衡测评工具 |

Rust workspace 包含三个 package：

| Package | Crate / 二进制 | 用途 |
| --- | --- | --- |
| `bloodflow-mahjong` | `bloodflow_mahjong` | Rust 库和诊断 benchmark |
| `bloodflow-mahjong-pybind` | `bloodflow_mahjong` | Python 扩展模块 |
| `bloodflow-mahjong-rule-tournament` | `rule-tournament` | 策略锦标赛 CLI |

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

`Game` 是权威模拟状态，为测试和重放提供全知接口。部署策略必须只读取观察者视角的 observation 和过滤后的 transition，见 [`IMPLEMENTATION.md`](IMPLEMENTATION.md) "信息边界"一节。更多 API 见 [`engine/core/README.md`](engine/core/README.md)。

## 内置策略

| CLI 标识符 | 设计 | 公开接口 |
| --- | --- | --- |
| `rule-fast` | 低成本、确定性的基准策略 | Rust `Game` 和 `Batch`；Python `Game` 和 `Batch` |
| `rule-ev` | 手牌价值、防守启发式和确定性有限前瞻 | Rust `Game` 和 `Batch` |
| `rule-planner` | 手牌图、公开状态价值、信念采样和配对 rollout 改进 | Rust `Game` |

三种策略的强弱对比用锦标赛工具评估，统计方法见 [`engine/tools/rule-tournament/README.md`](engine/tools/rule-tournament/README.md)。

## Python 绑定

需要 Python 3.10 或更高版本、NumPy 和 Maturin。绑定公开 `Game`、`Batch`、压缩合法动作 mask、observation、事件、信息集重采样和 `rule-fast` 策略，不公开 `rule-ev` 或 `rule-planner`。安装、数组格式和示例见 [`engine/pybind/README.md`](engine/pybind/README.md)。

## 构建与验证

完整的 fmt、test、clippy、doc 命令和诊断 benchmark 见 [`engine/README.md`](engine/README.md)。Python 扩展的构建与测试见 [`engine/pybind/README.md`](engine/pybind/README.md)。
