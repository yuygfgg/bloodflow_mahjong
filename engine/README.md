# Rust Workspace

## Workspace 成员

| 目录 | Package | 产物 |
| --- | --- | --- |
| [`core/`](core/) | `bloodflow-mahjong` | Rust 库和四个诊断 benchmark |
| [`pybind/`](pybind/) | `bloodflow-mahjong-pybind` | Python 扩展 `bloodflow_mahjong` |
| [`tools/rule-tournament/`](tools/rule-tournament/) | `bloodflow-mahjong-rule-tournament` | `rule-tournament` CLI |

workspace 使用 Rust 2024 edition，最低 Rust 版本为 1.85。release profile 使用单 codegen unit、fat LTO 和 `panic=abort`。

## 构建和检查

在本目录执行：

```bash
cargo fmt --all -- --check
cargo build --workspace --all-targets
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo doc \
  -p bloodflow-mahjong \
  -p bloodflow-mahjong-rule-tournament \
  --no-deps
```

PyBind cdylib 与 core crate 使用同一个 Rust lib 名称，不能共用 rustdoc 输出路径。Python API 以 [`pybind/README.md`](pybind/README.md) 和 `bloodflow_mahjong.pyi` 为准。

需要完整 release 产物时执行：

```bash
cargo build --release --workspace --all-targets
```

## Core 能力

`bloodflow_mahjong` 禁止 unsafe code，并公开以下接口组：

- `Game`、`LegalActions`、`StepOutcome` 和 `GameError`；
- `Action`、`ActionId`、`ActionMask` 和 115 维动作常量；
- `Tile`、`Suit`、`Seat`、`Meld` 和相关枚举；
- 和牌、牌型、向听数和最大待牌倍率分析；
- `Batch` 批量环境和 caller-owned `*_into` 缓冲区接口；
- `rule-fast`、`rule-ev` 和 `rule-planner` 三种策略。

详细 Rust API 和配置示例见 [`core/README.md`](core/README.md)。状态机和信息边界见 [`../IMPLEMENTATION.md`](../IMPLEMENTATION.md)。

## 策略测评

`rule-tournament` 支持三种内置策略的任意两两组合。每个 block 运行 6 局平衡座位分配，并按 block bootstrap 置信区间（以整个 block 为重采样单位）。

```bash
cargo run --release -p bloodflow-mahjong-rule-tournament -- \
  --blocks 1 \
  --bootstrap-samples 100 \
  --policy-a rule-ev \
  --policy-b rule-fast
```

该命令是 smoke test。统计方法、预算参数和长测命令见 [`tools/rule-tournament/README.md`](tools/rule-tournament/README.md)。

## 诊断 benchmark

四个 benchmark 都是普通二进制程序，不是 `cargo bench` harness。

```bash
# Single-game throughput. The positional argument is the game count.
cargo run --release -p bloodflow-mahjong --bin throughput-benchmark -- 50000

# Batch throughput. Positional arguments are environments and rounds.
cargo run --release -p bloodflow-mahjong --bin batch-throughput-benchmark -- 1024 4096

# Caller-owned flat-buffer hot path. This is not a C ABI benchmark.
cargo run --release -p bloodflow-mahjong --bin ffi-throughput-benchmark -- 1024 4096

# Maximum-wait analysis throughput. This program takes no arguments.
cargo run --release -p bloodflow-mahjong --bin max-wait-benchmark
```

benchmark 只用于定位性能变化。策略强弱必须通过平衡锦标赛测量。

## Python 绑定

安装、数组格式和示例见 [`pybind/README.md`](pybind/README.md)。绑定需要 Python 3.10 或更高版本，并公开 `rule-fast`、`rule-ev` 和 `rule-planner` 三种策略。
