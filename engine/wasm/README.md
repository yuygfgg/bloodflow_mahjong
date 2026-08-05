# WebAssembly 绑定

本 crate 将血流成河规则引擎暴露给 JavaScript / TypeScript。底层 API 与 Python 扩展对称；`js/` 中的 TypeScript 封装再把这些原语整理成客户端使用的观察者相对 UI 快照和 worker 协议类型。

## 目录

| 路径 | 作用 |
| --- | --- |
| `src/lib.rs` | `wasm-bindgen` cdylib；`Game` API 与常量导出 |
| `pkg/` | 构建产物：JS/TS glue 与 `.wasm`（不入库） |
| `js/` | 面向客户端的 TypeScript 封装与快照 schema |
| `js/tests/` | 完整 Node 测试 harness（非常量 smoke） |
| `build.sh` | release 构建并生成 `wasm-bindgen` 产物 |

## 构建

依赖：

- Rust 1.85+，并安装 `wasm32-unknown-unknown` target
- 与 crate 中 `wasm-bindgen` 版本一致的 `wasm-bindgen-cli`

```bash
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli
./engine/wasm/build.sh
```

## 底层 API

| Python | WASM / JS |
| --- | --- |
| `Game(seed)` | `new Game(seed)` |
| `game.step_id(action)` | `game.stepId(action)` |
| `game.observe_into(...)` | `game.observeInto(...)` |
| `game.events_into(viewer, buf)` | `game.eventsInto(viewer, buf)` |
| `game.simple_rule_action()` | `game.simpleRuleAction()` |
| `game.rule_ev_action(cfg)` | `game.ruleEvActionWithConfig(cfg)` |
| `RuleNn(bytes).action(game)` | `new RuleNn(bytes).action(game)` |
| `bm.ACTION_SPACE_SIZE` | `ACTION_SPACE_SIZE()`（init 之后） |

调用方自有缓冲区替代 NumPy 数组：

- `observeInto`：固定宽度的 `Uint8Array` / `Int32Array`
- `eventsInto` / `stepEventsInto`：长度为 `capacity * 8` 的扁平 `Int32Array`
- `stepInto`：长度为 `STEP_RECORD_WIDTH()` 的 `Int32Array`

不提供 `Batch`。浏览器客户端只驱动一局本地对局。

配置对象（`RuleEvConfig` / `RulePlannerConfig`）必须由 JS 侧持有所有权，并通过 `*WithConfig` 借用传入；不要把所有权交给会释放它的 WASM 方法。

## TypeScript 客户端层

```bash
cd engine/wasm/js
npm install
npm test
```

`npm test` 先 typecheck，再跑 `tests/` 下的完整 harness。

`buildUiSnapshot(game, viewer)` 产出观察者相对快照。Worker 消息类型定义在 `protocol.ts`。

## 完整测试 harness

`js/tests/` 覆盖引擎绑定与客户端封装的主路径：

| 文件 | 覆盖 |
| --- | --- |
| `constants.test.ts` | WASM 常量导出完整性与动作布局契约 |
| `game.test.ts` | 缓冲区、非法动作原子性、隐藏摸牌、IS 重采样、终局 |
| `legal.test.ts` | mask 展开与合法性判断 |
| `policies.test.ts` | rule-fast / rule-ev / rule-planner 配置与对局 |
| `rule_nn.test.ts` | ONNX 模型加载与完整对局 |
| `replay.test.ts` | 同 seed + 动作序列的确定性重放 |
| `snapshot.test.ts` | 观察者相对 UI 快照字段与隐藏信息 |

```bash
./engine/wasm/build.sh
cd engine/wasm/js && npm test
```
