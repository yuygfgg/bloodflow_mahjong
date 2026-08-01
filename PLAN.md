# 可见局面复盘：严格可形成校验与强类型 REPL

## 摘要

core 接受可见局面，构造一个与该局面投影一致的合法 `Game`，再调用现有策略给出最佳动作。REPL 用紧凑命令录入公开事件和 viewer 私有信息。

任何被接受的会话状态都必须满足以下条件：至少存在一个完整、规则合法的 `Game`，其公开信息与 viewer 私有信息投影等于当前可见局面。解析失败、转移非法或无法具体化时，REPL 拒绝整行输入并保持状态不变。

## 目标

- 常用事件只输入无法从牌局状态推导的信息。
- 每类命令只有一种签名。actor 要么必填，要么禁止输入。
- 建议输出使用 REPL 的规范记号，并可直接作为下一条输入执行。
- 用户可以在换三张、定缺、出牌和响应阶段请求一个最佳动作。
- viewer 和牌结构由已知手牌计算。非 viewer 和牌使用结构简称。

## 不包含的范围

- 不扩展 Python pybind。
- 不默认输出 `planner-analysis` 的详细分析。
- 不反推唯一开局 seed，也不重放隐藏牌历史。
- 不验证非 viewer 的隐藏手牌是否真能组成其声明的 `ShapeSpec`。
- 不实现第二套完整规则引擎。core 的既有规则仍是最终依据。

## 座位模型

### 决定：REPL 使用相对座位

REPL 的数字不绑定东、南、西、北。数字以 viewer 为原点，并沿引擎行牌方向递增：

| `RelativeSeat` | 含义 |
| --- | --- |
| `0` | viewer，也就是“我” |
| `1` | viewer 的下一家 |
| `2` | viewer 的对家 |
| `3` | viewer 的上一家 |

相对座位适合实时复盘。用户不需要先输入自己的绝对方位，同一套输入在换座后仍保持相同含义。策略 observation 也使用 viewer 相对视角。

绑定东、南、西、北只在跨 viewer 合并牌谱时更直接。本工具维护单一 viewer 的可见局面，不承担跨视角牌谱交换。若以后增加牌谱导入导出，序列化 `SeatMap` 即可保留绝对方位，无需改变 REPL 命令。

绝对座位只存在于 core 边界。`SeatMap` 负责双向转换：

```rust
#[repr(transparent)]
struct RelativeSeat(u8);

struct SeatMap {
    dealer: RelativeSeat,
}
```

引擎继续把绝对座位 `0` 作为庄家。若 REPL 中庄家是相对座位 `d`，则：

```text
absolute(relative) = (relative + 4 - d) mod 4
relative(absolute) = (absolute + d) mod 4
```

`VisiblePosition` 使用 `RelativeSeat`。`Game::from_visible_position` 在写入 `Game` 时转换为绝对 `Seat`，`project_visible` 在比较投影前转换回来。业务逻辑不得直接混用两种座位类型。

REPL 不提供 `viewer` 命令。viewer 永远是相对座位 `0`。`:dealer <Seat>` 设置庄家的相对座位，默认值为 `0`。

## Core 契约

### 可形成性

状态 `V` 可形成，当且仅当存在一个 seed，使以下条件全部成立：

1. `Game::from_visible_position(V, seed)` 成功。
2. 具体化后的 `G` 满足引擎状态不变量：
   - 手牌张数与阶段、origin、副露和已和状态一致
   - 副露槽稠密
   - `locked[t] <= concealed[t]`
   - 每种牌的物理副本数不超过 4
   - 对手暗手与牌墙填充后，牌集闭合
   - 定缺与阶段一致
3. `project_visible(G, SeatMap) == V`：
   - 副露、牌河、定缺、分数和历史最高结构倍率一致
   - viewer 暗手与锁牌一致
   - 当前行牌权、pending event 和来源一致
4. 若 `V` 存在 viewer 决策：
   - `G.decision().actor` 映射为相对座位 `0`
   - `G.legal_action_mask()` 与 `V` 可推出的 mask 一致
5. 对 `Turn`、`Exchange` 和 `ChooseMissing`，至少有一个 seed 能使 `G.resample_information_set(seed)` 成功。

响应窗的合法集合依赖暗手。具体化时遵循引擎现有约定，固定响应窗相关手牌并重采样牌墙。

### Core API

建议新增 `engine/core/src/review.rs`，并从 `lib.rs` 导出：

- `RelativeSeat`
- `SeatMap`
- `VisiblePosition`
- `PendingEvent`
- `ReviewPolicy`
  - `Fast`
  - `Ev(RuleEvConfig)`
  - `Planner(RulePlannerConfig)`
- `PositionError`
  - `Inventory`
  - `StateInvariant`
  - `Unformable`
  - `ProjectionMismatch`
  - `DecisionMismatch`
- `validate_visible_position(&VisiblePosition) -> Result<(), PositionError>`
- `Game::from_visible_position(&VisiblePosition, seed) -> Result<Game, PositionError>`
- `project_visible(&Game, SeatMap) -> VisiblePosition`
- `advise_visible_position(&VisiblePosition, policy, seed) -> Result<ActionId, PositionError>`

`PositionError` 不包含 REPL 语法错误。lexer、parser 和状态 reducer 使用 tool 自己的错误类型。

三种策略实现保持不变。REPL 状态机不能进入 core。

### 确定性校验

#### 牌的物理计数

每种牌统计以下已知副本：

- viewer 暗手
- 非 viewer 已公开的锁牌
- 牌河
- 碰牌者提供的 2 张
- 直杠或碰杠者提供的 3 张
- 暗杠的 4 张

被调用的弃牌仍保留在牌河中，因此碰和直杠只增加响应者提供的副本。点炮引用 pending discard，不增加新的物理副本。每种牌总数必须小于或等于 4。

四家暗手张数按阶段和副露推导：

- 未和：`13 - 3 * meld_count`
- 已和：`14 - 3 * meld_count`
- 当前 actor 在 `Draw` 或 `AfterPong` origin 时再加 1

`:len` 可以覆盖非 viewer 的推断值。覆盖后仍必须有足够牌填充其余暗手和牌墙。

其他库存约束：

- `wall_remaining` 默认取未知池填充对手暗手后的剩余张数。
- `:wall` 只覆盖默认值，且必须保持牌集闭合。
- 每家最多有 4 个副露。
- 碰杠必须先存在同牌的碰。
- 暗杠来源固定为 actor。

#### 阶段与公开事件

- 行牌阶段要求四家均已定缺。
- 换三张和定缺阶段不得存在行牌副露或牌河。
- 牌河按事件时间排序。
- 碰、直杠和点炮必须消费类型匹配的 pending event。
- tile 和 source 只从 pending event 读取。命令不得重复提供。
- 抢杠胡必须消费一个 pending 碰杠声明。
- viewer 输入 `+Tile` 前，该牌必须还有物理副本。
- `=<Tiles>` 设置的 viewer 手牌不得与公开牌和锁牌冲突。

#### Viewer 决策

- `+Tile` 只在相对座位 `0` 即将摸牌时合法。
- viewer 正常摸牌后，`?` 必须看到一个 `Turn` 决策。
- 他人弃牌后，core 用 viewer 手牌计算 `HuResponse` 或 `MeldResponse`。
- viewer 存在响应时，下一公开事件不能替 viewer 隐式选择 pass。用户必须输入对应动作或 `.`。
- viewer 已和后只能弃未锁牌。`:lock` 必须是 viewer 暗手的子集。

#### 非 Viewer 和牌

非 viewer 和牌只验证以下可见事实：

- 转移来源合法。
- `ShapeSpec` 语法和组合合法。
- 存在一个满足全局库存与张数约束的隐藏暗手填充。

REPL 不验证该隐藏暗手能否组成声明牌型。`ShapeSpec` 只用于 `has_won`、结构倍率和计分。

分数默认由已知杠和和牌事件自动结算，计分基数为 100。`:score` 覆盖要求每家分数非负，且总分为 40000。

### 自动推断

- viewer 固定为相对座位 `0`。
- dealer 默认是相对座位 `0`。
- 初始分数为每家 10000。
- 对手普通摸牌不可见，不需要输入。
- 弃牌 actor 由当前行牌权推导。
- 碰和直杠的 tile 与 source 从 pending discard 取得。
- 点炮和抢杠胡的 tile 与 source 从 pending event 取得；非 viewer 自摸必须输入公开和牌张。
- 碰杠的 source 从已有碰取得；暗杠的 source 固定为 actor。
- 对手暗手张数按阶段公式推导。
- `Turn.can_hu` 和 viewer 的和牌结构由引擎分析器计算。
- 事件番从 pending kind、前一事件、墙数和开局状态推导。

### 具体化算法

1. 用 `SeatMap` 把相对座位转换为引擎绝对座位。
2. 放入全部已知公开牌和 viewer 私有牌。
3. 计算未知池：`4 * 27 - 已知物理副本`。
4. 按推断值或 `:len` 覆盖填充非 viewer 暗手。
5. 把剩余牌放入牌墙，并满足定缺相关约束。
6. 写入 phase、turn origin 和 reaction kind。
7. 用引擎计算 viewer 的 `can_hu` 和 legal mask。
8. 比较 `project_visible(G, SeatMap)` 与输入 `V`。

任一步失败都返回 `PositionError`，且不返回部分构造的 `Game`。

## 强类型 REPL

路径：`engine/tools/review`
二进制：`review`

### 会话与事务

会话保存：

- `VisiblePosition`
- `ReviewPolicy`
- seed
- 深度至少为 32 的 undo 栈

每条变更命令在会话副本上执行。REPL 依次完成解析、类型检查、状态转移、确定性校验和具体化检查。全部成功后才替换当前会话并压入 undo 快照。失败不能改变状态、undo 栈或随机数进度。

### 交互提示符

prompt 显示 REPL 当前等待的事件。弃牌 prompt 使用相对座位：

| Prompt | 含义 |
| --- | --- |
| `0> ` | 下一弃牌者是 viewer |
| `1> ` | 下一弃牌者是相对座位 `1` |
| `2> ` | 下一弃牌者是相对座位 `2` |
| `3> ` | 下一弃牌者是相对座位 `3` |
| `0+> ` | 等待 viewer 输入可见摸牌 |
| `0?> ` | 等待 viewer 的胡、碰、直杠或 pass 响应 |
| `x> ` | 等待 viewer 选择换三张 |
| `d> ` | 等待 viewer 选择定缺 |
| `*> ` | 当前没有唯一可显示的下一弃牌者，等待其他公开事件 |
| `done> ` | 牌局已结束；动作命令拒绝 |

`0> ` 只表示 `VisiblePosition.next_discarder == 0`。在 viewer 的普通回合中，REPL 先显示 `0+> `；用户输入 `+5s` 后才显示 `0> `。碰后无需摸牌，viewer 碰牌成功后直接显示 `0> `。

非 viewer 的隐藏摸牌由 reducer 合成，因此 prompt 可以从 `0> ` 直接变为 `1> `。`1> ` 表示下一张弃牌若发生，其 actor 必须是相对座位 `1`；该座位仍可先声明杠或自摸。

prompt 由强类型状态生成：

```rust
enum PromptState {
    Exchange,
    ChooseMissing,
    ViewerDraw,
    ViewerResponse,
    Discard(RelativeSeat),
    PublicEvent,
    Finished,
}
```

prompt formatter 的优先级固定为：终局、viewer 开局决策、viewer 摸牌、viewer 响应、`next_discarder`、其他公开事件。formatter 不读取上一条命令文本。

交互模式把 prompt 和错误写到 stderr。`?` 的规范动作、`s` 的状态和帮助写到 stdout。只有 stdin 和 stderr 都是 TTY 时才默认输出 prompt；`--no-prompt` 可以关闭 prompt。prompt 使用固定 ASCII 文本，不包含 ANSI 颜色，也不进入命令 parser。

输入成功后，REPL 根据新状态重绘 prompt。输入失败时状态不变，并重绘相同 prompt。`Ctrl-C` 取消当前行且不修改状态；空行上的 `Ctrl-D` 等价于 `q`。

示例：

```text
0+> +5s
0> -5s
1> -3p
0?> .
2> -9m
```

### 命令签名规则

命令不存在“默认 actor”或“可选 actor”。每类命令固定采用以下一种规则：

- 弃牌不接受座位。reducer 根据行牌权确定 actor。
- 碰、杠和胡必须接受座位，包括 viewer 自己。
- viewer 摸牌不接受座位，因为只有 viewer 的摸牌牌面可见。
- viewer pass 不接受座位。只有 viewer 的 pass 需要录入。
- 换三张和单人定缺是 viewer 决策，不接受座位。
- 四家定缺使用固定长度的 `Suit4`，不逐个输入座位。

这条规则删除 `Actor = epsilon` 一类上下文默认值。parser 对每个 action 只构造一个明确签名。

### 词法类型

```text
Seat        ::= "0" | "1" | "2" | "3"
Other       ::= "1" | "2" | "3"
Suit        ::= "m" | "s" | "p"
Rank        ::= "1" | "2" | ... | "9"
Tile        ::= Rank Suit
TileGroup   ::= Rank { Rank } Suit
Tiles       ::= TileGroup { TileGroup }
Suit4       ::= Suit Suit Suit Suit
UInt        ::= Digit { Digit }
ParamName   ::= Lower { Lower | Digit | "-" | "_" }
ParamValue  ::= non-empty token without ASCII whitespace or "="
Param       ::= ParamName "=" ParamValue
ShapeSpec   ::= ShapeCode { SP ShapeCode }
```

`Tiles` 解析为按牌种计数的 multiset。构造时拒绝第五张同牌。`ExchangeTiles` 是 `Tiles` 的精化类型：同一花色、1 至 3 张，并且加上本阶段已选张后不超过 3 张。

只接受 `m/s/p`。不接受 `w/t/b` 等同义花色。

### Command grammar

以下 grammar 定义规范输出。parser 可以忽略 token 边界处的 ASCII 空白，但 formatter 只输出下列形式。`SP` 表示至少一个 ASCII 空格。

```text
Command         ::= Reset | Action | Directive | Control
Reset           ::= "=" Tiles

Action          ::= Exchange | Missing | Draw | Discard
                  | Pong | Kong | Hu | Pass
Exchange        ::= "x" Tiles
Missing         ::= "d" Suit | "d" Suit4
Draw            ::= "+" Tile
Discard         ::= "-" Tile
Pong            ::= Seat ":p"
Kong            ::= Seat ":k"
                  | Seat ":ak" Tile
                  | Seat ":ck" Tile
Hu              ::= "0:h"
                  | Other ":h" SP ShapeSpec
                  | Other ":h+" Tile SP ShapeSpec
Pass            ::= "."

Directive       ::= ":lock" SP (Tiles | "auto")
                  | ":len" SP Other SP (UInt | "auto")
                  | ":score" SP (UInt SP UInt SP UInt SP UInt | "auto")
                  | ":wall" SP (UInt | "auto")
                  | ":dealer" SP Seat
                  | ":seed" SP UInt
                  | ":policy" SP Policy { SP Param }

Policy          ::= "fast" | "ev" | "planner"
Control         ::= "?" | "s" | "u" | "??" | "q"
```

`dSuit` 与 `dSuit4` 按 suit 数量区分。其他数量属于语法错误。注释和尾随自由文本不属于 v1 grammar。

以下输入必须拒绝：

- `1:-3p`：弃牌不接受座位。
- `p`：碰必须带座位。
- `ck3p`：杠必须带座位。
- `h`：胡必须带座位。
- `0:h ph`：viewer 和牌结构必须由引擎计算。
- `1:h`：非 viewer 和牌必须带 `ShapeSpec`。

### 强类型 AST

lexer 和 parser 必须直接构造强类型值。reducer 不得重新解析字符串。

```rust
enum Command {
    Reset(ConcealedTiles),
    Action(ObservedAction),
    Directive(Directive),
    Advise,
    Status,
    Undo,
    Help,
    Quit,
}

#[repr(transparent)]
struct OtherSeat(RelativeSeat);

enum ObservedAction {
    SelectExchange(ExchangeTiles),
    ChooseMissing(Suit),
    ChooseAllMissing([Suit; 4]),
    Draw(Tile),
    Discard(Tile),
    Pong { actor: RelativeSeat },
    ExposedKong { actor: RelativeSeat },
    AddedKong { actor: RelativeSeat, tile: Tile },
    ConcealedKong { actor: RelativeSeat, tile: Tile },
    ViewerHu,
    HiddenClaimHu { winner: OtherSeat, shapes: ShapeSpec },
    HiddenSelfDrawHu {
        winner: OtherSeat,
        tile: Tile,
        shapes: ShapeSpec,
    },
    ViewerPass,
}

enum Override<T> {
    Auto,
    Value(T),
}

enum Directive {
    Lock(Override<ConcealedTiles>),
    ConcealedLength {
        seat: OtherSeat,
        value: Override<ConcealedCount>,
    },
    Scores(Override<[Score; 4]>),
    Wall(Override<WallCount>),
    Dealer(RelativeSeat),
    Seed(u64),
    Policy(ReviewPolicy),
}
```

`OtherSeat` 只接受 `1..3`。`HiddenClaimHu` 和 `HiddenSelfDrawHu` 无法保存 viewer。`ViewerHu` 固定来自 `0:h`。这些类型分支消除 `ShapeSpec` 是否必填的运行时猜测。

`ConcealedCount` 和 `WallCount` 是有界整数类型。`Score` 非负，`Scores` 构造器检查总分。`ReviewPolicy` 构造器检查参数名、类型和范围。

### 动作语义

| 输入 | AST | 语义 |
| --- | --- | --- |
| `=<Tiles>` | `Reset` | 清空牌局状态并设置 viewer 当前暗手；保留 dealer、seed 和 policy |
| `x<Tiles>` | `SelectExchange` | 原子提交 1 至 3 个 viewer 换牌选择 |
| `d<Suit>` | `ChooseMissing` | viewer 选择定缺，例如 `dp` |
| `d<Suit4>` | `ChooseAllMissing` | 按相对座位 `0..3` 设置四家定缺，例如 `dmspm` |
| `+<Tile>` | `Draw` | viewer 摸牌，例如 `+5s` |
| `-<Tile>` | `Discard` | 当前 actor 弃牌，例如 `-5s` |
| `<Seat>:p` | `Pong` | 指定座位碰当前 pending discard |
| `<Seat>:k` | `ExposedKong` | 指定座位直杠当前 pending discard |
| `<Seat>:ak<Tile>` | `AddedKong` | 指定座位碰杠 |
| `<Seat>:ck<Tile>` | `ConcealedKong` | 指定座位暗杠 |
| `0:h` | `ViewerHu` | viewer 执行当前合法胡动作，结构由引擎计算 |
| `<Other>:h <ShapeSpec>` | `HiddenClaimHu` | 非 viewer 对当前 pending 弃牌或碰杠声明胡牌 |
| `<Other>:h+<Tile> <ShapeSpec>` | `HiddenSelfDrawHu` | 非 viewer 自摸，并记录公开和牌张 |
| `.` | `ViewerPass` | viewer 在当前响应窗选择 pass |

`p` 和 `k` 不接受 tile 或 source。两者从 pending discard 读取这些值。`h` 点炮也从 pending event 读取 tile 和 source。缺少相应 pending event 时，reducer 返回明确的 `TransitionError`。

所有杠命令都带 actor。即使当前行牌权已经能推出碰杠或暗杠的 actor，REPL 仍要求座位。该规则使碰、杠、胡三类公开声明使用同一个 actor 约定。

### 弃牌 actor 推导

`Discard(Tile)` 不保存 actor。reducer 在执行命令时读取 `VisiblePosition.next_discarder`：

- viewer 输入 `+Tile` 后，下一弃牌者是 `0`。
- 碰成功后，下一弃牌者是碰牌者。
- 普通无人响应弃牌后，下一弃牌者是原出牌者的下一家。
- 和牌后，下一弃牌者按引擎既有轮转规则确定。
- 杠后，下一弃牌者仍是杠牌者，但必须先完成补牌。

当前 actor 是非 viewer 时，`-Tile` 会先合成一次牌面不可见的普通摸牌，再记录弃牌并减少墙数。当前 actor 是 viewer 时，普通回合必须先输入 `+Tile`；`AfterPong` origin 不要求摸牌。

非 viewer 的公开回合动作使用同一套摸牌规则。若状态要求先摸牌：

- `Seat:akTile` 和 `Seat:ckTile` 先合成未知摸牌，再校验 actor 和动作。
- `Other:h+Tile ShapeSpec` 把命令中的 tile 作为本次已公开的摸牌。
- 杠后的 `-Tile`、`Seat:akTile` 或 `Seat:ckTile` 先合成未知补牌，并从墙尾扣牌。

viewer 的普通摸牌和补牌都必须显式输入 `+Tile`。当前 phase 和 origin 决定该牌来自墙头还是墙尾。

actor 不确定时，`-Tile` 返回 `UnknownDiscarder`。REPL 不允许用户用座位前缀绕过缺失状态。

### Pending event

- 弃牌创建 `Pending::Discard { source, tile }`。
- 碰杠声明创建 `Pending::AddedKong { source, tile }`。
- `Seat:p` 和 `Seat:k` 消费 pending discard。
- `Other:h ShapeSpec` 消费 pending discard 或 pending added-kong。
- 一炮多响期间保留 pending event。每名 winner 只能记录一次，source 不能成为 winner。
- 第一条非胡公开事件关闭剩余的非 viewer 胡响应。
- viewer 有合法响应时，只有 `0:h`、`0:p`、`0:k` 或 `.` 可以关闭该响应。后续事件不能隐式代替 viewer pass。
- 非 viewer pass 不录入。状态机从下一条相容的公开事件推导。

### Directive 与控制命令

| 输入 | 作用 |
| --- | --- |
| `:lock <Tiles>` | 覆盖 viewer 锁牌 |
| `:lock auto` | 恢复空锁牌默认值 |
| `:len <Other> <UInt>` | 覆盖非 viewer 暗手张数 |
| `:len <Other> auto` | 恢复该座位的张数推导 |
| `:score <a> <b> <c> <d>` | 覆盖四家分数 |
| `:score auto` | 恢复事件自动计分 |
| `:wall <UInt>` | 覆盖墙数 |
| `:wall auto` | 恢复墙数推导 |
| `:dealer <Seat>` | 设置相对庄家；首个换牌、定缺或行牌事件后拒绝 |
| `:seed <UInt>` | 设置具体化和策略采样 seed |
| `:policy fast\|ev\|planner [Param...]` | 设置策略和参数 |
| `?` | 输出一个最佳动作的规范记号 |
| `s` | 输出手牌、定缺、副露、河尾、阶段、下一弃牌者、prompt state、墙数和策略 |
| `u` | 恢复上一条成功变更前的会话快照 |
| `??` | 输出帮助 |
| `q` | 退出 |

Directive 使用完整英文词根。高频牌局事件使用短记号，低频覆盖项优先保证含义明确。

`:dealer` 可以在 `=<Tiles>` 前后使用，但必须早于首个换牌、定缺或行牌事件。修改 dealer 会重新计算首个 actor，不改变相对座位标号。

### 错误分层与处理管线

1. Lexer 生成带源位置的 token。非法字符返回 `LexError`。
2. Parser 按 grammar 构造 `Command`。签名不匹配返回 `ParseError { span, expected }`。
3. 精化类型构造器检查范围、张数和 `ShapeSpec`。失败返回 `TypeError`。
4. Reducer 在会话副本上执行 typed command。行牌权或 pending event 不匹配时返回 `TransitionError`。
5. Core 执行确定性校验、具体化和投影检查。失败返回 `PositionError`。
6. REPL 只提交通过前五步的变更命令。

每层错误只描述该层事实。状态机不能根据错误字符串修正输入，也不能在 reducer 中再次解析 tile、seat 或 pattern。

## `ShapeSpec`

`ShapeSpec` 只用于非 viewer 的隐藏和牌结构。每个 code 映射到 core 的一个 `Pattern`：

| Code | `Pattern` | 牌型 |
| --- | --- | --- |
| `ph` | `Plain` | 平胡 |
| `dy` | `AllSimples` | 断幺九 |
| `pp` | `AllTriplets` | 碰碰胡 |
| `qpp` | `PureAllTriplets` | 清碰碰胡 |
| `7` | `SevenPairs` | 七对 |
| `q7` | `PureSevenPairs` | 清七对 |
| `j7` | `TwoFiveEightSevenPairs` | 将七对 |
| `l7` | `DragonSevenPairs` | 龙七对 |
| `ql7` | `PureDragonSevenPairs` | 清龙七对 |
| `2l7` | `DoubleDragonSevenPairs` | 双龙七对 |
| `3l7` | `TripleDragonSevenPairs` | 三龙七对 |
| `j2l7` | `TwoFiveEightDoubleDragonSevenPairs` | 将双龙七对 |
| `j3l7` | `TwoFiveEightTripleDragonSevenPairs` | 将三龙七对 |
| `18` | `EighteenArhats` | 十八罗汉 |
| `q18` | `PureEighteenArhats` | 清十八罗汉 |
| `yj` | `TerminalsInEveryGroup` | 幺九 |
| `qyj` | `AllTerminals` | 清幺九 |
| `qy` | `PureOneSuit` | 清一色 |
| `jg` | `GoldenHook` | 金钩钓 |
| `qjg` | `PureGoldenHook` | 清金钩钓 |

`ShapeSpec` 构造器执行以下校验：

- 至少包含一个 code。
- 重复 code 非法。
- `ph` 只能单独出现。
- 同一牌型族最多选择一个 code。
- 复合 code 不能与其已包含的 code 并列，例如 `qpp pp`、`qpp qy`、`q7 7` 和 `q18 18`。
- 结构冲突非法，例如 `pp 7`、`7 18`、`dy yj` 和 `dy qyj`。
- formatter 按 `Pattern` 稳定序号输出，不保留输入顺序。

牌型族、包含关系、冲突关系、倍率和帮助文本由一个 `ShapeSchema` 表维护。parser、validator、计分和帮助输出必须共用该表，不能各自维护条件链。

抢杠胡、杠上炮、杠上开花、海底捞月、天胡和地胡从事件上下文推导。REPL 不接受这些事件番简称。

## 阶段推进

- `=<Tiles>` 创建无行牌历史的新局面。
- `x<Tiles>` 用于换三张阶段。整行包含的 1 至 3 个选择原子提交。
- `d<Suit4>` 可直接建立换三张完成后的定缺状态。viewer 当前手牌视为换牌后的手牌。
- `+Tile` 进入 viewer 的 `Turn { origin = Draw }`。
- `-Tile` 记录当前 actor 弃牌并创建响应流程。
- `Seat:p` 成功后，`next_discarder = Seat`，origin 为 `AfterPong`。
- `Seat:k`、`Seat:akTile` 和 `Seat:ckTile` 成功后，actor 必须先补牌。
- viewer 补牌使用 `+Tile`。非 viewer 补牌在其下一公开动作前自动合成。
- 碰杠先进入抢杠响应。无人胡后才升级副露并结算杠分。

## `?` 输出

`?` 只打印一个规范动作，不排序，不解释。示例：

```text
x1m
dp
-5s
0:h
0:p
0:k
0:ck2m
0:ak3p
.
```

每一行都能在产生该建议的状态中直接输入。建议 formatter、命令 formatter 和帮助输出共用同一实现。

## 会话示例

以下脚本从行牌阶段开始。相对座位 `0` 是 viewer：

```text
=1123456m123789s
dpppp
+5s
?
-5s
-1m
?
0:h
```

第二个弃牌命令没有座位。第一个弃牌后的轮转状态将 actor 推导为相对座位 `1`。

其他公开动作示例：

```text
2:p
1:k
3:ak5s
0:ck2m
1:h qpp
3:h+5s 7
```

这些行展示独立命令形式，不表示同一段连续牌局。

## Workspace 与文档

- workspace 增加 `engine/tools/review`。
- `engine/tools/review/README.md` 包含 30 秒上手、prompt 表、命令表、`ShapeSpec` 和错误示例。
- 根 `README.md` 与 `engine/README.md` 增加 review 工具入口。
- `engine/core/README.md` 说明 `VisiblePosition`、`SeatMap` 和 `advise_visible_position`。

## 实现顺序

1. Core：`RelativeSeat`、`SeatMap`、`VisiblePosition` 和 `project_visible`。
2. Core：`validate_visible_position` 的确定性校验。
3. Core：`Game::from_visible_position` 的具体化与投影反查。
4. Core：`advise_visible_position` 和策略配置封装。
5. Core：合法、非法、可形成和不可形成单测。
6. Tool：lexer、parser、精化类型、`ShapeSchema` 和 formatter。
7. Tool：typed reducer、actor 推导和 pending event。
8. Tool：事务会话、prompt formatter、undo、集成测试和文档。

## 测试计划

### 座位

- `RelativeSeat` 只接受 `0..3`。
- 四种 dealer 相对位置都能与绝对 `Seat` 双向 round-trip。
- observation、牌河、meld source 和 decision actor 经两次映射后不变。
- REPL 不存在 `viewer` directive。
- `:dealer` 在只设置手牌后仍可执行，在首个牌局事件后拒绝。

### Parser 与类型

- 全部规范动作都能构造预期 AST。
- `parse(format(command)) == command`。
- formatter 不输出非规范空白。
- `1:-3p`、`p`、`ck3p`、`h`、`0:h ph` 和 `1:h` 均拒绝。
- `missme`、`miss`、`m`、`hu`、`undo` 和 `help` 均拒绝。
- 第五张同牌、非法 `ExchangeTiles` 和越界整数均拒绝。
- `ShapeSchema` 覆盖全部 code、倍率、包含关系和冲突关系。
- 帮助输出的 ShapeCode 集合等于 `ShapeSchema` 集合。

### 交互界面

- `next_discarder = 0..3` 分别格式化为 `0> `、`1> `、`2> ` 和 `3> `。
- viewer 普通回合先显示 `0+> `；输入摸牌后显示 `0> `。
- viewer 碰牌成功后直接显示 `0> `。
- viewer 响应、换三张、定缺、未知公开事件和终局使用各自的固定 prompt。
- 非法输入后 prompt 文本不变。
- stdin 或 stderr 不是 TTY 时不输出 prompt；`--no-prompt` 也关闭 prompt。
- prompt 与错误进入 stderr；建议和状态进入 stdout。

### Reducer

- 普通轮转、碰后弃牌、和牌后轮转和杠后补牌都能推导唯一 `next_discarder`。
- actor 不唯一时，`-Tile` 返回 `UnknownDiscarder`。
- 非 viewer 的 `-Tile` 合成隐藏摸牌；viewer 普通回合缺 `+Tile` 时拒绝。
- 非 viewer 的暗杠、碰杠和自摸按各自规则合成未知或已公开摸牌。
- non-viewer 杠后的下一公开回合动作从墙尾合成补牌。
- `Seat:p`、`Seat:k` 和 claim hu 只消费匹配的 pending event。
- viewer 有响应时，后续事件不能隐式 pass。
- non-viewer pass 可从下一条相容公开事件推导。
- 一炮多响保留 pending event，直到第一条非胡事件。

### Core 校验与具体化

- 库存、副露来源、张数公式、定缺阶段和分数和均有边界测试。
- 合法最短局具体化成功，且投影相等。
- 至少一个信息集重采样成功。
- 对手张数无法填充、响应窗矛盾和 `:wall` 不闭合时拒绝。
- `?` 返回 legal mask 中的动作。
- `?` 输出可解析，并能在原状态执行。

### 事务

- 非法行不改变状态、undo 栈或随机数进度。
- `u` 恢复完整会话快照。
- 恢复后再次执行 `?` 得到相同结果。
- 文档中的最短脚本作为 smoke test 执行。

## 默认值

- 策略：`rule-ev` 与 `RuleEvConfig::STANDARD`。
- viewer：相对座位 `0`，不可配置。
- dealer：相对座位 `0`，可在首个牌局事件前用 `:dealer` 修改。
- 分数：每家 10000。
- 锁牌：空。
- 墙数和非 viewer 手牌张数：自动推导。
- “可形成”表示存在一个合法完整具体化，并通过投影、决策和信息集检查。

## 不进入命令面的字段

用户不输入以下派生字段：

- phase
- actor
- `can_hu`
- 完整 response remaining 集合
- turn origin
- 弃牌 actor
- 碰、直杠或点炮的 tile 与 source
- 可从事件上下文推导的事件番
- 仅修改 `has_won` 的标记

状态机和 core 校验共同维护这些字段。无法推导时返回明确错误，不增加临时覆盖命令。
