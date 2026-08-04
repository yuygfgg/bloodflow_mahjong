# Web Client Rewrite Plan

## 1. Purpose and success criteria

This plan replaces the PySide6 desktop client with a browser client for the Rust Blood Flow Mahjong engine. The client must be easy to try from a static web deployment. A user must not need Python, Rust, a compiler, or a model-training environment to play one complete local game.

The UI requirement is to copy OpenRiichi as closely as possible. The Chinese shorthand for this requirement is “照抄”. Minimize free-form design work. The target is OpenRiichi's visual language and interaction rhythm:

- a full-screen four-sided table;
- a deep perspective camera with the local player at the bottom;
- a dark wood frame and a large textured table surface;
- physical-looking tiles with strong edge highlights and shadows;
- large player names, seats, scores, and wind indicators around the table;
- a central status panel;
- translucent action buttons along the lower edge;
- short, readable result overlays;
- tile movement, flip, draw, discard, meld, and score animations.

The client must copy OpenRiichi's general UI behavior whenever that behavior is reusable. Engineers must read the corresponding OpenRiichi code and translate the generic logic instead of inventing a new interaction model. Translate the implementation to TypeScript/Three.js and translate the text to the project's language; change rules only where Blood Flow Mahjong differs. Do not copy Japanese riichi rules into the game. Every tile, button, label, animation, and result panel must represent `GAME_RULES.md`. The engine remains the only authority for legal actions, hidden information, scoring, and termination.

### 1.1 Definition of done

The first public milestone is complete when all of the following are true:

1. A user can open a hosted page and start a game with one click.
2. The page runs one human seat against three deterministic local bots.
3. The bot settings include `rule-fast`, `rule-ev`, and `rule-nn`; the NN bot can play a complete seeded game.
4. The user can complete exchange, missing-suit choice, ordinary turns, wins, kongs, and response windows.
5. The UI shows every Blood Flow-specific transition without exposing hidden tiles.
6. The user can pause, restart, choose a seed, and copy a replay identifier.
7. The same seed and action sequence produce the same event log and final score.
8. The initial page is usable on a 1280x720 desktop viewport and remains playable on a 1024x768 viewport.
9. A production build does not require a local toolchain.

## 2. Scope and non-goals

### 2.1 In scope

- Local single-player games with `rule-fast` and `rule-ev` bots.
- `rule-nn` browser inference with the bundled micro model.
- OpenRiichi-faithful 3D table, tile, HUD, menu, animation, and sound presentation.
- Blood Flow-specific exchange, missing-suit, repeat-win, kong, payment, flower-pig, and dajiao presentation.
- Deterministic replay export and import.
- Static deployment and offline cache after the first load.

### 2.2 Out of scope for the first release

- Internet multiplayer. A client-side authoritative game would reveal hidden hands.
- Accounts, matchmaking, chat, ranking, and a persistent server database.
- A second rules implementation in TypeScript.
- Full parity with OpenRiichi's Vala networking or native rendering code.
- `rule-planner` browser inference. This is a later optional enhancement.

## 3. Rules authority and UI contract

`engine/core` is authoritative. The browser must submit an `ActionId` only after reading the current legal mask from WASM. The UI may group legal actions for presentation, but it must not infer legality from visual state.

The WASM wrapper must expose a UI snapshot separate from the training observation. The snapshot must use explicit fields and viewer-relative seats. It must not expose the full `Game` struct, opponent concealed hands, wall order, or unfiltered events.

### 3.1 Required snapshot fields

The snapshot protocol should contain:

- `engine_rules_version`;
- `phase`;
- `decision_actor`;
- `dealer`;
- `exchange_direction`;
- `wall_remaining`;
- `scores`;
- `missing_suits`;
- `hands` for the viewer and concealed tile counts for opponents;
- `locked_tiles` for all seats;
- `melds`;
- `rivers` with relative owners;
- `pending_source`, `pending_tile`, and pending response kind;
- `draw_tile` only when visible to the viewer;
- `has_won`;
- `legal_action_mask`;
- `event_history` and `step_events` after observer filtering;
- `termination_reason`;
- final rankings and payment summary when finished.

The protocol should be versioned. A transition contains the accepted action, filtered events, the next snapshot, and an animation hint list. Animation hints are derived from events and do not affect rules.

### 3.2 Worker boundary

`engine.worker.ts` owns the WASM `Game` instance. The main thread sends:

- `new_game(seed, human_seat, bot_profiles)`;
- `submit(action_id)`;
- `request_hint(policy)`;
- `pause` and `resume`;
- `export_replay`;
- `load_replay`.

The worker returns `ready`, `snapshot`, `transition`, `hint`, `replay`, and `error` messages. AI steps run inside the worker as the single owner of the WASM game. The worker boundary is an ownership and protocol decision; it is not a claim that the micro NN model needs special performance treatment. The first release includes `rule-fast`, `rule-ev`, and `rule-nn`. Bundle the NN model with the initial Web build and load it when the worker starts. Its CPU inference cost is expected to be small, so lazy loading and model splitting are not first-release requirements.

## 4. Recommended technology stack

### 4.1 Rust and WASM

- `bloodflow-mahjong` remains the rules crate.
- Add `engine/wasm` as a small `cdylib` wrapper.
- Use `wasm-bindgen` for typed JavaScript entry points.
- Use `serde`/`serde-wasm-bindgen` or a compact explicit binary encoder for snapshots. Start with typed JSON-compatible values for debuggability.
- Build with `wasm-pack` or `wasm-bindgen-cli` from a reproducible toolchain.
- Include the `rule-nn` feature and the bundled model in the required first-release build. Keep `rule-planner` separate until a later release.

The core uses an explicit seed and `ChaCha8Rng`. Keep this property. Do not introduce browser-dependent randomness into the rules path.

### 4.2 Browser application

- TypeScript, React, Vite, and pnpm.
- Zustand or a small reducer for UI-only state. Do not mirror authoritative game state in multiple stores.
- Three.js for the table, tiles, walls, discard ponds, melds, and camera.
- React DOM for menus, action bars, score panels, help, replay controls, and accessibility text.
- Web Worker for WASM and AI.
- Vitest for reducers and protocol tests.
- Playwright for browser smoke tests and replay tests.

Use React for composition and controls. Use Three.js for continuous visual transforms. Avoid a DOM element for every tile in the 3D table. When OpenRiichi already has a matching control, state transition, camera behavior, menu, animation timing, or layout rule, port that behavior first and only then apply the language and Blood Flow rule substitutions.

## 5. OpenRiichi-faithful visual language

### 5.0 Copy-first implementation constraint

This project must not treat OpenRiichi as a loose mood board. It must use OpenRiichi as the primary UI specification. The default decision is “copy the OpenRiichi behavior”; a new design needs a concrete reason such as a Blood Flow rule, browser limitation, accessibility requirement, or hidden-information constraint.

For every major UI subsystem, identify the corresponding OpenRiichi source before implementation:

- main window and scene composition;
- camera and table transforms;
- player and tile render objects;
- hand sorting and tile selection;
- action button construction and ordering;
- center status and scoring overlays;
- menu navigation and pause behavior;
- animation queue and sound triggers;
- replay and game-log views.

Translate common logic directly from the OpenRiichi code. Keep the same state transitions, control placement, visual hierarchy, timing, and interaction rhythm whenever the behavior is generic. For generic behavior, change only the implementation language (Vala/native APIs to TypeScript/Three.js/browser APIs) and the displayed language. Do not redesign the behavior. Blood Flow rule differences are the only product-level reason to add or remove an interaction. Change only:

1. implementation language and browser APIs;
2. displayed language and terminology;
3. tile inventory and model bindings;
4. Blood Flow rules and legal-action mapping;
5. browser-specific input, performance, and accessibility details.

Do not add a second visual system because it is easier to implement. A simplified debug renderer is allowed only as a temporary development tool and must not become the production design.

### 5.1 Table composition

The main scene uses a perspective camera looking toward the table center. The viewer's hand sits near the bottom edge. Opponent hands sit at the top, left, and right edges. The center contains a compact status board rather than a large rule explanation.

Use these visual layers:

1. Background: a dark blue-green or space-like texture with low contrast.
2. Table frame: dark wood with visible depth and bevel.
3. Felt: a bright, high-contrast material. Blood Flow variants may use blue-green or marble textures, but the tile faces must remain readable.
4. Tile layer: white or dark-backed physical tiles with bevels, highlights, and shadows.
5. HUD layer: large player labels and scores, centered status text, and lower action buttons.
6. Overlay layer: response prompts, scoring results, pause menu, replay controls, and errors.

The design should feel like a physical board with a theatrical game overlay. Keep action labels large and short. Do not reproduce OpenRiichi's Japanese-only action names.

### 5.2 Camera and responsive layout

- Use a fixed logical table aspect ratio and letterbox the scene when necessary.
- Preserve the bottom hand as the visual anchor.
- Scale tile geometry and labels together.
- Keep the action bar inside the safe area on short screens.
- Permit camera rotation and zoom only from the pause/settings menu in the first release.
- Keep all critical actions available by keyboard and accessible DOM controls.

### 5.3 Tile language

Use the OpenRiichi tile silhouette, proportions, shading, spacing, and depth language. Convert native models to GLB and use a shared material. Maintain one stable tile coordinate system so a tile can move between wall, hand, river, meld, and overlay positions without visual snapping.

Blood Flow uses 27 suited tiles: characters, bamboo, and dots. Do not display honor tiles. The tile atlas must include a back, all 27 faces, a selected state, a disabled state, and a locked/winning state.

## 6. Blood Flow-specific interaction design

This section is mandatory. The client must make each non-riichi rule visible and understandable.

### 6.1 Opening exchange: 换三张

Engine phase: `Exchange`. The opening hand contains 14 tiles for the dealer and 13 for each other player. Each player selects three tiles of one suit. The engine collects three sequential tile actions and applies the exchange only after all twelve selections are complete.

UI behavior:

- Show a centered banner: `换三张` and the exchange direction (`左`, `对家`, or `右`).
- Show the viewer's hand in the normal bottom position with a visible selection tray above it.
- Highlight the first selected tile with a gold outline. After the first selection, constrain the hand to tiles of the same suit and dim all other tiles.
- Show `已选 1/3`, `已选 2/3`, or `已选 3/3` beside the action bar.
- Do not let the user select a fourth tile or a different suit. The disabled appearance must match the OpenRiichi button and tile language.
- Provide `确认换牌` only when exactly three tiles are selected. The button maps to the third engine action; it does not perform a client-side exchange.
- Animate selected tiles lifting from the hand, moving along the chosen direction, rotating face-down while travelling, then settling into the recipient hand.
- Keep opponent selections face-down. Show progress badges such as `下家 2/3` only if the engine event stream makes that progress public.
- On completion, play one exchange-complete event and move to missing-suit choice.

The exchange direction is public. The selected tile identities of other players are not public until the rules expose them.

### 6.2 Missing suit: 定缺

Engine phase: `ChooseMissing`. Every player chooses one suit to declare missing. All four choices become public together.

UI behavior:

- Replace the exchange banner with a centered `定缺` banner.
- Present three large suit buttons: `万`, `条`, `筒`. Use the same textured button treatment as OpenRiichi's lower action bar.
- Add a short instruction: `选择本局定缺花色`. Do not use a radio form that looks unrelated to the table.
- Before submission, show the chosen suit as a gold wind-indicator-style marker near the local name.
- Disable submission until one suit is selected. Do not allow a second click to change a submitted choice.
- When all choices are collected, animate four suit markers appearing around the table at the same time.
- From this point, tiles of the declared missing suit receive a clear but restrained red/orange warning tint in the viewer hand.
- During the forced-missing discard period, dim non-missing tiles and make missing-suit tiles the only enabled discard targets.

Do not reveal opponent concealed counts or selected tiles as part of the missing-suit presentation.

### 6.3 Normal turn and missing-suit enforcement

Engine phase: `Turn`. A player draws, then must discard, self-draw win, concealed kong, or added kong when legal. Before all missing-suit tiles are discarded, only tiles from the declared missing suit may be discarded.

UI behavior:

- Animate the draw from the wall into the local hand. Label a replacement draw as `杠后补牌`, and label a last-wall draw as `海底` when the event flags require it.
- Mark the drawn tile with a small lift and a short glow. Do not use a permanent floating tooltip.
- Keep the local hand sorted unless the player has selected manual ordering in settings.
- Disable locked winning tiles. Show them in a separate, slightly raised row or with a gold edge.
- If missing-suit enforcement is active, use a red edge for the required suit and a muted gray edge for forbidden tiles.
- Use a single discard action: click an enabled tile. The action bar may show `出牌`, but the tile itself remains the primary control.
- If `胡`, `暗杠`, or `碰杠` is legal, show them as short lower buttons. Do not show disabled buttons unless the user enables an accessibility setting.
- For a self-draw, label the action `自摸` in the Blood Flow vocabulary.

### 6.4 Discard response: 胡、碰、直杠、过

After a discard, the engine first opens `HuResponse` for all eligible players. Only if nobody wins does it open `MeldResponse` for pong or exposed kong.

UI behavior:

- Pause the normal camera motion and add a compact center banner: `响应 6万` (using the actual tile).
- For the local player, show only legal response buttons: `胡`, `碰`, `直杠`, and `过`.
- Use OpenRiichi-style large horizontal buttons with distinct colors: warm red for `胡`, gold for `杠`, blue-green for `碰`, gray for `过`.
- Show the source seat and tile in the center, but do not show hidden response reasons.
- If the local player has multiple legal responses, order them `胡`, `直杠`, `碰`, `过`; keep the order stable.
- A response button must submit exactly one engine action. No local animation may imply a response before the worker accepts it.
- When one or more players win, suppress pong and kong animations for that discard.
- For multiple winners, queue winner overlays in seat order and show each payment as a separate transfer.

### 6.5 Blood Flow win: 和牌后继续

Winning does not end the hand. The winner's winning structure becomes locked and the player continues to participate according to engine rules.

UI behavior:

- Show a brief `和牌` overlay with the winning tile, shape multiplier, event multipliers, and immediate payment.
- Move the winning subset into a locked row. Locked tiles use a gold border and a subtle vertical offset.
- Keep the winner's seat active in the table. Do not replace the whole table with a terminal result screen.
- Show a small `已和 N 次` badge near the seat name.
- On the next turn, render only the unlocked hand as selectable.
- If another win is legal, keep the `和`/`自摸` action prominent.
- If the player cannot continue a winning structure, let the engine transition normally; the client must not invent a “pass after win” state.
- Use a short lock animation: the winning tiles move to the locked row, their face remains visible, and the active hand contracts around them.

The visual distinction between active and locked tiles is essential. A player must understand why a tile cannot be discarded or reused.

### 6.6 Kongs and replacement draws

Support concealed kong, exposed kong, and added kong. Added kong opens the same rob-kong (`抢杠胡`) response flow as the engine.

UI behavior:

- Use `暗杠`, `直杠`, and `碰杠` labels. Do not use the generic `Kan` label.
- Show a kong declaration as a four-tile group with the OpenRiichi meld spacing and orientation.
- For a concealed kong, hide the appropriate tile faces if the rules require a face-down presentation; preserve the actual viewer visibility contract.
- Animate the replacement draw from the wall tail and label it `杠后补牌`.
- For an added kong, show a short center banner `声明碰杠，等待抢杠胡`.
- During that response, show only `胡` and `过` when legal.
- If robbed, keep the kong incomplete in the visual history, show `抢杠胡`, and do not play a payment for the cancelled kong.
- If not robbed, complete the meld and play the replacement draw.

### 6.7 Last-wall and terminal events

- Mark a last-wall draw with a small `海底` tag next to the active seat.
- Keep the table visible through the final response window.
- For wall exhaustion, show `牌墙摸完` before settlement.
- Show flower-pig settlement first, then dajiao settlement, matching engine order.
- Render each payment as a moving score token from payer to receiver. Clamp displayed payments to the actual engine payment.
- Show the final four-seat score panel only after all settlement events complete.

### 6.8 Flower pig and dajiao settlement

These are Blood Flow terminal rules, not generic “draw” results.

- Use a two-step result overlay: `查花猪` followed by `查大叫`.
- For flower pig, mark the violating seat with a red suit icon and list the missing-suit tiles that remain.
- For dajiao, show the ready/waiting seats and the maximum structural multiplier used by the engine.
- Do not show event multipliers that the engine excludes from dajiao calculation.
- Animate payments after each stage, not as one merged number.
- Keep the score history accessible from the result panel.

## 7. Scene and component structure

### 7.1 Three.js scene

Suggested scene nodes:

```text
TableScene
├── CameraRig
├── TableFrame
├── TableSurface
├── CenterBoard
├── WallRing
├── Seat[0..3]
│   ├── Hand
│   ├── River
│   ├── Melds
│   ├── LockedWins
│   └── SeatMarker
├── AnimationLayer
└── EffectsLayer
```

Use object pools for tiles, score tokens, and text labels. A finished game must not retain an unbounded number of animation objects.

### 7.2 React components

Suggested components:

- `GamePage`;
- `TableCanvas`;
- `CenterStatus`;
- `SeatHud`;
- `ActionBar`;
- `ExchangePanel`;
- `MissingSuitPanel`;
- `ResponsePanel`;
- `WinOverlay`;
- `SettlementOverlay`;
- `PauseMenu`;
- `ReplayPanel`;
- `AccessibilityPanel`;
- `AssetLoadingOverlay`.

The action bar selects its contents from `legal_action_mask` and phase. Blood Flow panels are phase-specific views, not independent rule engines.

## 8. Asset migration and licensing

The repository is AGPLv3. See `LICENSE`. OpenRiichi is GPLv3. GPLv3 section 13 expressly permits combining a covered work with a work licensed under AGPLv3, so the OpenRiichi GPLv3 material and this AGPLv3 project may be combined under the compatible copyleft terms. Keep the applicable license and copyright notices in the distributed Web client.

Copy the OpenRiichi assets used by the native client as directly as possible. Convert them only when the Web runtime requires a format change:

Recommended conversion pipeline:

1. Import OBJ/MTL into Blender or a reproducible command-line converter.
2. Export GLB with a stable scale, origin, and orientation.
3. Pack tile faces into one atlas and generate a machine-readable tile map.
4. Compress textures to WebP or KTX2 after visual comparison.
5. Convert audio to Ogg Opus.
6. Subset the font and convert it to WOFF2.
7. Use the converted asset in the copied OpenRiichi scene and compare the result at the reference viewport.

## 9. Replay and protocol design

Store a replay as:

```json
{
  "protocol_version": 1,
  "engine_rules_version": 6,
  "seed": 42,
  "human_seat": 0,
  "actions": [0, 1, 2]
}
```

The action list is authoritative. Event history and UI animation hints are derived data. Reject a replay when the engine version is incompatible or when one action is illegal. Never trust a client-provided final score.

Replay tests must compare:

- phase after every action;
- legal action mask;
- filtered step events;
- score vector;
- final rankings and termination reason.

## 10. Delivery phases

### Phase A: Contract and build foundation

- Add `engine/wasm`.
- Define snapshot and worker message schemas.
- Add a no-UI worker test that plays a seeded game.
- Add TypeScript project, lint, format, and test commands.
- Build and verify the required WASM game path with `rule-nn`, including model loading and one deterministic inference step.
- Verify native and WASM deterministic replay.

Exit criteria: a browser worker can start, step, and finish a game without rendering.

### Phase B: Playable 2D/debug client

- Render every snapshot with plain HTML and debug tile shapes.
- Implement exchange, missing-suit, turn, response, win, kong, and settlement controls.
- Show event and score logs.
- Add keyboard navigation and an error boundary.

Exit criteria: all engine phases are playable and observable before 3D work starts.

### Phase C: OpenRiichi-style table

- Add the perspective camera, wood frame, table material, center board, wall, seats, rivers, melds, and hand geometry.
- Add responsive camera scaling and safe-area layout.
- Add tile atlas and material states.
- Preserve the Phase B controls as an accessibility/debug overlay.

Exit criteria: the table is recognizably an OpenRiichi translation, with only language, browser implementation, and Blood Flow rule differences.

### Phase D: Animation and audio

- Implement event-to-animation mapping.
- Add exchange, draw, discard, meld, lock, win, payment, and settlement animations.
- Add sound cues with a mute setting and user-gesture audio unlock.
- Add skip-animation and reduced-motion modes.

Exit criteria: a complete seeded game can be watched without debug controls.

### Phase E: Replay, performance, and release

- Add replay import/export.
- Add Playwright visual smoke tests.
- Measure first contentful paint, WASM load, memory, and frame time.
- Treat planner, music, and high-resolution texture splitting as optional post-release optimizations. NN lazy loading and model splitting are also optional later optimizations, not initial requirements.
- Add static hosting, cache headers, and a license notice page.

Exit criteria: a fresh browser can start a game quickly on a normal desktop connection and the release contains the required license notices.

## 11. Testing strategy

### 11.1 Engine and WASM

- Keep all existing Rust rule tests.
- Add snapshot schema tests for every phase.
- Add tests that opponent draws and concealed hands remain hidden.
- Add deterministic replay tests across native and WASM builds.
- Add action atomicity tests: an illegal action changes no state or RNG state.

### 11.2 UI state tests

Test each Blood Flow phase with fixtures:

- exchange at selection counts 0, 1, 2, and 3;
- each exchange direction;
- missing-suit selection and simultaneous reveal;
- forced missing-suit discard;
- normal draw and discard;
- self-draw, discard win, multiple winners, and repeat win;
- concealed, exposed, and added kongs;
- robbed added kong;
- last-wall draw;
- flower-pig and dajiao settlement.

Each fixture must assert enabled controls, visible tiles, labels, seat-relative positions, and event order.

### 11.3 Visual tests

- Capture the reference viewport at 1280x720 and 1024x768.
- Compare table camera, tile readability, action bar placement, and overlay contrast.
- Run reduced-motion snapshots so animation timing does not make tests flaky.
- Check that the copied OpenRiichi style does not reduce the contrast of Blood Flow warnings.

## 12. Performance and accessibility budgets

- First page shell visible within 1 second on a warm desktop load.
- Measure the WASM, JavaScript, and bundled NN model payload. Do not make a specific payload target or lazy-loading strategy a first-release gate.
- Keep steady-state rendering at 60 FPS on an integrated desktop GPU.
- Keep the required local bot path responsive. The NN path is part of the initial build; planner isolation and later inference optimizations are optional follow-up work.
- Use keyboard focus for every action.
- Provide text labels for all 3D actions.
- Support reduced motion, high contrast, and color-independent missing-suit indicators.
- Do not use color as the only signal for disabled, locked, or winning tiles.

## 13. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| A direct OpenRiichi translation becomes difficult in the browser | Keep the original subsystem structure and port the generic behavior before simplifying it |
| Optional planner increases load or compute cost | Defer it until after the first release; if implemented, isolate it in the existing worker and optimize later |
| Planner blocks the browser | Run all AI in a worker; add cancellation and time budgets |
| UI duplicates engine rules | Use legal mask and typed snapshots only |
| 3D scene is hard to test | Keep a 2D/debug renderer and event fixtures |
| Hidden information leaks | Filter snapshots and events in Rust before serialization |
| Small screens hide actions | Safe-area action bar, keyboard controls, and responsive scaling |
| Animation desynchronizes from rules | Animate accepted worker transitions only; allow skip and resync |

## 14. Immediate next tasks

1. Create `engine/wasm` and define the first snapshot schema.
2. Add the worker protocol and a deterministic no-render browser test.
3. Add a minimal TypeScript/Vite app with a debug renderer.
4. Implement exchange and missing-suit panels before ordinary-turn polish.
5. Add a Blood Flow fixture pack for repeat wins, locked tiles, kongs, and settlement.
6. Convert the first table, tile, and button assets and port their matching OpenRiichi render and interaction logic.
7. Keep the existing desktop client as a test oracle until the Web client reaches Phase D.
