//! Low-level WebAssembly bindings for `bloodflow-mahjong`.
//!
//! Callers own typed-array buffers for every `*_into` write path.
//! Higher-level UI snapshots and worker protocol types live in the TypeScript
//! package under `js/`.

#![deny(unsafe_code)]

use ::bloodflow_mahjong as core_engine;
#[cfg(feature = "rule-nn")]
use core_engine::RuleNn as CoreRuleNn;
use core_engine::{
    ActionId, ExchangeDirection, Game as CoreGame, GameError, MeldKind, RuleEvConfig,
    RuleEvDefense, RulePlannerConfig, Seat, StepOutcome, TerminationReason,
};
use wasm_bindgen::prelude::*;

/// Install a panic hook that prints to the browser/worker console.
#[wasm_bindgen(js_name = initPanicHook)]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

// Zero-arg getters on the WASM module, workaroud wasm-bindgen's limitation that
// cannot export plain numeric `const` items;
macro_rules! export_engine_constants {
    ($($name:ident = { $value:expr };)*) => {
        $(
            #[wasm_bindgen]
            #[allow(non_snake_case)]
            pub fn $name() -> i32 {
                let value: i64 = $value;
                i32::try_from(value).unwrap_or_else(|_| {
                    panic!(concat!(stringify!($name), " does not fit in i32"))
                })
            }
        )*
    };
}

core_engine::with_engine_constants!(export_engine_constants);

/// Immutable search budget for the `rule-ev` policy.
#[wasm_bindgen(js_name = RuleEvConfig)]
#[derive(Clone, Copy)]
pub struct JsRuleEvConfig {
    inner: RuleEvConfig,
}

#[wasm_bindgen(js_class = RuleEvConfig)]
impl JsRuleEvConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(search_depth: u8, defense: bool) -> Result<JsRuleEvConfig, JsValue> {
        let inner = RuleEvConfig::with_search_depth(search_depth)
            .ok_or_else(|| config_range_error("search_depth", search_depth, "0..=3"))?
            .with_defense(if defense {
                RuleEvDefense::Heuristic
            } else {
                RuleEvDefense::None
            });
        Ok(Self { inner })
    }

    #[wasm_bindgen(js_name = fast)]
    pub fn fast() -> JsRuleEvConfig {
        Self {
            inner: RuleEvConfig::FAST,
        }
    }

    #[wasm_bindgen(js_name = standard)]
    pub fn standard() -> JsRuleEvConfig {
        Self {
            inner: RuleEvConfig::STANDARD,
        }
    }

    #[wasm_bindgen(getter, js_name = searchDepth)]
    pub fn search_depth(&self) -> u8 {
        self.inner.search_depth()
    }

    #[wasm_bindgen(getter)]
    pub fn defense(&self) -> bool {
        self.inner.defense() == RuleEvDefense::Heuristic
    }
}

/// Immutable search budget for the `rule-planner` policy.
#[wasm_bindgen(js_name = RulePlannerConfig)]
#[derive(Clone, Copy)]
pub struct JsRulePlannerConfig {
    inner: RulePlannerConfig,
}

#[wasm_bindgen(js_class = RulePlannerConfig)]
impl JsRulePlannerConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(
        hand_changes: u8,
        draw_horizon: u8,
        candidate_states: u32,
        belief_worlds: u16,
        response_worlds: u16,
        search_iterations: u16,
    ) -> Result<JsRulePlannerConfig, JsValue> {
        Ok(Self {
            inner: build_planner_config(
                hand_changes,
                draw_horizon,
                candidate_states,
                belief_worlds,
                response_worlds,
                search_iterations,
            )?,
        })
    }

    #[wasm_bindgen(js_name = defaultConfig)]
    pub fn default_config() -> JsRulePlannerConfig {
        Self {
            inner: RulePlannerConfig::DEFAULT,
        }
    }

    #[wasm_bindgen(getter, js_name = handChanges)]
    pub fn hand_changes(&self) -> u8 {
        self.inner.hand_changes()
    }

    #[wasm_bindgen(getter, js_name = drawHorizon)]
    pub fn draw_horizon(&self) -> u8 {
        self.inner.draw_horizon()
    }

    #[wasm_bindgen(getter, js_name = candidateStates)]
    pub fn candidate_states(&self) -> u32 {
        self.inner.candidate_states()
    }

    #[wasm_bindgen(getter, js_name = beliefWorlds)]
    pub fn belief_worlds(&self) -> u16 {
        self.inner.belief_worlds()
    }

    #[wasm_bindgen(getter, js_name = responseWorlds)]
    pub fn response_worlds(&self) -> u16 {
        self.inner.response_worlds()
    }

    #[wasm_bindgen(getter, js_name = searchIterations)]
    pub fn search_iterations(&self) -> u16 {
        self.inner.search_iterations()
    }
}

/// ONNX-backed neural policy (`rule-nn` feature).
#[cfg(feature = "rule-nn")]
#[wasm_bindgen(js_name = RuleNn)]
pub struct JsRuleNn {
    inner: CoreRuleNn,
}

#[cfg(feature = "rule-nn")]
#[wasm_bindgen(js_class = RuleNn)]
impl JsRuleNn {
    #[wasm_bindgen(constructor)]
    pub fn new(onnx: &[u8]) -> Result<JsRuleNn, JsValue> {
        CoreRuleNn::from_onnx_bytes(onnx)
            .map(|inner| Self { inner })
            .map_err(rule_nn_error)
    }

    /// Select a legal action for the current decision, or `undefined` at terminal.
    pub fn action(&self, game: &JsGame) -> Result<Option<u8>, JsValue> {
        self.inner
            .action(&game.inner)
            .map(|action| action.map(|id| id.index() as u8))
            .map_err(rule_nn_error)
    }
}

/// Single-environment game state.
#[wasm_bindgen(js_name = Game)]
pub struct JsGame {
    inner: CoreGame,
}

#[wasm_bindgen(js_class = Game)]
impl JsGame {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: u64) -> JsGame {
        Self {
            inner: CoreGame::new(seed),
        }
    }

    #[wasm_bindgen(js_name = withExchangeDirection)]
    pub fn with_exchange_direction(seed: u64, direction: u8) -> Result<JsGame, JsValue> {
        let direction = exchange_direction(direction)?;
        Ok(Self {
            inner: CoreGame::new_with_direction(seed, direction),
        })
    }

    pub fn reset(&mut self, seed: u64) {
        self.inner.reset(seed);
    }

    #[wasm_bindgen(js_name = resampleInformationSet)]
    pub fn resample_information_set(&self, seed: u64) -> Result<JsGame, JsValue> {
        self.inner
            .resample_information_set(seed)
            .map(|inner| Self { inner })
            .map_err(game_error)
    }

    #[wasm_bindgen(getter)]
    pub fn phase(&self) -> u8 {
        self.inner.phase().code()
    }

    /// Current decision as `[actor, phase]`, or `undefined` when finished.
    #[wasm_bindgen(getter)]
    pub fn decision(&self) -> Option<Vec<u8>> {
        self.inner
            .decision()
            .map(|decision| vec![decision.actor.as_u8(), decision.phase.code()])
    }

    /// Legal-action mask as two little-endian `u64` words (`[low, high]`).
    #[wasm_bindgen(getter, js_name = legalActionMask)]
    pub fn legal_action_mask(&self) -> Vec<u64> {
        self.inner.legal_action_mask().map_or_else(
            || vec![0, 0],
            |mask| {
                let words = mask.words();
                vec![words[0], words[1]]
            },
        )
    }

    /// Deterministic `rule-fast` action, or `undefined` at terminal.
    #[wasm_bindgen(js_name = simpleRuleAction)]
    pub fn simple_rule_action(&self) -> Option<u8> {
        self.inner
            .simple_rule_action()
            .map(|action| action.index() as u8)
    }

    /// `rule-ev` action with the standard budget.
    #[wasm_bindgen(js_name = ruleEvAction)]
    pub fn rule_ev_action(&self) -> Option<u8> {
        self.inner
            .rule_ev_action_with_config(RuleEvConfig::STANDARD)
            .map(|action| action.index() as u8)
    }

    /// `rule-ev` action with an explicit borrowed config.
    #[wasm_bindgen(js_name = ruleEvActionWithConfig)]
    pub fn rule_ev_action_with_config(&self, config: &JsRuleEvConfig) -> Option<u8> {
        self.inner
            .rule_ev_action_with_config(config.inner)
            .map(|action| action.index() as u8)
    }

    /// `rule-planner` action with the default budget.
    #[wasm_bindgen(js_name = rulePlannerAction)]
    pub fn rule_planner_action(&self) -> Option<u8> {
        self.inner
            .rule_planner_action_with_config(RulePlannerConfig::DEFAULT)
            .map(|action| action.index() as u8)
    }

    /// `rule-planner` action with an explicit borrowed config.
    #[wasm_bindgen(js_name = rulePlannerActionWithConfig)]
    pub fn rule_planner_action_with_config(&self, config: &JsRulePlannerConfig) -> Option<u8> {
        self.inner
            .rule_planner_action_with_config(config.inner)
            .map(|action| action.index() as u8)
    }

    /// Apply an action id and return the 12-field step record as `i32` values.
    ///
    /// Score components stay well inside `i32` for Blood Flow Mahjong scoring.
    #[wasm_bindgen(js_name = stepId)]
    pub fn step_id(&mut self, action: u8) -> Result<Vec<i32>, JsValue> {
        let action = action_id(action)?;
        let outcome = self.inner.step_id(action).map_err(game_error)?;
        Ok(encode_outcome_i32(outcome).to_vec())
    }

    /// Write the 12-field step record into a caller-owned `Int32Array`.
    #[wasm_bindgen(js_name = stepInto)]
    pub fn step_into(&mut self, action: u8, output: &mut [i32]) -> Result<(), JsValue> {
        require_len(output, core_engine::STEP_RECORD_WIDTH, "output")?;
        let action = action_id(action)?;
        let outcome = self.inner.step_id(action).map_err(game_error)?;
        output.copy_from_slice(&encode_outcome_i32(outcome));
        Ok(())
    }

    #[wasm_bindgen(getter, js_name = eventCount)]
    pub fn event_count(&self) -> u32 {
        self.inner.event_count() as u32
    }

    #[wasm_bindgen(getter, js_name = eventDropped)]
    pub fn event_dropped(&self) -> u64 {
        self.inner.event_dropped()
    }

    /// Write viewer-filtered event history into a flat `[capacity * 8]` buffer.
    ///
    /// Returns the number of written records. Capacity must be `1..=512`.
    #[wasm_bindgen(js_name = eventsInto)]
    pub fn events_into(&self, viewer: u8, output: &mut [i32]) -> Result<u32, JsValue> {
        let viewer = seat_value(viewer)?;
        let capacity = validate_event_buffer(output, "output")?;
        let written = self.inner.events_into(viewer, output).map_err(game_error)?;
        debug_assert!(written <= capacity);
        Ok(written as u32)
    }

    /// Write events produced by the most recent step into a flat buffer.
    #[wasm_bindgen(js_name = stepEventsInto)]
    pub fn step_events_into(&self, viewer: u8, output: &mut [i32]) -> Result<u32, JsValue> {
        let viewer = seat_value(viewer)?;
        let capacity = validate_event_buffer(output, "output")?;
        let written = self
            .inner
            .step_events_into(viewer, output)
            .map_err(game_error)?;
        debug_assert!(written <= capacity);
        Ok(written as u32)
    }

    /// Write viewer-relative win summaries into `[4 * 5]` `i32` fields.
    ///
    /// Each row is `[winCount, lastShapeMultiplier, lastMultiplier,
    /// lastPatternBits, lastEventFlags]`. Last-win fields are `-1` when the
    /// relative player has not won.
    #[wasm_bindgen(js_name = playerUiStatsInto)]
    pub fn player_ui_stats_into(&self, viewer: u8, output: &mut [i32]) -> Result<(), JsValue> {
        let viewer = seat_value(viewer)?;
        require_len(output, core_engine::PLAYER_UI_STATS_WIDTH, "output")?;
        self.inner
            .player_ui_stats_into(viewer, output)
            .map_err(game_error)
    }

    /// Write viewer-relative wall-settlement metadata and revealed hands.
    ///
    /// Metadata rows are `[flowerPig, ready, maxShapeMultiplier]` and hands
    /// are four consecutive concealed 27-tile histograms. Returns `false`
    /// before a wall-exhaustion settlement is available.
    #[wasm_bindgen(js_name = wallSettlementInto)]
    pub fn wall_settlement_into(
        &self,
        viewer: u8,
        meta: &mut [i32],
        hands: &mut [u8],
    ) -> Result<bool, JsValue> {
        let viewer = seat_value(viewer)?;
        require_len(meta, core_engine::WALL_SETTLEMENT_META_WIDTH, "meta")?;
        require_len(hands, core_engine::WALL_SETTLEMENT_HANDS_WIDTH, "hands")?;
        self.inner
            .wall_settlement_into(viewer, meta, hands)
            .map_err(game_error)
    }

    /// Write the partial-information observation for one viewer.
    ///
    /// Buffer sizes must match the fixed training/observation contract:
    /// - `tile_obs`: `10 * 27` (`TILE_OBSERVATION_WIDTH`)
    /// - `melds`: `4 * 4 * 3` (`MELD_OBSERVATION_WIDTH`)
    /// - `river`: `108 * 2` (`RIVER_OBSERVATION_WIDTH`)
    /// - `meta`: `META_OBSERVATION_WIDTH`
    #[wasm_bindgen(js_name = observeInto)]
    pub fn observe_into(
        &self,
        viewer: u8,
        tile_obs: &mut [u8],
        melds: &mut [u8],
        river: &mut [u8],
        meta: &mut [i32],
    ) -> Result<(), JsValue> {
        let viewer = seat_value(viewer)?;
        require_len(tile_obs, core_engine::TILE_OBSERVATION_WIDTH, "tile_obs")?;
        require_len(melds, core_engine::MELD_OBSERVATION_WIDTH, "melds")?;
        require_len(river, core_engine::RIVER_OBSERVATION_WIDTH, "river")?;
        require_len(meta, core_engine::META_OBSERVATION_WIDTH, "meta")?;
        self.inner
            .observation_into(viewer, tile_obs, melds, river, meta)
            .map_err(game_error)
    }

    /// Write perfect-information tile counts. Training and diagnostics only.
    #[wasm_bindgen(js_name = oracleTileCountsInto)]
    pub fn oracle_tile_counts_into(&self, output: &mut [u8]) -> Result<(), JsValue> {
        require_len(output, core_engine::ORACLE_TILE_COUNT_PLANES * 27, "output")?;
        self.inner
            .oracle_tile_counts_into(output)
            .map_err(game_error)
    }

    #[wasm_bindgen(getter)]
    pub fn dealer(&self) -> u8 {
        self.inner.dealer().as_u8()
    }

    #[wasm_bindgen(getter, js_name = exchangeDirection)]
    pub fn exchange_direction(&self) -> u8 {
        self.inner.exchange_direction() as u8
    }

    #[wasm_bindgen(getter, js_name = wallRemaining)]
    pub fn wall_remaining(&self) -> u32 {
        self.inner.wall_remaining() as u32
    }

    /// Current draw as `[player, tile, replacementFlag]`, or `undefined`.
    #[wasm_bindgen(getter, js_name = currentDraw)]
    pub fn current_draw(&self) -> Option<Vec<u8>> {
        self.inner.current_draw().map(|draw| {
            vec![
                draw.player.as_u8(),
                draw.tile.as_u8(),
                u8::from(draw.replacement),
            ]
        })
    }

    /// Copy one seat's concealed histogram into a length-27 buffer.
    #[wasm_bindgen(js_name = concealedInto)]
    pub fn concealed_into(&self, seat: u8, output: &mut [u8]) -> Result<(), JsValue> {
        let seat = seat_value(seat)?;
        require_len(output, 27, "output")?;
        output.copy_from_slice(self.inner.concealed(seat));
        Ok(())
    }

    /// Copy one seat's locked histogram into a length-27 buffer.
    #[wasm_bindgen(js_name = lockedInto)]
    pub fn locked_into(&self, seat: u8, output: &mut [u8]) -> Result<(), JsValue> {
        let seat = seat_value(seat)?;
        require_len(output, 27, "output")?;
        output.copy_from_slice(self.inner.locked(seat));
        Ok(())
    }

    /// Copy one seat's stable winning base into a length-27 buffer.
    ///
    /// The browser requests this for the viewer seat so it can keep the base
    /// in place while presenting only newly accumulated winning references.
    #[wasm_bindgen(js_name = winBaseInto)]
    pub fn win_base_into(&self, seat: u8, output: &mut [u8]) -> Result<(), JsValue> {
        let seat = seat_value(seat)?;
        require_len(output, 27, "output")?;
        output.copy_from_slice(self.inner.win_base(seat));
        Ok(())
    }

    /// Copy one seat's exchange selection histogram into a length-27 buffer.
    #[wasm_bindgen(js_name = exchangeSelectionInto)]
    pub fn exchange_selection_into(&self, seat: u8, output: &mut [u8]) -> Result<(), JsValue> {
        let seat = seat_value(seat)?;
        require_len(output, 27, "output")?;
        output.copy_from_slice(self.inner.exchange_selection(seat));
        Ok(())
    }

    /// Structural shanten and improving-tile mask for one seat: `[shanten, mask]`.
    #[wasm_bindgen(js_name = handAnalysis)]
    pub fn hand_analysis(&self, seat: u8) -> Result<Vec<i32>, JsValue> {
        let analysis = self.inner.hand_analysis(seat_value(seat)?);
        Ok(vec![
            i32::from(analysis.shanten),
            analysis.improving_tiles as i32,
        ])
    }

    /// Absolute seat scores as a length-4 array.
    pub fn scores(&self) -> Vec<i32> {
        Seat::ALL
            .map(|seat| i32::try_from(self.inner.score(seat)).unwrap_or(i32::MAX))
            .to_vec()
    }

    /// Missing suits as length-4 array of `0..2`, or `-1` when unset.
    #[wasm_bindgen(js_name = missingSuits)]
    pub fn missing_suits(&self) -> Vec<i8> {
        Seat::ALL
            .map(|seat| self.inner.missing_suit(seat).map_or(-1, |suit| suit as i8))
            .to_vec()
    }

    #[wasm_bindgen(js_name = hasWon)]
    pub fn has_won(&self, seat: u8) -> Result<bool, JsValue> {
        Ok(self.inner.has_won(seat_value(seat)?))
    }

    /// Max win multipliers for absolute seats `0..3`.
    #[wasm_bindgen(js_name = maxWinMultipliers)]
    pub fn max_win_multipliers(&self) -> Vec<u32> {
        Seat::ALL
            .map(|seat| self.inner.max_win_multiplier(seat))
            .to_vec()
    }

    /// Melds for one absolute seat as a flat `[tile, kind, source] * count` array.
    pub fn melds(&self, seat: u8) -> Result<Vec<u8>, JsValue> {
        let seat = seat_value(seat)?;
        let mut out = Vec::with_capacity(self.inner.meld_count(seat) * 3);
        for index in 0..self.inner.meld_count(seat) {
            if let Some(meld) = self.inner.meld(seat, index) {
                out.push(meld.tile.as_u8());
                out.push(meld_kind_code(meld.kind));
                out.push(meld.source.as_u8());
            }
        }
        Ok(out)
    }

    /// Chronological discards as a flat `[seat, tile] * n` array.
    pub fn discards(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for (seat, tile) in self.inner.discards() {
            out.push(seat.as_u8());
            out.push(tile.as_u8());
        }
        out
    }

    /// Absolute seat ranking from first to last place.
    pub fn rankings(&self) -> Vec<u8> {
        self.inner.rankings().map(Seat::as_u8).to_vec()
    }

    #[wasm_bindgen(getter, js_name = terminationReason)]
    pub fn termination_reason(&self) -> Option<u8> {
        self.inner.termination_reason().map(TerminationReason::code)
    }
}

fn action_id(action: u8) -> Result<ActionId, JsValue> {
    ActionId::new(usize::from(action))
        .ok_or_else(|| JsValue::from_str(&format!("action id {action} is out of range")))
}

fn seat_value(seat: u8) -> Result<Seat, JsValue> {
    Seat::new(seat).ok_or_else(|| JsValue::from_str(&format!("seat {seat} is out of range")))
}

fn exchange_direction(direction: u8) -> Result<ExchangeDirection, JsValue> {
    match direction {
        1 => Ok(ExchangeDirection::Left),
        2 => Ok(ExchangeDirection::Across),
        3 => Ok(ExchangeDirection::Right),
        _ => Err(JsValue::from_str(&format!(
            "exchange direction must be 1, 2, or 3; got {direction}"
        ))),
    }
}

fn meld_kind_code(kind: MeldKind) -> u8 {
    kind.code()
}

fn game_error(error: GameError) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[cfg(feature = "rule-nn")]
fn rule_nn_error(error: core_engine::RuleNnError) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn config_range_error(name: &str, value: impl std::fmt::Display, range: &str) -> JsValue {
    JsValue::from_str(&format!("{name} must be in {range}, got {value}"))
}

fn build_planner_config(
    hand_changes: u8,
    draw_horizon: u8,
    candidate_states: u32,
    belief_worlds: u16,
    response_worlds: u16,
    search_iterations: u16,
) -> Result<RulePlannerConfig, JsValue> {
    RulePlannerConfig::DEFAULT
        .with_hand_changes(hand_changes)
        .ok_or_else(|| config_range_error("hand_changes", hand_changes, "0..=2"))?
        .with_draw_horizon(draw_horizon)
        .ok_or_else(|| config_range_error("draw_horizon", draw_horizon, "0..=32"))?
        .with_candidate_states(candidate_states)
        .ok_or_else(|| config_range_error("candidate_states", candidate_states, "1..=200000"))?
        .with_belief_worlds(belief_worlds)
        .ok_or_else(|| config_range_error("belief_worlds", belief_worlds, "0..=256"))?
        .with_response_worlds(response_worlds)
        .ok_or_else(|| config_range_error("response_worlds", response_worlds, "0..=256"))?
        .with_search_iterations(search_iterations)
        .ok_or_else(|| config_range_error("search_iterations", search_iterations, "0..=4096"))
}

fn require_len<T>(slice: &[T], expected: usize, name: &str) -> Result<(), JsValue> {
    if slice.len() != expected {
        return Err(JsValue::from_str(&format!(
            "{name} length must be {expected}, got {}",
            slice.len()
        )));
    }
    Ok(())
}

fn validate_event_buffer(output: &[i32], name: &str) -> Result<usize, JsValue> {
    let record_width = core_engine::EVENT_RECORD_WIDTH;
    let history_capacity = core_engine::EVENT_HISTORY_CAPACITY;
    if output.len() % record_width != 0 {
        return Err(JsValue::from_str(&format!(
            "{name} length must be a multiple of {record_width}"
        )));
    }
    let capacity = output.len() / record_width;
    if capacity == 0 || capacity > history_capacity {
        return Err(JsValue::from_str(&format!(
            "{name} capacity must be in 1..={history_capacity}, got {capacity}"
        )));
    }
    Ok(capacity)
}

fn encode_outcome_i32(outcome: StepOutcome) -> [i32; core_engine::STEP_RECORD_WIDTH] {
    let mut record = [0_i32; core_engine::STEP_RECORD_WIDTH];
    record[0] = outcome
        .draw
        .map_or(-1, |draw| i32::from(draw.player.as_u8()));
    record[1] = outcome.draw.map_or(-1, |draw| i32::from(draw.tile.as_u8()));
    record[2] = outcome.draw.map_or(0, |draw| i32::from(draw.replacement));
    record[3] = outcome
        .discard
        .map_or(-1, |discard| i32::from(discard.player.as_u8()));
    record[4] = outcome
        .discard
        .map_or(-1, |discard| i32::from(discard.tile.as_u8()));
    for (index, delta) in outcome.score_delta.iter().enumerate() {
        record[5 + index] =
            i32::try_from(*delta).unwrap_or(if *delta < 0 { i32::MIN } else { i32::MAX });
    }
    record[9] = outcome
        .next
        .map_or(-1, |decision| i32::from(decision.actor.as_u8()));
    record[10] = outcome
        .next
        .map_or(-1, |decision| i32::from(decision.phase.code()));
    record[11] = i32::from(outcome.terminal);
    record
}
