use ::bloodflow_mahjong as core_engine;
#[cfg(feature = "rule-nn")]
use core_engine::RuleNn;
use core_engine::{
    ACTION_ADDED_KONG_OFFSET, ACTION_CHOOSE_MISSING_OFFSET, ACTION_CONCEALED_KONG_OFFSET,
    ACTION_DISCARD_OFFSET, ACTION_EXCHANGE_TILE_OFFSET, ACTION_EXPOSED_KONG, ACTION_HU,
    ACTION_PASS, ACTION_PONG, ACTION_SPACE_SIZE, ActionId, Batch, ENGINE_RULES_VERSION,
    EVENT_FLAG_AFTER_KONG, EVENT_FLAG_EARTHLY, EVENT_FLAG_HEAVENLY, EVENT_FLAG_LAST_WALL_TILE,
    EVENT_FLAG_OPENING_DISCARD, EVENT_FLAG_REPLACEMENT_DRAW, EVENT_FLAG_ROB_KONG,
    EVENT_FLAG_SELF_DRAW, EVENT_HISTORY_CAPACITY, EVENT_RECORD_WIDTH, EventKind, ExchangeDirection,
    Game, GameError, LEGAL_ACTION_MASK_WORDS, MELD_OBSERVATION_WIDTH, META_OBSERVATION_WIDTH,
    MeldKind, ORACLE_TILE_COUNT_PLANES, Phase, RIVER_OBSERVATION_WIDTH, RULE_EV_ACTION_TERMINAL,
    RULE_PLANNER_ACTION_TERMINAL, RuleEvConfig, RuleEvDefense, RulePlannerConfig, SHANTEN_COMPLETE,
    SHANTEN_MAX, SHANTEN_TERMINAL, SIMPLE_RULE_ACTION_TERMINAL, STEP_RECORD_WIDTH, Seat,
    StepOutcome, TILE_OBSERVATION_WIDTH, TerminationReason,
};
use numpy::{
    PyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray,
    PyReadwriteArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

const PLAYER_COUNT: usize = 4;
const TILE_KIND_COUNT: usize = 27;
const TILE_OBSERVATION_PLANES: usize = 10;
const MELD_SLOTS: usize = 4;
const MELD_FIELDS: usize = 3;
const RIVER_TILE_CAPACITY: usize = 108;
const RIVER_FIELDS: usize = 2;

type StepRecordTuple = (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64);

#[cfg(feature = "rule-nn")]
#[pyclass(frozen, name = "RuleNn", module = "bloodflow_mahjong")]
struct PyRuleNn {
    inner: RuleNn,
}

#[cfg(feature = "rule-nn")]
#[pymethods]
impl PyRuleNn {
    #[new]
    fn new(py: Python<'_>, onnx: &[u8]) -> PyResult<Self> {
        py.detach(|| RuleNn::from_onnx_bytes(onnx))
            .map(|inner| Self { inner })
            .map_err(rule_nn_model_error)
    }

    #[staticmethod]
    fn from_file(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Self> {
        let onnx = std::fs::read(path)?;
        py.detach(move || RuleNn::from_onnx_bytes(&onnx))
            .map(|inner| Self { inner })
            .map_err(rule_nn_model_error)
    }

    fn action(&self, py: Python<'_>, game: &PyGame) -> PyResult<Option<u8>> {
        py.detach(|| self.inner.action(&game.inner))
            .map(|action| action.map(|id| id.index() as u8))
            .map_err(rule_nn_inference_error)
    }

    fn __repr__(&self) -> &'static str {
        "RuleNn()"
    }
}

#[pyclass(frozen, name = "RuleEvConfig", module = "bloodflow_mahjong")]
struct PyRuleEvConfig {
    inner: RuleEvConfig,
}

#[pymethods]
impl PyRuleEvConfig {
    #[new]
    #[pyo3(signature = (search_depth=1, defense=true))]
    fn new(search_depth: u8, defense: bool) -> PyResult<Self> {
        let inner = RuleEvConfig::with_search_depth(search_depth)
            .ok_or_else(|| config_range_error("search_depth", search_depth, "0..=3"))?
            .with_defense(if defense {
                RuleEvDefense::Heuristic
            } else {
                RuleEvDefense::None
            });
        Ok(Self { inner })
    }

    #[staticmethod]
    fn fast() -> Self {
        Self {
            inner: RuleEvConfig::FAST,
        }
    }

    #[staticmethod]
    fn standard() -> Self {
        Self {
            inner: RuleEvConfig::STANDARD,
        }
    }

    #[getter]
    fn search_depth(&self) -> u8 {
        self.inner.search_depth()
    }

    #[getter]
    fn defense(&self) -> bool {
        self.inner.defense() == RuleEvDefense::Heuristic
    }

    fn __repr__(&self) -> String {
        format!(
            "RuleEvConfig(search_depth={}, defense={})",
            self.search_depth(),
            python_bool(self.defense())
        )
    }
}

#[pyclass(frozen, name = "RulePlannerConfig", module = "bloodflow_mahjong")]
struct PyRulePlannerConfig {
    inner: RulePlannerConfig,
}

#[pymethods]
impl PyRulePlannerConfig {
    #[new]
    #[pyo3(signature = (
        hand_changes=0,
        draw_horizon=1,
        candidate_states=1,
        belief_worlds=64,
        response_worlds=0,
        search_iterations=64,
    ))]
    fn new(
        hand_changes: u8,
        draw_horizon: u8,
        candidate_states: u32,
        belief_worlds: u16,
        response_worlds: u16,
        search_iterations: u16,
    ) -> PyResult<Self> {
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

    #[getter]
    fn hand_changes(&self) -> u8 {
        self.inner.hand_changes()
    }

    #[getter]
    fn draw_horizon(&self) -> u8 {
        self.inner.draw_horizon()
    }

    #[getter]
    fn candidate_states(&self) -> u32 {
        self.inner.candidate_states()
    }

    #[getter]
    fn belief_worlds(&self) -> u16 {
        self.inner.belief_worlds()
    }

    #[getter]
    fn response_worlds(&self) -> u16 {
        self.inner.response_worlds()
    }

    #[getter]
    fn search_iterations(&self) -> u16 {
        self.inner.search_iterations()
    }

    fn __repr__(&self) -> String {
        format!(
            concat!(
                "RulePlannerConfig(hand_changes={}, draw_horizon={}, ",
                "candidate_states={}, belief_worlds={}, response_worlds={}, ",
                "search_iterations={})"
            ),
            self.hand_changes(),
            self.draw_horizon(),
            self.candidate_states(),
            self.belief_worlds(),
            self.response_worlds(),
            self.search_iterations(),
        )
    }
}

#[pyclass(name = "Game", module = "bloodflow_mahjong")]
struct PyGame {
    inner: Game,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (seed=0))]
    fn new(seed: u64) -> Self {
        Self {
            inner: Game::new(seed),
        }
    }

    #[staticmethod]
    fn with_exchange_direction(seed: u64, direction: u8) -> PyResult<Self> {
        let direction = exchange_direction(direction)?;
        Ok(Self {
            inner: Game::new_with_direction(seed, direction),
        })
    }

    fn reset(&mut self, seed: u64) {
        self.inner.reset(seed);
    }

    fn resample_information_set(&self, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: self
                .inner
                .resample_information_set(seed)
                .map_err(game_error)?,
        })
    }

    #[getter]
    fn phase(&self) -> u8 {
        phase_code(self.inner.phase())
    }

    #[getter]
    fn decision(&self) -> Option<(u8, u8)> {
        self.inner
            .decision()
            .map(|decision| (decision.actor.as_u8(), phase_code(decision.phase)))
    }

    #[getter]
    fn legal_action_mask(&self) -> (u64, u64) {
        self.inner
            .legal_action_mask()
            .map_or((0, 0), |mask| (mask.words()[0], mask.words()[1]))
    }

    /// Returns the deterministic baseline action, or `None` after terminal.
    fn simple_rule_action(&self) -> Option<u8> {
        self.inner
            .simple_rule_action()
            .map(|action| action.index() as u8)
    }

    /// Returns a rule-EV action, or `None` after terminal.
    #[pyo3(signature = (config=None))]
    fn rule_ev_action(
        &self,
        py: Python<'_>,
        config: Option<PyRef<'_, PyRuleEvConfig>>,
    ) -> Option<u8> {
        let config = config.map_or(RuleEvConfig::STANDARD, |value| value.inner);
        py.detach(|| self.inner.rule_ev_action_with_config(config))
            .map(|action| action.index() as u8)
    }

    /// Returns a rule-planner action, or `None` after terminal.
    #[pyo3(signature = (config=None))]
    fn rule_planner_action(
        &self,
        py: Python<'_>,
        config: Option<PyRef<'_, PyRulePlannerConfig>>,
    ) -> Option<u8> {
        let config = config.map_or(RulePlannerConfig::DEFAULT, |value| value.inner);
        py.detach(|| self.inner.rule_planner_action_with_config(config))
            .map(|action| action.index() as u8)
    }

    fn step_id(&mut self, action: u8) -> PyResult<StepRecordTuple> {
        let action = action_id(action)?;
        let outcome = self.inner.step_id(action).map_err(game_error)?;
        Ok(outcome_tuple(outcome))
    }

    fn step_into<'py>(&mut self, action: u8, output: &Bound<'py, PyArray1<i64>>) -> PyResult<()> {
        require_shape(output, &[STEP_RECORD_WIDTH], "output")?;
        require_c_contiguous(output, "output")?;
        let action = action_id(action)?;
        let mut output = try_readwrite(output, "output")?;
        let output = writable_slice(&mut output, "output")?;
        let outcome = self.inner.step_id(action).map_err(game_error)?;
        encode_outcome(outcome, output);
        Ok(())
    }

    #[getter]
    fn event_count(&self) -> usize {
        self.inner.event_count()
    }

    #[getter]
    fn event_dropped(&self) -> u64 {
        self.inner.event_dropped()
    }

    fn events_into<'py>(
        &self,
        py: Python<'py>,
        viewer: u8,
        output: &Bound<'py, PyArray2<i32>>,
    ) -> PyResult<usize> {
        let viewer = seat_value(viewer)?;
        validate_event_array(output, "output")?;
        require_c_contiguous(output, "output")?;
        let mut output = try_readwrite(output, "output")?;
        let output = writable_slice(&mut output, "output")?;
        py.detach(|| self.inner.events_into(viewer, output))
            .map_err(game_error)
    }

    fn step_events_into<'py>(
        &self,
        py: Python<'py>,
        viewer: u8,
        output: &Bound<'py, PyArray2<i32>>,
    ) -> PyResult<usize> {
        let viewer = seat_value(viewer)?;
        validate_event_array(output, "output")?;
        require_c_contiguous(output, "output")?;
        let mut output = try_readwrite(output, "output")?;
        let output = writable_slice(&mut output, "output")?;
        py.detach(|| self.inner.step_events_into(viewer, output))
            .map_err(game_error)
    }

    fn observe_into<'py>(
        &self,
        viewer: u8,
        tile_obs: &Bound<'py, PyArray2<u8>>,
        melds: &Bound<'py, PyArray3<u8>>,
        river: &Bound<'py, PyArray2<u8>>,
        meta: &Bound<'py, PyArray1<i32>>,
    ) -> PyResult<()> {
        let viewer = seat_value(viewer)?;
        require_shape(
            tile_obs,
            &[TILE_OBSERVATION_PLANES, TILE_KIND_COUNT],
            "tile_obs",
        )?;
        require_c_contiguous(tile_obs, "tile_obs")?;
        require_shape(melds, &[PLAYER_COUNT, MELD_SLOTS, MELD_FIELDS], "melds")?;
        require_c_contiguous(melds, "melds")?;
        require_shape(river, &[RIVER_TILE_CAPACITY, RIVER_FIELDS], "river")?;
        require_c_contiguous(river, "river")?;
        require_shape(meta, &[META_OBSERVATION_WIDTH], "meta")?;
        require_c_contiguous(meta, "meta")?;

        let mut tile_obs = try_readwrite(tile_obs, "tile_obs")?;
        let mut melds = try_readwrite(melds, "melds")?;
        let mut river = try_readwrite(river, "river")?;
        let mut meta = try_readwrite(meta, "meta")?;
        self.inner
            .observation_into(
                viewer,
                writable_slice(&mut tile_obs, "tile_obs")?,
                writable_slice(&mut melds, "melds")?,
                writable_slice(&mut river, "river")?,
                writable_slice(&mut meta, "meta")?,
            )
            .map_err(game_error)
    }

    fn oracle_tile_counts_into<'py>(&self, output: &Bound<'py, PyArray2<u8>>) -> PyResult<()> {
        require_shape(
            output,
            &[ORACLE_TILE_COUNT_PLANES, TILE_KIND_COUNT],
            "output",
        )?;
        require_c_contiguous(output, "output")?;
        let mut output = try_readwrite(output, "output")?;
        self.inner
            .oracle_tile_counts_into(writable_slice(&mut output, "output")?)
            .map_err(game_error)
    }

    #[getter]
    fn dealer(&self) -> u8 {
        self.inner.dealer().as_u8()
    }

    #[getter]
    fn exchange_direction(&self) -> u8 {
        self.inner.exchange_direction() as u8
    }

    #[getter]
    fn wall_remaining(&self) -> usize {
        self.inner.wall_remaining()
    }

    #[getter]
    fn current_draw(&self) -> Option<(u8, u8, bool)> {
        self.inner
            .current_draw()
            .map(|draw| (draw.player.as_u8(), draw.tile.as_u8(), draw.replacement))
    }

    fn concealed_into<'py>(&self, seat: u8, output: &Bound<'py, PyArray1<u8>>) -> PyResult<()> {
        copy_tiles_into(self.inner.concealed(seat_value(seat)?), output, "output")
    }

    fn locked_into<'py>(&self, seat: u8, output: &Bound<'py, PyArray1<u8>>) -> PyResult<()> {
        copy_tiles_into(self.inner.locked(seat_value(seat)?), output, "output")
    }

    fn exchange_selection_into<'py>(
        &self,
        seat: u8,
        output: &Bound<'py, PyArray1<u8>>,
    ) -> PyResult<()> {
        copy_tiles_into(
            self.inner.exchange_selection(seat_value(seat)?),
            output,
            "output",
        )
    }

    /// Returns conventional structural shanten and an improving-tile bitmask.
    fn hand_analysis(&self, seat: u8) -> PyResult<(i8, u32)> {
        let analysis = self.inner.hand_analysis(seat_value(seat)?);
        Ok((analysis.shanten, analysis.improving_tiles))
    }

    fn scores(&self) -> (i64, i64, i64, i64) {
        let scores = Seat::ALL.map(|seat| self.inner.score(seat));
        (scores[0], scores[1], scores[2], scores[3])
    }

    fn missing_suits(&self) -> (i8, i8, i8, i8) {
        let suits =
            Seat::ALL.map(|seat| self.inner.missing_suit(seat).map_or(-1, |suit| suit as i8));
        (suits[0], suits[1], suits[2], suits[3])
    }

    fn has_won(&self, seat: u8) -> PyResult<bool> {
        Ok(self.inner.has_won(seat_value(seat)?))
    }

    fn max_win_multipliers(&self) -> (u32, u32, u32, u32) {
        let values = Seat::ALL.map(|seat| self.inner.max_win_multiplier(seat));
        (values[0], values[1], values[2], values[3])
    }

    fn melds(&self, seat: u8) -> PyResult<Vec<(u8, u8, u8)>> {
        let seat = seat_value(seat)?;
        Ok((0..self.inner.meld_count(seat))
            .filter_map(|index| self.inner.meld(seat, index))
            .map(|meld| {
                (
                    meld.tile.as_u8(),
                    meld_kind_code(meld.kind),
                    meld.source.as_u8(),
                )
            })
            .collect())
    }

    fn discards(&self) -> Vec<(u8, u8)> {
        self.inner
            .discards()
            .map(|(seat, tile)| (seat.as_u8(), tile.as_u8()))
            .collect()
    }

    fn rankings(&self) -> (u8, u8, u8, u8) {
        let seats = self.inner.rankings().map(Seat::as_u8);
        (seats[0], seats[1], seats[2], seats[3])
    }

    #[getter]
    fn termination_reason(&self) -> Option<u8> {
        self.inner.termination_reason().map(TerminationReason::code)
    }
}

#[pyclass(name = "Batch", module = "bloodflow_mahjong")]
struct PyBatch {
    inner: Batch,
}

#[pymethods]
impl PyBatch {
    #[new]
    #[pyo3(signature = (size, seed=0))]
    fn new(py: Python<'_>, size: usize, seed: u64) -> Self {
        Self {
            inner: py.detach(|| Batch::new(size, seed)),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn event_dropped_into<'py>(
        &self,
        py: Python<'py>,
        output: &Bound<'py, PyArray1<u64>>,
    ) -> PyResult<()> {
        require_shape(output, &[self.inner.len()], "output")?;
        require_c_contiguous(output, "output")?;
        let mut output = try_readwrite(output, "output")?;
        let output = writable_slice(&mut output, "output")?;
        py.detach(|| self.inner.event_dropped_into(output))
            .map_err(game_error)
    }

    fn reset_all(&mut self, py: Python<'_>, seed: u64) {
        py.detach(|| self.inner.reset_all(seed));
    }

    fn reset_at(&mut self, py: Python<'_>, index: usize, seed: u64) -> PyResult<()> {
        py.detach(|| self.inner.reset_at(index, seed))
            .map_err(game_error)
    }

    fn reset_many<'py>(
        &mut self,
        py: Python<'py>,
        indices: &Bound<'py, PyArray1<u32>>,
        seeds: &Bound<'py, PyArray1<u64>>,
    ) -> PyResult<()> {
        require_c_contiguous(indices, "indices")?;
        require_shape(seeds, &[indices.len()], "seeds")?;
        require_c_contiguous(seeds, "seeds")?;

        let indices = try_readonly(indices, "indices")?;
        let seeds = try_readonly(seeds, "seeds")?;
        let indices = readonly_slice(&indices, "indices")?;
        let seeds = readonly_slice(&seeds, "seeds")?;
        if let Some(&index) = indices
            .iter()
            .find(|&&index| index as usize >= self.inner.len())
        {
            return Err(PyValueError::new_err(format!(
                "batch index {index} is out of range for batch size {}",
                self.inner.len()
            )));
        }

        py.detach(|| -> Result<(), GameError> {
            for (&index, &seed) in indices.iter().zip(seeds) {
                self.inner.reset_at(index as usize, seed)?;
            }
            Ok(())
        })
        .map_err(game_error)
    }

    fn clone_indices<'py>(
        &self,
        py: Python<'py>,
        indices: &Bound<'py, PyArray1<u32>>,
    ) -> PyResult<Self> {
        require_c_contiguous(indices, "indices")?;
        let indices = try_readonly(indices, "indices")?;
        let indices = readonly_slice(&indices, "indices")?;
        if let Some(&index) = indices
            .iter()
            .find(|&&index| index as usize >= self.inner.len())
        {
            return Err(PyValueError::new_err(format!(
                "batch index {index} is out of range for batch size {}",
                self.inner.len()
            )));
        }
        let indices = indices
            .iter()
            .map(|&index| index as usize)
            .collect::<Vec<_>>();
        let inner = py
            .detach(|| self.inner.clone_indices(&indices))
            .map_err(game_error)?;
        Ok(Self { inner })
    }

    fn remove_indices_swap<'py>(
        &mut self,
        py: Python<'py>,
        indices: &Bound<'py, PyArray1<u32>>,
    ) -> PyResult<Vec<u32>> {
        require_c_contiguous(indices, "indices")?;
        let indices = try_readonly(indices, "indices")?;
        let indices = readonly_slice(&indices, "indices")?;
        if let Some(&index) = indices
            .iter()
            .find(|&&index| index as usize >= self.inner.len())
        {
            return Err(PyValueError::new_err(format!(
                "batch index {index} is out of range for batch size {}",
                self.inner.len()
            )));
        }
        if indices.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PyValueError::new_err(
                "removed indices must be strictly increasing",
            ));
        }
        let indices = indices
            .iter()
            .map(|&index| index as usize)
            .collect::<Vec<_>>();
        py.detach(|| self.inner.remove_indices_swap(&indices))
            .map(|rows| rows.into_iter().map(|row| row as u32).collect())
            .map_err(game_error)
    }

    fn resample_information_sets<'py>(
        &self,
        py: Python<'py>,
        indices: &Bound<'py, PyArray1<u32>>,
        seeds: &Bound<'py, PyArray1<u64>>,
    ) -> PyResult<Self> {
        require_c_contiguous(indices, "indices")?;
        require_shape(seeds, &[indices.len()], "seeds")?;
        require_c_contiguous(seeds, "seeds")?;
        let indices = try_readonly(indices, "indices")?;
        let seeds = try_readonly(seeds, "seeds")?;
        let indices = readonly_slice(&indices, "indices")?;
        let seeds = readonly_slice(&seeds, "seeds")?;
        if let Some(&index) = indices
            .iter()
            .find(|&&index| index as usize >= self.inner.len())
        {
            return Err(PyValueError::new_err(format!(
                "batch index {index} is out of range for batch size {}",
                self.inner.len()
            )));
        }
        let indices = indices
            .iter()
            .map(|&index| index as usize)
            .collect::<Vec<_>>();
        let inner = py
            .detach(|| self.inner.resample_information_sets(&indices, seeds))
            .map_err(game_error)?;
        Ok(Self { inner })
    }

    fn resample_live_walls<'py>(
        &self,
        py: Python<'py>,
        indices: &Bound<'py, PyArray1<u32>>,
        seeds: &Bound<'py, PyArray1<u64>>,
    ) -> PyResult<Self> {
        require_c_contiguous(indices, "indices")?;
        require_shape(seeds, &[indices.len()], "seeds")?;
        require_c_contiguous(seeds, "seeds")?;
        let indices = try_readonly(indices, "indices")?;
        let seeds = try_readonly(seeds, "seeds")?;
        let indices = readonly_slice(&indices, "indices")?;
        let seeds = readonly_slice(&seeds, "seeds")?;
        if let Some(&index) = indices
            .iter()
            .find(|&&index| index as usize >= self.inner.len())
        {
            return Err(PyValueError::new_err(format!(
                "batch index {index} is out of range for batch size {}",
                self.inner.len()
            )));
        }
        let indices = indices
            .iter()
            .map(|&index| index as usize)
            .collect::<Vec<_>>();
        let inner = py
            .detach(|| self.inner.resample_live_walls(&indices, seeds))
            .map_err(game_error)?;
        Ok(Self { inner })
    }

    fn oracle_tile_counts_into<'py>(
        &self,
        py: Python<'py>,
        output: &Bound<'py, PyArray3<u8>>,
    ) -> PyResult<()> {
        require_shape(
            output,
            &[self.inner.len(), ORACLE_TILE_COUNT_PLANES, TILE_KIND_COUNT],
            "output",
        )?;
        require_c_contiguous(output, "output")?;
        let mut output = try_readwrite(output, "output")?;
        let output = writable_slice(&mut output, "output")?;
        py.detach(|| self.inner.oracle_tile_counts_into(output))
            .map_err(game_error)
    }

    fn legal_action_masks_into<'py>(
        &self,
        py: Python<'py>,
        output: &Bound<'py, PyArray2<u64>>,
    ) -> PyResult<()> {
        require_shape(
            output,
            &[self.inner.len(), LEGAL_ACTION_MASK_WORDS],
            "output",
        )?;
        require_c_contiguous(output, "output")?;
        let mut output = try_readwrite(output, "output")?;
        let output = writable_slice(&mut output, "output")?;
        py.detach(|| self.inner.legal_action_mask_words_into(output))
            .map_err(game_error)
    }

    fn simple_rule_actions_into<'py>(
        &self,
        py: Python<'py>,
        output: &Bound<'py, PyArray1<u8>>,
    ) -> PyResult<()> {
        require_shape(output, &[self.inner.len()], "output")?;
        require_c_contiguous(output, "output")?;
        let mut output = try_readwrite(output, "output")?;
        let output = writable_slice(&mut output, "output")?;
        py.detach(|| self.inner.simple_rule_actions_into(output))
            .map_err(game_error)
    }

    fn simple_rule_actions_masked_into<'py>(
        &self,
        py: Python<'py>,
        enabled: &Bound<'py, PyArray1<u8>>,
        output: &Bound<'py, PyArray1<u8>>,
    ) -> PyResult<()> {
        require_shape(enabled, &[self.inner.len()], "enabled")?;
        require_c_contiguous(enabled, "enabled")?;
        require_shape(output, &[self.inner.len()], "output")?;
        require_c_contiguous(output, "output")?;
        let enabled = try_readonly(enabled, "enabled")?;
        let mut output = try_readwrite(output, "output")?;
        let enabled = readonly_slice(&enabled, "enabled")?;
        let output = writable_slice(&mut output, "output")?;
        py.detach(|| self.inner.simple_rule_actions_masked_into(enabled, output))
            .map_err(game_error)
    }

    #[pyo3(signature = (output, config=None))]
    fn rule_ev_actions_into<'py>(
        &self,
        py: Python<'py>,
        output: &Bound<'py, PyArray1<u8>>,
        config: Option<PyRef<'py, PyRuleEvConfig>>,
    ) -> PyResult<()> {
        require_shape(output, &[self.inner.len()], "output")?;
        require_c_contiguous(output, "output")?;
        let config = config.map_or(RuleEvConfig::STANDARD, |value| value.inner);
        let mut output = try_readwrite(output, "output")?;
        let output = writable_slice(&mut output, "output")?;
        py.detach(|| self.inner.rule_ev_actions_with_config_into(config, output))
            .map_err(game_error)
    }

    #[pyo3(signature = (enabled, output, config=None))]
    fn rule_ev_actions_masked_into<'py>(
        &self,
        py: Python<'py>,
        enabled: &Bound<'py, PyArray1<u8>>,
        output: &Bound<'py, PyArray1<u8>>,
        config: Option<PyRef<'py, PyRuleEvConfig>>,
    ) -> PyResult<()> {
        require_shape(enabled, &[self.inner.len()], "enabled")?;
        require_c_contiguous(enabled, "enabled")?;
        require_shape(output, &[self.inner.len()], "output")?;
        require_c_contiguous(output, "output")?;
        let config = config.map_or(RuleEvConfig::STANDARD, |value| value.inner);
        let enabled = try_readonly(enabled, "enabled")?;
        let mut output = try_readwrite(output, "output")?;
        let enabled = readonly_slice(&enabled, "enabled")?;
        let output = writable_slice(&mut output, "output")?;
        py.detach(|| {
            self.inner
                .rule_ev_actions_masked_with_config_into(enabled, config, output)
        })
        .map_err(game_error)
    }

    #[pyo3(signature = (output, config=None))]
    fn rule_planner_actions_into<'py>(
        &self,
        py: Python<'py>,
        output: &Bound<'py, PyArray1<u8>>,
        config: Option<PyRef<'py, PyRulePlannerConfig>>,
    ) -> PyResult<()> {
        require_shape(output, &[self.inner.len()], "output")?;
        require_c_contiguous(output, "output")?;
        let config = config.map_or(RulePlannerConfig::DEFAULT, |value| value.inner);
        let mut output = try_readwrite(output, "output")?;
        let output = writable_slice(&mut output, "output")?;
        py.detach(|| {
            self.inner
                .rule_planner_actions_with_config_into(config, output)
        })
        .map_err(game_error)
    }

    #[pyo3(signature = (enabled, output, config=None))]
    fn rule_planner_actions_masked_into<'py>(
        &self,
        py: Python<'py>,
        enabled: &Bound<'py, PyArray1<u8>>,
        output: &Bound<'py, PyArray1<u8>>,
        config: Option<PyRef<'py, PyRulePlannerConfig>>,
    ) -> PyResult<()> {
        require_shape(enabled, &[self.inner.len()], "enabled")?;
        require_c_contiguous(enabled, "enabled")?;
        require_shape(output, &[self.inner.len()], "output")?;
        require_c_contiguous(output, "output")?;
        let config = config.map_or(RulePlannerConfig::DEFAULT, |value| value.inner);
        let enabled = try_readonly(enabled, "enabled")?;
        let mut output = try_readwrite(output, "output")?;
        let enabled = readonly_slice(&enabled, "enabled")?;
        let output = writable_slice(&mut output, "output")?;
        py.detach(|| {
            self.inner
                .rule_planner_actions_masked_with_config_into(enabled, config, output)
        })
        .map_err(game_error)
    }

    fn hand_analysis_into<'py>(
        &self,
        py: Python<'py>,
        shanten: &Bound<'py, PyArray1<i8>>,
        improving_tiles: &Bound<'py, PyArray1<u32>>,
    ) -> PyResult<()> {
        let batch_size = self.inner.len();
        require_shape(shanten, &[batch_size], "shanten")?;
        require_c_contiguous(shanten, "shanten")?;
        require_shape(improving_tiles, &[batch_size], "improving_tiles")?;
        require_c_contiguous(improving_tiles, "improving_tiles")?;

        let mut shanten = try_readwrite(shanten, "shanten")?;
        let mut improving_tiles = try_readwrite(improving_tiles, "improving_tiles")?;
        let shanten = writable_slice(&mut shanten, "shanten")?;
        let improving_tiles = writable_slice(&mut improving_tiles, "improving_tiles")?;
        py.detach(|| self.inner.hand_analysis_into(shanten, improving_tiles))
            .map_err(game_error)
    }

    fn hand_analysis_indices_into<'py>(
        &self,
        py: Python<'py>,
        indices: &Bound<'py, PyArray1<u32>>,
        shanten: &Bound<'py, PyArray1<i8>>,
        improving_tiles: &Bound<'py, PyArray1<u32>>,
    ) -> PyResult<()> {
        require_c_contiguous(indices, "indices")?;
        require_shape(shanten, &[indices.len()], "shanten")?;
        require_c_contiguous(shanten, "shanten")?;
        require_shape(improving_tiles, &[indices.len()], "improving_tiles")?;
        require_c_contiguous(improving_tiles, "improving_tiles")?;

        let indices = try_readonly(indices, "indices")?;
        let mut shanten = try_readwrite(shanten, "shanten")?;
        let mut improving_tiles = try_readwrite(improving_tiles, "improving_tiles")?;
        let indices = readonly_slice(&indices, "indices")?;
        let shanten = writable_slice(&mut shanten, "shanten")?;
        let improving_tiles = writable_slice(&mut improving_tiles, "improving_tiles")?;
        py.detach(|| {
            self.inner
                .hand_analysis_indices_into(indices, shanten, improving_tiles)
        })
        .map_err(game_error)
    }

    fn events_into<'py>(
        &self,
        py: Python<'py>,
        events: &Bound<'py, PyArray3<i32>>,
        lengths: &Bound<'py, PyArray1<u16>>,
    ) -> PyResult<()> {
        let capacity = validate_batch_event_array(self.inner.len(), events, "events")?;
        require_shape(lengths, &[self.inner.len()], "lengths")?;
        require_c_contiguous(lengths, "lengths")?;

        let mut events = try_readwrite(events, "events")?;
        let mut lengths = try_readwrite(lengths, "lengths")?;
        let events = writable_slice(&mut events, "events")?;
        let lengths = writable_slice(&mut lengths, "lengths")?;
        py.detach(|| self.inner.events_into(capacity, events, lengths))
            .map_err(game_error)
    }

    fn events_masked_into<'py>(
        &self,
        py: Python<'py>,
        history_seat_masks: &Bound<'py, PyArray1<u8>>,
        events: &Bound<'py, PyArray3<i32>>,
        lengths: &Bound<'py, PyArray1<u16>>,
    ) -> PyResult<()> {
        let batch_size = self.inner.len();
        require_shape(history_seat_masks, &[batch_size], "history_seat_masks")?;
        require_c_contiguous(history_seat_masks, "history_seat_masks")?;
        let capacity = validate_batch_event_array(batch_size, events, "events")?;
        require_shape(lengths, &[batch_size], "lengths")?;
        require_c_contiguous(lengths, "lengths")?;

        let history_seat_masks = try_readonly(history_seat_masks, "history_seat_masks")?;
        let mut events = try_readwrite(events, "events")?;
        let mut lengths = try_readwrite(lengths, "lengths")?;
        let history_seat_masks = readonly_slice(&history_seat_masks, "history_seat_masks")?;
        let events = writable_slice(&mut events, "events")?;
        let lengths = writable_slice(&mut lengths, "lengths")?;
        py.detach(|| {
            self.inner
                .events_masked_into(history_seat_masks, capacity, events, lengths)
        })
        .map_err(game_error)
    }

    fn step_events_into<'py>(
        &self,
        py: Python<'py>,
        events: &Bound<'py, PyArray3<i32>>,
        lengths: &Bound<'py, PyArray1<u16>>,
    ) -> PyResult<()> {
        let capacity = validate_batch_event_array(self.inner.len(), events, "events")?;
        require_shape(lengths, &[self.inner.len()], "lengths")?;
        require_c_contiguous(lengths, "lengths")?;

        let mut events = try_readwrite(events, "events")?;
        let mut lengths = try_readwrite(lengths, "lengths")?;
        let events = writable_slice(&mut events, "events")?;
        let lengths = writable_slice(&mut lengths, "lengths")?;
        py.detach(|| self.inner.step_events_into(capacity, events, lengths))
            .map_err(game_error)
    }

    fn step_into<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyArray1<u8>>,
        records: &Bound<'py, PyArray2<i64>>,
    ) -> PyResult<()> {
        require_shape(actions, &[self.inner.len()], "actions")?;
        require_c_contiguous(actions, "actions")?;
        require_shape(records, &[self.inner.len(), STEP_RECORD_WIDTH], "records")?;
        require_c_contiguous(records, "records")?;

        let actions = try_readonly(actions, "actions")?;
        let mut records = try_readwrite(records, "records")?;
        let actions = readonly_slice(&actions, "actions")?;
        let records = writable_slice(&mut records, "records")?;
        py.detach(|| self.inner.step_indices_into(actions, records))
            .map_err(game_error)
    }

    fn step_masked_into<'py>(
        &mut self,
        py: Python<'py>,
        enabled: &Bound<'py, PyArray1<u8>>,
        actions: &Bound<'py, PyArray1<u8>>,
        records: &Bound<'py, PyArray2<i64>>,
    ) -> PyResult<()> {
        require_shape(enabled, &[self.inner.len()], "enabled")?;
        require_c_contiguous(enabled, "enabled")?;
        require_shape(actions, &[self.inner.len()], "actions")?;
        require_c_contiguous(actions, "actions")?;
        require_shape(records, &[self.inner.len(), STEP_RECORD_WIDTH], "records")?;
        require_c_contiguous(records, "records")?;

        let enabled = try_readonly(enabled, "enabled")?;
        let actions = try_readonly(actions, "actions")?;
        let mut records = try_readwrite(records, "records")?;
        let enabled = readonly_slice(&enabled, "enabled")?;
        let actions = readonly_slice(&actions, "actions")?;
        let records = writable_slice(&mut records, "records")?;
        py.detach(|| {
            self.inner
                .step_masked_indices_into(enabled, actions, records)
        })
        .map_err(game_error)
    }

    fn observe_into<'py>(
        &self,
        py: Python<'py>,
        tile_obs: &Bound<'py, PyArray3<u8>>,
        melds: &Bound<'py, PyArray4<u8>>,
        river: &Bound<'py, PyArray3<u8>>,
        meta: &Bound<'py, PyArray2<i32>>,
    ) -> PyResult<()> {
        validate_observation_arrays(self.inner.len(), tile_obs, melds, river, meta)?;

        let mut tile_obs = try_readwrite(tile_obs, "tile_obs")?;
        let mut melds = try_readwrite(melds, "melds")?;
        let mut river = try_readwrite(river, "river")?;
        let mut meta = try_readwrite(meta, "meta")?;
        let tile_obs = writable_slice(&mut tile_obs, "tile_obs")?;
        let melds = writable_slice(&mut melds, "melds")?;
        let river = writable_slice(&mut river, "river")?;
        let meta = writable_slice(&mut meta, "meta")?;
        py.detach(|| self.inner.observations_into(tile_obs, melds, river, meta))
            .map_err(game_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn step_and_observe_history_into<'py>(
        &mut self,
        py: Python<'py>,
        actions: &Bound<'py, PyArray1<u8>>,
        history_seat_masks: &Bound<'py, PyArray1<u8>>,
        records: &Bound<'py, PyArray2<i64>>,
        mask_words: &Bound<'py, PyArray2<u64>>,
        tile_obs: &Bound<'py, PyArray3<u8>>,
        melds: &Bound<'py, PyArray4<u8>>,
        river: &Bound<'py, PyArray3<u8>>,
        meta: &Bound<'py, PyArray2<i32>>,
        events: &Bound<'py, PyArray3<i32>>,
        event_lengths: &Bound<'py, PyArray1<u16>>,
    ) -> PyResult<()> {
        let batch_size = self.inner.len();
        require_shape(actions, &[batch_size], "actions")?;
        require_c_contiguous(actions, "actions")?;
        require_shape(history_seat_masks, &[batch_size], "history_seat_masks")?;
        require_c_contiguous(history_seat_masks, "history_seat_masks")?;
        require_shape(records, &[batch_size, STEP_RECORD_WIDTH], "records")?;
        require_c_contiguous(records, "records")?;
        require_shape(
            mask_words,
            &[batch_size, LEGAL_ACTION_MASK_WORDS],
            "mask_words",
        )?;
        require_c_contiguous(mask_words, "mask_words")?;
        validate_observation_arrays(batch_size, tile_obs, melds, river, meta)?;
        let event_capacity = validate_batch_event_array(batch_size, events, "events")?;
        require_shape(event_lengths, &[batch_size], "event_lengths")?;
        require_c_contiguous(event_lengths, "event_lengths")?;

        let actions = try_readonly(actions, "actions")?;
        let history_seat_masks = try_readonly(history_seat_masks, "history_seat_masks")?;
        let mut records = try_readwrite(records, "records")?;
        let mut mask_words = try_readwrite(mask_words, "mask_words")?;
        let mut tile_obs = try_readwrite(tile_obs, "tile_obs")?;
        let mut melds = try_readwrite(melds, "melds")?;
        let mut river = try_readwrite(river, "river")?;
        let mut meta = try_readwrite(meta, "meta")?;
        let mut events = try_readwrite(events, "events")?;
        let mut event_lengths = try_readwrite(event_lengths, "event_lengths")?;
        let actions = readonly_slice(&actions, "actions")?;
        let history_seat_masks = readonly_slice(&history_seat_masks, "history_seat_masks")?;
        let records = writable_slice(&mut records, "records")?;
        let mask_words = writable_slice(&mut mask_words, "mask_words")?;
        let tile_obs = writable_slice(&mut tile_obs, "tile_obs")?;
        let melds = writable_slice(&mut melds, "melds")?;
        let river = writable_slice(&mut river, "river")?;
        let meta = writable_slice(&mut meta, "meta")?;
        let events = writable_slice(&mut events, "events")?;
        let event_lengths = writable_slice(&mut event_lengths, "event_lengths")?;

        py.detach(|| {
            self.inner.step_indices_observe_history_into(
                actions,
                history_seat_masks,
                records,
                mask_words,
                tile_obs,
                melds,
                river,
                meta,
                event_capacity,
                events,
                event_lengths,
            )
        })
        .map_err(game_error)
    }

    #[allow(clippy::too_many_arguments)]
    fn reset_and_observe_history_into<'py>(
        &mut self,
        py: Python<'py>,
        reset_flags: &Bound<'py, PyArray1<u8>>,
        seeds: &Bound<'py, PyArray1<u64>>,
        history_seat_masks: &Bound<'py, PyArray1<u8>>,
        mask_words: &Bound<'py, PyArray2<u64>>,
        tile_obs: &Bound<'py, PyArray3<u8>>,
        melds: &Bound<'py, PyArray4<u8>>,
        river: &Bound<'py, PyArray3<u8>>,
        meta: &Bound<'py, PyArray2<i32>>,
        events: &Bound<'py, PyArray3<i32>>,
        event_lengths: &Bound<'py, PyArray1<u16>>,
    ) -> PyResult<()> {
        let batch_size = self.inner.len();
        require_shape(reset_flags, &[batch_size], "reset_flags")?;
        require_c_contiguous(reset_flags, "reset_flags")?;
        require_shape(seeds, &[batch_size], "seeds")?;
        require_c_contiguous(seeds, "seeds")?;
        require_shape(history_seat_masks, &[batch_size], "history_seat_masks")?;
        require_c_contiguous(history_seat_masks, "history_seat_masks")?;
        require_shape(
            mask_words,
            &[batch_size, LEGAL_ACTION_MASK_WORDS],
            "mask_words",
        )?;
        require_c_contiguous(mask_words, "mask_words")?;
        validate_observation_arrays(batch_size, tile_obs, melds, river, meta)?;
        let event_capacity = validate_batch_event_array(batch_size, events, "events")?;
        require_shape(event_lengths, &[batch_size], "event_lengths")?;
        require_c_contiguous(event_lengths, "event_lengths")?;

        let reset_flags = try_readonly(reset_flags, "reset_flags")?;
        let seeds = try_readonly(seeds, "seeds")?;
        let history_seat_masks = try_readonly(history_seat_masks, "history_seat_masks")?;
        let mut mask_words = try_readwrite(mask_words, "mask_words")?;
        let mut tile_obs = try_readwrite(tile_obs, "tile_obs")?;
        let mut melds = try_readwrite(melds, "melds")?;
        let mut river = try_readwrite(river, "river")?;
        let mut meta = try_readwrite(meta, "meta")?;
        let mut events = try_readwrite(events, "events")?;
        let mut event_lengths = try_readwrite(event_lengths, "event_lengths")?;
        let reset_flags = readonly_slice(&reset_flags, "reset_flags")?;
        let seeds = readonly_slice(&seeds, "seeds")?;
        let history_seat_masks = readonly_slice(&history_seat_masks, "history_seat_masks")?;
        let mask_words = writable_slice(&mut mask_words, "mask_words")?;
        let tile_obs = writable_slice(&mut tile_obs, "tile_obs")?;
        let melds = writable_slice(&mut melds, "melds")?;
        let river = writable_slice(&mut river, "river")?;
        let meta = writable_slice(&mut meta, "meta")?;
        let events = writable_slice(&mut events, "events")?;
        let event_lengths = writable_slice(&mut event_lengths, "event_lengths")?;

        py.detach(|| {
            self.inner.reset_observe_history_into(
                reset_flags,
                seeds,
                history_seat_masks,
                mask_words,
                tile_obs,
                melds,
                river,
                meta,
                event_capacity,
                events,
                event_lengths,
            )
        })
        .map_err(game_error)
    }
}

fn action_id(action: u8) -> PyResult<ActionId> {
    ActionId::new(action as usize).ok_or_else(|| {
        PyValueError::new_err(format!(
            "action must be in 0..{ACTION_SPACE_SIZE}, got {action}"
        ))
    })
}

fn seat_value(seat: u8) -> PyResult<Seat> {
    Seat::new(seat)
        .ok_or_else(|| PyValueError::new_err(format!("seat must be in 0..4, got {seat}")))
}

fn exchange_direction(direction: u8) -> PyResult<ExchangeDirection> {
    match direction {
        1 => Ok(ExchangeDirection::Left),
        2 => Ok(ExchangeDirection::Across),
        3 => Ok(ExchangeDirection::Right),
        _ => Err(PyValueError::new_err(format!(
            "exchange direction must be 1 (left), 2 (across), or 3 (right), got {direction}"
        ))),
    }
}

fn phase_code(phase: Phase) -> u8 {
    match phase {
        Phase::Exchange => 0,
        Phase::ChooseMissing => 1,
        Phase::Turn => 2,
        Phase::HuResponse => 3,
        Phase::MeldResponse => 4,
        Phase::Finished => 5,
    }
}

fn meld_kind_code(kind: MeldKind) -> u8 {
    match kind {
        MeldKind::Pong => 0,
        MeldKind::ExposedKong => 1,
        MeldKind::AddedKong => 2,
        MeldKind::ConcealedKong => 3,
    }
}

fn game_error(error: GameError) -> PyErr {
    match error {
        GameError::Finished => PyRuntimeError::new_err(error.to_string()),
        GameError::InvalidAction
        | GameError::InvalidExchange
        | GameError::BatchLength
        | GameError::BatchIndex
        | GameError::EventCapacity
        | GameError::InformationSetUnavailable => PyValueError::new_err(error.to_string()),
    }
}

#[cfg(feature = "rule-nn")]
fn rule_nn_model_error(error: core_engine::RuleNnError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[cfg(feature = "rule-nn")]
fn rule_nn_inference_error(error: core_engine::RuleNnError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn require_shape<'py>(
    array: &impl PyUntypedArrayMethods<'py>,
    expected: &[usize],
    name: &str,
) -> PyResult<()> {
    let shape = array.shape();
    if shape != expected {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape {expected:?}, got {shape:?}"
        )));
    }
    Ok(())
}

fn require_c_contiguous<'py>(array: &impl PyUntypedArrayMethods<'py>, name: &str) -> PyResult<()> {
    if !array.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "{name} must be C-contiguous"
        )));
    }
    Ok(())
}

fn copy_tiles_into<'py>(
    source: &[u8; TILE_KIND_COUNT],
    output: &Bound<'py, PyArray1<u8>>,
    name: &str,
) -> PyResult<()> {
    require_shape(output, &[TILE_KIND_COUNT], name)?;
    require_c_contiguous(output, name)?;
    let mut output = try_readwrite(output, name)?;
    writable_slice(&mut output, name)?.copy_from_slice(source);
    Ok(())
}

fn writable_slice<'array, T, D>(
    array: &'array mut PyReadwriteArray<'_, T, D>,
    name: &str,
) -> PyResult<&'array mut [T]>
where
    T: numpy::Element,
    D: numpy::ndarray::Dimension,
{
    if array.len() == 0 {
        return Ok(&mut []);
    }
    require_slice_layout(&***array, name)?;
    array
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err(format!("{name} must be aligned and C-contiguous")))
}

fn readonly_slice<'array, T, D>(
    array: &'array PyReadonlyArray<'_, T, D>,
    name: &str,
) -> PyResult<&'array [T]>
where
    T: numpy::Element,
    D: numpy::ndarray::Dimension,
{
    if array.len() == 0 {
        return Ok(&[]);
    }
    require_slice_layout(&**array, name)?;
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be aligned and C-contiguous")))
}

fn require_slice_layout<'py, T, D>(array: &Bound<'py, PyArray<T, D>>, name: &str) -> PyResult<()>
where
    T: numpy::Element,
    D: numpy::ndarray::Dimension,
{
    let element_size = core::mem::size_of::<T>();
    let address = array.data() as usize;
    if address == 0 || address % core::mem::align_of::<T>() != 0 {
        return Err(PyValueError::new_err(format!(
            "{name} must have an aligned data pointer"
        )));
    }
    if element_size != 0 && array.len() > isize::MAX as usize / element_size {
        return Err(PyValueError::new_err(format!(
            "{name} is too large to expose as a Rust slice"
        )));
    }
    Ok(())
}

fn try_readonly<'py, T, D>(
    array: &Bound<'py, PyArray<T, D>>,
    name: &str,
) -> PyResult<PyReadonlyArray<'py, T, D>>
where
    T: numpy::Element,
    D: numpy::ndarray::Dimension,
{
    array.try_readonly().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} overlaps a writable array used by this call"
        ))
    })
}

fn try_readwrite<'py, T, D>(
    array: &Bound<'py, PyArray<T, D>>,
    name: &str,
) -> PyResult<PyReadwriteArray<'py, T, D>>
where
    T: numpy::Element,
    D: numpy::ndarray::Dimension,
{
    array.try_readwrite().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} overlaps another array used by this call or is not writable"
        ))
    })
}

fn validate_observation_arrays<'py>(
    batch_size: usize,
    tile_obs: &Bound<'py, PyArray3<u8>>,
    melds: &Bound<'py, PyArray4<u8>>,
    river: &Bound<'py, PyArray3<u8>>,
    meta: &Bound<'py, PyArray2<i32>>,
) -> PyResult<()> {
    require_shape(
        tile_obs,
        &[batch_size, TILE_OBSERVATION_PLANES, TILE_KIND_COUNT],
        "tile_obs",
    )?;
    require_c_contiguous(tile_obs, "tile_obs")?;
    require_shape(
        melds,
        &[batch_size, PLAYER_COUNT, MELD_SLOTS, MELD_FIELDS],
        "melds",
    )?;
    require_c_contiguous(melds, "melds")?;
    require_shape(
        river,
        &[batch_size, RIVER_TILE_CAPACITY, RIVER_FIELDS],
        "river",
    )?;
    require_c_contiguous(river, "river")?;
    require_shape(meta, &[batch_size, META_OBSERVATION_WIDTH], "meta")?;
    require_c_contiguous(meta, "meta")?;
    Ok(())
}

fn validate_event_array<'py>(output: &Bound<'py, PyArray2<i32>>, name: &str) -> PyResult<()> {
    let shape = output.shape();
    if shape.len() != 2 || shape[1] != EVENT_RECORD_WIDTH {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape [capacity, {EVENT_RECORD_WIDTH}], got {shape:?}"
        )));
    }
    if shape[0] > EVENT_HISTORY_CAPACITY {
        return Err(PyValueError::new_err(format!(
            "{name} capacity must be at most {EVENT_HISTORY_CAPACITY}, got {}",
            shape[0]
        )));
    }
    Ok(())
}

fn validate_batch_event_array<'py>(
    batch_size: usize,
    output: &Bound<'py, PyArray3<i32>>,
    name: &str,
) -> PyResult<usize> {
    let shape = output.shape();
    if shape.len() != 3 || shape[0] != batch_size || shape[2] != EVENT_RECORD_WIDTH {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape [{batch_size}, capacity, {EVENT_RECORD_WIDTH}], got {shape:?}"
        )));
    }
    if shape[1] > EVENT_HISTORY_CAPACITY {
        return Err(PyValueError::new_err(format!(
            "{name} capacity must be at most {EVENT_HISTORY_CAPACITY}, got {}",
            shape[1]
        )));
    }
    Ok(shape[1])
}

fn config_range_error(name: &str, value: impl std::fmt::Display, range: &str) -> PyErr {
    PyValueError::new_err(format!("{name} must be in {range}, got {value}"))
}

fn build_planner_config(
    hand_changes: u8,
    draw_horizon: u8,
    candidate_states: u32,
    belief_worlds: u16,
    response_worlds: u16,
    search_iterations: u16,
) -> PyResult<RulePlannerConfig> {
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

fn python_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

fn add_event_enums(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    let enum_module = py.import("enum")?;
    let int_enum = enum_module.getattr("IntEnum")?;
    let int_flag = enum_module.getattr("IntFlag")?;

    let kinds = PyDict::new(py);
    for (name, value) in [
        ("ACTION", EventKind::Action.code()),
        ("GAME_START", EventKind::GameStart.code()),
        ("TURN_START", EventKind::TurnStart.code()),
        ("DRAW", EventKind::Draw.code()),
        ("DISCARD", EventKind::Discard.code()),
        ("EXCHANGE_COMPLETE", EventKind::ExchangeComplete.code()),
        ("MISSING_REVEALED", EventKind::MissingRevealed.code()),
        ("MELD", EventKind::Meld.code()),
        ("HU", EventKind::Hu.code()),
        ("PAYMENT", EventKind::Payment.code()),
        ("GAME_END", EventKind::GameEnd.code()),
    ] {
        kinds.set_item(name, value)?;
    }
    let event_kind = int_enum.call1(("EventKind", kinds))?;
    event_kind.setattr("__module__", "bloodflow_mahjong")?;
    module.add("EventKind", event_kind)?;

    let flags = PyDict::new(py);
    for (name, value) in [
        ("REPLACEMENT_DRAW", EVENT_FLAG_REPLACEMENT_DRAW),
        ("LAST_WALL_TILE", EVENT_FLAG_LAST_WALL_TILE),
        ("AFTER_KONG", EVENT_FLAG_AFTER_KONG),
        ("OPENING_DISCARD", EVENT_FLAG_OPENING_DISCARD),
        ("SELF_DRAW", EVENT_FLAG_SELF_DRAW),
        ("ROB_KONG", EVENT_FLAG_ROB_KONG),
        ("HEAVENLY", EVENT_FLAG_HEAVENLY),
        ("EARTHLY", EVENT_FLAG_EARTHLY),
    ] {
        flags.set_item(name, value)?;
    }
    let event_flag = int_flag.call1(("EventFlag", flags))?;
    event_flag.setattr("__module__", "bloodflow_mahjong")?;
    module.add("EventFlag", event_flag)?;
    Ok(())
}

fn encode_outcome(outcome: StepOutcome, record: &mut [i64]) {
    debug_assert_eq!(record.len(), STEP_RECORD_WIDTH);
    record.fill(0);
    record[0] = outcome
        .draw
        .map_or(-1, |draw| i64::from(draw.player.as_u8()));
    record[1] = outcome.draw.map_or(-1, |draw| i64::from(draw.tile.as_u8()));
    record[2] = outcome.draw.map_or(0, |draw| i64::from(draw.replacement));
    record[3] = outcome
        .discard
        .map_or(-1, |discard| i64::from(discard.player.as_u8()));
    record[4] = outcome
        .discard
        .map_or(-1, |discard| i64::from(discard.tile.as_u8()));
    record[5..9].copy_from_slice(&outcome.score_delta);
    record[9] = outcome
        .next
        .map_or(-1, |decision| i64::from(decision.actor.as_u8()));
    record[10] = outcome
        .next
        .map_or(-1, |decision| i64::from(phase_code(decision.phase)));
    record[11] = i64::from(outcome.terminal);
}

fn outcome_tuple(outcome: StepOutcome) -> StepRecordTuple {
    let mut record = [0; STEP_RECORD_WIDTH];
    encode_outcome(outcome, &mut record);
    (
        record[0], record[1], record[2], record[3], record[4], record[5], record[6], record[7],
        record[8], record[9], record[10], record[11],
    )
}

#[pymodule]
fn bloodflow_mahjong(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "rule-nn")]
    module.add_class::<PyRuleNn>()?;
    module.add_class::<PyRuleEvConfig>()?;
    module.add_class::<PyRulePlannerConfig>()?;
    module.add_class::<PyGame>()?;
    module.add_class::<PyBatch>()?;
    add_event_enums(module)?;
    module.add("ACTION_SPACE_SIZE", ACTION_SPACE_SIZE)?;
    module.add("ACTION_EXCHANGE_TILE_OFFSET", ACTION_EXCHANGE_TILE_OFFSET)?;
    module.add("ACTION_CHOOSE_MISSING_OFFSET", ACTION_CHOOSE_MISSING_OFFSET)?;
    module.add("ACTION_DISCARD_OFFSET", ACTION_DISCARD_OFFSET)?;
    module.add("ACTION_HU", ACTION_HU)?;
    module.add("ACTION_PONG", ACTION_PONG)?;
    module.add("ACTION_EXPOSED_KONG", ACTION_EXPOSED_KONG)?;
    module.add("ACTION_CONCEALED_KONG_OFFSET", ACTION_CONCEALED_KONG_OFFSET)?;
    module.add("ACTION_ADDED_KONG_OFFSET", ACTION_ADDED_KONG_OFFSET)?;
    module.add("ACTION_PASS", ACTION_PASS)?;
    module.add("LEGAL_ACTION_MASK_WORDS", LEGAL_ACTION_MASK_WORDS)?;
    module.add("STEP_RECORD_WIDTH", STEP_RECORD_WIDTH)?;
    module.add("EVENT_RECORD_WIDTH", EVENT_RECORD_WIDTH)?;
    module.add("EVENT_HISTORY_CAPACITY", EVENT_HISTORY_CAPACITY)?;
    module.add("ENGINE_RULES_VERSION", ENGINE_RULES_VERSION)?;
    module.add("SHANTEN_COMPLETE", SHANTEN_COMPLETE)?;
    module.add("SHANTEN_MAX", SHANTEN_MAX)?;
    module.add("SHANTEN_TERMINAL", SHANTEN_TERMINAL)?;
    module.add("SIMPLE_RULE_ACTION_TERMINAL", SIMPLE_RULE_ACTION_TERMINAL)?;
    module.add("RULE_EV_ACTION_TERMINAL", RULE_EV_ACTION_TERMINAL)?;
    module.add("RULE_PLANNER_ACTION_TERMINAL", RULE_PLANNER_ACTION_TERMINAL)?;
    module.add("TILE_OBSERVATION_WIDTH", TILE_OBSERVATION_WIDTH)?;
    module.add("TILE_OBSERVATION_PLANES", TILE_OBSERVATION_PLANES)?;
    module.add("MELD_OBSERVATION_WIDTH", MELD_OBSERVATION_WIDTH)?;
    module.add("MELD_SLOTS", MELD_SLOTS)?;
    module.add("MELD_FIELDS", MELD_FIELDS)?;
    module.add("RIVER_OBSERVATION_WIDTH", RIVER_OBSERVATION_WIDTH)?;
    module.add("RIVER_TILE_CAPACITY", RIVER_TILE_CAPACITY)?;
    module.add("RIVER_FIELDS", RIVER_FIELDS)?;
    module.add("META_OBSERVATION_WIDTH", META_OBSERVATION_WIDTH)?;
    module.add("ORACLE_TILE_COUNT_PLANES", ORACLE_TILE_COUNT_PLANES)?;
    module.add("PLAYER_COUNT", PLAYER_COUNT)?;
    module.add("TILE_KIND_COUNT", TILE_KIND_COUNT)?;
    module.add("PHASE_EXCHANGE", phase_code(Phase::Exchange))?;
    module.add("PHASE_CHOOSE_MISSING", phase_code(Phase::ChooseMissing))?;
    module.add("PHASE_TURN", phase_code(Phase::Turn))?;
    module.add("PHASE_HU_RESPONSE", phase_code(Phase::HuResponse))?;
    module.add("PHASE_MELD_RESPONSE", phase_code(Phase::MeldResponse))?;
    module.add("PHASE_FINISHED", phase_code(Phase::Finished))?;
    module.add(
        "TERMINATION_WALL_EXHAUSTED",
        TerminationReason::WallExhausted.code(),
    )?;
    module.add(
        "TERMINATION_THREE_PLAYERS_BANKRUPT",
        TerminationReason::ThreePlayersBankrupt.code(),
    )?;
    Ok(())
}
