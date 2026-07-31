use core::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use crate::ActionId;
use crate::game::{
    Game, LegalActions, MELD_OBSERVATION_WIDTH, META_OBSERVATION_WIDTH, Phase,
    RIVER_OBSERVATION_WIDTH, TILE_OBSERVATION_WIDTH,
};
#[cfg(feature = "planner-analysis")]
use crate::rules::ev::RuleEvConfig;
use crate::rules::{
    hand::{Holding, mask_tiles},
    opening,
};
#[cfg(feature = "planner-analysis")]
use crate::types::PLAYER_COUNT;
use crate::types::{Seat, Tile};

mod belief;
mod graph;
mod history;
mod quality;
mod response;
mod search;
mod simulation;
mod state;
mod value;

use graph::{HandGraphPlanner, PlanningHorizon};
use quality::HandPotential;
use state::{PlannedDraw, PlanningHand as HandState, PlanningPublicState};
use value::PublicValueModel;

const MAX_HAND_CHANGES: u8 = 2;
const MAX_DRAW_HORIZON: u8 = 32;
const MAX_CANDIDATE_STATES: u32 = 200_000;
const MAX_BELIEF_WORLDS: u16 = 256;
const MAX_RESPONSE_WORLDS: u16 = 256;
const MAX_SEARCH_ITERATIONS: u16 = 4_096;
const MAX_ROLLOUT_ACTIONS: usize = 1_024;
const SEARCH_SEED_DOMAIN: u64 = 0x6b82_12f4_c938_d0a7;
const BELIEF_SEED_DOMAIN: u64 = 0xd14f_3a6c_92e7_580b;

static SEARCH_DECISIONS: AtomicU64 = AtomicU64::new(0);
static SEARCH_PROPOSALS: AtomicU64 = AtomicU64::new(0);
static SEARCH_VALIDATION_REJECTIONS: AtomicU64 = AtomicU64::new(0);
static SEARCH_OVERRIDES: AtomicU64 = AtomicU64::new(0);
static SEARCH_ROLLOUTS: AtomicU64 = AtomicU64::new(0);
static PLANNED_TURNS: AtomicU64 = AtomicU64::new(0);
static TURN_OVERRIDES: AtomicU64 = AtomicU64::new(0);
static HAZARD_CANDIDATES: AtomicU64 = AtomicU64::new(0);
static HAZARD_LOSS_MILLIPOINTS: AtomicU64 = AtomicU64::new(0);
static HAZARD_WON_LOSS_MILLIPOINTS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RulePlannerSearchStats {
    pub decisions: u64,
    pub proposals: u64,
    pub validation_rejections: u64,
    pub overrides: u64,
    pub rollouts: u64,
    pub planned_turns: u64,
    pub turn_overrides: u64,
    pub hazard_candidates: u64,
    pub hazard_loss_millipoints: u64,
    pub hazard_won_loss_millipoints: u64,
}

/// Root-particle distribution used by planner analysis builds.
///
/// `OracleHidden` reads the authoritative hidden allocation and is never a
/// deployable policy. It independently resamples the future wall order.
#[cfg(feature = "planner-analysis")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RulePlannerRootBelief {
    #[default]
    Posterior,
    Uniform,
    OracleHidden,
}

/// One observation-only policy used by a known continuation profile.
///
/// `PlannerBaseline` deliberately disables paired root search. It retains the
/// remaining planner configuration and represents the fixed policy improved
/// by one root-search decision.
#[cfg(feature = "planner-analysis")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RulePlannerContinuationPolicy {
    Fast,
    Ev(RuleEvConfig),
    PlannerBaseline(RulePlannerConfig),
}

/// Frozen continuation policy assigned to each seat in an analysis rollout.
///
/// Each policy receives only the current actor's ordinary policy inputs. The
/// profile reveals strategy identity to the diagnostic search, but it does not
/// expose another seat's concealed hand to the acting policy.
#[cfg(feature = "planner-analysis")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RulePlannerContinuationProfile {
    policies: [RulePlannerContinuationPolicy; PLAYER_COUNT],
}

#[cfg(feature = "planner-analysis")]
impl RulePlannerContinuationProfile {
    pub const fn new(policies: [RulePlannerContinuationPolicy; PLAYER_COUNT]) -> Self {
        Self { policies }
    }

    pub const fn for_seat(self, seat: Seat) -> RulePlannerContinuationPolicy {
        self.policies[seat.index()]
    }
}

/// Continuation model used by planner analysis.
///
/// `Current` preserves the production proxy ensemble. `KnownPolicies` is an
/// oracle diagnostic because deployed play cannot assume the opponents'
/// strategy identities.
#[cfg(feature = "planner-analysis")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RulePlannerContinuation {
    #[default]
    Current,
    KnownPolicies(RulePlannerContinuationProfile),
}

/// Orthogonal root-belief and continuation settings for planner analysis.
#[cfg(feature = "planner-analysis")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RulePlannerAnalysisOptions {
    root_belief: RulePlannerRootBelief,
    continuation: RulePlannerContinuation,
}

#[cfg(feature = "planner-analysis")]
impl RulePlannerAnalysisOptions {
    pub const fn new(root_belief: RulePlannerRootBelief) -> Self {
        Self {
            root_belief,
            continuation: RulePlannerContinuation::Current,
        }
    }

    pub const fn with_continuation(mut self, continuation: RulePlannerContinuation) -> Self {
        self.continuation = continuation;
        self
    }

    pub const fn root_belief(self) -> RulePlannerRootBelief {
        self.root_belief
    }

    pub const fn continuation(self) -> RulePlannerContinuation {
        self.continuation
    }
}

/// One planner decision and its optional root-search diagnostic.
///
/// This type is exported only by the `planner-analysis` feature.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RulePlannerAnalysis {
    action: ActionId,
    search: Option<RulePlannerSearchAnalysis>,
}

impl RulePlannerAnalysis {
    const fn without_search(action: ActionId) -> Self {
        Self {
            action,
            search: None,
        }
    }

    const fn from_search(search: RulePlannerSearchAnalysis) -> Self {
        Self {
            action: search.action(),
            search: Some(search),
        }
    }

    pub const fn action(self) -> ActionId {
        self.action
    }

    #[cfg(any(feature = "planner-analysis", test))]
    pub const fn search(self) -> Option<RulePlannerSearchAnalysis> {
        self.search
    }
}

/// Diagnostic for one root-search decision.
///
/// `outcome` determines the final action. An accepted proposal becomes the
/// action; every other outcome retains `baseline`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RulePlannerSearchAnalysis {
    baseline: ActionId,
    outcome: RulePlannerSearchOutcome,
    rollouts: u64,
}

impl RulePlannerSearchAnalysis {
    const fn new(baseline: ActionId, outcome: RulePlannerSearchOutcome, rollouts: u64) -> Self {
        Self {
            baseline,
            outcome,
            rollouts,
        }
    }

    const fn action(self) -> ActionId {
        match self.outcome {
            RulePlannerSearchOutcome::Accepted(proposal) => proposal,
            RulePlannerSearchOutcome::NoProposal | RulePlannerSearchOutcome::Rejected(_) => {
                self.baseline
            }
        }
    }

    #[cfg(any(feature = "planner-analysis", test))]
    pub const fn baseline(self) -> ActionId {
        self.baseline
    }

    #[cfg(any(feature = "planner-analysis", test))]
    pub const fn outcome(self) -> RulePlannerSearchOutcome {
        self.outcome
    }

    #[cfg(any(feature = "planner-analysis", test))]
    pub const fn rollouts(self) -> u64 {
        self.rollouts
    }
}

/// Result of the proposal and validation gates for one root search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RulePlannerSearchOutcome {
    NoProposal,
    Rejected(ActionId),
    Accepted(ActionId),
}

impl RulePlannerSearchOutcome {
    #[cfg(any(feature = "planner-analysis", test))]
    pub const fn proposal(self) -> Option<ActionId> {
        match self {
            Self::NoProposal => None,
            Self::Rejected(proposal) | Self::Accepted(proposal) => Some(proposal),
        }
    }

    #[cfg(any(feature = "planner-analysis", test))]
    pub const fn accepted(self) -> bool {
        matches!(self, Self::Accepted(_))
    }
}

pub fn rule_planner_search_stats() -> RulePlannerSearchStats {
    RulePlannerSearchStats {
        decisions: SEARCH_DECISIONS.load(AtomicOrdering::Relaxed),
        proposals: SEARCH_PROPOSALS.load(AtomicOrdering::Relaxed),
        validation_rejections: SEARCH_VALIDATION_REJECTIONS.load(AtomicOrdering::Relaxed),
        overrides: SEARCH_OVERRIDES.load(AtomicOrdering::Relaxed),
        rollouts: SEARCH_ROLLOUTS.load(AtomicOrdering::Relaxed),
        planned_turns: PLANNED_TURNS.load(AtomicOrdering::Relaxed),
        turn_overrides: TURN_OVERRIDES.load(AtomicOrdering::Relaxed),
        hazard_candidates: HAZARD_CANDIDATES.load(AtomicOrdering::Relaxed),
        hazard_loss_millipoints: HAZARD_LOSS_MILLIPOINTS.load(AtomicOrdering::Relaxed),
        hazard_won_loss_millipoints: HAZARD_WON_LOSS_MILLIPOINTS.load(AtomicOrdering::Relaxed),
    }
}

pub fn reset_rule_planner_search_stats() {
    SEARCH_DECISIONS.store(0, AtomicOrdering::Relaxed);
    SEARCH_PROPOSALS.store(0, AtomicOrdering::Relaxed);
    SEARCH_VALIDATION_REJECTIONS.store(0, AtomicOrdering::Relaxed);
    SEARCH_OVERRIDES.store(0, AtomicOrdering::Relaxed);
    SEARCH_ROLLOUTS.store(0, AtomicOrdering::Relaxed);
    PLANNED_TURNS.store(0, AtomicOrdering::Relaxed);
    TURN_OVERRIDES.store(0, AtomicOrdering::Relaxed);
    HAZARD_CANDIDATES.store(0, AtomicOrdering::Relaxed);
    HAZARD_LOSS_MILLIPOINTS.store(0, AtomicOrdering::Relaxed);
    HAZARD_WON_LOSS_MILLIPOINTS.store(0, AtomicOrdering::Relaxed);
}

/// Compute budget for the belief-aware planning policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RulePlannerConfig {
    hand_changes: u8,
    draw_horizon: u8,
    candidate_states: u32,
    belief_worlds: u16,
    response_worlds: u16,
    search_iterations: u16,
}

impl RulePlannerConfig {
    pub const FAST: Self = Self {
        hand_changes: 0,
        draw_horizon: 4,
        candidate_states: 256,
        belief_worlds: 0,
        response_worlds: 0,
        search_iterations: 0,
    };
    pub const STANDARD: Self = Self {
        hand_changes: 1,
        draw_horizon: 27,
        candidate_states: 4_096,
        belief_worlds: 0,
        response_worlds: 0,
        search_iterations: 0,
    };
    pub const DEEP: Self = Self {
        hand_changes: 2,
        draw_horizon: 32,
        candidate_states: 20_000,
        belief_worlds: 0,
        response_worlds: 0,
        search_iterations: 256,
    };
    const ROLLOUT: Self = Self {
        hand_changes: 0,
        draw_horizon: 0,
        candidate_states: 1,
        belief_worlds: 0,
        response_worlds: 0,
        search_iterations: 0,
    };

    pub const fn with_hand_changes(mut self, hand_changes: u8) -> Option<Self> {
        if hand_changes <= MAX_HAND_CHANGES {
            self.hand_changes = hand_changes;
            Some(self)
        } else {
            None
        }
    }

    pub const fn hand_changes(self) -> u8 {
        self.hand_changes
    }

    pub const fn with_draw_horizon(mut self, draw_horizon: u8) -> Option<Self> {
        if draw_horizon <= MAX_DRAW_HORIZON {
            self.draw_horizon = draw_horizon;
            Some(self)
        } else {
            None
        }
    }

    pub const fn draw_horizon(self) -> u8 {
        self.draw_horizon
    }

    pub const fn with_candidate_states(mut self, candidate_states: u32) -> Option<Self> {
        if candidate_states > 0 && candidate_states <= MAX_CANDIDATE_STATES {
            self.candidate_states = candidate_states;
            Some(self)
        } else {
            None
        }
    }

    pub const fn candidate_states(self) -> u32 {
        self.candidate_states
    }

    pub const fn with_belief_worlds(mut self, belief_worlds: u16) -> Option<Self> {
        if belief_worlds <= MAX_BELIEF_WORLDS {
            self.belief_worlds = belief_worlds;
            Some(self)
        } else {
            None
        }
    }

    pub const fn belief_worlds(self) -> u16 {
        self.belief_worlds
    }

    pub const fn with_response_worlds(mut self, response_worlds: u16) -> Option<Self> {
        if response_worlds <= MAX_RESPONSE_WORLDS {
            self.response_worlds = response_worlds;
            Some(self)
        } else {
            None
        }
    }

    pub const fn response_worlds(self) -> u16 {
        self.response_worlds
    }

    pub const fn with_search_iterations(mut self, search_iterations: u16) -> Option<Self> {
        if search_iterations <= MAX_SEARCH_ITERATIONS {
            self.search_iterations = search_iterations;
            Some(self)
        } else {
            None
        }
    }

    pub const fn search_iterations(self) -> u16 {
        self.search_iterations
    }

    const fn without_search(mut self) -> Self {
        self.search_iterations = 0;
        self
    }
}

impl Default for RulePlannerConfig {
    fn default() -> Self {
        Self::STANDARD
    }
}

impl Game {
    pub fn rule_planner_action(&self) -> Option<ActionId> {
        self.rule_planner_action_with_config(RulePlannerConfig::STANDARD)
    }

    pub fn rule_planner_action_with_config(&self, config: RulePlannerConfig) -> Option<ActionId> {
        let legal = self.legal_actions()?;
        if config.search_iterations != 0 && legal.decision.phase == Phase::Turn && !legal.can_hu {
            return search::paired_policy_improvement(self, &legal, config);
        }
        Some(planner_action_without_search(
            self,
            &legal,
            config.without_search(),
        ))
    }

    /// Runs a planner root-belief ablation.
    ///
    /// This method is diagnostic. `OracleHidden` reads authoritative hidden
    /// state and must not be used by a deployed policy.
    #[cfg(feature = "planner-analysis")]
    pub fn rule_planner_analysis_action_with_config(
        &self,
        config: RulePlannerConfig,
        root_belief: RulePlannerRootBelief,
    ) -> Option<ActionId> {
        self.rule_planner_analysis_with_config(config, root_belief)
            .map(RulePlannerAnalysis::action)
    }

    /// Runs a planner analysis with explicit root-belief and continuation
    /// settings and returns only the selected action.
    #[cfg(feature = "planner-analysis")]
    pub fn rule_planner_analysis_action_with_options(
        &self,
        config: RulePlannerConfig,
        options: RulePlannerAnalysisOptions,
    ) -> Option<ActionId> {
        self.rule_planner_analysis_with_options(config, options)
            .map(RulePlannerAnalysis::action)
    }

    /// Runs a planner root-belief ablation and returns its search diagnostic.
    ///
    /// This method is diagnostic. `OracleHidden` reads authoritative hidden
    /// state and must not be used by a deployed policy.
    #[cfg(feature = "planner-analysis")]
    pub fn rule_planner_analysis_with_config(
        &self,
        config: RulePlannerConfig,
        root_belief: RulePlannerRootBelief,
    ) -> Option<RulePlannerAnalysis> {
        self.rule_planner_analysis_with_options(
            config,
            RulePlannerAnalysisOptions::new(root_belief),
        )
    }

    /// Runs a planner analysis with explicit root-belief and continuation
    /// settings and returns its search diagnostic.
    #[cfg(feature = "planner-analysis")]
    pub fn rule_planner_analysis_with_options(
        &self,
        config: RulePlannerConfig,
        options: RulePlannerAnalysisOptions,
    ) -> Option<RulePlannerAnalysis> {
        let legal = self.legal_actions()?;
        if config.search_iterations != 0 && legal.decision.phase == Phase::Turn && !legal.can_hu {
            return search::paired_policy_improvement_analysis(
                self,
                &legal,
                config,
                options.root_belief.into(),
                options.continuation.into(),
            );
        }
        Some(RulePlannerAnalysis::without_search(
            planner_action_without_search(self, &legal, config.without_search()),
        ))
    }
}

#[cfg(feature = "planner-analysis")]
impl From<RulePlannerRootBelief> for belief::RootBeliefMode {
    fn from(value: RulePlannerRootBelief) -> Self {
        match value {
            RulePlannerRootBelief::Posterior => Self::Posterior,
            RulePlannerRootBelief::Uniform => Self::Uniform,
            RulePlannerRootBelief::OracleHidden => Self::OracleHidden,
        }
    }
}

fn planner_action_without_search(
    game: &Game,
    legal: &LegalActions,
    config: RulePlannerConfig,
) -> ActionId {
    planner_action_without_search_with_observation(game, legal, config, true)
}

/// Evaluates a frozen continuation without recording simulated policy work.
#[cfg(feature = "planner-analysis")]
fn planner_action_without_search_unobserved(
    game: &Game,
    legal: &LegalActions,
    config: RulePlannerConfig,
) -> ActionId {
    planner_action_without_search_with_observation(game, legal, config, false)
}

fn planner_action_without_search_with_observation(
    game: &Game,
    legal: &LegalActions,
    config: RulePlannerConfig,
    observe: bool,
) -> ActionId {
    match legal.decision.phase {
        Phase::Turn => choose_turn(game, legal, config, observe),
        Phase::HuResponse | Phase::MeldResponse => response::choose(game, legal, config),
        Phase::Exchange => opening::choose_exchange(
            game.concealed(legal.decision.actor),
            game.exchange_selection(legal.decision.actor),
            legal.exchange_mask,
        ),
        Phase::ChooseMissing => opening::choose_missing(game.concealed(legal.decision.actor)),
        Phase::Finished => unreachable!("a legal-action set is non-terminal"),
    }
}

fn choose_turn(
    game: &Game,
    legal: &LegalActions,
    config: RulePlannerConfig,
    observe: bool,
) -> ActionId {
    let actor = legal.decision.actor;
    if observe {
        PLANNED_TURNS.fetch_add(1, AtomicOrdering::Relaxed);
    }
    if legal.can_hu {
        return ActionId::HU;
    }
    let baseline = game.simple_rule_action();

    let best = plan_turn(game, legal, config, |hazards| {
        if observe {
            record_turn_hazards(game, actor, legal, hazards);
        }
    });
    if observe && baseline.is_some_and(|action| action != best.action) {
        TURN_OVERRIDES.fetch_add(1, AtomicOrdering::Relaxed);
    }
    best.action
}

fn plan_turn(
    game: &Game,
    legal: &LegalActions,
    config: RulePlannerConfig,
    observe_hazards: impl FnOnce(Option<&belief::HazardTable>),
) -> ValuedAction {
    debug_assert_eq!(legal.decision.phase, Phase::Turn);
    let actor = legal.decision.actor;
    let holding = Holding::from_game(game, actor);

    let normal_horizon = planning_horizon(game, config.draw_horizon, PlannedDraw::Normal);
    let supplement_horizon = planning_horizon(game, config.draw_horizon, PlannedDraw::Supplement);
    let estimate_hazards = if config.draw_horizon <= 1 {
        belief::estimate_immediate
    } else {
        belief::estimate
    };
    let hazards = estimate_hazards(
        game,
        actor,
        usize::from(config.belief_worlds),
        public_state_hash(game, actor) ^ BELIEF_SEED_DOMAIN,
    );
    observe_hazards(hazards.as_ref());
    let public = PlanningPublicState::from_game(game);
    let state = HandState::new(holding, game.has_won(actor), game.max_win_multiplier(actor));
    let value_model = PublicValueModel::new(game, actor, hazards.as_ref());
    let evaluator = HandGraphPlanner::new(
        game,
        actor,
        state,
        hazards.as_ref(),
        &value_model,
        config.candidate_states as usize,
    );
    let Some(discard) = evaluator.best_discard(
        state,
        legal.discard_mask,
        public,
        normal_horizon,
        config.hand_changes,
    ) else {
        let action = legal
            .discard_mask
            .trailing_zeros()
            .try_into()
            .ok()
            .and_then(Tile::new)
            .map(ActionId::discard)
            .expect("a turn without a special action has a legal discard");
        return ValuedAction {
            action,
            value: f64::NEG_INFINITY,
        };
    };
    let has_won = game.has_won(actor);
    let visible = game.visible_tile_counts(actor);
    let baseline_holding = holding
        .after_discard(discard.tile)
        .expect("the selected discard is legal for the current holding");
    let baseline_potential = HandPotential::evaluate(&baseline_holding, &visible, has_won);
    let mut best = ValuedAction {
        action: ActionId::discard(discard.tile),
        value: discard.value,
    };

    for tile in mask_tiles(legal.concealed_kong_mask) {
        let Some(after) = holding.after_concealed_kong(tile, actor) else {
            continue;
        };
        if !HandPotential::evaluate(&after, &visible, has_won)
            .permits_kong_from(baseline_potential, has_won)
        {
            continue;
        }
        let (kong_public, immediate) = value_model.concealed_kong_transition(public);
        let value = immediate
            + evaluator.value_before_draw(
                HandState::new(after, game.has_won(actor), game.max_win_multiplier(actor)),
                kong_public,
                supplement_horizon,
                config.hand_changes,
            );
        best.consider(ActionId::concealed_kong(tile), value);
    }

    for tile in mask_tiles(legal.added_kong_mask) {
        let Some(success) = holding.after_added_kong(tile) else {
            continue;
        };
        if !HandPotential::evaluate(&success, &visible, has_won)
            .permits_kong_from(baseline_potential, has_won)
        {
            continue;
        }
        let Some(robbed) = holding.after_robbed_added_kong(tile) else {
            continue;
        };
        let hazard = hazards.as_ref().and_then(|table| table.added_kong(tile));
        let value = value_model.hazard_transition_value(
            public,
            hazard,
            |after_hazard, is_robbed, next_actor| {
                if is_robbed {
                    evaluator.value_after_external_win(
                        HandState::new(robbed, game.has_won(actor), game.max_win_multiplier(actor)),
                        after_hazard,
                        next_actor,
                        normal_horizon,
                        config.hand_changes,
                    )
                } else {
                    let (kong_public, immediate) = value_model.added_kong_transition(after_hazard);
                    immediate
                        + evaluator.value_before_draw(
                            HandState::new(
                                success,
                                game.has_won(actor),
                                game.max_win_multiplier(actor),
                            ),
                            kong_public,
                            supplement_horizon,
                            config.hand_changes,
                        )
                }
            },
        );
        best.consider(ActionId::added_kong(tile), value);
    }

    best
}

fn record_turn_hazards(
    game: &Game,
    actor: Seat,
    legal: &LegalActions,
    hazards: Option<&belief::HazardTable>,
) {
    for tile in mask_tiles(legal.discard_mask) {
        HAZARD_CANDIDATES.fetch_add(1, AtomicOrdering::Relaxed);
        HAZARD_LOSS_MILLIPOINTS.fetch_add(
            (hazards
                .map(|table| table.immediate_discard(tile))
                .map_or(0.0, |hazard| {
                    hazard.expected_loss_points(actor, game.score(actor))
                })
                * 1_000.0)
                .round()
                .max(0.0) as u64,
            AtomicOrdering::Relaxed,
        );
        let won_loss = hazards
            .map(|table| table.immediate_discard(tile))
            .map_or(0.0, |hazard| {
                Seat::ALL
                    .into_iter()
                    .filter(|&opponent| opponent != actor && game.has_won(opponent))
                    .map(|opponent| {
                        hazard.opponent_expected_loss_points(actor, opponent, game.score(actor))
                    })
                    .sum()
            });
        HAZARD_WON_LOSS_MILLIPOINTS.fetch_add(
            (won_loss * 1_000.0).round().max(0.0) as u64,
            AtomicOrdering::Relaxed,
        );
    }
}

fn planning_horizon(game: &Game, draw_limit: u8, first_draw: PlannedDraw) -> PlanningHorizon {
    PlanningHorizon::new(game.wall_remaining(), draw_limit, first_draw)
}

#[derive(Clone, Copy, Debug)]
struct ValuedAction {
    action: ActionId,
    value: f64,
}

impl ValuedAction {
    fn consider(&mut self, action: ActionId, value: f64) {
        if value > self.value || (value == self.value && action < self.action) {
            self.action = action;
            self.value = value;
        }
    }
}

fn public_state_hash(game: &Game, actor: Seat) -> u64 {
    let mut tile_obs = [0; TILE_OBSERVATION_WIDTH];
    let mut melds = [0; MELD_OBSERVATION_WIDTH];
    let mut river = [0; RIVER_OBSERVATION_WIDTH];
    let mut meta = [0; META_OBSERVATION_WIDTH];
    game.observation_into(actor, &mut tile_obs, &mut melds, &mut river, &mut meta)
        .expect("fixed observation buffers have the engine widths");

    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut write = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for byte in tile_obs.into_iter().chain(melds).chain(river) {
        write(byte);
    }
    for value in meta {
        for byte in value.to_le_bytes() {
            write(byte);
        }
    }
    if let Some(mask) = game.legal_action_mask() {
        for word in mask.words() {
            for byte in word.to_le_bytes() {
                write(byte);
            }
        }
    }
    hash
}

fn mix_search_seed(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_out_of_range_budgets() {
        assert!(
            RulePlannerConfig::STANDARD
                .with_hand_changes(MAX_HAND_CHANGES + 1)
                .is_none()
        );
        assert!(
            RulePlannerConfig::STANDARD
                .with_draw_horizon(MAX_DRAW_HORIZON + 1)
                .is_none()
        );
        assert!(
            RulePlannerConfig::STANDARD
                .with_candidate_states(MAX_CANDIDATE_STATES + 1)
                .is_none()
        );
        assert!(
            RulePlannerConfig::STANDARD
                .with_belief_worlds(MAX_BELIEF_WORLDS + 1)
                .is_none()
        );
        assert!(
            RulePlannerConfig::STANDARD
                .with_response_worlds(MAX_RESPONSE_WORLDS + 1)
                .is_none()
        );
        assert!(
            RulePlannerConfig::STANDARD
                .with_search_iterations(MAX_SEARCH_ITERATIONS + 1)
                .is_none()
        );
    }

    #[test]
    fn planner_completes_games_with_legal_actions() {
        let config = RulePlannerConfig::ROLLOUT
            .with_draw_horizon(0)
            .expect("zero-horizon planning is supported")
            .with_candidate_states(1)
            .expect("one root candidate is supported");
        for seed in 0..2 {
            let mut game = Game::new(seed);
            for _ in 0..MAX_ROLLOUT_ACTIONS {
                let Some(action) = game.rule_planner_action_with_config(config) else {
                    break;
                };
                assert!(game.step_id(action).is_ok());
            }
            assert_eq!(game.phase(), Phase::Finished);
        }
    }
}
