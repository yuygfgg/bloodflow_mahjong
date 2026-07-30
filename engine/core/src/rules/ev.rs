use core::cmp::Ordering;
use core::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use rayon::prelude::*;

use crate::game::{Batch, Game, GameError, LegalActions, PARALLEL_BATCH_THRESHOLD, Phase};
use crate::types::{PLAYER_COUNT, Seat, Suit, TILE_KIND_COUNT, Tile};
use crate::{ACTION_SPACE_SIZE, Action, ActionId, WinFlags, analyze_shanten, evaluate_win};

use super::hand::{DUMMY_MELD, Holding, all_tiles, hand_structure_score, mask_tiles};
use super::opening;

/// Sentinel written for terminal batch slots by [`Batch::rule_ev_actions_into`].
pub const RULE_EV_ACTION_TERMINAL: u8 = u8::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuleEvDefense {
    None,
    Heuristic,
}

/// Statistical gate used to accept a Monte Carlo root-action override.
///
/// Production configurations use [`Self::WorldClustered`]. Scenario-level
/// gates are available to offline analysis builds for controlled ablations.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RuleEvSearchGate {
    #[cfg(feature = "rule-ev-analysis")]
    ScenarioStrict,
    #[cfg(feature = "rule-ev-analysis")]
    ScenarioRelaxed,
    #[default]
    WorldClustered,
}

#[cfg(feature = "rule-ev-analysis")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuleEvSearchCandidateSet {
    Online,
    AllDiscards,
}

#[cfg(feature = "rule-ev-analysis")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleEvSearchTraceConfig {
    pub worlds: u16,
    pub seed_domain: u64,
    pub candidate_set: RuleEvSearchCandidateSet,
}

#[cfg(feature = "rule-ev-analysis")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleEvSearchAction {
    pub action: ActionId,
    pub shanten: Option<i8>,
    pub attack_value: Option<u64>,
    pub online_candidate: bool,
}

#[cfg(feature = "rule-ev-analysis")]
#[derive(Clone, Debug)]
pub struct RuleEvSearchTrace {
    pub baseline: ActionId,
    pub actions: Vec<RuleEvSearchAction>,
    pub world_count: usize,
    pub continuation_count: usize,
    utilities: Vec<i64>,
}

#[cfg(feature = "rule-ev-analysis")]
impl RuleEvSearchTrace {
    pub fn utility(&self, world: usize, continuation: usize, action: usize) -> Option<i64> {
        if world >= self.world_count
            || continuation >= self.continuation_count
            || action >= self.actions.len()
        {
            return None;
        }
        let scenario = world * self.continuation_count + continuation;
        Some(self.utilities[scenario * self.actions.len() + action])
    }

    /// Applies the runtime selector to this trace with an explicit gate.
    pub fn select_action(&self, gate: RuleEvSearchGate) -> ActionId {
        let actions: Vec<_> = self.actions.iter().map(|action| action.action).collect();
        let baseline = actions
            .iter()
            .position(|&action| action == self.baseline)
            .expect("a search trace contains its baseline action");
        select_search_action(
            &self.utilities,
            &actions,
            baseline,
            self.world_count,
            self.continuation_count,
            gate,
        )
    }
}

#[cfg(feature = "rule-ev-analysis")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum RuleEvSearchTraceError {
    #[error("search trace requires a non-terminal turn decision without Hu")]
    UnsupportedDecision,
    #[error("search trace requires at least one information-set world")]
    NoWorlds,
    #[error("search trace requires at least one non-recursive continuation policy")]
    InvalidContinuations,
}

static SEARCH_DECISIONS: AtomicU64 = AtomicU64::new(0);
static SEARCH_OVERRIDES: AtomicU64 = AtomicU64::new(0);
static SEARCH_ROLLOUTS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuleEvSearchStats {
    pub decisions: u64,
    pub overrides: u64,
    pub rollouts: u64,
}

pub fn rule_ev_search_stats() -> RuleEvSearchStats {
    RuleEvSearchStats {
        decisions: SEARCH_DECISIONS.load(AtomicOrdering::Relaxed),
        overrides: SEARCH_OVERRIDES.load(AtomicOrdering::Relaxed),
        rollouts: SEARCH_ROLLOUTS.load(AtomicOrdering::Relaxed),
    }
}

pub fn reset_rule_ev_search_stats() {
    SEARCH_DECISIONS.store(0, AtomicOrdering::Relaxed);
    SEARCH_OVERRIDES.store(0, AtomicOrdering::Relaxed);
    SEARCH_ROLLOUTS.store(0, AtomicOrdering::Relaxed);
}

const WAIT_MULTIPLIER_CAP: u32 = 256;
const PONG_UKEIRE_MARGIN: u16 = 2;
const RISK_SCALE: u64 = 1_000_000;
const DEFENSE_RISK_WEIGHT: u64 = 15;
const MAX_SEARCH_DEPTH: u8 = 3;
const MAX_SEARCH_WORLDS: u16 = 256;
const DEEP_SEARCH_DISCOUNT: u64 = 1;

/// Compute budget for deterministic rule-EV lookahead.
///
/// Depth zero uses static hand evaluation. Each additional level enumerates
/// every live improving tile and the best discard which follows that draw.
/// The search never samples or reads the authoritative wall.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleEvConfig {
    search_depth: u8,
    search_worlds: u16,
    defense: RuleEvDefense,
    search_gate: RuleEvSearchGate,
}

impl RuleEvConfig {
    pub const FAST: Self = Self {
        search_depth: 0,
        search_worlds: 0,
        defense: RuleEvDefense::Heuristic,
        search_gate: RuleEvSearchGate::WorldClustered,
    };
    pub const STANDARD: Self = Self {
        search_depth: 1,
        search_worlds: 0,
        defense: RuleEvDefense::Heuristic,
        search_gate: RuleEvSearchGate::WorldClustered,
    };
    pub const DEEP: Self = Self {
        search_depth: 1,
        search_worlds: 32,
        defense: RuleEvDefense::Heuristic,
        search_gate: RuleEvSearchGate::WorldClustered,
    };

    pub const fn with_search_depth(search_depth: u8) -> Option<Self> {
        if search_depth <= MAX_SEARCH_DEPTH {
            Some(Self {
                search_depth,
                search_worlds: 0,
                defense: RuleEvDefense::Heuristic,
                search_gate: RuleEvSearchGate::WorldClustered,
            })
        } else {
            None
        }
    }

    pub const fn search_depth(self) -> u8 {
        self.search_depth
    }

    pub const fn with_search_worlds(mut self, search_worlds: u16) -> Option<Self> {
        if search_worlds <= MAX_SEARCH_WORLDS {
            self.search_worlds = search_worlds;
            Some(self)
        } else {
            None
        }
    }

    pub const fn search_worlds(self) -> u16 {
        self.search_worlds
    }

    pub const fn with_defense(mut self, defense: RuleEvDefense) -> Self {
        self.defense = defense;
        self
    }

    pub const fn defense(self) -> RuleEvDefense {
        self.defense
    }

    /// Selects a search gate for offline ablation.
    #[cfg(feature = "rule-ev-analysis")]
    pub const fn with_search_gate(mut self, search_gate: RuleEvSearchGate) -> Self {
        self.search_gate = search_gate;
        self
    }

    #[cfg(feature = "rule-ev-analysis")]
    pub const fn search_gate(self) -> RuleEvSearchGate {
        self.search_gate
    }
}

impl Default for RuleEvConfig {
    fn default() -> Self {
        Self::STANDARD
    }
}

impl Game {
    /// Chooses a deterministic action using only the actor's private state and
    /// information which is public to every player.
    pub fn rule_ev_action(&self) -> Option<ActionId> {
        self.rule_ev_action_with_config(RuleEvConfig::STANDARD)
    }

    /// Chooses a rule-EV action with an explicit deterministic search budget.
    pub fn rule_ev_action_with_config(&self, config: RuleEvConfig) -> Option<ActionId> {
        let legal = self.legal_actions()?;
        if config.search_worlds != 0
            && legal.decision.phase == Phase::Turn
            && !legal.can_hu
            && let Some(action) = information_set_search(self, &legal, config)
        {
            return Some(action);
        }
        let actor = legal.decision.actor;
        let action = match legal.decision.phase {
            Phase::Exchange => opening::choose_exchange(
                self.concealed(actor),
                self.exchange_selection(actor),
                legal.exchange_mask,
            ),
            Phase::ChooseMissing => opening::choose_missing(self.concealed(actor)),
            Phase::Turn => choose_turn(self, actor, &legal, config),
            Phase::HuResponse => ActionId::HU,
            Phase::MeldResponse => choose_meld_response(self, actor, &legal, config),
            Phase::Finished => return None,
        };
        debug_assert!(
            self.legal_action_mask()
                .is_some_and(|mask| mask.contains(action))
        );
        Some(action)
    }

    #[cfg(feature = "rule-ev-analysis")]
    pub fn rule_ev_search_trace(
        &self,
        baseline_config: RuleEvConfig,
        continuations: &[RuleEvConfig],
        trace_config: RuleEvSearchTraceConfig,
    ) -> Result<RuleEvSearchTrace, RuleEvSearchTraceError> {
        if trace_config.worlds == 0 {
            return Err(RuleEvSearchTraceError::NoWorlds);
        }
        if continuations.is_empty() || continuations.iter().any(|config| config.search_worlds != 0)
        {
            return Err(RuleEvSearchTraceError::InvalidContinuations);
        }
        let legal = self
            .legal_actions()
            .filter(|legal| legal.decision.phase == Phase::Turn && !legal.can_hu)
            .ok_or(RuleEvSearchTraceError::UnsupportedDecision)?;
        let baseline_config = RuleEvConfig {
            search_depth: baseline_config.search_depth,
            search_worlds: 0,
            defense: baseline_config.defense,
            search_gate: baseline_config.search_gate,
        };
        let baseline = heuristic_action(self, &legal, baseline_config);
        let online_actions = plausible_search_actions(self, &legal, baseline);
        let actions = match trace_config.candidate_set {
            RuleEvSearchCandidateSet::Online => online_actions.clone(),
            RuleEvSearchCandidateSet::AllDiscards => all_discard_search_actions(self, baseline),
        };
        let metadata = search_action_metadata(self, &legal, &actions, &online_actions);
        let worlds = sample_search_worlds(
            self,
            legal.decision.actor,
            usize::from(trace_config.worlds),
            trace_config.seed_domain,
        );
        let rollout_policies: Vec<_> = continuations
            .iter()
            .copied()
            .map(RolloutPolicy::homogeneous)
            .collect();
        let utilities =
            score_search_actions(&worlds, legal.decision.actor, &actions, &rollout_policies);
        Ok(RuleEvSearchTrace {
            baseline,
            actions: metadata,
            world_count: worlds.len(),
            continuation_count: continuations.len(),
            utilities,
        })
    }
}

impl Batch {
    /// Writes one rule-EV action per environment.
    pub fn rule_ev_actions_into(&self, output: &mut [u8]) -> Result<(), GameError> {
        if output.len() != self.len() {
            return Err(GameError::BatchLength);
        }
        let write = |game: &Game, action: &mut u8| {
            *action = game
                .rule_ev_action()
                .map_or(RULE_EV_ACTION_TERMINAL, |id| id.index() as u8);
        };
        if self.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games()
                .par_iter()
                .zip(output.par_iter_mut())
                .for_each(|(game, action)| write(game, action));
        } else {
            for (game, action) in self.games().iter().zip(output.iter_mut()) {
                write(game, action);
            }
        }
        Ok(())
    }

    /// Writes rule-EV actions only where `enabled` is one.
    pub fn rule_ev_actions_masked_into(
        &self,
        enabled: &[u8],
        output: &mut [u8],
    ) -> Result<(), GameError> {
        if enabled.len() != self.len() || output.len() != self.len() {
            return Err(GameError::BatchLength);
        }
        if enabled.iter().any(|&value| value > 1) {
            return Err(GameError::InvalidAction);
        }
        let write = |game: &Game, enabled: u8, action: &mut u8| {
            if enabled != 0 {
                *action = game
                    .rule_ev_action()
                    .map_or(RULE_EV_ACTION_TERMINAL, |id| id.index() as u8);
            }
        };
        if self.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games()
                .par_iter()
                .zip(enabled.par_iter().copied())
                .zip(output.par_iter_mut())
                .for_each(|((game, enabled), action)| write(game, enabled, action));
        } else {
            for ((game, enabled), action) in self
                .games()
                .iter()
                .zip(enabled.iter().copied())
                .zip(output.iter_mut())
            {
                write(game, enabled, action);
            }
        }
        Ok(())
    }
}

const MAX_ROLLOUT_ACTIONS: usize = 1_024;
const SEARCH_SEED_DOMAIN: u64 = 0x91e1_0da5_c79e_7b1d;
const INVALID_ROLLOUT_UTILITY: i64 = i64::MIN / 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpponentModel {
    SimpleRule,
    RuleEv(RuleEvConfig),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RolloutPolicy {
    root: RuleEvConfig,
    opponents: OpponentModel,
}

impl RolloutPolicy {
    #[cfg(feature = "rule-ev-analysis")]
    const fn homogeneous(config: RuleEvConfig) -> Self {
        Self {
            root: config,
            opponents: OpponentModel::RuleEv(config),
        }
    }
}

fn information_set_search(
    game: &Game,
    legal: &LegalActions,
    config: RuleEvConfig,
) -> Option<ActionId> {
    let heuristic = heuristic_action(
        game,
        legal,
        RuleEvConfig {
            search_depth: config.search_depth,
            search_worlds: 0,
            defense: config.defense,
            search_gate: config.search_gate,
        },
    );
    let actions = plausible_search_actions(game, legal, heuristic);
    if actions.len() <= 1 {
        return Some(heuristic);
    }
    SEARCH_DECISIONS.fetch_add(1, AtomicOrdering::Relaxed);

    let actor = legal.decision.actor;
    let worlds = usize::from(config.search_worlds);
    let sampled_worlds = sample_search_worlds(game, actor, worlds, SEARCH_SEED_DOMAIN);
    let root = RuleEvConfig::with_search_depth(config.search_depth)
        .expect("an existing search depth remains valid")
        .with_defense(config.defense);
    let rollout_policies = [
        RolloutPolicy {
            root,
            opponents: OpponentModel::SimpleRule,
        },
        RolloutPolicy {
            root,
            opponents: OpponentModel::RuleEv(RuleEvConfig::FAST.with_defense(config.defense)),
        },
        RolloutPolicy {
            root,
            opponents: OpponentModel::RuleEv(RuleEvConfig::STANDARD.with_defense(config.defense)),
        },
    ];
    let scores = score_search_actions(&sampled_worlds, actor, &actions, &rollout_policies);
    let task_count = scores.len();
    SEARCH_ROLLOUTS.fetch_add(task_count as u64, AtomicOrdering::Relaxed);

    let heuristic_index = actions
        .iter()
        .position(|&action| action == heuristic)
        .expect("the heuristic action is legal");
    let best = select_search_action(
        &scores,
        &actions,
        heuristic_index,
        worlds,
        rollout_policies.len(),
        config.search_gate,
    );
    if best != heuristic {
        SEARCH_OVERRIDES.fetch_add(1, AtomicOrdering::Relaxed);
    }
    Some(best)
}

fn select_search_action(
    scores: &[i64],
    actions: &[ActionId],
    baseline: usize,
    worlds: usize,
    rollout_policy_count: usize,
    gate: RuleEvSearchGate,
) -> ActionId {
    debug_assert!(!actions.is_empty());
    debug_assert!(baseline < actions.len());
    debug_assert!(worlds != 0);
    debug_assert!(rollout_policy_count != 0);
    debug_assert_eq!(scores.len(), worlds * rollout_policy_count * actions.len());

    let baseline_action = actions[baseline];
    if scores.contains(&INVALID_ROLLOUT_UTILITY) {
        return baseline_action;
    }
    if gate == RuleEvSearchGate::WorldClustered {
        return select_validated_world_action(
            scores,
            actions,
            baseline,
            worlds,
            rollout_policy_count,
        );
    }

    let mut best = baseline_action;
    let mut best_lcb = 0.0_f64;
    for (action_index, &action) in actions.iter().enumerate() {
        if action_index == baseline {
            continue;
        }
        let differences = search_gate_differences(
            scores,
            actions.len(),
            action_index,
            baseline,
            worlds,
            rollout_policy_count,
            gate,
        );
        let sample_count = differences.len();
        let mean = differences.iter().sum::<f64>() / sample_count as f64;
        let variance = if sample_count > 1 {
            differences
                .iter()
                .map(|difference| (difference - mean).powi(2))
                .sum::<f64>()
                / (sample_count - 1) as f64
        } else {
            f64::INFINITY
        };
        let positive = differences
            .iter()
            .filter(|&&difference| difference > 0.0)
            .count();
        let lcb = mean - 1.28 * (variance / sample_count as f64).sqrt();
        if gate.is_consistent(positive, sample_count)
            && lcb > 0.0
            && (lcb > best_lcb || (lcb == best_lcb && action.index() < best.index()))
        {
            best = action;
            best_lcb = lcb;
        }
    }
    best
}

fn select_validated_world_action(
    scores: &[i64],
    actions: &[ActionId],
    baseline: usize,
    worlds: usize,
    rollout_policy_count: usize,
) -> ActionId {
    let baseline_action = actions[baseline];
    let selection_worlds: Vec<_> = (0..worlds).step_by(2).collect();
    let validation_worlds: Vec<_> = (1..worlds).step_by(2).collect();
    if selection_worlds.is_empty() || validation_worlds.len() < 2 {
        return baseline_action;
    }

    let mut selected = baseline;
    let mut selected_score = 0.0_f64;
    for (candidate, &action) in actions.iter().enumerate() {
        if candidate == baseline {
            continue;
        }
        let score = (0..rollout_policy_count)
            .map(|policy| {
                mean_policy_difference(
                    scores,
                    actions.len(),
                    candidate,
                    baseline,
                    policy,
                    &selection_worlds,
                    rollout_policy_count,
                )
            })
            .fold(f64::INFINITY, f64::min);
        if score > selected_score
            || (score == selected_score && action.index() < actions[selected].index())
        {
            selected = candidate;
            selected_score = score;
        }
    }
    if selected == baseline {
        return baseline_action;
    }

    for policy in 0..rollout_policy_count {
        let differences: Vec<_> = validation_worlds
            .iter()
            .map(|&world| {
                policy_difference(
                    scores,
                    actions.len(),
                    selected,
                    baseline,
                    world,
                    policy,
                    rollout_policy_count,
                )
            })
            .collect();
        let positive = differences
            .iter()
            .filter(|&&difference| difference > 0.0)
            .count();
        let mean = differences.iter().sum::<f64>() / differences.len() as f64;
        if positive * 4 < differences.len() * 3 || mean <= 0.0 {
            return baseline_action;
        }
    }

    let differences: Vec<_> = validation_worlds
        .iter()
        .map(|&world| {
            (0..rollout_policy_count)
                .map(|policy| {
                    policy_difference(
                        scores,
                        actions.len(),
                        selected,
                        baseline,
                        world,
                        policy,
                        rollout_policy_count,
                    )
                })
                .sum::<f64>()
                / rollout_policy_count as f64
        })
        .collect();
    let mean = differences.iter().sum::<f64>() / differences.len() as f64;
    let variance = differences
        .iter()
        .map(|difference| (difference - mean).powi(2))
        .sum::<f64>()
        / (differences.len() - 1) as f64;
    let standard_error = (variance / differences.len() as f64).sqrt();
    let lower_bound = mean - student_t_95_one_sided(differences.len() - 1) * standard_error;
    if lower_bound > 0.0 {
        actions[selected]
    } else {
        baseline_action
    }
}

fn mean_policy_difference(
    scores: &[i64],
    action_count: usize,
    candidate: usize,
    baseline: usize,
    policy: usize,
    worlds: &[usize],
    rollout_policy_count: usize,
) -> f64 {
    worlds
        .iter()
        .map(|&world| {
            policy_difference(
                scores,
                action_count,
                candidate,
                baseline,
                world,
                policy,
                rollout_policy_count,
            )
        })
        .sum::<f64>()
        / worlds.len() as f64
}

fn policy_difference(
    scores: &[i64],
    action_count: usize,
    candidate: usize,
    baseline: usize,
    world: usize,
    policy: usize,
    rollout_policy_count: usize,
) -> f64 {
    let row = (world * rollout_policy_count + policy) * action_count;
    (scores[row + candidate] - scores[row + baseline]) as f64
}

fn student_t_95_one_sided(degrees_of_freedom: usize) -> f64 {
    const CRITICAL_VALUES: [f64; 30] = [
        6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782, 1.771,
        1.761, 1.753, 1.746, 1.740, 1.734, 1.729, 1.725, 1.721, 1.717, 1.714, 1.711, 1.708, 1.706,
        1.703, 1.701, 1.699, 1.697,
    ];
    CRITICAL_VALUES
        .get(degrees_of_freedom.saturating_sub(1))
        .copied()
        .unwrap_or(1.645)
}

impl RuleEvSearchGate {
    fn is_consistent(self, positive: usize, sample_count: usize) -> bool {
        match self {
            #[cfg(feature = "rule-ev-analysis")]
            Self::ScenarioStrict => positive * 4 >= sample_count * 3,
            #[cfg(feature = "rule-ev-analysis")]
            Self::ScenarioRelaxed => {
                if sample_count <= 8 {
                    positive * 4 >= sample_count * 3
                } else {
                    positive * 5 >= sample_count * 3
                }
            }
            Self::WorldClustered => {
                if sample_count <= 4 {
                    positive * 4 >= sample_count * 3
                } else {
                    positive * 5 >= sample_count * 3
                }
            }
        }
    }
}

fn search_gate_differences(
    scores: &[i64],
    action_count: usize,
    candidate: usize,
    baseline: usize,
    worlds: usize,
    rollout_policy_count: usize,
    gate: RuleEvSearchGate,
) -> Vec<f64> {
    match gate {
        #[cfg(feature = "rule-ev-analysis")]
        RuleEvSearchGate::ScenarioStrict | RuleEvSearchGate::ScenarioRelaxed => {
            paired_scenario_differences(scores, action_count, candidate, baseline)
        }
        RuleEvSearchGate::WorldClustered => paired_world_differences(
            scores,
            action_count,
            candidate,
            baseline,
            worlds,
            rollout_policy_count,
        ),
    }
}

#[cfg(feature = "rule-ev-analysis")]
fn paired_scenario_differences(
    scores: &[i64],
    action_count: usize,
    candidate: usize,
    baseline: usize,
) -> Vec<f64> {
    debug_assert_eq!(scores.len() % action_count, 0);
    scores
        .chunks_exact(action_count)
        .map(|row| (row[candidate] - row[baseline]) as f64)
        .collect()
}

fn paired_world_differences(
    scores: &[i64],
    action_count: usize,
    candidate: usize,
    baseline: usize,
    worlds: usize,
    rollout_policy_count: usize,
) -> Vec<f64> {
    debug_assert_eq!(scores.len(), worlds * rollout_policy_count * action_count);
    (0..worlds)
        .map(|world| {
            let first_scenario = world * rollout_policy_count;
            let total: i64 = (0..rollout_policy_count)
                .map(|policy| {
                    let row = (first_scenario + policy) * action_count;
                    scores[row + candidate] - scores[row + baseline]
                })
                .sum();
            total as f64 / rollout_policy_count as f64
        })
        .collect()
}

fn plausible_search_actions(
    game: &Game,
    legal: &LegalActions,
    heuristic: ActionId,
) -> Vec<ActionId> {
    let actor = legal.decision.actor;
    let holding = Holding::from_game(game, actor);
    let exposure = public_exposure(game, actor);
    let mut discards = Vec::new();
    for action in game
        .legal_action_mask()
        .expect("a legal-action set has a mask")
        .iter()
    {
        let Action::Discard(tile) = action.action() else {
            continue;
        };
        let Some(after) = holding.after_discard(tile) else {
            continue;
        };
        let quality = static_hand_quality(&after, &exposure);
        discards.push((
            action,
            quality,
            discard_attack_value(quality, game.has_won(actor)),
        ));
    }
    if discards.is_empty() {
        return vec![heuristic];
    }

    let has_won = game.has_won(actor);
    let best_shanten = discards
        .iter()
        .map(|(_, quality, _)| quality.shanten)
        .min()
        .expect("there is a discard");
    let best_attack = discards
        .iter()
        .filter(|(_, quality, _)| has_won || quality.shanten == best_shanten)
        .map(|(_, _, attack)| *attack)
        .max()
        .expect("there is a comparable discard");

    let mut actions = Vec::with_capacity(discards.len() + 1);
    actions.push(heuristic);
    for (action, quality, attack) in discards {
        let preserves_shanten = has_won || quality.shanten == best_shanten;
        let preserves_value = attack.saturating_mul(4) >= best_attack.saturating_mul(3);
        if preserves_shanten && preserves_value && action != heuristic {
            actions.push(action);
        }
    }
    actions.sort_unstable();
    actions
}

#[cfg(feature = "rule-ev-analysis")]
fn all_discard_search_actions(game: &Game, baseline: ActionId) -> Vec<ActionId> {
    let mut actions = vec![baseline];
    actions.extend(
        game.legal_action_mask()
            .expect("a search trace has a legal-action mask")
            .iter()
            .filter(|action| matches!(action.action(), Action::Discard(_))),
    );
    actions.sort_unstable();
    actions.dedup();
    actions
}

#[cfg(feature = "rule-ev-analysis")]
fn search_action_metadata(
    game: &Game,
    legal: &LegalActions,
    actions: &[ActionId],
    online_actions: &[ActionId],
) -> Vec<RuleEvSearchAction> {
    let actor = legal.decision.actor;
    let holding = Holding::from_game(game, actor);
    let exposure = public_exposure(game, actor);
    let has_won = game.has_won(actor);
    actions
        .iter()
        .map(|&action| {
            let quality = match action.action() {
                Action::Discard(tile) => holding
                    .after_discard(tile)
                    .map(|after| static_hand_quality(&after, &exposure)),
                _ => None,
            };
            RuleEvSearchAction {
                action,
                shanten: quality.map(|quality| quality.shanten),
                attack_value: quality.map(|quality| discard_attack_value(quality, has_won)),
                online_candidate: online_actions.binary_search(&action).is_ok(),
            }
        })
        .collect()
}

fn sample_search_worlds(game: &Game, actor: Seat, worlds: usize, seed_domain: u64) -> Vec<Game> {
    let public_seed = public_state_hash(game, actor) ^ seed_domain;
    (0..worlds)
        .map(|world_index| {
            let seed = mix_search_seed(public_seed.wrapping_add(world_index as u64));
            game.resample_information_set(seed)
                .expect("turn decisions have an information set")
        })
        .collect()
}

fn score_search_actions(
    worlds: &[Game],
    actor: Seat,
    actions: &[ActionId],
    rollout_policies: &[RolloutPolicy],
) -> Vec<i64> {
    let scenario_count = worlds.len() * rollout_policies.len();
    let task_count = scenario_count * actions.len();
    (0..task_count)
        .into_par_iter()
        .map(|task| {
            let action_index = task % actions.len();
            let scenario = task / actions.len();
            let rollout_policy = rollout_policies[scenario % rollout_policies.len()];
            let world = &worlds[scenario / rollout_policies.len()];
            rollout_action(world, actor, actions[action_index], rollout_policy)
        })
        .collect()
}

fn heuristic_action(game: &Game, legal: &LegalActions, config: RuleEvConfig) -> ActionId {
    let actor = legal.decision.actor;
    match legal.decision.phase {
        Phase::Exchange => opening::choose_exchange(
            game.concealed(actor),
            game.exchange_selection(actor),
            legal.exchange_mask,
        ),
        Phase::ChooseMissing => opening::choose_missing(game.concealed(actor)),
        Phase::Turn => choose_turn(game, actor, legal, config),
        Phase::HuResponse => ActionId::HU,
        Phase::MeldResponse => choose_meld_response(game, actor, legal, config),
        Phase::Finished => unreachable!("a legal-action set is non-terminal"),
    }
}

fn rollout_action(
    world: &Game,
    root_actor: Seat,
    action: ActionId,
    rollout_policy: RolloutPolicy,
) -> i64 {
    let mut game = world.clone();
    if game.step_id(action).is_err() {
        return INVALID_ROLLOUT_UTILITY;
    }
    for _ in 0..MAX_ROLLOUT_ACTIONS {
        let Some(decision) = game.decision() else {
            return terminal_utility(&game, root_actor);
        };
        let next = if decision.actor == root_actor {
            game.rule_ev_action_with_config(rollout_policy.root)
        } else {
            match rollout_policy.opponents {
                OpponentModel::SimpleRule => game.simple_rule_action(),
                OpponentModel::RuleEv(config) => game.rule_ev_action_with_config(config),
            }
        }
        .expect("a non-terminal rollout policy returns an action");
        if game.step_id(next).is_err() {
            return INVALID_ROLLOUT_UTILITY;
        }
    }
    INVALID_ROLLOUT_UTILITY
}

fn terminal_utility(game: &Game, actor: Seat) -> i64 {
    let rank = game
        .rankings()
        .iter()
        .position(|&seat| seat == actor)
        .expect("a terminal player has a rank");
    const RANK_UTILITY: [i64; PLAYER_COUNT] = [30_000, 10_000, -10_000, -30_000];
    let centered_score = (game.score(actor) - 10_000).clamp(-10_000, 30_000);
    RANK_UTILITY[rank] + centered_score / 4
}

fn public_state_hash(game: &Game, actor: Seat) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut write = |byte: u8| {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };

    write(actor.as_u8());
    write(game.phase() as u8);
    write(game.dealer().as_u8());
    write(game.exchange_direction() as u8);
    for byte in (game.wall_remaining() as u64).to_le_bytes() {
        write(byte);
    }
    for count in game.concealed(actor) {
        write(*count);
    }
    for count in game.exchange_selection(actor) {
        write(*count);
    }
    for seat in Seat::ALL {
        for count in game.locked(seat) {
            write(*count);
        }
        for byte in game.score(seat).to_le_bytes() {
            write(byte);
        }
        write(game.missing_suit(seat).map_or(u8::MAX, |suit| suit as u8));
        write(u8::from(game.has_won(seat)));
        write(game.concealed_len(seat).min(u8::MAX as usize) as u8);
        write(game.meld_count(seat) as u8);
        for index in 0..game.meld_count(seat) {
            let meld = game.meld(seat, index).expect("meld slots are dense");
            write(meld.tile.as_u8());
            write(meld.kind as u8);
            write(meld.source.as_u8());
        }
    }
    for (seat, tile) in game.discards() {
        write(seat.as_u8());
        write(tile.as_u8());
    }
    if let Some(draw) = game.current_draw()
        && draw.player == actor
    {
        write(draw.tile.as_u8());
        write(u8::from(draw.replacement));
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct WaitQuality {
    weighted_value: u64,
    live_copies: u16,
    max_multiplier: u32,
    distinct_tiles: u8,
}

impl WaitQuality {
    fn cmp_quality(self, other: Self) -> Ordering {
        (
            self.weighted_value,
            self.live_copies,
            self.max_multiplier,
            self.distinct_tiles,
        )
            .cmp(&(
                other.weighted_value,
                other.live_copies,
                other.max_multiplier,
                other.distinct_tiles,
            ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HandQuality {
    shanten: i8,
    live_improvements: u16,
    one_draw_value: u64,
    waits: WaitQuality,
    structure: i32,
}

fn hand_quality(
    holding: &Holding,
    exposure: &[u8; TILE_KIND_COUNT],
    search_depth: u8,
) -> HandQuality {
    let mut quality = static_hand_quality(holding, exposure);
    if search_depth != 0 && (1..=2).contains(&quality.shanten) {
        quality.one_draw_value = one_draw_value(
            holding,
            exposure,
            quality.shanten,
            analyze_shanten(&holding.concealed, holding.melds(), holding.missing).improving_tiles,
            search_depth,
        );
    }
    quality
}

fn static_hand_quality(holding: &Holding, exposure: &[u8; TILE_KIND_COUNT]) -> HandQuality {
    let analysis = analyze_shanten(&holding.concealed, holding.melds(), holding.missing);
    let waits = if analysis.shanten <= 0 {
        wait_quality(holding, exposure)
    } else {
        WaitQuality::default()
    };
    HandQuality {
        shanten: analysis.shanten,
        live_improvements: remaining_copies(analysis.improving_tiles, exposure),
        one_draw_value: 0,
        waits,
        structure: hand_structure_score(&holding.concealed),
    }
}

fn one_draw_value(
    holding: &Holding,
    exposure: &[u8; TILE_KIND_COUNT],
    current_shanten: i8,
    improving_tiles: u32,
    search_depth: u8,
) -> u64 {
    debug_assert!(search_depth != 0);
    let mut expected = 0_u64;
    let mut augmented = *holding;
    for tile in mask_tiles(improving_tiles) {
        let copies = u64::from(4_u8.saturating_sub(exposure[tile.index()].min(4)));
        if copies == 0 || augmented.concealed[tile.index()] >= 4 {
            continue;
        }
        augmented.concealed[tile.index()] += 1;
        let mut next_exposure = *exposure;
        next_exposure[tile.index()] = next_exposure[tile.index()].saturating_add(1).min(4);

        let direct_win = if augmented.missing_count() == 0 {
            evaluate_win(
                &augmented.concealed,
                augmented.melds(),
                Some(tile),
                WinFlags::NONE,
            )
        } else {
            None
        };
        let continuation = if let Some(win) = direct_win {
            2_000 + u64::from(win.multiplier.min(WAIT_MULTIPLIER_CAP)) * 200
        } else {
            let mut best_after_discard = 0_u64;
            for discard in mask_tiles(augmented.discard_mask()) {
                let Some(after) = augmented.after_discard(discard) else {
                    continue;
                };
                let quality = static_hand_quality(&after, &next_exposure);
                if quality.shanten < current_shanten {
                    let mut value = static_attack_value(quality);
                    if search_depth > 1 && (0..=2).contains(&quality.shanten) {
                        let analysis =
                            analyze_shanten(&after.concealed, after.melds(), after.missing);
                        value = value.saturating_add(
                            one_draw_value(
                                &after,
                                &next_exposure,
                                quality.shanten,
                                analysis.improving_tiles,
                                search_depth - 1,
                            ) / DEEP_SEARCH_DISCOUNT,
                        );
                    }
                    best_after_discard = best_after_discard.max(value);
                }
            }
            best_after_discard
        };
        if continuation != 0 {
            expected = expected.saturating_add(copies * continuation);
        }
        augmented.concealed[tile.index()] -= 1;
    }
    expected
}

fn static_attack_value(quality: HandQuality) -> u64 {
    if quality.shanten <= 0 {
        500 + quality.waits.weighted_value * 20
            + u64::from(quality.waits.live_copies) * 10
            + u64::from(quality.waits.max_multiplier) * 5
    } else {
        u64::from(quality.live_improvements) * 40 + quality.structure.max(0) as u64 / 4
    }
}

fn wait_quality(holding: &Holding, exposure: &[u8; TILE_KIND_COUNT]) -> WaitQuality {
    if holding.missing_count() != 0 {
        return WaitQuality::default();
    }
    let mut result = WaitQuality::default();
    let mut augmented = holding.concealed;
    for tile in all_tiles() {
        if holding.missing == Some(tile.suit()) {
            continue;
        }
        let copies = 4_u8.saturating_sub(exposure[tile.index()].min(4));
        if copies == 0 {
            continue;
        }
        augmented[tile.index()] = augmented[tile.index()].saturating_add(1);
        if let Some(evaluation) =
            evaluate_win(&augmented, holding.melds(), Some(tile), WinFlags::NONE)
        {
            let multiplier = evaluation.multiplier.min(WAIT_MULTIPLIER_CAP);
            result.weighted_value += u64::from(copies) * u64::from(multiplier);
            result.live_copies += u16::from(copies);
            result.max_multiplier = result.max_multiplier.max(multiplier);
            result.distinct_tiles += 1;
        }
        augmented[tile.index()] = holding.concealed[tile.index()];
    }
    result
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiscardDanger {
    certain_winners: u8,
    expected_loss: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DiscardCandidate {
    tile: Tile,
    quality: HandQuality,
    danger: DiscardDanger,
    exposure: u8,
}

fn best_discard(
    game: &Game,
    actor: Seat,
    holding: Holding,
    discard_mask: u32,
    exposure: &[u8; TILE_KIND_COUNT],
    config: RuleEvConfig,
) -> Option<DiscardCandidate> {
    let has_won = game.has_won(actor);
    let mut best = None;
    for tile in mask_tiles(discard_mask) {
        let Some(remaining) = holding.after_discard(tile) else {
            continue;
        };
        let candidate = DiscardCandidate {
            tile,
            quality: hand_quality(&remaining, exposure, config.search_depth),
            danger: discard_danger(game, actor, tile, exposure, config.defense),
            exposure: exposure[tile.index()],
        };
        if best.is_none_or(|current| discard_better(candidate, current, has_won)) {
            best = Some(candidate);
        }
    }
    best
}

fn discard_better(candidate: DiscardCandidate, current: DiscardCandidate, has_won: bool) -> bool {
    if !has_won && candidate.quality.shanten != current.quality.shanten {
        return candidate.quality.shanten < current.quality.shanten;
    }
    let candidate_attack = discard_attack_value(candidate.quality, has_won);
    let current_attack = discard_attack_value(current.quality, has_won);
    let candidate_utility = candidate_attack as i128 * i128::from(RISK_SCALE)
        - i128::from(candidate.danger.expected_loss) * i128::from(DEFENSE_RISK_WEIGHT);
    let current_utility = current_attack as i128 * i128::from(RISK_SCALE)
        - i128::from(current.danger.expected_loss) * i128::from(DEFENSE_RISK_WEIGHT);
    if candidate_utility != current_utility {
        return candidate_utility > current_utility;
    }
    (
        candidate.quality.structure,
        candidate.exposure,
        edge_distance(candidate.tile),
        u8::MAX - candidate.tile.as_u8(),
    ) > (
        current.quality.structure,
        current.exposure,
        edge_distance(current.tile),
        u8::MAX - current.tile.as_u8(),
    )
}

fn discard_attack_value(quality: HandQuality, has_won: bool) -> u64 {
    if has_won || quality.shanten <= 0 {
        quality.waits.weighted_value * 10
            + u64::from(quality.waits.live_copies) * 5
            + u64::from(quality.waits.max_multiplier)
    } else {
        u64::from(quality.live_improvements) * 100
            + quality.one_draw_value
            + quality.structure.max(0) as u64 / 4
    }
}

fn discard_danger(
    game: &Game,
    actor: Seat,
    tile: Tile,
    exposure: &[u8; TILE_KIND_COUNT],
    defense: RuleEvDefense,
) -> DiscardDanger {
    match defense {
        RuleEvDefense::None => DiscardDanger::default(),
        RuleEvDefense::Heuristic => heuristic_discard_danger(game, actor, tile, exposure),
    }
}

fn heuristic_discard_danger(
    game: &Game,
    actor: Seat,
    tile: Tile,
    exposure: &[u8; TILE_KIND_COUNT],
) -> DiscardDanger {
    const DEFENSE_ONSET_DISCARDS: f64 = 24.0;
    const DEFENSE_FULL_DISCARDS: f64 = 56.0;
    const RANK_FACTORS: [f64; PLAYER_COUNT] = [1.40, 1.00, 0.65, 0.35];

    let discards = game.discards().count() as f64;
    let progress = ((discards - DEFENSE_ONSET_DISCARDS)
        / (DEFENSE_FULL_DISCARDS - DEFENSE_ONSET_DISCARDS))
        .clamp(0.0, 1.0);
    let rank = game
        .rankings()
        .iter()
        .position(|&seat| seat == actor)
        .expect("the actor has a rank");
    let known_risk_factor = RANK_FACTORS[rank];
    let uncertain_risk_factor = progress * known_risk_factor;

    let mut danger = DiscardDanger::default();
    for opponent in Seat::ALL {
        if opponent == actor || game.missing_suit(opponent) == Some(tile.suit()) {
            continue;
        }
        if let Some(multiplier) = certain_wait_multiplier(game, opponent, tile) {
            danger.certain_winners += 1;
            danger.expected_loss = danger.expected_loss.saturating_add(
                (f64::from(multiplier.min(WAIT_MULTIPLIER_CAP))
                    * RISK_SCALE as f64
                    * known_risk_factor) as u64,
            );
            continue;
        }
        let posterior =
            opponent_win_posterior(game, opponent, tile, exposure) * uncertain_risk_factor;
        let multiplier = expected_discard_multiplier(game, opponent);
        danger.expected_loss = danger
            .expected_loss
            .saturating_add((posterior * f64::from(multiplier) * RISK_SCALE as f64) as u64);
    }
    danger
}

fn opponent_win_posterior(
    game: &Game,
    opponent: Seat,
    tile: Tile,
    exposure: &[u8; TILE_KIND_COUNT],
) -> f64 {
    let hidden = game.concealed_len(opponent).saturating_sub(
        game.locked(opponent)
            .iter()
            .map(|&count| usize::from(count))
            .sum(),
    );
    if hidden == 0 {
        return 0.0;
    }
    let unknown_total: usize = exposure
        .iter()
        .map(|&count| usize::from(4_u8.saturating_sub(count.min(4))))
        .sum();
    if unknown_total == 0 {
        return 0.0;
    }

    let available =
        |candidate: Tile| usize::from(4_u8.saturating_sub(exposure[candidate.index()].min(4)));
    let same = available(tile);
    let mut completion = contains_probability(same, 1, hidden, unknown_total) * 0.34;
    if same >= 2 {
        completion += contains_probability(same, 2, hidden, unknown_total) * 0.26;
    }
    let rank = tile.rank();
    let start_min = rank.saturating_sub(2);
    let start_max = rank.min(6);
    for start in start_min..=start_max {
        if rank < start || rank > start + 2 {
            continue;
        }
        let mut required = [0_usize; 2];
        let mut length = 0;
        for offset in 0..3 {
            let candidate = Tile::from_suit_rank(tile.suit(), start + offset)
                .expect("sequence ranks stay within one suit");
            if candidate == tile {
                continue;
            }
            required[length] = available(candidate);
            length += 1;
        }
        if length == 2 {
            completion +=
                contains_probability_pair(required[0], required[1], hidden, unknown_total) * 0.20;
        }
    }

    let total_discards = game.discards().count() as f64;
    let opponent_discards = game
        .discards()
        .filter(|(seat, _)| *seat == opponent)
        .count() as f64;
    let melds = game.meld_count(opponent) as f64;
    let readiness_logit = -3.7 + 0.085 * total_discards + 0.12 * opponent_discards + 0.48 * melds;
    let readiness = 1.0 / (1.0 + (-readiness_logit).exp());
    let suit_bonus = if game.meld_count(opponent) >= 2
        && (0..game.meld_count(opponent)).all(|index| {
            game.meld(opponent, index)
                .is_some_and(|meld| meld.tile.suit() == tile.suit())
        }) {
        1.35
    } else {
        1.0
    };
    (completion * readiness * suit_bonus).clamp(0.0, 0.85)
}

fn contains_probability(available: usize, required: usize, hand: usize, total: usize) -> f64 {
    if available < required || hand < required || total < required {
        return 0.0;
    }
    let denominator = binomial(total, required) as f64;
    if denominator == 0.0 {
        return 0.0;
    }
    let expected_subsets =
        binomial(available, required) as f64 * binomial(hand, required) as f64 / denominator;
    // Several physical-copy subsets can satisfy the same event. A Poisson
    // union gives a smooth lower-bound posterior without enumerating hands.
    1.0 - (-expected_subsets).exp()
}

fn contains_probability_pair(
    first_available: usize,
    second_available: usize,
    hand: usize,
    total: usize,
) -> f64 {
    if first_available == 0 || second_available == 0 || hand < 2 || total < 2 {
        return 0.0;
    }
    let pair_inclusion = (hand * (hand - 1)) as f64 / (total * (total - 1)) as f64;
    (pair_inclusion * first_available as f64 * second_available as f64).min(1.0)
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    (0..k).fold(1_usize, |value, index| value * (n - index) / (index + 1))
}

fn expected_discard_multiplier(game: &Game, opponent: Seat) -> u32 {
    // Exact waits use the scoring engine. Unknown hands use only public melds.
    1 + (game.meld_count(opponent) as u32).min(3)
}

fn certain_wait_multiplier(game: &Game, opponent: Seat, tile: Tile) -> Option<u32> {
    if !game.has_won(opponent) || game.missing_suit(opponent) == Some(tile.suit()) {
        return None;
    }
    let mut counts = *game.locked(opponent);
    counts[tile.index()] = counts[tile.index()].saturating_add(1);
    let mut melds = [DUMMY_MELD; 4];
    let meld_len = game.meld_count(opponent);
    for (index, meld) in melds.iter_mut().enumerate().take(meld_len) {
        *meld = game.meld(opponent, index).expect("meld slots are dense");
    }
    evaluate_win(&counts, &melds[..meld_len], Some(tile), WinFlags::NONE)
        .map(|evaluation| evaluation.multiplier)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KongCandidate {
    action: ActionId,
    quality: HandQuality,
    immediate_value: i64,
    tile: Tile,
}

fn choose_turn(game: &Game, actor: Seat, legal: &LegalActions, config: RuleEvConfig) -> ActionId {
    if legal.can_hu {
        return ActionId::HU;
    }

    let holding = Holding::from_game(game, actor);
    let exposure = public_exposure(game, actor);
    let baseline = best_discard(game, actor, holding, legal.discard_mask, &exposure, config)
        .expect("a turn has a legal discard");
    let seven_pairs = seven_pairs_shanten(&holding.concealed, holding.missing);
    let mut best_kong = None;

    if seven_pairs > 1 {
        for tile in mask_tiles(legal.concealed_kong_mask) {
            let Some(after) = holding.after_concealed_kong(tile, actor) else {
                continue;
            };
            let quality = hand_quality(&after, &exposure, config.search_depth);
            if kong_preserves_progress(quality, baseline.quality, game.has_won(actor)) {
                let candidate = KongCandidate {
                    action: ActionId::concealed_kong(tile),
                    quality,
                    immediate_value: Seat::ALL
                        .into_iter()
                        .filter(|&seat| seat != actor)
                        .map(|seat| game.score(seat).clamp(0, 200))
                        .sum(),
                    tile,
                };
                if best_kong
                    .is_none_or(|current| kong_better(candidate, current, game.has_won(actor)))
                {
                    best_kong = Some(candidate);
                }
            }
        }
    }

    for tile in mask_tiles(legal.added_kong_mask) {
        if discard_danger(game, actor, tile, &exposure, config.defense).certain_winners != 0 {
            continue;
        }
        let Some(after) = holding.after_added_kong(tile) else {
            continue;
        };
        let quality = hand_quality(&after, &exposure, config.search_depth);
        if kong_preserves_progress(quality, baseline.quality, game.has_won(actor)) {
            let candidate = KongCandidate {
                action: ActionId::added_kong(tile),
                quality,
                immediate_value: Seat::ALL
                    .into_iter()
                    .filter(|&seat| seat != actor)
                    .map(|seat| game.score(seat).clamp(0, 100))
                    .sum(),
                tile,
            };
            if best_kong.is_none_or(|current| kong_better(candidate, current, game.has_won(actor)))
            {
                best_kong = Some(candidate);
            }
        }
    }

    best_kong.map_or_else(|| ActionId::discard(baseline.tile), |kong| kong.action)
}

fn kong_preserves_progress(candidate: HandQuality, baseline: HandQuality, has_won: bool) -> bool {
    if has_won {
        candidate.waits.cmp_quality(baseline.waits) != Ordering::Less
    } else {
        candidate.shanten <= baseline.shanten
    }
}

fn kong_better(candidate: KongCandidate, current: KongCandidate, has_won: bool) -> bool {
    let progress = if has_won {
        candidate.quality.waits.cmp_quality(current.quality.waits)
    } else {
        current.quality.shanten.cmp(&candidate.quality.shanten)
    };
    progress == Ordering::Greater
        || (progress == Ordering::Equal
            && (candidate.immediate_value, u8::MAX - candidate.tile.as_u8())
                > (current.immediate_value, u8::MAX - current.tile.as_u8()))
}

fn choose_meld_response(
    game: &Game,
    actor: Seat,
    legal: &LegalActions,
    config: RuleEvConfig,
) -> ActionId {
    let Some((source, tile)) = game.discards().last() else {
        return ActionId::PASS;
    };
    let holding = Holding::from_game(game, actor);
    let exposure = public_exposure(game, actor);
    let pass = hand_quality(&holding, &exposure, config.search_depth);
    let seven_pairs = seven_pairs_shanten(&holding.concealed, holding.missing);

    if legal.can_exposed_kong
        && seven_pairs > 1
        && let Some(after) = holding.after_exposed_kong(tile, source)
    {
        let kong = hand_quality(&after, &exposure, config.search_depth);
        if meld_kong_is_worthwhile(pass, kong, game.has_won(actor)) {
            return ActionId::EXPOSED_KONG;
        }
    }

    if legal.can_pong
        && seven_pairs > 1
        && let Some(after_pong) = holding.after_pong(tile, source)
        && let Some(after) = best_discard(
            game,
            actor,
            after_pong,
            after_pong.discard_mask(),
            &exposure,
            config,
        )
        && pong_is_worthwhile(pass, after.quality, game.has_won(actor))
    {
        return ActionId::PONG;
    }
    ActionId::PASS
}

fn meld_kong_is_worthwhile(pass: HandQuality, kong: HandQuality, has_won: bool) -> bool {
    if has_won {
        kong.waits.cmp_quality(pass.waits) != Ordering::Less
    } else {
        kong.shanten <= pass.shanten
    }
}

fn pong_is_worthwhile(pass: HandQuality, pong: HandQuality, has_won: bool) -> bool {
    if has_won {
        return pong.waits.cmp_quality(pass.waits) == Ordering::Greater;
    }
    if pong.shanten != pass.shanten {
        return pong.shanten < pass.shanten;
    }
    if pass.shanten <= 0 {
        pong.waits.cmp_quality(pass.waits) == Ordering::Greater
    } else {
        pong.live_improvements >= pass.live_improvements.saturating_add(PONG_UKEIRE_MARGIN)
    }
}

fn seven_pairs_shanten(counts: &[u8; TILE_KIND_COUNT], missing: Option<Suit>) -> i8 {
    let mut pairs = 0_i8;
    let mut pair_slots = 0_i8;
    for tile in all_tiles() {
        if missing == Some(tile.suit()) {
            continue;
        }
        let count = counts[tile.index()].min(4);
        pairs += i8::try_from(count / 2).expect("a tile has at most two pairs");
        pair_slots += i8::try_from(count.div_ceil(2)).expect("a tile has at most two pair slots");
    }
    6 - pairs + (7 - pair_slots).max(0)
}

fn public_exposure(game: &Game, actor: Seat) -> [u8; TILE_KIND_COUNT] {
    game.visible_tile_counts(actor)
}

fn remaining_copies(mask: u32, exposure: &[u8; TILE_KIND_COUNT]) -> u16 {
    mask_tiles(mask)
        .map(|tile| u16::from(4_u8.saturating_sub(exposure[tile.index()].min(4))))
        .sum()
}

fn edge_distance(tile: Tile) -> u8 {
    tile.rank().abs_diff(4)
}

const _: () = assert!(RULE_EV_ACTION_TERMINAL as usize >= ACTION_SPACE_SIZE);
const _: () = assert!(PLAYER_COUNT == 4);

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank - 1).expect("test rank is valid")
    }

    fn add_sequence(counts: &mut [u8; TILE_KIND_COUNT], suit: Suit, first_rank: u8) {
        for rank in first_rank..first_rank + 3 {
            counts[tile(suit, rank).index()] += 1;
        }
    }

    fn holding(counts: [u8; TILE_KIND_COUNT]) -> Holding {
        Holding {
            concealed: counts,
            locked: [0; TILE_KIND_COUNT],
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: None,
        }
    }

    #[test]
    fn rollout_policies_are_averaged_within_each_world() {
        let scores = [10, 14, 20, 22, 5, 0, 7, 8];
        assert_eq!(
            paired_world_differences(&scores, 2, 1, 0, 2, 2),
            [3.0, -2.0]
        );
    }

    #[cfg(feature = "rule-ev-analysis")]
    fn two_action_scores(differences: &[(i64, i64)]) -> Vec<i64> {
        differences
            .iter()
            .flat_map(|&(first, second)| [0, first, 0, second])
            .collect()
    }

    #[cfg(feature = "rule-ev-analysis")]
    fn two_action_three_policy_scores(differences: &[[i64; 3]]) -> Vec<i64> {
        differences
            .iter()
            .flat_map(|world| world.iter().flat_map(|&difference| [0, difference]))
            .collect()
    }

    #[cfg(feature = "rule-ev-analysis")]
    #[test]
    fn relaxed_scenario_gate_does_not_replace_world_clustering() {
        let actions = [
            ActionId::discard(tile(Suit::Characters, 1)),
            ActionId::discard(tile(Suit::Characters, 2)),
        ];
        let scores = two_action_scores(&[
            (100, 100),
            (100, 100),
            (100, 100),
            (100, 100),
            (1, -100),
            (1, -100),
            (-1, -1),
            (-1, -1),
        ]);

        assert_eq!(
            select_search_action(&scores, &actions, 0, 8, 2, RuleEvSearchGate::ScenarioStrict,),
            actions[0]
        );
        assert_eq!(
            select_search_action(
                &scores,
                &actions,
                0,
                8,
                2,
                RuleEvSearchGate::ScenarioRelaxed,
            ),
            actions[1]
        );
        assert_eq!(
            select_search_action(&scores, &actions, 0, 8, 2, RuleEvSearchGate::WorldClustered,),
            actions[0]
        );
    }

    #[cfg(feature = "rule-ev-analysis")]
    #[test]
    fn world_validation_requires_every_rollout_policy_to_agree() {
        let actions = [
            ActionId::discard(tile(Suit::Characters, 1)),
            ActionId::discard(tile(Suit::Characters, 2)),
        ];
        let scores = two_action_scores(&[
            (100, -1),
            (100, -1),
            (100, -1),
            (100, -1),
            (100, -1),
            (-1, -1),
            (-1, -1),
            (-1, -1),
        ]);

        for gate in [
            RuleEvSearchGate::ScenarioStrict,
            RuleEvSearchGate::ScenarioRelaxed,
        ] {
            assert_eq!(
                select_search_action(&scores, &actions, 0, 8, 2, gate),
                actions[0]
            );
        }
        assert_eq!(
            select_search_action(&scores, &actions, 0, 8, 2, RuleEvSearchGate::WorldClustered,),
            actions[0]
        );
    }

    #[cfg(feature = "rule-ev-analysis")]
    #[test]
    fn world_validation_uses_worlds_independent_from_candidate_selection() {
        let actions = [
            ActionId::discard(tile(Suit::Characters, 1)),
            ActionId::discard(tile(Suit::Characters, 2)),
        ];
        let scores = two_action_three_policy_scores(&[
            [100, 100, 100],
            [-1, -1, -1],
            [100, 100, 100],
            [-1, -1, -1],
            [100, 100, 100],
            [-1, -1, -1],
            [100, 100, 100],
            [-1, -1, -1],
        ]);
        assert_eq!(
            select_search_action(&scores, &actions, 0, 8, 3, RuleEvSearchGate::WorldClustered,),
            actions[0]
        );
    }

    #[cfg(feature = "rule-ev-analysis")]
    #[test]
    fn world_validation_accepts_an_independently_confirmed_consensus() {
        let actions = [
            ActionId::discard(tile(Suit::Characters, 1)),
            ActionId::discard(tile(Suit::Characters, 2)),
        ];
        let scores = two_action_three_policy_scores(&[[10, 11, 12]; 8]);
        assert_eq!(
            select_search_action(&scores, &actions, 0, 8, 3, RuleEvSearchGate::WorldClustered,),
            actions[1]
        );
    }

    #[cfg(feature = "rule-ev-analysis")]
    #[test]
    fn failed_rollout_returns_the_baseline() {
        let actions = [
            ActionId::discard(tile(Suit::Characters, 1)),
            ActionId::discard(tile(Suit::Characters, 2)),
        ];
        let mut scores = two_action_three_policy_scores(&[[10, 11, 12]; 8]);
        scores[0] = INVALID_ROLLOUT_UTILITY;
        assert_eq!(
            select_search_action(&scores, &actions, 0, 8, 3, RuleEvSearchGate::WorldClustered,),
            actions[0]
        );
    }

    #[cfg(feature = "rule-ev-analysis")]
    #[test]
    fn production_config_uses_the_world_clustered_gate() {
        assert_eq!(
            RuleEvConfig::DEEP.search_gate(),
            RuleEvSearchGate::WorldClustered
        );
    }

    #[cfg(feature = "rule-ev-analysis")]
    #[test]
    fn search_trace_reuses_the_runtime_rollout_path() {
        let mut game = Game::new(173);
        while game.phase() != Phase::Turn {
            let action = game
                .simple_rule_action()
                .expect("setup remains non-terminal");
            game.step_id(action).expect("simple rule action is legal");
        }
        let continuations = [
            RuleEvConfig::FAST.with_defense(RuleEvDefense::None),
            RuleEvConfig::STANDARD.with_defense(RuleEvDefense::None),
        ];
        let trace = game
            .rule_ev_search_trace(
                RuleEvConfig::STANDARD.with_defense(RuleEvDefense::None),
                &continuations,
                RuleEvSearchTraceConfig {
                    worlds: 2,
                    seed_domain: 0x5ab4_32d1_9810_73ef,
                    candidate_set: RuleEvSearchCandidateSet::AllDiscards,
                },
            )
            .expect("a non-Hu turn has a search trace");
        assert_eq!(trace.world_count, 2);
        assert_eq!(trace.continuation_count, 2);
        assert!(
            trace
                .actions
                .iter()
                .any(|action| action.action == trace.baseline)
        );
        assert!(trace.utility(1, 1, trace.actions.len() - 1).is_some());
        assert!(trace.utility(2, 0, 0).is_none());
        assert!(trace.actions.iter().any(|action| {
            action.action == trace.select_action(RuleEvSearchGate::WorldClustered)
        }));
    }

    #[test]
    fn reentry_wait_requires_the_new_tile_in_a_winning_subset() {
        let mut counts = [0; TILE_KIND_COUNT];
        add_sequence(&mut counts, Suit::Characters, 1);
        add_sequence(&mut counts, Suit::Characters, 4);
        add_sequence(&mut counts, Suit::Bamboo, 1);
        add_sequence(&mut counts, Suit::Bamboo, 4);
        let pair = tile(Suit::Bamboo, 9);
        counts[pair.index()] = 2;
        let mut exposure = counts;
        let quality = wait_quality(&holding(counts), &exposure);
        assert!(quality.live_copies >= 2);
        assert!(quality.weighted_value > 0);

        exposure[pair.index()] = 4;
        let exhausted = wait_quality(&holding(counts), &exposure);
        assert!(exhausted.live_copies < quality.live_copies);
    }

    #[test]
    fn discard_meld_transforms_never_consume_locked_tiles() {
        let pong_tile = tile(Suit::Characters, 5);
        let mut counts = [0; TILE_KIND_COUNT];
        counts[pong_tile.index()] = 3;
        counts[tile(Suit::Bamboo, 1).index()] = 1;
        let mut state = holding(counts);
        state.locked[pong_tile.index()] = 2;
        assert!(state.after_pong(pong_tile, Seat::EAST).is_none());
        assert!(state.after_exposed_kong(pong_tile, Seat::EAST).is_none());

        state.concealed[pong_tile.index()] = 4;
        state.locked[pong_tile.index()] = 1;
        let kong = state
            .after_exposed_kong(pong_tile, Seat::EAST)
            .expect("three unlocked matching tiles form an exposed Kong");
        assert_eq!(kong.concealed[pong_tile.index()], 1);
        assert_eq!(kong.locked[pong_tile.index()], 1);
    }

    #[test]
    fn seven_pairs_route_counts_a_quad_as_two_pairs() {
        let mut counts = [0; TILE_KIND_COUNT];
        counts[tile(Suit::Characters, 2).index()] = 4;
        for rank in 3..=7 {
            counts[tile(Suit::Characters, rank).index()] = 2;
        }
        assert_eq!(seven_pairs_shanten(&counts, None), -1);
        assert!(seven_pairs_shanten(&counts, Some(Suit::Characters)) > 0);
    }

    #[test]
    fn guaranteed_deal_in_beats_ukeire_on_equal_shanten() {
        let safe = DiscardCandidate {
            tile: tile(Suit::Characters, 1),
            quality: HandQuality {
                shanten: 1,
                live_improvements: 2,
                one_draw_value: 0,
                waits: WaitQuality::default(),
                structure: 0,
            },
            danger: DiscardDanger::default(),
            exposure: 1,
        };
        let dangerous = DiscardCandidate {
            tile: tile(Suit::Characters, 2),
            quality: HandQuality {
                live_improvements: 12,
                ..safe.quality
            },
            danger: DiscardDanger {
                certain_winners: 1,
                expected_loss: 100 * RISK_SCALE,
            },
            exposure: 1,
        };
        assert!(discard_better(safe, dangerous, false));
        assert!(!discard_better(dangerous, safe, false));
    }

    #[test]
    fn pong_requires_real_progress() {
        let pass = HandQuality {
            shanten: 1,
            live_improvements: 8,
            one_draw_value: 0,
            waits: WaitQuality::default(),
            structure: 0,
        };
        assert!(!pong_is_worthwhile(
            pass,
            HandQuality {
                live_improvements: 9,
                ..pass
            },
            false,
        ));
        assert!(pong_is_worthwhile(
            pass,
            HandQuality {
                shanten: 0,
                live_improvements: 3,
                ..pass
            },
            false,
        ));
    }
}
