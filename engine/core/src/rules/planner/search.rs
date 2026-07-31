//! Paired terminal policy improvement over sampled information sets.
//!
//! Every sampled world evaluates every legal root action under the same
//! continuation models. Selection maximizes the worst model-specific paired
//! gain. An independent particle stream accepts an override only when every
//! continuation model improves and the pooled paired lower confidence bound
//! is positive.

use rayon::prelude::*;

use crate::types::PLAYER_COUNT;
use crate::{BeliefResidualError, BeliefResidualEvaluator};

use super::{
    belief::{RootBeliefMode, RootBeliefParticle, RootBeliefSampler},
    simulation::RolloutModels,
    *,
};

const SELECTION_WORLD_DOMAIN: u64 = SEARCH_SEED_DOMAIN ^ 0x34f5_172c_80ad_9e61;
const VALIDATION_WORLD_DOMAIN: u64 = SEARCH_SEED_DOMAIN ^ 0xa83d_c047_5be1_26f9;
const MIN_EFFECTIVE_WORLDS: f64 = 8.0;
const VALIDATION_Z_SCORE: f64 = 1.96;

#[derive(Clone, Debug)]
struct EvaluatedWorld {
    log_likelihood: f64,
    utilities: Vec<f64>,
}

#[derive(Clone, Debug)]
struct ValidationWorld {
    log_likelihood: f64,
    differences: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
struct WeightedEstimate {
    mean: f64,
    standard_error: f64,
    effective_worlds: f64,
}

impl WeightedEstimate {
    fn from_log_weighted(samples: &[(f64, f64)]) -> Option<Self> {
        if samples.len() < 2 || samples.iter().any(|(_, value)| !value.is_finite()) {
            return None;
        }
        let weights = normalized_log_weights(samples.iter().map(|(weight, _)| *weight))?;
        let mean = samples
            .iter()
            .zip(&weights)
            .map(|((_, value), weight)| value * weight)
            .sum::<f64>();
        let squared_weight_sum = weights.iter().map(|weight| weight * weight).sum::<f64>();
        if !squared_weight_sum.is_finite() || squared_weight_sum >= 1.0 {
            return None;
        }
        let squared_error = samples
            .iter()
            .zip(&weights)
            .map(|((_, value), weight)| weight * weight * (value - mean).powi(2))
            .sum::<f64>()
            / (1.0 - squared_weight_sum);
        Some(Self {
            mean,
            standard_error: squared_error.max(0.0).sqrt(),
            effective_worlds: squared_weight_sum.recip(),
        })
    }

    fn lower_confidence_bound(self) -> f64 {
        self.mean - VALIDATION_Z_SCORE * self.standard_error
    }
}

pub(super) fn paired_policy_improvement(
    game: &Game,
    legal: &LegalActions,
    config: RulePlannerConfig,
) -> Option<ActionId> {
    paired_policy_improvement_analysis_inner(
        game,
        legal,
        config,
        RootBeliefMode::Posterior,
        RolloutModels::Current,
        None,
    )
    .expect("root search without a learned evaluator cannot fail")
    .map(RulePlannerAnalysis::action)
}

pub(super) fn paired_policy_improvement_analysis_with_residual(
    game: &Game,
    legal: &LegalActions,
    config: RulePlannerConfig,
    evaluator: &dyn BeliefResidualEvaluator,
) -> Result<Option<RulePlannerAnalysis>, BeliefResidualError> {
    paired_policy_improvement_analysis_inner(
        game,
        legal,
        config,
        RootBeliefMode::Posterior,
        RolloutModels::Current,
        Some(evaluator),
    )
}

#[cfg(feature = "planner-analysis")]
pub(super) fn paired_policy_improvement_analysis(
    game: &Game,
    legal: &LegalActions,
    config: RulePlannerConfig,
    root_belief: RootBeliefMode,
    rollout_models: RolloutModels,
) -> Option<RulePlannerAnalysis> {
    paired_policy_improvement_analysis_inner(game, legal, config, root_belief, rollout_models, None)
        .expect("planner analysis without a learned evaluator cannot fail")
}

fn paired_policy_improvement_analysis_inner(
    game: &Game,
    legal: &LegalActions,
    config: RulePlannerConfig,
    root_belief: RootBeliefMode,
    rollout_models: RolloutModels,
    evaluator: Option<&dyn BeliefResidualEvaluator>,
) -> Result<Option<RulePlannerAnalysis>, BeliefResidualError> {
    let baseline = planner_action_without_search(game, legal, config.without_search());
    let Some(legal_mask) = game.legal_action_mask() else {
        return Ok(None);
    };
    let mut actions: Vec<_> = legal_mask.iter().collect();
    if actions.len() <= 1 {
        return Ok(actions
            .into_iter()
            .next()
            .map(RulePlannerAnalysis::without_search));
    }
    let Some(baseline_index) = actions.iter().position(|&action| action == baseline) else {
        return Ok(None);
    };
    actions.swap(0, baseline_index);

    let world_count = usize::from(config.search_iterations());
    if world_count < MIN_EFFECTIVE_WORLDS as usize {
        return Ok(Some(RulePlannerAnalysis::without_search(baseline)));
    }

    let actor = legal.decision.actor;
    let base_seed = public_state_hash(game, actor);
    let Some(selection_sampler) =
        RootBeliefSampler::new(game, actor, base_seed ^ SELECTION_WORLD_DOMAIN)
    else {
        return Ok(None);
    };
    let mut selection_particles = selection_sampler.sample_batch(world_count, root_belief);
    let mut belief_residual_fallback = None;
    let learned_validation_particles = if let Some(evaluator) = evaluator {
        let Some(validation_sampler) =
            RootBeliefSampler::new(game, actor, base_seed ^ VALIDATION_WORLD_DOMAIN)
        else {
            return Ok(None);
        };
        let mut validation_particles = validation_sampler.sample_batch(world_count, root_belief);
        let mut batches = [
            selection_particles.as_mut_slice(),
            validation_particles.as_mut_slice(),
        ];
        selection_sampler.apply_residuals(evaluator, &mut batches)?;
        belief_residual_fallback = Some(apply_joint_weight_fallback(
            &mut selection_particles,
            &mut validation_particles,
        ));
        Some(validation_particles)
    } else {
        None
    };
    SEARCH_DECISIONS.fetch_add(1, AtomicOrdering::Relaxed);
    let selection = evaluate_selection(selection_particles, actor, &actions, rollout_models);
    let model_count = rollout_models.len();
    let mut rollouts = (selection.len() * actions.len() * model_count) as u64;
    SEARCH_ROLLOUTS.fetch_add(rollouts, AtomicOrdering::Relaxed);

    let candidate_index = select_candidate(&selection, actions.len());
    let Some(candidate_index) = candidate_index.filter(|&index| index != 0) else {
        return Ok(Some(search_analysis(
            baseline,
            RulePlannerSearchOutcome::NoProposal,
            rollouts,
            belief_residual_fallback,
        )));
    };
    let candidate = actions[candidate_index];
    SEARCH_PROPOSALS.fetch_add(1, AtomicOrdering::Relaxed);

    let validation_particles = match learned_validation_particles {
        Some(particles) => particles,
        None => {
            let Some(validation_sampler) =
                RootBeliefSampler::new(game, actor, base_seed ^ VALIDATION_WORLD_DOMAIN)
            else {
                return Ok(None);
            };
            validation_sampler.sample_batch(world_count, root_belief)
        }
    };
    let validation = evaluate_validation(
        validation_particles,
        actor,
        candidate,
        baseline,
        rollout_models,
    );
    let validation_rollouts = (validation.len() * 2 * model_count) as u64;
    rollouts += validation_rollouts;
    SEARCH_ROLLOUTS.fetch_add(validation_rollouts, AtomicOrdering::Relaxed);

    let outcome = if validate_candidate(&validation) {
        SEARCH_OVERRIDES.fetch_add(1, AtomicOrdering::Relaxed);
        RulePlannerSearchOutcome::Accepted(candidate)
    } else {
        SEARCH_VALIDATION_REJECTIONS.fetch_add(1, AtomicOrdering::Relaxed);
        RulePlannerSearchOutcome::Rejected(candidate)
    };
    Ok(Some(search_analysis(
        baseline,
        outcome,
        rollouts,
        belief_residual_fallback,
    )))
}

fn search_analysis(
    baseline: ActionId,
    outcome: RulePlannerSearchOutcome,
    rollouts: u64,
    belief_residual_fallback: Option<bool>,
) -> RulePlannerAnalysis {
    let search = RulePlannerSearchAnalysis::new(baseline, outcome, rollouts);
    let search = match belief_residual_fallback {
        Some(fallback) => search.with_belief_residual(fallback),
        None => search,
    };
    RulePlannerAnalysis::from_search(search)
}

fn evaluate_selection(
    particles: Vec<RootBeliefParticle>,
    actor: Seat,
    actions: &[ActionId],
    rollout_models: RolloutModels,
) -> Vec<EvaluatedWorld> {
    let model_count = rollout_models.len();
    let row_width = actions.len() * model_count;
    let tasks = particles.len() * row_width;
    let utilities: Vec<_> = (0..tasks)
        .into_par_iter()
        .map(|task| {
            let world_index = task / row_width;
            let within_world = task % row_width;
            let action_index = within_world / model_count;
            let model_index = within_world % model_count;
            terminal_rollout(
                &particles[world_index].game,
                actor,
                actions[action_index],
                rollout_models,
                model_index,
            )
        })
        .collect();

    particles
        .into_iter()
        .zip(utilities.chunks_exact(row_width))
        .filter_map(|(particle, row)| {
            let utilities = row.iter().copied().collect::<Option<Vec<_>>>()?;
            Some(EvaluatedWorld {
                log_likelihood: particle.log_likelihood,
                utilities,
            })
        })
        .collect()
}

fn evaluate_validation(
    particles: Vec<RootBeliefParticle>,
    actor: Seat,
    candidate: ActionId,
    baseline: ActionId,
    rollout_models: RolloutModels,
) -> Vec<ValidationWorld> {
    let model_count = rollout_models.len();
    let row_width = 2 * model_count;
    let tasks = particles.len() * row_width;
    let utilities: Vec<_> = (0..tasks)
        .into_par_iter()
        .map(|task| {
            let world_index = task / row_width;
            let within_world = task % row_width;
            let action = if within_world / model_count == 0 {
                candidate
            } else {
                baseline
            };
            let model_index = within_world % model_count;
            terminal_rollout(
                &particles[world_index].game,
                actor,
                action,
                rollout_models,
                model_index,
            )
        })
        .collect();

    particles
        .into_iter()
        .zip(utilities.chunks_exact(row_width))
        .filter_map(|(particle, row)| {
            let row = row.iter().copied().collect::<Option<Vec<_>>>()?;
            Some(ValidationWorld {
                log_likelihood: particle.log_likelihood,
                differences: (0..model_count)
                    .map(|model| row[model] - row[model_count + model])
                    .collect(),
            })
        })
        .collect()
}

fn joint_weights_have_support(
    selection: &[RootBeliefParticle],
    validation: &[RootBeliefParticle],
) -> bool {
    [selection, validation].into_iter().all(|particles| {
        effective_worlds(particles.iter().map(|particle| particle.log_likelihood))
            .is_some_and(|count| count >= MIN_EFFECTIVE_WORLDS)
    })
}

/// Revert both particle streams when either stream has insufficient posterior
/// support. Selection and validation must use the same belief distribution.
fn apply_joint_weight_fallback(
    selection: &mut [RootBeliefParticle],
    validation: &mut [RootBeliefParticle],
) -> bool {
    if joint_weights_have_support(selection, validation) {
        return false;
    }
    selection
        .iter_mut()
        .for_each(RootBeliefParticle::use_handwritten_weight);
    validation
        .iter_mut()
        .for_each(RootBeliefParticle::use_handwritten_weight);
    true
}

fn effective_worlds(log_likelihoods: impl Iterator<Item = f64>) -> Option<f64> {
    let log_likelihoods: Vec<_> = log_likelihoods.collect();
    if !log_likelihoods.iter().any(|weight| weight.is_finite()) {
        return None;
    }
    let weights = normalized_log_weights(log_likelihoods.into_iter())?;
    Some(effective_worlds_from_normalized(&weights))
}

fn effective_worlds_from_normalized(weights: &[f64]) -> f64 {
    let squared_weight_sum = weights.iter().map(|weight| weight * weight).sum::<f64>();
    if !squared_weight_sum.is_finite() || squared_weight_sum <= 0.0 {
        return 0.0;
    }
    squared_weight_sum.recip()
}

fn select_candidate(worlds: &[EvaluatedWorld], action_count: usize) -> Option<usize> {
    let utility_count = worlds.first()?.utilities.len();
    let model_count = utility_count.checked_div(action_count)?;
    if action_count == 0
        || model_count == 0
        || utility_count != action_count * model_count
        || worlds.len() < 2
        || worlds.iter().any(|world| {
            world.utilities.len() != utility_count
                || world.utilities.iter().any(|value| !value.is_finite())
        })
    {
        return None;
    }
    let weights = normalized_log_weights(worlds.iter().map(|world| world.log_likelihood))?;
    let effective_worlds = effective_worlds_from_normalized(&weights);
    if effective_worlds < MIN_EFFECTIVE_WORLDS {
        return None;
    }

    let mut best = 0;
    let mut best_worst_gain = 0.0;
    for action in 1..action_count {
        let worst_gain = (0..model_count)
            .map(|model| {
                worlds
                    .iter()
                    .zip(&weights)
                    .map(|(world, weight)| {
                        weight
                            * (world.utilities[action * model_count + model]
                                - world.utilities[model])
                    })
                    .sum::<f64>()
            })
            .min_by(f64::total_cmp)
            .expect("there is a simulation policy");
        if worst_gain > best_worst_gain {
            best = action;
            best_worst_gain = worst_gain;
        }
    }
    Some(best)
}

fn validate_candidate(worlds: &[ValidationWorld]) -> bool {
    let Some(model_count) = worlds.first().map(|world| world.differences.len()) else {
        return false;
    };
    if model_count == 0
        || worlds.len() < 2
        || worlds
            .iter()
            .any(|world| world.differences.len() != model_count)
    {
        return false;
    }
    for model in 0..model_count {
        let samples: Vec<_> = worlds
            .iter()
            .map(|world| (world.log_likelihood, world.differences[model]))
            .collect();
        let Some(estimate) = WeightedEstimate::from_log_weighted(&samples) else {
            return false;
        };
        if estimate.effective_worlds < MIN_EFFECTIVE_WORLDS || estimate.mean <= 0.0 {
            return false;
        }
    }

    let pooled: Vec<_> = worlds
        .iter()
        .map(|world| {
            (
                world.log_likelihood,
                world.differences.iter().sum::<f64>() / model_count as f64,
            )
        })
        .collect();
    WeightedEstimate::from_log_weighted(&pooled).is_some_and(|estimate| {
        estimate.effective_worlds >= MIN_EFFECTIVE_WORLDS && estimate.lower_confidence_bound() > 0.0
    })
}

fn normalized_log_weights(log_weights: impl Iterator<Item = f64>) -> Option<Vec<f64>> {
    let log_weights: Vec<_> = log_weights.collect();
    if log_weights.is_empty() {
        return None;
    }
    let maximum = log_weights
        .iter()
        .copied()
        .filter(|weight| weight.is_finite())
        .max_by(f64::total_cmp);
    let Some(maximum) = maximum else {
        // Preserve the planner's established emergency behavior when the
        // proposal misses all posterior support. `effective_worlds` detects
        // this case before normalization, so learned-weight ESS cannot mistake
        // the uniform fallback for valid support.
        return Some(vec![1.0 / log_weights.len() as f64; log_weights.len()]);
    };
    let mut weights: Vec<_> = log_weights
        .into_iter()
        .map(|weight| {
            if weight.is_finite() {
                (weight - maximum).exp()
            } else {
                0.0
            }
        })
        .collect();
    let total = weights.iter().sum::<f64>();
    if !total.is_finite() || total <= f64::EPSILON {
        return None;
    }
    for weight in &mut weights {
        *weight /= total;
    }
    Some(weights)
}

fn terminal_rollout(
    world: &Game,
    actor: Seat,
    root_action: ActionId,
    rollout_models: RolloutModels,
    model_index: usize,
) -> Option<f64> {
    let mut game = world.clone();
    game.step_id(root_action).ok()?;
    for _ in 0..MAX_ROLLOUT_ACTIONS {
        let Some(legal) = game.legal_actions() else {
            return Some(terminal_utility(&game, actor) as f64);
        };
        let action = rollout_models.action(model_index, &game, &legal, actor);
        game.step_id(action).ok()?;
    }
    None
}

/// Encodes the tournament objective lexicographically: rank first, score
/// second. The total score plus one is a sufficient stride because engine
/// transfers conserve a non-negative score pool.
fn terminal_utility(game: &Game, actor: Seat) -> i64 {
    let rank = game
        .rankings()
        .iter()
        .position(|&seat| seat == actor)
        .expect("a terminal player has a rank");
    let total_score = Seat::ALL
        .into_iter()
        .map(|seat| game.score(seat))
        .sum::<i64>();
    let rank_priority = (PLAYER_COUNT - 1 - rank) as i64;
    rank_priority * (total_score + 1) + game.score(actor)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ZeroResidual;

    impl BeliefResidualEvaluator for ZeroResidual {
        fn evaluate_residuals(
            &self,
            _public: &crate::BeliefPublicFeatures,
            candidates: &[crate::BeliefCandidateFeatures],
            output: &mut [f32],
        ) -> Result<(), BeliefResidualError> {
            if candidates.len() != output.len() {
                return Err(BeliefResidualError::BatchLength {
                    candidates: candidates.len(),
                    outputs: output.len(),
                });
            }
            output.fill(0.0);
            Ok(())
        }
    }

    fn late_turn(seed: u64, remaining: usize) -> Game {
        let mut game = Game::new(seed);
        for _ in 0..MAX_ROLLOUT_ACTIONS {
            let legal = game.legal_actions().expect("setup game is non-terminal");
            if game.wall_remaining() <= remaining
                && legal.decision.phase == Phase::Turn
                && !legal.can_hu
            {
                return game;
            }
            let action = game
                .simple_rule_action()
                .expect("the setup policy handles every non-terminal phase");
            game.step_id(action).expect("the setup action is legal");
        }
        panic!("setup did not reach a late turn")
    }

    #[cfg(feature = "planner-analysis")]
    fn late_searchable_turn(seed: u64, remaining: usize) -> Game {
        let mut game = Game::new(seed);
        for _ in 0..MAX_ROLLOUT_ACTIONS {
            let legal = game.legal_actions().expect("setup game is non-terminal");
            if game.wall_remaining() <= remaining
                && legal.decision.phase == Phase::Turn
                && !legal.can_hu
                && game
                    .legal_action_mask()
                    .is_some_and(|mask| mask.iter().count() > 1)
            {
                return game;
            }
            let action = game
                .simple_rule_action()
                .expect("the setup policy handles every non-terminal phase");
            game.step_id(action).expect("the setup action is legal");
        }
        panic!("setup did not reach a searchable late turn")
    }

    fn synthetic_world(offset: f64, gains: [[f64; 2]; 2]) -> EvaluatedWorld {
        EvaluatedWorld {
            log_likelihood: 0.0,
            utilities: vec![offset, offset, offset + gains[1][0], offset + gains[1][1]],
        }
    }

    fn belief_particles(log_likelihoods: &[f64]) -> Vec<RootBeliefParticle> {
        log_likelihoods
            .iter()
            .enumerate()
            .map(|(index, &log_likelihood)| {
                RootBeliefParticle::new_for_test(
                    Game::new(0x4d58_0000 + index as u64),
                    log_likelihood,
                    -10.0,
                )
            })
            .collect()
    }

    fn concentrated_log_weights(count: usize) -> Vec<f64> {
        (0..count)
            .map(|index| if index == 0 { 100.0 } else { 0.0 })
            .collect()
    }

    #[test]
    fn joint_belief_keeps_learned_weights_when_both_streams_have_support() {
        let learned: Vec<_> = (0..16).map(|index| -(index as f64) * 0.01).collect();
        let mut selection = belief_particles(&learned);
        let mut validation = belief_particles(&learned);

        assert!(joint_weights_have_support(&selection, &validation));
        assert!(!apply_joint_weight_fallback(
            &mut selection,
            &mut validation
        ));
        assert_eq!(
            selection
                .iter()
                .map(|particle| particle.log_likelihood)
                .collect::<Vec<_>>(),
            learned,
        );
        assert_eq!(
            validation
                .iter()
                .map(|particle| particle.log_likelihood)
                .collect::<Vec<_>>(),
            learned,
        );
        assert_ne!(learned, vec![-10.0; 16]);
    }

    #[test]
    fn joint_belief_falls_back_both_streams_when_either_ess_is_low() {
        let uniform = vec![0.0; 16];
        for (selection_logs, validation_logs) in [
            (concentrated_log_weights(16), uniform.clone()),
            (uniform.clone(), concentrated_log_weights(16)),
        ] {
            let mut selection = belief_particles(&selection_logs);
            let mut validation = belief_particles(&validation_logs);

            assert!(!joint_weights_have_support(&selection, &validation));
            assert!(apply_joint_weight_fallback(&mut selection, &mut validation));
            assert_eq!(
                selection
                    .iter()
                    .map(|particle| particle.log_likelihood)
                    .collect::<Vec<_>>(),
                vec![-10.0; 16],
            );
            assert_eq!(
                validation
                    .iter()
                    .map(|particle| particle.log_likelihood)
                    .collect::<Vec<_>>(),
                vec![-10.0; 16],
            );
        }
    }

    #[test]
    fn log_weights_are_normalized_without_overflow() {
        let weights = normalized_log_weights([10_000.0, 9_999.0].into_iter())
            .expect("finite weights normalize");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights[0] > weights[1]);

        let fallback = normalized_log_weights([f64::NEG_INFINITY; 8].into_iter())
            .expect("zero posterior mass keeps the planner emergency fallback");
        assert_eq!(fallback, vec![0.125; 8]);
        assert!(effective_worlds([f64::NEG_INFINITY; 8].into_iter()).is_none());
    }

    #[test]
    fn weighted_confidence_bound_rejects_noise_and_accepts_gain() {
        let noisy: Vec<_> = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
            .into_iter()
            .map(|value| (0.0, value))
            .collect();
        let gain: Vec<_> = [2.0, 3.0, 2.0, 3.0, 2.0, 3.0, 2.0, 3.0]
            .into_iter()
            .map(|value| (0.0, value))
            .collect();
        assert!(
            WeightedEstimate::from_log_weighted(&noisy)
                .expect("eight samples have an estimate")
                .lower_confidence_bound()
                < 0.0
        );
        assert!(
            WeightedEstimate::from_log_weighted(&gain)
                .expect("eight samples have an estimate")
                .lower_confidence_bound()
                > 0.0
        );
    }

    #[test]
    fn candidate_selection_rejects_rollout_model_exploitation() {
        let robust: Vec<_> = (0..8)
            .map(|index| {
                synthetic_world(
                    if index % 2 == 0 {
                        1_000_000.0
                    } else {
                        -1_000_000.0
                    },
                    [[0.0; 2], [2.0, 1.0]],
                )
            })
            .collect();
        assert_eq!(select_candidate(&robust, 2), Some(1));

        let disagreement: Vec<_> = (0..8)
            .map(|_| synthetic_world(0.0, [[0.0; 2], [2.0, -1.0]]))
            .collect();
        assert_eq!(select_candidate(&disagreement, 2), Some(0));
    }

    #[test]
    fn candidate_selection_supports_one_continuation_model() {
        let worlds = vec![
            EvaluatedWorld {
                log_likelihood: 0.0,
                utilities: vec![0.0, 1.0],
            };
            8
        ];
        assert_eq!(select_candidate(&worlds, 2), Some(1));
    }

    #[test]
    fn validation_requires_both_rollout_models_to_improve() {
        let agreement = vec![
            ValidationWorld {
                log_likelihood: 0.0,
                differences: vec![2.0, 1.0],
            };
            8
        ];
        assert!(validate_candidate(&agreement));

        let disagreement = vec![
            ValidationWorld {
                log_likelihood: 0.0,
                differences: vec![2.0, -1.0],
            };
            8
        ];
        assert!(!validate_candidate(&disagreement));

        let one_model = vec![
            ValidationWorld {
                log_likelihood: 0.0,
                differences: vec![1.0],
            };
            8
        ];
        assert!(validate_candidate(&one_model));
    }

    #[test]
    fn terminal_utility_orders_players_by_rank() {
        let mut game = Game::new(37);
        while let Some(action) = game.simple_rule_action() {
            game.step_id(action)
                .expect("the simple policy action is legal");
        }
        let utilities: Vec<_> = game
            .rankings()
            .iter()
            .map(|&seat| terminal_utility(&game, seat))
            .collect();
        assert!(utilities.windows(2).all(|pair| pair[0] > pair[1]));
    }

    #[test]
    fn root_search_is_deterministic_across_thread_counts() {
        let game = late_turn(127, 2);
        let config = RulePlannerConfig::ROLLOUT
            .with_search_iterations(8)
            .expect("test search budget is supported");
        let action = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test thread pool builds")
                .install(|| game.rule_planner_action_with_config(config))
        };
        assert_eq!(action(1), action(2));
    }

    #[cfg(feature = "planner-analysis")]
    #[test]
    fn known_continuation_search_is_deterministic_across_thread_counts() {
        let game = late_turn(128, 2);
        let config = RulePlannerConfig::ROLLOUT
            .with_search_iterations(8)
            .expect("test search budget is supported");
        let profile = RulePlannerContinuationProfile::new(
            [RulePlannerContinuationPolicy::Fast; PLAYER_COUNT],
        );
        let options = RulePlannerAnalysisOptions::default()
            .with_continuation(RulePlannerContinuation::KnownPolicies(profile));
        let action = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test thread pool builds")
                .install(|| game.rule_planner_analysis_with_options(config, options))
        };
        assert_eq!(action(1), action(2));
    }

    #[test]
    fn root_search_returns_a_legal_action() {
        let game = late_turn(91, 2);
        let config = RulePlannerConfig::ROLLOUT
            .with_search_iterations(RULE_PLANNER_MIN_BELIEF_RESIDUAL_SEARCH_ITERATIONS)
            .expect("test search budget is supported");
        let action = game
            .rule_planner_action_with_config(config)
            .expect("a turn has a planner action");
        assert!(game.is_legal_action(action.action()));
    }

    #[test]
    fn zero_residual_preserves_the_handwritten_search_decision() {
        let game = late_turn(91, 2);
        let config = RulePlannerConfig::ROLLOUT
            .with_search_iterations(RULE_PLANNER_MIN_BELIEF_RESIDUAL_SEARCH_ITERATIONS)
            .expect("test search budget is supported");

        assert_eq!(
            game.rule_planner_action_with_config_and_belief_residual(config, &ZeroResidual)
                .expect("zero residual evaluation succeeds"),
            game.rule_planner_action_with_config(config),
        );
    }

    #[cfg(feature = "planner-analysis")]
    #[test]
    fn posterior_analysis_matches_production_search() {
        let game = late_turn(131, 2);
        let config = RulePlannerConfig::ROLLOUT
            .with_search_iterations(8)
            .expect("test search budget is supported");

        let analysis = game
            .rule_planner_analysis_with_config(config, RulePlannerRootBelief::Posterior)
            .expect("analysis returns a decision");
        let options_analysis = game
            .rule_planner_analysis_with_options(config, RulePlannerAnalysisOptions::default())
            .expect("default analysis options return a decision");
        assert_eq!(options_analysis, analysis);
        assert_eq!(
            Some(analysis.action()),
            game.rule_planner_action_with_config(config),
        );
        assert_eq!(
            game.rule_planner_analysis_action_with_config(
                config,
                RulePlannerRootBelief::Posterior,
            ),
            Some(analysis.action()),
        );
    }

    #[test]
    fn structured_search_outcomes_determine_consistent_actions() {
        let baseline = ActionId::PASS;
        let proposal = ActionId::HU;
        let cases = [
            (RulePlannerSearchOutcome::NoProposal, baseline, None, false),
            (
                RulePlannerSearchOutcome::Rejected(proposal),
                baseline,
                Some(proposal),
                false,
            ),
            (
                RulePlannerSearchOutcome::Accepted(proposal),
                proposal,
                Some(proposal),
                true,
            ),
        ];

        for (outcome, expected_action, expected_proposal, expected_accepted) in cases {
            let search = RulePlannerSearchAnalysis::new(baseline, outcome, 17);
            let analysis = RulePlannerAnalysis::from_search(search);
            assert_eq!(analysis.action(), expected_action);
            assert_eq!(analysis.search(), Some(search));
            assert_eq!(search.baseline(), baseline);
            assert_eq!(search.outcome().proposal(), expected_proposal);
            assert_eq!(search.outcome().accepted(), expected_accepted);
            assert_eq!(search.rollouts(), 17);
            assert_eq!(search.belief_residual_fallback(), None);
        }

        let residual =
            RulePlannerSearchAnalysis::new(baseline, RulePlannerSearchOutcome::NoProposal, 19)
                .with_belief_residual(true);
        assert_eq!(residual.belief_residual_fallback(), Some(true));
    }

    #[cfg(feature = "planner-analysis")]
    #[test]
    fn learned_residual_returns_structured_search_metadata() {
        let game = late_searchable_turn(132, 12);
        let config = RulePlannerConfig::ROLLOUT
            .with_search_iterations(RULE_PLANNER_MIN_BELIEF_RESIDUAL_SEARCH_ITERATIONS)
            .expect("minimum residual search budget is supported");

        let analysis = game
            .rule_planner_analysis_with_config_and_belief_residual(config, &ZeroResidual)
            .expect("zero residual evaluation succeeds")
            .expect("a turn has a planner analysis");
        let search = analysis.search().expect("a multi-action turn runs search");
        assert!(search.belief_residual_fallback().is_some());
        assert_eq!(
            Some(analysis.action()),
            game.rule_planner_action_with_config_and_belief_residual(config, &ZeroResidual)
                .expect("the action wrapper succeeds"),
        );
    }

    #[cfg(feature = "planner-analysis")]
    #[test]
    fn root_belief_ablations_return_legal_actions() {
        let game = late_turn(137, 2);
        let config = RulePlannerConfig::ROLLOUT
            .with_search_iterations(8)
            .expect("test search budget is supported");
        let legal = game
            .legal_action_mask()
            .expect("test state has legal actions");

        for belief in [
            RulePlannerRootBelief::Uniform,
            RulePlannerRootBelief::OracleHidden,
        ] {
            let analysis = game
                .rule_planner_analysis_with_config(config, belief)
                .expect("analysis policy returns an action");
            assert!(legal.contains(analysis.action()));
        }
    }

    #[cfg(feature = "planner-analysis")]
    #[test]
    fn known_continuation_ablations_return_legal_actions() {
        let game = late_turn(138, 2);
        let config = RulePlannerConfig::ROLLOUT
            .with_search_iterations(8)
            .expect("test search budget is supported");
        let profile = RulePlannerContinuationProfile::new([
            RulePlannerContinuationPolicy::Fast,
            RulePlannerContinuationPolicy::Ev(crate::RuleEvConfig::FAST),
            RulePlannerContinuationPolicy::PlannerBaseline(RulePlannerConfig::ROLLOUT),
            RulePlannerContinuationPolicy::Fast,
        ]);
        let legal = game
            .legal_action_mask()
            .expect("test state has legal actions");

        for belief in [
            RulePlannerRootBelief::Posterior,
            RulePlannerRootBelief::OracleHidden,
        ] {
            let options = RulePlannerAnalysisOptions::new(belief)
                .with_continuation(RulePlannerContinuation::KnownPolicies(profile));
            let analysis = game
                .rule_planner_analysis_with_options(config, options)
                .expect("known continuation returns an action");
            assert!(legal.contains(analysis.action()));
            assert_eq!(
                game.rule_planner_analysis_action_with_options(config, options),
                Some(analysis.action()),
            );
        }
    }

    #[cfg(feature = "planner-analysis")]
    #[test]
    fn continuation_model_count_controls_the_flat_utility_shape() {
        let game = late_turn(140, 2);
        let actor = game.decision().expect("test game has a decision").actor;
        let actions: Vec<_> = game
            .legal_action_mask()
            .expect("test game has legal actions")
            .iter()
            .collect();
        let sampler = RootBeliefSampler::new(&game, actor, 0x198a_4cc2)
            .expect("test game supports root particles");
        let profile = RulePlannerContinuationProfile::new(
            [RulePlannerContinuationPolicy::Fast; PLAYER_COUNT],
        );

        let current_particles = sampler.sample_batch(8, RootBeliefMode::Posterior);
        let known_particles = sampler.sample_batch(8, RootBeliefMode::Posterior);
        let current =
            evaluate_selection(current_particles, actor, &actions, RolloutModels::Current);
        let known = evaluate_selection(
            known_particles,
            actor,
            &actions,
            RolloutModels::KnownPolicies(profile),
        );

        assert_eq!(current.len(), known.len());
        assert!(
            current
                .iter()
                .all(|world| world.utilities.len() == actions.len() * 2)
        );
        assert!(
            known
                .iter()
                .all(|world| world.utilities.len() == actions.len())
        );
    }

    #[cfg(feature = "planner-analysis")]
    #[test]
    fn oracle_search_ignores_authoritative_future_wall_order() {
        let game = late_turn(139, 12);
        let reordered = game.resample_live_wall(0x59ad_c071);
        let config = RulePlannerConfig::ROLLOUT
            .with_search_iterations(8)
            .expect("test search budget is supported");

        assert_ne!(game, reordered);
        assert_eq!(
            game.rule_planner_analysis_with_config(config, RulePlannerRootBelief::OracleHidden),
            reordered
                .rule_planner_analysis_with_config(config, RulePlannerRootBelief::OracleHidden),
        );
    }
}
