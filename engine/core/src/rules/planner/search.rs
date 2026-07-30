//! Paired terminal policy improvement over sampled information sets.
//!
//! Every sampled world evaluates every legal root action under the same set
//! of identity-independent simulation policies. Selection maximizes the
//! worst policy-specific paired gain. An independent particle stream accepts
//! an override only when both simulation policies improve and the pooled
//! paired lower confidence bound is positive.

use rayon::prelude::*;

use crate::types::PLAYER_COUNT;

use super::{belief::RootBeliefSampler, simulation::OpponentModel, *};

const SELECTION_WORLD_DOMAIN: u64 = SEARCH_SEED_DOMAIN ^ 0x34f5_172c_80ad_9e61;
const VALIDATION_WORLD_DOMAIN: u64 = SEARCH_SEED_DOMAIN ^ 0xa83d_c047_5be1_26f9;
const POLICY_COUNT: usize = OpponentModel::ALL.len();
const MIN_EFFECTIVE_WORLDS: f64 = 8.0;
const VALIDATION_Z_SCORE: f64 = 1.96;

#[derive(Clone, Debug)]
struct EvaluatedWorld {
    log_likelihood: f64,
    utilities: Vec<[f64; POLICY_COUNT]>,
}

#[derive(Clone, Copy, Debug)]
struct ValidationWorld {
    log_likelihood: f64,
    differences: [f64; POLICY_COUNT],
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
    let baseline = planner_action_without_search(game, legal, config.without_search());
    let mut actions: Vec<_> = game.legal_action_mask()?.iter().collect();
    if actions.len() <= 1 {
        return actions.into_iter().next();
    }
    let baseline_index = actions.iter().position(|&action| action == baseline)?;
    actions.swap(0, baseline_index);

    let world_count = usize::from(config.search_iterations());
    if world_count < MIN_EFFECTIVE_WORLDS as usize {
        return Some(baseline);
    }

    let actor = legal.decision.actor;
    let base_seed = public_state_hash(game, actor);
    let selection_sampler =
        RootBeliefSampler::new(game, actor, base_seed ^ SELECTION_WORLD_DOMAIN)?;
    SEARCH_DECISIONS.fetch_add(1, AtomicOrdering::Relaxed);
    let selection = evaluate_selection(&selection_sampler, actor, &actions, world_count);
    SEARCH_ROLLOUTS.fetch_add(
        (selection.len() * actions.len() * POLICY_COUNT) as u64,
        AtomicOrdering::Relaxed,
    );

    let Some(candidate_index) = select_candidate(&selection, actions.len()) else {
        return Some(baseline);
    };
    if candidate_index == 0 {
        return Some(baseline);
    }
    let candidate = actions[candidate_index];
    SEARCH_PROPOSALS.fetch_add(1, AtomicOrdering::Relaxed);

    let validation_sampler =
        RootBeliefSampler::new(game, actor, base_seed ^ VALIDATION_WORLD_DOMAIN)?;
    let validation =
        evaluate_validation(&validation_sampler, actor, candidate, baseline, world_count);
    SEARCH_ROLLOUTS.fetch_add(
        (validation.len() * 2 * POLICY_COUNT) as u64,
        AtomicOrdering::Relaxed,
    );

    if validate_candidate(&validation) {
        SEARCH_OVERRIDES.fetch_add(1, AtomicOrdering::Relaxed);
        Some(candidate)
    } else {
        SEARCH_VALIDATION_REJECTIONS.fetch_add(1, AtomicOrdering::Relaxed);
        Some(baseline)
    }
}

fn evaluate_selection(
    sampler: &RootBeliefSampler<'_>,
    actor: Seat,
    actions: &[ActionId],
    world_count: usize,
) -> Vec<EvaluatedWorld> {
    let particles: Vec<_> = (0..world_count)
        .filter_map(|world_id| sampler.sample_weighted(world_id as u64))
        .collect();
    let row_width = actions.len() * POLICY_COUNT;
    let tasks = particles.len() * row_width;
    let utilities: Vec<_> = (0..tasks)
        .into_par_iter()
        .map(|task| {
            let world_index = task / row_width;
            let within_world = task % row_width;
            let action_index = within_world / POLICY_COUNT;
            let policy_index = within_world % POLICY_COUNT;
            terminal_rollout(
                &particles[world_index].game,
                actor,
                actions[action_index],
                OpponentModel::ALL[policy_index],
            )
        })
        .collect();

    particles
        .into_iter()
        .zip(utilities.chunks_exact(row_width))
        .filter_map(|(particle, row)| {
            let row = row.iter().copied().collect::<Option<Vec<_>>>()?;
            let utilities = row
                .chunks_exact(POLICY_COUNT)
                .map(|values| core::array::from_fn(|index| values[index]))
                .collect();
            Some(EvaluatedWorld {
                log_likelihood: particle.log_likelihood,
                utilities,
            })
        })
        .collect()
}

fn evaluate_validation(
    sampler: &RootBeliefSampler<'_>,
    actor: Seat,
    candidate: ActionId,
    baseline: ActionId,
    world_count: usize,
) -> Vec<ValidationWorld> {
    let particles: Vec<_> = (0..world_count)
        .filter_map(|world_id| sampler.sample_weighted(world_id as u64))
        .collect();
    let row_width = 2 * POLICY_COUNT;
    let tasks = particles.len() * row_width;
    let utilities: Vec<_> = (0..tasks)
        .into_par_iter()
        .map(|task| {
            let world_index = task / row_width;
            let within_world = task % row_width;
            let action = if within_world / POLICY_COUNT == 0 {
                candidate
            } else {
                baseline
            };
            let policy = OpponentModel::ALL[within_world % POLICY_COUNT];
            terminal_rollout(&particles[world_index].game, actor, action, policy)
        })
        .collect();

    particles
        .into_iter()
        .zip(utilities.chunks_exact(row_width))
        .filter_map(|(particle, row)| {
            let row = row.iter().copied().collect::<Option<Vec<_>>>()?;
            Some(ValidationWorld {
                log_likelihood: particle.log_likelihood,
                differences: core::array::from_fn(|policy| {
                    row[policy] - row[POLICY_COUNT + policy]
                }),
            })
        })
        .collect()
}

fn select_candidate(worlds: &[EvaluatedWorld], action_count: usize) -> Option<usize> {
    if worlds.len() < 2
        || worlds.iter().any(|world| {
            world.utilities.len() != action_count
                || world
                    .utilities
                    .iter()
                    .flatten()
                    .any(|value| !value.is_finite())
        })
    {
        return None;
    }
    let weights = normalized_log_weights(worlds.iter().map(|world| world.log_likelihood))?;
    let effective_worlds = weights
        .iter()
        .map(|weight| weight * weight)
        .sum::<f64>()
        .recip();
    if effective_worlds < MIN_EFFECTIVE_WORLDS {
        return None;
    }

    let mut best = 0;
    let mut best_worst_gain = 0.0;
    for action in 1..action_count {
        let worst_gain = (0..POLICY_COUNT)
            .map(|policy| {
                worlds
                    .iter()
                    .zip(&weights)
                    .map(|(world, weight)| {
                        weight * (world.utilities[action][policy] - world.utilities[0][policy])
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
    if worlds.len() < 2 {
        return false;
    }
    for policy in 0..POLICY_COUNT {
        let samples: Vec<_> = worlds
            .iter()
            .map(|world| (world.log_likelihood, world.differences[policy]))
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
                world.differences.iter().sum::<f64>() / POLICY_COUNT as f64,
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
    policy: OpponentModel,
) -> Option<f64> {
    let mut game = world.clone();
    game.step_id(root_action).ok()?;
    for _ in 0..MAX_ROLLOUT_ACTIONS {
        let Some(legal) = game.legal_actions() else {
            return Some(terminal_utility(&game, actor) as f64);
        };
        let action = policy.action(&game, &legal, actor);
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

    fn synthetic_world(offset: f64, gains: [[f64; POLICY_COUNT]; 2]) -> EvaluatedWorld {
        EvaluatedWorld {
            log_likelihood: 0.0,
            utilities: vec![
                [offset; POLICY_COUNT],
                core::array::from_fn(|policy| offset + gains[1][policy]),
            ],
        }
    }

    #[test]
    fn log_weights_are_normalized_without_overflow() {
        let weights = normalized_log_weights([10_000.0, 9_999.0].into_iter())
            .expect("finite weights normalize");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights[0] > weights[1]);

        let fallback = normalized_log_weights([f64::NEG_INFINITY; 8].into_iter())
            .expect("zero posterior mass uses maximum entropy");
        assert_eq!(fallback, vec![0.125; 8]);
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
                    [[0.0; POLICY_COUNT], [2.0, 1.0]],
                )
            })
            .collect();
        assert_eq!(select_candidate(&robust, 2), Some(1));

        let disagreement: Vec<_> = (0..8)
            .map(|_| synthetic_world(0.0, [[0.0; POLICY_COUNT], [2.0, -1.0]]))
            .collect();
        assert_eq!(select_candidate(&disagreement, 2), Some(0));
    }

    #[test]
    fn validation_requires_both_rollout_models_to_improve() {
        let agreement = vec![
            ValidationWorld {
                log_likelihood: 0.0,
                differences: [2.0, 1.0],
            };
            8
        ];
        assert!(validate_candidate(&agreement));

        let disagreement = vec![
            ValidationWorld {
                log_likelihood: 0.0,
                differences: [2.0, -1.0],
            };
            8
        ];
        assert!(!validate_candidate(&disagreement));
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

    #[test]
    fn root_search_returns_a_legal_action() {
        let game = late_turn(91, 2);
        let config = RulePlannerConfig::ROLLOUT
            .with_search_iterations(8)
            .expect("test search budget is supported");
        let action = game
            .rule_planner_action_with_config(config)
            .expect("a turn has a planner action");
        assert!(game.is_legal_action(action.action()));
    }
}
