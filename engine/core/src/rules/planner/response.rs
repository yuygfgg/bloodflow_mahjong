use crate::ActionId;
use crate::game::{Game, LegalActions, Phase};
use crate::rules::hand::{Holding, mask_tiles};
use crate::types::{PLAYER_COUNT, Seat};

use super::{
    MAX_ROLLOUT_ACTIONS, RulePlannerConfig, ValuedAction, mix_search_seed, plan_turn,
    public_state_hash, quality::HandPotential,
};

const RESPONSE_WORLD_DOMAIN: u64 = 0x7af1_e608_53cd_9b42;

/// Chooses one response from paired actor-observation-conditioned worlds.
///
/// Each candidate starts from the same sampled worlds. The engine applies the
/// candidate and every remaining response, so multi-win order, insufficient
/// funds, kong payments, and supplement draws use authoritative transitions.
pub(super) fn choose(game: &Game, legal: &LegalActions, config: RulePlannerConfig) -> ActionId {
    debug_assert!(matches!(
        legal.decision.phase,
        Phase::HuResponse | Phase::MeldResponse
    ));
    let fallback = fallback_action(legal);
    if config.response_worlds == 0 {
        return deterministic_response(game, legal).unwrap_or(fallback);
    }
    let Some(actions) = game
        .legal_action_mask()
        .map(|mask| mask.iter().collect::<Vec<_>>())
    else {
        return fallback;
    };
    if actions.len() == 1 {
        return actions[0];
    }

    let actor = legal.decision.actor;
    let world_count = usize::from(config.response_worlds);
    let Some(worlds) = sample_worlds(game, actor, world_count) else {
        return fallback;
    };
    let continuation_config = config
        .with_response_worlds(0)
        .expect("zero nested belief worlds are supported");
    let mut best: Option<ValuedAction> = None;
    for action in actions {
        let Some(total) = worlds.iter().try_fold(0.0, |total, world| {
            response_value(world, actor, action, continuation_config).map(|value| total + value)
        }) else {
            return fallback;
        };
        let candidate = ValuedAction {
            action,
            value: total / worlds.len() as f64,
        };
        match &mut best {
            Some(best) => best.consider(candidate.action, candidate.value),
            None => best = Some(candidate),
        }
    }
    best.map_or(fallback, |candidate| candidate.action)
}

fn deterministic_response(game: &Game, legal: &LegalActions) -> Option<ActionId> {
    match legal.decision.phase {
        Phase::HuResponse => return Some(ActionId::HU),
        Phase::MeldResponse => {}
        Phase::Exchange | Phase::ChooseMissing | Phase::Turn | Phase::Finished => return None,
    }

    let (source, tile) = game.discards().last()?;
    let actor = legal.decision.actor;
    let has_won = game.has_won(actor);
    let visible = game.visible_tile_counts(actor);
    let holding = Holding::from_game(game, actor);
    let pass = HandPotential::evaluate(&holding, &visible, has_won);
    let mut best = (ActionId::PASS, pass);

    if legal.can_exposed_kong
        && let Some(after) = holding.after_exposed_kong(tile, source)
    {
        let potential = HandPotential::evaluate(&after, &visible, has_won);
        if potential.permits_kong_from(pass, has_won) {
            return Some(ActionId::EXPOSED_KONG);
        }
    }

    if legal.can_pong
        && let Some(after_pong) = holding.after_pong(tile, source)
    {
        let pong = mask_tiles(after_pong.discard_mask())
            .filter_map(|discard| after_pong.after_discard(discard))
            .map(|after| HandPotential::evaluate(&after, &visible, has_won))
            .max_by(|left, right| left.cmp_for(*right, has_won));
        if let Some(potential) = pong
            && potential.cmp_for(pass, has_won).is_gt()
            && potential.cmp_for(best.1, has_won).is_gt()
        {
            best = (ActionId::PONG, potential);
        }
    }

    Some(best.0)
}

fn fallback_action(legal: &LegalActions) -> ActionId {
    match legal.decision.phase {
        Phase::HuResponse => ActionId::HU,
        Phase::MeldResponse if legal.can_exposed_kong => ActionId::EXPOSED_KONG,
        Phase::MeldResponse if legal.can_pong => ActionId::PONG,
        Phase::MeldResponse => ActionId::PASS,
        Phase::Exchange | Phase::ChooseMissing | Phase::Turn | Phase::Finished => {
            unreachable!("response fallback requires a response decision")
        }
    }
}

fn sample_worlds(game: &Game, actor: Seat, count: usize) -> Option<Vec<Game>> {
    let base = public_state_hash(game, actor) ^ RESPONSE_WORLD_DOMAIN;
    let expected_observation = public_state_hash(game, actor);
    let expected_legal = game.legal_action_mask()?;
    (0..count)
        .map(|index| {
            let sampled = game
                .resample_current_actor_response_information_set(mix_search_seed(
                    base.wrapping_add(index as u64),
                ))
                .ok()?;
            (sampled
                .decision()
                .is_some_and(|decision| decision.actor == actor)
                && sampled.legal_action_mask() == Some(expected_legal)
                && public_state_hash(&sampled, actor) == expected_observation)
                .then_some(sampled)
        })
        .collect()
}

fn response_value(
    world: &Game,
    actor: Seat,
    action: ActionId,
    config: RulePlannerConfig,
) -> Option<f64> {
    let initial_score = world.score(actor);
    let resolved = resolve_window(world, action)?;
    let continuation = advance_to_actor_turn(resolved, actor)?;
    let immediate = (continuation.score(actor) - initial_score) as f64;
    let Some(legal) = continuation.legal_actions() else {
        return Some(immediate);
    };
    debug_assert_eq!(legal.decision.actor, actor);
    debug_assert_eq!(legal.decision.phase, Phase::Turn);
    Some(immediate + plan_turn(&continuation, &legal, config, |_| {}).value)
}

/// Applies the root action and deterministically closes the same response
/// window. A legal Hu is accepted. A legal meld claimant takes Kong before
/// Pong. This policy is only for responders whose hidden choices must be
/// resolved before the candidate reaches a public continuation state.
fn resolve_window(world: &Game, root_action: ActionId) -> Option<Game> {
    let mut resolved = world.clone();
    resolved.step_id(root_action).ok()?;
    for _ in 0..PLAYER_COUNT {
        let phase = resolved.phase();
        if !matches!(phase, Phase::HuResponse | Phase::MeldResponse) {
            return Some(resolved);
        }
        let legal = resolved.legal_actions()?;
        let action = fallback_action(&legal);
        resolved.step_id(action).ok()?;
    }
    (!matches!(resolved.phase(), Phase::HuResponse | Phase::MeldResponse)).then_some(resolved)
}

/// Advances a sampled world to the actor's next turn.
///
/// Other players and intervening actor responses use one symmetric,
/// deterministic policy. The actor's turn is left unresolved and receives the
/// same Bellman continuation used by ordinary planner turns.
fn advance_to_actor_turn(mut game: Game, actor: Seat) -> Option<Game> {
    for _ in 0..MAX_ROLLOUT_ACTIONS {
        let Some(legal) = game.legal_actions() else {
            return Some(game);
        };
        if legal.decision.actor == actor && legal.decision.phase == Phase::Turn {
            return Some(game);
        }
        let action = if matches!(
            legal.decision.phase,
            Phase::HuResponse | Phase::MeldResponse
        ) {
            fallback_action(&legal)
        } else {
            game.simple_rule_action()?
        };
        game.step_id(action).ok()?;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_response(mut predicate: impl FnMut(&Game, &LegalActions) -> bool) -> Game {
        for seed in 0..512 {
            let mut game = Game::new(seed);
            for _ in 0..super::super::MAX_ROLLOUT_ACTIONS {
                let Some(legal) = game.legal_actions() else {
                    break;
                };
                if matches!(
                    legal.decision.phase,
                    Phase::HuResponse | Phase::MeldResponse
                ) && predicate(&game, &legal)
                {
                    return game;
                }
                let action = game
                    .simple_rule_action()
                    .expect("the setup policy handles a non-terminal game");
                game.step_id(action)
                    .expect("the setup policy returns a legal action");
            }
        }
        panic!("seed search did not find the requested response")
    }

    fn test_config() -> RulePlannerConfig {
        RulePlannerConfig::ROLLOUT
            .with_draw_horizon(0)
            .expect("zero draw horizon is supported")
            .with_candidate_states(1)
            .expect("one candidate state is supported")
            .with_response_worlds(2)
            .expect("two response worlds are supported")
    }

    #[test]
    fn sampled_response_choices_are_legal() {
        for phase in [Phase::HuResponse, Phase::MeldResponse] {
            let game = find_response(|_, legal| legal.decision.phase == phase);
            let legal = game.legal_actions().expect("response has legal actions");
            let action = choose(&game, &legal, test_config());
            assert!(game.is_legal_action(action.action()));
        }
    }

    #[test]
    fn every_response_candidate_closes_the_current_window() {
        let hu = find_response(|_, legal| legal.decision.phase == Phase::HuResponse);
        let hu_actions: Vec<_> = hu
            .legal_action_mask()
            .expect("Hu response has an action mask")
            .iter()
            .collect();
        assert!(hu_actions.contains(&ActionId::HU));
        assert!(hu_actions.contains(&ActionId::PASS));
        for action in hu_actions {
            let resolved = resolve_window(&hu, action).expect("a legal Hu response resolves");
            assert!(!matches!(
                resolved.phase(),
                Phase::HuResponse | Phase::MeldResponse
            ));
        }

        let meld =
            find_response(|_, legal| legal.decision.phase == Phase::MeldResponse && legal.can_pong);
        let meld_actor = meld.decision().expect("response has an actor").actor;
        for action in meld
            .legal_action_mask()
            .expect("meld response has an action mask")
            .iter()
        {
            let resolved = resolve_window(&meld, action).expect("a legal meld response resolves");
            assert!(!matches!(
                resolved.phase(),
                Phase::HuResponse | Phase::MeldResponse
            ));
            if action == ActionId::PONG {
                assert_eq!(
                    resolved.decision().map(|decision| decision.actor),
                    Some(meld_actor)
                );
                assert_eq!(resolved.phase(), Phase::Turn);
            }
        }
    }

    #[test]
    fn exposed_kong_candidate_uses_engine_payment_and_supplement_draw() {
        let game = find_response(|_, legal| {
            legal.decision.phase == Phase::MeldResponse && legal.can_exposed_kong
        });
        let actor = game.decision().expect("response has an actor").actor;
        let score_before = game.score(actor);
        let wall_before = game.wall_remaining();
        let resolved = resolve_window(&game, ActionId::EXPOSED_KONG)
            .expect("a legal exposed Kong response resolves");

        assert_eq!(resolved.phase(), Phase::Turn);
        assert_eq!(
            resolved.decision().map(|decision| decision.actor),
            Some(actor)
        );
        assert!(resolved.score(actor) > score_before);
        assert_eq!(resolved.wall_remaining() + 1, wall_before);
        assert!(
            resolved
                .current_draw()
                .is_some_and(|draw| draw.player == actor && draw.replacement)
        );
    }

    #[test]
    fn continuation_does_not_skip_a_non_actor_turn() {
        let game = find_response(|game, legal| {
            if legal.decision.phase != Phase::HuResponse {
                return false;
            }
            let actor = legal.decision.actor;
            resolve_window(game, ActionId::PASS).is_some_and(|resolved| {
                resolved.phase() != Phase::Finished
                    && resolved
                        .decision()
                        .is_some_and(|decision| decision.actor != actor)
            })
        });
        let actor = game.decision().expect("response has an actor").actor;
        let resolved = resolve_window(&game, ActionId::PASS).expect("Pass resolves the window");
        assert!(
            resolved
                .decision()
                .is_some_and(|decision| decision.actor != actor)
        );

        let continuation = advance_to_actor_turn(resolved, actor)
            .expect("the neutral policy reaches an actor turn or terminal state");
        assert!(
            continuation.phase() == Phase::Finished
                || continuation.decision().is_some_and(|decision| {
                    decision.actor == actor && decision.phase == Phase::Turn
                })
        );
    }
}
