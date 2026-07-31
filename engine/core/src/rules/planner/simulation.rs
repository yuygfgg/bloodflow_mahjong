//! Observation-pure policies used only inside counterfactual search.

use crate::ActionId;
use crate::game::{Game, LegalActions, Phase};
use crate::rules::{hand::Holding, opening};

use super::{RulePlannerConfig, quality::HandPotential, response};
use crate::rules::hand::mask_tiles;

#[cfg(feature = "planner-analysis")]
use super::{
    RulePlannerContinuation, RulePlannerContinuationPolicy, RulePlannerContinuationProfile,
    planner_action_without_search_unobserved,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum OpponentModel {
    Simple,
    Direct,
}

impl OpponentModel {
    pub(super) const ALL: [Self; 2] = [Self::Simple, Self::Direct];

    pub(super) fn action(self, game: &Game, legal: &LegalActions, target: crate::Seat) -> ActionId {
        if legal.decision.actor == target {
            return direct_action(game, legal);
        }
        match self {
            Self::Simple => game
                .simple_rule_action()
                .expect("a non-terminal game has a simple-rule action"),
            Self::Direct => direct_action(game, legal),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RolloutModels {
    Current,
    #[cfg(feature = "planner-analysis")]
    KnownPolicies(RulePlannerContinuationProfile),
}

impl RolloutModels {
    pub(super) const fn len(self) -> usize {
        match self {
            Self::Current => OpponentModel::ALL.len(),
            #[cfg(feature = "planner-analysis")]
            Self::KnownPolicies(_) => 1,
        }
    }

    pub(super) fn action(
        self,
        model_index: usize,
        game: &Game,
        legal: &LegalActions,
        target: crate::Seat,
    ) -> ActionId {
        match self {
            Self::Current => OpponentModel::ALL
                .get(model_index)
                .copied()
                .expect("current rollout model index is valid")
                .action(game, legal, target),
            #[cfg(feature = "planner-analysis")]
            Self::KnownPolicies(profile) => {
                assert_eq!(model_index, 0, "known continuation has one model");
                profile.for_seat(legal.decision.actor).action(game, legal)
            }
        }
    }
}

#[cfg(feature = "planner-analysis")]
impl From<RulePlannerContinuation> for RolloutModels {
    fn from(value: RulePlannerContinuation) -> Self {
        match value {
            RulePlannerContinuation::Current => Self::Current,
            RulePlannerContinuation::KnownPolicies(profile) => Self::KnownPolicies(profile),
        }
    }
}

#[cfg(feature = "planner-analysis")]
impl RulePlannerContinuationPolicy {
    fn action(self, game: &Game, legal: &LegalActions) -> ActionId {
        match self {
            Self::Fast => game
                .simple_rule_action()
                .expect("a non-terminal game has a simple-rule action"),
            Self::Ev(config) => game
                .rule_ev_action_with_config(config)
                .expect("a non-terminal game has a rule-EV action"),
            Self::PlannerBaseline(config) => {
                planner_action_without_search_unobserved(game, legal, config.without_search())
            }
        }
    }
}

fn direct_action(game: &Game, legal: &LegalActions) -> ActionId {
    match legal.decision.phase {
        Phase::Turn => direct_turn_action(game, legal),
        Phase::HuResponse | Phase::MeldResponse => {
            response::choose(game, legal, RulePlannerConfig::ROLLOUT)
        }
        Phase::Exchange => opening::choose_exchange(
            game.concealed(legal.decision.actor),
            game.exchange_selection(legal.decision.actor),
            legal.exchange_mask,
        ),
        Phase::ChooseMissing => opening::choose_missing(game.concealed(legal.decision.actor)),
        Phase::Finished => unreachable!("a legal-action set is non-terminal"),
    }
}

fn direct_turn_action(game: &Game, legal: &LegalActions) -> ActionId {
    if legal.can_hu {
        return ActionId::HU;
    }
    let actor = legal.decision.actor;
    let holding = Holding::from_game(game, actor);
    let visible = game.visible_tile_counts(actor);
    let has_won = game.has_won(actor);
    let mut best_discard = None::<(ActionId, HandPotential)>;
    for tile in mask_tiles(legal.discard_mask) {
        let Some(after) = holding.after_discard(tile) else {
            continue;
        };
        let action = ActionId::discard(tile);
        let potential = HandPotential::evaluate(&after, &visible, has_won);
        if best_discard.is_none_or(|(best_action, best_potential)| {
            potential.cmp_for(best_potential, has_won).is_gt()
                || (potential == best_potential && action < best_action)
        }) {
            best_discard = Some((action, potential));
        }
    }
    let (discard, discard_potential) = best_discard.expect("a turn without Hu has a discard");

    let mut best_kong = None::<(ActionId, HandPotential)>;
    let mut consider_kong = |action, potential: HandPotential| {
        if !potential.permits_kong_from(discard_potential, has_won) {
            return;
        }
        if best_kong.is_none_or(|(best_action, best_potential)| {
            potential.cmp_for(best_potential, has_won).is_gt()
                || (potential == best_potential && action < best_action)
        }) {
            best_kong = Some((action, potential));
        }
    };
    for tile in mask_tiles(legal.concealed_kong_mask) {
        if let Some(after) = holding.after_concealed_kong(tile, actor) {
            consider_kong(
                ActionId::concealed_kong(tile),
                HandPotential::evaluate(&after, &visible, has_won),
            );
        }
    }
    for tile in mask_tiles(legal.added_kong_mask) {
        if let Some(after) = holding.after_added_kong(tile) {
            consider_kong(
                ActionId::added_kong(tile),
                HandPotential::evaluate(&after, &visible, has_won),
            );
        }
    }
    best_kong.map_or(discard, |(action, _)| action)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulation_policies_complete_games_with_legal_actions() {
        for policy in OpponentModel::ALL {
            for seed in 0..4 {
                let mut game = Game::new(seed);
                for _ in 0..super::super::MAX_ROLLOUT_ACTIONS {
                    let Some(legal) = game.legal_actions() else {
                        break;
                    };
                    let action = policy.action(&game, &legal, crate::Seat::EAST);
                    game.step_id(action)
                        .expect("a simulation policy returns a legal action");
                }
                assert_eq!(game.phase(), Phase::Finished);
            }
        }
    }

    #[cfg(feature = "planner-analysis")]
    #[test]
    fn known_policy_profile_completes_games_with_legal_actions() {
        let profile = RulePlannerContinuationProfile::new([
            RulePlannerContinuationPolicy::Fast,
            RulePlannerContinuationPolicy::Ev(crate::RuleEvConfig::FAST),
            RulePlannerContinuationPolicy::PlannerBaseline(RulePlannerConfig::ROLLOUT),
            RulePlannerContinuationPolicy::Fast,
        ]);
        let models = RolloutModels::KnownPolicies(profile);

        for seed in 0..4 {
            let mut game = Game::new(seed);
            for _ in 0..super::super::MAX_ROLLOUT_ACTIONS {
                let Some(legal) = game.legal_actions() else {
                    break;
                };
                let action = models.action(0, &game, &legal, crate::Seat::EAST);
                game.step_id(action)
                    .expect("a known continuation policy returns a legal action");
            }
            assert_eq!(game.phase(), Phase::Finished);
        }
    }

    #[cfg(feature = "planner-analysis")]
    #[test]
    fn known_continuation_policies_are_information_set_invariant() {
        let policies = [
            RulePlannerContinuationPolicy::Fast,
            RulePlannerContinuationPolicy::Ev(crate::RuleEvConfig::FAST),
            RulePlannerContinuationPolicy::PlannerBaseline(RulePlannerConfig::ROLLOUT),
        ];
        let mut game = Game::new(71);

        for step in 0..64_u64 {
            let legal = game
                .legal_actions()
                .expect("setup game remains non-terminal");
            let actor = legal.decision.actor;
            let sampled = game
                .resample_information_set(0x8c71_40d9_u64.wrapping_add(step))
                .expect("an active decision has an information-set sample");
            let sampled_legal = sampled
                .legal_actions()
                .expect("sampled game remains non-terminal");
            assert_eq!(sampled_legal.decision, legal.decision);
            assert_eq!(sampled.legal_action_mask(), game.legal_action_mask());
            assert_eq!(
                super::super::public_state_hash(&sampled, actor),
                super::super::public_state_hash(&game, actor),
            );

            for policy in policies {
                assert_eq!(
                    policy.action(&sampled, &sampled_legal),
                    policy.action(&game, &legal),
                );
            }

            let action = game
                .simple_rule_action()
                .expect("the setup policy handles every active phase");
            game.step_id(action)
                .expect("the setup policy returns a legal action");
        }
    }

    #[cfg(feature = "planner-analysis")]
    #[test]
    fn planner_continuation_uses_the_non_recursive_baseline() {
        let mut game = Game::new(83);
        let legal = loop {
            let legal = game.legal_actions().expect("setup game is non-terminal");
            if legal.decision.phase == Phase::Turn && !legal.can_hu {
                break legal;
            }
            let action = game
                .simple_rule_action()
                .expect("the setup policy handles every active phase");
            game.step_id(action)
                .expect("the setup policy returns a legal action");
        };
        let config = RulePlannerConfig::ROLLOUT
            .with_search_iterations(8)
            .expect("test search budget is supported");
        let policy = RulePlannerContinuationPolicy::PlannerBaseline(config);

        assert_eq!(
            policy.action(&game, &legal),
            super::super::planner_action_without_search(&game, &legal, config.without_search(),),
        );
    }

    #[cfg(feature = "planner-analysis")]
    #[test]
    fn known_continuation_does_not_special_case_the_root_actor() {
        for seed in 0..32 {
            let mut game = Game::new(seed);
            for _ in 0..super::super::MAX_ROLLOUT_ACTIONS {
                let Some(legal) = game.legal_actions() else {
                    break;
                };
                let actor = legal.decision.actor;
                let ev = RulePlannerContinuationPolicy::Ev(crate::RuleEvConfig::FAST)
                    .action(&game, &legal);
                let direct = direct_action(&game, &legal);
                if ev != direct {
                    let mut policies =
                        [RulePlannerContinuationPolicy::Fast; crate::types::PLAYER_COUNT];
                    policies[actor.index()] =
                        RulePlannerContinuationPolicy::Ev(crate::RuleEvConfig::FAST);
                    let models =
                        RolloutModels::KnownPolicies(RulePlannerContinuationProfile::new(policies));
                    assert_eq!(models.action(0, &game, &legal, actor), ev);
                    return;
                }

                let action = game
                    .simple_rule_action()
                    .expect("the setup policy handles every active phase");
                game.step_id(action)
                    .expect("the setup policy returns a legal action");
            }
        }
        panic!("seed search did not find a rule-EV and Direct disagreement");
    }
}
