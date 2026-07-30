//! Observation-pure policies used only inside counterfactual search.

use crate::ActionId;
use crate::game::{Game, LegalActions, Phase};
use crate::rules::{hand::Holding, opening};

use super::{RulePlannerConfig, quality::HandPotential, response};
use crate::rules::hand::mask_tiles;

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
}
