use crate::game::{Game, SCORE_UNIT};
use crate::types::{PLAYER_COUNT, Seat};

use super::belief::{DiscardHazard, HazardTable, SampledWallOutcome, SampledWin};
use super::state::{PlanningHand, PlanningPublicState};

#[derive(Clone, Copy, Debug)]
enum WallOutcome {
    Flower,
    Ready(f64),
    Won(f64),
    Other,
}

/// Converts planner outcomes into the engine's score-point unit.
///
/// The wall value resolves joint posterior scenarios with the engine's
/// flower-pig and dajiao transfer order. This preserves hidden-hand
/// correlations, balance caps, and receipts which fund later payments.
pub(super) struct PublicValueModel {
    actor: Seat,
    wall_scenarios: Vec<([WallOutcome; PLAYER_COUNT], usize)>,
    wall_samples: usize,
}

impl PublicValueModel {
    pub(super) fn new(game: &Game, actor: Seat, belief: Option<&HazardTable>) -> Self {
        let mut wall_scenarios: Vec<_> = belief
            .into_iter()
            .flat_map(|table| table.wall_scenarios().transitions())
            .map(|scenario| (scenario.outcomes.map(sampled_wall_outcome), scenario.count))
            .collect();
        if wall_scenarios.is_empty() {
            wall_scenarios.push((
                Seat::ALL.map(|seat| {
                    if seat != actor && game.has_won(seat) {
                        WallOutcome::Won(f64::from(game.max_win_multiplier(seat).max(1)))
                    } else {
                        WallOutcome::Other
                    }
                }),
                1,
            ));
        }
        let wall_samples = wall_scenarios.iter().map(|(_, count)| count).sum();
        Self {
            actor,
            wall_scenarios,
            wall_samples,
        }
    }

    pub(super) fn self_draw_transition(
        &self,
        public: PlanningPublicState,
        multiplier: u32,
        shape_multiplier: u32,
    ) -> (PlanningPublicState, f64) {
        self.self_draw_by(public, self.actor, multiplier, shape_multiplier)
    }

    pub(super) fn self_draw_by(
        &self,
        mut public: PlanningPublicState,
        winner: Seat,
        multiplier: u32,
        shape_multiplier: u32,
    ) -> (PlanningPublicState, f64) {
        public.max_win_multipliers[winner.index()] =
            public.max_win_multipliers[winner.index()].max(shape_multiplier);
        let before = public.balances[self.actor.index()];
        let requested = SCORE_UNIT * i64::from(multiplier) * 2;
        for payer in seats_after(winner) {
            transfer(&mut public.balances, payer, winner, requested);
        }
        let gain = public.balances[self.actor.index()] - before;
        (public, gain as f64)
    }

    /// Resolves all winners of one discard in the engine's source-relative
    /// order. The returned delta is measured only from the planner actor's
    /// balance; winners still update their persistent shape multiplier when a
    /// later payment is capped at zero.
    pub(super) fn discard_winners_transition(
        &self,
        mut public: PlanningPublicState,
        source: Seat,
        wins: [Option<SampledWin>; PLAYER_COUNT],
    ) -> (PlanningPublicState, f64) {
        let before = public.balances[self.actor.index()];
        for winner in seats_after(source) {
            let Some(win) = wins[winner.index()] else {
                continue;
            };
            public.max_win_multipliers[winner.index()] =
                public.max_win_multipliers[winner.index()].max(win.shape_multiplier);
            transfer(
                &mut public.balances,
                source,
                winner,
                SCORE_UNIT * i64::from(win.payout_multiplier),
            );
        }
        let gain = public.balances[self.actor.index()] - before;
        (public, gain as f64)
    }

    pub(super) fn concealed_kong_transition(
        &self,
        public: PlanningPublicState,
    ) -> (PlanningPublicState, f64) {
        self.kong_from_all_transition(public, SCORE_UNIT * 2)
    }

    pub(super) fn added_kong_transition(
        &self,
        public: PlanningPublicState,
    ) -> (PlanningPublicState, f64) {
        self.kong_from_all_transition(public, SCORE_UNIT)
    }

    pub(super) fn exposed_kong_transition(
        &self,
        mut public: PlanningPublicState,
        source: Seat,
    ) -> (PlanningPublicState, f64) {
        let before = public.balances[self.actor.index()];
        transfer(&mut public.balances, source, self.actor, SCORE_UNIT);
        let gain = public.balances[self.actor.index()] - before;
        (public.with_supplement_draw(), gain as f64)
    }

    fn kong_from_all_transition(
        &self,
        mut public: PlanningPublicState,
        requested: i64,
    ) -> (PlanningPublicState, f64) {
        let before = public.balances[self.actor.index()];
        for payer in seats_after(self.actor) {
            transfer(&mut public.balances, payer, self.actor, requested);
        }
        let gain = public.balances[self.actor.index()] - before;
        (public.with_supplement_draw(), gain as f64)
    }

    pub(super) fn hazard_transition_value(
        &self,
        public: PlanningPublicState,
        hazard: Option<&DiscardHazard>,
        mut continuation: impl FnMut(PlanningPublicState, bool, Seat) -> f64,
    ) -> f64 {
        let Some(hazard) = hazard else {
            return continuation(public, false, self.actor);
        };
        let samples = hazard.sample_count();
        if samples == 0 {
            return continuation(public, false, self.actor);
        }

        hazard
            .transitions()
            .map(|atom| {
                let next = apply_hazard_atom(public, self.actor, atom, 1);
                let immediate =
                    next.balances[self.actor.index()] - public.balances[self.actor.index()];
                let any_win = atom.wins.iter().any(Option::is_some);
                let next_actor = seats_after(self.actor)
                    .filter(|winner| atom.wins[winner.index()].is_some())
                    .last()
                    .unwrap_or(self.actor)
                    .next();
                atom.count as f64 * (immediate as f64 + continuation(next, any_win, next_actor))
            })
            .sum::<f64>()
            / samples as f64
    }

    pub(super) fn discard_transition_value(
        &self,
        public: PlanningPublicState,
        hazard: Option<&DiscardHazard>,
        payout_scale: u32,
        mut continuation: impl FnMut(PlanningPublicState, Seat) -> f64,
    ) -> f64 {
        let Some(hazard) = hazard else {
            return continuation(public, self.actor.next());
        };
        let samples = hazard.sample_count();
        if samples == 0 {
            return continuation(public, self.actor.next());
        }

        hazard
            .transitions()
            .map(|atom| {
                let next = apply_hazard_atom(public, self.actor, atom, payout_scale);
                let immediate =
                    next.balances[self.actor.index()] - public.balances[self.actor.index()];
                let next_actor = seats_after(self.actor)
                    .filter(|winner| atom.wins[winner.index()].is_some())
                    .last()
                    .unwrap_or(self.actor)
                    .next();
                atom.count as f64 * (immediate as f64 + continuation(next, next_actor))
            })
            .sum::<f64>()
            / samples as f64
    }

    pub(super) fn wall_value(&self, hand: PlanningHand, public: PlanningPublicState) -> f64 {
        let actor_outcome = hand_wall_outcome(hand);
        let mut expected = 0.0;
        for &(mut outcomes, count) in &self.wall_scenarios {
            outcomes[self.actor.index()] = actor_outcome;
            for opponent in Seat::ALL.into_iter().filter(|&seat| seat != self.actor) {
                outcomes[opponent.index()] = with_prior_win(
                    outcomes[opponent.index()],
                    public.max_win_multipliers[opponent.index()],
                );
            }
            expected += count as f64 * settle_wall(public.balances, outcomes, self.actor) as f64;
        }
        expected / self.wall_samples as f64
    }
}

fn sampled_wall_outcome(outcome: SampledWallOutcome) -> WallOutcome {
    match outcome {
        SampledWallOutcome::Flower => WallOutcome::Flower,
        SampledWallOutcome::Ready(multiplier) => WallOutcome::Ready(f64::from(multiplier)),
        SampledWallOutcome::Won(multiplier) => WallOutcome::Won(f64::from(multiplier)),
        SampledWallOutcome::Other => WallOutcome::Other,
    }
}

fn with_prior_win(outcome: WallOutcome, prior_multiplier: u32) -> WallOutcome {
    if prior_multiplier == 0 {
        return outcome;
    }
    let prior = f64::from(prior_multiplier);
    match outcome {
        WallOutcome::Ready(multiplier) => WallOutcome::Ready(multiplier.max(prior)),
        WallOutcome::Won(multiplier) => WallOutcome::Won(multiplier.max(prior)),
        WallOutcome::Flower => WallOutcome::Flower,
        WallOutcome::Other => WallOutcome::Won(prior),
    }
}

fn hand_wall_outcome(hand: PlanningHand) -> WallOutcome {
    if hand.holding.suit_count() == 3 {
        return WallOutcome::Flower;
    }
    let wait_multiplier = hand
        .holding
        .max_wait()
        .map_or(0, |wait| wait.evaluation.multiplier);
    if wait_multiplier != 0 || hand.holding.has_won {
        let multiplier = f64::from(wait_multiplier.max(hand.max_win_multiplier).max(1));
        if wait_multiplier != 0 {
            WallOutcome::Ready(multiplier)
        } else {
            WallOutcome::Won(multiplier)
        }
    } else {
        WallOutcome::Other
    }
}

fn settle_wall(
    initial_scores: [i64; PLAYER_COUNT],
    outcomes: [WallOutcome; PLAYER_COUNT],
    actor: Seat,
) -> i64 {
    let mut scores = initial_scores;
    let flower = outcomes.map(|outcome| matches!(outcome, WallOutcome::Flower));
    let ready = outcomes.map(|outcome| matches!(outcome, WallOutcome::Ready(_)));
    let eligible =
        outcomes.map(|outcome| matches!(outcome, WallOutcome::Ready(_) | WallOutcome::Won(_)));
    let multiplier = outcomes.map(|outcome| match outcome {
        WallOutcome::Ready(value) | WallOutcome::Won(value) => value.max(1.0),
        WallOutcome::Flower | WallOutcome::Other => 1.0,
    });

    for payer in Seat::ALL {
        if !flower[payer.index()] {
            continue;
        }
        for offset in 1..PLAYER_COUNT as u8 {
            let payee = payer.offset(offset);
            if !flower[payee.index()] && eligible[payee.index()] {
                transfer(&mut scores, payer, payee, 10 * SCORE_UNIT);
            }
        }
        if bankrupt_count(&scores) >= 3 {
            return scores[actor.index()] - initial_scores[actor.index()];
        }
    }

    for payer in Seat::ALL {
        if flower[payer.index()] || ready[payer.index()] {
            continue;
        }
        for offset in 1..PLAYER_COUNT as u8 {
            let payee = payer.offset(offset);
            if !flower[payee.index()] && eligible[payee.index()] {
                transfer(
                    &mut scores,
                    payer,
                    payee,
                    (SCORE_UNIT as f64 * multiplier[payee.index()]).round() as i64,
                );
            }
        }
        if bankrupt_count(&scores) >= 3 {
            break;
        }
    }
    scores[actor.index()] - initial_scores[actor.index()]
}

fn transfer(scores: &mut [i64; PLAYER_COUNT], payer: Seat, payee: Seat, requested: i64) {
    let payment = scores[payer.index()].min(requested).max(0);
    scores[payer.index()] -= payment;
    scores[payee.index()] += payment;
}

fn bankrupt_count(scores: &[i64; PLAYER_COUNT]) -> usize {
    scores.iter().filter(|&&score| score == 0).count()
}

fn seats_after(source: Seat) -> impl Iterator<Item = Seat> {
    (1..PLAYER_COUNT as u8).map(move |offset| source.offset(offset))
}

fn apply_hazard_atom(
    public: PlanningPublicState,
    source: Seat,
    atom: super::belief::HazardAtom,
    payout_scale: u32,
) -> PlanningPublicState {
    let mut model_public = public;
    for winner in seats_after(source) {
        let Some(win) = atom.wins[winner.index()] else {
            continue;
        };
        model_public.max_win_multipliers[winner.index()] =
            model_public.max_win_multipliers[winner.index()].max(win.shape_multiplier);
        transfer(
            &mut model_public.balances,
            source,
            winner,
            SCORE_UNIT * i64::from(win.payout_multiplier.saturating_mul(payout_scale)),
        );
    }
    model_public
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flower_receipt_can_fund_later_dajiao_payment() {
        let scores = [1_000, 0, 10_000, 29_000];
        let outcomes = [
            WallOutcome::Flower,
            WallOutcome::Won(4.0),
            WallOutcome::Ready(1.0),
            WallOutcome::Ready(1.0),
        ];
        let south = Seat::new(1).unwrap();
        assert_eq!(settle_wall(scores, outcomes, south), 800);
    }

    #[test]
    fn a_qualified_actor_receives_dajiao_from_unready_players() {
        let scores = [10_000; PLAYER_COUNT];
        let outcomes = [
            WallOutcome::Ready(4.0),
            WallOutcome::Other,
            WallOutcome::Other,
            WallOutcome::Other,
        ];
        assert_eq!(settle_wall(scores, outcomes, Seat::EAST), 1_200);
    }

    #[test]
    fn kong_transitions_preserve_payment_order_and_caps() {
        let game = Game::new(17);
        let model = PublicValueModel::new(&game, Seat::EAST, None);
        let public = PlanningPublicState {
            balances: [10_000, 50, 300, 0],
            max_win_multipliers: [0; PLAYER_COUNT],
            next_draw: super::super::state::PlannedDraw::Normal,
        };

        let (concealed, gain) = model.concealed_kong_transition(public);
        assert_eq!(gain, 250.0);
        assert_eq!(concealed.balances, [10_250, 0, 100, 0]);
        assert_eq!(
            concealed.next_draw,
            super::super::state::PlannedDraw::Supplement
        );

        let (added, gain) = model.added_kong_transition(public);
        assert_eq!(gain, 150.0);
        assert_eq!(added.balances, [10_150, 0, 200, 0]);
        assert_eq!(
            added.next_draw,
            super::super::state::PlannedDraw::Supplement
        );
    }

    #[test]
    fn hazard_transition_keeps_payments_and_selects_the_last_winner() {
        let public = PlanningPublicState {
            balances: [10_000; PLAYER_COUNT],
            max_win_multipliers: [0; PLAYER_COUNT],
            next_draw: super::super::state::PlannedDraw::Normal,
        };
        let south = Seat::EAST.next();
        let north = Seat::EAST.offset(3);
        let atom = super::super::belief::HazardAtom {
            count: 1,
            wins: [
                None,
                Some(SampledWin {
                    payout_multiplier: 1,
                    shape_multiplier: 1,
                }),
                None,
                Some(SampledWin {
                    payout_multiplier: 2,
                    shape_multiplier: 2,
                }),
            ],
        };

        let next = apply_hazard_atom(public, Seat::EAST, atom, 2);
        assert_eq!(next.balances, [9_400, 10_200, 10_000, 10_400]);
        assert_eq!(next.max_win_multipliers[south.index()], 1);
        assert_eq!(next.max_win_multipliers[north.index()], 2);
        let next_actor = seats_after(Seat::EAST)
            .filter(|winner| atom.wins[winner.index()].is_some())
            .last()
            .unwrap_or(Seat::EAST)
            .next();
        assert_eq!(next_actor, Seat::EAST);
    }

    #[test]
    fn prior_win_makes_an_other_wall_outcome_eligible() {
        assert!(matches!(
            with_prior_win(WallOutcome::Other, 4),
            WallOutcome::Won(4.0)
        ));
        assert!(matches!(
            with_prior_win(WallOutcome::Ready(2.0), 4),
            WallOutcome::Ready(4.0)
        ));
        assert!(matches!(
            with_prior_win(WallOutcome::Flower, 4),
            WallOutcome::Flower
        ));
    }
}
