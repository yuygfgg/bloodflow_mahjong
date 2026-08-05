use crate::game::Game;
use crate::hand::apply_bloodflow_win;
use crate::rules::hand::Holding;
use crate::types::{PLAYER_COUNT, Seat, Tile};
use crate::{ShantenAnalysis, WinEvaluation, WinFlags};

/// Origin of the next draw represented by a planner public state.
///
/// A kong schedules one replacement draw. Consuming that draw restores the
/// ordinary origin, so event-only win multipliers cannot leak into later turns.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub(super) enum PlannedDraw {
    #[default]
    Normal,
    Supplement,
}

/// Public score and prior-win state carried by Bellman transitions.
///
/// Integer balances preserve insufficient-funds semantics and make the state
/// suitable for exact memoization. A zero multiplier means that the player has
/// not won yet.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) struct PlanningPublicState {
    pub(super) balances: [i64; PLAYER_COUNT],
    pub(super) max_win_multipliers: [u32; PLAYER_COUNT],
    pub(super) next_draw: PlannedDraw,
}

impl PlanningPublicState {
    pub(super) fn from_game(game: &Game) -> Self {
        Self {
            balances: Seat::ALL.map(|seat| game.score(seat).max(0)),
            max_win_multipliers: Seat::ALL.map(|seat| game.max_win_multiplier(seat)),
            next_draw: PlannedDraw::Normal,
        }
    }

    pub(super) const fn with_supplement_draw(mut self) -> Self {
        self.next_draw = PlannedDraw::Supplement;
        self
    }

    pub(super) const fn take_next_draw(mut self) -> (Self, PlannedDraw) {
        let draw = self.next_draw;
        self.next_draw = PlannedDraw::Normal;
        (self, draw)
    }

    pub(super) fn is_terminal(self) -> bool {
        self.balances
            .iter()
            .filter(|&&balance| balance == 0)
            .count()
            >= 3
    }
}

/// Complete private hand state used by planner transitions.
///
/// `Holding` describes physical tiles and melds. The two additional fields are
/// required because a Blood Flow win is not terminal: later wins must preserve
/// the locked winning subset and end-of-wall settlement uses the best prior
/// multiplier.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct PlanningHand {
    pub(super) holding: Holding,
    pub(super) max_win_multiplier: u32,
}

impl PlanningHand {
    pub(super) fn new(mut holding: Holding, has_won: bool, max_win_multiplier: u32) -> Self {
        holding.has_won = has_won;
        Self {
            holding,
            max_win_multiplier,
        }
    }

    pub(super) fn analysis(self) -> ShantenAnalysis {
        self.holding.analysis()
    }

    pub(super) fn shanten(self) -> i8 {
        self.holding.analysis().shanten
    }

    pub(super) fn with_draw(self, tile: Tile) -> Option<Self> {
        Some(Self {
            holding: self.holding.after_draw(tile)?,
            ..self
        })
    }

    pub(super) fn after_discard(self, tile: Tile) -> Option<Self> {
        Some(Self {
            holding: self.holding.after_discard(tile)?,
            ..self
        })
    }

    pub(super) fn win_on_draw(self, tile: Tile, flags: WinFlags) -> Option<WinEvaluation> {
        self.holding.evaluate_win(Some(tile), flags)
    }

    /// Applies the same stable-base transition as `Game::apply_win`.
    pub(super) fn after_win(mut self, required: Option<Tile>, evaluation: WinEvaluation) -> Self {
        let applied = apply_bloodflow_win(
            &self.holding.concealed,
            &mut self.holding.locked,
            &mut self.holding.win_base,
            self.holding.has_won,
            &evaluation.used,
            required,
        );
        debug_assert!(
            applied.is_some(),
            "a planned legal win must match its physical hand"
        );
        self.holding.has_won = true;
        self.max_win_multiplier = self.max_win_multiplier.max(evaluation.shape_multiplier);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::hand::DUMMY_MELD;
    use crate::types::{Suit, TILE_KIND_COUNT};

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank - 1).expect("test tile is valid")
    }

    #[test]
    fn a_blood_flow_win_expands_the_locked_subset() {
        let mut counts = [0; TILE_KIND_COUNT];
        for rank in [1, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 9, 9] {
            counts[tile(Suit::Characters, rank).index()] += 1;
        }
        let holding = Holding {
            concealed: counts,
            locked: [0; TILE_KIND_COUNT],
            win_base: [0; TILE_KIND_COUNT],
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: Some(Suit::Dots),
            has_won: false,
        };
        let state = PlanningHand::new(holding, false, 0);
        let winning_tile = tile(Suit::Characters, 9);
        let win = state
            .win_on_draw(winning_tile, WinFlags::NONE)
            .expect("the hand wins on 9m");
        let event_win = state
            .win_on_draw(
                winning_tile,
                WinFlags {
                    after_kong_draw: true,
                    last_wall_tile: true,
                    ..WinFlags::NONE
                },
            )
            .expect("event flags preserve a legal win");
        let after = state.after_win(Some(winning_tile), win);

        assert_eq!(event_win.multiplier, win.multiplier * 4);
        assert!(after.holding.has_won);
        assert!(after.holding.locked[winning_tile.index()] > 0);
        assert!(after.max_win_multiplier > 0);
    }

    #[test]
    fn scheduled_supplement_draw_is_consumed_once() {
        let game = Game::new(9);
        let public = PlanningPublicState::from_game(&game).with_supplement_draw();

        let (after, draw) = public.take_next_draw();
        assert_eq!(draw, PlannedDraw::Supplement);
        assert_eq!(after.next_draw, PlannedDraw::Normal);
    }
}
