use crate::game::Game;
use crate::types::{PLAYER_COUNT, Seat, Tile};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DiscardOrigin {
    Hand,
    Draw,
    ReplacementDraw,
    Pong,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RiverEntry {
    pub(super) tile: Tile,
    pub(super) origin: DiscardOrigin,
}

/// Reversible public discard suffix used by opponent-model features.
///
/// `Game::discards` is sufficient for tile exposure. The planner also needs
/// the chronological hand-discard/draw-discard distinction used by readiness
/// and wait-shape models. A public Hu, Pong, or Kong starts a new suffix because
/// reversing across that structural hand change requires different states.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct PublicHistory {
    by_player: [Vec<RiverEntry>; PLAYER_COUNT],
}

impl PublicHistory {
    pub(super) fn from_game(game: &Game) -> Self {
        let mut history = Self::default();
        for discard in game.public_discards() {
            if discard.hand_revision != game.public_hand_revision(discard.player) {
                continue;
            }
            let origin = if discard.after_pong {
                DiscardOrigin::Pong
            } else if discard.tsumogiri {
                if discard.after_kong {
                    DiscardOrigin::ReplacementDraw
                } else {
                    DiscardOrigin::Draw
                }
            } else {
                DiscardOrigin::Hand
            };
            let entry = RiverEntry {
                tile: discard.tile,
                origin,
            };
            history.by_player[discard.player.index()].push(entry);
        }
        history
    }

    pub(super) fn player(&self, player: Seat) -> &[RiverEntry] {
        &self.by_player[player.index()]
    }
}
