use crate::game::Game;
use crate::hand::{
    ShantenAnalysis, analyze_shanten, bloodflow_evaluation_counts, evaluate_bloodflow_win,
    evaluate_win, remove_tiles_for_meld, stabilize_win_base,
};
use crate::types::{Meld, MeldKind, Seat, Suit, TILE_KIND_COUNT, Tile};
use crate::{WinEvaluation, WinFlags};

pub(crate) const DUMMY_MELD: Meld = Meld {
    tile: Tile::from_index_unchecked(0),
    kind: MeldKind::Pong,
    source: Seat::EAST,
};

/// Rule-policy projection of one player's mutable hand state.
///
/// This type mirrors the engine transformations which affect concealed tiles,
/// the stable winning base, historical winning tiles, and exposed melds.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct Holding {
    pub(crate) concealed: [u8; TILE_KIND_COUNT],
    pub(crate) locked: [u8; TILE_KIND_COUNT],
    pub(crate) win_base: [u8; TILE_KIND_COUNT],
    pub(crate) melds: [Meld; 4],
    pub(crate) meld_len: usize,
    pub(crate) missing: Option<Suit>,
    pub(crate) has_won: bool,
}

impl Holding {
    pub(crate) fn from_game(game: &Game, actor: Seat) -> Self {
        let mut melds = [DUMMY_MELD; 4];
        let meld_len = game.meld_count(actor);
        for (index, meld) in melds.iter_mut().enumerate().take(meld_len) {
            *meld = game.meld(actor, index).expect("meld slots are dense");
        }
        Self {
            concealed: *game.concealed(actor),
            locked: *game.locked(actor),
            win_base: *game.win_base(actor),
            melds,
            meld_len,
            missing: game.missing_suit(actor),
            has_won: game.has_won(actor),
        }
    }

    pub(crate) fn melds(&self) -> &[Meld] {
        &self.melds[..self.meld_len]
    }

    pub(crate) fn evaluate_win(
        &self,
        required: Option<Tile>,
        flags: WinFlags,
    ) -> Option<WinEvaluation> {
        if self.missing_count() != 0 {
            return None;
        }
        if self.has_won {
            let counts = self.evaluation_counts()?;
            evaluate_bloodflow_win(&counts, &self.win_base, self.melds(), required, flags)
        } else {
            evaluate_win(&self.concealed, self.melds(), required, flags)
        }
    }

    pub(crate) fn analysis(&self) -> ShantenAnalysis {
        let counts = self.evaluation_counts().unwrap_or(self.concealed);
        analyze_shanten(&counts, self.melds(), self.missing)
    }

    pub(crate) fn evaluation_counts(&self) -> Option<[u8; TILE_KIND_COUNT]> {
        if self.has_won {
            bloodflow_evaluation_counts(&self.concealed, &self.locked, &self.win_base)
        } else {
            Some(self.concealed)
        }
    }

    pub(crate) fn unlocked_count(&self, tile: Tile) -> u8 {
        self.concealed[tile.index()].saturating_sub(self.locked[tile.index()])
    }

    pub(crate) fn missing_count(&self) -> u8 {
        self.missing.map_or(0, |suit| {
            let start = suit as usize * 9;
            self.evaluation_counts()
                .map_or(0, |counts| counts[start..start + 9].iter().sum())
        })
    }

    fn unlocked_len(&self) -> usize {
        self.concealed
            .iter()
            .zip(self.locked)
            .map(|(&concealed, locked)| usize::from(concealed.saturating_sub(locked)))
            .sum()
    }

    pub(crate) fn discard_mask(&self) -> u32 {
        all_tiles().fold(0_u32, |mask, tile| {
            mask | (u32::from(self.unlocked_count(tile) != 0) << tile.index())
        })
    }

    pub(crate) fn after_draw(mut self, tile: Tile) -> Option<Self> {
        self.concealed[tile.index()] = self.concealed[tile.index()].checked_add(1)?;
        Some(self)
    }

    pub(crate) fn after_discard(mut self, tile: Tile) -> Option<Self> {
        if self.unlocked_count(tile) == 0 {
            return None;
        }
        self.concealed[tile.index()] -= 1;
        if self.has_won {
            let target_len = 13_usize.saturating_sub(3 * self.meld_len);
            if !stabilize_win_base(
                &self.concealed,
                &mut self.locked,
                &mut self.win_base,
                target_len,
            ) {
                return None;
            }
        }
        Some(self)
    }

    fn remove_for_meld(&mut self, tile: Tile, amount: u8, allow_win_base: bool) -> bool {
        remove_tiles_for_meld(
            &mut self.concealed,
            &mut self.locked,
            &mut self.win_base,
            tile,
            amount,
            allow_win_base,
        )
    }

    fn push_meld(&mut self, meld: Meld) -> bool {
        if self.meld_len == self.melds.len() {
            return false;
        }
        self.melds[self.meld_len] = meld;
        self.meld_len += 1;
        true
    }

    pub(crate) fn after_pong(mut self, tile: Tile, source: Seat) -> Option<Self> {
        if self.has_won
            || self.missing == Some(tile.suit())
            || self.unlocked_len() < 3
            || !self.remove_for_meld(tile, 2, false)
            || !self.push_meld(Meld {
                tile,
                kind: MeldKind::Pong,
                source,
            })
        {
            return None;
        }
        Some(self)
    }

    pub(crate) fn after_exposed_kong(mut self, tile: Tile, source: Seat) -> Option<Self> {
        if self.has_won
            || self.missing == Some(tile.suit())
            || !self.remove_for_meld(tile, 3, false)
            || !self.push_meld(Meld {
                tile,
                kind: MeldKind::ExposedKong,
                source,
            })
        {
            return None;
        }
        Some(self)
    }

    pub(crate) fn after_concealed_kong(mut self, tile: Tile, actor: Seat) -> Option<Self> {
        if self.missing == Some(tile.suit())
            || !self.remove_for_meld(tile, 4, true)
            || !self.push_meld(Meld {
                tile,
                kind: MeldKind::ConcealedKong,
                source: actor,
            })
        {
            return None;
        }
        Some(self)
    }

    pub(crate) fn after_added_kong(mut self, tile: Tile) -> Option<Self> {
        if self.missing == Some(tile.suit()) || !self.remove_for_meld(tile, 1, true) {
            return None;
        }
        let meld = self.melds[..self.meld_len]
            .iter_mut()
            .find(|meld| meld.kind == MeldKind::Pong && meld.tile == tile)?;
        meld.kind = MeldKind::AddedKong;
        Some(self)
    }
}

pub(crate) fn all_tiles() -> impl Iterator<Item = Tile> {
    (0..TILE_KIND_COUNT).map(|index| Tile::from_index_unchecked(index as u8))
}

pub(crate) fn mask_tiles(mut mask: u32) -> impl Iterator<Item = Tile> {
    core::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let index = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        Some(Tile::from_index_unchecked(index))
    })
}

pub(crate) fn hand_structure_score(counts: &[u8; TILE_KIND_COUNT]) -> i32 {
    Suit::ALL
        .into_iter()
        .map(|suit| suit_structure_score(counts, suit))
        .sum()
}

pub(crate) fn suit_structure_score(counts: &[u8; TILE_KIND_COUNT], suit: Suit) -> i32 {
    let start = suit as usize * 9;
    let suit_counts = &counts[start..start + 9];
    let mut score = 0_i32;
    for (rank, &count) in suit_counts.iter().enumerate() {
        let count = i32::from(count);
        score += count * (4 - (rank as i32 - 4).abs());
        if count >= 2 {
            score += 8;
        }
        if count >= 3 {
            score += 12;
        }
        if count == 4 {
            score += 2;
        }
    }
    for rank in 0..8 {
        score += 3 * i32::from(suit_counts[rank].min(suit_counts[rank + 1]));
    }
    for rank in 0..7 {
        score += 2 * i32::from(suit_counts[rank].min(suit_counts[rank + 2]));
        score += 12
            * i32::from(
                suit_counts[rank]
                    .min(suit_counts[rank + 1])
                    .min(suit_counts[rank + 2]),
            );
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discard_mask_excludes_locked_tiles() {
        let one = Tile::from_index_unchecked(0);
        let two = Tile::from_index_unchecked(1);
        let mut holding = Holding {
            concealed: [0; TILE_KIND_COUNT],
            locked: [0; TILE_KIND_COUNT],
            win_base: [0; TILE_KIND_COUNT],
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: None,
            has_won: false,
        };
        holding.concealed[one.index()] = 2;
        holding.locked[one.index()] = 2;
        holding.concealed[two.index()] = 1;

        assert_eq!(holding.discard_mask(), 1 << two.index());
        assert!(holding.after_discard(one).is_none());
        assert!(holding.after_discard(two).is_some());
    }

    #[test]
    fn discard_mask_allows_every_unlocked_suit_when_missing_is_set() {
        let characters = Tile::from_suit_rank(Suit::Characters, 0).expect("test tile is valid");
        let bamboo = Tile::from_suit_rank(Suit::Bamboo, 0).expect("test tile is valid");
        let mut holding = Holding {
            concealed: [0; TILE_KIND_COUNT],
            locked: [0; TILE_KIND_COUNT],
            win_base: [0; TILE_KIND_COUNT],
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: Some(Suit::Characters),
            has_won: false,
        };
        holding.concealed[characters.index()] = 1;
        holding.concealed[bamboo.index()] = 1;

        let mask = holding.discard_mask();
        assert_ne!(mask & (1 << characters.index()), 0);
        assert_ne!(mask & (1 << bamboo.index()), 0);
    }

    #[test]
    fn meld_transforms_enforce_missing_suit_and_pong_discard_requirements() {
        let tile = Tile::from_suit_rank(Suit::Characters, 0).expect("test tile is valid");
        let mut holding = Holding {
            concealed: [0; TILE_KIND_COUNT],
            locked: [0; TILE_KIND_COUNT],
            win_base: [0; TILE_KIND_COUNT],
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: Some(Suit::Characters),
            has_won: false,
        };
        holding.concealed[tile.index()] = 4;

        assert!(holding.after_pong(tile, Seat::EAST.next()).is_none());
        assert!(
            holding
                .after_exposed_kong(tile, Seat::EAST.next())
                .is_none()
        );
        assert!(holding.after_concealed_kong(tile, Seat::EAST).is_none());

        holding.missing = Some(Suit::Dots);
        holding.concealed[tile.index()] = 2;
        assert!(holding.after_pong(tile, Seat::EAST.next()).is_none());
        holding.concealed[1] = 1;
        assert!(holding.after_pong(tile, Seat::EAST.next()).is_some());
    }

    #[test]
    fn repeated_win_ignores_the_historical_winning_tile() {
        let winning_tile = Tile::from_index_unchecked(17);
        let mut win_base = [0; TILE_KIND_COUNT];
        for count in win_base.iter_mut().take(6) {
            *count = 1;
        }
        for count in win_base.iter_mut().take(15).skip(9) {
            *count = 1;
        }
        win_base[winning_tile.index()] = 1;

        let mut holding = Holding {
            concealed: win_base,
            locked: win_base,
            win_base,
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: None,
            has_won: true,
        };
        holding.concealed[winning_tile.index()] += 1;
        holding.locked[winning_tile.index()] += 1;

        assert_eq!(
            holding.evaluation_counts().unwrap()[winning_tile.index()],
            1
        );
        assert!(
            holding
                .evaluate_win(Some(winning_tile), WinFlags::NONE)
                .is_none()
        );
        let after_draw = holding
            .after_draw(winning_tile)
            .expect("the active winning tile can be drawn");
        assert!(
            after_draw
                .evaluate_win(Some(winning_tile), WinFlags::NONE)
                .is_some()
        );
    }

    fn post_win_added_kong_holding(source: KongTileSource) -> (Holding, Tile, Tile, Tile) {
        let kong_tile = Tile::from_index_unchecked(0);
        let active_tile = Tile::from_index_unchecked(20);
        let replacement_tile = Tile::from_index_unchecked(21);
        let mut win_base = [0; TILE_KIND_COUNT];
        for count in win_base.iter_mut().take(11).skip(1) {
            *count = 1;
        }
        if source == KongTileSource::StableBase {
            win_base[1] = 0;
            win_base[kong_tile.index()] = 1;
        }
        let mut holding = Holding {
            concealed: win_base,
            locked: win_base,
            win_base,
            melds: [DUMMY_MELD; 4],
            meld_len: 1,
            missing: None,
            has_won: true,
        };
        holding.concealed[active_tile.index()] = 1;
        match source {
            KongTileSource::Active => holding.concealed[kong_tile.index()] += 1,
            KongTileSource::Historical => {
                holding.concealed[kong_tile.index()] += 1;
                holding.locked[kong_tile.index()] += 1;
            }
            KongTileSource::StableBase => {}
        }
        holding.melds[0] = Meld {
            tile: kong_tile,
            kind: MeldKind::Pong,
            source: Seat::EAST.next(),
        };
        (holding, kong_tile, active_tile, replacement_tile)
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum KongTileSource {
        Active,
        Historical,
        StableBase,
    }

    #[test]
    fn added_kong_rejects_a_historical_winning_tile() {
        let (holding, kong_tile, _, _) = post_win_added_kong_holding(KongTileSource::Historical);

        assert!(holding.after_added_kong(kong_tile).is_none());
    }

    #[test]
    fn concealed_kong_uses_active_and_stable_base_tiles() {
        let kong_tile = Tile::from_index_unchecked(0);
        let mut win_base = [0; TILE_KIND_COUNT];
        win_base[kong_tile.index()] = 3;
        for count in win_base.iter_mut().take(11).skip(1) {
            *count = 1;
        }
        let mut holding = Holding {
            concealed: win_base,
            locked: win_base,
            win_base,
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: None,
            has_won: true,
        };
        holding.concealed[kong_tile.index()] += 1;

        let after = holding
            .after_concealed_kong(kong_tile, Seat::EAST)
            .expect("three stable tiles and one active tile form a concealed Kong");

        assert_eq!(after.win_base[kong_tile.index()], 0);
        assert_eq!(after.locked[kong_tile.index()], 0);
        assert_eq!(after.concealed[kong_tile.index()], 0);
    }

    #[test]
    fn concealed_kong_rejects_historical_winning_tiles() {
        let kong_tile = Tile::from_index_unchecked(0);
        let mut win_base = [0; TILE_KIND_COUNT];
        for count in win_base.iter_mut().take(13).skip(1) {
            *count = 1;
        }
        win_base[13] = 1;
        let mut holding = Holding {
            concealed: win_base,
            locked: win_base,
            win_base,
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: None,
            has_won: true,
        };
        holding.concealed[kong_tile.index()] = 4;
        holding.locked[kong_tile.index()] = 4;

        assert!(
            holding
                .after_concealed_kong(kong_tile, Seat::EAST)
                .is_none()
        );
    }

    #[test]
    fn added_kong_accepts_an_active_fourth_tile() {
        let (holding, kong_tile, _, _) = post_win_added_kong_holding(KongTileSource::Active);
        let win_base = holding.win_base;

        let after = holding
            .after_added_kong(kong_tile)
            .expect("an active fourth tile extends the Pong");

        assert_eq!(after.win_base, win_base);
        assert_eq!(after.unlocked_count(kong_tile), 0);
    }

    #[test]
    fn post_kong_discard_restores_a_consumed_stable_base_tile() {
        let (holding, kong_tile, active_tile, replacement_tile) =
            post_win_added_kong_holding(KongTileSource::StableBase);

        let after = holding
            .after_added_kong(kong_tile)
            .and_then(|holding| holding.after_draw(replacement_tile))
            .and_then(|holding| holding.after_discard(active_tile))
            .expect("the replacement draw restores the shortened base");

        assert_eq!(after.win_base.iter().sum::<u8>(), 10);
        assert_eq!(after.win_base[kong_tile.index()], 0);
        assert_eq!(after.win_base[replacement_tile.index()], 1);
        assert_eq!(after.locked, after.concealed);
    }
}
