use crate::game::Game;
use crate::types::{Meld, MeldKind, Seat, Suit, TILE_KIND_COUNT, Tile};

pub(crate) const DUMMY_MELD: Meld = Meld {
    tile: Tile::from_index_unchecked(0),
    kind: MeldKind::Pong,
    source: Seat::EAST,
};

/// Rule-policy projection of one player's mutable hand state.
///
/// This type mirrors the engine transformations which affect concealed tiles,
/// locked winning subsets, and exposed melds. Keeping the transformations in
/// one place prevents evaluators from drifting away from legal play.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct Holding {
    pub(crate) concealed: [u8; TILE_KIND_COUNT],
    pub(crate) locked: [u8; TILE_KIND_COUNT],
    pub(crate) melds: [Meld; 4],
    pub(crate) meld_len: usize,
    pub(crate) missing: Option<Suit>,
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
            melds,
            meld_len,
            missing: game.missing_suit(actor),
        }
    }

    pub(crate) fn melds(&self) -> &[Meld] {
        &self.melds[..self.meld_len]
    }

    pub(crate) fn unlocked_count(&self, tile: Tile) -> u8 {
        self.concealed[tile.index()].saturating_sub(self.locked[tile.index()])
    }

    pub(crate) fn missing_count(&self) -> u8 {
        self.missing.map_or(0, |suit| {
            let start = suit as usize * 9;
            self.concealed[start..start + 9].iter().copied().sum()
        })
    }

    pub(crate) fn suit_count(&self) -> usize {
        Suit::ALL
            .into_iter()
            .filter(|suit| {
                let start = *suit as usize * 9;
                self.concealed[start..start + 9]
                    .iter()
                    .any(|&count| count != 0)
                    || self.melds().iter().any(|meld| meld.tile.suit() == *suit)
            })
            .count()
    }

    fn unlocked_len(&self) -> usize {
        self.concealed
            .iter()
            .zip(self.locked)
            .map(|(&concealed, locked)| usize::from(concealed.saturating_sub(locked)))
            .sum()
    }

    pub(crate) fn discard_mask(&self) -> u32 {
        let unlocked = all_tiles().fold(0_u32, |mask, tile| {
            mask | (u32::from(self.unlocked_count(tile) != 0) << tile.index())
        });
        let forced = self.missing.map_or(0, Suit::mask) & unlocked;
        if forced == 0 { unlocked } else { forced }
    }

    pub(crate) fn after_draw(mut self, tile: Tile) -> Option<Self> {
        if self.concealed[tile.index()] >= 4 {
            return None;
        }
        self.concealed[tile.index()] += 1;
        Some(self)
    }

    pub(crate) fn after_discard(mut self, tile: Tile) -> Option<Self> {
        if self.unlocked_count(tile) == 0 {
            return None;
        }
        self.concealed[tile.index()] -= 1;
        Some(self)
    }

    fn remove_for_meld(&mut self, tile: Tile, amount: u8, allow_locked: bool) -> bool {
        let index = tile.index();
        if self.concealed[index] < amount || (!allow_locked && self.unlocked_count(tile) < amount) {
            return false;
        }
        if allow_locked {
            self.locked[index] -= self.locked[index].min(amount);
        }
        self.concealed[index] -= amount;
        true
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
        if self.missing == Some(tile.suit())
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
        if self.missing == Some(tile.suit())
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

    pub(crate) fn after_robbed_added_kong(mut self, tile: Tile) -> Option<Self> {
        let index = tile.index();
        if self.concealed[index] == 0 {
            return None;
        }
        self.concealed[index] -= 1;
        self.locked[index] = self.locked[index].min(self.concealed[index]);
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
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: None,
        };
        holding.concealed[one.index()] = 2;
        holding.locked[one.index()] = 2;
        holding.concealed[two.index()] = 1;

        assert_eq!(holding.discard_mask(), 1 << two.index());
        assert!(holding.after_discard(one).is_none());
        assert!(holding.after_discard(two).is_some());
    }

    #[test]
    fn meld_transforms_enforce_missing_suit_and_pong_discard_requirements() {
        let tile = Tile::from_suit_rank(Suit::Characters, 0).expect("test tile is valid");
        let mut holding = Holding {
            concealed: [0; TILE_KIND_COUNT],
            locked: [0; TILE_KIND_COUNT],
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: Some(Suit::Characters),
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
    fn suit_count_includes_exposed_melds() {
        let characters = Tile::from_suit_rank(Suit::Characters, 0).expect("test tile is valid");
        let bamboo = Tile::from_suit_rank(Suit::Bamboo, 0).expect("test tile is valid");
        let dots = Tile::from_suit_rank(Suit::Dots, 0).expect("test tile is valid");
        let mut holding = Holding {
            concealed: [0; TILE_KIND_COUNT],
            locked: [0; TILE_KIND_COUNT],
            melds: [DUMMY_MELD; 4],
            meld_len: 1,
            missing: None,
        };
        holding.melds[0] = Meld {
            tile: characters,
            kind: MeldKind::Pong,
            source: Seat::EAST.next(),
        };
        holding.concealed[bamboo.index()] = 1;
        holding.concealed[dots.index()] = 1;

        assert_eq!(holding.suit_count(), 3);
    }
}
