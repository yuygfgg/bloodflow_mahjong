use core::cmp::Ordering;

use crate::types::{Suit, TILE_KIND_COUNT, Tile};
use crate::{ActionId, analyze_shanten};

use super::hand::{all_tiles, mask_tiles, suit_structure_score};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OpeningQuality {
    estimated_turns: u8,
    shanten: i8,
    missing_tiles: u8,
    live_improvements: u16,
    structure: i32,
}

impl OpeningQuality {
    fn cmp_quality(self, other: Self) -> Ordering {
        other
            .estimated_turns
            .cmp(&self.estimated_turns)
            .then_with(|| other.shanten.cmp(&self.shanten))
            .then_with(|| other.missing_tiles.cmp(&self.missing_tiles))
            .then_with(|| self.live_improvements.cmp(&other.live_improvements))
            .then_with(|| self.structure.cmp(&other.structure))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ExchangePlan {
    quality: OpeningQuality,
    next_tile: Tile,
}

struct ExchangeSearch<'a> {
    concealed: &'a [u8; TILE_KIND_COUNT],
    original_selected: &'a [u8; TILE_KIND_COUNT],
    legal_mask: u32,
    suit: Suit,
    best: Option<ExchangePlan>,
}

impl ExchangeSearch<'_> {
    fn search(&mut self, outgoing: &mut [u8; TILE_KIND_COUNT], start: usize, needed: u8) {
        if needed == 0 {
            self.consider(outgoing);
            return;
        }

        let end = self.suit as usize * 9 + 9;
        for index in start..end {
            if self.legal_mask & (1 << index) == 0 || outgoing[index] >= self.concealed[index] {
                continue;
            }
            outgoing[index] += 1;
            self.search(outgoing, index, needed - 1);
            outgoing[index] -= 1;
        }
    }

    fn consider(&mut self, outgoing: &[u8; TILE_KIND_COUNT]) {
        let mut remaining = *self.concealed;
        for (count, selected) in remaining.iter_mut().zip(outgoing.iter().copied()) {
            *count -= selected;
        }
        let (_, quality) = best_missing_choice(&remaining, self.concealed);
        let next_tile = all_tiles()
            .find(|tile| {
                outgoing[tile.index()] > self.original_selected[tile.index()]
                    && self.legal_mask & (1 << tile.index()) != 0
            })
            .expect("a completed exchange plan adds a legal tile");
        let candidate = ExchangePlan { quality, next_tile };
        if self
            .best
            .is_none_or(|current| exchange_plan_better(candidate, current))
        {
            self.best = Some(candidate);
        }
    }
}

/// Selects one tile from the best complete three-tile exchange plan.
pub(super) fn choose_exchange(
    concealed: &[u8; TILE_KIND_COUNT],
    selected: &[u8; TILE_KIND_COUNT],
    legal_mask: u32,
) -> ActionId {
    let selected_count: u8 = selected.iter().copied().sum();
    debug_assert!(selected_count < 3);
    debug_assert!(
        concealed
            .iter()
            .zip(selected)
            .all(|(&count, &reserved)| reserved <= count)
    );

    let mut best = None;
    if selected_count == 0 {
        for suit in Suit::ALL {
            if legal_mask & suit.mask() == 0 {
                continue;
            }
            let mut outgoing = *selected;
            let mut search = ExchangeSearch {
                concealed,
                original_selected: selected,
                legal_mask,
                suit,
                best: None,
            };
            search.search(&mut outgoing, suit as usize * 9, 3);
            if let Some(candidate) = search.best
                && best.is_none_or(|current| exchange_plan_better(candidate, current))
            {
                best = Some(candidate);
            }
        }
    } else {
        let suit = all_tiles()
            .find(|tile| selected[tile.index()] != 0)
            .map(Tile::suit)
            .expect("a partial exchange has a selected suit");
        let mut outgoing = *selected;
        let mut search = ExchangeSearch {
            concealed,
            original_selected: selected,
            legal_mask,
            suit,
            best: None,
        };
        search.search(&mut outgoing, suit as usize * 9, 3 - selected_count);
        best = search.best;
    }

    ActionId::select_exchange_tile(
        best.expect("an exchange decision has a complete legal plan")
            .next_tile,
    )
}

/// Selects the missing suit which gives the best post-purge hand.
pub(super) fn choose_missing(concealed: &[u8; TILE_KIND_COUNT]) -> ActionId {
    let (suit, _) = best_missing_choice(concealed, concealed);
    ActionId::choose_missing(suit)
}

fn opening_quality(
    counts: &[u8; TILE_KIND_COUNT],
    exposure: &[u8; TILE_KIND_COUNT],
    missing: Suit,
) -> OpeningQuality {
    let analysis = analyze_shanten(counts, &[], Some(missing));
    let missing_tiles = suit_count(counts, missing);
    let structural_turns =
        u8::try_from(analysis.shanten.max(0) + 1).expect("shanten has a small non-negative value");
    OpeningQuality {
        estimated_turns: missing_tiles.max(structural_turns),
        shanten: analysis.shanten,
        missing_tiles,
        live_improvements: remaining_copies(analysis.improving_tiles, exposure),
        structure: structure_without_suit(counts, missing),
    }
}

fn best_missing_choice(
    counts: &[u8; TILE_KIND_COUNT],
    exposure: &[u8; TILE_KIND_COUNT],
) -> (Suit, OpeningQuality) {
    let mut best = None;
    for suit in Suit::ALL {
        let quality = opening_quality(counts, exposure, suit);
        if best.is_none_or(|(current_suit, current)| {
            quality.cmp_quality(current) == Ordering::Greater
                || (quality == current && (suit as u8) < current_suit as u8)
        }) {
            best = Some((suit, quality));
        }
    }
    best.expect("three suits are always available")
}

fn exchange_plan_better(candidate: ExchangePlan, current: ExchangePlan) -> bool {
    candidate.quality.cmp_quality(current.quality) == Ordering::Greater
        || (candidate.quality == current.quality
            && (
                edge_distance(candidate.next_tile),
                u8::MAX - candidate.next_tile.as_u8(),
            ) > (
                edge_distance(current.next_tile),
                u8::MAX - current.next_tile.as_u8(),
            ))
}

fn remaining_copies(mask: u32, exposure: &[u8; TILE_KIND_COUNT]) -> u16 {
    mask_tiles(mask)
        .map(|tile| u16::from(4_u8.saturating_sub(exposure[tile.index()].min(4))))
        .sum()
}

fn structure_without_suit(counts: &[u8; TILE_KIND_COUNT], missing: Suit) -> i32 {
    Suit::ALL
        .into_iter()
        .filter(|&suit| suit != missing)
        .map(|suit| suit_structure_score(counts, suit))
        .sum()
}

fn suit_count(counts: &[u8; TILE_KIND_COUNT], suit: Suit) -> u8 {
    let start = suit as usize * 9;
    counts[start..start + 9].iter().copied().sum()
}

fn edge_distance(tile: Tile) -> u8 {
    tile.rank().abs_diff(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Action;

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank - 1).expect("test rank is valid")
    }

    fn opening_hand() -> [u8; TILE_KIND_COUNT] {
        let mut counts = [0; TILE_KIND_COUNT];
        for rank in 1..=5 {
            counts[tile(Suit::Characters, rank).index()] = 1;
        }
        for rank in 1..=4 {
            counts[tile(Suit::Bamboo, rank).index()] = 1;
            counts[tile(Suit::Dots, rank).index()] = 1;
        }
        counts
    }

    #[test]
    fn exchange_selects_a_legal_tile_from_a_complete_plan() {
        let counts = opening_hand();
        let legal = Suit::Characters.mask() | Suit::Bamboo.mask() | Suit::Dots.mask();
        let action = choose_exchange(&counts, &[0; TILE_KIND_COUNT], legal);
        let Action::SelectExchangeTile(selected) = action.action() else {
            panic!("exchange selection returned a non-exchange action");
        };
        assert!(legal & (1 << selected.index()) != 0);
    }

    #[test]
    fn partial_exchange_keeps_the_selected_suit() {
        let counts = opening_hand();
        let first = tile(Suit::Bamboo, 2);
        let mut selected = [0; TILE_KIND_COUNT];
        selected[first.index()] = 1;
        let legal = Suit::Bamboo.mask()
            & counts.iter().enumerate().fold(0, |mask, (index, count)| {
                mask | (u32::from(*count != 0) << index)
            });
        let action = choose_exchange(&counts, &selected, legal);
        let Action::SelectExchangeTile(selected) = action.action() else {
            panic!("exchange selection returned a non-exchange action");
        };
        assert_eq!(selected.suit(), Suit::Bamboo);
    }

    #[test]
    fn missing_choice_is_deterministic_for_equal_empty_suits() {
        let mut counts = [0; TILE_KIND_COUNT];
        counts[tile(Suit::Dots, 5).index()] = 1;
        assert_eq!(
            choose_missing(&counts),
            ActionId::choose_missing(Suit::Characters)
        );
    }
}
