use core::cmp::Ordering;
use std::collections::BTreeMap;

use crate::rules::hand::{Holding, all_tiles, mask_tiles};
use crate::types::{TILE_COPIES, TILE_KIND_COUNT, Tile};
use crate::{WinFlags, analyze_shanten, evaluate_win};

use super::history::{DiscardOrigin, RiverEntry};

const MAX_REVERSE_HISTORY_STATES: usize = 64;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct HandPotential {
    shanten: i8,
    live_improvements: u16,
    weighted_wait_value: u64,
    live_waits: u16,
    max_multiplier: u32,
    structure: i32,
}

impl HandPotential {
    pub(super) fn evaluate(
        holding: &Holding,
        visible: &[u8; TILE_KIND_COUNT],
        has_won: bool,
    ) -> Self {
        let analysis = analyze_shanten(&holding.concealed, holding.melds(), holding.missing);
        let mut potential = Self {
            shanten: analysis.shanten,
            live_improvements: remaining_copies(analysis.improving_tiles, visible),
            structure: structure_score(&holding.concealed),
            ..Self::default()
        };

        if analysis.shanten <= 0 || has_won {
            let mut augmented = holding.concealed;
            for tile in all_tiles() {
                if holding.missing == Some(tile.suit()) {
                    continue;
                }
                let copies = TILE_COPIES.saturating_sub(visible[tile.index()].min(TILE_COPIES));
                if copies == 0 || augmented[tile.index()] >= TILE_COPIES {
                    continue;
                }
                augmented[tile.index()] += 1;
                if let Some(win) =
                    evaluate_win(&augmented, holding.melds(), Some(tile), WinFlags::NONE)
                {
                    potential.live_waits += u16::from(copies);
                    potential.weighted_wait_value += u64::from(copies) * u64::from(win.multiplier);
                    potential.max_multiplier = potential.max_multiplier.max(win.multiplier);
                }
                augmented[tile.index()] -= 1;
            }
        }
        potential
    }

    pub(super) fn cmp_for(self, other: Self, has_won: bool) -> Ordering {
        if has_won {
            return (
                self.weighted_wait_value,
                self.live_waits,
                self.max_multiplier,
                self.structure,
            )
                .cmp(&(
                    other.weighted_wait_value,
                    other.live_waits,
                    other.max_multiplier,
                    other.structure,
                ));
        }
        (
            -self.shanten,
            self.live_improvements,
            self.weighted_wait_value,
            self.live_waits,
            self.max_multiplier,
            self.structure,
        )
            .cmp(&(
                -other.shanten,
                other.live_improvements,
                other.weighted_wait_value,
                other.live_waits,
                other.max_multiplier,
                other.structure,
            ))
    }

    /// Returns whether a Kong keeps the hand on a non-worsening route.
    ///
    /// Before the first win, shanten is the hard progress boundary. After a
    /// win, locked tiles prevent rebuilding, so the wait distribution becomes
    /// the boundary instead.
    pub(super) fn permits_kong_from(self, baseline: Self, has_won: bool) -> bool {
        if has_won {
            (
                self.weighted_wait_value,
                self.live_waits,
                self.max_multiplier,
            ) >= (
                baseline.weighted_wait_value,
                baseline.live_waits,
                baseline.max_multiplier,
            )
        } else {
            self.shanten <= baseline.shanten
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct ActionLikelihood {
    probability: f64,
    legal_actions: usize,
}

impl ActionLikelihood {
    pub(super) fn ratio_to_uniform(self) -> f64 {
        self.probability * self.legal_actions as f64
    }
}

/// Likelihood of one observed discard under a monotone rational policy.
///
/// Candidate probability is proportional to one plus the number of legal
/// alternatives it dominates. This Borda distribution has full support and
/// introduces no fitted temperature or arbitrary optimal-action threshold.
pub(super) fn discard_likelihood(
    after_discard: Holding,
    discarded: Tile,
    visible: &[u8; TILE_KIND_COUNT],
    has_won: bool,
) -> Option<ActionLikelihood> {
    let before_discard = after_discard.after_draw(discarded)?;
    let candidates: Vec<_> = mask_tiles(before_discard.discard_mask())
        .filter_map(|tile| {
            before_discard
                .after_discard(tile)
                .map(|holding| (tile, HandPotential::evaluate(&holding, visible, has_won)))
        })
        .collect();
    let actual = candidates
        .iter()
        .find(|(tile, _)| *tile == discarded)
        .map(|(_, potential)| *potential)?;

    let scores: Vec<_> = candidates
        .iter()
        .map(|(_, candidate)| {
            1_u32
                + candidates
                    .iter()
                    .filter(|(_, other)| candidate.cmp_for(*other, has_won).is_gt())
                    .count() as u32
        })
        .collect();
    let total: u32 = scores.iter().sum();
    let actual_score = candidates
        .iter()
        .zip(scores)
        .find(|((_, candidate), _)| *candidate == actual)
        .map(|(_, score)| score)?;
    Some(ActionLikelihood {
        probability: f64::from(actual_score) / f64::from(total),
        legal_actions: candidates.len(),
    })
}

/// Likelihood ratio of a public discard sequence under a sampled current hand.
///
/// The sequence is reversed from the sampled current holding. A draw-discard
/// leaves the previous holding unchanged. A hand-discard retains an unknown
/// drawn tile, so the reverse pass enumerates that tile and merges identical
/// predecessor holdings. Public Hu, Pong, and Kong events start a new hand
/// revision in `Game`; callers therefore never reverse across a structural
/// hand change.
pub(super) fn history_likelihood_ratio(
    current: Holding,
    entries: &[RiverEntry],
    visible: &[u8; TILE_KIND_COUNT],
    has_won: bool,
) -> f64 {
    if entries.is_empty() {
        return 1.0;
    }

    let mut states = vec![(current, 1.0_f64)];
    for entry in entries.iter().rev() {
        let mut previous = BTreeMap::<Holding, f64>::new();
        for (after, state_weight) in states {
            let Some(action) = discard_likelihood(after, entry.tile, visible, has_won) else {
                continue;
            };
            let action_weight = state_weight * action.ratio_to_uniform();
            match entry.origin {
                DiscardOrigin::Draw | DiscardOrigin::ReplacementDraw | DiscardOrigin::Pong => {
                    *previous.entry(after).or_default() += action_weight;
                }
                DiscardOrigin::Hand => {
                    let mut draws = Vec::new();
                    let mut total_draw_weight = 0_u16;
                    for drawn in all_tiles() {
                        if drawn == entry.tile || after.unlocked_count(drawn) == 0 {
                            continue;
                        }
                        let draw_weight = u16::from(
                            TILE_COPIES
                                .saturating_add(1)
                                .saturating_sub(visible[drawn.index()].min(TILE_COPIES)),
                        );
                        if draw_weight == 0 {
                            continue;
                        }
                        let Some(before_draw) = after
                            .after_discard(drawn)
                            .and_then(|holding| holding.after_draw(entry.tile))
                        else {
                            continue;
                        };
                        draws.push((before_draw, draw_weight));
                        total_draw_weight += draw_weight;
                    }
                    if total_draw_weight == 0 {
                        continue;
                    }
                    for (before_draw, draw_weight) in draws {
                        *previous.entry(before_draw).or_default() +=
                            action_weight * f64::from(draw_weight) / f64::from(total_draw_weight);
                    }
                }
            }
        }
        if previous.is_empty() {
            return 0.0;
        }
        let mut ranked: Vec<_> = previous.into_iter().collect();
        let total_weight: f64 = ranked.iter().map(|(_, weight)| weight).sum();
        ranked.sort_unstable_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(MAX_REVERSE_HISTORY_STATES);
        let retained_weight: f64 = ranked.iter().map(|(_, weight)| weight).sum();
        if retained_weight > 0.0 && retained_weight < total_weight {
            let scale = total_weight / retained_weight;
            for (_, weight) in &mut ranked {
                *weight *= scale;
            }
        }
        states = ranked;
    }
    states.into_iter().map(|(_, weight)| weight).sum()
}

fn remaining_copies(mask: u32, visible: &[u8; TILE_KIND_COUNT]) -> u16 {
    mask_tiles(mask)
        .map(|tile| u16::from(TILE_COPIES.saturating_sub(visible[tile.index()].min(TILE_COPIES))))
        .sum()
}

fn structure_score(counts: &[u8; TILE_KIND_COUNT]) -> i32 {
    let pairs: i32 = counts.iter().map(|&count| i32::from(count >= 2)).sum();
    let triples: i32 = counts.iter().map(|&count| i32::from(count >= 3)).sum();
    let adjacent: i32 = (0..TILE_KIND_COUNT)
        .filter(|index| index % 9 < 8)
        .map(|index| i32::from(counts[index].min(counts[index + 1])))
        .sum();
    let gapped: i32 = (0..TILE_KIND_COUNT)
        .filter(|index| index % 9 < 7)
        .map(|index| i32::from(counts[index].min(counts[index + 2])))
        .sum();
    8 * triples + 4 * pairs + 2 * adjacent + gapped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::hand::DUMMY_MELD;
    use crate::types::Suit;

    fn tile(rank: u8) -> Tile {
        Tile::from_suit_rank(Suit::Characters, rank - 1).expect("test rank is valid")
    }

    #[test]
    fn discard_model_has_full_support() {
        let mut concealed = [0; TILE_KIND_COUNT];
        for rank in [1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9, 9] {
            concealed[tile(rank).index()] += 1;
        }
        let holding = Holding {
            concealed,
            locked: [0; TILE_KIND_COUNT],
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: None,
        };
        let visible = concealed;
        let probability = discard_likelihood(holding, tile(5), &visible, false)
            .expect("the discarded tile reconstructs a legal hand");
        assert!(probability.probability > 0.0 && probability.probability <= 1.0);
        assert!(probability.ratio_to_uniform() > 0.0);
    }

    #[test]
    fn history_model_reverses_hand_and_draw_discards() {
        let mut concealed = [0; TILE_KIND_COUNT];
        for rank in [1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 9, 9] {
            concealed[tile(rank).index()] += 1;
        }
        let holding = Holding {
            concealed,
            locked: [0; TILE_KIND_COUNT],
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: None,
        };
        let entries = [
            RiverEntry {
                tile: tile(6),
                origin: DiscardOrigin::Hand,
            },
            RiverEntry {
                tile: tile(5),
                origin: DiscardOrigin::Draw,
            },
        ];

        let ratio = history_likelihood_ratio(holding, &entries, &concealed, false);
        assert!(ratio.is_finite() && ratio > 0.0);
    }
}
