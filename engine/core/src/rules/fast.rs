use crate::game::{Batch, Game, GameError, LegalActions, Phase};
use crate::types::{PLAYER_COUNT, Seat, Suit, TILE_KIND_COUNT, Tile};
use crate::{ACTION_SPACE_SIZE, ActionId};

use super::batch_policy_actions_into;
use super::hand::{Holding, hand_structure_score, suit_structure_score};

/// Sentinel written for terminal batch slots by
/// [`Batch::simple_rule_actions_into`].
pub const SIMPLE_RULE_ACTION_TERMINAL: u8 = u8::MAX;

impl Game {
    /// Chooses a deterministic, public-information-only baseline action.
    ///
    /// This policy is intended for cold-start opponents and regression tests,
    /// not as a strong Mahjong agent. It uses the current actor's concealed
    /// hand and exchange selection plus public winning tiles, melds, and
    /// discards. It never reads opponents' concealed hands or the wall.
    pub fn simple_rule_action(&self) -> Option<ActionId> {
        let legal = self.legal_actions()?;
        let actor = legal.decision.actor;
        let action = match legal.decision.phase {
            Phase::Exchange => choose_exchange(self, actor, legal.exchange_mask),
            Phase::ChooseMissing => choose_missing(self, actor),
            Phase::Turn => choose_turn(self, actor, &legal),
            Phase::HuResponse | Phase::MeldResponse => choose_response(&legal),
            Phase::Finished => return None,
        };
        debug_assert!(
            self.legal_action_mask()
                .is_some_and(|mask| mask.contains(action))
        );
        Some(action)
    }
}

fn choose_response(legal: &LegalActions) -> ActionId {
    if legal.can_hu {
        ActionId::HU
    } else if legal.can_exposed_kong {
        ActionId::EXPOSED_KONG
    } else if legal.can_pong {
        ActionId::PONG
    } else {
        ActionId::PASS
    }
}

impl Batch {
    /// Writes one deterministic simple-rule action per environment.
    ///
    /// Active slots receive an action in `0..ACTION_SPACE_SIZE`. Terminal
    /// slots receive [`SIMPLE_RULE_ACTION_TERMINAL`].
    pub fn simple_rule_actions_into(&self, output: &mut [u8]) -> Result<(), GameError> {
        batch_policy_actions_into(
            self,
            None,
            output,
            SIMPLE_RULE_ACTION_TERMINAL,
            Game::simple_rule_action,
        )
    }

    /// Writes simple-rule actions only where the byte mask is one.
    ///
    /// Disabled output rows are left untouched. The mask is validated before
    /// any action is computed or written.
    pub fn simple_rule_actions_masked_into(
        &self,
        enabled: &[u8],
        output: &mut [u8],
    ) -> Result<(), GameError> {
        batch_policy_actions_into(
            self,
            Some(enabled),
            output,
            SIMPLE_RULE_ACTION_TERMINAL,
            Game::simple_rule_action,
        )
    }
}

fn choose_exchange(game: &Game, actor: Seat, legal_mask: u32) -> ActionId {
    let mut counts = *game.concealed(actor);
    let selected_count: u8 = game.exchange_selection(actor).iter().copied().sum();
    for (count, selected) in counts
        .iter_mut()
        .zip(game.exchange_selection(actor).iter().copied())
    {
        *count = count.saturating_sub(selected);
    }

    let suit = if selected_count > 0 {
        mask_tiles(legal_mask)
            .next()
            .map(Tile::suit)
            .expect("an exchange decision has a legal tile")
    } else {
        Suit::ALL
            .into_iter()
            .filter(|&candidate| legal_mask & candidate.mask() != 0)
            .min_by_key(|&candidate| {
                (
                    suit_structure_score(&counts, candidate),
                    suit_count(&counts, candidate),
                    candidate as u8,
                )
            })
            .expect("an exchange decision has a legal suit")
    };

    let mut best: Option<(i32, u8, u8)> = None;
    for tile in mask_tiles(legal_mask & suit.mask()) {
        let mut remaining = counts;
        remaining[tile.index()] -= 1;
        let candidate = (
            hand_structure_score(&remaining),
            edge_distance(tile),
            u8::MAX - tile.as_u8(),
        );
        if best.is_none_or(|current| candidate > current) {
            best = Some(candidate);
        }
    }
    let tile_index = u8::MAX - best.expect("an exchange decision has a legal tile").2;
    ActionId::select_exchange_tile(Tile::new(tile_index).expect("chosen tile index is valid"))
}

fn choose_missing(game: &Game, actor: Seat) -> ActionId {
    let counts = game.concealed(actor);
    let suit = Suit::ALL
        .into_iter()
        .min_by_key(|&candidate| {
            (
                suit_count(counts, candidate),
                suit_structure_score(counts, candidate),
                candidate as u8,
            )
        })
        .expect("three suits are always available");
    ActionId::choose_missing(suit)
}

fn choose_turn(game: &Game, actor: Seat, legal: &LegalActions) -> ActionId {
    if legal.can_hu {
        return ActionId::HU;
    }
    if let Some(tile) = first_mask_tile(legal.concealed_kong_mask) {
        return ActionId::concealed_kong(tile);
    }
    if let Some(tile) = first_mask_tile(legal.added_kong_mask) {
        return ActionId::added_kong(tile);
    }

    let holding = Holding::from_game(game, actor);
    let exposure = public_exposure(game, actor);
    let mut best: Option<(i8, u16, i32, u8, u8, u8)> = None;
    for tile in mask_tiles(legal.discard_mask) {
        let remaining = holding
            .after_discard(tile)
            .expect("the legal discard mask contains an active tile");
        let counts = remaining
            .evaluation_counts()
            .expect("a legal holding has a consistent stable base");
        let analysis = remaining.analysis();
        let candidate = (
            -analysis.shanten,
            remaining_improving_copies(analysis.improving_tiles, &exposure),
            hand_structure_score(&counts),
            exposure[tile.index()],
            edge_distance(tile),
            u8::MAX - tile.as_u8(),
        );
        if best.is_none_or(|current| candidate > current) {
            best = Some(candidate);
        }
    }
    let tile_index = u8::MAX
        - best
            .expect("a turn always has a discard or special action")
            .5;
    ActionId::discard(Tile::new(tile_index).expect("chosen tile index is valid"))
}

fn remaining_improving_copies(mask: u32, exposure: &[u8; TILE_KIND_COUNT]) -> u16 {
    mask_tiles(mask)
        .map(|tile| u16::from(4_u8.saturating_sub(exposure[tile.index()].min(4))))
        .sum()
}

fn public_exposure(game: &Game, actor: Seat) -> [u8; TILE_KIND_COUNT] {
    game.visible_tile_counts(actor)
}

fn suit_count(counts: &[u8; TILE_KIND_COUNT], suit: Suit) -> u8 {
    let start = suit as usize * 9;
    counts[start..start + 9].iter().copied().sum()
}

fn edge_distance(tile: Tile) -> u8 {
    tile.rank().abs_diff(4)
}

fn first_mask_tile(mask: u32) -> Option<Tile> {
    (mask != 0).then(|| {
        Tile::new(mask.trailing_zeros() as u8).expect("tile masks only use the low 27 bits")
    })
}

fn mask_tiles(mut mask: u32) -> impl Iterator<Item = Tile> {
    core::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let index = mask.trailing_zeros() as u8;
        mask &= mask - 1;
        Some(Tile::new(index).expect("tile masks only use the low 27 bits"))
    })
}

const _: () = assert!(SIMPLE_RULE_ACTION_TERMINAL as usize >= ACTION_SPACE_SIZE);
const _: () = assert!(PLAYER_COUNT == 4);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structure_score_prefers_complete_shapes() {
        let mut isolated = [0; TILE_KIND_COUNT];
        isolated[0] = 1;
        isolated[3] = 1;
        isolated[8] = 1;
        let mut sequence = [0; TILE_KIND_COUNT];
        sequence[0..3].fill(1);
        assert!(hand_structure_score(&sequence) > hand_structure_score(&isolated));

        let mut pair = [0; TILE_KIND_COUNT];
        pair[4] = 2;
        let mut separate = [0; TILE_KIND_COUNT];
        separate[0] = 1;
        separate[8] = 1;
        assert!(hand_structure_score(&pair) > hand_structure_score(&separate));
    }

    #[test]
    fn simple_rule_policy_completes_games_with_legal_actions() {
        for seed in 0..128 {
            let mut game = Game::new(seed);
            for _ in 0..512 {
                let Some(action) = game.simple_rule_action() else {
                    break;
                };
                assert!(
                    game.legal_action_mask()
                        .expect("active game has a mask")
                        .contains(action)
                );
                game.step_id(action).expect("rule action is legal");
            }
            assert_eq!(game.phase(), Phase::Finished, "seed {seed} did not finish");
        }
    }

    #[test]
    fn batch_actions_match_each_game() {
        let batch = Batch::new(128, 23);
        let mut actions = [0; 128];
        batch
            .simple_rule_actions_into(&mut actions)
            .expect("batch output has the right length");
        for (game, action) in batch.games().iter().zip(actions) {
            assert_eq!(
                usize::from(action),
                game.simple_rule_action()
                    .expect("new games are active")
                    .index()
            );
        }
    }

    #[test]
    fn masked_batch_actions_touch_only_enabled_rows() {
        let batch = Batch::new(128, 29);
        let mut expected = [0; 128];
        batch
            .simple_rule_actions_into(&mut expected)
            .expect("batch output has the right length");
        let mut enabled = [0; 128];
        for (index, value) in enabled.iter_mut().enumerate() {
            *value = u8::from(index % 3 == 0);
        }
        let mut actions = [0xA5; 128];
        batch
            .simple_rule_actions_masked_into(&enabled, &mut actions)
            .expect("masked batch output is valid");
        for index in 0..128 {
            if enabled[index] == 1 {
                assert_eq!(actions[index], expected[index]);
            } else {
                assert_eq!(actions[index], 0xA5);
            }
        }

        let before = actions;
        enabled[1] = 2;
        assert!(matches!(
            batch.simple_rule_actions_masked_into(&enabled, &mut actions),
            Err(GameError::InvalidAction)
        ));
        assert_eq!(actions, before);
        assert!(matches!(
            batch.simple_rule_actions_masked_into(&enabled[..127], &mut actions),
            Err(GameError::BatchLength)
        ));
    }
}
