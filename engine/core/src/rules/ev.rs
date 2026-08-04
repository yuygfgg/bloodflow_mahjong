use core::cmp::Ordering;

use crate::game::{Batch, Game, GameError, LegalActions, Phase};
use crate::types::{PLAYER_COUNT, Seat, Suit, TILE_KIND_COUNT, Tile};
use crate::{ACTION_SPACE_SIZE, ActionId, WinFlags};

use super::batch_policy_actions_into;
#[cfg(test)]
use super::hand::DUMMY_MELD;
use super::hand::{Holding, all_tiles, hand_structure_score, mask_tiles};
use super::opening;

/// Sentinel written for terminal batch slots by [`Batch::rule_ev_actions_into`].
pub const RULE_EV_ACTION_TERMINAL: u8 = u8::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuleEvDefense {
    None,
    Heuristic,
}

const WAIT_MULTIPLIER_CAP: u32 = 256;
const PONG_UKEIRE_MARGIN: u16 = 2;
const RISK_SCALE: u64 = 1_000_000;
const DEFENSE_RISK_WEIGHT: u64 = 15;
const MAX_SEARCH_DEPTH: u8 = 3;
const DEEP_SEARCH_DISCOUNT: u64 = 1;

/// Compute budget for deterministic rule-EV lookahead.
///
/// Depth zero uses static hand evaluation. Each additional level enumerates
/// every live improving tile and the best discard which follows that draw.
/// The search never samples or reads the authoritative wall.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleEvConfig {
    search_depth: u8,
    defense: RuleEvDefense,
}

impl RuleEvConfig {
    pub const FAST: Self = Self {
        search_depth: 0,
        defense: RuleEvDefense::Heuristic,
    };
    pub const STANDARD: Self = Self {
        search_depth: 1,
        defense: RuleEvDefense::Heuristic,
    };

    pub const fn with_search_depth(search_depth: u8) -> Option<Self> {
        if search_depth <= MAX_SEARCH_DEPTH {
            Some(Self {
                search_depth,
                defense: RuleEvDefense::Heuristic,
            })
        } else {
            None
        }
    }

    pub const fn search_depth(self) -> u8 {
        self.search_depth
    }

    pub const fn with_defense(mut self, defense: RuleEvDefense) -> Self {
        self.defense = defense;
        self
    }

    pub const fn defense(self) -> RuleEvDefense {
        self.defense
    }
}

impl Default for RuleEvConfig {
    fn default() -> Self {
        Self::STANDARD
    }
}

impl Game {
    /// Chooses a deterministic action using only the actor's private state and
    /// information which is public to every player.
    pub fn rule_ev_action(&self) -> Option<ActionId> {
        self.rule_ev_action_with_config(RuleEvConfig::STANDARD)
    }

    /// Chooses a rule-EV action with an explicit deterministic search budget.
    pub fn rule_ev_action_with_config(&self, config: RuleEvConfig) -> Option<ActionId> {
        let legal = self.legal_actions()?;
        let actor = legal.decision.actor;
        let action = match legal.decision.phase {
            Phase::Exchange => opening::choose_exchange(
                self.concealed(actor),
                self.exchange_selection(actor),
                legal.exchange_mask,
            ),
            Phase::ChooseMissing => opening::choose_missing(self.concealed(actor)),
            Phase::Turn => choose_turn(self, actor, &legal, config),
            Phase::HuResponse => ActionId::HU,
            Phase::MeldResponse => choose_meld_response(self, actor, &legal, config),
            Phase::Finished => return None,
        };
        debug_assert!(
            self.legal_action_mask()
                .is_some_and(|mask| mask.contains(action))
        );
        Some(action)
    }
}

impl Batch {
    /// Writes one rule-EV action per environment.
    pub fn rule_ev_actions_into(&self, output: &mut [u8]) -> Result<(), GameError> {
        self.rule_ev_actions_with_config_into(RuleEvConfig::STANDARD, output)
    }

    /// Writes one configured rule-EV action per environment.
    pub fn rule_ev_actions_with_config_into(
        &self,
        config: RuleEvConfig,
        output: &mut [u8],
    ) -> Result<(), GameError> {
        batch_policy_actions_into(self, None, output, RULE_EV_ACTION_TERMINAL, |game| {
            game.rule_ev_action_with_config(config)
        })
    }

    /// Writes rule-EV actions only where `enabled` is one.
    pub fn rule_ev_actions_masked_into(
        &self,
        enabled: &[u8],
        output: &mut [u8],
    ) -> Result<(), GameError> {
        self.rule_ev_actions_masked_with_config_into(enabled, RuleEvConfig::STANDARD, output)
    }

    /// Writes configured rule-EV actions only where `enabled` is one.
    pub fn rule_ev_actions_masked_with_config_into(
        &self,
        enabled: &[u8],
        config: RuleEvConfig,
        output: &mut [u8],
    ) -> Result<(), GameError> {
        batch_policy_actions_into(
            self,
            Some(enabled),
            output,
            RULE_EV_ACTION_TERMINAL,
            |game| game.rule_ev_action_with_config(config),
        )
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct WaitQuality {
    weighted_value: u64,
    live_copies: u16,
    max_multiplier: u32,
    distinct_tiles: u8,
}

impl WaitQuality {
    fn cmp_quality(self, other: Self) -> Ordering {
        (
            self.weighted_value,
            self.live_copies,
            self.max_multiplier,
            self.distinct_tiles,
        )
            .cmp(&(
                other.weighted_value,
                other.live_copies,
                other.max_multiplier,
                other.distinct_tiles,
            ))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HandQuality {
    shanten: i8,
    live_improvements: u16,
    one_draw_value: u64,
    waits: WaitQuality,
    structure: i32,
}

fn hand_quality(
    holding: &Holding,
    exposure: &[u8; TILE_KIND_COUNT],
    search_depth: u8,
) -> HandQuality {
    let mut quality = static_hand_quality(holding, exposure);
    if search_depth != 0 && (1..=2).contains(&quality.shanten) {
        quality.one_draw_value = one_draw_value(
            holding,
            exposure,
            quality.shanten,
            holding.analysis().improving_tiles,
            search_depth,
        );
    }
    quality
}

fn static_hand_quality(holding: &Holding, exposure: &[u8; TILE_KIND_COUNT]) -> HandQuality {
    let analysis = holding.analysis();
    let waits = if analysis.shanten <= 0 {
        wait_quality(holding, exposure)
    } else {
        WaitQuality::default()
    };
    HandQuality {
        shanten: analysis.shanten,
        live_improvements: remaining_copies(analysis.improving_tiles, exposure),
        one_draw_value: 0,
        waits,
        structure: hand_structure_score(&holding.evaluation_counts().unwrap_or(holding.concealed)),
    }
}

fn one_draw_value(
    holding: &Holding,
    exposure: &[u8; TILE_KIND_COUNT],
    current_shanten: i8,
    improving_tiles: u32,
    search_depth: u8,
) -> u64 {
    debug_assert!(search_depth != 0);
    let mut expected = 0_u64;
    let mut augmented = *holding;
    for tile in mask_tiles(improving_tiles) {
        let copies = u64::from(4_u8.saturating_sub(exposure[tile.index()].min(4)));
        if copies == 0 || augmented.concealed[tile.index()] >= 4 {
            continue;
        }
        augmented.concealed[tile.index()] += 1;
        let mut next_exposure = *exposure;
        next_exposure[tile.index()] = next_exposure[tile.index()].saturating_add(1).min(4);

        let direct_win = augmented.evaluate_win(Some(tile), WinFlags::NONE);
        let continuation = if let Some(win) = direct_win {
            2_000 + u64::from(win.multiplier.min(WAIT_MULTIPLIER_CAP)) * 200
        } else {
            let mut best_after_discard = 0_u64;
            for discard in mask_tiles(augmented.discard_mask()) {
                let Some(after) = augmented.after_discard(discard) else {
                    continue;
                };
                let quality = static_hand_quality(&after, &next_exposure);
                if quality.shanten < current_shanten {
                    let mut value = static_attack_value(quality);
                    if search_depth > 1 && (0..=2).contains(&quality.shanten) {
                        let analysis = after.analysis();
                        value = value.saturating_add(
                            one_draw_value(
                                &after,
                                &next_exposure,
                                quality.shanten,
                                analysis.improving_tiles,
                                search_depth - 1,
                            ) / DEEP_SEARCH_DISCOUNT,
                        );
                    }
                    best_after_discard = best_after_discard.max(value);
                }
            }
            best_after_discard
        };
        if continuation != 0 {
            expected = expected.saturating_add(copies * continuation);
        }
        augmented.concealed[tile.index()] -= 1;
    }
    expected
}

fn static_attack_value(quality: HandQuality) -> u64 {
    if quality.shanten <= 0 {
        500 + quality.waits.weighted_value * 20
            + u64::from(quality.waits.live_copies) * 10
            + u64::from(quality.waits.max_multiplier) * 5
    } else {
        u64::from(quality.live_improvements) * 40 + quality.structure.max(0) as u64 / 4
    }
}

fn wait_quality(holding: &Holding, exposure: &[u8; TILE_KIND_COUNT]) -> WaitQuality {
    if holding.missing_count() != 0 {
        return WaitQuality::default();
    }
    let mut result = WaitQuality::default();
    for tile in all_tiles() {
        if holding.missing == Some(tile.suit()) {
            continue;
        }
        let copies = 4_u8.saturating_sub(exposure[tile.index()].min(4));
        if copies == 0 {
            continue;
        }
        let Some(augmented) = holding.after_draw(tile) else {
            continue;
        };
        if let Some(evaluation) = augmented.evaluate_win(Some(tile), WinFlags::NONE) {
            let multiplier = evaluation.multiplier.min(WAIT_MULTIPLIER_CAP);
            result.weighted_value += u64::from(copies) * u64::from(multiplier);
            result.live_copies += u16::from(copies);
            result.max_multiplier = result.max_multiplier.max(multiplier);
            result.distinct_tiles += 1;
        }
    }
    result
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct DiscardDanger {
    expected_loss: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DiscardCandidate {
    tile: Tile,
    quality: HandQuality,
    danger: DiscardDanger,
    exposure: u8,
}

fn best_discard(
    game: &Game,
    actor: Seat,
    holding: Holding,
    discard_mask: u32,
    exposure: &[u8; TILE_KIND_COUNT],
    config: RuleEvConfig,
) -> Option<DiscardCandidate> {
    let has_won = game.has_won(actor);
    let mut best = None;
    for tile in mask_tiles(discard_mask) {
        let Some(remaining) = holding.after_discard(tile) else {
            continue;
        };
        let candidate = DiscardCandidate {
            tile,
            quality: hand_quality(&remaining, exposure, config.search_depth),
            danger: discard_danger(game, actor, tile, exposure, config.defense),
            exposure: exposure[tile.index()],
        };
        if best.is_none_or(|current| discard_better(candidate, current, has_won)) {
            best = Some(candidate);
        }
    }
    best
}

fn discard_better(candidate: DiscardCandidate, current: DiscardCandidate, has_won: bool) -> bool {
    if !has_won && candidate.quality.shanten != current.quality.shanten {
        return candidate.quality.shanten < current.quality.shanten;
    }
    let candidate_attack = discard_attack_value(candidate.quality, has_won);
    let current_attack = discard_attack_value(current.quality, has_won);
    let candidate_utility = candidate_attack as i128 * i128::from(RISK_SCALE)
        - i128::from(candidate.danger.expected_loss) * i128::from(DEFENSE_RISK_WEIGHT);
    let current_utility = current_attack as i128 * i128::from(RISK_SCALE)
        - i128::from(current.danger.expected_loss) * i128::from(DEFENSE_RISK_WEIGHT);
    if candidate_utility != current_utility {
        return candidate_utility > current_utility;
    }
    (
        candidate.quality.structure,
        candidate.exposure,
        edge_distance(candidate.tile),
        u8::MAX - candidate.tile.as_u8(),
    ) > (
        current.quality.structure,
        current.exposure,
        edge_distance(current.tile),
        u8::MAX - current.tile.as_u8(),
    )
}

fn discard_attack_value(quality: HandQuality, has_won: bool) -> u64 {
    if has_won || quality.shanten <= 0 {
        quality.waits.weighted_value * 10
            + u64::from(quality.waits.live_copies) * 5
            + u64::from(quality.waits.max_multiplier)
    } else {
        u64::from(quality.live_improvements) * 100
            + quality.one_draw_value
            + quality.structure.max(0) as u64 / 4
    }
}

fn discard_danger(
    game: &Game,
    actor: Seat,
    tile: Tile,
    exposure: &[u8; TILE_KIND_COUNT],
    defense: RuleEvDefense,
) -> DiscardDanger {
    match defense {
        RuleEvDefense::None => DiscardDanger::default(),
        RuleEvDefense::Heuristic => heuristic_discard_danger(game, actor, tile, exposure),
    }
}

fn heuristic_discard_danger(
    game: &Game,
    actor: Seat,
    tile: Tile,
    exposure: &[u8; TILE_KIND_COUNT],
) -> DiscardDanger {
    const DEFENSE_ONSET_DISCARDS: f64 = 24.0;
    const DEFENSE_FULL_DISCARDS: f64 = 56.0;
    const RANK_FACTORS: [f64; PLAYER_COUNT] = [1.40, 1.00, 0.65, 0.35];

    let discards = game.discards().count() as f64;
    let progress = ((discards - DEFENSE_ONSET_DISCARDS)
        / (DEFENSE_FULL_DISCARDS - DEFENSE_ONSET_DISCARDS))
        .clamp(0.0, 1.0);
    let rank = game
        .rankings()
        .iter()
        .position(|&seat| seat == actor)
        .expect("the actor has a rank");
    let known_risk_factor = RANK_FACTORS[rank];
    let uncertain_risk_factor = progress * known_risk_factor;

    let mut danger = DiscardDanger::default();
    for opponent in Seat::ALL {
        if opponent == actor || game.missing_suit(opponent) == Some(tile.suit()) {
            continue;
        }
        let posterior =
            opponent_win_posterior(game, opponent, tile, exposure) * uncertain_risk_factor;
        let multiplier = expected_discard_multiplier(game, opponent);
        danger.expected_loss = danger
            .expected_loss
            .saturating_add((posterior * f64::from(multiplier) * RISK_SCALE as f64) as u64);
    }
    danger
}

fn opponent_win_posterior(
    game: &Game,
    opponent: Seat,
    tile: Tile,
    exposure: &[u8; TILE_KIND_COUNT],
) -> f64 {
    let public_win_tiles = game.public_win_tiles(opponent);
    let hidden = game.concealed_len(opponent).saturating_sub(
        public_win_tiles
            .iter()
            .map(|&count| usize::from(count))
            .sum(),
    );
    if hidden == 0 {
        return 0.0;
    }
    let unknown_total: usize = exposure
        .iter()
        .map(|&count| usize::from(4_u8.saturating_sub(count.min(4))))
        .sum();
    if unknown_total == 0 {
        return 0.0;
    }

    let available =
        |candidate: Tile| usize::from(4_u8.saturating_sub(exposure[candidate.index()].min(4)));
    let same = available(tile);
    let mut completion = contains_probability(same, 1, hidden, unknown_total) * 0.34;
    if same >= 2 {
        completion += contains_probability(same, 2, hidden, unknown_total) * 0.26;
    }
    let rank = tile.rank();
    let start_min = rank.saturating_sub(2);
    let start_max = rank.min(6);
    for start in start_min..=start_max {
        if rank < start || rank > start + 2 {
            continue;
        }
        let mut required = [0_usize; 2];
        let mut length = 0;
        for offset in 0..3 {
            let candidate = Tile::from_suit_rank(tile.suit(), start + offset)
                .expect("sequence ranks stay within one suit");
            if candidate == tile {
                continue;
            }
            required[length] = available(candidate);
            length += 1;
        }
        if length == 2 {
            completion +=
                contains_probability_pair(required[0], required[1], hidden, unknown_total) * 0.20;
        }
    }

    let total_discards = game.discards().count() as f64;
    let opponent_discards = game
        .discards()
        .filter(|(seat, _)| *seat == opponent)
        .count() as f64;
    let melds = game.meld_count(opponent) as f64;
    let readiness_logit = -3.7 + 0.085 * total_discards + 0.12 * opponent_discards + 0.48 * melds;
    let readiness = 1.0 / (1.0 + (-readiness_logit).exp());
    let suit_bonus = if game.meld_count(opponent) >= 2
        && (0..game.meld_count(opponent)).all(|index| {
            game.meld(opponent, index)
                .is_some_and(|meld| meld.tile.suit() == tile.suit())
        }) {
        1.35
    } else {
        1.0
    };
    (completion * readiness * suit_bonus).clamp(0.0, 0.85)
}

fn contains_probability(available: usize, required: usize, hand: usize, total: usize) -> f64 {
    if available < required || hand < required || total < required {
        return 0.0;
    }
    let denominator = binomial(total, required) as f64;
    if denominator == 0.0 {
        return 0.0;
    }
    let expected_subsets =
        binomial(available, required) as f64 * binomial(hand, required) as f64 / denominator;
    // Several physical-copy subsets can satisfy the same event. A Poisson
    // union gives a smooth lower-bound posterior without enumerating hands.
    1.0 - (-expected_subsets).exp()
}

fn contains_probability_pair(
    first_available: usize,
    second_available: usize,
    hand: usize,
    total: usize,
) -> f64 {
    if first_available == 0 || second_available == 0 || hand < 2 || total < 2 {
        return 0.0;
    }
    let pair_inclusion = (hand * (hand - 1)) as f64 / (total * (total - 1)) as f64;
    (pair_inclusion * first_available as f64 * second_available as f64).min(1.0)
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    (0..k).fold(1_usize, |value, index| value * (n - index) / (index + 1))
}

fn expected_discard_multiplier(game: &Game, opponent: Seat) -> u32 {
    // Hidden hands are estimated from public melds only.
    1 + (game.meld_count(opponent) as u32).min(3)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KongCandidate {
    action: ActionId,
    quality: HandQuality,
    immediate_value: i64,
    tile: Tile,
}

fn choose_turn(game: &Game, actor: Seat, legal: &LegalActions, config: RuleEvConfig) -> ActionId {
    if legal.can_hu {
        return ActionId::HU;
    }

    let holding = Holding::from_game(game, actor);
    let exposure = public_exposure(game, actor);
    let baseline = best_discard(game, actor, holding, legal.discard_mask, &exposure, config)
        .expect("a turn has a legal discard");
    let seven_pairs = seven_pairs_shanten(
        &holding.evaluation_counts().unwrap_or(holding.concealed),
        holding.missing,
    );
    let mut best_kong = None;

    if seven_pairs > 1 {
        for tile in mask_tiles(legal.concealed_kong_mask) {
            let Some(after) = holding.after_concealed_kong(tile, actor) else {
                continue;
            };
            let quality = hand_quality(&after, &exposure, config.search_depth);
            if kong_preserves_progress(quality, baseline.quality, game.has_won(actor)) {
                let candidate = KongCandidate {
                    action: ActionId::concealed_kong(tile),
                    quality,
                    immediate_value: Seat::ALL
                        .into_iter()
                        .filter(|&seat| seat != actor)
                        .map(|seat| game.score(seat).clamp(0, 200))
                        .sum(),
                    tile,
                };
                if best_kong
                    .is_none_or(|current| kong_better(candidate, current, game.has_won(actor)))
                {
                    best_kong = Some(candidate);
                }
            }
        }
    }

    for tile in mask_tiles(legal.added_kong_mask) {
        let Some(after) = holding.after_added_kong(tile) else {
            continue;
        };
        let quality = hand_quality(&after, &exposure, config.search_depth);
        if kong_preserves_progress(quality, baseline.quality, game.has_won(actor)) {
            let candidate = KongCandidate {
                action: ActionId::added_kong(tile),
                quality,
                immediate_value: Seat::ALL
                    .into_iter()
                    .filter(|&seat| seat != actor)
                    .map(|seat| game.score(seat).clamp(0, 100))
                    .sum(),
                tile,
            };
            if best_kong.is_none_or(|current| kong_better(candidate, current, game.has_won(actor)))
            {
                best_kong = Some(candidate);
            }
        }
    }

    best_kong.map_or_else(|| ActionId::discard(baseline.tile), |kong| kong.action)
}

fn kong_preserves_progress(candidate: HandQuality, baseline: HandQuality, has_won: bool) -> bool {
    if has_won {
        candidate.waits.cmp_quality(baseline.waits) != Ordering::Less
    } else {
        candidate.shanten <= baseline.shanten
    }
}

fn kong_better(candidate: KongCandidate, current: KongCandidate, has_won: bool) -> bool {
    let progress = if has_won {
        candidate.quality.waits.cmp_quality(current.quality.waits)
    } else {
        current.quality.shanten.cmp(&candidate.quality.shanten)
    };
    progress == Ordering::Greater
        || (progress == Ordering::Equal
            && (candidate.immediate_value, u8::MAX - candidate.tile.as_u8())
                > (current.immediate_value, u8::MAX - current.tile.as_u8()))
}

fn choose_meld_response(
    game: &Game,
    actor: Seat,
    legal: &LegalActions,
    config: RuleEvConfig,
) -> ActionId {
    let Some((source, tile)) = game.discards().last() else {
        return ActionId::PASS;
    };
    let holding = Holding::from_game(game, actor);
    let exposure = public_exposure(game, actor);
    let pass = hand_quality(&holding, &exposure, config.search_depth);
    let seven_pairs = seven_pairs_shanten(
        &holding.evaluation_counts().unwrap_or(holding.concealed),
        holding.missing,
    );

    if legal.can_exposed_kong
        && seven_pairs > 1
        && let Some(after) = holding.after_exposed_kong(tile, source)
    {
        let kong = hand_quality(&after, &exposure, config.search_depth);
        if meld_kong_is_worthwhile(pass, kong, game.has_won(actor)) {
            return ActionId::EXPOSED_KONG;
        }
    }

    if legal.can_pong
        && seven_pairs > 1
        && let Some(after_pong) = holding.after_pong(tile, source)
        && let Some(after) = best_discard(
            game,
            actor,
            after_pong,
            after_pong.discard_mask(),
            &exposure,
            config,
        )
        && pong_is_worthwhile(pass, after.quality, game.has_won(actor))
    {
        return ActionId::PONG;
    }
    ActionId::PASS
}

fn meld_kong_is_worthwhile(pass: HandQuality, kong: HandQuality, has_won: bool) -> bool {
    if has_won {
        kong.waits.cmp_quality(pass.waits) != Ordering::Less
    } else {
        kong.shanten <= pass.shanten
    }
}

fn pong_is_worthwhile(pass: HandQuality, pong: HandQuality, has_won: bool) -> bool {
    if has_won {
        return pong.waits.cmp_quality(pass.waits) == Ordering::Greater;
    }
    if pong.shanten != pass.shanten {
        return pong.shanten < pass.shanten;
    }
    if pass.shanten <= 0 {
        pong.waits.cmp_quality(pass.waits) == Ordering::Greater
    } else {
        pong.live_improvements >= pass.live_improvements.saturating_add(PONG_UKEIRE_MARGIN)
    }
}

fn seven_pairs_shanten(counts: &[u8; TILE_KIND_COUNT], missing: Option<Suit>) -> i8 {
    let mut pairs = 0_i8;
    let mut pair_slots = 0_i8;
    for tile in all_tiles() {
        if missing == Some(tile.suit()) {
            continue;
        }
        let count = counts[tile.index()].min(4);
        pairs += i8::try_from(count / 2).expect("a tile has at most two pairs");
        pair_slots += i8::try_from(count.div_ceil(2)).expect("a tile has at most two pair slots");
    }
    6 - pairs + (7 - pair_slots).max(0)
}

fn public_exposure(game: &Game, actor: Seat) -> [u8; TILE_KIND_COUNT] {
    game.visible_tile_counts(actor)
}

fn remaining_copies(mask: u32, exposure: &[u8; TILE_KIND_COUNT]) -> u16 {
    mask_tiles(mask)
        .map(|tile| u16::from(4_u8.saturating_sub(exposure[tile.index()].min(4))))
        .sum()
}

fn edge_distance(tile: Tile) -> u8 {
    tile.rank().abs_diff(4)
}

const _: () = assert!(RULE_EV_ACTION_TERMINAL as usize >= ACTION_SPACE_SIZE);
const _: () = assert!(PLAYER_COUNT == 4);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{
        MELD_OBSERVATION_WIDTH, META_OBSERVATION_WIDTH, RIVER_OBSERVATION_WIDTH,
        TILE_OBSERVATION_WIDTH,
    };

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank - 1).expect("test rank is valid")
    }

    fn add_sequence(counts: &mut [u8; TILE_KIND_COUNT], suit: Suit, first_rank: u8) {
        for rank in first_rank..first_rank + 3 {
            counts[tile(suit, rank).index()] += 1;
        }
    }

    fn holding(counts: [u8; TILE_KIND_COUNT]) -> Holding {
        Holding {
            concealed: counts,
            locked: [0; TILE_KIND_COUNT],
            win_base: [0; TILE_KIND_COUNT],
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: None,
            has_won: false,
        }
    }

    fn observation_for(
        game: &Game,
        viewer: Seat,
    ) -> (
        [u8; TILE_OBSERVATION_WIDTH],
        [u8; MELD_OBSERVATION_WIDTH],
        [u8; RIVER_OBSERVATION_WIDTH],
        [i32; META_OBSERVATION_WIDTH],
    ) {
        let mut observation = (
            [0; TILE_OBSERVATION_WIDTH],
            [0; MELD_OBSERVATION_WIDTH],
            [0; RIVER_OBSERVATION_WIDTH],
            [0; META_OBSERVATION_WIDTH],
        );
        game.observation_into(
            viewer,
            &mut observation.0,
            &mut observation.1,
            &mut observation.2,
            &mut observation.3,
        )
        .expect("test observation buffers have the required dimensions");
        observation
    }

    #[test]
    fn reentry_wait_requires_the_new_tile_in_a_winning_subset() {
        let mut counts = [0; TILE_KIND_COUNT];
        add_sequence(&mut counts, Suit::Characters, 1);
        add_sequence(&mut counts, Suit::Characters, 4);
        add_sequence(&mut counts, Suit::Bamboo, 1);
        add_sequence(&mut counts, Suit::Bamboo, 4);
        let pair = tile(Suit::Bamboo, 9);
        counts[pair.index()] = 2;
        let mut exposure = counts;
        let quality = wait_quality(&holding(counts), &exposure);
        assert!(quality.live_copies >= 2);
        assert!(quality.weighted_value > 0);

        exposure[pair.index()] = 4;
        let exhausted = wait_quality(&holding(counts), &exposure);
        assert!(exhausted.live_copies < quality.live_copies);
    }

    #[test]
    fn discard_meld_transforms_never_consume_locked_tiles() {
        let pong_tile = tile(Suit::Characters, 5);
        let mut counts = [0; TILE_KIND_COUNT];
        counts[pong_tile.index()] = 3;
        counts[tile(Suit::Bamboo, 1).index()] = 1;
        let mut state = holding(counts);
        state.locked[pong_tile.index()] = 2;
        assert!(state.after_pong(pong_tile, Seat::EAST).is_none());
        assert!(state.after_exposed_kong(pong_tile, Seat::EAST).is_none());

        state.concealed[pong_tile.index()] = 4;
        state.locked[pong_tile.index()] = 1;
        let kong = state
            .after_exposed_kong(pong_tile, Seat::EAST)
            .expect("three unlocked matching tiles form an exposed Kong");
        assert_eq!(kong.concealed[pong_tile.index()], 1);
        assert_eq!(kong.locked[pong_tile.index()], 1);
    }

    #[test]
    fn seven_pairs_route_counts_a_quad_as_two_pairs() {
        let mut counts = [0; TILE_KIND_COUNT];
        counts[tile(Suit::Characters, 2).index()] = 4;
        for rank in 3..=7 {
            counts[tile(Suit::Characters, rank).index()] = 2;
        }
        assert_eq!(seven_pairs_shanten(&counts, None), -1);
        assert!(seven_pairs_shanten(&counts, Some(Suit::Characters)) > 0);
    }

    #[test]
    fn guaranteed_deal_in_beats_ukeire_on_equal_shanten() {
        let safe = DiscardCandidate {
            tile: tile(Suit::Characters, 1),
            quality: HandQuality {
                shanten: 1,
                live_improvements: 2,
                one_draw_value: 0,
                waits: WaitQuality::default(),
                structure: 0,
            },
            danger: DiscardDanger::default(),
            exposure: 1,
        };
        let dangerous = DiscardCandidate {
            tile: tile(Suit::Characters, 2),
            quality: HandQuality {
                live_improvements: 12,
                ..safe.quality
            },
            danger: DiscardDanger {
                expected_loss: 100 * RISK_SCALE,
            },
            exposure: 1,
        };
        assert!(discard_better(safe, dangerous, false));
        assert!(!discard_better(dangerous, safe, false));
    }

    #[test]
    fn pong_requires_real_progress() {
        let pass = HandQuality {
            shanten: 1,
            live_improvements: 8,
            one_draw_value: 0,
            waits: WaitQuality::default(),
            structure: 0,
        };
        assert!(!pong_is_worthwhile(
            pass,
            HandQuality {
                live_improvements: 9,
                ..pass
            },
            false,
        ));
        assert!(pong_is_worthwhile(
            pass,
            HandQuality {
                shanten: 0,
                live_improvements: 3,
                ..pass
            },
            false,
        ));
    }

    #[test]
    fn rule_ev_does_not_read_opponent_hidden_winning_base() {
        let mut game = Game::new(8);
        for _ in 0..52 {
            let action = game
                .simple_rule_action()
                .expect("the fixture remains active");
            game.step_id(action).expect("the rule action is legal");
        }

        let actor = Seat::ALL[1];
        let opponent = Seat::ALL[2];
        assert_eq!(game.decision().map(|decision| decision.actor), Some(actor));
        assert!(game.has_won(opponent));

        let left = game
            .resample_information_set(0)
            .expect("the information set can be sampled");
        let right = game
            .resample_information_set(1)
            .expect("the information set can be sampled");
        assert_ne!(left.win_base(opponent), right.win_base(opponent));
        assert_ne!(left.locked(opponent), right.locked(opponent));
        assert_eq!(
            observation_for(&left, actor),
            observation_for(&right, actor)
        );

        let discard_dangers = |sampled: &Game| {
            let legal = sampled
                .legal_actions()
                .expect("the sampled game remains on the actor's turn");
            let exposure = public_exposure(sampled, actor);
            mask_tiles(legal.discard_mask)
                .map(|tile| {
                    (
                        tile,
                        heuristic_discard_danger(sampled, actor, tile, &exposure),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(discard_dangers(&left), discard_dangers(&right));
        for config in [RuleEvConfig::FAST, RuleEvConfig::STANDARD] {
            assert_eq!(
                left.rule_ev_action_with_config(config),
                right.rule_ev_action_with_config(config)
            );
        }
    }
}
