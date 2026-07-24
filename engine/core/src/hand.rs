use std::sync::OnceLock;

use crate::types::{Meld, Suit, TILE_KIND_COUNT, Tile};

const SUIT_COUNT: usize = 3;
const RANK_COUNT: usize = 9;
const SUIT_STATE_COUNT: usize = 1_953_125; // 5^9
const POW5: [usize; RANK_COUNT] = [1, 5, 25, 125, 625, 3_125, 15_625, 78_125, 390_625];
const GROUP_KIND_COUNT: usize = TILE_KIND_COUNT + SUIT_COUNT * 7;

// Bits 0..=4 mean 0..=4 melds without a pair. Bits 5..=9 mean the
// same meld counts with a pair. Each entry describes shapes contained in,
// rather than exactly equal to, the suit state.
static SUIT_SHAPES: OnceLock<Box<[u16]>> = OnceLock::new();

/// A named scoring pattern. Composite variants already include the base
/// patterns named by the rules and are not scored a second time.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum Pattern {
    Plain = 0,
    AllSimples,
    AllTriplets,
    PureAllTriplets,
    SevenPairs,
    PureSevenPairs,
    TwoFiveEightSevenPairs,
    DragonSevenPairs,
    PureDragonSevenPairs,
    DoubleDragonSevenPairs,
    TripleDragonSevenPairs,
    TwoFiveEightDoubleDragonSevenPairs,
    TwoFiveEightTripleDragonSevenPairs,
    EighteenArhats,
    PureEighteenArhats,
    TerminalsInEveryGroup,
    AllTerminals,
    PureOneSuit,
    GoldenHook,
    PureGoldenHook,
    RobKong,
    KongDiscard,
    KongDraw,
    LastWallTile,
    Heavenly,
    Earthly,
}

impl Pattern {
    pub const ALL: [Self; 26] = [
        Self::Plain,
        Self::AllSimples,
        Self::AllTriplets,
        Self::PureAllTriplets,
        Self::SevenPairs,
        Self::PureSevenPairs,
        Self::TwoFiveEightSevenPairs,
        Self::DragonSevenPairs,
        Self::PureDragonSevenPairs,
        Self::DoubleDragonSevenPairs,
        Self::TripleDragonSevenPairs,
        Self::TwoFiveEightDoubleDragonSevenPairs,
        Self::TwoFiveEightTripleDragonSevenPairs,
        Self::EighteenArhats,
        Self::PureEighteenArhats,
        Self::TerminalsInEveryGroup,
        Self::AllTerminals,
        Self::PureOneSuit,
        Self::GoldenHook,
        Self::PureGoldenHook,
        Self::RobKong,
        Self::KongDiscard,
        Self::KongDraw,
        Self::LastWallTile,
        Self::Heavenly,
        Self::Earthly,
    ];

    pub const fn multiplier(self) -> u32 {
        match self {
            Self::Plain => 1,
            Self::AllSimples => 2,
            Self::AllTriplets => 2,
            Self::PureAllTriplets => 8,
            Self::SevenPairs => 4,
            Self::PureSevenPairs => 16,
            Self::TwoFiveEightSevenPairs => 16,
            Self::DragonSevenPairs => 8,
            Self::PureDragonSevenPairs => 32,
            Self::DoubleDragonSevenPairs => 16,
            Self::TripleDragonSevenPairs => 32,
            Self::TwoFiveEightDoubleDragonSevenPairs => 64,
            Self::TwoFiveEightTripleDragonSevenPairs => 128,
            Self::EighteenArhats => 64,
            Self::PureEighteenArhats => 256,
            Self::TerminalsInEveryGroup => 4,
            Self::AllTerminals => 16,
            Self::PureOneSuit => 4,
            Self::GoldenHook => 4,
            Self::PureGoldenHook => 16,
            Self::RobKong | Self::KongDiscard | Self::KongDraw | Self::LastWallTile => 2,
            Self::Heavenly | Self::Earthly => 32,
        }
    }
}

/// Fixed-size set of [`Pattern`] values.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct PatternSet(u32);

impl PatternSet {
    pub const EMPTY: Self = Self(0);

    pub const fn from_pattern(pattern: Pattern) -> Self {
        Self(1_u32 << pattern as u32)
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn contains(self, pattern: Pattern) -> bool {
        self.0 & (1_u32 << pattern as u32) != 0
    }

    pub fn insert(&mut self, pattern: Pattern) {
        self.0 |= 1_u32 << pattern as u32;
    }

    pub fn iter(self) -> impl Iterator<Item = Pattern> {
        Pattern::ALL
            .into_iter()
            .filter(move |pattern| self.contains(*pattern))
    }
}

impl core::ops::BitOr for PatternSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl core::ops::BitOrAssign for PatternSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Event multipliers which are independent of the selected hand shape.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WinFlags {
    pub rob_kong: bool,
    pub after_kong_discard: bool,
    pub after_kong_draw: bool,
    pub last_wall_tile: bool,
    pub heavenly: bool,
    pub earthly: bool,
}

impl WinFlags {
    pub const NONE: Self = Self {
        rob_kong: false,
        after_kong_discard: false,
        after_kong_draw: false,
        last_wall_tile: false,
        heavenly: false,
        earthly: false,
    };
}

/// The highest-scoring deterministic interpretation of a winning hand.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WinEvaluation {
    /// Structural multiplier before self-draw payment and event multipliers.
    pub shape_multiplier: u32,
    pub multiplier: u32,
    pub patterns: PatternSet,
    /// Concealed tiles selected for this win. Exposed melds are not included.
    pub used: [u8; TILE_KIND_COUNT],
}

/// The highest-scoring structural win available after adding one tile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaxWaitEvaluation {
    pub winning_tile: Tile,
    pub evaluation: WinEvaluation,
}

/// Returns whether `counts` contains a winning substructure.
///
/// `counts` includes the winning tile. Values above four are allowed for
/// logical duplicate references created by repeated wins; shape availability
/// is capped at four copies. When `required` is present, at least one copy of
/// that tile kind must be selected into the winning structure.
pub fn is_winning(counts: &[u8; TILE_KIND_COUNT], melds: &[Meld], required: Option<Tile>) -> bool {
    if melds.len() > 4 {
        return false;
    }

    let counts = capped_counts(counts);
    if melds.is_empty() && can_seven_pairs(&counts, required) {
        return true;
    }

    can_standard(&counts, 4 - melds.len(), required)
}

/// Selects and scores the highest-multiplier winning substructure.
///
/// Equal-multiplier decompositions are resolved by the fixed search order:
/// lower pair tile first, followed by triplets and then sequences in tile
/// order. Standard hands precede seven-pair hands on an exact tie.
pub fn evaluate_win(
    counts: &[u8; TILE_KIND_COUNT],
    melds: &[Meld],
    required: Option<Tile>,
    flags: WinFlags,
) -> Option<WinEvaluation> {
    if melds.len() > 4 {
        return None;
    }

    let counts = capped_counts(counts);
    let mut best = None;

    if can_standard(&counts, 4 - melds.len(), required) {
        let mut search = StandardSearch::new(counts, melds, required);
        search.run();
        best = search.best;
    }

    if melds.is_empty() && can_seven_pairs(&counts, required) {
        search_seven_pairs(&counts, required, &mut best);
    }

    let mut result = best?;
    apply_win_flags(&mut result, flags);
    Some(result)
}

/// Finds the winning tile and structure with the highest possible multiplier.
///
/// Event multipliers are intentionally excluded because this is used for
/// end-of-wall dajiao settlement. The hypothetical tile is required to be in
/// the selected structure. A configured missing suit rejects hands or melds
/// which still contain that suit and is not searched for a winning tile.
/// Equal-multiplier waits are resolved in tile order.
pub fn evaluate_max_wait(
    counts: &[u8; TILE_KIND_COUNT],
    melds: &[Meld],
    missing_suit: Option<Suit>,
) -> Option<MaxWaitEvaluation> {
    if melds.len() > 4 {
        return None;
    }
    if let Some(suit) = missing_suit {
        let start = suit as usize * RANK_COUNT;
        if counts[start..start + RANK_COUNT]
            .iter()
            .any(|&count| count != 0)
            || melds.iter().any(|meld| meld.tile.suit() == suit)
        {
            return None;
        }
    }

    // Seven-pair search is compact and, unlike standard decomposition, does
    // not branch over every possible meld ordering. Search it first so dense
    // expanded holdings can finish without running the expensive standard
    // search for all 27 candidate tiles.
    let standard_upper_bound = standard_multiplier_upper_bound(melds);
    let seven_pairs = if melds.is_empty() {
        evaluate_max_seven_pairs(counts, missing_suit)
    } else {
        None
    };
    if seven_pairs
        .as_ref()
        .is_some_and(|wait| wait.evaluation.multiplier > standard_upper_bound)
    {
        return seven_pairs;
    }

    let mut augmented = *counts;
    let mut best: Option<MaxWaitEvaluation> = None;
    for index in 0..TILE_KIND_COUNT {
        let tile = Tile::from_index_unchecked(index as u8);
        if missing_suit == Some(tile.suit()) {
            continue;
        }

        augmented[index] = augmented[index].saturating_add(1);
        if let Some(evaluation) = evaluate_win(&augmented, melds, Some(tile), WinFlags::NONE)
            && best
                .as_ref()
                .is_none_or(|current| evaluation.multiplier > current.evaluation.multiplier)
        {
            best = Some(MaxWaitEvaluation {
                winning_tile: tile,
                evaluation,
            });
        }
        augmented[index] = counts[index];

        if best
            .as_ref()
            .is_some_and(|wait| wait.evaluation.multiplier >= standard_upper_bound)
        {
            break;
        }
    }

    match (best, seven_pairs) {
        (Some(standard), Some(seven)) => {
            if seven.evaluation.multiplier > standard.evaluation.multiplier
                || (seven.evaluation.multiplier == standard.evaluation.multiplier
                    && seven.winning_tile.index() < standard.winning_tile.index())
            {
                Some(seven)
            } else {
                Some(standard)
            }
        }
        (standard, seven) => standard.or(seven),
    }
}

fn standard_multiplier_upper_bound(melds: &[Meld]) -> u32 {
    if melds.len() == 4 {
        if melds.iter().all(|meld| meld.kind.is_kong()) {
            4_096
        } else {
            256
        }
    } else {
        64
    }
}

fn evaluate_max_seven_pairs(
    counts: &[u8; TILE_KIND_COUNT],
    missing_suit: Option<Suit>,
) -> Option<MaxWaitEvaluation> {
    let mut augmented = *counts;
    let mut best = None;
    for index in 0..TILE_KIND_COUNT {
        let tile = Tile::from_index_unchecked(index as u8);
        if missing_suit == Some(tile.suit()) {
            continue;
        }
        augmented[index] = augmented[index].saturating_add(1);
        let capped = capped_counts(&augmented);
        if can_seven_pairs(&capped, Some(tile)) {
            let mut evaluation = None;
            search_seven_pairs(&capped, Some(tile), &mut evaluation);
            if let Some(evaluation) = evaluation
                && best.as_ref().is_none_or(|current: &MaxWaitEvaluation| {
                    evaluation.multiplier > current.evaluation.multiplier
                })
            {
                best = Some(MaxWaitEvaluation {
                    winning_tile: tile,
                    evaluation,
                });
            }
        }
        augmented[index] = counts[index];
    }
    best
}

fn capped_counts(counts: &[u8; TILE_KIND_COUNT]) -> [u8; TILE_KIND_COUNT] {
    let mut capped = [0; TILE_KIND_COUNT];
    let mut tile = 0;
    while tile < TILE_KIND_COUNT {
        capped[tile] = counts[tile].min(4);
        tile += 1;
    }
    capped
}

const fn shape_bit(melds: usize, pair: bool) -> u16 {
    1_u16 << (melds + if pair { 5 } else { 0 })
}

const fn add_meld(shapes: u16) -> u16 {
    ((shapes & 0x000f) << 1) | ((shapes & 0x01e0) << 1)
}

const fn add_pair(shapes: u16) -> u16 {
    (shapes & 0x001f) << 5
}

fn suit_shapes() -> &'static [u16] {
    SUIT_SHAPES.get_or_init(build_suit_shapes)
}

fn build_suit_shapes() -> Box<[u16]> {
    let mut table = vec![0_u16; SUIT_STATE_COUNT];
    let mut counts = [0_u8; RANK_COUNT];

    for code in 0..SUIT_STATE_COUNT {
        let mut value = code;
        for count in &mut counts {
            *count = (value % 5) as u8;
            value /= 5;
        }

        // The empty selection makes every availability state useful as a
        // subset state without adding single-tile removal transitions.
        let mut shapes = shape_bit(0, false);
        for rank in 0..RANK_COUNT {
            if counts[rank] >= 2 {
                shapes |= add_pair(table[code - 2 * POW5[rank]]);
            }
            if counts[rank] >= 3 {
                shapes |= add_meld(table[code - 3 * POW5[rank]]);
            }
        }
        for start in 0..7 {
            if counts[start] != 0 && counts[start + 1] != 0 && counts[start + 2] != 0 {
                let previous = code - POW5[start] - POW5[start + 1] - POW5[start + 2];
                shapes |= add_meld(table[previous]);
            }
        }
        table[code] = shapes;
    }

    table.into_boxed_slice()
}

fn suit_codes(counts: &[u8; TILE_KIND_COUNT]) -> [usize; SUIT_COUNT] {
    let mut codes = [0; SUIT_COUNT];
    for (code, suit_counts) in codes.iter_mut().zip(counts.chunks_exact(RANK_COUNT)) {
        *code = suit_counts
            .iter()
            .zip(POW5)
            .map(|(&count, place)| count as usize * place)
            .sum();
    }
    codes
}

#[inline]
fn can_combine_shapes(
    table: &[u16],
    codes: [usize; SUIT_COUNT],
    target_melds: usize,
    pair: bool,
) -> bool {
    if target_melds > 4 {
        return false;
    }

    let shapes = [table[codes[0]], table[codes[1]], table[codes[2]]];
    if !pair {
        for first in 0..=target_melds {
            for second in 0..=target_melds - first {
                let third = target_melds - first - second;
                if shapes[0] & shape_bit(first, false) != 0
                    && shapes[1] & shape_bit(second, false) != 0
                    && shapes[2] & shape_bit(third, false) != 0
                {
                    return true;
                }
            }
        }
        return false;
    }

    for pair_suit in 0..SUIT_COUNT {
        for first in 0..=target_melds {
            for second in 0..=target_melds - first {
                let meld_counts = [first, second, target_melds - first - second];
                if (0..SUIT_COUNT)
                    .all(|suit| shapes[suit] & shape_bit(meld_counts[suit], suit == pair_suit) != 0)
                {
                    return true;
                }
            }
        }
    }
    false
}

fn can_standard(
    counts: &[u8; TILE_KIND_COUNT],
    needed_melds: usize,
    required: Option<Tile>,
) -> bool {
    let table = suit_shapes();
    let codes = suit_codes(counts);
    let Some(required) = required else {
        return can_combine_shapes(table, codes, needed_melds, true);
    };

    let tile = required.index();
    let suit = tile / RANK_COUNT;
    let rank = tile % RANK_COUNT;

    // Reserve a complete group containing the required tile, then use the
    // subset table for the rest. Copies of one tile kind are indistinguishable,
    // so the reserved group can always be considered to contain the win tile.
    if counts[tile] >= 2 {
        let mut remaining = codes;
        remaining[suit] -= 2 * POW5[rank];
        if can_combine_shapes(table, remaining, needed_melds, false) {
            return true;
        }
    }

    if needed_melds == 0 {
        return false;
    }

    if counts[tile] >= 3 {
        let mut remaining = codes;
        remaining[suit] -= 3 * POW5[rank];
        if can_combine_shapes(table, remaining, needed_melds - 1, true) {
            return true;
        }
    }

    let first_start = rank.saturating_sub(2);
    let last_start = rank.min(6);
    for start in first_start..=last_start {
        let offset = suit * RANK_COUNT;
        if counts[offset + start] != 0
            && counts[offset + start + 1] != 0
            && counts[offset + start + 2] != 0
        {
            let mut remaining = codes;
            remaining[suit] -= POW5[start] + POW5[start + 1] + POW5[start + 2];
            if can_combine_shapes(table, remaining, needed_melds - 1, true) {
                return true;
            }
        }
    }
    false
}

fn can_seven_pairs(counts: &[u8; TILE_KIND_COUNT], required: Option<Tile>) -> bool {
    let pair_units: usize = counts.iter().map(|count| (*count / 2) as usize).sum();
    if pair_units < 7 {
        return false;
    }
    required.is_none_or(|tile| counts[tile.index()] >= 2)
}

struct StandardSearch<'a> {
    available: [u8; TILE_KIND_COUNT],
    used: [u8; TILE_KIND_COUNT],
    codes: [usize; SUIT_COUNT],
    exposed: &'a [Meld],
    required: Option<usize>,
    needed_melds: usize,
    exposed_groups_have_terminal: bool,
    table: &'static [u16],
    best: Option<WinEvaluation>,
}

impl<'a> StandardSearch<'a> {
    fn new(counts: [u8; TILE_KIND_COUNT], exposed: &'a [Meld], required: Option<Tile>) -> Self {
        Self {
            codes: suit_codes(&counts),
            available: counts,
            used: [0; TILE_KIND_COUNT],
            exposed,
            required: required.map(Tile::index),
            needed_melds: 4 - exposed.len(),
            exposed_groups_have_terminal: exposed.iter().all(|meld| meld.tile.is_terminal()),
            table: suit_shapes(),
            best: None,
        }
    }

    fn run(&mut self) {
        for pair in 0..TILE_KIND_COUNT {
            if self.available[pair] < 2 {
                continue;
            }

            let suit = pair / RANK_COUNT;
            let rank = pair % RANK_COUNT;
            self.available[pair] -= 2;
            self.used[pair] += 2;
            self.codes[suit] -= 2 * POW5[rank];

            if can_combine_shapes(self.table, self.codes, self.needed_melds, false) {
                let required_used = self.required == Some(pair);
                let groups_have_terminal =
                    is_terminal_index(pair) && self.exposed_groups_have_terminal;
                self.search_melds(
                    0,
                    self.needed_melds,
                    false,
                    groups_have_terminal,
                    required_used,
                );
            }

            self.codes[suit] += 2 * POW5[rank];
            self.used[pair] -= 2;
            self.available[pair] += 2;
        }
    }

    fn search_melds(
        &mut self,
        first_group: usize,
        remaining: usize,
        has_sequence: bool,
        groups_have_terminal: bool,
        required_used: bool,
    ) {
        if remaining == 0 {
            if self.required.is_some() && !required_used {
                return;
            }
            let candidate =
                score_standard(&self.used, self.exposed, has_sequence, groups_have_terminal);
            consider_candidate(&mut self.best, candidate);
            return;
        }

        for group in first_group..GROUP_KIND_COUNT {
            if group < TILE_KIND_COUNT {
                let tile = group;
                if self.available[tile] < 3 {
                    continue;
                }
                let suit = tile / RANK_COUNT;
                let rank = tile % RANK_COUNT;
                self.available[tile] -= 3;
                self.used[tile] += 3;
                self.codes[suit] -= 3 * POW5[rank];

                if can_combine_shapes(self.table, self.codes, remaining - 1, false) {
                    self.search_melds(
                        group,
                        remaining - 1,
                        has_sequence,
                        groups_have_terminal && is_terminal_index(tile),
                        required_used || self.required == Some(tile),
                    );
                }

                self.codes[suit] += 3 * POW5[rank];
                self.used[tile] -= 3;
                self.available[tile] += 3;
                continue;
            }

            let sequence = group - TILE_KIND_COUNT;
            let suit = sequence / 7;
            let start = sequence % 7;
            let offset = suit * RANK_COUNT;
            if self.available[offset + start] == 0
                || self.available[offset + start + 1] == 0
                || self.available[offset + start + 2] == 0
            {
                continue;
            }

            let delta = POW5[start] + POW5[start + 1] + POW5[start + 2];
            for rank in start..start + 3 {
                self.available[offset + rank] -= 1;
                self.used[offset + rank] += 1;
            }
            self.codes[suit] -= delta;

            if can_combine_shapes(self.table, self.codes, remaining - 1, false) {
                let contains_required = self
                    .required
                    .is_some_and(|tile| (offset + start..offset + start + 3).contains(&tile));
                self.search_melds(
                    group,
                    remaining - 1,
                    true,
                    groups_have_terminal && matches!(start, 0 | 6),
                    required_used || contains_required,
                );
            }

            self.codes[suit] += delta;
            for rank in start..start + 3 {
                self.used[offset + rank] -= 1;
                self.available[offset + rank] += 1;
            }
        }
    }
}

fn score_standard(
    used: &[u8; TILE_KIND_COUNT],
    exposed: &[Meld],
    has_sequence: bool,
    groups_have_terminal: bool,
) -> WinEvaluation {
    let properties = tile_properties(used, exposed);
    let all_triplets = !has_sequence;
    let all_kongs = exposed.len() == 4 && exposed.iter().all(|meld| meld.kind.is_kong());
    let golden_hook = exposed.len() == 4;
    let mut multiplier = 1;
    let mut patterns = PatternSet::EMPTY;
    let mut pure_is_included = false;

    if all_kongs {
        let pattern = if properties.pure_one_suit {
            pure_is_included = true;
            Pattern::PureEighteenArhats
        } else {
            Pattern::EighteenArhats
        };
        add_pattern(&mut patterns, &mut multiplier, pattern);
    } else if golden_hook {
        let pattern = if properties.pure_one_suit {
            pure_is_included = true;
            Pattern::PureGoldenHook
        } else {
            Pattern::GoldenHook
        };
        add_pattern(&mut patterns, &mut multiplier, pattern);
    } else if all_triplets && !properties.all_terminals {
        let pattern = if properties.pure_one_suit {
            pure_is_included = true;
            Pattern::PureAllTriplets
        } else {
            Pattern::AllTriplets
        };
        add_pattern(&mut patterns, &mut multiplier, pattern);
    }

    if properties.all_terminals {
        add_pattern(&mut patterns, &mut multiplier, Pattern::AllTerminals);
    } else if groups_have_terminal {
        add_pattern(
            &mut patterns,
            &mut multiplier,
            Pattern::TerminalsInEveryGroup,
        );
    }

    if properties.pure_one_suit && !pure_is_included {
        add_pattern(&mut patterns, &mut multiplier, Pattern::PureOneSuit);
    }
    if properties.all_simples {
        add_pattern(&mut patterns, &mut multiplier, Pattern::AllSimples);
    }
    if patterns.is_empty() {
        add_pattern(&mut patterns, &mut multiplier, Pattern::Plain);
    }

    WinEvaluation {
        shape_multiplier: multiplier,
        multiplier,
        patterns,
        used: *used,
    }
}

#[derive(Clone, Copy)]
struct PairState {
    reachable: bool,
    used: [u8; TILE_KIND_COUNT],
}

impl PairState {
    const EMPTY: Self = Self {
        reachable: false,
        used: [0; TILE_KIND_COUNT],
    };
}

#[derive(Clone, Copy)]
struct PairFeatures {
    pairs: u8,
    dragons: u8,
    suit_mask: u8,
    all_two_five_eight: bool,
    all_simples: bool,
    all_terminals: bool,
    required_used: bool,
}

const PAIR_STATE_COUNT: usize = 4_096;

impl PairFeatures {
    const fn encode(self) -> usize {
        self.pairs as usize
            | (self.dragons as usize) << 3
            | (self.suit_mask as usize) << 5
            | (self.all_two_five_eight as usize) << 8
            | (self.all_simples as usize) << 9
            | (self.all_terminals as usize) << 10
            | (self.required_used as usize) << 11
    }

    const fn decode(value: usize) -> Self {
        Self {
            pairs: (value & 0x7) as u8,
            dragons: ((value >> 3) & 0x3) as u8,
            suit_mask: ((value >> 5) & 0x7) as u8,
            all_two_five_eight: value & (1 << 8) != 0,
            all_simples: value & (1 << 9) != 0,
            all_terminals: value & (1 << 10) != 0,
            required_used: value & (1 << 11) != 0,
        }
    }
}

fn search_seven_pairs(
    counts: &[u8; TILE_KIND_COUNT],
    required: Option<Tile>,
    best: &mut Option<WinEvaluation>,
) {
    let mut current = [PairState::EMPTY; PAIR_STATE_COUNT];
    let mut next = [PairState::EMPTY; PAIR_STATE_COUNT];
    let initial = PairFeatures {
        pairs: 0,
        dragons: 0,
        suit_mask: 0,
        all_two_five_eight: true,
        all_simples: true,
        all_terminals: true,
        required_used: false,
    };
    current[initial.encode()] = PairState {
        reachable: true,
        used: [0; TILE_KIND_COUNT],
    };

    let required = required.map(Tile::index);
    for (tile, &count) in counts.iter().enumerate() {
        next.fill(PairState::EMPTY);
        for (state_index, &state) in current.iter().enumerate() {
            if !state.reachable {
                continue;
            }
            let features = PairFeatures::decode(state_index);
            let capacity = (count / 2).min(7 - features.pairs);
            for take in (0..=capacity).rev() {
                let mut updated = features;
                updated.pairs += take;
                if take != 0 {
                    let rank = tile % RANK_COUNT;
                    updated.dragons += u8::from(take == 2);
                    updated.suit_mask |= 1 << (tile / RANK_COUNT);
                    updated.all_two_five_eight &= matches!(rank, 1 | 4 | 7);
                    updated.all_simples &= !matches!(rank, 0 | 8);
                    updated.all_terminals &= matches!(rank, 0 | 8);
                    updated.required_used |= required == Some(tile);
                }

                let destination = updated.encode();
                if !next[destination].reachable {
                    let mut selected = state;
                    selected.used[tile] = take * 2;
                    next[destination] = selected;
                }
            }
        }
        core::mem::swap(&mut current, &mut next);
    }

    for (state_index, &state) in current.iter().enumerate() {
        if !state.reachable {
            continue;
        }
        let features = PairFeatures::decode(state_index);
        if features.pairs != 7 || (required.is_some() && !features.required_used) {
            continue;
        }
        consider_candidate(best, score_seven_pairs(&state.used));
    }
}

fn score_seven_pairs(used: &[u8; TILE_KIND_COUNT]) -> WinEvaluation {
    let mut suit_mask = 0_u8;
    let mut all_two_five_eight = true;
    let mut all_simples = true;
    let mut all_terminals = true;
    let mut dragons = 0_u8;

    for (tile, count) in used.iter().copied().enumerate() {
        if count == 0 {
            continue;
        }
        let rank = tile % RANK_COUNT;
        suit_mask |= 1 << (tile / RANK_COUNT);
        all_two_five_eight &= matches!(rank, 1 | 4 | 7);
        all_simples &= !matches!(rank, 0 | 8);
        all_terminals &= matches!(rank, 0 | 8);
        dragons += u8::from(count == 4);
    }

    let pure = suit_mask.count_ones() == 1;
    let family = if all_two_five_eight && dragons >= 3 {
        Pattern::TwoFiveEightTripleDragonSevenPairs
    } else if all_two_five_eight && dragons >= 2 {
        Pattern::TwoFiveEightDoubleDragonSevenPairs
    } else if dragons >= 3 {
        Pattern::TripleDragonSevenPairs
    } else if pure && dragons >= 1 {
        Pattern::PureDragonSevenPairs
    } else if dragons >= 2 {
        Pattern::DoubleDragonSevenPairs
    } else if all_two_five_eight {
        Pattern::TwoFiveEightSevenPairs
    } else if pure {
        Pattern::PureSevenPairs
    } else if dragons >= 1 {
        Pattern::DragonSevenPairs
    } else {
        Pattern::SevenPairs
    };

    let mut patterns = PatternSet::EMPTY;
    let mut multiplier = 1;
    add_pattern(&mut patterns, &mut multiplier, family);

    // 258 variants already contain the all-simples property. Pure seven-pair
    // variants similarly consume pure-one-suit within their own family.
    if all_simples && !all_two_five_eight {
        add_pattern(&mut patterns, &mut multiplier, Pattern::AllSimples);
    }
    if all_terminals {
        add_pattern(&mut patterns, &mut multiplier, Pattern::AllTerminals);
    }

    WinEvaluation {
        shape_multiplier: multiplier,
        multiplier,
        patterns,
        used: *used,
    }
}

#[derive(Clone, Copy)]
struct TileProperties {
    pure_one_suit: bool,
    all_simples: bool,
    all_terminals: bool,
}

fn tile_properties(used: &[u8; TILE_KIND_COUNT], exposed: &[Meld]) -> TileProperties {
    let mut suit_mask = 0_u8;
    let mut all_simples = true;
    let mut all_terminals = true;

    for (tile, count) in used.iter().copied().enumerate() {
        if count == 0 {
            continue;
        }
        suit_mask |= 1 << (tile / RANK_COUNT);
        all_simples &= !is_terminal_index(tile);
        all_terminals &= is_terminal_index(tile);
    }
    for meld in exposed {
        suit_mask |= 1 << meld.tile.suit() as u8;
        all_simples &= !meld.tile.is_terminal();
        all_terminals &= meld.tile.is_terminal();
    }

    TileProperties {
        pure_one_suit: suit_mask.count_ones() == 1,
        all_simples,
        all_terminals,
    }
}

const fn is_terminal_index(tile: usize) -> bool {
    matches!(tile % RANK_COUNT, 0 | 8)
}

fn add_pattern(patterns: &mut PatternSet, multiplier: &mut u32, pattern: Pattern) {
    patterns.insert(pattern);
    *multiplier *= pattern.multiplier();
}

fn consider_candidate(best: &mut Option<WinEvaluation>, candidate: WinEvaluation) {
    if best
        .as_ref()
        .is_none_or(|current| candidate.multiplier > current.multiplier)
    {
        *best = Some(candidate);
    }
}

fn apply_win_flags(evaluation: &mut WinEvaluation, flags: WinFlags) {
    let events = [
        (flags.rob_kong, Pattern::RobKong),
        (flags.after_kong_discard, Pattern::KongDiscard),
        (flags.after_kong_draw, Pattern::KongDraw),
        (flags.last_wall_tile, Pattern::LastWallTile),
        (flags.heavenly, Pattern::Heavenly),
        (flags.earthly, Pattern::Earthly),
    ];
    for (applies, pattern) in events {
        if applies {
            add_pattern(
                &mut evaluation.patterns,
                &mut evaluation.multiplier,
                pattern,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MeldKind, Seat, Suit};

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank - 1).unwrap()
    }

    fn add(counts: &mut [u8; TILE_KIND_COUNT], suit: Suit, rank: u8, amount: u8) {
        counts[tile(suit, rank).index()] += amount;
    }

    fn exposed(suit: Suit, rank: u8, kind: MeldKind) -> Meld {
        Meld {
            tile: tile(suit, rank),
            kind,
            source: Seat::EAST,
        }
    }

    fn brute_standard(
        counts: &[u8; TILE_KIND_COUNT],
        exposed_melds: usize,
        required: Option<Tile>,
    ) -> bool {
        let mut available = capped_counts(counts);
        for pair in 0..TILE_KIND_COUNT {
            if available[pair] < 2 {
                continue;
            }
            available[pair] -= 2;
            if brute_melds(
                &mut available,
                0,
                4 - exposed_melds,
                required.map(Tile::index),
                required.is_some_and(|tile| tile.index() == pair),
            ) {
                return true;
            }
            available[pair] += 2;
        }
        false
    }

    fn brute_melds(
        available: &mut [u8; TILE_KIND_COUNT],
        first_group: usize,
        remaining: usize,
        required: Option<usize>,
        required_used: bool,
    ) -> bool {
        if remaining == 0 {
            return required.is_none() || required_used;
        }

        for group in first_group..GROUP_KIND_COUNT {
            if group < TILE_KIND_COUNT {
                if available[group] < 3 {
                    continue;
                }
                available[group] -= 3;
                let wins = brute_melds(
                    available,
                    group,
                    remaining - 1,
                    required,
                    required_used || required == Some(group),
                );
                available[group] += 3;
                if wins {
                    return true;
                }
                continue;
            }

            let sequence = group - TILE_KIND_COUNT;
            let suit = sequence / 7;
            let start = sequence % 7;
            let offset = suit * RANK_COUNT;
            if available[offset + start] == 0
                || available[offset + start + 1] == 0
                || available[offset + start + 2] == 0
            {
                continue;
            }
            for rank in start..start + 3 {
                available[offset + rank] -= 1;
            }
            let contains_required =
                required.is_some_and(|tile| (offset + start..offset + start + 3).contains(&tile));
            let wins = brute_melds(
                available,
                group,
                remaining - 1,
                required,
                required_used || contains_required,
            );
            for rank in start..start + 3 {
                available[offset + rank] += 1;
            }
            if wins {
                return true;
            }
        }
        false
    }

    #[test]
    fn standard_subset_must_contain_the_required_tile() {
        let mut counts = [0; TILE_KIND_COUNT];
        for rank in 1..=3 {
            add(&mut counts, Suit::Characters, rank, 1);
        }
        for rank in 4..=6 {
            add(&mut counts, Suit::Bamboo, rank, 1);
        }
        add(&mut counts, Suit::Dots, 7, 3);
        add(&mut counts, Suit::Dots, 8, 3);
        add(&mut counts, Suit::Dots, 9, 2);
        add(&mut counts, Suit::Bamboo, 9, 1); // unused extra tile

        assert!(is_winning(&counts, &[], Some(tile(Suit::Dots, 8))));
        assert!(!is_winning(&counts, &[], Some(tile(Suit::Bamboo, 9))));

        let evaluation =
            evaluate_win(&counts, &[], Some(tile(Suit::Dots, 8)), WinFlags::NONE).unwrap();
        assert_eq!(evaluation.used.iter().sum::<u8>(), 14);
        assert_eq!(evaluation.used[tile(Suit::Bamboo, 9).index()], 0);
    }

    #[test]
    fn logical_duplicate_counts_are_capped_without_rejecting_the_hand() {
        let mut counts = [0; TILE_KIND_COUNT];
        add(&mut counts, Suit::Characters, 1, 9);
        add(&mut counts, Suit::Characters, 2, 3);
        add(&mut counts, Suit::Bamboo, 3, 3);
        add(&mut counts, Suit::Dots, 4, 3);
        add(&mut counts, Suit::Dots, 5, 2);

        let evaluation = evaluate_win(&counts, &[], None, WinFlags::NONE).expect("winning subset");
        assert_eq!(evaluation.used[tile(Suit::Characters, 1).index()], 3);
        assert!(evaluation.used.iter().all(|count| *count <= 4));
    }

    #[test]
    fn ambiguous_standard_hand_uses_the_higher_scoring_decomposition() {
        let mut counts = [0; TILE_KIND_COUNT];
        for rank in 1..=4 {
            add(&mut counts, Suit::Characters, rank, 3);
        }
        add(&mut counts, Suit::Characters, 5, 2);

        let evaluation = evaluate_win(&counts, &[], None, WinFlags::NONE).unwrap();
        assert_eq!(evaluation.multiplier, 8);
        assert!(evaluation.patterns.contains(Pattern::PureAllTriplets));
        assert!(!evaluation.patterns.contains(Pattern::PureOneSuit));
    }

    #[test]
    fn max_wait_selects_the_higher_scoring_wait() {
        let mut counts = [0; TILE_KIND_COUNT];
        for rank in 1..=4 {
            add(&mut counts, Suit::Characters, rank, 3);
        }
        add(&mut counts, Suit::Characters, 5, 1);

        let five = tile(Suit::Characters, 5);
        let six = tile(Suit::Characters, 6);
        let mut completed = counts;
        completed[five.index()] += 1;
        let five_evaluation = evaluate_win(&completed, &[], Some(five), WinFlags::NONE).unwrap();
        completed = counts;
        completed[six.index()] += 1;
        let six_evaluation = evaluate_win(&completed, &[], Some(six), WinFlags::NONE).unwrap();

        assert_eq!(five_evaluation.multiplier, 8);
        assert!(five_evaluation.patterns.contains(Pattern::PureAllTriplets));
        assert_eq!(six_evaluation.multiplier, 4);
        assert!(six_evaluation.patterns.contains(Pattern::PureOneSuit));

        let best = evaluate_max_wait(&counts, &[], Some(Suit::Dots)).unwrap();
        assert_eq!(best.winning_tile, five);
        assert_eq!(best.evaluation, five_evaluation);
    }

    #[test]
    fn max_wait_searches_all_substructures_in_expanded_holdings() {
        let mut counts = [0; TILE_KIND_COUNT];
        add(&mut counts, Suit::Characters, 2, 4);
        add(&mut counts, Suit::Bamboo, 5, 4);
        add(&mut counts, Suit::Dots, 8, 4);
        add(&mut counts, Suit::Bamboo, 2, 1);
        add(&mut counts, Suit::Characters, 1, 1);

        let high_wait = tile(Suit::Bamboo, 2);
        let low_wait = tile(Suit::Characters, 1);
        let mut completed = counts;
        completed[low_wait.index()] += 1;
        let low_evaluation = evaluate_win(&completed, &[], Some(low_wait), WinFlags::NONE).unwrap();
        assert_eq!(low_evaluation.multiplier, 32);
        assert!(
            low_evaluation
                .patterns
                .contains(Pattern::TripleDragonSevenPairs)
        );

        let best = evaluate_max_wait(&counts, &[], None).unwrap();
        assert_eq!(best.winning_tile, high_wait);
        assert_eq!(best.evaluation.multiplier, 128);
        assert!(
            best.evaluation
                .patterns
                .contains(Pattern::TwoFiveEightTripleDragonSevenPairs)
        );
        assert_eq!(best.evaluation.used[high_wait.index()], 2);
        assert_eq!(best.evaluation.used[low_wait.index()], 0);
    }

    #[test]
    fn optimized_max_wait_matches_naive_randomized_search() {
        fn naive(counts: &[u8; TILE_KIND_COUNT], melds: &[Meld]) -> Option<MaxWaitEvaluation> {
            let mut augmented = *counts;
            let mut best: Option<MaxWaitEvaluation> = None;
            for index in 0..TILE_KIND_COUNT {
                let winning_tile = Tile::from_index_unchecked(index as u8);
                augmented[index] = augmented[index].saturating_add(1);
                if let Some(evaluation) =
                    evaluate_win(&augmented, melds, Some(winning_tile), WinFlags::NONE)
                    && best
                        .as_ref()
                        .is_none_or(|current| evaluation.multiplier > current.evaluation.multiplier)
                {
                    best = Some(MaxWaitEvaluation {
                        winning_tile,
                        evaluation,
                    });
                }
                augmented[index] = counts[index];
            }
            best
        }

        let mut random = 0x4f13_65a9_2bd0_c781_u64;
        for _ in 0..300 {
            let mut counts = [0_u8; TILE_KIND_COUNT];
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            for _ in 0..8 + random as usize % 24 {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let index = random as usize % TILE_KIND_COUNT;
                counts[index] = counts[index].saturating_add(1);
            }

            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let meld_count = random as usize % 5;
            let mut melds = Vec::with_capacity(meld_count);
            for _ in 0..meld_count {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let kind = match random & 3 {
                    0 => MeldKind::Pong,
                    1 => MeldKind::ExposedKong,
                    2 => MeldKind::AddedKong,
                    _ => MeldKind::ConcealedKong,
                };
                melds.push(Meld {
                    tile: Tile::from_index_unchecked((random as usize % TILE_KIND_COUNT) as u8),
                    kind,
                    source: Seat::EAST,
                });
            }

            assert_eq!(
                evaluate_max_wait(&counts, &melds, None),
                naive(&counts, &melds),
                "counts={counts:?}, melds={melds:?}"
            );
        }
    }

    #[test]
    fn two_five_eight_triple_dragon_seven_pairs_is_not_double_counted() {
        let mut counts = [0; TILE_KIND_COUNT];
        add(&mut counts, Suit::Characters, 2, 4);
        add(&mut counts, Suit::Bamboo, 5, 4);
        add(&mut counts, Suit::Dots, 8, 4);
        add(&mut counts, Suit::Bamboo, 2, 2);

        let evaluation =
            evaluate_win(&counts, &[], Some(tile(Suit::Bamboo, 2)), WinFlags::NONE).unwrap();
        assert_eq!(evaluation.multiplier, 128);
        assert!(
            evaluation
                .patterns
                .contains(Pattern::TwoFiveEightTripleDragonSevenPairs)
        );
        assert!(!evaluation.patterns.contains(Pattern::SevenPairs));
        assert!(!evaluation.patterns.contains(Pattern::AllSimples));
    }

    #[test]
    fn all_terminals_suppresses_the_implied_all_triplets() {
        let mut counts = [0; TILE_KIND_COUNT];
        add(&mut counts, Suit::Characters, 1, 3);
        add(&mut counts, Suit::Characters, 9, 3);
        add(&mut counts, Suit::Bamboo, 1, 3);
        add(&mut counts, Suit::Bamboo, 9, 3);
        add(&mut counts, Suit::Dots, 1, 2);

        let evaluation = evaluate_win(&counts, &[], None, WinFlags::NONE).unwrap();
        assert_eq!(evaluation.multiplier, 16);
        assert!(evaluation.patterns.contains(Pattern::AllTerminals));
        assert!(!evaluation.patterns.contains(Pattern::AllTriplets));
    }

    #[test]
    fn every_group_with_a_terminal_is_scored() {
        let mut counts = [0; TILE_KIND_COUNT];
        for rank in 1..=3 {
            add(&mut counts, Suit::Characters, rank, 1);
        }
        for rank in 7..=9 {
            add(&mut counts, Suit::Characters, rank, 1);
        }
        add(&mut counts, Suit::Bamboo, 1, 3);
        add(&mut counts, Suit::Bamboo, 9, 3);
        add(&mut counts, Suit::Dots, 1, 2);

        let evaluation = evaluate_win(&counts, &[], None, WinFlags::NONE).unwrap();
        assert_eq!(evaluation.multiplier, 4);
        assert!(evaluation.patterns.contains(Pattern::TerminalsInEveryGroup));
    }

    #[test]
    fn exposed_kongs_select_pure_eighteen_arhats() {
        let melds = [
            exposed(Suit::Characters, 1, MeldKind::ExposedKong),
            exposed(Suit::Characters, 3, MeldKind::AddedKong),
            exposed(Suit::Characters, 5, MeldKind::ConcealedKong),
            exposed(Suit::Characters, 7, MeldKind::ExposedKong),
        ];
        let mut counts = [0; TILE_KIND_COUNT];
        add(&mut counts, Suit::Characters, 5, 2);

        let evaluation = evaluate_win(
            &counts,
            &melds,
            Some(tile(Suit::Characters, 5)),
            WinFlags::NONE,
        )
        .unwrap();
        assert_eq!(evaluation.multiplier, 256);
        assert!(evaluation.patterns.contains(Pattern::PureEighteenArhats));
        assert!(!evaluation.patterns.contains(Pattern::PureGoldenHook));
    }

    #[test]
    fn independent_event_flags_multiply_the_shape() {
        let mut counts = [0; TILE_KIND_COUNT];
        for rank in 1..=3 {
            add(&mut counts, Suit::Characters, rank, 1);
        }
        for rank in 4..=6 {
            add(&mut counts, Suit::Bamboo, rank, 1);
        }
        for rank in 7..=9 {
            add(&mut counts, Suit::Dots, rank, 1);
        }
        add(&mut counts, Suit::Characters, 5, 3);
        add(&mut counts, Suit::Bamboo, 9, 2);

        let flags = WinFlags {
            rob_kong: true,
            after_kong_discard: true,
            heavenly: true,
            ..WinFlags::NONE
        };
        let evaluation = evaluate_win(&counts, &[], None, flags).unwrap();
        assert_eq!(evaluation.multiplier, 2 * 2 * 32);
        assert!(evaluation.patterns.contains(Pattern::RobKong));
        assert!(evaluation.patterns.contains(Pattern::KongDiscard));
        assert!(evaluation.patterns.contains(Pattern::Heavenly));
    }

    #[test]
    fn subset_table_matches_independent_randomized_search() {
        let mut random = 0x2d35_8dcc_aa6c_78a5_u64;
        for _ in 0..2_000 {
            let mut counts = [0_u8; TILE_KIND_COUNT];
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let draws = 10 + (random as usize % 15);
            for _ in 0..draws {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let tile = random as usize % TILE_KIND_COUNT;
                counts[tile] = counts[tile].saturating_add(1);
            }
            random ^= random << 13;
            random ^= random >> 7;
            random ^= random << 17;
            let exposed_melds = random as usize % 5;
            let required = if random & 4 == 0 {
                None
            } else {
                Some(Tile::new((random as u8) % TILE_KIND_COUNT as u8).unwrap())
            };
            let capped = capped_counts(&counts);

            assert_eq!(
                can_standard(&capped, 4 - exposed_melds, required),
                brute_standard(&counts, exposed_melds, required),
                "counts={counts:?}, melds={exposed_melds}, required={required:?}"
            );
        }
    }
}
