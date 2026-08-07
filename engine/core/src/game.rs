use crate::action::{
    Action, ActionId, ActionMask, LEGAL_ACTION_MASK_WORDS, added_kong_offset,
    concealed_kong_offset, discard_offset, exchange_offset,
};
use crate::hand::{
    SHANTEN_TERMINAL, ShantenAnalysis, StableWinBaseSearchOrder, WinEvaluation, WinFlags,
    analyze_shanten, apply_bloodflow_win, bloodflow_evaluation_counts, evaluate_bloodflow_max_wait,
    evaluate_bloodflow_win, evaluate_max_wait, evaluate_win, is_bloodflow_winning, is_winning,
    remove_tiles_for_meld, stabilize_win_base, visit_stable_win_bases,
};
use crate::rng::Rng;
use crate::types::{
    ExchangeDirection, Meld, MeldKind, PLAYER_COUNT, Seat, Suit, TILE_COPIES, TILE_KIND_COUNT,
    Tile, WALL_TILE_COUNT,
};
use rayon::prelude::*;
use thiserror::Error;

const STARTING_SCORE: i64 = 10_000;
pub(crate) const SCORE_UNIT: i64 = 100;
/// Increment whenever engine rule semantics or deterministic initialization
/// change in a way that can invalidate a stored action-sequence replay.
pub const ENGINE_RULES_VERSION: u32 = 10;
pub(crate) const PARALLEL_BATCH_THRESHOLD: usize = 64;
pub const STEP_RECORD_WIDTH: usize = 12;
/// Number of `i32` fields in one event-stream record.
pub const EVENT_RECORD_WIDTH: usize = 8;
/// Retained event history per environment. Older events are overwritten.
pub const EVENT_HISTORY_CAPACITY: usize = 512;
/// Fields stored per relative player in the UI-only win summary buffer.
pub const PLAYER_UI_STATS_FIELDS: usize = 5;
/// Four relative players and five fields per player.
pub const PLAYER_UI_STATS_WIDTH: usize = PLAYER_COUNT * PLAYER_UI_STATS_FIELDS;
/// Fields stored per relative player in the wall-settlement metadata buffer.
pub const WALL_SETTLEMENT_FIELDS: usize = 3;
/// Four relative players and three wall-settlement fields per player.
pub const WALL_SETTLEMENT_META_WIDTH: usize = PLAYER_COUNT * WALL_SETTLEMENT_FIELDS;
/// Four revealed terminal concealed hands of 27 tile kinds each.
pub const WALL_SETTLEMENT_HANDS_WIDTH: usize = PLAYER_COUNT * TILE_KIND_COUNT;
/// Viewer-scoped tile-count and private tile-state planes.
pub const TILE_OBSERVATION_PLANES: usize = 11;
/// Eleven viewer-scoped channels of 27 tile kinds each.
pub const TILE_OBSERVATION_WIDTH: usize = TILE_OBSERVATION_PLANES * TILE_KIND_COUNT;
/// Meld slots stored per relative seat in the ordinary observation tensor.
pub const MELD_SLOTS: usize = 4;
/// Fields stored per meld slot: tile, kind, source_relative.
pub const MELD_FIELDS: usize = 3;
/// Four relative players, four meld slots, and three fields per meld.
pub const MELD_OBSERVATION_WIDTH: usize = PLAYER_COUNT * MELD_SLOTS * MELD_FIELDS;
/// Maximum chronological river entries retained in the observation tensor.
pub const RIVER_TILE_CAPACITY: usize = WALL_TILE_COUNT;
/// Fields stored per river entry: tile, owner_relative.
pub const RIVER_FIELDS: usize = 2;
/// Up to 108 chronological discards with tile and relative owner fields.
pub const RIVER_OBSERVATION_WIDTH: usize = RIVER_TILE_CAPACITY * RIVER_FIELDS;
/// Scalar state plus four relative-player feature groups.
pub const META_OBSERVATION_WIDTH: usize = 34;
/// Training-only perfect-information tile-count planes: four concealed hands,
/// four complete locked subsets, and one unordered remaining-wall histogram.
pub const ORACLE_TILE_COUNT_PLANES: usize = PLAYER_COUNT * 2 + 1;
pub const ORACLE_TILE_COUNT_WIDTH: usize = ORACLE_TILE_COUNT_PLANES * TILE_KIND_COUNT;

/// Event kinds written to the fixed-width event stream.
///
/// The stream contains both private decision records and public rule events.
/// Visibility is applied when a stream is copied for a viewer; the canonical
/// event ring never exposes its internal visibility masks through FFI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum EventKind {
    Action = 0,
    GameStart = 1,
    TurnStart = 2,
    Draw = 3,
    Discard = 4,
    ExchangeComplete = 5,
    MissingRevealed = 6,
    Meld = 7,
    Hu = 8,
    Payment = 9,
    GameEnd = 10,
    SettlementStage = 11,
}

impl EventKind {
    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// Public stages of end-of-wall settlement.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum SettlementStage {
    FlowerPig = 0,
    Dajiao = 1,
}

impl SettlementStage {
    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// Event flags shared by the fixed-width stream schema.
///
/// For [`EventKind::SettlementStage`] records, the `flags` field contains one
/// of the [`SettlementStage`] codes. Other event kinds keep their existing
/// flag meanings.
pub const EVENT_FLAG_REPLACEMENT_DRAW: u8 = 1 << 0;
pub const EVENT_FLAG_LAST_WALL_TILE: u8 = 1 << 1;
pub const EVENT_FLAG_AFTER_KONG: u8 = 1 << 2;
pub const EVENT_FLAG_OPENING_DISCARD: u8 = 1 << 3;
pub const EVENT_FLAG_SELF_DRAW: u8 = 1 << 4;
pub const EVENT_FLAG_ROB_KONG: u8 = 1 << 5;
pub const EVENT_FLAG_HEAVENLY: u8 = 1 << 6;
pub const EVENT_FLAG_EARTHLY: u8 = 1 << 7;
/// A discard used the tile drawn at the start of the same turn.
///
/// Event flags are interpreted by event kind. Bit four is also used by Hu
/// events for self-draw, but the meanings cannot overlap in one record.
pub const EVENT_FLAG_TSUMOGIRI: u8 = 1 << 4;
/// A discard immediately followed a successful Pong.
///
/// Event flags are interpreted by event kind. Bit five is also used by Hu
/// events for robbing a kong, but the meanings cannot overlap in one record.
pub const EVENT_FLAG_AFTER_PONG: u8 = 1 << 5;

const ALL_PLAYER_MASK: u8 = (1 << PLAYER_COUNT) - 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StoredEvent {
    record: [i32; EVENT_RECORD_WIDTH],
    visible_to: u8,
    tile_visible_to: u8,
}

impl StoredEvent {
    const EMPTY: Self = Self {
        record: [-1; EVENT_RECORD_WIDTH],
        visible_to: 0,
        tile_visible_to: 0,
    };
}
const DUMMY_MELD: Meld = Meld {
    tile: Tile::from_index_unchecked(0),
    kind: MeldKind::Pong,
    source: Seat::EAST,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Phase {
    Exchange = 0,
    ChooseMissing = 1,
    Turn = 2,
    HuResponse = 3,
    MeldResponse = 4,
    Finished = 5,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TerminationReason {
    WallExhausted = 0,
    ThreePlayersBankrupt = 1,
}

impl TerminationReason {
    pub const fn code(self) -> u8 {
        self as u8
    }
}

impl Phase {
    /// Stable integer representation used by flat FFI transition records.
    pub const fn code(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Decision {
    pub actor: Seat,
    pub phase: Phase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrawEvent {
    pub player: Seat,
    pub tile: Tile,
    pub replacement: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiscardEvent {
    pub player: Seat,
    pub tile: Tile,
}

/// Public scoring details from one player's most recent completed win.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WinSummary {
    pub shape_multiplier: u32,
    pub multiplier: u32,
    pub patterns: u32,
    pub flags: u8,
}

/// Public facts calculated once when the live wall is exhausted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WallSettlementSummary {
    flower_pig: [bool; PLAYER_COUNT],
    ready: [bool; PLAYER_COUNT],
    max_shape_multipliers: [u32; PLAYER_COUNT],
    revealed_hands: [[u8; TILE_KIND_COUNT]; PLAYER_COUNT],
}

impl WallSettlementSummary {
    pub const fn is_flower_pig(&self, seat: Seat) -> bool {
        self.flower_pig[seat.index()]
    }

    pub const fn is_ready(&self, seat: Seat) -> bool {
        self.ready[seat.index()]
    }

    pub const fn max_shape_multiplier(&self, seat: Seat) -> u32 {
        self.max_shape_multipliers[seat.index()]
    }

    pub const fn revealed_hand(&self, seat: Seat) -> &[u8; TILE_KIND_COUNT] {
        &self.revealed_hands[seat.index()]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepOutcome {
    pub draw: Option<DrawEvent>,
    pub discard: Option<DiscardEvent>,
    pub score_delta: [i64; PLAYER_COUNT],
    pub next: Option<Decision>,
    pub terminal: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DrawNotice {
    pub player: Seat,
    pub tile: Option<Tile>,
    pub replacement: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlayerStepOutcome {
    pub draw: Option<DrawNotice>,
    pub discard: Option<DiscardEvent>,
    pub score_delta: [i64; PLAYER_COUNT],
    pub next: Option<Decision>,
    pub terminal: bool,
}

impl StepOutcome {
    /// Filters the authoritative transition before it is given to one policy.
    /// A draw is public, but its tile face is only visible to the drawing player.
    pub fn for_player(self, viewer: Seat) -> PlayerStepOutcome {
        PlayerStepOutcome {
            draw: self.draw.map(|draw| DrawNotice {
                player: draw.player,
                tile: (draw.player == viewer).then_some(draw.tile),
                replacement: draw.replacement,
            }),
            discard: self.discard,
            score_delta: self.score_delta,
            next: self.next,
            terminal: self.terminal,
        }
    }

    fn write_record(self, record: &mut [i64; STEP_RECORD_WIDTH]) {
        match self.draw {
            Some(draw) => {
                record[0] = i64::from(draw.player.as_u8());
                record[1] = i64::from(draw.tile.as_u8());
                record[2] = i64::from(u8::from(draw.replacement));
            }
            None => {
                record[0] = -1;
                record[1] = -1;
                record[2] = 0;
            }
        }
        match self.discard {
            Some(discard) => {
                record[3] = i64::from(discard.player.as_u8());
                record[4] = i64::from(discard.tile.as_u8());
            }
            None => {
                record[3] = -1;
                record[4] = -1;
            }
        }
        record[5..9].copy_from_slice(&self.score_delta);
        match self.next {
            Some(next) => {
                record[9] = i64::from(next.actor.as_u8());
                record[10] = i64::from(next.phase.code());
            }
            None => {
                record[9] = -1;
                record[10] = -1;
            }
        }
        record[11] = i64::from(u8::from(self.terminal));
    }
}

impl Default for StepOutcome {
    fn default() -> Self {
        Self {
            draw: None,
            discard: None,
            score_delta: [0; PLAYER_COUNT],
            next: None,
            terminal: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LegalActions {
    pub decision: Decision,
    pub exchange_mask: u32,
    pub discard_mask: u32,
    pub concealed_kong_mask: u32,
    pub added_kong_mask: u32,
    pub can_choose_missing: bool,
    pub can_hu: bool,
    pub can_pong: bool,
    pub can_exposed_kong: bool,
    pub can_pass: bool,
}

impl LegalActions {
    const fn empty(decision: Decision) -> Self {
        Self {
            decision,
            exchange_mask: 0,
            discard_mask: 0,
            concealed_kong_mask: 0,
            added_kong_mask: 0,
            can_choose_missing: false,
            can_hu: false,
            can_pong: false,
            can_exposed_kong: false,
            can_pass: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Error, PartialEq)]
pub enum GameError {
    #[error("the game is already finished")]
    Finished,
    #[error("the action is not legal in the current decision")]
    InvalidAction,
    #[error("the selected exchange tile is invalid")]
    InvalidExchange,
    #[error("batch input and output lengths must match the batch size")]
    BatchLength,
    #[error("batch index is out of range")]
    BatchIndex,
    #[error("event output capacity is larger than the retained event history")]
    EventCapacity,
    #[error("information-set sampling requires a non-terminal decision")]
    InformationSetUnavailable,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Player {
    concealed: [u8; TILE_KIND_COUNT],
    locked: [u8; TILE_KIND_COUNT],
    win_base: [u8; TILE_KIND_COUNT],
    melds: [Option<Meld>; 4],
    meld_count: u8,
    missing: Option<Suit>,
    score: i64,
    has_drawn: bool,
    has_discarded: bool,
    has_won: bool,
    win_count: u32,
    last_win: Option<WinSummary>,
    max_win_multiplier: u32,
    kong_forbidden_mask: u32,
}

impl Player {
    const fn new() -> Self {
        Self {
            concealed: [0; TILE_KIND_COUNT],
            locked: [0; TILE_KIND_COUNT],
            win_base: [0; TILE_KIND_COUNT],
            melds: [None; 4],
            meld_count: 0,
            missing: None,
            score: STARTING_SCORE,
            has_drawn: false,
            has_discarded: false,
            has_won: false,
            win_count: 0,
            last_win: None,
            max_win_multiplier: 0,
            kong_forbidden_mask: 0,
        }
    }

    fn meld_buffer(&self) -> ([Meld; 4], usize) {
        let mut out = [DUMMY_MELD; 4];
        let mut len = 0;
        for meld in self.melds.iter().flatten() {
            out[len] = *meld;
            len += 1;
        }
        (out, len)
    }

    fn add_meld(&mut self, meld: Meld) -> bool {
        if self.meld_count >= 4 {
            return false;
        }
        self.melds[self.meld_count as usize] = Some(meld);
        self.meld_count += 1;
        true
    }

    fn missing_count(&self) -> u8 {
        let Some(suit) = self.missing else {
            return 0;
        };
        let start = suit as usize * 9;
        self.evaluation_counts().unwrap_or(self.concealed)[start..start + 9]
            .iter()
            .copied()
            .sum()
    }

    fn unlocked_count(&self, tile: Tile) -> u8 {
        self.concealed[tile.index()].saturating_sub(self.locked[tile.index()])
    }

    fn unlocked_len(&self) -> usize {
        self.concealed
            .iter()
            .zip(self.locked.iter())
            .map(|(&concealed, &locked)| concealed.saturating_sub(locked) as usize)
            .sum()
    }

    fn evaluation_counts(&self) -> Option<[u8; TILE_KIND_COUNT]> {
        bloodflow_evaluation_counts(&self.concealed, &self.locked, &self.win_base)
    }

    fn win_base_target_len(&self) -> usize {
        13_usize.saturating_sub(3 * self.meld_count as usize)
    }

    fn stabilize_win_base(&mut self) -> bool {
        let target_len = self.win_base_target_len();
        stabilize_win_base(
            &self.concealed,
            &mut self.locked,
            &mut self.win_base,
            target_len,
        )
    }

    fn kong_count(&self, tile: Tile) -> u8 {
        self.unlocked_count(tile)
            .saturating_add(self.win_base[tile.index()])
    }

    const fn is_solvent(&self) -> bool {
        self.score > 0
    }

    fn can_claim_meld(&self, tile: Tile) -> bool {
        self.is_solvent()
            && !self.has_won
            && self.meld_count < 4
            && self.missing != Some(tile.suit())
    }

    fn can_pong(&self, tile: Tile) -> bool {
        self.can_claim_meld(tile) && self.unlocked_count(tile) >= 2 && self.unlocked_len() >= 3
    }

    fn can_exposed_kong(&self, tile: Tile) -> bool {
        self.can_claim_meld(tile) && self.unlocked_count(tile) >= 3
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum TurnOrigin {
    Initial,
    Draw {
        tile: Tile,
        after_kong: bool,
        last_wall_tile: bool,
        deferred_kong_forbidden_tile: Option<Tile>,
    },
    AfterPong,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReactionKind {
    Discard {
        after_kong: bool,
        opening_discard: bool,
        last_wall_tile: bool,
    },
    AddedKong,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PendingMeldKind {
    Pong,
    ExposedKong,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingMeld {
    actor: Seat,
    kind: PendingMeldKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PendingMelds {
    pong: Option<Seat>,
    exposed_kong: Option<Seat>,
}

impl PendingMelds {
    const NONE: Self = Self {
        pong: None,
        exposed_kong: None,
    };

    fn record(&mut self, actor: Seat, kind: PendingMeldKind) {
        let slot = match kind {
            PendingMeldKind::Pong => &mut self.pong,
            PendingMeldKind::ExposedKong => &mut self.exposed_kong,
        };
        slot.get_or_insert(actor);
    }

    fn selected(self) -> Option<PendingMeld> {
        self.exposed_kong
            .map(|actor| PendingMeld {
                actor,
                kind: PendingMeldKind::ExposedKong,
            })
            .or_else(|| {
                self.pong.map(|actor| PendingMeld {
                    actor,
                    kind: PendingMeldKind::Pong,
                })
            })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Stage {
    Exchange {
        actor: Seat,
        selected: u8,
        suit: Option<Suit>,
    },
    ChooseMissing {
        actor: Seat,
    },
    Turn {
        actor: Seat,
        origin: TurnOrigin,
        can_hu: bool,
    },
    HuResponse {
        source: Seat,
        tile: Tile,
        remaining: u8,
        winners: u8,
        pending_melds: PendingMelds,
        kind: ReactionKind,
    },
    MeldResponse {
        source: Seat,
        tile: Tile,
        remaining: u8,
        pending_melds: PendingMelds,
    },
    Finished,
}

/// Authoritative environment state.
///
/// This type intentionally exposes perfect information for simulation,
/// testing, and replay. Policy code must use viewer-scoped observations and
/// [`StepOutcome::for_player`] instead of forwarding raw state or transitions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Game {
    players: [Player; PLAYER_COUNT],
    wall: [u8; WALL_TILE_COUNT],
    wall_head: u8,
    wall_tail: u8,
    dealer: Seat,
    exchange_direction: ExchangeDirection,
    exchange: [[u8; TILE_KIND_COUNT]; PLAYER_COUNT],
    pending_missing: [Option<Suit>; PLAYER_COUNT],
    stage: Stage,
    discards: [u8; WALL_TILE_COUNT],
    discard_owners: [u8; WALL_TILE_COUNT],
    discard_flags: [u8; WALL_TILE_COUNT],
    discard_len: u8,
    final_wall_tile_drawn: bool,
    earthly_hu_open: bool,
    // Multiple winners can reference one physical discard or robbed-Kong tile.
    // Every reference after the first is logical and must not consume inventory.
    duplicate_win_references: [u8; TILE_KIND_COUNT],
    wall_settlement: Option<WallSettlementSummary>,
    transition_draw: Option<DrawEvent>,
    transition_discard: Option<DiscardEvent>,
    events: [StoredEvent; EVENT_HISTORY_CAPACITY],
    event_next: usize,
    event_len: usize,
    event_total: u64,
    event_dropped: u64,
    step_event_start: u64,
    step_event_end: u64,
}

impl Game {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let direction = match rng.bounded(3) {
            0 => ExchangeDirection::Left,
            1 => ExchangeDirection::Across,
            _ => ExchangeDirection::Right,
        };
        Self::with_rng(&mut rng, direction)
    }

    pub fn new_with_direction(seed: u64, direction: ExchangeDirection) -> Self {
        let mut rng = Rng::new(seed);
        Self::with_rng(&mut rng, direction)
    }

    fn with_rng(rng: &mut Rng, direction: ExchangeDirection) -> Self {
        let mut wall = [0_u8; WALL_TILE_COUNT];
        for (index, tile) in wall.iter_mut().enumerate() {
            *tile = (index / TILE_COPIES as usize) as u8;
        }
        rng.shuffle(&mut wall);

        let mut game = Self {
            players: [Player::new(); PLAYER_COUNT],
            wall,
            wall_head: 0,
            wall_tail: WALL_TILE_COUNT as u8,
            dealer: Seat::EAST,
            exchange_direction: direction,
            exchange: [[0; TILE_KIND_COUNT]; PLAYER_COUNT],
            pending_missing: [None; PLAYER_COUNT],
            stage: Stage::Exchange {
                actor: Seat::EAST,
                selected: 0,
                suit: None,
            },
            discards: [0; WALL_TILE_COUNT],
            discard_owners: [0; WALL_TILE_COUNT],
            discard_flags: [0; WALL_TILE_COUNT],
            discard_len: 0,
            final_wall_tile_drawn: false,
            earthly_hu_open: true,
            duplicate_win_references: [0; TILE_KIND_COUNT],
            wall_settlement: None,
            transition_draw: None,
            transition_discard: None,
            events: [StoredEvent::EMPTY; EVENT_HISTORY_CAPACITY],
            event_next: 0,
            event_len: 0,
            event_total: 0,
            event_dropped: 0,
            step_event_start: 0,
            step_event_end: 0,
        };

        for seat in Seat::ALL {
            for _ in 0..13 {
                let tile = game.draw_head_raw().expect("initial wall has enough tiles");
                game.players[seat.index()].concealed[tile.index()] += 1;
            }
        }
        let tile = game.draw_head_raw().expect("initial wall has dealer tile");
        game.players[game.dealer.index()].concealed[tile.index()] += 1;
        game.push_event(
            EventKind::GameStart,
            Some(game.dealer),
            None,
            None,
            game.exchange_direction as u8,
            0,
            0,
            ALL_PLAYER_MASK,
            0,
        );
        game
    }

    pub fn phase(&self) -> Phase {
        match self.stage {
            Stage::Exchange { .. } => Phase::Exchange,
            Stage::ChooseMissing { .. } => Phase::ChooseMissing,
            Stage::Turn { .. } => Phase::Turn,
            Stage::HuResponse { .. } => Phase::HuResponse,
            Stage::MeldResponse { .. } => Phase::MeldResponse,
            Stage::Finished => Phase::Finished,
        }
    }

    pub fn reset(&mut self, seed: u64) {
        *self = Self::new(seed);
    }

    /// Samples a determinization from the current actor's information set.
    ///
    /// The current actor's complete visible observation is held fixed. Other
    /// players' hidden active concealed tiles and the live wall are sampled from
    /// one unknown pool while preserving hand sizes and facts implied by public
    /// missing-suit discards. During exchange,
    /// players with already selected tiles are fixed so their pending exchange
    /// remains valid.
    /// During response windows all hands are fixed because the pending
    /// responder set encodes hand-dependent legality; the live wall is still
    /// independently resampled.
    pub fn resample_information_set(&self, seed: u64) -> Result<Self, GameError> {
        let viewer = self
            .decision()
            .ok_or(GameError::InformationSetUnavailable)?
            .actor;
        let mut fixed_players = seat_bit(viewer);
        match self.stage {
            Stage::Exchange { .. } => {
                for seat in Seat::ALL {
                    if self.exchange[seat.index()].iter().any(|&count| count != 0) {
                        fixed_players |= seat_bit(seat);
                    }
                }
            }
            Stage::HuResponse { .. } | Stage::MeldResponse { .. } => {
                // Candidate masks are derived from concealed hands. Keeping
                // them fixed avoids constructing a state with impossible
                // pending Hu/Pong/Kong responses.
                fixed_players = ALL_PLAYER_MASK;
            }
            Stage::ChooseMissing { .. } | Stage::Turn { .. } => {}
            Stage::Finished => return Err(GameError::InformationSetUnavailable),
        }

        self.resample_hidden_information(seed, fixed_players)
    }

    fn resample_hidden_information(&self, seed: u64, fixed_players: u8) -> Result<Self, GameError> {
        let mut sampled = self.clone();

        let mut unknown = Vec::with_capacity(self.wall_remaining() + 42);
        let mut movable_counts = [0_usize; PLAYER_COUNT];
        for seat in Seat::ALL {
            if fixed_players & seat_bit(seat) != 0 {
                continue;
            }
            let source = &self.players[seat.index()];
            // Historical winning tiles are public. The stable winning base
            // remains hidden and is sampled with the rest of the concealed
            // hand.
            let public_locked = if source.has_won {
                self.public_win_tiles(seat)
            } else {
                source.locked
            };
            let player = &mut sampled.players[seat.index()];
            for (tile_index, &fixed) in public_locked.iter().enumerate() {
                if fixed > player.concealed[tile_index] {
                    return Err(GameError::InformationSetUnavailable);
                }
                let movable = player.concealed[tile_index] - fixed;
                movable_counts[seat.index()] += movable as usize;
                unknown.extend(core::iter::repeat_n(tile_index as u8, movable as usize));
                player.concealed[tile_index] = fixed;
                player.locked[tile_index] = public_locked[tile_index];
                player.win_base[tile_index] = 0;
            }
        }
        unknown.extend(
            self.wall[self.wall_head as usize..self.wall_tail as usize]
                .iter()
                .copied(),
        );

        // Canonicalize the multiset before seeded sampling. Hidden source
        // assignments must not affect a sample from the same public state.
        unknown.sort_unstable();
        let mut rng = Rng::new(seed);

        let mut remaining = [0_u8; TILE_KIND_COUNT];
        for &tile in &unknown {
            remaining[tile as usize] = remaining[tile as usize]
                .checked_add(1)
                .ok_or(GameError::InformationSetUnavailable)?;
        }
        let hidden_winners = Seat::ALL
            .into_iter()
            .filter(|&seat| {
                fixed_players & seat_bit(seat) == 0 && self.players[seat.index()].has_won
            })
            .collect::<Vec<_>>();
        let mut sampled_bases = [[0_u8; TILE_KIND_COUNT]; PLAYER_COUNT];
        if !self.assign_hidden_win_bases(
            &hidden_winners,
            0,
            &mut remaining,
            &mut sampled_bases,
            &mut rng,
        ) {
            return Err(GameError::InformationSetUnavailable);
        }

        for &seat in &hidden_winners {
            let base = sampled_bases[seat.index()];
            let player = &mut sampled.players[seat.index()];
            player.win_base = base;
            for (index, &count) in base.iter().enumerate() {
                player.concealed[index] = player.concealed[index]
                    .checked_add(count)
                    .ok_or(GameError::InformationSetUnavailable)?;
                player.locked[index] = player.locked[index]
                    .checked_add(count)
                    .ok_or(GameError::InformationSetUnavailable)?;
            }
        }

        unknown.clear();
        for (index, &count) in remaining.iter().enumerate() {
            unknown.extend(core::iter::repeat_n(index as u8, count as usize));
        }
        rng.shuffle(&mut unknown);
        let mut cursor = 0;
        for seat in Seat::ALL {
            if fixed_players & seat_bit(seat) != 0 {
                continue;
            }
            let reserved = if self.players[seat.index()].has_won {
                self.players[seat.index()].win_base_target_len()
            } else {
                0
            };
            let remaining_movable = movable_counts[seat.index()]
                .checked_sub(reserved)
                .ok_or(GameError::InformationSetUnavailable)?;
            for _ in 0..remaining_movable {
                let Some(&tile) = unknown.get(cursor) else {
                    return Err(GameError::InformationSetUnavailable);
                };
                sampled.players[seat.index()].concealed[tile as usize] += 1;
                cursor += 1;
            }
        }
        let wall_len = self.wall_remaining();
        let Some(wall_end) = cursor.checked_add(wall_len) else {
            return Err(GameError::InformationSetUnavailable);
        };
        if wall_end != unknown.len() {
            return Err(GameError::InformationSetUnavailable);
        }
        sampled.wall[self.wall_head as usize..self.wall_tail as usize]
            .copy_from_slice(&unknown[cursor..wall_end]);

        Ok(sampled)
    }

    fn assign_hidden_win_bases(
        &self,
        seats: &[Seat],
        position: usize,
        remaining: &mut [u8; TILE_KIND_COUNT],
        sampled_bases: &mut [[u8; TILE_KIND_COUNT]; PLAYER_COUNT],
        rng: &mut Rng,
    ) -> bool {
        let Some(&seat) = seats.get(position) else {
            return true;
        };
        let player = &self.players[seat.index()];
        if player.win_base_target_len() > remaining.iter().map(|&count| count as usize).sum() {
            return false;
        }
        let (melds, meld_count) = player.meld_buffer();
        let public_wins = self.public_win_tiles(seat);
        let order = StableWinBaseSearchOrder::shuffled(rng);
        let available = *remaining;

        visit_stable_win_bases(
            &available,
            &melds[..meld_count],
            player.missing,
            &public_wins,
            &order,
            &mut |base| {
                for index in 0..TILE_KIND_COUNT {
                    remaining[index] -= base[index];
                }
                sampled_bases[seat.index()] = base;

                if self.assign_hidden_win_bases(seats, position + 1, remaining, sampled_bases, rng)
                {
                    return true;
                }

                for index in 0..TILE_KIND_COUNT {
                    remaining[index] += base[index];
                }
                sampled_bases[seat.index()] = [0; TILE_KIND_COUNT];
                false
            },
        )
    }

    /// Clones the state and independently shuffles only the unseen live wall.
    ///
    /// Every hand, public event, pending decision, score, and the consumed
    /// part of the wall remains unchanged. This makes the operation suitable
    /// for rejuvenating future draws after particle-filter resampling.
    pub fn resample_live_wall(&self, seed: u64) -> Self {
        let mut sampled = self.clone();
        let mut rng = Rng::new(seed);
        let live_wall = &mut sampled.wall[sampled.wall_head as usize..sampled.wall_tail as usize];
        // A sample is a function of the remaining multiset and the seed. It
        // must not retain information from the authoritative wall order.
        live_wall.sort_unstable();
        rng.shuffle(live_wall);
        sampled
    }

    pub fn decision(&self) -> Option<Decision> {
        let phase = self.phase();
        let actor = match self.stage {
            Stage::Exchange { actor, .. }
            | Stage::ChooseMissing { actor }
            | Stage::Turn { actor, .. } => actor,
            Stage::HuResponse {
                source, remaining, ..
            }
            | Stage::MeldResponse {
                source, remaining, ..
            } => first_seat_in_mask(source, remaining)?,
            Stage::Finished => return None,
        };
        Some(Decision { actor, phase })
    }

    pub fn legal_actions(&self) -> Option<LegalActions> {
        let decision = self.decision()?;
        let mut legal = LegalActions::empty(decision);
        match self.stage {
            Stage::Exchange {
                actor,
                selected,
                suit,
            } => legal.exchange_mask = self.legal_exchange_mask(actor, selected, suit),
            Stage::ChooseMissing { .. } => legal.can_choose_missing = true,
            Stage::Turn { actor, can_hu, .. } => {
                legal.can_hu = can_hu;
                legal.discard_mask = self.legal_discard_mask(actor);
                (legal.concealed_kong_mask, legal.added_kong_mask) =
                    self.self_declared_kong_masks(actor);
            }
            Stage::HuResponse { tile, kind, .. } => {
                let player = &self.players[decision.actor.index()];
                legal.can_hu = self.can_player_win_with_added_tile(decision.actor, tile);
                if matches!(kind, ReactionKind::Discard { .. }) {
                    legal.can_pong = player.can_pong(tile);
                    legal.can_exposed_kong = player.can_exposed_kong(tile);
                }
                legal.can_pass = true;
            }
            Stage::MeldResponse { tile, .. } => {
                let player = &self.players[decision.actor.index()];
                legal.can_pong = player.can_pong(tile);
                legal.can_exposed_kong = player.can_exposed_kong(tile);
                legal.can_pass = true;
            }
            Stage::Finished => return None,
        }
        Some(legal)
    }

    /// Returns the fixed-size policy mask for the current decision.
    pub fn legal_action_mask(&self) -> Option<ActionMask> {
        let legal = self.legal_actions()?;
        let mut mask = ActionMask::EMPTY;
        mask.insert_tile_mask(legal.exchange_mask, exchange_offset());
        if legal.can_choose_missing {
            for suit in Suit::ALL {
                mask.insert(ActionId::choose_missing(suit));
            }
        }
        mask.insert_tile_mask(legal.discard_mask, discard_offset());
        if legal.can_hu {
            mask.insert(ActionId::HU);
        }
        if legal.can_pong {
            mask.insert(ActionId::PONG);
        }
        if legal.can_exposed_kong {
            mask.insert(ActionId::EXPOSED_KONG);
        }
        mask.insert_tile_mask(legal.concealed_kong_mask, concealed_kong_offset());
        mask.insert_tile_mask(legal.added_kong_mask, added_kong_offset());
        if legal.can_pass {
            mask.insert(ActionId::PASS);
        }
        Some(mask)
    }

    /// Applies an action selected directly from a policy's fixed output head.
    pub fn step_id(&mut self, action: ActionId) -> Result<StepOutcome, GameError> {
        self.step(action.action())
    }

    pub fn step(&mut self, action: Action) -> Result<StepOutcome, GameError> {
        if !self.is_legal_action(action) {
            return match (self.phase(), action) {
                (Phase::Finished, _) => Err(GameError::Finished),
                (Phase::Exchange, Action::SelectExchangeTile(_)) => Err(GameError::InvalidExchange),
                _ => Err(GameError::InvalidAction),
            };
        }
        self.apply_legal_action(action)
    }

    fn apply_legal_action(&mut self, action: Action) -> Result<StepOutcome, GameError> {
        self.transition_draw = None;
        self.transition_discard = None;
        let event_start = self.event_total;
        let decision = self.decision().expect("a legal action has a decision");
        let (action_tile, action_visibility) = match action {
            Action::AddedKong(tile) => (Some(tile), ALL_PLAYER_MASK),
            _ => (None, seat_bit(decision.actor)),
        };
        self.push_event(
            EventKind::Action,
            Some(decision.actor),
            None,
            action_tile,
            0,
            action.id().index() as i32,
            i32::from(decision.phase.code()),
            action_visibility,
            action_visibility,
        );
        let scores_before: [i64; PLAYER_COUNT] =
            core::array::from_fn(|index| self.players[index].score);
        let stage = self.stage;
        match stage {
            Stage::Exchange {
                actor,
                selected,
                suit,
            } => self.step_exchange(actor, selected, suit, action),
            Stage::ChooseMissing { actor } => self.step_missing(actor, action),
            Stage::Turn {
                actor,
                origin,
                can_hu,
            } => self.step_turn(actor, origin, can_hu, action),
            Stage::HuResponse { .. } => self.step_hu_response(action),
            Stage::MeldResponse {
                source,
                tile,
                remaining,
                pending_melds,
            } => self.step_meld_response(source, tile, remaining, pending_melds, action),
            Stage::Finished => Err(GameError::Finished),
        }?;
        if self.phase() == Phase::Finished {
            self.push_event(
                EventKind::GameEnd,
                None,
                None,
                None,
                u8::from(self.wall_remaining() == 0) * EVENT_FLAG_LAST_WALL_TILE,
                0,
                0,
                ALL_PLAYER_MASK,
                0,
            );
        }
        self.step_event_start = event_start;
        self.step_event_end = self.event_total;
        Ok(StepOutcome {
            draw: self.transition_draw,
            discard: self.transition_discard,
            score_delta: core::array::from_fn(|index| {
                self.players[index].score - scores_before[index]
            }),
            next: self.decision(),
            terminal: self.phase() == Phase::Finished,
        })
    }

    pub fn is_legal_action(&self, action: Action) -> bool {
        let Some(legal) = self.legal_actions() else {
            return false;
        };
        match action {
            Action::SelectExchangeTile(tile) => legal.exchange_mask & (1 << tile.index()) != 0,
            Action::ChooseMissing(_) => legal.can_choose_missing,
            Action::Discard(tile) => legal.discard_mask & (1 << tile.index()) != 0,
            Action::Hu => legal.can_hu,
            Action::Pong => legal.can_pong,
            Action::ExposedKong => legal.can_exposed_kong,
            Action::ConcealedKong(tile) => legal.concealed_kong_mask & (1 << tile.index()) != 0,
            Action::AddedKong(tile) => legal.added_kong_mask & (1 << tile.index()) != 0,
            Action::Pass => legal.can_pass,
        }
    }

    pub fn dealer(&self) -> Seat {
        self.dealer
    }

    pub fn exchange_direction(&self) -> ExchangeDirection {
        self.exchange_direction
    }

    /// Tiles already selected by a player during the exchange phase.
    pub fn exchange_selection(&self, seat: Seat) -> &[u8; TILE_KIND_COUNT] {
        &self.exchange[seat.index()]
    }

    pub fn wall_remaining(&self) -> usize {
        self.wall_tail.saturating_sub(self.wall_head) as usize
    }

    pub fn current_draw(&self) -> Option<DrawEvent> {
        match self.stage {
            Stage::Turn {
                actor,
                origin:
                    TurnOrigin::Draw {
                        tile, after_kong, ..
                    },
                ..
            } => Some(DrawEvent {
                player: actor,
                tile,
                replacement: after_kong,
            }),
            _ => None,
        }
    }

    /// Number of retained viewer-independent events in the ring.
    pub fn event_count(&self) -> usize {
        self.event_len
    }

    /// Number of events overwritten since the game was created or reset.
    pub fn event_dropped(&self) -> u64 {
        self.event_dropped
    }

    /// Number of events emitted by the most recent successful step.
    pub fn step_event_count(&self) -> usize {
        (self.step_event_end - self.step_event_start) as usize
    }

    /// Copies the most recent visible events into a caller-owned flat buffer.
    ///
    /// The buffer length must be a multiple of [`EVENT_RECORD_WIDTH`] and its
    /// capacity may be smaller than [`EVENT_HISTORY_CAPACITY`], in which case
    /// only the newest visible records are copied. Records are ordered from
    /// oldest to newest, use relative seats for fields 1 and 2, and are filled
    /// with `-1` before writing. A draw event remains public while its tile
    /// field is masked to `-1` for every viewer other than the drawer.
    pub fn events_into(&self, viewer: Seat, output: &mut [i32]) -> Result<usize, GameError> {
        if output.len() % EVENT_RECORD_WIDTH != 0 {
            return Err(GameError::BatchLength);
        }
        let capacity = output.len() / EVENT_RECORD_WIDTH;
        if capacity > EVENT_HISTORY_CAPACITY {
            return Err(GameError::EventCapacity);
        }
        self.write_events(viewer, output)
    }

    /// Copies only the events emitted by the most recent successful step.
    /// This is the compact form intended for a training loop.
    pub fn step_events_into(&self, viewer: Seat, output: &mut [i32]) -> Result<usize, GameError> {
        if output.len() % EVENT_RECORD_WIDTH != 0 {
            return Err(GameError::BatchLength);
        }
        let capacity = output.len() / EVENT_RECORD_WIDTH;
        if capacity > EVENT_HISTORY_CAPACITY {
            return Err(GameError::EventCapacity);
        }
        self.write_event_range(viewer, self.step_event_start, self.step_event_end, output)
    }

    #[allow(clippy::too_many_arguments)]
    fn push_event(
        &mut self,
        kind: EventKind,
        actor: Option<Seat>,
        target: Option<Seat>,
        tile: Option<Tile>,
        flags: u8,
        value: i32,
        aux: i32,
        visible_to: u8,
        tile_visible_to: u8,
    ) {
        let mut record = [-1_i32; EVENT_RECORD_WIDTH];
        record[0] = i32::from(kind.code());
        record[1] = actor.map_or(-1, |seat| i32::from(seat.as_u8()));
        record[2] = target.map_or(-1, |seat| i32::from(seat.as_u8()));
        record[3] = tile.map_or(-1, |tile| i32::from(tile.as_u8()));
        record[4] = i32::from(flags);
        record[5] = value;
        record[6] = aux;
        self.events[self.event_next] = StoredEvent {
            record,
            visible_to,
            tile_visible_to,
        };
        self.event_next = (self.event_next + 1) % EVENT_HISTORY_CAPACITY;
        self.event_total = self.event_total.saturating_add(1);
        if self.event_len < EVENT_HISTORY_CAPACITY {
            self.event_len += 1;
        } else {
            self.event_dropped = self.event_dropped.saturating_add(1);
        }
    }

    fn write_events(&self, viewer: Seat, output: &mut [i32]) -> Result<usize, GameError> {
        debug_assert_eq!(output.len() % EVENT_RECORD_WIDTH, 0);
        self.write_event_range(viewer, 0, self.event_total, output)
    }

    fn write_event_range(
        &self,
        viewer: Seat,
        requested_start: u64,
        requested_end: u64,
        output: &mut [i32],
    ) -> Result<usize, GameError> {
        debug_assert_eq!(output.len() % EVENT_RECORD_WIDTH, 0);
        let capacity = output.len() / EVENT_RECORD_WIDTH;
        output.fill(-1);
        if capacity == 0 {
            return Ok(0);
        }

        let oldest = self.event_total.saturating_sub(self.event_len as u64);
        let start = requested_start.max(oldest).min(self.event_total);
        let end = requested_end.max(start).min(self.event_total);
        let mut retained_start = start;
        let mut retained = 0;
        for sequence in (start..end).rev() {
            if self.event_at(sequence).visible_to & seat_bit(viewer) == 0 {
                continue;
            }
            retained += 1;
            retained_start = sequence;
            if retained == capacity {
                break;
            }
        }
        let mut written = 0;
        for sequence in retained_start..end {
            let event = self.event_at(sequence);
            if event.visible_to & seat_bit(viewer) == 0 {
                continue;
            }
            if written >= capacity {
                break;
            }
            let mut record = event.record;
            if record[1] >= 0 {
                let seat = Seat::new(record[1] as u8).expect("event actor seat is valid");
                record[1] = i32::from(relative_seat(viewer, seat));
            }
            if record[2] >= 0 {
                let seat = Seat::new(record[2] as u8).expect("event target seat is valid");
                record[2] = i32::from(relative_seat(viewer, seat));
            }
            if record[0] == i32::from(EventKind::Draw.code())
                && event.tile_visible_to & seat_bit(viewer) == 0
            {
                record[3] = -1;
            }
            let offset = written * EVENT_RECORD_WIDTH;
            output[offset..offset + EVENT_RECORD_WIDTH].copy_from_slice(&record);
            written += 1;
        }
        Ok(written)
    }

    fn event_at(&self, sequence: u64) -> &StoredEvent {
        let oldest = self.event_total - self.event_len as u64;
        let offset = (sequence - oldest) as usize;
        let start =
            (self.event_next + EVENT_HISTORY_CAPACITY - self.event_len) % EVENT_HISTORY_CAPACITY;
        &self.events[(start + offset) % EVENT_HISTORY_CAPACITY]
    }

    pub fn concealed(&self, seat: Seat) -> &[u8; TILE_KIND_COUNT] {
        &self.players[seat.index()].concealed
    }

    /// Returns the number of concealed tile references held by a player.
    ///
    /// The count is public in the observation schema. It does not reveal tile
    /// identities and remains valid after repeated blood-flow wins.
    pub(crate) fn concealed_len(&self, seat: Seat) -> usize {
        self.players[seat.index()]
            .concealed
            .iter()
            .map(|&count| usize::from(count))
            .sum()
    }

    pub fn locked(&self, seat: Seat) -> &[u8; TILE_KIND_COUNT] {
        &self.players[seat.index()].locked
    }

    /// Returns the winning tiles accumulated by this player.
    ///
    /// The stable continuation base remains concealed. Only the separated
    /// winning-tile references are public.
    pub fn public_win_tiles(&self, seat: Seat) -> [u8; TILE_KIND_COUNT] {
        let player = &self.players[seat.index()];
        core::array::from_fn(|index| {
            player.locked[index]
                .checked_sub(player.win_base[index])
                .expect("the stable win base must be a subset of locked tiles")
        })
    }

    /// Returns the hidden stable continuation base from authoritative state.
    ///
    /// This is perfect information. Policy code must use viewer-scoped
    /// observations instead of reading this value.
    pub fn win_base(&self, seat: Seat) -> &[u8; TILE_KIND_COUNT] {
        &self.players[seat.index()].win_base
    }

    pub fn score(&self, seat: Seat) -> i64 {
        self.players[seat.index()].score
    }

    pub fn missing_suit(&self, seat: Seat) -> Option<Suit> {
        self.players[seat.index()].missing
    }

    pub fn meld_count(&self, seat: Seat) -> usize {
        self.players[seat.index()].meld_count as usize
    }

    pub fn meld(&self, seat: Seat, index: usize) -> Option<Meld> {
        self.players[seat.index()]
            .melds
            .get(index)
            .copied()
            .flatten()
    }

    pub fn has_won(&self, seat: Seat) -> bool {
        self.players[seat.index()].has_won
    }

    /// Number of completed wins for one player in this game.
    pub fn win_count(&self, seat: Seat) -> u32 {
        self.players[seat.index()].win_count
    }

    /// Most recent completed win for one player, if any.
    pub fn last_win(&self, seat: Seat) -> Option<WinSummary> {
        self.players[seat.index()].last_win
    }

    /// Highest shape multiplier from this player's completed wins.
    pub fn max_win_multiplier(&self, seat: Seat) -> u32 {
        self.players[seat.index()].max_win_multiplier
    }

    /// Viewer-relative win summaries for UI presentation.
    ///
    /// Each player record is `[win_count, last_shape_multiplier,
    /// last_multiplier, last_pattern_bits, last_event_flags]`. A player with
    /// no completed win has `-1` in the four last-win fields.
    pub fn player_ui_stats_into(&self, viewer: Seat, output: &mut [i32]) -> Result<(), GameError> {
        if output.len() != PLAYER_UI_STATS_WIDTH {
            return Err(GameError::BatchLength);
        }
        output.fill(-1);
        for relative in 0..PLAYER_COUNT {
            let player = &self.players[viewer.offset(relative as u8).index()];
            let base = relative * PLAYER_UI_STATS_FIELDS;
            output[base] = player.win_count.min(i32::MAX as u32) as i32;
            if let Some(last_win) = player.last_win {
                output[base + 1] = last_win.shape_multiplier.min(i32::MAX as u32) as i32;
                output[base + 2] = last_win.multiplier.min(i32::MAX as u32) as i32;
                output[base + 3] = last_win.patterns as i32;
                output[base + 4] = i32::from(last_win.flags);
            }
        }
        Ok(())
    }

    /// End-of-wall settlement facts, available only after wall settlement.
    pub fn wall_settlement(&self) -> Option<&WallSettlementSummary> {
        self.wall_settlement.as_ref()
    }

    /// Writes viewer-relative end-of-wall settlement metadata and hands.
    ///
    /// Each metadata record is `[flower_pig, ready, max_shape_multiplier]`.
    /// Revealed concealed hands are four consecutive 27-tile histograms. If
    /// no wall settlement is public, this method clears both buffers and
    /// returns `false`.
    pub fn wall_settlement_into(
        &self,
        viewer: Seat,
        meta: &mut [i32],
        hands: &mut [u8],
    ) -> Result<bool, GameError> {
        if meta.len() != WALL_SETTLEMENT_META_WIDTH || hands.len() != WALL_SETTLEMENT_HANDS_WIDTH {
            return Err(GameError::BatchLength);
        }
        meta.fill(-1);
        hands.fill(0);
        let Some(summary) = self.wall_settlement() else {
            return Ok(false);
        };
        debug_assert_eq!(self.phase(), Phase::Finished);
        for relative in 0..PLAYER_COUNT {
            let seat = viewer.offset(relative as u8);
            let meta_base = relative * WALL_SETTLEMENT_FIELDS;
            meta[meta_base] = i32::from(u8::from(summary.is_flower_pig(seat)));
            meta[meta_base + 1] = i32::from(u8::from(summary.is_ready(seat)));
            meta[meta_base + 2] = summary.max_shape_multiplier(seat).min(i32::MAX as u32) as i32;
            let hand_base = relative * TILE_KIND_COUNT;
            hands[hand_base..hand_base + TILE_KIND_COUNT]
                .copy_from_slice(summary.revealed_hand(seat));
        }
        Ok(true)
    }

    /// Returns conventional structural shanten and improving tiles for a seat.
    ///
    /// This authoritative accessor can inspect any seat, just like
    /// [`Game::concealed`]. Policy code should request only its own seat.
    pub fn hand_analysis(&self, seat: Seat) -> ShantenAnalysis {
        let player = &self.players[seat.index()];
        let (melds, len) = player.meld_buffer();
        player.evaluation_counts().map_or(
            ShantenAnalysis {
                shanten: SHANTEN_TERMINAL,
                improving_tiles: 0,
            },
            |counts| analyze_shanten(&counts, &melds[..len], player.missing),
        )
    }

    pub fn discards(&self) -> impl Iterator<Item = (Seat, Tile)> + '_ {
        (0..self.discard_len as usize).map(|index| {
            (
                Seat::new(self.discard_owners[index]).expect("stored seat is valid"),
                Tile::new(self.discards[index]).expect("stored tile is valid"),
            )
        })
    }

    /// Counts tiles known to `viewer` from private and public information.
    ///
    /// The result includes the viewer's concealed hand, other players' public
    /// winning tiles, exposed meld contributions, and the discard river. A
    /// discarded tile claimed by one or more winners is counted once even
    /// though the authoritative winner hands retain a reference to it.
    pub(crate) fn visible_tile_counts(&self, viewer: Seat) -> [u8; TILE_KIND_COUNT] {
        let mut counts = [0_u16; TILE_KIND_COUNT];
        add_tile_counts(&mut counts, &self.players[viewer.index()].concealed);

        for seat in Seat::ALL {
            let player = &self.players[seat.index()];
            if seat != viewer {
                add_tile_counts(&mut counts, &self.public_win_tiles(seat));
            }
            for meld in player.melds.iter().flatten() {
                let amount = match meld.kind {
                    MeldKind::Pong => 2,
                    MeldKind::ExposedKong | MeldKind::AddedKong => 3,
                    MeldKind::ConcealedKong => 4,
                };
                counts[meld.tile.index()] += amount;
            }
        }
        for &tile in &self.discards[..self.discard_len as usize] {
            counts[tile as usize] += 1;
        }

        core::array::from_fn(|index| {
            let known =
                counts[index].saturating_sub(u16::from(self.duplicate_win_references[index]));
            debug_assert!(
                known <= u16::from(TILE_COPIES),
                "visible tile count exceeds physical inventory: viewer={viewer:?}, tile={index}, raw={}, own_concealed={}, concealed_by_seat={:?}, public_wins_by_seat={:?}, meld_extra_by_seat={:?}, river={}, duplicate_win_references={}, known={known}, phase={:?}, discards={}",
                counts[index],
                self.players[viewer.index()].concealed[index],
                Seat::ALL.map(|seat| self.players[seat.index()].concealed[index]),
                Seat::ALL.map(|seat| self.public_win_tiles(seat)[index]),
                Seat::ALL.map(|seat| self.players[seat.index()]
                    .melds
                    .iter()
                    .flatten()
                    .filter(|meld| meld.tile.index() == index)
                    .map(|meld| match meld.kind {
                        MeldKind::Pong => 2_u16,
                        MeldKind::ExposedKong | MeldKind::AddedKong => 3,
                        MeldKind::ConcealedKong => 4,
                    })
                    .sum::<u16>()),
                self.discards[..self.discard_len as usize]
                    .iter()
                    .filter(|&&tile| usize::from(tile) == index)
                    .count(),
                self.duplicate_win_references[index],
                self.phase(),
                self.discard_len,
            );
            known.min(u16::from(u8::MAX)) as u8
        })
    }

    pub fn rankings(&self) -> [Seat; PLAYER_COUNT] {
        let mut seats = Seat::ALL;
        seats.sort_by(|left, right| {
            self.score(*right)
                .cmp(&self.score(*left))
                .then_with(|| left.index().cmp(&right.index()))
        });
        seats
    }

    pub fn termination_reason(&self) -> Option<TerminationReason> {
        if self.phase() != Phase::Finished {
            return None;
        }
        Some(if self.wall_remaining() == 0 {
            TerminationReason::WallExhausted
        } else {
            TerminationReason::ThreePlayersBankrupt
        })
    }

    fn write_observation(
        &self,
        viewer: Seat,
        tile_obs: &mut [u8],
        melds: &mut [u8],
        river: &mut [u8],
        meta: &mut [i32],
    ) {
        debug_assert_eq!(tile_obs.len(), TILE_OBSERVATION_WIDTH);
        debug_assert_eq!(melds.len(), MELD_OBSERVATION_WIDTH);
        debug_assert_eq!(river.len(), RIVER_OBSERVATION_WIDTH);
        debug_assert_eq!(meta.len(), META_OBSERVATION_WIDTH);

        let decision = self.decision();

        tile_obs.fill(0);
        tile_obs[..TILE_KIND_COUNT].copy_from_slice(&self.players[viewer.index()].concealed);
        tile_obs[TILE_KIND_COUNT..2 * TILE_KIND_COUNT]
            .copy_from_slice(&self.exchange[viewer.index()]);
        for relative in 0..PLAYER_COUNT {
            let seat = viewer.offset(relative as u8);
            let start = (2 + relative) * TILE_KIND_COUNT;
            if seat == viewer {
                tile_obs[start..start + TILE_KIND_COUNT]
                    .copy_from_slice(&self.players[seat.index()].locked);
            } else {
                tile_obs[start..start + TILE_KIND_COUNT]
                    .copy_from_slice(&self.public_win_tiles(seat));
            }
        }
        for index in 0..self.discard_len as usize {
            let owner = Seat::new(self.discard_owners[index]).expect("stored seat is valid");
            let relative = relative_seat(viewer, owner) as usize;
            let tile = self.discards[index] as usize;
            let offset = (6 + relative) * TILE_KIND_COUNT + tile;
            tile_obs[offset] = tile_obs[offset].saturating_add(1);
        }
        let mut kong_forbidden_mask = self.players[viewer.index()].kong_forbidden_mask;
        if let Stage::Turn {
            actor,
            origin:
                TurnOrigin::Draw {
                    deferred_kong_forbidden_tile: Some(tile),
                    ..
                },
            ..
        } = self.stage
            && actor == viewer
        {
            kong_forbidden_mask |= 1 << tile.index();
        }
        let kong_forbidden_start = 10 * TILE_KIND_COUNT;
        for index in 0..TILE_KIND_COUNT {
            if kong_forbidden_mask & (1 << index) != 0 {
                tile_obs[kong_forbidden_start + index] = 1;
            }
        }

        melds.fill(u8::MAX);
        for relative in 0..PLAYER_COUNT {
            let seat = viewer.offset(relative as u8);
            for index in 0..self.players[seat.index()].meld_count as usize {
                let meld = self.players[seat.index()].melds[index]
                    .expect("occupied meld slots are contiguous");
                let offset = (relative * 4 + index) * 3;
                melds[offset] = meld.tile.as_u8();
                melds[offset + 1] = meld.kind.code();
                melds[offset + 2] = relative_seat(viewer, meld.source);
            }
        }

        river.fill(u8::MAX);
        for index in 0..self.discard_len as usize {
            let owner = Seat::new(self.discard_owners[index]).expect("stored seat is valid");
            river[index * 2] = self.discards[index];
            river[index * 2 + 1] = relative_seat(viewer, owner);
        }

        let draw = self.current_draw();
        let (pending_source, pending_tile) = match self.stage {
            Stage::HuResponse { source, tile, .. } | Stage::MeldResponse { source, tile, .. } => (
                i32::from(relative_seat(viewer, source)),
                i32::from(tile.as_u8()),
            ),
            _ => (-1, -1),
        };
        let (exchange_selected, exchange_suit) = match self.stage {
            Stage::Exchange { selected, suit, .. } => (
                i32::from(selected),
                suit.map_or(-1, |suit| i32::from(suit as u8)),
            ),
            _ => (0, -1),
        };
        let reaction_flags = match self.stage {
            Stage::HuResponse {
                kind: ReactionKind::AddedKong,
                ..
            } => 1,
            Stage::HuResponse {
                kind:
                    ReactionKind::Discard {
                        after_kong,
                        opening_discard,
                        last_wall_tile,
                    },
                ..
            } => {
                i32::from(u8::from(after_kong)) << 1
                    | i32::from(u8::from(opening_discard)) << 2
                    | i32::from(u8::from(last_wall_tile)) << 3
            }
            _ => 0,
        };
        meta[0] = i32::from(self.phase().code());
        meta[1] = decision.map_or(-1, |decision| i32::from(decision.actor.as_u8()));
        meta[2] = i32::from(relative_seat(viewer, self.dealer));
        meta[3] = i32::from(self.exchange_direction as u8);
        meta[4] = self.wall_remaining() as i32;
        meta[5] = draw.map_or(-1, |draw| {
            if draw.player == viewer {
                i32::from(draw.tile.as_u8())
            } else {
                -1
            }
        });
        meta[6] = draw.map_or(0, |draw| i32::from(u8::from(draw.replacement)));
        meta[7] = pending_source;
        meta[8] = pending_tile;
        meta[9] = i32::from(self.discard_len);
        meta[10] = exchange_selected;
        meta[11] = exchange_suit;
        for relative in 0..PLAYER_COUNT {
            let player = &self.players[viewer.offset(relative as u8).index()];
            meta[12 + relative] = player.score as i32;
            meta[16 + relative] = player.missing.map_or(-1, |suit| i32::from(suit as u8));
            meta[20 + relative] = i32::from(u8::from(player.has_won));
            meta[24 + relative] = player.concealed.iter().map(|&count| i32::from(count)).sum();
            meta[30 + relative] = player.max_win_multiplier as i32;
        }
        meta[28] = i32::from(u8::from(self.phase() == Phase::Finished));
        meta[29] = reaction_flags;
    }

    /// Writes the ordinary partial-information observation for one viewer.
    pub fn observation_into(
        &self,
        viewer: Seat,
        tile_obs: &mut [u8],
        melds: &mut [u8],
        river: &mut [u8],
        meta: &mut [i32],
    ) -> Result<(), GameError> {
        if tile_obs.len() != TILE_OBSERVATION_WIDTH
            || melds.len() != MELD_OBSERVATION_WIDTH
            || river.len() != RIVER_OBSERVATION_WIDTH
            || meta.len() != META_OBSERVATION_WIDTH
        {
            return Err(GameError::BatchLength);
        }
        self.write_observation(viewer, tile_obs, melds, river, meta);
        Ok(())
    }

    /// Writes perfect-information tile counts for Critic training only.
    ///
    /// Planes `0..4` are concealed hands. Planes `4..8` are their complete
    /// locked subsets, including the stable winning base and historical
    /// winning-tile references. Plane `8` is an unordered live-wall histogram.
    pub fn oracle_tile_counts_into(&self, output: &mut [u8]) -> Result<(), GameError> {
        if output.len() != ORACLE_TILE_COUNT_WIDTH {
            return Err(GameError::BatchLength);
        }
        output.fill(0);
        for seat in Seat::ALL {
            let concealed_start = seat.index() * TILE_KIND_COUNT;
            output[concealed_start..concealed_start + TILE_KIND_COUNT]
                .copy_from_slice(&self.players[seat.index()].concealed);
            let locked_start = (PLAYER_COUNT + seat.index()) * TILE_KIND_COUNT;
            output[locked_start..locked_start + TILE_KIND_COUNT]
                .copy_from_slice(&self.players[seat.index()].locked);
        }
        let wall = &mut output[(ORACLE_TILE_COUNT_PLANES - 1) * TILE_KIND_COUNT..];
        for &tile in &self.wall[self.wall_head as usize..self.wall_tail as usize] {
            wall[tile as usize] += 1;
        }
        Ok(())
    }

    fn step_exchange(
        &mut self,
        actor: Seat,
        selected: u8,
        suit: Option<Suit>,
        action: Action,
    ) -> Result<(), GameError> {
        let Action::SelectExchangeTile(tile) = action else {
            return Err(GameError::InvalidAction);
        };
        if self.legal_exchange_mask(actor, selected, suit) & (1 << tile.index()) == 0 {
            return Err(GameError::InvalidExchange);
        }
        self.exchange[actor.index()][tile.index()] += 1;
        if selected < 2 {
            self.stage = Stage::Exchange {
                actor,
                selected: selected + 1,
                suit: Some(suit.unwrap_or(tile.suit())),
            };
        } else if actor.index() + 1 < PLAYER_COUNT {
            self.stage = Stage::Exchange {
                actor: actor.next(),
                selected: 0,
                suit: None,
            };
        } else {
            self.apply_exchange();
            self.push_event(
                EventKind::ExchangeComplete,
                None,
                None,
                None,
                self.exchange_direction as u8,
                0,
                0,
                ALL_PLAYER_MASK,
                0,
            );
            self.stage = Stage::ChooseMissing { actor: Seat::EAST };
        }
        Ok(())
    }

    fn legal_exchange_mask(&self, actor: Seat, selected: u8, suit: Option<Suit>) -> u32 {
        let concealed = &self.players[actor.index()].concealed;
        let already_selected = &self.exchange[actor.index()];
        let mut mask = 0_u32;
        for index in 0..TILE_KIND_COUNT {
            if concealed[index] <= already_selected[index] {
                continue;
            }
            let tile = Tile::from_index_unchecked(index as u8);
            if let Some(suit) = suit {
                if tile.suit() == suit {
                    mask |= 1 << index;
                }
                continue;
            }
            debug_assert_eq!(selected, 0);
            let start = tile.suit() as usize * 9;
            let held_in_suit: u8 = concealed[start..start + 9].iter().copied().sum();
            if held_in_suit >= 3 {
                mask |= 1 << index;
            }
        }
        mask
    }

    fn apply_exchange(&mut self) {
        for seat in Seat::ALL {
            for tile in 0..TILE_KIND_COUNT {
                self.players[seat.index()].concealed[tile] -= self.exchange[seat.index()][tile];
            }
        }
        for sender in Seat::ALL {
            let receiver = sender.offset(self.exchange_direction.offset());
            for tile in 0..TILE_KIND_COUNT {
                self.players[receiver.index()].concealed[tile] +=
                    self.exchange[sender.index()][tile];
            }
        }
    }

    fn step_missing(&mut self, actor: Seat, action: Action) -> Result<(), GameError> {
        let Action::ChooseMissing(suit) = action else {
            return Err(GameError::InvalidAction);
        };
        self.pending_missing[actor.index()] = Some(suit);
        if actor.index() + 1 < PLAYER_COUNT {
            self.stage = Stage::ChooseMissing {
                actor: actor.next(),
            };
        } else {
            for seat in Seat::ALL {
                self.players[seat.index()].missing = self.pending_missing[seat.index()];
                self.push_event(
                    EventKind::MissingRevealed,
                    Some(seat),
                    None,
                    None,
                    0,
                    self.pending_missing[seat.index()].map_or(-1, |suit| suit as i32),
                    0,
                    ALL_PLAYER_MASK,
                    0,
                );
            }
            let can_hu = self.can_player_win(self.dealer, None);
            self.push_event(
                EventKind::TurnStart,
                Some(self.dealer),
                None,
                None,
                0,
                0,
                1,
                ALL_PLAYER_MASK,
                0,
            );
            self.stage = Stage::Turn {
                actor: self.dealer,
                origin: TurnOrigin::Initial,
                can_hu,
            };
        }
        Ok(())
    }

    fn step_turn(
        &mut self,
        actor: Seat,
        origin: TurnOrigin,
        can_hu: bool,
        action: Action,
    ) -> Result<(), GameError> {
        if let TurnOrigin::Draw {
            deferred_kong_forbidden_tile: Some(tile),
            ..
        } = origin
        {
            self.players[actor.index()].kong_forbidden_mask |= 1 << tile.index();
        }
        match action {
            Action::Hu if can_hu => self.resolve_self_draw(actor, origin),
            Action::Discard(tile) => self.discard(actor, tile, origin),
            Action::ConcealedKong(tile) => self.concealed_kong(actor, tile),
            Action::AddedKong(tile) => self.propose_added_kong(actor, tile),
            _ => Err(GameError::InvalidAction),
        }
    }

    fn step_hu_response(&mut self, action: Action) -> Result<(), GameError> {
        let Stage::HuResponse {
            source,
            tile,
            mut remaining,
            mut winners,
            mut pending_melds,
            kind,
        } = self.stage
        else {
            unreachable!("Hu response dispatch requires a HuResponse stage");
        };
        let actor = first_seat_in_mask(source, remaining).ok_or(GameError::InvalidAction)?;
        match action {
            Action::Hu if self.can_player_win_with_added_tile(actor, tile) => {
                winners |= seat_bit(actor);
            }
            Action::Pong
                if matches!(kind, ReactionKind::Discard { .. })
                    && self.players[actor.index()].can_pong(tile) =>
            {
                pending_melds.record(actor, PendingMeldKind::Pong);
            }
            Action::ExposedKong
                if matches!(kind, ReactionKind::Discard { .. })
                    && self.players[actor.index()].can_exposed_kong(tile) =>
            {
                pending_melds.record(actor, PendingMeldKind::ExposedKong);
            }
            Action::Pass => {}
            _ => return Err(GameError::InvalidAction),
        }
        remaining &= !seat_bit(actor);
        if remaining != 0 {
            self.stage = Stage::HuResponse {
                source,
                tile,
                remaining,
                winners,
                pending_melds,
                kind,
            };
            return Ok(());
        }

        if winners != 0 {
            self.resolve_discard_wins(source, tile, winners, kind);
        } else if let Some(claim) = pending_melds.selected() {
            debug_assert!(matches!(kind, ReactionKind::Discard { .. }));
            return match claim.kind {
                PendingMeldKind::Pong => self.pong(claim.actor, source, tile),
                PendingMeldKind::ExposedKong => self.exposed_kong(claim.actor, source, tile),
            };
        } else {
            match kind {
                ReactionKind::Discard { .. } => self.begin_next_normal_turn(source.next()),
                ReactionKind::AddedKong => self.complete_added_kong(source, tile),
            }
        }
        Ok(())
    }

    fn step_meld_response(
        &mut self,
        source: Seat,
        tile: Tile,
        mut remaining: u8,
        mut pending_melds: PendingMelds,
        action: Action,
    ) -> Result<(), GameError> {
        let actor = first_seat_in_mask(source, remaining).ok_or(GameError::InvalidAction)?;
        let player = &self.players[actor.index()];
        let can_pong = player.can_pong(tile);
        let can_kong = player.can_exposed_kong(tile);
        match action {
            Action::Pong if can_pong => pending_melds.record(actor, PendingMeldKind::Pong),
            Action::ExposedKong if can_kong => {
                pending_melds.record(actor, PendingMeldKind::ExposedKong);
            }
            Action::Pass => {}
            _ => return Err(GameError::InvalidAction),
        }
        remaining &= !seat_bit(actor);
        if remaining == 0 {
            if let Some(claim) = pending_melds.selected() {
                return match claim.kind {
                    PendingMeldKind::Pong => self.pong(claim.actor, source, tile),
                    PendingMeldKind::ExposedKong => self.exposed_kong(claim.actor, source, tile),
                };
            }
            self.begin_next_normal_turn(source.next());
        } else {
            self.stage = Stage::MeldResponse {
                source,
                tile,
                remaining,
                pending_melds,
            };
        }
        Ok(())
    }

    fn self_declared_kong_masks(&self, actor: Seat) -> (u32, u32) {
        let player = &self.players[actor.index()];
        let mut concealed = 0_u32;
        let mut added = 0_u32;

        // Self-declared Kongs use active tiles plus the stable base. Historical
        // winning-tile references never participate in a later Kong.
        if player.meld_count < 4 {
            for index in 0..TILE_KIND_COUNT {
                let tile = Tile::from_index_unchecked(index as u8);
                if player.missing == Some(tile.suit())
                    || player.kong_forbidden_mask & (1 << index) != 0
                {
                    continue;
                }
                if player.kong_count(tile) >= 4 {
                    concealed |= 1 << index;
                }
            }
        }
        for meld in player.melds.iter().flatten() {
            if meld.kind == MeldKind::Pong
                && player.kong_count(meld.tile) > 0
                && player.missing != Some(meld.tile.suit())
                && player.kong_forbidden_mask & (1 << meld.tile.index()) == 0
            {
                added |= 1 << meld.tile.index();
            }
        }
        (concealed, added)
    }

    fn legal_discard_mask(&self, actor: Seat) -> u32 {
        let player = &self.players[actor.index()];
        if player.has_won {
            let Stage::Turn {
                actor: turn_actor,
                origin: TurnOrigin::Draw { tile, .. },
                ..
            } = self.stage
            else {
                return 0;
            };
            return if turn_actor == actor && player.unlocked_count(tile) != 0 {
                1 << tile.index()
            } else {
                0
            };
        }

        let mut unlocked = 0_u32;
        for index in 0..TILE_KIND_COUNT {
            if player.concealed[index] > player.locked[index] {
                unlocked |= 1 << index;
            }
        }
        unlocked
    }

    fn discard(&mut self, actor: Seat, tile: Tile, origin: TurnOrigin) -> Result<(), GameError> {
        self.players[actor.index()].concealed[tile.index()] -= 1;
        self.players[actor.index()].has_discarded = true;
        if self.players[actor.index()].has_won {
            let stabilized = self.players[actor.index()].stabilize_win_base();
            debug_assert!(stabilized, "a post-Kong discard must restore the win base");
        }
        let after_kong = matches!(
            origin,
            TurnOrigin::Draw {
                after_kong: true,
                ..
            }
        );
        let last_wall_tile = self.final_wall_tile_drawn;
        let tsumogiri = matches!(origin, TurnOrigin::Draw { tile: drawn, .. } if drawn == tile);
        let opening_discard =
            actor == self.dealer && origin == TurnOrigin::Initial && self.discard_len == 0;
        let mut discard_flags = 0;
        if after_kong {
            discard_flags |= EVENT_FLAG_AFTER_KONG;
        }
        if opening_discard {
            discard_flags |= EVENT_FLAG_OPENING_DISCARD;
        }
        if last_wall_tile {
            discard_flags |= EVENT_FLAG_LAST_WALL_TILE;
        }
        if tsumogiri {
            discard_flags |= EVENT_FLAG_TSUMOGIRI;
        }
        if origin == TurnOrigin::AfterPong {
            discard_flags |= EVENT_FLAG_AFTER_PONG;
        }

        let discard_index = self.discard_len as usize;
        self.discards[discard_index] = tile.as_u8();
        self.discard_owners[discard_index] = actor.as_u8();
        self.discard_flags[discard_index] = discard_flags;
        self.discard_len += 1;
        self.transition_discard = Some(DiscardEvent {
            player: actor,
            tile,
        });
        self.push_event(
            EventKind::Discard,
            Some(actor),
            None,
            Some(tile),
            discard_flags,
            0,
            0,
            ALL_PLAYER_MASK,
            ALL_PLAYER_MASK,
        );
        let mut hu_candidates = 0_u8;
        let mut response_candidates = 0_u8;
        for seat in Seat::ALL {
            if seat == actor {
                continue;
            }
            if self.players[seat.index()].can_exposed_kong(tile) {
                self.players[seat.index()].kong_forbidden_mask |= 1 << tile.index();
            }
            if self.can_player_win_with_added_tile(seat, tile) {
                hu_candidates |= seat_bit(seat);
            }
            if self.can_respond_to_reaction(
                seat,
                tile,
                ReactionKind::Discard {
                    after_kong,
                    opening_discard,
                    last_wall_tile,
                },
            ) {
                response_candidates |= seat_bit(seat);
            }
        }
        if hu_candidates != 0 {
            self.stage = Stage::HuResponse {
                source: actor,
                tile,
                remaining: response_candidates,
                winners: 0,
                pending_melds: PendingMelds::NONE,
                kind: ReactionKind::Discard {
                    after_kong,
                    opening_discard,
                    last_wall_tile,
                },
            };
        } else {
            self.begin_meld_responses(actor, tile);
        }
        Ok(())
    }

    fn begin_meld_responses(&mut self, source: Seat, tile: Tile) {
        let mut candidates = 0_u8;
        for seat in Seat::ALL {
            if seat == source {
                continue;
            }
            let player = &self.players[seat.index()];
            if player.can_pong(tile) || player.can_exposed_kong(tile) {
                candidates |= seat_bit(seat);
            }
        }
        if candidates == 0 {
            self.begin_next_normal_turn(source.next());
        } else {
            self.stage = Stage::MeldResponse {
                source,
                tile,
                remaining: candidates,
                pending_melds: PendingMelds::NONE,
            };
        }
    }

    fn pong(&mut self, actor: Seat, source: Seat, tile: Tile) -> Result<(), GameError> {
        self.earthly_hu_open = false;
        self.remove_for_meld(actor, tile, 2, false);
        let added = self.players[actor.index()].add_meld(Meld {
            tile,
            kind: MeldKind::Pong,
            source,
        });
        debug_assert!(added);
        self.push_event(
            EventKind::Meld,
            Some(actor),
            Some(source),
            Some(tile),
            MeldKind::Pong.code(),
            0,
            0,
            ALL_PLAYER_MASK,
            ALL_PLAYER_MASK,
        );
        self.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::AfterPong,
            can_hu: false,
        };
        Ok(())
    }

    fn exposed_kong(&mut self, actor: Seat, source: Seat, tile: Tile) -> Result<(), GameError> {
        self.earthly_hu_open = false;
        self.remove_for_meld(actor, tile, 3, false);
        let added = self.players[actor.index()].add_meld(Meld {
            tile,
            kind: MeldKind::ExposedKong,
            source,
        });
        debug_assert!(added);
        self.push_event(
            EventKind::Meld,
            Some(actor),
            Some(source),
            Some(tile),
            MeldKind::ExposedKong.code(),
            0,
            0,
            ALL_PLAYER_MASK,
            ALL_PLAYER_MASK,
        );
        self.transfer(source, actor, SCORE_UNIT * 3);
        if self.finish_if_one_solvent() {
            return Ok(());
        }
        self.begin_supplement_turn(actor);
        Ok(())
    }

    fn concealed_kong(&mut self, actor: Seat, tile: Tile) -> Result<(), GameError> {
        self.earthly_hu_open = false;
        self.remove_for_meld(actor, tile, 4, true);
        let added = self.players[actor.index()].add_meld(Meld {
            tile,
            kind: MeldKind::ConcealedKong,
            source: actor,
        });
        debug_assert!(added);
        self.push_event(
            EventKind::Meld,
            Some(actor),
            None,
            Some(tile),
            MeldKind::ConcealedKong.code(),
            0,
            0,
            ALL_PLAYER_MASK,
            ALL_PLAYER_MASK,
        );
        for payer in seats_after(actor) {
            self.transfer(payer, actor, SCORE_UNIT * 2);
        }
        if self.finish_if_one_solvent() {
            return Ok(());
        }
        self.begin_supplement_turn(actor);
        Ok(())
    }

    fn propose_added_kong(&mut self, actor: Seat, tile: Tile) -> Result<(), GameError> {
        self.earthly_hu_open = false;
        let mut candidates = 0_u8;
        for seat in Seat::ALL {
            if seat != actor && self.can_player_win_with_added_tile(seat, tile) {
                candidates |= seat_bit(seat);
            }
        }
        if candidates == 0 {
            self.complete_added_kong(actor, tile);
        } else {
            self.stage = Stage::HuResponse {
                source: actor,
                tile,
                remaining: candidates,
                winners: 0,
                pending_melds: PendingMelds::NONE,
                kind: ReactionKind::AddedKong,
            };
        }
        Ok(())
    }

    fn complete_added_kong(&mut self, actor: Seat, tile: Tile) {
        self.remove_for_meld(actor, tile, 1, true);
        let meld = self.players[actor.index()]
            .melds
            .iter_mut()
            .flatten()
            .find(|meld| meld.kind == MeldKind::Pong && meld.tile == tile)
            .expect("legal added kong has matching pong");
        meld.kind = MeldKind::AddedKong;
        self.push_event(
            EventKind::Meld,
            Some(actor),
            None,
            Some(tile),
            MeldKind::AddedKong.code(),
            0,
            0,
            ALL_PLAYER_MASK,
            ALL_PLAYER_MASK,
        );
        for payer in seats_after(actor) {
            self.transfer(payer, actor, SCORE_UNIT);
        }
        if !self.finish_if_one_solvent() {
            self.begin_supplement_turn(actor);
        }
    }

    fn resolve_self_draw(&mut self, actor: Seat, origin: TurnOrigin) -> Result<(), GameError> {
        let opening_self_draw = {
            let player = &self.players[actor.index()];
            !player.has_discarded && !player.has_won
        };
        let (required, mut flags) = match origin {
            TurnOrigin::Initial => (None, WinFlags::NONE),
            TurnOrigin::Draw {
                tile,
                after_kong,
                last_wall_tile,
                ..
            } => (
                Some(tile),
                WinFlags {
                    after_kong_draw: after_kong,
                    last_wall_tile,
                    ..WinFlags::NONE
                },
            ),
            TurnOrigin::AfterPong => return Err(GameError::InvalidAction),
        };
        flags.heavenly = actor == self.dealer && opening_self_draw;
        flags.earthly = actor != self.dealer && opening_self_draw && self.earthly_hu_open;
        let mut evaluation = self
            .evaluate_player(actor, required, flags)
            .ok_or(GameError::InvalidAction)?;
        evaluation.multiplier = evaluation.multiplier.max(2);
        let event_flags = win_event_flags(flags, true);
        let winning_tile = self.apply_win(actor, required, evaluation, event_flags);
        self.push_event(
            EventKind::Hu,
            Some(actor),
            None,
            Some(winning_tile),
            event_flags,
            evaluation.multiplier as i32,
            evaluation.patterns.bits() as i32,
            ALL_PLAYER_MASK,
            ALL_PLAYER_MASK,
        );
        let payment = SCORE_UNIT * i64::from(evaluation.multiplier);
        for payer in seats_after(actor) {
            self.transfer(payer, actor, payment);
        }
        if !self.finish_if_one_solvent() {
            self.begin_next_normal_turn(actor.next());
        }
        Ok(())
    }

    fn resolve_discard_wins(&mut self, source: Seat, tile: Tile, winners: u8, kind: ReactionKind) {
        match kind {
            ReactionKind::Discard { .. } => self.consume_latest_discard(source, tile),
            ReactionKind::AddedKong => self.remove_robbed_added_kong_tile(source, tile),
        }

        let next_actor = if winners.count_ones() > 1 {
            source.next()
        } else {
            first_seat_in_mask(source, winners)
                .expect("at least one discard winner")
                .next()
        };
        for (resolved_winners, winner) in seats_in_mask_after(source, winners).enumerate() {
            if resolved_winners > 0 {
                self.duplicate_win_references[tile.index()] =
                    self.duplicate_win_references[tile.index()].saturating_add(1);
            }
            self.players[winner.index()].concealed[tile.index()] =
                self.players[winner.index()].concealed[tile.index()].saturating_add(1);
            let flags = match kind {
                ReactionKind::Discard {
                    after_kong,
                    opening_discard: _,
                    last_wall_tile,
                } => WinFlags {
                    after_kong_discard: after_kong,
                    last_wall_tile,
                    earthly: false,
                    ..WinFlags::NONE
                },
                ReactionKind::AddedKong => WinFlags {
                    rob_kong: true,
                    ..WinFlags::NONE
                },
            };
            let evaluation = self
                .evaluate_player(winner, Some(tile), flags)
                .expect("response legality was checked");
            let event_flags = win_event_flags(flags, false);
            let winning_tile = self.apply_win(winner, Some(tile), evaluation, event_flags);
            self.push_event(
                EventKind::Hu,
                Some(winner),
                Some(source),
                Some(winning_tile),
                event_flags,
                evaluation.multiplier as i32,
                evaluation.patterns.bits() as i32,
                ALL_PLAYER_MASK,
                ALL_PLAYER_MASK,
            );
            self.transfer(
                source,
                winner,
                SCORE_UNIT * i64::from(evaluation.multiplier),
            );
        }
        if !self.finish_if_one_solvent() {
            self.begin_next_normal_turn(next_actor);
        }
    }

    fn consume_latest_discard(&mut self, source: Seat, tile: Tile) {
        let index = self
            .discard_len
            .checked_sub(1)
            .expect("a discard win must claim the latest river tile") as usize;
        debug_assert_eq!(self.discards[index], tile.as_u8());
        debug_assert_eq!(self.discard_owners[index], source.as_u8());
        self.discard_len -= 1;
    }

    fn apply_win(
        &mut self,
        actor: Seat,
        required: Option<Tile>,
        evaluation: WinEvaluation,
        event_flags: u8,
    ) -> Tile {
        self.earthly_hu_open = false;
        let player = &mut self.players[actor.index()];
        let winning_tile = apply_bloodflow_win(
            &player.concealed,
            &mut player.locked,
            &mut player.win_base,
            player.has_won,
            &evaluation.used,
            required,
        )
        .expect("a legal win must match the player's physical hand");
        player.has_won = true;
        player.win_count = player.win_count.saturating_add(1);
        player.last_win = Some(WinSummary {
            shape_multiplier: evaluation.shape_multiplier,
            multiplier: evaluation.multiplier,
            patterns: evaluation.patterns.bits(),
            flags: event_flags,
        });
        player.max_win_multiplier = player.max_win_multiplier.max(evaluation.shape_multiplier);
        winning_tile
    }

    fn can_player_win(&self, actor: Seat, required: Option<Tile>) -> bool {
        let player = &self.players[actor.index()];
        if !player.is_solvent() || player.missing_count() != 0 {
            return false;
        }
        let (melds, len) = player.meld_buffer();
        if player.has_won {
            player.evaluation_counts().is_some_and(|counts| {
                is_bloodflow_winning(&counts, &player.win_base, &melds[..len], required)
            })
        } else {
            is_winning(&player.concealed, &melds[..len], required)
        }
    }

    pub(crate) fn can_player_win_with_added_tile(&self, actor: Seat, tile: Tile) -> bool {
        let player = &self.players[actor.index()];
        if !player.is_solvent()
            || player.missing_count() != 0
            || player.missing == Some(tile.suit())
        {
            return false;
        }
        let (melds, len) = player.meld_buffer();
        if player.has_won {
            let Some(mut counts) = player.evaluation_counts() else {
                return false;
            };
            let Some(count) = counts[tile.index()].checked_add(1) else {
                return false;
            };
            counts[tile.index()] = count;
            is_bloodflow_winning(&counts, &player.win_base, &melds[..len], Some(tile))
        } else {
            let mut counts = player.concealed;
            counts[tile.index()] = counts[tile.index()].saturating_add(1);
            is_winning(&counts, &melds[..len], Some(tile))
        }
    }

    fn can_respond_to_reaction(&self, actor: Seat, tile: Tile, kind: ReactionKind) -> bool {
        if self.can_player_win_with_added_tile(actor, tile) {
            return true;
        }
        if kind == ReactionKind::AddedKong {
            return false;
        }
        let player = &self.players[actor.index()];
        player.can_pong(tile) || player.can_exposed_kong(tile)
    }

    fn evaluate_player(
        &self,
        actor: Seat,
        required: Option<Tile>,
        flags: WinFlags,
    ) -> Option<WinEvaluation> {
        let player = &self.players[actor.index()];
        if player.missing_count() != 0 {
            return None;
        }
        let (melds, len) = player.meld_buffer();
        if player.has_won {
            let counts = player.evaluation_counts()?;
            evaluate_bloodflow_win(&counts, &player.win_base, &melds[..len], required, flags)
        } else {
            evaluate_win(&player.concealed, &melds[..len], required, flags)
        }
    }

    fn begin_next_normal_turn(&mut self, start: Seat) {
        if self
            .players
            .iter()
            .filter(|player| player.is_solvent())
            .count()
            <= 1
        {
            self.stage = Stage::Finished;
            return;
        }
        let Some(actor) = self.next_solvent_from(start) else {
            self.stage = Stage::Finished;
            return;
        };
        self.begin_normal_turn(actor);
    }

    fn next_solvent_from(&self, start: Seat) -> Option<Seat> {
        (0..PLAYER_COUNT)
            .map(|offset| start.offset(offset as u8))
            .find(|seat| self.players[seat.index()].is_solvent())
    }

    fn begin_normal_turn(&mut self, actor: Seat) {
        debug_assert!(self.players[actor.index()].is_solvent());
        let Some(tile) = self.draw_head_raw() else {
            self.finish_wall_game();
            return;
        };
        self.players[actor.index()].concealed[tile.index()] =
            self.players[actor.index()].concealed[tile.index()].saturating_add(1);
        self.players[actor.index()].has_drawn = true;
        self.transition_draw = Some(DrawEvent {
            player: actor,
            tile,
            replacement: false,
        });
        let last_wall_tile = self.wall_remaining() == 0;
        self.final_wall_tile_drawn |= last_wall_tile;
        self.push_event(
            EventKind::Draw,
            Some(actor),
            None,
            Some(tile),
            u8::from(last_wall_tile) * EVENT_FLAG_LAST_WALL_TILE,
            self.wall_remaining() as i32,
            0,
            ALL_PLAYER_MASK,
            seat_bit(actor),
        );
        let can_hu = self.can_player_win(actor, Some(tile));
        let deferred_kong_forbidden_tile =
            (self.self_declared_kong_masks(actor).1 != 0).then_some(tile);
        self.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::Draw {
                tile,
                after_kong: false,
                last_wall_tile,
                deferred_kong_forbidden_tile,
            },
            can_hu,
        };
    }

    fn begin_supplement_turn(&mut self, actor: Seat) {
        let Some(tile) = self.draw_tail_raw() else {
            self.finish_wall_game();
            return;
        };
        self.players[actor.index()].concealed[tile.index()] =
            self.players[actor.index()].concealed[tile.index()].saturating_add(1);
        self.players[actor.index()].has_drawn = true;
        self.transition_draw = Some(DrawEvent {
            player: actor,
            tile,
            replacement: true,
        });
        let last_wall_tile = self.wall_remaining() == 0;
        self.final_wall_tile_drawn |= last_wall_tile;
        self.push_event(
            EventKind::Draw,
            Some(actor),
            None,
            Some(tile),
            EVENT_FLAG_REPLACEMENT_DRAW | (u8::from(last_wall_tile) * EVENT_FLAG_LAST_WALL_TILE),
            self.wall_remaining() as i32,
            0,
            ALL_PLAYER_MASK,
            seat_bit(actor),
        );
        let can_hu = self.can_player_win(actor, Some(tile));
        let deferred_kong_forbidden_tile =
            (self.self_declared_kong_masks(actor).1 != 0).then_some(tile);
        self.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::Draw {
                tile,
                after_kong: true,
                last_wall_tile,
                deferred_kong_forbidden_tile,
            },
            can_hu,
        };
    }

    fn draw_head_raw(&mut self) -> Option<Tile> {
        if self.wall_head >= self.wall_tail {
            return None;
        }
        let tile = Tile::new(self.wall[self.wall_head as usize]);
        self.wall_head += 1;
        tile
    }

    fn draw_tail_raw(&mut self) -> Option<Tile> {
        if self.wall_head >= self.wall_tail {
            return None;
        }
        self.wall_tail -= 1;
        Tile::new(self.wall[self.wall_tail as usize])
    }

    fn remove_for_meld(&mut self, actor: Seat, tile: Tile, amount: u8, allow_win_base: bool) {
        let player = &mut self.players[actor.index()];
        let removed = remove_tiles_for_meld(
            &mut player.concealed,
            &mut player.locked,
            &mut player.win_base,
            tile,
            amount,
            allow_win_base,
        );
        debug_assert!(removed, "a legal meld must have enough matching tiles");
    }

    fn remove_robbed_added_kong_tile(&mut self, actor: Seat, tile: Tile) {
        let player = &mut self.players[actor.index()];
        let removed = remove_tiles_for_meld(
            &mut player.concealed,
            &mut player.locked,
            &mut player.win_base,
            tile,
            1,
            true,
        );
        debug_assert!(removed, "a proposed added Kong owns its fourth tile");
        if player.has_won {
            let stabilized = player.stabilize_win_base();
            debug_assert!(stabilized, "a robbed added Kong must restore the win base");
        }
    }

    fn transfer(&mut self, payer: Seat, payee: Seat, requested: i64) -> i64 {
        if !self.players[payer.index()].is_solvent() || !self.players[payee.index()].is_solvent() {
            return 0;
        }
        let amount = requested.min(self.players[payer.index()].score).max(0);
        self.players[payer.index()].score -= amount;
        self.players[payee.index()].score += amount;
        if amount > 0 {
            self.push_event(
                EventKind::Payment,
                Some(payer),
                Some(payee),
                None,
                0,
                amount.min(i64::from(i32::MAX)) as i32,
                0,
                ALL_PLAYER_MASK,
                0,
            );
        }
        amount
    }

    fn finish_if_one_solvent(&mut self) -> bool {
        if self
            .players
            .iter()
            .filter(|player| player.is_solvent())
            .count()
            <= 1
        {
            self.stage = Stage::Finished;
            true
        } else {
            false
        }
    }

    fn finish_wall_game(&mut self) {
        let mut flower_pig = [false; PLAYER_COUNT];

        for seat in Seat::ALL {
            let player = &self.players[seat.index()];
            flower_pig[seat.index()] = player.is_solvent() && player.missing_count() != 0;
        }

        self.wall_settlement = Some(WallSettlementSummary {
            flower_pig,
            ready: [false; PLAYER_COUNT],
            max_shape_multipliers: [0; PLAYER_COUNT],
            revealed_hands: self
                .players
                .map(|player| player.evaluation_counts().unwrap_or([0; TILE_KIND_COUNT])),
        });
        self.push_event(
            EventKind::SettlementStage,
            None,
            None,
            None,
            SettlementStage::FlowerPig.code(),
            0,
            0,
            ALL_PLAYER_MASK,
            0,
        );

        for payer in Seat::ALL {
            if !flower_pig[payer.index()] {
                continue;
            }
            for payee in seats_after(payer) {
                if !flower_pig[payee.index()] {
                    self.transfer(payer, payee, SCORE_UNIT * 10);
                }
            }
            if self.finish_if_one_solvent() {
                return;
            }
        }

        let mut ready = [false; PLAYER_COUNT];
        let mut max_shape_multipliers = [0_u32; PLAYER_COUNT];
        for seat in Seat::ALL {
            let wait_multiplier = self.max_wait_multiplier(seat);
            ready[seat.index()] = wait_multiplier > 0;
            max_shape_multipliers[seat.index()] = wait_multiplier;
        }
        let summary = self
            .wall_settlement
            .as_mut()
            .expect("wall settlement was initialized before scoring");
        summary.ready = ready;
        summary.max_shape_multipliers = max_shape_multipliers;

        self.push_event(
            EventKind::SettlementStage,
            None,
            None,
            None,
            SettlementStage::Dajiao.code(),
            0,
            0,
            ALL_PLAYER_MASK,
            0,
        );

        for payer in Seat::ALL {
            if ready[payer.index()] {
                continue;
            }
            for payee in seats_after(payer) {
                if ready[payee.index()] {
                    let multiplier = max_shape_multipliers[payee.index()].max(1);
                    self.transfer(payer, payee, SCORE_UNIT * i64::from(multiplier));
                }
            }
            if self.finish_if_one_solvent() {
                return;
            }
        }
        self.stage = Stage::Finished;
    }

    fn max_wait_multiplier(&self, actor: Seat) -> u32 {
        let player = &self.players[actor.index()];
        if !player.is_solvent() {
            return 0;
        }
        let (melds, len) = player.meld_buffer();
        if player.has_won {
            player
                .evaluation_counts()
                .and_then(|counts| {
                    evaluate_bloodflow_max_wait(
                        &counts,
                        &player.win_base,
                        &melds[..len],
                        player.missing,
                    )
                })
                .map_or(0, |wait| wait.evaluation.multiplier)
        } else {
            evaluate_max_wait(&player.concealed, &melds[..len], player.missing)
                .map_or(0, |wait| wait.evaluation.multiplier)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Batch {
    games: Vec<Game>,
}

impl Batch {
    pub fn new(size: usize, seed: u64) -> Self {
        let create = |index: usize| Game::new(batch_seed(seed, index));
        let games = if size >= PARALLEL_BATCH_THRESHOLD {
            (0..size).into_par_iter().map(create).collect()
        } else {
            (0..size).map(create).collect()
        };
        Self { games }
    }

    pub fn len(&self) -> usize {
        self.games.len()
    }

    pub fn is_empty(&self) -> bool {
        self.games.is_empty()
    }

    /// Clones selected environments into an independent batch.
    ///
    /// Repeated indices intentionally create identical counterfactual states,
    /// which is useful for branching several legal actions from one decision.
    pub fn clone_indices(&self, indices: &[usize]) -> Result<Self, GameError> {
        if indices.iter().any(|&index| index >= self.games.len()) {
            return Err(GameError::BatchIndex);
        }
        let clone_game = |&index: &usize| self.games[index].clone();
        let games = if indices.len() >= PARALLEL_BATCH_THRESHOLD {
            indices.par_iter().map(clone_game).collect()
        } else {
            indices.iter().map(clone_game).collect()
        };
        Ok(Self { games })
    }

    /// Removes a strictly increasing set of rows with at most one game move
    /// per removal and returns each survivor's original row index.
    pub fn remove_indices_swap(&mut self, indices: &[usize]) -> Result<Vec<usize>, GameError> {
        if indices.iter().any(|&index| index >= self.games.len())
            || indices.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(GameError::BatchIndex);
        }
        let mut original_rows = (0..self.games.len()).collect::<Vec<_>>();
        for &index in indices.iter().rev() {
            self.games.swap_remove(index);
            original_rows.swap_remove(index);
        }
        Ok(original_rows)
    }

    /// Clones selected environments and independently resamples each clone
    /// from its current actor's information set.
    pub fn resample_information_sets(
        &self,
        indices: &[usize],
        seeds: &[u64],
    ) -> Result<Self, GameError> {
        if indices.len() != seeds.len() {
            return Err(GameError::BatchLength);
        }
        if indices.iter().any(|&index| index >= self.games.len()) {
            return Err(GameError::BatchIndex);
        }
        let sample =
            |(&index, &seed): (&usize, &u64)| self.games[index].resample_information_set(seed);
        let games = if indices.len() >= PARALLEL_BATCH_THRESHOLD {
            indices
                .par_iter()
                .zip(seeds.par_iter())
                .map(sample)
                .collect::<Result<Vec<_>, _>>()?
        } else {
            indices
                .iter()
                .zip(seeds.iter())
                .map(sample)
                .collect::<Result<Vec<_>, _>>()?
        };
        Ok(Self { games })
    }

    /// Clones selected environments and independently shuffles their live walls.
    pub fn resample_live_walls(&self, indices: &[usize], seeds: &[u64]) -> Result<Self, GameError> {
        if indices.len() != seeds.len() {
            return Err(GameError::BatchLength);
        }
        if indices.iter().any(|&index| index >= self.games.len()) {
            return Err(GameError::BatchIndex);
        }
        let sample = |(&index, &seed): (&usize, &u64)| self.games[index].resample_live_wall(seed);
        let games = if indices.len() >= PARALLEL_BATCH_THRESHOLD {
            indices
                .par_iter()
                .zip(seeds.par_iter())
                .map(sample)
                .collect()
        } else {
            indices.iter().zip(seeds.iter()).map(sample).collect()
        };
        Ok(Self { games })
    }

    /// Writes perfect-information Critic inputs as `[batch, 9, 27]` planes.
    pub fn oracle_tile_counts_into(&self, output: &mut [u8]) -> Result<(), GameError> {
        if self.games.len().checked_mul(ORACLE_TILE_COUNT_WIDTH) != Some(output.len()) {
            return Err(GameError::BatchLength);
        }
        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(output.par_chunks_mut(ORACLE_TILE_COUNT_WIDTH))
                .try_for_each(|(game, row)| game.oracle_tile_counts_into(row))?;
        } else {
            for (game, row) in self
                .games
                .iter()
                .zip(output.chunks_mut(ORACLE_TILE_COUNT_WIDTH))
            {
                game.oracle_tile_counts_into(row)?;
            }
        }
        Ok(())
    }

    /// Writes the number of overwritten events for each environment.
    pub fn event_dropped_into(&self, output: &mut [u64]) -> Result<(), GameError> {
        if output.len() != self.games.len() {
            return Err(GameError::BatchLength);
        }
        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(output.par_iter_mut())
                .for_each(|(game, dropped)| *dropped = game.event_dropped());
        } else {
            for (game, dropped) in self.games.iter().zip(output.iter_mut()) {
                *dropped = game.event_dropped();
            }
        }
        Ok(())
    }

    pub fn games(&self) -> &[Game] {
        &self.games
    }

    pub fn games_mut(&mut self) -> &mut [Game] {
        &mut self.games
    }

    pub fn reset_at(&mut self, index: usize, seed: u64) -> Result<(), GameError> {
        let Some(game) = self.games.get_mut(index) else {
            return Err(GameError::BatchIndex);
        };
        game.reset(seed);
        Ok(())
    }

    pub fn reset_all(&mut self, seed: u64) {
        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, game)| game.reset(batch_seed(seed, index)));
        } else {
            for (index, game) in self.games.iter_mut().enumerate() {
                game.reset(batch_seed(seed, index));
            }
        }
    }

    pub fn legal_actions_into(&self, output: &mut [Option<LegalActions>]) -> Result<(), GameError> {
        if output.len() != self.games.len() {
            return Err(GameError::BatchLength);
        }
        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(output.par_iter_mut())
                .for_each(|(game, slot)| *slot = game.legal_actions());
        } else {
            for (game, slot) in self.games.iter().zip(output.iter_mut()) {
                *slot = game.legal_actions();
            }
        }
        Ok(())
    }

    pub fn legal_action_masks_into(
        &self,
        output: &mut [Option<ActionMask>],
    ) -> Result<(), GameError> {
        if output.len() != self.games.len() {
            return Err(GameError::BatchLength);
        }
        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(output.par_iter_mut())
                .for_each(|(game, slot)| *slot = game.legal_action_mask());
        } else {
            for (game, slot) in self.games.iter().zip(output.iter_mut()) {
                *slot = game.legal_action_mask();
            }
        }
        Ok(())
    }

    /// Writes two packed legal-action mask words per environment.
    ///
    /// Terminal environments are represented by two zero words. The caller
    /// owns the output allocation, so this method does not allocate.
    pub fn legal_action_mask_words_into(&self, output: &mut [u64]) -> Result<(), GameError> {
        if self.games.len().checked_mul(LEGAL_ACTION_MASK_WORDS) != Some(output.len()) {
            return Err(GameError::BatchLength);
        }
        let write = |game: &Game, words: &mut [u64]| {
            if let Some(mask) = game.legal_action_mask() {
                words.copy_from_slice(mask.words());
            } else {
                words.fill(0);
            }
        };
        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(output.par_chunks_mut(LEGAL_ACTION_MASK_WORDS))
                .for_each(|(game, words)| write(game, words));
        } else {
            for (game, words) in self
                .games
                .iter()
                .zip(output.chunks_mut(LEGAL_ACTION_MASK_WORDS))
            {
                write(game, words);
            }
        }
        Ok(())
    }

    /// Writes current-actor conventional shanten and improving-tile masks.
    ///
    /// Terminal slots receive [`SHANTEN_TERMINAL`] and an empty mask. The
    /// output buffers are `[batch]` arrays owned by the caller.
    pub fn hand_analysis_into(
        &self,
        shanten: &mut [i8],
        improving_tiles: &mut [u32],
    ) -> Result<(), GameError> {
        if shanten.len() != self.games.len() || improving_tiles.len() != self.games.len() {
            return Err(GameError::BatchLength);
        }
        let write = |game: &Game, shanten: &mut i8, improving_tiles: &mut u32| {
            let Some(decision) = game.decision() else {
                *shanten = SHANTEN_TERMINAL;
                *improving_tiles = 0;
                return;
            };
            let analysis = game.hand_analysis(decision.actor);
            *shanten = analysis.shanten;
            *improving_tiles = analysis.improving_tiles;
        };
        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(shanten.par_iter_mut())
                .zip(improving_tiles.par_iter_mut())
                .for_each(|((game, shanten), improving_tiles)| {
                    write(game, shanten, improving_tiles);
                });
        } else {
            for ((game, shanten), improving_tiles) in self
                .games
                .iter()
                .zip(shanten.iter_mut())
                .zip(improving_tiles.iter_mut())
            {
                write(game, shanten, improving_tiles);
            }
        }
        Ok(())
    }

    /// Writes hand analysis only for selected batch rows.
    ///
    /// Indices are validated before any output is changed. Outputs are compact
    /// arrays in the same order as `indices`.
    pub fn hand_analysis_indices_into(
        &self,
        indices: &[u32],
        shanten: &mut [i8],
        improving_tiles: &mut [u32],
    ) -> Result<(), GameError> {
        if shanten.len() != indices.len() || improving_tiles.len() != indices.len() {
            return Err(GameError::BatchLength);
        }
        if indices
            .iter()
            .any(|&index| index as usize >= self.games.len())
        {
            return Err(GameError::BatchIndex);
        }
        let write = |index: u32, shanten: &mut i8, improving_tiles: &mut u32| {
            let game = &self.games[index as usize];
            let Some(decision) = game.decision() else {
                *shanten = SHANTEN_TERMINAL;
                *improving_tiles = 0;
                return;
            };
            let analysis = game.hand_analysis(decision.actor);
            *shanten = analysis.shanten;
            *improving_tiles = analysis.improving_tiles;
        };
        if indices.len() >= PARALLEL_BATCH_THRESHOLD {
            indices
                .par_iter()
                .copied()
                .zip(shanten.par_iter_mut())
                .zip(improving_tiles.par_iter_mut())
                .for_each(|((index, shanten), improving_tiles)| {
                    write(index, shanten, improving_tiles);
                });
        } else {
            for ((&index, shanten), improving_tiles) in indices
                .iter()
                .zip(shanten.iter_mut())
                .zip(improving_tiles.iter_mut())
            {
                write(index, shanten, improving_tiles);
            }
        }
        Ok(())
    }

    /// Writes current-actor observations into four caller-owned flat buffers.
    ///
    /// Seat-indexed channels and metadata are rotated so relative seat zero is
    /// the current actor. A terminal game uses the dealer as relative seat zero.
    /// Tile channels are actor concealed, actor exchange selection, the
    /// actor's complete locked subset, three opponents' public historical
    /// winning-tile histograms, and four per-owner discard counts.
    /// Meld records are
    /// `[tile, kind, source_relative]`; unused records contain `255`. River
    /// records are chronological `[tile, owner_relative]` discards and retain
    /// claimed tiles; unused records contain `255`.
    ///
    /// Metadata is `[phase, actor_absolute, dealer_relative,
    /// exchange_direction, wall_remaining, current_draw_tile, replacement,
    /// pending_source_relative, pending_tile, discard_len,
    /// exchange_selected_count, exchange_suit, scores[4], missing_suits[4],
    /// has_won[4], concealed_sizes[4], terminal, reaction_flags,
    /// max_win_multipliers[4]]`. Missing optional values use `-1`.
    /// `reaction_flags` uses bit 0 for rob-kong, bit 1 for an after-kong
    /// discard, bit 2 for the dealer's opening discard, and bit 3 for a discard
    /// of the final live-wall tile.
    pub fn observations_into(
        &self,
        tile_obs: &mut [u8],
        melds: &mut [u8],
        river: &mut [u8],
        meta: &mut [i32],
    ) -> Result<(), GameError> {
        if self.games.len().checked_mul(TILE_OBSERVATION_WIDTH) != Some(tile_obs.len())
            || self.games.len().checked_mul(MELD_OBSERVATION_WIDTH) != Some(melds.len())
            || self.games.len().checked_mul(RIVER_OBSERVATION_WIDTH) != Some(river.len())
            || self.games.len().checked_mul(META_OBSERVATION_WIDTH) != Some(meta.len())
        {
            return Err(GameError::BatchLength);
        }

        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(tile_obs.par_chunks_mut(TILE_OBSERVATION_WIDTH))
                .zip(melds.par_chunks_mut(MELD_OBSERVATION_WIDTH))
                .zip(river.par_chunks_mut(RIVER_OBSERVATION_WIDTH))
                .zip(meta.par_chunks_mut(META_OBSERVATION_WIDTH))
                .for_each(|((((game, tile_obs), melds), river), meta)| {
                    let viewer = game
                        .decision()
                        .map_or(game.dealer(), |decision| decision.actor);
                    game.write_observation(viewer, tile_obs, melds, river, meta);
                });
        } else {
            for ((((game, tile_obs), melds), river), meta) in self
                .games
                .iter()
                .zip(tile_obs.chunks_mut(TILE_OBSERVATION_WIDTH))
                .zip(melds.chunks_mut(MELD_OBSERVATION_WIDTH))
                .zip(river.chunks_mut(RIVER_OBSERVATION_WIDTH))
                .zip(meta.chunks_mut(META_OBSERVATION_WIDTH))
            {
                let viewer = game
                    .decision()
                    .map_or(game.dealer(), |decision| decision.actor);
                game.write_observation(viewer, tile_obs, melds, river, meta);
            }
        }
        Ok(())
    }

    /// Writes the latest viewer-scoped event history for every environment.
    ///
    /// `capacity` is the number of records reserved per environment. The
    /// output is `[batch, capacity, EVENT_RECORD_WIDTH]` when viewed as a
    /// multidimensional array and `lengths` receives the number of visible
    /// records written to each row. The viewer is the current decision actor,
    /// or the dealer for a terminal environment.
    pub fn events_into(
        &self,
        capacity: usize,
        output: &mut [i32],
        lengths: &mut [u16],
    ) -> Result<(), GameError> {
        if capacity > EVENT_HISTORY_CAPACITY
            || self.games.len().checked_mul(capacity * EVENT_RECORD_WIDTH) != Some(output.len())
            || lengths.len() != self.games.len()
        {
            return if capacity > EVENT_HISTORY_CAPACITY {
                Err(GameError::EventCapacity)
            } else {
                Err(GameError::BatchLength)
            };
        }
        if capacity == 0 {
            lengths.fill(0);
            return Ok(());
        }
        let write = |game: &Game, records: &mut [i32], length: &mut u16| {
            let viewer = game
                .decision()
                .map_or(game.dealer(), |decision| decision.actor);
            *length = game
                .write_events(viewer, records)
                .expect("batch event buffers were validated") as u16;
        };
        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(output.par_chunks_mut(capacity * EVENT_RECORD_WIDTH))
                .zip(lengths.par_iter_mut())
                .for_each(|((game, records), length)| write(game, records, length));
        } else {
            for ((game, records), length) in self
                .games
                .iter()
                .zip(output.chunks_mut(capacity * EVENT_RECORD_WIDTH))
                .zip(lengths.iter_mut())
            {
                write(game, records, length);
            }
        }
        Ok(())
    }

    /// Writes viewer-scoped event history only for selected current viewers.
    ///
    /// Each byte in `history_seat_masks` is a seat bitset. An environment's
    /// history is written only when its current viewer's bit is set. Otherwise
    /// its event row is untouched and its returned length is zero.
    pub fn events_masked_into(
        &self,
        history_seat_masks: &[u8],
        capacity: usize,
        output: &mut [i32],
        lengths: &mut [u16],
    ) -> Result<(), GameError> {
        if capacity > EVENT_HISTORY_CAPACITY
            || history_seat_masks.len() != self.games.len()
            || self.games.len().checked_mul(capacity * EVENT_RECORD_WIDTH) != Some(output.len())
            || lengths.len() != self.games.len()
        {
            return if capacity > EVENT_HISTORY_CAPACITY {
                Err(GameError::EventCapacity)
            } else {
                Err(GameError::BatchLength)
            };
        }
        if capacity == 0 {
            lengths.fill(0);
            return Ok(());
        }
        let write = |game: &Game, history_seat_mask: u8, records: &mut [i32], length: &mut u16| {
            let viewer = game
                .decision()
                .map_or(game.dealer(), |decision| decision.actor);
            if history_seat_mask & seat_bit(viewer) == 0 {
                *length = 0;
                return;
            }
            *length = game
                .write_events(viewer, records)
                .expect("batch event buffers were validated") as u16;
        };
        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(history_seat_masks.par_iter().copied())
                .zip(output.par_chunks_mut(capacity * EVENT_RECORD_WIDTH))
                .zip(lengths.par_iter_mut())
                .for_each(|(((game, history_seat_mask), records), length)| {
                    write(game, history_seat_mask, records, length);
                });
        } else {
            for (((game, &history_seat_mask), records), length) in self
                .games
                .iter()
                .zip(history_seat_masks.iter())
                .zip(output.chunks_mut(capacity * EVENT_RECORD_WIDTH))
                .zip(lengths.iter_mut())
            {
                write(game, history_seat_mask, records, length);
            }
        }
        Ok(())
    }

    /// Writes only the events emitted by each environment's most recent step.
    pub fn step_events_into(
        &self,
        capacity: usize,
        output: &mut [i32],
        lengths: &mut [u16],
    ) -> Result<(), GameError> {
        if capacity > EVENT_HISTORY_CAPACITY
            || self.games.len().checked_mul(capacity * EVENT_RECORD_WIDTH) != Some(output.len())
            || lengths.len() != self.games.len()
        {
            return if capacity > EVENT_HISTORY_CAPACITY {
                Err(GameError::EventCapacity)
            } else {
                Err(GameError::BatchLength)
            };
        }
        if capacity == 0 {
            lengths.fill(0);
            return Ok(());
        }
        let write = |game: &Game, records: &mut [i32], length: &mut u16| {
            let viewer = game
                .decision()
                .map_or(game.dealer(), |decision| decision.actor);
            *length = game
                .step_events_into(viewer, records)
                .expect("batch event buffers were validated") as u16;
        };
        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(output.par_chunks_mut(capacity * EVENT_RECORD_WIDTH))
                .zip(lengths.par_iter_mut())
                .for_each(|((game, records), length)| write(game, records, length));
        } else {
            for ((game, records), length) in self
                .games
                .iter()
                .zip(output.chunks_mut(capacity * EVENT_RECORD_WIDTH))
                .zip(lengths.iter_mut())
            {
                write(game, records, length);
            }
        }
        Ok(())
    }

    pub fn step(
        &mut self,
        actions: &[Action],
        outcomes: &mut [StepOutcome],
    ) -> Result<(), GameError> {
        self.step_with(actions, outcomes, core::convert::identity)
    }

    pub fn step_ids(
        &mut self,
        actions: &[ActionId],
        outcomes: &mut [StepOutcome],
    ) -> Result<(), GameError> {
        self.step_with(actions, outcomes, ActionId::action)
    }

    /// Applies raw policy-head indices and writes one flat transition record
    /// per environment into caller-owned memory.
    ///
    /// Each record contains:
    /// `[draw_player, draw_tile, replacement, discard_player, discard_tile,
    /// score_delta_0..score_delta_3, next_actor, next_phase, terminal]`.
    /// Missing players, tiles, and next decisions use `-1` sentinels.
    pub fn step_indices_into(
        &mut self,
        actions: &[u8],
        records: &mut [i64],
    ) -> Result<(), GameError> {
        if actions.len() != self.games.len()
            || self.games.len().checked_mul(STEP_RECORD_WIDTH) != Some(records.len())
        {
            return Err(GameError::BatchLength);
        }

        let all_legal = if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(actions.par_iter().copied())
                .all(|(game, index)| {
                    ActionId::new(index as usize)
                        .is_some_and(|id| game.is_legal_action(id.action()))
                })
        } else {
            self.games
                .iter()
                .zip(actions.iter().copied())
                .all(|(game, index)| {
                    ActionId::new(index as usize)
                        .is_some_and(|id| game.is_legal_action(id.action()))
                })
        };
        if !all_legal {
            return Err(GameError::InvalidAction);
        }

        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter_mut()
                .zip(actions.par_iter().copied())
                .zip(records.par_chunks_mut(STEP_RECORD_WIDTH))
                .try_for_each(|((game, index), record)| -> Result<(), GameError> {
                    let action = ActionId::new(index as usize)
                        .expect("action index was validated")
                        .action();
                    let outcome = game.apply_legal_action(action)?;
                    outcome.write_record(record.try_into().expect("record width was validated"));
                    Ok(())
                })?;
        } else {
            for ((game, index), record) in self
                .games
                .iter_mut()
                .zip(actions.iter().copied())
                .zip(records.chunks_mut(STEP_RECORD_WIDTH))
            {
                let action = ActionId::new(index as usize)
                    .expect("action index was validated")
                    .action();
                let outcome = game.apply_legal_action(action)?;
                outcome.write_record(record.try_into().expect("record width was validated"));
            }
        }
        Ok(())
    }

    /// Applies raw policy-head indices only where the byte mask is one.
    ///
    /// Disabled rows are left unchanged and receive an all-`-1` record. The
    /// whole operation is atomic with respect to invalid masks or actions.
    pub fn step_masked_indices_into(
        &mut self,
        enabled: &[u8],
        actions: &[u8],
        records: &mut [i64],
    ) -> Result<(), GameError> {
        if enabled.len() != self.games.len()
            || actions.len() != self.games.len()
            || self.games.len().checked_mul(STEP_RECORD_WIDTH) != Some(records.len())
        {
            return Err(GameError::BatchLength);
        }
        if enabled.iter().any(|&value| value > 1) {
            return Err(GameError::InvalidAction);
        }
        let all_legal = if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(enabled.par_iter().copied())
                .zip(actions.par_iter().copied())
                .all(|((game, enabled), index)| {
                    enabled == 0
                        || ActionId::new(index as usize)
                            .is_some_and(|id| game.is_legal_action(id.action()))
                })
        } else {
            self.games
                .iter()
                .zip(enabled.iter().copied())
                .zip(actions.iter().copied())
                .all(|((game, enabled), index)| {
                    enabled == 0
                        || ActionId::new(index as usize)
                            .is_some_and(|id| game.is_legal_action(id.action()))
                })
        };
        if !all_legal {
            return Err(GameError::InvalidAction);
        }

        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter_mut()
                .zip(enabled.par_iter().copied())
                .zip(actions.par_iter().copied())
                .zip(records.par_chunks_mut(STEP_RECORD_WIDTH))
                .try_for_each(
                    |(((game, enabled), index), record)| -> Result<(), GameError> {
                        if enabled == 0 {
                            record.fill(-1);
                            return Ok(());
                        }
                        let action = ActionId::new(index as usize)
                            .expect("action index was validated")
                            .action();
                        let outcome = game.apply_legal_action(action)?;
                        outcome
                            .write_record(record.try_into().expect("record width was validated"));
                        Ok(())
                    },
                )?;
        } else {
            for (((game, enabled), index), record) in self
                .games
                .iter_mut()
                .zip(enabled.iter().copied())
                .zip(actions.iter().copied())
                .zip(records.chunks_mut(STEP_RECORD_WIDTH))
            {
                if enabled == 0 {
                    record.fill(-1);
                    continue;
                }
                let action = ActionId::new(index as usize)
                    .expect("action index was validated")
                    .action();
                let outcome = game.apply_legal_action(action)?;
                outcome.write_record(record.try_into().expect("record width was validated"));
            }
        }
        Ok(())
    }

    /// Applies actions and writes every next-state training buffer in one
    /// batch traversal. Full event history is emitted only when the next
    /// actor's bit is set in the corresponding `history_seat_masks` row.
    #[allow(clippy::too_many_arguments)]
    pub fn step_indices_observe_history_into(
        &mut self,
        actions: &[u8],
        history_seat_masks: &[u8],
        records: &mut [i64],
        mask_words: &mut [u64],
        tile_obs: &mut [u8],
        melds: &mut [u8],
        river: &mut [u8],
        meta: &mut [i32],
        event_capacity: usize,
        events: &mut [i32],
        event_lengths: &mut [u16],
    ) -> Result<(), GameError> {
        let size = self.games.len();
        if actions.len() != size
            || history_seat_masks.len() != size
            || size.checked_mul(STEP_RECORD_WIDTH) != Some(records.len())
            || size.checked_mul(LEGAL_ACTION_MASK_WORDS) != Some(mask_words.len())
            || size.checked_mul(TILE_OBSERVATION_WIDTH) != Some(tile_obs.len())
            || size.checked_mul(MELD_OBSERVATION_WIDTH) != Some(melds.len())
            || size.checked_mul(RIVER_OBSERVATION_WIDTH) != Some(river.len())
            || size.checked_mul(META_OBSERVATION_WIDTH) != Some(meta.len())
            || event_capacity == 0
            || event_capacity > EVENT_HISTORY_CAPACITY
            || size.checked_mul(event_capacity * EVENT_RECORD_WIDTH) != Some(events.len())
            || event_lengths.len() != size
        {
            return if event_capacity == 0 || event_capacity > EVENT_HISTORY_CAPACITY {
                Err(GameError::EventCapacity)
            } else {
                Err(GameError::BatchLength)
            };
        }

        let all_legal = if size >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(actions.par_iter().copied())
                .all(|(game, index)| {
                    ActionId::new(index as usize)
                        .is_some_and(|id| game.is_legal_action(id.action()))
                })
        } else {
            self.games
                .iter()
                .zip(actions.iter().copied())
                .all(|(game, index)| {
                    ActionId::new(index as usize)
                        .is_some_and(|id| game.is_legal_action(id.action()))
                })
        };
        if !all_legal {
            return Err(GameError::InvalidAction);
        }

        let write = |game: &mut Game,
                     index: u8,
                     history_seat_mask: u8,
                     record: &mut [i64],
                     words: &mut [u64],
                     tile_obs: &mut [u8],
                     melds: &mut [u8],
                     river: &mut [u8],
                     meta: &mut [i32],
                     events: &mut [i32],
                     event_length: &mut u16|
         -> Result<(), GameError> {
            let action = ActionId::new(index as usize)
                .expect("action index was validated")
                .action();
            let outcome = game.apply_legal_action(action)?;
            outcome.write_record(record.try_into().expect("record width was validated"));
            let viewer = game
                .decision()
                .map_or(game.dealer(), |decision| decision.actor);
            game.write_observation(viewer, tile_obs, melds, river, meta);
            if let Some(mask) = game.legal_action_mask() {
                words.copy_from_slice(mask.words());
            } else {
                words.fill(0);
            }
            let viewer = game
                .decision()
                .map_or(game.dealer(), |decision| decision.actor);
            if history_seat_mask & seat_bit(viewer) != 0 {
                *event_length =
                    game.write_events(viewer, events)
                        .expect("batch event buffers were validated") as u16;
            } else {
                *event_length = 0;
            }
            Ok(())
        };

        if size >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter_mut()
                .zip(actions.par_iter().copied())
                .zip(history_seat_masks.par_iter().copied())
                .zip(records.par_chunks_mut(STEP_RECORD_WIDTH))
                .zip(mask_words.par_chunks_mut(LEGAL_ACTION_MASK_WORDS))
                .zip(tile_obs.par_chunks_mut(TILE_OBSERVATION_WIDTH))
                .zip(melds.par_chunks_mut(MELD_OBSERVATION_WIDTH))
                .zip(river.par_chunks_mut(RIVER_OBSERVATION_WIDTH))
                .zip(meta.par_chunks_mut(META_OBSERVATION_WIDTH))
                .zip(events.par_chunks_mut(event_capacity * EVENT_RECORD_WIDTH))
                .zip(event_lengths.par_iter_mut())
                .try_for_each(
                    |(
                        (
                            (
                                (
                                    (
                                        (
                                            ((((game, index), history_seat_mask), record), words),
                                            tile_obs,
                                        ),
                                        melds,
                                    ),
                                    river,
                                ),
                                meta,
                            ),
                            events,
                        ),
                        event_length,
                    )| {
                        write(
                            game,
                            index,
                            history_seat_mask,
                            record,
                            words,
                            tile_obs,
                            melds,
                            river,
                            meta,
                            events,
                            event_length,
                        )
                    },
                )?;
        } else {
            for (
                (
                    (
                        (
                            (
                                (((((game, &index), &history_seat_mask), record), words), tile_obs),
                                melds,
                            ),
                            river,
                        ),
                        meta,
                    ),
                    events,
                ),
                event_length,
            ) in self
                .games
                .iter_mut()
                .zip(actions.iter())
                .zip(history_seat_masks.iter())
                .zip(records.chunks_mut(STEP_RECORD_WIDTH))
                .zip(mask_words.chunks_mut(LEGAL_ACTION_MASK_WORDS))
                .zip(tile_obs.chunks_mut(TILE_OBSERVATION_WIDTH))
                .zip(melds.chunks_mut(MELD_OBSERVATION_WIDTH))
                .zip(river.chunks_mut(RIVER_OBSERVATION_WIDTH))
                .zip(meta.chunks_mut(META_OBSERVATION_WIDTH))
                .zip(events.chunks_mut(event_capacity * EVENT_RECORD_WIDTH))
                .zip(event_lengths.iter_mut())
            {
                write(
                    game,
                    index,
                    history_seat_mask,
                    record,
                    words,
                    tile_obs,
                    melds,
                    river,
                    meta,
                    events,
                    event_length,
                )?;
            }
        }
        Ok(())
    }

    /// Resets selected environments and refreshes only their caller-owned
    /// observation rows. Non-selected rows and their output buffers are left
    /// untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn reset_observe_history_into(
        &mut self,
        reset_flags: &[u8],
        seeds: &[u64],
        history_seat_masks: &[u8],
        mask_words: &mut [u64],
        tile_obs: &mut [u8],
        melds: &mut [u8],
        river: &mut [u8],
        meta: &mut [i32],
        event_capacity: usize,
        events: &mut [i32],
        event_lengths: &mut [u16],
    ) -> Result<(), GameError> {
        let size = self.games.len();
        if reset_flags.len() != size
            || seeds.len() != size
            || history_seat_masks.len() != size
            || size.checked_mul(LEGAL_ACTION_MASK_WORDS) != Some(mask_words.len())
            || size.checked_mul(TILE_OBSERVATION_WIDTH) != Some(tile_obs.len())
            || size.checked_mul(MELD_OBSERVATION_WIDTH) != Some(melds.len())
            || size.checked_mul(RIVER_OBSERVATION_WIDTH) != Some(river.len())
            || size.checked_mul(META_OBSERVATION_WIDTH) != Some(meta.len())
            || event_capacity == 0
            || event_capacity > EVENT_HISTORY_CAPACITY
            || size.checked_mul(event_capacity * EVENT_RECORD_WIDTH) != Some(events.len())
            || event_lengths.len() != size
        {
            return if event_capacity == 0 || event_capacity > EVENT_HISTORY_CAPACITY {
                Err(GameError::EventCapacity)
            } else {
                Err(GameError::BatchLength)
            };
        }

        let write = |game: &mut Game,
                     reset: u8,
                     seed: u64,
                     history_seat_mask: u8,
                     words: &mut [u64],
                     tile_obs: &mut [u8],
                     melds: &mut [u8],
                     river: &mut [u8],
                     meta: &mut [i32],
                     events: &mut [i32],
                     event_length: &mut u16| {
            if reset == 0 {
                return;
            }
            game.reset(seed);
            let viewer = game
                .decision()
                .map_or(game.dealer(), |decision| decision.actor);
            game.write_observation(viewer, tile_obs, melds, river, meta);
            if let Some(mask) = game.legal_action_mask() {
                words.copy_from_slice(mask.words());
            } else {
                words.fill(0);
            }
            let viewer = game
                .decision()
                .map_or(game.dealer(), |decision| decision.actor);
            if history_seat_mask & seat_bit(viewer) != 0 {
                *event_length =
                    game.write_events(viewer, events)
                        .expect("batch event buffers were validated") as u16;
            } else {
                *event_length = 0;
            }
        };

        if size >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter_mut()
                .zip(reset_flags.par_iter().copied())
                .zip(seeds.par_iter().copied())
                .zip(history_seat_masks.par_iter().copied())
                .zip(mask_words.par_chunks_mut(LEGAL_ACTION_MASK_WORDS))
                .zip(tile_obs.par_chunks_mut(TILE_OBSERVATION_WIDTH))
                .zip(melds.par_chunks_mut(MELD_OBSERVATION_WIDTH))
                .zip(river.par_chunks_mut(RIVER_OBSERVATION_WIDTH))
                .zip(meta.par_chunks_mut(META_OBSERVATION_WIDTH))
                .zip(events.par_chunks_mut(event_capacity * EVENT_RECORD_WIDTH))
                .zip(event_lengths.par_iter_mut())
                .for_each(
                    |(
                        (
                            (
                                (
                                    (
                                        (
                                            ((((game, reset), seed), history_seat_mask), words),
                                            tile_obs,
                                        ),
                                        melds,
                                    ),
                                    river,
                                ),
                                meta,
                            ),
                            events,
                        ),
                        event_length,
                    )| {
                        write(
                            game,
                            reset,
                            seed,
                            history_seat_mask,
                            words,
                            tile_obs,
                            melds,
                            river,
                            meta,
                            events,
                            event_length,
                        );
                    },
                );
        } else {
            for (
                (
                    (
                        (
                            (
                                (((((game, &reset), &seed), &history_seat_mask), words), tile_obs),
                                melds,
                            ),
                            river,
                        ),
                        meta,
                    ),
                    events,
                ),
                event_length,
            ) in self
                .games
                .iter_mut()
                .zip(reset_flags.iter())
                .zip(seeds.iter())
                .zip(history_seat_masks.iter())
                .zip(mask_words.chunks_mut(LEGAL_ACTION_MASK_WORDS))
                .zip(tile_obs.chunks_mut(TILE_OBSERVATION_WIDTH))
                .zip(melds.chunks_mut(MELD_OBSERVATION_WIDTH))
                .zip(river.chunks_mut(RIVER_OBSERVATION_WIDTH))
                .zip(meta.chunks_mut(META_OBSERVATION_WIDTH))
                .zip(events.chunks_mut(event_capacity * EVENT_RECORD_WIDTH))
                .zip(event_lengths.iter_mut())
            {
                write(
                    game,
                    reset,
                    seed,
                    history_seat_mask,
                    words,
                    tile_obs,
                    melds,
                    river,
                    meta,
                    events,
                    event_length,
                );
            }
        }
        Ok(())
    }

    fn step_with<A, F>(
        &mut self,
        actions: &[A],
        outcomes: &mut [StepOutcome],
        decode: F,
    ) -> Result<(), GameError>
    where
        A: Copy + Send + Sync,
        F: Copy + Fn(A) -> Action + Sync,
    {
        if actions.len() != self.games.len() || outcomes.len() != self.games.len() {
            return Err(GameError::BatchLength);
        }
        let all_legal = if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter()
                .zip(actions.par_iter().copied())
                .all(|(game, action)| game.is_legal_action(decode(action)))
        } else {
            self.games
                .iter()
                .zip(actions.iter().copied())
                .all(|(game, action)| game.is_legal_action(decode(action)))
        };
        if !all_legal {
            return Err(GameError::InvalidAction);
        }
        if self.games.len() >= PARALLEL_BATCH_THRESHOLD {
            self.games
                .par_iter_mut()
                .zip(actions.par_iter().copied())
                .zip(outcomes.par_iter_mut())
                .try_for_each(|((game, action), outcome)| -> Result<(), GameError> {
                    *outcome = game.apply_legal_action(decode(action))?;
                    Ok(())
                })?;
        } else {
            for ((game, action), outcome) in self
                .games
                .iter_mut()
                .zip(actions.iter().copied())
                .zip(outcomes.iter_mut())
            {
                *outcome = game.apply_legal_action(decode(action))?;
            }
        }
        Ok(())
    }
}

fn batch_seed(seed: u64, index: usize) -> u64 {
    seed.wrapping_add((index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

fn seat_bit(seat: Seat) -> u8 {
    1 << seat.index()
}

fn relative_seat(viewer: Seat, seat: Seat) -> u8 {
    ((seat.index() + PLAYER_COUNT - viewer.index()) % PLAYER_COUNT) as u8
}

fn add_tile_counts(target: &mut [u16; TILE_KIND_COUNT], source: &[u8; TILE_KIND_COUNT]) {
    for (target, source) in target.iter_mut().zip(source.iter().copied()) {
        *target += u16::from(source);
    }
}

fn first_seat_in_mask(source: Seat, mask: u8) -> Option<Seat> {
    (1..=PLAYER_COUNT as u8)
        .map(|offset| source.offset(offset))
        .find(|&seat| mask & seat_bit(seat) != 0)
}

fn seats_after(source: Seat) -> [Seat; 3] {
    [source.offset(1), source.offset(2), source.offset(3)]
}

fn seats_in_mask_after(source: Seat, mask: u8) -> impl Iterator<Item = Seat> {
    (1..=PLAYER_COUNT as u8)
        .map(move |offset| source.offset(offset))
        .filter(move |&seat| mask & seat_bit(seat) != 0)
}

fn win_event_flags(flags: WinFlags, self_draw: bool) -> u8 {
    let mut output = 0;
    if self_draw {
        output |= EVENT_FLAG_SELF_DRAW;
    }
    if flags.rob_kong {
        output |= EVENT_FLAG_ROB_KONG;
    }
    if flags.after_kong_discard || flags.after_kong_draw {
        output |= EVENT_FLAG_AFTER_KONG;
    }
    if flags.last_wall_tile {
        output |= EVENT_FLAG_LAST_WALL_TILE;
    }
    if flags.heavenly {
        output |= EVENT_FLAG_HEAVENLY;
    }
    if flags.earthly {
        output |= EVENT_FLAG_EARTHLY;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_three_same_suit(game: &Game, seat: Seat) -> [Tile; 3] {
        for suit in Suit::ALL {
            let mut tiles = [Tile::from_index_unchecked(0); 3];
            let mut len = 0;
            for index in suit as usize * 9..suit as usize * 9 + 9 {
                for _ in 0..game.concealed(seat)[index] {
                    if len < 3 {
                        tiles[len] = Tile::from_index_unchecked(index as u8);
                        len += 1;
                    }
                }
            }
            if len == 3 {
                return tiles;
            }
        }
        panic!("every initial hand has three tiles in some suit")
    }

    fn complete_setup(game: &mut Game) {
        for seat in Seat::ALL {
            let exchange = first_three_same_suit(game, seat);
            for tile in exchange {
                game.step(Action::SelectExchangeTile(tile)).unwrap();
            }
        }
        for _ in Seat::ALL {
            game.step(Action::ChooseMissing(Suit::Dots)).unwrap();
        }
    }

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank - 1).unwrap()
    }

    fn add_tile(player: &mut Player, tile: Tile, count: u8) {
        player.concealed[tile.index()] += count;
    }

    fn add_sequence(player: &mut Player, suit: Suit, first_rank: u8) {
        for rank in first_rank..first_rank + 3 {
            add_tile(player, tile(suit, rank), 1);
        }
    }

    fn make_plain_wait(player: &mut Player) -> Tile {
        add_sequence(player, Suit::Characters, 1);
        add_sequence(player, Suit::Characters, 4);
        add_sequence(player, Suit::Bamboo, 1);
        add_sequence(player, Suit::Bamboo, 4);
        let winning_tile = tile(Suit::Bamboo, 9);
        add_tile(player, winning_tile, 1);
        player.missing = Some(Suit::Dots);
        winning_tile
    }

    fn make_zero_copy_wait(player: &mut Player) -> Tile {
        add_sequence(player, Suit::Characters, 1);
        add_sequence(player, Suit::Characters, 4);
        add_sequence(player, Suit::Bamboo, 1);
        let pair = tile(Suit::Bamboo, 5);
        add_tile(player, pair, 2);
        add_tile(player, tile(Suit::Characters, 7), 1);
        add_tile(player, tile(Suit::Characters, 8), 1);
        player.missing = Some(Suit::Dots);
        tile(Suit::Characters, 9)
    }

    fn make_quad_seven_pairs_wait(player: &mut Player) -> Tile {
        let winning_tile = tile(Suit::Characters, 1);
        add_tile(player, winning_tile, 3);
        for rank in 2..=6 {
            add_tile(player, tile(Suit::Characters, rank), 2);
        }
        player.missing = Some(Suit::Dots);
        winning_tile
    }

    fn make_flower_pig(player: &mut Player) {
        add_tile(player, tile(Suit::Characters, 1), 1);
        add_tile(player, tile(Suit::Bamboo, 1), 1);
        add_tile(player, tile(Suit::Dots, 1), 1);
    }

    fn constructed_game() -> Game {
        let mut game = Game::new(0);
        game.players = [Player::new(); PLAYER_COUNT];
        game.wall = [0; WALL_TILE_COUNT];
        game.wall_head = 0;
        game.wall_tail = 0;
        game.stage = Stage::Finished;
        game.discard_len = 0;
        game.transition_draw = None;
        game.transition_discard = None;
        game
    }

    fn set_wall(game: &mut Game, tiles: &[Tile]) {
        game.wall_head = 0;
        game.wall_tail = tiles.len() as u8;
        game.final_wall_tile_drawn = false;
        for (slot, tile) in game.wall.iter_mut().zip(tiles) {
            *slot = tile.as_u8();
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
        .unwrap();
        observation
    }

    fn assert_stable_base_supports_public_wins(game: &Game, seat: Seat) {
        let player = &game.players[seat.index()];
        let base = *game.win_base(seat);
        let public_wins = game.public_win_tiles(seat);
        let (melds, meld_count) = player.meld_buffer();
        assert_eq!(
            base.iter().map(|&count| usize::from(count)).sum::<usize>(),
            player.win_base_target_len()
        );
        if let Some(missing) = player.missing {
            let start = missing as usize * 9;
            assert!(base[start..start + 9].iter().all(|&count| count == 0));
            assert!(
                melds[..meld_count]
                    .iter()
                    .all(|meld| meld.tile.suit() != missing)
            );
        }

        for (index, &win_count) in public_wins.iter().enumerate() {
            for _ in 0..win_count {
                let mut completed = base;
                completed[index] += 1;
                let winning_tile = Tile::from_index_unchecked(index as u8);
                assert!(
                    is_bloodflow_winning(
                        &completed,
                        &base,
                        &melds[..meld_count],
                        Some(winning_tile),
                    ),
                    "seat {seat:?} has an invalid sampled base for tile {winning_tile:?}"
                );
            }
        }
    }

    #[test]
    fn deal_is_deterministic_and_conserves_tiles() {
        let left = Game::new(42);
        let right = Game::new(42);
        assert_eq!(left.wall, right.wall);
        assert_eq!(left.wall_remaining(), 55);
        let held: usize = Seat::ALL
            .iter()
            .map(|&seat| {
                left.concealed(seat)
                    .iter()
                    .map(|&n| n as usize)
                    .sum::<usize>()
            })
            .sum();
        assert_eq!(held, 53);
    }

    #[test]
    fn exchange_is_three_sequential_same_suit_decisions() {
        let mut game = Game::new(7);
        let selected = first_three_same_suit(&game, Seat::EAST);

        game.step(Action::SelectExchangeTile(selected[0])).unwrap();
        assert_eq!(game.decision().unwrap().actor, Seat::EAST);
        assert_eq!(game.exchange_selection(Seat::EAST)[selected[0].index()], 1);
        let legal = game.legal_actions().unwrap();
        assert_ne!(legal.exchange_mask, 0);
        assert!(legal.exchange_mask & !selected[0].suit().mask() == 0);

        game.step(Action::SelectExchangeTile(selected[1])).unwrap();
        assert_eq!(game.decision().unwrap().actor, Seat::EAST);
        game.step(Action::SelectExchangeTile(selected[2])).unwrap();
        assert_eq!(game.decision().unwrap().actor, Seat::ALL[1]);
    }

    #[test]
    fn setup_reaches_dealer_turn() {
        let mut game = Game::new(9);
        complete_setup(&mut game);
        assert_eq!(game.phase(), Phase::Turn);
        assert_eq!(game.decision().unwrap().actor, Seat::EAST);
        assert!(game.legal_actions().unwrap().discard_mask != 0);
    }

    #[test]
    fn missing_suits_are_published_together() {
        let mut game = Game::new(11);
        for seat in Seat::ALL {
            let exchange = first_three_same_suit(&game, seat);
            for tile in exchange {
                game.step(Action::SelectExchangeTile(tile)).unwrap();
            }
        }
        game.step(Action::ChooseMissing(Suit::Characters)).unwrap();
        assert!(
            Seat::ALL
                .iter()
                .all(|&seat| game.missing_suit(seat).is_none())
        );
        game.step(Action::ChooseMissing(Suit::Bamboo)).unwrap();
        game.step(Action::ChooseMissing(Suit::Dots)).unwrap();
        assert!(
            Seat::ALL
                .iter()
                .all(|&seat| game.missing_suit(seat).is_none())
        );
        game.step(Action::ChooseMissing(Suit::Characters)).unwrap();
        assert_eq!(game.missing_suit(Seat::ALL[0]), Some(Suit::Characters));
        assert_eq!(game.missing_suit(Seat::ALL[1]), Some(Suit::Bamboo));
        assert_eq!(game.missing_suit(Seat::ALL[2]), Some(Suit::Dots));
        assert_eq!(game.missing_suit(Seat::ALL[3]), Some(Suit::Characters));
    }

    #[test]
    fn missing_suit_does_not_restrict_discard_choices() {
        let mut game = constructed_game();
        let actor = Seat::EAST;
        let missing_tile = tile(Suit::Characters, 1);
        let other_tile = tile(Suit::Bamboo, 1);
        add_tile(&mut game.players[actor.index()], missing_tile, 1);
        add_tile(&mut game.players[actor.index()], other_tile, 1);
        game.players[actor.index()].missing = Some(Suit::Characters);
        set_wall(&mut game, &[tile(Suit::Dots, 9)]);
        game.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        let legal = game.legal_actions().expect("turn has legal actions");
        assert_ne!(legal.discard_mask & (1 << missing_tile.index()), 0);
        assert_ne!(legal.discard_mask & (1 << other_tile.index()), 0);
        assert!(game.is_legal_action(Action::Discard(other_tile)));

        game.step(Action::Discard(other_tile))
            .expect("a non-missing tile remains a legal discard");
        assert_eq!(game.concealed(actor)[other_tile.index()], 0);
        assert_eq!(game.concealed(actor)[missing_tile.index()], 1);
    }

    #[test]
    fn pass_is_never_legal_on_a_normal_turn() {
        let mut game = Game::new(12);
        complete_setup(&mut game);
        let decision = game.decision();
        let wall = game.wall_remaining();
        let hand = *game.concealed(Seat::EAST);
        assert!(!game.is_legal_action(Action::Pass));
        assert_eq!(game.step(Action::Pass), Err(GameError::InvalidAction));
        assert_eq!(game.decision(), decision);
        assert_eq!(game.wall_remaining(), wall);
        assert_eq!(*game.concealed(Seat::EAST), hand);
    }

    #[test]
    fn pong_is_rejected_when_it_would_leave_no_discard() {
        let mut game = constructed_game();
        let actor = Seat::ALL[1];
        let called = tile(Suit::Characters, 1);
        let next_draw = tile(Suit::Bamboo, 1);
        game.players[actor.index()].concealed[called.index()] = 2;
        game.wall[0] = next_draw.as_u8();
        game.wall_head = 0;
        game.wall_tail = 1;
        game.stage = Stage::MeldResponse {
            source: Seat::EAST,
            tile: called,
            remaining: seat_bit(actor),
            pending_melds: PendingMelds::NONE,
        };

        let legal = game.legal_actions().expect("meld response has a decision");
        assert!(!legal.can_pong);
        assert!(legal.can_pass);
        let outcome = game.step(Action::Pass).unwrap();
        assert_eq!(outcome.draw.unwrap().tile, next_draw);
        assert_eq!(game.phase(), Phase::Turn);
    }

    #[test]
    fn impossible_pong_does_not_create_a_response_window() {
        let mut game = constructed_game();
        let actor = Seat::ALL[1];
        let called = tile(Suit::Characters, 1);
        let next_draw = tile(Suit::Bamboo, 1);
        game.players[actor.index()].concealed[called.index()] = 2;
        set_wall(&mut game, &[next_draw]);

        game.begin_meld_responses(Seat::EAST, called);

        assert_eq!(
            game.decision(),
            Some(Decision {
                actor: Seat::ALL[1],
                phase: Phase::Turn,
            })
        );
        assert_eq!(game.current_draw().unwrap().tile, next_draw);
    }

    #[test]
    fn bankrupt_player_cannot_respond_and_is_skipped_for_the_next_draw() {
        let mut game = constructed_game();
        let source = Seat::EAST;
        let bankrupt = Seat::ALL[1];
        let next = Seat::ALL[2];
        let called = tile(Suit::Characters, 1);
        let next_draw = tile(Suit::Bamboo, 1);
        add_tile(&mut game.players[source.index()], called, 1);
        add_tile(&mut game.players[bankrupt.index()], called, 3);
        game.players[bankrupt.index()].score = 0;
        set_wall(&mut game, &[next_draw]);
        game.stage = Stage::Turn {
            actor: source,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        let outcome = game.step(Action::Discard(called)).unwrap();

        assert_eq!(game.players[bankrupt.index()].concealed[called.index()], 3);
        assert_eq!(outcome.draw.map(|draw| draw.player), Some(next));
        assert_eq!(game.decision().map(|decision| decision.actor), Some(next));
    }

    #[test]
    fn locked_triplet_cannot_form_an_exposed_kong() {
        let mut game = constructed_game();
        let actor = Seat::ALL[1];
        let called = tile(Suit::Characters, 1);
        let next_draw = tile(Suit::Bamboo, 1);
        let player = &mut game.players[actor.index()];
        player.concealed[called.index()] = 3;
        player.locked[called.index()] = 3;
        player.has_won = true;
        set_wall(&mut game, &[next_draw]);

        let mut response = game.clone();
        response.stage = Stage::MeldResponse {
            source: Seat::EAST,
            tile: called,
            remaining: seat_bit(actor),
            pending_melds: PendingMelds::NONE,
        };
        let before = response.clone();
        let legal = response.legal_actions().expect("response has a decision");
        assert!(!legal.can_pong);
        assert!(!legal.can_exposed_kong);
        assert!(legal.can_pass);
        assert_eq!(response.simple_rule_action(), Some(ActionId::PASS));
        assert_eq!(
            response.step(Action::ExposedKong),
            Err(GameError::InvalidAction)
        );
        assert_eq!(response, before);

        game.begin_meld_responses(Seat::EAST, called);
        assert_eq!(
            game.decision(),
            Some(Decision {
                actor,
                phase: Phase::Turn,
            })
        );
        assert_eq!(game.current_draw().unwrap().tile, next_draw);
    }

    #[test]
    fn concealed_kong_after_hu_uses_the_stable_base_but_not_historical_wins() {
        let mut game = constructed_game();
        let actor = Seat::ALL[1];
        let historical_quad = tile(Suit::Characters, 1);
        let base_and_draw_quad = tile(Suit::Characters, 2);
        let replacement = tile(Suit::Bamboo, 1);

        let player = &mut game.players[actor.index()];
        player.missing = None;
        player.concealed[historical_quad.index()] = 4;
        player.locked[historical_quad.index()] = 4;
        player.concealed[base_and_draw_quad.index()] = 4;
        player.locked[base_and_draw_quad.index()] = 3;
        player.win_base[base_and_draw_quad.index()] = 3;
        player.has_won = true;
        set_wall(&mut game, &[replacement]);
        game.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::Draw {
                tile: base_and_draw_quad,
                after_kong: false,
                last_wall_tile: false,
                deferred_kong_forbidden_tile: None,
            },
            can_hu: false,
        };

        let legal = game.legal_actions().unwrap();
        assert_eq!(legal.concealed_kong_mask, 1 << base_and_draw_quad.index());
        assert_eq!(
            game.step(Action::ConcealedKong(historical_quad)),
            Err(GameError::InvalidAction)
        );
        let outcome = game
            .step(Action::ConcealedKong(base_and_draw_quad))
            .unwrap();
        assert_eq!(
            game.meld(actor, 0).map(|meld| (meld.tile, meld.kind)),
            Some((base_and_draw_quad, MeldKind::ConcealedKong))
        );
        assert_eq!(game.locked(actor)[base_and_draw_quad.index()], 0);
        assert_eq!(game.public_win_tiles(actor)[historical_quad.index()], 4);
        assert_eq!(
            outcome.draw,
            Some(DrawEvent {
                player: actor,
                tile: replacement,
                replacement: true,
            })
        );
        assert!(!outcome.terminal);
        assert_ne!(
            game.legal_actions().unwrap().discard_mask & (1 << replacement.index()),
            0
        );
    }

    #[test]
    fn draw_face_is_filtered_for_other_players() {
        let outcome = StepOutcome {
            draw: Some(DrawEvent {
                player: Seat::ALL[1],
                tile: Tile::from_index_unchecked(7),
                replacement: false,
            }),
            ..StepOutcome::default()
        };
        assert_eq!(
            outcome.for_player(Seat::ALL[1]).draw.unwrap().tile,
            Some(Tile::from_index_unchecked(7))
        );
        assert_eq!(outcome.for_player(Seat::EAST).draw.unwrap().tile, None);
    }

    #[test]
    fn batch_rejects_invalid_actions_atomically_and_can_reset_one_game() {
        let mut batch = Batch::new(2, 31);
        let first_action = exchange_action_for_test(&batch.games()[0], Seat::EAST);
        let decisions = [batch.games()[0].decision(), batch.games()[1].decision()];
        let hands = [
            *batch.games()[0].concealed(Seat::EAST),
            *batch.games()[1].concealed(Seat::EAST),
        ];
        let mut outcomes = [StepOutcome::default(); 2];
        assert_eq!(
            batch.step(&[first_action, Action::Pass], &mut outcomes),
            Err(GameError::InvalidAction)
        );
        for index in 0..2 {
            assert_eq!(batch.games()[index].decision(), decisions[index]);
            assert_eq!(*batch.games()[index].concealed(Seat::EAST), hands[index]);
        }

        batch.reset_at(1, 99).unwrap();
        assert_eq!(batch.games()[1].phase(), Phase::Exchange);
        assert_eq!(batch.reset_at(2, 99), Err(GameError::BatchIndex));
    }

    #[test]
    fn heavenly_hu_uses_the_post_exchange_hand_without_a_required_tile() {
        let mut game = constructed_game();
        let winning_tile = make_plain_wait(&mut game.players[Seat::EAST.index()]);
        add_tile(&mut game.players[Seat::EAST.index()], winning_tile, 1);
        let next_draw = tile(Suit::Characters, 9);
        set_wall(&mut game, &[next_draw]);
        game.stage = Stage::Turn {
            actor: Seat::EAST,
            origin: TurnOrigin::Initial,
            can_hu: true,
        };

        let outcome = game.step(Action::Hu).unwrap();

        assert_eq!(outcome.score_delta, [9_600, -3_200, -3_200, -3_200]);
        assert_eq!(game.score(Seat::EAST), 19_600);
        assert!(game.has_won(Seat::EAST));
        assert_eq!(game.locked(Seat::EAST).iter().sum::<u8>(), 14);
        assert_eq!(game.win_base(Seat::EAST).iter().sum::<u8>(), 13);
        assert_eq!(game.players[Seat::EAST.index()].unlocked_len(), 0);
        let mut events = [0_i32; EVENT_HISTORY_CAPACITY * EVENT_RECORD_WIDTH];
        let event_count = game
            .step_events_into(Seat::EAST, &mut events)
            .expect("event buffer is valid");
        let hu = events[..event_count * EVENT_RECORD_WIDTH]
            .chunks_exact(EVENT_RECORD_WIDTH)
            .find(|event| event[0] == i32::from(EventKind::Hu.code()))
            .expect("the step includes the heavenly win");
        let reference_tile = usize::try_from(hu[3]).expect("the win exposes a tile reference");
        assert_eq!(game.public_win_tiles(Seat::EAST)[reference_tile], 1);
        assert_eq!(
            outcome.draw,
            Some(DrawEvent {
                player: Seat::ALL[1],
                tile: next_draw,
                replacement: false,
            })
        );
        assert_eq!(
            outcome.next,
            Some(Decision {
                actor: Seat::ALL[1],
                phase: Phase::Turn,
            })
        );
    }

    #[test]
    fn dealers_opening_discard_does_not_create_earthly_hu() {
        let mut game = constructed_game();
        let winner = Seat::ALL[1];
        let winning_tile = make_plain_wait(&mut game.players[winner.index()]);
        add_tile(&mut game.players[Seat::EAST.index()], winning_tile, 1);
        let next_draw = tile(Suit::Characters, 9);
        set_wall(&mut game, &[next_draw]);
        game.stage = Stage::Turn {
            actor: Seat::EAST,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        let discard = game.step(Action::Discard(winning_tile)).unwrap();
        assert_eq!(
            discard.discard,
            Some(DiscardEvent {
                player: Seat::EAST,
                tile: winning_tile,
            })
        );
        assert_eq!(game.decision().unwrap().actor, winner);

        let hu = game.step(Action::Hu).unwrap();
        assert_eq!(hu.score_delta, [-100, 100, 0, 0]);
        assert_eq!(game.locked(winner).iter().sum::<u8>(), 14);
        assert_eq!(game.win_base(winner).iter().sum::<u8>(), 13);
        assert_eq!(game.players[winner.index()].unlocked_len(), 0);
        assert_eq!(
            hu.next,
            Some(Decision {
                actor: Seat::ALL[2],
                phase: Phase::Turn,
            })
        );
        assert_eq!(hu.draw.unwrap().tile, next_draw);
    }

    #[test]
    fn a_non_dealers_first_self_draw_can_be_earthly_hu() {
        let mut game = constructed_game();
        let winner = Seat::ALL[1];
        let winning_tile = make_plain_wait(&mut game.players[winner.index()]);
        let next_draw = tile(Suit::Characters, 9);
        set_wall(&mut game, &[winning_tile, next_draw]);

        game.begin_normal_turn(winner);
        let hu = game.step(Action::Hu).unwrap();

        assert_eq!(hu.score_delta, [-3_200, 9_600, -3_200, -3_200]);
        let summary = game.players[winner.index()]
            .last_win
            .expect("the first self draw completed a win");
        assert_eq!(summary.multiplier, 32);
        assert_ne!(summary.flags & EVENT_FLAG_EARTHLY, 0);
    }

    #[test]
    fn self_draw_does_not_double_a_higher_shape_multiplier() {
        let mut game = constructed_game();
        let winner = Seat::ALL[1];
        let winning_tile = make_quad_seven_pairs_wait(&mut game.players[winner.index()]);
        let next_draw = tile(Suit::Dots, 9);
        game.players[winner.index()].has_discarded = true;
        set_wall(&mut game, &[winning_tile, next_draw]);

        game.begin_normal_turn(winner);
        let hu = game.step(Action::Hu).unwrap();

        assert_eq!(hu.score_delta, [-3_200, 9_600, -3_200, -3_200]);
        let summary = game.players[winner.index()]
            .last_win
            .expect("the self draw completed a win");
        assert_eq!(summary.shape_multiplier, 32);
        assert_eq!(summary.multiplier, 32);
    }

    #[test]
    fn a_pong_can_be_followed_by_other_self_declared_kongs() {
        let mut game = constructed_game();
        let source = Seat::EAST;
        let actor = Seat::ALL[1];
        let called = tile(Suit::Characters, 5);
        let existing_pong = tile(Suit::Characters, 7);
        let concealed_quad = tile(Suit::Bamboo, 5);
        let replacement = tile(Suit::Bamboo, 9);
        add_tile(&mut game.players[source.index()], called, 1);
        add_tile(&mut game.players[actor.index()], called, 3);
        add_tile(&mut game.players[actor.index()], existing_pong, 1);
        add_tile(&mut game.players[actor.index()], concealed_quad, 4);
        assert!(game.players[actor.index()].add_meld(Meld {
            tile: existing_pong,
            kind: MeldKind::Pong,
            source: Seat::ALL[3],
        }));
        game.players[actor.index()].missing = Some(Suit::Dots);
        game.discard_len = 1;
        set_wall(&mut game, &[replacement]);
        game.stage = Stage::Turn {
            actor: source,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::Discard(called)).unwrap();
        assert_ne!(
            game.players[actor.index()].kong_forbidden_mask & (1 << called.index()),
            0,
            "offering an exposed Kong permanently forbids self-declared Kongs of that tile"
        );
        game.step(Action::Pong).unwrap();

        let legal = game
            .legal_actions()
            .expect("the Pong winner must discard or Kong");
        assert_eq!(legal.added_kong_mask & (1 << called.index()), 0);
        assert_ne!(legal.added_kong_mask & (1 << existing_pong.index()), 0);
        assert_ne!(legal.concealed_kong_mask & (1 << concealed_quad.index()), 0);

        let mut added = game.clone();
        let added_outcome = added.step(Action::AddedKong(existing_pong)).unwrap();
        assert_eq!(
            added_outcome.draw,
            Some(DrawEvent {
                player: actor,
                tile: replacement,
                replacement: true,
            })
        );

        let concealed_outcome = game.step(Action::ConcealedKong(concealed_quad)).unwrap();
        assert_eq!(
            concealed_outcome.draw,
            Some(DrawEvent {
                player: actor,
                tile: replacement,
                replacement: true,
            })
        );
    }

    #[test]
    fn a_dealer_kong_closes_earthly_hu_before_the_first_discard() {
        let mut game = constructed_game();
        let dealer = Seat::EAST;
        let winner = Seat::ALL[1];
        let kong_tile = tile(Suit::Characters, 9);
        let winning_tile = make_plain_wait(&mut game.players[winner.index()]);
        let next_draw = tile(Suit::Dots, 9);
        let replacement = tile(Suit::Characters, 8);
        add_tile(&mut game.players[dealer.index()], kong_tile, 4);
        add_tile(&mut game.players[dealer.index()], winning_tile, 1);
        game.players[dealer.index()].missing = Some(Suit::Dots);
        set_wall(&mut game, &[next_draw, replacement]);
        game.stage = Stage::Turn {
            actor: dealer,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::ConcealedKong(kong_tile)).unwrap();
        assert!(!game.earthly_hu_open);
        game.step(Action::Discard(winning_tile)).unwrap();

        assert_eq!(
            game.discard_flags[0] & EVENT_FLAG_OPENING_DISCARD,
            0,
            "a Kong ends the dealer's uninterrupted opening turn"
        );
        let hu = game.step(Action::Hu).unwrap();
        assert_eq!(hu.score_delta, [-200, 200, 0, 0]);
        let flags = game.players[winner.index()]
            .last_win
            .expect("the discard completed a win")
            .flags;
        assert_ne!(flags & EVENT_FLAG_AFTER_KONG, 0);
        assert_eq!(flags & EVENT_FLAG_EARTHLY, 0);
    }

    #[test]
    fn a_dealer_supplement_draw_before_the_first_discard_can_be_heavenly_hu() {
        let mut game = constructed_game();
        let dealer = Seat::EAST;
        let winning_tile = make_plain_wait(&mut game.players[dealer.index()]);
        let kong_tile = tile(Suit::Characters, 9);
        let next_draw = tile(Suit::Dots, 9);
        add_tile(&mut game.players[dealer.index()], kong_tile, 4);
        set_wall(&mut game, &[next_draw, winning_tile]);
        game.stage = Stage::Turn {
            actor: dealer,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::ConcealedKong(kong_tile)).unwrap();
        let hu = game.step(Action::Hu).unwrap();

        assert_eq!(hu.score_delta, [9_600, -3_200, -3_200, -3_200]);
        let summary = game.players[dealer.index()]
            .last_win
            .expect("the supplement draw completed a win");
        assert_eq!(summary.multiplier, 32);
        assert_ne!(summary.flags & EVENT_FLAG_HEAVENLY, 0);
        assert_ne!(summary.flags & EVENT_FLAG_AFTER_KONG, 0);
    }

    #[test]
    fn a_win_on_the_discarded_final_wall_tile_keeps_the_last_wall_multiplier() {
        let mut game = constructed_game();
        let source = Seat::EAST;
        let winner = Seat::ALL[1];
        let winning_tile = make_plain_wait(&mut game.players[winner.index()]);
        set_wall(&mut game, &[winning_tile]);

        game.begin_normal_turn(source);
        game.step(Action::Discard(winning_tile)).unwrap();

        assert_ne!(game.discard_flags[0] & EVENT_FLAG_LAST_WALL_TILE, 0);
        let hu = game.step(Action::Hu).unwrap();
        let summary = game.players[winner.index()]
            .last_win
            .expect("the final discard completed a win");
        assert_eq!(summary.multiplier, 2);
        assert_ne!(summary.flags & EVENT_FLAG_LAST_WALL_TILE, 0);
        assert!(hu.terminal);
    }

    #[test]
    fn the_last_wall_multiplier_survives_a_pong_before_the_winning_discard() {
        let mut game = constructed_game();
        let source = Seat::EAST;
        let ponger = Seat::ALL[1];
        let winner = Seat::ALL[2];
        let called = tile(Suit::Characters, 5);
        let winning_tile = make_plain_wait(&mut game.players[winner.index()]);
        add_tile(&mut game.players[ponger.index()], called, 2);
        add_tile(&mut game.players[ponger.index()], winning_tile, 1);
        game.players[ponger.index()].missing = Some(Suit::Dots);
        set_wall(&mut game, &[called]);

        game.begin_normal_turn(source);
        game.step(Action::Discard(called)).unwrap();
        game.step(Action::Pong).unwrap();
        game.step(Action::Discard(winning_tile)).unwrap();

        assert_ne!(game.discard_flags[0] & EVENT_FLAG_LAST_WALL_TILE, 0);
        assert_ne!(game.discard_flags[1] & EVENT_FLAG_LAST_WALL_TILE, 0);
        game.step(Action::Hu).unwrap();
        let summary = game.players[winner.index()]
            .last_win
            .expect("the post-Pong discard completed a win");
        assert_eq!(summary.multiplier, 2);
        assert_ne!(summary.flags & EVENT_FLAG_LAST_WALL_TILE, 0);
    }

    #[test]
    fn one_discard_response_exposes_hu_kong_pong_and_pass() {
        let mut game = constructed_game();
        let source = Seat::EAST;
        let responder = Seat::ALL[1];
        let response_tile = make_quad_seven_pairs_wait(&mut game.players[responder.index()]);
        let next_draw = tile(Suit::Bamboo, 9);
        add_tile(&mut game.players[source.index()], response_tile, 1);
        set_wall(&mut game, &[next_draw]);
        game.stage = Stage::Turn {
            actor: source,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::Discard(response_tile)).unwrap();

        let legal = game
            .legal_actions()
            .expect("the responder has one combined choice");
        assert_eq!(legal.decision.actor, responder);
        assert_eq!(legal.decision.phase, Phase::HuResponse);
        assert!(legal.can_hu);
        assert!(legal.can_pong);
        assert!(legal.can_exposed_kong);
        assert!(legal.can_pass);
        assert_eq!(
            game.legal_action_mask().unwrap().iter().collect::<Vec<_>>(),
            vec![
                ActionId::HU,
                ActionId::PONG,
                ActionId::EXPOSED_KONG,
                ActionId::PASS,
            ]
        );

        let mut hu = game.clone();
        hu.step(Action::Hu).unwrap();
        assert!(hu.has_won(responder));
        assert_eq!(hu.players[responder.index()].meld_count, 0);

        let mut pong = game.clone();
        pong.step(Action::Pong).unwrap();
        assert_eq!(pong.phase(), Phase::Turn);
        assert_eq!(
            pong.decision().map(|decision| decision.actor),
            Some(responder)
        );
        assert_eq!(
            pong.meld(responder, 0).map(|meld| meld.kind),
            Some(MeldKind::Pong)
        );

        let mut kong = game.clone();
        let outcome = kong.step(Action::ExposedKong).unwrap();
        assert_eq!(kong.phase(), Phase::Turn);
        assert_eq!(
            kong.decision().map(|decision| decision.actor),
            Some(responder)
        );
        assert_eq!(
            kong.meld(responder, 0).map(|meld| meld.kind),
            Some(MeldKind::ExposedKong)
        );
        assert_eq!(
            outcome.draw,
            Some(DrawEvent {
                player: responder,
                tile: next_draw,
                replacement: true,
            })
        );

        let mut pass = game;
        let outcome = pass.step(Action::Pass).unwrap();
        assert_eq!(pass.phase(), Phase::Turn);
        assert_eq!(
            pass.decision().map(|decision| decision.actor),
            Some(responder)
        );
        assert_eq!(pass.players[responder.index()].meld_count, 0);
        assert_eq!(
            outcome.draw,
            Some(DrawEvent {
                player: responder,
                tile: next_draw,
                replacement: false,
            })
        );
    }

    #[test]
    fn pending_pong_waits_for_later_hu_and_is_overridden_only_by_hu() {
        let mut game = constructed_game();
        let source = Seat::EAST;
        let meld_responder = Seat::ALL[1];
        let hu_responder = Seat::ALL[2];
        let response_tile = make_zero_copy_wait(&mut game.players[hu_responder.index()]);
        add_tile(&mut game.players[meld_responder.index()], response_tile, 2);
        add_tile(
            &mut game.players[meld_responder.index()],
            tile(Suit::Bamboo, 9),
            1,
        );
        add_tile(&mut game.players[source.index()], response_tile, 1);
        set_wall(&mut game, &[tile(Suit::Dots, 9)]);
        game.stage = Stage::Turn {
            actor: source,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::Discard(response_tile)).unwrap();
        let legal = game.legal_actions().unwrap();
        assert_eq!(legal.decision.actor, meld_responder);
        assert!(!legal.can_hu);
        assert!(legal.can_pong);
        game.step(Action::Pong).unwrap();

        assert_eq!(game.phase(), Phase::HuResponse);
        assert_eq!(
            game.decision().map(|decision| decision.actor),
            Some(hu_responder)
        );
        assert_eq!(game.players[meld_responder.index()].meld_count, 0);
        assert!(game.legal_actions().unwrap().can_hu);

        let mut hu = game.clone();
        hu.step(Action::Hu).unwrap();
        assert!(hu.has_won(hu_responder));
        assert_eq!(hu.players[meld_responder.index()].meld_count, 0);

        let mut pass = game;
        pass.step(Action::Pass).unwrap();
        assert!(!pass.has_won(hu_responder));
        assert_eq!(
            pass.meld(meld_responder, 0).map(|meld| meld.kind),
            Some(MeldKind::Pong)
        );
        assert_eq!(pass.phase(), Phase::Turn);
        assert_eq!(
            pass.decision().map(|decision| decision.actor),
            Some(meld_responder)
        );
    }

    #[test]
    fn multiple_hu_results_override_an_earlier_pending_meld() {
        let mut base = constructed_game();
        let source = Seat::EAST;
        let meld_responder = Seat::ALL[1];
        let first_winner = Seat::ALL[2];
        let second_winner = Seat::ALL[3];
        let response_tile = make_zero_copy_wait(&mut base.players[first_winner.index()]);
        assert_eq!(
            make_zero_copy_wait(&mut base.players[second_winner.index()]),
            response_tile
        );
        add_tile(&mut base.players[meld_responder.index()], response_tile, 3);
        add_tile(&mut base.players[source.index()], response_tile, 1);
        let next_draw = tile(Suit::Dots, 9);
        set_wall(&mut base, &[next_draw]);
        base.stage = Stage::Turn {
            actor: source,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        for pending_action in [Action::Pong, Action::ExposedKong] {
            let mut game = base.clone();
            game.step(Action::Discard(response_tile)).unwrap();
            assert!(game.is_legal_action(pending_action));
            game.step(pending_action).unwrap();
            game.step(Action::Hu).unwrap();
            let outcome = game.step(Action::Hu).unwrap();

            assert!(game.has_won(first_winner));
            assert!(game.has_won(second_winner));
            assert_eq!(game.players[meld_responder.index()].meld_count, 0);
            assert_eq!(
                game.players[meld_responder.index()].unlocked_count(response_tile),
                3
            );
            let winner_payment = outcome.score_delta[first_winner.index()];
            assert!(winner_payment > 0);
            assert_eq!(outcome.score_delta[second_winner.index()], winner_payment);
            assert_eq!(outcome.score_delta[meld_responder.index()], 0);
            assert_eq!(outcome.score_delta[source.index()], -2 * winner_payment);
            assert_eq!(game.phase(), Phase::Turn);
            assert_eq!(
                game.decision().map(|decision| decision.actor),
                Some(meld_responder)
            );
            assert_eq!(
                outcome.draw,
                Some(DrawEvent {
                    player: meld_responder,
                    tile: next_draw,
                    replacement: false,
                })
            );
        }
    }

    #[test]
    fn exposed_kong_outranks_nearer_pong_after_all_hu_candidates_pass() {
        let mut game = constructed_game();
        let source = Seat::EAST;
        let pong_responder = Seat::ALL[1];
        let kong_responder = Seat::ALL[2];
        let hu_responder = Seat::ALL[3];
        let response_tile = make_zero_copy_wait(&mut game.players[hu_responder.index()]);
        add_tile(&mut game.players[kong_responder.index()], response_tile, 3);
        add_tile(&mut game.players[pong_responder.index()], response_tile, 2);
        add_tile(
            &mut game.players[pong_responder.index()],
            tile(Suit::Bamboo, 9),
            1,
        );
        add_tile(&mut game.players[source.index()], response_tile, 1);
        let supplement = tile(Suit::Dots, 9);
        set_wall(&mut game, &[supplement]);
        game.stage = Stage::Turn {
            actor: source,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::Discard(response_tile)).unwrap();
        game.step(Action::Pong).unwrap();
        game.step(Action::ExposedKong).unwrap();
        let outcome = game.step(Action::Pass).unwrap();

        assert_eq!(
            game.meld(kong_responder, 0).map(|meld| meld.kind),
            Some(MeldKind::ExposedKong)
        );
        assert_eq!(game.players[pong_responder.index()].meld_count, 0);
        assert_eq!(
            game.decision().map(|decision| decision.actor),
            Some(kong_responder)
        );
        assert_eq!(outcome.score_delta, [-300, 0, 300, 0]);
        assert_eq!(
            outcome.draw,
            Some(DrawEvent {
                player: kong_responder,
                tile: supplement,
                replacement: true,
            })
        );
    }

    #[test]
    fn exposed_kong_outranks_nearer_pong_without_hu_candidates() {
        let mut game = constructed_game();
        let source = Seat::EAST;
        let pong_responder = Seat::ALL[1];
        let kong_responder = Seat::ALL[2];
        let response_tile = tile(Suit::Characters, 5);
        add_tile(&mut game.players[pong_responder.index()], response_tile, 2);
        add_tile(
            &mut game.players[pong_responder.index()],
            tile(Suit::Bamboo, 9),
            1,
        );
        add_tile(&mut game.players[kong_responder.index()], response_tile, 3);
        add_tile(&mut game.players[source.index()], response_tile, 1);
        let supplement = tile(Suit::Dots, 9);
        set_wall(&mut game, &[supplement]);
        game.stage = Stage::Turn {
            actor: source,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::Discard(response_tile)).unwrap();
        let pending_pong = game.step(Action::Pong).unwrap();
        assert_eq!(pending_pong.score_delta, [0; PLAYER_COUNT]);
        assert_eq!(game.players[pong_responder.index()].meld_count, 0);
        assert_eq!(
            game.decision().map(|decision| decision.actor),
            Some(kong_responder)
        );

        let outcome = game.step(Action::ExposedKong).unwrap();
        assert_eq!(game.players[pong_responder.index()].meld_count, 0);
        assert_eq!(
            game.meld(kong_responder, 0).map(|meld| meld.kind),
            Some(MeldKind::ExposedKong)
        );
        assert_eq!(outcome.score_delta, [-300, 0, 300, 0]);
        assert_eq!(
            outcome.draw,
            Some(DrawEvent {
                player: kong_responder,
                tile: supplement,
                replacement: true,
            })
        );
    }

    #[test]
    fn observation_reveals_only_an_opponents_winning_tile() {
        let mut game = constructed_game();
        let source = Seat::EAST;
        let winner = Seat::ALL[1];
        let winning_tile = make_plain_wait(&mut game.players[winner.index()]);
        add_tile(&mut game.players[source.index()], winning_tile, 1);
        set_wall(&mut game, &[tile(Suit::Characters, 9)]);
        game.stage = Stage::Turn {
            actor: source,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::Discard(winning_tile)).unwrap();
        game.step(Action::Hu).unwrap();

        let public_wins = game.public_win_tiles(winner);
        assert_eq!(public_wins.iter().sum::<u8>(), 1);
        assert_eq!(public_wins[winning_tile.index()], 1);
        assert_eq!(game.locked(winner).iter().sum::<u8>(), 14);
        assert_eq!(game.win_base(winner).iter().sum::<u8>(), 13);

        let source_observation = observation_for(&game, source).0;
        let winner_relative = usize::from(relative_seat(source, winner));
        let opponent_start = (2 + winner_relative) * TILE_KIND_COUNT;
        assert_eq!(
            &source_observation[opponent_start..opponent_start + TILE_KIND_COUNT],
            &public_wins
        );

        let winner_observation = observation_for(&game, winner).0;
        let own_locked_start = 2 * TILE_KIND_COUNT;
        assert_eq!(
            &winner_observation[own_locked_start..own_locked_start + TILE_KIND_COUNT],
            game.locked(winner)
        );
    }

    #[test]
    fn visible_tile_counts_deduplicate_each_discard_win_reference() {
        let mut game = constructed_game();
        let first_winner = Seat::ALL[1];
        let second_winner = Seat::ALL[2];
        let viewer = Seat::ALL[3];
        let winning_tile = make_plain_wait(&mut game.players[first_winner.index()]);
        assert_eq!(
            make_plain_wait(&mut game.players[second_winner.index()]),
            winning_tile
        );
        add_tile(&mut game.players[Seat::EAST.index()], winning_tile, 1);
        game.stage = Stage::Turn {
            actor: Seat::EAST,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::Discard(winning_tile)).unwrap();
        game.step(Action::Hu).unwrap();
        game.step(Action::Hu).unwrap();

        assert_eq!(game.discard_len, 0);
        assert_eq!(game.duplicate_win_references[winning_tile.index()], 1);
        assert_eq!(game.public_win_tiles(first_winner)[winning_tile.index()], 1);
        assert_eq!(
            game.public_win_tiles(second_winner)[winning_tile.index()],
            1
        );
        assert_eq!(game.visible_tile_counts(viewer)[winning_tile.index()], 1);
    }

    #[test]
    fn robbed_added_kong_does_not_create_a_discard_win_reference() {
        let mut game = constructed_game();
        let winner = Seat::ALL[1];
        let winning_tile = make_plain_wait(&mut game.players[winner.index()]);
        add_tile(&mut game.players[Seat::EAST.index()], winning_tile, 1);
        let next_draw = tile(Suit::Dots, 9);
        set_wall(&mut game, &[next_draw]);

        game.resolve_discard_wins(
            Seat::EAST,
            winning_tile,
            seat_bit(winner),
            ReactionKind::AddedKong,
        );

        assert_eq!(game.duplicate_win_references[winning_tile.index()], 0);
        assert_eq!(
            game.visible_tile_counts(Seat::ALL[2])[winning_tile.index()],
            1
        );
        assert_eq!(game.decision().unwrap().actor, winner.next());
        assert_eq!(
            game.transition_draw,
            Some(DrawEvent {
                player: winner.next(),
                tile: next_draw,
                replacement: false,
            })
        );
    }

    #[test]
    fn multiple_robbed_added_kong_wins_share_one_physical_tile() {
        let mut game = constructed_game();
        let source = Seat::EAST;
        let first_winner = Seat::ALL[1];
        let second_winner = Seat::ALL[2];
        let viewer = Seat::ALL[3];
        let winning_tile = make_zero_copy_wait(&mut game.players[first_winner.index()]);
        assert_eq!(
            make_zero_copy_wait(&mut game.players[second_winner.index()]),
            winning_tile,
        );
        add_tile(&mut game.players[source.index()], winning_tile, 1);
        assert!(game.players[source.index()].add_meld(Meld {
            tile: winning_tile,
            kind: MeldKind::Pong,
            source: viewer,
        }));
        game.discards[0] = winning_tile.as_u8();
        game.discard_owners[0] = viewer.as_u8();
        game.discard_len = 1;
        let next_draw = tile(Suit::Dots, 9);
        set_wall(&mut game, &[next_draw]);

        game.resolve_discard_wins(
            source,
            winning_tile,
            seat_bit(first_winner) | seat_bit(second_winner),
            ReactionKind::AddedKong,
        );

        assert_eq!(game.duplicate_win_references[winning_tile.index()], 1);
        assert_eq!(game.locked(first_winner)[winning_tile.index()], 1);
        assert_eq!(game.locked(second_winner)[winning_tile.index()], 1);
        assert_eq!(game.visible_tile_counts(viewer)[winning_tile.index()], 4);
        assert_eq!(game.decision().unwrap().actor, source.next());
        assert_eq!(
            game.transition_draw,
            Some(DrawEvent {
                player: source.next(),
                tile: next_draw,
                replacement: false,
            })
        );
    }

    #[test]
    fn multiple_hu_suppresses_melds_and_resumes_after_the_discarder() {
        let mut game = constructed_game();
        let first_winner = Seat::ALL[1];
        let last_winner = Seat::ALL[2];
        let meld_candidate = Seat::ALL[3];
        let winning_tile = make_plain_wait(&mut game.players[first_winner.index()]);
        assert_eq!(
            make_plain_wait(&mut game.players[last_winner.index()]),
            winning_tile
        );
        add_tile(&mut game.players[meld_candidate.index()], winning_tile, 2);
        add_tile(&mut game.players[Seat::EAST.index()], winning_tile, 1);
        game.discard_len = 1;
        let next_draw = tile(Suit::Characters, 9);
        set_wall(&mut game, &[next_draw]);
        game.stage = Stage::Turn {
            actor: Seat::EAST,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::Discard(winning_tile)).unwrap();
        let first_response = game.step(Action::Hu).unwrap();
        assert_eq!(first_response.score_delta, [0; PLAYER_COUNT]);
        assert_eq!(first_response.next.unwrap().actor, last_winner);

        let resolved = game.step(Action::Hu).unwrap();
        assert_eq!(resolved.score_delta, [-200, 100, 100, 0]);
        assert_eq!(game.meld_count(meld_candidate), 0);
        assert_eq!(game.concealed(meld_candidate)[winning_tile.index()], 2);
        assert_eq!(
            resolved.next,
            Some(Decision {
                actor: first_winner,
                phase: Phase::Turn,
            })
        );
        assert_eq!(
            resolved.draw,
            Some(DrawEvent {
                player: first_winner,
                tile: next_draw,
                replacement: false,
            })
        );
    }

    #[test]
    fn exposed_kong_charges_only_the_discarder_and_empty_tail_finishes() {
        let mut game = constructed_game();
        for seat in Seat::ALL {
            make_flower_pig(&mut game.players[seat.index()]);
        }
        let actor = Seat::ALL[1];
        let kong_tile = tile(Suit::Characters, 5);
        add_tile(&mut game.players[actor.index()], kong_tile, 3);
        game.stage = Stage::MeldResponse {
            source: Seat::EAST,
            tile: kong_tile,
            remaining: seat_bit(actor),
            pending_melds: PendingMelds::NONE,
        };

        let outcome = game.step(Action::ExposedKong).unwrap();

        assert_eq!(outcome.score_delta, [-300, 300, 0, 0]);
        assert!(outcome.terminal);
        assert_eq!(outcome.draw, None);
        assert_eq!(game.phase(), Phase::Finished);
        assert_eq!(
            game.meld(actor, 0),
            Some(Meld {
                tile: kong_tile,
                kind: MeldKind::ExposedKong,
                source: Seat::EAST,
            })
        );
    }

    #[test]
    fn added_kong_can_use_a_previously_held_tile() {
        let mut game = constructed_game();
        for seat in Seat::ALL {
            make_flower_pig(&mut game.players[seat.index()]);
        }
        let actor = Seat::ALL[1];
        let kong_tile = tile(Suit::Characters, 5);
        let current_draw = tile(Suit::Bamboo, 5);
        add_tile(&mut game.players[actor.index()], kong_tile, 1);
        add_tile(&mut game.players[actor.index()], current_draw, 1);
        assert!(game.players[actor.index()].add_meld(Meld {
            tile: kong_tile,
            kind: MeldKind::Pong,
            source: Seat::EAST,
        }));
        game.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::Draw {
                tile: current_draw,
                after_kong: false,
                last_wall_tile: false,
                deferred_kong_forbidden_tile: None,
            },
            can_hu: false,
        };

        let outcome = game.step(Action::AddedKong(kong_tile)).unwrap();

        assert_eq!(outcome.score_delta, [-100, 300, -100, -100]);
        assert!(outcome.terminal);
        assert_eq!(outcome.draw, None);
        assert_eq!(game.phase(), Phase::Finished);
        assert_eq!(game.meld(actor, 0).unwrap().kind, MeldKind::AddedKong);
    }

    #[test]
    fn a_drawn_added_kong_offer_is_visible_and_persistently_forbidden() {
        let mut game = constructed_game();
        let actor = Seat::ALL[1];
        let kong_tile = tile(Suit::Characters, 5);
        let next_draw = tile(Suit::Bamboo, 9);
        assert!(game.players[actor.index()].add_meld(Meld {
            tile: kong_tile,
            kind: MeldKind::Pong,
            source: Seat::EAST,
        }));
        game.players[actor.index()].missing = Some(Suit::Dots);
        set_wall(&mut game, &[kong_tile, next_draw]);

        game.begin_normal_turn(actor);

        let legal = game.legal_actions().expect("the draw creates a Kong offer");
        assert_ne!(legal.added_kong_mask & (1 << kong_tile.index()), 0);
        assert_eq!(
            game.players[actor.index()].kong_forbidden_mask & (1 << kong_tile.index()),
            0,
            "the current offer remains legal until the decision is applied"
        );
        let observation = observation_for(&game, actor);
        assert_eq!(
            observation.0[10 * TILE_KIND_COUNT + kong_tile.index()],
            1,
            "the viewer observes the persistent state created by the offer"
        );

        game.step(Action::Discard(kong_tile)).unwrap();

        assert_ne!(
            game.players[actor.index()].kong_forbidden_mask & (1 << kong_tile.index()),
            0
        );
    }

    #[test]
    fn added_kong_declaration_is_public_before_rob_kong_response() {
        let mut game = constructed_game();
        let actor = Seat::EAST;
        let winner = Seat::ALL[1];
        let observer = Seat::ALL[2];
        let kong_tile = make_plain_wait(&mut game.players[winner.index()]);
        add_tile(&mut game.players[actor.index()], kong_tile, 1);
        assert!(game.players[actor.index()].add_meld(Meld {
            tile: kong_tile,
            kind: MeldKind::Pong,
            source: Seat::ALL[3],
        }));
        game.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::AddedKong(kong_tile)).unwrap();

        let legal = game
            .legal_actions()
            .expect("the rob-Kong response is pending");
        assert!(legal.can_hu);
        assert!(legal.can_pass);
        assert!(!legal.can_pong);
        assert!(!legal.can_exposed_kong);

        assert_eq!(
            game.stage,
            Stage::HuResponse {
                source: actor,
                tile: kong_tile,
                remaining: seat_bit(winner),
                winners: 0,
                pending_melds: PendingMelds::NONE,
                kind: ReactionKind::AddedKong,
            }
        );
        let mut events = [0_i32; EVENT_RECORD_WIDTH];
        assert_eq!(game.step_events_into(observer, &mut events).unwrap(), 1);
        assert_eq!(events[0], i32::from(EventKind::Action.code()));
        assert_eq!(events[1], i32::from(relative_seat(observer, actor)));
        assert_eq!(events[3], i32::from(kong_tile.as_u8()));
        assert_eq!(events[5], ActionId::added_kong(kong_tile).index() as i32);
    }

    #[test]
    fn concealed_kong_charges_two_units_each_and_empty_tail_finishes() {
        let mut game = constructed_game();
        for seat in Seat::ALL {
            make_flower_pig(&mut game.players[seat.index()]);
        }
        let actor = Seat::ALL[1];
        let kong_tile = tile(Suit::Characters, 5);
        add_tile(&mut game.players[actor.index()], kong_tile, 4);
        game.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        let outcome = game.step(Action::ConcealedKong(kong_tile)).unwrap();

        assert_eq!(
            outcome.score_delta,
            [-200, 600, -200, -200],
            "wall exhaustion must not refund an already settled Kong"
        );
        assert!(outcome.terminal);
        assert_eq!(outcome.draw, None);
        assert_eq!(game.phase(), Phase::Finished);
        assert_eq!(game.meld(actor, 0).unwrap().kind, MeldKind::ConcealedKong);
    }

    #[test]
    fn repeated_hu_preserves_the_base_and_adds_one_public_winning_tile() {
        let mut game = constructed_game();
        let actor = Seat::ALL[1];
        let player = &mut game.players[actor.index()];
        add_sequence(player, Suit::Characters, 1);
        add_sequence(player, Suit::Bamboo, 4);
        add_tile(player, tile(Suit::Bamboo, 7), 3);
        add_tile(player, tile(Suit::Bamboo, 8), 2);
        add_tile(player, tile(Suit::Bamboo, 9), 2);
        player.missing = Some(Suit::Dots);

        let first_winning_tile = tile(Suit::Bamboo, 8);
        add_tile(&mut game.players[Seat::EAST.index()], first_winning_tile, 1);
        game.discard_len = 1;
        set_wall(&mut game, &[tile(Suit::Characters, 9)]);
        game.stage = Stage::Turn {
            actor: Seat::EAST,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::Discard(first_winning_tile)).unwrap();
        game.step(Action::Hu).unwrap();
        let first_locks = *game.locked(actor);
        let first_base = *game.win_base(actor);
        assert_eq!(first_locks.iter().sum::<u8>(), 14);
        assert_eq!(first_base.iter().sum::<u8>(), 13);
        assert_eq!(game.public_win_tiles(actor)[first_winning_tile.index()], 1);
        assert_eq!(game.players[actor.index()].unlocked_len(), 0);

        let second_winning_tile = tile(Suit::Bamboo, 9);
        let following_draw = tile(Suit::Characters, 8);
        set_wall(&mut game, &[second_winning_tile, following_draw]);
        game.begin_normal_turn(actor);
        assert!(game.legal_actions().unwrap().can_hu);
        assert_ne!(
            game.legal_actions().unwrap().discard_mask & (1 << second_winning_tile.index()),
            0
        );
        assert_eq!(
            game.legal_actions().unwrap().discard_mask,
            1 << second_winning_tile.index()
        );

        let second_hu = game.step(Action::Hu).unwrap();
        assert_eq!(game.locked(actor).iter().sum::<u8>(), 15);
        assert_eq!(game.win_base(actor), &first_base);
        for (before, after) in first_locks.iter().zip(game.locked(actor)) {
            assert!(after >= before);
        }
        assert_eq!(game.public_win_tiles(actor).iter().sum::<u8>(), 2);
        assert_eq!(game.public_win_tiles(actor)[second_winning_tile.index()], 1);
        assert_eq!(game.players[actor.index()].unlocked_len(), 0);
        assert_eq!(game.win_count(actor), 2);
        let last_win = game.last_win(actor).expect("repeat win has a summary");
        assert!(last_win.shape_multiplier > 0);
        assert!(last_win.multiplier >= last_win.shape_multiplier);
        assert_ne!(last_win.flags & EVENT_FLAG_SELF_DRAW, 0);
        assert_eq!(second_hu.draw.unwrap().tile, following_draw);
        assert_eq!(second_hu.next.unwrap().actor, actor.next());
    }

    #[test]
    fn repeated_self_draw_hu_keeps_the_base_locked() {
        let mut game = constructed_game();
        let actor = Seat::ALL[1];
        let winning_tile = make_plain_wait(&mut game.players[actor.index()]);
        add_tile(&mut game.players[actor.index()], winning_tile, 1);
        game.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::Draw {
                tile: winning_tile,
                after_kong: false,
                last_wall_tile: false,
                deferred_kong_forbidden_tile: None,
            },
            can_hu: true,
        };

        game.step(Action::Hu).unwrap();
        assert_eq!(game.players[actor.index()].unlocked_len(), 0);
        assert_eq!(game.locked(actor).iter().sum::<u8>(), 14);
        assert_eq!(game.win_base(actor).iter().sum::<u8>(), 13);

        set_wall(&mut game, &[winning_tile, tile(Suit::Characters, 9)]);
        game.begin_normal_turn(actor);
        assert!(game.legal_actions().unwrap().can_hu);
        assert_eq!(
            game.legal_actions().unwrap().discard_mask,
            1 << winning_tile.index()
        );
        game.step(Action::Hu).unwrap();

        assert_eq!(game.players[actor.index()].unlocked_len(), 0);
        assert_eq!(game.locked(actor).iter().sum::<u8>(), 15);
        assert_eq!(game.win_base(actor).iter().sum::<u8>(), 13);
        assert_eq!(
            game.concealed(actor)
                .iter()
                .map(|&count| count as usize)
                .sum::<usize>(),
            15
        );
    }

    #[test]
    fn historical_win_references_do_not_expand_repeat_waits_or_change_shape() {
        let mut game = constructed_game();
        let source = Seat::EAST;
        let winner = Seat::ALL[1];
        let winning_tile = make_plain_wait(&mut game.players[winner.index()]);
        add_tile(&mut game.players[source.index()], winning_tile, 1);
        set_wall(&mut game, &[tile(Suit::Characters, 9)]);
        game.stage = Stage::Turn {
            actor: source,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        game.step(Action::Discard(winning_tile)).unwrap();
        game.step(Action::Hu).unwrap();

        let repeat_waits: Vec<_> = (0..TILE_KIND_COUNT)
            .map(|index| Tile::from_index_unchecked(index as u8))
            .filter(|&tile| game.can_player_win_with_added_tile(winner, tile))
            .collect();
        assert_eq!(repeat_waits, vec![winning_tile]);
        let active_hand = game.players[winner.index()]
            .evaluation_counts()
            .expect("historical references are a concealed subset");
        assert_eq!(active_hand.iter().sum::<u8>(), 13);

        let mut shape_multipliers = Vec::new();
        for historical_tile in [tile(Suit::Characters, 7), tile(Suit::Bamboo, 8)] {
            game.players[winner.index()].concealed[winning_tile.index()] += 1;
            let evaluation = game
                .evaluate_player(winner, Some(winning_tile), WinFlags::NONE)
                .expect("the active hand keeps the same repeat wait");
            shape_multipliers.push(evaluation.shape_multiplier);
            game.apply_win(winner, Some(winning_tile), evaluation, 0);
            assert_eq!(
                game.players[winner.index()].evaluation_counts(),
                Some(active_hand)
            );

            game.players[winner.index()].concealed[historical_tile.index()] += 1;
            game.players[winner.index()].locked[historical_tile.index()] += 1;
            assert_eq!(
                game.players[winner.index()].evaluation_counts(),
                Some(active_hand)
            );
        }
        assert_eq!(shape_multipliers[0], shape_multipliers[1]);
    }

    #[test]
    fn historical_win_reference_does_not_complete_the_active_hand() {
        let mut game = constructed_game();
        let actor = Seat::ALL[1];
        let player = &mut game.players[actor.index()];
        let winning_tile = make_plain_wait(player);
        player.win_base = player.concealed;
        player.locked = player.win_base;
        add_tile(player, winning_tile, 1);
        player.locked[winning_tile.index()] += 1;
        player.has_won = true;

        assert!(!game.can_player_win(actor, None));
        assert!(game.can_player_win_with_added_tile(actor, winning_tile));
    }

    #[test]
    fn bankrupt_players_are_excluded_from_wall_settlement() {
        let mut game = constructed_game();
        make_flower_pig(&mut game.players[Seat::EAST.index()]);
        game.players[Seat::EAST.index()].missing = Some(Suit::Characters);
        make_flower_pig(&mut game.players[Seat::ALL[1].index()]);
        game.players[Seat::ALL[1].index()].missing = Some(Suit::Characters);
        make_plain_wait(&mut game.players[Seat::ALL[2].index()]);
        make_plain_wait(&mut game.players[Seat::ALL[3].index()]);
        game.players[Seat::EAST.index()].score = 1_000;
        game.players[Seat::ALL[1].index()].score = 0;
        game.players[Seat::ALL[2].index()].score = 10_000;
        game.players[Seat::ALL[3].index()].score = 29_000;

        assert_eq!(game.max_wait_multiplier(Seat::ALL[1]), 0);
        assert_eq!(game.max_wait_multiplier(Seat::ALL[2]), 1);
        assert_eq!(game.max_wait_multiplier(Seat::ALL[3]), 1);

        game.finish_wall_game();

        let settlement = game.wall_settlement().expect("wall settlement is public");
        assert!(settlement.is_flower_pig(Seat::EAST));
        assert!(!settlement.is_flower_pig(Seat::ALL[1]));
        assert!(!settlement.is_ready(Seat::EAST));
        assert!(!settlement.is_ready(Seat::ALL[1]));
        assert_eq!(settlement.max_shape_multiplier(Seat::ALL[1]), 0);
        assert_eq!(
            settlement.revealed_hand(Seat::EAST),
            game.concealed(Seat::EAST)
        );

        let mut settlement_events = [0_i32; EVENT_HISTORY_CAPACITY * EVENT_RECORD_WIDTH];
        let event_count = game
            .events_into(Seat::EAST, &mut settlement_events)
            .expect("event buffer is valid");
        let stages: Vec<_> = settlement_events[..event_count * EVENT_RECORD_WIDTH]
            .chunks_exact(EVENT_RECORD_WIDTH)
            .filter(|record| record[0] == i32::from(EventKind::SettlementStage.code()))
            .map(|record| record[4])
            .collect();
        assert_eq!(
            stages,
            vec![
                i32::from(SettlementStage::FlowerPig.code()),
                i32::from(SettlementStage::Dajiao.code()),
            ]
        );

        assert_eq!(
            Seat::ALL.map(|seat| game.score(seat)),
            [0, 0, 11_000, 29_000]
        );
        assert_eq!(game.phase(), Phase::Finished);
        assert_eq!(
            Seat::ALL.iter().map(|&seat| game.score(seat)).sum::<i64>(),
            40_000
        );
    }

    #[test]
    fn ui_buffers_filter_win_and_terminal_settlement_information() {
        let mut game = Game::new(9);
        let mut stats = [99_i32; PLAYER_UI_STATS_WIDTH];
        game.player_ui_stats_into(Seat::ALL[2], &mut stats)
            .expect("stats buffer is valid");
        assert!(
            stats
                .chunks_exact(PLAYER_UI_STATS_FIELDS)
                .all(|row| { row[0] == 0 && row[1..].iter().all(|&value| value == -1) })
        );

        let mut meta = [99_i32; WALL_SETTLEMENT_META_WIDTH];
        let mut hands = [99_u8; WALL_SETTLEMENT_HANDS_WIDTH];
        assert!(
            !game
                .wall_settlement_into(Seat::EAST, &mut meta, &mut hands)
                .expect("settlement buffers are valid")
        );
        assert!(meta.iter().all(|&value| value == -1));
        assert!(hands.iter().all(|&value| value == 0));

        game.wall_head = game.wall_tail;
        game.finish_wall_game();
        assert!(
            game.wall_settlement_into(Seat::ALL[2], &mut meta, &mut hands)
                .expect("settlement buffers are valid")
        );
        for relative in 0..PLAYER_COUNT {
            let seat = Seat::ALL[2].offset(relative as u8);
            let base = relative * WALL_SETTLEMENT_FIELDS;
            let summary = game.wall_settlement().expect("summary exists");
            assert_eq!(meta[base], i32::from(u8::from(summary.is_flower_pig(seat))));
            assert_eq!(meta[base + 1], i32::from(u8::from(summary.is_ready(seat))));
            let hand_base = relative * TILE_KIND_COUNT;
            assert_eq!(
                &hands[hand_base..hand_base + TILE_KIND_COUNT],
                summary.revealed_hand(seat),
            );
        }
    }

    #[test]
    fn robbed_added_kong_removes_only_the_active_fourth_tile() {
        let mut game = Game::new(77);
        let tile = Tile::from_index_unchecked(4);
        game.players[Seat::EAST.index()].concealed[tile.index()] = 2;
        game.players[Seat::EAST.index()].locked[tile.index()] = 1;

        game.remove_robbed_added_kong_tile(Seat::EAST, tile);
        assert_eq!(game.players[Seat::EAST.index()].concealed[tile.index()], 1);
        assert_eq!(game.players[Seat::EAST.index()].locked[tile.index()], 1);
    }

    #[test]
    fn observation_pending_fields_cover_discard_and_added_kong_reactions() {
        let source = Seat::EAST;
        let actor = Seat::ALL[2];
        let pending_tile = tile(Suit::Bamboo, 5);
        for (kind, expected_flags) in [
            (
                ReactionKind::Discard {
                    after_kong: true,
                    opening_discard: true,
                    last_wall_tile: true,
                },
                0b1110,
            ),
            (ReactionKind::AddedKong, 0b001),
        ] {
            let mut game = constructed_game();
            game.stage = Stage::HuResponse {
                source,
                tile: pending_tile,
                remaining: seat_bit(actor),
                winners: seat_bit(Seat::ALL[1]),
                pending_melds: PendingMelds::NONE,
                kind,
            };
            let batch = Batch { games: vec![game] };
            let mut tile_obs = [0; TILE_OBSERVATION_WIDTH];
            let mut melds = [0; MELD_OBSERVATION_WIDTH];
            let mut river = [0; RIVER_OBSERVATION_WIDTH];
            let mut meta = [0; META_OBSERVATION_WIDTH];

            batch
                .observations_into(&mut tile_obs, &mut melds, &mut river, &mut meta)
                .unwrap();

            assert_eq!(meta[0], i32::from(Phase::HuResponse.code()));
            assert_eq!(meta[1], i32::from(actor.as_u8()));
            assert_eq!(meta[7], i32::from(relative_seat(actor, source)));
            assert_eq!(meta[8], i32::from(pending_tile.as_u8()));
            assert_eq!(meta[29], expected_flags);
            assert!(river.iter().all(|&value| value == u8::MAX));
        }
    }

    #[test]
    fn information_set_resampling_preserves_actor_observation_and_tile_inventory() {
        let mut game = Game::new(913);
        while matches!(game.phase(), Phase::Exchange | Phase::ChooseMissing) {
            let action = game.simple_rule_action().expect("setup is non-terminal");
            game.step_id(action).unwrap();
        }
        let viewer = game.decision().unwrap().actor;
        let mut tile_obs = [0; TILE_OBSERVATION_WIDTH];
        let mut melds = [0; MELD_OBSERVATION_WIDTH];
        let mut river = [0; RIVER_OBSERVATION_WIDTH];
        let mut meta = [0; META_OBSERVATION_WIDTH];
        game.observation_into(viewer, &mut tile_obs, &mut melds, &mut river, &mut meta)
            .unwrap();
        let mut oracle = [0; ORACLE_TILE_COUNT_WIDTH];
        game.oracle_tile_counts_into(&mut oracle).unwrap();

        let mut sampled = game.resample_information_set(71).unwrap();
        let mut sampled_tile_obs = [0; TILE_OBSERVATION_WIDTH];
        let mut sampled_melds = [0; MELD_OBSERVATION_WIDTH];
        let mut sampled_river = [0; RIVER_OBSERVATION_WIDTH];
        let mut sampled_meta = [0; META_OBSERVATION_WIDTH];
        sampled
            .observation_into(
                viewer,
                &mut sampled_tile_obs,
                &mut sampled_melds,
                &mut sampled_river,
                &mut sampled_meta,
            )
            .unwrap();
        assert_eq!(sampled_tile_obs, tile_obs);
        assert_eq!(sampled_melds, melds);
        assert_eq!(sampled_river, river);
        assert_eq!(sampled_meta, meta);

        let mut sampled_oracle = [0; ORACLE_TILE_COUNT_WIDTH];
        sampled
            .oracle_tile_counts_into(&mut sampled_oracle)
            .unwrap();
        for tile in 0..TILE_KIND_COUNT {
            let inventory = (0..PLAYER_COUNT)
                .map(|seat| oracle[seat * TILE_KIND_COUNT + tile] as u16)
                .sum::<u16>()
                + oracle[(ORACLE_TILE_COUNT_PLANES - 1) * TILE_KIND_COUNT + tile] as u16;
            let sampled_inventory = (0..PLAYER_COUNT)
                .map(|seat| sampled_oracle[seat * TILE_KIND_COUNT + tile] as u16)
                .sum::<u16>()
                + sampled_oracle[(ORACLE_TILE_COUNT_PLANES - 1) * TILE_KIND_COUNT + tile] as u16;
            assert_eq!(sampled_inventory, inventory);
        }
        assert_ne!(sampled_oracle, oracle);

        for _ in 0..512 {
            let Some(action) = sampled.simple_rule_action() else {
                break;
            };
            sampled.step_id(action).unwrap();
        }
        assert_eq!(sampled.phase(), Phase::Finished);
    }

    #[test]
    fn information_set_resampling_canonicalizes_hidden_active_hands() {
        let viewer = Seat::ALL[2];
        let winner = Seat::ALL[1];
        let mut first_active_player = Player::new();
        let winning_tile = make_plain_wait(&mut first_active_player);
        let first_active = first_active_player.concealed;

        let mut second_active_player = Player::new();
        add_tile(&mut second_active_player, tile(Suit::Characters, 1), 2);
        add_tile(&mut second_active_player, tile(Suit::Characters, 2), 3);
        add_tile(&mut second_active_player, tile(Suit::Characters, 9), 3);
        add_tile(&mut second_active_player, tile(Suit::Bamboo, 1), 3);
        add_tile(&mut second_active_player, tile(Suit::Bamboo, 9), 2);
        second_active_player.missing = Some(Suit::Dots);
        let second_active = second_active_player.concealed;

        let make_winner = |active: [u8; TILE_KIND_COUNT]| {
            let mut player = Player::new();
            player.concealed = active;
            player.win_base = active;
            player.locked = active;
            player.concealed[winning_tile.index()] += 1;
            player.locked[winning_tile.index()] += 1;
            player.missing = Some(Suit::Dots);
            player.has_won = true;
            player
        };
        let tile_list = |counts: &[u8; TILE_KIND_COUNT]| {
            let mut tiles = Vec::with_capacity(13);
            for (index, &count) in counts.iter().enumerate() {
                tiles.extend(core::iter::repeat_n(index as u8, count as usize));
            }
            tiles
        };

        // Swap the two hidden active hands between the hand and wall. Both worlds
        // therefore have the same public state and the same unknown pool.
        let filler = core::array::from_fn(|index| (index / TILE_COPIES as usize) as u8);
        let mut left_wall = filler;
        let mut right_wall = filler;
        for (index, tile) in tile_list(&second_active).into_iter().enumerate() {
            left_wall[index] = tile;
        }
        for (index, tile) in tile_list(&first_active).into_iter().enumerate() {
            right_wall[index] = tile;
        }

        let mut left = constructed_game();
        left.players[winner.index()] = make_winner(first_active);
        left.players[viewer.index()].concealed[0] = 1;
        left.wall = left_wall;
        left.wall_head = 0;
        left.wall_tail = WALL_TILE_COUNT as u8;
        left.stage = Stage::Turn {
            actor: viewer,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };
        let mut right = left.clone();
        right.players[winner.index()] = make_winner(second_active);
        right.wall = right_wall;

        assert_eq!(
            observation_for(&left, viewer),
            observation_for(&right, viewer)
        );
        for seed in 0..8 {
            let sampled_left = left
                .resample_information_set(seed)
                .expect("a valid hidden active hand is available in the pool");
            let sampled_right = right
                .resample_information_set(seed)
                .expect("a valid hidden active hand is available in the pool");
            assert_eq!(sampled_left, sampled_right);
            assert_eq!(
                sampled_left.public_win_tiles(winner),
                left.public_win_tiles(winner)
            );
            assert_eq!(sampled_left.players[winner.index()].unlocked_len(), 0);
            assert_eq!(sampled_left.win_base(winner).iter().sum::<u8>(), 13);
            for index in 0..TILE_KIND_COUNT {
                assert!(
                    sampled_left.players[winner.index()].locked[index]
                        <= sampled_left.players[winner.index()].concealed[index]
                );
            }
        }
    }

    #[test]
    fn information_set_resampling_conditions_every_hidden_win_base() {
        let first_winner = Seat::EAST;
        let second_winner = Seat::ALL[1];
        let viewer = Seat::ALL[2];
        let mut game = constructed_game();

        let first = &mut game.players[first_winner.index()];
        add_sequence(first, Suit::Characters, 1);
        add_sequence(first, Suit::Characters, 7);
        add_tile(first, tile(Suit::Bamboo, 4), 1);
        add_tile(first, tile(Suit::Bamboo, 5), 1);
        add_tile(first, tile(Suit::Bamboo, 9), 2);
        first.win_base = first.concealed;
        first.locked = first.win_base;
        for winning_tile in [tile(Suit::Bamboo, 3), tile(Suit::Bamboo, 6)] {
            add_tile(first, winning_tile, 1);
            first.locked[winning_tile.index()] += 1;
        }
        first.missing = Some(Suit::Dots);
        first.has_won = true;
        first.win_count = 2;
        assert!(first.add_meld(Meld {
            tile: tile(Suit::Characters, 5),
            kind: MeldKind::Pong,
            source: Seat::ALL[3],
        }));

        let second = &mut game.players[second_winner.index()];
        let second_winning_tile = make_plain_wait(second);
        second.win_base = second.concealed;
        second.locked = second.win_base;
        add_tile(second, second_winning_tile, 1);
        second.locked[second_winning_tile.index()] += 1;
        second.has_won = true;
        second.win_count = 1;

        game.players[viewer.index()].concealed[tile(Suit::Dots, 4).index()] = 1;
        let mut wall = Vec::with_capacity(12);
        for rank in 1..=3 {
            for _ in 0..TILE_COPIES {
                wall.push(tile(Suit::Dots, rank));
            }
        }
        set_wall(&mut game, &wall);
        game.stage = Stage::Turn {
            actor: viewer,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        let expected_observation = observation_for(&game, viewer);
        let expected_public_wins = [
            game.public_win_tiles(first_winner),
            game.public_win_tiles(second_winner),
        ];
        for seed in 0..16 {
            let sampled = game
                .resample_information_set(seed)
                .expect("the public state has a joint legal hidden assignment");
            assert_eq!(observation_for(&sampled, viewer), expected_observation);
            for (winner, expected_public) in [first_winner, second_winner]
                .into_iter()
                .zip(expected_public_wins)
            {
                assert_eq!(sampled.public_win_tiles(winner), expected_public);
                assert_eq!(sampled.players[winner.index()].unlocked_len(), 0);
                assert_stable_base_supports_public_wins(&sampled, winner);
            }
        }
    }

    #[test]
    fn information_set_resampling_preserves_public_wins_after_a_kong() {
        let mut game = constructed_game();
        let winner = Seat::EAST;
        let viewer = winner.next();
        let player = &mut game.players[winner.index()];
        player.missing = Some(Suit::Dots);
        for rank in 1..=9 {
            add_tile(player, tile(Suit::Characters, rank), 1);
        }
        add_tile(player, tile(Suit::Bamboo, 1), 1);
        player.win_base = player.concealed;
        player.locked = player.win_base;
        let winning_tile = tile(Suit::Bamboo, 9);
        add_tile(player, winning_tile, 1);
        player.locked[winning_tile.index()] += 1;
        player.has_won = true;
        assert!(player.add_meld(Meld {
            tile: tile(Suit::Characters, 5),
            kind: MeldKind::AddedKong,
            source: Seat::ALL[3],
        }));

        game.players[viewer.index()].concealed[0] = 1;
        game.discards[0] = tile(Suit::Characters, 1).as_u8();
        game.discard_owners[0] = winner.as_u8();
        game.discard_len = 1;
        game.wall = core::array::from_fn(|index| (index / TILE_COPIES as usize) as u8);
        game.wall_head = 0;
        game.wall_tail = WALL_TILE_COUNT as u8;
        game.stage = Stage::Turn {
            actor: viewer,
            origin: TurnOrigin::Initial,
            can_hu: false,
        };

        let public_wins = game.public_win_tiles(winner);
        assert_eq!(public_wins.iter().sum::<u8>(), 1);
        assert_eq!(public_wins[winning_tile.index()], 1);
        let expected_observation = observation_for(&game, viewer);
        let sampled = game.resample_information_set(17).unwrap();
        assert_eq!(observation_for(&sampled, viewer), expected_observation);
        assert_eq!(sampled.public_win_tiles(winner), public_wins);
        assert_eq!(sampled.players[winner.index()].unlocked_len(), 0);
        assert_eq!(sampled.win_base(winner).iter().sum::<u8>(), 10);
        for index in 0..TILE_KIND_COUNT {
            assert!(
                sampled.players[winner.index()].locked[index]
                    <= sampled.players[winner.index()].concealed[index]
            );
        }
    }

    #[test]
    fn information_set_batch_sampling_is_deterministic_and_rejects_terminal_states() {
        let batch = Batch::new(3, 101);
        let indices = [2, 0, 2];
        let seeds = [17, 19, 17];
        let first = batch.resample_information_sets(&indices, &seeds).unwrap();
        let second = batch.resample_information_sets(&indices, &seeds).unwrap();
        let mut first_oracle = vec![0; first.len() * ORACLE_TILE_COUNT_WIDTH];
        let mut second_oracle = vec![0; second.len() * ORACLE_TILE_COUNT_WIDTH];
        first.oracle_tile_counts_into(&mut first_oracle).unwrap();
        second.oracle_tile_counts_into(&mut second_oracle).unwrap();
        assert_eq!(first_oracle, second_oracle);
        assert!(matches!(
            batch.resample_information_sets(&[3], &[1]),
            Err(GameError::BatchIndex)
        ));

        let mut terminal = constructed_game();
        terminal.stage = Stage::Finished;
        assert!(matches!(
            terminal.resample_information_set(1),
            Err(GameError::InformationSetUnavailable)
        ));
    }

    #[test]
    fn live_wall_resampling_preserves_state_and_is_deterministic() {
        let mut game = Game::new(211);
        for _ in 0..24 {
            let action = game.simple_rule_action().expect("game remains active");
            game.step_id(action).unwrap();
        }
        let first = game.resample_live_wall(17);
        let repeated = game.resample_live_wall(17);
        let different = game.resample_live_wall(19);
        let mut reordered = game.clone();
        reordered.wall[reordered.wall_head as usize..reordered.wall_tail as usize]
            .sort_unstable_by(|left, right| right.cmp(left));
        let from_reordered = reordered.resample_live_wall(17);
        assert_eq!(first.wall, repeated.wall);
        assert_eq!(first.wall, from_reordered.wall);
        assert_ne!(first.wall, different.wall);
        assert_eq!(first.wall_head, game.wall_head);
        assert_eq!(first.wall_tail, game.wall_tail);
        assert_eq!(
            first.wall[..first.wall_head as usize],
            game.wall[..game.wall_head as usize]
        );
        let mut original_live =
            game.wall[game.wall_head as usize..game.wall_tail as usize].to_vec();
        let mut sampled_live =
            first.wall[first.wall_head as usize..first.wall_tail as usize].to_vec();
        original_live.sort_unstable();
        sampled_live.sort_unstable();
        assert_eq!(sampled_live, original_live);

        let mut original_oracle = [0; ORACLE_TILE_COUNT_WIDTH];
        let mut sampled_oracle = [0; ORACLE_TILE_COUNT_WIDTH];
        game.oracle_tile_counts_into(&mut original_oracle).unwrap();
        first.oracle_tile_counts_into(&mut sampled_oracle).unwrap();
        assert_eq!(sampled_oracle, original_oracle);

        let batch = Batch::new(2, 313);
        let sampled = batch.resample_live_walls(&[1, 1], &[23, 23]).unwrap();
        assert_eq!(sampled.games[0].wall, sampled.games[1].wall);
        assert!(matches!(
            batch.resample_live_walls(&[2], &[1]),
            Err(GameError::BatchIndex)
        ));
        assert!(matches!(
            batch.resample_live_walls(&[0], &[1, 2]),
            Err(GameError::BatchLength)
        ));
    }

    #[test]
    fn swap_removing_indices_returns_the_survivor_order_atomically() {
        let mut batch = Batch::new(4, 317);
        let original = batch.games.iter().map(|game| game.wall).collect::<Vec<_>>();
        let order = batch.remove_indices_swap(&[0, 2]).unwrap();
        assert_eq!(order, vec![3, 1]);
        assert_eq!(batch.len(), 2);
        for (game, &original_row) in batch.games.iter().zip(&order) {
            assert_eq!(game.wall, original[original_row]);
        }

        let before = batch.games[0].wall;
        assert_eq!(
            batch.remove_indices_swap(&[1, 0]),
            Err(GameError::BatchIndex)
        );
        assert_eq!(
            batch.remove_indices_swap(&[0, 0]),
            Err(GameError::BatchIndex)
        );
        assert_eq!(batch.remove_indices_swap(&[2]), Err(GameError::BatchIndex));
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.games[0].wall, before);
    }

    #[test]
    fn oracle_accessor_does_not_mutate_partial_observation() {
        let game = Game::new(31);
        let viewer = game.decision().unwrap().actor;
        let mut before_tile = [0; TILE_OBSERVATION_WIDTH];
        let mut before_melds = [0; MELD_OBSERVATION_WIDTH];
        let mut before_river = [0; RIVER_OBSERVATION_WIDTH];
        let mut before_meta = [0; META_OBSERVATION_WIDTH];
        game.observation_into(
            viewer,
            &mut before_tile,
            &mut before_melds,
            &mut before_river,
            &mut before_meta,
        )
        .unwrap();
        let mut oracle = [0; ORACLE_TILE_COUNT_WIDTH];
        game.oracle_tile_counts_into(&mut oracle).unwrap();
        assert_eq!(
            oracle[..PLAYER_COUNT * TILE_KIND_COUNT]
                .iter()
                .map(|&count| count as usize)
                .sum::<usize>()
                + oracle[(ORACLE_TILE_COUNT_PLANES - 1) * TILE_KIND_COUNT..]
                    .iter()
                    .map(|&count| count as usize)
                    .sum::<usize>(),
            WALL_TILE_COUNT
        );

        let mut after_tile = [0; TILE_OBSERVATION_WIDTH];
        let mut after_melds = [0; MELD_OBSERVATION_WIDTH];
        let mut after_river = [0; RIVER_OBSERVATION_WIDTH];
        let mut after_meta = [0; META_OBSERVATION_WIDTH];
        game.observation_into(
            viewer,
            &mut after_tile,
            &mut after_melds,
            &mut after_river,
            &mut after_meta,
        )
        .unwrap();
        assert_eq!(after_tile, before_tile);
        assert_eq!(after_melds, before_melds);
        assert_eq!(after_river, before_river);
        assert_eq!(after_meta, before_meta);
    }

    #[test]
    fn scores_are_zero_sum() {
        let game = Game::new(1);
        assert_eq!(
            Seat::ALL.iter().map(|&seat| game.score(seat)).sum::<i64>(),
            40_000
        );
    }

    fn exchange_action_for_test(game: &Game, seat: Seat) -> Action {
        Action::SelectExchangeTile(first_three_same_suit(game, seat)[0])
    }

    #[test]
    fn long_games_preserve_bloodflow_state_invariants() {
        for seed in 0..1_000 {
            let mut game = Game::new(seed);
            for step in 0..512 {
                let Some(action_id) = game.simple_rule_action() else {
                    break;
                };
                game.step_id(action_id)
                    .expect("simple policy action is legal");
                for (index, &seat) in Seat::ALL.iter().enumerate() {
                    let player = &game.players[index];
                    for tile in 0..TILE_KIND_COUNT {
                        assert!(
                            player.win_base[tile] <= player.locked[tile]
                                && player.locked[tile] <= player.concealed[tile],
                            "seed {seed}, step {step}, player {index}, tile {tile}"
                        );
                    }
                    if !player.has_won {
                        assert_eq!(player.win_base, [0; TILE_KIND_COUNT]);
                        assert_eq!(player.locked, [0; TILE_KIND_COUNT]);
                    }
                    assert_eq!(
                        player.kong_forbidden_mask >> TILE_KIND_COUNT,
                        0,
                        "seed {seed}, step {step}, player {index}"
                    );
                    assert_eq!(
                        game.public_win_tiles(seat).iter().sum::<u8>(),
                        player.win_count.min(u32::from(u8::MAX)) as u8,
                        "seed {seed}, step {step}, player {index}"
                    );

                    let stable_len = 13_usize.saturating_sub(3 * player.meld_count as usize);
                    let extra_tile = match game.stage {
                        Stage::Exchange { .. } | Stage::ChooseMissing { .. } => {
                            usize::from(seat == game.dealer)
                        }
                        Stage::Turn { actor, .. } => usize::from(seat == actor),
                        Stage::HuResponse {
                            source,
                            kind: ReactionKind::AddedKong,
                            ..
                        } => usize::from(seat == source),
                        Stage::HuResponse {
                            kind: ReactionKind::Discard { .. },
                            ..
                        }
                        | Stage::MeldResponse { .. }
                        | Stage::Finished => 0,
                    };
                    assert_eq!(
                        player
                            .evaluation_counts()
                            .expect("blood-flow partitions stay consistent")
                            .iter()
                            .map(|&count| usize::from(count))
                            .sum::<usize>(),
                        stable_len + extra_tile,
                        "seed {seed}, step {step}, player {index}"
                    );
                }
            }
            assert_eq!(game.phase(), Phase::Finished, "seed {seed} did not finish");
        }
    }
}
