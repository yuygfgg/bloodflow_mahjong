use crate::action::{
    Action, ActionId, ActionMask, LEGAL_ACTION_MASK_WORDS, added_kong_offset,
    concealed_kong_offset, discard_offset, exchange_offset,
};
use crate::hand::{
    SHANTEN_TERMINAL, ShantenAnalysis, WinEvaluation, WinFlags, analyze_shanten, evaluate_max_wait,
    evaluate_win, is_winning,
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
pub const ENGINE_RULES_VERSION: u32 = 1;
pub(crate) const PARALLEL_BATCH_THRESHOLD: usize = 64;
pub const STEP_RECORD_WIDTH: usize = 12;
/// Number of `i32` fields in one event-stream record.
pub const EVENT_RECORD_WIDTH: usize = 8;
/// Retained event history per environment. Older events are overwritten.
pub const EVENT_HISTORY_CAPACITY: usize = 512;
/// Ten tile-count channels of 27 tile kinds each.
pub const TILE_OBSERVATION_WIDTH: usize = 10 * TILE_KIND_COUNT;
/// Four relative players, four meld slots, and three fields per meld.
pub const MELD_OBSERVATION_WIDTH: usize = PLAYER_COUNT * 4 * 3;
/// Up to 108 chronological discards with tile and relative owner fields.
pub const RIVER_OBSERVATION_WIDTH: usize = WALL_TILE_COUNT * 2;
/// Scalar state plus four relative-player feature groups.
pub const META_OBSERVATION_WIDTH: usize = 34;
/// Training-only perfect-information tile-count planes: four concealed hands,
/// four locked subsets, and one unordered remaining-wall histogram.
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
}

impl EventKind {
    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// Event flags shared by the fixed-width stream schema.
pub const EVENT_FLAG_REPLACEMENT_DRAW: u8 = 1 << 0;
pub const EVENT_FLAG_LAST_WALL_TILE: u8 = 1 << 1;
pub const EVENT_FLAG_AFTER_KONG: u8 = 1 << 2;
pub const EVENT_FLAG_OPENING_DISCARD: u8 = 1 << 3;
pub const EVENT_FLAG_SELF_DRAW: u8 = 1 << 4;
pub const EVENT_FLAG_ROB_KONG: u8 = 1 << 5;
pub const EVENT_FLAG_HEAVENLY: u8 = 1 << 6;
pub const EVENT_FLAG_EARTHLY: u8 = 1 << 7;

const ALL_PLAYER_MASK: u8 = (1 << PLAYER_COUNT) - 1;

#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, Debug)]
struct Player {
    concealed: [u8; TILE_KIND_COUNT],
    locked: [u8; TILE_KIND_COUNT],
    melds: [Option<Meld>; 4],
    meld_count: u8,
    missing: Option<Suit>,
    score: i64,
    has_drawn: bool,
    has_won: bool,
    max_win_multiplier: u32,
}

impl Player {
    const fn new() -> Self {
        Self {
            concealed: [0; TILE_KIND_COUNT],
            locked: [0; TILE_KIND_COUNT],
            melds: [None; 4],
            meld_count: 0,
            missing: None,
            score: STARTING_SCORE,
            has_drawn: false,
            has_won: false,
            max_win_multiplier: 0,
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
        self.concealed[start..start + 9].iter().copied().sum()
    }

    fn unlocked_count(&self, tile: Tile) -> u8 {
        self.concealed[tile.index()].saturating_sub(self.locked[tile.index()])
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TurnOrigin {
    Initial,
    Draw {
        tile: Tile,
        after_kong: bool,
        last_wall_tile: bool,
    },
    AfterPong,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReactionKind {
    Discard {
        after_kong: bool,
        opening_discard: bool,
    },
    AddedKong,
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
        kind: ReactionKind,
    },
    MeldResponse {
        source: Seat,
        tile: Tile,
        remaining: u8,
    },
    Finished,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct HiddenHandConstraint {
    seat: Seat,
    tile_count: usize,
    forbidden_suit: Suit,
}

/// Authoritative environment state.
///
/// This type intentionally exposes perfect information for simulation,
/// testing, and replay. Policy code must use viewer-scoped observations and
/// [`StepOutcome::for_player`] instead of forwarding raw state or transitions.
#[derive(Clone, Debug)]
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
    discard_len: u8,
    // A discard remains in the river after a discard win, while the winner's
    // recorded hand also contains that same physical tile. One entry is kept
    // for every such hand reference so public tile counts can remove it.
    discard_win_references: [u8; TILE_KIND_COUNT],
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
            discard_len: 0,
            discard_win_references: [0; TILE_KIND_COUNT],
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
    /// players' non-locked concealed tiles and the live wall are sampled from
    /// one unknown pool while preserving concealed sizes and facts implied by
    /// public missing-suit discards. During exchange, players with already
    /// selected tiles are fixed so their pending exchange remains valid.
    /// During response windows all hands are fixed because the pending
    /// responder set encodes hand-dependent legality; the live wall is still
    /// independently resampled.
    pub fn resample_information_set(&self, seed: u64) -> Result<Self, GameError> {
        let viewer = self
            .decision()
            .ok_or(GameError::InformationSetUnavailable)?
            .actor;
        let mut sampled = self.clone();
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

        let mut unknown = Vec::with_capacity(self.wall_remaining() + 42);
        let mut movable_counts = [0_usize; PLAYER_COUNT];
        for seat in Seat::ALL {
            if fixed_players & seat_bit(seat) != 0 {
                continue;
            }
            let player = &mut sampled.players[seat.index()];
            for tile_index in 0..TILE_KIND_COUNT {
                let movable =
                    player.concealed[tile_index].saturating_sub(player.locked[tile_index]);
                movable_counts[seat.index()] += movable as usize;
                unknown.extend(core::iter::repeat_n(tile_index as u8, movable as usize));
                player.concealed[tile_index] = player.locked[tile_index];
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

        let constraints: Vec<_> = Seat::ALL
            .into_iter()
            .filter(|&seat| fixed_players & seat_bit(seat) == 0)
            .filter_map(|seat| {
                self.known_empty_missing_suit(seat)
                    .map(|forbidden_suit| HiddenHandConstraint {
                        seat,
                        tile_count: movable_counts[seat.index()],
                        forbidden_suit,
                    })
            })
            .collect();
        let constrained_allocations =
            sample_constrained_suit_allocations(&unknown, &constraints, &mut rng);
        let mut suit_pools: [Vec<u8>; 3] = core::array::from_fn(|_| Vec::new());
        for tile in unknown {
            suit_pools[usize::from(tile / 9)].push(tile);
        }
        for pool in &mut suit_pools {
            rng.shuffle(pool);
        }

        let mut constrained_players = 0_u8;
        for (constraint, allocation) in constraints.iter().zip(constrained_allocations) {
            constrained_players |= seat_bit(constraint.seat);
            let player = &mut sampled.players[constraint.seat.index()];
            for (suit, count) in allocation.into_iter().enumerate() {
                for _ in 0..count {
                    let tile = suit_pools[suit]
                        .pop()
                        .expect("a sampled suit allocation fits the unknown pool");
                    player.concealed[tile as usize] += 1;
                }
            }
        }

        let mut remaining: Vec<_> = suit_pools.into_iter().flatten().collect();
        rng.shuffle(&mut remaining);
        let mut cursor = 0;
        for seat in Seat::ALL {
            if constrained_players & seat_bit(seat) != 0 {
                continue;
            }
            for _ in 0..movable_counts[seat.index()] {
                let tile = remaining[cursor] as usize;
                sampled.players[seat.index()].concealed[tile] += 1;
                cursor += 1;
            }
        }
        let wall_len = self.wall_remaining();
        sampled.wall[self.wall_head as usize..self.wall_tail as usize]
            .copy_from_slice(&remaining[cursor..cursor + wall_len]);
        debug_assert_eq!(cursor + wall_len, remaining.len());
        Ok(sampled)
    }

    fn known_empty_missing_suit(&self, seat: Seat) -> Option<Suit> {
        let missing = self.players[seat.index()].missing?;
        (0..self.discard_len as usize)
            .rev()
            .find(|&index| self.discard_owners[index] == seat.as_u8())
            .and_then(|index| {
                let tile = Tile::new(self.discards[index]).expect("stored tile is valid");
                (tile.suit() != missing).then_some(missing)
            })
    }

    /// Clones the state and independently shuffles only the unseen live wall.
    ///
    /// Every hand, public event, pending decision, score, and the consumed
    /// part of the wall remains unchanged. This makes the operation suitable
    /// for rejuvenating future draws after particle-filter resampling.
    pub fn resample_live_wall(&self, seed: u64) -> Self {
        let mut sampled = self.clone();
        let mut rng = Rng::new(seed);
        rng.shuffle(&mut sampled.wall[sampled.wall_head as usize..sampled.wall_tail as usize]);
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
            Stage::Turn {
                actor,
                origin,
                can_hu,
            } => {
                let player = &self.players[actor.index()];
                legal.can_hu = can_hu;
                legal.discard_mask = self.legal_discard_mask(actor);
                // After a player has won, only a self-drawn fourth tile for
                // an existing pong may be used as an added kong.  Concealed
                // kongs are no longer legal for that player.
                if !player.has_won && player.meld_count < 4 && origin != TurnOrigin::AfterPong {
                    for index in 0..TILE_KIND_COUNT {
                        let tile = Tile::from_index_unchecked(index as u8);
                        if player.missing == Some(tile.suit()) {
                            continue;
                        }
                        if player.concealed[index] >= 4 {
                            legal.concealed_kong_mask |= 1 << index;
                        }
                    }
                }
                if origin != TurnOrigin::AfterPong {
                    for meld in player.melds.iter().flatten() {
                        if meld.kind == MeldKind::Pong
                            && player.concealed[meld.tile.index()] > 0
                            && player.missing != Some(meld.tile.suit())
                        {
                            legal.added_kong_mask |= 1 << meld.tile.index();
                        }
                    }
                }
            }
            Stage::HuResponse { .. } => {
                legal.can_hu = true;
                legal.can_pass = true;
            }
            Stage::MeldResponse { tile, .. } => {
                let player = &self.players[decision.actor.index()];
                legal.can_pong = self.can_pong(decision.actor, tile);
                legal.can_exposed_kong =
                    player.meld_count < 4 && player.concealed[tile.index()] >= 3;
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
        self.push_event(
            EventKind::Action,
            Some(decision.actor),
            None,
            None,
            0,
            action.id().index() as i32,
            i32::from(decision.phase.code()),
            seat_bit(decision.actor),
            0,
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
            Stage::HuResponse {
                source,
                tile,
                remaining,
                winners,
                kind,
            } => self.step_hu_response(source, tile, remaining, winners, kind, action),
            Stage::MeldResponse {
                source,
                tile,
                remaining,
            } => self.step_meld_response(source, tile, remaining, action),
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

    /// Highest shape multiplier from this player's completed wins.
    pub fn max_win_multiplier(&self, seat: Seat) -> u32 {
        self.players[seat.index()].max_win_multiplier
    }

    /// Returns conventional structural shanten and improving tiles for a seat.
    ///
    /// This authoritative accessor can inspect any seat, just like
    /// [`Game::concealed`]. Policy code should request only its own seat.
    pub fn hand_analysis(&self, seat: Seat) -> ShantenAnalysis {
        let player = &self.players[seat.index()];
        let (melds, len) = player.meld_buffer();
        analyze_shanten(&player.concealed, &melds[..len], player.missing)
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
    /// The result includes the viewer's concealed hand, other players' locked
    /// tiles, exposed meld contributions, and the discard river. A discarded
    /// tile claimed by one or more winners is counted once even though the
    /// authoritative winner hands retain a reference to it.
    pub(crate) fn visible_tile_counts(&self, viewer: Seat) -> [u8; TILE_KIND_COUNT] {
        let mut counts = [0_u16; TILE_KIND_COUNT];
        add_tile_counts(&mut counts, &self.players[viewer.index()].concealed);

        for seat in Seat::ALL {
            let player = &self.players[seat.index()];
            if seat != viewer {
                add_tile_counts(&mut counts, &player.locked);
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
            let known = counts[index].saturating_sub(u16::from(self.discard_win_references[index]));
            debug_assert!(known <= u16::from(TILE_COPIES));
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
            tile_obs[start..start + TILE_KIND_COUNT]
                .copy_from_slice(&self.players[seat.index()].locked);
        }
        for index in 0..self.discard_len as usize {
            let owner = Seat::new(self.discard_owners[index]).expect("stored seat is valid");
            let relative = relative_seat(viewer, owner) as usize;
            let tile = self.discards[index] as usize;
            let offset = (6 + relative) * TILE_KIND_COUNT + tile;
            tile_obs[offset] = tile_obs[offset].saturating_add(1);
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
                    },
                ..
            } => i32::from(u8::from(after_kong)) << 1 | i32::from(u8::from(opening_discard)) << 2,
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
    /// Planes `0..4` are concealed hands, `4..8` are their locked subsets,
    /// and plane `8` is an unordered histogram of the live wall.
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
        match action {
            Action::Hu if can_hu => self.resolve_self_draw(actor, origin),
            Action::Discard(tile) if self.legal_discard_mask(actor) & (1 << tile.index()) != 0 => {
                self.discard(actor, tile, origin)
            }
            Action::ConcealedKong(tile)
                if self
                    .legal_actions()
                    .is_some_and(|legal| legal.concealed_kong_mask & (1 << tile.index()) != 0) =>
            {
                self.concealed_kong(actor, tile)
            }
            Action::AddedKong(tile)
                if self
                    .legal_actions()
                    .is_some_and(|legal| legal.added_kong_mask & (1 << tile.index()) != 0) =>
            {
                self.propose_added_kong(actor, tile)
            }
            _ => Err(GameError::InvalidAction),
        }
    }

    fn step_hu_response(
        &mut self,
        source: Seat,
        tile: Tile,
        mut remaining: u8,
        mut winners: u8,
        kind: ReactionKind,
        action: Action,
    ) -> Result<(), GameError> {
        let actor = first_seat_in_mask(source, remaining).ok_or(GameError::InvalidAction)?;
        match action {
            Action::Hu => winners |= seat_bit(actor),
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
                kind,
            };
            return Ok(());
        }

        if winners != 0 {
            self.resolve_discard_wins(source, tile, winners, kind);
        } else {
            match kind {
                ReactionKind::Discard { .. } => self.begin_meld_responses(source, tile),
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
        action: Action,
    ) -> Result<(), GameError> {
        let actor = first_seat_in_mask(source, remaining).ok_or(GameError::InvalidAction)?;
        let player = &self.players[actor.index()];
        let can_pong = self.can_pong(actor, tile);
        let can_kong = player.meld_count < 4 && player.concealed[tile.index()] >= 3;
        match action {
            Action::Pong if can_pong => return self.pong(actor, source, tile),
            Action::ExposedKong if can_kong => return self.exposed_kong(actor, source, tile),
            Action::Pass => {}
            _ => return Err(GameError::InvalidAction),
        }
        remaining &= !seat_bit(actor);
        if remaining == 0 {
            self.begin_normal_turn(source.next());
        } else {
            self.stage = Stage::MeldResponse {
                source,
                tile,
                remaining,
            };
        }
        Ok(())
    }

    fn legal_discard_mask(&self, actor: Seat) -> u32 {
        let player = &self.players[actor.index()];
        let mut unlocked = 0_u32;
        for index in 0..TILE_KIND_COUNT {
            if player.concealed[index] > player.locked[index] {
                unlocked |= 1 << index;
            }
        }
        if player.missing_count() == 0 {
            return unlocked;
        }
        let missing_mask = player.missing.map_or(0, Suit::mask);
        let forced = unlocked & missing_mask;
        if forced == 0 { unlocked } else { forced }
    }

    fn discard(&mut self, actor: Seat, tile: Tile, origin: TurnOrigin) -> Result<(), GameError> {
        self.players[actor.index()].concealed[tile.index()] -= 1;
        let discard_index = self.discard_len as usize;
        self.discards[discard_index] = tile.as_u8();
        self.discard_owners[discard_index] = actor.as_u8();
        self.discard_len += 1;
        self.transition_discard = Some(DiscardEvent {
            player: actor,
            tile,
        });

        let after_kong = matches!(
            origin,
            TurnOrigin::Draw {
                after_kong: true,
                ..
            }
        );
        let opening_discard = actor == self.dealer && self.discard_len == 1;
        let mut discard_flags = 0;
        if after_kong {
            discard_flags |= EVENT_FLAG_AFTER_KONG;
        }
        if opening_discard {
            discard_flags |= EVENT_FLAG_OPENING_DISCARD;
        }
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
        let mut candidates = 0_u8;
        for seat in Seat::ALL {
            if seat != actor && self.can_player_win_with_added_tile(seat, tile) {
                candidates |= seat_bit(seat);
            }
        }
        if candidates != 0 {
            self.stage = Stage::HuResponse {
                source: actor,
                tile,
                remaining: candidates,
                winners: 0,
                kind: ReactionKind::Discard {
                    after_kong,
                    opening_discard,
                },
            };
        } else {
            self.begin_meld_responses(actor, tile);
        }
        Ok(())
    }

    fn can_pong(&self, actor: Seat, tile: Tile) -> bool {
        let player = &self.players[actor.index()];
        player.meld_count < 4
            && player.unlocked_count(tile) >= 2
            && player
                .concealed
                .iter()
                .zip(player.locked.iter())
                .map(|(&concealed, &locked)| concealed.saturating_sub(locked) as usize)
                .sum::<usize>()
                >= 3
    }

    fn begin_meld_responses(&mut self, source: Seat, tile: Tile) {
        let mut candidates = 0_u8;
        for seat in Seat::ALL {
            if seat == source {
                continue;
            }
            let player = &self.players[seat.index()];
            if player.missing == Some(tile.suit()) || player.meld_count >= 4 {
                continue;
            }
            // A pong must leave one unlocked tile for the immediate discard
            // that follows it.  Keep the response window aligned with the
            // same predicate used by `legal_actions` and `step_meld_response`.
            if self.can_pong(seat, tile) || player.concealed[tile.index()] >= 3 {
                candidates |= seat_bit(seat);
            }
        }
        if candidates == 0 {
            self.begin_normal_turn(source.next());
        } else {
            self.stage = Stage::MeldResponse {
                source,
                tile,
                remaining: candidates,
            };
        }
    }

    fn pong(&mut self, actor: Seat, source: Seat, tile: Tile) -> Result<(), GameError> {
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
        self.remove_for_meld(actor, tile, 3, true);
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
        self.transfer(source, actor, SCORE_UNIT);
        if self.finish_if_three_zero() {
            return Ok(());
        }
        self.begin_supplement_turn(actor);
        Ok(())
    }

    fn concealed_kong(&mut self, actor: Seat, tile: Tile) -> Result<(), GameError> {
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
        if self.finish_if_three_zero() {
            return Ok(());
        }
        self.begin_supplement_turn(actor);
        Ok(())
    }

    fn propose_added_kong(&mut self, actor: Seat, tile: Tile) -> Result<(), GameError> {
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
        if !self.finish_if_three_zero() {
            self.begin_supplement_turn(actor);
        }
    }

    fn resolve_self_draw(&mut self, actor: Seat, origin: TurnOrigin) -> Result<(), GameError> {
        let (required, flags) = match origin {
            TurnOrigin::Initial => (
                None,
                WinFlags {
                    heavenly: true,
                    ..WinFlags::NONE
                },
            ),
            TurnOrigin::Draw {
                tile,
                after_kong,
                last_wall_tile,
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
        let evaluation = self
            .evaluate_player(actor, required, flags)
            .ok_or(GameError::InvalidAction)?;
        self.apply_win(actor, required, evaluation);
        self.push_event(
            EventKind::Hu,
            Some(actor),
            None,
            required,
            win_event_flags(flags, true),
            evaluation.multiplier as i32,
            evaluation.patterns.bits() as i32,
            ALL_PLAYER_MASK,
            ALL_PLAYER_MASK,
        );
        let payment = SCORE_UNIT * i64::from(evaluation.multiplier) * 2;
        for payer in seats_after(actor) {
            self.transfer(payer, actor, payment);
        }
        if !self.finish_if_three_zero() {
            self.begin_normal_turn(actor.next());
        }
        Ok(())
    }

    fn resolve_discard_wins(&mut self, source: Seat, tile: Tile, winners: u8, kind: ReactionKind) {
        if kind == ReactionKind::AddedKong {
            self.remove_lost_tile_preserving_locks(source, tile);
        }

        let mut last_winner = source;
        for winner in seats_in_mask_after(source, winners) {
            if matches!(kind, ReactionKind::Discard { .. }) {
                self.discard_win_references[tile.index()] =
                    self.discard_win_references[tile.index()].saturating_add(1);
            }
            self.players[winner.index()].concealed[tile.index()] =
                self.players[winner.index()].concealed[tile.index()].saturating_add(1);
            let flags = match kind {
                ReactionKind::Discard {
                    after_kong,
                    opening_discard,
                } => WinFlags {
                    after_kong_discard: after_kong,
                    earthly: opening_discard
                        && winner != self.dealer
                        && !self.players[winner.index()].has_drawn,
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
            self.apply_win(winner, Some(tile), evaluation);
            self.push_event(
                EventKind::Hu,
                Some(winner),
                Some(source),
                Some(tile),
                win_event_flags(flags, false),
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
            last_winner = winner;
        }
        if !self.finish_if_three_zero() {
            self.begin_normal_turn(last_winner.next());
        }
    }

    fn apply_win(&mut self, actor: Seat, required: Option<Tile>, evaluation: WinEvaluation) {
        let player = &mut self.players[actor.index()];
        for index in 0..TILE_KIND_COUNT {
            let selected = evaluation.used[index];
            if selected == 0 {
                continue;
            }
            let target = if required.is_some_and(|tile| tile.index() == index) {
                selected.max(player.locked[index].saturating_add(1))
            } else {
                selected.max(player.locked[index])
            };
            player.locked[index] = target.min(player.concealed[index]);
        }
        player.has_won = true;
        player.max_win_multiplier = player.max_win_multiplier.max(evaluation.shape_multiplier);
    }

    fn can_player_win(&self, actor: Seat, required: Option<Tile>) -> bool {
        let player = &self.players[actor.index()];
        if player.missing_count() != 0 {
            return false;
        }
        let (melds, len) = player.meld_buffer();
        is_winning(&player.concealed, &melds[..len], required)
    }

    pub(crate) fn can_player_win_with_added_tile(&self, actor: Seat, tile: Tile) -> bool {
        let player = &self.players[actor.index()];
        if player.missing_count() != 0 || player.missing == Some(tile.suit()) {
            return false;
        }
        let mut counts = player.concealed;
        counts[tile.index()] = counts[tile.index()].saturating_add(1);
        let (melds, len) = player.meld_buffer();
        is_winning(&counts, &melds[..len], Some(tile))
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
        evaluate_win(&player.concealed, &melds[..len], required, flags)
    }

    fn begin_normal_turn(&mut self, actor: Seat) {
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
        self.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::Draw {
                tile,
                after_kong: false,
                last_wall_tile,
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
        self.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::Draw {
                tile,
                after_kong: true,
                last_wall_tile,
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

    fn remove_for_meld(&mut self, actor: Seat, tile: Tile, amount: u8, allow_locked: bool) {
        let player = &mut self.players[actor.index()];
        let index = tile.index();
        if allow_locked {
            let removed_locked = player.locked[index].min(amount);
            player.locked[index] -= removed_locked;
        } else {
            debug_assert!(player.unlocked_count(tile) >= amount);
        }
        player.concealed[index] -= amount;
    }

    fn remove_lost_tile_preserving_locks(&mut self, actor: Seat, tile: Tile) {
        let player = &mut self.players[actor.index()];
        let index = tile.index();
        player.concealed[index] -= 1;
        player.locked[index] = player.locked[index].min(player.concealed[index]);
    }

    fn transfer(&mut self, payer: Seat, payee: Seat, requested: i64) -> i64 {
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

    fn finish_if_three_zero(&mut self) -> bool {
        if self
            .players
            .iter()
            .filter(|player| player.score == 0)
            .count()
            >= 3
        {
            self.stage = Stage::Finished;
            true
        } else {
            false
        }
    }

    fn finish_wall_game(&mut self) {
        let mut flower_pig = [false; PLAYER_COUNT];
        let mut ready = [false; PLAYER_COUNT];
        let mut max_multiplier = [0_u32; PLAYER_COUNT];

        for seat in Seat::ALL {
            flower_pig[seat.index()] = self.suit_count(seat) == 3;
            if !flower_pig[seat.index()] {
                max_multiplier[seat.index()] = self.max_wait_multiplier(seat);
                ready[seat.index()] = max_multiplier[seat.index()] > 0;
            }
        }

        for payer in Seat::ALL {
            if !flower_pig[payer.index()] {
                continue;
            }
            for payee in seats_after(payer) {
                if !flower_pig[payee.index()]
                    && (ready[payee.index()] || self.players[payee.index()].has_won)
                {
                    self.transfer(payer, payee, SCORE_UNIT * 10);
                }
            }
            if self.finish_if_three_zero() {
                return;
            }
        }

        for payer in Seat::ALL {
            if flower_pig[payer.index()] || ready[payer.index()] {
                continue;
            }
            for payee in seats_after(payer) {
                if !flower_pig[payee.index()]
                    && (ready[payee.index()] || self.players[payee.index()].has_won)
                {
                    let multiplier = max_multiplier[payee.index()]
                        .max(self.players[payee.index()].max_win_multiplier)
                        .max(1);
                    self.transfer(payer, payee, SCORE_UNIT * i64::from(multiplier));
                }
            }
            if self.finish_if_three_zero() {
                return;
            }
        }
        self.stage = Stage::Finished;
    }

    fn max_wait_multiplier(&self, actor: Seat) -> u32 {
        let player = &self.players[actor.index()];
        let (melds, len) = player.meld_buffer();
        evaluate_max_wait(&player.concealed, &melds[..len], player.missing)
            .map_or(0, |wait| wait.evaluation.multiplier)
    }

    fn suit_count(&self, actor: Seat) -> usize {
        let player = &self.players[actor.index()];
        Suit::ALL
            .iter()
            .filter(|&&suit| {
                let start = suit as usize * 9;
                player.concealed[start..start + 9]
                    .iter()
                    .any(|&count| count > 0)
                    || player
                        .melds
                        .iter()
                        .flatten()
                        .any(|meld| meld.tile.suit() == suit)
            })
            .count()
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
    /// Tile channels are actor concealed, actor exchange selection, four locked
    /// hands, and four per-owner discard counts. Meld records are
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
    /// discard, and bit 2 for the dealer's opening discard.
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

fn sample_constrained_suit_allocations(
    unknown: &[u8],
    constraints: &[HiddenHandConstraint],
    rng: &mut Rng,
) -> Vec<[usize; 3]> {
    let mut remaining = [0_usize; 3];
    for &tile in unknown {
        remaining[usize::from(tile / 9)] += 1;
    }

    let mut sampled = Vec::with_capacity(constraints.len());
    for (index, constraint) in constraints.iter().enumerate() {
        let options =
            allowed_suit_allocations(remaining, constraint.tile_count, constraint.forbidden_suit);
        let weights: Vec<_> = options
            .iter()
            .map(|&allocation| {
                allocation_weight(remaining, allocation)
                    * constrained_completion_weight(
                        constraints,
                        index + 1,
                        subtract_allocation(remaining, allocation),
                    )
            })
            .collect();
        let total: f64 = weights.iter().sum();
        assert!(
            total.is_finite() && total > 0.0,
            "public hand constraints are feasible"
        );

        let mut draw = rng.unit_f64() * total;
        let selected = weights
            .iter()
            .position(|&weight| {
                if draw < weight {
                    true
                } else {
                    draw -= weight;
                    false
                }
            })
            .unwrap_or(weights.len() - 1);
        let allocation = options[selected];
        remaining = subtract_allocation(remaining, allocation);
        sampled.push(allocation);
    }
    sampled
}

fn constrained_completion_weight(
    constraints: &[HiddenHandConstraint],
    index: usize,
    remaining: [usize; 3],
) -> f64 {
    let Some(constraint) = constraints.get(index) else {
        return 1.0;
    };
    allowed_suit_allocations(remaining, constraint.tile_count, constraint.forbidden_suit)
        .into_iter()
        .map(|allocation| {
            allocation_weight(remaining, allocation)
                * constrained_completion_weight(
                    constraints,
                    index + 1,
                    subtract_allocation(remaining, allocation),
                )
        })
        .sum()
}

fn allowed_suit_allocations(
    remaining: [usize; 3],
    tile_count: usize,
    forbidden_suit: Suit,
) -> Vec<[usize; 3]> {
    let allowed = match forbidden_suit {
        Suit::Characters => [Suit::Bamboo as usize, Suit::Dots as usize],
        Suit::Bamboo => [Suit::Characters as usize, Suit::Dots as usize],
        Suit::Dots => [Suit::Characters as usize, Suit::Bamboo as usize],
    };
    let first = allowed[0];
    let second = allowed[1];
    let minimum_first = tile_count.saturating_sub(remaining[second]);
    let maximum_first = tile_count.min(remaining[first]);
    (minimum_first..=maximum_first)
        .map(|first_count| {
            let mut allocation = [0_usize; 3];
            allocation[first] = first_count;
            allocation[second] = tile_count - first_count;
            allocation
        })
        .collect()
}

fn allocation_weight(remaining: [usize; 3], allocation: [usize; 3]) -> f64 {
    remaining
        .into_iter()
        .zip(allocation)
        .map(|(available, selected)| combination(available, selected))
        .product()
}

fn subtract_allocation(remaining: [usize; 3], allocation: [usize; 3]) -> [usize; 3] {
    core::array::from_fn(|suit| remaining[suit] - allocation[suit])
}

fn combination(total: usize, selected: usize) -> f64 {
    if selected > total {
        return 0.0;
    }
    let selected = selected.min(total - selected);
    (0..selected).fold(1.0, |value, index| {
        value * (total - index) as f64 / (index + 1) as f64
    })
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
        for (slot, tile) in game.wall.iter_mut().zip(tiles) {
            *slot = tile.as_u8();
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
    fn concealed_kong_is_rejected_after_hu() {
        let mut game = constructed_game();
        let actor = Seat::ALL[1];
        let locked_quad = tile(Suit::Characters, 1);
        let fresh_quad = tile(Suit::Characters, 2);

        // Both shapes can appear in a post-win holding: the first is an
        // all-locked seven-pairs quad, while the second is three locked tiles
        // plus one newly drawn tile. Neither may be declared as a concealed
        // kong after the player has already won; only an added kong remains.
        let player = &mut game.players[actor.index()];
        player.missing = None;
        player.concealed[locked_quad.index()] = 4;
        player.locked[locked_quad.index()] = 4;
        player.concealed[fresh_quad.index()] = 4;
        player.locked[fresh_quad.index()] = 3;
        player.has_won = true;
        game.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::Draw {
                tile: fresh_quad,
                after_kong: false,
                last_wall_tile: false,
            },
            can_hu: false,
        };

        let legal = game.legal_actions().unwrap();
        assert_eq!(legal.concealed_kong_mask, 0);
        assert_eq!(
            game.step(Action::ConcealedKong(locked_quad)),
            Err(GameError::InvalidAction)
        );
        assert_eq!(
            game.step(Action::ConcealedKong(fresh_quad)),
            Err(GameError::InvalidAction)
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
    fn heavenly_hu_pays_locks_and_advances_to_the_dealers_next_seat() {
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

        assert_eq!(outcome.score_delta, [19_200, -6_400, -6_400, -6_400]);
        assert_eq!(game.score(Seat::EAST), 29_200);
        assert_eq!(game.locked(Seat::EAST), game.concealed(Seat::EAST));
        assert!(game.has_won(Seat::EAST));
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
    fn dealers_opening_discard_gives_an_undrawn_non_dealer_earthly_hu() {
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
        assert_eq!(hu.score_delta, [-3_200, 3_200, 0, 0]);
        assert_eq!(game.locked(winner), game.concealed(winner));
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

        assert_eq!(game.discard_win_references[winning_tile.index()], 2);
        assert_eq!(game.locked(first_winner)[winning_tile.index()], 2);
        assert_eq!(game.locked(second_winner)[winning_tile.index()], 2);
        assert_eq!(game.visible_tile_counts(viewer)[winning_tile.index()], 3);
    }

    #[test]
    fn robbed_added_kong_does_not_create_a_discard_win_reference() {
        let mut game = constructed_game();
        let winner = Seat::ALL[1];
        let winning_tile = make_plain_wait(&mut game.players[winner.index()]);
        add_tile(&mut game.players[Seat::EAST.index()], winning_tile, 1);

        game.resolve_discard_wins(
            Seat::EAST,
            winning_tile,
            seat_bit(winner),
            ReactionKind::AddedKong,
        );

        assert_eq!(game.discard_win_references[winning_tile.index()], 0);
        assert_eq!(
            game.visible_tile_counts(Seat::ALL[2])[winning_tile.index()],
            2
        );
    }

    #[test]
    fn multiple_hu_suppresses_melds_and_resumes_after_the_last_winner() {
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
                actor: meld_candidate,
                phase: Phase::Turn,
            })
        );
        assert_eq!(
            resolved.draw,
            Some(DrawEvent {
                player: meld_candidate,
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
        };

        let outcome = game.step(Action::ExposedKong).unwrap();

        assert_eq!(outcome.score_delta, [-100, 100, 0, 0]);
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
    fn added_kong_charges_each_opponent_and_empty_tail_finishes() {
        let mut game = constructed_game();
        for seat in Seat::ALL {
            make_flower_pig(&mut game.players[seat.index()]);
        }
        let actor = Seat::ALL[1];
        let kong_tile = tile(Suit::Characters, 5);
        add_tile(&mut game.players[actor.index()], kong_tile, 1);
        assert!(game.players[actor.index()].add_meld(Meld {
            tile: kong_tile,
            kind: MeldKind::Pong,
            source: Seat::EAST,
        }));
        game.stage = Stage::Turn {
            actor,
            origin: TurnOrigin::Initial,
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

        assert_eq!(outcome.score_delta, [-200, 600, -200, -200]);
        assert!(outcome.terminal);
        assert_eq!(outcome.draw, None);
        assert_eq!(game.phase(), Phase::Finished);
        assert_eq!(game.meld(actor, 0).unwrap().kind, MeldKind::ConcealedKong);
    }

    #[test]
    fn repeated_hu_reuses_old_locks_and_unions_in_the_new_winning_tile() {
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
        assert_eq!(
            first_locks
                .iter()
                .map(|&count| count as usize)
                .sum::<usize>(),
            14
        );
        assert_eq!(game.locked(actor), game.concealed(actor));

        let second_winning_tile = tile(Suit::Bamboo, 9);
        let following_draw = tile(Suit::Characters, 8);
        set_wall(&mut game, &[second_winning_tile, following_draw]);
        game.begin_normal_turn(actor);
        assert!(game.legal_actions().unwrap().can_hu);
        assert_ne!(
            game.legal_actions().unwrap().discard_mask & (1 << second_winning_tile.index()),
            0
        );

        let second_hu = game.step(Action::Hu).unwrap();
        assert_eq!(
            game.locked(actor)
                .iter()
                .map(|&count| count as usize)
                .sum::<usize>(),
            15
        );
        for (before, after) in first_locks.iter().zip(game.locked(actor)) {
            assert!(after >= before);
        }
        assert_eq!(game.locked(actor)[second_winning_tile.index()], 3);
        assert_eq!(game.locked(actor), game.concealed(actor));
        assert_eq!(second_hu.draw.unwrap().tile, following_draw);
        assert_eq!(second_hu.next.unwrap().actor, actor.next());
    }

    #[test]
    fn flower_pig_settlement_precedes_and_can_fund_dajiao_payments() {
        let mut game = constructed_game();
        make_flower_pig(&mut game.players[Seat::EAST.index()]);
        game.players[Seat::ALL[1].index()].has_won = true;
        game.players[Seat::ALL[1].index()].max_win_multiplier = 4;
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

        // East first pays its only 1,000 points to the first eligible seat.
        // South then uses 200 of that receipt for its two dajiao payments.
        assert_eq!(
            Seat::ALL.map(|seat| game.score(seat)),
            [0, 800, 10_100, 29_100]
        );
        assert_eq!(game.phase(), Phase::Finished);
        assert_eq!(
            Seat::ALL.iter().map(|&seat| game.score(seat)).sum::<i64>(),
            40_000
        );
    }

    #[test]
    fn robbed_added_kong_preserves_an_equivalent_locked_tile() {
        let mut game = Game::new(77);
        let tile = Tile::from_index_unchecked(4);
        game.players[Seat::EAST.index()].concealed[tile.index()] = 2;
        game.players[Seat::EAST.index()].locked[tile.index()] = 1;

        game.remove_lost_tile_preserving_locks(Seat::EAST, tile);
        assert_eq!(game.players[Seat::EAST.index()].concealed[tile.index()], 1);
        assert_eq!(game.players[Seat::EAST.index()].locked[tile.index()], 1);

        game.remove_lost_tile_preserving_locks(Seat::EAST, tile);
        assert_eq!(game.players[Seat::EAST.index()].concealed[tile.index()], 0);
        assert_eq!(game.players[Seat::EAST.index()].locked[tile.index()], 0);
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
                },
                0b110,
            ),
            (ReactionKind::AddedKong, 0b001),
        ] {
            let mut game = constructed_game();
            game.stage = Stage::HuResponse {
                source,
                tile: pending_tile,
                remaining: seat_bit(actor),
                winners: seat_bit(Seat::ALL[1]),
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
    fn information_set_resampling_respects_proven_empty_missing_suits() {
        let mut game = Game::new(913);
        let constrained = loop {
            if game.phase() == Phase::Turn {
                let actor = game.decision().unwrap().actor;
                if let Some(constraint) = Seat::ALL
                    .into_iter()
                    .filter(|&seat| seat != actor)
                    .find_map(|seat| game.known_empty_missing_suit(seat).map(|suit| (seat, suit)))
                {
                    break constraint;
                }
            }
            let action = game
                .simple_rule_action()
                .expect("the seeded game reaches a constrained turn");
            game.step_id(action).unwrap();
        };
        let (seat, forbidden_suit) = constrained;
        let start = forbidden_suit as usize * 9;

        for seed in 0..128 {
            let sampled = game.resample_information_set(seed).unwrap();
            assert_eq!(sampled.concealed(seat)[start..start + 9], [0; 9]);
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
        assert_eq!(first.wall, repeated.wall);
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
}
