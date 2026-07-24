use crate::{Suit, Tile};

/// Number of logits expected from a policy head.
///
/// The layout is stable and deliberately gives each action one global meaning:
///
/// - `0..27`: select one tile during the three-tile exchange
/// - `27..30`: choose the missing suit
/// - `30..57`: discard a tile
/// - `57`: hu
/// - `58`: pong
/// - `59`: exposed kong
/// - `60..87`: concealed kong by tile
/// - `87..114`: added kong by tile
/// - `114`: pass in a response window
pub const ACTION_SPACE_SIZE: usize = 115;

/// Number of packed `u64` words used by one environment's legal-action mask.
pub const LEGAL_ACTION_MASK_WORDS: usize = 2;

pub const ACTION_EXCHANGE_TILE_OFFSET: usize = 0;
pub const ACTION_CHOOSE_MISSING_OFFSET: usize = 27;
pub const ACTION_DISCARD_OFFSET: usize = 30;
pub const ACTION_HU: usize = 57;
pub const ACTION_PONG: usize = 58;
pub const ACTION_EXPOSED_KONG: usize = 59;
pub const ACTION_CONCEALED_KONG_OFFSET: usize = 60;
pub const ACTION_ADDED_KONG_OFFSET: usize = 87;
pub const ACTION_PASS: usize = 114;

const EXCHANGE_OFFSET: u8 = ACTION_EXCHANGE_TILE_OFFSET as u8;
const MISSING_OFFSET: u8 = ACTION_CHOOSE_MISSING_OFFSET as u8;
const DISCARD_OFFSET: u8 = ACTION_DISCARD_OFFSET as u8;
const HU_INDEX: u8 = ACTION_HU as u8;
const PONG_INDEX: u8 = ACTION_PONG as u8;
const EXPOSED_KONG_INDEX: u8 = ACTION_EXPOSED_KONG as u8;
const CONCEALED_KONG_OFFSET: u8 = ACTION_CONCEALED_KONG_OFFSET as u8;
const ADDED_KONG_OFFSET: u8 = ACTION_ADDED_KONG_OFFSET as u8;
const PASS_INDEX: u8 = ACTION_PASS as u8;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Action {
    SelectExchangeTile(Tile),
    ChooseMissing(Suit),
    Discard(Tile),
    Hu,
    Pong,
    ExposedKong,
    ConcealedKong(Tile),
    AddedKong(Tile),
    Pass,
}

impl Action {
    pub const fn id(self) -> ActionId {
        match self {
            Self::SelectExchangeTile(tile) => ActionId(EXCHANGE_OFFSET + tile.as_u8()),
            Self::ChooseMissing(suit) => ActionId(MISSING_OFFSET + suit as u8),
            Self::Discard(tile) => ActionId(DISCARD_OFFSET + tile.as_u8()),
            Self::Hu => ActionId::HU,
            Self::Pong => ActionId::PONG,
            Self::ExposedKong => ActionId::EXPOSED_KONG,
            Self::ConcealedKong(tile) => ActionId(CONCEALED_KONG_OFFSET + tile.as_u8()),
            Self::AddedKong(tile) => ActionId(ADDED_KONG_OFFSET + tile.as_u8()),
            Self::Pass => ActionId::PASS,
        }
    }
}

/// A validated index into the fixed policy action space.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct ActionId(u8);

impl ActionId {
    pub const HU: Self = Self(HU_INDEX);
    pub const PONG: Self = Self(PONG_INDEX);
    pub const EXPOSED_KONG: Self = Self(EXPOSED_KONG_INDEX);
    pub const PASS: Self = Self(PASS_INDEX);

    pub const fn new(index: usize) -> Option<Self> {
        if index < ACTION_SPACE_SIZE {
            Some(Self(index as u8))
        } else {
            None
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn select_exchange_tile(tile: Tile) -> Self {
        Self(EXCHANGE_OFFSET + tile.as_u8())
    }

    pub const fn choose_missing(suit: Suit) -> Self {
        Self(MISSING_OFFSET + suit as u8)
    }

    pub const fn discard(tile: Tile) -> Self {
        Self(DISCARD_OFFSET + tile.as_u8())
    }

    pub const fn concealed_kong(tile: Tile) -> Self {
        Self(CONCEALED_KONG_OFFSET + tile.as_u8())
    }

    pub const fn added_kong(tile: Tile) -> Self {
        Self(ADDED_KONG_OFFSET + tile.as_u8())
    }

    pub const fn action(self) -> Action {
        match self.0 {
            EXCHANGE_OFFSET..=26 => {
                Action::SelectExchangeTile(Tile::from_index_unchecked(self.0 - EXCHANGE_OFFSET))
            }
            MISSING_OFFSET => Action::ChooseMissing(Suit::Characters),
            28 => Action::ChooseMissing(Suit::Bamboo),
            29 => Action::ChooseMissing(Suit::Dots),
            DISCARD_OFFSET..=56 => {
                Action::Discard(Tile::from_index_unchecked(self.0 - DISCARD_OFFSET))
            }
            HU_INDEX => Action::Hu,
            PONG_INDEX => Action::Pong,
            EXPOSED_KONG_INDEX => Action::ExposedKong,
            CONCEALED_KONG_OFFSET..=86 => {
                Action::ConcealedKong(Tile::from_index_unchecked(self.0 - CONCEALED_KONG_OFFSET))
            }
            ADDED_KONG_OFFSET..=113 => {
                Action::AddedKong(Tile::from_index_unchecked(self.0 - ADDED_KONG_OFFSET))
            }
            PASS_INDEX => Action::Pass,
            _ => unreachable!(),
        }
    }
}

impl From<Action> for ActionId {
    fn from(action: Action) -> Self {
        action.id()
    }
}

impl From<ActionId> for Action {
    fn from(id: ActionId) -> Self {
        id.action()
    }
}

/// Compact legal-action mask for [`ACTION_SPACE_SIZE`] fixed policy outputs.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct ActionMask {
    words: [u64; LEGAL_ACTION_MASK_WORDS],
}

impl ActionMask {
    pub const EMPTY: Self = Self {
        words: [0; LEGAL_ACTION_MASK_WORDS],
    };

    pub const fn contains(self, action: ActionId) -> bool {
        let index = action.index();
        self.words[index / 64] & (1_u64 << (index % 64)) != 0
    }

    pub const fn words(&self) -> &[u64; LEGAL_ACTION_MASK_WORDS] {
        &self.words
    }

    pub const fn count_ones(self) -> u32 {
        self.words[0].count_ones() + self.words[1].count_ones()
    }

    pub const fn is_empty(self) -> bool {
        self.words[0] == 0 && self.words[1] == 0
    }

    pub fn iter(self) -> impl Iterator<Item = ActionId> {
        let mut words = self.words;
        let mut word_index = 0;
        core::iter::from_fn(move || {
            while word_index < words.len() {
                if words[word_index] == 0 {
                    word_index += 1;
                    continue;
                }
                let bit = words[word_index].trailing_zeros() as usize;
                words[word_index] &= words[word_index] - 1;
                return Some(ActionId((word_index * 64 + bit) as u8));
            }
            None
        })
    }

    pub fn to_dense(self) -> [u8; ACTION_SPACE_SIZE] {
        core::array::from_fn(|index| u8::from(self.contains(ActionId(index as u8))))
    }

    pub(crate) fn insert(&mut self, action: ActionId) {
        let index = action.index();
        self.words[index / 64] |= 1_u64 << (index % 64);
    }

    pub(crate) fn insert_tile_mask(&mut self, mut tiles: u32, offset: u8) {
        while tiles != 0 {
            let tile = tiles.trailing_zeros() as u8;
            self.insert(ActionId(offset + tile));
            tiles &= tiles - 1;
        }
    }
}

pub(crate) const fn exchange_offset() -> u8 {
    EXCHANGE_OFFSET
}

pub(crate) const fn discard_offset() -> u8 {
    DISCARD_OFFSET
}

pub(crate) const fn concealed_kong_offset() -> u8 {
    CONCEALED_KONG_OFFSET
}

pub(crate) const fn added_kong_offset() -> u8 {
    ADDED_KONG_OFFSET
}
