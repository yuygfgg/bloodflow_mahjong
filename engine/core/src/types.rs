use core::fmt;

pub const PLAYER_COUNT: usize = 4;
pub const TILE_KIND_COUNT: usize = 27;
pub const TILE_COPIES: u8 = 4;
pub const WALL_TILE_COUNT: usize = 108;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Tile(u8);

impl Tile {
    pub(crate) const fn from_index_unchecked(index: u8) -> Self {
        Self(index)
    }

    pub const fn new(index: u8) -> Option<Self> {
        if index < TILE_KIND_COUNT as u8 {
            Some(Self(index))
        } else {
            None
        }
    }

    pub const fn from_suit_rank(suit: Suit, rank: u8) -> Option<Self> {
        if rank < 9 {
            Some(Self(suit as u8 * 9 + rank))
        } else {
            None
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn as_u8(self) -> u8 {
        self.0
    }

    pub const fn suit(self) -> Suit {
        match self.0 / 9 {
            0 => Suit::Characters,
            1 => Suit::Bamboo,
            _ => Suit::Dots,
        }
    }

    pub const fn rank(self) -> u8 {
        self.0 % 9
    }

    pub const fn is_terminal(self) -> bool {
        self.rank() == 0 || self.rank() == 8
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const SUITS: [char; 3] = ['m', 's', 'p'];
        write!(f, "{}{}", self.rank() + 1, SUITS[self.suit() as usize])
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(u8)]
pub enum Suit {
    Characters = 0,
    Bamboo = 1,
    Dots = 2,
}

impl Suit {
    pub const ALL: [Self; 3] = [Self::Characters, Self::Bamboo, Self::Dots];

    pub const fn mask(self) -> u32 {
        0x1ff << (self as u32 * 9)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct Seat(u8);

impl Seat {
    pub const EAST: Self = Self(0);
    pub const ALL: [Self; PLAYER_COUNT] = [Self(0), Self(1), Self(2), Self(3)];

    pub const fn new(index: u8) -> Option<Self> {
        if index < PLAYER_COUNT as u8 {
            Some(Self(index))
        } else {
            None
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn as_u8(self) -> u8 {
        self.0
    }

    pub const fn next(self) -> Self {
        Self((self.0 + 1) & 3)
    }

    pub const fn offset(self, amount: u8) -> Self {
        Self((self.0 + amount) & 3)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ExchangeDirection {
    Left = 1,
    Across = 2,
    Right = 3,
}

impl ExchangeDirection {
    pub const fn offset(self) -> u8 {
        self as u8
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum MeldKind {
    Pong = 0,
    ExposedKong = 1,
    AddedKong = 2,
    ConcealedKong = 3,
}

impl MeldKind {
    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn is_kong(self) -> bool {
        !matches!(self, Self::Pong)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Meld {
    pub tile: Tile,
    pub kind: MeldKind,
    pub source: Seat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WinSource {
    SelfDraw,
    Discard { from: Seat },
    RobKong { from: Seat },
}
