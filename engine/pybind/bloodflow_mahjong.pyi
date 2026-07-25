from enum import IntEnum, IntFlag
from typing import Final
import numpy as np
import numpy.typing as npt

ACTION_SPACE_SIZE: Final[int]
ACTION_EXCHANGE_TILE_OFFSET: Final[int]
ACTION_CHOOSE_MISSING_OFFSET: Final[int]
ACTION_DISCARD_OFFSET: Final[int]
ACTION_HU: Final[int]
ACTION_PONG: Final[int]
ACTION_EXPOSED_KONG: Final[int]
ACTION_CONCEALED_KONG_OFFSET: Final[int]
ACTION_ADDED_KONG_OFFSET: Final[int]
ACTION_PASS: Final[int]
LEGAL_ACTION_MASK_WORDS: Final[int]
STEP_RECORD_WIDTH: Final[int]
EVENT_RECORD_WIDTH: Final[int]
EVENT_HISTORY_CAPACITY: Final[int]
SHANTEN_COMPLETE: Final[int]
SHANTEN_MAX: Final[int]
SHANTEN_TERMINAL: Final[int]
SIMPLE_RULE_ACTION_TERMINAL: Final[int]

class EventKind(IntEnum):
    ACTION: Final[int]
    GAME_START: Final[int]
    TURN_START: Final[int]
    DRAW: Final[int]
    DISCARD: Final[int]
    EXCHANGE_COMPLETE: Final[int]
    MISSING_REVEALED: Final[int]
    MELD: Final[int]
    HU: Final[int]
    PAYMENT: Final[int]
    GAME_END: Final[int]

class EventFlag(IntFlag):
    REPLACEMENT_DRAW: Final[int]
    LAST_WALL_TILE: Final[int]
    AFTER_KONG: Final[int]
    OPENING_DISCARD: Final[int]
    SELF_DRAW: Final[int]
    ROB_KONG: Final[int]
    HEAVENLY: Final[int]
    EARTHLY: Final[int]

TILE_OBSERVATION_WIDTH: Final[int]
TILE_OBSERVATION_PLANES: Final[int]
MELD_OBSERVATION_WIDTH: Final[int]
MELD_SLOTS: Final[int]
MELD_FIELDS: Final[int]
RIVER_OBSERVATION_WIDTH: Final[int]
RIVER_TILE_CAPACITY: Final[int]
RIVER_FIELDS: Final[int]
META_OBSERVATION_WIDTH: Final[int]
PLAYER_COUNT: Final[int]
TILE_KIND_COUNT: Final[int]
PHASE_EXCHANGE: Final[int]
PHASE_CHOOSE_MISSING: Final[int]
PHASE_TURN: Final[int]
PHASE_HU_RESPONSE: Final[int]
PHASE_MELD_RESPONSE: Final[int]
PHASE_FINISHED: Final[int]

class Game:
    def __init__(self, seed: int = 0) -> None: ...
    @staticmethod
    def with_exchange_direction(seed: int, direction: int) -> Game: ...
    def reset(self, seed: int) -> None: ...
    @property
    def phase(self) -> int: ...
    @property
    def decision(self) -> tuple[int, int] | None: ...
    @property
    def legal_action_mask(self) -> tuple[int, int]: ...
    def simple_rule_action(self) -> int | None: ...
    def step_id(self, action: int) -> tuple[int, ...]: ...
    def step_into(self, action: int, output: npt.NDArray[np.int64]) -> None: ...
    @property
    def event_count(self) -> int: ...
    @property
    def event_dropped(self) -> int: ...
    def events_into(
        self,
        viewer: int,
        output: npt.NDArray[np.int32],
    ) -> int: ...
    def step_events_into(
        self,
        viewer: int,
        output: npt.NDArray[np.int32],
    ) -> int: ...
    @property
    def dealer(self) -> int: ...
    @property
    def exchange_direction(self) -> int: ...
    @property
    def wall_remaining(self) -> int: ...
    @property
    def current_draw(self) -> tuple[int, int, bool] | None: ...
    def concealed_into(self, seat: int, output: npt.NDArray[np.uint8]) -> None: ...
    def locked_into(self, seat: int, output: npt.NDArray[np.uint8]) -> None: ...
    def exchange_selection_into(
        self, seat: int, output: npt.NDArray[np.uint8]
    ) -> None: ...
    def hand_analysis(self, seat: int) -> tuple[int, int]: ...
    def scores(self) -> tuple[int, int, int, int]: ...
    def missing_suits(self) -> tuple[int, int, int, int]: ...
    def has_won(self, seat: int) -> bool: ...
    def max_win_multipliers(self) -> tuple[int, int, int, int]: ...
    def melds(self, seat: int) -> list[tuple[int, int, int]]: ...
    def discards(self) -> list[tuple[int, int]]: ...
    def rankings(self) -> tuple[int, int, int, int]: ...

class Batch:
    def __init__(self, size: int, seed: int = 0) -> None: ...
    def __len__(self) -> int: ...
    @property
    def is_empty(self) -> bool: ...
    def event_dropped_into(self, output: npt.NDArray[np.uint64]) -> None: ...
    def reset_all(self, seed: int) -> None: ...
    def reset_at(self, index: int, seed: int) -> None: ...
    def reset_many(
        self,
        indices: npt.NDArray[np.uint32],
        seeds: npt.NDArray[np.uint64],
    ) -> None: ...
    def legal_action_masks_into(self, output: npt.NDArray[np.uint64]) -> None: ...
    def simple_rule_actions_into(self, output: npt.NDArray[np.uint8]) -> None: ...
    def hand_analysis_into(
        self,
        shanten: npt.NDArray[np.int8],
        improving_tiles: npt.NDArray[np.uint32],
    ) -> None: ...
    def hand_analysis_indices_into(
        self,
        indices: npt.NDArray[np.uint32],
        shanten: npt.NDArray[np.int8],
        improving_tiles: npt.NDArray[np.uint32],
    ) -> None: ...
    def events_into(
        self,
        events: npt.NDArray[np.int32],
        lengths: npt.NDArray[np.uint16],
    ) -> None: ...
    def step_events_into(
        self,
        events: npt.NDArray[np.int32],
        lengths: npt.NDArray[np.uint16],
    ) -> None: ...
    def step_into(
        self,
        actions: npt.NDArray[np.uint8],
        records: npt.NDArray[np.int64],
    ) -> None: ...
    def observe_into(
        self,
        tile_obs: npt.NDArray[np.uint8],
        melds: npt.NDArray[np.uint8],
        river: npt.NDArray[np.uint8],
        meta: npt.NDArray[np.int32],
    ) -> None: ...
    def step_and_observe_history_into(
        self,
        actions: npt.NDArray[np.uint8],
        history_seat_masks: npt.NDArray[np.uint8],
        records: npt.NDArray[np.int64],
        mask_words: npt.NDArray[np.uint64],
        tile_obs: npt.NDArray[np.uint8],
        melds: npt.NDArray[np.uint8],
        river: npt.NDArray[np.uint8],
        meta: npt.NDArray[np.int32],
        events: npt.NDArray[np.int32],
        event_lengths: npt.NDArray[np.uint16],
    ) -> None: ...
    def reset_and_observe_history_into(
        self,
        reset_flags: npt.NDArray[np.uint8],
        seeds: npt.NDArray[np.uint64],
        history_seat_masks: npt.NDArray[np.uint8],
        mask_words: npt.NDArray[np.uint64],
        tile_obs: npt.NDArray[np.uint8],
        melds: npt.NDArray[np.uint8],
        river: npt.NDArray[np.uint8],
        meta: npt.NDArray[np.int32],
        events: npt.NDArray[np.int32],
        event_lengths: npt.NDArray[np.uint16],
    ) -> None: ...
