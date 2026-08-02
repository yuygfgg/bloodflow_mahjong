use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use rayon::prelude::*;

use crate::WinFlags;
use crate::game::{Game, SCORE_UNIT};
use crate::rules::hand::{all_tiles, hand_structure_score, mask_tiles};
use crate::types::{PLAYER_COUNT, Seat, TILE_COPIES, TILE_KIND_COUNT, Tile};

use super::belief::{HazardTable, OpponentTurnEvent, SampledWin};
use super::state::{PlannedDraw, PlanningHand, PlanningPublicState};
use super::value::PublicValueModel;

// A physical tile contributes at most 10.67 intrinsic points, 6 adjacent
// points, 4 gapped points, and 36 sequence points in `hand_structure_score`.
// Rounding that proven per-tile upper bound up keeps the normalized value in
// [0, 1] without a fitted scale.
const MAX_STRUCTURE_SCORE_PER_TILE: u16 = 57;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DiscardPlan {
    pub(super) tile: Tile,
    pub(super) value: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PlanningHorizon {
    pub(super) draws: u8,
    /// Whether this horizon reaches the actual end of the live wall.
    ///
    /// A configured draw limit is normally an evaluation cutoff, not a game
    /// termination. Keeping this bit next to the clamped draw count prevents
    /// truncated branches from being scored with end-of-wall settlement.
    pub(super) reaches_wall: bool,
}

impl PlanningHorizon {
    /// Builds a horizon from the physical live-wall size and the first draw
    /// origin after the current turn.
    ///
    /// `draws` counts only actor opportunities. `reaches_wall` uses the
    /// physical number of tiles consumed by that many scheduled turns, so a
    /// short actor horizon is not mistaken for actual wall exhaustion.
    pub(super) fn new(wall_remaining: usize, draw_limit: u8, first_draw: PlannedDraw) -> Self {
        let available_draws = match first_draw {
            PlannedDraw::Normal => wall_remaining / PLAYER_COUNT,
            PlannedDraw::Supplement => {
                if wall_remaining == 0 {
                    0
                } else {
                    1 + (wall_remaining - 1) / PLAYER_COUNT
                }
            }
        };
        let draws_usize = available_draws.min(usize::from(draw_limit));
        let draws = draws_usize.min(usize::from(u8::MAX)) as u8;
        let physical_draws = match first_draw {
            PlannedDraw::Normal => draws_usize.saturating_mul(PLAYER_COUNT),
            PlannedDraw::Supplement => draws_usize
                .checked_sub(1)
                .map_or(0, |ordinary| 1 + ordinary.saturating_mul(PLAYER_COUNT)),
        };
        Self {
            draws,
            reaches_wall: physical_draws >= wall_remaining,
        }
    }
}

#[derive(Clone, Copy)]
enum DiscardContext {
    Root,
    Simulated { after_kong: bool },
}

impl DiscardContext {
    const fn payout_scale(self) -> u32 {
        match self {
            Self::Root | Self::Simulated { after_kong: false } => 1,
            Self::Simulated { after_kong: true } => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct DrawTimeline {
    wall_draws: u8,
    normal_draws: u8,
}

/// Tiles consumed by simulated draws whose identity is known to the planner.
///
/// `HandGraphPlanner::base_unknown` is the inventory that remains unknown at
/// the root decision.  A tile that is drawn and then discarded is no longer in
/// that inventory, even though it is absent from the current hand.  Keeping
/// this histogram in the Bellman state prevents such a tile from being drawn
/// again on a later branch.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct DrawnInventory {
    counts: [u8; TILE_KIND_COUNT],
}

impl DrawnInventory {
    fn after_known_draw(self, tile: Tile) -> Self {
        let mut counts = self.counts;
        counts[tile.index()] = counts[tile.index()].saturating_add(1);
        Self { counts }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct HandChangeBudget {
    transitions_left: u8,
    detours_left: u8,
}

impl HandChangeBudget {
    fn new(hand: PlanningHand, extra_detours: u8, turns_left: u8) -> Self {
        Self {
            transitions_left: hand_change_distance(hand)
                .saturating_add(extra_detours)
                .min(turns_left),
            detours_left: extra_detours,
        }
    }

    fn after_change(self, before: PlanningHand, after: PlanningHand) -> Option<Self> {
        let transitions_left = self.transitions_left.checked_sub(1)?;
        let detour_cost = hand_change_distance(after)
            .saturating_add(1)
            .saturating_sub(hand_change_distance(before));
        Some(Self {
            transitions_left,
            detours_left: self.detours_left.checked_sub(detour_cost)?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct BellmanState {
    hand: PlanningHand,
    public: PlanningPublicState,
    turns_left: u8,
    timeline: DrawTimeline,
    drawn: DrawnInventory,
}

impl BellmanState {
    const fn with_hand(self, hand: PlanningHand) -> Self {
        Self { hand, ..self }
    }

    const fn with_public(self, public: PlanningPublicState) -> Self {
        Self { public, ..self }
    }

    fn after_known_draw(self, tile: Tile) -> Self {
        Self {
            drawn: self.drawn.after_known_draw(tile),
            ..self
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct OwnDraw {
    hand: PlanningHand,
    tile: Tile,
    origin: PlannedDraw,
    flags: WinFlags,
}

impl DrawTimeline {
    const fn after(self, draw: PlannedDraw) -> Self {
        let normal_draw = match draw {
            PlannedDraw::Normal => 1,
            PlannedDraw::Supplement => 0,
        };
        Self {
            wall_draws: self.wall_draws.saturating_add(1),
            normal_draws: self.normal_draws.saturating_add(normal_draw),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct OpponentCycle {
    active: Seat,
    events_left: u8,
}

impl OpponentCycle {
    const fn start(actor: Seat, active: Seat) -> Self {
        Self {
            active,
            events_left: events_before_actor(actor, active),
        }
    }

    const fn after_turn(self, actor: Seat, next: Seat) -> Self {
        Self {
            active: next,
            events_left: events_before_actor(actor, next),
        }
    }

    fn is_complete(self, actor: Seat) -> bool {
        self.active == actor || self.events_left == 0
    }
}

const fn events_before_actor(actor: Seat, active: Seat) -> u8 {
    actor.as_u8().wrapping_sub(active.as_u8()) & 3
}

/// Akochan-style own-hand candidate graph evaluated by remaining draw count.
///
/// Nodes are post-action private hand states. A draw edge can win immediately
/// or choose any legal discard edge. Non-changing draws return to the same
/// node without consuming the hand-change budget. Useful hand replacements
/// consume one unit and cannot move above the root shanten boundary.
pub(super) struct HandGraphPlanner<'a> {
    actor: crate::Seat,
    base_unknown: [u8; TILE_KIND_COUNT],
    root_concealed: [u8; TILE_KIND_COUNT],
    root_locked: [u8; TILE_KIND_COUNT],
    hazards: Option<&'a HazardTable>,
    value_model: &'a PublicValueModel,
    candidate_limit: usize,
    wall_remaining: u8,
}

impl<'a> HandGraphPlanner<'a> {
    pub(super) fn new(
        game: &Game,
        actor: crate::Seat,
        root: PlanningHand,
        hazards: Option<&'a HazardTable>,
        value_model: &'a PublicValueModel,
        candidate_limit: usize,
    ) -> Self {
        let visible = game.visible_tile_counts(actor);
        Self {
            actor,
            base_unknown: core::array::from_fn(|index| {
                TILE_COPIES.saturating_sub(visible[index].min(TILE_COPIES))
            }),
            root_concealed: root.holding.concealed,
            root_locked: root.holding.locked,
            hazards,
            value_model,
            candidate_limit,
            wall_remaining: u8::try_from(game.wall_remaining()).unwrap_or(u8::MAX),
        }
    }

    pub(super) fn best_discard(
        &self,
        root: PlanningHand,
        discard_mask: u32,
        public: PlanningPublicState,
        horizon: PlanningHorizon,
        extra_hand_changes: u8,
    ) -> Option<DiscardPlan> {
        self.best_discard_at(
            BellmanState {
                hand: root,
                public,
                turns_left: horizon.draws,
                timeline: DrawTimeline::default(),
                drawn: DrawnInventory::default(),
            },
            discard_mask,
            extra_hand_changes,
            DiscardContext::Root,
            horizon.reaches_wall,
        )
    }

    fn best_discard_at(
        &self,
        state: BellmanState,
        discard_mask: u32,
        extra_hand_changes: u8,
        context: DiscardContext,
        reaches_wall: bool,
    ) -> Option<DiscardPlan> {
        let mut candidates: Vec<_> = mask_tiles(discard_mask)
            .filter_map(|tile| state.hand.after_discard(tile).map(|hand| (tile, hand)))
            .collect();
        let root_shanten = candidates
            .iter()
            .map(|(_, hand)| hand.analysis().shanten)
            .min()?;
        if !state.hand.holding.has_won {
            // Before the first win, shanten is a dominance constraint. The
            // sampled immediate hazard ranks equally advanced routes; only
            // terminal search may validate sacrificing hand progress.
            candidates.retain(|(_, hand)| hand.analysis().shanten == root_shanten);
        }
        let roots: Vec<_> = candidates
            .iter()
            .map(|&(_, hand)| {
                (
                    hand,
                    HandChangeBudget::new(hand, extra_hand_changes, state.turns_left),
                )
            })
            .collect();
        let admitted = Arc::new(self.candidate_hands(&roots, self.candidate_limit));
        let evaluate = |branch: &mut BranchEvaluator<'_, '_>, (tile, hand)| {
            let budget = HandChangeBudget::new(hand, extra_hand_changes, state.turns_left);
            let plan = DiscardPlan {
                tile,
                value: branch.after_discard(
                    state.with_hand(hand),
                    tile,
                    Continuation::Graph { budget },
                    context,
                ),
            };
            (plan, hand)
        };
        let evaluated: Vec<_> = if state.turns_left <= 1 {
            candidates
                .into_par_iter()
                .map(|candidate| {
                    evaluate(
                        &mut BranchEvaluator::with_admitted(
                            self,
                            Arc::clone(&admitted),
                            reaches_wall,
                            true,
                        ),
                        candidate,
                    )
                })
                .collect()
        } else {
            let mut branch = BranchEvaluator::with_admitted(self, admitted, reaches_wall, true);
            candidates
                .into_iter()
                .map(|candidate| evaluate(&mut branch, candidate))
                .collect()
        };
        evaluated
            .into_iter()
            .max_by(|(left, left_hand), (right, right_hand)| {
                left.value.total_cmp(&right.value).then_with(|| {
                    (self.progress(*left_hand), Reverse(left.tile))
                        .cmp(&(self.progress(*right_hand), Reverse(right.tile)))
                })
            })
            .map(|(plan, _)| plan)
    }

    pub(super) fn value_before_draw(
        &self,
        hand: PlanningHand,
        public: PlanningPublicState,
        horizon: PlanningHorizon,
        extra_hand_changes: u8,
    ) -> f64 {
        let budget = HandChangeBudget::new(hand, extra_hand_changes, horizon.draws);
        BranchEvaluator::new(
            self,
            self.candidate_limit,
            &[(hand, budget)],
            horizon.reaches_wall,
        )
        .before_draw(
            BellmanState {
                hand,
                public,
                turns_left: horizon.draws,
                timeline: DrawTimeline::default(),
                drawn: DrawnInventory::default(),
            },
            budget,
        )
    }

    /// Evaluates a continuation after another player wins from this actor.
    ///
    /// The authoritative engine resumes with the seat after the last winner.
    /// Starting at the actor's own draw would skip every intervening turn and
    /// use the wrong live-wall offset.
    pub(super) fn value_after_external_win(
        &self,
        hand: PlanningHand,
        public: PlanningPublicState,
        next_actor: Seat,
        horizon: PlanningHorizon,
        extra_hand_changes: u8,
    ) -> f64 {
        let budget = HandChangeBudget::new(hand, extra_hand_changes, horizon.draws);
        let mut branch = BranchEvaluator::new(
            self,
            self.candidate_limit,
            &[(hand, budget)],
            horizon.reaches_wall,
        );
        branch.opponent_cycle(
            BellmanState {
                hand,
                public,
                turns_left: horizon.draws,
                timeline: DrawTimeline::default(),
                drawn: DrawnInventory::default(),
            },
            Continuation::Graph { budget },
            OpponentCycle::start(self.actor, next_actor),
        )
    }

    fn available_copies(&self, hand: PlanningHand, tile: Tile) -> u8 {
        let acquired =
            hand.holding.concealed[tile.index()].saturating_sub(self.root_concealed[tile.index()]);
        self.base_unknown[tile.index()].saturating_sub(acquired)
    }

    fn available_copies_in_state(&self, state: BellmanState, tile: Tile) -> u8 {
        self.base_unknown[tile.index()].saturating_sub(state.drawn.counts[tile.index()])
    }

    fn available_copies_after_draws(&self, drawn: DrawnInventory, tile: Tile) -> u8 {
        self.base_unknown[tile.index()].saturating_sub(drawn.counts[tile.index()])
    }

    fn progress(&self, hand: PlanningHand) -> HandProgress {
        let analysis = hand.analysis();
        let live_improvements = mask_tiles(analysis.improving_tiles)
            .map(|tile| u16::from(self.available_copies(hand, tile)))
            .sum();
        let unlocked_tiles = all_tiles()
            .map(|tile| u16::from(hand.holding.unlocked_count(tile)))
            .sum();
        HandProgress {
            shanten: Reverse(if hand.holding.has_won {
                0
            } else {
                analysis.shanten
            }),
            live_improvements,
            distinct_improvements: analysis.improving_tiles.count_ones() as u8,
            structure: hand_structure_score(
                &hand
                    .holding
                    .evaluation_counts()
                    .unwrap_or(hand.holding.concealed),
            ),
            unlocked_tiles,
            state: hand,
        }
    }

    fn candidate_hands(
        &self,
        roots: &[(PlanningHand, HandChangeBudget)],
        candidate_limit: usize,
    ) -> HashSet<PlanningHand> {
        let mut admitted: HashSet<_> = roots.iter().map(|&(hand, _)| hand).collect();
        let mut frontier: Vec<_> = roots
            .iter()
            .copied()
            .map(|(hand, budget)| CandidateNode {
                hand,
                drawn: DrawnInventory::default(),
                budget,
            })
            .collect();
        let total_limit = candidate_limit.max(admitted.len());
        let max_depth = roots
            .iter()
            .map(|(_, budget)| budget.transitions_left)
            .max()
            .unwrap_or_default();

        for depth in 0..max_depth {
            if admitted.len() >= total_limit || frontier.is_empty() {
                break;
            }
            let mut generated = HashSet::<CandidateNode>::new();
            for previous in frontier {
                for draw in all_tiles() {
                    if self.available_copies_after_draws(previous.drawn, draw) == 0 {
                        continue;
                    }
                    let Some(drawn) = previous.hand.with_draw(draw) else {
                        continue;
                    };
                    let next_drawn = previous.drawn.after_known_draw(draw);
                    for discard in mask_tiles(drawn.holding.discard_mask()) {
                        let Some(next) = drawn.after_discard(discard) else {
                            continue;
                        };
                        if next == previous.hand {
                            continue;
                        }
                        let Some(budget) = previous.budget.after_change(previous.hand, next) else {
                            continue;
                        };
                        if admitted.contains(&next) {
                            continue;
                        }
                        generated.insert(CandidateNode {
                            hand: next,
                            drawn: next_drawn,
                            budget,
                        });
                    }
                }
            }

            let mut ranked: Vec<_> = generated
                .into_iter()
                .map(|node| (Reverse(self.progress(node.hand)), node))
                .collect();
            ranked.sort_unstable_by_key(|(progress, _)| *progress);
            let remaining_layers = usize::from(max_depth - depth);
            let remaining_slots = total_limit - admitted.len();
            let layer_quota = remaining_slots.div_ceil(remaining_layers);
            ranked.truncate(layer_quota);
            frontier = ranked.into_iter().map(|(_, node)| node).collect();
            admitted.extend(frontier.iter().map(|node| node.hand));
        }
        admitted
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CandidateNode {
    hand: PlanningHand,
    drawn: DrawnInventory,
    budget: HandChangeBudget,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct HandProgress {
    shanten: Reverse<i8>,
    live_improvements: u16,
    distinct_improvements: u8,
    structure: i32,
    unlocked_tiles: u16,
    state: PlanningHand,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum EvaluationPhase {
    OwnDraw,
    Opponent {
        active: Seat,
        events_left: u8,
        continuation: Continuation,
    },
    RepeatWinTail,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Continuation {
    Graph { budget: HandChangeBudget },
    Repeat { discard_win_available: bool },
}

impl Continuation {
    const fn repeat_cycle() -> Self {
        Self::Repeat {
            discard_win_available: true,
        }
    }

    const fn budget(self) -> HandChangeBudget {
        match self {
            Self::Graph { budget } => budget,
            Self::Repeat { .. } => HandChangeBudget {
                transitions_left: 0,
                detours_left: 0,
            },
        }
    }

    const fn after_actor_win(self) -> Self {
        match self {
            Self::Graph { .. } | Self::Repeat { .. } => Self::Repeat {
                discard_win_available: false,
            },
        }
    }

    const fn permits_actor_win(self) -> bool {
        matches!(
            self,
            Self::Graph { .. }
                | Self::Repeat {
                    discard_win_available: true
                }
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CacheKey {
    hand: PlanningHand,
    public: PlanningPublicState,
    turns_left: u8,
    budget: HandChangeBudget,
    timeline: DrawTimeline,
    drawn: DrawnInventory,
    phase: EvaluationPhase,
}

struct BranchEvaluator<'a, 'model> {
    planner: &'a HandGraphPlanner<'model>,
    admitted: Arc<HashSet<PlanningHand>>,
    cache: HashMap<CacheKey, f64>,
    reaches_wall: bool,
    parallelize_draws: bool,
}

impl<'a, 'model> BranchEvaluator<'a, 'model> {
    fn new(
        planner: &'a HandGraphPlanner<'model>,
        candidate_limit: usize,
        roots: &[(PlanningHand, HandChangeBudget)],
        reaches_wall: bool,
    ) -> Self {
        Self::with_admitted(
            planner,
            Arc::new(planner.candidate_hands(roots, candidate_limit)),
            reaches_wall,
            true,
        )
    }

    fn with_admitted(
        planner: &'a HandGraphPlanner<'model>,
        admitted: Arc<HashSet<PlanningHand>>,
        reaches_wall: bool,
        parallelize_draws: bool,
    ) -> Self {
        Self {
            planner,
            admitted,
            cache: HashMap::new(),
            reaches_wall,
            parallelize_draws,
        }
    }

    fn serial_fork(&self) -> Self {
        Self {
            planner: self.planner,
            admitted: Arc::clone(&self.admitted),
            cache: HashMap::new(),
            reaches_wall: self.reaches_wall,
            parallelize_draws: false,
        }
    }

    fn before_draw(&mut self, state: BellmanState, budget: HandChangeBudget) -> f64 {
        let hand = state.hand;
        let public = state.public;
        let turns_left = state.turns_left;
        let timeline = state.timeline;
        if public.is_terminal() {
            return 0.0;
        }
        if timeline.wall_draws >= self.planner.wall_remaining {
            return self.planner.value_model.wall_value(hand, public);
        }
        if hand.holding.locked != self.planner.root_locked {
            return self.repeat_win_tail(state);
        }
        let key = CacheKey {
            hand,
            public,
            turns_left,
            budget,
            timeline,
            drawn: state.drawn,
            phase: EvaluationPhase::OwnDraw,
        };
        if let Some(&value) = self.cache.get(&key) {
            return value;
        }

        if turns_left == 0 {
            let value = self.leaf_value(BellmanState {
                hand,
                public,
                turns_left,
                timeline,
                drawn: state.drawn,
            });
            self.cache.insert(key, value);
            return value;
        }

        let (public, draw_origin) = public.take_next_draw();
        let flags = self.draw_win_flags(draw_origin, timeline);
        let next_timeline = timeline.after(draw_origin);
        let draws: Vec<_> = all_tiles()
            .filter_map(|tile| {
                let copies = self.planner.available_copies_in_state(state, tile);
                let drawn = hand.with_draw(tile)?;
                (copies != 0).then_some((tile, copies, drawn))
            })
            .collect();
        let evaluate = |branch: &mut Self, tile, drawn| {
            branch.after_draw(
                BellmanState {
                    hand,
                    public,
                    turns_left: turns_left.saturating_sub(1),
                    timeline: next_timeline,
                    drawn: state.drawn,
                },
                OwnDraw {
                    hand: drawn,
                    tile,
                    origin: draw_origin,
                    flags,
                },
                budget,
            )
        };
        let outcomes: Vec<_> = if self.parallelize_draws && turns_left == 1 {
            draws
                .into_par_iter()
                .map(|(tile, copies, drawn)| {
                    let value = evaluate(&mut self.serial_fork(), tile, drawn);
                    (copies, value)
                })
                .collect()
        } else {
            draws
                .into_iter()
                .map(|(tile, copies, drawn)| {
                    let value = evaluate(self, tile, drawn);
                    (copies, value)
                })
                .collect()
        };
        let (total_copies, weighted_value) =
            outcomes
                .into_iter()
                .fold((0_u16, 0.0), |(total, weighted), (copies, value)| {
                    (
                        total + u16::from(copies),
                        weighted + f64::from(copies) * value,
                    )
                });
        let value = if total_copies == 0 {
            self.planner.value_model.wall_value(hand, public)
        } else {
            weighted_value / f64::from(total_copies)
        };
        self.cache.insert(key, value);
        value
    }

    fn repeat_win_tail(&mut self, state: BellmanState) -> f64 {
        let hand = state.hand;
        let public = state.public;
        let turns_left = state.turns_left;
        let timeline = state.timeline;
        if public.is_terminal() {
            return 0.0;
        }
        if timeline.wall_draws >= self.planner.wall_remaining {
            return self.planner.value_model.wall_value(hand, public);
        }
        let key = CacheKey {
            hand,
            public,
            turns_left,
            budget: HandChangeBudget::default(),
            timeline,
            drawn: state.drawn,
            phase: EvaluationPhase::RepeatWinTail,
        };
        if let Some(&value) = self.cache.get(&key) {
            return value;
        }
        if turns_left == 0 {
            let value = self.leaf_value(BellmanState {
                hand,
                public,
                turns_left,
                timeline,
                drawn: state.drawn,
            });
            self.cache.insert(key, value);
            return value;
        }

        let (public, draw_origin) = public.take_next_draw();
        let flags = self.draw_win_flags(draw_origin, timeline);
        let next_timeline = timeline.after(draw_origin);
        let discard_context = DiscardContext::Simulated {
            after_kong: draw_origin == PlannedDraw::Supplement,
        };
        let draws: Vec<_> = all_tiles()
            .filter_map(|tile| {
                let copies = self.planner.available_copies_in_state(state, tile);
                (copies != 0).then_some((tile, copies))
            })
            .collect();
        let outcomes: Vec<_> = if self.parallelize_draws && turns_left == 1 {
            draws
                .into_par_iter()
                .map(|(tile, copies)| {
                    let value = self.serial_fork().repeat_draw_value(
                        state,
                        public,
                        flags,
                        next_timeline,
                        discard_context,
                        tile,
                    );
                    (copies, value)
                })
                .collect()
        } else {
            draws
                .into_iter()
                .map(|(tile, copies)| {
                    let value = self.repeat_draw_value(
                        state,
                        public,
                        flags,
                        next_timeline,
                        discard_context,
                        tile,
                    );
                    (copies, value)
                })
                .collect()
        };
        let (total_copies, weighted_value) =
            outcomes
                .into_iter()
                .fold((0_u16, 0.0), |(total, weighted), (copies, value)| {
                    (
                        total + u16::from(copies),
                        weighted + f64::from(copies) * value,
                    )
                });
        let value = if total_copies == 0 {
            self.planner.value_model.wall_value(hand, public)
        } else {
            weighted_value / f64::from(total_copies)
        };
        self.cache.insert(key, value);
        value
    }

    fn repeat_draw_value(
        &mut self,
        state: BellmanState,
        public: PlanningPublicState,
        flags: WinFlags,
        next_timeline: DrawTimeline,
        discard_context: DiscardContext,
        tile: Tile,
    ) -> f64 {
        let after_draw = BellmanState {
            turns_left: state.turns_left - 1,
            timeline: next_timeline,
            public,
            ..state
        }
        .after_known_draw(tile);
        let decline = self.after_discard(
            after_draw,
            tile,
            Continuation::repeat_cycle(),
            discard_context,
        );
        let take = state
            .hand
            .with_draw(tile)
            .and_then(|drawn| drawn.win_on_draw(tile, flags).map(|win| (drawn, win)))
            .map_or(decline, |(drawn, win)| {
                let (next_public, gain) = self.planner.value_model.self_draw_transition(
                    public,
                    win.multiplier,
                    win.shape_multiplier,
                );
                let next_hand = drawn.after_win(Some(tile), win);
                gain + self.opponent_cycle(
                    BellmanState {
                        hand: next_hand,
                        public: next_public,
                        ..after_draw
                    },
                    Continuation::repeat_cycle(),
                    OpponentCycle::start(self.planner.actor, self.planner.actor.next()),
                )
            });
        take.max(decline)
    }

    fn leaf_value(&self, state: BellmanState) -> f64 {
        let hand = state.hand;
        let public = state.public;
        if state.timeline.wall_draws >= self.planner.wall_remaining {
            self.planner.value_model.wall_value(hand, public)
        } else {
            // A configured horizon is an evaluation cutoff, not a game
            // termination. Preserve the end-of-wall value as a prior and add
            // the value of reaching a useful hand before the cutoff.
            self.frontier_value(state)
        }
    }

    fn frontier_value(&self, state: BellmanState) -> f64 {
        let hand = state.hand;
        let public = state.public;
        // End-of-wall settlement is a reliable prior only when the configured
        // horizon covers the nominal wall. A shorter cutoff starts from a
        // neutral score and uses the hand's continuation potential below.
        let baseline = if self.reaches_wall {
            self.planner.value_model.wall_value(hand, public)
        } else {
            0.0
        };
        let total_copies: u16 = all_tiles()
            .map(|tile| u16::from(self.planner.available_copies_in_state(state, tile)))
            .sum();
        if total_copies == 0 {
            return baseline;
        }

        // `wall_draws` counts every physical tile consumed by all seats. The
        // frontier is reached immediately before the actor's next draw, so
        // only roughly one in four remaining ordinary tiles belongs to the
        // actor. A pending replacement draw is the one exception: it belongs
        // to the actor before the ordinary rotation resumes.
        let remaining_wall = self
            .planner
            .wall_remaining
            .saturating_sub(state.timeline.wall_draws);
        let remaining_draws = actor_draw_opportunities(remaining_wall, public.next_draw)
            .min(u8::try_from(total_copies).unwrap_or(u8::MAX));
        if remaining_draws == 0 {
            return baseline;
        }

        let (_, nominal_gain) =
            self.planner
                .value_model
                .self_draw_by(public, self.planner.actor, 1, 1);
        let score_unit = SCORE_UNIT as f64;
        let win_scale = nominal_gain.max(score_unit);

        let analysis = (!hand.holding.has_won).then(|| hand.holding.analysis());
        let wait_mask = analysis.map_or_else(
            || {
                all_tiles().fold(0_u32, |mask, tile| {
                    let wins = self.planner.available_copies_in_state(state, tile) != 0
                        && hand
                            .with_draw(tile)
                            .and_then(|drawn| drawn.win_on_draw(tile, WinFlags::NONE))
                            .is_some();
                    mask | (u32::from(wins) << tile.index())
                })
            },
            |analysis| {
                if analysis.shanten <= 0 {
                    analysis.improving_tiles
                } else {
                    0
                }
            },
        );
        if wait_mask != 0 {
            let mut wait_copies = 0_u16;
            let mut weighted_delta = 0.0;
            for tile in mask_tiles(wait_mask) {
                let copies = self.planner.available_copies_in_state(state, tile);
                if copies == 0 {
                    continue;
                }
                let Some(drawn) = hand.with_draw(tile) else {
                    continue;
                };
                let Some(win) = drawn.win_on_draw(tile, WinFlags::NONE) else {
                    continue;
                };
                let (next_public, gain) = self.planner.value_model.self_draw_transition(
                    public,
                    win.multiplier,
                    win.shape_multiplier,
                );
                let next_hand = drawn.after_win(Some(tile), win);
                let delta =
                    gain + self.planner.value_model.wall_value(next_hand, next_public) - baseline;
                wait_copies = wait_copies.saturating_add(u16::from(copies));
                weighted_delta += f64::from(copies) * delta;
            }
            if wait_copies != 0 {
                let wait_probability = hit_probability(total_copies, wait_copies, remaining_draws);
                let average_delta = weighted_delta / f64::from(wait_copies);
                return cap_frontier_delta(
                    baseline,
                    baseline + wait_probability * average_delta,
                    win_scale,
                );
            }
        }

        // Conventional shanten describes the already completed subset of an
        // expanded post-win hand, not its next Blood Flow win. If exact wait
        // enumeration above found no repeat win, there is no calibrated shape
        // continuation to add at this frontier.
        let Some(analysis) = analysis else {
            return baseline;
        };
        let improvement_copies: u16 = mask_tiles(analysis.improving_tiles)
            .map(|tile| u16::from(self.planner.available_copies_in_state(state, tile)))
            .sum();
        if improvement_copies == 0 {
            return baseline;
        }
        let improvement_probability =
            hit_probability(total_copies, improvement_copies, remaining_draws);

        // For a hand above tenpai, an improving draw has no immediate score
        // event. Use the same one-multiplier self-draw value as the unit of
        // progress, discounted by the remaining shanten distance. The
        // normalized structure score estimates how much of a non-improving
        // draw can be retained as useful shape instead of being discarded.
        let distance = f64::from(analysis.shanten.max(1));
        let hand_len = hand.holding.concealed.iter().copied().sum::<u8>();
        let structure_probability = f64::from(hand_structure_score(&hand.holding.concealed))
            / f64::from(u16::from(hand_len.max(1)) * MAX_STRUCTURE_SCORE_PER_TILE);
        let progress_value = improvement_probability * win_scale / distance;
        cap_frontier_delta(
            baseline,
            baseline + progress_value * (1.0 + structure_probability.clamp(0.0, 1.0)),
            win_scale,
        )
    }

    fn after_draw(&mut self, state: BellmanState, draw: OwnDraw, budget: HandChangeBudget) -> f64 {
        let state = state.after_known_draw(draw.tile);
        let decline = self.best_after_draw_discard(state, draw, budget);
        let Some(win) = draw.hand.win_on_draw(draw.tile, draw.flags) else {
            return decline;
        };
        let after_win = draw.hand.after_win(Some(draw.tile), win);
        let (next_public, gain) = self.planner.value_model.self_draw_transition(
            state.public,
            win.multiplier,
            win.shape_multiplier,
        );
        let continuation = self.opponent_cycle(
            BellmanState {
                hand: after_win,
                public: next_public,
                ..state
            },
            Continuation::repeat_cycle(),
            OpponentCycle::start(self.planner.actor, self.planner.actor.next()),
        );
        let take = gain + continuation;
        take.max(decline)
    }

    fn best_after_draw_discard(
        &mut self,
        state: BellmanState,
        draw: OwnDraw,
        budget: HandChangeBudget,
    ) -> f64 {
        let mut best = None::<(f64, HandProgress)>;
        for discard in mask_tiles(draw.hand.holding.discard_mask()) {
            let Some(next) = draw.hand.after_discard(discard) else {
                continue;
            };
            let changed = next != state.hand;
            let next_budget = if changed {
                if !self.admitted.contains(&next) {
                    continue;
                }
                let Some(next_budget) = budget.after_change(state.hand, next) else {
                    continue;
                };
                next_budget
            } else {
                budget
            };
            let value = self.after_discard(
                state.with_hand(next),
                discard,
                Continuation::Graph {
                    budget: next_budget,
                },
                DiscardContext::Simulated {
                    after_kong: draw.origin == PlannedDraw::Supplement,
                },
            );
            let progress = self.planner.progress(next);
            if best.is_none_or(|current| {
                value.total_cmp(&current.0).is_gt() || (value == current.0 && progress > current.1)
            }) {
                best = Some((value, progress));
            }
        }
        best.map_or_else(|| self.leaf_value(state), |(value, _)| value)
    }

    fn best_called_discard(
        &mut self,
        state: BellmanState,
        discard_mask: u32,
        budget: HandChangeBudget,
    ) -> Option<f64> {
        let mut candidates: Vec<_> = mask_tiles(discard_mask)
            .filter_map(|tile| state.hand.after_discard(tile).map(|hand| (tile, hand)))
            .collect();
        let root_shanten = candidates
            .iter()
            .map(|(_, hand)| hand.analysis().shanten)
            .min()?;
        if !state.hand.holding.has_won {
            candidates.retain(|(_, hand)| hand.analysis().shanten == root_shanten);
        }
        candidates
            .into_iter()
            .map(|(tile, hand)| {
                let value = self.after_discard(
                    state.with_hand(hand),
                    tile,
                    Continuation::Graph { budget },
                    DiscardContext::Simulated { after_kong: false },
                );
                (value, self.planner.progress(hand))
            })
            .max_by(|left, right| {
                left.0
                    .total_cmp(&right.0)
                    .then_with(|| left.1.cmp(&right.1))
            })
            .map(|(value, _)| value)
    }

    fn after_discard(
        &mut self,
        state: BellmanState,
        tile: Tile,
        continuation: Continuation,
        context: DiscardContext,
    ) -> f64 {
        let hazard = self.planner.hazards.map(|table| match context {
            DiscardContext::Root => table.immediate_discard(tile),
            DiscardContext::Simulated { after_kong } => {
                let draw_offset = state
                    .timeline
                    .normal_draws
                    .saturating_sub(u8::from(!after_kong));
                table
                    .future_discard_at(tile, draw_offset)
                    .unwrap_or_else(|| table.future_discard(tile))
            }
        });
        self.planner.value_model.discard_transition_value(
            state.public,
            hazard,
            context.payout_scale(),
            |next_public, next_actor| {
                self.opponent_cycle(
                    state.with_public(next_public),
                    continuation,
                    OpponentCycle::start(self.planner.actor, next_actor),
                )
            },
        )
    }

    fn opponent_cycle(
        &mut self,
        state: BellmanState,
        continuation: Continuation,
        cursor: OpponentCycle,
    ) -> f64 {
        if state.public.is_terminal() {
            return 0.0;
        }
        if cursor.is_complete(self.planner.actor) {
            return match continuation {
                Continuation::Graph { budget } => self.before_draw(state, budget),
                Continuation::Repeat { .. } => self.repeat_win_tail(state),
            };
        }
        if state.timeline.wall_draws >= self.planner.wall_remaining {
            return self
                .planner
                .value_model
                .wall_value(state.hand, state.public);
        }

        let key = CacheKey {
            hand: state.hand,
            public: state.public,
            turns_left: state.turns_left,
            budget: continuation.budget(),
            timeline: state.timeline,
            drawn: state.drawn,
            phase: EvaluationPhase::Opponent {
                active: cursor.active,
                events_left: cursor.events_left,
                continuation,
            },
        };
        if let Some(&value) = self.cache.get(&key) {
            return value;
        }

        let Some(table) = self.planner.hazards.and_then(|hazards| {
            hazards.opponent_turn_at(cursor.active, state.timeline.normal_draws)
        }) else {
            // A call turn does not consume a head-wall tile. Belief keeps
            // these events under a positive call index; consume the first
            // one when no ordinary draw table is available for this seat and
            // offset instead of silently advancing the wall clock.
            if let Some(value) = self.opponent_call_value(state, continuation, cursor, 1) {
                self.cache.insert(key, value);
                return value;
            }
            let value = if state.turns_left == 0 {
                // With no actor draw left, the symmetric discard prior has no
                // calibrated continuation value. Preserve the static policy
                // baseline and only advance the physical clock.
                self.opponent_cycle(
                    BellmanState {
                        timeline: state.timeline.after(PlannedDraw::Normal),
                        ..state
                    },
                    continuation,
                    cursor.after_turn(self.planner.actor, cursor.active.next()),
                )
            } else {
                self.fallback_opponent_cycle(state, continuation, cursor)
            };
            self.cache.insert(key, value);
            return value;
        };
        let samples = table.sample_count();
        if samples == 0 {
            let value = self.opponent_cycle(
                BellmanState {
                    timeline: state.timeline.after(PlannedDraw::Normal),
                    ..state
                },
                continuation,
                cursor.after_turn(self.planner.actor, cursor.active.next()),
            );
            self.cache.insert(key, value);
            return value;
        }
        let transitions: Vec<_> = table.transitions().collect();

        // The actor sees the discarded tile, but does not see hidden winners
        // until the response window has finished. Grouping by tile keeps the
        // Hu/Pass comparison outside the hidden winner sampling.
        let mut grouped =
            HashMap::<Tile, Vec<(usize, [Option<SampledWin>; crate::types::PLAYER_COUNT])>>::new();
        let mut expected = 0.0;
        let after_opponent_draw = BellmanState {
            timeline: state.timeline.after(PlannedDraw::Normal),
            ..state
        };
        for transition in transitions {
            match transition.event {
                OpponentTurnEvent::SelfDraw(win) => {
                    let payout_multiplier = win.payout_multiplier.saturating_mul(
                        if state.timeline.wall_draws + 1 == self.planner.wall_remaining {
                            2
                        } else {
                            1
                        },
                    );
                    let (next_public, gain) = self.planner.value_model.self_draw_by(
                        state.public,
                        cursor.active,
                        payout_multiplier,
                        win.shape_multiplier,
                    );
                    let continuation = if next_public.is_terminal() {
                        0.0
                    } else {
                        self.opponent_cycle(
                            after_opponent_draw.with_public(next_public),
                            continuation,
                            cursor.after_turn(self.planner.actor, cursor.active.next()),
                        )
                    };
                    expected += transition.count as f64 * (gain + continuation);
                }
                OpponentTurnEvent::Discard { tile, other_wins } => {
                    grouped
                        .entry(tile)
                        .or_default()
                        .push((transition.count, other_wins));
                }
            }
        }

        for (tile, outcomes) in grouped {
            let actor_win = continuation
                .permits_actor_win()
                .then(|| state.hand.with_draw(tile))
                .flatten()
                .and_then(|drawn| drawn.win_on_draw(tile, WinFlags::NONE));
            let mut pass_sum = 0.0;
            let mut hu_sum = 0.0;
            for (count, other_wins) in outcomes {
                let pass = self.resolve_opponent_discard(
                    after_opponent_draw,
                    cursor,
                    tile,
                    other_wins,
                    None,
                    continuation,
                );
                pass_sum += count as f64 * pass;
                let take = actor_win.map_or(pass, |win| {
                    self.resolve_opponent_discard(
                        after_opponent_draw,
                        cursor,
                        tile,
                        other_wins,
                        Some(win),
                        continuation,
                    )
                });
                hu_sum += count as f64 * take;
            }
            expected += if actor_win.is_some() {
                pass_sum.max(hu_sum)
            } else {
                pass_sum
            };
        }

        let value = expected / samples as f64;
        self.cache.insert(key, value);
        value
    }

    /// Evaluates one opponent turn when no posterior world table is available.
    ///
    /// The planner must not treat this case as a free clock advance: an
    /// opponent can discard the actor's winning tile or offer a useful meld.
    /// Hidden opponent choices are represented by the symmetric unknown-tile
    /// prior. Only response tiles are expanded individually; all other tiles
    /// share one passive continuation. This keeps zero-world search rollouts
    /// cheap while preserving the two events that materially affect the
    /// actor's value.
    fn fallback_opponent_cycle(
        &mut self,
        state: BellmanState,
        continuation: Continuation,
        cursor: OpponentCycle,
    ) -> f64 {
        let after_draw = BellmanState {
            timeline: state.timeline.after(PlannedDraw::Normal),
            ..state
        };
        let next_cursor = cursor.after_turn(self.planner.actor, cursor.active.next());
        if !continuation.permits_actor_win() {
            return self.opponent_cycle(after_draw, continuation, next_cursor);
        }
        let mut total_copies = 0_u16;
        let mut passive_copies = 0_u16;
        let mut weighted = 0.0;

        for tile in all_tiles() {
            let copies = self.planner.available_copies_in_state(state, tile);
            if copies == 0 {
                continue;
            }
            total_copies = total_copies.saturating_add(u16::from(copies));

            let actor_win = continuation
                .permits_actor_win()
                .then(|| state.hand.with_draw(tile))
                .flatten()
                .and_then(|drawn| drawn.win_on_draw(tile, WinFlags::NONE));
            if actor_win.is_none() {
                passive_copies = passive_copies.saturating_add(u16::from(copies));
                continue;
            }

            let empty_wins = [None; crate::types::PLAYER_COUNT];
            let pass = self.resolve_opponent_discard(
                after_draw,
                cursor,
                tile,
                empty_wins,
                None,
                continuation,
            );
            let value = actor_win.map_or(pass, |win| {
                pass.max(self.resolve_opponent_discard(
                    after_draw,
                    cursor,
                    tile,
                    empty_wins,
                    Some(win),
                    continuation,
                ))
            });
            weighted += f64::from(copies) * value;
        }

        if passive_copies != 0 {
            let passive = self.opponent_cycle(after_draw, continuation, next_cursor);
            weighted += f64::from(passive_copies) * passive;
        }

        if total_copies == 0 {
            self.opponent_cycle(after_draw, continuation, next_cursor)
        } else {
            weighted / f64::from(total_copies)
        }
    }

    fn opponent_call_value(
        &mut self,
        state: BellmanState,
        continuation: Continuation,
        cursor: OpponentCycle,
        call_index: u8,
    ) -> Option<f64> {
        let table = self.planner.hazards?.opponent_call_at(
            cursor.active,
            state.timeline.normal_draws,
            call_index,
        )?;
        let samples = table.sample_count();
        if samples == 0 {
            return None;
        }
        let transitions: Vec<_> = table.transitions().collect();
        let mut expected = 0.0;
        for transition in transitions {
            let value = match transition.event {
                OpponentTurnEvent::SelfDraw(win) => {
                    // A no-draw self-draw event can only be the replacement
                    // draw after a kong. The tail draw consumes one wall tile
                    // but does not advance the normal head offset.
                    let (next_public, gain) = self.planner.value_model.self_draw_by(
                        state.public,
                        cursor.active,
                        win.payout_multiplier,
                        win.shape_multiplier,
                    );
                    let next_state = BellmanState {
                        public: next_public,
                        timeline: state.timeline.after(PlannedDraw::Supplement),
                        ..state
                    };
                    if next_public.is_terminal() {
                        gain
                    } else {
                        gain + self.opponent_cycle(
                            next_state,
                            continuation,
                            cursor.after_turn(self.planner.actor, cursor.active.next()),
                        )
                    }
                }
                OpponentTurnEvent::Discard { tile, other_wins } => self.resolve_opponent_discard(
                    state.after_known_draw(tile),
                    cursor,
                    tile,
                    other_wins,
                    None,
                    continuation,
                ),
            };
            expected += transition.count as f64 * value;
        }
        Some(expected / samples as f64)
    }

    fn resolve_opponent_discard(
        &mut self,
        state: BellmanState,
        cursor: OpponentCycle,
        tile: Tile,
        other_wins: [Option<SampledWin>; crate::types::PLAYER_COUNT],
        actor_win: Option<crate::WinEvaluation>,
        continuation: Continuation,
    ) -> f64 {
        let source = cursor.active;
        let mut wins = other_wins;
        if let Some(win) = actor_win {
            wins[self.planner.actor.index()] = Some(SampledWin {
                payout_multiplier: win.multiplier,
                shape_multiplier: win.shape_multiplier,
            });
        }
        if wins.iter().any(Option::is_some) {
            let last_winner = seats_after(source)
                .filter(|seat| wins[seat.index()].is_some())
                .last()
                .expect("a non-empty winner set has a source-relative winner");
            let (next_public, gain) =
                self.planner
                    .value_model
                    .discard_winners_transition(state.public, source, wins);
            let next_hand = actor_win.map_or(state.hand, |win| {
                state
                    .hand
                    .with_draw(tile)
                    .expect("a winning tile can be added to the hand")
                    .after_win(Some(tile), win)
            });
            let next_value = if next_public.is_terminal() {
                0.0
            } else {
                let next_continuation = if actor_win.is_some() {
                    continuation.after_actor_win()
                } else {
                    continuation
                };
                self.opponent_cycle(
                    BellmanState {
                        hand: next_hand,
                        public: next_public,
                        ..state
                    },
                    next_continuation,
                    cursor.after_turn(self.planner.actor, last_winner.next()),
                )
            };
            return gain + next_value;
        }

        self.meld_response_value(state, cursor, tile, continuation)
    }

    fn meld_response_value(
        &mut self,
        state: BellmanState,
        cursor: OpponentCycle,
        tile: Tile,
        continuation: Continuation,
    ) -> f64 {
        let source = cursor.active;
        let pass = self.opponent_cycle(
            state,
            continuation,
            cursor.after_turn(self.planner.actor, source.next()),
        );
        let Continuation::Graph { budget } = continuation else {
            return pass;
        };
        if budget.transitions_left == 0 {
            // A zero-change rollout deliberately models only the current
            // discard/Hu response. Expanding every Pong/Kong discard here
            // creates a large tree without any legal hand-change budget.
            return pass;
        }
        let mut best = pass;

        if let Some(after_pong) = state.hand.holding.after_pong(tile, source) {
            let pong_hand = PlanningHand::new(
                after_pong,
                state.hand.holding.has_won,
                state.hand.max_win_multiplier,
            );
            if let Some(value) = self.best_called_discard(
                state.with_hand(pong_hand),
                pong_hand.holding.discard_mask(),
                budget,
            ) {
                best = best.max(value);
            }
        }
        if let Some(after_kong) = state.hand.holding.after_exposed_kong(tile, source) {
            let kong_hand = PlanningHand::new(
                after_kong,
                state.hand.holding.has_won,
                state.hand.max_win_multiplier,
            );
            let (next_public, immediate) = self
                .planner
                .value_model
                .exposed_kong_transition(state.public, source);
            best = best.max(
                immediate
                    + self.before_draw(
                        BellmanState {
                            hand: kong_hand,
                            public: next_public,
                            ..state
                        },
                        budget,
                    ),
            );
        }
        best
    }

    fn draw_win_flags(&self, draw: PlannedDraw, timeline: DrawTimeline) -> WinFlags {
        WinFlags {
            after_kong_draw: draw == PlannedDraw::Supplement,
            last_wall_tile: timeline.wall_draws + 1 == self.planner.wall_remaining,
            ..WinFlags::NONE
        }
    }
}

fn seats_after(source: Seat) -> impl Iterator<Item = Seat> {
    (1..PLAYER_COUNT as u8).map(move |offset| source.offset(offset))
}

/// Minimum number of hand-changing draws required before a hand can win.
///
/// A zero-shanten hand can win without replacing another tile. Keeping this
/// lower bound inside the transition relation ensures that optional shape
/// changes cannot consume progress which the remaining horizon still needs.
fn hand_change_distance(hand: PlanningHand) -> u8 {
    u8::try_from(hand.shanten().max(0)).unwrap_or_default()
}

/// Probability of drawing at least one of `hits` remaining copies in a
/// sample without replacement. The planner uses this only as a frontier
/// estimate; exact wall order remains hidden from the policy.
fn hit_probability(total: u16, hits: u16, draws: u8) -> f64 {
    if total == 0 || hits == 0 || draws == 0 {
        return 0.0;
    }
    let draws = u16::from(draws).min(total);
    let mut miss = 1.0;
    for drawn in 0..draws {
        let remaining = total - drawn;
        let misses = remaining.saturating_sub(hits);
        miss *= f64::from(misses) / f64::from(remaining);
    }
    (1.0 - miss).clamp(0.0, 1.0)
}

/// Number of future draws available to the planner actor at a frontier node.
///
/// A frontier node is evaluated before the actor's next draw. For a normal
/// draw, the actor receives the first tile and then one tile per complete
/// four-seat rotation, which is `ceil(W / 4)`. For a pending supplement draw,
/// one tail tile is guaranteed to the actor first; the remaining head-wall
/// tiles then provide `floor((W - 1) / 4)` ordinary opportunities.
fn actor_draw_opportunities(remaining_wall: u8, next_draw: PlannedDraw) -> u8 {
    match next_draw {
        PlannedDraw::Normal => remaining_wall.div_ceil(PLAYER_COUNT as u8),
        PlannedDraw::Supplement => {
            if remaining_wall == 0 {
                0
            } else {
                1 + (remaining_wall - 1) / PLAYER_COUNT as u8
            }
        }
    }
}

fn cap_frontier_delta(baseline: f64, value: f64, one_win_value: f64) -> f64 {
    let limit = one_win_value.abs().max(SCORE_UNIT as f64);
    baseline + (value - baseline).clamp(-limit, limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::hand::{DUMMY_MELD, Holding};
    use crate::types::{Seat, Suit};

    fn tile(suit: Suit, rank: u8) -> Tile {
        Tile::from_suit_rank(suit, rank - 1).expect("test tile is valid")
    }

    fn repeat_ready_hand() -> (PlanningHand, Tile) {
        let mut concealed = [0; TILE_KIND_COUNT];
        for (suit, rank) in [
            (Suit::Characters, 1),
            (Suit::Characters, 2),
            (Suit::Characters, 3),
            (Suit::Bamboo, 4),
            (Suit::Bamboo, 5),
            (Suit::Bamboo, 6),
            (Suit::Bamboo, 7),
            (Suit::Bamboo, 7),
            (Suit::Bamboo, 7),
            (Suit::Bamboo, 8),
            (Suit::Bamboo, 8),
            (Suit::Bamboo, 8),
            (Suit::Bamboo, 9),
            (Suit::Bamboo, 9),
        ] {
            concealed[tile(suit, rank).index()] += 1;
        }
        let winning_tile = tile(Suit::Bamboo, 9);
        let mut win_base = concealed;
        win_base[winning_tile.index()] -= 1;
        let hand = PlanningHand::new(
            Holding {
                concealed,
                locked: concealed,
                win_base,
                melds: [DUMMY_MELD; 4],
                meld_len: 0,
                missing: Some(Suit::Dots),
                has_won: true,
            },
            true,
            1,
        );
        assert!(
            hand.with_draw(winning_tile)
                .and_then(|drawn| drawn.win_on_draw(winning_tile, WinFlags::NONE))
                .is_some()
        );
        (hand, winning_tile)
    }

    #[test]
    fn hit_probability_matches_basic_without_replacement_cases() {
        assert_eq!(hit_probability(10, 0, 4), 0.0);
        assert_eq!(hit_probability(10, 10, 1), 1.0);
        let one = hit_probability(10, 1, 1);
        assert!((one - 0.1).abs() < 1e-12);
        let two = hit_probability(10, 2, 2);
        assert!((two - 1.0 + 8.0 / 10.0 * 7.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn actor_draw_opportunities_follow_rotation_and_replacement_rules() {
        // At an actor turn, the next normal draw is preceded by three
        // opponents. A pending supplement draw is immediate.
        assert_eq!(actor_draw_opportunities(0, PlannedDraw::Normal), 0);
        assert_eq!(actor_draw_opportunities(1, PlannedDraw::Normal), 1);
        assert_eq!(actor_draw_opportunities(3, PlannedDraw::Normal), 1);
        assert_eq!(actor_draw_opportunities(4, PlannedDraw::Normal), 1);
        assert_eq!(actor_draw_opportunities(5, PlannedDraw::Normal), 2);

        assert_eq!(actor_draw_opportunities(0, PlannedDraw::Supplement), 0);
        assert_eq!(actor_draw_opportunities(1, PlannedDraw::Supplement), 1);
        assert_eq!(actor_draw_opportunities(4, PlannedDraw::Supplement), 1);
        assert_eq!(actor_draw_opportunities(5, PlannedDraw::Supplement), 2);
    }

    #[test]
    fn post_win_frontier_uses_exact_repeat_waits() {
        let game = Game::new(8);
        let (hand, winning_tile) = repeat_ready_hand();
        let values = PublicValueModel::new(&game, Seat::EAST, None);
        let planner = HandGraphPlanner::new(&game, Seat::EAST, hand, None, &values, 8);
        let evaluator =
            BranchEvaluator::new(&planner, 8, &[(hand, HandChangeBudget::default())], false);
        let state = BellmanState {
            hand,
            public: PlanningPublicState::from_game(&game),
            turns_left: 0,
            timeline: DrawTimeline::default(),
            drawn: DrawnInventory::default(),
        };

        assert_ne!(
            hand.analysis().improving_tiles & (1 << winning_tile.index()),
            0
        );
        assert!(planner.available_copies_in_state(state, winning_tile) > 0);
        assert!(evaluator.frontier_value(state) > 0.0);
    }

    #[test]
    fn repeat_cycle_expands_at_most_one_discard_win() {
        let game = Game::new(8);
        let (hand, winning_tile) = repeat_ready_hand();
        let values = PublicValueModel::new(&game, Seat::EAST, None);
        let planner = HandGraphPlanner::new(&game, Seat::EAST, hand, None, &values, 8);
        let mut evaluator =
            BranchEvaluator::new(&planner, 8, &[(hand, HandChangeBudget::default())], false);
        let state = BellmanState {
            hand,
            public: PlanningPublicState::from_game(&game),
            turns_left: 0,
            timeline: DrawTimeline::default(),
            drawn: DrawnInventory::default(),
        };
        let cursor = OpponentCycle::start(Seat::EAST, Seat::EAST.next());
        let available = Continuation::repeat_cycle();
        let spent = available.after_actor_win();

        assert!(available.permits_actor_win());
        assert!(!spent.permits_actor_win());
        assert!(planner.available_copies_in_state(state, winning_tile) > 0);
        assert!(
            evaluator.fallback_opponent_cycle(state, available, cursor)
                > evaluator.fallback_opponent_cycle(state, spent, cursor)
        );
    }

    #[test]
    fn graph_can_follow_a_nonchanging_draw_then_a_win() {
        let game = Game::new(8);
        let mut counts = [0; TILE_KIND_COUNT];
        for rank in [1, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 9] {
            counts[tile(Suit::Characters, rank).index()] += 1;
        }
        let holding = Holding {
            concealed: counts,
            locked: [0; TILE_KIND_COUNT],
            win_base: [0; TILE_KIND_COUNT],
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: Some(Suit::Dots),
            has_won: false,
        };
        let hand = PlanningHand::new(holding, false, 0);
        let values = PublicValueModel::new(&game, Seat::EAST, None);
        let mut planner = HandGraphPlanner::new(&game, Seat::EAST, hand, None, &values, 1);
        // This unit test needs only a draw-discard self-loop and the later win.
        // Restricting the synthetic inventory keeps unrelated tile sequences
        // out of the state graph.
        planner.base_unknown = [0; TILE_KIND_COUNT];
        planner.base_unknown[tile(Suit::Characters, 1).index()] = 1;
        planner.base_unknown[tile(Suit::Characters, 9).index()] = 1;
        planner.wall_remaining = 8;

        let public = PlanningPublicState::from_game(&game);
        let one_draw = planner.value_before_draw(
            hand,
            public,
            PlanningHorizon::new(8, 1, PlannedDraw::Normal),
            0,
        );
        let two_draws = planner.value_before_draw(
            hand,
            public,
            PlanningHorizon::new(8, 2, PlannedDraw::Normal),
            0,
        );
        assert!(one_draw.is_finite());
        assert!(two_draws > one_draw);
    }

    #[test]
    fn hand_change_budget_spends_detours_only_without_shanten_progress() {
        let mut counts = [0; TILE_KIND_COUNT];
        for (suit, rank) in [
            (Suit::Characters, 1),
            (Suit::Characters, 1),
            (Suit::Characters, 1),
            (Suit::Characters, 2),
            (Suit::Characters, 3),
            (Suit::Characters, 4),
            (Suit::Bamboo, 5),
            (Suit::Bamboo, 5),
            (Suit::Bamboo, 6),
            (Suit::Bamboo, 7),
            (Suit::Characters, 8),
            (Suit::Characters, 9),
            (Suit::Bamboo, 2),
        ] {
            counts[tile(suit, rank).index()] += 1;
        }
        let root = PlanningHand::new(
            Holding {
                concealed: counts,
                locked: [0; TILE_KIND_COUNT],
                win_base: [0; TILE_KIND_COUNT],
                melds: [DUMMY_MELD; 4],
                meld_len: 0,
                missing: Some(Suit::Dots),
                has_won: false,
            },
            false,
            0,
        );
        assert_eq!(hand_change_distance(root), 1);

        let mut improving = None;
        let mut same_distance = None;
        for draw in all_tiles() {
            let Some(drawn) = root.with_draw(draw) else {
                continue;
            };
            for discard in mask_tiles(drawn.holding.discard_mask()) {
                let Some(next) = drawn.after_discard(discard) else {
                    continue;
                };
                if next == root {
                    continue;
                }
                match hand_change_distance(next).cmp(&hand_change_distance(root)) {
                    std::cmp::Ordering::Less => {
                        improving.get_or_insert(next);
                    }
                    std::cmp::Ordering::Equal => {
                        same_distance.get_or_insert(next);
                    }
                    std::cmp::Ordering::Greater => {}
                }
            }
        }
        let improving = improving.expect("the one-shanten hand has an improving replacement");
        let same_distance = same_distance.expect("the hand also has a shape-only replacement");

        let direct = HandChangeBudget {
            transitions_left: 1,
            detours_left: 0,
        };
        assert_eq!(
            direct.after_change(root, improving),
            Some(HandChangeBudget::default())
        );
        assert_eq!(direct.after_change(root, same_distance), None);

        let with_detour = HandChangeBudget {
            transitions_left: 1,
            detours_left: 1,
        };
        assert_eq!(
            with_detour.after_change(root, same_distance),
            Some(HandChangeBudget::default())
        );
    }

    #[test]
    fn zero_horizon_returns_a_legal_discard() {
        let game = Game::new(9);
        let holding = Holding::from_game(&game, Seat::EAST);
        let hand = PlanningHand::new(holding, false, 0);
        let values = PublicValueModel::new(&game, Seat::EAST, None);
        let planner = HandGraphPlanner::new(&game, Seat::EAST, hand, None, &values, 1_024);

        let plan = planner
            .best_discard(
                hand,
                hand.holding.discard_mask(),
                PlanningPublicState::from_game(&game),
                PlanningHorizon::new(1, 0, PlannedDraw::Normal),
                0,
            )
            .expect("the hand has a discard");
        assert_ne!(hand.holding.discard_mask() & (1 << plan.tile.index()), 0);
        assert!(plan.value.is_finite());
    }

    #[test]
    fn planning_horizon_clamps_to_available_draws() {
        assert_eq!(
            PlanningHorizon::new(9, 8, PlannedDraw::Normal),
            PlanningHorizon {
                draws: 2,
                reaches_wall: false,
            }
        );
        assert_eq!(
            PlanningHorizon::new(8, 8, PlannedDraw::Normal),
            PlanningHorizon {
                draws: 2,
                reaches_wall: true,
            }
        );
    }

    #[test]
    fn planning_horizon_uses_physical_wall_consumption() {
        assert!(!PlanningHorizon::new(3, 0, PlannedDraw::Normal).reaches_wall);
        assert!(!PlanningHorizon::new(9, 2, PlannedDraw::Normal).reaches_wall);
        assert!(PlanningHorizon::new(8, 2, PlannedDraw::Normal).reaches_wall);

        assert!(!PlanningHorizon::new(3, 1, PlannedDraw::Supplement).reaches_wall);
        assert!(PlanningHorizon::new(5, 2, PlannedDraw::Supplement).reaches_wall);
        assert!(PlanningHorizon::new(1, 1, PlannedDraw::Supplement).reaches_wall);
    }

    #[test]
    fn initial_live_wall_gives_thirteen_normal_or_fourteen_supplement_draws() {
        let game = Game::new(8);
        assert_eq!(game.wall_remaining(), 55);
        let normal = PlanningHorizon::new(game.wall_remaining(), 32, PlannedDraw::Normal);
        assert_eq!(normal.draws, 13);
        assert!(!normal.reaches_wall);
        let supplement = PlanningHorizon::new(game.wall_remaining(), 32, PlannedDraw::Supplement);
        assert_eq!(supplement.draws, 14);
        assert!(!supplement.reaches_wall);
    }

    #[test]
    fn truncated_horizon_does_not_settle_wall() {
        let game = Game::new(8);
        let hand = PlanningHand::new(Holding::from_game(&game, Seat::EAST), false, 0);
        let values = PublicValueModel::new(&game, Seat::EAST, None);
        let planner = HandGraphPlanner::new(&game, Seat::EAST, hand, None, &values, 8);
        let evaluator =
            BranchEvaluator::new(&planner, 8, &[(hand, HandChangeBudget::default())], false);

        let state = BellmanState {
            hand,
            public: PlanningPublicState::from_game(&game),
            turns_left: 0,
            timeline: DrawTimeline::default(),
            drawn: DrawnInventory::default(),
        };
        let value = evaluator.leaf_value(state);
        let wall_value = values.wall_value(hand, PlanningPublicState::from_game(&game));
        assert!(value.is_finite());
        assert!((value - wall_value).abs() > 1e-6);
    }

    #[test]
    fn discarded_simulated_draw_stays_consumed() {
        let game = Game::new(8);
        let hand = PlanningHand::new(Holding::from_game(&game, Seat::EAST), false, 0);
        let values = PublicValueModel::new(&game, Seat::EAST, None);
        let planner = HandGraphPlanner::new(&game, Seat::EAST, hand, None, &values, 8);
        let state = BellmanState {
            hand,
            public: PlanningPublicState::from_game(&game),
            turns_left: 2,
            timeline: DrawTimeline::default(),
            drawn: DrawnInventory::default(),
        };
        let draw_tile = all_tiles()
            .find(|&tile| {
                planner.available_copies_in_state(state, tile) > 0
                    && hand
                        .with_draw(tile)
                        .and_then(|drawn| drawn.after_discard(tile))
                        .is_some_and(|returned| returned == hand)
            })
            .expect("the initial information set has a drawable tile");
        let before = planner.available_copies_in_state(state, draw_tile);
        let returned = hand
            .with_draw(draw_tile)
            .and_then(|drawn| drawn.after_discard(draw_tile))
            .expect("the simulated draw can be discarded");
        let after_discard = state.after_known_draw(draw_tile).with_hand(returned);

        assert_eq!(returned, hand);
        assert_eq!(
            planner.available_copies_in_state(after_discard, draw_tile),
            before - 1
        );
    }

    #[test]
    fn opponent_cycle_consumes_all_three_turns_before_actor_draws() {
        let actor = Seat::EAST;
        let south = actor.next();
        let west = south.next();
        let north = west.next();
        let start = OpponentCycle::start(actor, south);
        assert_eq!(start.events_left, 3);

        let after_south = start.after_turn(actor, west);
        assert_eq!(after_south.events_left, 2);
        let after_west = after_south.after_turn(actor, north);
        assert_eq!(after_west.events_left, 1);
        let after_north = after_west.after_turn(actor, actor);
        assert!(after_north.is_complete(actor));
        let timeline = DrawTimeline::default()
            .after(PlannedDraw::Normal)
            .after(PlannedDraw::Normal)
            .after(PlannedDraw::Normal);
        assert_eq!(timeline.wall_draws, 3);
        assert_eq!(timeline.normal_draws, 3);
    }

    #[test]
    fn draw_timeline_separates_head_and_tail_consumption() {
        let timeline = DrawTimeline::default()
            .after(PlannedDraw::Supplement)
            .after(PlannedDraw::Normal)
            .after(PlannedDraw::Supplement);
        assert_eq!(timeline.wall_draws, 3);
        assert_eq!(timeline.normal_draws, 1);
    }

    #[test]
    fn actor_discard_win_restarts_rotation_from_their_next_seat() {
        let actor = Seat::EAST;
        let source = actor.next();
        let before_response = OpponentCycle::start(actor, source);

        // The current opponent drew once, then discarded into the actor's Hu.
        let after_response = before_response.after_turn(actor, actor.next());
        assert_eq!(after_response.active, actor.next());
        assert_eq!(after_response.events_left, 3);
    }

    #[test]
    fn self_draw_terminal_state_is_propagated_to_the_continuation() {
        let game = Game::new(8);
        let mut counts = [0; TILE_KIND_COUNT];
        for rank in [1, 1, 1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 9] {
            counts[tile(Suit::Characters, rank).index()] += 1;
        }
        let holding = Holding {
            concealed: counts,
            locked: [0; TILE_KIND_COUNT],
            win_base: [0; TILE_KIND_COUNT],
            melds: [DUMMY_MELD; 4],
            meld_len: 0,
            missing: Some(Suit::Dots),
            has_won: false,
        };
        let hand = PlanningHand::new(holding, false, 0);
        let winning_tile = tile(Suit::Characters, 9);
        let drawn = hand.with_draw(winning_tile).expect("the tile can be drawn");
        let values = PublicValueModel::new(&game, Seat::EAST, None);
        let planner = HandGraphPlanner::new(&game, Seat::EAST, hand, None, &values, 128);
        let mut evaluator =
            BranchEvaluator::new(&planner, 128, &[(hand, HandChangeBudget::default())], false);
        let public = PlanningPublicState {
            balances: [10_000, 100, 100, 100],
            max_win_multipliers: [0; crate::types::PLAYER_COUNT],
            next_draw: PlannedDraw::Normal,
        };

        assert_eq!(
            evaluator.after_draw(
                BellmanState {
                    hand,
                    public,
                    turns_left: 0,
                    timeline: DrawTimeline {
                        wall_draws: 1,
                        normal_draws: 1,
                    },
                    drawn: DrawnInventory::default(),
                },
                OwnDraw {
                    hand: drawn,
                    tile: winning_tile,
                    origin: PlannedDraw::Normal,
                    flags: WinFlags::NONE,
                },
                HandChangeBudget::default(),
            ),
            300.0
        );
    }
}
