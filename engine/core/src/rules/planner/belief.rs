use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::game::{Game, Phase, SCORE_UNIT, TerminationReason};
use crate::rules::hand::Holding;
use crate::types::{PLAYER_COUNT, Seat, TILE_KIND_COUNT, Tile};
use crate::{Action, BeliefResidualError, BeliefResidualEvaluator};
#[cfg(any(feature = "belief-training", test))]
use crate::{BeliefPublicFeatures, BeliefRootCandidate};
use crate::{WinFlags, analyze_shanten, evaluate_max_wait, evaluate_win};

use super::history::PublicHistory;
use super::quality::{HandPotential, history_likelihood_ratio};

const POSTERIOR_CANDIDATES_PER_WORLD: usize = 4;
const MAX_PROJECTION_ACTIONS: usize = 1_024;

pub(super) struct RootBeliefSampler<'a> {
    root: &'a Game,
    actor: Seat,
    observation: u64,
    legal: crate::ActionMask,
    base_seed: u64,
    history: PublicHistory,
}

pub(super) struct RootBeliefParticle {
    pub(super) game: Game,
    pub(super) log_likelihood: f64,
    handwritten_log_likelihood: f64,
}

impl RootBeliefParticle {
    pub(super) fn use_handwritten_weight(&mut self) {
        self.log_likelihood = self.handwritten_log_likelihood;
    }

    #[cfg(test)]
    pub(super) fn new_for_test(
        game: Game,
        log_likelihood: f64,
        handwritten_log_likelihood: f64,
    ) -> Self {
        Self {
            game,
            log_likelihood,
            handwritten_log_likelihood,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) enum RootBeliefMode {
    #[default]
    Posterior,
    #[cfg(any(feature = "planner-analysis", test))]
    Uniform,
    #[cfg(any(feature = "planner-analysis", test))]
    OracleHidden,
}

impl<'a> RootBeliefSampler<'a> {
    /// Creates a stable maximum-entropy sampler for one root information set.
    pub(super) fn new(root: &'a Game, actor: Seat, base_seed: u64) -> Option<Self> {
        if root.phase() != Phase::Turn || root.decision()?.actor != actor {
            return None;
        }
        Some(Self {
            root,
            actor,
            observation: super::public_state_hash(root, actor),
            legal: root.legal_action_mask()?,
            base_seed,
            history: PublicHistory::from_game(root),
        })
    }

    /// Samples one stable particle. Existing IDs never change when the search
    /// budget grows.
    pub(super) fn sample(&self, id: u64) -> Option<Game> {
        let seed = mix_seed(self.base_seed.wrapping_add(id));
        let state = self.root.resample_information_set(seed).ok()?;
        (state.phase() == Phase::Turn
            && state
                .decision()
                .is_some_and(|decision| decision.actor == self.actor)
            && state.legal_action_mask() == Some(self.legal)
            && super::public_state_hash(&state, self.actor) == self.observation)
            .then_some(state)
    }

    /// Samples one stable root-search particle under the selected diagnostic
    /// belief. Search normalizes log-likelihoods after incomplete paired
    /// evaluations have been discarded.
    pub(super) fn sample_weighted(
        &self,
        id: u64,
        mode: RootBeliefMode,
    ) -> Option<RootBeliefParticle> {
        let (game, log_likelihood) = match mode {
            RootBeliefMode::Posterior => {
                let game = self.sample(id)?;
                let log_likelihood = world_log_likelihood_ratio(&game, self.actor, &self.history);
                (game, log_likelihood)
            }
            #[cfg(any(feature = "planner-analysis", test))]
            RootBeliefMode::Uniform => (self.sample(id)?, 0.0),
            #[cfg(any(feature = "planner-analysis", test))]
            RootBeliefMode::OracleHidden => {
                let seed = mix_seed(self.base_seed.wrapping_add(id));
                (self.root.resample_live_wall(seed), 0.0)
            }
        };
        Some(RootBeliefParticle {
            game,
            log_likelihood,
            handwritten_log_likelihood: log_likelihood,
        })
    }

    /// Builds one stable prefix of weighted root particles.
    pub(super) fn sample_batch(
        &self,
        count: usize,
        mode: RootBeliefMode,
    ) -> Vec<RootBeliefParticle> {
        // A public state can reject a determinization when its missing-suit
        // constraints have no feasible allocation. Keep the normal path at
        // the requested budget, and only spend extra sampling work when a
        // rejected draw actually leaves the batch short.
        let mut particles: Vec<_> = (0..count)
            .into_par_iter()
            .filter_map(|id| self.sample_weighted(id as u64, mode))
            .collect();
        let missing = count.saturating_sub(particles.len());
        if missing != 0 {
            let extra_attempts = missing.saturating_mul(4);
            let extra: Vec<_> = (count as u64..count as u64 + extra_attempts as u64)
                .into_par_iter()
                .filter_map(|id| self.sample_weighted(id, mode))
                .collect();
            particles.extend(extra);
        }
        particles.truncate(count);
        particles
    }

    /// Applies one learned residual evaluation to one or more particle
    /// streams. Candidate and particle order remain unchanged.
    pub(super) fn apply_residuals(
        &self,
        evaluator: &dyn BeliefResidualEvaluator,
        batches: &mut [&mut [RootBeliefParticle]],
    ) -> Result<(), BeliefResidualError> {
        let public = self.root.belief_public_features(self.actor);
        let candidates: Vec<_> = batches
            .iter()
            .flat_map(|batch| batch.iter())
            .map(|particle| particle.game.belief_candidate_features(self.actor))
            .collect();
        let mut residuals = vec![0.0_f32; candidates.len()];
        evaluator.evaluate_residuals(&public, &candidates, &mut residuals)?;
        for (index, (particle, residual)) in batches
            .iter_mut()
            .flat_map(|batch| batch.iter_mut())
            .zip(residuals)
            .enumerate()
        {
            if !residual.is_finite() {
                return Err(BeliefResidualError::NonFinite { index });
            }
            particle.log_likelihood += f64::from(residual);
        }
        Ok(())
    }
}

impl Game {
    /// Returns public belief features for the current turn root.
    #[cfg(any(feature = "belief-training", test))]
    pub fn belief_root_public_features(&self) -> Option<BeliefPublicFeatures> {
        let decision = self.decision()?;
        (decision.phase == Phase::Turn).then(|| self.belief_public_features(decision.actor))
    }

    /// Returns the authoritative hidden allocation and its hand-written
    /// posterior offset. This method is intended for analysis and training
    /// labels, not for deployed policy input.
    #[cfg(any(feature = "belief-training", test))]
    pub fn belief_root_candidate(&self) -> Option<BeliefRootCandidate> {
        let decision = self.decision()?;
        if decision.phase != Phase::Turn {
            return None;
        }
        let history = PublicHistory::from_game(self);
        Some(BeliefRootCandidate {
            features: self.belief_candidate_features(decision.actor),
            handwritten_log_weight: world_log_likelihood_ratio(self, decision.actor, &history),
        })
    }

    /// Samples a stable prefix from the current turn information-set proposal.
    /// Existing proposal IDs do not change when `count` grows.
    #[cfg(any(feature = "belief-training", test))]
    pub fn belief_root_proposals(
        &self,
        base_seed: u64,
        count: usize,
    ) -> Option<Vec<BeliefRootCandidate>> {
        let decision = self.decision()?;
        let sampler = RootBeliefSampler::new(self, decision.actor, base_seed)?;
        let particles = sampler.sample_batch(count, RootBeliefMode::Posterior);
        Some(
            particles
                .into_iter()
                .map(|particle| BeliefRootCandidate {
                    features: particle.game.belief_candidate_features(decision.actor),
                    handwritten_log_weight: particle.log_likelihood,
                })
                .collect(),
        )
    }
}

/// One winner in an information-set hazard sample.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct SampledWin {
    pub(super) payout_multiplier: u32,
    pub(super) shape_multiplier: u32,
}

/// One exact joint response atom and its sample multiplicity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct HazardAtom {
    pub(super) count: usize,
    pub(super) wins: [Option<SampledWin>; PLAYER_COUNT],
}

impl HazardAtom {
    #[cfg(test)]
    pub(super) fn winner_mask(self) -> u8 {
        self.wins
            .iter()
            .enumerate()
            .fold(0_u8, |mask, (index, win)| {
                mask | (u8::from(win.is_some()) << index)
            })
    }
}

/// Information-set distribution of immediate responses to one tile.
///
/// Atoms retain exact payout and persistent shape multipliers. Balance caps
/// are deliberately not applied here because a Bellman branch can reach the
/// same discard with balances different from those at the root decision.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct DiscardHazard {
    samples: usize,
    atoms: BTreeMap<[Option<SampledWin>; PLAYER_COUNT], usize>,
}

impl DiscardHazard {
    pub(super) const fn sample_count(&self) -> usize {
        self.samples
    }

    pub(super) fn transitions(&self) -> impl Iterator<Item = HazardAtom> + '_ {
        self.atoms
            .iter()
            .map(|(&wins, &count)| HazardAtom { count, wins })
    }

    #[cfg(test)]
    pub(super) fn any_win_probability(&self) -> f64 {
        self.probability(
            self.atoms
                .iter()
                .filter(|(wins, _)| wins.iter().any(Option::is_some))
                .map(|(_, &count)| count)
                .sum(),
        )
    }

    #[cfg(test)]
    pub(super) fn multiple_win_probability(&self) -> f64 {
        self.probability(
            self.atoms
                .iter()
                .filter(|(wins, _)| wins.iter().filter(|win| win.is_some()).count() >= 2)
                .map(|(_, &count)| count)
                .sum(),
        )
    }

    /// Expected score paid after applying the engine's insufficient-funds cap.
    pub(super) fn expected_loss_points(&self, actor: Seat, score: i64) -> f64 {
        self.expected_payments(actor, score)[actor.index()]
    }

    /// Expected score requested by all winners before insufficient-funds caps.
    #[cfg(test)]
    pub(super) fn expected_nominal_loss_points(&self) -> f64 {
        if self.samples == 0 {
            return 0.0;
        }
        let requested: u128 = self
            .atoms
            .iter()
            .map(|(wins, &count)| {
                let points: u128 = wins
                    .iter()
                    .flatten()
                    .map(|win| u128::from(win.payout_multiplier) * SCORE_UNIT as u128)
                    .sum();
                points * count as u128
            })
            .sum();
        requested as f64 / self.samples as f64
    }

    #[cfg(test)]
    pub(super) fn opponent_win_probability(&self, opponent: Seat) -> f64 {
        self.probability(
            self.atoms
                .iter()
                .filter(|(wins, _)| wins[opponent.index()].is_some())
                .map(|(_, &count)| count)
                .sum(),
        )
    }

    #[cfg(test)]
    pub(super) fn opponent_conditional_multiplier(&self, opponent: Seat) -> Option<f64> {
        let (samples, multiplier_sum) = self
            .atoms
            .iter()
            .filter_map(|(wins, &count)| {
                wins[opponent.index()].map(|win| (count, win.payout_multiplier))
            })
            .fold((0_usize, 0_u128), |(samples, sum), (count, multiplier)| {
                (
                    samples + count,
                    sum + u128::from(multiplier) * count as u128,
                )
            });
        (samples != 0).then(|| multiplier_sum as f64 / samples as f64)
    }

    pub(super) fn opponent_expected_loss_points(
        &self,
        actor: Seat,
        opponent: Seat,
        score: i64,
    ) -> f64 {
        self.expected_payments(actor, score)[opponent.index()]
    }

    #[cfg(test)]
    fn probability(&self, matching_samples: usize) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            matching_samples as f64 / self.samples as f64
        }
    }

    fn expected_payments(&self, actor: Seat, score: i64) -> [f64; PLAYER_COUNT] {
        if self.samples == 0 {
            return [0.0; PLAYER_COUNT];
        }
        let mut payment_sums = [0_u128; PLAYER_COUNT];
        for atom in self.transitions() {
            let mut available = score.max(0) as u64;
            let mut total = 0_u64;
            for offset in 1..PLAYER_COUNT as u8 {
                let winner = actor.offset(offset);
                let Some(win) = atom.wins[winner.index()] else {
                    continue;
                };
                let requested = u64::from(win.payout_multiplier) * SCORE_UNIT as u64;
                let payment = requested.min(available);
                available -= payment;
                total += payment;
                payment_sums[winner.index()] += u128::from(payment) * atom.count as u128;
            }
            payment_sums[actor.index()] += u128::from(total) * atom.count as u128;
        }
        payment_sums.map(|sum| sum as f64 / self.samples as f64)
    }

    fn observe(&mut self, wins: [Option<SampledWin>; PLAYER_COUNT]) {
        self.samples += 1;
        *self.atoms.entry(wins).or_default() += 1;
    }
}

/// One sampled public event during an opponent's projected turn.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum OpponentTurnEvent {
    SelfDraw(SampledWin),
    Discard {
        tile: Tile,
        other_wins: [Option<SampledWin>; PLAYER_COUNT],
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct WeightedOpponentTurn {
    pub(super) count: usize,
    pub(super) event: OpponentTurnEvent,
}

/// Posterior distribution of one opponent draw-and-action event.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct OpponentTurnTable {
    samples: usize,
    events: BTreeMap<OpponentTurnEvent, usize>,
}

/// Identifies one future normal draw without assuming a fixed seat order.
///
/// `draw_offset == 0` is the first tile drawn from the head of the live wall
/// after the root decision. Kongs draw from the tail and therefore do not
/// advance this offset. Keeping the source explicit is necessary because a
/// Hu changes the next actor.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct OpponentTurnKey {
    pub(super) source: Seat,
    pub(super) draw_offset: u8,
    /// Zero for a normal draw. Positive values order no-draw call turns before
    /// the normal draw at `draw_offset`.
    pub(super) call_index: u8,
}

/// Marginal event tables indexed by their shared live-wall position.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct OpponentTurnTimeline {
    turns: BTreeMap<OpponentTurnKey, OpponentTurnTable>,
}

impl OpponentTurnTimeline {
    fn get(&self, source: Seat, draw_offset: u8) -> Option<&OpponentTurnTable> {
        self.get_call(source, draw_offset, 0)
    }

    fn get_call(
        &self,
        source: Seat,
        draw_offset: u8,
        call_index: u8,
    ) -> Option<&OpponentTurnTable> {
        self.turns.get(&OpponentTurnKey {
            source,
            draw_offset,
            call_index,
        })
    }

    fn observe(&mut self, key: OpponentTurnKey, event: OpponentTurnEvent) {
        self.turns.entry(key).or_default().observe(event);
    }
}

/// Tracks head-wall positions independently from the player who receives one.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct FutureDrawClock {
    next_offset: u8,
    pending: [Option<u8>; PLAYER_COUNT],
    calls_before_next_draw: u8,
}

impl FutureDrawClock {
    fn observe_draw(&mut self, player: Seat, replacement: bool) {
        if replacement {
            return;
        }
        self.pending[player.index()] = Some(self.next_offset);
        self.next_offset = self.next_offset.saturating_add(1);
        self.calls_before_next_draw = 0;
    }

    fn pending_draw(&self, player: Seat) -> Option<u8> {
        self.pending[player.index()]
    }

    fn complete_turn(&mut self, player: Seat) -> OpponentTurnKey {
        if let Some(draw_offset) = self.pending[player.index()].take() {
            OpponentTurnKey {
                source: player,
                draw_offset,
                call_index: 0,
            }
        } else {
            self.calls_before_next_draw = self.calls_before_next_draw.saturating_add(1);
            OpponentTurnKey {
                source: player,
                draw_offset: self.next_offset,
                call_index: self.calls_before_next_draw,
            }
        }
    }
}

type WorldWinTable = [[Option<SampledWin>; PLAYER_COUNT]; TILE_KIND_COUNT];

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct DiscardHazardTimeline {
    turns: BTreeMap<u8, [DiscardHazard; TILE_KIND_COUNT]>,
}

impl DiscardHazardTimeline {
    fn get(&self, tile: Tile, draw_offset: u8) -> Option<&DiscardHazard> {
        self.turns
            .get(&draw_offset)
            .map(|hazards| &hazards[tile.index()])
    }

    fn observe(&mut self, draw_offset: u8, wins: &WorldWinTable) {
        let hazards = self
            .turns
            .entry(draw_offset)
            .or_insert_with(|| core::array::from_fn(|_| DiscardHazard::default()));
        for tile in all_tiles() {
            hazards[tile.index()].observe(wins[tile.index()]);
        }
    }
}

impl OpponentTurnTable {
    pub(super) const fn sample_count(&self) -> usize {
        self.samples
    }

    pub(super) fn transitions(&self) -> impl Iterator<Item = WeightedOpponentTurn> + '_ {
        self.events
            .iter()
            .map(|(&event, &count)| WeightedOpponentTurn { count, event })
    }

    fn observe(&mut self, event: OpponentTurnEvent) {
        self.samples += 1;
        *self.events.entry(event).or_default() += 1;
    }
}

/// One player's status in a sampled end-of-wall scenario.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum SampledWallOutcome {
    Flower,
    Ready(u32),
    Won(u32),
    Other,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct WeightedWallScenario {
    pub(super) count: usize,
    pub(super) outcomes: [SampledWallOutcome; PLAYER_COUNT],
}

/// Joint projected hidden-hand statuses from posterior worlds.
///
/// Keeping all opponents in one atom preserves correlations and nonlinear
/// balance caps during wall settlement.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct WallScenarioTable {
    scenarios: BTreeMap<[SampledWallOutcome; PLAYER_COUNT], usize>,
}

impl WallScenarioTable {
    pub(super) fn transitions(&self) -> impl Iterator<Item = WeightedWallScenario> + '_ {
        self.scenarios
            .iter()
            .map(|(&outcomes, &count)| WeightedWallScenario { count, outcomes })
    }

    fn observe(&mut self, world: &Game, actor: Seat) {
        let outcomes = Seat::ALL.map(|seat| {
            if seat == actor {
                SampledWallOutcome::Other
            } else {
                sampled_wall_outcome(world, seat)
            }
        });
        *self.scenarios.entry(outcomes).or_default() += 1;
    }
}

fn sampled_wall_outcome(world: &Game, seat: Seat) -> SampledWallOutcome {
    let holding = Holding::from_game(world, seat);
    if holding.suit_count() == 3 {
        return SampledWallOutcome::Flower;
    }

    let wait_multiplier = evaluate_max_wait(&holding.concealed, holding.melds(), holding.missing)
        .map_or(0, |wait| wait.evaluation.multiplier);
    if wait_multiplier != 0 {
        SampledWallOutcome::Ready(wait_multiplier.max(world.max_win_multiplier(seat)).max(1))
    } else if world.has_won(seat) {
        SampledWallOutcome::Won(world.max_win_multiplier(seat).max(1))
    } else {
        SampledWallOutcome::Other
    }
}

/// Hazard estimates for tile responses in the current information set.
///
/// Immediate and future discards use separate tables because event flags from
/// the root decision do not apply to later ordinary discards. Both discard
/// tables cover every tile kind. Added-kong hazards remain limited to actions
/// that are legal at the root decision.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct HazardTable {
    added_kong_mask: u32,
    immediate_discards: [DiscardHazard; TILE_KIND_COUNT],
    future_discards: [DiscardHazard; TILE_KIND_COUNT],
    future_discard_timeline: DiscardHazardTimeline,
    added_kongs: [DiscardHazard; TILE_KIND_COUNT],
    opponent_turns: OpponentTurnTimeline,
    wall_scenarios: WallScenarioTable,
}

impl HazardTable {
    /// Returns the hazard under the root decision's discard event flags.
    pub(super) fn immediate_discard(&self, tile: Tile) -> &DiscardHazard {
        &self.immediate_discards[tile.index()]
    }

    /// Returns the hazard for an ordinary discard reached by a future branch.
    pub(super) fn future_discard(&self, tile: Tile) -> &DiscardHazard {
        &self.future_discards[tile.index()]
    }

    /// Returns discard responses after the actor receives a specified future
    /// normal draw. The table uses projected opponent hands at that offset.
    pub(super) fn future_discard_at(&self, tile: Tile, draw_offset: u8) -> Option<&DiscardHazard> {
        self.future_discard_timeline.get(tile, draw_offset)
    }

    pub(super) fn added_kong(&self, tile: Tile) -> Option<&DiscardHazard> {
        (self.added_kong_mask & (1 << tile.index()) != 0).then_some(&self.added_kongs[tile.index()])
    }

    /// Returns a future event table by live-wall position and actual actor.
    pub(super) fn opponent_turn_at(
        &self,
        source: Seat,
        draw_offset: u8,
    ) -> Option<&OpponentTurnTable> {
        self.opponent_turns.get(source, draw_offset)
    }

    /// Returns a no-draw turn after a call at the same live-wall offset.
    /// `call_index == 0` is the ordinary draw turn and is exposed through
    /// `opponent_turn_at`; positive indices identify consecutive Pong/Kong
    /// response turns before the next normal draw.
    pub(super) fn opponent_call_at(
        &self,
        source: Seat,
        draw_offset: u8,
        call_index: u8,
    ) -> Option<&OpponentTurnTable> {
        if call_index == 0 {
            None
        } else {
            self.opponent_turns
                .get_call(source, draw_offset, call_index)
        }
    }

    pub(super) const fn wall_scenarios(&self) -> &WallScenarioTable {
        &self.wall_scenarios
    }
}

/// Samples hidden hands and estimates root and future response hazards.
///
/// Sampling is all-or-nothing. Returning a partial table would silently change
/// the requested distribution and make comparisons between actions biased.
pub(super) fn estimate(
    game: &Game,
    actor: Seat,
    worlds: usize,
    base_seed: u64,
) -> Option<HazardTable> {
    estimate_with_scope(game, actor, worlds, base_seed, true)
}

/// Samples only hazards caused directly by the root action.
///
/// A shallow Bellman horizon has no need for a complete projected opponent
/// timeline. Keeping future tables empty lets deeper nodes use the symmetric
/// fallback model without multiplying every node by the sampled world count.
pub(super) fn estimate_immediate(
    game: &Game,
    actor: Seat,
    worlds: usize,
    base_seed: u64,
) -> Option<HazardTable> {
    estimate_with_scope(game, actor, worlds, base_seed, false)
}

fn estimate_with_scope(
    game: &Game,
    actor: Seat,
    worlds: usize,
    base_seed: u64,
    include_future: bool,
) -> Option<HazardTable> {
    if worlds == 0 || game.phase() != Phase::Turn || game.decision()?.actor != actor {
        return None;
    }
    let legal = game.legal_actions()?;
    let added_kong_mask = legal.added_kong_mask;
    let mut immediate_discards = core::array::from_fn(|_| DiscardHazard::default());
    let mut future_discards = core::array::from_fn(|_| DiscardHazard::default());
    let mut future_discard_timeline = DiscardHazardTimeline::default();
    let mut added_kongs = core::array::from_fn(|_| DiscardHazard::default());
    let mut opponent_turns = OpponentTurnTimeline::default();
    let mut wall_scenarios = WallScenarioTable::default();

    let history = PublicHistory::from_game(game);
    let immediate_flags = discard_win_flags(game, actor);
    let sampled_worlds = posterior_worlds(game, actor, &history, worlds, base_seed)?;
    for sampled in sampled_worlds {
        let future_wins = world_win_table(&sampled, actor, WinFlags::NONE);
        let immediate_wins = (immediate_flags != WinFlags::NONE)
            .then(|| world_win_table(&sampled, actor, immediate_flags));
        for tile in all_tiles() {
            let future = future_wins[tile.index()];
            if include_future {
                future_discards[tile.index()].observe(future);
            }
            immediate_discards[tile.index()].observe(
                immediate_wins
                    .as_ref()
                    .map_or(future, |wins| wins[tile.index()]),
            );
        }
        if added_kong_mask != 0 {
            let rob_kong_wins = world_win_table(
                &sampled,
                actor,
                WinFlags {
                    rob_kong: true,
                    ..WinFlags::NONE
                },
            );
            for tile in mask_tiles(added_kong_mask) {
                added_kongs[tile.index()].observe(rob_kong_wins[tile.index()]);
            }
        }

        if !include_future {
            continue;
        }

        let projection = project_world(sampled.clone(), actor)?;
        let actual_turns: BTreeMap<_, _> = projection.opponent_turns.into_iter().collect();
        let fallback_turns: [[Option<OpponentTurnEvent>; TILE_KIND_COUNT]; PLAYER_COUNT] =
            core::array::from_fn(|source_index| {
                let source = Seat::new(source_index as u8).expect("array index is a valid seat");
                core::array::from_fn(|tile_index| {
                    (source != actor)
                        .then(|| {
                            fallback_opponent_turn_event(
                                &sampled,
                                actor,
                                source,
                                Tile::from_index_unchecked(tile_index as u8),
                                &future_wins,
                            )
                        })
                        .flatten()
                })
            });
        let wall_draws = sampled.wall_remaining().min(usize::from(u8::MAX));
        for draw_offset in 0..wall_draws {
            let draw_offset = draw_offset as u8;
            let drawn_tile = sampled
                .live_wall_tile(usize::from(draw_offset))
                .expect("a sampled offset is inside the live wall");
            for source in Seat::ALL.into_iter().filter(|&source| source != actor) {
                let key = OpponentTurnKey {
                    source,
                    draw_offset,
                    call_index: 0,
                };
                let event = actual_turns
                    .get(&key)
                    .copied()
                    .or(fallback_turns[source.index()][drawn_tile.index()])?;
                opponent_turns.observe(key, event);
            }

            let wins = projection
                .future_discards
                .get(&draw_offset)
                .unwrap_or(&future_wins);
            future_discard_timeline.observe(draw_offset, wins);
        }
        for (key, event) in actual_turns {
            if key.call_index != 0 {
                opponent_turns.observe(key, event);
            }
        }
        observe_projected_wall(&mut wall_scenarios, &projection.terminal, actor);
    }

    Some(HazardTable {
        added_kong_mask,
        immediate_discards,
        future_discards,
        future_discard_timeline,
        added_kongs,
        opponent_turns,
        wall_scenarios,
    })
}

fn observe_projected_wall(table: &mut WallScenarioTable, terminal: &Game, actor: Seat) {
    if is_wall_projection(terminal.termination_reason()) {
        table.observe(terminal, actor);
    }
}

const fn is_wall_projection(reason: Option<TerminationReason>) -> bool {
    matches!(reason, Some(TerminationReason::WallExhausted))
}

#[derive(Debug)]
struct WorldProjection {
    terminal: Game,
    opponent_turns: Vec<(OpponentTurnKey, OpponentTurnEvent)>,
    future_discards: BTreeMap<u8, WorldWinTable>,
}

/// Advances one complete posterior world with a deterministic legal policy.
///
/// The rollout serves two related estimates. It records future opponent
/// events from the hand state at the corresponding turn, and it projects the
/// same joint hidden world to the end of the wall. A single rollout therefore
/// preserves correlations between all opponent hands and their future draws.
fn project_world(mut world: Game, actor: Seat) -> Option<WorldProjection> {
    let mut clock = FutureDrawClock::default();
    let mut opponent_turns = Vec::new();
    let mut future_discards = BTreeMap::new();
    let mut root_turn_pending = true;

    for _ in 0..MAX_PROJECTION_ACTIONS {
        let Some(decision) = world.decision() else {
            return Some(WorldProjection {
                terminal: world,
                opponent_turns,
                future_discards,
            });
        };
        let action = world.simple_rule_action()?;
        let event = if decision.phase == Phase::Turn && decision.actor != actor {
            projected_turn_event(&world, actor, decision.actor, action.action())
        } else {
            None
        };
        if decision.phase == Phase::Turn
            && decision.actor == actor
            && let Some(draw_offset) = clock.pending_draw(actor)
        {
            future_discards
                .entry(draw_offset)
                .or_insert_with(|| world_win_table(&world, actor, WinFlags::NONE));
        }
        if matches!(action.action(), Action::Hu | Action::Discard(_)) {
            if root_turn_pending && decision.actor == actor && clock.pending_draw(actor).is_none() {
                root_turn_pending = false;
            } else {
                let key = clock.complete_turn(decision.actor);
                if let Some(event) = event {
                    opponent_turns.push((key, event));
                }
            }
        }

        let outcome = world.step_id(action).ok()?;
        if let Some(draw) = outcome.draw {
            if root_turn_pending && !draw.replacement {
                root_turn_pending = false;
            }
            clock.observe_draw(draw.player, draw.replacement);
        }
    }
    None
}

fn projected_turn_event(
    world: &Game,
    actor: Seat,
    source: Seat,
    action: Action,
) -> Option<OpponentTurnEvent> {
    match action {
        Action::Hu => {
            let drawn = world.current_draw()?;
            let holding = Holding::from_game(world, source);
            let evaluation = evaluate_win(
                &holding.concealed,
                holding.melds(),
                Some(drawn.tile),
                WinFlags {
                    after_kong_draw: drawn.replacement,
                    ..WinFlags::NONE
                },
            )?;
            Some(OpponentTurnEvent::SelfDraw(SampledWin {
                payout_multiplier: evaluation.multiplier,
                shape_multiplier: evaluation.shape_multiplier,
            }))
        }
        Action::Discard(tile) => {
            let mut other_wins =
                world_wins_for_tile(world, source, tile, discard_win_flags(world, source));
            other_wins[source.index()] = None;
            other_wins[actor.index()] = None;
            Some(OpponentTurnEvent::Discard { tile, other_wins })
        }
        Action::SelectExchangeTile(_)
        | Action::ChooseMissing(_)
        | Action::Pong
        | Action::ExposedKong
        | Action::ConcealedKong(_)
        | Action::AddedKong(_)
        | Action::Pass => None,
    }
}

/// Constructs a counterfactual event when the baseline rollout did not put
/// `source` at this wall offset. The hidden hand comes from the same posterior
/// world. It is intentionally the nearest available snapshot, while the tile
/// and all response hands remain from that exact joint world.
fn fallback_opponent_turn_event(
    world: &Game,
    actor: Seat,
    source: Seat,
    drawn_tile: Tile,
    actor_future_wins: &WorldWinTable,
) -> Option<OpponentTurnEvent> {
    let holding = Holding::from_game(world, source).after_draw(drawn_tile)?;
    if holding.missing_count() == 0
        && let Some(evaluation) = evaluate_win(
            &holding.concealed,
            holding.melds(),
            Some(drawn_tile),
            WinFlags::NONE,
        )
    {
        return Some(OpponentTurnEvent::SelfDraw(SampledWin {
            payout_multiplier: evaluation.multiplier,
            shape_multiplier: evaluation.shape_multiplier,
        }));
    }

    let visible = world.visible_tile_counts(source);
    let has_won = world.has_won(source);
    let discard = mask_tiles(holding.discard_mask())
        .filter_map(|tile| {
            holding.after_discard(tile).map(|after| {
                (
                    HandPotential::evaluate(&after, &visible, has_won),
                    core::cmp::Reverse(tile),
                    tile,
                )
            })
        })
        .max_by(|(left, left_tile, _), (right, right_tile, _)| {
            left.cmp_for(*right, has_won)
                .then_with(|| left_tile.cmp(right_tile))
        })
        .map(|(_, _, tile)| tile)?;
    let mut other_wins = actor_future_wins[discard.index()];
    other_wins[source.index()] = None;
    other_wins[actor.index()] = None;
    Some(OpponentTurnEvent::Discard {
        tile: discard,
        other_wins,
    })
}

fn posterior_worlds(
    game: &Game,
    actor: Seat,
    history: &PublicHistory,
    worlds: usize,
    base_seed: u64,
) -> Option<Vec<Game>> {
    let candidate_count = worlds.checked_mul(POSTERIOR_CANDIDATES_PER_WORLD)?;
    let sampled: Vec<_> = (0..candidate_count)
        .into_par_iter()
        .map(|candidate_index| {
            let seed = mix_seed(base_seed.wrapping_add(candidate_index as u64));
            game.resample_information_set(seed).map(|sampled| {
                let weight = world_likelihood_ratio(&sampled, actor, history);
                (sampled, weight)
            })
        })
        .collect();
    let sampled: Vec<_> = sampled.into_iter().collect::<Result<_, _>>().ok()?;
    let (candidates, weights): (Vec<_>, Vec<_>) = sampled.into_iter().unzip();

    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= f64::EPSILON {
        return Some(candidates.into_iter().take(worlds).collect());
    }

    // Stochastic universal resampling with a fixed midpoint is deterministic,
    // preserves posterior mass, and avoids independent resampling noise.
    let step = total / worlds as f64;
    let mut target = 0.5 * step;
    let mut cumulative = weights[0];
    let mut candidate_index = 0;
    let mut selected = Vec::with_capacity(worlds);
    for _ in 0..worlds {
        while cumulative < target && candidate_index + 1 < candidates.len() {
            candidate_index += 1;
            cumulative += weights[candidate_index];
        }
        selected.push(candidates[candidate_index].clone());
        target += step;
    }
    Some(selected)
}

fn world_likelihood_ratio(world: &Game, actor: Seat, history: &PublicHistory) -> f64 {
    Seat::ALL
        .into_iter()
        .filter(|&opponent| opponent != actor)
        .map(|opponent| {
            let holding = crate::rules::hand::Holding::from_game(world, opponent);
            let visible = world.visible_tile_counts(opponent);
            let entries = history.player(opponent);
            history_likelihood_ratio(holding, entries, &visible, world.has_won(opponent))
        })
        .product()
}

fn world_log_likelihood_ratio(world: &Game, actor: Seat, history: &PublicHistory) -> f64 {
    Seat::ALL
        .into_iter()
        .filter(|&opponent| opponent != actor)
        .try_fold(0.0, |total, opponent| {
            let holding = crate::rules::hand::Holding::from_game(world, opponent);
            let visible = world.visible_tile_counts(opponent);
            let entries = history.player(opponent);
            let ratio =
                history_likelihood_ratio(holding, entries, &visible, world.has_won(opponent));
            (ratio.is_finite() && ratio > 0.0).then(|| total + ratio.ln())
        })
        .unwrap_or(f64::NEG_INFINITY)
}

fn mix_seed(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn world_win_table(
    world: &Game,
    source: Seat,
    flags: WinFlags,
) -> [[Option<SampledWin>; PLAYER_COUNT]; TILE_KIND_COUNT] {
    let mut table = [[None; PLAYER_COUNT]; TILE_KIND_COUNT];
    for opponent in Seat::ALL.into_iter().filter(|&seat| seat != source) {
        let holding = Holding::from_game(world, opponent);
        if holding.missing_count() != 0 {
            continue;
        }
        let candidate_mask = win_candidate_mask(world, opponent, &holding);
        for tile in mask_tiles(candidate_mask) {
            let Some(with_tile) = holding.after_draw(tile) else {
                continue;
            };
            let Some(evaluation) =
                evaluate_win(&with_tile.concealed, with_tile.melds(), Some(tile), flags)
            else {
                continue;
            };
            table[tile.index()][opponent.index()] = Some(SampledWin {
                payout_multiplier: evaluation.multiplier,
                shape_multiplier: evaluation.shape_multiplier,
            });
        }
    }
    table
}

fn world_wins_for_tile(
    world: &Game,
    source: Seat,
    tile: Tile,
    flags: WinFlags,
) -> [Option<SampledWin>; PLAYER_COUNT] {
    let mut wins = [None; PLAYER_COUNT];
    for opponent in Seat::ALL.into_iter().filter(|&seat| seat != source) {
        let holding = Holding::from_game(world, opponent);
        if holding.missing_count() != 0
            || holding.missing == Some(tile.suit())
            || win_candidate_mask(world, opponent, &holding) & (1 << tile.index()) == 0
        {
            continue;
        }
        let Some(with_tile) = holding.after_draw(tile) else {
            continue;
        };
        let Some(evaluation) =
            evaluate_win(&with_tile.concealed, with_tile.melds(), Some(tile), flags)
        else {
            continue;
        };
        wins[opponent.index()] = Some(SampledWin {
            payout_multiplier: evaluation.multiplier,
            shape_multiplier: evaluation.shape_multiplier,
        });
    }
    wins
}

fn win_candidate_mask(world: &Game, opponent: Seat, holding: &Holding) -> u32 {
    if world.has_won(opponent) {
        return (1_u32 << TILE_KIND_COUNT) - 1;
    }
    let analysis = analyze_shanten(&holding.concealed, holding.melds(), holding.missing);
    if analysis.shanten == 0 {
        analysis.improving_tiles
    } else {
        0
    }
}

fn discard_win_flags(game: &Game, actor: Seat) -> WinFlags {
    let after_kong_discard = game
        .current_draw()
        .is_some_and(|draw| draw.player == actor && draw.replacement);
    let earthly = actor == game.dealer() && game.discards().next().is_none();
    WinFlags {
        after_kong_discard,
        earthly,
        ..WinFlags::NONE
    }
}

#[cfg(test)]
const fn seat_bit(seat: Seat) -> u8 {
    1 << seat.index()
}

fn mask_tiles(mask: u32) -> impl Iterator<Item = Tile> {
    (0..TILE_KIND_COUNT)
        .filter(move |&index| mask & (1 << index) != 0)
        .map(|index| Tile::from_index_unchecked(index as u8))
}

fn all_tiles() -> impl Iterator<Item = Tile> {
    (0..TILE_KIND_COUNT).map(|index| Tile::from_index_unchecked(index as u8))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct WallResidual;

    impl BeliefResidualEvaluator for WallResidual {
        fn evaluate_residuals(
            &self,
            _public: &BeliefPublicFeatures,
            candidates: &[crate::BeliefCandidateFeatures],
            output: &mut [f32],
        ) -> Result<(), BeliefResidualError> {
            if candidates.len() != output.len() {
                return Err(BeliefResidualError::BatchLength {
                    candidates: candidates.len(),
                    outputs: output.len(),
                });
            }
            for (candidate, residual) in candidates.iter().zip(output) {
                *residual = f32::from(candidate.live_wall[0]) * 0.25;
            }
            Ok(())
        }
    }

    struct NonFiniteResidual;

    impl BeliefResidualEvaluator for NonFiniteResidual {
        fn evaluate_residuals(
            &self,
            _public: &BeliefPublicFeatures,
            _candidates: &[crate::BeliefCandidateFeatures],
            output: &mut [f32],
        ) -> Result<(), BeliefResidualError> {
            output.fill(f32::NAN);
            Ok(())
        }
    }

    fn observed_turn(seed: u64) -> Game {
        let mut game = Game::new(seed);
        for _ in 0..MAX_PROJECTION_ACTIONS {
            let legal = game.legal_actions().expect("setup game is non-terminal");
            if legal.decision.phase == Phase::Turn && game.wall_remaining() <= 48 {
                return game;
            }
            let action = game
                .simple_rule_action()
                .expect("the setup policy handles a non-terminal game");
            game.step_id(action).expect("the setup action is legal");
        }
        panic!("setup did not reach an observed turn")
    }

    const fn win(multiplier: u32) -> SampledWin {
        SampledWin {
            payout_multiplier: multiplier,
            shape_multiplier: multiplier,
        }
    }

    #[test]
    fn posterior_particles_preserve_root_information_and_stable_prefixes() {
        let game = observed_turn(1_019);
        let actor = game.decision().expect("turn has an actor").actor;
        let base_seed = super::super::public_state_hash(&game, actor) ^ 0x41f2_6da8_339c_107b;
        let sampler = RootBeliefSampler::new(&game, actor, base_seed)
            .expect("the observed turn has an information set");
        let short: Vec<_> = (0..4)
            .map(|id| sampler.sample(id).expect("particle has support"))
            .collect();
        let long: Vec<_> = (0..8)
            .map(|id| sampler.sample(id).expect("particle has support"))
            .collect();

        for world in &long {
            assert_eq!(world.phase(), game.phase());
            assert_eq!(world.decision(), game.decision());
            assert_eq!(world.legal_action_mask(), game.legal_action_mask());
            assert_eq!(
                super::super::public_state_hash(world, actor),
                super::super::public_state_hash(&game, actor)
            );
        }
        assert_eq!(short, long[..4]);
    }

    #[test]
    fn public_root_proposals_share_the_production_sampler_prefix() {
        let game = observed_turn(1_021);
        let short = game
            .belief_root_proposals(0x4ae8_20cd, 4)
            .expect("turn root supports belief proposals");
        let long = game
            .belief_root_proposals(0x4ae8_20cd, 8)
            .expect("turn root supports belief proposals");
        let truth = game
            .belief_root_candidate()
            .expect("turn root exposes one training label");

        assert_eq!(short, long[..4]);
        assert_eq!(
            truth.features,
            game.belief_candidate_features(game.decision().expect("turn root has an actor").actor,),
        );
        assert_eq!(
            game.belief_root_public_features(),
            game.decision()
                .map(|decision| game.belief_public_features(decision.actor)),
        );
    }

    #[test]
    fn residual_batch_updates_multiple_particle_streams() {
        let game = observed_turn(1_022);
        let actor = game.decision().expect("turn root has an actor").actor;
        let sampler = RootBeliefSampler::new(&game, actor, 0x052f_901b)
            .expect("turn root supports belief particles");
        let baseline = sampler.sample_batch(8, RootBeliefMode::Posterior);
        let mut learned = sampler.sample_batch(8, RootBeliefMode::Posterior);
        let mut validation = sampler.sample_batch(4, RootBeliefMode::Posterior);
        let validation_baseline = sampler.sample_batch(4, RootBeliefMode::Posterior);
        let mut batches = [learned.as_mut_slice(), validation.as_mut_slice()];
        sampler
            .apply_residuals(&WallResidual, &mut batches)
            .expect("finite residuals are accepted");

        assert_eq!(baseline.len(), learned.len());
        for (base, adjusted) in baseline.iter().zip(&learned) {
            assert_eq!(base.game, adjusted.game);
            let residual =
                f64::from(base.game.belief_candidate_features(actor).live_wall[0]) * 0.25;
            assert_eq!(adjusted.log_likelihood, base.log_likelihood + residual);
        }
        for (base, adjusted) in validation_baseline.iter().zip(&validation) {
            let residual =
                f64::from(base.game.belief_candidate_features(actor).live_wall[0]) * 0.25;
            assert_eq!(adjusted.log_likelihood, base.log_likelihood + residual);
        }

        let mut invalid = sampler.sample_batch(8, RootBeliefMode::Posterior);
        let mut invalid_batches = [invalid.as_mut_slice()];
        assert!(matches!(
            sampler.apply_residuals(&NonFiniteResidual, &mut invalid_batches),
            Err(BeliefResidualError::NonFinite { index: 0 }),
        ));
    }

    #[test]
    fn oracle_particles_preserve_hidden_allocation_and_resample_future() {
        let game = observed_turn(1_023);
        let actor = game.decision().expect("turn has an actor").actor;
        let sampler = RootBeliefSampler::new(&game, actor, 0x7f83_a019)
            .expect("the observed turn has an information set");
        let first = sampler
            .sample_weighted(3, RootBeliefMode::OracleHidden)
            .expect("oracle particle has support");
        let repeated = sampler
            .sample_weighted(3, RootBeliefMode::OracleHidden)
            .expect("repeated oracle particle has support");
        let second = sampler
            .sample_weighted(4, RootBeliefMode::OracleHidden)
            .expect("second oracle particle has support");

        assert_eq!(first.log_likelihood, 0.0);
        assert_eq!(first.game, repeated.game);
        assert_ne!(first.game, second.game);
        for seat in Seat::ALL {
            assert_eq!(first.game.concealed(seat), game.concealed(seat));
            assert_eq!(first.game.locked(seat), game.locked(seat));
        }
        let mut expected = [0; crate::game::ORACLE_TILE_COUNT_WIDTH];
        let mut actual = [0; crate::game::ORACLE_TILE_COUNT_WIDTH];
        game.oracle_tile_counts_into(&mut expected)
            .expect("root oracle buffer is valid");
        first
            .game
            .oracle_tile_counts_into(&mut actual)
            .expect("sample oracle buffer is valid");
        assert_eq!(actual, expected);
    }

    #[test]
    fn uniform_particles_ignore_public_history_likelihood() {
        let game = observed_turn(1_027);
        let actor = game.decision().expect("turn has an actor").actor;
        let sampler = RootBeliefSampler::new(&game, actor, 0x392c_61a8)
            .expect("the observed turn has an information set");
        let particle = sampler
            .sample_weighted(0, RootBeliefMode::Uniform)
            .expect("uniform particle has support");

        assert_eq!(particle.log_likelihood, 0.0);
        assert_eq!(
            super::super::public_state_hash(&particle.game, actor),
            super::super::public_state_hash(&game, actor)
        );
    }

    #[test]
    fn posterior_sampling_is_independent_of_rayon_thread_count() {
        let game = observed_turn(1_031);
        let actor = game.decision().expect("turn has an actor").actor;
        let sample = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test thread pool builds")
                .install(|| {
                    let sampler = RootBeliefSampler::new(&game, actor, 0x8cb9_45e1)
                        .expect("the observed turn has an information set");
                    (0..4)
                        .map(|id| sampler.sample(id).expect("particle has support"))
                        .collect::<Vec<_>>()
                })
        };
        let serial = sample(1);
        let parallel = sample(4);
        assert_eq!(serial, parallel);
    }

    fn naive_world_win_table(world: &Game, source: Seat) -> WorldWinTable {
        let mut table = [[None; PLAYER_COUNT]; TILE_KIND_COUNT];
        for opponent in Seat::ALL.into_iter().filter(|&seat| seat != source) {
            let holding = Holding::from_game(world, opponent);
            if holding.missing_count() != 0 {
                continue;
            }
            for tile in all_tiles() {
                let Some(with_tile) = holding.after_draw(tile) else {
                    continue;
                };
                let Some(evaluation) = evaluate_win(
                    &with_tile.concealed,
                    with_tile.melds(),
                    Some(tile),
                    WinFlags::NONE,
                ) else {
                    continue;
                };
                table[tile.index()][opponent.index()] = Some(SampledWin {
                    payout_multiplier: evaluation.multiplier,
                    shape_multiplier: evaluation.shape_multiplier,
                });
            }
        }
        table
    }

    #[test]
    fn empty_observation_has_zero_risk() {
        let mut hazard = DiscardHazard::default();
        hazard.observe([None; PLAYER_COUNT]);

        assert_eq!(hazard.sample_count(), 1);
        assert_eq!(hazard.any_win_probability(), 0.0);
        assert_eq!(hazard.multiple_win_probability(), 0.0);
        assert_eq!(hazard.expected_loss_points(Seat::EAST, 10_000), 0.0);

        let atoms: Vec<_> = hazard.transitions().collect();
        assert_eq!(atoms.len(), 1);
        assert_eq!(atoms[0].winner_mask(), 0);
        assert_eq!(atoms[0].count, 1);
    }

    #[test]
    fn observations_preserve_joint_and_conditional_statistics() {
        let south = Seat::EAST.offset(1);
        let west = Seat::EAST.offset(2);
        let mut hazard = DiscardHazard::default();
        hazard.observe([None, Some(win(2)), Some(win(4)), None]);
        hazard.observe([None, Some(win(6)), None, None]);
        hazard.observe([None; PLAYER_COUNT]);

        assert_eq!(hazard.sample_count(), 3);
        assert_eq!(hazard.any_win_probability(), 2.0 / 3.0);
        assert_eq!(hazard.multiple_win_probability(), 1.0 / 3.0);
        assert_eq!(hazard.expected_loss_points(Seat::EAST, 10_000), 400.0);
        assert_eq!(hazard.expected_nominal_loss_points(), 400.0);
        assert_eq!(hazard.opponent_win_probability(south), 2.0 / 3.0);
        assert_eq!(hazard.opponent_conditional_multiplier(south), Some(4.0));
        assert_eq!(hazard.opponent_win_probability(west), 1.0 / 3.0);
        assert_eq!(hazard.opponent_conditional_multiplier(west), Some(4.0));

        let winner_mask = seat_bit(south) | seat_bit(west);
        let multiple = hazard
            .transitions()
            .find(|atom| atom.winner_mask() == winner_mask)
            .expect("the multi-winner outcome was observed");
        assert_eq!(multiple.count, 1);
        assert_eq!(multiple.wins[south.index()], Some(win(2)));
        assert_eq!(multiple.wins[west.index()], Some(win(4)));

        let mut outcome_masks: Vec<_> = hazard
            .transitions()
            .map(|atom| (atom.winner_mask(), atom.count))
            .collect();
        outcome_masks.sort_unstable();
        assert_eq!(
            outcome_masks,
            vec![(0, 1), (seat_bit(south), 1), (winner_mask, 1)]
        );
    }

    #[test]
    fn insufficient_funds_follow_response_order() {
        let south = Seat::EAST.offset(1);
        let west = Seat::EAST.offset(2);
        let mut hazard = DiscardHazard::default();
        hazard.observe([None, Some(win(2)), Some(win(4)), None]);

        assert_eq!(hazard.expected_nominal_loss_points(), 600.0);
        assert_eq!(hazard.expected_loss_points(Seat::EAST, 500), 500.0);
        assert_eq!(
            hazard.opponent_expected_loss_points(Seat::EAST, south, 500),
            200.0
        );
        assert_eq!(
            hazard.opponent_expected_loss_points(Seat::EAST, west, 500),
            300.0
        );
    }

    #[test]
    fn estimate_rejects_unsupported_decisions_and_empty_budget() {
        let game = Game::new(7);

        assert_eq!(estimate(&game, Seat::EAST, 0, 11), None);
        assert_eq!(estimate(&game, Seat::EAST, 4, 11), None);
    }

    #[test]
    fn estimate_populates_future_hazards_outside_the_root_discard_mask() {
        let mut game = Game::new(7);
        while game.phase() != Phase::Turn {
            let action = game
                .simple_rule_action()
                .expect("setup game has a decision before its first turn");
            game.step_id(action)
                .expect("the simple rule returns a legal setup action");
        }
        let legal = game.legal_actions().expect("setup reached a turn");
        let tile = all_tiles()
            .find(|tile| legal.discard_mask & (1 << tile.index()) == 0)
            .expect("a hand cannot contain all tile kinds");
        let table = estimate(&game, legal.decision.actor, 1, 11)
            .expect("one complete information-set sample is available");

        assert_eq!(table.immediate_discard(tile).sample_count(), 1);
        assert_eq!(table.future_discard(tile).sample_count(), 1);
        assert_eq!(
            table
                .future_discard_at(tile, 0)
                .expect("the first future draw has a hazard table")
                .sample_count(),
            1
        );
        let last_offset = u8::try_from(game.wall_remaining() - 1)
            .expect("the live wall fits in the offset representation");
        for source in Seat::ALL
            .into_iter()
            .filter(|&source| source != legal.decision.actor)
        {
            assert_eq!(
                table
                    .opponent_turn_at(source, last_offset)
                    .expect("counterfactual actor offsets have fallback events")
                    .sample_count(),
                1
            );
        }
    }

    #[test]
    fn discard_contexts_cover_all_tiles_and_remain_independent() {
        let immediate_tile = Tile::from_index_unchecked(0);
        let future_tile = Tile::from_index_unchecked((TILE_KIND_COUNT - 1) as u8);
        let mut immediate_discards = core::array::from_fn(|_| DiscardHazard::default());
        let mut future_discards = core::array::from_fn(|_| DiscardHazard::default());
        immediate_discards[immediate_tile.index()].observe([None, Some(win(8)), None, None]);
        future_discards[future_tile.index()].observe([None; PLAYER_COUNT]);
        let table = HazardTable {
            added_kong_mask: 0,
            immediate_discards,
            future_discards,
            future_discard_timeline: DiscardHazardTimeline::default(),
            added_kongs: core::array::from_fn(|_| DiscardHazard::default()),
            opponent_turns: OpponentTurnTimeline::default(),
            wall_scenarios: WallScenarioTable::default(),
        };

        for tile in all_tiles() {
            let _ = table.immediate_discard(tile);
            let _ = table.future_discard(tile);
        }
        assert_eq!(table.immediate_discard(immediate_tile).sample_count(), 1);
        assert_eq!(table.future_discard(immediate_tile).sample_count(), 0);
        assert_eq!(table.immediate_discard(future_tile).sample_count(), 0);
        assert_eq!(table.future_discard(future_tile).sample_count(), 1);
    }

    #[test]
    fn draw_offsets_follow_the_wall_after_a_winner_changes_the_actor() {
        let south = Seat::EAST.next();
        let west = south.next();
        let north = west.next();
        let mut clock = FutureDrawClock::default();

        clock.observe_draw(south, false);
        assert_eq!(
            clock.complete_turn(south),
            OpponentTurnKey {
                source: south,
                draw_offset: 0,
                call_index: 0,
            }
        );

        // A discard win can make the winner's successor, rather than the
        // discarder or the next fixed seat, receive the next wall tile.
        clock.observe_draw(north, false);
        assert_eq!(clock.pending_draw(west), None);
        assert_eq!(
            clock.complete_turn(north),
            OpponentTurnKey {
                source: north,
                draw_offset: 1,
                call_index: 0,
            }
        );
    }

    #[test]
    fn replacement_draws_keep_the_normal_wall_offset() {
        let south = Seat::EAST.next();
        let west = south.next();
        let mut clock = FutureDrawClock::default();

        clock.observe_draw(south, false);
        clock.observe_draw(south, true);
        assert_eq!(clock.complete_turn(south).draw_offset, 0);
        clock.observe_draw(west, false);
        assert_eq!(clock.complete_turn(west).draw_offset, 1);
    }

    #[test]
    fn no_draw_call_turns_do_not_alias_the_next_wall_draw() {
        let south = Seat::EAST.next();
        let mut clock = FutureDrawClock::default();

        assert_eq!(
            clock.complete_turn(south),
            OpponentTurnKey {
                source: south,
                draw_offset: 0,
                call_index: 1,
            }
        );
        clock.observe_draw(south, false);
        assert_eq!(
            clock.complete_turn(south),
            OpponentTurnKey {
                source: south,
                draw_offset: 0,
                call_index: 0,
            }
        );
    }

    #[test]
    fn wall_projection_advances_hidden_hands_instead_of_using_the_root_snapshot() {
        let mut game = Game::new(7);
        while game.phase() != Phase::Turn {
            let action = game
                .simple_rule_action()
                .expect("setup game has a decision before its first turn");
            game.step_id(action)
                .expect("the simple rule returns a legal setup action");
        }
        let actor = game.decision().expect("setup reached a turn").actor;
        let current = Seat::ALL.map(|seat| sampled_wall_outcome(&game, seat));
        let current_hazards = world_win_table(&game, actor, WinFlags::NONE);

        let projection = project_world(game, actor).expect("a legal game reaches a terminal state");
        let projected = Seat::ALL.map(|seat| sampled_wall_outcome(&projection.terminal, seat));

        assert_eq!(projection.terminal.phase(), Phase::Finished);
        assert_ne!(projected, current);
        assert!(
            projection
                .future_discards
                .values()
                .any(|hazards| hazards != &current_hazards),
            "future actor hazards must use projected opponent hands"
        );
    }

    #[test]
    fn bankruptcy_is_not_a_wall_projection() {
        assert!(is_wall_projection(Some(TerminationReason::WallExhausted)));
        assert!(!is_wall_projection(Some(
            TerminationReason::ThreePlayersBankrupt
        )));
        assert!(!is_wall_projection(None));
    }

    #[test]
    fn shanten_prefilter_preserves_exact_win_tables() {
        let mut game = Game::new(19);
        let mut checked = 0;
        for _ in 0..192 {
            let Some(decision) = game.decision() else {
                break;
            };
            if decision.phase == Phase::Turn && checked < 12 {
                assert_eq!(
                    world_win_table(&game, decision.actor, WinFlags::NONE),
                    naive_world_win_table(&game, decision.actor)
                );
                checked += 1;
            }
            let action = game
                .simple_rule_action()
                .expect("the simple rule handles a non-terminal game");
            game.step_id(action)
                .expect("the simple rule returns a legal action");
        }
        assert_eq!(checked, 12);
    }
}
