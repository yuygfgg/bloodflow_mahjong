use rayon::prelude::*;

use crate::ActionId;
use crate::game::{Batch, Game, GameError, PARALLEL_BATCH_THRESHOLD};

pub(crate) mod ev;
pub(crate) mod fast;
pub(crate) mod hand;
#[cfg(feature = "rule-nn")]
pub(crate) mod nn;
pub(crate) mod opening;
pub(crate) mod planner;

fn batch_policy_actions_into(
    batch: &Batch,
    enabled: Option<&[u8]>,
    output: &mut [u8],
    terminal: u8,
    choose: impl Fn(&Game) -> Option<ActionId> + Sync,
) -> Result<(), GameError> {
    if output.len() != batch.len() || enabled.is_some_and(|mask| mask.len() != batch.len()) {
        return Err(GameError::BatchLength);
    }
    if enabled.is_some_and(|mask| mask.iter().any(|&value| value > 1)) {
        return Err(GameError::InvalidAction);
    }

    let write = |game: &Game, action: &mut u8| {
        *action = choose(game).map_or(terminal, |id| id.index() as u8);
    };
    match enabled {
        Some(mask) if batch.len() >= PARALLEL_BATCH_THRESHOLD => batch
            .games()
            .par_iter()
            .zip(mask.par_iter().copied())
            .zip(output.par_iter_mut())
            .for_each(|((game, enabled), action)| {
                if enabled != 0 {
                    write(game, action);
                }
            }),
        Some(mask) => {
            for ((game, &enabled), action) in batch.games().iter().zip(mask).zip(output.iter_mut())
            {
                if enabled != 0 {
                    write(game, action);
                }
            }
        }
        None if batch.len() >= PARALLEL_BATCH_THRESHOLD => batch
            .games()
            .par_iter()
            .zip(output.par_iter_mut())
            .for_each(|(game, action)| write(game, action)),
        None => {
            for (game, action) in batch.games().iter().zip(output) {
                write(game, action);
            }
        }
    }
    Ok(())
}
