use std::error::Error;
use std::path::PathBuf;

use bloodflow_mahjong::{Game, RuleNn, Seat};

fn main() -> Result<(), Box<dyn Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: rule_nn_smoke <model.onnx> [seed]")?;
    let seed = std::env::args()
        .nth(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(7);
    let bytes = std::fs::read(&path)?;
    let policy = RuleNn::from_onnx_bytes(&bytes)?;
    let mut game = Game::new(seed);
    let mut decisions = 0_usize;

    while let Some(action) = policy.action(&game)? {
        let legal = game
            .legal_action_mask()
            .expect("active policy result has a legal-action mask");
        assert!(legal.contains(action), "rule-nn selected an illegal action");
        game.step_id(action)?;
        decisions += 1;
    }

    let scores = Seat::ALL.map(|seat| game.score(seat));
    println!(
        "model={} seed={seed} decisions={decisions} scores={scores:?}",
        path.display()
    );
    Ok(())
}
