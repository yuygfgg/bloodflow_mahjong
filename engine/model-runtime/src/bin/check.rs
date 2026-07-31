use std::path::PathBuf;

use bloodflow_mahjong_model_runtime::CandleBeliefModel;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(about = "Validate a belief model artifact against its golden vector")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = 1e-4)]
    tolerance: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let model = CandleBeliefModel::load(&args.model)?;
    model.verify_golden(args.tolerance)?;
    println!(
        "RESULT model {} beta {:.6} golden tolerance {:.2e} passed",
        model.artifact_dir().display(),
        model.beta(),
        args.tolerance,
    );
    Ok(())
}
