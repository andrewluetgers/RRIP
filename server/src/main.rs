use anyhow::Result;
use clap::{Parser, Subcommand};

mod core;
mod fast_upsample_ycbcr;
mod turbojpeg_optimized;
mod serve;
mod encode;
mod decode;

#[derive(Parser)]
#[command(name = "origami", about = "ORIGAMI tile server and residual encoder")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the tile server
    Serve(serve::ServeArgs),
    /// Encode residuals from a DZI pyramid
    Encode(encode::EncodeArgs),
    /// Decode (reconstruct) tiles from residuals or pack files
    Decode(decode::DecodeArgs),
}

fn main() -> Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve(args) => serve::run(args),
        Commands::Encode(args) => encode::run(args),
        Commands::Decode(args) => decode::run(args),
    }
}
