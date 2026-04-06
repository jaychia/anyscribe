use clap::{Parser, Subcommand};
use tokio_util::sync::CancellationToken;

use scribe_rs::audio::cpal_input::CpalAudioInput;
use scribe_rs::config::{config_path, first_run_setup, load_config, save_config};
use scribe_rs::output::markdown::MarkdownOutputSink;
use scribe_rs::output::multi::MultiOutputSink;
use scribe_rs::output::stdout::StdoutOutputSink;
use scribe_rs::pipeline::traits::AudioInput;
use scribe_rs::pipeline::PipelineRunner;
use scribe_rs::postprocess::NoopPostprocessor;
use scribe_rs::preprocess::DefaultPreprocessor;
use scribe_rs::transcribe::whisper::WhisperTranscriptionEngine;
use scribe_rs::types::Metadata;

#[derive(Parser)]
#[command(
    name = "scribe-rs",
    about = "Simple, modular, single-binary transcription from audio to text."
)]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start a live recording and transcribe in real-time
    Record,
    /// List recent notes
    List,
    /// View or modify configuration
    #[command(subcommand)]
    Config(ConfigCommands),
}

#[derive(Subcommand)]
enum ConfigCommands {
    /// Show current configuration
    Show,
    /// Set a configuration value
    Set {
        /// Config key to set
        #[arg(value_parser = ["notes_path", "whisper_model", "language", "sample_rate"])]
        key: String,
        /// New value
        value: String,
    },
    /// Run interactive first-time setup
    Setup,
}

fn load_and_validate_config() -> anyhow::Result<scribe_rs::config::Config> {
    let mut cfg = load_config()?;
    if cfg.notes_path.is_empty() {
        cfg = first_run_setup()?;
    }
    let errors = cfg.validate();
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("Config error: {e}");
        }
        anyhow::bail!("Configuration is invalid");
    }
    Ok(cfg)
}

async fn cmd_record() -> anyhow::Result<()> {
    let config = load_and_validate_config()?;

    let input = CpalAudioInput::new()?;
    let info = input.info();
    eprintln!(
        "Audio device: {}ch @ {}Hz → {}Hz mono",
        info.channels, info.sample_rate, config.sample_rate
    );
    eprintln!("Recording... Press Enter or Ctrl+C to stop.\n");

    let engine = WhisperTranscriptionEngine::new(&config.whisper_model, config.sample_rate)?;

    let cancel = CancellationToken::new();

    // Stop on Enter or Ctrl+C
    let cancel_clone = cancel.clone();
    tokio::spawn(async move {
        let stdin_line = async {
            let mut buf = String::new();
            let stdin = tokio::io::stdin();
            let mut reader = tokio::io::BufReader::new(stdin);
            let _ = tokio::io::AsyncBufReadExt::read_line(&mut reader, &mut buf).await;
        };
        tokio::select! {
            biased;
            _ = tokio::signal::ctrl_c() => {}
            _ = stdin_line => {}
        }
        cancel_clone.cancel();
    });

    let recorded_at = chrono::Local::now().naive_local();

    let sink = MultiOutputSink {
        sinks: vec![
            Box::new(StdoutOutputSink),
            Box::new(MarkdownOutputSink {
                notes_dir: config.notes_dir(),
                recorded_at,
                title: None,
            }),
        ],
    };

    let runner = PipelineRunner {
        input: Box::new(input),
        preprocessor: Box::new(DefaultPreprocessor {
            target_sample_rate: config.sample_rate,
        }),
        engine: Box::new(engine),
        postprocessor: Box::new(NoopPostprocessor),
        sink: Box::new(sink),
        cancel,
        metadata: Metadata {
            model: config.whisper_model.clone(),
            language: config.language.clone(),
        },
    };

    runner.run().await?;

    Ok(())
}

fn cmd_list() -> anyhow::Result<()> {
    let config = load_and_validate_config()?;
    let notes_dir = config.notes_dir();

    if !notes_dir.exists() {
        println!("No notes yet. Run `scribe-rs record` to get started.");
        return Ok(());
    }

    let mut notes: Vec<_> = std::fs::read_dir(&notes_dir)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("md") {
                Some(path)
            } else {
                None
            }
        })
        .collect();

    if notes.is_empty() {
        println!("No notes yet. Run `scribe-rs record` to get started.");
        return Ok(());
    }

    notes.sort_by(|a, b| b.cmp(a));

    println!("Notes in {}:\n", notes_dir.display());
    for note in &notes {
        if let Some(stem) = note.file_stem().and_then(|s| s.to_str()) {
            let name = stem.replace(['_', '-'], " ");
            println!("  {name}");
        }
    }
    Ok(())
}

fn cmd_config_show() -> anyhow::Result<()> {
    let cfg = load_config()?;
    let path = config_path();
    println!("Config file: {}\n", path.display());
    println!(
        "  notes_path:     {}",
        if cfg.notes_path.is_empty() {
            "(not set)"
        } else {
            &cfg.notes_path
        }
    );
    println!("  whisper_model:  {}", cfg.whisper_model);
    println!(
        "  language:       {}",
        cfg.language.as_deref().unwrap_or("(auto-detect)")
    );
    println!("  sample_rate:    {}", cfg.sample_rate);
    Ok(())
}

fn cmd_config_set(key: &str, value: &str) -> anyhow::Result<()> {
    let mut cfg = load_config()?;

    match key {
        "notes_path" => cfg.notes_path = value.to_string(),
        "whisper_model" => cfg.whisper_model = value.to_string(),
        "language" => {
            let lower = value.to_lowercase();
            if lower == "none" || lower == "auto" || value.is_empty() {
                cfg.language = None;
            } else {
                cfg.language = Some(value.to_string());
            }
        }
        "sample_rate" => {
            cfg.sample_rate = value
                .parse()
                .map_err(|_| anyhow::anyhow!("sample_rate must be an integer"))?;
        }
        _ => unreachable!(),
    }

    let errors = cfg.validate();
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("Config error: {e}");
        }
        anyhow::bail!("Configuration is invalid");
    }

    save_config(&cfg)?;

    match key {
        "language" => println!("Set {key} = {:?}", cfg.language),
        _ => println!("Set {key} = {value:?}"),
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    env_logger::init();

    match cli.command {
        Commands::Record => cmd_record().await,
        Commands::List => cmd_list(),
        Commands::Config(cmd) => match cmd {
            ConfigCommands::Show => cmd_config_show(),
            ConfigCommands::Set { key, value } => cmd_config_set(&key, &value),
            ConfigCommands::Setup => {
                first_run_setup()?;
                Ok(())
            }
        },
    }
}
