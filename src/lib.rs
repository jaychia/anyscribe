//! # anyscribe
//!
//! A modular, trait-based real-time transcription pipeline.
//!
//! anyscribe defines four pipeline stages connected by bounded async channels,
//! with a broadcast channel that fans segments out to any number of subscribers:
//!
//! ```text
//! AudioInput --> Preprocessor --> TranscriptionEngine --> Postprocessor -->> [subscribers]
//! ```
//!
//! Each stage is an `#[async_trait]` with a uniform `async fn run()` interface.
//! Stages are spawned as independent tokio tasks and joined with `tokio::join!`.
//! Backpressure propagates naturally through bounded channel sends.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use anyscribe::pipeline::PipelineRunner;
//! use anyscribe::preprocess::DefaultPreprocessor;
//! use anyscribe::postprocess::NoopPostprocessor;
//! use anyscribe::output::stdout::StdoutOutputSink;
//! use anyscribe::audio::cpal_input::CpalAudioInput;
//! use anyscribe::transcribe::whisper::WhisperTranscriptionEngine;
//! use anyscribe::types::Metadata;
//! use tokio_util::sync::CancellationToken;
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let runner = PipelineRunner::new(
//!     Box::new(CpalAudioInput::new(PathBuf::from("/tmp/recording.wav"))?),
//!     Box::new(DefaultPreprocessor { target_sample_rate: 16000 }),
//!     Box::new(WhisperTranscriptionEngine::new("base", 16000)?),
//!     Box::new(NoopPostprocessor),
//!     CancellationToken::new(),
//!     Metadata { model: "base".into(), language: None },
//! );
//!
//! // Subscribe before running — each subscriber gets its own stream of segments.
//! let rx = runner.subscribe();
//! tokio::spawn(async move { StdoutOutputSink.run(rx).await });
//!
//! runner.run().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Implementing custom stages
//!
//! Every pipeline stage is a trait. Implement [`pipeline::traits::AudioInput`] for
//! a custom audio source, [`pipeline::traits::Postprocessor`] for custom filtering, etc.
//! The [`pipeline::PipelineRunner`] accepts any combination of trait objects.
//! Subscribers are registered via [`PipelineRunner::subscribe`] before running.

pub mod audio;
pub mod config;
pub mod constants;
pub mod error;
pub mod output;
pub mod pipeline;
pub mod postprocess;
pub mod preprocess;
pub mod transcribe;
pub mod types;
