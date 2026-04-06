//! # scribe-rs
//!
//! A modular, trait-based real-time transcription pipeline.
//!
//! scribe-rs defines five pipeline stages connected by bounded async channels:
//!
//! ```text
//! AudioInput --> Preprocessor --> TranscriptionEngine --> Postprocessor --> OutputSink
//! ```
//!
//! Each stage is an `#[async_trait]` with a uniform `async fn run()` interface.
//! Stages are spawned as independent tokio tasks and joined with `tokio::join!`.
//! Backpressure propagates naturally through bounded channel sends.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use scribe_rs::pipeline::PipelineRunner;
//! use scribe_rs::preprocess::DefaultPreprocessor;
//! use scribe_rs::postprocess::NoopPostprocessor;
//! use scribe_rs::output::stdout::StdoutOutputSink;
//! use scribe_rs::audio::cpal_input::CpalAudioInput;
//! use scribe_rs::transcribe::whisper::WhisperTranscriptionEngine;
//! use scribe_rs::types::Metadata;
//! use tokio_util::sync::CancellationToken;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let runner = PipelineRunner {
//!     input: Box::new(CpalAudioInput::new()?),
//!     preprocessor: Box::new(DefaultPreprocessor { target_sample_rate: 16000 }),
//!     engine: Box::new(WhisperTranscriptionEngine::new("base", 16000)?),
//!     postprocessor: Box::new(NoopPostprocessor),
//!     sink: Box::new(StdoutOutputSink),
//!     cancel: CancellationToken::new(),
//!     metadata: Metadata { model: "base".into(), language: None },
//! };
//!
//! runner.run().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Implementing custom stages
//!
//! Every stage is a trait. Implement [`pipeline::traits::OutputSink`] for a custom
//! output target, [`pipeline::traits::AudioInput`] for a custom audio source, etc.
//! The [`pipeline::PipelineRunner`] accepts any combination of trait objects.

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
