//! Pipeline trait definitions.
//!
//! These five traits define the contract for each stage of the transcription
//! pipeline. Every stage follows the same pattern: `async fn run()` takes
//! channel endpoints, processes data until the input closes or cancellation
//! is requested, then returns `Result<(), ScribeError>`.

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::ScribeError;
use crate::types::{AudioChunk, Metadata, Segment};

// ── Source ─────────────────────────────────────────────────────────

/// Device format reported by an [`AudioInput`] implementation.
///
/// Available synchronously via [`AudioInput::info`] after construction,
/// before [`AudioInput::run`] is called. The [`Preprocessor`] uses this
/// to know the source sample rate and channel count for conversion.
#[derive(Debug, Clone)]
pub struct AudioInputInfo {
    /// Native sample rate of the audio device (e.g., 44100, 48000).
    pub sample_rate: u32,
    /// Number of interleaved channels (1 = mono, 2 = stereo).
    pub channels: u16,
}

/// Push-based audio source.
///
/// # Contract
///
/// - **`info()`** returns device metadata. Must be valid after construction,
///   before `run()` is called. The pipeline reads this to configure downstream
///   stages.
///
/// - **`run()`** starts capturing audio and pushes raw interleaved `f32` PCM
///   samples into `output`. Runs until `cancel` is triggered. Implementations
///   should use `try_send` (not `send().await`) since hardware callbacks cannot
///   block — drop frames and track the count internally.
///
/// # Example
///
/// See [`crate::audio::cpal_input::CpalAudioInput`] for the default
/// implementation using the system microphone.
#[async_trait]
pub trait AudioInput: Send {
    /// Returns the audio device format. Available immediately after construction.
    fn info(&self) -> &AudioInputInfo;

    /// Starts capturing audio and pushing into `output` until `cancel` fires.
    async fn run(
        &mut self,
        output: mpsc::Sender<Vec<f32>>,
        cancel: CancellationToken,
    ) -> Result<(), ScribeError>;
}

// ── Processing stages ──────────────────────────────────────────────

/// Transforms raw device audio into normalized mono chunks at a target sample rate.
///
/// Reads raw interleaved PCM from `input`, applies format conversion (mono
/// downmix, resampling, normalization), and sends [`AudioChunk`]s to `output`.
/// Runs until the input channel closes.
///
/// See [`crate::preprocess::DefaultPreprocessor`] for the built-in implementation.
#[async_trait]
pub trait Preprocessor: Send {
    async fn run(
        &mut self,
        input: mpsc::Receiver<Vec<f32>>,
        output: mpsc::Sender<AudioChunk>,
        info: AudioInputInfo,
    ) -> Result<(), ScribeError>;
}

/// Converts preprocessed audio into timestamped text segments.
///
/// Accumulates [`AudioChunk`]s into transcription windows, runs inference,
/// and sends [`Segment`]s to `output`. Respects `cancel` for graceful shutdown
/// (drains the remaining buffer before exiting).
///
/// After `run()` completes, call [`updated_metadata()`](TranscriptionEngine::updated_metadata)
/// to retrieve detected language and other metadata discovered during transcription.
///
/// # CPU-bound work
///
/// Implementations that perform CPU-intensive inference (e.g., whisper) should
/// use `tokio::task::spawn_blocking` internally to avoid starving the async runtime.
///
/// See [`crate::transcribe::whisper::WhisperTranscriptionEngine`] for the built-in implementation.
#[async_trait]
pub trait TranscriptionEngine: Send {
    async fn run(
        &mut self,
        input: mpsc::Receiver<AudioChunk>,
        output: mpsc::Sender<Segment>,
        cancel: CancellationToken,
        metadata: Metadata,
    ) -> Result<(), ScribeError>;

    /// Returns metadata updated during transcription (e.g., detected language).
    ///
    /// Only meaningful after `run()` has completed.
    fn updated_metadata(&self) -> Metadata;
}

/// Filters or transforms transcript segments between the engine and subscribers.
///
/// Reads [`Segment`]s from `input`, optionally transforms them, and sends
/// results to `output`. Runs until the input channel closes.
///
/// The built-in [`crate::postprocess::NoopPostprocessor`] passes segments
/// through unchanged. Custom implementations could filter noise tokens
/// (e.g., `[music]`), merge short segments, apply punctuation correction, etc.
#[async_trait]
pub trait Postprocessor: Send {
    async fn run(
        &mut self,
        input: mpsc::Receiver<Segment>,
        output: mpsc::Sender<Segment>,
    ) -> Result<(), ScribeError>;
}
