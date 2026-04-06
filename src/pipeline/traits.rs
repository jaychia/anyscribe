use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::ScribeError;
use crate::types::{AudioChunk, Metadata, Segment};

// ── Source ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AudioInputInfo {
    pub sample_rate: u32,
    pub channels: u16,
}

/// Push-based audio source.
/// - `info()` returns device info (available after construction, before run)
/// - `run()` starts capture, pushes into output until cancelled
/// - Backpressure: try_send fails when full; dropped internally
#[async_trait]
pub trait AudioInput: Send {
    fn info(&self) -> &AudioInputInfo;

    async fn run(
        &mut self,
        output: mpsc::Sender<Vec<f32>>,
        cancel: CancellationToken,
    ) -> Result<(), ScribeError>;
}

// ── Processing stages ──────────────────────────────────────────────

/// Reads raw audio, preprocesses (mono + resample + normalize), sends AudioChunks.
/// Runs until input channel closes.
#[async_trait]
pub trait Preprocessor: Send {
    async fn run(
        &mut self,
        input: mpsc::Receiver<Vec<f32>>,
        output: mpsc::Sender<AudioChunk>,
        info: AudioInputInfo,
    ) -> Result<(), ScribeError>;
}

/// Buffers preprocessed audio into windows, transcribes each, sends Segments.
/// After run() completes, call `updated_metadata()` to get detected language etc.
#[async_trait]
pub trait TranscriptionEngine: Send {
    async fn run(
        &mut self,
        input: mpsc::Receiver<AudioChunk>,
        output: mpsc::Sender<Segment>,
        cancel: CancellationToken,
        metadata: Metadata,
    ) -> Result<(), ScribeError>;

    fn updated_metadata(&self) -> Metadata;
}

/// Filters/transforms segments. Owns its async channel loop.
#[async_trait]
pub trait Postprocessor: Send {
    async fn run(
        &mut self,
        input: mpsc::Receiver<Segment>,
        output: mpsc::Sender<Segment>,
    ) -> Result<(), ScribeError>;
}

// ── Sink ───────────────────────────────────────────────────────────

/// Owns its async loop: reads segments from input channel until closed.
#[async_trait]
pub trait OutputSink: Send {
    async fn run(
        &mut self,
        input: mpsc::Receiver<Segment>,
        metadata: Metadata,
    ) -> Result<(), ScribeError>;
}
