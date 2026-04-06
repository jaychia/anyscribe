//! Pipeline runner and trait re-exports.

pub mod traits;

pub use traits::*;

use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

use crate::constants::{AUDIO_CHANNEL_CAPACITY, SEGMENT_CHANNEL_CAPACITY};
use crate::error::ScribeError;
use crate::types::{AudioChunk, AudioNotification, Metadata, Segment};

/// Wires pipeline stages together and broadcasts segments to subscribers.
///
/// Each stage is spawned as an independent `tokio::spawn` task. Bounded
/// channels connect adjacent stages. Segments emitted by the postprocessor
/// are broadcast to all subscribers registered via [`subscribe`](PipelineRunner::subscribe).
///
/// # Cancellation
///
/// Pass a [`CancellationToken`] and call `.cancel()` on it to trigger
/// graceful shutdown. The audio source stops, channels drain in sequence,
/// and all tasks complete.
///
/// # Example
///
/// ```rust,no_run
/// # use anyscribe::pipeline::PipelineRunner;
/// # use anyscribe::types::Metadata;
/// # use tokio_util::sync::CancellationToken;
/// # async fn example(
/// #     input: Box<dyn anyscribe::pipeline::AudioInput>,
/// #     pre: Box<dyn anyscribe::pipeline::Preprocessor>,
/// #     chk: Box<dyn anyscribe::pipeline::Chunker>,
/// #     eng: Box<dyn anyscribe::pipeline::TranscriptionEngine>,
/// #     post: Box<dyn anyscribe::pipeline::Postprocessor>,
/// # ) -> Result<(), anyscribe::error::ScribeError> {
/// let runner = PipelineRunner::new(
///     input,
///     pre,
///     chk,
///     eng,
///     post,
///     CancellationToken::new(),
///     Metadata::default(),
/// );
///
/// // Subscribe before running — each call returns an independent receiver.
/// let rx = runner.subscribe();
/// tokio::spawn(async move {
///     // consume segments from rx ...
/// });
///
/// runner.run().await
/// # }
/// ```
pub struct PipelineRunner {
    pub input: Box<dyn AudioInput>,
    pub preprocessor: Box<dyn Preprocessor>,
    pub chunker: Box<dyn Chunker>,
    pub engine: Box<dyn TranscriptionEngine>,
    pub postprocessor: Box<dyn Postprocessor>,
    pub cancel: CancellationToken,
    pub metadata: Metadata,
    segment_tx: broadcast::Sender<Segment>,
}

impl PipelineRunner {
    /// Creates a new pipeline runner with a broadcast channel for subscribers.
    pub fn new(
        input: Box<dyn AudioInput>,
        preprocessor: Box<dyn Preprocessor>,
        chunker: Box<dyn Chunker>,
        engine: Box<dyn TranscriptionEngine>,
        postprocessor: Box<dyn Postprocessor>,
        cancel: CancellationToken,
        metadata: Metadata,
    ) -> Self {
        let (segment_tx, _) = broadcast::channel(SEGMENT_CHANNEL_CAPACITY);
        Self {
            input,
            preprocessor,
            chunker,
            engine,
            postprocessor,
            cancel,
            metadata,
            segment_tx,
        }
    }

    /// Returns a new subscriber that receives all segments emitted by the pipeline.
    ///
    /// Call this before [`run`](PipelineRunner::run) — once the pipeline starts,
    /// any segments sent before a subscriber is created are lost to that subscriber.
    pub fn subscribe(&self) -> broadcast::Receiver<Segment> {
        self.segment_tx.subscribe()
    }

    /// Runs the full pipeline to completion.
    ///
    /// Creates bounded channels, spawns all stages, and blocks until
    /// every task finishes. Segments are broadcast to all subscribers.
    /// Returns the first error encountered, or `Ok(())` if all stages
    /// completed successfully.
    #[tracing::instrument(name = "pipeline", skip_all)]
    pub async fn run(self) -> Result<(), ScribeError> {
        let (raw_tx, raw_rx) = mpsc::channel::<AudioNotification>(AUDIO_CHANNEL_CAPACITY);
        let (proc_tx, proc_rx) = mpsc::channel::<AudioChunk>(AUDIO_CHANNEL_CAPACITY);
        let (chunked_tx, chunked_rx) = mpsc::channel::<AudioChunk>(AUDIO_CHANNEL_CAPACITY);
        let (seg_tx, seg_rx) = mpsc::channel::<Segment>(SEGMENT_CHANNEL_CAPACITY);
        let (post_tx, post_rx) = mpsc::channel::<Segment>(SEGMENT_CHANNEL_CAPACITY);

        let info = self.input.info().clone();

        let mut input = self.input;
        let cancel_a = self.cancel.clone();
        let input_h = tokio::spawn(
            async move { input.run(raw_tx, cancel_a).await }
                .instrument(tracing::info_span!("stage", name = "audio_input")),
        );

        let mut pre = self.preprocessor;
        let pre_h = tokio::spawn(
            async move { pre.run(raw_rx, proc_tx, info).await }
                .instrument(tracing::info_span!("stage", name = "preprocessor")),
        );

        let mut chk = self.chunker;
        let cancel_c = self.cancel.clone();
        let chunk_h = tokio::spawn(
            async move { chk.run(proc_rx, chunked_tx, cancel_c).await }
                .instrument(tracing::info_span!("stage", name = "chunker")),
        );

        let mut eng = self.engine;
        let cancel_e = self.cancel.clone();
        let meta = self.metadata.clone();
        let eng_h = tokio::spawn(
            async move {
                eng.run(chunked_rx, seg_tx, cancel_e, meta).await?;
                Ok::<_, ScribeError>(eng)
            }
            .instrument(tracing::info_span!("stage", name = "transcription_engine")),
        );

        let mut post = self.postprocessor;
        let post_h = tokio::spawn(
            async move { post.run(seg_rx, post_tx).await }
                .instrument(tracing::info_span!("stage", name = "postprocessor")),
        );

        // Broadcast segments from the postprocessor to all subscribers.
        let segment_tx = self.segment_tx;
        let bcast_h = tokio::spawn(
            async move {
                let mut post_rx = post_rx;
                while let Some(seg) = post_rx.recv().await {
                    // Ignore errors — no active subscribers is fine.
                    let _ = segment_tx.send(seg);
                }
                Ok::<_, ScribeError>(())
            }
            .instrument(tracing::info_span!("stage", name = "broadcast")),
        );

        let (input_r, pre_r, chunk_r, eng_r, post_r, bcast_r) =
            tokio::join!(input_h, pre_h, chunk_h, eng_h, post_h, bcast_h);

        input_r.map_err(|e| ScribeError::Pipeline(format!("audio panicked: {e}")))??;
        pre_r.map_err(|e| ScribeError::Pipeline(format!("preprocess panicked: {e}")))??;
        chunk_r.map_err(|e| ScribeError::Pipeline(format!("chunker panicked: {e}")))??;
        eng_r.map_err(|e| ScribeError::Pipeline(format!("transcribe panicked: {e}")))??;
        post_r.map_err(|e| ScribeError::Pipeline(format!("postprocess panicked: {e}")))??;
        bcast_r.map_err(|e| ScribeError::Pipeline(format!("broadcast panicked: {e}")))??;

        Ok(())
    }
}
