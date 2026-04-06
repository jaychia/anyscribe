//! Pipeline runner and trait re-exports.

pub mod traits;

pub use traits::*;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::constants::{AUDIO_CHANNEL_CAPACITY, SEGMENT_CHANNEL_CAPACITY};
use crate::error::ScribeError;
use crate::types::{AudioChunk, Metadata, Segment};

/// Wires five pipeline stages together and runs them concurrently.
///
/// Each stage is spawned as an independent `tokio::spawn` task. Bounded
/// channels connect adjacent stages. All tasks are joined with `tokio::join!`
/// — no task is leaked on error or cancellation.
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
/// #     eng: Box<dyn anyscribe::pipeline::TranscriptionEngine>,
/// #     post: Box<dyn anyscribe::pipeline::Postprocessor>,
/// #     sink: Box<dyn anyscribe::pipeline::OutputSink>,
/// # ) -> Result<(), anyscribe::error::ScribeError> {
/// let runner = PipelineRunner {
///     input,
///     preprocessor: pre,
///     engine: eng,
///     postprocessor: post,
///     sink,
///     cancel: CancellationToken::new(),
///     metadata: Metadata::default(),
/// };
/// runner.run().await
/// # }
/// ```
pub struct PipelineRunner {
    pub input: Box<dyn AudioInput>,
    pub preprocessor: Box<dyn Preprocessor>,
    pub engine: Box<dyn TranscriptionEngine>,
    pub postprocessor: Box<dyn Postprocessor>,
    pub sink: Box<dyn OutputSink>,
    pub cancel: CancellationToken,
    pub metadata: Metadata,
}

impl PipelineRunner {
    /// Runs the full pipeline to completion.
    ///
    /// Creates bounded channels, spawns all five stages, and blocks until
    /// every task finishes. Returns the first error encountered, or `Ok(())`
    /// if all stages completed successfully.
    pub async fn run(self) -> Result<(), ScribeError> {
        let (raw_tx, raw_rx) = mpsc::channel::<Vec<f32>>(AUDIO_CHANNEL_CAPACITY);
        let (proc_tx, proc_rx) = mpsc::channel::<AudioChunk>(AUDIO_CHANNEL_CAPACITY);
        let (seg_tx, seg_rx) = mpsc::channel::<Segment>(SEGMENT_CHANNEL_CAPACITY);
        let (post_tx, post_rx) = mpsc::channel::<Segment>(SEGMENT_CHANNEL_CAPACITY);

        let info = self.input.info().clone();

        let mut input = self.input;
        let cancel_a = self.cancel.clone();
        let input_h = tokio::spawn(async move { input.run(raw_tx, cancel_a).await });

        let mut pre = self.preprocessor;
        let pre_h = tokio::spawn(async move { pre.run(raw_rx, proc_tx, info).await });

        let mut eng = self.engine;
        let cancel_e = self.cancel.clone();
        let meta = self.metadata.clone();
        let eng_h = tokio::spawn(async move {
            eng.run(proc_rx, seg_tx, cancel_e, meta).await?;
            Ok::<_, ScribeError>(eng)
        });

        let mut post = self.postprocessor;
        let post_h = tokio::spawn(async move { post.run(seg_rx, post_tx).await });

        let mut sink = self.sink;
        let meta_clone = self.metadata.clone();
        let sink_h = tokio::spawn(async move { sink.run(post_rx, meta_clone).await });

        let (input_r, pre_r, eng_r, post_r, sink_r) =
            tokio::join!(input_h, pre_h, eng_h, post_h, sink_h);

        input_r.map_err(|e| ScribeError::Pipeline(format!("audio panicked: {e}")))??;
        pre_r.map_err(|e| ScribeError::Pipeline(format!("preprocess panicked: {e}")))??;
        eng_r.map_err(|e| ScribeError::Pipeline(format!("transcribe panicked: {e}")))??;
        post_r.map_err(|e| ScribeError::Pipeline(format!("postprocess panicked: {e}")))??;
        sink_r.map_err(|e| ScribeError::Pipeline(format!("sink panicked: {e}")))??;

        Ok(())
    }
}
