//! Postprocessor implementations.

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::ScribeError;
use crate::pipeline::traits::Postprocessor;
use crate::types::Segment;

/// Passthrough postprocessor that forwards segments unchanged.
///
/// Use this as the default. Replace with a custom [`Postprocessor`] to
/// filter noise tokens, merge short segments, apply punctuation, etc.
pub struct NoopPostprocessor;

#[async_trait]
impl Postprocessor for NoopPostprocessor {
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<Segment>,
        output: mpsc::Sender<Segment>,
    ) -> Result<(), ScribeError> {
        while let Some(seg) = input.recv().await {
            if output.send(seg).await.is_err() {
                break;
            }
        }
        Ok(())
    }
}
