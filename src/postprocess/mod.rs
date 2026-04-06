use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::ScribeError;
use crate::pipeline::traits::Postprocessor;
use crate::types::Segment;

/// Passthrough postprocessor — forwards segments unchanged.
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
