//! Real-time stdout output sink.

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::ScribeError;
use crate::pipeline::traits::OutputSink;
use crate::types::{format_timestamp, Metadata, Segment};

/// Prints each transcript segment to stdout as `[MM:SS] text` in real time.
pub struct StdoutOutputSink;

#[async_trait]
impl OutputSink for StdoutOutputSink {
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<Segment>,
        _metadata: Metadata,
    ) -> Result<(), ScribeError> {
        while let Some(seg) = input.recv().await {
            let text = seg.text.trim();
            if !text.is_empty() {
                println!("  [{}] {text}", format_timestamp(seg.start));
            }
        }
        Ok(())
    }
}
