//! Real-time stdout output sink.

use tokio::sync::broadcast;

use crate::error::ScribeError;
use crate::types::{format_timestamp, Segment};

/// Prints each transcript segment to stdout as `[MM:SS] text` in real time.
pub struct StdoutOutputSink;

impl StdoutOutputSink {
    /// Consumes segments from a broadcast receiver and prints them to stdout.
    pub async fn run(
        self,
        mut input: broadcast::Receiver<Segment>,
    ) -> Result<(), ScribeError> {
        loop {
            match input.recv().await {
                Ok(seg) => {
                    let text = seg.text.trim();
                    if !text.is_empty() {
                        println!("  [{}] {text}", format_timestamp(seg.start));
                    }
                }
                Err(broadcast::error::RecvError::Closed) => break,
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    log::warn!("stdout subscriber lagged, skipped {n} segments");
                }
            }
        }
        Ok(())
    }
}
