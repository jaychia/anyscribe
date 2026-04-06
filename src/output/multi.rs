use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::constants::SEGMENT_CHANNEL_CAPACITY;
use crate::error::ScribeError;
use crate::pipeline::traits::OutputSink;
use crate::types::{Metadata, Segment};

/// Fans out segments to multiple inner sinks, each running in its own task.
pub struct MultiOutputSink {
    pub sinks: Vec<Box<dyn OutputSink>>,
}

#[async_trait]
impl OutputSink for MultiOutputSink {
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<Segment>,
        metadata: Metadata,
    ) -> Result<(), ScribeError> {
        // Give each inner sink its own channel and spawn it
        let mut txs = Vec::new();
        let mut handles = Vec::new();

        for mut sink in self.sinks.drain(..) {
            let (tx, rx) = mpsc::channel::<Segment>(SEGMENT_CHANNEL_CAPACITY);
            txs.push(tx);
            let meta = metadata.clone();
            handles.push(tokio::spawn(async move { sink.run(rx, meta).await }));
        }

        // Fan out incoming segments to all inner sinks
        while let Some(seg) = input.recv().await {
            for tx in &txs {
                let _ = tx.send(seg.clone()).await;
            }
        }

        // Close all inner channels
        drop(txs);

        // Wait for all inner sinks to finish
        for h in handles {
            h.await
                .map_err(|e| ScribeError::Pipeline(format!("sink panicked: {e}")))??;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct CollectingSink {
        collected: Arc<Mutex<Vec<Segment>>>,
    }

    #[async_trait]
    impl OutputSink for CollectingSink {
        async fn run(
            &mut self,
            mut input: mpsc::Receiver<Segment>,
            _metadata: Metadata,
        ) -> Result<(), ScribeError> {
            while let Some(seg) = input.recv().await {
                self.collected.lock().unwrap().push(seg);
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_multi_sink_fans_out() {
        let collected_a = Arc::new(Mutex::new(Vec::new()));
        let collected_b = Arc::new(Mutex::new(Vec::new()));

        let sink_a = CollectingSink {
            collected: collected_a.clone(),
        };
        let sink_b = CollectingSink {
            collected: collected_b.clone(),
        };

        let mut multi = MultiOutputSink {
            sinks: vec![Box::new(sink_a), Box::new(sink_b)],
        };

        let (tx, rx) = mpsc::channel(10);
        tx.send(Segment {
            start: 0.0,
            end: 1.0,
            text: "hello".to_string(),
        })
        .await
        .unwrap();
        tx.send(Segment {
            start: 1.0,
            end: 2.0,
            text: "world".to_string(),
        })
        .await
        .unwrap();
        drop(tx);

        multi.run(rx, Metadata::default()).await.unwrap();

        let a = collected_a.lock().unwrap();
        let b = collected_b.lock().unwrap();
        assert_eq!(a.len(), 2);
        assert_eq!(b.len(), 2);
        assert_eq!(a[0].text, "hello");
        assert_eq!(b[1].text, "world");
    }
}
