use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use scribe_rs::error::ScribeError;
use scribe_rs::pipeline::traits::*;
use scribe_rs::pipeline::PipelineRunner;
use scribe_rs::types::*;

// ── Mock AudioInput ────────────────────────────────────────────────

struct MockAudioInput {
    info: AudioInputInfo,
    chunks: Vec<Vec<f32>>,
}

#[async_trait]
impl AudioInput for MockAudioInput {
    fn info(&self) -> &AudioInputInfo {
        &self.info
    }

    async fn run(
        &mut self,
        output: mpsc::Sender<Vec<f32>>,
        cancel: CancellationToken,
    ) -> Result<(), ScribeError> {
        for chunk in &self.chunks {
            if cancel.is_cancelled() {
                break;
            }
            if output.send(chunk.clone()).await.is_err() {
                break;
            }
        }
        // Drop output sender to signal end of audio
        drop(output);
        Ok(())
    }
}

// ── Mock Preprocessor (passthrough) ────────────────────────────────

struct MockPreprocessor;

#[async_trait]
impl Preprocessor for MockPreprocessor {
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<Vec<f32>>,
        output: mpsc::Sender<AudioChunk>,
        info: AudioInputInfo,
    ) -> Result<(), ScribeError> {
        while let Some(raw) = input.recv().await {
            let chunk = AudioChunk {
                samples: raw,
                sample_rate: info.sample_rate,
            };
            if output.send(chunk).await.is_err() {
                break;
            }
        }
        Ok(())
    }
}

// ── Mock TranscriptionEngine ───────────────────────────────────────

struct MockTranscriptionEngine {
    updated_meta: Metadata,
}

#[async_trait]
impl TranscriptionEngine for MockTranscriptionEngine {
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<AudioChunk>,
        output: mpsc::Sender<Segment>,
        _cancel: CancellationToken,
        metadata: Metadata,
    ) -> Result<(), ScribeError> {
        let mut seg_idx = 0;
        while let Some(chunk) = input.recv().await {
            let duration = chunk.samples.len() as f64 / chunk.sample_rate as f64;
            let seg = Segment {
                start: seg_idx as f64 * duration,
                end: (seg_idx + 1) as f64 * duration,
                text: format!("segment {seg_idx}"),
            };
            seg_idx += 1;
            if output.send(seg).await.is_err() {
                break;
            }
        }

        self.updated_meta = Metadata {
            model: metadata.model,
            language: Some("en".to_string()),
        };

        Ok(())
    }

    fn updated_metadata(&self) -> Metadata {
        self.updated_meta.clone()
    }
}

// ── Mock Postprocessor (passthrough) ───────────────────────────────

struct MockPostprocessor;

#[async_trait]
impl Postprocessor for MockPostprocessor {
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

// ── Mock OutputSink (collecting) ───────────────────────────────────

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

// ── Tests ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_pipeline_end_to_end() {
    let collected = Arc::new(Mutex::new(Vec::new()));

    let input = MockAudioInput {
        info: AudioInputInfo {
            sample_rate: 16000,
            channels: 1,
        },
        chunks: vec![
            vec![0.1; 1600], // 100ms at 16kHz
            vec![0.2; 1600],
            vec![0.3; 1600],
        ],
    };

    let runner = PipelineRunner {
        input: Box::new(input),
        preprocessor: Box::new(MockPreprocessor),
        engine: Box::new(MockTranscriptionEngine {
            updated_meta: Metadata::default(),
        }),
        postprocessor: Box::new(MockPostprocessor),
        sink: Box::new(CollectingSink {
            collected: collected.clone(),
        }),
        cancel: CancellationToken::new(),
        metadata: Metadata {
            model: "test".to_string(),
            language: None,
        },
    };

    runner.run().await.unwrap();

    let segs = collected.lock().unwrap();
    assert_eq!(segs.len(), 3);
    assert_eq!(segs[0].text, "segment 0");
    assert_eq!(segs[1].text, "segment 1");
    assert_eq!(segs[2].text, "segment 2");
}

#[tokio::test]
async fn test_pipeline_empty_input() {
    let collected = Arc::new(Mutex::new(Vec::new()));

    let input = MockAudioInput {
        info: AudioInputInfo {
            sample_rate: 16000,
            channels: 1,
        },
        chunks: vec![], // No audio
    };

    let runner = PipelineRunner {
        input: Box::new(input),
        preprocessor: Box::new(MockPreprocessor),
        engine: Box::new(MockTranscriptionEngine {
            updated_meta: Metadata::default(),
        }),
        postprocessor: Box::new(MockPostprocessor),
        sink: Box::new(CollectingSink {
            collected: collected.clone(),
        }),
        cancel: CancellationToken::new(),
        metadata: Metadata::default(),
    };

    runner.run().await.unwrap();

    let segs = collected.lock().unwrap();
    assert!(segs.is_empty());
}

#[tokio::test]
async fn test_pipeline_cancellation() {
    let collected = Arc::new(Mutex::new(Vec::new()));
    let cancel = CancellationToken::new();

    // AudioInput that sends one chunk then waits for cancel
    struct CancelWaitInput {
        info: AudioInputInfo,
    }

    #[async_trait]
    impl AudioInput for CancelWaitInput {
        fn info(&self) -> &AudioInputInfo {
            &self.info
        }

        async fn run(
            &mut self,
            output: mpsc::Sender<Vec<f32>>,
            cancel: CancellationToken,
        ) -> Result<(), ScribeError> {
            let _ = output.send(vec![0.5; 1600]).await;
            cancel.cancelled().await;
            Ok(())
        }
    }

    let cancel_clone = cancel.clone();
    let runner = PipelineRunner {
        input: Box::new(CancelWaitInput {
            info: AudioInputInfo {
                sample_rate: 16000,
                channels: 1,
            },
        }),
        preprocessor: Box::new(MockPreprocessor),
        engine: Box::new(MockTranscriptionEngine {
            updated_meta: Metadata::default(),
        }),
        postprocessor: Box::new(MockPostprocessor),
        sink: Box::new(CollectingSink {
            collected: collected.clone(),
        }),
        cancel,
        metadata: Metadata::default(),
    };

    // Cancel after a short delay
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        cancel_clone.cancel();
    });

    runner.run().await.unwrap();

    let segs = collected.lock().unwrap();
    // Should have at least the one segment from the chunk sent before cancel
    assert!(segs.len() >= 1);
}
