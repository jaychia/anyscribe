use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::{broadcast, mpsc};
use tokio_util::sync::CancellationToken;

use anyscribe::audio::wav::WavWriter;
use anyscribe::chunk::PassthroughChunker;
use anyscribe::error::ScribeError;
use anyscribe::pipeline::traits::*;
use anyscribe::pipeline::PipelineRunner;
use anyscribe::types::*;

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
        output: mpsc::Sender<AudioNotification>,
        cancel: CancellationToken,
    ) -> Result<(), ScribeError> {
        let mut wav_writer =
            WavWriter::new(&self.info.wav_path, self.info.sample_rate, self.info.channels)?;

        for chunk in &self.chunks {
            if cancel.is_cancelled() {
                break;
            }
            wav_writer.write_samples(chunk)?;
            if output
                .send(AudioNotification {
                    num_samples: chunk.len(),
                })
                .await
                .is_err()
            {
                break;
            }
        }

        wav_writer.finalize()?;
        drop(output);
        Ok(())
    }
}

// ── Mock Preprocessor (passthrough, reads from WAV) ───────────────

struct MockPreprocessor;

#[async_trait]
impl Preprocessor for MockPreprocessor {
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<AudioNotification>,
        output: mpsc::Sender<AudioChunk>,
        info: AudioInputInfo,
    ) -> Result<(), ScribeError> {
        let mut wav_reader = anyscribe::audio::wav::WavReader::new(&info.wav_path)?;

        while let Some(notification) = input.recv().await {
            let raw = wav_reader.read_samples(notification.num_samples)?;
            let chunk = AudioChunk {
                samples: raw,
                sample_rate: info.sample_rate,
                offset_secs: 0.0,
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
                start: chunk.offset_secs,
                end: chunk.offset_secs + duration,
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

// ── Collecting subscriber helper ──────────────────────────────────

async fn collect_segments(
    mut rx: broadcast::Receiver<Segment>,
    collected: Arc<Mutex<Vec<Segment>>>,
) {
    loop {
        match rx.recv().await {
            Ok(seg) => collected.lock().unwrap().push(seg),
            Err(broadcast::error::RecvError::Closed) => break,
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
        }
    }
}

// ── Helper: create mock input with temp WAV ───────────────────────

fn mock_input(dir: &tempfile::TempDir, chunks: Vec<Vec<f32>>) -> MockAudioInput {
    MockAudioInput {
        info: AudioInputInfo {
            sample_rate: 16000,
            channels: 1,
            wav_path: dir.path().join("test.wav"),
        },
        chunks,
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_pipeline_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let collected = Arc::new(Mutex::new(Vec::new()));

    let input = mock_input(
        &dir,
        vec![
            vec![0.1; 1600], // 100ms at 16kHz
            vec![0.2; 1600],
            vec![0.3; 1600],
        ],
    );

    let runner = PipelineRunner::new(
        Box::new(input),
        Box::new(MockPreprocessor),
        Box::new(PassthroughChunker),
        Box::new(MockTranscriptionEngine {
            updated_meta: Metadata::default(),
        }),
        Box::new(MockPostprocessor),
        CancellationToken::new(),
        Metadata {
            model: "test".to_string(),
            language: None,
        },
    );

    let rx = runner.subscribe();
    let collected_clone = collected.clone();
    let sub_h = tokio::spawn(collect_segments(rx, collected_clone));

    runner.run().await.unwrap();
    sub_h.await.unwrap();

    let segs = collected.lock().unwrap();
    assert_eq!(segs.len(), 3);
    assert_eq!(segs[0].text, "segment 0");
    assert_eq!(segs[1].text, "segment 1");
    assert_eq!(segs[2].text, "segment 2");

    // WAV file should exist on disk
    assert!(dir.path().join("test.wav").exists());
}

#[tokio::test]
async fn test_pipeline_multiple_subscribers() {
    let dir = tempfile::tempdir().unwrap();
    let collected_a = Arc::new(Mutex::new(Vec::new()));
    let collected_b = Arc::new(Mutex::new(Vec::new()));

    let input = mock_input(&dir, vec![vec![0.1; 1600], vec![0.2; 1600]]);

    let runner = PipelineRunner::new(
        Box::new(input),
        Box::new(MockPreprocessor),
        Box::new(PassthroughChunker),
        Box::new(MockTranscriptionEngine {
            updated_meta: Metadata::default(),
        }),
        Box::new(MockPostprocessor),
        CancellationToken::new(),
        Metadata::default(),
    );

    let rx_a = runner.subscribe();
    let rx_b = runner.subscribe();
    let ha = tokio::spawn(collect_segments(rx_a, collected_a.clone()));
    let hb = tokio::spawn(collect_segments(rx_b, collected_b.clone()));

    runner.run().await.unwrap();
    ha.await.unwrap();
    hb.await.unwrap();

    let a = collected_a.lock().unwrap();
    let b = collected_b.lock().unwrap();
    assert_eq!(a.len(), 2);
    assert_eq!(b.len(), 2);
    assert_eq!(a[0].text, "segment 0");
    assert_eq!(b[1].text, "segment 1");
}

#[tokio::test]
async fn test_pipeline_empty_input() {
    let dir = tempfile::tempdir().unwrap();
    let collected = Arc::new(Mutex::new(Vec::new()));

    let input = mock_input(&dir, vec![]);

    let runner = PipelineRunner::new(
        Box::new(input),
        Box::new(MockPreprocessor),
        Box::new(PassthroughChunker),
        Box::new(MockTranscriptionEngine {
            updated_meta: Metadata::default(),
        }),
        Box::new(MockPostprocessor),
        CancellationToken::new(),
        Metadata::default(),
    );

    let rx = runner.subscribe();
    let collected_clone = collected.clone();
    let sub_h = tokio::spawn(collect_segments(rx, collected_clone));

    runner.run().await.unwrap();
    sub_h.await.unwrap();

    let segs = collected.lock().unwrap();
    assert!(segs.is_empty());
}

#[tokio::test]
async fn test_pipeline_cancellation() {
    let dir = tempfile::tempdir().unwrap();
    let collected = Arc::new(Mutex::new(Vec::new()));
    let cancel = CancellationToken::new();
    let wav_path = dir.path().join("cancel.wav");

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
            output: mpsc::Sender<AudioNotification>,
            cancel: CancellationToken,
        ) -> Result<(), ScribeError> {
            let mut wav_writer =
                WavWriter::new(&self.info.wav_path, self.info.sample_rate, self.info.channels)?;

            let chunk = vec![0.5; 1600];
            wav_writer.write_samples(&chunk)?;
            let _ = output
                .send(AudioNotification {
                    num_samples: chunk.len(),
                })
                .await;

            cancel.cancelled().await;
            wav_writer.finalize()?;
            Ok(())
        }
    }

    let cancel_clone = cancel.clone();
    let runner = PipelineRunner::new(
        Box::new(CancelWaitInput {
            info: AudioInputInfo {
                sample_rate: 16000,
                channels: 1,
                wav_path,
            },
        }),
        Box::new(MockPreprocessor),
        Box::new(PassthroughChunker),
        Box::new(MockTranscriptionEngine {
            updated_meta: Metadata::default(),
        }),
        Box::new(MockPostprocessor),
        cancel,
        Metadata::default(),
    );

    let rx = runner.subscribe();
    let collected_clone = collected.clone();
    let sub_h = tokio::spawn(collect_segments(rx, collected_clone));

    // Cancel after a short delay
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        cancel_clone.cancel();
    });

    runner.run().await.unwrap();
    sub_h.await.unwrap();

    let segs = collected.lock().unwrap();
    // Should have at least the one segment from the chunk sent before cancel
    assert!(segs.len() >= 1);
}

// ── Mock TranscriptionEngine that skips certain chunks ────────────

struct FailingMockTranscriptionEngine {
    fail_on: HashSet<usize>,
    updated_meta: Metadata,
}

#[async_trait]
impl TranscriptionEngine for FailingMockTranscriptionEngine {
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<AudioChunk>,
        output: mpsc::Sender<Segment>,
        _cancel: CancellationToken,
        metadata: Metadata,
    ) -> Result<(), ScribeError> {
        let mut chunk_idx = 0;
        while let Some(chunk) = input.recv().await {
            if self.fail_on.contains(&chunk_idx) {
                // Simulate internal error handling: log and skip
                tracing::warn!(chunk_idx, "mock: skipping chunk");
            } else {
                let duration = chunk.samples.len() as f64 / chunk.sample_rate as f64;
                let seg = Segment {
                    start: chunk.offset_secs,
                    end: chunk.offset_secs + duration,
                    text: format!("segment {chunk_idx}"),
                };
                if output.send(seg).await.is_err() {
                    break;
                }
            }
            chunk_idx += 1;
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

#[tokio::test]
async fn test_pipeline_skips_failing_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let collected = Arc::new(Mutex::new(Vec::new()));

    let input = mock_input(
        &dir,
        vec![
            vec![0.1; 1600], // chunk 0 — will succeed
            vec![0.2; 1600], // chunk 1 — will fail
            vec![0.3; 1600], // chunk 2 — will succeed
            vec![0.4; 1600], // chunk 3 — will fail
            vec![0.5; 1600], // chunk 4 — will succeed
        ],
    );

    let fail_on: HashSet<usize> = [1, 3].into_iter().collect();

    let runner = PipelineRunner::new(
        Box::new(input),
        Box::new(MockPreprocessor),
        Box::new(PassthroughChunker),
        Box::new(FailingMockTranscriptionEngine {
            fail_on,
            updated_meta: Metadata::default(),
        }),
        Box::new(MockPostprocessor),
        CancellationToken::new(),
        Metadata::default(),
    );

    let rx = runner.subscribe();
    let collected_clone = collected.clone();
    let sub_h = tokio::spawn(collect_segments(rx, collected_clone));

    // Pipeline should complete successfully despite chunk failures
    runner.run().await.unwrap();
    sub_h.await.unwrap();

    let segs = collected.lock().unwrap();
    // Only 3 of 5 chunks should produce segments
    assert_eq!(segs.len(), 3);
    assert_eq!(segs[0].text, "segment 0");
    assert_eq!(segs[1].text, "segment 2");
    assert_eq!(segs[2].text, "segment 4");
}
