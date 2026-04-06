//! Audio chunking strategies for the transcription pipeline.
//!
//! The chunker sits between the preprocessor and the transcription engine,
//! accumulating small audio chunks into larger windows suitable for
//! transcription. Each emitted chunk carries an `offset_secs` so the
//! engine can produce correctly timestamped segments.

use async_trait::async_trait;
use tracing::{debug, warn};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::ScribeError;
use crate::pipeline::traits::Chunker;
use crate::types::AudioChunk;

/// Chunker that accumulates audio into fixed-duration windows with overlap.
///
/// This is the default strategy for local Whisper inference:
/// - Accumulates preprocessed audio into a buffer
/// - Emits windows of `chunk_duration_secs` (e.g., 30s)
/// - Retains `overlap_secs` (e.g., 5s) from the end of each window for
///   cross-boundary context in the next window
/// - Enforces a maximum buffer size to prevent unbounded memory growth
/// - Emits any remaining audio (> 1s) as a final chunk on shutdown
pub struct OverlapChunker {
    pub chunk_duration_secs: f64,
    pub overlap_secs: f64,
    pub max_buffer_secs: f64,
    pub sample_rate: u32,
}

#[async_trait]
impl Chunker for OverlapChunker {
    #[tracing::instrument(name = "chunker", skip_all, fields(
        chunk_secs = self.chunk_duration_secs,
        overlap_secs = self.overlap_secs,
        sample_rate = self.sample_rate,
    ))]
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<AudioChunk>,
        output: mpsc::Sender<AudioChunk>,
        cancel: CancellationToken,
    ) -> Result<(), ScribeError> {
        let chunk_samples = (self.chunk_duration_secs * self.sample_rate as f64) as usize;
        let overlap_samples = (self.overlap_secs * self.sample_rate as f64) as usize;
        let max_buffer_samples = (self.max_buffer_secs * self.sample_rate as f64) as usize;
        let min_final_samples = self.sample_rate as usize; // 1 second

        let mut buffer: Vec<f32> = Vec::new();
        let mut offset: f64 = 0.0;

        loop {
            tokio::select! {
                biased;
                chunk = input.recv() => {
                    match chunk {
                        Some(data) => {
                            buffer.extend_from_slice(&data.samples);

                            if buffer.len() > max_buffer_samples {
                                let excess = buffer.len() - max_buffer_samples;
                                buffer.drain(..excess);
                                warn!(cap_secs = self.max_buffer_secs, "audio buffer exceeded cap, oldest audio dropped");
                            }
                        }
                        None => break,
                    }
                }
                _ = cancel.cancelled() => {
                    while let Ok(data) = input.try_recv() {
                        buffer.extend_from_slice(&data.samples);
                    }
                    break;
                }
            }

            while buffer.len() >= chunk_samples {
                let window: Vec<f32> = buffer[..chunk_samples].to_vec();
                buffer = buffer[chunk_samples - overlap_samples..].to_vec();

                debug!(
                    "emitting window ({:.1}s), offset={offset:.1}s",
                    window.len() as f64 / self.sample_rate as f64,
                );

                let chunk = AudioChunk {
                    samples: window,
                    sample_rate: self.sample_rate,
                    offset_secs: offset,
                };

                if output.send(chunk).await.is_err() {
                    return Ok(());
                }

                offset += self.chunk_duration_secs - self.overlap_secs;
            }
        }

        // Emit remaining buffer if it has enough audio (> 1 second).
        if buffer.len() > min_final_samples {
            debug!(
                "final chunk: {} samples ({:.1}s)",
                buffer.len(),
                buffer.len() as f64 / self.sample_rate as f64,
            );

            let chunk = AudioChunk {
                samples: buffer,
                sample_rate: self.sample_rate,
                offset_secs: offset,
            };

            let _ = output.send(chunk).await;
        }

        Ok(())
    }
}

/// Chunker that passes audio through without windowing.
///
/// Each incoming chunk is forwarded directly with a cumulative `offset_secs`.
/// Useful for remote APIs that accept arbitrary-length audio, or for testing.
pub struct PassthroughChunker;

#[async_trait]
impl Chunker for PassthroughChunker {
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<AudioChunk>,
        output: mpsc::Sender<AudioChunk>,
        cancel: CancellationToken,
    ) -> Result<(), ScribeError> {
        let mut offset: f64 = 0.0;

        loop {
            tokio::select! {
                biased;
                chunk = input.recv() => {
                    match chunk {
                        Some(mut data) => {
                            let duration = data.samples.len() as f64 / data.sample_rate as f64;
                            data.offset_secs = offset;
                            offset += duration;
                            if output.send(data).await.is_err() {
                                break;
                            }
                        }
                        None => break,
                    }
                }
                _ = cancel.cancelled() => {
                    while let Ok(mut data) = input.try_recv() {
                        let duration = data.samples.len() as f64 / data.sample_rate as f64;
                        data.offset_secs = offset;
                        offset += duration;
                        if output.send(data).await.is_err() {
                            break;
                        }
                    }
                    break;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_overlap_chunker_produces_correct_windows() {
        let (in_tx, in_rx) = mpsc::channel(10);
        let (out_tx, mut out_rx) = mpsc::channel(10);

        let mut chunker = OverlapChunker {
            chunk_duration_secs: 3.0,
            overlap_secs: 1.0,
            max_buffer_secs: 30.0,
            sample_rate: 100, // 100 samples/sec for easy math
        };

        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();

        let handle = tokio::spawn(async move {
            chunker.run(in_rx, out_tx, cancel_clone).await
        });

        // Send 7 seconds of audio in 1-second chunks
        for i in 0..7 {
            let samples = vec![i as f32; 100]; // 1 second each
            in_tx.send(AudioChunk { samples, sample_rate: 100, offset_secs: 0.0 }).await.unwrap();
        }
        drop(in_tx);

        handle.await.unwrap().unwrap();

        // 7 chunks of 1s each arrive sequentially. With 3s windows + 1s overlap:
        // Chunks accumulate in buffer; every time buffer reaches 300, a window is emitted.
        // After each window, 100 samples (1s overlap) are retained.
        //
        // w0: after chunks 0-2, buffer=300 → emit, keep 100. offset=0.0
        // w1: after chunks 3-4, buffer=300 → emit, keep 100. offset=2.0
        // w2: after chunks 5-6, buffer=300 → emit, keep 100. offset=4.0
        // Input closes. 100 samples remain, not > 100 (1s) threshold → no final chunk.

        let w0 = out_rx.recv().await.unwrap();
        assert_eq!(w0.samples.len(), 300);
        assert!((w0.offset_secs - 0.0).abs() < f64::EPSILON);

        let w1 = out_rx.recv().await.unwrap();
        assert_eq!(w1.samples.len(), 300);
        assert!((w1.offset_secs - 2.0).abs() < f64::EPSILON);

        let w2 = out_rx.recv().await.unwrap();
        assert_eq!(w2.samples.len(), 300);
        assert!((w2.offset_secs - 4.0).abs() < f64::EPSILON);

        assert!(out_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn test_overlap_chunker_no_overlap() {
        let (in_tx, in_rx) = mpsc::channel(10);
        let (out_tx, mut out_rx) = mpsc::channel(10);

        let mut chunker = OverlapChunker {
            chunk_duration_secs: 2.0,
            overlap_secs: 0.0,
            max_buffer_secs: 30.0,
            sample_rate: 100,
        };

        let cancel = CancellationToken::new();
        let handle = tokio::spawn(async move {
            chunker.run(in_rx, out_tx, cancel).await
        });

        // Send 5 seconds
        for _ in 0..5 {
            in_tx.send(AudioChunk { samples: vec![0.5; 100], sample_rate: 100, offset_secs: 0.0 }).await.unwrap();
        }
        drop(in_tx);

        handle.await.unwrap().unwrap();

        let w0 = out_rx.recv().await.unwrap();
        assert_eq!(w0.samples.len(), 200);
        assert!((w0.offset_secs - 0.0).abs() < f64::EPSILON);

        let w1 = out_rx.recv().await.unwrap();
        assert_eq!(w1.samples.len(), 200);
        assert!((w1.offset_secs - 2.0).abs() < f64::EPSILON);

        // Remaining 1s = 100 samples, which is NOT > 100 (min_final_samples = sample_rate = 100)
        // So no final chunk emitted
        assert!(out_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn test_overlap_chunker_final_buffer_threshold() {
        let (in_tx, in_rx) = mpsc::channel(10);
        let (out_tx, mut out_rx) = mpsc::channel(10);

        let mut chunker = OverlapChunker {
            chunk_duration_secs: 3.0,
            overlap_secs: 0.0,
            max_buffer_secs: 30.0,
            sample_rate: 100,
        };

        let cancel = CancellationToken::new();
        let handle = tokio::spawn(async move {
            chunker.run(in_rx, out_tx, cancel).await
        });

        // Send 3.5 seconds → one 3s window + 0.5s remainder (< 1s, should be dropped)
        in_tx.send(AudioChunk { samples: vec![0.1; 350], sample_rate: 100, offset_secs: 0.0 }).await.unwrap();
        drop(in_tx);

        handle.await.unwrap().unwrap();

        let w0 = out_rx.recv().await.unwrap();
        assert_eq!(w0.samples.len(), 300);

        // 50 samples < 100 (1s), no final chunk
        assert!(out_rx.recv().await.is_none());
    }

    #[tokio::test]
    async fn test_passthrough_chunker_cumulative_offset() {
        let (in_tx, in_rx) = mpsc::channel(10);
        let (out_tx, mut out_rx) = mpsc::channel(10);

        let mut chunker = PassthroughChunker;
        let cancel = CancellationToken::new();

        let handle = tokio::spawn(async move {
            chunker.run(in_rx, out_tx, cancel).await
        });

        // 3 chunks of 0.5s each at 100 samples/sec
        for _ in 0..3 {
            in_tx.send(AudioChunk { samples: vec![0.1; 50], sample_rate: 100, offset_secs: 0.0 }).await.unwrap();
        }
        drop(in_tx);

        handle.await.unwrap().unwrap();

        let c0 = out_rx.recv().await.unwrap();
        assert!((c0.offset_secs - 0.0).abs() < f64::EPSILON);

        let c1 = out_rx.recv().await.unwrap();
        assert!((c1.offset_secs - 0.5).abs() < f64::EPSILON);

        let c2 = out_rx.recv().await.unwrap();
        assert!((c2.offset_secs - 1.0).abs() < f64::EPSILON);

        assert!(out_rx.recv().await.is_none());
    }
}
