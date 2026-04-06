//! Audio preprocessing: mono downmix, resampling, and peak normalization.
//!
//! The [`DefaultPreprocessor`] chains these three operations in its async
//! run loop. The individual functions ([`to_mono`], [`resample`], [`normalize`])
//! are also public for use in custom preprocessing pipelines.

use async_trait::async_trait;
use tracing::debug;
use tokio::sync::mpsc;

use crate::error::ScribeError;
use crate::pipeline::traits::{AudioInputInfo, Preprocessor};
use crate::types::AudioChunk;

const TARGET_PEAK: f32 = 0.95;

/// Downmixes multi-channel interleaved audio to mono by averaging channels.
pub fn to_mono(data: &[f32], channels: u16) -> Vec<f32> {
    if channels == 1 {
        return data.to_vec();
    }
    let ch = channels as usize;
    data.chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Linear interpolation resample from `from_rate` to `to_rate`.
pub fn resample(data: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || data.is_empty() {
        return data.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (data.len() as f64 / ratio) as usize;
    (0..out_len)
        .map(|i| {
            let src = i as f64 * ratio;
            let idx = src as usize;
            let frac = src - idx as f64;
            let a = data[idx];
            let b = data.get(idx + 1).copied().unwrap_or(a);
            a + (b - a) * frac as f32
        })
        .collect()
}

/// Peak-normalize audio to TARGET_PEAK. Skips silence.
pub fn normalize(audio: &[f32]) -> Vec<f32> {
    let peak = audio.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

    if peak < 1e-8 {
        debug!("normalize: peak={peak:.8} (silence), no gain applied");
        return audio.to_vec();
    }

    let gain = TARGET_PEAK / peak;
    debug!("normalize: peak={peak:.6}, gain={gain:.1}x");
    audio.iter().map(|s| s * gain).collect()
}

/// Default preprocessor: mono downmix -> linear resample -> peak normalize.
///
/// Converts raw device audio (arbitrary channels and sample rate) into
/// normalized mono [`AudioChunk`]s at `target_sample_rate`.
pub struct DefaultPreprocessor {
    /// Target sample rate for output chunks (typically 16000 for Whisper).
    pub target_sample_rate: u32,
}

#[async_trait]
impl Preprocessor for DefaultPreprocessor {
    #[tracing::instrument(name = "preprocessor", skip_all, fields(target_rate = self.target_sample_rate))]
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<Vec<f32>>,
        output: mpsc::Sender<AudioChunk>,
        info: AudioInputInfo,
    ) -> Result<(), ScribeError> {
        let mut chunk_count: u64 = 0;

        while let Some(raw) = input.recv().await {
            chunk_count += 1;

            let mono = to_mono(&raw, info.channels);
            let resampled = resample(&mono, info.sample_rate, self.target_sample_rate);
            let normalized = normalize(&resampled);

            if !normalized.is_empty() {
                let chunk = AudioChunk {
                    samples: normalized,
                    sample_rate: self.target_sample_rate,
                };
                if output.send(chunk).await.is_err() {
                    break;
                }
            }
        }

        debug!("preprocessor finished: {chunk_count} chunks processed");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_mono_single_channel() {
        let data = vec![0.5, -0.5, 0.3];
        assert_eq!(to_mono(&data, 1), data);
    }

    #[test]
    fn test_to_mono_stereo() {
        let data = vec![1.0, 0.0, 0.0, 1.0];
        let mono = to_mono(&data, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.5).abs() < f32::EPSILON);
        assert!((mono[1] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_resample_same_rate() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(resample(&data, 48000, 48000), data);
    }

    #[test]
    fn test_resample_downsample() {
        let data: Vec<f32> = (0..48000).map(|i| i as f32).collect();
        let out = resample(&data, 48000, 16000);
        assert_eq!(out.len(), 16000);
        assert!(out[0].abs() < 0.01);
    }

    #[test]
    fn test_resample_empty() {
        assert!(resample(&[], 48000, 16000).is_empty());
    }

    #[test]
    fn test_normalize_silence() {
        let data = vec![0.0, 0.0, 0.0];
        let out = normalize(&data);
        assert_eq!(out, data);
    }

    #[test]
    fn test_normalize_scales_to_target() {
        let data = vec![0.5, -0.5, 0.25];
        let out = normalize(&data);
        let peak = out.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!((peak - TARGET_PEAK).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_preserves_relative_amplitudes() {
        let data = vec![0.5, -0.25, 0.1];
        let out = normalize(&data);
        // Ratios should be preserved
        assert!((out[1] / out[0] - (-0.5)).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_preprocessor_run() {
        let (raw_tx, raw_rx) = mpsc::channel(10);
        let (proc_tx, mut proc_rx) = mpsc::channel(10);

        let mut pre = DefaultPreprocessor {
            target_sample_rate: 16000,
        };

        let info = AudioInputInfo {
            sample_rate: 48000,
            channels: 2,
        };

        let handle = tokio::spawn(async move { pre.run(raw_rx, proc_tx, info).await });

        // Send 480 stereo samples = 240 mono frames at 48kHz
        // After resample to 16kHz: 240 / 3 = 80 samples
        let stereo: Vec<f32> = (0..480).map(|i| (i % 2) as f32).collect();
        raw_tx.send(stereo).await.unwrap();
        drop(raw_tx);

        let chunk = proc_rx.recv().await.unwrap();
        assert_eq!(chunk.samples.len(), 80);
        assert_eq!(chunk.sample_rate, 16000);

        assert!(proc_rx.recv().await.is_none());
        handle.await.unwrap().unwrap();
    }
}
