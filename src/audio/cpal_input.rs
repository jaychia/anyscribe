use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use log::debug;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::error::ScribeError;
use crate::pipeline::traits::{AudioInput, AudioInputInfo};

struct SharedState {
    total_frames: AtomicU64,
    dropped_chunks: AtomicU32,
}

impl SharedState {
    fn new() -> Self {
        Self {
            total_frames: AtomicU64::new(0),
            dropped_chunks: AtomicU32::new(0),
        }
    }
}

pub struct CpalAudioInput {
    info: AudioInputInfo,
    state: Arc<SharedState>,
}

impl CpalAudioInput {
    pub fn new() -> Result<Self, ScribeError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| ScribeError::Audio("no input device available".to_string()))?;

        let default_config = device
            .default_input_config()
            .map_err(|e| ScribeError::Audio(format!("failed to get default input config: {e}")))?;

        let info = AudioInputInfo {
            sample_rate: default_config.sample_rate().0,
            channels: default_config.channels(),
        };

        Ok(Self {
            info,
            state: Arc::new(SharedState::new()),
        })
    }
}

#[async_trait]
impl AudioInput for CpalAudioInput {
    fn info(&self) -> &AudioInputInfo {
        &self.info
    }

    async fn run(
        &mut self,
        output: mpsc::Sender<Vec<f32>>,
        cancel: CancellationToken,
    ) -> Result<(), ScribeError> {
        let info = self.info.clone();
        let state = Arc::clone(&self.state);

        // Reset counters
        state.total_frames.store(0, Ordering::Relaxed);
        state.dropped_chunks.store(0, Ordering::Relaxed);

        // Signal channel to tell the audio thread to stop
        let (stop_tx, stop_rx) = std::sync::mpsc::channel::<()>();

        // Spawn a dedicated thread that owns the cpal Stream (not Send)
        let audio_thread = std::thread::spawn(move || -> Result<(), ScribeError> {
            let host = cpal::default_host();
            let device = host
                .default_input_device()
                .ok_or_else(|| ScribeError::Audio("no input device available".to_string()))?;

            let config = cpal::StreamConfig {
                channels: info.channels,
                sample_rate: cpal::SampleRate(info.sample_rate),
                buffer_size: cpal::BufferSize::Default,
            };

            let channels = info.channels;

            let stream = device
                .build_input_stream(
                    &config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        let n_frames = data.len() / channels as usize;
                        state
                            .total_frames
                            .fetch_add(n_frames as u64, Ordering::Relaxed);

                        if output.try_send(data.to_vec()).is_err() {
                            state.dropped_chunks.fetch_add(1, Ordering::Relaxed);
                        }
                    },
                    |err| {
                        eprintln!("Audio stream error: {err}");
                    },
                    None,
                )
                .map_err(|e| ScribeError::Audio(e.to_string()))?;

            stream
                .play()
                .map_err(|e| ScribeError::Audio(e.to_string()))?;

            // Block until signaled to stop. Stream stays alive via this scope.
            let _ = stop_rx.recv();
            Ok(())
        });

        // Wait for cancellation in async context
        cancel.cancelled().await;

        // Signal the audio thread to stop
        let _ = stop_tx.send(());

        // Wait for thread to finish
        audio_thread
            .join()
            .map_err(|_| ScribeError::Audio("audio thread panicked".to_string()))??;

        let dropped = self.state.dropped_chunks.load(Ordering::Relaxed);
        let elapsed =
            self.state.total_frames.load(Ordering::Relaxed) as f64 / self.info.sample_rate as f64;

        debug!("audio input stopped: elapsed={elapsed:.1}s dropped_chunks={dropped}");

        if dropped > 0 {
            eprintln!(
                "WARNING: {dropped} audio chunk(s) dropped — transcription couldn't keep up."
            );
        }

        Ok(())
    }
}
