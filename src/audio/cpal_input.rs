//! Default audio input using the system microphone via cpal.
//!
//! Audio is captured from the default system input device, written to a WAV
//! file on disk, and notifications are sent downstream so the preprocessor
//! can read the data back.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::audio::wav::WavWriter;
use crate::constants::AUDIO_CHANNEL_CAPACITY;
use crate::error::ScribeError;
use crate::pipeline::traits::{AudioInput, AudioInputInfo};
use crate::types::AudioNotification;

/// Atomic counters shared between the cpal callback thread and the async runtime.
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

/// Captures audio from the default system input device using cpal.
///
/// # Disk buffer
///
/// Audio is written to a WAV file on disk as it arrives. The pipeline channel
/// carries lightweight [`AudioNotification`]s instead of raw sample data.
/// This decouples audio capture from transcription speed — the disk acts as
/// an unbounded buffer, and the WAV file is kept for offline playback.
///
/// # Threading model
///
/// cpal's `Stream` is `!Send`, so it cannot live in a tokio task. Instead,
/// [`run()`](AudioInput::run) spawns a dedicated OS thread that owns the
/// stream. The cpal hardware callback pushes raw PCM via `try_send` into
/// an internal bounded channel. An async loop in `run()` drains this channel,
/// writes to the WAV file, and sends notifications downstream.
///
/// # Construction
///
/// [`CpalAudioInput::new()`] enumerates the default input device, reads
/// its native format, and creates the WAV file (so it exists before any
/// pipeline task tries to read from it).
pub struct CpalAudioInput {
    info: AudioInputInfo,
    state: Arc<SharedState>,
    wav_writer: WavWriter,
}

impl CpalAudioInput {
    /// Creates a new audio input by probing the default system input device.
    ///
    /// Creates the WAV file at `wav_path` with a valid header. The file will
    /// be finalized with correct sizes when recording ends.
    ///
    /// Fails if no input device is available, the device format cannot be read,
    /// or the WAV file cannot be created.
    pub fn new(wav_path: PathBuf) -> Result<Self, ScribeError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| ScribeError::Audio("no input device available".to_string()))?;

        let default_config = device
            .default_input_config()
            .map_err(|e| ScribeError::Audio(format!("failed to get default input config: {e}")))?;

        let sample_rate = default_config.sample_rate().0;
        let channels = default_config.channels();

        let wav_writer = WavWriter::new(&wav_path, sample_rate, channels)?;

        let info = AudioInputInfo {
            sample_rate,
            channels,
            wav_path,
        };

        Ok(Self {
            info,
            state: Arc::new(SharedState::new()),
            wav_writer,
        })
    }
}

#[async_trait]
impl AudioInput for CpalAudioInput {
    fn info(&self) -> &AudioInputInfo {
        &self.info
    }

    #[tracing::instrument(name = "audio_input", skip_all, fields(sample_rate = self.info.sample_rate, channels = self.info.channels))]
    async fn run(
        &mut self,
        output: mpsc::Sender<AudioNotification>,
        cancel: CancellationToken,
    ) -> Result<(), ScribeError> {
        let info = self.info.clone();
        let state = Arc::clone(&self.state);

        // Reset counters
        state.total_frames.store(0, Ordering::Relaxed);
        state.dropped_chunks.store(0, Ordering::Relaxed);

        // Internal channel: cpal callback -> disk writer loop
        let (internal_tx, mut internal_rx) = mpsc::channel::<Vec<f32>>(AUDIO_CHANNEL_CAPACITY);

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

                        if internal_tx.try_send(data.to_vec()).is_err() {
                            state.dropped_chunks.fetch_add(1, Ordering::Relaxed);
                        }
                    },
                    |err| {
                        error!(%err, "audio stream error");
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

        // Disk writer loop: drain internal channel, write to WAV, send notifications
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => {
                    // Drain remaining audio from internal channel
                    while let Ok(chunk) = internal_rx.try_recv() {
                        let n = chunk.len();
                        self.wav_writer.write_samples(&chunk)?;
                        let _ = output.try_send(AudioNotification { num_samples: n });
                    }
                    break;
                }
                chunk = internal_rx.recv() => {
                    match chunk {
                        Some(data) => {
                            let n = data.len();
                            self.wav_writer.write_samples(&data)?;
                            if output.send(AudioNotification { num_samples: n }).await.is_err() {
                                break;
                            }
                        }
                        None => break,
                    }
                }
            }
        }

        // Signal the audio thread to stop
        let _ = stop_tx.send(());

        // Wait for thread to finish
        audio_thread
            .join()
            .map_err(|_| ScribeError::Audio("audio thread panicked".to_string()))??;

        // Finalize WAV header with correct data sizes
        self.wav_writer.finalize()?;

        let dropped = self.state.dropped_chunks.load(Ordering::Relaxed);
        let elapsed =
            self.state.total_frames.load(Ordering::Relaxed) as f64 / self.info.sample_rate as f64;

        debug!(elapsed_secs = format!("{elapsed:.1}"), dropped, "audio input stopped");
        info!(path = %self.info.wav_path.display(), "WAV recording saved");

        if dropped > 0 {
            warn!(dropped, "audio chunks dropped — transcription couldn't keep up");
        }

        Ok(())
    }
}
