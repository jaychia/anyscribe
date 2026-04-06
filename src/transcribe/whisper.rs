//! Whisper-based transcription engine.
//!
//! Accumulates audio into 30-second windows with 5-second overlap,
//! runs whisper inference on a blocking thread pool, and emits
//! timestamped [`Segment`]s.

use std::sync::Arc;

use async_trait::async_trait;
use log::debug;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::constants::{CHUNK_DURATION_SECS, MAX_BUFFER_SECS, OVERLAP_SECS};
use crate::error::ScribeError;
use crate::pipeline::traits::TranscriptionEngine;
use crate::preprocess::normalize;
use crate::transcribe::model::resolve_model_path;
use crate::types::{AudioChunk, Metadata, Segment};

/// Inner whisper-rs wrapper. Shared via `Arc` so `spawn_blocking` closures
/// can call `transcribe()` without moving the context.
struct WhisperEngine {
    ctx: WhisperContext,
}

impl WhisperEngine {
    fn new(model_size: &str) -> Result<Self, ScribeError> {
        let model_path = resolve_model_path(model_size)?;
        let path_str = model_path.to_string_lossy();

        eprintln!("Loading Whisper model ({model_size})...");
        let ctx = WhisperContext::new_with_params(&path_str, WhisperContextParameters::default())
            .map_err(|e| ScribeError::Transcription(format!("failed to load model: {e}")))?;

        Ok(Self { ctx })
    }

    fn transcribe(
        &self,
        audio: &[f32],
        language: Option<&str>,
        initial_prompt: Option<&str>,
    ) -> Result<TranscribeResult, ScribeError> {
        if audio.is_empty() {
            return Ok(TranscribeResult {
                segments: vec![],
                language: String::new(),
            });
        }

        let mut state = self.ctx.create_state().map_err(|e| {
            ScribeError::Transcription(format!("failed to create whisper state: {e}"))
        })?;

        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        });

        params.set_language(language);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_print_special(false);

        if let Some(prompt) = initial_prompt {
            params.set_initial_prompt(prompt);
        }

        state
            .full(params, audio)
            .map_err(|e| ScribeError::Transcription(format!("transcription failed: {e}")))?;

        let n_segments = state
            .full_n_segments()
            .map_err(|e| ScribeError::Transcription(format!("failed to get segment count: {e}")))?;

        let mut segments = Vec::new();
        for i in 0..n_segments {
            let t0 = state.full_get_segment_t0(i).unwrap_or(0);
            let t1 = state.full_get_segment_t1(i).unwrap_or(0);
            let text = state.full_get_segment_text_lossy(i).unwrap_or_default();

            segments.push(Segment {
                start: t0 as f64 / 100.0,
                end: t1 as f64 / 100.0,
                text,
            });
        }

        let lang_id = state.full_lang_id_from_state().unwrap_or(-1);
        let language_str = if lang_id >= 0 {
            whisper_rs::get_lang_str(lang_id)
                .unwrap_or("unknown")
                .to_string()
        } else {
            String::new()
        };

        Ok(TranscribeResult {
            segments,
            language: language_str,
        })
    }
}

struct TranscribeResult {
    segments: Vec<Segment>,
    language: String,
}

/// Local Whisper transcription engine using whisper-rs (whisper.cpp).
///
/// # Windowing strategy
///
/// Audio is accumulated into a buffer. When the buffer reaches 30 seconds,
/// a window is extracted and transcribed. After transcription, the last 5
/// seconds are kept as overlap for cross-boundary context. This continues
/// until the input channel closes or cancellation is requested. Any remaining
/// audio (> 1 second) is transcribed as a final chunk.
///
/// # Blocking work
///
/// Whisper inference is CPU-bound and runs on tokio's blocking thread pool
/// via `spawn_blocking`. The async `run()` loop stays responsive to new
/// audio chunks and cancellation while inference is in progress.
///
/// # Model management
///
/// Models are resolved via [`crate::transcribe::model::resolve_model_path`]:
/// environment variable override, local cache, or automatic download from
/// HuggingFace.
pub struct WhisperTranscriptionEngine {
    engine: Arc<WhisperEngine>,
    sample_rate: u32,
    updated_metadata: Metadata,
}

impl WhisperTranscriptionEngine {
    /// Creates a new engine, loading the specified whisper model.
    ///
    /// Downloads the model from HuggingFace if not already cached.
    /// Valid model sizes: `tiny`, `base`, `small`, `medium`, `large-v3`.
    pub fn new(model_size: &str, sample_rate: u32) -> Result<Self, ScribeError> {
        let engine = WhisperEngine::new(model_size)?;
        Ok(Self {
            engine: Arc::new(engine),
            sample_rate,
            updated_metadata: Metadata::default(),
        })
    }

    async fn transcribe_chunk(
        &self,
        audio: &[f32],
        language: &Option<String>,
        initial_prompt: &str,
        offset: f64,
    ) -> Result<(Vec<Segment>, String), ScribeError> {
        let engine = Arc::clone(&self.engine);
        let lang = language.clone();
        let prompt = if initial_prompt.is_empty() {
            None
        } else {
            Some(initial_prompt.to_string())
        };

        let audio = normalize(audio);

        let result = tokio::task::spawn_blocking(move || {
            engine.transcribe(&audio, lang.as_deref(), prompt.as_deref())
        })
        .await
        .map_err(|e| ScribeError::Transcription(format!("transcription task panicked: {e}")))??;

        let segments = result
            .segments
            .into_iter()
            .map(|seg| Segment {
                start: seg.start + offset,
                end: seg.end + offset,
                text: seg.text,
            })
            .collect();

        Ok((segments, result.language))
    }
}

#[async_trait]
impl TranscriptionEngine for WhisperTranscriptionEngine {
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<AudioChunk>,
        output: mpsc::Sender<Segment>,
        cancel: CancellationToken,
        metadata: Metadata,
    ) -> Result<(), ScribeError> {
        let chunk_samples = (CHUNK_DURATION_SECS * self.sample_rate as f64) as usize;
        let overlap_samples = (OVERLAP_SECS * self.sample_rate as f64) as usize;
        let max_buffer_samples = (MAX_BUFFER_SECS * self.sample_rate as f64) as usize;

        let mut buffer: Vec<f32> = Vec::new();
        let mut offset: f64 = 0.0;
        let mut last_text = String::new();
        let mut detected_language = metadata.language.clone();

        loop {
            tokio::select! {
                biased;
                chunk = input.recv() => {
                    match chunk {
                        Some(data) => {
                            buffer.extend_from_slice(&data.samples);

                            // Enforce max buffer cap
                            if buffer.len() > max_buffer_samples {
                                let excess = buffer.len() - max_buffer_samples;
                                buffer.drain(..excess);
                                eprintln!("WARNING: audio buffer exceeded {MAX_BUFFER_SECS}s cap, oldest audio dropped");
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
                let audio_chunk: Vec<f32> = buffer[..chunk_samples].to_vec();
                buffer = buffer[chunk_samples - overlap_samples..].to_vec();

                debug!(
                    "transcribing window ({:.1}s), offset={offset:.1}s",
                    audio_chunk.len() as f64 / self.sample_rate as f64,
                );

                let (segments, lang) = self
                    .transcribe_chunk(&audio_chunk, &metadata.language, &last_text, offset)
                    .await?;

                if detected_language.is_none() && !lang.is_empty() {
                    detected_language = Some(lang);
                }

                if let Some(last_seg) = segments.last() {
                    last_text = last_seg.text.clone();
                }
                for seg in segments {
                    if output.send(seg).await.is_err() {
                        return Ok(());
                    }
                }

                offset += CHUNK_DURATION_SECS - OVERLAP_SECS;
            }
        }

        // Process remaining buffer (if > 1 second of audio)
        let min_samples = self.sample_rate as usize;
        if buffer.len() > min_samples {
            debug!(
                "final buffer: {} samples ({:.1}s)",
                buffer.len(),
                buffer.len() as f64 / self.sample_rate as f64,
            );

            let (segments, lang) = self
                .transcribe_chunk(&buffer, &metadata.language, &last_text, offset)
                .await?;

            if detected_language.is_none() && !lang.is_empty() {
                detected_language = Some(lang);
            }

            for seg in segments {
                if output.send(seg).await.is_err() {
                    break;
                }
            }
        }

        self.updated_metadata = Metadata {
            model: metadata.model,
            language: detected_language,
        };

        Ok(())
    }

    fn updated_metadata(&self) -> Metadata {
        self.updated_metadata.clone()
    }
}
