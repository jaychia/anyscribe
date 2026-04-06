//! OpenAI-compatible transcription engine.
//!
//! Sends pre-chunked audio to any API that implements the OpenAI
//! `/v1/audio/transcriptions` endpoint (OpenAI, Groq, local whisper
//! servers, etc.) and parses the `verbose_json` response into
//! timestamped [`Segment`]s.

use async_trait::async_trait;
use tracing::{debug, info, warn};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use ureq::unversioned::multipart::{Form, Part};

use crate::audio::wav::encode_wav_bytes;
use crate::error::ScribeError;
use crate::pipeline::traits::TranscriptionEngine;
use crate::types::{AudioChunk, Metadata, Segment};

/// Transcription engine that calls an OpenAI-compatible audio API.
///
/// Receives pre-chunked audio from the [`Chunker`](crate::pipeline::traits::Chunker)
/// stage, encodes each chunk as WAV, POSTs it to the configured endpoint,
/// and parses the `verbose_json` response into timestamped segments.
///
/// Works with any service implementing the OpenAI `/v1/audio/transcriptions`
/// API: OpenAI itself, Groq, local faster-whisper servers, etc.
pub struct OpenAiTranscriptionEngine {
    api_key: String,
    base_url: String,
    model: String,
    sample_rate: u32,
    updated_metadata: Metadata,
    skipped_chunks: u32,
}

impl OpenAiTranscriptionEngine {
    /// Creates a new engine targeting the OpenAI API.
    ///
    /// Defaults to `https://api.openai.com` and model `whisper-1`.
    pub fn new(api_key: String, sample_rate: u32) -> Self {
        Self {
            api_key,
            base_url: "https://api.openai.com".to_string(),
            model: "whisper-1".to_string(),
            sample_rate,
            updated_metadata: Metadata::default(),
            skipped_chunks: 0,
        }
    }

    /// Overrides the API base URL (e.g., for a local whisper server).
    pub fn with_base_url(mut self, url: String) -> Self {
        self.base_url = url;
        self
    }

    /// Overrides the model name.
    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    async fn transcribe_and_send(
        &mut self,
        chunk: &AudioChunk,
        language: &Option<String>,
        detected_language: &mut Option<String>,
        output: &mpsc::Sender<Segment>,
    ) {
        let offset = chunk.offset_secs;
        debug!(
            "transcribing chunk ({:.1}s) via API, offset={offset:.1}s",
            chunk.samples.len() as f64 / self.sample_rate as f64,
        );

        match self.call_api(chunk, language).await {
            Ok((segments, lang)) => {
                if detected_language.is_none() && !lang.is_empty() {
                    *detected_language = Some(lang);
                }
                for mut seg in segments {
                    seg.start += offset;
                    seg.end += offset;
                    if output.send(seg).await.is_err() {
                        return;
                    }
                }
            }
            Err(e) => {
                self.skipped_chunks += 1;
                warn!(
                    offset = offset,
                    skipped_total = self.skipped_chunks,
                    error = %e,
                    "skipping transcription chunk"
                );
            }
        }
    }

    async fn call_api(
        &self,
        chunk: &AudioChunk,
        language: &Option<String>,
    ) -> Result<(Vec<Segment>, String), ScribeError> {
        let wav_bytes = encode_wav_bytes(&chunk.samples, chunk.sample_rate);
        let url = format!("{}/v1/audio/transcriptions", self.base_url);
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        let language = language.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut form = Form::new()
                .text("model", &model)
                .text("response_format", "verbose_json")
                .part(
                    "file",
                    Part::bytes(&wav_bytes)
                        .file_name("audio.wav")
                        .mime_str("audio/wav")
                        .map_err(|e| ScribeError::Transcription(format!("mime error: {e}")))?,
                );

            if let Some(ref lang) = language {
                form = form.text("language", lang);
            }

            let response = ureq::post(&url)
                .header("Authorization", &format!("Bearer {api_key}"))
                .send(form)
                .map_err(|e| ScribeError::Transcription(format!("API request failed: {e}")))?;

            let body: serde_json::Value = response
                .into_body()
                .read_json()
                .map_err(|e| ScribeError::Transcription(format!("failed to parse response: {e}")))?;

            parse_verbose_json(&body)
        })
        .await
        .map_err(|e| ScribeError::Transcription(format!("API task panicked: {e}")))?;

        result
    }
}

/// Parses the OpenAI `verbose_json` transcription response.
///
/// Expected format:
/// ```json
/// {
///   "language": "english",
///   "segments": [
///     { "start": 0.0, "end": 5.2, "text": "Hello world" }
///   ]
/// }
/// ```
///
/// Returns segments with timestamps relative to the chunk (not offset-adjusted)
/// and the detected language string.
fn parse_verbose_json(body: &serde_json::Value) -> Result<(Vec<Segment>, String), ScribeError> {
    let language = body["language"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let segments = match body["segments"].as_array() {
        Some(arr) => arr
            .iter()
            .filter_map(|s| {
                let start = s["start"].as_f64()?;
                let end = s["end"].as_f64()?;
                let text = s["text"].as_str()?.to_string();
                Some(Segment { start, end, text })
            })
            .collect(),
        None => {
            // Fallback: some APIs return just "text" without segments
            let text = body["text"]
                .as_str()
                .unwrap_or("")
                .to_string();
            if text.is_empty() {
                vec![]
            } else {
                let duration = body["duration"].as_f64().unwrap_or(0.0);
                vec![Segment {
                    start: 0.0,
                    end: duration,
                    text,
                }]
            }
        }
    };

    Ok((segments, language))
}

#[async_trait]
impl TranscriptionEngine for OpenAiTranscriptionEngine {
    #[tracing::instrument(name = "openai_engine", skip_all, fields(
        base_url = %self.base_url,
        model = %self.model,
    ))]
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<AudioChunk>,
        output: mpsc::Sender<Segment>,
        cancel: CancellationToken,
        metadata: Metadata,
    ) -> Result<(), ScribeError> {
        let mut detected_language = metadata.language.clone();

        info!(base_url = %self.base_url, model = %self.model, "OpenAI engine started");

        loop {
            let chunk = tokio::select! {
                biased;
                chunk = input.recv() => {
                    match chunk {
                        Some(data) => data,
                        None => break,
                    }
                }
                _ = cancel.cancelled() => {
                    while let Ok(data) = input.try_recv() {
                        self.transcribe_and_send(
                            &data, &metadata.language,
                            &mut detected_language, &output,
                        ).await;
                    }
                    break;
                }
            };

            self.transcribe_and_send(
                &chunk, &metadata.language,
                &mut detected_language, &output,
            ).await;
        }

        if self.skipped_chunks > 0 {
            warn!(skipped = self.skipped_chunks, "transcription chunks skipped due to errors");
        }

        self.updated_metadata = Metadata {
            model: self.model.clone(),
            language: detected_language,
        };

        Ok(())
    }

    fn updated_metadata(&self) -> Metadata {
        self.updated_metadata.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_verbose_json_with_segments() {
        let body: serde_json::Value = serde_json::json!({
            "task": "transcribe",
            "language": "english",
            "duration": 10.5,
            "segments": [
                { "start": 0.0, "end": 5.2, "text": "Hello world" },
                { "start": 5.2, "end": 10.5, "text": "How are you?" }
            ]
        });

        let (segments, lang) = parse_verbose_json(&body).unwrap();
        assert_eq!(lang, "english");
        assert_eq!(segments.len(), 2);
        assert!((segments[0].start - 0.0).abs() < f64::EPSILON);
        assert!((segments[0].end - 5.2).abs() < f64::EPSILON);
        assert_eq!(segments[0].text, "Hello world");
        assert_eq!(segments[1].text, "How are you?");
    }

    #[test]
    fn test_parse_verbose_json_text_only_fallback() {
        let body: serde_json::Value = serde_json::json!({
            "text": "Hello world",
            "duration": 3.0
        });

        let (segments, lang) = parse_verbose_json(&body).unwrap();
        assert_eq!(lang, "");
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].text, "Hello world");
        assert!((segments[0].end - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_parse_verbose_json_empty() {
        let body: serde_json::Value = serde_json::json!({});

        let (segments, lang) = parse_verbose_json(&body).unwrap();
        assert_eq!(lang, "");
        assert!(segments.is_empty());
    }
}
