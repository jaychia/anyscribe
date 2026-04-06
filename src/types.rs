//! Domain types shared across pipeline stages.

/// A timestamped span of transcribed text.
///
/// Timestamps are in seconds from the start of the recording.
#[derive(Debug, Clone)]
pub struct Segment {
    /// Start time in seconds.
    pub start: f64,
    /// End time in seconds.
    pub end: f64,
    /// Transcribed text for this time span.
    pub text: String,
}

/// A preprocessed chunk of audio ready for transcription.
///
/// Contains mono, resampled, normalized `f32` PCM samples.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Mono PCM samples, normalized to peak 0.95.
    pub samples: Vec<f32>,
    /// Sample rate of the contained audio (typically 16 kHz for Whisper).
    pub sample_rate: u32,
}

/// Metadata carried through the pipeline.
///
/// Passed into the pipeline at construction, and updated by the
/// [`TranscriptionEngine`](crate::pipeline::traits::TranscriptionEngine)
/// during inference (e.g., detected language).
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    /// Whisper model name (e.g., "base", "small").
    pub model: String,
    /// Language code (e.g., "en"), or `None` for auto-detect.
    pub language: Option<String>,
}

/// Complete transcript with metadata.
///
/// Useful for programmatic access to transcription results.
#[derive(Debug, Clone)]
pub struct TranscriptResult {
    pub segments: Vec<Segment>,
    pub language: String,
    pub duration: f64,
}

impl TranscriptResult {
    /// Joins all segment text into a single string.
    pub fn full_text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Formats seconds as `MM:SS` or `HH:MM:SS` for display.
pub fn format_timestamp(seconds: f64) -> String {
    let total = seconds as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h:02}:{m:02}:{s:02}")
    } else {
        format!("{m:02}:{s:02}")
    }
}

pub fn format_duration(seconds: f64) -> String {
    let total = seconds as u64;
    let h = total / 3600;
    let m = (total % 3600) / 60;
    let s = total % 60;
    if h > 0 {
        format!("{h}:{m:02}:{s:02}")
    } else {
        format!("{m}:{s:02}")
    }
}
