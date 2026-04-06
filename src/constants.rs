//! Compile-time constants for pipeline tuning.

use std::ops::RangeInclusive;

/// Whisper model sizes available for download from HuggingFace.
pub const VALID_MODELS: &[&str] = &["tiny", "base", "small", "medium", "large-v3"];

/// Bounded channel capacity for raw audio and preprocessed audio channels.
pub const AUDIO_CHANNEL_CAPACITY: usize = 100;
/// Bounded channel capacity for segment channels (engine -> postprocessor -> sink).
pub const SEGMENT_CHANNEL_CAPACITY: usize = 256;

/// Valid range for configured sample rates.
pub const SAMPLE_RATE_RANGE: RangeInclusive<u32> = 8000..=48000;
/// Default target sample rate for Whisper (16 kHz).
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;

/// Duration of each transcription window in seconds.
pub const CHUNK_DURATION_SECS: f64 = 30.0;
/// Overlap between consecutive transcription windows in seconds.
/// Provides context continuity across chunk boundaries.
pub const OVERLAP_SECS: f64 = 5.0;
/// Maximum audio buffer before oldest samples are dropped.
/// Prevents unbounded memory growth when transcription falls behind.
pub const MAX_BUFFER_SECS: f64 = 300.0;
