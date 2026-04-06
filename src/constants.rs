use std::ops::RangeInclusive;

pub const VALID_MODELS: &[&str] = &["tiny", "base", "small", "medium", "large-v3"];

pub const AUDIO_CHANNEL_CAPACITY: usize = 100;
pub const SEGMENT_CHANNEL_CAPACITY: usize = 256;

pub const SAMPLE_RATE_RANGE: RangeInclusive<u32> = 8000..=48000;
pub const DEFAULT_SAMPLE_RATE: u32 = 16000;

pub const CHUNK_DURATION_SECS: f64 = 30.0;
pub const OVERLAP_SECS: f64 = 5.0;
pub const MAX_BUFFER_SECS: f64 = 300.0;
