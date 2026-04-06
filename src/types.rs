#[derive(Debug, Clone)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Metadata {
    pub model: String,
    pub language: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TranscriptResult {
    pub segments: Vec<Segment>,
    pub language: String,
    pub duration: f64,
}

impl TranscriptResult {
    pub fn full_text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

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
