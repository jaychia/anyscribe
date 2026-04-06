use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ScribeError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Audio error: {0}")]
    Audio(String),

    #[error("Transcription error: {0}")]
    Transcription(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Output error: {path}: {source}")]
    Output {
        path: PathBuf,
        source: std::io::Error,
    },

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
