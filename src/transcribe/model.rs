//! Whisper model resolution, caching, and download.
//!
//! Models are resolved in order:
//!
//! 1. **`ANYSCRIBE_MODEL_PATH` env var** — use an explicit file path.
//! 2. **Local cache** (`~/.cache/anyscribe/models/`) — reuse a previously downloaded model.
//! 3. **Download** from HuggingFace — fetches the ggml binary with a progress bar.
//!
//! Downloads use a temp-file guard: on failure, the `.tmp` file is cleaned up
//! automatically. On success, it is atomically renamed to the final path.

use std::io::{Read, Write};
use std::path::PathBuf;

use indicatif::{ProgressBar, ProgressStyle};

use crate::constants::VALID_MODELS;
use crate::error::ScribeError;

const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

fn model_filename(model_size: &str) -> String {
    format!("ggml-{model_size}.bin")
}

fn cache_dir() -> PathBuf {
    dirs::cache_dir()
        .or_else(dirs::home_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("anyscribe")
        .join("models")
}

/// Resolve the path to a whisper model file.
///
/// Checks (in order):
/// 1. `ANYSCRIBE_MODEL_PATH` env var
/// 2. Cache directory (`~/.cache/anyscribe/models/`)
/// 3. Downloads from HuggingFace if not found
pub fn resolve_model_path(model_size: &str) -> Result<PathBuf, ScribeError> {
    if !VALID_MODELS.contains(&model_size) {
        return Err(ScribeError::Transcription(format!(
            "invalid model: {model_size}"
        )));
    }

    // 1. Environment variable override
    if let Ok(path) = std::env::var("ANYSCRIBE_MODEL_PATH") {
        let p = PathBuf::from(&path);
        if p.is_file() {
            return Ok(p);
        }
        return Err(ScribeError::Transcription(format!(
            "ANYSCRIBE_MODEL_PATH does not exist: {path}"
        )));
    }

    // 2. Check cache
    let dir = cache_dir();
    let filename = model_filename(model_size);
    let cached = dir.join(&filename);
    if cached.is_file() {
        return Ok(cached);
    }

    // 3. Download
    download_model(model_size, &dir, &filename)
}

/// Drop guard that cleans up temp file on failure.
struct TempFileGuard {
    path: PathBuf,
    keep: bool,
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if !self.keep {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn download_model(model_size: &str, dir: &PathBuf, filename: &str) -> Result<PathBuf, ScribeError> {
    let url = format!("{MODEL_BASE_URL}/{filename}");
    let dest = dir.join(filename);

    eprintln!("Downloading whisper model '{model_size}'...");
    eprintln!("  From: {url}");
    eprintln!("  To:   {}", dest.display());

    std::fs::create_dir_all(dir).map_err(|e| {
        ScribeError::Transcription(format!(
            "failed to create model cache directory {}: {e}",
            dir.display()
        ))
    })?;

    let response = ureq::get(&url)
        .call()
        .map_err(|e| ScribeError::Transcription(format!("failed to download model: {e}")))?;

    let content_length = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(0);

    let pb = if content_length > 0 {
        let pb = ProgressBar::new(content_length);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("  [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb
    } else {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("  {spinner} {bytes}")
                .unwrap(),
        );
        pb
    };

    // Write to temp file, rename on success. Guard cleans up on failure.
    let tmp_dest = dir.join(format!("{filename}.tmp"));
    let mut guard = TempFileGuard {
        path: tmp_dest.clone(),
        keep: false,
    };

    let mut file = std::fs::File::create(&tmp_dest)
        .map_err(|e| ScribeError::Transcription(format!("failed to create temp file: {e}")))?;

    let mut reader = response.into_body().into_reader();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| ScribeError::Transcription(format!("download read error: {e}")))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])
            .map_err(|e| ScribeError::Transcription(format!("failed to write model file: {e}")))?;
        pb.inc(n as u64);
    }

    pb.finish_and_clear();

    std::fs::rename(&tmp_dest, &dest)
        .map_err(|e| ScribeError::Transcription(format!("failed to finalize model file: {e}")))?;

    guard.keep = true;

    eprintln!("  Download complete.\n");
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_filename() {
        assert_eq!(model_filename("base"), "ggml-base.bin");
        assert_eq!(model_filename("large-v3"), "ggml-large-v3.bin");
    }

    #[test]
    fn test_cache_dir_is_reasonable() {
        let dir = cache_dir();
        let s = dir.to_string_lossy();
        assert!(s.contains("anyscribe") && s.contains("models"));
    }

    #[test]
    fn test_env_override_nonexistent() {
        std::env::set_var("ANYSCRIBE_MODEL_PATH", "/tmp/nonexistent_model_file.bin");
        let result = resolve_model_path("base");
        std::env::remove_var("ANYSCRIBE_MODEL_PATH");
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_model_name() {
        let result = resolve_model_path("nonexistent");
        assert!(result.is_err());
    }
}
