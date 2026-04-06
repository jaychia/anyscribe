use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::constants::{DEFAULT_SAMPLE_RATE, SAMPLE_RATE_RANGE, VALID_MODELS};
use crate::error::ScribeError;

fn default_model() -> String {
    "base".to_string()
}

fn default_sample_rate() -> u32 {
    DEFAULT_SAMPLE_RATE
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub notes_path: String,
    #[serde(default = "default_model")]
    pub whisper_model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            notes_path: String::new(),
            whisper_model: default_model(),
            language: None,
            sample_rate: default_sample_rate(),
        }
    }
}

impl Config {
    pub fn notes_dir(&self) -> PathBuf {
        PathBuf::from(&self.notes_path)
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.notes_path.is_empty() {
            errors.push("notes_path is not set".to_string());
        } else if !Path::new(&self.notes_path).is_dir() {
            errors.push(format!("notes_path does not exist: {}", self.notes_path));
        }
        if !VALID_MODELS.contains(&self.whisper_model.as_str()) {
            errors.push(format!("invalid whisper_model: {}", self.whisper_model));
        }
        if !SAMPLE_RATE_RANGE.contains(&self.sample_rate) {
            errors.push(format!(
                "sample_rate {} out of valid range ({}-{})",
                self.sample_rate,
                SAMPLE_RATE_RANGE.start(),
                SAMPLE_RATE_RANGE.end()
            ));
        }
        errors
    }
}

pub fn config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config")
        .join("scribe-rs")
}

pub fn config_path() -> PathBuf {
    config_dir().join("config.toml")
}

pub fn default_notes_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".scribe-rs")
}

pub fn load_config() -> Result<Config, ScribeError> {
    let path = config_path();
    if !path.exists() {
        return Ok(Config::default());
    }
    let contents = std::fs::read_to_string(&path)?;
    let config: Config = toml::from_str(&contents)
        .map_err(|e| ScribeError::Config(format!("invalid config TOML: {e}")))?;
    Ok(config)
}

pub fn save_config(config: &Config) -> Result<(), ScribeError> {
    let dir = config_dir();
    std::fs::create_dir_all(&dir)?;
    let contents = toml::to_string_pretty(config)
        .map_err(|e| ScribeError::Config(format!("failed to serialize config: {e}")))?;
    std::fs::write(config_path(), contents)?;
    Ok(())
}

pub fn first_run_setup() -> Result<Config, ScribeError> {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();

    println!("\n  scribe-rs — First Run Setup\n");

    let default_path = default_notes_path();
    let default_display = default_path.display();

    let notes_path = {
        print!("  Notes directory [{default_display}]: ");
        stdout.flush()?;
        let mut input = String::new();
        stdin.read_line(&mut input)?;
        let input = input.trim();

        let path = if input.is_empty() {
            default_path.clone()
        } else {
            PathBuf::from(input)
        };

        if !path.is_dir() {
            std::fs::create_dir_all(&path)?;
            println!("  Created {}", path.display());
        }
        path.to_string_lossy().to_string()
    };

    print!("  Whisper model (tiny/base/small/medium/large-v3) [base]: ");
    stdout.flush()?;
    let mut model_input = String::new();
    stdin.read_line(&mut model_input)?;
    let model = model_input.trim();
    let whisper_model = if model.is_empty() { "base" } else { model }.to_string();

    let config = Config {
        notes_path,
        whisper_model,
        ..Config::default()
    };
    save_config(&config)?;
    println!("\n  Config saved to {}\n", config_path().display());
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = Config::default();
        assert_eq!(cfg.notes_path, "");
        assert_eq!(cfg.whisper_model, "base");
        assert!(cfg.language.is_none());
        assert_eq!(cfg.sample_rate, 16000);
    }

    #[test]
    fn test_config_notes_dir() {
        let cfg = Config {
            notes_path: "/tmp/notes".to_string(),
            ..Config::default()
        };
        assert_eq!(cfg.notes_dir(), PathBuf::from("/tmp/notes"));
    }

    #[test]
    fn test_validate_empty_notes_path() {
        let cfg = Config::default();
        let errors = cfg.validate();
        assert!(errors.iter().any(|e| e.contains("notes_path")));
    }

    #[test]
    fn test_validate_invalid_model() {
        let cfg = Config {
            notes_path: "/tmp".to_string(),
            whisper_model: "nonexistent".to_string(),
            ..Config::default()
        };
        let errors = cfg.validate();
        assert!(errors.iter().any(|e| e.contains("whisper_model")));
    }

    #[test]
    fn test_validate_invalid_sample_rate() {
        let cfg = Config {
            notes_path: "/tmp".to_string(),
            sample_rate: 0,
            ..Config::default()
        };
        let errors = cfg.validate();
        assert!(errors.iter().any(|e| e.contains("sample_rate")));
    }

    #[test]
    fn test_validate_valid() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = Config {
            notes_path: dir.path().to_string_lossy().to_string(),
            whisper_model: "base".to_string(),
            ..Config::default()
        };
        assert!(cfg.validate().is_empty());
    }

    #[test]
    fn test_roundtrip_serialization() {
        let cfg = Config {
            notes_path: "/tmp/notes".to_string(),
            whisper_model: "small".to_string(),
            language: Some("en".to_string()),
            sample_rate: 16000,
        };

        let contents = toml::to_string_pretty(&cfg).unwrap();
        let loaded: Config = toml::from_str(&contents).unwrap();
        assert_eq!(loaded.notes_path, "/tmp/notes");
        assert_eq!(loaded.whisper_model, "small");
        assert_eq!(loaded.language, Some("en".to_string()));
        assert_eq!(loaded.sample_rate, 16000);
    }

    #[test]
    fn test_language_none_not_serialized() {
        let cfg = Config::default();
        let s = toml::to_string_pretty(&cfg).unwrap();
        assert!(!s.contains("language"));
    }
}
