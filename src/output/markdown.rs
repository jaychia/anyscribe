//! Markdown output sink with YAML frontmatter.

use std::path::PathBuf;

use chrono::NaiveDateTime;
use tokio::sync::broadcast;

use crate::error::ScribeError;
use crate::types::{format_duration, format_timestamp, Metadata, Segment};

/// Collects all segments and writes a markdown file with YAML frontmatter.
///
/// The file is written atomically when the broadcast channel closes (i.e., when
/// the pipeline completes). The output path is derived from `recorded_at`
/// and an optional `title`.
///
/// # Output format
///
/// ```markdown
/// ---
/// date: 2026-03-14T14:30:00
/// duration: "5:00"
/// language: en
/// model: base
/// tags:
///   - meeting
///   - anyscribe
/// ---
///
/// # Meeting Notes — 2026-03-14 14:30
///
/// ## Transcript
///
/// **[00:00]** First segment text
///
/// **[00:30]** Second segment text
/// ```
pub struct MarkdownOutputSink {
    /// Directory where the markdown file will be written.
    pub notes_dir: PathBuf,
    /// Timestamp used for the filename and frontmatter.
    pub recorded_at: NaiveDateTime,
    /// Optional title for the filename (sanitized to filesystem-safe characters).
    pub title: Option<String>,
}

impl MarkdownOutputSink {
    pub fn generate_markdown(&self, segments: &[Segment], metadata: &Metadata) -> String {
        let date_str = self.recorded_at.format("%Y-%m-%dT%H:%M:%S");
        let display_date = self.recorded_at.format("%Y-%m-%d %H:%M");
        let duration = segments.last().map(|s| s.end).unwrap_or(0.0);
        let language = metadata.language.as_deref().unwrap_or("unknown");

        let mut lines = vec![
            "---".to_string(),
            format!("date: {date_str}"),
            format!("duration: \"{}\"", format_duration(duration)),
            format!("language: {language}"),
            format!("model: {}", metadata.model),
            "tags:".to_string(),
            "  - meeting".to_string(),
            "  - anyscribe".to_string(),
            "---".to_string(),
            String::new(),
            format!("# Meeting Notes — {display_date}"),
            String::new(),
            "## Transcript".to_string(),
            String::new(),
        ];

        for seg in segments {
            let text = seg.text.trim();
            if !text.is_empty() {
                lines.push(format!("**[{}]** {text}", format_timestamp(seg.start)));
                lines.push(String::new());
            }
        }

        lines.join("\n")
    }

    fn output_path(&self) -> PathBuf {
        let date_prefix = self.recorded_at.format("%Y-%m-%d_%H-%M");
        let filename = if let Some(title) = &self.title {
            let safe: String = title
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || c == ' ' || c == '-' || c == '_' {
                        c
                    } else {
                        ' '
                    }
                })
                .collect();
            let safe = safe.trim().replace(' ', "-");
            format!("{date_prefix}_{safe}.md")
        } else {
            format!("{date_prefix}_meeting-notes.md")
        };
        self.notes_dir.join(filename)
    }

    /// Consumes segments from a broadcast receiver and writes a markdown file on completion.
    pub async fn run(
        self,
        mut input: broadcast::Receiver<Segment>,
        metadata: Metadata,
    ) -> Result<(), ScribeError> {
        let mut segments = Vec::new();
        loop {
            match input.recv().await {
                Ok(seg) => segments.push(seg),
                Err(broadcast::error::RecvError::Closed) => break,
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    log::warn!("markdown subscriber lagged, skipped {n} segments");
                }
            }
        }

        let content = self.generate_markdown(&segments, &metadata);
        let path = self.output_path();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| ScribeError::Output {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }

        std::fs::write(&path, &content).map_err(|e| ScribeError::Output {
            path: path.clone(),
            source: e,
        })?;

        eprintln!("Saved: {}", path.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn test_sink(dir: PathBuf) -> MarkdownOutputSink {
        MarkdownOutputSink {
            notes_dir: dir,
            recorded_at: NaiveDate::from_ymd_opt(2026, 3, 14)
                .unwrap()
                .and_hms_opt(14, 30, 0)
                .unwrap(),
            title: None,
        }
    }

    #[test]
    fn test_generate_markdown_frontmatter() {
        let sink = test_sink(PathBuf::from("/tmp"));
        let metadata = Metadata {
            model: "base".to_string(),
            language: Some("en".to_string()),
        };
        let segments = vec![Segment {
            start: 0.0,
            end: 5.0,
            text: "Hello world".to_string(),
        }];

        let md = sink.generate_markdown(&segments, &metadata);
        assert!(md.contains("---"));
        assert!(md.contains("date: 2026-03-14T14:30:00"));
        assert!(md.contains("duration: \"0:05\""));
        assert!(md.contains("language: en"));
        assert!(md.contains("model: base"));
        assert!(md.contains("anyscribe"));
    }

    #[test]
    fn test_generate_markdown_segments() {
        let sink = test_sink(PathBuf::from("/tmp"));
        let metadata = Metadata::default();
        let segments = vec![
            Segment {
                start: 0.0,
                end: 5.0,
                text: "First segment".to_string(),
            },
            Segment {
                start: 5.0,
                end: 10.0,
                text: "Second segment".to_string(),
            },
        ];

        let md = sink.generate_markdown(&segments, &metadata);
        assert!(md.contains("**[00:00]** First segment"));
        assert!(md.contains("**[00:05]** Second segment"));
    }

    #[test]
    fn test_output_path_default() {
        let sink = test_sink(PathBuf::from("/tmp/notes"));
        let path = sink.output_path();
        assert_eq!(
            path,
            PathBuf::from("/tmp/notes/2026-03-14_14-30_meeting-notes.md")
        );
    }

    #[test]
    fn test_output_path_custom_title() {
        let mut sink = test_sink(PathBuf::from("/tmp/notes"));
        sink.title = Some("Team Standup".to_string());
        let path = sink.output_path();
        assert!(path.to_string_lossy().contains("Team-Standup"));
    }

    #[tokio::test]
    async fn test_sink_run_writes_file() {
        let dir = tempfile::tempdir().unwrap();
        let sink = test_sink(dir.path().to_path_buf());
        let metadata = Metadata {
            model: "base".to_string(),
            language: Some("en".to_string()),
        };

        let (tx, rx) = broadcast::channel(10);
        tx.send(Segment {
            start: 0.0,
            end: 5.0,
            text: "Hello".to_string(),
        })
        .unwrap();
        drop(tx);

        let output_path = sink.output_path();
        sink.run(rx, metadata).await.unwrap();

        assert!(output_path.exists());
        let content = std::fs::read_to_string(&output_path).unwrap();
        assert!(content.contains("Hello"));
    }
}
