# scribe-rs

Simple, modular, single-binary transcription from audio to text. Written in Rust.

This repository is broken down into modules which can be written for different use-cases for a simple streaming transcription pipeline.

```
AudioInput -> Preprocessor -> TranscriptionEngine -> Postprocessor -> OutputSink
``` 

We also provide default implementations:

1. `WhisperTranscriptionEngine` using `whisper-rs`
2. `CpalAudioInput` using `cpal` to capture audio from the current device
3. `MultiOutputSink` allows for multiplexing to multiple output sinks
4. `StdoutOutputSink` writes outputs to stdout
5. `MarkdownOutputSink` writes outputs to a markdown file with YAML frontmatter and some reasonable defaults

