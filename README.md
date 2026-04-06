# anyscribe

anyscribe is a simple modular, portable transcription pipeline that runs audio-to-markdown processing pipelines. It allows for easily composing these pipelines to run on various hardware configurations (e.g. running on-device with Whisper, or using a remote transcription service. Hooking up to an audio source that is the current machine's audio inputs, or a file WAV source.) and produces an opinionated Markdown with YAML frontmatter format afterwards, making the data suitable for AI consumption.

## Architecture

The core abstraction is a five-stage async pipeline connected by bounded `tokio::sync::mpsc` channels:

```
AudioInput --> Preprocessor --> TranscriptionEngine --> Postprocessor --> OutputSink
    [ch A]         [ch B]             [ch C]               [ch D]
```

Every stage implements the same interface:

```rust
#[async_trait]
pub trait StageName: Send {
    async fn run(&mut self, input: Receiver<T>, output: Sender<U>) -> Result<(), ScribeError>;
}
```

Each stage is spawned as its own `tokio::spawn` task. The `PipelineRunner` creates channels, spawns all five stages, and joins them with `tokio::join!`. There is no central polling loop — data flows through channels, and backpressure propagates naturally via bounded sends.

### Data flow

| Channel | Type | Capacity | Backpressure |
|---------|------|----------|--------------|
| A | `Vec<f32>` (raw PCM) | 100 | `try_send` drops + warns |
| B | `AudioChunk` (mono, resampled, normalized) | 100 | `.await` suspends preprocessor |
| C | `Segment` (timestamped text) | 256 | `.await` suspends engine |
| D | `Segment` (post-processed) | 256 | `.await` suspends postprocessor |

The audio source is the only stage that uses `try_send` — hardware callbacks cannot block, so chunks are dropped and counted when the pipeline falls behind. All other stages use `send().await`, which creates cascading backpressure from sink back to source.

### Traits

Five traits define the pipeline contract. All are `#[async_trait]` with `async fn run() -> Result<(), ScribeError>`:

- **[`AudioInput`](src/pipeline/traits.rs)** — Push-based audio source. Provides `info()` (sample rate, channels) synchronously before `run()` starts. `CpalAudioInput` captures from the default microphone via a dedicated thread (cpal's `Stream` is `!Send`).

- **[`Preprocessor`](src/pipeline/traits.rs)** — Transforms raw device audio into normalized mono chunks at the target sample rate. `DefaultPreprocessor` applies `to_mono` -> `resample` (linear interpolation) -> `normalize` (peak normalization to 0.95).

- **[`TranscriptionEngine`](src/pipeline/traits.rs)** — Converts audio chunks into timestamped text segments. Provides `updated_metadata()` after `run()` completes (detected language, etc.). `WhisperTranscriptionEngine` accumulates 30-second windows with 5-second overlap and runs inference via `spawn_blocking`.

- **[`Postprocessor`](src/pipeline/traits.rs)** — Transforms or filters segments between the engine and the sink. `NoopPostprocessor` passes through unchanged. Extend this to filter noise tokens, merge segments, apply punctuation, etc.

- **[`OutputSink`](src/pipeline/traits.rs)** — Consumes segments. Owns its async loop. `StdoutOutputSink` prints `[MM:SS] text` in real time. `MarkdownOutputSink` collects all segments and writes a markdown file with YAML frontmatter on completion. `MultiOutputSink` fans out to multiple inner sinks concurrently.

### Pipeline runner

`PipelineRunner` wires the five stages together:

```rust
let runner = PipelineRunner {
    input: Box::new(my_audio_input),
    preprocessor: Box::new(DefaultPreprocessor { target_sample_rate: 16000 }),
    engine: Box::new(WhisperTranscriptionEngine::new("base", 16000)?),
    postprocessor: Box::new(NoopPostprocessor),
    sink: Box::new(StdoutOutputSink),
    cancel: CancellationToken::new(),
    metadata: Metadata { model: "base".into(), language: None },
};

runner.run().await?;
```

Cancellation is cooperative — pass a `CancellationToken` and cancel it to gracefully shut down. The audio source stops capturing, the engine drains its buffer, and channels close in sequence.

## Default implementations

| Trait | Implementation | Notes |
|-------|---------------|-------|
| `AudioInput` | `CpalAudioInput` | Default microphone via cpal. Device enumeration at construction time, stream on dedicated OS thread. |
| `Preprocessor` | `DefaultPreprocessor` | `to_mono()` + `resample()` (linear interpolation) + `normalize()` (peak to 0.95). |
| `TranscriptionEngine` | `WhisperTranscriptionEngine` | whisper-rs with beam search (size 5). 30s windows, 5s overlap. `spawn_blocking` for CPU inference. Auto-downloads models from HuggingFace. |
| `Postprocessor` | `NoopPostprocessor` | Passthrough. |
| `OutputSink` | `StdoutOutputSink` | Prints `[MM:SS] text` to stdout. |
| `OutputSink` | `MarkdownOutputSink` | Writes markdown with YAML frontmatter on completion. |
| `OutputSink` | `MultiOutputSink` | Fans out to multiple sinks via spawned tasks + channels. |

### Output format

The `MarkdownOutputSink` produces files designed for AI consumption:

```markdown
---
date: 2026-03-14T14:30:00
duration: "5:00"
language: en
model: base
tags:
  - meeting
  - anyscribe
---

# Meeting Notes — 2026-03-14 14:30

## Transcript

**[00:00]** First segment text

**[00:30]** Second segment text
```

## Usage

```bash
# Record and transcribe (first run downloads the whisper model)
anyscribe record

# List saved notes
anyscribe list

# View configuration
anyscribe config show

# Change model (tiny/base/small/medium/large-v3)
anyscribe config set whisper_model small

# Set language (or "auto" for auto-detect)
anyscribe config set language en
```

Configuration is stored in `~/.config/anyscribe/config.toml`. Models are cached in `~/.cache/anyscribe/models/`. Set `ANYSCRIBE_MODEL_PATH` to override model resolution.

## Building

```bash
cargo build --release
```

Requires a C/C++ toolchain for whisper-rs (whisper.cpp is compiled from source). On Linux, you also need `libasound2-dev` for cpal.

## Extending

Every stage is a trait. To add a custom component, implement the trait and pass it to `PipelineRunner`:

```rust
// Custom sink that sends segments to a webhook
struct WebhookSink { url: String }

#[async_trait]
impl OutputSink for WebhookSink {
    async fn run(
        &mut self,
        mut input: mpsc::Receiver<Segment>,
        metadata: Metadata,
    ) -> Result<(), ScribeError> {
        while let Some(seg) = input.recv().await {
            // POST to webhook...
        }
        Ok(())
    }
}
```

The same pattern works for custom audio sources (network streams, file playback), alternative engines (cloud APIs, faster-whisper), or processing stages (speaker diarization, translation).

## Project structure

```
src/
  lib.rs                      Crate root — module declarations
  types.rs                    Segment, AudioChunk, Metadata, TranscriptResult
  error.rs                    ScribeError (thiserror)
  constants.rs                Channel capacities, model list, buffer limits
  config.rs                   TOML config load/save/validate
  pipeline/
    traits.rs                 5 trait definitions
    mod.rs                    PipelineRunner
  audio/
    cpal_input.rs             CpalAudioInput
  preprocess/
    mod.rs                    DefaultPreprocessor, to_mono, resample, normalize
  transcribe/
    whisper.rs                WhisperTranscriptionEngine
    model.rs                  Model download/cache with temp-file guard
  postprocess/
    mod.rs                    NoopPostprocessor
  output/
    stdout.rs                 StdoutOutputSink
    markdown.rs               MarkdownOutputSink
    multi.rs                  MultiOutputSink
  main.rs                     CLI (clap) — record, list, config
tests/
  pipeline_test.rs            Integration tests with mock implementations
```

## License

Apache-2.0
