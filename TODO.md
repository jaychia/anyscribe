# TODO: Server-Ready Architecture

Preparing anyscribe to run as a long-lived server engine behind an AI agent, targeting Raspberry Pi and Mac Mini.

## Priority 1: Streaming output

- [ ] Add a `ChannelOutputSink` (or `broadcast::Sender<Segment>`) that server request handlers can subscribe to
- [ ] Restructure so the pipeline can expose a segment stream rather than only running to completion
- [ ] `PipelineRunner::run()` consumes `self` — make it possible to query state and stream segments while running

## Priority 2: Error resilience

- [ ] Skip/log bad transcription chunks instead of killing the entire pipeline
- [ ] Add retry logic for transient failures (critical for remote transcription engines)
- [ ] Audio capture should never stop because of a downstream failure

## Priority 3: Incremental persistence

- [ ] `MarkdownOutputSink` buffers all segments in memory until shutdown — write incrementally instead
- [ ] Add file rotation (e.g., hourly) for all-day transcription
- [ ] A crash at 5pm currently loses everything since morning

## Priority 4: Extract windowing from transcription engine

- [ ] The 30s window + 5s overlap logic is baked into `WhisperTranscriptionEngine::run()`
- [ ] Remote APIs (Deepgram, AssemblyAI, OpenAI) need different chunking strategies
- [ ] Make windowing a reusable component so engine impls only handle "audio in, segments out"

## Priority 5: Memory stability for all-day operation

- [ ] Replace `Vec<f32>` accumulator in whisper.rs with `VecDeque` or ring buffer (avoids copy every 30s)
- [ ] Reuse buffers in preprocessing — `normalize` and `to_mono` can mutate in place
- [ ] `data.to_vec()` in cpal callback allocates on the real-time audio thread; consider a pre-allocated pool
- [ ] Remove double normalization (preprocessor + `transcribe_chunk` both normalize)

## Priority 6: Fix metadata propagation bug

- [ ] `PipelineRunner` returns the engine from its join handle but never calls `updated_metadata()`
- [ ] `MarkdownOutputSink` receives initial metadata, not the detected language from transcription

## Priority 7: Better resampler ✅

- [x] Linear interpolation (`preprocess/mod.rs`) introduces aliasing that degrades transcription accuracy
- [x] Switch to `rubato` crate for meaningful quality improvement

## Other

- [ ] Replace `std::fs::write`/`create_dir_all` in `MarkdownOutputSink` with `tokio::fs` (blocks async runtime)
- [ ] Add observability — periodic metrics logging (drop rate, latency, segments processed) for headless operation
- [ ] Consider making `PipelineRunner` generic over stages instead of `Box<dyn Trait>` for Pi performance
- [ ] Add config hot-reload for model/language switching without restart
- [ ] Add disk management — rotation or cleanup policy for accumulated markdown files
