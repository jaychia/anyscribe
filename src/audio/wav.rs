//! WAV file writing and reading for the disk-backed audio buffer.
//!
//! Audio is written as 32-bit IEEE float PCM in a standard RIFF/WAVE container.
//! The [`WavWriter`] creates the file with a placeholder header, appends samples
//! during recording, and [`finalize`](WavWriter::finalize)s the header when
//! recording ends. A separate [`WavReader`] reads samples back from disk using
//! its own file handle, allowing concurrent write+read.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::ScribeError;

/// Size of the RIFF/WAVE header for 32-bit float PCM (bytes).
const WAV_HEADER_SIZE: u64 = 44;

/// IEEE float format tag.
const FORMAT_IEEE_FLOAT: u16 = 3;

/// Bits per sample for f32 audio.
const BITS_PER_SAMPLE: u16 = 32;

// ── Writer ────────────────────────────────────────────────────────

/// Writes 32-bit float PCM audio to a WAV file.
///
/// The file is created with a RIFF/WAVE header containing placeholder sizes.
/// Call [`write_samples`](WavWriter::write_samples) to append audio data,
/// then [`finalize`](WavWriter::finalize) to update the header with the
/// correct data length.
pub struct WavWriter {
    file: std::fs::File,
    total_data_bytes: u32,
}

impl WavWriter {
    /// Creates a new WAV file and writes the initial header.
    pub fn new(path: &Path, sample_rate: u32, channels: u16) -> Result<Self, ScribeError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                ScribeError::Audio(format!(
                    "failed to create WAV directory {}: {e}",
                    parent.display()
                ))
            })?;
        }

        let mut file = std::fs::File::create(path)
            .map_err(|e| ScribeError::Audio(format!("failed to create WAV file: {e}")))?;

        let byte_rate = sample_rate * channels as u32 * (BITS_PER_SAMPLE / 8) as u32;
        let block_align = channels * (BITS_PER_SAMPLE / 8);

        // RIFF header
        file.write_all(b"RIFF")?;
        file.write_all(&0u32.to_le_bytes())?; // placeholder: file_size - 8
        file.write_all(b"WAVE")?;

        // fmt subchunk
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?; // subchunk size
        file.write_all(&FORMAT_IEEE_FLOAT.to_le_bytes())?;
        file.write_all(&channels.to_le_bytes())?;
        file.write_all(&sample_rate.to_le_bytes())?;
        file.write_all(&byte_rate.to_le_bytes())?;
        file.write_all(&block_align.to_le_bytes())?;
        file.write_all(&BITS_PER_SAMPLE.to_le_bytes())?;

        // data subchunk
        file.write_all(b"data")?;
        file.write_all(&0u32.to_le_bytes())?; // placeholder: data size

        file.flush()?;

        Ok(Self {
            file,
            total_data_bytes: 0,
        })
    }

    /// Appends interleaved f32 samples to the WAV file and flushes.
    pub fn write_samples(&mut self, samples: &[f32]) -> Result<(), ScribeError> {
        for &s in samples {
            self.file.write_all(&s.to_le_bytes())?;
        }
        self.total_data_bytes += (samples.len() * 4) as u32;
        self.file.flush()?;
        Ok(())
    }

    /// Updates the RIFF and data chunk sizes in the header.
    ///
    /// Must be called after all samples have been written.
    pub fn finalize(&mut self) -> Result<(), ScribeError> {
        // RIFF chunk size = total_data_bytes + 36 (header minus first 8 bytes)
        let riff_size = self.total_data_bytes + 36;

        self.file.seek(SeekFrom::Start(4))?;
        self.file.write_all(&riff_size.to_le_bytes())?;

        self.file.seek(SeekFrom::Start(40))?;
        self.file.write_all(&self.total_data_bytes.to_le_bytes())?;

        self.file.flush()?;
        Ok(())
    }
}

// ── In-memory encoding ───────────────────────────────────────────

/// Encodes mono f32 PCM samples as a complete WAV file in memory.
///
/// Returns the raw bytes of a valid RIFF/WAVE container with 32-bit IEEE
/// float PCM. Useful for remote transcription APIs that accept WAV uploads.
pub fn encode_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let channels: u16 = 1;
    let data_size = (samples.len() * 4) as u32;
    let byte_rate = sample_rate * channels as u32 * (BITS_PER_SAMPLE / 8) as u32;
    let block_align = channels * (BITS_PER_SAMPLE / 8);

    let mut buf = Vec::with_capacity(WAV_HEADER_SIZE as usize + data_size as usize);

    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(data_size + 36).to_le_bytes());
    buf.extend_from_slice(b"WAVE");

    // fmt subchunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&FORMAT_IEEE_FLOAT.to_le_bytes());
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());

    // data subchunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        buf.extend_from_slice(&s.to_le_bytes());
    }

    buf
}

// ── Reader ────────────────────────────────────────────────────────

/// Reads f32 samples from a WAV file, starting after the 44-byte header.
///
/// Maintains a read cursor that advances with each call to
/// [`read_samples`](WavReader::read_samples). Designed for concurrent use
/// with a [`WavWriter`] on the same file (via separate file handles).
pub struct WavReader {
    file: std::fs::File,
}

impl WavReader {
    /// Opens a WAV file for reading, seeking past the header.
    pub fn new(path: &Path) -> Result<Self, ScribeError> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| ScribeError::Audio(format!("failed to open WAV file: {e}")))?;
        file.seek(SeekFrom::Start(WAV_HEADER_SIZE))?;
        Ok(Self { file })
    }

    /// Reads `num_samples` interleaved f32 values from the current position.
    pub fn read_samples(&mut self, num_samples: usize) -> Result<Vec<f32>, ScribeError> {
        let byte_count = num_samples * 4;
        let mut buf = vec![0u8; byte_count];
        self.file.read_exact(&mut buf).map_err(|e| {
            ScribeError::Audio(format!("failed to read WAV samples: {e}"))
        })?;

        let samples = buf
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_write_and_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wav");

        let samples: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();

        let mut writer = WavWriter::new(&path, 16000, 1).unwrap();
        writer.write_samples(&samples).unwrap();
        writer.finalize().unwrap();

        let mut reader = WavReader::new(&path).unwrap();
        let read_back = reader.read_samples(100).unwrap();

        assert_eq!(samples.len(), read_back.len());
        for (a, b) in samples.iter().zip(read_back.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_wav_header_structure() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("header.wav");

        let samples = vec![1.0f32; 10];
        let mut writer = WavWriter::new(&path, 44100, 2).unwrap();
        writer.write_samples(&samples).unwrap();
        writer.finalize().unwrap();

        let bytes = std::fs::read(&path).unwrap();

        // RIFF header
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");

        // fmt subchunk
        assert_eq!(&bytes[12..16], b"fmt ");
        let fmt_size = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        assert_eq!(fmt_size, 16);
        let format = u16::from_le_bytes(bytes[20..22].try_into().unwrap());
        assert_eq!(format, FORMAT_IEEE_FLOAT);
        let channels = u16::from_le_bytes(bytes[22..24].try_into().unwrap());
        assert_eq!(channels, 2);
        let sample_rate = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        assert_eq!(sample_rate, 44100);
        let bits = u16::from_le_bytes(bytes[34..36].try_into().unwrap());
        assert_eq!(bits, 32);

        // data subchunk
        assert_eq!(&bytes[36..40], b"data");
        let data_size = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        assert_eq!(data_size, 10 * 4); // 10 samples * 4 bytes each

        // RIFF size
        let riff_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(riff_size, data_size + 36);
    }

    #[test]
    fn test_wav_concurrent_write_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("concurrent.wav");

        let mut writer = WavWriter::new(&path, 16000, 1).unwrap();

        // Write first batch
        let batch1: Vec<f32> = vec![0.1, 0.2, 0.3];
        writer.write_samples(&batch1).unwrap();

        // Open reader before writer is finalized
        let mut reader = WavReader::new(&path).unwrap();
        let read1 = reader.read_samples(3).unwrap();
        assert_eq!(read1.len(), 3);
        assert!((read1[0] - 0.1).abs() < f32::EPSILON);

        // Write second batch
        let batch2: Vec<f32> = vec![0.4, 0.5];
        writer.write_samples(&batch2).unwrap();

        // Read second batch from same reader
        let read2 = reader.read_samples(2).unwrap();
        assert_eq!(read2.len(), 2);
        assert!((read2[0] - 0.4).abs() < f32::EPSILON);

        writer.finalize().unwrap();
    }

    #[test]
    fn test_wav_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.wav");

        let mut writer = WavWriter::new(&path, 16000, 1).unwrap();
        writer.finalize().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), WAV_HEADER_SIZE as usize);

        let data_size = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        assert_eq!(data_size, 0);
    }

    #[test]
    fn test_encode_wav_bytes_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("encoded.wav");

        let samples: Vec<f32> = (0..160).map(|i| (i as f32 * 0.005).sin()).collect();
        let bytes = encode_wav_bytes(&samples, 16000);

        // Write to disk so WavReader can read it back
        std::fs::write(&path, &bytes).unwrap();

        let mut reader = WavReader::new(&path).unwrap();
        let read_back = reader.read_samples(160).unwrap();

        assert_eq!(samples.len(), read_back.len());
        for (a, b) in samples.iter().zip(read_back.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }

        // Verify header correctness
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        let data_size = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        assert_eq!(data_size, 160 * 4);
    }
}
