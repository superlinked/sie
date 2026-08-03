use std::io::Cursor;

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};
use symphonia::core::audio::sample::Sample;
use symphonia::core::audio::Channels;
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::codecs::registry::RegisterableAudioDecoder;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia_adapter_libopus::OpusDecoder;
use thiserror::Error;

#[cfg(feature = "msgpack")]
pub mod msgpack;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub const DEFAULT_MAX_COMPRESSED_BYTES: usize = 24 * 1024 * 1024;
pub const DEFAULT_MAX_DURATION_MS: u64 = 12 * 60 * 1_000;
pub const DEFAULT_MIN_SAMPLE_RATE: u32 = 8_000;
pub const DEFAULT_MAX_SAMPLE_RATE: u32 = 48_000;
pub const DEFAULT_MAX_CHANNELS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioLimits {
    pub max_compressed_bytes: usize,
    pub max_duration_ms: u64,
    pub min_sample_rate: u32,
    pub max_sample_rate: u32,
    pub max_channels: usize,
}

impl Default for AudioLimits {
    fn default() -> Self {
        Self {
            max_compressed_bytes: DEFAULT_MAX_COMPRESSED_BYTES,
            max_duration_ms: DEFAULT_MAX_DURATION_MS,
            min_sample_rate: DEFAULT_MIN_SAMPLE_RATE,
            max_sample_rate: DEFAULT_MAX_SAMPLE_RATE,
            max_channels: DEFAULT_MAX_CHANNELS,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioContainer {
    Wav,
    Mp3,
    Flac,
    Ogg,
    IsoMp4,
    WebM,
}

impl AudioContainer {
    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
            Self::Ogg => "ogg",
            Self::IsoMp4 => "m4a",
            Self::WebM => "webm",
        }
    }

    fn hint_extension(self) -> &'static str {
        self.canonical_name()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub sample_count: u64,
    pub duration_ms: u64,
    pub source_sample_rate: u32,
    pub source_channels: usize,
    pub source_sample_count: u64,
    pub container: AudioContainer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedPcm16 {
    pub pcm_s16le: Vec<u8>,
    pub sample_rate: u32,
    pub sample_count: u64,
    pub duration_ms: u64,
    pub source_sample_rate: u32,
    pub source_channels: usize,
    pub source_sample_count: u64,
    pub container: AudioContainer,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InspectedAudio {
    pub duration_ms: u64,
    pub source_sample_rate: u32,
    pub source_channels: usize,
    pub source_sample_count: u64,
    pub container: AudioContainer,
}

#[derive(Clone, Copy)]
enum DecodeMode {
    Inspect,
    Prepare,
}

struct DecodedSource {
    mono: Option<Vec<f32>>,
    duration_ms: u64,
    source_sample_rate: u32,
    source_channels: usize,
    source_sample_count: u64,
    container: AudioContainer,
}

impl PreparedAudio {
    pub fn pcm_s16le(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.samples.len() * size_of::<i16>());
        for sample in &self.samples {
            let scaled = (sample.clamp(-1.0, 1.0) * 32_768.0).round();
            let pcm_sample = scaled.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
            bytes.extend_from_slice(&pcm_sample.to_le_bytes());
        }
        bytes
    }

    pub fn into_pcm_s16le(self) -> PreparedPcm16 {
        let Self {
            samples,
            sample_rate,
            sample_count,
            duration_ms,
            source_sample_rate,
            source_channels,
            source_sample_count,
            container,
        } = self;
        let mut pcm_s16le = Vec::with_capacity(samples.len() * size_of::<i16>());
        for sample in samples {
            let scaled = (sample.clamp(-1.0, 1.0) * 32_768.0).round();
            let pcm_sample = scaled.clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16;
            pcm_s16le.extend_from_slice(&pcm_sample.to_le_bytes());
        }
        PreparedPcm16 {
            pcm_s16le,
            sample_rate,
            sample_count,
            duration_ms,
            source_sample_rate,
            source_channels,
            source_sample_count,
            container,
        }
    }

    #[cfg(any(test, feature = "evaluation"))]
    pub fn pcm_f32le_for_evaluation(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.samples.len() * size_of::<f32>());
        for sample in &self.samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AudioPrepError {
    #[error("audio input is empty")]
    Empty,
    #[error("compressed audio exceeds {max_bytes} bytes")]
    CompressedTooLarge { max_bytes: usize },
    #[error("unsupported or unrecognized audio container")]
    UnsupportedContainer,
    #[error("declared audio format does not match probed container")]
    FormatMismatch,
    #[error("audio container has no decodable audio track")]
    NoAudioTrack,
    #[error("audio container has multiple audio tracks")]
    MultipleAudioTracks,
    #[error("unsupported audio codec: {0}")]
    UnsupportedCodec(String),
    #[error("audio sample rate {actual} is outside {min}..={max} Hz")]
    InvalidSampleRate { actual: u32, min: u32, max: u32 },
    #[error("audio channel count {actual} is outside 1..={max}")]
    InvalidChannels { actual: usize, max: usize },
    #[error("decoded audio exceeds {max_duration_ms} ms")]
    DurationTooLong { max_duration_ms: u64 },
    #[error("malformed audio: {0}")]
    Malformed(String),
    #[error("audio decode failed: {0}")]
    Decode(String),
    #[error("audio resampling failed: {0}")]
    Resample(String),
}

pub fn decode_audio(
    data: Vec<u8>,
    declared_format: Option<&str>,
    limits: AudioLimits,
) -> Result<PreparedAudio, AudioPrepError> {
    let decoded = decode_source(data, declared_format, limits, DecodeMode::Prepare)?;
    let mono = decoded
        .mono
        .expect("prepare mode always retains decoded mono samples");
    let samples = resample_to_target(mono, decoded.source_sample_rate)?;
    let sample_count = samples.len() as u64;
    Ok(PreparedAudio {
        samples,
        sample_rate: TARGET_SAMPLE_RATE,
        sample_count,
        duration_ms: decoded.duration_ms,
        source_sample_rate: decoded.source_sample_rate,
        source_channels: decoded.source_channels,
        source_sample_count: decoded.source_sample_count,
        container: decoded.container,
    })
}

/// Decode and validate every source packet while retaining only authoritative
/// duration metadata. Worker preparation uses [`decode_audio`] to additionally
/// downmix, resample, and materialize canonical PCM.
pub fn inspect_audio(
    data: Vec<u8>,
    declared_format: Option<&str>,
    limits: AudioLimits,
) -> Result<InspectedAudio, AudioPrepError> {
    let decoded = decode_source(data, declared_format, limits, DecodeMode::Inspect)?;
    Ok(InspectedAudio {
        duration_ms: decoded.duration_ms,
        source_sample_rate: decoded.source_sample_rate,
        source_channels: decoded.source_channels,
        source_sample_count: decoded.source_sample_count,
        container: decoded.container,
    })
}

fn decode_source(
    data: Vec<u8>,
    declared_format: Option<&str>,
    limits: AudioLimits,
    mode: DecodeMode,
) -> Result<DecodedSource, AudioPrepError> {
    validate_limits(limits)?;
    if data.is_empty() {
        return Err(AudioPrepError::Empty);
    }
    if data.len() > limits.max_compressed_bytes {
        return Err(AudioPrepError::CompressedTooLarge {
            max_bytes: limits.max_compressed_bytes,
        });
    }

    let container = detect_container(&data).ok_or(AudioPrepError::UnsupportedContainer)?;
    validate_declared_format(container, declared_format)?;

    let source = MediaSourceStream::new(Box::new(Cursor::new(data)), Default::default());
    let mut hint = Hint::new();
    hint.with_extension(container.hint_extension());
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            source,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|error| AudioPrepError::Decode(error.to_string()))?;

    let audio_tracks = format
        .tracks()
        .iter()
        .filter(|track| track.track_type() == Some(TrackType::Audio))
        .collect::<Vec<_>>();
    if audio_tracks.is_empty() {
        return Err(AudioPrepError::NoAudioTrack);
    }
    if audio_tracks.len() != 1 {
        return Err(AudioPrepError::MultipleAudioTracks);
    }
    let track = audio_tracks[0];
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or(AudioPrepError::NoAudioTrack)?;
    let declared_sample_rate = audio_params.sample_rate.ok_or_else(|| {
        AudioPrepError::Malformed("the audio stream does not declare a sample rate".to_string())
    })?;
    let declared_channels = audio_params
        .channels
        .as_ref()
        .map(|channels| channels.count())
        .ok_or_else(|| {
            AudioPrepError::Malformed(
                "the audio stream does not declare a channel layout".to_string(),
            )
        })?;
    validate_audio_shape(declared_sample_rate, declared_channels, limits)?;
    let track_id = track.id;

    let decoder_options = AudioDecoderOptions::default();
    let mut decoder =
        match symphonia::default::get_codecs().make_audio_decoder(audio_params, &decoder_options) {
            Ok(decoder) => decoder,
            Err(default_error) => OpusDecoder::try_registry_new(audio_params, &decoder_options)
                .map_err(|_| AudioPrepError::UnsupportedCodec(default_error.to_string()))?,
        };

    let mut decoded_sample_rate = None;
    let mut decoded_channels = None;
    let mut mono = match mode {
        DecodeMode::Inspect => None,
        DecodeMode::Prepare => Some(Vec::new()),
    };
    let mut source_sample_count = 0usize;
    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(error) => return Err(AudioPrepError::Decode(error.to_string())),
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet).map_err(map_decode_error)?;
        let decoded_spec = decoded.spec();
        let actual_sample_rate = decoded_spec.rate();
        let channels = validate_decoded_spec(
            &mut decoded_sample_rate,
            &mut decoded_channels,
            actual_sample_rate,
            decoded_spec.channels(),
            limits,
        )?;
        let max_source_frames = max_source_frames(actual_sample_rate, limits.max_duration_ms);
        let frame_count = decoded.frames();
        if source_sample_count.saturating_add(frame_count) > max_source_frames {
            return Err(AudioPrepError::DurationTooLong {
                max_duration_ms: limits.max_duration_ms,
            });
        }
        source_sample_count += frame_count;
        if let Some(mono) = mono.as_mut() {
            let mut interleaved = vec![f32::MID; decoded.samples_interleaved()];
            decoded.copy_to_slice_interleaved(&mut interleaved);
            if channels == 1 {
                mono.extend_from_slice(&interleaved);
            } else {
                mono.extend(
                    interleaved
                        .chunks_exact(channels)
                        .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32),
                );
            }
        }
    }
    if source_sample_count == 0 {
        return Err(AudioPrepError::Decode(
            "audio track decoded to zero samples".into(),
        ));
    }
    let source_sample_rate = decoded_sample_rate
        .ok_or_else(|| AudioPrepError::Malformed("decoded audio has no sample rate".to_string()))?;
    let source_channels = decoded_channels
        .as_ref()
        .map(Channels::count)
        .ok_or_else(|| {
            AudioPrepError::Malformed("decoded audio has no channel layout".to_string())
        })?;

    let source_sample_count = source_sample_count as u64;
    let duration_ms =
        duration_ms_ceil(source_sample_count, source_sample_rate).ok_or_else(|| {
            AudioPrepError::Malformed("decoded audio duration exceeds supported range".into())
        })?;
    if duration_ms > limits.max_duration_ms {
        return Err(AudioPrepError::DurationTooLong {
            max_duration_ms: limits.max_duration_ms,
        });
    }
    Ok(DecodedSource {
        mono,
        duration_ms,
        source_sample_rate,
        source_channels,
        source_sample_count,
        container,
    })
}

pub fn decode_audio_to_pcm16(
    data: Vec<u8>,
    declared_format: Option<&str>,
    limits: AudioLimits,
) -> Result<PreparedPcm16, AudioPrepError> {
    decode_audio(data, declared_format, limits).map(PreparedAudio::into_pcm_s16le)
}

fn validate_limits(limits: AudioLimits) -> Result<(), AudioPrepError> {
    if limits.max_compressed_bytes == 0
        || limits.max_duration_ms == 0
        || limits.min_sample_rate == 0
        || limits.min_sample_rate > limits.max_sample_rate
        || limits.max_channels == 0
    {
        return Err(AudioPrepError::Decode(
            "invalid audio preparation limits".into(),
        ));
    }
    Ok(())
}

fn validate_audio_shape(
    sample_rate: u32,
    channels: usize,
    limits: AudioLimits,
) -> Result<(), AudioPrepError> {
    if !(limits.min_sample_rate..=limits.max_sample_rate).contains(&sample_rate) {
        return Err(AudioPrepError::InvalidSampleRate {
            actual: sample_rate,
            min: limits.min_sample_rate,
            max: limits.max_sample_rate,
        });
    }
    if channels == 0 || channels > limits.max_channels {
        return Err(AudioPrepError::InvalidChannels {
            actual: channels,
            max: limits.max_channels,
        });
    }
    Ok(())
}

fn validate_decoded_spec(
    expected_sample_rate: &mut Option<u32>,
    expected_channels: &mut Option<Channels>,
    sample_rate: u32,
    channels: &Channels,
    limits: AudioLimits,
) -> Result<usize, AudioPrepError> {
    let channel_count = channels.count();
    validate_audio_shape(sample_rate, channel_count, limits)?;

    if let Some(expected) = expected_sample_rate {
        if *expected != sample_rate {
            return Err(AudioPrepError::Malformed(format!(
                "decoded sample rate changed from {expected} to {sample_rate} Hz"
            )));
        }
    } else {
        *expected_sample_rate = Some(sample_rate);
    }

    if let Some(expected) = expected_channels.as_ref() {
        if expected != channels {
            return Err(AudioPrepError::Malformed(format!(
                "decoded channel layout changed from {expected:?} to {channels:?}"
            )));
        }
    } else {
        *expected_channels = Some(channels.clone());
    }

    Ok(channel_count)
}
fn max_source_frames(sample_rate: u32, max_duration_ms: u64) -> usize {
    (u64::from(sample_rate)
        .saturating_mul(max_duration_ms)
        .saturating_add(999)
        / 1_000)
        .min(usize::MAX as u64) as usize
}

fn duration_ms_ceil(source_sample_count: u64, source_sample_rate: u32) -> Option<u64> {
    let sample_rate = u64::from(source_sample_rate);
    if sample_rate == 0 {
        return None;
    }
    let whole_ms = (source_sample_count / sample_rate).checked_mul(1_000)?;
    let remainder = source_sample_count % sample_rate;
    let fractional_ms = remainder.checked_mul(1_000)?.checked_add(sample_rate - 1)? / sample_rate;
    whole_ms.checked_add(fractional_ms)
}

fn map_decode_error(error: SymphoniaError) -> AudioPrepError {
    match error {
        SymphoniaError::Unsupported(message) => AudioPrepError::UnsupportedCodec(message.into()),
        other => AudioPrepError::Decode(other.to_string()),
    }
}

fn resample_to_target(mono: Vec<f32>, source_sample_rate: u32) -> Result<Vec<f32>, AudioPrepError> {
    if source_sample_rate == TARGET_SAMPLE_RATE {
        return Ok(mono);
    }
    let input_frames = mono.len();
    let input = InterleavedSlice::new(&mono, 1, input_frames)
        .map_err(|error| AudioPrepError::Resample(error.to_string()))?;
    let mut resampler = Fft::<f32>::new(
        source_sample_rate as usize,
        TARGET_SAMPLE_RATE as usize,
        1_024,
        1,
        FixedSync::Input,
    )
    .map_err(|error| AudioPrepError::Resample(error.to_string()))?;
    let output = resampler
        .process_all(&input, input_frames, None)
        .map_err(|error| AudioPrepError::Resample(error.to_string()))?;
    Ok(output.take_data())
}

fn validate_declared_format(
    container: AudioContainer,
    declared_format: Option<&str>,
) -> Result<(), AudioPrepError> {
    let Some(declared) = declared_format
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(());
    };
    let matches = match declared.to_ascii_lowercase().as_str() {
        "wav" | "wave" | "audio/wav" | "audio/x-wav" => container == AudioContainer::Wav,
        "mp3" | "mpeg" | "mpga" | "audio/mpeg" => container == AudioContainer::Mp3,
        "flac" | "audio/flac" => container == AudioContainer::Flac,
        "ogg" | "oga" | "audio/ogg" => container == AudioContainer::Ogg,
        "m4a" | "mp4" | "audio/mp4" | "audio/x-m4a" => container == AudioContainer::IsoMp4,
        "webm" | "audio/webm" => container == AudioContainer::WebM,
        _ => false,
    };
    if matches {
        return Ok(());
    }
    Err(AudioPrepError::FormatMismatch)
}

pub fn detect_container(data: &[u8]) -> Option<AudioContainer> {
    if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WAVE" {
        return Some(AudioContainer::Wav);
    }
    if data.starts_with(b"fLaC") {
        return Some(AudioContainer::Flac);
    }
    if data.starts_with(b"OggS") {
        return Some(AudioContainer::Ogg);
    }
    if data.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        return Some(AudioContainer::WebM);
    }
    if data.len() >= 12 && &data[4..8] == b"ftyp" {
        return Some(AudioContainer::IsoMp4);
    }
    if data.starts_with(b"ID3") || (data.len() >= 2 && data[0] == 0xff && data[1] & 0xe0 == 0xe0) {
        return Some(AudioContainer::Mp3);
    }
    None
}

#[cfg(feature = "python")]
mod python {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::{PyBytes, PyDict};

    #[cfg(feature = "evaluation")]
    use super::decode_audio;
    use super::{decode_audio_to_pcm16, AudioLimits};

    fn decode_pcm16_without_gil(
        py: Python<'_>,
        data: Vec<u8>,
        format: Option<&str>,
    ) -> PyResult<super::PreparedPcm16> {
        let declared_format = format.map(str::to_owned);
        py.detach(move || {
            decode_audio_to_pcm16(data, declared_format.as_deref(), AudioLimits::default())
        })
        .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[cfg(feature = "evaluation")]
    fn decode_f32_without_gil(
        py: Python<'_>,
        data: Vec<u8>,
        format: Option<&str>,
    ) -> PyResult<super::PreparedAudio> {
        let declared_format = format.map(str::to_owned);
        py.detach(move || decode_audio(data, declared_format.as_deref(), AudioLimits::default()))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[pyfunction(name = "decode_audio")]
    #[pyo3(signature = (data, format=None))]
    fn decode_audio_py<'py>(
        py: Python<'py>,
        data: Vec<u8>,
        format: Option<&str>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let prepared = decode_pcm16_without_gil(py, data, format)?;
        let result = PyDict::new(py);
        result.set_item("encoding", "pcm_s16le")?;
        result.set_item("pcm_s16le", PyBytes::new(py, &prepared.pcm_s16le))?;
        result.set_item("sample_rate", prepared.sample_rate)?;
        result.set_item("sample_count", prepared.sample_count)?;
        result.set_item("duration_ms", prepared.duration_ms)?;
        result.set_item("source_sample_rate", prepared.source_sample_rate)?;
        result.set_item("source_sample_count", prepared.source_sample_count)?;
        result.set_item("source_channels", prepared.source_channels)?;
        result.set_item("container", prepared.container.canonical_name())?;
        Ok(result)
    }

    #[cfg(feature = "evaluation")]
    #[pyfunction]
    #[pyo3(signature = (data, format=None))]
    fn _decode_audio_f32_for_evaluation<'py>(
        py: Python<'py>,
        data: Vec<u8>,
        format: Option<&str>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let prepared = decode_f32_without_gil(py, data, format)?;
        let result = PyDict::new(py);
        result.set_item(
            "pcm_f32le",
            PyBytes::new(py, &prepared.pcm_f32le_for_evaluation()),
        )?;
        result.set_item("sample_rate", prepared.sample_rate)?;
        result.set_item("sample_count", prepared.sample_count)?;
        result.set_item("duration_ms", prepared.duration_ms)?;
        result.set_item("source_sample_rate", prepared.source_sample_rate)?;
        result.set_item("source_channels", prepared.source_channels)?;
        result.set_item("source_sample_count", prepared.source_sample_count)?;
        result.set_item("container", prepared.container.canonical_name())?;
        Ok(result)
    }

    #[pymodule]
    fn sie_audio_prep(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(decode_audio_py, module)?)?;
        #[cfg(feature = "evaluation")]
        module.add_function(wrap_pyfunction!(_decode_audio_f32_for_evaluation, module)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symphonia::core::audio::layouts::{CHANNEL_LAYOUT_MONO, CHANNEL_LAYOUT_STEREO};

    fn wav_pcm16(sample_rate: u32, channels: u16, frames: &[[i16; 2]]) -> Vec<u8> {
        let samples = frames
            .iter()
            .flat_map(|frame| frame.iter().take(channels as usize))
            .copied()
            .collect::<Vec<_>>();
        let data_len = (samples.len() * size_of::<i16>()) as u32;
        let mut wav = Vec::with_capacity(44 + data_len as usize);
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&channels.to_le_bytes());
        wav.extend_from_slice(&sample_rate.to_le_bytes());
        wav.extend_from_slice(&(sample_rate * u32::from(channels) * 2).to_le_bytes());
        wav.extend_from_slice(&(channels * 2).to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        for sample in samples {
            wav.extend_from_slice(&sample.to_le_bytes());
        }
        wav
    }

    fn mp3_silence(frame_count: usize) -> Vec<u8> {
        // MPEG-1 Layer III, 128 kbps, 44.1 kHz, mono. Zero side information
        // and main data decode deterministically to silence.
        let mut frame = vec![0; 417];
        frame[..4].copy_from_slice(&[0xff, 0xfb, 0x90, 0xc4]);
        frame.repeat(frame_count)
    }

    fn assert_inspection_matches_preparation(
        data: Vec<u8>,
        format: Option<&str>,
        limits: AudioLimits,
    ) {
        let inspected = inspect_audio(data.clone(), format, limits).unwrap();
        let prepared = decode_audio(data, format, limits).unwrap();
        assert_eq!(inspected.duration_ms, prepared.duration_ms);
        assert_eq!(inspected.source_sample_rate, prepared.source_sample_rate);
        assert_eq!(inspected.source_channels, prepared.source_channels);
        assert_eq!(inspected.source_sample_count, prepared.source_sample_count);
        assert_eq!(inspected.container, prepared.container);
    }

    #[test]
    fn detects_supported_container_magic() {
        assert_eq!(
            detect_container(b"RIFF\x00\x00\x00\x00WAVE"),
            Some(AudioContainer::Wav)
        );
        assert_eq!(detect_container(b"fLaC"), Some(AudioContainer::Flac));
        assert_eq!(detect_container(b"OggS"), Some(AudioContainer::Ogg));
        assert_eq!(
            detect_container(b"\x1a\x45\xdf\xa3"),
            Some(AudioContainer::WebM)
        );
        assert_eq!(
            detect_container(b"\x00\x00\x00\x18ftypM4A "),
            Some(AudioContainer::IsoMp4)
        );
        assert_eq!(detect_container(b"ID3"), Some(AudioContainer::Mp3));
        assert_eq!(detect_container(b"\xff\xfb"), Some(AudioContainer::Mp3));
        assert_eq!(detect_container(b"not audio"), None);
    }

    #[test]
    fn computes_conservative_duration_without_overflow() {
        assert_eq!(duration_ms_ceil(1, 16_000), Some(1));
        assert_eq!(duration_ms_ceil(1_600, 16_000), Some(100));
        assert_eq!(duration_ms_ceil(1_601, 16_000), Some(101));
        assert_eq!(duration_ms_ceil(u64::MAX, 1), None);
    }

    #[test]
    fn inspection_matches_preparation_across_wav_shapes() {
        for sample_rate in [8_000, 16_000, 44_100, 48_000] {
            for channels in [1, 2] {
                for frame_count in [1, 31, 1_601] {
                    let frames = (0..frame_count)
                        .map(|index| [index as i16, -(index as i16)])
                        .collect::<Vec<_>>();
                    assert_inspection_matches_preparation(
                        wav_pcm16(sample_rate, channels, &frames),
                        Some("wav"),
                        AudioLimits::default(),
                    );
                }
            }
        }
    }

    #[test]
    fn inspection_matches_preparation_across_mp3_frame_counts() {
        for frame_count in [2, 4, 8, 31] {
            assert_inspection_matches_preparation(
                mp3_silence(frame_count),
                Some("mp3"),
                AudioLimits::default(),
            );
        }
    }

    #[test]
    fn inspection_matches_preparation_decode_errors() {
        let cases = [
            (Vec::new(), None, AudioLimits::default()),
            (b"not audio".to_vec(), None, AudioLimits::default()),
            (
                b"RIFF\x00\x00\x00\x00WAVE".to_vec(),
                Some("wav"),
                AudioLimits::default(),
            ),
            (
                b"\xff\xfb\x90\xc4".to_vec(),
                Some("mp3"),
                AudioLimits::default(),
            ),
            (
                wav_pcm16(16_000, 1, &[[0, 0]; 32]),
                Some("mp3"),
                AudioLimits::default(),
            ),
            (
                wav_pcm16(16_000, 1, &[[0, 0]; 32]),
                Some("wav"),
                AudioLimits {
                    max_duration_ms: 1,
                    ..AudioLimits::default()
                },
            ),
        ];
        for (data, format, limits) in cases {
            assert_eq!(
                inspect_audio(data.clone(), format, limits).unwrap_err(),
                decode_audio(data, format, limits).unwrap_err(),
            );
        }
    }

    #[test]
    fn decodes_mono_pcm_with_ceil_elapsed_milliseconds() {
        let frames = (0..1_601)
            .map(|index| {
                let sample = if index % 2 == 0 { 16_384 } else { -16_384 };
                [sample, 0]
            })
            .collect::<Vec<_>>();
        let prepared = decode_audio(
            wav_pcm16(16_000, 1, &frames),
            Some("wav"),
            AudioLimits::default(),
        )
        .unwrap();
        assert_eq!(prepared.sample_rate, 16_000);
        assert_eq!(prepared.sample_count, 1_601);
        assert_eq!(prepared.duration_ms, 101);
        assert_eq!(prepared.source_channels, 1);
        assert!((prepared.samples[0] - 0.5).abs() < 0.001);
        assert!((prepared.samples[1] + 0.5).abs() < 0.001);
        let pcm = prepared.into_pcm_s16le();
        assert_eq!(pcm.pcm_s16le.len(), 1_601 * 2);
        assert_eq!(pcm.sample_count, 1_601);
        assert_eq!(pcm.duration_ms, 101);
    }

    #[test]
    fn pcm16_candidate_quantization_error_is_bounded() {
        let frames = [
            [i16::MIN, 0],
            [-12_345, 0],
            [0, 0],
            [12_345, 0],
            [i16::MAX, 0],
        ];
        let prepared =
            decode_audio(wav_pcm16(16_000, 1, &frames), None, AudioLimits::default()).unwrap();
        let pcm = prepared.pcm_s16le();
        let reconstructed = pcm
            .chunks_exact(2)
            .map(|bytes| f32::from(i16::from_le_bytes([bytes[0], bytes[1]])) / 32_768.0)
            .collect::<Vec<_>>();
        assert_eq!(reconstructed.len(), prepared.samples.len());
        for (original, quantized) in prepared.samples.iter().zip(reconstructed) {
            assert!((original - quantized).abs() <= 1.0 / 32_768.0);
        }
    }

    #[test]
    fn downmixes_stereo_by_averaging_channels() {
        let frames = vec![[16_384, -16_384]; 160];
        let prepared =
            decode_audio(wav_pcm16(16_000, 2, &frames), None, AudioLimits::default()).unwrap();
        assert_eq!(prepared.source_channels, 2);
        assert!(prepared.samples.iter().all(|sample| sample.abs() < 0.000_1));
    }

    #[test]
    fn resamples_to_16khz_without_changing_source_duration() {
        let frames = vec![[8_192, 0]; 801];
        let prepared =
            decode_audio(wav_pcm16(8_000, 1, &frames), None, AudioLimits::default()).unwrap();
        assert_eq!(prepared.source_sample_rate, 8_000);
        assert_eq!(prepared.source_sample_count, 801);
        assert_eq!(prepared.sample_rate, 16_000);
        assert_eq!(prepared.duration_ms, 101);
    }

    #[test]
    fn sub_millisecond_audio_has_one_billable_millisecond() {
        let prepared = decode_audio(
            wav_pcm16(16_000, 1, &[[8_192, 0]]),
            None,
            AudioLimits::default(),
        )
        .unwrap();
        assert_eq!(prepared.source_sample_count, 1);
        assert_eq!(prepared.duration_ms, 1);
    }

    #[test]
    fn rejects_midstream_sample_rate_and_layout_changes() {
        let mut sample_rate = None;
        let mut channels = None;
        assert_eq!(
            validate_decoded_spec(
                &mut sample_rate,
                &mut channels,
                16_000,
                &CHANNEL_LAYOUT_MONO,
                AudioLimits::default()
            ),
            Ok(1)
        );
        let rate_error = validate_decoded_spec(
            &mut sample_rate,
            &mut channels,
            48_000,
            &CHANNEL_LAYOUT_MONO,
            AudioLimits::default(),
        )
        .unwrap_err();
        assert!(
            matches!(rate_error, AudioPrepError::Malformed(message) if message.contains("sample rate changed"))
        );
        let layout_error = validate_decoded_spec(
            &mut sample_rate,
            &mut channels,
            16_000,
            &CHANNEL_LAYOUT_STEREO,
            AudioLimits::default(),
        )
        .unwrap_err();
        assert!(
            matches!(layout_error, AudioPrepError::Malformed(message) if message.contains("channel layout changed"))
        );
    }

    #[test]
    fn rejects_declared_format_mismatch() {
        let error = decode_audio(
            wav_pcm16(16_000, 1, &[[0, 0]; 16]),
            Some("mp3"),
            AudioLimits::default(),
        )
        .unwrap_err();
        assert_eq!(error, AudioPrepError::FormatMismatch);
    }

    #[test]
    fn rejects_compressed_size_before_probe() {
        assert_eq!(
            AudioLimits::default().max_compressed_bytes,
            24 * 1024 * 1024
        );
        let limits = AudioLimits {
            max_compressed_bytes: 8,
            ..AudioLimits::default()
        };
        let error = decode_audio(vec![0; 9], None, limits).unwrap_err();
        assert_eq!(error, AudioPrepError::CompressedTooLarge { max_bytes: 8 });
    }

    #[test]
    fn rejects_rate_and_channel_limits() {
        let rate_error = decode_audio(
            wav_pcm16(4_000, 1, &[[0, 0]; 16]),
            None,
            AudioLimits::default(),
        )
        .unwrap_err();
        assert!(matches!(
            rate_error,
            AudioPrepError::InvalidSampleRate { actual: 4_000, .. }
        ));

        let high_rate_error = decode_audio(
            wav_pcm16(96_000, 1, &[[0, 0]; 16]),
            None,
            AudioLimits::default(),
        )
        .unwrap_err();
        assert!(matches!(
            high_rate_error,
            AudioPrepError::InvalidSampleRate { actual: 96_000, .. }
        ));

        let channel_error = decode_audio(
            wav_pcm16(16_000, 2, &[[0, 0]; 16]),
            None,
            AudioLimits {
                max_channels: 1,
                ..AudioLimits::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            channel_error,
            AudioPrepError::InvalidChannels { actual: 2, max: 1 }
        ));
    }

    #[test]
    fn rejects_duration_during_decode() {
        let error = decode_audio(
            wav_pcm16(16_000, 1, &[[0, 0]; 32]),
            None,
            AudioLimits {
                max_duration_ms: 1,
                ..AudioLimits::default()
            },
        )
        .unwrap_err();
        assert_eq!(
            error,
            AudioPrepError::DurationTooLong { max_duration_ms: 1 }
        );
    }

    #[test]
    fn rejects_malformed_matching_magic() {
        let error = decode_audio(
            b"RIFF\x00\x00\x00\x00WAVE".to_vec(),
            None,
            AudioLimits::default(),
        )
        .unwrap_err();
        assert!(matches!(error, AudioPrepError::Decode(_)));
    }
}
