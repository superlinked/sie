# SIE audio preparation

`sie-audio-prep` is the single implementation of generic audio preparation
used by the Rust worker sidecar and Python direct execution. Its inspection
mode decodes and validates every packet while retaining only authoritative
source duration metadata, without materializing canonical PCM.

It probes untrusted in-memory media, decodes one audio track, validates
compressed and decoded expansion limits, averages mono/stereo input to mono,
and resamples to 16 kHz PCM. The sidecar wire candidate is signed PCM16 LE;
model-specific feature extraction remains in the model adapter.

Authoritative duration is the checked integer ceiling of decoded source samples
times 1,000 divided by the source sample rate, so every valid nonempty clip carries at least 1 ms.

Supported launch containers/codecs:

- WAV with PCM or ADPCM
- MP3
- FLAC
- Ogg with Vorbis or Opus
- M4A/ISO-MP4 with AAC-LC or ALAC
- WebM/Matroska with Vorbis or Opus

The implementation does not invoke `ffmpeg`. Opus is decoded through the exact
`symphonia-adapter-libopus` 0.3.0 release and bundled libopus 1.6.1 source pinned
by `opusic-sys` 0.7.3.

Dependency licenses:

- Symphonia 0.6: MPL-2.0
- symphonia-adapter-libopus: MIT OR Apache-2.0
- opusic-sys/libopus: BSD-3-Clause and Opus licenses
- Rubato 4.0 and audioadapter-buffers 4.0: MIT OR Apache-2.0
- PyO3 0.29 (optional Python extension): MIT OR Apache-2.0
