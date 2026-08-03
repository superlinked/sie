use std::hint::black_box;
use std::time::{Duration, Instant};

use sie_audio_prep::{decode_audio, inspect_audio, AudioLimits};

const RUNS: usize = 7;
const MAX_INSPECTION_P95: Duration = Duration::from_secs(1);

fn wav_pcm16_silence(sample_rate: u32, frames: usize) -> Vec<u8> {
    let data_len = (frames * size_of::<i16>()) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.resize(44 + data_len as usize, 0);
    wav
}

fn mp3_silence(frame_count: usize) -> Vec<u8> {
    let mut frame = vec![0; 417];
    frame[..4].copy_from_slice(&[0xff, 0xfb, 0x90, 0xc4]);
    frame.repeat(frame_count)
}

fn percentile(samples: &mut [Duration], percentile: f64) -> Duration {
    samples.sort_unstable();
    let index = ((samples.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(samples.len() - 1);
    samples[index]
}

fn benchmark_shape(name: &str, data: Vec<u8>, format: &str) {
    let limits = AudioLimits::default();
    let warm_inspection =
        inspect_audio(data.clone(), Some(format), limits).expect("warm inspection succeeds");
    let warm_preparation =
        decode_audio(data.clone(), Some(format), limits).expect("warm preparation succeeds");
    assert_eq!(warm_inspection.duration_ms, warm_preparation.duration_ms);
    assert_eq!(
        warm_inspection.source_sample_count,
        warm_preparation.source_sample_count
    );
    assert_eq!(
        warm_inspection.source_sample_rate,
        warm_preparation.source_sample_rate
    );

    let mut inspection = Vec::with_capacity(RUNS);
    let mut preparation = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let inspect_input = data.clone();
        let started = Instant::now();
        let inspected = inspect_audio(black_box(inspect_input), Some(format), limits)
            .expect("inspection succeeds");
        inspection.push(started.elapsed());

        let prepare_input = data.clone();
        let started = Instant::now();
        let prepared = decode_audio(black_box(prepare_input), Some(format), limits)
            .expect("preparation succeeds");
        preparation.push(started.elapsed());

        assert_eq!(inspected.duration_ms, prepared.duration_ms);
        assert_eq!(inspected.source_sample_count, prepared.source_sample_count);
        assert_eq!(inspected.source_sample_rate, prepared.source_sample_rate);
        assert_eq!(inspected.source_channels, prepared.source_channels);
        assert_eq!(inspected.container, prepared.container);
    }

    let inspection_p50 = percentile(&mut inspection.clone(), 0.50);
    let inspection_p95 = percentile(&mut inspection, 0.95);
    let preparation_p50 = percentile(&mut preparation.clone(), 0.50);
    let preparation_p95 = percentile(&mut preparation, 0.95);
    println!(
        "AUDIO_DUPLICATE_DECODE_BENCH shape={name} bytes={} duration_ms={} inspection_p50_us={} inspection_p95_us={} preparation_p50_us={} preparation_p95_us={}",
        data.len(),
        warm_inspection.duration_ms,
        inspection_p50.as_micros(),
        inspection_p95.as_micros(),
        preparation_p50.as_micros(),
        preparation_p95.as_micros(),
    );
    assert!(
        inspection_p95 <= MAX_INSPECTION_P95,
        "{name} inspection p95 {inspection_p95:?} exceeds {MAX_INSPECTION_P95:?}",
    );
}

#[test]
#[ignore = "release-only maximum-duration duplicate-decode benchmark"]
fn maximum_duration_duplicate_decode_overhead() {
    benchmark_shape("wav", wav_pcm16_silence(8_000, 5_760_000), "wav");
    benchmark_shape("mp3", mp3_silence(27_562), "mp3");
}
