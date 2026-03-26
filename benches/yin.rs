use criterion::{black_box, criterion_group, criterion_main, Criterion};

use yin_no_std::Yin;

struct Case {
    name: &'static str,
    frame: Vec<f32>,
    tau_max: usize,
    diff: Vec<f32>,
    cmnd: Vec<f32>,
}

fn make_sine_frame(sample_rate: f32, freq_hz: f32, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    let phase_inc = 2.0 * core::f32::consts::PI * freq_hz / sample_rate;
    let mut phase = 0.0f32;

    for _ in 0..len {
        out.push(phase.sin());
        phase += phase_inc;
    }

    out
}

fn bench_yin_detect(c: &mut Criterion) {
    let sample_rate = 48_000.0f32;
    let yin = Yin::new(sample_rate, 0.10, 0.15);

    let mut cases = vec![
        {
            let tau_max = 128;
            let frame = make_sine_frame(sample_rate, 440.0, 256);
            Case {
                name: "256_440hz",
                frame,
                tau_max,
                diff: vec![0.0; tau_max + 1],
                cmnd: vec![0.0; tau_max + 1],
            }
        },
        {
            let tau_max = 256;
            let frame = make_sine_frame(sample_rate, 220.0, 512);
            Case {
                name: "512_220hz",
                frame,
                tau_max,
                diff: vec![0.0; tau_max + 1],
                cmnd: vec![0.0; tau_max + 1],
            }
        },
        {
            let tau_max = 512;
            let frame = make_sine_frame(sample_rate, 110.0, 1024);
            Case {
                name: "1024_110hz",
                frame,
                tau_max,
                diff: vec![0.0; tau_max + 1],
                cmnd: vec![0.0; tau_max + 1],
            }
        },
    ];

    let mut group = c.benchmark_group("yin_detect");

    for case in &mut cases {
        group.bench_function(case.name, |b| {
            b.iter(|| {
                let result = yin.detect(
                    black_box(case.frame.as_slice()),
                    black_box(case.tau_max),
                    black_box(case.diff.as_mut_slice()),
                    black_box(case.cmnd.as_mut_slice()),
                );
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_yin_detect);
criterion_main!(benches);
