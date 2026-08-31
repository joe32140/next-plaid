//! Residual-LUT stage-2 kernel A/B on one host: scalar vs AVX2 (pshufb +
//! maddubs) vs AVX-512 VNNI (vpdpbusd) on x86, NEON sdot on aarch64.
//! Shapes mirror the SciFact profile: 22 query tokens x 230 doc tokens x
//! dim 128, nbits 4, 256 centroids. All paths are pinned bit-exact to the
//! scalar reference by the test suite; this prints their speed.
use ndarray::{Array1, Array2};
use next_plaid::binary::quantize_query_i8;
use next_plaid::residual_lut::{
    active_kernel_name, build_query_planes, compute_inv_norms, maxsim_residual_lut_i8,
    maxsim_residual_lut_i8_force_avx2, maxsim_residual_lut_i8_force_avx512,
    maxsim_residual_lut_i8_force_neon, maxsim_residual_lut_scalar, quantize_lut,
};
use next_plaid::ResidualCodec;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

const DIM: usize = 128;
const NQ: usize = 22;
const ND: usize = 230;
const NDOCS: usize = 200;
const K: usize = 256;
const REPS: usize = 7;

struct Doc {
    packed: Array2<u8>,
    codes: Vec<i64>,
    inv: Vec<f32>,
}

fn main() {
    let mut rng = StdRng::seed_from_u64(42);
    let centroids = Array2::from_shape_fn((K, DIM), |_| rng.gen_range(-1.0f32..1.0));
    let sample: Vec<f32> = (0..40_000).map(|_| rng.gen_range(-0.3f32..0.3)).collect();
    let mut sorted = sample.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = 16usize;
    let q = |p: f64| sorted[((sorted.len() - 1) as f64 * p) as usize];
    let cutoffs: Array1<f32> = (1..n).map(|i| q(i as f64 / n as f64)).collect();
    let weights: Array1<f32> = (0..n).map(|i| q((i as f64 + 0.5) / n as f64)).collect();
    let codec =
        ResidualCodec::new(4, centroids, Array1::zeros(DIM), Some(cutoffs), Some(weights)).unwrap();
    let lut = quantize_lut(&codec).unwrap();

    let query = Array2::from_shape_fn((NQ, DIM), |_| rng.gen_range(-1.0f32..1.0));
    let q8 = quantize_query_i8(&query.view());
    let planes = build_query_planes(&q8, &lut, DIM);
    let cdot_t = Array2::from_shape_fn((K, NQ), |(c, qi)| {
        codec
            .centroids
            .row(c)
            .iter()
            .zip(query.row(qi))
            .map(|(a, b)| a * b)
            .sum::<f32>()
    });

    let docs: Vec<Doc> = (0..NDOCS)
        .map(|_| {
            let res = Array2::from_shape_fn((ND, DIM), |_| rng.gen_range(-0.3f32..0.3));
            let packed = codec.quantize_residuals(&res).unwrap();
            let codes: Vec<i64> = (0..ND).map(|_| rng.gen_range(0..K as i64)).collect();
            let inv = compute_inv_norms(&codec, &codes, &packed.view()).unwrap();
            Doc { packed, codes, inv }
        })
        .collect();

    // warmup through the safe dispatcher so shape bugs panic on its asserts
    let _ = maxsim_residual_lut_i8(
        &q8, Some(&planes), &docs[0].packed.view(), &docs[0].codes, &cdot_t.view(), &lut,
        &docs[0].inv, DIM,
    );
    println!("dispatcher would pick: {}", active_kernel_name(DIM, lut.nibble.is_some()));
    let macs = (NQ * ND * DIM * NDOCS) as f64;

    let mut arm = |name: &str, f: &dyn Fn(&Doc) -> Option<f32>| {
        if f(&docs[0]).is_none() {
            println!("{name:>14}: unsupported on this host");
            return;
        }
        let mut best = f64::INFINITY;
        let mut chk = 0f64;
        for _ in 0..REPS {
            let t0 = Instant::now();
            let mut acc = 0f64;
            for d in &docs {
                acc += f(d).unwrap() as f64;
            }
            let dt = t0.elapsed().as_secs_f64();
            if dt < best {
                best = dt;
            }
            chk = acc;
        }
        println!(
            "{name:>14}: {:8.0} ns/doc  {:6.1} GMAC/s  chk={chk:.3}",
            best * 1e9 / NDOCS as f64,
            macs / best / 1e9
        );
    };

    arm("scalar", &|d| {
        Some(maxsim_residual_lut_scalar(
            &q8, &d.packed.view(), &d.codes, &cdot_t.view(), &lut, &d.inv, DIM,
        ))
    });
    arm("avx2-maddubs", &|d| {
        maxsim_residual_lut_i8_force_avx2(
            &q8, &planes, &d.packed.view(), &d.codes, &cdot_t.view(), &lut, &d.inv, DIM,
        )
    });
    arm("avx512-vnni", &|d| {
        maxsim_residual_lut_i8_force_avx512(
            &q8, &planes, &d.packed.view(), &d.codes, &cdot_t.view(), &lut, &d.inv, DIM,
        )
    });
    arm("neon-sdot", &|d| {
        maxsim_residual_lut_i8_force_neon(
            &q8, &planes, &d.packed.view(), &d.codes, &cdot_t.view(), &lut, &d.inv, DIM,
        )
    });
    arm("dispatch", &|d| {
        Some(maxsim_residual_lut_i8(
            &q8, Some(&planes), &d.packed.view(), &d.codes, &cdot_t.view(), &lut, &d.inv, DIM,
        ))
    });
}
