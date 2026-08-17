//! Worked example: ternary quantize -> base-3 pack -> decompress for ONE token
//! at dim = 7 (so we get two bytes: a full group of 5 trits + a padded group).
//! Run: cargo run -p next-plaid --release --example ternary_trace

use ndarray::{array, Array1, Array2};
use next_plaid::ResidualCodec;

fn main() {
    let dim = 7usize;

    // The token's centroid lives at centroids[0]; `codes = [0]` selects it.
    let c = [0.10f32, 0.20, 0.00, 0.50, -0.30, 0.05, -0.10];
    let mut centroids = Array2::<f32>::zeros((1, dim));
    for (j, &v) in c.iter().enumerate() {
        centroids[[0, j]] = v;
    }

    // Dead-zone codec: 2 cutoffs split the residual into 3 trits; 3 weights are
    // the reconstruction levels {-m, 0, +m}.
    let codec = ResidualCodec::new_ternary(
        2,
        centroids,
        Array1::zeros(dim),
        Some(array![-0.05f32, 0.05]),    // cutoffs
        Some(array![-0.1f32, 0.0, 0.1]), // weights = {-m, 0, +m}
    )
    .unwrap();

    // One token's residual (x - centroid), chosen near the ±m levels.
    let residual: Array2<f32> = array![[-0.09, 0.11, 0.00, -0.12, 0.08, -0.02, 0.10]];

    let packed = codec.quantize_residuals(&residual).unwrap();
    println!("residual      = {:?}", residual.row(0).to_vec());
    println!(
        "packed bytes  = {:?}   ({} bytes for {} dims)",
        packed.row(0).to_vec(),
        packed.ncols(),
        dim
    );

    let codes = Array1::from_vec(vec![0usize]);
    let recon = codec.decompress(&packed, &codes.view()).unwrap();
    println!("centroid      = {:?}", c);
    println!(
        "reconstructed = {:?}   (L2-normalized, the codec's last step)",
        recon
            .row(0)
            .iter()
            .map(|v| (v * 10000.0).round() / 10000.0)
            .collect::<Vec<_>>()
    );
}
