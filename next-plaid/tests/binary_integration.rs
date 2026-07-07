//! End-to-end tests for asymmetric binary quantization: build a binary index,
//! confirm the on-disk document store shrinks, and confirm the asymmetric binary
//! MaxSim scoring path still retrieves the right documents.

use ndarray::{Array2, Axis};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::StandardNormal;
use ndarray_rand::RandomExt;
use next_plaid::index::MmapIndex;
use next_plaid::{binary, IndexConfig, SearchParameters};
use rand::rngs::StdRng;
use tempfile::TempDir;

/// Distinct L2-normalized documents so self-retrieval is well defined.
fn random_docs(num_docs: usize, tokens: usize, dim: usize) -> Vec<Array2<f32>> {
    let mut rng = StdRng::seed_from_u64(7);
    (0..num_docs)
        .map(|_| {
            let mut emb: Array2<f32> =
                Array2::random_using((tokens, dim), StandardNormal, &mut rng);
            for mut row in emb.axis_iter_mut(Axis(0)) {
                let norm = row.dot(&row).sqrt().max(1e-12);
                row /= norm;
            }
            emb
        })
        .collect()
}

fn config(binary: bool) -> IndexConfig {
    IndexConfig {
        nbits: 4,
        batch_size: 64,
        seed: Some(42),
        binary,
        ..Default::default()
    }
}

fn params() -> SearchParameters {
    SearchParameters {
        top_k: 3,
        n_ivf_probe: 16,
        ..Default::default()
    }
}

#[test]
fn binary_index_stores_one_bit_per_dimension() {
    let (dim, nbits) = (64usize, 4usize);
    let docs = random_docs(40, 8, dim);

    let float_dir = TempDir::new().unwrap();
    let bin_dir = TempDir::new().unwrap();
    let float_ix =
        MmapIndex::create_with_kmeans(&docs, float_dir.path().to_str().unwrap(), &config(false))
            .unwrap();
    let bin_ix =
        MmapIndex::create_with_kmeans(&docs, bin_dir.path().to_str().unwrap(), &config(true))
            .unwrap();

    // Float stores dim*nbits/8 bytes/token; binary stores ceil(dim/8).
    assert!(!float_ix.metadata.binary);
    assert!(bin_ix.metadata.binary);
    assert_eq!(float_ix.mmap_residuals.ncols(), dim * nbits / 8); // 32
    assert_eq!(bin_ix.mmap_residuals.ncols(), binary::packed_dim(dim)); // 8

    // The document store is 4x narrower here (and 32x versus raw f32).
    assert_eq!(
        float_ix.mmap_residuals.ncols() / bin_ix.mmap_residuals.ncols(),
        nbits
    );
}

#[test]
fn binary_index_retrieves_the_query_document() {
    let docs = random_docs(50, 8, 64);
    let dir = TempDir::new().unwrap();
    let index =
        MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &config(true)).unwrap();

    // Use each document's own tokens as the query; the document itself should
    // rank first under asymmetric binary MaxSim (q is dotted with sign(q)).
    let mut hits = 0;
    for (doc_id, doc) in docs.iter().enumerate() {
        let result = index.search(doc, &params(), None).unwrap();
        if result.passage_ids.first() == Some(&(doc_id as i64)) {
            hits += 1;
        }
    }
    let recall_at_1 = hits as f32 / docs.len() as f32;
    assert!(recall_at_1 >= 0.9, "binary recall@1 too low: {recall_at_1}");
}

#[test]
fn binary_reconstruct_returns_signs_not_garbage() {
    let docs = random_docs(20, 8, 64);
    let dir = TempDir::new().unwrap();
    let index =
        MmapIndex::create_with_kmeans(&docs, dir.path().to_str().unwrap(), &config(true)).unwrap();

    // reconstruct() on a binary index must decode the stored ±1 signs, matching
    // sign(original), not misread them as residual codes.
    let recon = index.reconstruct(&[3]).unwrap();
    let doc = &recon[0];
    assert_eq!(doc.dim(), docs[3].dim());
    for (got, orig) in doc.iter().zip(docs[3].iter()) {
        assert_eq!(*got, if *orig >= 0.0 { 1.0 } else { -1.0 });
    }
}

#[test]
fn binary_index_rejects_incremental_update() {
    // update_index re-encodes through the residual codec; appending residual
    // rows to a 1-bit sign store would corrupt it and flip metadata.binary
    // off. Every update entry point must refuse — update_append reached
    // update_index unguarded once.
    let dim = 64usize;
    let docs = random_docs(20, 8, dim);
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_str().unwrap();
    MmapIndex::create_with_kmeans(&docs, path, &config(true)).unwrap();

    let more = random_docs(4, 8, dim);
    let err = MmapIndex::update_append(&more, path, &next_plaid::update::UpdateConfig::default())
        .unwrap_err();
    assert!(
        err.to_string().contains("binary"),
        "expected binary-index rejection, got: {err}"
    );

    // The failed attempt must not have flipped the on-disk flag.
    let reloaded = MmapIndex::load(path).unwrap();
    assert!(reloaded.metadata.binary);
}
