//! `name_cluster` rust core.
//!
//! Public surface is exposed via PyO3 in `_lowlevel`:
//!   - `normalize(name)` — single-name normalization
//!   - `cluster_lists(names, **opts)` — full pipeline; takes/returns Python
//!     lists (the Python wrapper layer plugs this into narwhals/Arrow).

use pyo3::prelude::*;

mod builder;
mod cluster;
mod lsh;
mod minhash;
mod ngram;
mod normalize;
mod tfidf;

#[pyfunction(name = "normalize")]
fn py_normalize(name: &str) -> String {
    normalize::normalize(name)
}

/// Full clustering pipeline, list-based marshalling at the Py↔Rust boundary.
/// Returns `(cluster_ids, canonical, flagged_cluster_ids)`.
///
/// `names` is a list where each element is a non-empty str (clusters) or
/// `None` (skipped — receives `cluster_id=None, canonical=None` in output).
///
/// When `return_canonical=False` the canonical Vec is returned empty —
/// avoids 10M String allocs + Py conversions when callers (`cluster_names`)
/// only need cluster_ids.
#[pyfunction(name = "cluster_lists")]
#[pyo3(signature = (
    names,
    *,
    threshold = 0.85,
    seed = 0,
    ngram_size = 3,
    lsh_bands = 32,
    lsh_rows = 4,
    hub_radius_max = 2,
    diameter_check_min_size = 5,
    max_name_length = 256,
    return_canonical = true,
))]
#[allow(clippy::too_many_arguments)]
fn py_cluster_lists(
    names: Vec<Option<String>>,
    threshold: f32,
    seed: u64,
    ngram_size: usize,
    lsh_bands: usize,
    lsh_rows: usize,
    hub_radius_max: usize,
    diameter_check_min_size: usize,
    max_name_length: usize,
    return_canonical: bool,
) -> PyResult<(Vec<Option<u32>>, Vec<Option<String>>, Vec<u32>)> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "threshold must be in [0.0, 1.0], got {}",
            threshold
        )));
    }
    if ngram_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ngram_size must be >= 1",
        ));
    }
    if lsh_bands == 0 || lsh_rows == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "lsh_bands and lsh_rows must both be >= 1",
        ));
    }
    let opts = builder::ClusterOpts {
        threshold,
        seed,
        ngram_size,
        lsh_bands,
        lsh_rows,
        hub_radius_max,
        diameter_check_min_size,
        max_name_length,
    };
    let mut b = builder::ClusterBuilder::new(opts);
    // Consume `names` by value: the input Vec drops at end of loop scope,
    // before finalize() runs its big stages. Saves ~3 GB peak RSS at 10M.
    for name in names {
        b.add(name.as_deref());
    }
    let r = b.finalize();
    let canonical = if return_canonical { r.canonical } else { Vec::new() };
    Ok((r.cluster_ids, canonical, r.flagged_cluster_ids))
}

#[pymodule]
fn _lowlevel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(py_cluster_lists, m)?)?;
    Ok(())
}

#[cfg(test)]
mod pipeline_smoke {
    use super::{
        builder::{ClusterBuilder, ClusterOpts},
        lsh::LshIndex,
        minhash::MinHasher,
        ngram,
        normalize::normalize,
        tfidf::{cosine, Vocabulary},
    };
    use std::collections::HashSet;

    /// 11-name fixture with three real clusters + two singletons.
    /// IBM↔INTERNATIONAL BUSINESS MACHINES is intentionally a hard case
    /// (acronym/expansion: task #17) — should NOT pair via plain LSH.
    const FIXTURE: &[&str] = &[
        "Acme Corporation",                       // 0
        "ACME Corp",                              // 1
        "Acme Corporation Inc",                   // 2
        "International Business Machines",        // 3
        "IBM",                                    // 4
        "INTERNATIONAL BUSINESS MACHINES INC",    // 5
        "Sherwin-Williams Co",                    // 6
        "Sherwin Williams Company",               // 7
        "The Sherwin-Williams Co",                // 8
        "Foothill Industries",                    // 9
        "Brightspoke",                            // 10
    ];

    fn run_pipeline() -> (Vec<String>, Vec<(u32, u32)>) {
        let normalized: Vec<String> = FIXTURE.iter().map(|n| normalize(n)).collect();
        let hasher = MinHasher::new(128, 42);
        let mut idx = LshIndex::new(32, 4, 42);
        for (i, n) in normalized.iter().enumerate() {
            idx.insert(i as u32, &hasher.signature(ngram::ngrams(n, 3)));
        }
        (normalized, idx.candidate_pairs_distinct())
    }

    #[test]
    fn end_to_end_finds_intuitive_pairs() {
        let (_normalized, candidate_pairs) = run_pipeline();
        let pairs: HashSet<(u32, u32)> = candidate_pairs.into_iter().collect();

        // Acme: all 3 names normalize to "acme" -> all 3 pairs.
        assert!(pairs.contains(&(0, 1)));
        assert!(pairs.contains(&(0, 2)));
        assert!(pairs.contains(&(1, 2)));

        // Sherwin-Williams: at least one of the 3 pairs collides.
        let sherwin = [(6u32, 7u32), (6, 8), (7, 8)];
        assert!(
            sherwin.iter().any(|p| pairs.contains(p)),
            "no Sherwin-Williams pair collided"
        );

        // IBM acronym must NOT pair with the long form (acronym is task #17).
        assert!(!pairs.contains(&(3, 4)));

        // Foothill / Brightspoke have no real twin -> no pairs involving them.
        assert!(pairs.iter().all(|&(a, b)| a != 9 && b != 9));
        assert!(pairs.iter().all(|&(a, b)| a != 10 && b != 10));
    }

    #[test]
    fn end_to_end_with_tfidf_rerank() {
        let (normalized, pairs) = run_pipeline();

        // Build vocabulary across the normalized corpus + vectorize each.
        let vocab = Vocabulary::build(normalized.iter(), 3);
        let vectors: Vec<_> = normalized.iter().map(|n| vocab.vectorize(n)).collect();

        // Score each candidate pair; partition by intuitive truth.
        let mut high_score_pairs: Vec<(u32, u32, f32)> = Vec::new();
        let mut low_score_pairs: Vec<(u32, u32, f32)> = Vec::new();
        for &(a, b) in &pairs {
            let c = cosine(&vectors[a as usize], &vectors[b as usize]);
            if c >= 0.7 {
                high_score_pairs.push((a, b, c));
            } else {
                low_score_pairs.push((a, b, c));
            }
        }

        // The 3 Acme pairs all normalize to "acme" -> cosine = 1.0
        let acme_pairs = [(0u32, 1u32), (0, 2), (1, 2)];
        for p in &acme_pairs {
            assert!(
                high_score_pairs.iter().any(|(a, b, _)| (*a, *b) == *p),
                "Acme pair {:?} should rerank above 0.7", p
            );
        }
        // At least one Sherwin-Williams pair should rerank high
        let sherwin = [(6u32, 7u32), (6, 8), (7, 8)];
        assert!(
            sherwin.iter().any(|p| high_score_pairs.iter().any(|(a, b, _)| (*a, *b) == *p)),
            "no Sherwin-Williams pair scored above 0.7"
        );
    }

    #[test]
    fn cluster_builder_full_pipeline() {
        let mut b = ClusterBuilder::new(ClusterOpts::default());
        for n in FIXTURE {
            b.add(Some(n));
        }
        let r = b.finalize();
        assert_eq!(r.cluster_ids.len(), FIXTURE.len());

        let acme_id = r.cluster_ids[0].expect("acme should cluster");
        assert_eq!(r.cluster_ids[1], Some(acme_id));
        assert_eq!(r.cluster_ids[2], Some(acme_id));

        // Rows 3 + 5 both normalize to "intl business machines"; row 4 ("IBM")
        // stays alone — acronym↔expansion bridging is task #17.
        assert_eq!(r.cluster_ids[3], r.cluster_ids[5]);
        assert_ne!(r.cluster_ids[4], r.cluster_ids[3]);

        // Sherwin-Williams variants don't cluster at default threshold because
        // hyphen-drop produces 3 distinct normalized forms whose cosines don't
        // cross 0.85. Cosine threshold tuning is task #3, not a phase-4 claim.

        let foothill_id = r.cluster_ids[9].unwrap();
        let brightspoke_id = r.cluster_ids[10].unwrap();
        assert_eq!(r.cluster_ids.iter().filter(|c| **c == Some(foothill_id)).count(), 1);
        assert_eq!(r.cluster_ids.iter().filter(|c| **c == Some(brightspoke_id)).count(), 1);

        assert_ne!(acme_id, r.cluster_ids[3].unwrap());
        assert!(r.canonical.iter().all(|c| c.is_some()));
        assert!(r.flagged_cluster_ids.is_empty());
    }
}
