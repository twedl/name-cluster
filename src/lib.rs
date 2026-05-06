//! `name_cluster` rust core.
//!
//! Public surface is exposed via PyO3 in `_lowlevel`:
//!   - `normalize(name)` — single-name normalization
//!   - `cluster_lists(names, **opts)` — full pipeline; takes/returns Python
//!     lists (the Python wrapper layer plugs this into narwhals/Arrow).

// pyo3 0.22's `#[pyfunction]` macro emits a `.into()` for the Err branch of
// `PyResult` that clippy reads as a same-type round-trip. Quiet at module level.
#![allow(clippy::useless_conversion)]

use ahash::AHashMap;
use pyo3::prelude::*;
use std::collections::HashMap;

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

/// Normalize each (canonical, [alias]) pair and flatten to alias_norm ->
/// canon_norm. Pairs whose normalized form is empty are dropped silently.
/// Iteration is sorted by canonical so duplicate-alias behavior (last wins)
/// is deterministic across runs regardless of input dict order.
fn build_alias_map(raw: Option<HashMap<String, Vec<String>>>) -> AHashMap<String, String> {
    let mut out = AHashMap::new();
    let Some(d) = raw else {
        return out;
    };
    let mut entries: Vec<(String, Vec<String>)> = d.into_iter().collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (canon_raw, alias_list) in entries {
        let canon_norm = normalize::normalize(&canon_raw);
        if canon_norm.is_empty() {
            continue;
        }
        for alias_raw in alias_list {
            let alias_norm = normalize::normalize(&alias_raw);
            if alias_norm.is_empty() {
                continue;
            }
            out.insert(alias_norm, canon_norm.clone());
        }
    }
    out
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
    aliases = None,
    n_threads = None,
    return_canonical = true,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn py_cluster_lists(
    py: Python<'_>,
    names: Vec<Option<String>>,
    threshold: f32,
    seed: u64,
    ngram_size: usize,
    lsh_bands: usize,
    lsh_rows: usize,
    hub_radius_max: usize,
    diameter_check_min_size: usize,
    max_name_length: usize,
    aliases: Option<HashMap<String, Vec<String>>>,
    n_threads: Option<usize>,
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
        aliases: build_alias_map(aliases),
        n_threads,
    };
    let r = py.allow_threads(|| {
        let mut b = builder::ClusterBuilder::new(opts);
        // add_batch consumes `names` and runs Phase A/C of insertion in parallel.
        b.add_batch(names);
        b.finalize()
    });
    let canonical = if return_canonical {
        r.canonical
    } else {
        Vec::new()
    };
    Ok((r.cluster_ids, canonical, r.flagged_cluster_ids))
}

/// Run the pipeline through TF-IDF rerank only. Returns scored candidate
/// pairs over UNIQUE normalized names. Used by the public `candidates()`
/// debug API for threshold tuning. No clustering, no canonical assignment.
///
/// Returns `(idx_a, idx_b, score, unique_normalized, original_to_unique)`
/// where idx_{a,b} index into `unique_normalized`. Pairs are filtered to
/// `score >= min_score` (default 0.0 = all candidates).
#[pyfunction(name = "candidate_pairs_lists")]
#[pyo3(signature = (
    names,
    *,
    min_score = 0.0,
    seed = 0,
    ngram_size = 3,
    lsh_bands = 32,
    lsh_rows = 4,
    max_name_length = 256,
    aliases = None,
    n_threads = None,
))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn py_candidate_pairs_lists(
    py: Python<'_>,
    names: Vec<Option<String>>,
    min_score: f32,
    seed: u64,
    ngram_size: usize,
    lsh_bands: usize,
    lsh_rows: usize,
    max_name_length: usize,
    aliases: Option<HashMap<String, Vec<String>>>,
    n_threads: Option<usize>,
) -> PyResult<(Vec<u32>, Vec<u32>, Vec<f32>, Vec<String>, Vec<Option<u32>>)> {
    if !(0.0..=1.0).contains(&min_score) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "min_score must be in [0.0, 1.0], got {}",
            min_score
        )));
    }
    if ngram_size == 0 || lsh_bands == 0 || lsh_rows == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ngram_size, lsh_bands, lsh_rows must all be >= 1",
        ));
    }
    let opts = builder::ClusterOpts {
        threshold: 0.0,
        seed,
        ngram_size,
        lsh_bands,
        lsh_rows,
        hub_radius_max: 2,
        diameter_check_min_size: 5,
        max_name_length,
        aliases: build_alias_map(aliases),
        n_threads,
    };
    let r = py.allow_threads(|| {
        let mut b = builder::ClusterBuilder::new(opts);
        b.add_batch(names);
        b.candidate_pairs(min_score)
    });
    let mut idx_a = Vec::with_capacity(r.scored_pairs.len());
    let mut idx_b = Vec::with_capacity(r.scored_pairs.len());
    let mut scores = Vec::with_capacity(r.scored_pairs.len());
    for (a, b, s) in r.scored_pairs {
        idx_a.push(a);
        idx_b.push(b);
        scores.push(s);
    }
    Ok((
        idx_a,
        idx_b,
        scores,
        r.unique_normalized,
        r.original_to_unique,
    ))
}

/// Pairwise cosine over a small ad-hoc list of names. Used by `explain()`
/// to score edges within a single cluster on demand. No LSH blocking — all
/// O(n²) pairs returned.
#[pyfunction(name = "pairwise_cosines_lists")]
#[pyo3(signature = (names, *, ngram_size = 3))]
fn py_pairwise_cosines_lists(
    names: Vec<String>,
    ngram_size: usize,
) -> PyResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    if ngram_size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "ngram_size must be >= 1",
        ));
    }
    let pairs = builder::pairwise_cosines(&names, ngram_size);
    let mut a = Vec::with_capacity(pairs.len());
    let mut b = Vec::with_capacity(pairs.len());
    let mut s = Vec::with_capacity(pairs.len());
    for (i, j, sc) in pairs {
        a.push(i);
        b.push(j);
        s.push(sc);
    }
    Ok((a, b, s))
}

#[pymodule]
fn _lowlevel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(py_cluster_lists, m)?)?;
    m.add_function(wrap_pyfunction!(py_candidate_pairs_lists, m)?)?;
    m.add_function(wrap_pyfunction!(py_pairwise_cosines_lists, m)?)?;
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
        "Acme Corporation",                    // 0
        "ACME Corp",                           // 1
        "Acme Corporation Inc",                // 2
        "International Business Machines",     // 3
        "IBM",                                 // 4
        "INTERNATIONAL BUSINESS MACHINES INC", // 5
        "Sherwin-Williams Co",                 // 6
        "Sherwin Williams Company",            // 7
        "The Sherwin-Williams Co",             // 8
        "Foothill Industries",                 // 9
        "Brightspoke",                         // 10
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
                "Acme pair {:?} should rerank above 0.7",
                p
            );
        }
        // At least one Sherwin-Williams pair should rerank high
        let sherwin = [(6u32, 7u32), (6, 8), (7, 8)];
        assert!(
            sherwin
                .iter()
                .any(|p| high_score_pairs.iter().any(|(a, b, _)| (*a, *b) == *p)),
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
        assert_eq!(
            r.cluster_ids
                .iter()
                .filter(|c| **c == Some(foothill_id))
                .count(),
            1
        );
        assert_eq!(
            r.cluster_ids
                .iter()
                .filter(|c| **c == Some(brightspoke_id))
                .count(),
            1
        );

        assert_ne!(acme_id, r.cluster_ids[3].unwrap());
        assert!(r.canonical.iter().all(|c| c.is_some()));
        assert!(r.flagged_cluster_ids.is_empty());
    }
}
