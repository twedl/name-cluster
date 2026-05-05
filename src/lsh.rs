//! Banded LSH bucketing on MinHash signatures.
//!
//! Splits each signature into `bands` slices of `rows` consecutive values.
//! Items whose band-slice hashes match (in any one of the bands) become
//! candidate pairs. Tunable for any target Jaccard threshold via the
//! classic LSH formula `P(candidate | jaccard=s) = 1 - (1 - s^r)^b`.
//!
//! Memory: `bands × n_items` bucket entries (HashMap<key, Vec<idx>>).
//! At default 32×4 with 10M items: ~320M entries, ~10-15 GB peak. The
//! LSH index is the dominant memory consumer in the pipeline.
//!
//! Determinism: bucket-key hashing uses `ahash::RandomState` with explicit
//! u64 seeds derived from construction `seed`, NOT `AHasher::default()`
//! (which is randomly seeded per process and breaks cross-machine
//! determinism). Candidate-pair iteration sorts buckets and within-bucket
//! items, giving deterministic output order.

use ahash::RandomState;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};

#[derive(Clone)]
pub struct LshIndex {
    bands: usize,
    rows: usize,
    buckets: HashMap<u64, Vec<u32>>,
    random_state: RandomState,
}

impl LshIndex {
    pub fn new(bands: usize, rows: usize, seed: u64) -> Self {
        assert!(bands > 0 && rows > 0);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let random_state = RandomState::with_seeds(
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        );
        Self {
            bands,
            rows,
            buckets: HashMap::new(),
            random_state,
        }
    }

    /// Insert one signature, indexed by `item_idx`.
    pub fn insert(&mut self, item_idx: u32, sig: &[u64]) {
        assert_eq!(
            sig.len(),
            self.bands * self.rows,
            "signature length must equal bands*rows"
        );
        for b in 0..self.bands {
            let band_slice = &sig[b * self.rows..(b + 1) * self.rows];
            let key = self.bucket_key(b, band_slice);
            self.buckets.entry(key).or_default().push(item_idx);
        }
    }

    /// Distinct candidate pairs, sorted ascending — `(min_idx, max_idx)`.
    /// Pairs that collide on multiple bands appear exactly once. Memory
    /// peak: `O(distinct-pair-count)` HashSet, smaller than the raw
    /// (bucket × pair) cross-product when band-collisions are dense.
    pub fn candidate_pairs_distinct(&self) -> Vec<(u32, u32)> {
        let mut seen: HashSet<(u32, u32)> = HashSet::new();
        for items in self.buckets.values() {
            let mut sorted = items.clone();
            sorted.sort_unstable();
            sorted.dedup();
            for i in 0..sorted.len() {
                for j in (i + 1)..sorted.len() {
                    seen.insert((sorted[i], sorted[j]));
                }
            }
        }
        let mut out: Vec<(u32, u32)> = seen.into_iter().collect();
        out.sort_unstable();
        out
    }

    fn bucket_key(&self, band_idx: usize, band_slice: &[u64]) -> u64 {
        let mut h = self.random_state.build_hasher();
        h.write_usize(band_idx);
        for &v in band_slice {
            h.write_u64(v);
        }
        h.finish()
    }
}

/// Probability that two items with true Jaccard `s` share at least one band.
/// Useful for tuning bands/rows against a target recall.
pub fn lsh_collision_prob(jaccard: f64, bands: usize, rows: usize) -> f64 {
    1.0 - (1.0 - jaccard.powi(rows as i32)).powi(bands as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minhash::MinHasher;
    use crate::ngram::ngrams;

    #[test]
    fn identical_items_collide() {
        let h = MinHasher::new(32, 0);
        let mut idx = LshIndex::new(8, 4, 0);
        let sig = h.signature(ngrams("acme corporation", 3));
        idx.insert(0, &sig);
        idx.insert(1, &sig);
        assert_eq!(idx.candidate_pairs_distinct(), vec![(0, 1)]);
    }

    #[test]
    fn disjoint_items_dont_collide() {
        let h = MinHasher::new(32, 0);
        let mut idx = LshIndex::new(8, 4, 0);
        idx.insert(0, &h.signature(ngrams("aaaa bbbb cccc", 3)));
        idx.insert(1, &h.signature(ngrams("xxxx yyyy zzzz", 3)));
        assert!(idx.candidate_pairs_distinct().is_empty());
    }

    #[test]
    fn similar_items_likely_collide() {
        let h = MinHasher::new(128, 0);
        let mut idx = LshIndex::new(32, 4, 0);
        idx.insert(0, &h.signature(ngrams("acme corporation", 3)));
        idx.insert(1, &h.signature(ngrams("acme corporations", 3)));
        assert_eq!(idx.candidate_pairs_distinct(), vec![(0, 1)]);
    }

    #[test]
    fn deterministic_pair_order() {
        let h = MinHasher::new(32, 0);
        let mut idx = LshIndex::new(8, 4, 0);
        idx.insert(5, &h.signature(ngrams("acme corp", 3)));
        idx.insert(2, &h.signature(ngrams("acme corp", 3)));
        idx.insert(8, &h.signature(ngrams("acme corp", 3)));
        assert_eq!(idx.candidate_pairs_distinct(), vec![(2, 5), (2, 8), (5, 8)]);
    }

    #[test]
    fn signature_length_mismatch_panics() {
        let mut idx = LshIndex::new(8, 4, 0);
        let result = std::panic::catch_unwind(move || {
            idx.insert(0, &vec![0u64; 31]);
        });
        assert!(result.is_err());
    }

    #[test]
    fn collision_prob_formula() {
        let p = lsh_collision_prob(0.5, 32, 4);
        assert!((p - 0.85).abs() < 0.05, "expected ~0.85 at j=0.5 b=32 r=4, got {}", p);
        assert_eq!(lsh_collision_prob(0.0, 32, 4), 0.0);
        assert_eq!(lsh_collision_prob(1.0, 32, 4), 1.0);
    }
}
