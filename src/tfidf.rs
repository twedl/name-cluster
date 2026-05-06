//! Sparse char-n-gram TF-IDF + cosine for rerank.
//!
//! Pipeline role: after MinHash-LSH emits candidate pairs (high-recall, may
//! include false positives), TF-IDF cosine reranks each candidate to decide
//! whether to keep the edge. IDF down-weights common n-grams (legal-form
//! suffixes, frequent letter trigrams) so the cosine score reflects the
//! distinctive content shared between two names — not the boilerplate.
//!
//! Storage: per-vector `Vec<(u32, f32)>` sorted by ID, L2-normalized at build
//! time so cosine(a, b) is just a sorted-merge dot product. At ~30 trigrams
//! per normalized name, ~12 bytes per entry: ~360 B per vector → 3.6 GB at
//! 10M names. Acceptable inside our memory budget.
//!
//! Determinism: vocabulary IDs are assigned in sorted-by-bytes order, IDF
//! is a pure function of doc frequency, and cosine sums are computed in
//! sorted-ID order. Bit-exact across runs and machines.

use crate::ngram::{distinct_ngrams, ngrams};
use ahash::AHashMap;

#[derive(Clone)]
pub struct Vocabulary {
    n: usize,
    ids: AHashMap<Box<[u8]>, u32>,
    idf: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SparseVector {
    /// Sorted by ID ascending; weights are L2-normalized.
    entries: Vec<(u32, f32)>,
}

#[cfg(test)]
impl SparseVector {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// L2 norm of the vector. Always 1.0 for non-empty vectors built by
    /// `Vocabulary::vectorize`; 0.0 for empty.
    pub fn l2_norm(&self) -> f32 {
        self.entries.iter().map(|(_, w)| w * w).sum::<f32>().sqrt()
    }
}

impl Vocabulary {
    /// Build vocabulary + IDF table from an iterator of normalized names.
    /// Each name contributes to a single document's frequency count
    /// (n-gram repeats within a name don't inflate document frequency).
    pub fn build<I, S>(names: I, n: usize) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        assert!(n > 0, "n must be > 0");

        let mut df: AHashMap<Box<[u8]>, u32> = AHashMap::new();
        let mut n_docs: u32 = 0;
        for name in names {
            n_docs += 1;
            for g in distinct_ngrams(name.as_ref(), n) {
                *df.entry(g.to_vec().into_boxed_slice()).or_insert(0) += 1;
            }
        }

        // Sort vocab keys for deterministic ID assignment.
        let mut pairs: Vec<(Box<[u8]>, u32)> = df.into_iter().collect();
        pairs.sort_unstable_by(|a, b| a.0.cmp(&b.0));

        let n_docs_f = n_docs as f32;
        let mut ids = AHashMap::with_capacity(pairs.len());
        let mut idf = Vec::with_capacity(pairs.len());
        for (id, (gram, df_count)) in pairs.into_iter().enumerate() {
            ids.insert(gram, id as u32);
            idf.push((n_docs_f / df_count as f32).ln());
        }
        Self { n, ids, idf }
    }

    #[cfg(test)]
    pub fn vocab_size(&self) -> usize {
        self.idf.len()
    }

    /// IDF weight for one n-gram, or `None` if the n-gram is not in vocab.
    #[cfg(test)]
    pub fn idf_for(&self, gram: &[u8]) -> Option<f32> {
        self.ids.get(gram).map(|&id| self.idf[id as usize])
    }

    /// Vectorize one name. N-grams not in the vocabulary are silently dropped
    /// (they have no IDF — appropriate when scoring a query against an
    /// already-built corpus). Output is sorted by ID and L2-normalized.
    pub fn vectorize(&self, name: &str) -> SparseVector {
        let mut tf: AHashMap<u32, u32> = AHashMap::new();
        for g in ngrams(name, self.n) {
            if let Some(&id) = self.ids.get(g) {
                *tf.entry(id).or_insert(0) += 1;
            }
        }
        let mut entries: Vec<(u32, f32)> = tf
            .into_iter()
            .map(|(id, count)| (id, count as f32 * self.idf[id as usize]))
            .collect();
        entries.sort_unstable_by_key(|&(id, _)| id);

        let norm: f32 = entries.iter().map(|(_, w)| w * w).sum::<f32>().sqrt();
        if norm > 0.0 {
            for (_, w) in entries.iter_mut() {
                *w /= norm;
            }
        }
        SparseVector { entries }
    }
}

/// Cosine similarity for two L2-normalized SparseVectors. With both already
/// unit-normalized, cosine = dot product (no division at scoring time).
pub fn cosine(a: &SparseVector, b: &SparseVector) -> f32 {
    let mut sum = 0.0f32;
    let (mut i, mut j) = (0usize, 0usize);
    let (av, bv) = (&a.entries, &b.entries);
    while i < av.len() && j < bv.len() {
        let (id_a, w_a) = av[i];
        let (id_b, w_b) = bv[j];
        match id_a.cmp(&id_b) {
            std::cmp::Ordering::Equal => {
                sum += w_a * w_b;
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(names: &[&str], n: usize) -> Vocabulary {
        Vocabulary::build(names.iter().copied(), n)
    }

    #[test]
    fn identical_inputs_cosine_one() {
        let v = vocab(&["acme corp", "acme corporation", "acme inc"], 3);
        let a = v.vectorize("acme corp");
        let b = v.vectorize("acme corp");
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn fully_disjoint_cosine_zero() {
        let v = vocab(&["aaa bbb", "xxx yyy"], 3);
        let a = v.vectorize("aaa bbb");
        let b = v.vectorize("xxx yyy");
        assert_eq!(cosine(&a, &b), 0.0);
    }

    #[test]
    fn known_overlap_cosine_in_range() {
        let v = vocab(
            &[
                "acme corporation",
                "acme corporations",
                "acme inc",
                "other thing",
            ],
            3,
        );
        let a = v.vectorize("acme corporation");
        let b = v.vectorize("acme corporations");
        let c = cosine(&a, &b);
        assert!(
            c > 0.7 && c < 1.0,
            "near-duplicate cosine {} should be 0.7..1.0",
            c
        );
    }

    #[test]
    fn idf_downweights_common_grams() {
        // "abc" is in every doc (df=N → IDF=0); "xyz" only in one (IDF=ln(N)).
        let docs = ["abc def", "abc ghi", "abc jkl", "abc xyz"];
        let v = Vocabulary::build(docs.iter().copied(), 3);

        assert_eq!(
            v.idf_for(b"abc"),
            Some(0.0),
            "common n-gram should have IDF 0"
        );
        let xyz = v.idf_for(b"xyz").expect("xyz should be in vocab");
        assert!(xyz > 1.0, "rare n-gram should have IDF > 1, got {}", xyz);
    }

    #[test]
    fn deterministic_vocab_ids() {
        // Same docs in different order -> same vocab + same IDF table
        // (vocab IDs assigned sorted-by-bytes regardless of input order).
        let v1 = vocab(&["abc def", "ghi jkl", "mno pqr"], 3);
        let v2 = vocab(&["mno pqr", "abc def", "ghi jkl"], 3);
        assert_eq!(v1.idf, v2.idf);
        assert_eq!(v1.vocab_size(), v2.vocab_size());
        // Spot-check a few specific n-grams across both vocabs
        for gram in [b"abc".as_ref(), b"def".as_ref(), b"mno".as_ref()] {
            assert_eq!(v1.idf_for(gram), v2.idf_for(gram));
        }
    }

    #[test]
    fn out_of_vocab_grams_dropped() {
        let v = vocab(&["acme corp"], 3);
        // "xyz" trigrams aren't in vocab — vector should ignore them
        let vec = v.vectorize("xyz xyz xyz");
        assert!(vec.is_empty());
    }

    #[test]
    fn empty_vector_cosine_zero() {
        let v = vocab(&["acme corp"], 3);
        let empty = v.vectorize("?!"); // no valid n-grams (single non-ascii)
        let nonempty = v.vectorize("acme corp");
        assert_eq!(cosine(&empty, &nonempty), 0.0);
    }

    #[test]
    fn vectorize_is_l2_normalized() {
        let v = vocab(
            &["acme corporation", "acme corp ltd", "other thing here"],
            3,
        );
        let vec = v.vectorize("acme corporation");
        assert!(
            (vec.l2_norm() - 1.0).abs() < 1e-5,
            "L2 norm {} should be 1.0",
            vec.l2_norm()
        );
    }

    #[test]
    fn cosine_symmetric() {
        let v = vocab(&["acme corp", "beta inc", "gamma ltd"], 3);
        let a = v.vectorize("acme corp");
        let b = v.vectorize("beta inc");
        assert_eq!(cosine(&a, &b), cosine(&b, &a));
    }
}
