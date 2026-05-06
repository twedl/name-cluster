//! MinHash signatures via universal hashing.
//!
//! For a set S of items, the MinHash signature is `K` minimum hash values
//! computed under K independent hash functions. Two sets have estimated
//! Jaccard similarity equal to the fraction of equal positions in their
//! signatures.
//!
//! Implementation: hash each item ONCE with a deterministically-seeded
//! `ahash::RandomState`, then derive K permutations via universal hashing
//!
//!   h_i(x) = ((a_i * x + b_i) mod p)
//!
//! where p = 2^61 - 1 (a Mersenne prime; reduction uses the identity
//! `2^61 ≡ 1 (mod p)` for two cheap shift+add passes). This gives the
//! same Jaccard-estimation guarantees as K independent hashes, but with
//! K multiply-adds per item instead of K full hash computations —
//! typically 3-5× faster.
//!
//! Determinism: the `ahash::RandomState` is seeded from explicit u64
//! coefficients, NOT from `AHasher::default()` (which uses runtime-random
//! getrandom keys and is NOT deterministic across program runs). Same
//! `seed` + same input set → byte-identical signature on any machine.

use ahash::RandomState;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::hash::{BuildHasher, Hasher};

const MERSENNE_P: u64 = (1u64 << 61) - 1;

#[derive(Clone)]
pub struct MinHasher {
    a: Vec<u64>,
    b: Vec<u64>,
    random_state: RandomState,
}

impl MinHasher {
    /// Build a hasher producing `num_perm`-length signatures.
    pub fn new(num_perm: usize, seed: u64) -> Self {
        assert!(num_perm > 0, "num_perm must be > 0");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut a = Vec::with_capacity(num_perm);
        let mut b = Vec::with_capacity(num_perm);
        for _ in 0..num_perm {
            // a must be non-zero for the permutation to be a bijection mod p.
            a.push((rng.next_u64() % (MERSENNE_P - 1)) + 1);
            b.push(rng.next_u64() % MERSENNE_P);
        }
        let random_state = RandomState::with_seeds(
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        );
        Self { a, b, random_state }
    }

    pub fn signature<I, T>(&self, items: I) -> Vec<u64>
    where
        I: IntoIterator<Item = T>,
        T: AsRef<[u8]>,
    {
        let mut sig = vec![u64::MAX; self.a.len()];
        self.signature_into(items, &mut sig);
        sig
    }

    /// Buffer-reusing variant. `out` is resized to `num_perm` and refilled.
    /// Lets callers (or future rayon workers) avoid one Vec allocation per name.
    pub fn signature_into<I, T>(&self, items: I, out: &mut Vec<u64>)
    where
        I: IntoIterator<Item = T>,
        T: AsRef<[u8]>,
    {
        out.clear();
        out.resize(self.a.len(), u64::MAX);
        for item in items {
            let h = self.base_hash(item.as_ref());
            for ((slot, a_i), b_i) in out.iter_mut().zip(&self.a).zip(&self.b) {
                let hi = mersenne_mul_add(*a_i, h, *b_i);
                if hi < *slot {
                    *slot = hi;
                }
            }
        }
    }

    fn base_hash(&self, bytes: &[u8]) -> u64 {
        let mut h = self.random_state.build_hasher();
        h.write(bytes);
        h.finish()
    }
}

#[inline]
fn mersenne_mul_add(a: u64, x: u64, b: u64) -> u64 {
    const P: u128 = MERSENNE_P as u128;
    let prod = (a as u128).wrapping_mul(x as u128);
    let mut r = prod.wrapping_add(b as u128);
    r = (r >> 61) + (r & P);
    r = (r >> 61) + (r & P);
    if r >= P {
        r -= P;
    }
    r as u64
}

#[cfg(test)]
pub fn estimate_jaccard(sig_a: &[u64], sig_b: &[u64]) -> f64 {
    assert_eq!(sig_a.len(), sig_b.len(), "signature length mismatch");
    if sig_a.is_empty() {
        return 0.0;
    }
    let eq = sig_a
        .iter()
        .zip(sig_b.iter())
        .filter(|(a, b)| a == b)
        .count();
    eq as f64 / sig_a.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ngrams(text: &str, n: usize) -> Vec<&[u8]> {
        crate::ngram::ngrams(text, n).collect()
    }

    #[test]
    fn deterministic_within_run() {
        let h = MinHasher::new(64, 42);
        let s1 = h.signature(ngrams("acme corporation", 3));
        let s2 = h.signature(ngrams("acme corporation", 3));
        assert_eq!(s1, s2);
    }

    #[test]
    fn signature_length_matches_num_perm() {
        let h = MinHasher::new(128, 0);
        assert_eq!(h.signature(ngrams("test name", 3)).len(), 128);
    }

    #[test]
    fn identical_inputs_jaccard_one() {
        let h = MinHasher::new(128, 0);
        let s = h.signature(ngrams("acme corp ltd", 3));
        assert_eq!(estimate_jaccard(&s, &s), 1.0);
    }

    #[test]
    fn disjoint_inputs_jaccard_low() {
        let h = MinHasher::new(128, 0);
        let s1 = h.signature(ngrams("aaaa bbbb", 3));
        let s2 = h.signature(ngrams("xxxx yyyy", 3));
        let j = estimate_jaccard(&s1, &s2);
        assert!(j < 0.1, "disjoint sets should have low Jaccard, got {}", j);
    }

    #[test]
    fn estimate_close_to_true_jaccard() {
        // A trigrams of "abcdefgh" (6) ∩ B trigrams of "abcdefij" (6) = "abc","bcd","cde","def" (4)
        // True Jaccard = 4/8 = 0.5
        let h = MinHasher::new(256, 7);
        let s_a = h.signature(ngrams("abcdefgh", 3));
        let s_b = h.signature(ngrams("abcdefij", 3));
        let j = estimate_jaccard(&s_a, &s_b);
        assert!((j - 0.5).abs() < 0.1, "estimated {} too far from 0.5", j);
    }

    #[test]
    fn duplicates_dont_affect_signature() {
        let h = MinHasher::new(64, 0);
        let s_a = h.signature(vec![b"abc", b"def", b"ghi"]);
        let s_b = h.signature(vec![b"def", b"ghi", b"abc", b"abc", b"def"]);
        assert_eq!(s_a, s_b);
    }

    #[test]
    fn different_seed_different_signature() {
        let s1 = MinHasher::new(64, 0).signature(ngrams("acme", 3));
        let s2 = MinHasher::new(64, 1).signature(ngrams("acme", 3));
        assert_ne!(s1, s2);
    }

    #[test]
    fn signature_into_matches_signature() {
        let h = MinHasher::new(64, 99);
        let items = ngrams("foothill industries", 3);
        let direct = h.signature(items.clone());
        let mut buf = Vec::new();
        h.signature_into(items, &mut buf);
        assert_eq!(direct, buf);
        // Reuse: refill should overwrite, not append
        h.signature_into(ngrams("foothill industries", 3), &mut buf);
        assert_eq!(buf.len(), 64);
    }

    #[test]
    fn mersenne_reduction_bounds() {
        for &(a, x, b) in &[
            (1u64, 0u64, 0u64),
            (MERSENNE_P, MERSENNE_P, MERSENNE_P),
            (u64::MAX, u64::MAX, u64::MAX),
            (12345, 67890, 13579),
        ] {
            let r = mersenne_mul_add(a, x, b);
            assert!(r < MERSENNE_P, "result {} >= p {}", r, MERSENNE_P);
        }
    }
}
