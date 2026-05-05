//! Char-n-gram extraction.
//!
//! Operates on byte slices. After `normalize()` the input is guaranteed
//! ASCII (`[a-z0-9 ]` only), so byte n-grams ARE char n-grams — no UTF-8
//! decoding needed in the hot loop.
//!
//! Space-containing n-grams (e.g. `"e c"` from `"acme corp"`) capture word
//! boundaries; downstream TF-IDF weighting separates "ibm" alone from
//! "ibm" embedded in a longer name via IDF on the boundary trigrams.

use std::collections::HashSet;

/// Yield all overlapping n-grams of byte length `n` from `text`.
/// Empty iterator when `text.len() < n` or `n == 0`.
pub fn ngrams(text: &str, n: usize) -> impl Iterator<Item = &[u8]> + '_ {
    // `slice::windows(0)` panics, so when n=0 we pass an empty slice with
    // n=1 to get a safe empty iterator out.
    let bytes = if n == 0 { &[][..] } else { text.as_bytes() };
    bytes.windows(n.max(1))
}

/// Yield distinct n-grams (set semantics). Order unspecified.
pub fn distinct_ngrams(text: &str, n: usize) -> impl Iterator<Item = &[u8]> {
    ngrams(text, n).collect::<HashSet<_>>().into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect(text: &str, n: usize) -> Vec<&[u8]> {
        ngrams(text, n).collect()
    }

    #[test]
    fn basic() {
        let g = collect("acme corp", 3);
        let s: Vec<&str> = g.iter().map(|b| std::str::from_utf8(b).unwrap()).collect();
        assert_eq!(s, vec!["acm", "cme", "me ", "e c", " co", "cor", "orp"]);
    }

    #[test]
    fn shorter_than_n() {
        assert_eq!(collect("ab", 3).len(), 0);
        assert_eq!(collect("", 3).len(), 0);
    }

    #[test]
    fn equal_to_n() {
        assert_eq!(collect("abc", 3), vec![b"abc"]);
    }

    #[test]
    fn n_zero() {
        assert_eq!(collect("acme", 0).len(), 0);
    }

    #[test]
    fn distinct_dedups_repeats() {
        let d: Vec<&[u8]> = distinct_ngrams("abababab", 3).collect();
        assert_eq!(d.len(), 2);
    }

    #[test]
    fn distinct_unique_text() {
        let all: Vec<_> = ngrams("abcdef", 3).collect();
        let dist: Vec<_> = distinct_ngrams("abcdef", 3).collect();
        assert_eq!(all.len(), dist.len());
    }
}
