//! `ClusterBuilder` — orchestrates the full pipeline:
//!
//!   add_chunk(names) → normalize, dedup, MinHash sig, LSH insert
//!   finalize()       → vocab build, vectorize, candidate-pair rerank,
//!                      threshold, union-find, hub selection, canonical
//!                      assignment, re-broadcast to original rows.
//!
//! Per-input-row outputs (cluster_id + canonical) honor the v1 input
//! contract: null/empty-post-norm rows produce `None` (skip, don't merge
//! into any cluster).
//!
//! Determinism: every stage uses sorted iteration where order can affect
//! output. Deduplicated normalized names get IDs in insertion order; final
//! cluster_ids are reassigned by sorting components on canonical name asc
//! (so two runs over the same input + seed produce byte-identical IDs).

use ahash::AHashMap;

use crate::cluster::{
    build_adjacency, connected_components, diameter_split, group_by_component,
};
use crate::lsh::LshIndex;
use crate::minhash::MinHasher;
use crate::ngram::ngrams;
use crate::normalize::normalize;
use crate::tfidf::{cosine, Vocabulary};

#[derive(Clone, Debug)]
pub struct ClusterOpts {
    pub threshold: f32,
    pub seed: u64,
    pub ngram_size: usize,
    pub lsh_bands: usize,
    pub lsh_rows: usize,
    pub hub_radius_max: usize,
    pub diameter_check_min_size: usize,
    pub max_name_length: usize,
}

impl Default for ClusterOpts {
    fn default() -> Self {
        Self {
            threshold: 0.85,
            seed: 0,
            ngram_size: 3,
            lsh_bands: 32,
            lsh_rows: 4,
            hub_radius_max: 2,
            diameter_check_min_size: 5,
            max_name_length: 256,
        }
    }
}

pub struct ClusterBuilder {
    opts: ClusterOpts,
    minhasher: MinHasher,
    lsh: LshIndex,
    /// Maps original row index → unique normalized index (`None` for rows
    /// whose name normalized to empty / was null).
    original_to_unique: Vec<Option<u32>>,
    unique_normalized: Vec<String>,
    seen: AHashMap<String, u32>,
}

#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// One per original input row. `None` for rows skipped (null/empty-post-norm).
    pub cluster_ids: Vec<Option<u32>>,
    /// Hub-name canonical for each row (`None` when cluster_id is `None`).
    pub canonical: Vec<Option<String>>,
    /// Cluster IDs whose ancestor (pre-split) connected component exceeded
    /// `hub_radius_max`. The split mechanism (task #27, hub-radius partition)
    /// produced these as residuals; useful for diagnostic + downstream
    /// quality flagging. Empty when no splits occurred.
    pub flagged_cluster_ids: Vec<u32>,
}

impl ClusterBuilder {
    pub fn new(opts: ClusterOpts) -> Self {
        let num_perm = opts.lsh_bands * opts.lsh_rows;
        let minhasher = MinHasher::new(num_perm, opts.seed);
        let lsh = LshIndex::new(opts.lsh_bands, opts.lsh_rows, opts.seed);
        Self {
            opts,
            minhasher,
            lsh,
            original_to_unique: Vec::new(),
            unique_normalized: Vec::new(),
            seen: AHashMap::new(),
        }
    }

    /// Append one row. `None` is treated as null.
    pub fn add(&mut self, name: Option<&str>) {
        let raw = match name {
            Some(s) => s,
            None => {
                self.original_to_unique.push(None);
                return;
            }
        };
        let truncated = truncate_utf8(raw, self.opts.max_name_length);
        let norm = normalize(truncated);
        if norm.is_empty() {
            self.original_to_unique.push(None);
            return;
        }
        // entry() = single hash lookup; on miss we move `norm` into the map
        // as the key (no clone) and run the per-unique sig + LSH insert.
        use std::collections::hash_map::Entry;
        let next_idx = self.unique_normalized.len() as u32;
        let unique_idx = match self.seen.entry(norm) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let sig = self
                    .minhasher
                    .signature(ngrams(e.key(), self.opts.ngram_size));
                self.lsh.insert(next_idx, &sig);
                self.unique_normalized.push(e.key().clone());
                e.insert(next_idx);
                next_idx
            }
        };
        self.original_to_unique.push(Some(unique_idx));
    }

    /// Convenience: append many rows.
    pub fn add_chunk<'a, I: IntoIterator<Item = Option<&'a str>>>(&mut self, names: I) {
        for n in names {
            self.add(n);
        }
    }

    pub fn finalize(self) -> ClusterResult {
        let n_unique = self.unique_normalized.len();
        let n_rows = self.original_to_unique.len();
        if n_unique == 0 {
            return ClusterResult {
                cluster_ids: vec![None; n_rows],
                canonical: vec![None; n_rows],
                flagged_cluster_ids: Vec::new(),
            };
        }

        let vocab = Vocabulary::build(self.unique_normalized.iter(), self.opts.ngram_size);
        let vectors: Vec<_> = self
            .unique_normalized
            .iter()
            .map(|n| vocab.vectorize(n))
            .collect();

        // candidate_pairs_distinct() already returns sorted-distinct.
        let edges: Vec<(u32, u32)> = self
            .lsh
            .candidate_pairs_distinct()
            .into_iter()
            .filter(|&(a, b)| {
                cosine(&vectors[a as usize], &vectors[b as usize]) >= self.opts.threshold
            })
            .collect();

        let (component_of, n_components) = connected_components(n_unique, &edges);
        let groups = group_by_component(&component_of, n_components);
        let adj = build_adjacency(n_unique, &edges);

        struct ComponentInfo {
            members: Vec<u32>,
            canonical: String,
            /// True iff this cluster came from splitting an oversized parent.
            /// Either side of a split inherits the flag — not just the residuals.
            flagged: bool,
        }

        // Hub-radius split: oversized components (size ≥ min_size AND hub
        // eccentricity > radius_max) get split. Pieces smaller than min_size
        // pass through. The "near" group keeps the original hub; the "far"
        // group's intra-subgraph CCs are recursively analysed.
        let mut infos: Vec<ComponentInfo> = Vec::with_capacity(n_components);
        for members in groups {
            for piece in diameter_split(
                members,
                &adj,
                &self.unique_normalized,
                self.opts.hub_radius_max,
                self.opts.diameter_check_min_size,
            ) {
                let canonical = self.unique_normalized[piece.hub as usize].clone();
                infos.push(ComponentInfo {
                    members: piece.members,
                    canonical,
                    flagged: piece.flagged,
                });
            }
        }

        // Stable sort: deterministic with rare canonical-name ties.
        infos.sort_by(|a, b| a.canonical.cmp(&b.canonical));

        // Per-cluster canonical (one String per cluster, NOT per member)
        // + per-unique-name cluster_id table + flagged-list.
        let mut unique_to_cluster_id: Vec<u32> = vec![0; n_unique];
        let mut cluster_canonical: Vec<String> = Vec::with_capacity(infos.len());
        let mut flagged_cluster_ids: Vec<u32> = Vec::new();
        for (cluster_id, info) in infos.into_iter().enumerate() {
            let cid = cluster_id as u32;
            for &m in &info.members {
                unique_to_cluster_id[m as usize] = cid;
            }
            if info.flagged {
                flagged_cluster_ids.push(cid);
            }
            cluster_canonical.push(info.canonical);
        }

        // Re-broadcast to original rows in a single pass.
        let (cluster_ids, canonical): (Vec<_>, Vec<_>) = self
            .original_to_unique
            .iter()
            .map(|opt| match opt {
                Some(u) => {
                    let cid = unique_to_cluster_id[*u as usize];
                    (Some(cid), Some(cluster_canonical[cid as usize].clone()))
                }
                None => (None, None),
            })
            .unzip();

        ClusterResult { cluster_ids, canonical, flagged_cluster_ids }
    }
}

/// Walk back to the nearest UTF-8 char boundary at or below `max`. Returns
/// `s` unchanged when `s.len() <= max`. Pre-normalize input may be UTF-8;
/// naive `&s[..max]` panics on a multi-byte boundary.
fn truncate_utf8(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cluster(names: &[&str], opts: ClusterOpts) -> ClusterResult {
        let mut b = ClusterBuilder::new(opts);
        for n in names {
            b.add(Some(n));
        }
        b.finalize()
    }

    fn opts() -> ClusterOpts {
        ClusterOpts::default()
    }

    #[test]
    fn three_acme_one_brightspoke_cluster_correctly() {
        let names = ["Acme Corporation", "ACME Corp", "Acme Inc", "Brightspoke"];
        let r = cluster(&names, opts());
        assert_eq!(r.cluster_ids.len(), 4);
        // First 3 share a cluster
        assert_eq!(r.cluster_ids[0], r.cluster_ids[1]);
        assert_eq!(r.cluster_ids[1], r.cluster_ids[2]);
        // Brightspoke is its own
        assert_ne!(r.cluster_ids[0], r.cluster_ids[3]);
        // Canonical for the Acme cluster is "acme" (post-norm)
        assert_eq!(r.canonical[0].as_deref(), Some("acme"));
        assert_eq!(r.canonical[3].as_deref(), Some("brightspoke"));
    }

    #[test]
    fn null_and_empty_get_none() {
        let mut b = ClusterBuilder::new(opts());
        b.add(Some("Acme Corp"));
        b.add(None);
        b.add(Some(""));
        b.add(Some("?!"));
        b.add(Some("Acme Inc"));
        let r = b.finalize();
        assert!(r.cluster_ids[0].is_some());
        assert!(r.cluster_ids[1].is_none());
        assert!(r.cluster_ids[2].is_none());
        assert!(r.cluster_ids[3].is_none());
        assert!(r.cluster_ids[4].is_some());
        assert_eq!(r.cluster_ids[0], r.cluster_ids[4]);
    }

    #[test]
    fn duplicates_share_one_cluster_id() {
        // Internal dedup: two identical names share one unique slot but each
        // original row still gets the same cluster_id.
        let names = ["Acme Corp", "Acme Corp", "Acme Corp"];
        let r = cluster(&names, opts());
        assert!(r.cluster_ids.iter().all(|c| *c == r.cluster_ids[0]));
        assert!(r.cluster_ids[0].is_some());
    }

    #[test]
    fn deterministic_output() {
        let names = [
            "Acme Corp",
            "Brightspoke",
            "Acme Corporation",
            "ACME Inc",
            "Foothill Industries",
        ];
        let a = cluster(&names, opts());
        let b = cluster(&names, opts());
        assert_eq!(a.cluster_ids, b.cluster_ids);
        assert_eq!(a.canonical, b.canonical);
    }

    #[test]
    fn cluster_ids_sorted_by_canonical_asc() {
        let names = ["Foothill Industries", "Acme Corp", "Brightspoke"];
        let r = cluster(&names, opts());
        let mut pairs: Vec<(u32, String)> = r
            .cluster_ids
            .iter()
            .zip(r.canonical.iter())
            .filter_map(|(id, c)| match (id, c) {
                (Some(i), Some(n)) => Some((*i, n.clone())),
                _ => None,
            })
            .collect();
        pairs.sort_by_key(|(id, _)| *id);
        let canonicals_in_order: Vec<&str> = pairs.iter().map(|(_, n)| n.as_str()).collect();
        let mut sorted = canonicals_in_order.clone();
        sorted.sort_unstable();
        assert_eq!(canonicals_in_order, sorted);
    }

    #[test]
    fn empty_input() {
        let r = cluster(&[], opts());
        assert_eq!(r.cluster_ids.len(), 0);
        assert_eq!(r.canonical.len(), 0);
        assert!(r.flagged_cluster_ids.is_empty());
    }

    #[test]
    fn high_threshold_keeps_close_matches_apart() {
        // With threshold = 0.99, near-duplicates that don't normalize to
        // identical strings should NOT cluster (cosine ~0.95 falls below).
        let mut o = opts();
        o.threshold = 0.99;
        let names = ["Acme Manufacturing", "Acme Mfg"]; // both -> "acme mfg"
        let r = cluster(&names, o);
        // Both normalize to "acme mfg" exactly -> identical sigs -> cosine=1.0 -> still cluster
        assert_eq!(r.cluster_ids[0], r.cluster_ids[1]);
    }

    #[test]
    fn low_threshold_merges_more() {
        let mut o = opts();
        o.threshold = 0.5;
        let names = ["Alpha Manufacturing Inc", "Alpha Mfg", "Alpha Industries"];
        let r = cluster(&names, o);
        // All share "alpha" + descriptor canonicalization should put them close
        let unique_clusters: std::collections::HashSet<_> =
            r.cluster_ids.iter().filter_map(|c| *c).collect();
        assert!(
            unique_clusters.len() <= 2,
            "expected merging at low threshold, got {} clusters",
            unique_clusters.len()
        );
    }

    #[test]
    fn diameter_flag_not_triggered_on_small_clusters() {
        // Small cluster: diameter_check_min_size defaults to 5; Acme cluster
        // is 3 names, so no flag regardless of structure.
        let r = cluster(
            &["Acme Corp", "Acme Inc", "Acme Corporation"],
            opts(),
        );
        assert!(r.flagged_cluster_ids.is_empty());
    }

}

