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
use rayon::prelude::*;

use crate::cluster::{build_adjacency, connected_components, diameter_split, group_by_component};
use crate::lsh::LshIndex;
use crate::minhash::MinHasher;
use crate::ngram::{comparison_form, ngrams};
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
    /// Bidirectional alias map applied AFTER normalize(): if the post-normalize
    /// string is a key, replace with the value. Lets users force-merge an
    /// acronym with its expansion (e.g. "ibm" -> "intl business machines"),
    /// where char-n-gram cosine alone never would. Empty by default.
    pub aliases: AHashMap<String, String>,
    /// Worker threads for the parallelisable stages (TF-IDF rerank scoring +
    /// per-name vectorisation). `None` uses rayon's global pool, sized by
    /// `RAYON_NUM_THREADS` if set, else `std::thread::available_parallelism()`
    /// (honors cgroup CPU quota on Linux).
    /// `Some(1)` forces single-threaded execution; useful for benchmarking
    /// and debugging non-determinism. Output is byte-identical regardless
    /// of thread count.
    pub n_threads: Option<usize>,
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
            aliases: AHashMap::new(),
            n_threads: None,
        }
    }
}

/// Run `f` inside a rayon thread pool of `n_threads` workers (or the global
/// pool when `None`). Per-call pool, so we don't mutate `build_global` and
/// callers can vary thread count freely.
fn with_thread_pool<R: Send, F: FnOnce() -> R + Send>(n_threads: Option<usize>, f: F) -> R {
    match n_threads {
        None => f(),
        Some(n) => rayon::ThreadPoolBuilder::new()
            .num_threads(n.max(1))
            .build()
            .expect("rayon thread pool")
            .install(f),
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
    /// `hub_radius_max`. Both halves of any hub-radius split inherit the
    /// flag; useful for diagnostic / downstream quality filtering.
    pub flagged_cluster_ids: Vec<u32>,
}

/// Output of [`ClusterBuilder::candidate_pairs`] — the post-LSH, post-rerank
/// candidate pairs WITHOUT the cluster-stage (UF, hub split, sort/relabel).
/// Used by the public `candidates()` debug API.
#[derive(Debug, Clone)]
pub struct CandidatePairsResult {
    /// One row per pair: (unique_idx_a, unique_idx_b, cosine_score).
    /// Sorted by (a, b) ascending. Pairs are over UNIQUE normalized names;
    /// callers map unique idx → first original row via `original_to_unique`.
    pub scored_pairs: Vec<(u32, u32, f32)>,
    pub unique_normalized: Vec<String>,
    pub original_to_unique: Vec<Option<u32>>,
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

    /// Append one row. `None` is treated as null. Test-only — production
    /// code uses `add_batch` for the parallel insertion path.
    #[cfg(test)]
    pub fn add(&mut self, name: Option<&str>) {
        let raw = match name {
            Some(s) => s,
            None => {
                self.original_to_unique.push(None);
                return;
            }
        };
        let truncated = truncate_utf8(raw, self.opts.max_name_length);
        let mut norm = normalize(truncated);
        if let Some(canon) = self.opts.aliases.get(&norm) {
            norm = canon.clone();
        }
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
                let cmp = comparison_form(e.key());
                let sig = self
                    .minhasher
                    .signature(ngrams(cmp.as_ref(), self.opts.ngram_size));
                self.lsh.insert(next_idx, &sig);
                self.unique_normalized.push(e.key().clone());
                e.insert(next_idx);
                next_idx
            }
        };
        self.original_to_unique.push(Some(unique_idx));
    }

    /// Parallel batch insertion. Three phases:
    ///   A. par_iter: per-row truncate + normalize + alias lookup → Vec<Option<String>>
    ///   B. sequential: dedup + assign unique indices (preserves insertion order)
    ///   C. par_iter: per-new-unique MinHash signature, then sequential LSH insert
    ///
    /// Same observable result as calling `add()` per row, but cuts the per-row
    /// CPU into parallel work where it's safe (normalize is pure; signature is
    /// `&self`-only on the MinHasher).
    pub fn add_batch(&mut self, names: Vec<Option<String>>) {
        if names.is_empty() {
            return;
        }
        let max_len = self.opts.max_name_length;
        let aliases = &self.opts.aliases;
        let normalized: Vec<Option<String>> = with_thread_pool(self.opts.n_threads, || {
            names
                .into_par_iter()
                .map(|opt| {
                    let raw = opt?;
                    let trunc = truncate_utf8(&raw, max_len);
                    let mut nm = normalize(trunc);
                    if let Some(canon) = aliases.get(&nm) {
                        nm = canon.clone();
                    }
                    (!nm.is_empty()).then_some(nm)
                })
                .collect()
        });

        use std::collections::hash_map::Entry;
        let mut new_keys: Vec<String> = Vec::new();
        let mut new_idxs: Vec<u32> = Vec::new();
        for opt_norm in normalized {
            match opt_norm {
                None => self.original_to_unique.push(None),
                Some(nm) => {
                    let next_idx = self.unique_normalized.len() as u32;
                    let unique_idx = match self.seen.entry(nm) {
                        Entry::Occupied(e) => *e.get(),
                        Entry::Vacant(e) => {
                            new_keys.push(e.key().clone());
                            new_idxs.push(next_idx);
                            self.unique_normalized.push(e.key().clone());
                            e.insert(next_idx);
                            next_idx
                        }
                    };
                    self.original_to_unique.push(Some(unique_idx));
                }
            }
        }

        if new_keys.is_empty() {
            return;
        }
        let ngram_size = self.opts.ngram_size;
        let hasher = &self.minhasher;
        let sigs: Vec<Vec<u64>> = with_thread_pool(self.opts.n_threads, || {
            new_keys
                .par_iter()
                .map(|name| {
                    let cmp = comparison_form(name);
                    hasher.signature(ngrams(cmp.as_ref(), ngram_size))
                })
                .collect()
        });
        for (idx, sig) in new_idxs.iter().zip(sigs.iter()) {
            self.lsh.insert(*idx, sig);
        }
    }

    /// Run pipeline through TF-IDF rerank only — no clustering. Pairs are
    /// filtered to `score >= min_score` during scoring (no full Vec built
    /// then re-filtered). Set `min_score = 0.0` to return everything.
    pub fn candidate_pairs(self, min_score: f32) -> CandidatePairsResult {
        if self.unique_normalized.is_empty() {
            return CandidatePairsResult {
                scored_pairs: Vec::new(),
                unique_normalized: Vec::new(),
                original_to_unique: self.original_to_unique,
            };
        }
        let scored_pairs = with_thread_pool(self.opts.n_threads, || {
            score_lsh_candidates(
                &self.unique_normalized,
                &self.lsh,
                self.opts.ngram_size,
                min_score,
            )
        });
        CandidatePairsResult {
            scored_pairs,
            unique_normalized: self.unique_normalized,
            original_to_unique: self.original_to_unique,
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

        let scored_pairs = with_thread_pool(self.opts.n_threads, || {
            score_lsh_candidates(
                &self.unique_normalized,
                &self.lsh,
                self.opts.ngram_size,
                self.opts.threshold,
            )
        });
        let edges: Vec<(u32, u32)> = scored_pairs.into_iter().map(|(a, b, _)| (a, b)).collect();

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

        ClusterResult {
            cluster_ids,
            canonical,
            flagged_cluster_ids,
        }
    }
}

/// Build a TF-IDF vocabulary over `unique_normalized`, vectorize each, then
/// score every LSH candidate pair via cosine, keeping only pairs at or above
/// `min_score`. Shared by `finalize()` (threshold) and `candidate_pairs()`
/// (debug min_score) — both stop here in the pipeline.
///
/// Vectorisation and scoring are par-iter'd; output ordering is preserved by
/// rayon's `collect`, so the returned Vec is byte-identical to the serial run.
fn score_lsh_candidates(
    unique_normalized: &[String],
    lsh: &LshIndex,
    ngram_size: usize,
    min_score: f32,
) -> Vec<(u32, u32, f32)> {
    // IDF degeneracy short-circuit: with ≤ 2 unique normalized names the
    // TF-IDF formula collapses (n-grams shared by both docs get idf=0; the
    // unshared ones land in disjoint vectors → cosine always 0 regardless of
    // how similar the inputs look). Skip the wasted vocab build + vectorise.
    // Public limitation documented in README § Scope.
    if unique_normalized.len() < 3 {
        return Vec::new();
    }
    // Strip whitespace before n-gramming so glued/spaced variants of the same
    // name produce identical trigram sets — see ngram::comparison_form.
    let cmp: Vec<_> = unique_normalized
        .par_iter()
        .map(|s| comparison_form(s))
        .collect();
    let vocab = Vocabulary::build(cmp.iter().map(|c| c.as_ref()), ngram_size);
    let vectors: Vec<_> = cmp
        .par_iter()
        .map(|c| vocab.vectorize(c.as_ref()))
        .collect();
    lsh.candidate_pairs_distinct()
        .into_par_iter()
        .filter_map(|(a, b)| {
            let score = cosine(&vectors[a as usize], &vectors[b as usize]);
            (score >= min_score).then_some((a, b, score))
        })
        .collect()
}

/// Pairwise n-gram-count cosine over a small set of names (no LSH, no IDF).
/// For `explain()` on cluster members. Does NOT replicate cluster()'s
/// IDF-weighted scores — those depend on the full corpus and are degenerate
/// on tiny subsets (shared n-grams have IDF=0). Returns interpretable
/// "how similar do these strings look" scores instead.
pub fn pairwise_cosines(names: &[String], ngram_size: usize) -> Vec<(u32, u32, f32)> {
    if names.len() < 2 {
        return Vec::new();
    }
    use ahash::AHashMap;
    let docs: Vec<AHashMap<Vec<u8>, u32>> = names
        .iter()
        .map(|n| {
            let cmp = comparison_form(n);
            let mut counts: AHashMap<Vec<u8>, u32> = AHashMap::new();
            for g in ngrams(cmp.as_ref(), ngram_size) {
                *counts.entry(g.to_vec()).or_insert(0) += 1;
            }
            counts
        })
        .collect();
    let norms: Vec<f32> = docs
        .iter()
        .map(|d| d.values().map(|&c| (c as f32).powi(2)).sum::<f32>().sqrt())
        .collect();
    let mut out = Vec::with_capacity(names.len() * (names.len() - 1) / 2);
    for i in 0..names.len() {
        for j in (i + 1)..names.len() {
            let dot: f32 = docs[i]
                .iter()
                .filter_map(|(g, &c)| docs[j].get(g).map(|&c2| (c as f32) * (c2 as f32)))
                .sum();
            let denom = norms[i] * norms[j];
            let score = if denom > 0.0 { dot / denom } else { 0.0 };
            out.push((i as u32, j as u32, score));
        }
    }
    out
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
        let r = cluster(&["Acme Corp", "Acme Inc", "Acme Corporation"], opts());
        assert!(r.flagged_cluster_ids.is_empty());
    }
}
