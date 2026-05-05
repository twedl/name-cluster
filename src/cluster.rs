//! Cluster-stage primitives: union-find, BFS, hub selection.
//!
//! Pipeline role: after rerank emits a thresholded edge list, these
//! primitives turn it into named clusters.
//!
//!   edges -> UnionFind -> components -> per-component hub -> canonical
//!
//! Determinism: all operations are pure functions of their inputs and
//! produce output in sorted order (component members ascending, hub
//! tie-breaks lex-ascending on canonical name).

use ahash::AHashMap;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Union-find
// ---------------------------------------------------------------------------

/// Disjoint-set with path-halving find + union-by-rank.
pub struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            // Path-halving: every other step, point at grandparent.
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = self.parent[x as usize];
        }
        x
    }

    pub fn union(&mut self, a: u32, b: u32) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        let (ra, rb) = if self.rank[ra as usize] < self.rank[rb as usize] {
            (rb, ra)
        } else {
            (ra, rb)
        };
        self.parent[rb as usize] = ra;
        if self.rank[ra as usize] == self.rank[rb as usize] {
            self.rank[ra as usize] += 1;
        }
        true
    }
}

/// Component label per node. Labels assigned `0..n_components` in ascending
/// root-index order — deterministic regardless of edge insertion order.
pub fn connected_components(n: usize, edges: &[(u32, u32)]) -> (Vec<u32>, usize) {
    let mut uf = UnionFind::new(n);
    for &(a, b) in edges {
        uf.union(a, b);
    }
    let mut roots: Vec<u32> = (0..n as u32).map(|i| uf.find(i)).collect();
    let mut unique_roots: Vec<u32> = roots.clone();
    unique_roots.sort_unstable();
    unique_roots.dedup();
    let mut root_to_label: AHashMap<u32, u32> = AHashMap::with_capacity(unique_roots.len());
    for (i, &r) in unique_roots.iter().enumerate() {
        root_to_label.insert(r, i as u32);
    }
    for r in roots.iter_mut() {
        *r = root_to_label[r];
    }
    (roots, unique_roots.len())
}

pub fn group_by_component(component_of: &[u32], n_components: usize) -> Vec<Vec<u32>> {
    let mut groups: Vec<Vec<u32>> = vec![Vec::new(); n_components];
    for (node, &comp) in component_of.iter().enumerate() {
        groups[comp as usize].push(node as u32);
    }
    groups
}

// ---------------------------------------------------------------------------
// Adjacency + BFS
// ---------------------------------------------------------------------------

/// Each neighbor list is sorted-deduplicated for deterministic BFS order.
pub fn build_adjacency(n: usize, edges: &[(u32, u32)]) -> Vec<Vec<u32>> {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(a, b) in edges {
        adj[a as usize].push(b);
        adj[b as usize].push(a);
    }
    for nbrs in adj.iter_mut() {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    adj
}

/// BFS from `source`, return the maximum distance reached (eccentricity).
/// Distances are in number of edges. Unreachable nodes don't contribute.
pub fn bfs_eccentricity(adj: &[Vec<u32>], source: u32) -> usize {
    let n = adj.len();
    let mut dist: Vec<i32> = vec![-1; n];
    dist[source as usize] = 0;
    let mut queue: VecDeque<u32> = VecDeque::new();
    queue.push_back(source);
    let mut max_dist = 0;
    while let Some(u) = queue.pop_front() {
        let d = dist[u as usize];
        for &v in &adj[u as usize] {
            if dist[v as usize] < 0 {
                dist[v as usize] = d + 1;
                queue.push_back(v);
                if (d + 1) as usize > max_dist {
                    max_dist = (d + 1) as usize;
                }
            }
        }
    }
    max_dist
}

// ---------------------------------------------------------------------------
// Hub selection
// ---------------------------------------------------------------------------

/// Highest-degree node in `members`, lex-asc tie-break on `names[i]`.
/// `members` must be non-empty; `adj` covers the full graph.
pub fn pick_hub(members: &[u32], adj: &[Vec<u32>], names: &[String]) -> u32 {
    *members
        .iter()
        .min_by(|&&a, &&b| {
            let deg_a = adj[a as usize].len();
            let deg_b = adj[b as usize].len();
            // Higher degree wins -> compare by Reverse(degree); lex-asc name secondary.
            deg_b
                .cmp(&deg_a)
                .then_with(|| names[a as usize].cmp(&names[b as usize]))
        })
        .expect("pick_hub called with empty member list")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uf_basic() {
        let mut uf = UnionFind::new(5);
        assert!(uf.union(0, 1));
        assert!(uf.union(1, 2));
        assert!(!uf.union(0, 2));
        assert_eq!(uf.find(0), uf.find(2));
        assert_ne!(uf.find(0), uf.find(3));
    }

    #[test]
    fn cc_simple() {
        // 0-1-2 connected; 3-4 connected; 5 alone.
        let edges = vec![(0, 1), (1, 2), (3, 4)];
        let (comps, n) = connected_components(6, &edges);
        assert_eq!(n, 3);
        assert_eq!(comps[0], comps[1]);
        assert_eq!(comps[1], comps[2]);
        assert_eq!(comps[3], comps[4]);
        assert_ne!(comps[0], comps[3]);
        assert_ne!(comps[0], comps[5]);
        // Labels assigned in ascending root order: comp(0) < comp(3) < comp(5)
        assert!(comps[0] < comps[3]);
        assert!(comps[3] < comps[5]);
    }

    #[test]
    fn cc_deterministic_label_order() {
        let edges_a = vec![(0, 1), (3, 4)];
        let edges_b = vec![(3, 4), (1, 0)];
        let (a, _) = connected_components(5, &edges_a);
        let (b, _) = connected_components(5, &edges_b);
        assert_eq!(a, b);
    }

    #[test]
    fn group_by_component_basic() {
        let comps = vec![0u32, 0, 1, 0, 2, 1];
        let groups = group_by_component(&comps, 3);
        assert_eq!(groups[0], vec![0, 1, 3]);
        assert_eq!(groups[1], vec![2, 5]);
        assert_eq!(groups[2], vec![4]);
    }

    #[test]
    fn bfs_star_eccentricity_two() {
        // Star: hub = 0, spokes = 1,2,3 (each only connected to 0).
        let adj = build_adjacency(4, &[(0, 1), (0, 2), (0, 3)]);
        // From hub: max distance = 1 (each spoke is 1 hop)
        assert_eq!(bfs_eccentricity(&adj, 0), 1);
        // From a spoke: max = 2 (spoke -> hub -> other spoke)
        assert_eq!(bfs_eccentricity(&adj, 1), 2);
    }

    #[test]
    fn bfs_chain_eccentricity_grows_with_length() {
        // Chain: 0-1-2-3-4
        let adj = build_adjacency(5, &[(0, 1), (1, 2), (2, 3), (3, 4)]);
        assert_eq!(bfs_eccentricity(&adj, 0), 4);
        assert_eq!(bfs_eccentricity(&adj, 2), 2);
    }

    #[test]
    fn bfs_disconnected_node() {
        let adj = build_adjacency(3, &[(0, 1)]);
        // Source 2 is alone -> max distance 0
        assert_eq!(bfs_eccentricity(&adj, 2), 0);
    }

    #[test]
    fn pick_hub_max_degree_wins() {
        // Star with hub=2 (degree 3), spokes 0,1,3 (each degree 1)
        let adj = build_adjacency(4, &[(2, 0), (2, 1), (2, 3)]);
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let members = vec![0, 1, 2, 3];
        assert_eq!(pick_hub(&members, &adj, &names), 2);
    }

    #[test]
    fn pick_hub_tie_break_by_name() {
        // Both 0 and 1 have degree 1 (path of length 1)
        let adj = build_adjacency(2, &[(0, 1)]);
        let names_asc = vec!["aaa".to_string(), "bbb".to_string()];
        assert_eq!(pick_hub(&[0, 1], &adj, &names_asc), 0); // "aaa" < "bbb"
        let names_desc = vec!["zzz".to_string(), "bbb".to_string()];
        assert_eq!(pick_hub(&[0, 1], &adj, &names_desc), 1); // "bbb" < "zzz"
    }

    #[test]
    fn pick_hub_singleton_returns_self() {
        let adj = build_adjacency(1, &[]);
        let names = vec!["only".to_string()];
        assert_eq!(pick_hub(&[0], &adj, &names), 0);
    }
}
