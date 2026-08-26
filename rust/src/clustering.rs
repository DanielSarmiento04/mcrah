//! Phase 2 Step 4: Semantic clustering of static 3DGS nodes into hyperedges.
//!
//! Rather than building an O(N^2) KNN graph (rules.md Rule 2), we cluster the
//! static Gaussian attributes into hyperedges using cosine similarity on
//! a compact per-Gaussian semantic feature. This yields the initial
//! hypergraph G = (V, E) consumed by the SIMGNN/DHGC modules.
//!
//! The implementation here is a deterministic, centroid-based assignment:
//!   1. Each Gaussian is described by a feature vector f_i = [mu / scale,
//!      spherical harmonics DC color, opacity, log-scale].
//!   2. K centroids are initialized via k-means++.
//!   3. Assignment is by **cosine** similarity (not L2), matching Rule 2.
//!   4. Each cluster -> one hyperedge; the incidence matrix H is exported.
//!
//! This is intentionally a CPU preprocess: it runs once per scene at t=0 and
//! the structure is then propagated autoregressively.

use crate::npy;
use ndarray::{Array1, Array2};
use std::path::Path;

/// Knobs for the clustering preprocessor.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Number of hyperedges (clusters). If None, derived from N via sqrt-law.
    pub k: Option<usize>,
    /// Max k-means iterations.
    pub max_iter: usize,
    /// Cosine-similarity threshold for "hard" assignment to nearest centroid.
    pub sim_eps: f32,
    /// Seed for deterministic centroid init (k-means++).
    pub seed: u64,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            k: None,
            max_iter: 50,
            sim_eps: 1e-4,
            seed: 42,
        }
    }
}

/// Per-Gaussian static attributes at t=0 (the static 3DGS init, Phase 1 Step 3).
/// Each row is one Gaussian; feature dim is consumed by the clustering.
#[derive(Debug, Clone)]
pub struct StaticGaussians {
    /// N x D feature matrix (already normalized to a compact semantic space).
    pub feats: Array2<f32>,
}

impl StaticGaussians {
    pub fn n(&self) -> usize {
        self.feats.nrows()
    }
}

/// Result of clustering: per-node cluster id + incidence matrix.
#[derive(Debug, Clone)]
pub struct Hypergraph {
    pub n_nodes: usize,
    pub n_edges: usize,
    /// length N, assignment[node] = edge id
    pub assignment: Vec<usize>,
}

impl Hypergraph {
    /// Dense incidence matrix H (N x E), H[i,j]=1 if node i in edge j.
    pub fn incidence_dense(&self) -> Array2<f32> {
        let mut h = Array2::<f32>::zeros((self.n_nodes, self.n_edges));
        for (i, &e) in self.assignment.iter().enumerate() {
            h[(i, e)] = 1.0;
        }
        h
    }

    /// CSR-like edge lists: for each edge, the sorted list of member nodes.
    pub fn edge_lists(&self) -> Vec<Vec<usize>> {
        let mut edges: Vec<Vec<usize>> = vec![Vec::new(); self.n_edges];
        for (node, &e) in self.assignment.iter().enumerate() {
            edges[e].push(node);
        }
        for e in &mut edges {
            e.sort();
        }
        edges
    }
}

/// Run cosine-similarity k-means clustering and return the hypergraph.
pub fn cosine_clusters(g: &StaticGaussians, cfg: &ClusterConfig) -> Hypergraph {
    let n = g.n();
    let k = cfg.k.unwrap_or_else(|| ((n as f32).sqrt().ceil() as usize).max(8));
    let k = k.min(n).max(1);

    let (assignment, _centroids) = cosine_kmeans(&g.feats, k, cfg);
    Hypergraph {
        n_nodes: n,
        n_edges: k,
        assignment,
    }
}

/// L2-normalize each row in place.
fn row_normalize(m: &mut Array2<f32>) {
    for i in 0..m.nrows() {
        let mut row = m.row_mut(i);
        let mut norm = 0.0f32;
        for &v in row.iter() {
            norm += v * v;
        }
        norm = norm.sqrt().max(1e-12);
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
}

/// Deterministic splitmix64 PRNG (seeded).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / ((1u64 << 24) as f32)
    }
}

/// k-means++ init followed by Lloyd iterations using **cosine** similarity
/// (equivalent to Euclidean k-means on L2-normalized vectors).
fn cosine_kmeans(
    feats: &Array2<f32>,
    k: usize,
    cfg: &ClusterConfig,
) -> (Vec<usize>, Array2<f32>) {
    let n = feats.nrows();
    let d = feats.ncols();
    let mut x = feats.clone();
    row_normalize(&mut x);

    // k-means++ initialization.
    let mut rng = Rng(cfg.seed);
    let first = (rng.next() as usize) % n;
    let mut centroids = Array2::<f32>::zeros((k, d));
    centroids.row_mut(0).assign(&x.row(first));

    let mut dist2 = Array1::<f32>::from_elem(n, f32::MAX);
    for ci in 1..k {
        // update dist2 to nearest chosen centroid
        let c = centroids.row(ci - 1).to_owned();
        for i in 0..n {
            let xi = x.row(i);
            // cosine sim = dot (since normalized); dist2 = 1 - sim
            let sim: f32 = xi.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
            let d2 = (1.0 - sim).max(0.0);
            if d2 < dist2[i] {
                dist2[i] = d2;
            }
        }
        let total: f32 = dist2.iter().sum();
        let r = (rng.unit() * total).min(total - 1e-6).max(0.0);
        let mut acc = 0.0f32;
        let mut pick = n - 1;
        for i in 0..n {
            acc += dist2[i];
            if acc >= r {
                pick = i;
                break;
            }
        }
        centroids.row_mut(ci).assign(&x.row(pick));
    }

    // Lloyd iterations
    let mut assignment = vec![0usize; n];
    for _ in 0..cfg.max_iter {
        let mut changed = false;
        for i in 0..n {
            let xi = x.row(i);
            let mut best = 0usize;
            let mut best_sim = f32::MIN;
            for c in 0..k {
                let cc = centroids.row(c);
                let sim: f32 = xi.iter().zip(cc.iter()).map(|(a, b)| a * b).sum();
                if sim > best_sim {
                    best_sim = sim;
                    best = c;
                }
            }
            if assignment[i] != best {
                changed = true;
                assignment[i] = best;
            }
        }
        if !changed {
            break;
        }
        // recompute centroids = mean of assigned (normalized) vectors
        let mut new_c = Array2::<f32>::zeros((k, d));
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let c = assignment[i];
            new_c.row_mut(c).scaled_add(1.0, &x.row(i));
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                for v in new_c.row_mut(c).iter_mut() {
                    *v *= inv;
                }
            } else {
                // reseed dead centroid to a random node
                let r = (rng.next() as usize) % n;
                new_c.row_mut(c).assign(&x.row(r));
            }
        }
        // re-normalize centroids
        row_normalize(&mut new_c);
        centroids = new_c;
    }
    (assignment, centroids)
}

/// Persist a hypergraph to disk as:
///   <out>/assignment.npy   (i64, N)
///   <out>/incidence.npy     (f32, N x E)
///   <out>/edge_lists.npy    (i64, nnz x 2: [node, edge])
pub fn save_hypergraph(hg: &Hypergraph, out_dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let assign_i64: Vec<i64> = hg.assignment.iter().map(|&x| x as i64).collect();
    npy::save_i64_2d(&out_dir.join("assignment.npy"), hg.n_nodes, 1, &assign_i64)?;

    let inc = hg.incidence_dense();
    let inc_flat: Vec<f32> = inc.iter().copied().collect();
    npy::save_f32_2d(&out_dir.join("incidence.npy"), hg.n_nodes, hg.n_edges, &inc_flat)?;

    let mut edges = Vec::new();
    for (node, &e) in hg.assignment.iter().enumerate() {
        edges.push(node as i64);
        edges.push(e as i64);
    }
    npy::save_i64_2d(&out_dir.join("edge_lists.npy"), edges.len() / 2, 2, &edges)?;
    Ok(())
}
