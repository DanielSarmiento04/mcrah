//! Graph utilities shared between the Rust preprocessor and (via .npy export)
//! the Python training side. Currently exposes multi-scale dilated subgraph
//! construction used by the DHGC module (Phase 2 Step 5).

use crate::npy;

/// Build a dilated neighborhood index for a hypergraph.
///
/// Given the edge membership of each node, a node's "dilation-1" neighbors are
/// all nodes sharing an edge with it; dilation-k neighbors are nodes reachable
/// in exactly k edge-hops (multi-scale receptive field, rules.md Rule 3).
///
/// Returns CSR: `neighbors` (concatenated node ids) + `offsets` (length N+1).
pub fn dilated_neighbors(hg: &crate::clustering::Hypergraph, dilation: usize) -> (Vec<i64>, Vec<i64>) {
    let n = hg.n_nodes;
    let edges = hg.edge_lists();

    // node -> edges it belongs to
    let mut node_edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (e, members) in edges.iter().enumerate() {
        for &m in members {
            node_edges[m].push(e);
        }
    }

    // BFS expansion of `dilation` hops over edges.
    let mut offsets = vec![0i64; n + 1];
    let mut all_neighbors: Vec<Vec<i64>> = vec![Vec::new(); n];
    for i in 0..n {
        let mut visited = std::collections::HashSet::new();
        visited.insert(i);
        let mut frontier: Vec<usize> = node_edges[i].clone();
        for _ in 0..dilation {
            let mut next_frontier = Vec::new();
            for e in frontier.drain(..) {
                for &m in &edges[e] {
                    if visited.insert(m) {
                        all_neighbors[i].push(m as i64);
                        next_frontier.extend(node_edges[m].iter().copied());
                    }
                }
            }
            // dedup next_frontier edges
            let mut seen = std::collections::HashSet::new();
            next_frontier.retain(|e| seen.insert(*e));
            frontier = next_frontier;
        }
        all_neighbors[i].sort_unstable();
    }
    let mut cum = 0i64;
    for (i, nb) in all_neighbors.iter().enumerate() {
        cum += nb.len() as i64;
        offsets[i + 1] = cum;
    }
    let neighbors: Vec<i64> = all_neighbors.into_iter().flatten().collect();
    (neighbors, offsets)
}

/// Persist dilated neighbor sets for several dilation levels.
/// Output files:
///   <out>/dil_neighbors_<k>.npy  (i64, nnz)
///   <out>/dil_offsets_<k>.npy    (i64, N+1)
pub fn save_dilated_sets(hg: &crate::clustering::Hypergraph, dilations: &[usize], out_dir: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    for &k in dilations {
        let (neighbors, offsets) = dilated_neighbors(hg, k);
        npy::save_i64_2d(&out_dir.join(format!("dil_neighbors_{k}.npy")), neighbors.len(), 1, &neighbors)?;
        npy::save_i64_2d(&out_dir.join(format!("dil_offsets_{k}.npy")), offsets.len(), 1, &offsets)?;
    }
    Ok(())
}
