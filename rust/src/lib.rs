//! mcrah-data: high-performance D-NeRF data ingestion and hypergraph clustering
//! preprocessing, written in Rust (per rules.md Rule 7) to maximize throughput
//! before tensors are handed to the PyTorch training loop.
//!
//! The library exposes three concerns:
//!   1. [`transforms`]   - parse the D-NeRF `transforms_*.json` schema.
//!   2. [`image_io`]     - decode PNG frames into packed HWC u8 tensors.
//!   3. [`npy`]          - serialize tensors to numpy `.npy` (consumed by torch).
//!   4. [`clustering`]   - cosine-similarity hyperedge clustering (Phase 2 Step 4).

pub mod transforms;
pub mod image_io;
pub mod npy;
pub mod clustering;
pub mod graph;

pub use transforms::{DNeRFDataset, Frame, SceneSplit, load_scene, list_categories};
pub use image_io::load_image_u8;
pub use clustering::{cosine_clusters, ClusterConfig};
