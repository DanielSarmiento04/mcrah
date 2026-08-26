//! `preprocess` binary: parses D-NeRF transforms, dumps per-scene metadata,
//! decodes images, and (when a static-Gaussian snapshot exists) builds the
//! cosine hypergraph + dilated DHGC neighbor sets.
//!
//! Usage:
//!   preprocess --data-root ./data --category trex \
//!              --out ./processed/trex [--images] [--cluster-k 64]
//!
//! Running with no args processes all categories under ./data.

use clap::{Parser, ValueEnum};
use mcrah_data::{
    clustering::{cosine_clusters, save_hypergraph, ClusterConfig},
    graph::save_dilated_sets,
    list_categories, load_scene, load_image_u8, npy,
    transforms::SceneSplit,
};
use std::path::PathBuf;

#[derive(Clone, Debug, ValueEnum)]
enum SplitArg {
    All,
    Train,
    Val,
    Test,
}

#[derive(Parser, Debug)]
#[command(name = "preprocess", version, about = "D-NeRF -> .npy ingestion + hypergraph clustering for MCRAH")]
struct Args {
    /// Path to the data/ directory holding the 8 D-NeRF categories.
    #[arg(long, default_value = "./data")]
    data_root: PathBuf,

    /// Single category to process. If omitted, processes all categories.
    #[arg(long)]
    category: Option<String>,

    /// Output directory.
    #[arg(long, default_value = "./processed")]
    out: PathBuf,

    /// Also decode and dump all images as .npy (large).
    #[arg(long, default_value_t = false)]
    images: bool,

    /// Build the static hypergraph from a synthetic uniform Gaussian cloud
    /// (used until the real static-3DGS init in Phase 1 Step 3 lands).
    #[arg(long, default_value_t = false)]
    cluster: bool,

    /// Number of hyperedges (clusters). 0 => auto (sqrt(N)).
    #[arg(long, default_value_t = 0)]
    cluster_k: usize,

    /// Which split to process images for.
    #[arg(long, value_enum, default_value_t = SplitArg::All)]
    split: SplitArg,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let cats: Vec<String> = match &args.category {
        Some(c) => vec![c.clone()],
        None => list_categories(&args.data_root)?,
    };

    for cat in &cats {
        println!("[{cat}] loading scene...");
        let scene = load_scene(&args.data_root, cat)?;
        println!(
            "[{cat}] camera_angle_x={:.6} train={} val={} test={} t_steps={}",
            scene.camera_angle_x,
            scene.train.len(),
            scene.val.len(),
            scene.test.len(),
            scene.temporal_steps().len()
        );

        // Always dump per-frame metadata (poses + times).
        let scene_out = args.out.join(cat);
        std::fs::create_dir_all(&scene_out)?;
        dump_metadata(&scene, &scene_out)?;

        if args.images {
            let splits: Vec<SceneSplit> = match args.split {
                SplitArg::All => vec![SceneSplit::Train, SceneSplit::Val, SceneSplit::Test],
                SplitArg::Train => vec![SceneSplit::Train],
                SplitArg::Val => vec![SceneSplit::Val],
                SplitArg::Test => vec![SceneSplit::Test],
            };
            for sp in splits {
                dump_images(&scene, sp, &scene_out)?;
            }
        }

        if args.cluster {
            // Placeholder static cloud: a regular grid of N Gaussians so the
            // hypergraph machinery is exercised end-to-end. Phase 1 Step 3 will
            // replace this with the real static-3DGS fit.
            let n = 1024usize;
            let d = 8usize;
            let feats: Vec<f32> = (0..(n * d))
                .map(|i| ((i as u32).wrapping_mul(2654435761) as f32) / u32::MAX as f32)
                .collect();
            let gs = mcrah_data::clustering::StaticGaussians {
                feats: ndarray::Array2::from_shape_vec((n, d), feats)?,
            };
            let cfg = ClusterConfig {
                k: if args.cluster_k == 0 { None } else { Some(args.cluster_k) },
                ..Default::default()
            };
            let hg = cosine_clusters(&gs, &cfg);
            println!("[{cat}] hypergraph: {} nodes -> {} edges", hg.n_nodes, hg.n_edges);
            save_hypergraph(&hg, &scene_out.join("hypergraph"))?;
            save_dilated_sets(&hg, &[1, 2, 4], &scene_out.join("hypergraph"))?;
        }
    }
    println!("done.");
    Ok(())
}

/// Dump poses (N x 4x4 flattened = N x 16) and times (N) per split.
fn dump_metadata(
    scene: &mcrah_data::transforms::DNeRFDataset,
    out: &std::path::Path,
) -> anyhow::Result<()> {
    for sp in [SceneSplit::Train, SceneSplit::Val, SceneSplit::Test] {
        let frames = scene.split(sp);
        let n = frames.len();
        if n == 0 {
            continue;
        }
        let mut poses = Vec::with_capacity(n * 16);
        let mut times = Vec::with_capacity(n);
        let mut file_idx = Vec::with_capacity(n);
        for (i, f) in frames.iter().enumerate() {
            for r in 0..4 {
                for c in 0..4 {
                    poses.push(f.transform_matrix[r][c]);
                }
            }
            times.push(f.time);
            file_idx.push(i as i64);
        }
        let d = out.join(sp.name());
        std::fs::create_dir_all(&d)?;
        npy::save_f32_2d(&d.join("poses.npy"), n, 16, &poses)?;
        npy::save_f32_1d(&d.join("times.npy"), &times)?;
        npy::save_i64_2d(&d.join("file_idx.npy"), n, 1, &file_idx)?;
    }
    Ok(())
}

/// Decode every PNG for a split into a stacked .npy tensor.
fn dump_images(
    scene: &mcrah_data::transforms::DNeRFDataset,
    sp: SceneSplit,
    out: &std::path::Path,
) -> anyhow::Result<()> {
    let frames = scene.split(sp);
    let n = frames.len();
    if n == 0 {
        return Ok(());
    }
    let img_dir = out.join(sp.name()).join("images");
    std::fs::create_dir_all(&img_dir)?;
    println!("[{}] dumping {} images...", scene.category, n);
    for (i, f) in frames.iter().enumerate() {
        let path = scene.image_path(f);
        let img = load_image_u8(&path)?;
        npy::save_u8_3d(
            &img_dir.join(format!("{i:04}.npy")),
            img.height as usize,
            img.width as usize,
            img.channels as usize,
            &img.data,
        )?;
        if i == 0 {
            println!("[{}] img shape = {}x{}x{}", scene.category, img.width, img.height, img.channels);
        }
    }
    Ok(())
}
