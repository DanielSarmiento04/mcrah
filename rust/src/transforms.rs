//! Parsing of the D-NeRF `transforms_{train,val,test}.json` schema.
//!
//! Verified schema (all 8 categories, e.g. data/trex/transforms_train.json):
//! ```jsonc
//! {
//!   "camera_angle_x": 0.6911112070083618,
//!   "frames": [
//!     { "file_path": "./train/r_000",
//!       "rotation": 0.031415926535897934,
//!       "time": 0.0,
//!       "transform_matrix": [ [..4], [..4], [..4], [0,0,0,1] ] }
//!   ]
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A single D-NeRF frame: camera pose + normalized timestamp + image path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    /// Relative path, e.g. "./train/r_000" (no extension).
    #[serde(rename = "file_path")]
    pub file_path: String,
    /// Per-frame rotation delta (radians).
    pub rotation: f32,
    /// Normalized time in [0, 1].
    pub time: f32,
    /// 4x4 camera-to-world transform (OpenGL/NeRF convention; column-major in file).
    #[serde(rename = "transform_matrix")]
    pub transform_matrix: [[f32; 4]; 4],
}

/// Top-level D-NeRF transforms file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformsFile {
    #[serde(rename = "camera_angle_x")]
    pub camera_angle_x: f32,
    pub frames: Vec<Frame>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneSplit {
    Train,
    Val,
    Test,
}

impl SceneSplit {
    pub fn name(self) -> &'static str {
        match self {
            SceneSplit::Train => "train",
            SceneSplit::Val => "val",
            SceneSplit::Test => "test",
        }
    }
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "train" => Some(Self::Train),
            "val" => Some(Self::Val),
            "test" => Some(Self::Test),
            _ => None,
        }
    }
}

/// One D-NeRF scene (category) with its three splits loaded in memory.
#[derive(Debug, Clone)]
pub struct DNeRFDataset {
    pub category: String,
    pub root: PathBuf,
    pub camera_angle_x: f32,
    pub train: Vec<Frame>,
    pub val: Vec<Frame>,
    pub test: Vec<Frame>,
}

impl DNeRFDataset {
    pub fn split(&self, split: SceneSplit) -> &[Frame] {
        match split {
            SceneSplit::Train => &self.train,
            SceneSplit::Val => &self.val,
            SceneSplit::Test => &self.test,
        }
    }

    /// Absolute image path for a frame (appends `.png`).
    pub fn image_path(&self, f: &Frame) -> PathBuf {
        self.root.join(format!("{}.png", f.file_path))
    }

    /// Number of distinct temporal steps. The D-NeRF convention encodes time
    /// per frame; we canonicalize it to a monotonic t-axis for autoregressive
    /// rollouts by taking unique sorted `time` values.
    pub fn temporal_steps(&self) -> Vec<f32> {
        let mut ts: Vec<f32> = self
            .train
            .iter()
            .map(|f| f.time)
            .chain(self.val.iter().map(|f| f.time))
            .chain(self.test.iter().map(|f| f.time))
            .collect();
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ts.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
        ts
    }
}

/// Load all three splits of a single D-NeRF category from `data/<category>/`.
pub fn load_scene(data_root: &Path, category: &str) -> std::io::Result<DNeRFDataset> {
    let root = data_root.join(category);

    let read_split = |split: SceneSplit| -> std::io::Result<(Vec<Frame>, f32)> {
        let path = root.join(format!("transforms_{}.json", split.name()));
        let raw = std::fs::read_to_string(&path).map_err(|e| {
            std::io::Error::new(
                e.kind(),
                format!("reading {}: {e}", path.display()),
            )
        })?;
        let parsed: TransformsFile = serde_json::from_str(&raw).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("json parse: {e}"))
        })?;
        Ok((parsed.frames, parsed.camera_angle_x))
    };

    let (train, cax_train) = read_split(SceneSplit::Train)?;
    let (val, cax_val) = read_split(SceneSplit::Val)?;
    let (test, cax_test) = read_split(SceneSplit::Test)?;

    // camera_angle_x is identical across splits (verified: 0.6911...); sanity-assert.
    let camera_angle_x = cax_train;
    debug_assert!((cax_train - cax_val).abs() < 1e-4 && (cax_train - cax_test).abs() < 1e-4,
        "camera_angle_x differs across splits for {category}");

    Ok(DNeRFDataset {
        category: category.to_string(),
        root,
        camera_angle_x,
        train,
        val,
        test,
    })
}

/// Enumerate every D-NeRF category present under `data_root` (sorted).
pub fn list_categories(data_root: &Path) -> std::io::Result<Vec<String>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(data_root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                // Must have at least transforms_train.json to count.
                if entry.path().join("transforms_train.json").exists() {
                    out.push(name.to_string());
                }
            }
        }
    }
    out.sort();
    Ok(out)
}
