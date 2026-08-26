//! Minimal numpy `.npy` writer (v1.0, little-endian) so Rust-preprocessed
//! tensors load directly with `np.load` / `torch.from_numpy`.
//!
//! Format:
//!   magic = b"\x93NUMPY" (6 bytes)
//!   version (1 byte major=1, 1 byte minor=0)
//!   header_len (2 bytes, little endian, v1.0)
//!   header  (ASCII, padded with spaces, ending in '\n')
//!   raw data (row-major C-order)

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

fn dtype_of<T>() -> &'static str {
    std::any::type_name::<T>()
        .rsplit("::")
        .next()
        .unwrap_or("V")
}

/// Write a 1-D f32 array.
pub fn save_f32_1d(path: &Path, data: &[f32]) -> std::io::Result<()> {
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}),}}",
        data.len()
    );
    write_npy(path, &header, 4, bytemuck_cast_f32(data))
}

/// Write a 2-D f32 array (rows, cols), row-major.
pub fn save_f32_2d(path: &Path, rows: usize, cols: usize, data: &[f32]) -> std::io::Result<()> {
    debug_assert_eq!(data.len(), rows * cols);
    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}),}}",
        rows, cols
    );
    write_npy(path, &header, 4, bytemuck_cast_f32(data))
}

/// Write a 1-D u8 array.
pub fn save_u8_1d(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let header = format!(
        "{{'descr': '|u1', 'fortran_order': False, 'shape': ({}),}}",
        data.len()
    );
    write_npy(path, &header, 1, data)
}

/// Write a 3-D u8 array (h, w, c), row-major. Used for image tensors.
pub fn save_u8_3d(path: &Path, h: usize, w: usize, c: usize, data: &[u8]) -> std::io::Result<()> {
    debug_assert_eq!(data.len(), h * w * c);
    let header = format!(
        "{{'descr': '|u1', 'fortran_order': False, 'shape': ({}, {}, {}),}}",
        h, w, c
    );
    write_npy(path, &header, 1, data)
}

/// Write a 2-D i64 array, e.g. hyperedge incidence / index tensors.
pub fn save_i64_2d(path: &Path, rows: usize, cols: usize, data: &[i64]) -> std::io::Result<()> {
    debug_assert_eq!(data.len(), rows * cols);
    let header = format!(
        "{{'descr': '<i8', 'fortran_order': False, 'shape': ({}, {}),}}",
        rows, cols
    );
    write_npy(path, &header, 8, bytemuck_cast_i64(data))
}

fn write_npy(path: &Path, header: &str, _item_bytes: usize, raw: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = BufWriter::new(File::create(path)?);

    // Magic + version
    f.write_all(b"\x93NUMPY")?;
    f.write_all(&[1u8, 0u8])?;

    // Header is padded with spaces so that total (10 + header_len) % 16 == 0,
    // matching numpy's alignment.
    let mut header_bytes = header.as_bytes().to_vec();
    // ensure trailing newline
    if !header_bytes.ends_with(b"\n") {
        header_bytes.push(b'\n');
    }
    // pad to make (10 + len) divisible by 16
    while (10 + header_bytes.len()) % 16 != 0 {
        header_bytes.insert(header_bytes.len() - 1, b' ');
    }
    let hlen = header_bytes.len() as u16;
    f.write_all(&hlen.to_le_bytes())?;
    f.write_all(&header_bytes)?;
    f.write_all(raw)?;
    f.flush()?;
    Ok(())
}

// ---- bytemuck-free reinterpret casts (safe because we own the slices) ----

fn bytemuck_cast_f32(v: &[f32]) -> &[u8] {
    let _ = dtype_of::<f32>();
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<f32>())
    }
}

fn bytemuck_cast_i64(v: &[i64]) -> &[u8] {
    let _ = dtype_of::<i64>();
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<i64>())
    }
}
