//! Minimal PNG decoding into packed HWC u8 tensors using the `png` crate.
//! We do not depend on any image resampling here; frames are loaded at their
//! native resolution and handed to PyTorch as-is.

use png::{BitDepth, ColorType, Decoder};
use std::fs::File;
use std::path::Path;

/// Packed image: HxWxC row-major u8.
#[derive(Debug, Clone)]
pub struct ImageU8 {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    /// length = (width * height * channels)
    pub data: Vec<u8>,
}

impl ImageU8 {
    pub fn shape(&self) -> (u32, u32, u32) {
        (self.height, self.width, self.channels)
    }
}

/// Decode a PNG into HWC u8. RGB is kept RGB; RGBA is kept RGBA.
pub fn load_image_u8(path: &Path) -> std::io::Result<ImageU8> {
    let file = File::open(path)?;
    let mut decoder = Decoder::new(file);
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info().map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, format!("png decode: {e}"))
    })?;

    // Allocate the output buffer using the read-time info.
    let info = reader.info();
    let width = info.width;
    let height = info.height;
    let channels = match info.color_type {
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Indexed => 3, // transformed to RGB by normalize_to_color8
    };

    let mut buf = vec![0u8; reader.output_buffer_size()];
    let frame = reader
        .next_frame(&mut buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("png read: {e}")))?;
    buf.truncate(frame.buffer_size());

    // Sanity: bit depth must be 8 after normalization.
    debug_assert_eq!(frame.bit_depth, BitDepth::Eight);

    Ok(ImageU8 {
        width,
        height,
        channels: channels as u32,
        data: buf,
    })
}
