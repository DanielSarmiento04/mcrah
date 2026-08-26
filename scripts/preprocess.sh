#!/usr/bin/env bash
# Preprocess D-NeRF data with the Rust binary: metadata + hypergraph + images.
# Usage: scripts/preprocess.sh [category] [--images]
#   category: single D-NeRF category, or "all" (default: all)
set -euo pipefail

cd "$(dirname "$0")/.."

CAT="${1:-all}"
EXTRA=""
if [[ "${2:-}" == "--images" ]]; then
  EXTRA="--images"
fi

if [[ "$CAT" == "all" ]]; then
  cargo run --release --manifest-path rust/Cargo.toml --bin preprocess -- --data-root ./data --out ./processed --cluster $EXTRA
else
  cargo run --release --manifest-path rust/Cargo.toml --bin preprocess -- --data-root ./data --category "$CAT" --out ./processed --cluster $EXTRA
fi
