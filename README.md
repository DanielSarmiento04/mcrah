# MCRAH: Motion-Coherent Rigidity-Adaptive Hypergraph for Dynamic 3D Gaussian Splatting

> Targeting CVPR 2027. Real-time dynamic novel-view synthesis from a static 3D
> Gaussian Splatting substrate, driven by an adaptive hypergraph neural network
> whose topology evolves with the scene, with noise-injected PDE regularization
> against autoregressive error drift.

## Novel Contribution: MCRAH

The baseline architecture (cosine-clustering hypergraph + DHGC far-field
attention) is from DVHGNN (Li et al., CVPR 2025), a 2D image recognition
backbone. Our contribution is **MCRAH — the Motion-Coherent Rigidity-Adaptive
Hypergraph**, which addresses the core limitation of the static hypergraph:
cluster membership is fixed at t=0 and cannot adapt as rigid body parts split
or merge during motion.

MCRAH adds three components absent from the literature:

1. **Motion-coherent topology evolution** — a learned gate recomputes a soft
   membership matrix M (N, E) at each autoregressive step from the current
   deformation state, not from static appearance similarity. The hypergraph
   structure evolves with the scene.

2. **Rigidity prior loss** — within each soft hyperedge, per-node displacement is
   pulled toward the cluster-centroid displacement (quasi-rigid body
   assumption). A hyperedge represents a quasi-rigid part; the loss penalizes
   intra-cluster deformation deviation.

3. **Topology temporal smoothness** — membership reassignment is regularized so
   the hyperedge structure does not jump abruptly between time steps.

The soft membership replaces the hard incidence matrix in the Θ propagation
operator, making the topology *differentiable* and trainable end-to-end through
the autoregressive rollout. Ablation: set `hypergraph.adaptive=False` to revert
to the static DVHGNN baseline.

## Overview

DVHGNN predicts the physical deformation of a static 3D Gaussian Splatting
(3DGS) scene over time via an autoregressive graph network, with MCRAH
providing the adaptive topology:

1. **Static init** — fit a 3DGS cloud to the t=0 frames (Phase 1).
2. **Hypergraph** — cluster Gaussians into hyperedges by cosine similarity;
   build multi-scale dilated neighbor sets for far-field attention (Phase 2).
3. **Autoregressive rollout** — at each step:
   - `h = SIMGNN.encode(cloud_t, hg)` — dense local-field propagation
   - `h = DHGC(h, dilated)` — dilated far-field attention (Stage 2)
   - `Δpos, Δrot = heads(h)` — predict physical offsets
   - `cloud_{t+1} = apply_offsets(cloud_t, Δpos, Δrot)`
4. **Differentiable rendering** — render `cloud_{t+1}` at the target camera,
   supervise with L1+SSIM, and backprop through the rasterizer to the offsets.
5. **Error mitigation** — relative-L2 regularization + Gaussian noise injection
   on the input state + PDE temporal smoothness, combatting rollout drift.

## Architecture

```
src/mcrah/
├── config.py              Central dataclass config (all architectural knobs)
├── gs/                    Gaussian primitives + differentiable rasterizer
│   ├── gaussian.py        Cloud bookkeeping, quaternion math, apply_offsets
│   └── rasterizer.py      Pure-torch EWA splatter (MPS/CPU) + CUDA adapter
├── models/
│   ├── hypergraph.py      Hypergraph + dilated structure (Θ propagation)
│   ├── simgnn.py          Stage 1: dense local-field propagation GNN
│   ├── dhgc.py            Stage 2: dilated far-field attention
│   ├── mcrah.py           Novel: motion-coherent rigidity-adaptive hypergraph core
│   └── network.py         Full autoregressive network + rollout
├── losses/
│   ├── rendering.py       L1+SSIM, PSNR, SSIM metric
│   └── pde.py            Relative-L2, noise injection, PDE smoothness
├── data/                  D-NeRF dataset (raw JSON+PNG or Rust .npy)
└── training/
    ├── static_init.py     Phase 1: static 3DGS fit at t=0
    ├── trainer.py         Phase 3: decoupled two-stage training
    └── evaluator.py       Phase 4: NVS metrics + rollout stability + ablations
```

## Installation

```bash
# Python (training + model)
pip install -e .

# Rust (fast data preprocessing + clustering)
cd rust && cargo build --release
```

## Usage

### Preprocess D-NeRF data (Rust)

```bash
# All categories: metadata + hypergraph + dilated sets
./scripts/preprocess.sh all

# Single category with image dump
./scripts/preprocess.sh trex --images
```

### Train (end-to-end pipeline)

```bash
# Quick smoke run on one category
python scripts/train.py --category trex --quick

# Full two-stage training
python scripts/train.py --category trex \
    --iterations 10000 --static-iters 5000

# All categories
python scripts/train.py --all --iterations 10000
```

### Evaluate

```bash
python scripts/evaluate.py --category trex \
    --checkpoint runs/trex/runs/checkpoint_joint.pt
```

## Testing

```bash
pytest tests/ -q
```

- `tests/test_smoke.py` — component tests (hypergraph, SIMGNN, DHGC, MCRAH
  rollout, rasterizer).
- `tests/test_training.py` — training/eval infrastructure tests (static init,
  gradient flow, differentiable rendering, evaluator).
- `tests/test_mcrah.py` — MCRAH novel module (gate, soft propagation,
  rigidity/topology losses, gradient flow, adaptive vs static ablation).

## Dataset

D-NeRF (dynamic NeRF) — 8 synthetic categories of non-rigid motion:
`bouncingballs, hellwarrior, hook, jumpingjacks, lego, mutant, standup, trex`.
Each has `train/val/test` splits with camera poses (`transforms_*.json`) and
RGB images. 
# mcrah
