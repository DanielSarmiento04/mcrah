"""MCRAH: Motion-Coherent Rigidity-Adaptive Hypergraph for drift-stable
Dynamic 3D Gaussian Splatting. CVPR 2027 target.

Package layout follows workflow.md:
    data/    - D-NeRF dataset wrappers consuming the Rust-preprocessed .npy.
    models/  - SIMGNN, DHGC, MCRAH adaptive hypergraph, and the full
               autoregressive network.
    gs/      - differentiable Gaussian splatting rasterization bridge.
    losses/  - L1+SSIM, relative L2, and noise-injected PDE regularization.
"""

__version__ = "0.1.0"
