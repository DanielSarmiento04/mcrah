"""Model definitions for MCRAH.

Implements the architecture from workflow.md Phase 2/3:
  * :class:`Hypergraph` / :class:`DilatedStructure` - graph structures.
  * :class:`SIMGNN`  - Stage 1, dense local-field propagation.
  * :class:`DHGC`    - Stage 2, dilated far-field attention.
  * :class:`MCRAH`   - the full autoregressive network integrating both stages.
  * :class:`MCRAHCore` - the adaptive motion-coherent hypergraph component.
"""

from .hypergraph import (
    Hypergraph, DilatedStructure,
    load_hypergraph, load_dilated_structure,
    build_hypergraph_from_features,
)
from .simgnn import SIMGNN, FeatureEncoder, HGNNBlock, OffsetHeads
from .dhgc import DHGC, DHGCBlock, DilatedAttention
from .network import MCRAH, RolloutStep
from .mcrah import (
    MCRAHCore, MCRAHGate, AdaptiveHypergraph,
    RigidityLoss, TopologySmoothnessLoss, compute_centroids,
)

__all__ = [
    "Hypergraph", "DilatedStructure",
    "load_hypergraph", "load_dilated_structure",
    "build_hypergraph_from_features",
    "SIMGNN", "FeatureEncoder", "HGNNBlock", "OffsetHeads",
    "DHGC", "DHGCBlock", "DilatedAttention",
    "MCRAH", "RolloutStep",
    "MCRAHCore", "MCRAHGate", "AdaptiveHypergraph",
    "RigidityLoss", "TopologySmoothnessLoss", "compute_centroids",
]
