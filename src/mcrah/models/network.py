"""MCRAH: the full autoregressive network (Phase 3 Step 7).

Integrates Stage 1 (SIMGNN dense propagation) and Stage 2 (DHGC far-field
attention) into the autoregressive rollout described in workflow.md:
  1. From the t=0 static Gaussian cloud, build (or load) the hypergraph + DHGC.
  2. At each step t -> t+1:
        h = SIMGNN(cloud_t, hg)          # dense local propagation
        h = DHGC(h, dilated)             # far-field attention
        Δpos, Δrot = heads(h)
        cloud_{t+1} = apply_offsets(cloud_t, Δpos, Δrot)
  3. Render cloud_{t+1} and supervise against the GT image (losses/).

When MCRAH is enabled, the hypergraph membership is recomputed from the
current deformation state before propagation, making the topology evolve
with the scene (novel contribution).

The decoupled two-stage training (rules.md Rule 8) is handled at the optimizer
level via the ``stage`` config; here we expose a single module and the caller
freezes the appropriate submodules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import Config
from ..gs.gaussian import GaussianCloud, apply_offsets
from .simgnn import SIMGNN
from .dhgc import DHGC
from .hypergraph import (
    Hypergraph, DilatedStructure,
    build_hypergraph_from_features,
)
from .mcrah import MCRAHCore, compute_centroids


@dataclass
class RolloutStep:
    """One predicted step of the autoregressive rollout."""
    cloud: GaussianCloud          # predicted cloud at this time
    delta_pos: torch.Tensor       # (N,3)
    delta_rot: Optional[torch.Tensor]  # (N,4) or None
    membership: Optional[torch.Tensor] = None  # (N,E) MCRAH soft membership


class MCRAH(nn.Module):
    """The complete MCRAH model.

    Holds the static (t=0) cloud, the hypergraph structures, and the two-stage
    network. ``rollout`` produces a sequence of predicted clouds, which the
    training loop renders and supervises.
    """

    def __init__(self, cfg: Config, cloud: GaussianCloud,
                 hypergraph: Optional[Hypergraph] = None,
                 dilated: Optional[DilatedStructure] = None):
        super().__init__()
        self.cfg = cfg
        self.simgnn = SIMGNN(cfg)
        self.dhgc = DHGC(cfg)

        # Static reference cloud (t=0). Stored as a buffer-like attribute; not
        # a learnable parameter (the static 3DGS fit is a separate Phase 1 step).
        self.register_cloud(cloud)

        # Build or accept the graph structures.
        if hypergraph is None:
            hypergraph = build_hypergraph_from_features(
                cloud.means.detach(), seed=cfg.hypergraph.seed
            )
        self.hypergraph = hypergraph
        self.dilated = dilated  # may be None in Stage-1-only training

        cluster_id = self.hypergraph.assignment.to(cloud.means.device)
        self.register_buffer("cluster_id", cluster_id, persistent=False)

        # MCRAH: novel motion-coherent rigidity-adaptive hypergraph.
        self.use_mcrah = cfg.hypergraph.adaptive
        if self.use_mcrah:
            self.mcrah = MCRAHCore(
                assignment_0=cluster_id,
                n_clusters=self.hypergraph.n_edges,
                feat_dim=cfg.model.feat_dim,
                tau=cfg.hypergraph.mcrae_tau,
                adaptive=True,
            )
            centroids_0 = compute_centroids(cloud.means.detach(),
                                             cluster_id, self.hypergraph.n_edges)
            self.mcrah.set_reference(cloud.means.detach(), centroids_0)
        else:
            self.mcrah = None

    # -- cloud management -------------------------------------------------- #
    def register_cloud(self, cloud: GaussianCloud) -> None:
        """Store the static cloud's tensors as non-persistent buffers so they
        move with .to(device) but aren't saved in the checkpoint."""
        for name, t in [("means", cloud.means), ("scales", cloud.scales),
                        ("rotations", cloud.rotations),
                        ("opacities", cloud.opacities), ("sh", cloud.sh)]:
            self.register_buffer(f"_cloud_{name}", t.clone(), persistent=False)

    @property
    def cloud(self) -> GaussianCloud:
        return GaussianCloud(
            means=self._cloud_means, scales=self._cloud_scales,
            rotations=self._cloud_rotations, opacities=self._cloud_opacities,
            sh=self._cloud_sh,
        )

    # -- forward ----------------------------------------------------------- #
    def step(self, cloud: GaussianCloud, time: torch.Tensor) -> RolloutStep:
        """Predict one autoregressive step t -> t+1.

        Ordering follows workflow.md Phase 3 Step 7: dense local propagation
        (SIMGNN.encode) feeds the far-field attention (DHGC), and the offset
        heads are applied to the *refined* features. When DHGC is absent
        (Stage-1-only training) the heads act on the SIMGNN features directly.

        When MCRAH is enabled, the hypergraph membership is recomputed from
        the current deformation state before propagation, making the topology
        evolve with the scene (novel contribution).
        """
        dev = cloud.means.device

        if self.use_mcrah and self.mcrah is not None:
            self.mcrah = self.mcrah.to(dev)
            membership = self.mcrah.update_membership(cloud.means)
            # Propagate through the adaptive (soft) hypergraph.
            h = self.simgnn.encoder(cloud, self.cluster_id, time)
            for blk in self.simgnn.blocks:
                h = blk._propagate_mcrah(h, self.mcrah.hypergraph)
            h = self.simgnn.final_norm(h)
        else:
            membership = None
            hg = self.hypergraph.to(dev)
            h = self.simgnn.encode(cloud, hg, self.cluster_id, time)

        if self.dilated is not None:
            h = self.dhgc(h, self.dilated.to(dev))
        delta_pos, delta_rot = self.simgnn.heads(h)
        new_cloud = apply_offsets(cloud, delta_pos, delta_rot)
        return RolloutStep(cloud=new_cloud, delta_pos=delta_pos,
                          delta_rot=delta_rot, membership=membership)

    def rollout(self, times: List[torch.Tensor],
                noise_injector=None) -> List[RolloutStep]:
        """Autoregressive rollout over ``len(times)`` steps.

        Args:
            times: list of time tensors, one per step.
            noise_injector: optional :class:`NoiseInjector` applied to the input
                cloud state at each step (rules.md Rule 4).
        Returns: list of :class:`RolloutStep`, length == len(times).
        """
        steps: List[RolloutStep] = []
        cloud = self.cloud
        for t in times:
            cur = cloud
            if (self.training and noise_injector is not None):
                cur = self._inject_noise(cloud, noise_injector)
            step = self.step(cur, t)
            steps.append(step)
            cloud = step.cloud
        return steps

    def _inject_noise(self, cloud: GaussianCloud, noise) -> GaussianCloud:
        """Add Gaussian noise to the input state (means only, to start)."""
        noisy_means = noise(cloud.means)
        return GaussianCloud(
            means=noisy_means, scales=cloud.scales, rotations=cloud.rotations,
            opacities=cloud.opacities, sh=cloud.sh,
        )

    # -- stage control (rules.md Rule 8 decoupled training) --------------- #
    def configure_stage(self, stage: str) -> None:
        """Freeze submodules for decoupled two-stage training.
        ``"dense"`` trains SIMGNN only (DHGC frozen).
        ``"farfield"`` trains DHGC only (SIMGNN frozen).
        ``"joint"`` trains everything.
        """
        if stage not in ("dense", "farfield", "joint"):
            raise ValueError(f"unknown stage: {stage}")
        for p in self.simgnn.parameters():
            p.requires_grad = stage in ("dense", "joint")
        for p in self.dhgc.parameters():
            p.requires_grad = stage in ("farfield", "joint")

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        dev = self._cloud_means.device
        if self.hypergraph is not None:
            self.hypergraph = self.hypergraph.to(dev)
        if self.dilated is not None:
            self.dilated = self.dilated.to(dev)
        self.cluster_id = self.cluster_id.to(dev)
        return self
