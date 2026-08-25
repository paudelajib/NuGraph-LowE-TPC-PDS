"""NuGraph3 geometry-aware direction decoder.

Two modes, selected with AXIS_MODE:

"weighted_pca" (default, vertex-independent):
    The axis-only loss (1 - cos^2) is sign-symmetric, so no vertex is needed
    to define the axis. Candidate geometry is built around the attention-
    weighted spacepoint centroid instead of the vertex:

        c      = sum_i w_i sp_pos_i          (weighted centroid)
        u_i    = normalize(sp_pos_i - c)
        M      = sum_i w_i u_i u_i^T         (weighted orientation tensor)
        loss   = 1 - t^T M t                 (t = true axis; trace(M) = 1)

    The predicted axis is the principal eigenvector of M. The predicted
    vertex (evt.v_pred, if present) enters only as a soft attention feature
    and to resolve the axis sign — it never anchors the geometry, so a few-cm
    vertex error cannot corrupt the axis estimate.

"vertex_rays" (previous behavior):
    Candidate directions are rays from a vertex reference to each
    spacepoint, u_i = normalize(sp_pos_i - vertex), combined with learned
    attention weights. Accurate with the true vertex, but degrades badly
    when the vertex error is comparable to the vertex-to-spacepoint
    distance (~11 cm in this sample vs ~5.5 cm median v_pred error).
"""

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import softmax

from pytorch_lightning.loggers import TensorBoardLogger

from ..types import Data


class DirectionDecoder(nn.Module):
    """
    Geometry-aware event-level direction decoder.

    AXIS_MODE:
      "weighted_pca"  vertex-independent weighted orientation tensor (default)
      "vertex_rays"   previous vertex-anchored attention decoder

    USE_TRUE_VERTEX_REFERENCE only affects "vertex_rays" mode:
      True  -> anchor rays at evt.y_vtx (diagnostic ceiling)
      False -> anchor rays at evt.v_pred / evt.v

    USE_VPRED_FEATURES only affects "weighted_pca" mode: when evt.v_pred is
    present, the unit vector and log distance from v_pred to each spacepoint
    are appended to the attention features as a soft prior.
    """

    AXIS_MODE = "weighted_pca"
    USE_TRUE_VERTEX_REFERENCE = True
    USE_VPRED_FEATURES = True

    def __init__(self, interaction_features: int):
        super().__init__()

        self.evt_norm = nn.LayerNorm(interaction_features)

        # LazyLinear lets PyTorch infer the input dimension at first forward pass.
        # Input will be:
        #   normalized sp.x
        #   normalized evt.x for the parent event
        #   unit vector from reference point to sp: ux, uy, uz
        #   log distance from reference point to sp
        #   (weighted_pca + USE_VPRED_FEATURES: also unit vector and log
        #    distance from evt.v_pred to sp)
        self.score_net = nn.Sequential(
            nn.LazyLinear(64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def _get_node_batch(self, data: Data, node: str) -> torch.Tensor:
        """Return PyG batch vector for a node type."""
        store = data[node]

        if hasattr(store, "batch"):
            return store.batch

        if hasattr(store, "ptr"):
            ptr = store.ptr
            counts = ptr[1:] - ptr[:-1]
            return torch.repeat_interleave(
                torch.arange(len(counts), device=ptr.device),
                counts,
            )

        raise RuntimeError(f"Could not find batch or ptr for node type {node}")

    def _get_vertex_reference(self, data: Data) -> torch.Tensor:
        """Return vertex reference for each event (vertex_rays mode)."""
        evt = data["evt"]

        if self.USE_TRUE_VERTEX_REFERENCE:
            if not hasattr(evt, "y_vtx"):
                raise RuntimeError(
                    "USE_TRUE_VERTEX_REFERENCE = True, but evt.y_vtx is missing."
                )
            return evt.y_vtx

        if hasattr(evt, "v_pred"):
            return evt.v_pred

        if hasattr(evt, "v"):
            return evt.v.detach()

        if hasattr(evt, "vertex"):
            return evt.vertex.detach()

        raise RuntimeError(
            "Predicted vertex not found. Run with --vertex, provide evt.v_pred, "
            "or temporarily set USE_TRUE_VERTEX_REFERENCE = True."
        )

    def _get_sign_reference(self, evt) -> torch.Tensor | None:
        """Reference point used only to orient the axis (weighted_pca mode).

        Prefer the predicted vertex so deployment never needs truth; fall
        back to the true vertex for truth-only samples.
        """
        if hasattr(evt, "v_pred"):
            return evt.v_pred.float()
        if hasattr(evt, "v"):
            return evt.v.detach().float()
        if hasattr(evt, "y_vtx"):
            return evt.y_vtx.float()
        return None

    def forward(self, data: Data, stage: str = None) -> tuple[torch.Tensor, dict[str, Any]]:
        evt = data["evt"]
        sp = data["sp"]

        if not hasattr(sp, "pos"):
            raise RuntimeError(
                "data['sp'].pos is missing. Need spacepoint positions for "
                "geometry-aware direction decoding."
            )

        if not hasattr(evt, "y_dir"):
            raise RuntimeError("data['evt'].y_dir is missing.")

        # ------------------------------------------------------------
        # Event and spacepoint bookkeeping
        # ------------------------------------------------------------
        sp_batch = self._get_node_batch(data, "sp")
        n_evt = evt.x.shape[0]

        sp_pos = sp.pos[:, :3].float()
        sp_x = sp.x.float()

        sp_x_norm = F.layer_norm(sp_x, (sp_x.shape[-1],))
        evt_x_norm = self.evt_norm(evt.x.float())

        target = F.normalize(evt.y_dir.float(), dim=-1, eps=1.0e-8)

        if self.AXIS_MODE == "weighted_pca":
            # --------------------------------------------------------
            # Pass 1 geometry: uniform centroid per event
            # --------------------------------------------------------
            ones = torch.ones(sp_pos.shape[0], device=sp_pos.device, dtype=sp_pos.dtype)
            cnt = torch.zeros(n_evt, device=sp_pos.device, dtype=sp_pos.dtype)
            cnt.index_add_(0, sp_batch, ones)
            cnt = cnt.clamp_min(1.0)

            c0 = torch.zeros(n_evt, 3, device=sp_pos.device, dtype=sp_pos.dtype)
            c0.index_add_(0, sp_batch, sp_pos)
            c0 = c0 / cnt.unsqueeze(-1)

            rel0 = sp_pos - c0[sp_batch]
            dist = rel0.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            rel0_hat = rel0 / dist
            log_dist = torch.log1p(dist)

            feats = [sp_x_norm, evt_x_norm[sp_batch], rel0_hat, log_dist]

            if self.USE_VPRED_FEATURES and hasattr(evt, "v_pred"):
                relv = sp_pos - evt.v_pred.float()[sp_batch]
                dv = relv.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
                feats += [relv / dv, torch.log1p(dv)]

            score = self.score_net(torch.cat(feats, dim=-1)).squeeze(-1)
            weight = softmax(score, sp_batch)

            # --------------------------------------------------------
            # Pass 2 geometry: weighted centroid and orientation tensor
            # --------------------------------------------------------
            # sum_i w_i = 1 per event, so no division needed.
            c = torch.zeros(n_evt, 3, device=sp_pos.device, dtype=weight.dtype)
            c.index_add_(0, sp_batch, weight.unsqueeze(-1) * sp_pos)

            rel = sp_pos - c[sp_batch]
            d = rel.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            u = rel / d

            outer = weight.unsqueeze(-1).unsqueeze(-1) * (
                u.unsqueeze(-1) * u.unsqueeze(-2)
            )
            m = torch.zeros(n_evt, 3, 3, device=sp_pos.device, dtype=weight.dtype)
            m.index_add_(0, sp_batch, outer)

            # Rayleigh-quotient axis loss: trace(M) = 1, so
            # t^T M t = sum_i w_i cos^2(theta_i) and loss = weighted mean sin^2.
            # Equals 1 - cos^2 in the rank-1 limit; no eigh in the loss path.
            rayleigh = torch.einsum("ei,eij,ej->e", target, m, target)
            rayleigh = rayleigh.clamp(0.0, 1.0)
            loss = (1.0 - rayleigh).mean()

            # Predicted axis for outputs/metrics: principal eigenvector of M.
            # eigh gradients are unstable near degenerate spectra, so keep it
            # out of the loss path entirely.
            with torch.no_grad():
                _, evecs = torch.linalg.eigh(m)
                pred = F.normalize(evecs[..., -1], dim=-1, eps=1.0e-8)

                sign_ref = self._get_sign_reference(evt)
                if sign_ref is not None:
                    outward = c - sign_ref
                    flip = (pred * outward).sum(dim=-1, keepdim=True) < 0
                    pred = torch.where(flip, -pred, pred)

            dist_metric = d
        elif self.AXIS_MODE == "vertex_rays":
            vertex = self._get_vertex_reference(data).float()

            rel = sp_pos - vertex[sp_batch]
            dist = rel.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            rel_hat = rel / dist
            log_dist = torch.log1p(dist)

            score_input = torch.cat(
                [sp_x_norm, evt_x_norm[sp_batch], rel_hat, log_dist],
                dim=-1,
            )

            score = self.score_net(score_input).squeeze(-1)
            weight = softmax(score, sp_batch)

            weighted_vec = weight.unsqueeze(-1) * rel_hat

            pred = torch.zeros(
                n_evt,
                3,
                device=weighted_vec.device,
                dtype=weighted_vec.dtype,
            )
            pred.index_add_(0, sp_batch, weighted_vec)
            pred = F.normalize(pred, dim=-1, eps=1.0e-8)

            cos_loss = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
            loss = (1.0 - cos_loss.pow(2)).mean()

            dist_metric = dist
        else:
            raise RuntimeError(f"Unknown AXIS_MODE: {self.AXIS_MODE}")

        cos = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
        axis_cos = cos.abs()

        evt.d = pred

        if isinstance(data, Batch):
            data._slice_dict["evt"]["d"] = data["evt"].ptr
            inc = torch.zeros(data.num_graphs, device=evt.x.device)
            data._inc_dict["evt"]["d"] = inc

        metrics: dict[str, Any] = {}

        if stage:
            angle_deg = torch.rad2deg(torch.acos(cos))
            axis_angle_deg = torch.rad2deg(torch.acos(axis_cos))
            diff = pred - target
            xyz_abs = diff.abs().mean(dim=0)
            axis_sign = torch.where(
                cos[:, None] >= 0,
                torch.ones_like(cos[:, None]),
                -torch.ones_like(cos[:, None]),
            )
            axis_diff = pred - axis_sign * target
            axis_xyz_abs = axis_diff.abs().mean(dim=0)

            # Main direction metrics
            metrics[f"direction/loss-{stage}"] = loss
            metrics[f"direction/loss-cosine-{stage}"] = loss
            metrics[f"direction/cosine-{stage}"] = cos.mean()
            metrics[f"direction/axis-cosine-{stage}"] = axis_cos.mean()
            metrics[f"direction/angle-deg-{stage}"] = angle_deg.mean()
            metrics[f"direction/angle-deg-median-{stage}"] = angle_deg.median()
            metrics[f"direction/axis-angle-deg-{stage}"] = axis_angle_deg.mean()
            metrics[f"direction/axis-angle-deg-median-{stage}"] = axis_angle_deg.median()

            # Component diagnostics
            metrics[f"direction/ux-resolution-{stage}"] = xyz_abs[0]
            metrics[f"direction/uy-resolution-{stage}"] = xyz_abs[1]
            metrics[f"direction/uz-resolution-{stage}"] = xyz_abs[2]
            metrics[f"direction/axis-ux-resolution-{stage}"] = axis_xyz_abs[0]
            metrics[f"direction/axis-uy-resolution-{stage}"] = axis_xyz_abs[1]
            metrics[f"direction/axis-uz-resolution-{stage}"] = axis_xyz_abs[2]

            # Attention diagnostics
            # max weight close to 1 means decoder selects one/few spacepoints.
            # max weight small means it averages many spacepoints.
            max_weight = torch.zeros(
                n_evt,
                device=weight.device,
                dtype=weight.dtype,
            )
            max_weight.scatter_reduce_(
                0,
                sp_batch,
                weight,
                reduce="amax",
                include_self=False,
            )

            metrics[f"direction/attention-max-weight-{stage}"] = max_weight.mean()

            # Geometry diagnostics
            metrics[f"direction/sp-dist-mean-{stage}"] = dist_metric.mean()
            metrics[f"direction/evtx-feature-std-{stage}"] = evt.x.std(dim=0).mean()
            metrics[f"direction/evtx-norm-{stage}"] = evt.x.norm(dim=1).mean()

        return loss, metrics

    def on_epoch_end(
        self,
        logger: TensorBoardLogger,
        stage: str,
        epoch: int,
    ) -> None:
        pass
