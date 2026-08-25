"""NuGraph3 geometry-aware direction decoder.

Spacepoint-attention direction decoder.

Instead of predicting direction from evt.x alone, this decoder builds candidate
direction vectors from spacepoint positions relative to the vertex:

    u_i = normalize(sp_pos_i - vertex)

Then it learns attention weights over spacepoints and predicts

    u_pred = normalize(sum_i w_i u_i)

This tests whether geometry in the graph can predict the G4 lepton direction.
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

    Diagnostic mode:
      USE_TRUE_VERTEX_REFERENCE = True

    This uses evt.v_pred as the reference point. That is intentional for the
    first test. It tells us whether the direction information is present in
    the graph geometry at all.

    Later, after this works, switch to:
      USE_TRUE_VERTEX_REFERENCE = True

    Then it will use evt.v from the vertex decoder.
    """

    USE_TRUE_VERTEX_REFERENCE = True

    def __init__(self, interaction_features: int):
        super().__init__()

        self.evt_norm = nn.LayerNorm(interaction_features)

        # LazyLinear lets PyTorch infer the input dimension at first forward pass.
        # Input will be:
        #   normalized sp.x
        #   normalized evt.x for the parent event
        #   unit vector from vertex to sp: ux, uy, uz
        #   log distance from vertex to sp
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
        """Return vertex reference for each event."""
        evt = data["evt"]

        if self.USE_TRUE_VERTEX_REFERENCE:
            if not hasattr(evt, "v_pred"):
                raise RuntimeError(
                    "USE_TRUE_VERTEX_REFERENCE = True, but evt.v_pred is missing."
                )
            return evt.v_pred

        # Practical mode: use predicted vertex from VertexDecoder.
        # This requires running with --vertex --direction, and the vertex decoder
        # must run before the direction decoder.
        if hasattr(evt, "v"):
            return evt.v.detach()

        if hasattr(evt, "vertex"):
            return evt.vertex.detach()

        raise RuntimeError(
            "Predicted vertex not found. Run with --vertex, or temporarily set "
            "USE_TRUE_VERTEX_REFERENCE = True for diagnostic mode."
        )

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

        vertex = self._get_vertex_reference(data).float()

        # ------------------------------------------------------------
        # Geometry: vector from vertex to each spacepoint
        # ------------------------------------------------------------
        rel = sp_pos - vertex[sp_batch]
        dist = rel.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        rel_hat = rel / dist

        # ------------------------------------------------------------
        # Features for attention score
        # ------------------------------------------------------------
        sp_x_norm = F.layer_norm(sp_x, (sp_x.shape[-1],))
        evt_x_norm = self.evt_norm(evt.x.float())

        # log distance is safer than raw distance because detector coordinates
        # may be large.
        log_dist = torch.log1p(dist)

        score_input = torch.cat(
            [
                sp_x_norm,
                evt_x_norm[sp_batch],
                rel_hat,
                log_dist,
            ],
            dim=-1,
        )

        score = self.score_net(score_input).squeeze(-1)

        # Softmax separately within each event.
        weight = softmax(score, sp_batch)

        # Weighted sum of candidate directions.
        weighted_vec = weight.unsqueeze(-1) * rel_hat

        pred = torch.zeros(
            n_evt,
            3,
            device=weighted_vec.device,
            dtype=weighted_vec.dtype,
        )
        pred.index_add_(0, sp_batch, weighted_vec)

        pred = F.normalize(pred, dim=-1, eps=1.0e-8)

        target = F.normalize(evt.y_dir.float(), dim=-1, eps=1.0e-8)

        cos = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
        axis_cos = cos.abs()
        loss = (1.0 - cos.pow(2)).mean()

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
            metrics[f"direction/sp-dist-mean-{stage}"] = dist.mean()
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
