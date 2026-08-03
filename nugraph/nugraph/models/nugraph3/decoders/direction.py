"""NuGraph3 direction decoder"""

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from pytorch_lightning.loggers import TensorBoardLogger

from ..types import Data


class DirectionDecoder(nn.Module):
    """
    NuGraph3 event-level lepton direction decoder.

    Predicts the outgoing primary lepton direction as a 3D unit vector:

        y_dir = [ux, uy, uz]

    This version uses direct cosine loss:

        loss = mean(1 - u_pred dot u_true)

    Perfect prediction:
        cosine = 1
        angle = 0 deg
        loss = 0

    Random direction:
        cosine ~ 0
        angle ~ 90 deg
        loss ~ 1
    """

    def __init__(self, interaction_features: int):
        super().__init__()

        # Network predicts raw 3-vector, then we normalize it
        self.net = nn.Linear(interaction_features, 3)

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        # Raw prediction
        x_raw = self.net(data["evt"].x)

        # Predicted unit direction
        x = F.normalize(x_raw, dim=-1, eps=1.0e-8)

        # True unit direction
        y = F.normalize(data["evt"].y_dir, dim=-1, eps=1.0e-8)

        # Cosine agreement
        cos = (x * y).sum(dim=-1).clamp(-1.0, 1.0)

        # Direct angular/cosine loss
        loss = (1.0 - cos).mean()

        # Store predicted direction
        data["evt"].d = x

        if isinstance(data, Batch):
            data._slice_dict["evt"]["d"] = data["evt"].ptr
            inc = torch.zeros(data.num_graphs, device=data["evt"].x.device)
            data._inc_dict["evt"]["d"] = inc

        metrics = {}

        if stage:
            diff = x - y
            xyz_abs = diff.abs().mean(dim=0)

            angle_deg = torch.rad2deg(torch.acos(cos))

            metrics[f"direction/loss-{stage}"] = loss
            metrics[f"direction/cosine-{stage}"] = cos.mean()
            metrics[f"direction/angle-deg-{stage}"] = angle_deg.mean()

            metrics[f"direction/ux-resolution-{stage}"] = xyz_abs[0]
            metrics[f"direction/uy-resolution-{stage}"] = xyz_abs[1]
            metrics[f"direction/uz-resolution-{stage}"] = xyz_abs[2]

        return loss, metrics

    def on_epoch_end(
        self,
        logger: TensorBoardLogger,
        stage: str,
        epoch: int,
    ) -> None:
        """
        NuGraph3 decoder end-of-epoch callback function.
        """
        pass
