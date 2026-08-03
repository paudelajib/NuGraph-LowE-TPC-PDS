"""NuGraph3 vertex decoder"""

from typing import Any

import torch
from torch import nn
from torch_geometric.data import Batch
from pytorch_lightning.loggers import TensorBoardLogger

from ....util import LogCoshLoss
from ..types import Data


class VertexDecoder(nn.Module):
    """
    NuGraph3 vertex decoder module.

    The network predicts normalized vertex coordinates.

    Normalization:
        x_norm = (x_cm - mean_x) / std_x
        y_norm = (y_cm - mean_y) / std_y
        z_norm = (z_cm - mean_z) / std_z

    The loss is computed in normalized coordinates.

    The stored prediction data["evt"].v is converted back to cm,
    so plotting DeltaX, DeltaY, DeltaZ still works in physical units.
    """

    def __init__(self, interaction_features: int):
        super().__init__()

        # ------------------------------------------------------------
        # Vertex normalization constants from training graph
        # mean = [-1.1677353382110596, -2.3839571475982666, 692.0067749023438]
        # std  = [211.81112670898438, 348.0730285644531, 402.8901062011719]
        # ------------------------------------------------------------
        vtx_mean_cm = torch.tensor(
            [
                -1.1677353382110596,
                -2.3839571475982666,
                692.0067749023438,
            ],
            dtype=torch.float,
        )

        vtx_std_cm = torch.tensor(
            [
                211.81112670898438,
                348.0730285644531,
                402.8901062011719,
            ],
            dtype=torch.float,
        )

        # Buffers move automatically with model.to(device),
        # but they are not trainable parameters.
        self.register_buffer("vtx_mean_cm", vtx_mean_cm.reshape(1, 3))
        self.register_buffer("vtx_std_cm", vtx_std_cm.reshape(1, 3))

        # Original NuGraph-style loss
        self.loss = LogCoshLoss()

        # Keep original temperature mechanism
        self.temp = nn.Parameter(torch.tensor(5.))

        # Network predicts normalized vertex coordinates
        self.net = nn.Linear(interaction_features, 3)

    def forward(self, data: Data, stage: str = None) -> dict[str, Any]:
        # True vertex in cm
        y_cm = data["evt"].y_vtx

        # True vertex in normalized coordinates
        y_norm = (y_cm - self.vtx_mean_cm) / self.vtx_std_cm

        # Predicted vertex in normalized coordinates
        x_norm = self.net(data["evt"].x)

        # Loss in normalized coordinate space
        w = (-1.0 * self.temp).exp()
        loss = w * self.loss(x_norm, y_norm) + self.temp

        # Convert prediction back to cm
        x_cm = x_norm * self.vtx_std_cm + self.vtx_mean_cm

        # Store physical prediction in cm
        data["evt"].v = x_cm

        if isinstance(data, Batch):
            data._slice_dict["evt"]["v"] = data["evt"].ptr
            inc = torch.zeros(data.num_graphs, device=data["evt"].x.device)
            data._inc_dict["evt"]["v"] = inc

        metrics = {}

        if stage:
            # Physical residuals in cm
            diff_cm = x_cm - y_cm

            xyz_abs_cm = diff_cm.abs().mean(dim=0)
            dr_cm = diff_cm.square().sum(dim=1).sqrt().mean()

            # Diagnostic residuals in normalized units
            diff_norm = x_norm - y_norm
            xyz_abs_norm = diff_norm.abs().mean(dim=0)
            dr_norm = diff_norm.square().sum(dim=1).sqrt().mean()

            metrics[f"vertex/loss-{stage}"] = loss

            # Main physical metrics
            metrics[f"vertex/x-resolution-{stage}"] = xyz_abs_cm[0]
            metrics[f"vertex/y-resolution-{stage}"] = xyz_abs_cm[1]
            metrics[f"vertex/z-resolution-{stage}"] = xyz_abs_cm[2]
            metrics[f"vertex/resolution-{stage}"] = dr_cm

            # Extra diagnostic normalized metrics
            metrics[f"vertex/x-resolution-norm-{stage}"] = xyz_abs_norm[0]
            metrics[f"vertex/y-resolution-norm-{stage}"] = xyz_abs_norm[1]
            metrics[f"vertex/z-resolution-norm-{stage}"] = xyz_abs_norm[2]
            metrics[f"vertex/resolution-norm-{stage}"] = dr_norm

        if stage == "train":
            metrics["temperature/vertex"] = self.temp

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
