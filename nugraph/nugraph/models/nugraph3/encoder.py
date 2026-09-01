"""NuGraph3 encoder"""
import torch
from pynuml.data import NuGraphData
from ...util import InputNorm

class Encoder(torch.nn.Module):
    """
    NuGraph3 encoder
    
    Args:
        in_features: Number of input node features
        planar_features: Number of planar node features
        nexus_feature: Number of nexus node features
        interaction_features: Number of interaction node features
    """
    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 ophit_features: int,
                 pmt_features: int,
                 flash_features: int,
                 use_optical: bool,
                 sp_features: int = 0):
        super().__init__()
        self.input_norm = InputNorm(in_features)
        self.planar_net = torch.nn.Linear(in_features, planar_features)
        self.nexus_features = nexus_features
        self.interaction_features = interaction_features

        # optional spacepoint/nexus input encoder
        # used when --3dfeatext provides [delta_T, chi2, x, y, z] as data["sp"].x
        if sp_features > 0:
            self.sp_input_norm = InputNorm(sp_features)
            self.sp_net = torch.nn.Sequential(
                torch.nn.Linear(sp_features, nexus_features),
                torch.nn.Mish())
        else:
            self.sp_input_norm = None
            self.sp_net = None

        # hardcode optical features pending redesign
        #
        # These carry an InputNorm for the same reason the hit and spacepoint
        # encoders do: the raw ranges are wild (ophit area spans 8 to ~540000,
        # positions are hundreds of cm, peaktime spans +/-2245 us), and feeding
        # that straight into a Linear leaves the optical branch badly scaled.
        # It matters more now that ophit.x is rewritten twice per iteration -
        # once by ophit_to_ophit and once by pmt_to_ophit.
        if use_optical:
            self.ophit_input_norm = InputNorm(9)
            self.pmt_input_norm = InputNorm(4)
            self.flash_input_norm = InputNorm(10)
            self.ophit_net = torch.nn.Linear(9, ophit_features)
            self.pmt_net = torch.nn.Linear(4, pmt_features)
            self.flash_net = torch.nn.Linear(10, flash_features)

    def forward(self, data: NuGraphData) -> None:
        """
        NuGraph3 encoder forward pass
        
        Args:
            data: Graph data object
        """
        data["hit"].x = self.input_norm(data["hit"].x)
        data["hit"].x = self.planar_net(data["hit"].x)
        if self.sp_net is not None:
            # Encode real spacepoint/nexus input features as the initial sp embedding.
            data["sp"].x = self.sp_net(self.sp_input_norm(data["sp"].x))
        else:
            data["sp"].x = torch.zeros(data["sp"].num_nodes,
                                       self.nexus_features,
                                       device=data["hit"].x.device)
        data["evt"].x = torch.zeros(data["evt"].num_nodes,
                                    self.interaction_features,
                                    device=data["hit"].x.device)

        if hasattr(self, "ophit_net"):
            data["ophit"].x = self.ophit_net(self.ophit_input_norm(data["ophit"].x))
            data["pmt"].x = self.pmt_net(self.pmt_input_norm(data["pmt"].x))
            data["flash"].x = self.flash_net(self.flash_input_norm(data["flash"].x))
