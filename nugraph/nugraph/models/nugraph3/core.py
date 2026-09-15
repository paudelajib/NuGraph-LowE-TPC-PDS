"""NuGraph core message-passing engine"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from .types import T, TD, Data
from .edge_geometry import PLANE_FEATURES

class NuGraphBlock(MessagePassing): # pylint: disable=abstract-method
    """
    Standard NuGraph message-passing block
    
    This block generates attention weights for each graph edge based on both
    the source and target node features, and then applies those weights to
    the source node features in order to form messages. These messages are
    then aggregated into the target nodes using softmax aggregation, and
    then fed into a two-layer MLP to generate updated target node features.

    Args:
        source_features: Number of source node input features
        target_features: Number of target node input features
        out_features: Number of target node output features
        edge_features: Number of edge attributes fed to the attention weight.
            0 (default) reproduces the original block exactly, so checkpoints
            trained without edge attributes still load.
    """
    def __init__(self, source_features: int, target_features: int,
                 out_features: int, edge_features: int = 0):
        super().__init__(aggr="softmax")

        self.edge_features = edge_features
        self.edge_net = nn.Sequential(
            nn.Linear(source_features+target_features+edge_features, 1),
            nn.Sigmoid())

        self.net = nn.Sequential(
            nn.Linear(source_features+target_features, out_features),
            nn.Mish(),
            nn.Linear(out_features, out_features),
            nn.Mish())

    def forward(self, x: T, edge_index: T, edge_attr: T = None) -> T: # pylint: disable=arguments-differ
        """
        NuGraphBlock forward pass

        Args:
            x: Node feature tensor
            edge_index: Edge index tensor
            edge_attr: Edge attribute tensor, required iff edge_features > 0
        """
        if self.edge_features:
            if edge_attr is None:
                raise RuntimeError(
                    f"block built with edge_features={self.edge_features} "
                    "but called without edge attributes")
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.propagate(edge_index, x=x)

    def message(self, x_i: T, x_j: T, edge_attr: T = None) -> T: # pylint: disable=arguments-differ
        """
        NuGraphBlock message function

        This function constructs messages on graph edges. Features from the
        source and target nodes, plus any edge attributes, are concatenated
        and fed into a linear layer to construct attention weights. Messages
        are then formed on edges by weighting the source node features by
        these attention weights.

        Args:
            x_i: Edge features from target nodes
            x_j: Edge features from source nodes
            edge_attr: Optional geometric attributes of each edge
        """
        z = (x_i, x_j) if edge_attr is None else (x_i, x_j, edge_attr)
        return self.edge_net(torch.cat(z, dim=1).detach()) * x_j

    def update(self, aggr_out: T, x: T) -> T: # pylint: disable=arguments-differ
        """
        NuGraphBlock update function

        This function takes the output node features and combines them with
        the input features

        Args:
            aggr_out: Tensor of aggregated node features
            x: Target node features
        """
        if isinstance(x, tuple):
            _, x = x
        return self.net(torch.cat((aggr_out, x), dim=1))

class NuGraphCore(nn.Module):
    """
    NuGraph core message-passing engine
    
    This is the core NuGraph message-passing loop

    Args:
        hit_features: Number of features in planar embedding
        nexus_features: Number of features in nexus embedding
        interaction_features: Number of features in interaction embedding
        use_checkpointing: Whether to use checkpointing
        use_edge_attr: Whether planar (Delaunay) edges carry geometric attributes
    """
    def __init__(self,
                 hit_features: int,
                 nexus_features: int,
                 interaction_features: int,
                 use_checkpointing: bool = True,
                 use_edge_attr: bool = False):
        super().__init__()

        self.use_checkpointing = use_checkpointing
        self.use_edge_attr = use_edge_attr

        # internal planar message-passing; the only core edge with geometry,
        # since the others are containment
        self.plane_net = NuGraphBlock(hit_features, hit_features,
                                      hit_features,
                                      edge_features=PLANE_FEATURES if use_edge_attr else 0)

        # message-passing from planar nodes to nexus nodes
        self.plane_to_nexus = NuGraphBlock(hit_features, nexus_features,
                                           nexus_features)

        # message-passing from nexus nodes to interaction nodes
        self.nexus_to_interaction = NuGraphBlock(nexus_features,
                                                 interaction_features,
                                                 interaction_features)

        # message-passing from interaction nodes to nexus nodes
        self.interaction_to_nexus = NuGraphBlock(interaction_features,
                                                 nexus_features,
                                                 nexus_features)

        # message-passing from nexus nodes to planar nodes
        self.nexus_to_plane = NuGraphBlock(nexus_features, hit_features,
                                           hit_features)

    def checkpoint(self, net: nn.Module, *args) -> TD:
        """
        Checkpoint module, if enabled.
        
        Args:
            net: Network module
            args: Arguments to network module
        """
        if self.use_checkpointing and self.training:
            return checkpoint(net, *args, use_reentrant=False)
        else:
            return net(*args)

    def forward(self, data: Data) -> None:
        """
        NuGraphCore forward pass
        
        Args:
            data: Graph data object
        """

        # message-passing in hits
        planar = data["hit", "delaunay-planar", "hit"]
        data["hit"].x = self.checkpoint(
            self.plane_net, data["hit"].x, planar.edge_index,
            *((planar.edge_attr,) if self.use_edge_attr else ()))

        # message-passing from hits to nexus
        data["sp"].x = self.checkpoint(
            self.plane_to_nexus, (data["hit"].x, data["sp"].x),
            data["hit", "nexus", "sp"].edge_index)

        # message-passing from nexus to interaction
        data["evt"].x = self.checkpoint(
            self.nexus_to_interaction, (data["sp"].x, data["evt"].x),
            data["sp", "in", "evt"].edge_index)

        # message-passing from interaction to nexus
        data["sp"].x = self.checkpoint(
            self.interaction_to_nexus, (data["evt"].x, data["sp"].x),
            data["sp", "in", "evt"].edge_index[(1,0), :])

        # message-passing from nexus to hits
        data["hit"].x = self.checkpoint(
            self.nexus_to_plane, (data["sp"].x, data["hit"].x),
            data["hit", "nexus", "sp"].edge_index[(1,0), :])
