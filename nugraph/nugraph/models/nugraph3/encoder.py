"""NuGraph3 encoder"""
import torch
from pynuml.data import NuGraphData
from ...util import InputNorm

# [n_hit, n_spacepoint, n_ophit, total_pe] per event, all log-scaled
EVT_SEED_FEATURES = 4

# index of "pe" within the 9 raw OpHit features
# [x, y, z, amplitude, area, pe, peaktime, width, amplitude/area]
OPHIT_PE_COL = 5


def _per_event(data, node_type: str, n_evt: int, device,
               weights: torch.Tensor = None) -> torch.Tensor:
    """Sum `weights` (or count nodes) of `node_type` within each event.

    Returns zeros when the node type is absent or empty, so the seed keeps a
    fixed width whether or not the optical branch is enabled.
    """
    if node_type not in data.node_types:
        return torch.zeros(n_evt, device=device)
    store = data[node_type]
    n = store.num_nodes
    if not n:
        return torch.zeros(n_evt, device=device)

    # a batched graph carries per-node event indices; a single graph does not
    idx = getattr(store, "batch", None)
    if idx is None:
        idx = torch.zeros(n, dtype=torch.long, device=device)
    w = torch.ones(n, device=device) if weights is None else weights.to(device)

    out = torch.zeros(n_evt, device=device)
    out.index_add_(0, idx, w)
    return out


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
                 sp_features: int = 0,
                 use_evt_seed: bool = False):
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

        # Event node seed.
        #
        # This used to start at zeros, so everything the event node learned
        # arrived through softmax aggregation - a weighted MEAN, which is
        # count-invariant. Yet OpHit multiplicity is the single most
        # discriminating quantity we measured (AUC 0.757, against 0.725 for the
        # whole network), and no path through flash.totalpe or pmt.sumpe carries
        # a count: those are sums of photoelectrons, not numbers of hits.
        #
        # Seeding the event node with log-scaled extensive quantities gives it
        # that information directly and lets message passing add topology on
        # top. Log scaling because these span orders of magnitude and only their
        # ratios carry meaning. Always four features, zero-filled when the
        # optical branch is absent, so the seed width does not depend on flags.
        #
        # Opt-in, because building it unconditionally puts evt_net and
        # evt_input_norm in the state dict and makes every checkpoint trained
        # before this existed fail to load with strict=True. Behind the flag,
        # an older checkpoint reconstructs its original zero-init architecture,
        # and a newer one is loaded by passing use_evt_seed=True.
        if use_evt_seed:
            self.evt_input_norm = InputNorm(EVT_SEED_FEATURES)
            self.evt_net = torch.nn.Linear(EVT_SEED_FEATURES, interaction_features)

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
        # Seed the event node with per-event extensive quantities. Computed from
        # the RAW optical features, so this must run before ophit_net rewrites
        # data["ophit"].x below.
        device = data["hit"].x.device
        n_evt = data["evt"].num_nodes
        has_optical = ("ophit" in data.node_types
                       and data["ophit"].x is not None
                       and data["ophit"].x.numel() > 0)
        if hasattr(self, "evt_net"):
            seed = torch.stack([
                _per_event(data, "hit", n_evt, device),
                _per_event(data, "sp", n_evt, device),
                _per_event(data, "ophit", n_evt, device),
                _per_event(data, "ophit", n_evt, device,
                           weights=data["ophit"].x[:, OPHIT_PE_COL].clamp(min=0))
                if has_optical else torch.zeros(n_evt, device=device),
            ], dim=1)
            data["evt"].x = self.evt_net(self.evt_input_norm(torch.log1p(seed)))
        else:
            data["evt"].x = torch.zeros(n_evt, self.interaction_features,
                                        device=device)

        if hasattr(self, "ophit_net"):
            data["ophit"].x = self.ophit_net(self.ophit_input_norm(data["ophit"].x))
            data["pmt"].x = self.pmt_net(self.pmt_input_norm(data["pmt"].x))
            data["flash"].x = self.flash_net(self.flash_input_norm(data["flash"].x))
