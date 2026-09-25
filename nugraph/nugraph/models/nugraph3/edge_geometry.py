"""Static geometric attributes for NuGraph3 edges"""
import torch
from torch import nn
from ...util import InputNorm

# raw ophit.x = [x, y, z, amplitude, area, pe, peaktime, width, amplitude/area]
OPHIT_TIME_COL = 6

PLANE = ("hit", "delaunay-planar", "hit")
NEXUS_PMT = ("sp", "knn", "pmt")
PMT_PMT = ("pmt", "knn", "pmt")
OPHIT_OPHIT = ("ophit", "knn", "ophit")

# attributes per edge type: [signed deltas, absolute deltas] over the
# coordinates the edge is defined in
# Delaunay: [d_integral, d_rms, d_wire, d_time, distance], built in the
# encoder from normalised hit inputs (nugraph/nugraph#169)
PLANE_FEATURES = 5
NEXUS_PMT_FEATURES = 4    # (y, z)
PMT_PMT_FEATURES = 4      # (y, z)
OPHIT_OPHIT_FEATURES = 6  # (y, z, time)


def deltas(src: torch.Tensor, dst: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Signed and absolute coordinate differences along each edge.

    Both are needed: the attention layer that consumes these is linear, so it
    cannot form |d| from d itself.
    """
    d = dst[edge_index[1]] - src[edge_index[0]]
    return torch.cat((d, d.abs()), dim=1)


class EdgeGeometry(nn.Module):
    """
    Attach geometric attributes to the edges whose meaning is geometric.

    Every NuGraphBlock previously weighted an edge using only the two endpoint
    embeddings, so it never saw how far apart two connected nodes are. The
    endpoints do carry position at the first iteration, but after a round or
    two of message passing the embeddings encode what a node looks like rather
    than where it is. These attributes are computed once from the raw
    positions and re-supplied at every iteration.

    Must run before the encoder: OpHit time is read from the raw ophit.x,
    which the encoder overwrites. Runs once per forward pass, outside the
    message-passing loop and outside checkpointing, so each normalisation
    updates its running statistics once per batch.

    Covers the optical edge types only; the Delaunay edges are handled in the
    encoder, where the normalised hit inputs are available. Containment edges
    (hit-nexus-sp, sp-in-evt, ophit-in-pmt, pmt-in-flash) have no natural
    geometry and are left without attributes.

    Args:
        use_optical: whether the optical branch is present
        use_pmt_pmt: whether pmt <-> pmt message passing is enabled
        use_ophit_ophit: whether ophit <-> ophit message passing is enabled
    """
    def __init__(self, use_optical: bool, use_pmt_pmt: bool, use_ophit_ophit: bool):
        super().__init__()
        self.norms = nn.ModuleDict()
        if use_optical:
            self.norms["nexus_pmt"] = InputNorm(NEXUS_PMT_FEATURES)
            if use_pmt_pmt:
                self.norms["pmt_pmt"] = InputNorm(PMT_PMT_FEATURES)
            if use_ophit_ophit:
                self.norms["ophit_ophit"] = InputNorm(OPHIT_OPHIT_FEATURES)

    def _norm(self, key: str, raw: torch.Tensor) -> torch.Tensor:
        if raw.size(0) == 0:
            return raw
        norm = self.norms[key]
        if raw.size(0) < 2:
            # a single row has no variance; normalise without updating stats
            previous, norm.update = norm.update, False
            try:
                return norm(raw)
            finally:
                norm.update = previous
        return norm(raw)

    def forward(self, data) -> None:
        """
        Compute and attach edge attributes.

        Args:
            data: batched graph, before the encoder has run
        """
        if "nexus_pmt" in self.norms and NEXUS_PMT in data.edge_types:
            sp_yz, pmt_yz = data["sp"].pos[:, 1:3], data["pmt"].pos
            ei = data[NEXUS_PMT].edge_index
            fwd = deltas(sp_yz, pmt_yz, ei)
            rev = deltas(pmt_yz, sp_yz, ei.flip(0))  # the pmt -> sp direction
            # one normalisation over both directions: the edge type is used
            # both ways, so its statistics should be direction-symmetric
            both = self._norm("nexus_pmt", torch.cat((fwd, rev), dim=0))
            n = fwd.size(0)
            data[NEXUS_PMT].edge_attr = both[:n]
            data[NEXUS_PMT].edge_attr_rev = both[n:]

        if "pmt_pmt" in self.norms and PMT_PMT in data.edge_types:
            pos = data["pmt"].pos
            data[PMT_PMT].edge_attr = self._norm(
                "pmt_pmt", deltas(pos, pos, data[PMT_PMT].edge_index))

        if "ophit_ophit" in self.norms and OPHIT_OPHIT in data.edge_types:
            yzt = torch.cat((data["ophit"].pos[:, 1:3],
                             data["ophit"].x[:, OPHIT_TIME_COL:OPHIT_TIME_COL + 1]), dim=1)
            data[OPHIT_OPHIT].edge_attr = self._norm(
                "ophit_ophit", deltas(yzt, yzt, data[OPHIT_OPHIT].edge_index))

    @staticmethod
    def clear(data) -> None:
        """Remove the attributes, so the returned batch matches its input
        and Batch.to_data_list() is unaffected."""
        for et in (PLANE, NEXUS_PMT, PMT_PMT, OPHIT_OPHIT):
            if et in data.edge_types:
                store = data[et]
                for key in ("edge_attr", "edge_attr_rev"):
                    if key in store:
                        del store[key]
