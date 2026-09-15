"""NuGraph optical convolution module"""
import torch
from pynuml.data import NuGraphData
from .core import NuGraphBlock
from .types import TD
from .edge_geometry import NEXUS_PMT_FEATURES, PMT_PMT_FEATURES, OPHIT_OPHIT_FEATURES

class NuGraphOptical(torch.nn.Module):
        """
        NuGraph optical message-passing engine

        This module incorporates optical information into NuGraph

        Args:
        interaction_features: Number of features in interaction embedding
        ophit_features: Number of features in optical hit embedding
        pmt_features: Number of features in PMT (flashsumpe) embedding
        flash_features: Number of features in optical flash embedding
        use_checkpointing: Whether to use checkpointing
        """
        def __init__(self, # pylint: disable=too-many-arguments,too-many-positional-arguments
                     interaction_features: int,
                     nexus_features: int,
                     ophit_features: int,
                     pmt_features: int,
                     flash_features: int,
                     use_checkpointing: bool = True,
                     optical_only: bool = False,
                     use_pmt_pmt: bool = False,
                     use_ophit_ophit: bool = False,
                     use_edge_attr: bool = False):
                super().__init__()

                self.use_checkpointing = use_checkpointing
                self.optical_only = optical_only
                self.use_pmt_pmt = use_pmt_pmt
                self.use_ophit_ophit = use_ophit_ophit
                self.use_edge_attr = use_edge_attr
                # geometric edge types get attributes; containment edges do not
                ef = lambda n: n if use_edge_attr else 0

                # hierarchical message-passing for optical system
                self.ophit_to_pmt = NuGraphBlock(ophit_features, pmt_features, pmt_features)
                self.pmt_to_flash = NuGraphBlock(pmt_features, flash_features, flash_features)
                self.flash_to_interaction = NuGraphBlock(flash_features,
                                                         interaction_features,
                                                         interaction_features)
                self.interaction_to_flash = NuGraphBlock(interaction_features,
                                                         flash_features, flash_features)
                self.flash_to_pmt = NuGraphBlock(flash_features, pmt_features, pmt_features)
                self.pmt_to_ophit = NuGraphBlock(pmt_features, ophit_features, ophit_features)

                # message-passing between PMT nodes
                if self.use_pmt_pmt:
                        self.pmt_to_pmt = NuGraphBlock(pmt_features, pmt_features, pmt_features,
                                                       edge_features=ef(PMT_PMT_FEATURES))

                # message-passing between OpHit nodes. The edges are built from
                # separation in (y, z) and time; with edge attributes the
                # network sees that separation instead of just the connection.
                if self.use_ophit_ophit:
                        self.ophit_to_ophit = NuGraphBlock(ophit_features, ophit_features,
                                                           ophit_features,
                                                           edge_features=ef(OPHIT_OPHIT_FEATURES))

                # message-passing between nexus nodes and PMT nodes (opflashsumpe)
                self.nexus_to_pmt = NuGraphBlock(nexus_features, pmt_features, pmt_features,
                                                 edge_features=ef(NEXUS_PMT_FEATURES))
                self.pmt_to_nexus = NuGraphBlock(pmt_features, nexus_features, nexus_features,
                                                 edge_features=ef(NEXUS_PMT_FEATURES))

        def _attr(self, store, key: str = "edge_attr") -> tuple:
                """Edge attributes as extra positional args, or nothing when disabled."""
                return (store[key],) if self.use_edge_attr else ()

        def checkpoint(self, net: torch.nn.Module, *args) -> TD:
                """
                Checkpoint module, if enabled.

                Args:
                net: Network module
                args: Arguments to network module
                """
                if self.use_checkpointing and self.training:
                        return torch.utils.checkpoint.checkpoint(net, *args, use_reentrant=False)
                return net(*args)

        def forward(self, data: NuGraphData) -> None:
                """
                NuGraphCore forward pass

                Args:
                data: Graph data object
                """

                # message-passing between OpHits.
                #
                # This MUST run before ophit -> pmt. Around two thirds of
                # OpHits have no ophit -> pmt edge, so without a prior step
                # among OpHits themselves their features never enter the
                # hierarchy at all. Running it first lets an unattached OpHit
                # pass its features to a neighbour that IS attached, which then
                # carries them up to pmt -> flash -> evt.
                if self.use_ophit_ophit:
                        edge_type = ("ophit", "knn", "ophit")
                        if edge_type not in data.edge_types:
                                raise RuntimeError(
                                        "use_ophit_ophit=True, but graph does not contain "
                                        "('ophit', 'knn', 'ophit') edges. Reprocess with "
                                        "OpHit-to-OpHit edges."
                                )
                        ophit_edges = data[edge_type]
                        if ophit_edges.edge_index.numel() == 0:
                                raise RuntimeError(
                                        "use_ophit_ophit=True, but ('ophit', 'knn', 'ophit') "
                                        "edge_index is empty."
                                )
                        data["ophit"].x = self.checkpoint(
                                self.ophit_to_ophit,
                                (data["ophit"].x, data["ophit"].x),
                                ophit_edges.edge_index,
                                *self._attr(ophit_edges))

                # message-passing from ophit to pmt
                data["pmt"].x = self.checkpoint(
                        self.ophit_to_pmt, (data["ophit"].x, data["pmt"].x),
                        data["ophit", "in", "pmt"].edge_index)

                # message-passing from space points to PMTs
                # Skip this in optical-only mode so TPC/spacepoint information
                # does not enter the PDS branch.
                if not self.optical_only:
                        bridge = data["sp", "knn", "pmt"]
                        data["pmt"].x = self.checkpoint(
                                self.nexus_to_pmt, (data["sp"].x, data["pmt"].x),
                                bridge.edge_index, *self._attr(bridge))

                # message-passing from PMTs to PMTs
                if self.use_pmt_pmt:
                        edge_type = ("pmt", "knn", "pmt")
                        if edge_type not in data.edge_types:
                                raise RuntimeError(
                                        "use_pmt_pmt=True, but graph does not contain "
                                        "('pmt', 'knn', 'pmt') edges. Reprocess with PMT-to-PMT edges."
                                )
                        pmt_edges = data[edge_type]
                        if pmt_edges.edge_index.numel() == 0:
                                raise RuntimeError(
                                        "use_pmt_pmt=True, but ('pmt', 'knn', 'pmt') edge_index is empty."
                                )
                        data["pmt"].x = self.checkpoint(
                                self.pmt_to_pmt,
                                (data["pmt"].x, data["pmt"].x),
                                pmt_edges.edge_index, *self._attr(pmt_edges))

                # message-passing from pmt to flash
                data["flash"].x = self.checkpoint(
                        self.pmt_to_flash, (data["pmt"].x, data["flash"].x),
                        data["pmt", "in", "flash"].edge_index)

                # message-passing from flash to interaction
                data["evt"].x = self.checkpoint(
                        self.flash_to_interaction, (data["flash"].x, data["evt"].x),
                        data["flash", "in", "evt"].edge_index)

                # message-passing from interaction to flash
                data["flash"].x = self.checkpoint(
                        self.interaction_to_flash, (data["evt"].x, data["flash"].x),
                        data["flash", "in", "evt"].edge_index[(1,0), :])

                # message-passing from flash to pmt
                data["pmt"].x = self.checkpoint(
                        self.flash_to_pmt, (data["flash"].x, data["pmt"].x),
                        data["pmt", "in", "flash"].edge_index[(1,0), :])

                # message-passing from PMTs to space points
                # Skip this in optical-only mode so PDS does not update TPC/spacepoint nodes.
                if not self.optical_only:
                        # reversed edges need the reversed attributes: the
                        # signed deltas flip, the absolute ones do not
                        bridge = data["sp", "knn", "pmt"]
                        data["sp"].x = self.checkpoint(
                                self.pmt_to_nexus, (data["pmt"].x, data["sp"].x),
                                bridge.edge_index[(1,0), :],
                                *self._attr(bridge, "edge_attr_rev"))

                # message-passing from pmt to ophit
                data["ophit"].x = self.checkpoint(
                        self.pmt_to_ophit, (data["pmt"].x, data["ophit"].x),
                        data["ophit", "in", "pmt"].edge_index[(1,0), :])
