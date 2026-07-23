from typing import Any, Callable

import torch
import torch_geometric as pyg
import pandas as pd

from ..data import NuGraphData
from .base import ProcessorBase


def _unique_extend(dst: list[str], src: list[str]) -> None:
    """Append keys without duplicating them."""
    for key in src:
        if key not in dst:
            dst.append(key)


def _xyz_columns(frame: pd.DataFrame, base: str) -> list[str]:
    """Return the 3 expanded column names for an HDF5 vector column.

    pynuml expands some 3-vectors as base_x/base_y/base_z and some as
    base_0/base_1/base_2, depending on the source column name.  This helper
    makes the vertex fallback work for both conventions.
    """
    candidates = [
        [f"{base}_x", f"{base}_y", f"{base}_z"],
        [f"{base}_0", f"{base}_1", f"{base}_2"],
    ]
    for cols in candidates:
        if all(col in frame.columns for col in cols):
            return cols
    raise KeyError(
        f'Could not find expanded columns for "{base}". '
        f'Available columns are: {list(frame.columns)}'
    )


class HitGraphProducer(ProcessorBase):
    '''Process event into graphs'''

    def __init__(self,
                 file: 'pynuml.io.File',
                 semantic_labeller: Callable = None,
                 event_labeller: Callable = None,
                 label_vertex: bool = False,
                 label_position: bool = False,
                 optical: bool = False,
                 planes: list[str] = ['u','v','y'],
                 node_feats: list[str] = ['integral','rms','tpc'],
                 lower_bound: int = 3,
                 store_detailed_truth: bool = False):

        self.semantic_labeller = semantic_labeller
        self.event_labeller = event_labeller
        self.label_vertex = label_vertex
        self.label_position = label_position
        self.optical = optical
        self.planes = planes
        self.node_feats = node_feats
        self.lower_bound = lower_bound
        self.store_detailed_truth = store_detailed_truth

        self.transform = pyg.transforms.Compose((
            pyg.transforms.Delaunay(),
            pyg.transforms.FaceToEdge()))

        # Vertex truth in the original beam-neutrino files lives in
        # /event_table/nu_vtx_corr.  Some low-energy / supernova samples do not
        # write that dataset.  In that case, fall back to the start_position of
        # the primary particle (parent_id == 0), which is the best available
        # true interaction vertex without changing the HDF5Maker.
        self.vertex_source = None
        if self.label_vertex:
            event_keys = set(file._fd["event_table"].keys()) if "event_table" in file._fd else set()
            particle_keys = set(file._fd["particle_table"].keys()) if "particle_table" in file._fd else set()

            if "nu_vtx_corr" in event_keys:
                self.vertex_source = "event_table"
            elif {"parent_id", "start_position"}.issubset(particle_keys):
                self.vertex_source = "primary_particle_start"
            else:
                raise Exception(
                    "label_vertex=True was requested, but neither "
                    "/event_table/nu_vtx_corr nor "
                    "/particle_table/{parent_id,start_position} exists in this HDF5 file. "
                    "Use label_vertex=False, add vertex datasets in HDF5Maker, or provide "
                    "particle_table/start_position."
                )

        super().__init__(file)

    @property
    def columns(self) -> dict[str, list[str]]:
        groups = {
            'hit_table': [],
            'spacepoint_table': []
        }
        if self.semantic_labeller:
            groups['particle_table'] = ['g4_id','parent_id','type','momentum','start_process','end_process']
            groups['edep_table'] = []
        if self.event_labeller:
            groups['event_table'] = ['is_cc', 'is_es', 'nu_pdg']
        if self.label_vertex:
            if self.vertex_source == "event_table":
                # Only nu_vtx_corr is used below.  Do not require
                # nu_vtx_wire_pos/time for samples that do not write them.
                keys = ['nu_vtx_corr']
                if 'event_table' in groups:
                    _unique_extend(groups['event_table'], keys)
                else:
                    groups['event_table'] = keys
            elif self.vertex_source == "primary_particle_start":
                keys = ['parent_id', 'start_position']
                if 'particle_table' in groups:
                    _unique_extend(groups['particle_table'], keys)
                else:
                    groups['particle_table'] = keys
        if self.label_position:
            groups["edep_table"] = []
        if self.optical:
            groups["ophit_table"] = []
            groups["opflash_table"] = []
            groups["opflashsumpe_table"] = []
        return groups

    @property
    def metadata(self):
        metadata = dict(planes=self.planes, gen=torch.tensor([2]))
        if self.semantic_labeller is not None:
            metadata['semantic_classes'] = self.semantic_labeller.labels[:-1]
        if self.event_labeller is not None:
            metadata['event_classes'] = self.event_labeller.labels
        return metadata

    def __call__(self, evt: 'pynuml.io.Event') -> tuple[str, Any]:

        if self.event_labeller or (self.label_vertex and self.vertex_source == "event_table"):
            event = evt['event_table'].squeeze()
        else:
            event = None

        # support different generations of event HDF5 format
        hits = evt['hit_table']
        if "global_plane" in hits.columns:
            plane_key, proj_key, drift_key = "global_plane", "global_wire", "global_time"
        elif "local_plane" in hits.columns:
            plane_key, proj_key, drift_key = "local_plane", "local_wire", "local_time"
        else:
            plane_key, proj_key, drift_key = "view", "proj", "drift"

        spacepoints = evt['spacepoint_table'].reset_index(drop=True)

        # discard any events with pathologically large hit integrals
        # this is a hotfix that should be removed once the dataset is fixed
        if hits.integral.max() > 1e6:
            print('found event with pathologically large hit integral, skipping')
            return evt.name, None

        # handle energy depositions
        if self.semantic_labeller:
            edeps = evt['edep_table']
            energy_col = 'energy' if 'energy' in edeps.columns else 'energy_fraction' # for backwards compatibility

            # get ID of max particle
            g4_id = edeps[[energy_col, 'g4_id', 'hit_id']]
            g4_id = g4_id.sort_values(by=[energy_col],
                                      ascending=False,
                                      kind='mergesort').drop_duplicates('hit_id')
            hits = g4_id.merge(hits, on='hit_id', how='right')

            # charge-weighted average of 3D position
            if self.label_position:
                edeps = edeps[["hit_id", "energy", "x_position", "y_position", "z_position"]]
                for col in ["x_position", "y_position", "z_position"]:
                    edeps.loc[:, col] *= edeps.energy
                edeps = edeps.groupby("hit_id").sum()
                for col in ["x_position", "y_position", "z_position"]:
                    edeps.loc[:, col] /= edeps.energy
                edeps = edeps.drop("energy", axis="columns")
                hits = edeps.merge(hits, on="hit_id", how="right")

            hits['filter_label'] = ~hits[energy_col].isnull()
            hits = hits.drop(energy_col, axis='columns')

        # reset spacepoint index
        spacepoints = spacepoints.reset_index(names='index_3d')

        # Skip events with no usable 3D spacepoints.
        # Otherwise empty sp tensors/edges can be written as scalar HDF5 fields
        # and NuGraphData.load fails during sample generation.
        if spacepoints.shape[0] == 0:
            return evt.name, None

        # skip events with fewer than lower_bnd simulated hits in any plane.
        # note that we can't just do a pandas groupby here, because that will
        # skip over any planes with zero hits
        for i in range(len(self.planes)):
            planehits = hits[hits[plane_key]==i]
            nhits = planehits.filter_label.sum() if self.semantic_labeller else planehits.shape[0]
            if nhits < self.lower_bound:
                return evt.name, None

        # get labels for each particle
        if self.semantic_labeller:
            particles = self.semantic_labeller(evt['particle_table'])
            try:
                hits = hits.merge(particles, on='g4_id', how='left')
            except:
                print('exception occurred when merging hits and particles')
                print('hit table:', hits)
                print('particle table:', particles)
                print('skipping this event')
                return evt.name, None
            mask = (~hits.g4_id.isnull()) & (hits.semantic_label.isnull())
            if mask.any():
                print(f'found {mask.sum()} orphaned hits.')
                return evt.name, None
            del mask

        data = NuGraphData()

        # event metadata
        r, sr, e = evt.event_id
        data['metadata'].run = r
        data['metadata'].subrun = sr
        data['metadata'].event = e

        # spacepoint nodes
        # Always define sp.pos so batched graphs have consistent attributes.
        if "position_x" in spacepoints.keys():
            sp_pos = (
                spacepoints[[f"position_{c}" for c in ("x", "y", "z")]]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype="float32")
            )
            data["sp"].pos = torch.tensor(sp_pos, dtype=torch.float).reshape((-1, 3))
        else:
            data["sp"].pos = torch.zeros((spacepoints.shape[0], 3), dtype=torch.float)

        hits = hits.reset_index(names="index_2d")

        node_pos = [proj_key, drift_key]

        # node position
        data["hit"].plane = torch.tensor(hits[plane_key].values, dtype=torch.long)
        data["hit"].pos = torch.tensor(hits[node_pos].values, dtype=torch.float)

        # node features
        node_feats = self.node_feats + [plane_key, proj_key, drift_key]
        data["hit"].x = torch.tensor(hits[node_feats].values).float()

        # node true position
        if self.label_position:
            data["hit"].y_position = torch.tensor(hits[["x_position", "y_position", "z_position"]].values).float()

        # hit indices
        data["hit"].id = torch.tensor(hits['hit_id'].values).long()

        # 2D graph edges
        data["hit", "delaunay", "hit"].edge_index = self.transform(data["hit"]).edge_index
        edge_plane = []
        for i, view_hits in hits.groupby(plane_key):
        
            # Delaunay needs at least 3 points in a plane
            if view_hits.shape[0] < 3:
                continue
        
            tmp = pyg.data.Data()
            tmp.index_2d = torch.tensor(view_hits.index_2d.values).long()
            tmp.pos = torch.tensor(view_hits[node_pos].values).float()
        
            try:
                edge_plane.append(tmp.index_2d[self.transform(tmp).edge_index])
            except AssertionError:
                continue
        
        if len(edge_plane) > 0:
            data["hit", "delaunay-planar", "hit"].edge_index = torch.cat(edge_plane, dim=1)
        else:
            data["hit", "delaunay-planar", "hit"].edge_index = torch.empty((2, 0), dtype=torch.long)

        # 3D graph edges
        edge_nexus = []
        for i, view_hits in hits.groupby(plane_key):
            p = self.planes[i]
            edge = spacepoints.merge(hits[['hit_id','index_2d']].add_suffix(f'_{p}'),
                                     on=f'hit_id_{p}',
                                     how='inner')
            edge = edge[[f'index_2d_{p}','index_3d']].values.transpose()
            edge = torch.tensor(edge) if edge.size else torch.empty((2,0))
            edge_nexus.append(edge.long())
        data["hit", "nexus", "sp"].edge_index = torch.cat(edge_nexus, dim=1)

        # add edges to event node
        data["evt"].num_nodes = 1
        lo = torch.arange(data["hit"].num_nodes, dtype=torch.long)
        hi = torch.zeros(data["hit"].num_nodes, dtype=torch.long)
        data["hit", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)
        lo = torch.arange(data["sp"].num_nodes, dtype=torch.long)
        hi = torch.zeros(data["sp"].num_nodes, dtype=torch.long)
        data["sp", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)

        # truth information
        if self.semantic_labeller:
            data["hit"].y_semantic = torch.tensor(hits['semantic_label'].fillna(-1).values).long()
            y = torch.tensor(hits['instance_label'].fillna(-1).values).long()
            mask = y != -1
            y = y[mask]
            instances = y.unique()
            # remap instances
            imax = instances.max() + 1 if instances.size(0) else 0
            if instances.size(0) != imax:
                remap = torch.full((imax,), -1, dtype=torch.long)
                remap[instances] = torch.arange(instances.size(0))
                y = remap[y]
            data["particle-truth"].num_nodes = instances.size(0)
            edges = torch.stack((mask.nonzero().squeeze(1), y), dim=0).long()
            data["hit", "cluster-truth", "particle-truth"].edge_index = edges
            if self.store_detailed_truth:
                data["hit"].g4_id = torch.tensor(hits['g4_id'].fillna(-1).values).long()
                data["hit"].parent_id = torch.tensor(hits['parent_id'].fillna(-1).values).long()
                data["hit"].pdg = torch.tensor(hits['type'].fillna(-1).values).long()

        # optical system
        if self.optical:

            ophits = evt["ophit_table"]
            sum_pe = evt["opflashsumpe_table"]
            opflash = evt["opflash_table"]

            # skip events with no flash
            if opflash.shape[0]==0:
                return evt.name, None

            # node position
            data["ophit"].pos = torch.tensor(ophits[["wire_pos_0", "wire_pos_1", "wire_pos_2"]].values).float()
            data["flash"].pos = torch.tensor(opflash[["wire_pos_0", "wire_pos_1", "wire_pos_2"]].values).float()

            if "pos_y" in sum_pe.columns:
                data["pmt"].pos = torch.tensor(sum_pe[["pos_y", "pos_z"]].values).float()
            else:
                # hardcoded positions for DUNE FD-HD optical detectors
                # extracted from opdet_geometry_raw.txt: Channel N => ... at (x,y,z) cm
                opdet_pos_y = torch.tensor([591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            591.569, 529.298, 467.028, 404.758, 342.487, 280.217, 217.946, 155.676, 93.4056, 31.1352,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569,
                                            -31.1352, -93.4056, -155.676, -217.946, -280.217, -342.487, -404.758, -467.028, -529.298, -591.569])
                opdet_pos_z = torch.tensor([1357.47, 1357.47, 1357.47, 1357.47, 1357.47, 1357.47, 1357.47, 1357.47, 1357.47, 1357.47,
                                            1308.67, 1308.67, 1308.67, 1308.67, 1308.67, 1308.67, 1308.67, 1308.67, 1308.67, 1308.67,
                                            1245.87, 1245.87, 1245.87, 1245.87, 1245.87, 1245.87, 1245.87, 1245.87, 1245.87, 1245.87,
                                            1197.07, 1197.07, 1197.07, 1197.07, 1197.07, 1197.07, 1197.07, 1197.07, 1197.07, 1197.07,
                                            1125.08, 1125.08, 1125.08, 1125.08, 1125.08, 1125.08, 1125.08, 1125.08, 1125.08, 1125.08,
                                            1076.28, 1076.28, 1076.28, 1076.28, 1076.28, 1076.28, 1076.28, 1076.28, 1076.28, 1076.28,
                                            1013.48, 1013.48, 1013.48, 1013.48, 1013.48, 1013.48, 1013.48, 1013.48, 1013.48, 1013.48,
                                            964.679, 964.679, 964.679, 964.679, 964.679, 964.679, 964.679, 964.679, 964.679, 964.679,
                                            892.689, 892.689, 892.689, 892.689, 892.689, 892.689, 892.689, 892.689, 892.689, 892.689,
                                            843.889, 843.889, 843.889, 843.889, 843.889, 843.889, 843.889, 843.889, 843.889, 843.889,
                                            781.089, 781.089, 781.089, 781.089, 781.089, 781.089, 781.089, 781.089, 781.089, 781.089,
                                            732.289, 732.289, 732.289, 732.289, 732.289, 732.289, 732.289, 732.289, 732.289, 732.289,
                                            660.299, 660.299, 660.299, 660.299, 660.299, 660.299, 660.299, 660.299, 660.299, 660.299,
                                            611.499, 611.499, 611.499, 611.499, 611.499, 611.499, 611.499, 611.499, 611.499, 611.499,
                                            548.699, 548.699, 548.699, 548.699, 548.699, 548.699, 548.699, 548.699, 548.699, 548.699,
                                            499.899, 499.899, 499.899, 499.899, 499.899, 499.899, 499.899, 499.899, 499.899, 499.899,
                                            427.909, 427.909, 427.909, 427.909, 427.909, 427.909, 427.909, 427.909, 427.909, 427.909,
                                            379.109, 379.109, 379.109, 379.109, 379.109, 379.109, 379.109, 379.109, 379.109, 379.109,
                                            316.309, 316.309, 316.309, 316.309, 316.309, 316.309, 316.309, 316.309, 316.309, 316.309,
                                            267.509, 267.509, 267.509, 267.509, 267.509, 267.509, 267.509, 267.509, 267.509, 267.509,
                                            195.519, 195.519, 195.519, 195.519, 195.519, 195.519, 195.519, 195.519, 195.519, 195.519,
                                            146.719, 146.719, 146.719, 146.719, 146.719, 146.719, 146.719, 146.719, 146.719, 146.719,
                                            83.9188, 83.9188, 83.9188, 83.9188, 83.9188, 83.9188, 83.9188, 83.9188, 83.9188, 83.9188,
                                            35.1188, 35.1188, 35.1188, 35.1188, 35.1188, 35.1188, 35.1188, 35.1188, 35.1188, 35.1188,
                                            1357.47, 1357.47, 1357.47, 1357.47, 1357.47, 1357.47, 1357.47, 1357.47, 1357.47, 1357.47,
                                            1308.67, 1308.67, 1308.67, 1308.67, 1308.67, 1308.67, 1308.67, 1308.67, 1308.67, 1308.67,
                                            1245.87, 1245.87, 1245.87, 1245.87, 1245.87, 1245.87, 1245.87, 1245.87, 1245.87, 1245.87,
                                            1197.07, 1197.07, 1197.07, 1197.07, 1197.07, 1197.07, 1197.07, 1197.07, 1197.07, 1197.07,
                                            1125.08, 1125.08, 1125.08, 1125.08, 1125.08, 1125.08, 1125.08, 1125.08, 1125.08, 1125.08,
                                            1076.28, 1076.28, 1076.28, 1076.28, 1076.28, 1076.28, 1076.28, 1076.28, 1076.28, 1076.28,
                                            1013.48, 1013.48, 1013.48, 1013.48, 1013.48, 1013.48, 1013.48, 1013.48, 1013.48, 1013.48,
                                            964.679, 964.679, 964.679, 964.679, 964.679, 964.679, 964.679, 964.679, 964.679, 964.679,
                                            892.689, 892.689, 892.689, 892.689, 892.689, 892.689, 892.689, 892.689, 892.689, 892.689,
                                            843.889, 843.889, 843.889, 843.889, 843.889, 843.889, 843.889, 843.889, 843.889, 843.889,
                                            781.089, 781.089, 781.089, 781.089, 781.089, 781.089, 781.089, 781.089, 781.089, 781.089,
                                            732.289, 732.289, 732.289, 732.289, 732.289, 732.289, 732.289, 732.289, 732.289, 732.289,
                                            660.299, 660.299, 660.299, 660.299, 660.299, 660.299, 660.299, 660.299, 660.299, 660.299,
                                            611.499, 611.499, 611.499, 611.499, 611.499, 611.499, 611.499, 611.499, 611.499, 611.499,
                                            548.699, 548.699, 548.699, 548.699, 548.699, 548.699, 548.699, 548.699, 548.699, 548.699,
                                            499.899, 499.899, 499.899, 499.899, 499.899, 499.899, 499.899, 499.899, 499.899, 499.899,
                                            427.909, 427.909, 427.909, 427.909, 427.909, 427.909, 427.909, 427.909, 427.909, 427.909,
                                            379.109, 379.109, 379.109, 379.109, 379.109, 379.109, 379.109, 379.109, 379.109, 379.109,
                                            316.309, 316.309, 316.309, 316.309, 316.309, 316.309, 316.309, 316.309, 316.309, 316.309,
                                            267.509, 267.509, 267.509, 267.509, 267.509, 267.509, 267.509, 267.509, 267.509, 267.509,
                                            195.519, 195.519, 195.519, 195.519, 195.519, 195.519, 195.519, 195.519, 195.519, 195.519,
                                            146.719, 146.719, 146.719, 146.719, 146.719, 146.719, 146.719, 146.719, 146.719, 146.719,
                                            83.9188, 83.9188, 83.9188, 83.9188, 83.9188, 83.9188, 83.9188, 83.9188, 83.9188, 83.9188,
                                            35.1188, 35.1188, 35.1188, 35.1188, 35.1188, 35.1188, 35.1188, 35.1188, 35.1188, 35.1188])
                data["pmt"].pos = torch.stack([opdet_pos_y[sum_pe["pmt_channel"].values], opdet_pos_z[sum_pe["pmt_channel"].values]], dim=1)

            # optical node features (not including the positions)
            # OpHit feature order:
            #   [wire_pos_0, wire_pos_1, wire_pos_2,
            #    amplitude, area_or_integral, pe, peaktime, width,
            #    amplitude_over_area_or_integral]
            #
            # Prefer "area" if available; otherwise use "integral".
            if "area" in ophits.columns:
                ophit_area_col = "area"
            elif "integral" in ophits.columns:
                ophit_area_col = "integral"
            else:
                raise KeyError(
                    "ophit_table needs either an 'area' or 'integral' column "
                    "to build amplitude/(area or integral)."
                )

            ophit_feature_cols = ["amplitude", ophit_area_col, "pe", "peaktime", "width"]
            missing_ophit_cols = [col for col in ophit_feature_cols if col not in ophits.columns]
            if missing_ophit_cols:
                raise KeyError(f"Missing required ophit_table columns: {missing_ophit_cols}")

            ophit_features = torch.tensor(ophits[ophit_feature_cols].values).float()

            amplitude = ophit_features[:, 0:1]
            area_or_integral = ophit_features[:, 1:2]

            # Safe ratio: use zero when the denominator is exactly/near zero.
            amplitude_over_area_or_integral = torch.where(
                area_or_integral.abs() > 1.0e-6,
                amplitude / area_or_integral,
                torch.zeros_like(amplitude),
            )

            data["ophit"].x = torch.cat(
                [
                    data["ophit"].pos,
                    ophit_features,
                    amplitude_over_area_or_integral,
                ],
                dim=1,
            )
            data["flash"].x = torch.cat([data["flash"].pos,torch.tensor(opflash[["time", "time_width", "totalpe", "y_center", "y_width", "z_center", "z_width"]].values).float()],dim=1)
            data["pmt"].x = torch.cat([data["pmt"].pos,torch.tensor(sum_pe[["pmt_channel", "sumpe"]].values).float()],dim=1)

            # ophit to pmt edges
            # PyG edge_index must use LOCAL node indices, not raw hit_id/sumpe_id.
            sumpe_id_to_local = {
                int(v): i for i, v in enumerate(sum_pe["sumpe_id"].values)
                if not pd.isna(v)
            }

            ophit_src = []
            pmt_dst = []

            for local_ophit, sid in enumerate(ophits["sumpe_id"].values):
                if pd.isna(sid):
                    continue
                sid = int(sid)
                if sid >= 0 and sid in sumpe_id_to_local:
                    ophit_src.append(local_ophit)
                    pmt_dst.append(sumpe_id_to_local[sid])

            if len(ophit_src) > 0:
                data["ophit", "in", "pmt"].edge_index = torch.tensor(
                    [ophit_src, pmt_dst],
                    dtype=torch.long,
                )
            else:
                data["ophit", "in", "pmt"].edge_index = torch.empty(
                    (2, 0),
                    dtype=torch.long,
                )

            # pmt to pmt edges
            n_pmt = data["pmt"].pos.size(0)
            if n_pmt > 1:
                distances = torch.cdist(data["pmt"].pos, data["pmt"].pos, p=2)
                distances.fill_diagonal_(float("inf"))
                knn = min(3, n_pmt - 1)
                _, neighbor_idx = torch.topk(distances, knn, largest=False, dim=1)
                source = torch.arange(n_pmt, dtype=torch.long).repeat_interleave(knn)
                target = neighbor_idx.flatten()
                edge4 = torch.stack((source, target), dim=0)
                edge4 = torch.cat((edge4, edge4.flip(0)), dim=1)
            else:
                edge4 = torch.empty((2, 0), dtype=torch.long)

            data["pmt", "knn", "pmt"].edge_index = edge4

            # pmt to flash edges
            # Also use LOCAL pmt and flash node indices.
            flash_id_to_local = {
                int(v): i for i, v in enumerate(opflash["flash_id"].values)
                if not pd.isna(v)
            }

            pmt_src = []
            flash_dst = []

            for local_pmt, fid in enumerate(sum_pe["flash_id"].values):
                if pd.isna(fid):
                    continue
                fid = int(fid)
                if fid >= 0 and fid in flash_id_to_local:
                    pmt_src.append(local_pmt)
                    flash_dst.append(flash_id_to_local[fid])

            if len(pmt_src) > 0:
                data["pmt", "in", "flash"].edge_index = torch.tensor(
                    [pmt_src, flash_dst],
                    dtype=torch.long,
                )
            else:
                data["pmt", "in", "flash"].edge_index = torch.empty(
                    (2, 0),
                    dtype=torch.long,
                )

            # flash to event edges
            lo = torch.arange(data["flash"].num_nodes, dtype=torch.long)
            hi = torch.zeros(data["flash"].num_nodes, dtype=torch.long)
            data["flash", "in", "evt"].edge_index = torch.stack((lo, hi), dim=0)

            # nexus to pmt edges
            spacepoints_nodes_np = (
                spacepoints[["position_y", "position_z"]]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype="float32")
            )
            spacepoints_nodes = torch.tensor(spacepoints_nodes_np, dtype=torch.float)
            # spacepoint -> nearest PMT edges
            if spacepoints_nodes.shape[0] > 0 and data["pmt"].pos.shape[0] > 0:
                distances = torch.cdist(spacepoints_nodes.float(), data["pmt"].pos.float())
                k_requested = 10
                k = min(k_requested, int(distances.shape[1]))
                if k > 0:
                    _, nearest_indices = torch.topk(distances, k, largest=False, dim=1)
                    spacepoints_indices = torch.arange(spacepoints_nodes.shape[0], dtype=torch.long).repeat_interleave(k)
                    opflashsumpe_indices = nearest_indices.reshape(-1).long()
                    data["sp", "knn", "pmt"].edge_index = torch.stack([spacepoints_indices, opflashsumpe_indices], dim=0).long()
                else:
                    data["sp", "knn", "pmt"].edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                data["sp", "knn", "pmt"].edge_index = torch.empty((2, 0), dtype=torch.long)

        # event label
        if self.event_labeller:
            # pylint: disable=possibly-used-before-assignment
            data['evt'].y = torch.tensor(self.event_labeller(event)).long().reshape([1])

        # 3D vertex truth
        if self.label_vertex:
            if self.vertex_source == "event_table":
                vtx_3d = [[event.nu_vtx_corr_x, event.nu_vtx_corr_y, event.nu_vtx_corr_z]]
            elif self.vertex_source == "primary_particle_start":
                particles_for_vtx = evt['particle_table']
                start_cols = _xyz_columns(particles_for_vtx, 'start_position')

                primary = particles_for_vtx[particles_for_vtx['parent_id'] == 0]
                if primary.empty:
                    # Fallback: take the first particle if parent_id==0 is absent.
                    primary = particles_for_vtx.iloc[[0]]

                vtx_3d = [primary.iloc[0][start_cols].astype(float).values.tolist()]
            else:
                raise RuntimeError(f"Unknown vertex_source: {self.vertex_source}")

            data['evt'].y_vtx = torch.tensor(vtx_3d).float()

        return evt.name, data
