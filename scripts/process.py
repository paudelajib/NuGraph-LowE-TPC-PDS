#!/usr/bin/env python

import argparse
import os
import h5py
import pynuml
from mpi4py import MPI

from pynuml.labels.lowe import StandardLabelsLowE
from pynuml.labels.flavor_lowe import LowEFlavorLabels


def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", type=str, required=True)
    parser.add_argument("-o", "--outfile", type=str, required=True)

    # Need at least 3 points for Delaunay triangulation.
    parser.add_argument("--lower-bound", type=int, default=3)

    # Same idea as tar lowe.py.
    parser.add_argument("--e-thr", type=float, default=0.01)

    return parser.parse_args()


def registered_group_names(f):
    names = []
    for g in f._groups:
        try:
            names.append(g[0])
        except Exception:
            names.append(str(g))
    return names


def add_group_if_needed(f, infile, group_name):
    with h5py.File(infile, "r") as h5:
        exists_in_file = group_name in h5

    if not exists_in_file:
        print(f"WARNING: {group_name} not found in HDF5 file; skipping.")
        return

    names = registered_group_names(f)

    if group_name not in names:
        print(f"Adding missing group registration: {group_name}")
        f.add_group(group_name)


def process(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0 and os.path.exists(args.outfile):
        print(f"Removing existing output file: {args.outfile}")
        os.remove(args.outfile)

    comm.Barrier()

    f = pynuml.io.File(args.infile)

    # Tar-style semantic labeler.
    semantic_labeller = StandardLabelsLowE(e_thr=args.e_thr)

    # Tar-style ES/CC event labeler.
    event_labeller = LowEFlavorLabels()

    processor = pynuml.process.HitGraphProducer(
        file=f,
        semantic_labeller=semantic_labeller,
        event_labeller=event_labeller,
        label_vertex=True,
        label_position=False,

        # Keep this true for now because your graph should include PDS.
        # We will make hitgraph.py decide what optical nodes/edges are added.
        optical=True,

        lower_bound=args.lower_bound,
    )

    add_group_if_needed(f, args.infile, "hit_table")
    add_group_if_needed(f, args.infile, "spacepoint_table")
    add_group_if_needed(f, args.infile, "particle_table")
    add_group_if_needed(f, args.infile, "edep_table")
    add_group_if_needed(f, args.infile, "event_table")

    # PDS groups.
    add_group_if_needed(f, args.infile, "ophit_table")
    add_group_if_needed(f, args.infile, "opflash_table")
    add_group_if_needed(f, args.infile, "opflashsumpe_table")

    if rank == 0:
        print("")
        print("Registered groups before processing:")
        for name in registered_group_names(f):
            print("  ", name)
        print("")

        print("Labelers:")
        print("  semantic_labeller = StandardLabelsLowE")
        print("  event_labeller    = LowEFlavorLabels")
        print("")

        print("Event labels:")
        print("  0 = cc_nue")
        print("  1 = es_nue")
        print("")

        print("Processing options:")
        print(f"  lower_bound = {args.lower_bound}")
        print(f"  e_thr       = {args.e_thr}")
        print("  optical     = True")
        print("")

    out = pynuml.io.H5Out(args.outfile)

    f.process(processor, out)


if __name__ == "__main__":
    args = configure()
    process(args)
