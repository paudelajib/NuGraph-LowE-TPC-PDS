import io
import h5py
import torch
import numpy as np

f = "merged_large.graph.h5.0000.h5"

def decode(x):
    return x.decode() if isinstance(x, bytes) else str(x)

def raw_to_bytes(raw):
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, np.void):
        return raw.tobytes()
    return bytes(raw)

def load_graph(h5, name):
    raw = h5["dataset"][name][()]
    return torch.load(io.BytesIO(raw_to_bytes(raw)), map_location="cpu", weights_only=False)

with h5py.File(f, "r") as h5:
    print("dataset events =", len(h5["dataset"].keys()))

    # inspect one graph
    first_name = decode(h5["samples"]["train"][0])
    data = load_graph(h5, first_name)

    print("\nExample graph:", first_name)
    for node in ["hit", "sp", "ophit", "flash", "pmt", "evt"]:
        if node in data.node_types:
            print(node, "nodes =", data[node].num_nodes)
            if hasattr(data[node], "x"):
                print(" ", node + ".x", tuple(data[node].x.shape))
            if hasattr(data[node], "pos"):
                print(" ", node + ".pos", tuple(data[node].pos.shape))
            if hasattr(data[node], "y"):
                print(" ", node + ".y", data[node].y.tolist())

    print("\nSplit label balance:")
    for split in ["train", "validation", "test"]:
        labels = []
        for x in h5["samples"][split][:]:
            name = decode(x)
            data = load_graph(h5, name)
            labels.append(int(data["evt"].y.item()))

        labels = np.asarray(labels)
        print(split)
        print("  N  =", len(labels))
        print("  ES =", int(np.sum(labels == 0)))
        print("  CC =", int(np.sum(labels == 1)))
