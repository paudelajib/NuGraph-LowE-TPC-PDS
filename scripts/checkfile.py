import h5py

fname = "merged.graph.h5.0000.h5"

with h5py.File(fname, "r") as f:
    print(f"\nOpened: {fname}")
    print("Top-level keys:")
    for k in f.keys():
        print("  ", k)

    print("\nDatasets:")
    n_datasets = 0

    def visitor(name, obj):
        global n_datasets
        if isinstance(obj, h5py.Dataset):
            n_datasets += 1
            print(f"  {name:60s} shape={obj.shape}, dtype={obj.dtype}")

    f.visititems(visitor)

    print(f"\nTotal datasets: {n_datasets}")

    if n_datasets == 0:
        print("WARNING: file opened, but no datasets found.")
    else:
        print("Looks non-empty.")
