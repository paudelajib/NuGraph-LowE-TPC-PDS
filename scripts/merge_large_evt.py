import sys
import os
import h5py
import numpy as np

if len(sys.argv) != 3:
    print("Usage: python merge_large_evt.py large_evt_files.txt merged_large.evt.h5")
    sys.exit(1)

list_file = sys.argv[1]
out_file = sys.argv[2]

files = [x.strip() for x in open(list_file) if x.strip()]

if os.path.exists(out_file):
    raise RuntimeError(f"Output already exists: {out_file}. Move/delete it first.")

print(f"Merging {len(files)} files")
print(f"Output: {out_file}")

def list_datasets(h5):
    paths = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append(name)
    h5.visititems(visitor)
    return sorted(paths)

with h5py.File(files[0], "r") as h0:
    dataset_paths = list_datasets(h0)

print(f"Found {len(dataset_paths)} datasets")

# Quick schema check
for f in files[1:]:
    with h5py.File(f, "r") as h5:
        paths = list_datasets(h5)
        if paths != dataset_paths:
            raise RuntimeError(f"Dataset structure mismatch in {f}")

with h5py.File(out_file, "w") as hout:
    # copy top-level attrs from first file
    with h5py.File(files[0], "r") as h0:
        for k, v in h0.attrs.items():
            hout.attrs[k] = v

    for path in dataset_paths:
        print(f"Merging dataset: {path}")

        created = False
        offset = 0

        for f in files:
            with h5py.File(f, "r") as hin:
                dset = hin[path]
                data = dset[...]

                # scalar dataset: copy only once
                if dset.shape == ():
                    if not created:
                        grp_path = os.path.dirname(path)
                        if grp_path:
                            hout.require_group(grp_path)
                        out = hout.create_dataset(path, data=data)
                        for k, v in dset.attrs.items():
                            out.attrs[k] = v
                        created = True
                    continue

                n = dset.shape[0]

                if not created:
                    grp_path = os.path.dirname(path)
                    if grp_path:
                        hout.require_group(grp_path)

                    maxshape = (None,) + dset.shape[1:]
                    chunks = (max(1, min(n, 1024)),) + dset.shape[1:]

                    out = hout.create_dataset(
                        path,
                        shape=(0,) + dset.shape[1:],
                        maxshape=maxshape,
                        dtype=dset.dtype,
                        chunks=chunks,
                    )

                    for k, v in dset.attrs.items():
                        out.attrs[k] = v

                    created = True

                out = hout[path]
                out.resize((offset + n,) + dset.shape[1:])
                out[offset:offset+n] = data
                offset += n

print("DONE")
