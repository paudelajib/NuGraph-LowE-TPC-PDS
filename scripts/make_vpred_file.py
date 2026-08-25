import os
import h5py
import torch
import numpy as np
import nugraph as ng
from collections import Counter, defaultdict

torch.set_float32_matmul_precision("medium")

INFILE = os.environ["INFILE"]
OUTFILE = os.environ["OUTFILE"]
VTX_CKPT = os.environ["VTX_CKPT"]

print("INFILE   =", INFILE)
print("OUTFILE  =", OUTFILE)
print("VTX_CKPT =", VTX_CKPT)

Data = ng.data.H5DataModule
Model = ng.models.NuGraph3

# Check checkpoint is nexus-only, not 3D-message.
raw = torch.load(VTX_CKPT, map_location="cpu")
hp = raw.get("hyper_parameters", {})
sd = raw.get("state_dict", {})

print("\nCheckpoint sanity:")
print("  vertex_head      =", hp.get("vertex_head"))
print("  direction_head   =", hp.get("direction_head"))
print("  semantic_head    =", hp.get("semantic_head"))
print("  event_head       =", hp.get("event_head"))
print("  sp_features      =", hp.get("sp_features"))
print("  has sp_to_sp 3D  =", any(k.startswith("core_net.sp_to_sp") for k in sd))

assert hp.get("vertex_head") is True, "Checkpoint does not have vertex_head=True"
assert hp.get("sp_features") == 5, "Checkpoint is not nexus-feature sp_features=5"
assert not any(k.startswith("core_net.sp_to_sp") for k in sd), "This is a 3D-message checkpoint; do not use it here"

nudata = Data(
    INFILE,
    batch_size=64,
    num_workers=0,
    model=Model,
    shuffle="random",
    featext3d=True,
)

model = Model.load_from_checkpoint(VTX_CKPT)
model.eval()

# Important: checkpoint has direction_head=True, but for v_pred production
# we run only the vertex decoder, so direction code cannot interfere.
assert hasattr(model, "vertex_decoder"), "Checkpoint has no vertex_decoder"
model.decoders = [model.vertex_decoder]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("\nModel sanity:")
print("  model.encoder.sp_net is None =", model.encoder.sp_net is None)
print("  has vertex decoder           =", hasattr(model, "vertex_decoder"))
print("  has direction decoder        =", hasattr(model, "direction_decoder"))
print("  active decoders              =", [type(d).__name__ for d in model.decoders])

assert model.encoder.sp_net is not None, "Nexus features are not active in the model"

def get_event_keys(batch):
    run = subrun = event = None

    for store in batch.stores:
        keys = list(store.keys())

        def get_any(names):
            for n in names:
                if n in store:
                    return store[n]
            return None

        r = get_any(["run", "metadata/run"])
        sr = get_any(["subrun", "metadata/subrun"])
        ev = get_any(["event", "metadata/event"])

        if r is not None and sr is not None and ev is not None:
            run, subrun, event = r, sr, ev
            break

    if run is None or subrun is None or event is None:
        print("\nCould not locate run/subrun/event in batch.")
        print("Available stores:")
        for store in batch.stores:
            print("  store =", getattr(store, "_key", None), "keys =", list(store.keys()))
        raise RuntimeError("No metadata run/subrun/event found")

    run = run.detach().cpu().view(-1).numpy()
    subrun = subrun.detach().cpu().view(-1).numpy()
    event = event.detach().cpu().view(-1).numpy()

    return [f"r{int(r)}_sr{int(sr)}_evt{int(e)}" for r, sr, e in zip(run, subrun, event)]

preds = {}
seen_total = 0
duplicate_keys = Counter()

for split, loader in [
    ("train", nudata.train_dataloader()),
    ("validation", nudata.val_dataloader()),
    ("test", nudata.test_dataloader()),
]:
    print("Predicting:", split)

    with torch.no_grad():
        for batch in loader:
            keys = get_event_keys(batch)
            seen_total += len(keys)

            batch = batch.to(device)
            model(batch)

            assert hasattr(batch["evt"], "v"), "Vertex prediction batch['evt'].v not found"

            v = batch["evt"].v.detach().cpu().numpy()
            assert len(keys) == v.shape[0], (len(keys), v.shape)

            for k, vv in zip(keys, v):
                if k in preds:
                    duplicate_keys[k] += 1
                preds[k] = vv.astype("float32")

print("\nPrediction bookkeeping:")
print("  total graphs seen by dataloader =", seen_total)
print("  unique predicted vertices       =", len(preds))
print("  duplicate keys                  =", sum(duplicate_keys.values()))

with h5py.File(INFILE, "r") as f:
    original_splits = {}
    used = set()

    for split in ["train", "validation", "test"]:
        arr = f[f"samples/{split}"].asstr()[()].tolist()
        original_splits[split] = arr
        used.update(arr)

missing = sorted(used - set(preds))
extra = sorted(set(preds) - used)

print("  original used samples           =", len(used))
print("  missing predictions             =", len(missing))
print("  extra predictions               =", len(extra))

missing_by_split = {}
for split, arr in original_splits.items():
    m = [s for s in arr if s in missing]
    missing_by_split[split] = m
    print(f"  missing in {split:10s}          = {len(m)}")

if missing:
    print("\nFirst missing examples:")
    for s in missing[:20]:
        print(" ", s)

# This is deliberate: do not truth-fill missing vertices and do not write NaNs into used samples.
# Drop missing samples from split lists and dataset.
kept_splits = {}
for split, arr in original_splits.items():
    kept_splits[split] = [s for s in arr if s in preds]

kept = set()
for arr in kept_splits.values():
    kept.update(arr)

print("\nOutput split sizes after dropping missing:")
for split, arr in kept_splits.items():
    print(f"  {split:10s} = {len(arr)}")

print("  total kept =", len(kept))

# Sanity: compare predicted vertex to truth vertex on kept events.
errs = []

with h5py.File(INFILE, "r") as f:
    for s in kept:
        rec = f[f"dataset/{s}"][()]
        y = np.asarray(rec["evt/y_vtx"]).reshape(3)
        p = preds[s].reshape(3)
        errs.append(np.linalg.norm(p - y))

errs = np.asarray(errs)
print("\nVertex residual |v_pred - y_vtx| on kept samples:")
print("  median =", np.median(errs))
print("  mean   =", np.mean(errs))
print("  p68    =", np.percentile(errs, 68))
print("  p90    =", np.percentile(errs, 90))

if os.path.exists(OUTFILE):
    raise RuntimeError(f"OUTFILE already exists: {OUTFILE}")

strdt = h5py.string_dtype(encoding="utf-8")

with h5py.File(INFILE, "r") as fi, h5py.File(OUTFILE, "w") as fo:
    # Copy all metadata groups/datasets except dataset and samples.
    # We recreate samples because 33 missing events are dropped.
    for k in fi.keys():
        if k not in ("dataset", "samples"):
            fi.copy(k, fo)

    # Recreate samples with missing events removed.
    gs = fo.create_group("samples")
    for split, arr in kept_splits.items():
        gs.create_dataset(split, data=np.asarray(arr, dtype=object), dtype=strdt)

    # Recreate dataset using only retained events, adding evt/v_pred.
    gd = fo.create_group("dataset")

    for ak, av in fi["dataset"].attrs.items():
        gd.attrs[ak] = av

    copied = 0
    for s in fi["dataset"].keys():
        if s not in kept:
            continue

        rec = fi[f"dataset/{s}"][()]
        old_dtype = rec.dtype

        assert "evt/v_pred" not in old_dtype.names, "Input already has evt/v_pred"

        new_dtype = np.dtype(old_dtype.descr + [("evt/v_pred", "<f4", (1, 3))])
        newrec = np.empty((), dtype=new_dtype)

        for name in old_dtype.names:
            newrec[name] = rec[name]

        newrec["evt/v_pred"] = preds[s].reshape(1, 3)

        gd.create_dataset(s, data=newrec)
        copied += 1

        if copied % 5000 == 0:
            print("  copied", copied)

print("\nWROTE:", OUTFILE)

# Final HDF5 checks.
with h5py.File(OUTFILE, "r") as f:
    n_dataset = len(f["dataset"])
    n_splits = sum(len(f[f"samples/{split}"]) for split in ["train", "validation", "test"])

    print("\nFinal file checks:")
    print("  dataset N       =", n_dataset)
    print("  split total     =", n_splits)

    assert n_dataset == n_splits == len(kept), "dataset and split counts disagree"

    s = f["samples/test"].asstr()[0]
    fields = f[f"dataset/{s}"].dtype.names
    rec = f[f"dataset/{s}"][()]

    print("  check sample    =", s)
    print("  has evt/y_vtx   =", "evt/y_vtx" in fields)
    print("  has evt/v_pred  =", "evt/v_pred" in fields)
    print("  has evt/y_dir   =", "evt/y_dir" in fields)
    print("  |v_pred-y_vtx|  =", np.linalg.norm(rec["evt/v_pred"].reshape(3) - rec["evt/y_vtx"].reshape(3)))

    assert "evt/v_pred" in fields
    assert "evt/y_dir" in fields

# Final dataloader check: confirm output file exposes batch["evt"].v_pred.
nudata2 = Data(
    OUTFILE,
    batch_size=64,
    num_workers=0,
    model=Model,
    shuffle="random",
    featext3d=True,
)

batch = next(iter(nudata2.train_dataloader()))

print("\nOutput dataloader check:")
print("  has batch['evt'].v_pred =", hasattr(batch["evt"], "v_pred"))
print("  batch['evt'].v_pred shape =", tuple(batch["evt"].v_pred.shape))
print("  batch['sp'].x shape       =", tuple(batch["sp"].x.shape))

assert hasattr(batch["evt"], "v_pred"), "Output dataloader does not expose evt/v_pred"
assert batch["evt"].v_pred.shape[1] == 3, "evt/v_pred should be shape (N,3)"

print("\nSUCCESS: fixed-vpred graph is ready.")
