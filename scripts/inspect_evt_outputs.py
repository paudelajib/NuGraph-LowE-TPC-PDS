import torch
from nugraph.data import H5DataModule
from nugraph.models import NuGraph3

graph_file = "merged_large.graph.h5.0000.h5"
ckpt = "/home/apaudel/NuGraph/logs/merged_large_tpc_20e_fixed/version_0/checkpoints/epoch=19-step=21420.ckpt"

dm = H5DataModule(
    graph_file,
    batch_size=16,
    model=NuGraph3,
    num_workers=0,
)
dm.setup("test")

model = NuGraph3.load_from_checkpoint(ckpt, map_location="cpu")
model.eval()

batch = next(iter(dm.test_dataloader()))

with torch.no_grad():
    out = model(batch)

print("model forward returned:", type(out))
print("\nevt object:")
print(batch["evt"])

print("\nevt keys/fields:")
for k in batch["evt"].keys():
    v = batch["evt"][k]
    if torch.is_tensor(v):
        print(k, tuple(v.shape), v[:5])
    else:
        print(k, type(v), v)

print("\nFull batch:")
print(batch)
