import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl

from nugraph.data import H5DataModule
from nugraph.models import NuGraph3

GRAPH = "merged1.graph.h5.0000.h5"
CKPT = "/home/apaudel/NuGraph/logs/merged1_escc_fixed_b1/version_0/checkpoints/epoch=19-step=8680.ckpt"

dm = H5DataModule(
    data_path=GRAPH,
    model=NuGraph3,
    batch_size=1,
    num_workers=0,
    shuffle="random",
)

model = NuGraph3.load_from_checkpoint(CKPT, map_location="cpu")
model.eval()

accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices = 1

trainer = pl.Trainer(
    accelerator=accelerator,
    devices=devices,
    logger=False,
)

pred_batches = trainer.predict(model, dataloaders=dm.test_dataloader())

y_true = []
y_pred = []

for batch in pred_batches:
    true = batch["evt"].y.detach().cpu()
    prob = batch["evt"].e.detach().cpu()
    pred = prob.argmax(dim=1)

    y_true.extend(true.numpy().tolist())
    y_pred.extend(pred.numpy().tolist())

y_true = np.asarray(y_true, dtype=int)
y_pred = np.asarray(y_pred, dtype=int)

classes = dm.event_classes
print("Event classes:", classes)
print("N test events:", len(y_true))

nclass = len(classes)
cm = np.zeros((nclass, nclass), dtype=int)

for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

acc = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
print("Confusion matrix:")
print(cm)
print(f"Accuracy: {acc:.3f}")

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm)

ax.set_xticks(np.arange(nclass))
ax.set_yticks(np.arange(nclass))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"ES/CC confusion matrix, accuracy = {acc:.3f}")

for i in range(nclass):
    for j in range(nclass):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")

fig.colorbar(im, ax=ax, label="Events")
fig.tight_layout()

out = "merged1_escc_confusion.png"
fig.savefig(out, dpi=200)
print("Saved:", out)
