import torch
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pathlib import Path

from nugraph.data import H5DataModule
from nugraph.models import NuGraph3

GRAPH = "merged1.graph.h5.0000.h5"

# Use newest checkpoint from fixed-label training
CKPT = "/home/apaudel/NuGraph/logs/merged1_TRUE_TPC_PDS_b4_20epoch/version_0/checkpoints/epoch=19-step=2160.ckpt"

print("GRAPH =", GRAPH)
print("CKPT  =", CKPT)

dm = H5DataModule(
    data_path=GRAPH,
    model=NuGraph3,
    batch_size=1,
    num_workers=0,
    shuffle="random",
)

model = NuGraph3.load_from_checkpoint(CKPT, map_location="cpu")
model.eval()

trainer = pl.Trainer(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=False,
)

# TEST split only: should be 24 events = 12 ES + 12 CC
loader = dm.train_dataloader()
pred_batches = trainer.predict(model, dataloaders=loader)

y_true = []
y_pred = []

for batch in pred_batches:
    true = batch["evt"].y.detach().cpu().numpy().reshape(-1)
    score = batch["evt"].e.detach().cpu()
    pred = score.argmax(dim=1).numpy().reshape(-1)

    y_true.extend(true.tolist())
    y_pred.extend(pred.tolist())

y_true = np.asarray(y_true, dtype=int)
y_pred = np.asarray(y_pred, dtype=int)

print("N test events:", len(y_true))
print("True ES:", int(np.sum(y_true == 0)))
print("True CC:", int(np.sum(y_true == 1)))
print("Pred ES:", int(np.sum(y_pred == 0)))
print("Pred CC:", int(np.sum(y_pred == 1)))

classes = ["ES", "CC"]
cm = np.zeros((2, 2), dtype=int)

for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

acc = np.trace(cm) / cm.sum()
print("Confusion matrix:")
print(cm)
print("Accuracy:", acc)

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"ES/CC test confusion, acc={acc:.3f}")

for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

fig.colorbar(im, ax=ax, label="Events")
fig.tight_layout()

out = "merged1_escc_confusion_FIXED_Train_june23.png"
fig.savefig(out, dpi=200)
print("Saved:", out)
