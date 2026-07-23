import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from nugraph.data import H5DataModule
from nugraph.models import NuGraph3

GRAPH = "merged_large.graph.h5.0000.h5"

MODELS = {
    "TPC-only": "/home/apaudel/NuGraph/logs/merged_large_tpc_20e_fixed/version_0/checkpoints/epoch=19-step=21420.ckpt",
    "TPC+PDS": "/home/apaudel/NuGraph/logs/merged_large_tpc_pds_20e_fixed/version_0/checkpoints/epoch=19-step=21420.ckpt",
}

OUTDIR = "escc_final_report"
os.makedirs(OUTDIR, exist_ok=True)

def confusion(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def plot_cm(cm, title, outfile, normalize=False):
    if normalize:
        mat = cm / cm.sum(axis=1, keepdims=True)
        fmt = ".3f"
    else:
        mat = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], ["ES", "CC"])
    ax.set_yticks([0, 1], ["ES", "CC"])

    for i in range(2):
        for j in range(2):
            text = format(mat[i, j], fmt)
            ax.text(j, i, text, ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)

def evaluate(name, ckpt):
    print(f"\nEvaluating {name}")
    print("checkpoint:", ckpt)

    dm = H5DataModule(
        GRAPH,
        batch_size=16,
        model=NuGraph3,
        num_workers=0,
    )
    dm.setup("test")

    model = NuGraph3.load_from_checkpoint(ckpt, map_location="cpu")
    model.eval()

    y_true_all = []
    y_pred_all = []
    prob_all = []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            model(batch)

            y_true = batch["evt"].y.detach().cpu().numpy()
            prob = batch["evt"].e.detach().cpu().numpy()
            y_pred = np.argmax(prob, axis=1)

            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
            prob_all.append(prob)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    prob = np.concatenate(prob_all)

    cm = confusion(y_true, y_pred)
    acc = np.mean(y_true == y_pred)
    es_eff = cm[0, 0] / cm[0].sum()
    cc_eff = cm[1, 1] / cm[1].sum()

    print("N test =", len(y_true))
    print("Raw confusion matrix:")
    print(cm)
    print("Accuracy =", acc)
    print("ES efficiency =", es_eff)
    print("CC efficiency =", cc_eff)

    tag = name.lower().replace("+", "_").replace("-", "_").replace(" ", "_")

    np.savetxt(f"{OUTDIR}/{tag}_confusion_raw.txt", cm, fmt="%d")
    np.savetxt(f"{OUTDIR}/{tag}_probabilities.txt", prob)

    plot_cm(
        cm,
        f"{name}: raw confusion matrix",
        f"{OUTDIR}/{tag}_confusion_raw.png",
        normalize=False,
    )

    plot_cm(
        cm,
        f"{name}: normalized confusion matrix",
        f"{OUTDIR}/{tag}_confusion_normalized.png",
        normalize=True,
    )

    return {
        "model": name,
        "N_test": len(y_true),
        "accuracy": acc,
        "ES_efficiency": es_eff,
        "CC_efficiency": cc_eff,
        "ES_true": int(cm[0].sum()),
        "CC_true": int(cm[1].sum()),
        "ES_to_ES": int(cm[0, 0]),
        "ES_to_CC": int(cm[0, 1]),
        "CC_to_ES": int(cm[1, 0]),
        "CC_to_CC": int(cm[1, 1]),
    }

rows = []
for name, ckpt in MODELS.items():
    rows.append(evaluate(name, ckpt))

import pandas as pd
df = pd.DataFrame(rows)
print("\nComparison table:")
print(df)

df.to_csv(f"{OUTDIR}/comparison_table.csv", index=False)

print("\nWrote outputs to:", OUTDIR)
