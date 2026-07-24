# NuGraph Low-E CC/ES Training and Testing Commands

This file records the main run commands used for the low-energy CC/ES classification studies.

Repository mode summary:

```text
no optical flag   -> TPC only
--optical         -> TPC + PDS
--opticalonly     -> PDS only
```

Notebook case tags:

```python
CASE_TAG = "tpc_only"
CASE_TAG = "tpc_pds"
CASE_TAG = "tpc_pds_pmtpmt"
CASE_TAG = "pds_only_pmtpmt"
```

---

## Common setup

```bash
cd /home/apaudel/NuGraph/scripts
conda activate nugraph-gpu
export NUGRAPH_LOG=/home/apaudel/NuGraph/logs
```

---

## Graph files

### Original graph file

Used for the original TPC-only and original TPC+PDS runs:

```text
/home/apaudel/NuGraph/scripts/merged1_July21_40k
```

This graph does not include the PMT↔PMT edge.

### PMT↔PMT graph file

Used for the new TPC+PDS+PMT↔PMT and PDS-only runs:

```bash
cd /home/apaudel/NuGraph/scripts
conda activate nugraph-gpu
export NUGRAPH_LOG=/home/apaudel/NuGraph/logs

python process.py \
  -i /home/apaudel/NuGraph/data/data_40k/merged_large.evt.h5 \
  -o merged1_July21_40k_pmtpmt \
  --lower-bound 3

python merge.py -f merged1_July21_40k_pmtpmt
```

Check that the PMT↔PMT edge exists:

```bash
cd /home/apaudel/NuGraph/scripts

python - <<'PY'
from nugraph.data import H5DataModule

GRAPH = "/home/apaudel/NuGraph/scripts/merged1_July21_40k_pmtpmt"

dm = H5DataModule(
    data_path=GRAPH,
    batch_size=1,
    num_workers=0,
)

try:
    dm.setup("fit")
except TypeError:
    dm.setup()

batch = next(iter(dm.train_dataloader()))

key = ("pmt", "knn", "pmt")

print("Node types:")
print(batch.node_types)

print("\nEdge types:")
for edge_type in batch.edge_types:
    print(" ", edge_type)

print("\nChecking PMT <-> PMT edge:")
if key in batch.edge_types:
    edge_index = batch[key].edge_index
    print("OK: found", key)
    print("edge_index shape:", tuple(edge_index.shape))
    print("number of edges:", edge_index.shape[1])
else:
    raise RuntimeError("Missing ('pmt', 'knn', 'pmt') edge.")
PY
```

---

# 1. TPC-only training

Notebook tag:

```python
CASE_TAG = "tpc_only"
```

Training command:

```bash
cd /home/apaudel/NuGraph/scripts
conda activate nugraph-gpu
export NUGRAPH_LOG=/home/apaudel/NuGraph/logs

python train.py \
  --device 0 \
  --data-path /home/apaudel/NuGraph/scripts/merged1_July21_40k \
  --event \
  --logger tensorboard \
  --name merged1_July21_40k_tpc_only_ep45_bs64_in8_repo_infeatures \
  --epochs 45 \
  --batch-size 64 \
  --num-workers 0 \
  --in-feats 8
```

Example test command:

```bash
cd /home/apaudel/NuGraph/scripts
conda activate nugraph-gpu
export NUGRAPH_LOG=/home/apaudel/NuGraph/logs

CKPT=/home/apaudel/NuGraph/logs/merged1_July21_40k_tpc_only_ep45_bs64_in8_repo_infeatures/version_0/checkpoints/epoch=44-step=21105.ckpt

python test_event.py \
  --device 0 \
  --data-path /home/apaudel/NuGraph/scripts/merged1_July21_40k \
  --checkpoint $CKPT \
  --outfile merged1_July21_40k_tpc_only_ep45_bs64_in8_repo_infeatures_version_0_test_predictions.h5 \
  --batch-size 64 \
  --num-workers 0
```

---

# 2. TPC + PDS training

Notebook tag:

```python
CASE_TAG = "tpc_pds"
```

Training command:

```bash
cd /home/apaudel/NuGraph/scripts
conda activate nugraph-gpu
export NUGRAPH_LOG=/home/apaudel/NuGraph/logs

python train.py \
  --device 0 \
  --data-path /home/apaudel/NuGraph/scripts/merged1_July21_40k \
  --event \
  --optical \
  --logger tensorboard \
  --name merged1_July21_40k_tpc_pds_ep45_bs64_in8_repo_infeatures \
  --epochs 45 \
  --batch-size 64 \
  --num-workers 0 \
  --in-feats 8
```

Example test command:

```bash
cd /home/apaudel/NuGraph/scripts
conda activate nugraph-gpu
export NUGRAPH_LOG=/home/apaudel/NuGraph/logs

CKPT=/home/apaudel/NuGraph/logs/merged1_July21_40k_tpc_pds_ep45_bs64_in8_repo_infeatures/version_0/checkpoints/epoch=44-step=21105.ckpt

python test_event.py \
  --device 0 \
  --data-path /home/apaudel/NuGraph/scripts/merged1_July21_40k \
  --checkpoint $CKPT \
  --outfile merged1_July21_40k_tpc_pds_ep45_bs64_in8_repo_infeatures_version_0_test_predictions.h5 \
  --batch-size 64 \
  --num-workers 0
```

---

# 3. TPC + PDS + PMT↔PMT training

Notebook tag:

```python
CASE_TAG = "tpc_pds_pmtpmt"
```

Training command:

```bash
cd /home/apaudel/NuGraph/scripts
conda activate nugraph-gpu
export NUGRAPH_LOG=/home/apaudel/NuGraph/logs

python train.py \
  --device 0 \
  --data-path /home/apaudel/NuGraph/scripts/merged1_July21_40k_pmtpmt \
  --event \
  --optical \
  --logger tensorboard \
  --name merged1_July21_40k_pmtpmt_tpc_pds_ep45_bs64_in8_int32 \
  --epochs 45 \
  --batch-size 64 \
  --num-workers 0 \
  --in-feats 8
```

Example test command:

```bash
cd /home/apaudel/NuGraph/scripts
conda activate nugraph-gpu
export NUGRAPH_LOG=/home/apaudel/NuGraph/logs

CKPT=/home/apaudel/NuGraph/logs/merged1_July21_40k_pmtpmt_tpc_pds_ep45_bs64_in8_int32/version_0/checkpoints/epoch=44-step=21105.ckpt

python test_event.py \
  --device 0 \
  --data-path /home/apaudel/NuGraph/scripts/merged1_July21_40k_pmtpmt \
  --checkpoint $CKPT \
  --outfile merged1_July21_40k_pmtpmt_tpc_pds_ep45_bs64_in8_int32_version_0_test_predictions.h5 \
  --batch-size 64 \
  --num-workers 0
```

---

# 4. PDS-only training

Notebook tag:

```python
CASE_TAG = "pds_only_pmtpmt"
```

Training command:

```bash
cd /home/apaudel/NuGraph/scripts
conda activate nugraph-gpu
export NUGRAPH_LOG=/home/apaudel/NuGraph/logs

python train.py \
  --device 0 \
  --data-path /home/apaudel/NuGraph/scripts/merged1_July21_40k_pmtpmt \
  --event \
  --opticalonly \
  --logger tensorboard \
  --name merged1_July21_40k_pmtpmt_pds_only_ep45_bs64_in8_int32 \
  --epochs 45 \
  --batch-size 64 \
  --num-workers 0 \
  --in-feats 8
```

Example test command:

```bash
cd /home/apaudel/NuGraph/scripts
conda activate nugraph-gpu
export NUGRAPH_LOG=/home/apaudel/NuGraph/logs

CKPT=/home/apaudel/NuGraph/logs/merged1_July21_40k_pmtpmt_pds_only_ep45_bs64_in8_int32/version_0/checkpoints/epoch=44-step=21105.ckpt

python test_event.py \
  --device 0 \
  --data-path /home/apaudel/NuGraph/scripts/merged1_July21_40k_pmtpmt \
  --checkpoint $CKPT \
  --outfile merged1_July21_40k_pmtpmt_pds_only_ep45_bs64_in8_int32_version_0_test_predictions.h5 \
  --batch-size 64 \
  --num-workers 0
```

---

## Universal notebook workflow

Use:

```text
notebooks/July24_test_and_plot_final.ipynb
```

In Cell 1, set one of:

```python
CASE_TAG = "tpc_only"
CASE_TAG = "tpc_pds"
CASE_TAG = "tpc_pds_pmtpmt"
CASE_TAG = "pds_only_pmtpmt"
```

The notebook will:

```text
1. Find the latest checkpoint under the selected run directory.
2. Run scripts/test_event.py.
3. Save the prediction H5 file.
4. Read TensorBoard scalars and images.
5. Make the final test confusion matrix.
6. Make P(CC) and P(ES) histograms.
7. Save PNG, CSV, and summary outputs into a tag-based plot directory.
```

---

## Detaching a long training job

After the training starts:

```text
Ctrl+Z
```

Then:

```bash
bg
disown -h %1
```

Check it is still running:

```bash
pgrep -af "python train.py"
nvidia-smi
```

---

## Git commands to add this file

From the repository root:

```bash
cd /home/apaudel/NuGraph

git add RUN_COMMANDS.md
git status
git diff --cached --stat

git commit -m "Add run command record for CC ES training modes"
git push
```

If the final plotting notebook and code changes are still not committed, use:

```bash
cd /home/apaudel/NuGraph

git add \
  RUN_COMMANDS.md \
  notebooks/July24_test_and_plot_final.ipynb \
  nugraph/nugraph/models/nugraph3/nugraph3.py \
  nugraph/nugraph/models/nugraph3/optical.py \
  scripts/test_event.py

git status
git diff --cached --stat

git commit -m "Add optical-only mode and run command record"
git push
```
