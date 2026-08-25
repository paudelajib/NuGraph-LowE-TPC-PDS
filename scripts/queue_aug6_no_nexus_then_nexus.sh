#!/bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nugraph-gpu

export NUGRAPH_LOG=/home/apaudel/NuGraph/logs
mkdir -p "$NUGRAPH_LOG"

echo "Starting Aug6 no-nexus run at:"
date

python train.py \
  --device 0 \
  --data-path /home/apaudel/NuGraph/scripts/merged1_Aug6_40k_pmtpmt_g4lepdir \
  --semantic \
  --event \
  --optical \
  --logger tensorboard \
  --name merged1_Aug18_MERGE_CERATI_NEXUS_on_Aug6_g4lepdir_no_new_features_pmtpmtfix_ep45_bs64_in8 \
  --epochs 45 \
  --batch-size 64 \
  --num-workers 0 \
  --in-feats 8

echo "Finished Aug6 no-nexus run at:"
date

echo "Starting Aug6 nexus-only run at:"
date

python train.py \
  --device 0 \
  --data-path /home/apaudel/NuGraph/scripts/merged1_Aug6_40k_pmtpmt_g4lepdir \
  --semantic \
  --event \
  --optical \
  --logger tensorboard \
  --name merged1_Aug18_MERGE_CERATI_NEXUS_on_Aug6_g4lepdir_nexusfeat_only_pmtpmtfix_ep45_bs64_in8_sp5 \
  --epochs 45 \
  --batch-size 64 \
  --num-workers 0 \
  --in-feats 8 \
  --3dfeatext \
  --in-sp-feats 5

echo "Finished Aug6 nexus-only run at:"
date
