#!/bin/bash
set -euo pipefail

# Phase-1 SANA 1.5 latent-to-pixel training. Environment variables may
# override the defaults, e.g. NP=1 WORK_DIR=output/pixel_smoke bash ...
CONFIG="${CONFIG:-configs/sana_pixel/Sana_1600M_1024px_webdataset_bf16_lr2e5.yaml}"
NP="${NP:-4}"
WORK_DIR="${WORK_DIR:-output/sana_pixel_1024_10k}"
MASTER_PORT="${MASTER_PORT:-29510}"
TORCHRUN="${TORCHRUN:-torchrun}"

"$TORCHRUN" \
  --nproc_per_node="$NP" \
  --master_port="$MASTER_PORT" \
  train_scripts/train.py \
  --config_path="$CONFIG" \
  --work_dir="$WORK_DIR" \
  --name=sana_pixel_1024_10k \
  --report_to=tensorboard \
  "$@"
