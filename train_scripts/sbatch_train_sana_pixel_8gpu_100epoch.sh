#!/usr/bin/env bash
#SBATCH --job-name=sana-pixel-8gpu-1000ep-resume
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=6-00:00:00
#SBATCH --exclusive
#SBATCH --output=/home/ganchangyi/code/Sana-pixel/Sana/output/slurm-%x-%j.out
#SBATCH --error=/home/ganchangyi/code/Sana-pixel/Sana/output/slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="/fs1/private/user/ganchangyi/code/Sana-pixel/Sana"
CONFIG="configs/sana_pixel/Sana_1600M_1024px_webdataset_bf16_lr2e5.yaml"
WORK_DIR="output/sana_pixel_8gpu_1000epoch"
TORCHRUN="/home/ganchangyi/software/miniconda3/envs/pixel/bin/torchrun"
MASTER_PORT="${MASTER_PORT:-29520}"

source /home/ganchangyi/software/miniconda3/etc/profile.d/conda.sh
conda activate pixel

cd "$REPO_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT"
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR MASTER_PORT
export CONFIG WORK_DIR TORCHRUN

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
echo "master=${MASTER_ADDR}:${MASTER_PORT}"
echo "started=$(date -Is)"
nvidia-smi -L || true

# One torchrun process per node; each process launches four local GPU workers.
srun --nodes="$SLURM_NNODES" \
  --ntasks="$SLURM_NNODES" \
  --ntasks-per-node=1 \
  --kill-on-bad-exit=1 \
  bash -lc '
    exec "$TORCHRUN" \
      --nnodes="$SLURM_NNODES" \
      --nproc_per_node=4 \
      --node_rank="$SLURM_PROCID" \
      --master_addr="$MASTER_ADDR" \
      --master_port="$MASTER_PORT" \
      train_scripts/train.py \
      --config_path="$CONFIG" \
      --work_dir="$WORK_DIR" \
      --name=sana_pixel_8gpu_1000epoch \
      --resume_from=latest \
      --report_to=tensorboard \
      --train.train_batch_size=4 \
      --train.gradient_accumulation_steps=4 \
      --train.num_epochs=1000 \
      --train.early_stop_hours=0 \
      --train.visualize=true \
      --train.eval_sampling_epochs=15 \
      --train.eval_sampling_steps=1000000000 \
      --train.save_model_epochs=15 \
      --train.save_model_steps=1000000000
  '
