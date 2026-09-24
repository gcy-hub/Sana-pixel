#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Slurm 提交脚本：SANA-1.5 1.6B 1024px 批量 T2I 数据集生成（4 卡 / 1 天）
#
# 提交:
#   sbatch tools/dataset_gen/sbatch_t2i_dataset.sh
#
# 查看:
#   squeue -u $USER
#   tail -f /home/ganchangyi/dataset/SANA-Pixel-Dataset/logs/slurm/slurm-<jobid>.out
#
# 说明:
#   * 用 sbatch 而不是仓库自带的 `sana-run`（后者底层是阻塞式 srun，会话一断
#     整个进程树被杀 —— 上一轮 2511 张就是这么丢的）。sbatch 提交后立即返回，
#     作业独立存活，且脚本本身可断点续传。
#   * 作业被 Slurm 超时/抢占后，重新 sbatch 一次即可从断点继续。
# ---------------------------------------------------------------------------
#SBATCH --job-name=sana-t2i-1.6b-1024px
#SBATCH --account=students
#SBATCH --partition=gpujl
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
# 关键: 不显式申请时 Slurm 只给 32G 内存(见 ReqTRES=mem=32G)，而 4 个 worker
# 同时 torch.load 6.4GB 权重 + 模型/文本编码器的峰值约 48-64GB，会被 OOM killer
# SIGKILL 掉（无 traceback）。--mem=0 表示使用整机内存(486GB)。
# 同理 --cpus-per-task 不设会退化成 CPUs/Task=1，令 tokenizer/PIL/VAE 等
# CPU 侧工作被 cgroup 限流。
#SBATCH --mem=0
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=/home/ganchangyi/dataset/SANA-Pixel-Dataset/logs/slurm/slurm-%j.out
#SBATCH --error=/home/ganchangyi/dataset/SANA-Pixel-Dataset/logs/slurm/slurm-%j.err

set -euo pipefail

REPO_ROOT="/fs1/private/user/ganchangyi/code/Sana-pixel/Sana"
DATASET_ROOT="/home/ganchangyi/dataset/SANA-Pixel-Dataset"

echo "=============================================================================="
echo "job id    : ${SLURM_JOB_ID:-<none>}"
echo "node      : $(hostname)"
echo "started   : $(date -Is)"
echo "gpus      : ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
echo "=============================================================================="

source /home/ganchangyi/software/miniconda3/etc/profile.d/conda.sh
conda activate pixel

cd "$REPO_ROOT"

# 全离线加载: gemma-2-2b-it 与 dc-ae 已软链进 HF 缓存。
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO_ROOT"
# 本机 relay 的 NO_PROXY 含 "[::1]"，httpx 解析会报 Invalid port。
export NO_PROXY="127.0.0.1,localhost"
export no_proxy="127.0.0.1,localhost"

# 进度条按行输出，避免 sbatch 日志被 \r 刷成一行。
export PYTHONUNBUFFERED=1

set +e
# 额外参数原样透传，例如:
#   sbatch tools/dataset_gen/sbatch_t2i_dataset.sh --repeats 2 --variant 2
python tools/dataset_gen/generate_t2i_dataset.py \
    --dataset-root "$DATASET_ROOT" \
    --gpu-ids 0,1,2,3 \
    --progress-interval 10 \
    "$@"
rc=$?
set -e

echo "=============================================================================="
echo "finished  : $(date -Is)"
echo "exit code : $rc"
echo "on disk   : $(find "$DATASET_ROOT/images" -name '*.png' | wc -l) png"
echo "=============================================================================="
exit $rc