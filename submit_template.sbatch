#!/bin/bash
#SBATCH --account=hai_1259
#SBATCH --nodes=1
#SBATCH --partition=develbooster
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:10:00
#SBATCH --gres=gpu:4
#SBATCH -o ./.cache/slurm_output/%j.log

set -x
ulimit -c 0

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$PROJECT_DIR"
mkdir -p ./.cache/slurm_output

# Without this, srun does not inherit cpus-per-task from sbatch.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Single-node defaults for the auto-spawned torch.distributed launcher.
echo "----------------------------------"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST}"
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-7010}"
export GPUS_PER_NODE=4
export DEVICES_PER_NODE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_NODES="${SLURM_JOB_NUM_NODES}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib0}"

# Try to reduce link flips.
export NCCL_IB_TIMEOUT=100
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10

echo "MASTER_ADDR:MASTER_PORT=${MASTER_ADDR}:${MASTER_PORT}"
echo "Job id: ${SLURM_JOB_ID}"
echo "----------------------------------"

source ./llm_env/activate.sh

CACHE_ROOT="${CACHE_ROOT:-${PROJECT_DIR}/.cache/autoresearch}"
DATA_DIR="${DATA_DIR:-${CACHE_ROOT}/data}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${CACHE_ROOT}/tokenizer}"
KERNEL_DIR="${KERNEL_DIR:-${CACHE_ROOT}/kernels}"
KERNEL_MANIFEST="${KERNEL_DIR}/manifest.json"

echo "CACHE_ROOT=${CACHE_ROOT}"
echo "DATA_DIR=${DATA_DIR}"
echo "TOKENIZER_DIR=${TOKENIZER_DIR}"
echo "KERNEL_DIR=${KERNEL_DIR}"

if [ ! -f "${TOKENIZER_DIR}/tokenizer.pkl" ]; then
    echo "Missing tokenizer cache at ${TOKENIZER_DIR}/tokenizer.pkl"
    echo "Run 'source ./llm_env/activate.sh && python prepare.py' on a login node first."
    exit 1
fi

if [ ! -f "${TOKENIZER_DIR}/token_bytes.pt" ]; then
    echo "Missing token_bytes cache at ${TOKENIZER_DIR}/token_bytes.pt"
    echo "Run 'source ./llm_env/activate.sh && python prepare.py' on a login node first."
    exit 1
fi

if ! find "${DATA_DIR}" -maxdepth 1 -name 'shard_*.parquet' -print -quit | grep -q .; then
    echo "Missing parquet shards under ${DATA_DIR}"
    echo "Run 'source ./llm_env/activate.sh && python prepare.py' on a login node first."
    exit 1
fi

if [ ! -f "${KERNEL_MANIFEST}" ]; then
    echo "Missing kernel manifest at ${KERNEL_MANIFEST}"
    echo "Run 'source ./llm_env/activate.sh && python prepare.py' on a login node first."
    exit 1
fi

python train.py "$@"
