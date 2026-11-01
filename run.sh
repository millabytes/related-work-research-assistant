#!/bin/bash
#SBATCH --partition=frida
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gpus=H100:1
#SBATCH --job-name=rw_qwen_lora
#SBATCH --signal=B:USR2@90
#SBATCH --open-mode=append
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -Eeuo pipefail

# Submit from the repository root, or override explicitly:
#   RWA_PROJECT_DIR=/shared/workspace/<account>/rw-assistant sbatch run.sh
PROJECT_DIR="${RWA_PROJECT_DIR:-${SLURM_SUBMIT_DIR}}"
cd "${PROJECT_DIR}"

# FRIDA shared storage persists across jobs. $SCRATCH is job-local and is deleted
# when the allocation ends, so model checkpoints and datasets remain in the repo.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/rw-assistant/uv}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/rw-assistant/huggingface}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-nvcr.io#nvidia/pytorch:25.06-py3}"

echo "[$(date --iso-8601=seconds)] job=${SLURM_JOB_ID:-local} host=$(hostname) project=${PROJECT_DIR}"
echo "[$(date --iso-8601=seconds)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[$(date --iso-8601=seconds)] container=${CONTAINER_IMAGE}"
nvidia-smi

# Space-separated steps; evaluation is intentionally opt-in. Examples:
#   PIPELINE_STEPS="smoke" sbatch run.sh
#   PIPELINE_STEPS="prepare train status eval-sample" sbatch run.sh
#   PIPELINE_STEPS="eval-full" sbatch run.sh
read -r -a STEPS <<< "${PIPELINE_STEPS:-prepare train status}"

PIPELINE_ARGS=(
    "${STEPS[@]}"
    --device cuda
    --mixed-precision "${MIXED_PRECISION:-bf16}"
    --epochs "${EPOCHS:-3.0}"
    --batch-size "${BATCH_SIZE:-1}"
    --gradient-accumulation "${GRADIENT_ACCUMULATION:-16}"
    --learning-rate "${LEARNING_RATE:-0.0002}"
    --resume "${RESUME:-auto}"
    --eval-limit "${EVAL_LIMIT:-10}"
    --seed "${SEED:-42}"
)

if [[ "${FORCE_PREPARE:-0}" == "1" ]]; then
    PIPELINE_ARGS+=(--force-prepare)
fi
if [[ "${FORCE_TRAIN:-0}" == "1" ]]; then
    PIPELINE_ARGS+=(--force-train)
fi

pipeline_pid=""
handle_usr2() {
    echo "[$(date --iso-8601=seconds)] Wall-time signal received; stopping child and requeueing job."
    if [[ -n "${pipeline_pid}" ]]; then
        # Terminating srun propagates termination to the Python task. The next
        # allocation resumes from the latest persisted Trainer checkpoint.
        kill -TERM "${pipeline_pid}" 2>/dev/null || true
        wait "${pipeline_pid}" 2>/dev/null || true
    fi
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap handle_usr2 USR2

# FRIDA is container-first. The image provides CUDA, PyTorch and Python 3.12;
# uv then creates/reuses the locked project .venv on persistent shared storage.
# Installing the small uv executable inside the ephemeral container is intentional.
srun --unbuffered \
    --container-image="${CONTAINER_IMAGE}" \
    --container-mount-home \
    --container-mounts="${PROJECT_DIR}:${PROJECT_DIR}" \
    --container-workdir="${PROJECT_DIR}" \
    bash -lc '
        set -Eeuo pipefail
        python -m pip install --disable-pip-version-check --quiet uv
        if [[ "${RWA_SYNC:-1}" == "1" ]]; then
            uv sync --locked
        fi
        exec uv run --no-sync python run_pipeline.py "$@"
    ' bash "${PIPELINE_ARGS[@]}" &
pipeline_pid=$!
wait "${pipeline_pid}"
