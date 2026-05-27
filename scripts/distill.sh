#!/bin/bash
# SD-Zero Phase 2 — On-policy Self-Distillation via Revision Feedback.
#
# Paper Section 2.2. For each problem x:
#   1. student samples y ~ pi_theta(. | x)
#   2. binary reward r = 1[answer(y) == ground_truth]
#   3. P_r = "Let me rephrase the above solution."           (r=1)
#         or "Wait, this response is wrong. Let me correct it." (r=0)
#   4. frozen SRT teacher sees  chat(x) + y + P_r + y  and provides per-token
#      logits over the SECOND copy of y
#   5. student is trained to minimize
#        D_KL( pi_theta(. | x, y_<t) || pi_theta_SRT(. | x, y, P_r, y_<t) )
#
# Both student and teacher should be initialized from the Phase 1 SRT
# checkpoint produced by scripts/sft.sh.
#
# Usage:
#   bash scripts/distill.sh
# Override defaults via env vars (see DEFAULTS block below).

set -ex

# --- defaults -------------------------------------------------------------
STUDENT_MODEL="${STUDENT_MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
TEACHER_MODEL="${TEACHER_MODEL:-$STUDENT_MODEL}"      # default: SRT teacher = student
TRAIN_DATASET="${TRAIN_DATASET:-DeepScaler}"
VAL_DATASET="${VAL_DATASET:-AIME2024}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-8192}"
MAX_TEACHER_TOKENS="${MAX_TEACHER_TOKENS:-32768}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-128}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-128}"
MAX_NUM_STEPS="${MAX_NUM_STEPS:-1000}"
MAX_NUM_EPOCHS="${MAX_NUM_EPOCHS:-5}"
LR="${LR:-5e-6}"
TOPK="${TOPK:-64}"
KL_TYPE="${KL_TYPE:-mixed}"           # forward | reverse | mixed
MIXED_KL_WEIGHT="${MIXED_KL_WEIGHT:-0.5}"

# --- paths ---------------------------------------------------------------
ROOT="$(cd "$(dirname "$0")/../self-distillation" && pwd)"
cd "$ROOT"

# Load the Phase 2 CUDA/cuDNN/NCCL environment (requires `uv venv` to have
# been run inside self-distillation/ at least once — see SD-Zero/README.md).
# shellcheck disable=SC1091
source "$ROOT/setup_env.sh"

STUDENT_NAME=$(basename "$STUDENT_MODEL")
TEACHER_NAME=$(basename "$TEACHER_MODEL")
RUN_TAG="${RUN_TAG:-${STUDENT_NAME}_teacher-${TEACHER_NAME}_${TRAIN_DATASET}_${MAX_TOTAL_SEQUENCE_LENGTH}tok}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/$RUN_TAG}"
mkdir -p "$OUTPUT_DIR"

gpu_count=$(nvidia-smi -L | wc -l)
echo "Number of GPUs: $gpu_count"
echo "Student: $STUDENT_MODEL"
echo "Teacher: $TEACHER_MODEL  (SRT model from Phase 1)"
echo "Output : $OUTPUT_DIR"

# --- Ray isolation (multi-job-safe) --------------------------------------
export RAY_DEFAULT_PORT=$(shuf -i 10000-29000 -n 1)
export RAY_TEMP_DIR="/tmp/ray_job_${SLURM_JOB_ID:-$$}_${RAY_DEFAULT_PORT}"
mkdir -p "$RAY_TEMP_DIR"
export RAY_ADDRESS=""

# --- launch ---------------------------------------------------------------
uv run examples/run_distillation.py \
    --config examples/configs/distillation_math.yaml \
    policy.model_name="$STUDENT_MODEL" \
    teacher.model_name="$TEACHER_MODEL" \
    policy.max_total_sequence_length="$MAX_TOTAL_SEQUENCE_LENGTH" \
    distillation.max_teacher_tokens="$MAX_TEACHER_TOKENS" \
    distillation.num_prompts_per_step="$NUM_PROMPTS_PER_STEP" \
    distillation.num_generations_per_prompt=1 \
    distillation.max_num_steps="$MAX_NUM_STEPS" \
    distillation.max_num_epochs="$MAX_NUM_EPOCHS" \
    distillation.topk_logits_k="$TOPK" \
    distillation.val_at_start=true \
    distillation.val_at_end=true \
    loss_fn.kl_type="$KL_TYPE" \
    loss_fn.mixed_kl_weight="$MIXED_KL_WEIGHT" \
    data.train.dataset_name="$TRAIN_DATASET" \
    data.validation.dataset_name="$VAL_DATASET" \
    policy.train_global_batch_size="$TRAIN_GLOBAL_BATCH_SIZE" \
    policy.optimizer.kwargs.lr="$LR" \
    cluster.gpus_per_node="$gpu_count" \
    cluster.num_nodes=1 \
    policy.dtensor_cfg.tensor_parallel_size="$gpu_count" \
    policy.dtensor_cfg.context_parallel_size=1 \
    teacher.dtensor_cfg.tensor_parallel_size="$gpu_count" \
    teacher.dtensor_cfg.context_parallel_size=1 \
    policy.generation.vllm_cfg.tensor_parallel_size="$gpu_count" \
    checkpointing.checkpoint_dir="$OUTPUT_DIR" \
    logger.wandb.name="$RUN_TAG"
