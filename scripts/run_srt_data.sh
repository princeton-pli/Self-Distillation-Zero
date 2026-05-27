#!/bin/bash
# End-to-end SRT data generation (paper Section 2.1, Phase 1).
#
# Runs in one invocation:
#   1. self_critique_pipeline.py  -> collects D_revision = {(x, y_init, P_r, y_revised)}
#   2. prepare_data.py            -> stage1 (all remaining) + stage2 (~1k) SFT files
#
# Usage:
#   bash scripts/run_srt_data.sh                       # qwen + math (defaults)
#   MODE=code bash scripts/run_srt_data.sh             # qwen + code
#   MODEL=olmo bash scripts/run_srt_data.sh            # olmo + math
#   MODEL=olmo MODE=code bash scripts/run_srt_data.sh  # olmo + code
#
# MODEL and MODE are independent; either model runs either task.
# Override defaults via env vars (shown with their defaults below).

set -ex

MODE="${MODE:-math}"   # math | code
case "$MODE" in
    math|code) ;;
    *) echo "MODE must be 'math' or 'code' (got: $MODE)" >&2; exit 1 ;;
esac

MODEL="${MODEL:-qwen}"   # qwen | olmo  (independent of MODE)
case "$MODEL" in
    qwen|olmo) ;;
    *) echo "MODEL must be 'qwen' or 'olmo' (got: $MODEL)" >&2; exit 1 ;;
esac

# --- shared knobs ---------------------------------------------------------
TARGET_REVISION_SAMPLES="${TARGET_REVISION_SAMPLES:-6000}"
STAGE2_SIZE="${STAGE2_SIZE:-1000}"   # stage 1 takes all remaining incorrect-init tuples
NUM_RESPONSES="${NUM_RESPONSES:-3}"
NUM_REVISIONS="${NUM_REVISIONS:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"       # empty = run until target met or dataset end
SEED="${SEED:-0}"
REVISE_CORRECT="${REVISE_CORRECT:-0}"   # 1 to also revise correct y_init (paper r=1 path)

# --- model selection (independent of MODE) -------------------------------
if [ "$MODEL" = "qwen" ]; then
    MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
else
    MODEL_PATH="${MODEL_PATH:-allenai/Olmo-3-7B-Instruct}"
fi

# --- mode-specific dataset -----------------------------------------------
if [ "$MODE" = "math" ]; then
    DATASET="${DATASET:-open-r1/OpenR1-Math-220k}"
    SUBSET_ARG=""
else
    DATASET="${DATASET:-open-r1/codeforces-cots}"
    SUBSET="${SUBSET:-solutions}"
    SUBSET_ARG="--subset $SUBSET"
fi

ROOT="$(cd "$(dirname "$0")/../self-revision-training" && pwd)"
cd "$ROOT"

MODEL_NAME="$(basename "$MODEL_PATH")"
TAG="${MODEL_NAME}_${MODE}_${TARGET_REVISION_SAMPLES}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/$TAG}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-$ROOT/train_data}"
mkdir -p "$OUTPUT_DIR" "$TRAIN_DATA_DIR"

REVISION_FILE="${REVISION_FILE:-$OUTPUT_DIR/d_revision.json}"

REVISE_CORRECT_FLAG=""
[ "$REVISE_CORRECT" = "1" ] && REVISE_CORRECT_FLAG="--revise_correct"

END_FLAG=""
[ -n "$END_INDEX" ] && END_FLAG="--end_index $END_INDEX"

# ------------------------------------------------------------------ Stage A
echo "=== [1/2] Collecting D_revision (mode=$MODE, target=$TARGET_REVISION_SAMPLES) ==="
python self_critique_pipeline.py \
    --mode "$MODE" \
    --model_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    $SUBSET_ARG \
    --start_index "$START_INDEX" \
    $END_FLAG \
    --num_responses "$NUM_RESPONSES" \
    --num_revisions "$NUM_REVISIONS" \
    --batch_size "$BATCH_SIZE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_tokens "$MAX_TOKENS" \
    --gpu_memory_utilization "$GPU_MEM_UTIL" \
    --target_revision_samples "$TARGET_REVISION_SAMPLES" \
    $REVISE_CORRECT_FLAG \
    --output "$REVISION_FILE"

# ------------------------------------------------------------------ Stage B
echo "=== [2/2] Splitting: stage1 = all remaining incorrect-init, stage2 ~= $STAGE2_SIZE ==="
# Default prefix must contain "r1" (math) or "code" so sft.py's filename
# substring matching picks the right system message / response key.
if [ "$MODE" = "math" ]; then
    DEFAULT_PREFIX="r1_${MODEL_NAME}_srt"
else
    DEFAULT_PREFIX="code_${MODEL_NAME}_srt"
fi
PREFIX="${PREFIX:-$DEFAULT_PREFIX}"

python prepare_data.py \
    --input "$REVISION_FILE" \
    --output_dir "$TRAIN_DATA_DIR" \
    --stage2_size "$STAGE2_SIZE" \
    --seed "$SEED" \
    --prefix "$PREFIX"

echo
echo "Done."
echo "  D_revision    : $REVISION_FILE"
echo "  stage 1 SFT   : $TRAIN_DATA_DIR/${PREFIX}_stage1.json"
echo "  stage 2 SFT   : $TRAIN_DATA_DIR/${PREFIX}_stage2.json"
