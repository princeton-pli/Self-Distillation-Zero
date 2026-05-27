

cd /scratch/gpfs/ARORA/yh0068/Self-Distillation/nemo-rl

source /scratch/gpfs/ARORA/yh0068/Self-Distillation/nemo-rl/setup_env.sh

CKPT_PATH="/scratch/gpfs/ARORA/yh0068/Self-Distillation/nemo-rl/results/r1_Qwen3-4B_logit_distill_12k_samples_12k/step_60"


# NRL_FORCE_REBUILD_VENVS=true \
uv run python examples/converters/convert_dcp_to_hf.py \
    --config ${CKPT_PATH}/config.yaml \
    --dcp-ckpt-path ${CKPT_PATH}/policy/weights \
    --hf-ckpt-path ${CKPT_PATH}/r1_Qwen3-4B_logit_distill_12k_samples_step_60