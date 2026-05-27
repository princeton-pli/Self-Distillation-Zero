# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.
# limitations under the License.
import os
import warnings
from pathlib import Path
from typing import Any, NotRequired, Optional, TypedDict, TypeVar, cast

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.grpo import _should_use_async_rollouts, refit_policy_generation
from nemo_rl.algorithms.loss_functions import (
    DistillationLossConfig,
    DistillationLossDataDict,
    DistillationLossFn,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import (
    batched_message_log_to_flat_message,
    get_keys_from_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
)
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import (
    GenerationInterface,
)
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import (
    Logger,
    LoggerConfig,
    print_message_log_samples,
)
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

# ===============================================================================
# SD-Zero Phase 2: Self-Distillation control phrases
# ===============================================================================
# Outcome-conditioned delimiter prompts P_r from SD-Zero (Section 2.1 / 2.2).
# These MUST match the strings used by the Phase 1 SRT data collection at
# self-revision-training/self_critique_pipeline.py (_P_R_PHRASES / select_p_r)
# because the SRT teacher was trained to revise immediately after seeing them.
# The cue is model-specific, so it is resolved from the teacher's model family.
_P_R_PHRASES = {
    "qwen": {
        "correct": "Let me rephrase the above solution.",
        "incorrect": "Wait, this response is wrong. Let me correct it.",
    },
    "olmo": {
        "correct": "Let me rephrase the above solution.",
        "incorrect": "Wrong. I need to redo all.",
    },
}
_DEFAULT_P_R = _P_R_PHRASES["qwen"]


def select_p_r(model_name: str) -> dict[str, str]:
    """Return the {correct, incorrect} P_r control phrases for the model family
    in `model_name` (substring match; falls back to the Qwen phrasing)."""
    name = str(model_name).lower()
    for family, phrases in _P_R_PHRASES.items():
        if family in name:
            return phrases
    return _DEFAULT_P_R

# ===============================================================================
# Configuration
# ===============================================================================
TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class DistillationConfig(TypedDict):
    # Training configuration
    num_prompts_per_step: int
    num_generations_per_prompt: int
    max_rollout_turns: int  # for multi-turn rollouts. Math Environments just have 1 turn (answering the question)
    max_num_steps: int  # maximum number of steps to train for
    max_num_epochs: int  # maximum number of epochs to train for
    val_batch_size: int
    val_period: int
    val_at_start: bool
    # Whether to run validation on the last training step. Setting this to True ensures the
    # final checkpoint has validation metrics, which is required for get_best_checkpoint_path().
    val_at_end: bool
    max_val_samples: int
    val_max_total_sequence_length: NotRequired[int]
    topk_logits_k: int
    seed: int


class DistillationSaveState(TypedDict):
    total_steps: int  # Track total number of steps across all epochs
    current_epoch: int  # Track current epoch
    current_step: int  # Track step within current epoch
    val_reward: NotRequired[
        float
    ]  # Can be any metric. Setted to 'accuracy' by default in validation.
    consumed_samples: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training


def _default_distillation_save_state() -> DistillationSaveState:
    return {
        "current_epoch": 0,
        "current_step": 0,
        "total_steps": 0,
        "val_reward": -99999999.0,  # Aligned with GRPO
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


class MasterConfig(TypedDict):
    """Main configuration structure."""

    policy: PolicyConfig  # Student model configuration
    teacher: PolicyConfig  # Teacher model configuration
    loss_fn: DistillationLossConfig  # Loss function configuration
    env: dict[str, Any]  # Environment configuration
    data: DataConfig  # Data configuration
    distillation: DistillationConfig  # Distillation configuration
    logger: LoggerConfig  # Logger configuration
    cluster: ClusterConfig  # Cluster configuration
    checkpointing: CheckpointingConfig  # Checkpointing configuration


# ===============================================================================
# Setup & Initialization
# ===============================================================================
def check_vocab_equality(
    tokenizer: TokenizerType, student_model_name: str, teacher_model_name: str
) -> None:
    """Check if the vocab of the tokenizer (student) and the teacher tokenizer are equal."""
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)

    skip_hint = "Set NRL_SKIP_DISTILLATION_TOKENIZER_CHECK=true to skip this check."

    # 1) Exact token->id mapping equality
    vocab_a = tokenizer.get_vocab()
    vocab_b = teacher_tokenizer.get_vocab()
    assert vocab_a == vocab_b, (
        f"Token->ID mapping differs between student and teacher. {skip_hint}"
    )

    # 2) Size consistency (sanity checks)
    assert len(tokenizer) == len(teacher_tokenizer), (
        f"Effective vocab sizes differ between student and teacher. {skip_hint}"
    )

    # 3) Chech model.config.vocab_size to guarantee the last dimension of the logits is the same
    student_config = AutoConfig.from_pretrained(student_model_name)
    teacher_config = AutoConfig.from_pretrained(teacher_model_name)
    assert student_config.vocab_size == teacher_config.vocab_size, (
        f"Model config vocab sizes differ between student and teacher. {skip_hint}"
    )


def setup(
    master_config: MasterConfig,
    tokenizer: TokenizerType,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    ColocatablePolicyInterface,  # student_policy
    ColocatablePolicyInterface,  # teacher_policy
    Optional[GenerationInterface],  # student_generation
    StatefulDataLoader,
    Optional[StatefulDataLoader],
    DistillationLossFn,
    Logger,
    CheckpointManager,
    DistillationSaveState,
    MasterConfig,
]:
    """Main entry point for distillation algorithm.

    Returns:
        tuple of student_policy, teacher_policy, student_generation,
        train_dataloader, val_dataloader,
        loss_fn, logger, checkpointer, distillation_save_state, master_config
    """
    # Extract configuration
    policy_config = master_config["policy"]
    teacher_config = master_config["teacher"]
    generation_config = master_config["policy"]["generation"]
    loss_config = master_config["loss_fn"]
    distillation_config = master_config["distillation"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]

    assert generation_config is not None, (
        "A generation config in the PolicyConfig is required for distillation"
    )

    # Disallow SP + packing for dtensor path
    for cfg, who in ((policy_config, "student"), (teacher_config, "teacher")):
        # DTensor sequence parallel is supported; ensure CP and SP are not enabled together
        # This incompatibility is enforced in DTensor workers during initialization.
        # Additionally, SP may not be compatible with sequence packing for some models.
        # Refer to https://github.com/NVIDIA-NeMo/RL/issues/1178 for more details.
        # Therefore, we disable SP + packing for distillation.
        dtensor_enabled = cfg["dtensor_cfg"]["enabled"]
        sequence_packing_enabled = (
            "sequence_packing" in cfg and cfg["sequence_packing"]["enabled"]
        )
        sequence_parallel_enabled = (
            "sequence_parallel" in cfg["dtensor_cfg"]
            and cfg["dtensor_cfg"]["sequence_parallel"]
        )

        if dtensor_enabled and sequence_packing_enabled and sequence_parallel_enabled:
            raise AssertionError(
                f"Distillation does not support DTensor sequence parallel + sequence packing ({who} policy). "
                "Please refer to https://github.com/NVIDIA-NeMo/RL/issues/1178 for more details."
            )

    # Set random seed
    set_seed(distillation_config["seed"])

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    distillation_save_state: Optional[DistillationSaveState] = cast(
        Optional[DistillationSaveState],
        checkpointer.load_training_info(last_checkpoint_path),
    )
    if distillation_save_state is None:
        distillation_save_state = _default_distillation_save_state()

    # ==========================
    #           Data
    # ==========================
    dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=distillation_config["num_prompts_per_step"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
    )

    if last_checkpoint_path:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        dataloader.load_state_dict(dataloader_state_dict)

    print(
        f"  ✓ Training dataloader loaded with {len(train_dataset)} samples", flush=True
    )

    # Load validation dataset if provided
    val_dataloader: Optional[StatefulDataLoader] = None
    # If validation is enabled, load the validation dataloader
    if (
        distillation_config["val_period"] > 0
        or distillation_config["val_at_start"]
        or distillation_config["val_at_end"]
    ):
        assert val_dataset is not None, (
            "Validation dataset is required if validation is enabled"
        )
        val_dataloader = StatefulDataLoader(
            val_dataset,
            batch_size=distillation_config["val_batch_size"],
            shuffle=False,
            collate_fn=rl_collate_fn,
        )
        print(
            f"  ✓ Validation dataloader loaded with {len(val_dataset)} samples",
            flush=True,
        )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...", flush=True)
    colocated_inference = generation_config["colocated"]["enabled"]

    if colocated_inference:
        cluster = RayVirtualCluster(
            name="distillation_cluster",
            bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
            * cluster_config["num_nodes"],
            use_gpus=True,
            num_gpus_per_node=cluster_config["gpus_per_node"],
            max_colocated_worker_groups=1
            if generation_config["backend"] == "megatron"
            else 3,
        )
        train_cluster = cluster
        inference_cluster = cluster
        print(
            f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes",
            flush=True,
        )
    else:
        assert generation_config["backend"] != "megatron", (
            "Non-colocated inference is not supported for Megatron generation backends. "
            "Please use vLLM backend for generation."
        )

        # train resources will be updated through overall and inference resources below
        train_gpus_per_node = cluster_config["gpus_per_node"]
        train_nodes = cluster_config["num_nodes"]

        inference_resources = generation_config["colocated"]["resources"]
        inference_gpus_per_node = inference_resources["gpus_per_node"]
        inference_nodes = inference_resources["num_nodes"]

        # validate and configure resources
        if cluster_config["num_nodes"] == 1:
            assert (
                inference_gpus_per_node is not None and inference_gpus_per_node > 0
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set to a value > 0 "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_gpus_per_node}."
            )
            assert inference_nodes is None or inference_nodes == 1, (
                "policy.generation.colocated.resources.num_nodes must be 1 or set to null "
                "when cluster.num_nodes = 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            inference_nodes = 1
            train_gpus_per_node -= inference_gpus_per_node
        else:
            assert inference_nodes > 0, (
                "policy.generation.colocated.resources.num_nodes must be > 0 "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got {inference_nodes}."
            )
            assert (
                inference_gpus_per_node is not None
                and inference_gpus_per_node == cluster_config["gpus_per_node"]
            ), (
                "policy.generation.colocated.resources.gpus_per_node must be explicitly set and equal to cluster.gpus_per_node "
                "when cluster.num_nodes > 1 and inference is non-colocated, "
                f"but got inference_gpus_per_node={inference_gpus_per_node}, cluster.gpus_per_node={cluster_config['gpus_per_node']}."
            )
            train_nodes -= inference_nodes

        # create clusters
        train_cluster = RayVirtualCluster(
            name="distillation_train_cluster",
            bundle_ct_per_node_list=[train_gpus_per_node] * train_nodes,
            use_gpus=True,
            num_gpus_per_node=train_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        inference_cluster = RayVirtualCluster(
            name="distillation_inference_cluster",
            bundle_ct_per_node_list=[inference_gpus_per_node] * inference_nodes,
            use_gpus=True,
            num_gpus_per_node=inference_gpus_per_node,
            max_colocated_worker_groups=3,
        )
        print(
            f"  ✓ Separate clusters created: train={train_nodes}x{train_gpus_per_node}GPUs, inference={inference_nodes}x{inference_gpus_per_node}GPUs",
            flush=True,
        )

    # ==========================
    #      Teacher Policy
    # ==========================
    print("\n▶ Setting up teacher policy...", flush=True)
    # Checkpoint paths
    weights_path = None
    optimizer_path = None

    if not bool(os.getenv("NRL_SKIP_DISTILLATION_TOKENIZER_CHECK", False)):
        check_vocab_equality(
            tokenizer, policy_config["model_name"], teacher_config["model_name"]
        )

    if "megatron_cfg" in teacher_config and teacher_config["megatron_cfg"]["enabled"]:
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        teacher_config["megatron_cfg"]["train_iters"] = total_train_iters

    teacher_policy = Policy(
        name_prefix="teacher",
        cluster=train_cluster,
        config=teacher_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=False,
        init_reference_model=False,
    )
    teacher_policy.offload_after_refit()

    # ==========================
    #    Student Generation Interface
    # ==========================
    backend = generation_config["backend"]
    generation_config["model_name"] = policy_config["model_name"]  # Needed for vLLM

    if backend == "megatron":
        student_generation = None
    elif backend == "vllm":
        generation_config = cast(VllmConfig, generation_config)
        if "vllm_cfg" in generation_config:
            ## make vllm hf overrides match the training policy
            generation_config["vllm_cfg"]["hf_overrides"] = policy_config.get(
                "hf_config_overrides", {}
            )

            # Ensure max_model_len is large enough for validation if val_max_total_sequence_length is set
            if "val_max_total_sequence_length" in distillation_config:
                val_max_seq_len = distillation_config["val_max_total_sequence_length"]
                if generation_config["vllm_cfg"]["max_model_len"] < val_max_seq_len:
                    print(
                        f"  [INFO] Updating vLLM max_model_len from {generation_config['vllm_cfg']['max_model_len']} "
                        f"to {val_max_seq_len} to support validation.",
                        flush=True,
                    )
                    generation_config["vllm_cfg"]["max_model_len"] = val_max_seq_len

        student_generation = VllmGeneration(
            cluster=inference_cluster, config=generation_config
        )
        student_generation.finish_generation()
        print(
            f"  ✓ Using vLLM backend for generation with {policy_config['model_name']}",
            flush=True,
        )

    # ==========================
    #      Student Policy
    # ==========================
    print("\n▶ Setting up student policy...", flush=True)

    # Checkpoint paths
    if last_checkpoint_path:
        weights_path = Path(last_checkpoint_path) / "policy" / "weights"
        optimizer_path = Path(last_checkpoint_path) / "policy" / "optimizer"
        print(f"  Loading student weights from checkpoint: {weights_path}", flush=True)
    else:
        weights_path = None
        optimizer_path = None
        print(f"  Loading student weights from base model: {policy_config['model_name']}", flush=True)

    if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
        ## NOTE: this is equal to the total number of scheduler steps
        total_train_iters = min(
            distillation_config["max_num_steps"],
            distillation_config["max_num_epochs"] * len(dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters

    student_policy = Policy(
        name_prefix="student",
        cluster=train_cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
    )

    if student_generation is not None:
        state_dict_info = student_policy.prepare_refit_info()
        student_generation.prepare_refit_info(state_dict_info)

    # if it is not colocated inference, initialize collective communication for update weights
    if not colocated_inference:
        ip, port = train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication", flush=True)
        train_world_size = train_cluster.world_size()
        # inference cluster + head node of the train cluster
        world_size = train_world_size + inference_nodes * inference_gpus_per_node
        # init collective
        futures_train = student_policy.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        futures_inference = student_generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )  # type: ignore
        # wait for all futures to complete
        ray.get(futures_train + futures_inference)

    loss_fn = DistillationLossFn(loss_config)

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n", flush=True)

    return (
        student_policy,
        teacher_policy,
        student_generation,
        dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        distillation_save_state,
        master_config,
    )


# ===============================================================================
# SD-Zero Phase 2: build teacher inputs from (x, y, r)
# ===============================================================================
def _phase2_select_p_r(reward: Any, p_r_phrases: dict[str, str]) -> str:
    """Pick the outcome-conditioned delimiter P_r from the binary reward of y."""
    r = float(reward.item()) if torch.is_tensor(reward) else float(reward)
    return p_r_phrases["correct"] if r > 0.5 else p_r_phrases["incorrect"]


def _phase2_truncate_student_responses(
    repeated_batch: BatchedDataDict[Any],
    tokenizer: TokenizerType,
    max_seq_len: int,
    p_r_phrases: dict[str, str],
) -> None:
    """In-place truncate y so the Phase 2 teacher input fits within max_seq_len.

    The SRT teacher input is |x| + 2|y| + |delim| tokens (x in the user turn, then y,
    P_r, and y again teacher-forced inside the assistant turn). If that exceeds
    max_seq_len, shrink y symmetrically. Samples whose |x|+|delim| alone exceed the
    budget are masked out via loss_multiplier=0.
    """
    if "total_reward" not in repeated_batch:
        raise RuntimeError(
            "Phase 2 self-distillation requires the binary reward of y in "
            "repeated_batch['total_reward'] (produced by run_multi_turn_rollout). "
            f"Got keys: {list(repeated_batch.keys())}"
        )
    total_rewards = repeated_batch["total_reward"]

    for i, message_log in enumerate(repeated_batch["message_log"]):
        user_msg = next((m for m in message_log if m["role"] == "user"), None)
        assistant_msg = next(
            (m for m in message_log if m["role"] == "assistant"), None
        )
        if user_msg is None or assistant_msg is None:
            continue

        p_r = _phase2_select_p_r(total_rewards[i], p_r_phrases)
        delim_tokens = tokenizer(
            f"\n\n{p_r}\n\n", return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        user_len = user_msg["token_ids"].numel()
        y_len = assistant_msg["token_ids"].numel()
        delim_len = delim_tokens.numel()
        teacher_total = user_len + 2 * y_len + delim_len
        if teacher_total <= max_seq_len:
            continue

        allowed_y_len = (max_seq_len - user_len - delim_len) // 2
        if allowed_y_len <= 0:
            print(
                f"  [WARNING] Phase 2: |x|+|delim|={user_len + delim_len} exceeds "
                f"max_teacher_tokens={max_seq_len} for sample {i}; zeroing "
                f"loss_multiplier.",
                flush=True,
            )
            if "loss_multiplier" in repeated_batch:
                repeated_batch["loss_multiplier"][i] = 0.0
            # Keep at least one token in y so flattening doesn't break.
            keep = max(1, min(4, y_len))
            assistant_msg["token_ids"] = assistant_msg["token_ids"][:keep]
            assistant_msg["content"] = tokenizer.decode(
                assistant_msg["token_ids"], skip_special_tokens=True
            )
            if "generation_logprobs" in assistant_msg:
                assistant_msg["generation_logprobs"] = (
                    assistant_msg["generation_logprobs"][:keep]
                )
            continue

        print(
            f"  [INFO] Phase 2: truncating y for sample {i} from {y_len} to "
            f"{allowed_y_len} tokens so |x|+2|y|+|delim| <= {max_seq_len}.",
            flush=True,
        )
        assistant_msg["token_ids"] = assistant_msg["token_ids"][:allowed_y_len]
        assistant_msg["content"] = tokenizer.decode(
            assistant_msg["token_ids"], skip_special_tokens=True
        )
        if "generation_logprobs" in assistant_msg:
            assistant_msg["generation_logprobs"] = (
                assistant_msg["generation_logprobs"][:allowed_y_len]
            )


def _phase2_build_teacher_message_log(
    repeated_batch: BatchedDataDict[Any],
    tokenizer: TokenizerType,
    p_r_phrases: dict[str, str],
) -> list[Any]:
    """Build the SRT-teacher message log for SD-Zero Phase 2 (Section 2.2).

    For each sample, the teacher's input sequence is

        chat_template(user=x, add_generation_prompt=True)  # ends with assistant-start marker
        + y                                                # student's full response (y_init in SRT)
        + "\\n\\n" + P_r + "\\n\\n"                        # outcome-conditioned delimiter
        + y                                                # teacher-forced y_<t for t=1..|y|

    The token_loss_mask is 1 only on the SECOND copy of y. The teacher's logits at those
    positions provide pi_theta_SRT(. | x, y, P_r, y_<t) — the distillation target in the
    Section 2.2 KL loss. P_r is chosen from the binary reward r(y) on each sample.
    """
    total_rewards = repeated_batch["total_reward"]
    teacher_message_log: list[Any] = []

    for i, message_log in enumerate(repeated_batch["message_log"]):
        user_msg = next((m for m in message_log if m["role"] == "user"), None)
        assistant_msg = next(
            (m for m in message_log if m["role"] == "assistant"), None
        )
        if user_msg is None or assistant_msg is None:
            import copy as _copy
            teacher_message_log.append(_copy.deepcopy(message_log))
            continue

        p_r = _phase2_select_p_r(total_rewards[i], p_r_phrases)
        delim_text = f"\n\n{p_r}\n\n"
        delim_tokens = tokenizer(
            delim_text, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]

        user_tokens = user_msg["token_ids"]
        y_tokens = assistant_msg["token_ids"]

        # Concatenate chat_user(x) + y + delim into the teacher's "user-side"
        # prefix; the SECOND y becomes the assistant-side tokens where the loss
        # mask (and thus the alignment to student logits) is applied.
        teacher_prefix_tokens = torch.cat(
            [user_tokens, y_tokens, delim_tokens]
        )
        user_content = (
            user_msg.get("content", "")
            if isinstance(user_msg.get("content"), str)
            else ""
        )
        assistant_content = (
            assistant_msg.get("content", "")
            if isinstance(assistant_msg.get("content"), str)
            else ""
        )
        teacher_prefix_text = user_content + assistant_content + delim_text

        teacher_message_log.append(
            [
                {
                    "role": "user",
                    "content": teacher_prefix_text,
                    "token_ids": teacher_prefix_tokens,
                    "token_loss_mask": torch.zeros_like(teacher_prefix_tokens),
                },
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "token_ids": y_tokens,
                    "token_loss_mask": torch.ones_like(y_tokens),
                },
            ]
        )

    return teacher_message_log


# ===============================================================================
# Training & Validation
# ===============================================================================


def distillation_train(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    student_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: DistillationLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: DistillationSaveState,
    master_config: MasterConfig,
) -> None:
    """Run Distillation training algorithm."""
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    NEED_REFIT = True
    # If student_generation is None, use the student_policy as the generation interface (megatron framework backend)
    if student_generation is None:
        student_generation = student_policy  # type: ignore
        NEED_REFIT = False
    POLICY_GENERATION_STALE = True  # tracks if generation needs a refit before running
    assert student_generation is not None  # for mypy type check

    # common config/state items
    current_epoch = distillation_save_state["current_epoch"]  # current epoch
    current_step = distillation_save_state[
        "current_step"
    ]  # current step within current epoch
    total_steps = distillation_save_state[
        "total_steps"
    ]  # total number of steps across all epochs
    consumed_samples = distillation_save_state["consumed_samples"]
    total_valid_tokens = distillation_save_state["total_valid_tokens"]
    val_period = master_config["distillation"]["val_period"]
    val_at_start = master_config["distillation"]["val_at_start"]
    val_at_end = master_config["distillation"]["val_at_end"]
    colocated_inference = master_config["policy"]["generation"]["colocated"]["enabled"]
    max_epochs = master_config["distillation"][
        "max_num_epochs"
    ]  # max number of epochs to train for
    max_steps = master_config["distillation"][
        "max_num_steps"
    ]  # max number of steps to train for

    # P_r delimiter must match the SRT teacher's model family (Phase 1 _P_R_PHRASES).
    p_r_phrases = select_p_r(master_config["teacher"]["model_name"])
    print(f"Phase 2 P_r (incorrect): {p_r_phrases['incorrect']!r}", flush=True)

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...", flush=True)
        if NEED_REFIT and POLICY_GENERATION_STALE:
            refit_policy_generation(
                student_policy, student_generation, colocated_inference
            )
            POLICY_GENERATION_STALE = False
        else:
            student_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            student_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=total_steps,
            master_config=master_config,
        )
        student_generation.finish_generation()
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    # Run distillation training (multi-epoch until reaching max_num_steps or max_num_epochs)
    batch: BatchedDataDict[DatumSpec]

    while total_steps < max_steps and current_epoch < max_epochs:
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}",
            flush=True,
        )

        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(student_policy, total_steps + 1)
            if student_policy != student_generation:
                maybe_gpu_profile_step(student_generation, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # Prepare batch
                print("▶ Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    # Repeat batch items
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(
                            master_config["distillation"]["num_generations_per_prompt"]
                        )
                    )

                # Generate responses - this updates the LLMMessageLogType in repeated_batch
                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation"):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            student_policy,
                            student_generation,
                            colocated_inference,
                            timer=timer,
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()

                with timer.time("generation"):
                    # Use async rollouts if vLLM async engine is enabled
                    if _should_use_async_rollouts(master_config):
                        (
                            repeated_batch,
                            rollout_metrics,
                        ) = run_async_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["distillation"][
                                "max_rollout_turns"
                            ],
                            greedy=False,
                        )
                    else:
                        repeated_batch, rollout_metrics = run_multi_turn_rollout(
                            policy_generation=student_generation,
                            input_batch=repeated_batch,
                            tokenizer=tokenizer,
                            task_to_env=task_to_env,
                            max_seq_len=master_config["policy"][
                                "max_total_sequence_length"
                            ],
                            max_rollout_turns=master_config["distillation"][
                                "max_rollout_turns"
                            ],
                            greedy=False,
                        )
                    student_generation.finish_generation()

                with timer.time("data_processing"):
                    # SD-Zero Phase 2 (Section 2.2): the teacher conditions on
                    # (x, y, P_r, y_<t), where P_r is chosen from the binary reward of y.
                    # The teacher input has length |x| + 2|y| + |delim|; truncate y here
                    # so both student and teacher data agree on the same shortened y.
                    max_seq_len = master_config["distillation"]["max_teacher_tokens"]
                    _phase2_truncate_student_responses(
                        repeated_batch, tokenizer, max_seq_len, p_r_phrases
                    )

                    # Add loss mask and advantages to each message in LLMMessageLogType
                    for message_log in repeated_batch["message_log"]:
                        for message in message_log:
                            if message["role"] == "assistant":
                                message["token_loss_mask"] = torch.ones_like(
                                    message["token_ids"]
                                )
                            else:
                                message["token_loss_mask"] = torch.zeros_like(
                                    message["token_ids"]
                                )

                    # Convert updated LLMMessageLogType to FlatMessagesType for training
                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        repeated_batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    # Create training data from flattened messages
                    train_data = BatchedDataDict[DistillationLossDataDict](
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    # this will be mini-batched inside the policy, so maintain the packed multimodal structure
                    train_data.update(
                        flat_messages.get_multimodal_dict(as_tensors=False)
                    )
                    train_data.to("cpu")

                    # SD-Zero Phase 2 (Section 2.2): build the SRT-teacher input
                    # chat(x) + y + sep + P_r + sep + y, with the loss mask on the
                    # SECOND copy of y. P_r is selected per-sample from the binary
                    # reward of y. The existing logit-alignment block below maps the
                    # teacher's per-token distributions at the masked positions back
                    # to the student's y positions for the KL loss.
                    print(
                        "  Building Phase 2 teacher input: chat(x) + y + P_r + y "
                        "(P_r from binary reward of y)...",
                        flush=True,
                    )
                    teacher_message_log = _phase2_build_teacher_message_log(
                        repeated_batch, tokenizer, p_r_phrases
                    )
                    teacher_flat_messages, teacher_input_lengths = (
                        batched_message_log_to_flat_message(
                            teacher_message_log,
                            pad_value_dict={"token_ids": tokenizer.pad_token_id},
                            make_sequence_length_divisible_by=master_config["policy"][
                                "make_sequence_length_divisible_by"
                            ],
                        )
                    )
                    teacher_train_data = BatchedDataDict(
                        {
                            "input_ids": teacher_flat_messages["token_ids"],
                            "input_lengths": teacher_input_lengths,
                            "token_mask": teacher_flat_messages["token_loss_mask"],
                            "sample_mask": repeated_batch["loss_multiplier"],
                        }
                    )
                    teacher_train_data.update(
                        teacher_flat_messages.get_multimodal_dict(as_tensors=False)
                    )
                    teacher_train_data.to("cpu")

                    if current_step == 0:
                        student_resp_len = train_data["token_mask"][0].sum().item()
                        teacher_resp_len = teacher_train_data["token_mask"][0].sum().item()
                        print(
                            f"  Phase 2 sample 0: |y| (student) = {student_resp_len}, "
                            f"|y| (teacher masked) = {teacher_resp_len}",
                            flush=True,
                        )

                print("▶ Preparing for teacher logprob inference...", flush=True)
                with timer.time("teacher_logprob_inference_prep"):
                    teacher_policy.prepare_for_lp_inference()

                print("▶ Computing teacher logprobs...", flush=True)
                with timer.time("teacher_logprob_inference"):
                    teacher_topk = teacher_policy.get_topk_logits(
                        teacher_train_data,
                        k=master_config["distillation"]["topk_logits_k"],
                        timer=timer,
                    )

                    # Align teacher logits at the SECOND-y positions in the teacher
                    # input to the corresponding y positions in the student input.
                    # Logits at index i predict the token at i+1, so for each masked
                    # token at position p the predicting logit lives at p-1.
                    aligned_logits = torch.zeros(
                        train_data["input_ids"].shape
                        + (teacher_topk["topk_logits"].shape[-1],),
                        dtype=teacher_topk["topk_logits"].dtype,
                        device=teacher_topk["topk_logits"].device,
                    )
                    aligned_indices = torch.zeros(
                        train_data["input_ids"].shape
                        + (teacher_topk["topk_indices"].shape[-1],),
                        dtype=teacher_topk["topk_indices"].dtype,
                        device=teacher_topk["topk_indices"].device,
                    )

                    batch_size = train_data["input_ids"].shape[0]
                    for i in range(batch_size):
                        student_mask = train_data["token_mask"][i].bool()
                        teacher_mask = teacher_train_data["token_mask"][i].bool()
                        if not student_mask.any() or not teacher_mask.any():
                            print(
                                f"  [WARNING] Sample {i}: empty response mask, "
                                "skipping alignment.",
                                flush=True,
                            )
                            continue

                        student_indices = torch.nonzero(student_mask).squeeze()
                        teacher_indices = torch.nonzero(teacher_mask).squeeze()
                        if student_indices.dim() == 0:
                            student_indices = student_indices.unsqueeze(0)
                        if teacher_indices.dim() == 0:
                            teacher_indices = teacher_indices.unsqueeze(0)

                        s_len = student_indices.numel()
                        t_len = teacher_indices.numel()
                        if s_len != t_len:
                            print(
                                f"  [WARNING] Sample {i}: |y| mismatch student="
                                f"{s_len} vs teacher={t_len}; truncating to min.",
                                flush=True,
                            )
                            m = min(s_len, t_len)
                            student_indices = student_indices[:m]
                            teacher_indices = teacher_indices[:m]

                        student_logit_idx = student_indices - 1
                        teacher_logit_idx = teacher_indices - 1
                        if (student_logit_idx < 0).any() or (teacher_logit_idx < 0).any():
                            print(
                                f"  [ERROR] Sample {i}: logit index < 0; prompt "
                                "must have length >= 1.",
                                flush=True,
                            )
                            continue

                        aligned_logits[i, student_logit_idx] = teacher_topk[
                            "topk_logits"
                        ][i, teacher_logit_idx]
                        aligned_indices[i, student_logit_idx] = teacher_topk[
                            "topk_indices"
                        ][i, teacher_logit_idx]

                    train_data["teacher_topk_logits"] = aligned_logits
                    train_data["teacher_topk_indices"] = aligned_indices

                print("▶ Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    teacher_policy.offload_after_refit()
                    student_policy.prepare_for_training()  # set model train and reload optim to GPU
                    POLICY_GENERATION_STALE = True

                print("▶ Training policy...", flush=True)
                with timer.time("policy_training"):
                    train_results = student_policy.train(
                        train_data,
                        loss_fn,
                        timer=timer,
                    )

                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                # Run validation if it's a validation step or last step with val_at_end
                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    if NEED_REFIT and POLICY_GENERATION_STALE:
                        refit_policy_generation(
                            student_policy, student_generation, colocated_inference
                        )
                        POLICY_GENERATION_STALE = False
                    else:
                        student_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        student_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                    )
                    student_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "mean_prompt_length": repeated_batch["length"].numpy(),
                    "total_num_tokens": input_lengths.numpy(),
                }
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {
                        "lr",
                        "wd",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()
                metrics.update(rollout_metrics)
                total_valid_tokens += metrics["global_valid_toks"]

                ## Checkpointing
                consumed_samples += master_config["distillation"][
                    "num_prompts_per_step"
                ]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                # +1 because total_steps is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    student_policy.prepare_for_training()

                    distillation_save_state["current_epoch"] = current_epoch
                    distillation_save_state["current_step"] = current_step + 1
                    distillation_save_state["total_steps"] = total_steps + 1
                    distillation_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        distillation_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in distillation_save_state:
                        del distillation_save_state["val_reward"]
                    distillation_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                            f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                            f" e.g. 'val_reward --> 'val:accuracy'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in distillation_save_state:
                                del distillation_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            distillation_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, distillation_save_state, master_config
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # Log training data
            log_data = {
                "content": flat_messages["content"],
                "input_lengths": input_lengths.tolist(),
                "teacher_content": teacher_flat_messages["content"],
                "teacher_input_lengths": teacher_input_lengths.tolist(),
            }

            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps + 1}.jsonl"
            )

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )  # type: ignore

            print("\n📊 Training Results:")

            print(f"  • Loss: {metrics['loss']:.4f}")
            print(
                f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
            )
            if "total_flops" in train_results:
                total_tflops = (
                    train_results["total_flops"]
                    / timing_metrics["policy_training"]
                    / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)",
                    flush=True,
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%",
                        flush=True,
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops

            print("\n⏱️  Timing:", flush=True)
            # Display total time first, separately
            total_time = timing_metrics.get("total_step_time", 0)

            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            metrics.update(
                {
                    "tokens_per_sec_per_gpu": metrics["total_num_tokens"]
                    / total_time
                    / total_num_gpus
                }
            )

            print(f"  • Total step time: {total_time:.2f}s", flush=True)

            # Display all other timing metrics
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= max_steps:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        # End of epoch
        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch


def validate(
    policy_generation: GenerationInterface,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    step: int,
    master_config: MasterConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation", flush=True)
        return {}, {}

    if val_task_to_env is None:
        print(
            "  ⚠️ No validation task to environment mapping provided, skipping validation",
            flush=True,
        )
        return {}, {}

    timer = Timer()
    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...", flush=True)

        total_rewards = []  # Can be any metric. Setted to 'accuracy' by default.
        total_lengths = []
        all_message_logs = []  # Collect all message logs

        max_batches = (
            master_config["distillation"]["max_val_samples"]
            // master_config["distillation"]["val_batch_size"]
        )

        # Get validation max sequence length, defaulting to policy max sequence length
        val_max_seq_len = master_config["distillation"].get(
            "val_max_total_sequence_length",
            32768
        )
        print(f"  Using validation max sequence length: {val_max_seq_len}", flush=True)

        for batch_idx, val_batch in enumerate(val_dataloader):
            if batch_idx >= max_batches:
                break

            # Generate responses (updates the LLMMessageLogType in batch_with_msg_logs)
            # Use async rollouts if vLLM async engine is enabled
            if _should_use_async_rollouts(master_config):
                val_batch, gen_metrics = run_async_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=val_max_seq_len,
                    max_rollout_turns=master_config["distillation"][
                        "max_rollout_turns"
                    ],
                    greedy=False,
                )
            else:
                val_batch, gen_metrics = run_multi_turn_rollout(
                    policy_generation,
                    val_batch,
                    tokenizer,
                    val_task_to_env,
                    max_seq_len=val_max_seq_len,
                    max_rollout_turns=master_config["distillation"][
                        "max_rollout_turns"
                    ],
                    greedy=False,
                )
            rewards = val_batch["total_reward"]

            total_rewards.extend(rewards.tolist())
            total_lengths.append(gen_metrics["mean_gen_tokens_per_sample"])

            # Collect message logs for later display
            to_env = [
                get_keys_from_message_log(
                    val_batch["message_log"][i], ["role", "content"]
                )
                for i in range(len(val_batch["message_log"]))
            ]

            all_message_logs.extend(to_env)

        # Calculate validation metrics
        accuracy = (
            sum(total_rewards) / len(total_rewards) if len(total_rewards) > 0 else 0
        )
        avg_length = (
            sum(total_lengths) / len(total_lengths) if len(total_lengths) > 0 else 0
        )

        val_metrics = {
            "accuracy": accuracy,
            "avg_length": avg_length,
        }

        # Print sample conversations only once at the end of validation
        try:
            print_message_log_samples(
                all_message_logs,
                total_rewards,
                num_samples=min(
                    master_config["logger"]["num_val_samples_to_print"],
                    len(all_message_logs),
                ),
                step=step,
            )
        except Exception as e:
            print(f"\n  ⚠️ Error displaying message samples: {str(e)}")
            print("  ⚠️ Continuing validation without displaying samples...", flush=True)

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    # Print summary of validation results
    print("\n📊 Validation Results:")
    print(f"    • Accuracy: {accuracy:.4f}")
    print(f"    • Average response length: {avg_length:.1f} tokens")
    print(f"    • Samples processed: {len(total_rewards)}", flush=True)

    # Print timing information
    print("\n  ⏱️  Validation Timing:")
    validation_time = timing_metrics.get("total_validation_time", 0)
    print(f"    • Total validation time: {validation_time:.2f}s", flush=True)

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics
