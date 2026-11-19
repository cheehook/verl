#!/usr/bin/env bash
# GRPO + LoRA | text | vLLM rollout | FSDP training | NVIDIA GPUs or Intel GPUs (XPU)
# Canonical LoRA fine-tuning baseline for dense text LLMs (GSM8K).
# Intel GPUs: run inside docker/Dockerfile.xpu; tested on 4x Arc Pro B60 (24 GB) with NGPUS_PER_NODE=4.

set -xeuo pipefail

# ---- user-adjustable ----
# DEVICE (gpu|intel) is auto-detected by probing torch.xpu; override only for special cases.
if [ -z "${DEVICE:-}" ]; then
    if python3 -c 'import torch,sys; sys.exit(not torch.xpu.is_available())' 2>/dev/null; then
        DEVICE=intel
    else
        DEVICE=gpu
    fi
fi
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32}
max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}
actor_param_offload=${ACTOR_PARAM_OFFLOAD:-False}
actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-False}

actor_lr=${ACTOR_LR:-3e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

lora_rank=${LORA_RANK:-64}
lora_alpha=${LORA_ALPHA:-32}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_n=${ROLLOUT_N:-5}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

project_name=${PROJECT_NAME:-verl_grpo_lora_gsm8k}
experiment_name=${EXPERIMENT_NAME:-qwen3_8b_lora_vllm_fsdp}
# ---- end user-adjustable ----

EXTRA=()

if [ "${DEVICE}" = intel ]; then
    # 24 GB cards: offload the actor between phases so vLLM can take its share at startup, and cap
    # the per-GPU token budgets (fp32 logits of a 152k vocab are ~0.6 MB per token).
    actor_param_offload=${ACTOR_PARAM_OFFLOAD:-True}
    actor_optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-True}
    ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-2048}
    EXTRA+=(
        trainer.device=intel
        # Ray detects Intel GPUs only through dpctl, which the image does not ship
        +ray_kwargs.ray_init.num_gpus=${NGPUS_PER_NODE}
        # the LoRA base model defaults to fp32 (32 GB for 8B); bf16 halves every copy
        actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
        # torch.compile with FSDP collectives killed a rank in oneCCL on multi-card XPU
        actor_rollout_ref.actor.use_torch_compile=False
        # otherwise vLLM sizes its KV cache for Qwen3's 40960-token native context
        actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length))
    )
fi

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files=$HOME/data/gsm8k/train.parquet
    data.val_files=$HOME/data/gsm8k/test.parquet
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.lora_rank=${lora_rank}
    actor_rollout_ref.model.lora_alpha=${lora_alpha}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_param_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_optimizer_offload}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.load_format=safetensors
    actor_rollout_ref.rollout.layered_summon=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

########################### launch ###########################
# uv (set VERL_USE_UV=0 for system python): on GPU, the driver and every Ray worker
# (runtime_env.py_executable) run through `uv run` on the vllm × fsdp extras of the committed uv.lock;
# NPU falls back to ambient python. Run from the verl repo root.
LAUNCH=(python3)
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)
if [ "${VERL_USE_UV:-1}" != 0 ] && [ "${DEVICE:-gpu}" = gpu ]; then
    LAUNCH=(uv run --frozen --all-packages --extra vllm --extra fsdp python3)
    RAY=(ray_kwargs.ray_init.runtime_env.py_executable="uv -v run --frozen --all-packages --extra vllm --extra fsdp")
fi
"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "${RAY[@]}" \
    "$@"
