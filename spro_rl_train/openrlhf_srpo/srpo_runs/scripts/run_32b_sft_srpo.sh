set -x

export RAY_MASTER_PORT=6388
export RAY_DASHBOARD_PORT=8269
export NCCL_TIMEOUT=7200
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# OUTPUT_DIR='/mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/output_ckpts/qwen_2_5vl_sft_lowmem'
OUTPUT_DIR='/opt/tiger/mariana/zhongwei_tmp_checkpoints/srpo_qwen_2_5vl_sft_lowmem_filter_data'

export REWARD_LOG_PATH="${OUTPUT_DIR}/reward.log"
# export WORKING_DIR=$PWD
# export NODE_RANK=${NODE_RANK:-0}
# export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

if [ "$PET_NODE_RANK" -eq 0 ]; then
    ray start --head  --port=$RAY_MASTER_PORT --dashboard-host=0.0.0.0 --dashboard-port=$RAY_DASHBOARD_PORT --num-gpus 8
else
    sleep 30
    ray start --address="$MASTER_ADDR:$RAY_MASTER_PORT" --num-gpus 8 --block
fi

sleep 30

# /mnt/bn/seed-aws-va/zhongweiwan/SR_SFT/output_dir_qwen_2_5_7B_srsft/checkpoint-1000
# Qwen/Qwen2.5-VL-7B-Instruct
# /opt/tiger/mariana/zhongwei_tmp_checkpoints

if [ "$NODE_RANK" -eq 0 ]; then
  RAY_ADDRESS="http://127.0.0.1:$RAY_DASHBOARD_PORT" ray job submit \
  --working-dir $WORKING_DIR \
  --runtime-env-json '{"excludes": ["/mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/MM-EUREKA/.git"]}' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 8 \
  --remote_rm_url examples/scripts/srpo_reward_func.py \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 8 \
  --vllm_num_engines 8 \
  --vllm_tensor_parallel_size 1 \
  --colocate_all_models \
  --vllm_enable_sleep \
  --vllm_gpu_memory_utilization 0.25 \
  --vllm_sync_backend nccl \
  --pretrain /mnt/bn/seed-aws-va/zhongweiwan/SR_SFT/output_dir_qwen_2_5_32B_srsft/checkpoint-1000 \
  --save_path ${OUTPUT_DIR} \
  --micro_train_batch_size 1 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 1 \
  --rollout_batch_size 128 \
  --temperature 1.0 \
  --n_samples_per_prompt 8 \
  --lambd 1.0 \
  --gamma 1.0 \
  --max_epochs 1 \
  --num_episodes 3 \
  --prompt_max_len 1024 \
  --max_samples 100000 \
  --generate_max_len 2048 \
  --advantage_estimator group_norm \
  --zero_stage 3 \
  --bf16 \
  --actor_learning_rate 1e-6 \
  --init_kl_coef 0.0 \
  --prompt_data /mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO/Dataset_and_models/modified_37Krelease_filtered.jsonl \
  --disable_fast_tokenizer \
  --input_key message \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --save_steps 50 \
  --ckpt_path "${OUTPUT_DIR}/ckpt" \
  --max_ckpt_num 1000000 \
  --save_hf_ckpt \
  --freeze_prefix visual \
  --enable_accuracy_filter \
  --accuracy_lower_bound 0.1 \
  --accuracy_upper_bound 0.9 \
  --kl_estimator "k3" \
  --use_wandb True \
  --wandb_project "SR_GRPO" \
  --wandb_run_name "2_new_srpo_qwen_2_5vl_sft_lowmem_filter_data" \
  --system_prompt "Solve the user's question step by step. First, think about the reasoning process internally and write it inside <think> and </think> tags. Then provide the first answer in LaTeX format, wrapped with $...$, and the final result must use \boxed{}. Wrap this answer inside <answer> and </answer> tags. After that, perform a critical self-reflection on the previous reasoning and answer, writing the reflection inside <reflection> and </reflection> tags. Then, based on the reflection, generate a new reasoning process and a new answer: the new reasoning is again inside <think> and </think>, and the new answer is inside <answer> and </answer>, still using LaTeX $...$ and \boxed{}. Make sure both reasoning steps are clear and detailed. Even if the final answer does not change, the second reasoning must incorporate improvements based on the reflection. Always strictly follow the sequence: <think>...</think> <answer>...</answer> <reflection>...</reflection> <think>...</think> <answer>...</answer>. Example: <think> Since $1+1=2$, so the answer is $2$. </think><answer> The answer is $\boxed{2}$. </answer><reflection> The reasoning is correct but too brief; I could have explained the addition more explicitly. </reflection><think> Adding $1$ and $1$ together results in $2$ because $1$ plus $1$ means taking one and adding another one, leading to $2$. </think><answer> The answer is $\boxed{2}$. </answer>. All reasoning, answer, and reflection steps must be included without omission." \
  --load_checkpoint | tee ${OUTPUT_DIR}/training.log
fi

  # --use_tensorboard "${OUTPUT_DIR}/tensorboard" \

# sh examples/scripts/run_7b_sft_srpo_filter_data.sh