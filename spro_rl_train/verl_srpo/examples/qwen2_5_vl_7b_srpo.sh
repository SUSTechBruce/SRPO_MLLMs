#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH='Qwen2.5-VL-7B-SRPO_sft'  # replace it with your local file path
# hiyouga/geometry3k@train 
# /home/tiger/.cache/huggingface/datasets/hiyouga___geometry3k/default/0.0.0/fd21e533e1e50d0662a2bf7b223e60511bd5f8b7/
# /mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO_verl/data_and_models/wwttt/45K
# wwttt/45K@train


# replace 'hiyouga/geometry3k@train' with 'srpo_rl data' after transformation of data format 
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/mnt/bn/seed-aws-va/zhongweiwan/SR_GRPO_verl/data_and_models/wwttt/45K  \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo_normal_reward_45k_test \
    trainer.n_gpus_per_node=8

# bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
# 