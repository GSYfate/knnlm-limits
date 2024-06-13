#!/bin/bash

# 定义参数变量
# MODEL=meta-llama/Meta-Llama-3-8B
# MODEL=mistralai/Mistral-7B-v0.3
# MODEL=meta-llama/Llama-2-7b-hf
# KNN_TEMP=5.0
# K=2048
# LMBDA=0.1
# DSTORE_SIZE=609687689
# DS=knn-prompt
MODEL=meta-llama/Meta-Llama-3-8B
KNN_TEMP=5.0
K=2048
LMBDA=0.1
DSTORE_SIZE=187174908
DS=math
# 设置使用 GPU 0 和 1
# export CUDA_VISIBLE_DEVICES=1

# eval_gsm8k.py 调用
python -u eval_gsm8k.py \
  --model_name_or_path ${MODEL} \
  --dataset_name gsm8k --dataset_config_name main \
  --per_device_eval_batch_size=1 \
  --output_dir output/gsm8k/test/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --batch_size 16 \
  --do_eval \
  --eval_subset test \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}

# eval_drop.py 调用

# python -u eval_drop.py \
#   --model_name_or_path ${MODEL} \
#   --dataset_name drop \
#   --per_device_eval_batch_size=1 \
#   --output_dir output/drop/val/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
#   --do_eval \
#   --eval_subset validation \
#   --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
#   --knn \
#   --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
#   --dstore_size ${DSTORE_SIZE}
