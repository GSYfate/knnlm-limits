#!/bin/bash

# 定义参数变量
# MODEL=mistralai/Mistral-7B-v0.3
# MODEL=meta-llama/Meta-Llama-3-8B
# MODEL=meta-llama/Llama-2-7b-hf
# KNN_TEMP=3.0
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
# export CUDA_VISIBLE_DEVICES=0

# python -u eval_pmi.py \
#   --model_name_or_path ${MODEL} \
#   --dataset_name openbookqa --dataset_config_name additional \
#   --ignore_pad_token_for_loss \
#   --per_device_eval_batch_size=1 \
#   --output_dir output/obqa_pmi/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
#   --do_eval \
#   --eval_subset test \
#   --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
#   --knn \
#   --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
#   --dstore_size ${DSTORE_SIZE}

# python -u eval_pmi.py \
#   --model_name_or_path ${MODEL} \
#   --dataset_name allenai/ai2_arc --dataset_config_name ARC-Challenge \
#   --ignore_pad_token_for_loss \
#   --per_device_eval_batch_size=1 \
#   --output_dir arc/Challenge/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
#   --do_eval \
#   --eval_subset test \
#   --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
#   --knn \
#   --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
#   --dstore_size ${DSTORE_SIZE}

# python -u eval_pmi.py \
#   --model_name_or_path ${MODEL} \
#   --dataset_name allenai/ai2_arc --dataset_config_name ARC-Easy \
#   --ignore_pad_token_for_loss \
#   --per_device_eval_batch_size=1 \
#   --output_dir arc/Easy/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
#   --do_eval \
#   --eval_subset test \
#   --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
#   --knn \
#   --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
#   --dstore_size ${DSTORE_SIZE}

# # 第四个 eval_pmi.py 调用
# python -u eval_pmi.py \
#   --model_name_or_path ${MODEL} \
#   --dataset_name cais/mmlu --dataset_config_name all \
#   --ignore_pad_token_for_loss \
#   --per_device_eval_batch_size=1 \
#   --output_dir output/mmlu/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
#   --max_target_length 512 \
#   --do_eval \
#   --eval_subset test \
#   --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
#   --knn \
#   --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
#   --dstore_size ${DSTORE_SIZE}

# 第五个 eval_pmi.py 调用

python -u eval_pmi.py \
  --model_name_or_path ${MODEL} \
  --dataset_name Rowan/hellaswag \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/hella/val/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}
