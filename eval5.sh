#!/bin/bash

# 定义参数变量

# MODEL=meta-llama/Meta-Llama-3-8B
# # MODEL=mistralai/Mistral-7B-v0.3
# # MODEL=meta-llama/Llama-2-7b-hf
# KNN_TEMP=3.0
# K=2048
# LMBDA=0.1
# DSTORE_SIZE=187174908
# DS=math
# 设置使用 GPU 0 和 1
# export CUDA_VISIBLE_DEVICES=1

MODEL=meta-llama/Llama-2-7b-hf
KNN_TEMP=5.0
K=2048
LMBDA=0.2
DSTORE_SIZE=609687689
DS=knn-prompt

# eval_fuzzy.py 调用 (rte)
python -u eval_fuzzy.py \
  --model_name_or_path ${MODEL} \
  --validation_file data/rte/val.jsonl \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/rte/fuzzy/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}

# eval_fuzzy.py 调用 (rotten_tomatoes)
python -u eval_fuzzy.py \
  --model_name_or_path ${MODEL} \
  --validation_file data/rotten_tomatoes/dev.jsonl \
  --test_file data/rotten_tomatoes/test.jsonl \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/rt/fuzzy/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset test \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}

# eval_fuzzy.py 调用 (cb)
python -u eval_fuzzy.py \
  --model_name_or_path ${MODEL} \
  --validation_file data/cb/dev.jsonl \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/cb/fuzzy/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}

# eval_fuzzy.py 调用 (yahoo_answers_topics)
python -u eval_fuzzy.py \
  --model_name_or_path ${MODEL} \
  --dataset_name yahoo_answers_topics \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/yahoo/fuzzy/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset test \
  --max_eval_samples 3000 \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}

# eval_fuzzy.py 调用 (cr)
python -u eval_fuzzy.py \
  --model_name_or_path ${MODEL} \
  --validation_file data/cr/test.jsonl \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/cr/fuzzy/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}

# eval_fuzzy.py 调用 (agn)
python -u eval_fuzzy.py \
  --model_name_or_path ${MODEL} \
  --validation_file data/agn/test.jsonl \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/agn/fuzzy/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}

# eval_fuzzy.py 调用 (hyp)
python -u eval_fuzzy.py \
  --model_name_or_path ${MODEL} \
  --validation_file data/hyp/test.jsonl \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/hyp/fuzzy/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}

# eval_fuzzy.py 调用 (mr)
python -u eval_fuzzy.py \
  --model_name_or_path ${MODEL} \
  --validation_file data/mr/test.jsonl \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/mr/fuzzy/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}

# eval_fuzzy.py 调用 (sst2)
python -u eval_fuzzy.py \
  --model_name_or_path ${MODEL} \
  --validation_file data/sst2/test.jsonl \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/sst2/fuzzy/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}
