#!/bin/sh
MODEL=meta-llama/Llama-2-7b-hf
# MODEL=meta-llama/Meta-Llama-3-8B
# MODEL=mistralai/Mistral-7B-v0.3
python -u eval_gsm8k.py  \
  --model_name_or_path ${MODEL} \
  --dataset_name gsm8k --dataset_config_name main \
  --per_device_eval_batch_size=1 \
  --output_dir output/gsm8k/base/${MODEL} \
  --batch_size 16 \
  --do_eval \
  --eval_subset test \
  # --dstore_dir {path of datastore} \
  # --knn \
  # --knn_temp 3.0 --k 1600 --lmbda 0.2 \
  # --dstore_size 201842684 \
 