#!/bin/sh
# MODEL=meta-llama/Llama-2-7b-hf
MODEL=mistralai/Mistral-7B-v0.3
python -u eval_gsm8k.py  \
  --model_name_or_path ${MODEL} \
  --dataset_name gsm8k --dataset_config_name main \
  --per_device_eval_batch_size=1 \
  --output_dir output/gsm8k/test/base/${MODEL} \
  --batch_size 16 \
  --do_eval \
  --eval_subset test \
  # --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  # --knn \
  # --knn_temp 5.0 --k 2048 --lmbda 0.1 \
  # --dstore_size 609687689 \
  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 2048 --lmbda 0.2 \
  # --dstore_size 135989494 \
  # --dstore_dir /share/rush/datastore/math/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 2048 --lmbda 0.2 \
  # --dstore_size 201201660

  # --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 609687689 \
  # --dstore_dir /share/rush/datastore/math/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 201842684 \

  # --dstore_dir /share/rush/datastore/redpajama/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 1377593437 \



 