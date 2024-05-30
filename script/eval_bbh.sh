#!/bin/sh
# MODEL=meta-llama/Llama-2-7b-hf
MODEL=meta-llama/Meta-Llama-3-8B
python -u eval_bbh.py  \
  --model_name_or_path ${MODEL} \
  --dataset_name lukaemon/bbh\
  --per_device_eval_batch_size=1 \
  --output_dir output/bbh/test/wiki103/${MODEL} \
  --do_eval \
  --eval_subset test \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --knn \
  --knn_temp 5 --k 2048 --lmbda 0.2 \
  --dstore_size 120966390 \
  # --dstore_dir /share/rush/datastore/math/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 201842684 \
  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 140210422 \

 
  # --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 609687689 \
  # --dstore_dir /share/rush/datastore/redpajama/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 1377593437 \





  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 140210422 \

# --dataset_config_name disambiguation_qa
  # 
