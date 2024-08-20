MODEL=meta-llama/Llama-2-7b-hf
# MODEL=meta-llama/Meta-Llama-3-8B
# MODEL=mistralai/Mistral-7B-v0.3
DATASET_NAME=wentingzhao/math-textbooks

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name ${DATASET_NAME} \
  --output_dir {output dir} \
  --do_eval --eval_subset validation \
  # --dstore_dir {path of datastore} \
  # --knn \
  # --knn_temp 5.0 --k 2048 --lmbda 0.1 \
  # --dstore_size 609687689 \
