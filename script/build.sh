MODEL=meta-llama/Llama-2-7b-hf
# MODEL=meta-llama/Meta-Llama-3-8B
# MODEL=mistralai/Mistral-7B-v0.3
DATASET_NAME=wentingzhao/knn-prompt-datastore
python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name  ${DATASET_NAME} \
  --do_eval --eval_subset train \
  --output_dir {path of datastore} \
  --dstore_dir {path of datastore} \
  --dstore_size {dstore size} \
  --build_index

  