# MODEL=meta-llama/Llama-2-7b-hf
MODEL=meta-llama/Meta-Llama-3-8B
# MODEL=mistralai/Mistral-7B-v0.3
DATASET_NAME=wikitext
DATASET_CONFIG_NAME=wikitext-103-raw-v1
DS=wiki
python -u run_clm_wiki.py \
  --model_name_or_path ${MODEL} \
  --dataset_name ${DATASET_NAME} --dataset_config_name ${DATASET_CONFIG_NAME} \
  --output_dir output/wiki103/base/${MODEL} \
  --do_eval --eval_subset validation \
  # --dstore_dir {path of datastore} \
  # --knn \
  # --knn_temp 5.0 --k 2048 --lmbda 0.2 \
  # --dstore_size 609687689  \
  