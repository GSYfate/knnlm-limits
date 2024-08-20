MODEL=meta-llama/Llama-2-7b-hf
# MODEL=meta-llama/Meta-Llama-3-8B
# MODEL=mistralai/Mistral-7B-v0.3
DS=wiki
python -u eval_pmi.py  \
  --model_name_or_path ${MODEL} \
  --dataset_name openbookqa --dataset_config_name additional \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/obqa/${DS}/${MODEL} \
  --do_eval \
  --eval_subset test \
  # --dstore_dir {path of datastore} \
  # --knn \
  # --knn_temp 5.0 --k 2048 --lmbda 0.2 \
  # --dstore_size 609687689  \

