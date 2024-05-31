# MODEL=meta-llama/Llama-2-7b-hf
# MODEL=meta-llama/Meta-Llama-3-8B
MODEL=mistralai/Mistral-7B-v0.3
python -u eval_fuzzy.py  \
  --model_name_or_path ${MODEL} \
  --validation_file data/agn/test.jsonl \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/agn/fuzzy/base/${MODEL} \
  --do_eval \
  --eval_subset validation \
  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 1600 --lmbda 0.1 \
  # --dstore_size 135989494 \
  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 1600 --lmbda 0.2 \
  # --dstore_size 135989494 \
  # --dstore_dir /share/rush/datastore/math/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 1600 --lmbda 0.3 \
  # --dstore_size 201842684 \
