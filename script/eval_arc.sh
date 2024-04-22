# MODEL=mistralai/Mistral-7B-v0.1
MODEL=meta-llama/Llama-2-7b-hf

python -u eval_pmi.py  \
  --model_name_or_path ${MODEL} \
  --dataset_name allenai/ai2_arc --dataset_config_name ARC-Challenge \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir arc/Challenge/math/${MODEL} \
  --do_eval \
  --eval_subset test \
  --dstore_dir /share/rush/datastore/math/${MODEL} \
  --knn \
  --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  --dstore_size 201842684 \
  # --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 1377593437 \