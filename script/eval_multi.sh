MODEL=meta-llama/Llama-2-7b-hf

python -u run_multihop.py  \
  --model_name_or_path ${MODEL} \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir hotpot_multi_knn_1_1000/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir checkpoints/${MODEL} \
  --knn \
  --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  --dstore_size 140210422 \