# MODEL=meta-llama/Llama-2-7b-hf
MODEL=meta-llama/Meta-Llama-3-8B
python -u eval_fuzzy.py  \
  --model_name_or_path ${MODEL} \
  --validation_file data/cr/test.jsonl \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/cr/wiki/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --knn \
  --knn_temp 5.0 --k 1600 --lmbda 0.3 \
  --dstore_size 120966390 \
  # --dstore_dir /share/rush/datastore/redpajama/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 1600 --lmbda 0.3 \
  # --dstore_size 1377593437 \
  # --dstore_dir /share/rush/datastore/math/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 1600 --lmbda 0.3 \
  # --dstore_size 201842684 \


  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 1600 --lmbda 0.3 \
  # --dstore_size 140210422 \

  # --max_eval_samples 10 \








