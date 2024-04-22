MODEL=meta-llama/Llama-2-7b-hf

python -u eval_fuzzy.py  \
  --model_name_or_path ${MODEL} \
  --validation_file data/rotten_tomatoes/dev.jsonl \
  --test_file data/rotten_tomatoes/test.jsonl \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/rt/fuzzy/test${MODEL} \
  --do_eval \
  --eval_subset test \
  --dstore_dir /share/rush/datastore/math/${MODEL} \
  --knn \
  --knn_temp 3.0 --k 1600 --lmbda 0.3 \
  --dstore_size 201842684 \
  # --dstore_dir /share/rush/datastore/redpajama/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 1600 --lmbda 0.3 \
  # --dstore_size 1377593437 \
  # --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 1600 --lmbda 0.3 \
  # --dstore_size 609687689 \


  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 1600 --lmbda 0.3 \
  # --dstore_size 140210422 \





  # --max_eval_samples 600 \

