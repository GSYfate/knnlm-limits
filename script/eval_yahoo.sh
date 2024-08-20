# MODEL=meta-llama/Llama-2-7b-hf
MODEL=meta-llama/Meta-Llama-3-8B
# MODEL=mistralai/Mistral-7B-v0.3
python -u eval_fuzzy.py  \
  --model_name_or_path ${MODEL} \
  --dataset_name yahoo_answers_topics \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/yahoo/fuzzy/wiki/${MODEL} \
  --do_eval \
  --eval_subset test \
  --max_eval_samples 3000 \
 # --dstore_dir {path of datastore} \
  # --knn \
  # --knn_temp 5.0 --k 2048 --lmbda 0.1 \
  # --dstore_size 609687689 \

  








