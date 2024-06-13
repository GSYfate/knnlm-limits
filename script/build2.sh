MODEL=mistralai/Mistral-7B-v0.3
python -u run_clm3.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wentingzhao/knn-prompt-datastore \
  --per_device_eval_batch_size 8 \
  --output_dir /share/rush/datastore/knn-prompt/${MODEL} \
  --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  --dstore_size 576872732 \
  --build_index