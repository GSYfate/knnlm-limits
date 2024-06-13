MODEL=mistralai/Mistral-7B-v0.3
# MODEL=meta-llama/Meta-Llama-3-8B
python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wentingzhao/knn-prompt-datastore \
  --do_eval --eval_subset train \
  --per_device_eval_batch_size 8 \
  --output_dir /share/rush/datastore/knn-prompt/${MODEL} \
  --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  --save_knnlm_dstore

# python -u run_clm.py \
#   --model_name_or_path ${MODEL} \
#   --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
#   --output_dir /share/rush/datastore/knn-prompt/${MODEL} \
#   --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
#   --dstore_size 576872732 \
#   --build_index