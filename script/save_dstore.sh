MODEL=meta-llama/Llama-2-7b-hf

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name {name of the dataset} \
  --do_eval \
  --eval_subset train \
  --per_device_eval_batch_size 32 \
  --output_dir output/${MODEL} \
  --dstore_dir {path of your datastore}\
  --save_knnlm_dstore \