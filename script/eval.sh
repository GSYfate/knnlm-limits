MODEL=meta-llama/Llama-2-7b-hf

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name {name of the dataset} \
  --dataset_config_name {Optional} \
  --output_dir output/${MODEL} \
  --do_eval --eval_subset validation