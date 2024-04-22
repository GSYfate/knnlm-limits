  MODEL=meta-llama/Llama-2-7b-hf

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name {name of the dataset} \
  --output_dir output/${MODEL} \
  --dstore_dir {path of your datastore}\
  --build_index