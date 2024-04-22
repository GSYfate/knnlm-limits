MODEL=mistralai/Mistral-7B-v0.1

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir /share/rush/datastore/wiki103/${MODEL} \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --dstore_size 137105654 \
  --build_index