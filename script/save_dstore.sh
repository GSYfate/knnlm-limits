MODEL=meta-llama/Llama-2-7b-hf

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name {name of dataset} \
  --eval_subset train \
  --per_device_eval_batch_size 32 \
  --output_dir /share/rush/datastore/math/${MODEL} \
  --dstore_dir /share/rush/datastore/math/${MODEL} \
  --build_index
  #--save_knnlm_dstore \
