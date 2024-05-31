# MODEL=mistralai/Mistral-7B-v0.1
# MODEL=meta-llama/Llama-2-7b-hf
# MODEL=meta-llama/Meta-Llama-3-8B
MODEL=mistralai/Mistral-7B-v0.3
python -u eval_winogrande.py  \
  --model_name_or_path ${MODEL} \
  --dataset_name winogrande --dataset_config_name winogrande_xl \
  --per_device_eval_batch_size=1 \
  --output_dir output/winogrande/xl/wiki103/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --knn \
  --knn_temp 3.0 --k 2048 --lmbda 0.1 \
  --dstore_size 135989494 \
  # --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 609687689 \
  # --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 1377593437 \