MODEL=meta-llama/Llama-2-7b-hf

python -u eval_drop.py  \
  --model_name_or_path ${MODEL} \
  --dataset_name drop \
  --per_device_eval_batch_size=1 \
  --output_dir output/drop/val/test/${MODEL} \
  --do_eval \
  --eval_subset validation \
  # --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 609687689 \
  # --dstore_dir /share/rush/datastore/math/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 201842684 \


