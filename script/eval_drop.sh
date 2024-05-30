MODEL=meta-llama/Meta-Llama-3-8B

python -u eval_drop.py  \
  --model_name_or_path ${MODEL} \
  --dataset_name drop \
  --per_device_eval_batch_size=1 \
  --output_dir output/drop/val/base/${MODEL} \
  --do_eval \
  --eval_subset validation \
  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 5.0 --k 2048 --lmbda 0.2 \
  # --dstore_size 120966390 \
  # --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 609687689 \
  # --dstore_dir /share/rush/datastore/math/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 201842684 \


