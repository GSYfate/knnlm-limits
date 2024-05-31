# MODEL=mistralai/Mistral-7B-v0.1
# MODEL=meta-llama/Llama-2-7b-hf
# MODEL=meta-llama/Meta-Llama-3-8B
MODEL=mistralai/Mistral-7B-v0.3
python -u eval_pmi.py  \
  --model_name_or_path ${MODEL} \
  --dataset_name cais/mmlu --dataset_config_name all \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/mmlu/wiki103/${MODEL} \
  --max_target_length 512 \
  --do_eval \
  --eval_subset test \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --knn \
  --knn_temp 3.0 --k 2048 --lmbda 0.2 \
  --dstore_size 135989494 \
  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 5.0 --k 2048 --lmbda 0.2 \
  # --dstore_size 120966390 \
  #--dstore_dir /share/rush/datastore/math/${MODEL} \
  #--knn \
  #--knn_temp 1.0 --k 1024 --lmbda 0.2 \
  #--dstore_size 201842684 \
  #--dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  #--knn \
  #--knn_temp 1.0 --k 1024 --lmbda 0.2 \
  #--dstore_size 140210422 \
  # --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 1377593437 \

 

