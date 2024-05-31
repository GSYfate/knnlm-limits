# MODEL=meta-llama/Llama-2-7b-hf
# wentingzhao/math-textbooks
# wentingzhao/redpajama-test
export CUDA_VISIBLE_DEVICES=0
# MODEL=meta-llama/Meta-Llama-3-8B
  # --dataset_name wentingzhao/math-textbooks \
MODEL=mistralai/Mistral-7B-v0.3
python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir output/math/base/${MODEL} \
  --do_eval --eval_subset validation \
  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 2048 --lmbda 0.1 \
  # --dstore_size 135989494 \
  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 5.0 --k 2048 --lmbda 0.2 \
  # --dstore_size 120966390 \



