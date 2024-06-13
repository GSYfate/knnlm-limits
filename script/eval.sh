
# wentingzhao/math-textbooks
# wentingzhao/redpajama-test
# --dataset_name wentingzhao/math-textbooks \
# MODEL=meta-llama/Meta-Llama-3-8B
# MODEL=mistralai/Mistral-7B-v0.3
# MODEL=meta-llama/Llama-2-7b-hf
# 187174908
# 201201660
# 201842684
# 609687689
# export CUDA_VISIBLE_DEVICES=0
# MODEL=mistralai/Mistral-7B-v0.3
# DATASET_NAME=wikitext
# DATASET_CONFIG_NAME=wikitext-103-raw-v1
# DSTORE_DIR=/share/rush/datastore/math/${MODEL}
# DSTORE_SIZE=201201660
MODEL=meta-llama/Llama-2-7b-hf
DATASET_NAME=wikitext
DATASET_CONFIG_NAME=wikitext-103-raw-v1
DSTORE_DIR=/share/rush/datastore/redpajama/${MODEL}
DSTORE_SIZE=1377593437

lambdas=(0.1 0.15 0.2)
ks=(2048)
temps=(1 3 5)

# Loop through all combinations
for lmbda in "${lambdas[@]}"; do
  for k in "${ks[@]}"; do
    for temp in "${temps[@]}"; do
      OUTPUT_DIR=output/wiki/base/${MODEL}/lambda_${lmbda}_k_${k}_temp_${temp}
      python -u run_clm.py \
        --model_name_or_path ${MODEL} \
        --dataset_name ${DATASET_NAME} --dataset_config_name ${DATASET_CONFIG_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --do_eval --eval_subset validation \
        --dstore_dir ${DSTORE_DIR} \
        --knn \
        --knn_temp ${temp} --k ${k} --lmbda ${lmbda} \
        --dstore_size ${DSTORE_SIZE}
    done
  done
done

  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 3.0 --k 2048 --lmbda 0.1 \
  # --dstore_size 135989494 \
  # --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  # --knn \
  # --knn_temp 5.0 --k 2048 --lmbda 0.2 \
  # --dstore_size 120966390 \



