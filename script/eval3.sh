# wentingzhao/math-textbooks
# wentingzhao/redpajama-test
# --dataset_name wentingzhao/math-textbooks \
# MODEL=meta-llama/Meta-Llama-3-8B
# MODEL=mistralai/Mistral-7B-v0.3
# MODEL=meta-llama/Llama-2-7b-hf
# 187174908
# 201201660
# 201842684

MODEL=meta-llama/Llama-2-7b-hf
DATASET_NAME=wentingzhao/math-textbooks
DSTORE_DIR=/share/rush/datastore/knn-prompt/${MODEL}
DSTORE_SIZE=609687689

lambdas=(0.1 0.15 0.2 0.25)
ks=(2048)
temps=(1 3 5 10)

# Loop through all combinations
for lmbda in "${lambdas[@]}"; do
  for k in "${ks[@]}"; do
    for temp in "${temps[@]}"; do
      OUTPUT_DIR=output/math/knn-prompt/${MODEL}/lambda_${lmbda}_k_${k}_temp_${temp}
      python -u run_clm.py \
        --model_name_or_path ${MODEL} \
        --dataset_name ${DATASET_NAME} \
        --output_dir ${OUTPUT_DIR} \
        --do_eval --eval_subset validation \
        --dstore_dir ${DSTORE_DIR} \
        --knn \
        --knn_temp ${temp} --k ${k} --lmbda ${lmbda} \
        --dstore_size ${DSTORE_SIZE}
    done
  done
done







