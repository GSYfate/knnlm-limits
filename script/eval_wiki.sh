MODEL=$1
DS=$2
DS_PATH=$3
DATASET_NAME=wikitext
DATASET_CONFIG_NAME=wikitext-103-raw-v1

if [[ $MODEL == "meta-llama/Llama-2-7b-hf" && $DS == "wiki" ]]; then
  KNN_TEMP=5
  K=2048
  LAMBDA=0.2
  DSTORE_SIZE=609687689
elif [[ $MODEL == "meta-llama/Llama-2-7b-hf" && $DS == "math" ]]; then
  KNN_TEMP=3
  K=1600
  LAMBDA=0.2
  DSTORE_SIZE=201842684
elif [[ $MODEL == "meta-llama/Meta-Llama-3-8B" && $DS == "wiki" ]]; then
  KNN_TEMP=5
  K=2048
  LAMBDA=0.1
  DSTORE_SIZE=513504393
elif [[ $MODEL == "meta-llama/Meta-Llama-3-8B" && $DS == "math" ]]; then
  KNN_TEMP=3
  K=2048
  LAMBDA=0.1
  DSTORE_SIZE=187174908
elif [[ $MODEL == "mistralai/Mistral-7B-v0.3" && $DS == "wiki" ]]; then
  KNN_TEMP=10
  K=2048
  LAMBDA=0.1
  DSTORE_SIZE=587870345
elif [[ $MODEL == "mistralai/Mistral-7B-v0.3" && $DS == "math" ]]; then
  KNN_TEMP=10
  K=2048
  LAMBDA=0.1
  DSTORE_SIZE=201201660
fi


if [[ $DS == "base" ]]; then
  python -u run_clm_wiki.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} --dataset_config_name ${DATASET_CONFIG_NAME} \
    --output_dir output/wiki103/${DS} \
    --do_eval --eval_subset validation
else
  python -u run_clm_wiki.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} --dataset_config_name ${DATASET_CONFIG_NAME} \
    --output_dir output/wiki103/${DS} \
    --do_eval --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE
fi

  
  