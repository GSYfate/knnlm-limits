MODEL=$1
DATASET_NAME=$2
DS=$3
DS_PATH=$4

if [[ $DATASET_NAME == "wikitext" ]]; then
  DATASET_CONFIG_NAME="wikitext-103-raw-v1"
else
  DATASET_CONFIG_NAME=""
fi

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
  CMD="python -u run_clm.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} \
    --output_dir output/${DATASET_NAME}/${DS} \
    --do_eval --eval_subset validation"

  if [[ -n $DATASET_CONFIG_NAME ]]; then
    CMD+=" --dataset_config_name ${DATASET_CONFIG_NAME}"
  fi

  eval $CMD
else
  CMD="python -u run_clm.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} \
    --output_dir output/${DATASET_NAME}/${DS} \
    --do_eval --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE"
    
  if [[ -n $DATASET_CONFIG_NAME ]]; then
    CMD+=" --dataset_config_name ${DATASET_CONFIG_NAME}"
  fi

  eval $CMD
fi