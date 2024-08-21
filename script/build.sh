MODEL=$1
DS=$2
DS_PATH=$3

if [[  $DS == "wiki" ]]; then
  DATASET_NAME=wentingzhao/knn-prompt-datastore
elif [[  $DS == "math" ]]; then
  DATASET_NAME=wentingzhao/math-textbooks
fi
  

if [[ $MODEL == "meta-llama/Llama-2-7b-hf" && $DS == "wiki" ]]; then
  DSTORE_SIZE=609687689
elif [[ $MODEL == "meta-llama/Llama-2-7b-hf" && $DS == "math" ]]; then
  DSTORE_SIZE=201842684
elif [[ $MODEL == "meta-llama/Meta-Llama-3-8B" && $DS == "wiki" ]]; then
  DSTORE_SIZE=513504393
elif [[ $MODEL == "meta-llama/Meta-Llama-3-8B" && $DS == "math" ]]; then
  DSTORE_SIZE=187174908
elif [[ $MODEL == "mistralai/Mistral-7B-v0.3" && $DS == "wiki" ]]; then
  DSTORE_SIZE=587870345
elif [[ $MODEL == "mistralai/Mistral-7B-v0.3" && $DS == "math" ]]; then
  DSTORE_SIZE=201201660
fi


python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name  ${DATASET_NAME} \
  --output_dir ${DS_PATH} \
  --dstore_dir ${DS_PATH} \
  --dstore_size ${DSTORE_SIZE} \
  --build_index

  