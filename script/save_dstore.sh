
MODEL=$1
DS=$2
DS_PATH=$3

if [[  $DS == "wiki" ]]; then
  DATASET_NAME=wentingzhao/knn-prompt-datastore
elif [[  $DS == "math" ]]; then
  DATASET_NAME=wentingzhao/math-textbooks
fi

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name  ${DATASET_NAME} \
  --do_eval --eval_subset train \
  --output_dir ${DS_PATH} \
  --dstore_dir ${DS_PATH} \
  --save_knnlm_dstore
