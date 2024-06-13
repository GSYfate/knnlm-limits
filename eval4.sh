# MODEL=meta-llama/Meta-Llama-3-8B
# MODEL=mistralai/Mistral-7B-v0.3
MODEL=meta-llama/Llama-2-7b-hf
KNN_TEMP=3.0
K=2048
LMBDA=0.1
DSTORE_SIZE=609687689
DS=knn-prompt
# export CUDA_VISIBLE_DEVICES=0
# MODEL=meta-llama/Meta-Llama-3-8B
# KNN_TEMP=5.0
# K=2048
# LMBDA=0.1
# DSTORE_SIZE=187174908
# DS=math

# eval_qa.py 调用 (hotpot_qa)
python -u eval_qa.py \
  --model_name_or_path ${MODEL} \
  --dataset_name hotpot_qa --dataset_config_name distractor \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/hotpot/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}

# eval_qa.py 调用 (nq_open)
python -u eval_qa.py \
  --model_name_or_path ${MODEL} \
  --dataset_name nq_open \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/nq/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}

# eval_winogrande.py 调用
python -u eval_winogrande.py \
  --model_name_or_path ${MODEL} \
  --dataset_name winogrande --dataset_config_name winogrande_xl \
  --per_device_eval_batch_size=1 \
  --output_dir output/winogrande/xl/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}

# eval_bbh.py 调用
python -u eval_bbh.py \
  --model_name_or_path ${MODEL} \
  --dataset_name lukaemon/bbh \
  --per_device_eval_batch_size=1 \
  --output_dir output/bbh/test/${DS}_${KNN_TEMP}_${K}_${LMBDA}/${MODEL} \
  --do_eval \
  --eval_subset test \
  --dstore_dir /share/rush/datastore/${DS}/${MODEL} \
  --knn \
  --knn_temp ${KNN_TEMP} --k ${K} --lmbda ${LMBDA} \
  --dstore_size ${DSTORE_SIZE}