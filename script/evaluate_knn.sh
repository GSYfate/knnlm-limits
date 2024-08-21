#!/bin/bash

MODEL=$1
TASK=$2
DS=$3
DS_PATH=$4

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

if [[ $TASK == "obqa" ]]; then
  python -u eval_pmi.py \
    --model_name_or_path ${MODEL} \
    --dataset_name openbookqa --dataset_config_name additional \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/obqa/${DS}/${MODEL} \
    --do_eval \
    --eval_subset test \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "arc_easy" ]]; then
  python -u eval_pmi.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name allenai/ai2_arc --dataset_config_name ARC-Easy \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/arc/Easy/${DS}/${MODEL} \
    --do_eval \
    --eval_subset test \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "arc_challenge" ]]; then
  python -u eval_pmi.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name allenai/ai2_arc --dataset_config_name ARC-Challenge \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/arc/Challenge/${DS}/${MODEL} \
    --do_eval \
    --eval_subset test \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "mmlu" ]]; then
  python -u eval_pmi.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name cais/mmlu --dataset_config_name all \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/mmlu/${DS}/${MODEL} \
    --do_eval \
    --eval_subset test \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "hellaswag" ]]; then
  python -u eval_pmi.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name  Rowan/hellaswag \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/hellaswag/${DS}/${MODEL} \
    --do_eval \
    --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "drop" ]]; then
  python -u eval_drop.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name drop \
    --per_device_eval_batch_size=1 \
    --output_dir output/drop/${DS}/${MODEL} \
    --do_eval \
    --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "nq" ]]; then
  python -u eval_qa.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name nq_open \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/nq/${DS}/${MODEL} \
    --do_eval \
    --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "hotpotqa" ]]; then
  python -u eval_qa.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/hotpot/${DS}/${MODEL} \
    --do_eval \
    --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "gsm8k" ]]; then
  python -u eval_gsm8k.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name gsm8k --dataset_config_name main \
    --per_device_eval_batch_size=1 \
    --output_dir output/gsm8k/${DS}/${MODEL} \
    --batch_size 16 \
    --do_eval \
    --eval_subset test \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "bbh" ]]; then
  python -u eval_bbh.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name lukaemon/bbh  \
    --per_device_eval_batch_size=1 \
    --output_dir output/bbh/${DS}/${MODEL} \
    --do_eval \
    --eval_subset test \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "winogrande" ]]; then
  python -u eval_winogrande.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name winogrande --dataset_config_name winogrande_xl \
    --per_device_eval_batch_size=1 \
    --output_dir output/winogrande/xl/${DS}/${MODEL} \
    --do_eval \
    --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "sst2" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/sst2/test.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/sst2/${DS}/${MODEL} \
    --do_eval \
    --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "rt" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/rotten_tomatoes/dev.jsonl \
    --test_file data/rotten_tomatoes/test.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/rt/${DS}/${MODEL} \
    --do_eval \
    --eval_subset test \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "rte" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/rte/val.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/rte/${DS}/${MODEL} \
    --do_eval \
    --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "agn" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/agn/test.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/agn/${DS}/${MODEL} \
    --do_eval \
    --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "cb" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/cb/dev.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/cb/${DS}/${MODEL} \
    --do_eval \
    --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "cr" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/cr/test.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/cr/${DS}/${MODEL} \
    --do_eval \
    --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "hyp" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/hyp/test.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/hyp/${DS}/${MODEL} \
    --do_eval \
    --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "mr" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/mr/test.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/mr/${DS}/${MODEL} \
    --do_eval \
    --eval_subset validation \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE

elif [[ $TASK == "yahoo" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name yahoo_answers_topics \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/yahoo/${DS}/${MODEL} \
    --do_eval \
    --eval_subset test \
    --max_eval_samples 3000 \
    --dstore_dir ${DS_PATH} \
    --knn \
    --knn_temp $KNN_TEMP --k $K --lmbda $LAMBDA \
    --dstore_size $DSTORE_SIZE
    
fi
