#!/bin/bash

MODEL=$1
TASK=$2

if [[ $TASK == "obqa" ]]; then
  python -u eval_pmi.py \
    --model_name_or_path ${MODEL} \
    --dataset_name openbookqa --dataset_config_name additional \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/obqa/base/${MODEL} \
    --do_eval \
    --eval_subset test \

elif [[ $TASK == "arc_easy" ]]; then
  python -u eval_pmi.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name allenai/ai2_arc --dataset_config_name ARC-Easy \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/arc/Easy/base/${MODEL} \
    --do_eval \
    --eval_subset test \

elif [[ $TASK == "arc_challenge" ]]; then
  python -u eval_pmi.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name allenai/ai2_arc --dataset_config_name ARC-Challenge \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/arc/Challenge/base/${MODEL} \
    --do_eval \
    --eval_subset test \

elif [[ $TASK == "mmlu" ]]; then
  python -u eval_pmi.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name cais/mmlu --dataset_config_name all \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/mmlu/base/${MODEL} \
    --do_eval \
    --eval_subset test \

elif [[ $TASK == "hellaswag" ]]; then
  python -u eval_pmi.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name  Rowan/hellaswag \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/hellaswag/base/${MODEL} \
    --do_eval \
    --eval_subset validation \

elif [[ $TASK == "drop" ]]; then
  python -u eval_drop.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name drop \
    --per_device_eval_batch_size=1 \
    --output_dir output/drop/base/${MODEL} \
    --do_eval \
    --eval_subset validation \

elif [[ $TASK == "nq" ]]; then
  python -u eval_qa.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name nq_open \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/nq/base/${MODEL} \
    --do_eval \
    --eval_subset validation \

elif [[ $TASK == "hotpotqa" ]]; then
  python -u eval_qa.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name hotpot_qa --dataset_config_name distractor \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/hotpot/base/${MODEL} \
    --do_eval \
    --eval_subset validation \

elif [[ $TASK == "gsm8k" ]]; then
  python -u eval_gsm8k.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name gsm8k --dataset_config_name main \
    --per_device_eval_batch_size=1 \
    --output_dir output/gsm8k/base/${MODEL} \
    --batch_size 16 \
    --do_eval \
    --eval_subset test \

elif [[ $TASK == "bbh" ]]; then
  python -u eval_bbh.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name lukaemon/bbh  \
    --per_device_eval_batch_size=1 \
    --output_dir output/bbh/base/${MODEL} \
    --do_eval \
    --eval_subset test \

elif [[ $TASK == "winogrande" ]]; then
  python -u eval_winogrande.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name winogrande --dataset_config_name winogrande_xl \
    --per_device_eval_batch_size=1 \
    --output_dir output/winogrande/xl/base/${MODEL} \
    --do_eval \
    --eval_subset validation \

elif [[ $TASK == "sst2" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/sst2/test.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/sst2/base/${MODEL} \
    --do_eval \
    --eval_subset validation \

elif [[ $TASK == "rt" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/rotten_tomatoes/dev.jsonl \
    --test_file data/rotten_tomatoes/test.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/rt/base/${MODEL} \
    --do_eval \
    --eval_subset test \

elif [[ $TASK == "rte" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/rte/val.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/rte/base/${MODEL} \
    --do_eval \
    --eval_subset validation \

elif [[ $TASK == "agn" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/agn/test.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/agn/base/${MODEL} \
    --do_eval \
    --eval_subset validation \

elif [[ $TASK == "cb" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/cb/dev.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/cb/base/${MODEL} \
    --do_eval \
    --eval_subset validation \

elif [[ $TASK == "cr" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/cr/test.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/cr/base/${MODEL} \
    --do_eval \
    --eval_subset validation \

elif [[ $TASK == "hyp" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/hyp/test.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/hyp/base/${MODEL} \
    --do_eval \
    --eval_subset validation \

elif [[ $TASK == "mr" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --validation_file data/mr/test.jsonl \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/mr/base/${MODEL} \
    --do_eval \
    --eval_subset validation \

elif [[ $TASK == "yahoo" ]]; then
  python -u eval_fuzzy.py  \
    --model_name_or_path ${MODEL} \
    --dataset_name yahoo_answers_topics \
    --ignore_pad_token_for_loss \
    --per_device_eval_batch_size=1 \
    --output_dir output/yahoo/base/${MODEL} \
    --do_eval \
    --eval_subset test \
    --max_eval_samples 3000
fi
