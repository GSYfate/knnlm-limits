#!/bin/bash

MODEL=$1
DS=$2
DS_PATH=$3

python -u eval_pmi.py \
  --model_name_or_path ${MODEL} \
  --dataset_name openbookqa --dataset_config_name additional \
  --ignore_pad_token_for_loss \
  --per_device_eval_batch_size=1 \
  --output_dir output/obqa/${DS}/${MODEL} \
  --do_eval \
  --eval_subset test \
  $(if [ "${DS}" = "base" ]; then
      echo ""
    else
      case ${MODEL} in
        "meta-llama/Llama-2-7b-hf")
          if [ "${DS}" = "math" ]; then
            echo "--dstore_dir ${DS_PATH} \
                  --knn \
                  --knn_temp 3.0 --k 1600 --lmbda 0.2 \
                  --dstore_size 201842684"
          elif [ "${DS}" = "wiki" ]; then
            echo "--dstore_dir ${DS_PATH} \
                  --knn \
                  --knn_temp 5.0 --k 2048 --lmbda 0.2 \
                  --dstore_size 609687689"
          fi
          ;;
        "meta-llama/Meta-Llama-3-8B")
          if [ "${DS}" = "math" ]; then
            echo "--dstore_dir ${DS_PATH} \
                  --knn \
                  --knn_temp 3.0 --k 2048 --lmbda 0.1 \
                  --dstore_size 187174908"
          elif [ "${DS}" = "wiki" ]; then
            echo "--dstore_dir ${DS_PATH} \
                  --knn \
                  --knn_temp 5 --k 2048 --lmbda 0.1 \
                  --dstore_size 513504393"
          fi
          ;;
        "mistralai/Mistral-7B-v0.3")
          if [ "${DS}" = "math" ]; then
            echo "--dstore_dir ${DS_PATH} \
                  --knn \
                  --knn_temp 10 --k 2048 --lmbda 0.1 \
                  --dstore_size 201201660"
          elif [ "${DS}" = "wiki" ]; then
            echo "--dstore_dir ${DS_PATH} \
                  --knn \
                  --knn_temp 10 --k 2048 --lmbda 0.1 \
                  --dstore_size 587870345"
          fi
          ;;
      esac
    fi)
