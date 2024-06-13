MODEL=meta-llama/Llama-2-7b-hf
DATASET_NAME=wikitext
DATASET_CONFIG_NAME=wikitext-103-raw-v1
DSTORE_DIR=/share/rush/datastore/knn-prompt/${MODEL}
DSTORE_SIZE=609687689


OUTPUT_DIR=output/wiki/base/${MODEL}
python -u run_clm.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} --dataset_config_name ${DATASET_CONFIG_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --do_eval --eval_subset validation \
    # --dstore_dir ${DSTORE_DIR} \
    # --knn \
    # --knn_temp ${temp} --k ${k} --lmbda ${lmbda} \
    # --dstore_size ${DSTORE_SIZE}

