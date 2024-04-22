MODEL=meta-llama/Llama-2-7b-hf

python -u run_clm_2.py \
  --model_name_or_path ${MODEL} \
  --output_dir output/dataset/cc_news/${MODEL} \
  --per_device_eval_batch_size=1 \
  --do_eval --eval_subset train \
  --dataset_name stanfordnlp/imdb \
  # --dataset_config_name wikitext-103-raw-v1
  # --dstore_dir /share/rush/datastore/knn-prompt/${MODEL} \
  # --knn \
  # --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  # --dstore_size 609687689 \
# MODEL=meta-llama/Llama-2-7b-hf
# --max_eval_samples 100 \
# stas/c4-en-10k 
# ackmin108/c4-en-validation-mini
  # --dataset_name ptb_text_only \
#   --dataset_name ptb_text_only \
  # --dataset_name NeelNanda/wiki-10k \
# MODEL=meta-llama/Llama-2-7b-hf  
# python -u run_clm.py \
#   --model_name_or_path ${MODEL} \
#   --dataset_name Jackmin108/c4-en-validation-mini \
#   --per_device_eval_batch_size=1 \
#   --output_dir c4_knn/${MODEL} \
#   --do_eval --eval_subset validation \
#   --dstore_dir checkpoints/${MODEL} \
#   --retomaton \
#   --dstore_size 140210422 \
#   --knn_temp 5 --k 1024 --lmbda 0.2 \