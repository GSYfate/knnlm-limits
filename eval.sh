MODEL=mistralai/Mistral-7B-v0.3
python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir output/wiki/math/${MODEL} \
  --do_eval --eval_subset validation \
  --dstore_dir /share/rush/datastore/math/${MODEL} \
  --knn \
  --knn_temp 3.0 --k 2048 --lmbda 0.2 \
  --dstore_size 201201660

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wentingzhao/redpajama-test \
  --output_dir output/redpajama/math/${MODEL} \
  --do_eval --eval_subset train \
  --dstore_dir /share/rush/datastore/math/${MODEL} \
  --knn \
  --knn_temp 3.0 --k 2048 --lmbda 0.2 \
  --dstore_size 201201660

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wentingzhao/math-textbooks \
  --output_dir output/math/math/${MODEL} \
  --do_eval --eval_subset validation \ 
  --dstore_dir /share/rush/datastore/math/${MODEL} \
  --knn \
  --knn_temp 3.0 --k 2048 --lmbda 0.2 \
  --dstore_size 201201660