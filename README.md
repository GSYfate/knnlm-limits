


Table of Contents
=================
  * [Quickstart](#quickstart)
    * [Step 1: Setup  the Environment](#step-1-setup-environment)
    * [Step 2: Saving a Datastore](#step-2-saving-a-datastore)
    * [Step 3: Building the FAISS index](#step-3-building-the-faiss-index)
    * [Step 4: Evaluating Models](#step-4-evaluating-models)
  * [Evaluation](#evaluating-models-on-downstream-tasks)
  
## Quickstart

### Step 1: Setup Environment

#### Clone this repository:
```bash
git clone https://github.com/GSYfate/knnlm-limits.git
cd knnlm-limits
```

Run:
```bash
conda create --name faiss python=3.9
```

```bash
pip install -r requirements.txt
```

* The project also relies on the faiss library. To install the GPU version of faiss, use the following command.
```
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
```


### Step 2: Saving a Datastore

To save a datastore(for example, wikitext), run:
```bash
MODEL=meta-llama/Llama-2-7b-hf

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name ${DATASET_NAME}\
  --do_eval --eval_subset train \
  --output_dir output/${MODEL} \
  --dstore_dir { path of your datastore } \
  --save_knnlm_dstore
```
or run

```
  bash script/save_dstore.sh
```

### Step 3: Building the FAISS index


To build the FAISS index yourself:
```bash
MODEL=meta-llama/Llama-2-7b-hf

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --dstore_dir checkpoints/${MODEL} \
  --build_index
```

or run:

```
  bash script/build.sh
```

You can also directly access our built datastore through the link below.

https://huggingface.co/datasets/Reset23/math

https://huggingface.co/datasets/Reset23/wiki

### Step 4: Evaluating Models

To evaluate kNN-LM on the validation set:

```bash
MODEL=meta-llama/Llama-2-7b-hf

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name ${DATASET_NAME} \
  --output_dir couput/${MODEL} \
  --do_eval --eval_subset validation \
```
or run:

```
  bash script/eval.sh
```


For the wikitext-103 dataset, we performed some preprocessing. The evaluation code runs as follows:
```bash
MODEL=meta-llama/Llama-2-7b-hf

python -u run_clm_wiki.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir output/${MODEL} \
  --do_eval --eval_subset validation \
```
or run:

```
  bash script/eval_wiki.sh
```

To use kNN-LM, use the `--knn` flag
Additional possible test-time tunable hyperparameters are `--lmbda` (the interpolation factor between the datastore and the base LM), `--k` (the number of retrieved nearest neighbors) and `--knn_temp` (the softmax temperature when converting the nearest-neighbor distances into a probability distribution).

E.g.

```
  --dstore_dir {path of datastore} \
  --knn \
  --knn_temp 5.0 --k 2048 --lmbda 0.2 \
  --dstore_size 609687689 \
```


## Evaluating Models on Downstream Tasks

### Reasoning Tasks
 
#### OpenbookQA

Datasets: https://huggingface.co/datasets/openbookqa

Evaluation command: `bash script/eval_obqa.sh`

Eval Program: `eval_pmi.py`

Metrics: dcpmi



#### MMLU

Datasets: https://huggingface.co/datasets/cais/mmlu

Evaluation command: `bash script/eval_mmlu.sh`

Eval Program: `eval_pmi.py`

Metrics: dcpmi

#### Arc

Datasets: https://huggingface.co/datasets/allenai/ai2_arc

Evaluation command: `bash script/eval_arc.sh`

Eval Program: `eval_pmi.py`

Metrics: dcpmi


Hyperparameter settings:
ARC-Challenge: 
```
  --dataset_name allenai/ai2_arc --dataset_config_name ARC-Challenge\
```

ARC-Easy: 
```
  --dataset_name allenai/ai2_arc --dataset_config_name ARC-Easy \
```

#### HellaSwg

Datasets: https://huggingface.co/datasets/Rowan/hellaswag

Evaluation command: `bash script/eval_hellaswag.sh`

Eval Program: `evl_pmi.py`

Metrics: dcpmi


#### Drop

Datasets: https://huggingface.co/datasets/drop

Evaluation command: `bash script/eval_drop.sh`

Eval Program: `eval_drop.py`

Metrics: F1 score


#### NQ

Datasets: https://huggingface.co/datasets/nq_open

Evaluation command: `bash script/eval_nq.sh`

Eval Program: `eval_qa.py`

Metrics: F1 score


#### HotpotQA

Datasets: https://huggingface.co/datasets/hotpot_qa

Evaluation command: `bash script/eval_hotpot.sh`

Eval Program: `eval_qa.py`

Metrics: F1 score


#### GSM8k

Datasets: https://huggingface.co/datasets/gsm8k

Evaluation command: `bash script/eval_gsm.sh`

Eval Program: `eval_gsm8k.py`

Metrics: Accuracy

#### BBH

Datasets: https://huggingface.co/datasets/lukaemon/bbh

Evaluation command: `bash script/eval_bbh.sh`

Eval Program: `eval_bbh.py`

Metrics: Accuracy

#### Winogrande

Datasets: https://huggingface.co/datasets/allenai/winogrande

Evaluation command: `bash script/eval_winogrande.sh`

Eval Program: `eval_winogrande.py`

Metrics: Accuracy


### Memory-intensive Tasks 

Datasets: The corresponding datasets are stored in the data folder.

Eval Program: `eval_fuzzy.py`

Metrics: dcpmi

Evaluation command:

`bash script/eval_sst2.sh`

`bash script/eval_rt.sh`

`bash script/eval_rte.sh`

`bash script/eval_yahoo.sh`

`bash script/eval_mr.sh`

`bash script/eval_hyp.sh`

`bash script/eval_cr.sh`

`bash script/eval_cb.sh`

`bash script/eval_agn.sh`

