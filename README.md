


Table of Contents
=================
  * [Background](#background)
  * [Available Models](#available-models)
  * [Results](#results)
  * [Quickstart](#quickstart)
    * [Requirements](#requirements)
    * [Step 1: Evaluating the base Language Model](#step-1-evaluating-the-base-language-model)
    * [Step 2: Saving a Datastore](#step-2-saving-a-datastore)
    * [Step 3: Building the FAISS index](#step-3-building-the-faiss-index)
    * [Step 4: Evaluating Models](#step-4-evaluating-models)
  * [Evaluation](#evaluation)



## Background

### kNN-LM 
The k-nearest neighbor language model takes an already-trained model, performs a single forward pass over the entire training set, and creates a datastore of `(key,value)` pairs, where `key` is a hidden representation of the trained model after reading a training example, and `value` is the token that should be predicted next.

At test time, for every predicted token, the model performs a k-nearest neighbors search in the datastore, retrieves the `(key,value)` pairs that are closest to the test hidden representation, and normalizes their distances using softmax. Finally, the model interpolates the base LM's probability with the probability formed by the retrieved nearest neighbors and their normalized distances.
For more details, see the [paper by Khandelwal et al., ICLR'2020](https://arxiv.org/pdf/1911.00172.pdf)


## Quickstart - Language Modeling

### Step 0: Clone this repository:
```bash
git clone https://github.com/neulab/knn-transformers
cd knn-transformers
```

#### Requirements 
Run:
```bash
pip install requirements.txt`
```

* The project also depends on the `faiss` library. In MacOS, use the Anaconda installation instead:
```
conda install -c conda-forge faiss-cpu
```

### Step 1: Evaluating the base Language Model

To evaluate the fine-tuned model (for example, `neulab/gpt2-finetuned-wikitext103`) on the validation set (without any retrieval):

```bash
MODEL=meta-llama/Llama-2-7b-hf

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --do_eval --eval_subset validation
```

### Step 2: Saving a Datastore

To save a datastore, run:
```bash
MODEL=meta-llama/Llama-2-7b-hf

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --do_eval --eval_subset train \
  --output_dir checkpoints/${MODEL} \
  --dstore_dir checkpoints/${MODEL} \
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
  bash build.sh
```



### Step 4: Evaluating Models

To evaluate kNN-LM and RetoMaton on the validation set:

```bash
MODEL=meta-llama/Llama-2-7b-hf

python -u run_clm.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --do_eval --eval_subset validation \
  --dstore_dir checkpoints/${MODEL} --retomaton
```

To use kNN-LM, use the `--knn` flag instead of `--retomaton`.

To encourage the RetoMaton model to perform a full kNN search more frequently and thus increase accuracy and reduce perplexity, use a larger value of `--min-knns` such as `100`. Using `--min-knns 9999999` makes the model perform kNN search at every step (`FoSS = 0` in Figure 3 of the paper), and achieves the best results at the cost of slower speed.

Additional possible test-time tunable hyperparameters are `--lmbda` (the interpolation factor between the datastore and the base LM), `--k` (the number of retrieved nearest neighbors) and `--knn_temp` (the softmax temperature when converting the nearest-neighbor distances into a probability distribution).



## Evaluation: 

### OpenbookQA

**Datasets:** https://huggingface.co/datasets/openbookqa

**Evaluation command:** `bash script/eval_obqa_pmi.sh`

**Eval Program:** `eval_pmi.py`

**Metrics:** dcpmi

**Hyperparameter settings:**
```
  --eval_subset test \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --knn \
  --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  --dstore_size 140210422 \
```

### MMLU

**Datasets:** https://huggingface.co/datasets/cais/mmlu

**Evaluation command:** `bash script/eval_mmlu_pmi.sh`

**Eval Program:** `eval_pmi.py`

**Metrics:** dcpmi

**Hyperparameter settings:**
```
  --eval_subset test \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --knn \
  --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  --dstore_size 140210422 \
```
### Arc

**Datasets:** https://huggingface.co/datasets/allenai/ai2_arc

**Evaluation command:** `bash script/eval_arc_pmi.sh`

**Eval Program:** `eval_pmi.py`

**Metrics:** 

ARC-Challenge:  dcpmi

ARC-Easy: Acc

**Hyperparameter settings:**
```
  --eval_subset test \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --knn \
  --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  --dstore_size 140210422 \
```

### HellaSwg

**Datasets:** https://huggingface.co/datasets/Rowan/hellaswag

**Evaluation command:** `bash script/eval_hellaswag.sh`

**Eval Program:** `evl_pmi.py`

**Metrics:** Length normalized Acc

**Hyperparameter settings:**
```
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --knn \
  --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  --dstore_size 140210422 \
```



### Drop

**Datasets:** https://huggingface.co/datasets/drop

**Evaluation command:** `bash script/eval_drop.sh`

**Eval Program:** `eval_drop.py`

**Metrics:** F1 score

**Hyperparameter settings:**
```
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --knn \
  --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  --dstore_size 140210422 \
```



### NQ

**Datasets:** https://huggingface.co/datasets/nq_open

**Evaluation command:** `bash script/eval_nq.sh`

**Eval Program:** `run_qa.py`

**Metrics:** F1 score

**Hyperparameter settings:**
```
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --knn \
  --knn_temp 5.0 --k 1024 --lmbda 0.2 \
  --dstore_size 140210422 \
```


### HotpotQA

**Datasets:** https://huggingface.co/datasets/hotpot_qa

**Evaluation command:** `bash script/eval_hotpot.sh`

**Eval Program:** `run_qa.py`

**Metrics:** F1 score

**Hyperparameter settings:**
```
  --eval_subset validation \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --knn \
  --knn_temp 5.0 --k 1024 --lmbda 0.2 \
  --dstore_size 140210422 \
```

### GSM8k

**Datasets:** https://huggingface.co/datasets/gsm8k

**Evaluation command:** `bash script/eval_gsm.sh`

**Eval Program:** `eval_gsm8k.py`

**Metrics:** Acc

**Hyperparameter settings:**
```
  --eval_subset test \
  --dstore_dir /share/rush/datastore/wiki103/${MODEL} \
  --knn \
  --knn_temp 1.0 --k 1024 --lmbda 0.2 \
  --dstore_size 140210422 \
```

### knn_prompt dataset 

`bash script/eval_sst2.sh`

`bash script/eval_rt.sh`

`bash script/eval_rte.sh`

`bash script/eval_yahoo.sh`

`bash script/eval_mr.sh`

`bash script/eval_hyp.sh`

`bash script/eval_cr.sh`

`bash script/eval_cb.sh`

`bash script/eval_agn.sh`

