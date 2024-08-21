


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

#### Models Used in the Experiment

- **Llama-2-7b**: [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- **Meta-Llama-3-8B**: [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- **Mistral-7B-v0.3**: [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)

#### Datasets Used to build the datastore
- **wiki**: [wentingzhao/knn-prompt-datastore](https://huggingface.co/datasets/wentingzhao/knn-prompt-datastore)
- **math**: [wentingzhao/math-textbooks](https://huggingface.co/datasets/wentingzhao/math-textbooks)


To save a datastore, run:
```
  bash script/save_dstore.sh {model} {datastore} {path of datastore}
```

e.g.
```
  bash script/save_dstore.sh meta-llama/Llama-2-7b-hf wiki ds/wiki/Llama-2-7b-hf
```


### Step 3: Building the FAISS index


To build the FAISS index yourself, run:

```
  bash script/build.sh {model} {datastore} {path of datastore}
```
#### Download the built datastore
You can also directly access our built datastore through the link below.

Math datastore: https://huggingface.co/datasets/Reset23/math

Wiki datastore: https://huggingface.co/datasets/Reset23/wiki

**How can I download these built datastores?**

For example, to download the math datastore, run:
```
  git clone https://huggingface.co/datasets/Reset23/math
  cd math
  git lfs install
  git lfs pull
```

### Step 4: Evaluating Models

To evaluate kNN-LM on the validation set, run:

```
  bash script/eval.sh {model} {dataset name} {datastore} {path of datastore}
```
(for kNN-LM)

or
```
  bash script/eval.sh {model} {dataset name} base
```
(for base model)



For the wikitext-103 dataset, we performed some preprocessing. The evaluation code runs as follows:
```
  bash script/eval_wiki.sh {model}{datastore} {path of datastore}
```
(for kNN-LM)

or
```
  bash script/eval_wiki.sh {model} base
```
(for base model)

## Evaluating Models on Downstream Tasks
We evaluate both the base model and kNN-LMs on downstream tasks. For each task, we will provide the scripts used for evaluation with the base model or kNN-LM.

For base model we use `script/evaluate.sh {model} {task}`

For kNN-LM we use `script/evaluate_knn.sh {model} {task} {datastore} {path of datastore}`

### Reasoning Tasks
 
#### OpenbookQA

Dataset: https://huggingface.co/datasets/openbookqa

Evaluation command: 

base: `script/evaluate.sh {model} obqa`

kNN-LM: `script/evaluate_knn.sh {model} obqa {datastore} {path of datastore}`

Eval Program: `eval_pmi.py`

Metrics: dcpmi



#### MMLU

Dataset: https://huggingface.co/datasets/cais/mmlu

Evaluation command:

base: `script/evaluate.sh {model} mmlu`

kNN-LM: `script/evaluate_knn.sh {model} mmlu {datastore} {path of datastore}`

Eval Program: `eval_pmi.py`

Metrics: dcpmi

#### Arc

Dataset: https://huggingface.co/datasets/allenai/ai2_arc

Evaluation command: 

**ARC-Challenge**

base: `script/evaluate.sh {model} arc_challenge`

kNN-LM: `script/evaluate_knn.sh {model} arc_challenge {datastore} {path of datastore}`

**ARC-Easy**

base: `script/evaluate.sh {model }arc_easy`

kNN-LM: `script/evaluate_knn.sh {model} arc_easy {datastore} {path of datastore}`

Eval Program: `eval_pmi.py`

Metrics: dcpmi

#### HellaSwag

Dataset: https://huggingface.co/datasets/Rowan/hellaswag

Evaluation command: 

base: `script/evaluate.sh {model} hellaswag`

kNN-LM: `script/evaluate_knn.sh {model} hellaswag {datastore} {path of datastore}`

Eval Program: `evl_pmi.py`

Metrics: dcpmi


#### Drop

Dataset: https://huggingface.co/datasets/drop

Evaluation command:

base: `script/evaluate.sh {model} drop`

kNN-LM: `script/evaluate_knn.sh {model} drop {datastore} {path of datastore}`

Eval Program: `eval_drop.py`

Metrics: F1 score


#### NQ

Dataset: https://huggingface.co/datasets/nq_open

Evaluation command: 

base: `script/evaluate.sh {model} nq`

kNN-LM: `script/evaluate_knn.sh {model} nq {datastore} {path of datastore}`

Eval Program: `eval_qa.py`

Metrics: F1 score


#### HotpotQA

Dataset: https://huggingface.co/datasets/hotpot_qa

Evaluation command:

base: `script/evaluate.sh {model} hotpotqa`

kNN-LM: `script/evaluate_knn.sh {model} hotpotqa {datastore} {path of datastore}`

Eval Program: `eval_qa.py`

Metrics: F1 score


#### GSM8k

Dataset: https://huggingface.co/datasets/gsm8k

Evaluation command: 
base: `script/evaluate.sh {model} gsm8k`

kNN-LM: `script/evaluate_knn.sh {model} gsm8k {datastore} {path of datastore}`

Eval Program: `eval_gsm8k.py`

Metrics: Accuracy

#### BBH

Dataset: https://huggingface.co/datasets/lukaemon/bbh

Evaluation command:

base: `script/evaluate.sh {model} bbh`

kNN-LM: `script/evaluate_knn.sh {model} bbh {datastore} {path of datastore}`

Eval Program: `eval_bbh.py`

Metrics: Accuracy

#### Winogrande

Dataset: https://huggingface.co/datasets/allenai/winogrande

Evaluation command:

base: `script/evaluate.sh {model} winogrande`

kNN-LM: `script/evaluate_knn.sh {model} winogrande {datastore} {path of datastore}`

Eval Program: `eval_winogrande.py`

Metrics: Accuracy


### Memory-intensive Tasks 

Datasets: The corresponding datasets are stored in the data folder.

Eval Program: `eval_fuzzy.py`

Metrics: dcpmi

Evaluation command:
#### SST-2


base: `script/evaluate.sh {model} sst2`

kNN-LM: `script/evaluate_knn.sh {model} sst2 {datastore} {path of datastore}`

#### RT

base: `script/evaluate.sh {model} rt`

kNN-LM: `script/evaluate_knn.sh {model} rt {datastore} {path of datastore}`

#### RTE

base: `script/evaluate.sh {model} rte`

kNN-LM: `script/evaluate_knn.sh {model} rte {datastore} {path of datastore}`

#### Yahoo

base: `script/evaluate.sh {model} yahoo`

kNN-LM: `script/evaluate_knn.sh {model} yahoo {datastore} {path of datastore}`

#### MR

base: `script/evaluate.sh {model} mr`

kNN-LM: `script/evaluate_knn.sh {model} mr {datastore} {path of datastore}`

#### HYP

base: `script/evaluate.sh {model} hyp`

kNN-LM: `script/evaluate_knn.sh {model} hyp {datastore} {path of datastore}`

#### CR

base: `script/evaluate.sh {model} cr}`

kNN-LM: `script/evaluate_knn.sh {model} cr {datastore} {path of datastore}`

#### CB

base: `script/evaluate.sh {model} cb`

kNN-LM: `script/evaluate_knn.sh {model} cb {datastore} {path of datastore}`

#### AGN

base: `script/evaluate.sh {model} agn`

kNN-LM: `script/evaluate_knn.sh {model} agn {datastore} {path of datastore}`




## Acknowledgement
- **knnlm Implementation**: The knnlm is implemented based on the code available at [Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval](https://github.com/neulab/knn-transformers).
- **Data for Memory-Intensive Tasks**: The data used for memory-intensive tasks is sourced from [kNN-Prompt: Nearest Neighbor Zero-Shot Inference](https://github.com/swj0419/kNN_prompt/tree/main/task_data).