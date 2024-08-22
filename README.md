<p align="center">
     <h1>Great Memory, Shallow Reasoning: Limits of kNN-LMs </h1> 
</p>
<div align="center">

| [Paper](https://arxiv.org/abs/2408.11815) |

</div>


Table of Contents
=================
  * [Quickstart](#quickstart)
    * [Step 1: Setup  the Environment](#step-1-setup-environment)
    * [Step 2: Saving a Datastore](#step-2-saving-a-datastore)
    * [Step 3: Building the FAISS index](#step-3-building-the-faiss-index)
    * [Step 4: Evaluating Perplexity](#step-4-evaluating-perplexity)
  * [Evaluating Downstream Tasks](#evaluating-models-on-downstream-tasks)
  * [Acknowledgement](#acknowledgement)
  * [Citation](#citation)
  
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

### Step 4: Evaluating Perplexity

To evaluate kNN-LM on the validation set, run:

```
  bash script/eval_perplexity.sh {model} {dataset name} {datastore} {path of datastore}
```
for kNN-LM

or
```
  bash script/eval_perplexity.sh {model} {dataset name} base
```
for base model

## Evaluating Models on Downstream Tasks
We evaluate both the base model and kNN-LMs on downstream tasks. For each task, we will provide the scripts used for evaluation with the base model or kNN-LM.

### Reasoning Tasks

For base model, run ` bash script/evaluate_downstream.sh {model} {obqa, mmlu, arc_challenge, arc_easy, hellaswag, drop, nq, hotpotqa, gsm8k, bbh, winogrande}`

For kNN-LM, run ` bash script/evaluate_downstream_knn.sh {model} {obqa, mmlu, arc_challenge, arc_easy, hellaswag, drop, nq, hotpotqa, gsm8k, bbh, winogrande} {datastore} {path of datastore}`
 



### Memory-intensive Tasks 

Datasets: The corresponding datasets are stored in the data folder.

Eval Program: `eval_fuzzy.py`

Metrics: dcpmi

Evaluation command:

base: ` bash script/evaluate_downstream.sh {model} {sst2,rt,rte,yahoo,mr,hyp,cr,cb,agn}`

kNN-LM: ` bash script/evaluate_downstream_knn.sh {model} {sst2,rt,rte,yahoo,mr,hyp,cr,cb,agn} {datastore} {path of datastore}`





## Acknowledgement
- **knnlm Implementation**: The knnlm is implemented based on the code available at [Neuro-Symbolic Language Modeling with Automaton-augmented Retrieval](https://github.com/neulab/knn-transformers).
- **Data for Memory-Intensive Tasks**: The data used for memory-intensive tasks is sourced from [kNN-Prompt: Nearest Neighbor Zero-Shot Inference](https://github.com/swj0419/kNN_prompt/tree/main/task_data).

## Citation
If you find our work helpful, please use the following citations.

```
@misc{geng2024greatmemoryshallowreasoning,
      title={Great Memory, Shallow Reasoning: Limits of $k$NN-LMs}, 
      author={Shangyi Geng, Wenting Zhao, Alexander M Rush},
      journal ={arXiv preprint arXiv:2408.11815},
      year={2024}
}
```