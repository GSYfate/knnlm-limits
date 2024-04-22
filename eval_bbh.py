#!/usr/bin/python
import logging
import os
import sys
import json
import re
import string
import csv

from typing import Any, Dict, List, Set, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import datasets
from datasets import load_dataset, get_dataset_config_names
from dataclasses import dataclass, field
from pydantic import BaseModel
import transformers
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from knnlm import KNNWrapper, KNNSaver, KEY_TYPE, DIST

from tqdm import tqdm


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    batch_size: Optional[int] = field(
        default=4,
        metadata={
            "help": (
                "batch size"
            )
        },
    )
    max_input_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_output_length: Optional[int] = field(
        default=32,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    eval_subset: str = field(default='validation')

@dataclass
class KNNArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    knn: bool = field(default=False)
    knn_gpu: bool = field(default=True)
    dstore_size: int = field(default=None, metadata={"help": "The size of the dstore."})
    knn_keytype: KEY_TYPE.from_string = field(default=KEY_TYPE.last_ffn_input)
    save_knnlm_dstore: bool = field(default=False)
    dstore_dir: str = field(default="checkpoints")
    knn_sim_func: DIST.from_string = field(default=DIST.l2)
    lmbda: float = field(default=0.2)
    k: int = field(default=1024)
    knn_temp: float = field(default=1)
    # Args for building the faiss index:
    build_index: bool = field(default=False)
    # faiss_index: str = field(default="checkpoints/index")
    ncentroids: int = field(default=4096)
    code_size: int = field(default=64)
    probe: int = field(default=32)
    num_keys_to_add_at_a_time: int = field(default=1000000)
    move_dstore_to_mem: bool = field(default=True)
    no_load_keys: bool = field(default=True)
    recompute_dists: bool = field(default=False)


class BBHSample(BaseModel):
    input: str
    target: str

    def as_prompt(self, include_answer: bool = True):
        prompt = self.input
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " {}\n\n".format(self.target)
        return prompt


class BBHData(BaseModel):
    samples: List[BBHSample]

    @classmethod
    def get_config_names(cls, path: str = "lukaemon/bbh") -> List[str]:
        return get_dataset_config_names(path)

    @classmethod
    def load_from_huggingface(
        cls, path: str = "lukaemon/bbh", config: str = "", split: str = "test"
    ):
        data = load_dataset(path, config, split=split)
        samples = [BBHSample(**raw) for raw in tqdm(data, desc=str((path, split)))]
        return cls(samples=samples)

def gen_prompt(data: BBHData, k=-1):
    prompt = ""
    if k == -1:
        k = len(data.samples)
    for i in range(k):
        prompt += data.samples[i].as_prompt()
    return prompt

def evaluate(model, tokenizer, device, data, name, ntrain, max_output_length, output_dir) -> dict:
    data_train = BBHData(samples=data.samples[:ntrain])
    data_test = BBHData(samples=data.samples[ntrain:])
    is_correct = []
    results = []
    for i in tqdm(range(len(data_test.samples)), desc='Evaluating'):
        # get prompt and make sure it fits
        k = int(ntrain)
        prompt_end = data_test.samples[i].as_prompt(include_answer=False)
        train_prompt = gen_prompt(data_train, k)
        prompt = train_prompt + prompt_end

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_output_length,
            pad_token_id= tokenizer.eos_token_id
        )
        batch_size, length = inputs.input_ids.shape
        pred = tokenizer.decode(outputs[0, length:], skip_special_tokens=True)
        label = data_test.samples[i].target
        results.append({
            "question": prompt_end.strip(),
            "prediction": pred.strip(),
            "label": label.strip()
        })
        is_correct.append(pred.strip().startswith(label))
        if i == 0:
            print(dict(prompt=prompt, label=label, pred=pred))
            
    accuracy = sum(is_correct) / len(is_correct)
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{name}_results.json")
    with open(file_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    return dict(acc=sum(is_correct) / len(is_correct))


def seed_everything(seed=42):
    """
    Seed everything to make results replicable.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, KNNArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, knn_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, knn_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    seed = 42
    set_seed(seed)
    seed_everything(seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        padding_side='left',
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch.float16,
    )

    model.resize_token_embeddings(len(tokenizer))
    #if llama
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if training_args.do_eval:
        max_output_length = data_args.max_output_length
        max_input_length = data_args.max_input_length
    
 

    # Injecting KNN
    dimension = model.config.hidden_size
    knn_wrapper = None
    knn_args.seed = training_args.seed

    if knn_args.knn:
        knn_wrapper = KNNWrapper(dstore_size=knn_args.dstore_size, dstore_dir=knn_args.dstore_dir, 
            dimension= dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys, move_dstore_to_mem=knn_args.move_dstore_to_mem, knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k, lmbda=knn_args.lmbda, knn_temp=knn_args.knn_temp, probe=knn_args.probe)
    elif knn_args.save_knnlm_dstore or knn_args.build_index:
        training_args.predict_with_generate = False
        knn_wrapper = KNNSaver(dstore_size=knn_args.dstore_size, dstore_dir=knn_args.dstore_dir, 
            dimension=dimension, knn_keytype=knn_args.knn_keytype)
    
    if knn_wrapper is not None:
        knn_wrapper.break_into(model)

    # Evaluation
    if training_args.do_eval:
        model.eval()
        model.to(training_args.device)
        all_results = []
        total_test_samples = 0
        ntrain = 3
        if data_args.dataset_config_name is not None:
            name_list = [data_args.dataset_config_name]
        else:
            name_list = BBHData.get_config_names(path=data_args.dataset_name)
        for name in name_list:
            print(f'task: {name}')
            data = BBHData.load_from_huggingface(config=name)
            n_test_samples = len(data.samples) - ntrain
            total_test_samples += n_test_samples 
            result = evaluate(
                model, tokenizer, training_args.device, data, name, ntrain=ntrain, max_output_length=max_output_length, output_dir=training_args.output_dir)
            result_dict = {
                "name": name,
                "acc": result["acc"],
                "test_samples": n_test_samples
            }
            all_results.append(result_dict)
            print(result_dict)
        
        score = sum(res["acc"] for res in all_results) / len(all_results)
        print(f"Cumulative score: {score}")
        output_dir = training_args.output_dir  # Make sure this directory exists or create it
        os.makedirs(output_dir, exist_ok=True)
        csv_file_path = os.path.join(output_dir, 'bbh_results.csv')
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['name', 'acc', 'test_samples']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in all_results:
                writer.writerow(result)

            writer.writerow({'name': 'Cumulative Score', 'acc': score, 'test_samples': total_test_samples})

    if knn_args.build_index:
        knn_wrapper.build_index()
    
    if knn_wrapper is not None:
        knn_wrapper.break_out()
    


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
