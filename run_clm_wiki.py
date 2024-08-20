#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=causal-lm
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import itertools
import logging
import math
import os
    
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
import json
import datasets
from datasets import load_dataset
from tqdm import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from knnlm import KNNWrapper, KNNSaver, KEY_TYPE, DIST

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0.dev0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

padding_index = -100

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
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
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    eval_subset: str = field(default='validation')
    stride: int = field(default=512)
    patience: int = field(default=None)
    prompt: str = field(default=None)


    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

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
    lmbda: float = field(default=0.25)
    k: int = field(default=1024)
    knn_temp: float = field(default=1.0)
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


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, KNNArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, knn_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, knn_args = parser.parse_args_into_dataclasses()
    training_args._n_gpu = 1
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
    logger.info(f"kNN parameters {knn_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))
    model.to(training_args.device)
    # Injecting KNN
    dimension = model.config.hidden_size
    knn_wrapper = None
    knn_args.seed = training_args.seed

    if knn_args.knn:
        print("dstore_size: ",knn_args.dstore_size)
        print(knn_args.knn_gpu)
        knn_wrapper = KNNWrapper(dstore_size=knn_args.dstore_size, dstore_dir=knn_args.dstore_dir, 
            dimension= dimension, 
            knn_sim_func=knn_args.knn_sim_func, knn_keytype=knn_args.knn_keytype,
            no_load_keys=knn_args.no_load_keys, move_dstore_to_mem=knn_args.move_dstore_to_mem, knn_gpu=knn_args.knn_gpu,
            recompute_dists=knn_args.recompute_dists,
            k=knn_args.k, lmbda=knn_args.lmbda, knn_temp=knn_args.knn_temp, probe=knn_args.probe)
    elif knn_args.save_knnlm_dstore or knn_args.build_index:
        knn_wrapper = KNNSaver(dstore_size=knn_args.dstore_size, dstore_dir=knn_args.dstore_dir, 
            dimension=dimension, knn_keytype=knn_args.knn_keytype)
    
    if knn_wrapper is not None:
        knn_wrapper.break_into(model)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # Evaluation
    def process_texts(test_data, tokenizer):
        word_count = 0
        current_title = None
        current_texts = []
        all_input_ids = []
        all_attention_masks = []

        for text in test_data:
            stripped_text = text.strip()
            
            if stripped_text.startswith("= = =") and stripped_text.endswith("= = ="):
                if current_title and current_texts:
                    combined_text = "\n".join([current_title, "\n".join(current_texts)])         
                    word_count += len(combined_text.split())
                    encoding = tokenizer(combined_text, return_tensors="pt")
                    all_input_ids.append(encoding['input_ids'])
                    all_attention_masks.append(encoding['attention_mask'])
                    current_texts = []
                elif current_title:
                    word_count += len(current_title.split())
                    encoding = tokenizer(current_title, return_tensors="pt")
                    all_input_ids.append(encoding['input_ids'])
                    all_attention_masks.append(encoding['attention_mask'])
                current_title = stripped_text
            else:
                if stripped_text != "":
                    current_texts.append(stripped_text)

        if current_title and current_texts:
            combined_text = "\n".join([current_title, "\n".join(current_texts)])
            word_count += len(combined_text.split())
            encoding = tokenizer(combined_text, return_tensors="pt")
            all_input_ids.append(encoding['input_ids'])
            all_attention_masks.append(encoding['attention_mask'])
        elif current_title:
            word_count += len(current_title.split())
            encoding = tokenizer(current_title, return_tensors="pt")
            all_input_ids.append(encoding['input_ids'])
            all_attention_masks.append(encoding['attention_mask'])

        merged_input_ids = torch.cat(all_input_ids, dim=1)
        merged_attention_masks = torch.cat(all_attention_masks, dim=1)
        
        return {'input_ids': merged_input_ids, 'attention_mask': merged_attention_masks}, word_count

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        print("begin evaluation")
        if knn_args.knn:
            print("parameters")
            print("k",knn_args.k)
            print("lmbda",knn_args.lmbda)
            print("temp",knn_args.knn_temp)
        
     
        ds = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=data_args.eval_subset)
        # Extract text data
        text_data = ds["text"]
        encodings, word_count= process_texts(text_data, tokenizer)

        token_count = encodings['input_ids'].size(1)

        print(f"Word count: {word_count}")
        print(f"Token count: {token_count}")

        # Set parameters for sequence length and stride
        max_length = 1024
        stride = 512
        seq_len = encodings['input_ids'].size(1)

        nlls = []
        prev_end_loc = 0

        # Calculate negative log likelihood for each segment
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc 
            input_ids = encodings['input_ids'][:, begin_loc:end_loc].to(training_args.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                # Loss is calculated using CrossEntropyLoss
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        # Calculate perplexity
        ppl = torch.exp(torch.stack(nlls).mean())
        print("Perplexity:", ppl.cpu().item())
        ratio = token_count / word_count
        word_ppl = torch.exp((torch.stack(nlls).mean()) * ratio)
        print("Word Perplexity:", word_ppl.cpu().item())

        perplexities = {
            "Perplexity": ppl.cpu().item(),
            "Word Perplexity": word_ppl.cpu().item()
        }
        if knn_args.knn:
            file_name = f"{knn_args.k}_{knn_args.lmbda}_{knn_args.knn_temp}_ppl.json"
        else:
            file_name = f"base_ppl.json"
        full_path = os.path.join(training_args.output_dir, file_name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(perplexities, f)

    if knn_args.build_index:
        knn_wrapper.build_index()
    
    if knn_wrapper is not None:
        knn_wrapper.break_out()


if __name__ == "__main__":
    main()