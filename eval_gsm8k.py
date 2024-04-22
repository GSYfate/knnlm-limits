#!/usr/bin/python
import logging
import os
import sys
import json
from dataclasses import dataclass, field

import re
import csv
from typing import Any, Dict, List, Set, Tuple, Union, Optional

import evaluate
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
import datasets
import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from knnlm import KNNWrapper, KNNSaver, KEY_TYPE, DIST


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
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on a jsonlines file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (sacreblue) on a jsonlines file."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
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
    max_source_length: Optional[int] = field(
        default=4096,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    eval_subset: str = field(default='validation')
    patience: int = field(default=None)

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        valid_extensions = ["json", "jsonl"]
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in valid_extensions, "`train_file` should be a jsonlines file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in valid_extensions, "`validation_file` should be a jsonlines file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

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

#   "Question: Jesse and Mia are competing in a week long race. They have one week to run 30 miles. On the first three days, Jesse averages (2/3) of a mile. On day four she runs 10 miles. Mia averages 3 miles a day over the first 4 days. What is the average of their average that they have to run over the final three days?\nAnswer: Jesse runs 2 miles in the first three days because 3 x (2/3) = <<3*(2/3)=2>>2.\nJesse has 18 miles left to run because 30 - 10 - 2 = <<30-10-2=18>>18.\nJesse has to run an average of 6 miles a day because 18 / 3 = <<18/3=6>>6.\nMia runs 12 miles over the first four days because 4 x 3 = <<4*3=12>>12.\nShe has 18 miles left to run because 30 - 12 = <<30-12=18>>18.\nShe has to run six miles a day because 18 / 3 = <<18/3=6>>6.\nThe total they both have to run is <<12=12>>12 miles a day.\nThe average they have to run per day on average is 6 miles because 12 / 2 = <<12/2=6>>6.\n#### 6"
#         "Question: Paul went to a shop to buy some groceries. He bought some bread for $2, butter for $3, and juice for two times the price of the bread. He had $15 for his shopping. How much money did Paul have left?"

def build_qa_prompt(question):
    system_prompt = (
        "Your task is to solve a primary school level math problem. "
        "You should provide both a chain of reasoning and a final answer. "
        "The final answer should be clearly indicated, preceded by '####', for instance, #### 72.\n"
        "Here are some examples:\n"
    )

    ex_prompt = [
        "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\n"
        "Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. #### 6",
        
        "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\n"
        "Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
        
        "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n"
        "Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. #### 39",
        
        "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\n"
        "Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. #### 8",
        
        "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\n"
        "Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. #### 9",
        
        "Question: There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?\n"
        "Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. #### 29",
        
        "Question: Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?\n"
        "Answer: Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. #### 33",
        
        "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\n"
        "Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 = 8. #### 8"
    ]


    question_prompt = f"Question: {question}\nAnswer: "

    prompt = system_prompt + "\n".join(ex_prompt) + '\n' + question_prompt
    return prompt

def get_answer_from_model_output(outputs, tokenizer, prompt):
        generation_str = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
        generation_str = generation_str[len(prompt):]
        answer = generation_str.split("\n")[0]
        return answer, generation_str

def extract_answer_and_chain(response):
    answer_pattern = re.compile(r"#### ([0-9.,$]+)")
    match = answer_pattern.search(response)
    if not match:
        return None, None
    try:
        answer = re.search(r'[0-9.]+', match.group(1)).group()
    except Exception:
        return None, None
    answer = [x for x in answer.split('.') if x != '']
    if len(answer) == 1:
        answer = float(answer[0])
    elif len(answer) == 2:
        answer = float(f'{answer[0]}.{answer[1]}')
    else:
        answer = 0
    reasoning = response.split("####")[0]
    reasoning = [one.strip() for one in reasoning.split('\n')]
    reasoning = [one for one in reasoning if one != ""]
    return answer, reasoning

def check_answer(predict_response, actual_result):
    cur_answer, _ = extract_answer_and_chain(predict_response)
    real_answer, _ = extract_answer_and_chain(actual_result)
    return cur_answer == real_answer

def evaluate_dataset(model, tokenizer, device, eval_dataset, max_length, batch_size=4, output_dir=None):
    idx = 0
    num_correct = 0
    num_too_long = 0
    sample_prompt = None
    results = []
    model.to(device)
    tq = tqdm(total=len(eval_dataset), desc="Acc:  0.0%")
    # Process examples in batches
    for batch_start in range(0, len(eval_dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(eval_dataset))
        batch = eval_dataset[batch_start:batch_end]
        prompts = [build_qa_prompt(question) for question in batch['question']]
        if idx == 0:
            sample_prompt = prompts[0]
        answers = batch['answer']
        
        # Tokenize all prompts in the batch
        batch_encoding = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = batch_encoding['input_ids'].to(device)
        attention_mask = batch_encoding['attention_mask'].to(device)
    
        with torch.no_grad():
            if 'mistral' in model.name_or_path.lower():
                attention_mask = torch.ones(input_ids.shape, device=device)
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_length=max_length)
            else:
                outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=512)
        
        # Process each example in the batch
        for i, output in enumerate(outputs):
            prediction, generation = get_answer_from_model_output(output.unsqueeze(0), tokenizer, prompts[i])
            is_correct = check_answer(prediction, answers[i])
            num_correct += is_correct
            results.append({'Question': batch['question'][i], 'Prediction': prediction, 'Answer': answers[i], 'Correct': is_correct})
        
        idx += batch_size
        tq.update(batch_size)
        tq.set_description(f"Acc:  {num_correct / idx:.2%}")
        
    tq.close()
    acc = num_correct / idx * 100
    print(f"Acc: {acc:.2f}%")
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir,'gsm8k_results.csv'), 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Question', 'Prediction', 'Answer', 'Correct']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        d = {"Acc": acc, "num_examples": idx, "num_correct": num_correct}
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")
        if sample_prompt is not None:
            with open(os.path.join(output_dir, "example_prompt.txt"), "w") as f:
                f.write(sample_prompt)
    return results

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

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_translation", model_args, data_args)

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

    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )

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


    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets[data_args.eval_subset].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    elif knn_args.build_index:
        logger.info("Building index")
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    # padding = "max_length" if data_args.pad_to_max_length else "max_length"
    padding = "max_length" if data_args.pad_to_max_length else False
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
    
        eval_dataset = raw_datasets[data_args.eval_subset]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        if knn_args.dstore_size is None and knn_args.save_knnlm_dstore:
            knn_args.dstore_size = total_eval_tokens

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

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
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    if training_args.do_eval:
        # print("eval dataset is:",eval_dxataset[0])
        evaluate_dataset(
            model, tokenizer, training_args.device, eval_dataset,
            max_length=max_length,
            batch_size=data_args.batch_size,
            output_dir=training_args.output_dir,
        )

    if knn_args.build_index:
        knn_wrapper.build_index()
    
    if knn_wrapper is not None:
        knn_wrapper.break_out()
    
    
def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()
