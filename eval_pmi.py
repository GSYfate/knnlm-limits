import logging
import os
import sys
import re
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import transformers
import json


from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset
from tqdm import tqdm


from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

from transformers.utils.versions import require_version

from knnlm import KNNWrapper, KNNSaver, KEY_TYPE, DIST


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0.dev0")

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
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=1024,
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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
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

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, KNNArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
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
    print("seed")
    print(training_args.seed)
    set_seed(training_args.seed)
    # seed = 42
    # set_seed(seed)
    seed_everything(training_args.seed)

    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
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
 
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id


    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
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
    

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        total_eval_tokens = 0        
        for chunk in train_dataset['labels']:
            total_eval_tokens += len([x for x in chunk[1:] if x != -100])
        logger.info(f'[train] Total eval tokens: {total_eval_tokens}')

    if training_args.do_eval:
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
    print("dimension is", dimension)
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
        

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=data_args.patience)] if data_args.patience is not None else None,
    )

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams


    def cross_entropy_list(sources, targets, model, encoder):
        '''
        Gets a list of CE values, where the ith item is a list of cross-entropies
        for targets[i] with sources[i] as contexts

        targets and sources are lists of lists of tokens (integers)

        model is a language model
        
        '''
        assert(len(sources ) == len(targets))
        n_seqs = len(sources)
    
        torch.cuda.empty_cache()
        device = next(model.parameters()).device

        # initialize input tensors
        max_len = max([len(s + t) for s,t in zip(sources, targets)])
        input_ids = torch.zeros((n_seqs, max_len)).long() 
        labels = -100 * torch.ones((n_seqs, max_len)).long()
        
        for i, (source, target) in enumerate(zip(sources,targets)):
            s = torch.tensor(source).long()
            if "Meta-Llama-3-8B" in model.config._name_or_path:
                t = torch.tensor(target[1:]).long()
            elif "Llama-2-7b-hf" in model.config._name_or_path:
                t = torch.tensor(target[2:]).long()
            else:
                t = torch.tensor(target[1:]).long()
            input_ids[i,:len(s)] = s
            input_ids[i,len(s):len(s) + len(t)] = t
            labels[i,len(s):len(s) + len(t)] = t
        
        with torch.no_grad():
            input_ids = input_ids.to(device)
            outputs = model(input_ids, labels=labels)
            logits = outputs.logits.cpu()
            logits = logits[:, :-1, :].contiguous()

        # get cross-entropies given the logits
        logit_shape = logits.shape
        logits = logits.view(-1, logit_shape[-1])

        ce_list = F.cross_entropy(logits, labels[:,1:].contiguous().view(-1), reduction='none',ignore_index = -100)
        ce_list = ce_list.view(n_seqs, max_len -1).sum(dim=1).squeeze().tolist()
        
        # if one element (i.e. len(sources) == 1), nest it into a list. Otherwise, give full list
        # this just handles an idiosyncracy of the .tolist() function
        try:
            len(ce_list)
        except:
            ce_list = [ce_list]
        return ce_list

    def inference_autobatch( model, encoder, example, max_len = 1024):
        #####
        ## input data handling
        options = []
        for opt_raw in example['options']:
            opt = { key: encoder.encode(opt_raw[key]) for key in opt_raw.keys() }
            ## trim the option to the max length for gpt2
            opt['premise'] = opt['premise'][-(max_len - len(opt['hypothesis'])):]
            assert(len(opt['premise'] + opt['hypothesis']) <= max_len)
            # then add the encoded, trimmed option
            options.append(opt)

        #####
        ## cross-entropy calculation
        #####
        cond_ce = cross_entropy_list([opt['premise'] for opt in options], [opt['hypothesis'] for opt in options],model, encoder)

        ## get domain conditional CEs
        domain_cond_ce  = cross_entropy_list([opt['uncond_premise'] for opt in options],[opt['uncond_hypothesis'] for opt in options], model, encoder)
        
        ## get unconditional CEs
        uncond_ce = cross_entropy_list([[584] for opt in options], [opt['uncond_hypothesis'] for opt in options], model, encoder)

        ## get average CE by token
        avg_cond_ce = [ce / len(opt['hypothesis']) for ce, opt in zip(cond_ce, options)]
        
        # calculate dcpmi
        dcpmi = [ce_0 - ce_1 for ce_0,ce_1 in zip(domain_cond_ce, cond_ce)]
        pmi = [ce_0 - ce_1 for ce_0,ce_1 in zip(uncond_ce, cond_ce)]

        
        ## make predictions based on different scores
        lm_pred = cond_ce.index(min(cond_ce))
        lm_avg_pred = avg_cond_ce.index(min(avg_cond_ce))
        lm_domain_cond_pred = domain_cond_ce.index(min(domain_cond_ce))
        dcpmi_pred = dcpmi.index(max(dcpmi))
        pmi_pred = pmi.index(max(pmi))
        pred = {
                    'lm': lm_pred,
                    'tok_mean': lm_avg_pred,
                    'dcpmi' : dcpmi_pred,
                    'pmi': pmi_pred,
                    'domain_cond': lm_domain_cond_pred,
            }
        return pred

            
    def fwd(model, encoder, examples, max_len):
        '''
        This is designed for gpt2-style language models
        
        Inputs: (any you don't know)
            model - a HuggingFace Transformers gpt-2 model

            encoder - a HuggingFace Transformers tokenizer

            examples = [ex1, ex2, ...]
                where ex = [opt1, opt2, ...] (multiple choice options)
                where opt = (premise, hypothesis) 
            
            batch: is the max allowed batch size (set to 1 for no batching)
        '''
        
        if type(model) != str:
            print("Not Str")
            # print the first example to make sure the format is ok
            print('='*50)
            print('MAKE SURE TOKENIZATION AND FORMATTING LOOKS OK')
            print('\nprint example 0 of {}:'.format(len(examples)))
            ex = examples[0]
            options = ex['options']
            opt = options[0]
            print('CONDITIONAL:')
            print(encoder.decode(encoder.encode(opt['premise'])) + '<BREAK>' + encoder.decode(encoder.encode(opt['hypothesis'])))
            print('UNCONDITIONAL:')
            print(encoder.decode(encoder.encode(opt['uncond_premise'])) + '<BREAK>' + encoder.decode(encoder.encode(opt['uncond_hypothesis'])))
            print('='*50)
        else:
            # print the first example to make sure the format is ok
            print('='*50)
            print('MAKE SURE TOKENIZATION AND FORMATTING LOOKS OK')
            print('\nprint example 0 of {}:'.format(len(examples)))
            ex = examples[0]
            options = ex['options']
            opt = options[0]
            print('CONDITIONAL:')
            print(opt['premise'] + '<BREAK>' + opt['hypothesis'])
            print('UNCONDITIONAL:')
            print(opt['uncond_premise'] + '<BREAK>' + opt['uncond_hypothesis'])
            print('='*50)

        predictions_list = []
        

        print('actually calculating')
        print('max length is ', max_len)
        for example in tqdm(examples):
            pred = inference_autobatch(model, encoder, example, max_len = max_len)
            predictions_list.append(pred)

            
        labels = [ex['label'] for ex in examples]
 
        # get predictions into list by scoring key
        predictions_dict = {key:list(map(lambda v: v[key], predictions_list)) for key in predictions_list[0].keys()}

        # calculate accuracies
        results = {key: sum(list(map(lambda v: v[0] == v[1], zip(predictions_dict[key] , labels) )))/len(labels) for key in predictions_dict.keys()}

        # save labels for later
        predictions_dict['labels'] = labels
        return results, predictions_dict



    def preprocess_dataset(dataset, name):
        if 'openbookqa' in name:
            print("obqa")
            idx2abc = { 0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D' }
            abc2idx = { 'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3 }
            examples = []
            for ex in dataset:
                d = {}
                label = ex['answerKey']
                correct_hypothesis = abc2idx[label]
                stem = ex['question_stem']
                choices = ex['choices']
                hypotheses = []
                for idx in range(4):
                    text = choices['text'][idx]
                    label = choices['label'][idx]
                    assert(abc2idx[label] == idx)
                    hypotheses.append(text)

                d['premise'] = stem
                d['hypotheses'] = hypotheses
                d['correct_hypothesis'] = correct_hypothesis
 

                premise = d['premise']
                options = []
                for h in d['hypotheses']:
                    o = {}
                    h = ' ' + h
                    o['premise'] = premise
                    o['hypothesis'] = h
                    o['uncond_premise'] = ' the answer is:'
                    o['uncond_hypothesis'] = h
                    options.append(o)
                label = d['correct_hypothesis']
                examples.append({'options' : options, 'label' : label })

            return examples

        elif 'mmlu' in name:
            print('mmlu')
            examples = []
            for ex in dataset:
                d = {}
                correct_hypothesis = ex['answer']
                stem = ex['question']
                choices = ex['choices']
                hypotheses = []
                for idx in range(4):
                    text = choices[idx]
                    hypotheses.append(text)
                d['premise'] = stem
                d['hypotheses'] = hypotheses
                d['correct_hypothesis'] = correct_hypothesis
 
                premise = d['premise']
                options = []
                for h in d['hypotheses']:
                    o = {}
                    h = ' ' + h
                    o['premise'] = premise
                    o['hypothesis'] = h
                    o['uncond_premise'] = ' the answer is:'
                    o['uncond_hypothesis'] = h
                    options.append(o)
                label = d['correct_hypothesis']
                examples.append({'options' : options, 'label' : label })

            return examples

        elif 'arc' in name:
            print('arc')
            idx2abc = { 0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D' }
            abc2idx = { 'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3 }
            examples = []
            for ex in dataset:
                d = {}
                label = ex['answerKey']
                if label in "ABCD":
                    correct_hypothesis = abc2idx[label]
                else:
                    correct_hypothesis = label

                stem = ex['question']
                choices = ex['choices']
                hypotheses = []
                for idx in range(len(choices['text'])):
                    text = choices['text'][idx]
                    label = choices['label'][idx]
                    hypotheses.append(text)

                d['premise'] = stem
                d['hypotheses'] = hypotheses
                d['correct_hypothesis'] = correct_hypothesis
 

                premise = d['premise']
                options = []
                for h in d['hypotheses']:
                    o = {}
                    h = ' ' + h
                    o['premise'] = premise
                    o['hypothesis'] = h
                    o['uncond_premise'] = ' the answer is:'
                    o['uncond_hypothesis'] = h
                    options.append(o)
                label = d['correct_hypothesis']
                examples.append({'options' : options, 'label' : label })

            return examples

        elif 'hellaswag' in name:
            print("hella")
            def preprocess(text):
                text = text.strip()
                # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
                text = text.replace(" [title]", ". ")
                text = re.sub('\\[.*?\\]', '', text)
                text = text.replace("  ", " ")
                return text

            examples = []
            for ex in dataset:
                d = {}
                label = int(ex['label'])
                correct_hypothesis = label

                stem = preprocess(ex['activity_label'] + ': ' + ex['ctx'])
                choices = ex['endings']
                hypotheses = []
                for idx in range(len(choices)):
                    text = preprocess(choices[idx])
                    hypotheses.append(text)

                d['premise'] = stem
                d['hypotheses'] = hypotheses
                d['correct_hypothesis'] = correct_hypothesis

                premise = d['premise']
                options = []
                for h in d['hypotheses']:
                    o = {}
                    h = ' ' + h
                    o['premise'] = premise
                    o['hypothesis'] = h
                    o['uncond_premise'] = ' the answer is:'
                    o['uncond_hypothesis'] = h
                    options.append(o)
                label = d['correct_hypothesis']
                examples.append({'options' : options, 'label' : label })
            return examples


    
    def score(model, model_name, encoder, examples, stem, split, max_len):
        accs, preds = fwd(model, encoder, examples,  max_len)
        # save scores
        results_path = f'{stem}/{split}.accs'
        with open(results_path,'w') as out:
            out.write(json.dumps(accs))
        # save predicted labels
        preds_path = f'{stem}/{split}.preds'
        with open(preds_path, 'w') as out:
            out.write(json.dumps(preds))
        return accs

    if training_args.do_eval:
        model.eval()
        examples = preprocess_dataset(eval_dataset, data_args.dataset_name)
        name = model_args.model_name_or_path,
        accs = score(model, name, tokenizer, examples, training_args.output_dir, data_args.eval_subset, max_length)
        print(f'{name} gets {accs}% on {data_args.dataset_name}')
        print(f"{accs['domain_cond']} & {accs['lm']} & {accs['tok_mean']} & {accs['pmi']} & {accs['dcpmi']}")
        if knn_args.knn:
            print("parameters")
            print("k",knn_args.k)
            print("lmbda",knn_args.lmbda)
            print("temp",knn_args.knn_temp)
        d = {
            "acc": accs['lm'],
            "token_mean_acc": accs['tok_mean'],
            "pmi": accs['pmi'], 
            "dcpmi": accs['dcpmi'], 
            "num_examples": len(examples), 
        }
        with open(os.path.join(training_args.output_dir, "eval.json"), "w") as f:
            f.write(json.dumps(d) + "\n")
  

    kwargs = {}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if knn_args.build_index:
        knn_wrapper.build_index()
    
    if knn_wrapper is not None:
        knn_wrapper.break_out()
    
    return results


if __name__ == "__main__":
    main()
