import os

import logging
import time
import numpy as np
import torch
from torch import nn
from enum import Enum, auto
from pathlib import Path

import faiss
import faiss.contrib.torch_utils

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
logger = logging.getLogger(__name__)
logger.setLevel(20)

class DIST(Enum):
    l2 = auto()
    dot = auto()

    @staticmethod
    def from_string(s):
        try:
            return DIST[s.lower()]
        except KeyError:
            raise ValueError()


class KEY_TYPE(Enum):
    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s):
        try:
            return KEY_TYPE[s.lower()]
        except KeyError:
            raise ValueError()

class KNNWrapper(object):
    def __init__(self, dstore_size, dstore_dir, dimension, 
            knn_sim_func=None, knn_keytype=None,
            no_load_keys=False, move_dstore_to_mem=False, knn_gpu=True,
            recompute_dists = False,
            k=1024, lmbda=0.25, knn_temp=1.0, probe=32):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.lmbda = lmbda
        self.k = k
        self.knn_temperature = knn_temp
        self.probe = probe
        self.knn_sim_func = DIST.l2 if knn_sim_func is None else knn_sim_func
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype
        self.no_load_keys = no_load_keys
        self.recompute_dists = recompute_dists
        self.move_dstore_to_mem = move_dstore_to_mem
        self.knn_gpu = knn_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prompt_input_ids = None
        self.keys = None
        self.values = None
        self.prompt_attention_mask = None
        self.model = None
        self.vocab_size = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.hook_handles = []

        dist_type_to_dist_func = {
            DIST.l2: KNNWrapper.l2,
            DIST.dot: KNNWrapper.dotprod,
        }
        self.dist_func = dist_type_to_dist_func[knn_sim_func] # l2 or dot product function


    def setup_faiss(self):
        if not self.dstore_dir:
            raise ValueError('Cannot build a datastore without the data.')

        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension) 
        interval = 160000000
        cpu_indices = []
        for i in range(0, self.dstore_size, interval):
            start_time = time.time()
            start, end = i, min(self.dstore_size, i+interval)
            curr_index_name = index_name.replace('.indexed', f'_{start}_{end}.indexed')
            cpu_indices.append(faiss.read_index(curr_index_name, faiss.IO_FLAG_ONDISK_SAME_DIR))
            cpu_indices[-1].nprobe = self.probe
            logger.info(f'Reading {curr_index_name} took {time.time() - start_time} s')

        if self.knn_gpu:
            print("use knn_gpu")
            start_time = time.time()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_resource = faiss.StandardGpuResources()
            gpu_indices = []
            for i, cpu_index in enumerate(cpu_indices):
                if len(cpu_indices) >= 4 and torch.cuda.device_count() > 0:
                    if i in [0, 1, 2, 8]:
                        gpu_indices.append(faiss.index_cpu_to_gpu(gpu_resource, 1, cpu_index, co))
                    else:
                        gpu_indices.append(faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index, co))
                else:
                    gpu_indices.append(faiss.index_cpu_to_gpu(gpu_resource, 0, cpu_index, co))
                logger.info(f'Moving index {i} to GPU took {time.time() - start_time} s')
        else:
            gpu_indices = cpu_indices

        # make_direct_map() allows calling reconstruct(n), 
        # and reconstructing key vectors given their ids
        # currently, this is implemented only for CPU indexes:
        # https://github.com/facebookresearch/faiss/issues/2181
        for cpu_index in cpu_indices:
            cpu_index.make_direct_map()
        print("finish direct map")
        keys_vals_prefix = get_dstore_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension)
        if not self.no_load_keys:
            self.keys = np.memmap(f'{keys_vals_prefix}_keys.npy', dtype=np.float16, mode='r',
                                  shape=(self.dstore_size, self.dimension))
        self.vals = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r',
                              shape=(self.dstore_size, 1))
        # self.vals = torch.from_numpy(self.vals).to(self.device)

        # If you wish to load all the keys into memory
        # CAUTION: Only do this if your RAM can handle it!
        if self.move_dstore_to_mem:
            logger.info('Loading to memory...')
            start = time.time()

            if not self.no_load_keys:
                del self.keys
                self.keys_from_memmap = np.memmap(f'{keys_vals_prefix}_keys.npy', 
                    dtype=np.float16, mode='r', shape=(self.dstore_size, self.dimension))
                self.keys = self.keys_from_memmap[:].astype(np.float16)

            del self.vals
            vals_from_memmap = np.memmap(f'{keys_vals_prefix}_vals.npy', dtype=np.int32, mode='r', shape=(self.dstore_size, 1))
            self.vals = torch.from_numpy(vals_from_memmap[:]).long().to(self.device)
            del vals_from_memmap
            logger.info('Loading to memory took {} s'.format(time.time() - start))

        return cpu_indices, gpu_indices

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.reconstruct_index, self.indices = self.setup_faiss()
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)

        self.register_hook(final_layer, self.post_forward_hook)
        self.vocab_size = final_layer.out_features


    def get_knns(self, queries):
        if not self.knn_gpu:
            queries = queries.cpu()
        all_dists, all_knns = [], []
        queries = queries.float()
        for idx, one in enumerate(self.indices):
            curr_dists, curr_knns = one.search(queries, self.k)
            tot = 0
            for x in self.indices[:idx]:
                tot += x.ntotal
            curr_knns[0, 0].cpu().item()
            curr_knns[0, 0].cpu().item()
            curr_knns = curr_knns + tot
            all_dists.append(curr_dists)
            all_knns.append(curr_knns)
        knns = torch.cat(all_knns, dim=-1)
        dists = torch.cat(all_dists, dim=-1)
        indices = torch.topk(-dists, self.k, dim=-1).indices
        knns = torch.gather(knns, -1, indices)
        dists = torch.gather(dists, -1, indices)
        #dists, knns = self.index.search(queries, self.k)
        dists, knns = dists.to(self.device), knns.to(self.device)
        return dists, knns

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        # print("call post forward")
        batch, time_dim, vocab_size = output.shape
        shift = 0 if self.is_encoder_decoder else 1
        lm_logits = output
        lm_logits = torch.nn.functional.log_softmax(lm_logits, dim=-1) # (batch, time, vocab)

        queries = self.activation_capturer.captured # (batch, time, dim)

        if self.labels is None:
            nonpad_mask = torch.cat([
                torch.zeros([batch, time_dim - 1], dtype=torch.bool),
                torch.ones([batch, 1], dtype=torch.bool),
            ], axis=-1).to(self.device)
        else:
            self.labels = self.labels.to(self.device)
            nonpad_mask = torch.cat([
                self.labels[:, shift:] != -100, 
                torch.zeros([self.labels.shape[0], shift], dtype=torch.bool).to(self.device)
            ], axis=-1)

        lm_logits = lm_logits[nonpad_mask]
        queries = queries[nonpad_mask] # (nonpad, dim)

        dists, knns = self.get_knns(queries) # (nonpad batch * time, k)

        if self.recompute_dists:
            knns_vecs = torch.from_numpy(self.keys[knns]).to(self.device)
            dists = self.dist_func(queries, knns_vecs) 
        
        neg_dists = -dists

        knn_log_probs, _ = self.knns_to_log_prob(knns, neg_dists)
        interpolated_scores = KNNWrapper.interpolate(knn_log_probs, lm_logits, self.lmbda) # (nonpad, vocab)
        #output[nonpad_mask] = interpolated_scores
        output[nonpad_mask] = interpolated_scores.half()
        return output 

    def knns_to_log_prob(self, knns, neg_dists):
        probs = torch.nn.functional.softmax(neg_dists / self.knn_temperature, dim=-1)
        vals_at_knns = self.vals[knns].squeeze(-1) # (nonpad batch * time, k)
        knn_log_probs = torch.full(size=(vals_at_knns.shape[:-1] + (self.vocab_size,)), fill_value=0.0).to(self.device)
        knn_log_probs = knn_log_probs.scatter_add(dim=-1, index=vals_at_knns, src=probs).log()
        knn_log_probs = torch.nan_to_num(knn_log_probs, nan=None, neginf=-10000.0)
        return knn_log_probs, vals_at_knns
        
    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)

    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None
    
    def get_metrics(self):
        return {}
    
    @staticmethod
    def l2(query, keys):
        # query: (batch*time, dim)
        # keys:  (batch*time, k, dim)
        # returns: (batch*time, k)
        return torch.sum((query.unsqueeze(-2) - keys)**2, dim=-1)

    @staticmethod
    def dotprod(query, keys):
        # query: (batch, beams, dim)
        # keys:  (batch, 1, time, dim)
        # returns: (batch, beams, time)
        return torch.sum((query.unsqueeze(-2) * keys), dim=-1)


    @staticmethod
    def interpolate(knn_log_probs, lm_log_probs, lmbda):
        interpolated = torch.logaddexp(
            #lm_log_probs + torch.log(1 - lmbda),
            #knn_log_probs + torch.log(lmbda))
            lm_log_probs + np.log(1 - lmbda),
            knn_log_probs + np.log(lmbda))

        return interpolated

    @staticmethod
    def get_model_last_layer(model_type):
        # works for gpt2, marian, t5. If a model does not have an ".lm_head" layer, 
        # add an "if model_type is ..." statement here, and return the output embedding layer
        print("call get_model_last_layer")
        return lambda model: model.lm_head

    @staticmethod
    def get_model_embedding_layer(model_type):
        if model_type.startswith('gpt2'):
            return lambda model: model.transformer.wte

    # For every model name and key type, returns a lambda that returns the relevant layer in the model, 
    # and whether the input of that layer should be captured (True) or the output (False)
    model_layer_to_capture = {
        'bart': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        'gpt2': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.h[-1], False),
        },
        'marian': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
        },
        't5': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.decoder.block[-1].layer[2].DenseReluDense, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.decoder.block[-1].layer[2], False),
        },
        'llama': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.layers[-1], False),
        },
        'mistral': {
            KEY_TYPE.last_ffn_input: (lambda model: model.base_model.layers[-1].mlp, True),
            KEY_TYPE.last_ffn_output: (lambda model: model.base_model.layers[-1], False),
        }

}
    

class KNNSaver(object):
    def __init__(self, dstore_size, dstore_dir, dimension, knn_keytype=None):
        self.dstore_size = dstore_size
        self.dstore_dir = dstore_dir
        self.dimension = dimension
        self.knn_keytype = KEY_TYPE.last_ffn_input if knn_keytype is None else knn_keytype

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.activation_capturer = None
        self.is_encoder_decoder = None
        self.dstore_idx = 0
        self.dstore_keys = None
        self.dstore_vals = None
        self.labels = None
        self.hook_handles = []

        logger.info(f'keytype being saved: {self.knn_keytype}')
        logger.info('Saving fp16')

    def break_into(self, model):
        self.model = model
        model.broken_into = True
        self.is_encoder_decoder = model.config.is_encoder_decoder
        
        # Inject our activation_capturer to capture the activations at every forward pass
        layer_to_capture_fn, capture_input = KNNWrapper.model_layer_to_capture[model.config.model_type][self.knn_keytype]
        layer_to_capture = layer_to_capture_fn(model)
        self.activation_capturer = ActivationCapturer(layer_to_capture, capture_input=capture_input)
        self.register_hook(layer_to_capture, self.activation_capturer)

        # Inject our pre_forward_hook to capture the labels at every forward pass
        self.original_forward_func = model.forward
        model.forward = self.pre_forward_hook
        
        # Inject our main function after the model's final layer
        final_layer = KNNWrapper.get_model_last_layer(model.config.model_type)(model)
        self.register_hook(final_layer, self.post_forward_hook)
        keys_vals_prefix = get_dstore_path(self.dstore_dir, model.config.model_type, self.dstore_size, self.dimension)
        keys_filename = f'{keys_vals_prefix}_keys.npy'
        vals_filename = f'{keys_vals_prefix}_vals.npy'
        if os.path.exists(keys_filename) and os.path.exists(vals_filename):
            mode = 'r'
        else:
            mode = 'w+'
            Path(keys_filename).parent.mkdir(parents=True, exist_ok=True)
        self.dstore_keys = np.memmap(keys_filename, dtype=np.float16, mode=mode, shape=(self.dstore_size, self.dimension))
        self.dstore_vals = np.memmap(vals_filename, dtype=np.int32, mode=mode, shape=(self.dstore_size, 1))

    def pre_forward_hook(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if labels is None:
            print("labels is none")
            labels=input_ids
            #raise ValueError('labels must be provided when saving a datastore. Are you using --predict_with_generate by mistake? If so, disable it')
        self.labels = labels
        return self.original_forward_func(input_ids=input_ids, labels=labels, attention_mask=attention_mask, **kwargs)

    def post_forward_hook(self, module, input, output):
        shift = 0 if self.is_encoder_decoder else 1
        captured_keys = self.activation_capturer.captured
        if shift == 1:
            captured_keys = captured_keys[:, :-shift]
        captured_keys = captured_keys.flatten(0, 1) # (batch * time, dim)
        captured_values = self.labels[:, shift:].flatten(0, 1) # (batch * time)

        nonpad_mask = captured_values != -100
        keys = captured_keys[nonpad_mask]
        values = captured_values[nonpad_mask]
 
        batch_time_size = keys.shape[0]
        # if shape[0] == args.tokens_per_sample:
        if self.dstore_idx + batch_time_size > self.dstore_size:
            batch_time_size = max(self.dstore_size - self.dstore_idx, 0)
            keys = keys[:batch_time_size]
            values = values[:batch_time_size]
        try:
            self.dstore_keys[self.dstore_idx:(batch_time_size + self.dstore_idx)] = keys.cpu().numpy().astype(np.float16)
            self.dstore_vals[self.dstore_idx:(batch_time_size + self.dstore_idx)] = values.unsqueeze(-1).cpu().numpy().astype(np.int32)
        except ValueError as ex:
            logger.error(f'Error saving datastore with mode {self.dstore_keys.mode}, did you try to save an already existing datastore?')
            logger.error(f'Delete the files {self.dstore_keys.filename} and {self.dstore_vals.filename} and try again')
            raise ex

        self.dstore_idx += batch_time_size
 
        return output

    def register_hook(self, layer, func, pre=False):
        handle = layer.register_forward_pre_hook(func) if pre else layer.register_forward_hook(func)
        self.hook_handles.append(handle)
    
    def break_out(self):
        for h in self.hook_handles:
            h.remove()
        if self.model is not None and self.model.broken_into is not None:
            self.model.forward = self.original_forward_func
            self.model.broken_into = None

    def build_index(self, num_keys_to_add_at_a_time=1000000, 
            ncentroids=4096, seed=1, code_size=64, probe=32):
        logger.info('Building index')
        index_name = get_index_path(self.dstore_dir, self.model.config.model_type, self.dstore_size, self.dimension) 

        # Initialize faiss index
        quantizer = faiss.IndexFlatL2(self.dimension)
        index = faiss.IndexIVFPQ(quantizer, self.dimension,
            ncentroids, code_size, 8)
        index.nprobe = probe

        np.random.seed(seed)
        index = torch.load('index.pt')
        # index = faiss.read_index(, faiss.IO_FLAG_ONDISK_SAME_DIR)
        # random_sample = np.random.choice(np.arange(self.dstore_vals.shape[0]), size=[min(1000000, self.dstore_vals.shape[0])], replace=False)
        # random_sample = np.sort(random_sample)
        # logger.info('Training Index')
        # start = time.time()
        # index.train(self.dstore_keys[random_sample])
        # torch.save(index, 'index.pt')
        # logger.info(f'Training took {time.time() - start} s')

        logger.info('Adding Keys')
        interval = 160000000
        for i in range(480000000, len(self.dstore_keys), interval):
            start_time = time.time()
            index_copy = faiss.clone_index(index)
            start, end = i, min(len(self.dstore_keys), i+interval)
            curr_index_name = index_name.replace('.indexed', f'_{start}_{end}.indexed')
            curr_len = end - start
            for j in range(0, curr_len, num_keys_to_add_at_a_time):
            #for j in range(index_copy.ntotal, curr_len, num_keys_to_add_at_a_time):
                start_j, end_j = j, min(curr_len, j+num_keys_to_add_at_a_time)
                logger.info(f'Adding Index {start_j}')
                to_add = self.dstore_keys[start+start_j:start+end_j].copy()
                index_copy.add_with_ids(torch.tensor(to_add.astype(np.float32)), torch.arange(start_j, end_j))

                logger.info(f'Writing Index {start_j}')
                faiss.write_index(index_copy, curr_index_name)
            print(f'Writing index took {time.time() - start_time} s')
            print("current ntotal:", index_copy.ntotal)
            index_copy.reset()

    def get_metrics(self):
        return {}

class ActivationCapturer(nn.Module):
    def __init__(self, layer, capture_input=False):
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input

        self.captured = None

    
    def forward(self, module, input, output):
        if self.capture_input:
            self.captured = input[0].detach()
        else:
            self.captured = output.detach()


def get_dstore_path(dstore_dir, model_type, dstore_size, dimension):
    return f'{dstore_dir}/dstore_{model_type}_{dstore_size}_{dimension}'

def get_index_path(dstore_dir, model_type, dstore_size, dimension):
    return f'{dstore_dir}/index_{model_type}_{dstore_size}_{dimension}.indexed'