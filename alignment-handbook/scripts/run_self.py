#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import logging
import random
import sys
import random
import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

##### ATTENTION: modify 'alignment' to 'src.alignment', to redirect the import to src/alignment #####
from src.alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel
from trl import DPOTrainer
#import wandb
import tqdm
import datasets
datasets.disable_caching()
from datasets import Dataset

logger = logging.getLogger(__name__)


import threading

class ThreadSafeDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def __delitem__(self, key):
        with self._lock:
            super().__delitem__(key)

    def __contains__(self, key):
        with self._lock:
            return super().__contains__(key)

    def get(self, key, default=None):
        with self._lock:
            return super().get(key, default)

    def setdefault(self, key, default=None):
        with self._lock:
            return super().setdefault(key, default)

    def pop(self, key, default=None):
        with self._lock:
            return super().pop(key, default)

    def popitem(self):
        with self._lock:
            return super().popitem()

    def clear(self):
        with self._lock:
            super().clear()

    def update(self, *args, **kwargs):
        with self._lock:
            super().update(*args, **kwargs)

    def keys(self):
        with self._lock:
            return super().keys()

    def values(self):
        with self._lock:
            return super().values()

    def items(self):
        with self._lock:
            return super().items()

from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    save_confidence_name: Optional[str] = field(
        default="save_data",
        metadata={"help": "the location of the SFT model name or path"},
    )


def prepare_pairwise_data(dataset):
    """Convert (index, prompt, response) format to (index, prompt, chosen, rejected) pairs"""
    processed_data = []
    
    # Group by prompt to get all responses for each prompt
    grouped_data = {}
    for item in dataset:
        if item['index'] not in grouped_data:
            grouped_data[item['index']] = []
        grouped_data[item['index']] = {
            'prompt': item['prompt'],
            'response': item['response']
        }
    
    # Create pairs for each group
    for index, responses in grouped_data.items():
        n_responses = len(responses['response'])
        # For even number of responses
        for i in range(0, n_responses - 1, 2):
            processed_data.append({
                'index': f"{i}_{i+1}_{index}",
                'prompt': responses['prompt'],
                'chosen':   [{"content":responses['prompt'],"role":"user"},{"content":responses['response'][i],"role":"assistant"}],
                'rejected': [{"content":responses['prompt'],"role":"user"},{"content":responses['response'][i+1],"role":"assistant"}],
            })
            
        # For odd number of responses, use the last response twice if needed
        if n_responses % 2 == 1:
            processed_data.append({
                'index': f"-1_{0}_{index}",
                'prompt': responses['prompt'],
                'chosen':   [{"content":responses['prompt'],"role":"user"},{"content":responses['response'][-1],"role":"assistant"}],
                'rejected': [{"content":responses['prompt'],"role":"user"},{"content":responses['response'][0],"role":"assistant"}],
            })

    return Dataset.from_dict({
        'index': [item['index'] for item in processed_data],
        'prompt': [item['prompt'] for item in processed_data],
        'chosen': [item['chosen'] for item in processed_data],
        'rejected': [item['rejected'] for item in processed_data]
    })

def process_evaluation_results(raw_datasets, save_confidence, strategy="min_max"):
    """Process evaluation results to get final rankings
    
    Args:
        raw_datasets: Dataset containing original indices and lists of responses
        save_confidence: Dictionary mapping comparison indices to (chosen_reward, rejected_reward) pairs
        
    Returns:
        Dataset containing best and worst responses for each index
    """
    # Initialize response rewards mapping for each index
    index_response_rewards = {}  # {index: {response_idx: reward}}
    
    # Initialize rewards for all responses
    for idx in raw_datasets['index']:
        index_response_rewards[idx] = {}
    
    # Process comparison results
    for comp_index, rewards in save_confidence.items():
        chosen_idx, rejected_idx, original_idx = comp_index.split('_')
        chosen_reward, rejected_reward = rewards
        index_response_rewards[int(original_idx)][int(chosen_idx)] = float(chosen_reward)
        index_response_rewards[int(original_idx)][int(rejected_idx)] = float(rejected_reward)
    
    # Create final dataset
    final_data = []
    for idx in tqdm.tqdm(index_response_rewards.keys()):
        if len(index_response_rewards[idx]) == 0:  # Skip if no rewards recorded
            continue
        dataset_idx = raw_datasets['index'].index(idx)
        responses = raw_datasets['response'][dataset_idx]
        rewards = index_response_rewards[idx]
        
        # Create list of (response_idx, response, reward) tuples
        response_data = []
        for response_idx, response in enumerate(responses):
            str_idx = response_idx
            if str_idx in rewards:
                response_data.append((str_idx, response, rewards[str_idx]))
        
        if len(response_data) >= 2:
            # Sort by reward
            sorted_responses = sorted(response_data, key=lambda x: x[2], reverse=True)
            if strategy == 'min_max' :
                best = sorted_responses[0]
                worst = sorted_responses[-1]
            elif strategy == 'max_chosen' :
                best = sorted_responses[0]
                worst = random.choice(sorted_responses[1:])            
            final_data.append({
                'index': idx,  # Format: chosen_idx_rejected_idx_original_idx
                'prompt': raw_datasets['prompt'][dataset_idx],
                'chosen': [{"content":raw_datasets['prompt'][dataset_idx],"role":"user"},{"content": best[1],"role":"assistant"}],    # best[1] is the response text
                'rejected': [{"content":raw_datasets['prompt'][dataset_idx],"role":"user"},{"content": worst[1],"role":"assistant"}],   # worst[1] is the response text
                'chosen_reward': best[2],    # best[1] is the response text
                'rejected_reward': worst[2]  # worst[1] is the response text
            })
            print(idx,best[2], worst[2])
    
    return Dataset.from_dict({
        'index': [item['index'] for item in final_data],
        'prompt': [item['prompt'] for item in final_data],
        'chosen': [item['chosen'] for item in final_data],
        'rejected': [item['rejected'] for item in final_data],
        'chosen_reward': [item['chosen_reward'] for item in final_data],
        'rejected_reward': [item['rejected_reward'] for item in final_data]
    })

save_dataset = ThreadSafeDict()
def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig, ScriptArguments))
    model_args, data_args, training_args , scrips_args = parser.parse()
    if type(data_args.dataset_mixer) is str :
        data_args.dataset_mixer = {data_args.dataset_mixer: 1}
    #wandb.init(project='DPO-iteration',name=training_args.output_dir)
    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    print("data_args : ", data_args)
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    column_names.remove("index")

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)
    initial_training_raw = raw_datasets['train']
    initial_testing_raw = raw_datasets['test']
    print("raw_datasets", raw_datasets)
    
    #####################
    # Apply chat template
    #####################
    training_raw = prepare_pairwise_data(initial_training_raw)
    print(training_raw)
    raw_datasets = training_raw.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "dpo"},
        num_proc=data_args.preprocessing_num_workers,
    #    remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
        ##### ATTENTION: disable cache, for in-time dataset updating #####
        keep_in_memory=True
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train"]:
        raw_datasets= raw_datasets.rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )
        
    print("raw_datasets", raw_datasets)
    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets)), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets[index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets[index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets[index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path

    ref_model = training_args.ref_model_for_refine
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None
        
    
    from datasets import load_dataset, Dataset, DatasetDict
    #########################
    # Instantiate DPO trainer
    #########################
    save_confidence_name = scrips_args.save_confidence_name.split("/")[1] + ".json" 
    print("save_condience_name", save_confidence_name)
    trainer = DPOTrainer(
        model,
        ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets,
        eval_dataset=raw_datasets,
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        peft_config=get_peft_config(model_args),
        loss_type=training_args.loss_type,
        save_dataset = save_dataset,
        save_confidence_name = save_confidence_name,
        get_confidence=True
        #precompute_ref_log_probs=True
    )
    
    print("before start evalutation")


    ##########
    # Evaluate
    ##########

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(raw_datasets)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    
    ##################################
    # Save model and create model card
    ##################################    
    # import json
    # with open('save_confidence/' + save_confidence_name, 'r') as json_file:
    #     confidence = json.load(json_file)
    # from datasets import load_dataset, Dataset, DatasetDict
    # save_path = "datasets/"+save_confidence_name.rstrip(".json") 
    # final_dataset = process_evaluation_results(initial_training_raw, confidence)
    # train_dataset = DatasetDict({
    #     "train": final_dataset,
    #     "test": initial_testing_raw,
    # })
    # train_dataset.save_to_disk(save_path)


if __name__ == "__main__":
    main()