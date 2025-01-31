# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import os
from typing import List, Literal, Optional

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError

from .configs import DataArguments

sys = ""#"You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\nFocus on essential information and keep explanations brief to ensure clarity and conciseness."
DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template:
        messages.insert(0, {"role": "system", "content": sys})

def insert_system_message(messages, sys , tokenizer):
    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template
    # confirm the jinja template refers to a system message before inserting
    #if "system" in chat_template:
    messages.insert(0, {"role": "system", "content": sys})

def is_openai_format(messages) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False


def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
    prompt_templete=None,
    auto_insert_empty_system_msg=True,
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True if task == "generation" else False
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            maybe_insert_system_message(chosen_messages, tokenizer)
            maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages  = example["chosen"][:-1]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
                
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]
            if "sys_prompt" in example :
                system = example["sys_prompt"][0]
                insert_system_message(prompt_messages, system,tokenizer)
            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)
                #maybe_insert_system_message(chosen_messages, tokenizer)
                #maybe_insert_system_message(rejected_messages, tokenizer)
            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            if "prompt" in example:
                example["original_prompt"] =  example["prompt"]
            ##### ATTENTION: these code below is essential for llama and any other model which has bos_token #####
            if example["text_chosen"].startswith(tokenizer.bos_token):
                example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):]
            if example["text_rejected"].startswith(tokenizer.bos_token):
                example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):]

    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of {['sft', 'generation', 'rm', 'dpo']}"
        )
    return example


def get_datasets(
    data_config: DataArguments | dict,
    splits: List[str] = ["train", "test"],
    shuffle: bool = True,
) -> DatasetDict:
    """
    Loads one or more datasets with varying training set proportions.

    Args:
        data_config (`DataArguments` or `dict`):
            Dataset configuration and split proportions.
        splits (`List[str]`, *optional*, defaults to `['train', 'test']`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.

    Returns
        [`DatasetDict`]: The dataset dictionary containing the loaded datasets.
    """

    if type(data_config) is DataArguments:
        # Structure of the config to read the datasets and their mix
        # datasets_mixer:
        #     - 'dataset1': 0.5
        #     - 'dataset2': 0.3
        #     - 'dataset3': 0.2
        if type(data_config.dataset_mixer) is str :
            dataset_mixer = {data_config.dataset_mixer: 1}
        else :
            dataset_mixer = data_config.dataset_mixer
    elif isinstance(data_config, dict):
        # Structure of the input is:
        #     dataset_mixer = {
        #             "dataset1": 0.5,
        #             "dataset1": 0.3,
        #             "dataset1": 0.2,
        #         }
        dataset_mixer = data_config

    else:
        raise ValueError(f"Data config {data_config} not recognized.")

    raw_datasets = mix_datasets(dataset_mixer, splits=splits, shuffle=shuffle)
    return raw_datasets


def mix_datasets(dataset_mixer: dict, splits: Optional[List[str]] = None, shuffle=True) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions. By default, all test proportions are 1.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in all datasets and have a `train_` or `test_` prefix.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
    """
    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    fracs = []
    for ds, frac in dataset_mixer.items():
        fracs.append(frac)
        for split in splits:
            #try:
                # Try first if dataset on a Hub repo
            #    dataset = load_dataset(ds, split=split)
            #except DatasetGenerationError:
                # If not, check local dataset
            dataset = load_from_disk(os.path.join(ds, split))

            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if any(frac < 0 for frac in fracs):
        raise ValueError("Dataset fractions cannot be negative.")

    if len(raw_train_datasets) > 0:
        train_subsets = []
        for dataset, frac in zip(raw_train_datasets, fracs):
            train_subset = dataset.select(range(int(frac * len(dataset))))
            train_subsets.append(train_subset)
        if shuffle:
            raw_datasets["train"] = concatenate_datasets(train_subsets).shuffle(seed=42)
        else:
            raw_datasets["train"] = concatenate_datasets(train_subsets)
    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        if shuffle:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets).shuffle(seed=42)
        else:
            raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with split {split}. Check the dataset has been correctly formatted."
        )

    return raw_datasets
