#!/usr/bin/env python
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
import json

def maybe_insert_system_message(messages):
    # confirm the jinja template refers to a system message before inserting
    messages.insert(0, {"role": "system", "content": ""})
    return messages


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "the location of the dataset name or path"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=2,
        metadata={"help": "the number of generations per prompt"},
    )
    tp: Optional[int] = field(
        default=4,
        metadata={"help": "tensor_parallel_size"},
    )
    max_input_length: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=300,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="context_messages",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

model_path = script_args.model_name_or_path
print("model_path", model_path)
seed = script_args.seed
# set seed
torch.manual_seed(seed)
np.random.seed(seed)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.truncation_side = "left"
tokenizer.padding_side  = 'left'
sampling_params = SamplingParams(
    temperature=script_args.temperature,
    top_p=1.0,
    max_tokens=script_args.max_new_tokens,
    n=script_args.K,
    stop_token_ids=[tokenizer.eos_token_id] + script_args.eos_ids,
    
    #stop=["<|user|>"],
)

if script_args.dataset_name_or_path.split("/")[0] == 'vangard703' :
    ds = load_dataset(script_args.dataset_name_or_path, split="train")
    ds_test = load_dataset(script_args.dataset_name_or_path, split="test")
else :
    # print(f"load from disk : {script_args.dataset_name_or_path}")
    # import os
    # print(os.getcwd())
    ds = load_from_disk(script_args.dataset_name_or_path)['train']
    ds_test = load_from_disk(script_args.dataset_name_or_path)['test']

from datasets import Dataset, DatasetDict
def add_index_to_dataset(dataset: Dataset) -> Dataset:
    def add_index(example, idx):
        example['index'] = idx
        return example
    
    return dataset.map(add_index, with_indices=True)

if not "index" in ds.features :
    ds = add_index_to_dataset(ds)

original_prompt = ds['prompt']
index = ds['index']
sys_prompt = None
print(ds)

##### attention: do NOT insert system message for llama3-8b-instruct. #####
ds = ds.map(
lambda x: {
    "prompt": tokenizer.apply_chat_template(maybe_insert_system_message(x["chosen"][:-1]), tokenize=False, add_generation_prompt=True)
    if 'llama' not in model_path
    else tokenizer.apply_chat_template(x["chosen"][:-1], tokenize=False, add_generation_prompt=True)
}
)
prompt_chat_tem = ds.map(
    lambda x: 
        {"prompt_chat_tem": x["chosen"][:-1]})
    
data_size = len(ds["prompt"])
prompts = ds["prompt"]

llm = LLM(
    model=model_path,
    tokenizer=model_path,
    dtype="bfloat16",
    max_model_len=script_args.max_input_length,
    load_format="auto",
    seed=42,
    tensor_parallel_size=script_args.tp
)
outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)


completions = []
used_prompts = []
new_dataset = {}
new_dataset['prompt'] = []
new_dataset['response'] = []
new_dataset['original_prompt'] = []
new_dataset['index'] = []

for i, output in enumerate(outputs):
    tmp_data = {"prompt": prompts[i], "response": [out.text for out in output.outputs] ,"original_prompt" :original_prompt[i]}
    new_dataset['prompt'].append(original_prompt[i])
    new_dataset['response'].append([out.text for out in output.outputs])
    new_dataset['original_prompt'].append(original_prompt[i])
    new_dataset['index'].append(index[i])

# print(new_dataset)

from datasets import Dataset,DatasetDict
new_dataset = Dataset.from_dict(new_dataset)

def del_base_len(sample) :
    return len(sample['response']) > (script_args.K -1)

new_dataset = new_dataset.filter(del_base_len)
save_dataset = DatasetDict({
    "train": new_dataset,
    "test": ds_test
})
# print(save_dataset)

save_path = script_args.output_dir
save_dataset.save_to_disk(save_path)