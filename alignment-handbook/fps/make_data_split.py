from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
import datasets
import random
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import argparse
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    set_seed
)
import os
from tqdm import tqdm
os.chdir('../')

@dataclass
class ScriptArguments:
    dataset_name_or_path: Optional[str] = field(
        default="argilla/ultrafeedback-binarized-preferences-cleaned",
        metadata={"help": "the location of the dataset name or path"},
    )
    hidden_path: Optional[str] = field(
        default="save_hidden/phi2/matrix_M.pt",
        metadata={"help": "the path to the saved matrix M"},
    )
    save_dir: Optional[str] = field(
        default="datasets/fps_phi2",
        metadata={"help": "the path to the saved datasets(spa_0, spa_1, spa_2 and spa_3)"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    mode: Optional[Literal['random', 'fps']] = field(
        default='fps',
        metadata={"help": "how to select seed data"},
    )
    

def fps(D: torch.Tensor, n: int):
    assert len(list(D.shape)) == 2
    assert D.shape[0] == D.shape[1]
    # assert torch.all(D == D.T)
    
    total = D.shape[0]
    first = random.randint(0, total-1)
    selected = [first]
    remain = list(range(total))
    remain = remain[:first] + remain[first+1:]
    
    for _ in tqdm(range(n-1), desc="Running farthest point sampling"):
        dis = torch.index_select(D, 0, torch.tensor(remain).to(D.device))
        dis = torch.index_select(dis, 1, torch.tensor(selected).to(dis.device))
        cur = remain[dis.sum(dim=-1).argmax()]
        selected.append(cur)
        remain = set(remain)
        remain.remove(cur)
        remain = list(remain)
        
    assert len(selected) + len(remain) == total
    assert list(set(selected + remain)) == list(range(total))
    return selected, remain
    
    
parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

original_data = datasets.load_from_disk(script_args.dataset_name_or_path)
def add_index(example, idx):
    example['index'] = idx
    return example
original_data = original_data.map(add_index, with_indices=True)

if script_args.mode == 'random':
    suffle = original_data['train'].shuffle(seed=script_args.seed)
    train_data_2K = suffle[:2000]
    train_data_8K = suffle[2000:10000]
    train_data_20K = suffle[10000:30000]
    train_data_30K = suffle[30000:60000]
    test_data = suffle[60000:]
    train_data_2K = Dataset.from_dict(train_data_2K)
    
elif script_args.mode == 'fps':
    M = torch.load(script_args.hidden_path)
    assert M.shape[0] == len(original_data['train'])
    selected, remain = fps(M, 2000)
    train_data_2K = original_data['train'].select(selected)
    suffle = original_data['train'].select(remain).shuffle(seed=script_args.seed)
    train_data_8K = suffle[:8000]
    train_data_20K = suffle[8000:28000]
    train_data_30K = suffle[28000:58000]
    test_data = suffle[58000:]
    
else:
    raise NotImplementedError()

train_data_8K = Dataset.from_dict(train_data_8K)
train_data_20K = Dataset.from_dict(train_data_20K)
train_data_30K = Dataset.from_dict(train_data_30K)
test_data = Dataset.from_dict(test_data)
new_dataset_2K = DatasetDict({
    "train": train_data_2K,
    "test": test_data,
})

new_dataset_8K = DatasetDict({
    "train": train_data_8K,
    "test": test_data,
})

new_dataset_20K = DatasetDict({
    "train": train_data_20K,
    "test": test_data,
})
new_dataset_30K = DatasetDict({
    "train": train_data_30K,
    "test": test_data,
})

os.makedirs(script_args.save_dir, exist_ok=True)

save_path = os.path.join(script_args.save_dir, "spa_0")
new_dataset_2K.save_to_disk(save_path)

save_path = os.path.join(script_args.save_dir, "spa_1")
new_dataset_8K.save_to_disk(save_path)

save_path = os.path.join(script_args.save_dir, "spa_2")
new_dataset_20K.save_to_disk(save_path)

save_path = os.path.join(script_args.save_dir, "spa_3")
new_dataset_30K.save_to_disk(save_path)
