import os
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk, concatenate_datasets
from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.utils.data.distributed import DistributedSampler
from distributed_sequential_sampler import DistributedSequentialSampler
import torch.distributed as dist
from tqdm import tqdm
from transformers.hf_argparser import HfArgumentParser
import argparse  # parse --local-rank arguments
import sys
import random

def maybe_insert_system_message(messages):
    # confirm the jinja template refers to a system message before inserting
    messages.insert(0, {"role": "system", "content": ""})
    return messages

    
@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(
        default="save_model/zephyr-2K",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_name_or_path: Optional[str] = field(
        default="datasets/spa_0",
        metadata={"help": "the location of the dataset name or path"},
    )
    output_dir: Optional[str] = field(
        default="save_hidden",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    tp: Optional[int] = field(
        default=4,
        metadata={"help": "tensor_parallel_size"},
    )
    max_input_length: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the input tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    eos_ids: List[int] = field(default_factory=lambda: [], metadata={"help": "the ids of the end of sentence tokens"})


def main():
    # **Step 1: 使用 argparse 解析 torch.distributed.launch 传递的 `--local-rank` 参数**
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank passed by torch.distributed.launch")
    known_args, remaining_args = parser.parse_known_args()

    local_rank = known_args.local_rank  # 获取 local rank
    if local_rank != -1:
        torch.cuda.set_device(local_rank)  # 设置当前 GPU 设备

    # **Step 2: 使用 HfArgumentParser 解析剩余参数**
    hf_parser = HfArgumentParser(ScriptArguments)
    script_args = hf_parser.parse_args_into_dataclasses(args=remaining_args)[0]

    # **Step 3: 初始化分布式训练**
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # 设置随机种子
    seed = script_args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 加载模型和分词器
    model_path = script_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=None,
    )
    model = model.to(local_rank)  # 将模型加载到当前 GPU
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)  # 使用 DDP 包装模型

    # 加载数据集
    print("Start loading dataset")
    ds = load_from_disk(script_args.dataset_name_or_path)['train']

    # 如果数据集没有索引，则为其添加索引
    if "index" not in ds.features:
        def add_index_to_dataset(dataset):
            def add_index(example, idx):
                example["index"] = idx
                return example
            return dataset.map(add_index, with_indices=True)

        ds = add_index_to_dataset(ds)

    # 处理数据集的 prompt
    ds = ds.map(
        lambda x: {
            "prompt": tokenizer.apply_chat_template(
                maybe_insert_system_message(x["chosen"][:-1]), tokenize=False, add_generation_prompt=True
            )
        }
    )
    prompts = ds["prompt"]
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=script_args.max_input_length,
        return_tensors="pt"
    )

    # 分布式采样器和数据加载器
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"], torch.tensor(ds['index'], dtype=torch.float))
    sampler = DistributedSequentialSampler(dataset, shuffle=False, drop_last=False)  # 按 GPU 划分数据. 注意使用保顺序的采样器 DistributedSequentialSampler。
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    # 收集隐藏层状态
    last_token_hidden_states = []
    index = []
    print("Start generating")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            batch_input_ids, batch_attention_mask, batch_idx = batch
            batch_input_ids = batch_input_ids.to(local_rank)
            batch_attention_mask = batch_attention_mask.to(local_rank)

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=True,
            )

            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                last_layer_hidden_states = outputs.hidden_states[-1]  # 获取最后一层隐藏状态
                last_token_hidden_state = last_layer_hidden_states[:, -1, :]  # 提取最后一个 token 的隐藏状态
                last_token_hidden_states.append(last_token_hidden_state.detach().cpu())
                index.append(batch_idx.detach().cpu())
            else:
                print(f"No hidden states returned for batch on device {local_rank}.")

    # 从所有进程收集隐藏层状态
    if last_token_hidden_states:
        # 确保 `final_hidden_states` 在 GPU 上
        final_hidden_states = torch.cat(last_token_hidden_states, dim=0).to(local_rank)
        final_index = torch.cat(index, dim=0).to(local_rank)
        
        # 创建在 GPU 上的空张量列表，用于收集所有进程的结果
        gathered_hidden_states = [torch.zeros_like(final_hidden_states, device=local_rank) for _ in range(world_size)]
        gathered_index  = [torch.zeros_like(final_index , device=local_rank) for _ in range(world_size)]
        
        # 使用 `dist.all_gather` 在 GPU 上收集张量
        dist.all_gather(gathered_hidden_states, final_hidden_states)
        dist.all_gather(gathered_index, final_index)
        
        # 将结果拼接起来
        gathered_hidden_states = torch.cat(gathered_hidden_states, dim=0)[:len(ds)] # 注意删除末尾的padding data
        gathered_index = torch.cat(gathered_index, dim=0)[:len(ds)] # 注意删除末尾的padding data
        # 检查顺序，以防意外
        assert torch.all(gathered_index == torch.tensor(ds['index'], dtype=torch.float).to(local_rank))
        
    else:
        gathered_hidden_states = None

    print("start saving")
    del model
    torch.cuda.empty_cache()
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 仅在 rank 0 上保存结果
    if rank == 0:
        save_dir = script_args.output_dir
        os.makedirs(save_dir, exist_ok=True)

        e = gathered_hidden_states.to(local_rank)
        print(f"Shape of e: {e.shape}")  # 打印形状
        print(f"Device of e: {e.device}")  # 打印设备
        print(f"Data type of e: {e.dtype}")  # 打印数据类型
        e_float32 = e.float()
        # 计算余弦相似度矩阵, size=(data_num, data_num)
        M_float32 = torch.matmul(e, e.T)  # (data_num, data_num)
        norm = e.pow(2).sum(dim=-1).sqrt().unsqueeze(-1) # (data_num, 1)
        M_float32 /= torch.matmul(norm, norm.T) # (data_num, data_num)
        M_float32 = 1 - M_float32 # (data_num, data_num)
        M_float16 = M_float32.half()
        M_float16 = M_float32

        print("Matrix M shape:", M_float16.shape)

        hidden_states_path = os.path.join(save_dir, "hidden_states.pt")
        torch.save(gathered_hidden_states, hidden_states_path)
        print(f"Hidden states saved to {hidden_states_path}")

        matrix_M_path = os.path.join(save_dir, "matrix_M.pt")
        torch.save(M_float16, matrix_M_path)
        print(f"Matrix M saved to {matrix_M_path}")

    # 清理分布式进程组
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
