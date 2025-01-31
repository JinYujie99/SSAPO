import logging
import random
import sys

##### ATTENTION: modify 'alignment' to 'src.alignment', to redirect the import to src/alignment #####
from alignment import H4ArgumentParser
import datasets
datasets.disable_caching()
from datasets import Dataset, DatasetDict
logger = logging.getLogger(__name__)
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class ScriptArguments:
    dataset_mixer: Optional[str] = field(
        default="dataset",
        metadata={"help": "the location of the dataset"},
    )
    save_confidence_name: Optional[str] = field(
        default="save_data",
        metadata={"help": "the location of the SFT model name or path"},
    )

from collections import defaultdict
import random
from multiprocessing import Pool
import time

global_raw_datasets = None
global_index_response_rewards = None
global_strategy = None
global_index_map = None

def init_worker(raw_datasets, index_response_rewards, strategy, index_map):
    global global_raw_datasets, global_index_response_rewards, global_strategy, global_index_map
    global_raw_datasets = raw_datasets
    global_index_response_rewards = index_response_rewards
    global_strategy = strategy
    global_index_map = index_map

def process_single_index(idx):
    
    if len(global_index_response_rewards[idx]) == 0:
        return None
    
    dataset_idx = global_index_map[idx]
    responses = global_raw_datasets['response'][dataset_idx]
    rewards = global_index_response_rewards[idx]
    
    response_data = [(response_idx, response, rewards.get(response_idx, float('-inf')))
                     for response_idx, response in enumerate(responses)]
    
    if len(response_data) >= 2:
        sorted_responses = sorted(response_data, key=lambda x: x[2], reverse=True)
        best = sorted_responses[0]
        worst = sorted_responses[-1] if global_strategy == 'min_max' else random.choice(sorted_responses[1:])
        
        result = {
            'index': idx,
            'prompt': global_raw_datasets['prompt'][dataset_idx],
            'chosen': [{"content": global_raw_datasets['prompt'][dataset_idx], "role": "user"}, {"content": best[1], "role": "assistant"}],
            'rejected': [{"content": global_raw_datasets['prompt'][dataset_idx], "role": "user"}, {"content": worst[1], "role": "assistant"}],
            'chosen_reward': best[2],
            'rejected_reward': worst[2]
        }
        # logger.info(f"Processed dataset_idx {dataset_idx}: chosen={best[2]}, rejected={worst[2]}")
        return result
    else:
        return None

def process_evaluation_results(raw_datasets, save_confidence, strategy="min_max"):
    start_time = time.time()

    index_map = {idx: i for i, idx in enumerate(raw_datasets['index'])}
    index_response_rewards = defaultdict(dict)

    for comp_index, rewards in save_confidence.items():
        chosen_idx, rejected_idx, original_idx = comp_index.split('_')
        chosen_reward, rejected_reward = rewards
        index_response_rewards[int(original_idx)][int(chosen_idx)] = float(chosen_reward)
        index_response_rewards[int(original_idx)][int(rejected_idx)] = float(rejected_reward)
    
    # logger.info(f"Number of indices to process: {len(index_response_rewards.keys())}")

    with Pool(processes=64, initializer=init_worker, initargs=(raw_datasets, index_response_rewards, strategy, index_map)) as pool:
        results = []
        total_indices = len(index_response_rewards.keys())
        processed_count = 0
        
        for result in pool.imap_unordered(process_single_index, index_response_rewards.keys()):
            processed_count += 1
            if result is not None:
                results.append(result)
            if processed_count % 100 == 0:
                logger.info(f"Processed {processed_count}/{total_indices} items: Processed dataset_idx {result['index']}: chosen={result['chosen_reward']}, rejected={result['rejected_reward']}")

    final_data = [item for item in results if item is not None]
    elapsed_time = time.time() - start_time
    logger.info(f"Number of processed items: {len(final_data)}")
    logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
    
    return Dataset.from_dict({
        'index': [item['index'] for item in final_data],
        'prompt': [item['prompt'] for item in final_data],
        'chosen': [item['chosen'] for item in final_data],
        'rejected': [item['rejected'] for item in final_data],
        'chosen_reward': [item['chosen_reward'] for item in final_data],
        'rejected_reward': [item['rejected_reward'] for item in final_data]
    })

def main():
    parser = H4ArgumentParser((ScriptArguments))
    scrips_args = parser.parse()
    save_path = scrips_args.save_confidence_name
        
    #######
    # Setup
    #######
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    ###############
    # Load datasets
    ###############
    raw_datasets = datasets.load_from_disk(scrips_args.dataset_mixer)
    initial_training_raw = raw_datasets['train']
    initial_testing_raw = raw_datasets['test']
    
    save_confidence_name = scrips_args.save_confidence_name.split("/")[1] + ".json" 
     
    import json
    with open('save_confidence/' + save_confidence_name, 'r') as json_file:
        confidence = json.load(json_file)

    final_dataset = process_evaluation_results(initial_training_raw, confidence)
    train_dataset = DatasetDict({
        "train": final_dataset,
        "test": initial_testing_raw,
    })

    train_dataset.save_to_disk(save_path)
    logger.info(f"Dataset saved to disk at: {save_path}")


if __name__ == "__main__":
    main()