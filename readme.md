# Stackelberg Self-Annotated Preference Optimization (SSAPO)

## Condition

- CUDA: 12.1.1
- PyTorch: 2.1.2

## Setup Script

1. **Create and activate the conda environment**:

   ```bash
   conda create -n vllm python=3.10
   conda activate vllm
   pip install vllm datasets
   conda deactivate

   conda create -n spa python=3.10
   conda activate spa
   ```
2. **Install CUDA toolkit (if necessary)**:

   ```bash
   conda install nvidia/label/cuda-12.1.1::cuda-toolkit
   ```
3. **Install PyTorch and related packages**:

   ```bash
   conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
   ```
4. **Install additional required packages**:

   ```bash
   pip install -e trl/.
   pip install -e alignment-handbook/.
   pip install transformers==4.36.2
   pip install numpy==1.26.4
   pip install torchmetrics
   pip install huggingface-hub==0.24.7
   ```
5. **Navigate to the alignment-handbook directory**:

   ```bash
   cd alignment-handbook
   ```

## Data Preparation

1. **Split the data**:
   ```bash
   # 1. get seed data by random sampling
   python make_data_split.py
   # 2. get seed data through fps
   bash fps/make_data_split.sh
   ```

## Model Training

```bash
bash scripts/run_train.sh
```

## Notes

- The current setup is based on the **zephyr-7b-beta** model. Sampling code may need to be adjusted for different models or chat templates.
- Models trained to align with the current setup include:

  - `princeton-nlp/Llama-3-Base-8B-SFT`
  - `alignment-handbook/zephyr-7b-sft-full`
- For initial DPO training:

  - Using **3 epochs** is effective for datasets with fewer than 2,000 samples.
  - For larger datasets, even if initial performance is lower, **1 epoch** training has been observed to yield better trends.
- Hyperparameters can be customized using the configuration files:

  - `alignment-handbook/recipes/zephyr-7b-beta/dpo/config_full_initial.yaml`
  - `alignment-handbook/recipes/zephyr-7b-beta/dpo/config_full_upd.yaml`
  - `alignment-handbook/recipes/llama3-8b-instruct/dpo/config_full_initial.yaml`
  - `alignment-handbook/recipes/llama3-8b-instruct/dpo/config_full_upd.yaml`
