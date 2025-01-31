eval "$(conda shell.bash hook)"

Seed=42

###### initial DPO training ########
conda activate spa
CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_full_initial.yaml --seed=${Seed} &
wait
sleep 30

####### loop start ########
base_model=zephyr-2K
infer_model=save_model/zephyr-2K

for iteration in 1 2 3 
do
    prompt_dir=datasets/spa_${iteration}
    sample_output_dir=datasets/sample-${base_model}-spa_${iteration}
    judge_output_dir=datasets/training-${base_model}-spa_${iteration}
    final_model_path=save_model/${base_model}-spa_${iteration}_task2
    train_output_dir=datasets/add-args-${base_model}-spa_${iteration}
    
    if [ ${iteration} -eq 1 ]; then
        infer_model=save_model/zephyr-2K
        echo "Setting infer_model for iteration 1: $infer_model"
    fi
    
    if [ ${iteration} -eq 2 ]; then
        infer_model=save_model/zephyr-2K-spa_1
        echo "Setting infer_model for iteration 2: $infer_model"
    fi
    
    if [ ${iteration} -eq 3 ]; then
        infer_model=save_model/zephyr-2K-spa_2
        echo "Setting infer_model for iteration 3: $infer_model"
    fi
    
    wait
    
    conda deactivate 
    conda activate vllm
    CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/online_generation.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${sample_output_dir} &
    wait
    sleep 30
    
    conda deactivate 
    conda activate spa
    CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_self.py recipes/zephyr-7b-beta/dpo/config_full.yaml --dataset_mixer=${sample_output_dir}  --model_name_or_path=${infer_model} --output_dir=${final_model_path} --save_confidence_name=${judge_output_dir} --seed=${Seed} &
    
    wait
    sleep 30
    python scripts/make_training_samples.py --dataset_mixer=${sample_output_dir} --save_confidence_name=${judge_output_dir} &
    
    wait
    sleep 30
    python scripts/run_wst.py --input_dir=${judge_output_dir} --output_dir=${train_output_dir} --wst_epsilon 0.05 --fit_K 6
    
    wait
    sleep 30
    CUDA_VISIBLE_DEVICES=4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml scripts/run_dpo_upd.py recipes/zephyr-7b-beta/dpo/config_full_upd.yaml --dataset_mixer=${train_output_dir}  --model_name_or_path=${infer_model} --output_dir=${final_model_path} --seed=${Seed} &
    wait
    sleep 30
    
done