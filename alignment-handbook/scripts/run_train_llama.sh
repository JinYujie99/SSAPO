eval "$(conda shell.bash hook)"

####### initial DPO training ########
conda activate spa
CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --main_process_port 29501 scripts/run_dpo.py recipes/llama3-8b-instruct/dpo/config_full_initial.yaml &
wait
sleep 30

####### loop start ########
base_model=llama3-2K
infer_model=save_model/llama3-2K

##### ATTENTION: according to the paper, experiments conducted on llama3-8b-instruct use 2 iterations instead of 3 #####
for iteration in 1 2
do
    prompt_dir=datasets/spa_${iteration}
    sample_output_dir=datasets/sample-${base_model}-spa_${iteration}
    judge_output_dir=datasets/training-${base_model}-spa_${iteration}
    final_model_path=save_model/${base_model}-spa_${iteration}
    train_output_dir=datasets/add-args-${base_model}-spa_${iteration}
    
    if [ ${iteration} -ne 1 ]; then
        conda deactivate 
        conda activate vllm
        CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/online_generation.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${sample_output_dir} &

        wait
        sleep 30
        conda deactivate 
        conda activate spa
        CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --main_process_port 29501 scripts/run_self.py recipes/llama3-8b-instruct/dpo/config_full.yaml --dataset_mixer=${sample_output_dir}  --model_name_or_path=${infer_model} --output_dir=${final_model_path} --save_confidence_name=${judge_output_dir} &
    
        wait
        sleep 30
        python scripts/make_training_samples.py --dataset_mixer=${sample_output_dir} --save_confidence_name=${judge_output_dir} &
    
        wait
        sleep 30
        python scripts/run_wst.py --input_dir=${judge_output_dir} --output_dir=${train_output_dir} --wst_epsilon 0.01 --fit_K 6
    
        wait
        sleep 30
    fi
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/deepspeed_zero3.yaml --main_process_port 29501 scripts/run_dpo_upd.py recipes/llama3-8b-instruct/dpo/config_full_upd.yaml --dataset_mixer=${train_output_dir}  --model_name_or_path=${infer_model} --output_dir=${final_model_path} &
    wait
    sleep 30
    
    # ######## Update infer_model for the next iteration ########
    if [ ${iteration} -ne 10 ]; then
        infer_model=${final_model_path}
    fi
done