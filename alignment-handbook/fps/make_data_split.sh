eval "$(conda shell.bash hook)"

infer_model=model/zephyr-7b-sft-full
trained_dataset_path=datasets/argilla/ultrafeedback-binarized-preferences-cleaned
hidden_output_dir=fps/save_hidden/zephyr
save_dir=datasets

## calculate embeddings and distances
conda activate spa
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=12344 --nproc_per_node=4 run_hidden_ddp.py --model_name_or_path ${infer_model}  --dataset_name_or_path ${trained_dataset_path}  --output_dir ${hidden_output_dir} &
wait
sleep 30

## split datasets
python make_data_split.py --dataset_name_or_path ${trained_dataset_path}  --hidden_path ${hidden_output_dir}/matrix_M.pt --save_dir ${save_dir} &
wait
sleep 30