CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
    --master_port 8119 \
    --nproc_per_node=2 \
    --use_env train.py \
    --sync-bn \
    --cfg config_files/adapkc_theta.json \
    >train_adapkc_theta.log 2>&1