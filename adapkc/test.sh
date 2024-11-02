CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
    --master_port 8120 \
    --nproc_per_node=1 \
    --use_env test.py \
    --cfg config_files/adapkc_theta.json \
    --model-path /home/logs/carrada/adapkc_theta/name_of_the_model/results/model.pt \
    >test_adapkc_theta.log 2>&1