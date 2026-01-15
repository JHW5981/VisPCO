export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node=8 main.py \
    --model_name qwen2.5vl-3b \
    --trdata_path xxx/workspace/Qwen-Fastv-VisPCO/train_10000_uniform_area_single_image.json \
    --valdata_path xxx/workspace/Qwen-Fastv-VisPCO/eval_max_100_dataset.json \
    --kernel_type linear \
    --batch_size 2 \
    --num_workers 12 \
    --pin_memory \
    --seed 42 \
    --opt adamw \
    --lr 1e-4 \
    --min_lr 1e-6 \
    --weight_decay 0.0001 \
    --warmup_steps 0.001 \
    --epochs 1000 \
    --device cuda \
    --precision bf16 \
    --eval 1 \
    --update_freq 1 \
    --w 1 \
    --sigma 100 \
    --alpha 5 \
    --epsilon 0.01 \
    --beta 0.5 \
    --log_dir x x x/workspace/Qwen-Fastv-MultiLayer-VisPCO/train_logs/