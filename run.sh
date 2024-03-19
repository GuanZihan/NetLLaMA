torchrun --nproc_per_node=1 \
        instruction_tuning.py \
        --model_name_or_path "EleutherAI/gpt-neo-2.7B" \
        --data_path ./datasets/TeleQnA.json \
        --bf16 True \
        --output_dir ./output_GPT_NEO_2.7B/ \
        --num_train_epochs 2 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1\
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        # --fsdp "full_shard auto_wrap" \
        # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --ddp_timeout 18000