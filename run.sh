# torchrun --nproc_per_node=4 \
#         instruction_tuning.py \
#         --model_name_or_path "EleutherAI/gpt-neo-2.7B" \
#         --data_path ./datasets/TeleQnA.json \
#         --bf16 True \
#         --output_dir ./output_GPT_NEO_2.7B_/ \
#         --num_train_epochs 1 \
#         --per_device_train_batch_size 2 \
#         --per_device_eval_batch_size 1 \
#         --gradient_accumulation_steps 4 \
#         --evaluation_strategy "no" \
#         --save_strategy "epoch" \
#         --learning_rate 2e-5 \
#         --weight_decay 0. \
#         --warmup_ratio 0.03 \
#         --lr_scheduler_type "cosine" \
#         --logging_steps 1 \
#         --model_max_length 2048 \
#         --gradient_checkpointing True \
#         --ddp_timeout 18000

# EleutherAI/gpt-neo-125m
# EleutherAI/pythia-70m
# EleutherAI/pythia-1.4b
# EleutherAI/pythia-6.9b

# ablation studies on number of pretraining dataset
# models: EleutherAI/pythia-6.9b, LLama-7b, GPT_NEO 2.7B
# dataset: 0.1 / 0.3/ 0.5/ 0.7 / 0.9 / 1.0
# performance evaluation: accuracy


torchrun --nproc_per_node=4 \
        instruction_tuning.py \
        --model_name_or_path "EleutherAI/gpt-neo-1.3B" \
        --data_path ./datasets/TeleQnA.json \
        --bf16 True \
        --output_dir ./output_gpt-neo-1.3B/ \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --ddp_timeout 18000 \
        # --fsdp "full_shard auto_wrap" \
        # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        # --tf32 True \