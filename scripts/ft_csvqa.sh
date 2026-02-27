#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=16
EVAL_BATCH_PER_DEVICE=2
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# You should freeze the the merger also, becuase the merger is included in the vision_tower.
#     --modules_to_save "['moe_lora_layernorm', 'moe_lora']" \
#    --deepspeed scripts/zero2.json \



CUDA_VISIBLE_DEVICES=2 python src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens', 'lora_layernorm', 'custom_lora']" \
    --modules_to_save "['lora_layernorm', 'custom_lora']" \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --model_id /root/model/Qwen/Qwen2.5-VL-3B-Instruct \
    --data_path /root/autodl-tmp/data/Skywork/CSVQA/train/train.json \
    --eval_data_path /root/autodl-tmp/data/Skywork/CSVQA/test/test.json \
    --image_folder /root/autodl-tmp/data/Skywork/CSVQA/ \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --bf16_full_eval True \
    --disable_flash_attn2 False \
    --output_dir ./output/csvqa/qwen2_5_vl_loha64_v01 \
    --max_steps 8 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --per_device_eval_batch_size $EVAL_BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --label_names "labels" \
    --dataloader_num_workers 4 \
    --do_final_eval True \
    --test_data_path /root/autodl-tmp/data/Skywork/CSVQA/test/test.json


    # --eval_strategy "steps" \
    # --eval_steps 8 \
    # --save_strategy "steps" \
    # --save_steps 8 \
    # --save_total_limit 5 \
    # --batch_eval_metrics True \
    # --metric_for_best_model "accuracy" \