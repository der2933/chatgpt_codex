#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
export LD_LIBRARY_PATH=/home/jnu/anaconda3/envs/deepspeed/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH"

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="/home/jnu/model/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=8
EVAL_BATCH_PER_DEVICE=4
NUM_DEVICES=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# You should freeze the the merger also, becuase the merger is included in the vision_tower.
#     --modules_to_save "['moe_lora_layernorm', 'moe_lora']" \

CUDA_VISIBLE_DEVICES=1 deepspeed src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /home/jnu/gxw/data/ScienceQA/QCM-ALE/train/train.json \
    --eval_data_path /home/jnu/gxw/data/ScienceQA/QCM-ALE/test/test.json \
    --image_folder /home/jnu/gxw/data/ScienceQA/image \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --bf16_full_eval True \
    --disable_flash_attn2 False \
    --output_dir /home/jnu/gxw/output/qwen2_5_vl_lora8 \
    --max_steps 305 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --per_device_eval_batch_size $EVAL_BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1024 * 28 * 28)) \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --eval_strategy "steps" \
    --eval_steps 30 \
    --batch_eval_metrics True \
    --metric_for_best_model "accuracy" \
    --label_names "labels" \
    --save_strategy "steps" \
    --save_steps 30 \
    --save_total_limit 5 \
    --dataloader_num_workers 4 \
    --do_final_eval True \
    --test_data_path /home/jnu/gxw/data/ScienceQA/QCM-ALE/test/test.json

CUDA_VISIBLE_DEVICES=1 deepspeed src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /home/jnu/gxw/data/ScienceQA/QCM-ALE/train/train.json \
    --eval_data_path /home/jnu/gxw/data/ScienceQA/QCM-ALE/test/test.json \
    --image_folder /home/jnu/gxw/data/ScienceQA/image \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --bf16_full_eval True \
    --disable_flash_attn2 False \
    --output_dir /home/jnu/gxw/output/qwen2_5_vl_lora64 \
    --max_steps 305 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --per_device_eval_batch_size $EVAL_BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1024 * 28 * 28)) \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --eval_strategy "steps" \
    --eval_steps 30 \
    --batch_eval_metrics True \
    --metric_for_best_model "accuracy" \
    --label_names "labels" \
    --save_strategy "steps" \
    --save_steps 30 \
    --save_total_limit 5 \
    --dataloader_num_workers 4 \
    --do_final_eval True \
    --test_data_path /home/jnu/gxw/data/ScienceQA/QCM-ALE/test/test.json


CUDA_VISIBLE_DEVICES=1 deepspeed src/train/train_sft.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_method "mamoelora" \
    --modules_to_save "['lora_layernorm', 'custom_lora']" \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero2.json \
    --model_id $MODEL_NAME \
    --data_path /home/jnu/gxw/data/ScienceQA/QCM-ALE/train/train.json \
    --eval_data_path /home/jnu/gxw/data/ScienceQA/QCM-ALE/test/test.json \
    --image_folder /home/jnu/gxw/data/ScienceQA/image \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_merger True \
    --bf16 True \
    --fp16 False \
    --bf16_full_eval True \
    --disable_flash_attn2 False \
    --output_dir /home/jnu/gxw/output/qwen2_5_vl_mamoelora \
    --max_steps 305 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --per_device_eval_batch_size $EVAL_BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((1024 * 28 * 28)) \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --eval_strategy "steps" \
    --eval_steps 30 \
    --batch_eval_metrics True \
    --metric_for_best_model "accuracy" \
    --label_names "labels" \
    --save_strategy "steps" \
    --save_steps 30 \
    --save_total_limit 5 \
    --dataloader_num_workers 4 \
    --do_final_eval True \
    --test_data_path /home/jnu/gxw/data/ScienceQA/QCM-ALE/test/test.json
