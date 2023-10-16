#!/bin/bash

epoch=2
lr=1e-4
pretrained_model="EleutherAI/polyglot-ko-12.8b"
dataset_dir=data/kullm_v2

per_device_train_batch_size=4
per_device_eval_batch_size=4
gradient_accumulation_steps=8

# output_dir=
validation_file=data/kullm_v2_dev.jsonl
prediction_file=data/user_oriented_instructions_eval_wo_instances.jsonl

deepspeed_config_file=ds_zero3_no_offload.json

deepspeed --include  localhost:4,5,6,7 --master_port 24324 run_clm_sft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --tokenizer_name ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_predict \
    --seed 42 \
    --fp16 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.05 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy epoch \
    --num_train_epochs $epoch \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 64 \
    --max_seq_length 1024 \
    --output_dir ${output_dir} \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype float16 \
    --validation_file ${validation_file} \
    --prediction_file ${prediction_file} \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --ddp_find_unused_parameters False

    # --early_stopping_patience 3 \
    # --load_best_model_at_end \

