# lr=1e-4
# lr=3e-5
lr=1e-5
lora_rank=8
lora_alpha=32
# lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
# modules_to_save="embed_tokens,lm_head"
modules_to_save="embed_in,embed_out"
lora_dropout=0.05

pretrained_model="EleutherAI/polyglot-ko-12.8b"
dataset_dir=data/kullm_v3/train
per_device_train_batch_size=64
per_device_eval_batch_size=64
gradient_accumulation_steps=8
output_dir=ckpt/polyglot-13b-kullm_v3-${lr}-ep5
peft_model=path/to/peft/model/dir # if you did pre-training with PEFT
validation_file=data/kullm_v3/kullm_v2_2_alpaca_dolly_vicuna_shuffled_valid.json

deepspeed_config_file=ds_zero2_no_offload.json

torchrun --nproc_per_node 8 run_clm_sft_with_peft.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${pretrained_model} \
    --dataset_dir ${dataset_dir} \
    --validation_file ${validation_file} \
    --validation_split_percentage 0.001 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --per_device_eval_batch_size ${per_device_eval_batch_size} \
    --do_train \
    --do_eval \
    --seed $RANDOM \
    --fp16 \
    --num_train_epochs 5 \
    --lr_scheduler_type cosine \
    --learning_rate ${lr} \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --preprocessing_num_workers 64 \
    --max_seq_length 2028 \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank ${lora_rank} \
    --lora_alpha ${lora_alpha} \
    --trainable ${lora_trainable} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --torch_dtype float16 \
    --gradient_checkpointing \
    --report_to wandb \
    --ddp_find_unused_parameters False
    # --save_steps 30 \
    # --peft_path ${peft_model} \
    # --num_train_epochs 1 \
