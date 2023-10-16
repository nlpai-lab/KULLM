#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh

ckpt=$1
gpus=$2

model_name=$(echo $ckpt | awk -F'/' '{print $2}')
ckpt_name=$(echo $ckpt | awk -F'/' '{print $3}')
output_name=$model_name-$ckpt_name-beam4

conda activate sft
python infer.py --base_model EleutherAI/polyglot-ko-12.8b  \
--lora_model $ckpt  --with_prompt  --gpus $gpus \
 --predictions_file eval/${output_name}.json  \
 --data_file data/user_oriented_instructions_eval_wo_instances.jsonl

conda activate openai
python eval/postprocess.py --input eval/${output_name}.json --output eval/${output_name}.jsonl
python eval.py --input eval/${output_name}.jsonl
