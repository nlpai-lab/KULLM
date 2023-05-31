<p align="center" width="100%">
<img src="assets/logo.png" alt="NLP Logo" style="width: 90%;">
</p>

## Update Logs

- 2023.05.31:
  - [🤗Polyglot-ko 12.8B 기반 KULLM-Polyglot-12.8B-v2 fp16 모델](https://huggingface.co/taeminlee/kullm-polyglot-12.8b-v2) 공개
  - 구름(KULLM) 데이터셋 v2 공개
- 2023.05.30: [🤗Polyglot-ko 12.8B 기반 KULLM-Polyglot-12.8B fp16 모델](https://huggingface.co/metterian/kullm-polyglot-12.8b) 공개

---

<br>

# ☁️ KULLM (구름): Korea University Large Langauge Model

KULLM(구름)은 고려대학교 [NLP & AI 연구실](http://blp.korea.ac.kr/)과 [HIAI 연구소](http://hiai.korea.ac.kr)에서 개발한, 한국어에 특화된 LLM (Large Language Model) 프로젝트입니다.

구름 프로젝트는 한국어에 특화된 데이터셋을 공개하여 다양한 태스크를 아우르는 AI 모델을 제공하고자 합니다.

<br/>

## Example

<img src="assets/example.png" width="65%" >

<br/>

## 한국어 기반 모델(Polyglot-ko)

KULLM(구름)은 백본 모델로 한국어 모델은 Polyglot-ko(12.8B)모델을 사용하여 학습을 진행했습니다.

1. Polyglot-ko 12.8B 기반-v2 -> 🤗 [taeminlee/kullm-polyglot-12.8b-v2](https://huggingface.co/taeminlee/kullm-polyglot-12.8b-v2)
    - 데이터셋 v2: [GPT4ALL](https://github.com/nomic-ai/gpt4all), [Dolly](https://github.com/databrickslabs/dolly), [Vicuna](https://github.com/lm-sys/FastChat)
2. Polyglot-ko 12.8B 기반-v1 -> 🤗 [metterian/kullm-polyglot-12.8b-v1](https://huggingface.co/metterian/kullm-polyglot-12.8b-v1)
    - 데이터셋 v1: GPT4ALL

Meta의 LLAMA 모델과 Polyglot의 12.8B 이하의 모델은 테스트 결과 한국어 성능이 좋지 못하여 공개하지 않기로 했습니다. 추후 여러 좋은 한국어 성능을 보여주는 LLM 모델을 학습하여 공개하고자 합니다.

<br/>

## KULLM 모델 실행 예시 코드

### Huggingface Pipeline으로 실행

- 최신버전 torch / HF 라이브러리 설치

```bash
pip install -U torch transformers tokenizers accelerate
```

아래 예제 코드로 실행해볼 수 있습니다.

```python
import torch
from transformers import pipeline, AutoModelForCausalLM

from utils.prompter import Prompter


MODEL = 'taeminlee/kullm-polyglot-12.8b-v2'

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model.eval()

pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=MODEL,
    device=0
)

prompter = Prompter("kullm")

def infer(instruction="", input_text='', is_input_full=False):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(
        prompt,
        do_sample=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        num_beams=5,
        return_full_text=False,
        eos_token_id=2,
    )
    s = output.sequences[0]
    result = tokenizer.decode(s)

infer(input_text="고려대학교에 대해서 알려줘")
```

<br/>

## Dataset

### 구름 데이터셋 v2

구름 데이터셋 v2는 [GPT4ALL](https://github.com/nomic-ai/gpt4all), [Vicuna](https://github.com/lm-sys/FastChat), 그리고 Databricks의 [Dolly](https://github.com/databrickslabs/dolly) 데이터셋을 병합한 것입니다. 이 모든 데이터셋은 DEEPL을 이용하여 한국어로 번역되었습니다.

GPT4ALL은 instruction tuned assistant-style language model이며, Vicuna와 Dolly 데이터셋은 다양한 자연어 처리 문제를 해결하는 데 활용됩니다. 특히, Dolly는 instruction/response fine tuning records를 훈련 데이터로 사용한 언어 모델입니다.

구름 데이터셋은 이들 데이터셋을 활용하여 다양한 태스크를 아우르는 AI 모델을 제공합니다.

### 구름 데이터셋 v1

구름 데이터셋 v1은 GPT4ALL을 기반으로 합니다.

#### 데이터셋 예시

GPT4ALL 데이터셋은 다음과 같이 Instruct 부분과 Input, 그리고 Output 부분으로 구성되어있습니다.

```json
...
{
    "id": "user_oriented_task_235",
    "motivation_app": "Yelp",
    "instruction": "전문 분야에 따라 레스토랑, 홈 서비스, 자동차 서비스, 기타 중 하나로 비즈니스를 분류합니다.",
    "instances": [
        {
            "input": "견적을 받으려면 650-636-4884로 전화하거나 웹사이트를 방문하세요. 이 매장은 신품 타이어 및 일반 자동차 수리를 전문으로 합니다. 모든 타이어를 자체적으로 보유하고 있으며 예산이나 차량 특성에 맞는 다양한 타이어를 보유하고 있습니다. 어떤 타이어가 필요한지 잘 모르시겠다면 전문가가 상주하여 고객의 요구에 가장 적합한 타이어를 선택할 수 있도록 도와드립니다. 또한 상용차 타이어도 취급하고 있어 다양한 차량에 맞는 타이어를 제공할 수 있습니다.",
            "output": "Auto Services"
        }
    ]
},
...
```

한국어로 번역된 데이터셋은 [`user_oriented_instructions_train.jsonl`](README.md
data/user_oriented_instructions_train.jsonl)에 저장되어 있습니다.

<br>

## Training with LoRA

KULLM은 한국어 모델로 Polyglot 12.8B 모델을 LoRA (Low Rank Adaptation)를 사용하여 학습하였습니다.

모델 학습은 A100 80GB 4대로 진행했습니다. 학습에 사용한 코드는 [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)을 기반으로 사용하였습니다.

### KULLM v2

🤗 Huggingface Repo: [https://huggingface.co/metterian/kullm-polyglot-12.8b-v2](https://huggingface.co/metterian/kullm-polyglot-12.8b-v2)

모델 학습은 구름 데이터셋 v2 (GPT4ALL, Dolly, Vicuna)을 사용하여 진행했습니다. 총 8 epoch 학습하였으며, A100 80GB 4대를 사용했습니다.

### KULLM v1

🤗 Huggingface Repo: 🤗 [https://huggingface.co/metterian/kullm-polyglot-12.8b-v1](https://huggingface.co/metterian/kullm-polyglot-12.8b-v1)

모델 학습은 구름 데이터셋 v1 (GPT4ALL)을 사용하여 진행했습니다. 총 5 epoch 학습하였으며, A100 80GB 4대를 사용했습니다.

### Dependency

1. 다음 명령어를 통해 필요한 패키지를 설치:

```bash
pip install -r requirements.txt
```

2. 만약 bitsandbytes가 작동하지 않는다면, [소스에서 직접 설치하세요](https://github.com/TimDettmers/bitsandbytes/blob/main/compile_from_source.md). 윈도우 사용자는 [다음의 설명서](https://github.com/tloen/alpaca-lora/issues/17)를 참조하세요.

### Traning (`finetune_polyglot.py`)

이 코드는 Polyglot 모델에 PEFT를 직접적으로 적용하고, 프롬프트 구성 및 토크나이징에 관련된 코드가 들어있는 파일입니다.

사용 예시:

```
finetune_polyglot.py \
--base_model='EleutherAI/polyglot-ko-12.8b' \
--data_path='/data/persuade/01_KuAlpaca/alpaca_data_gpt4_deepl+gpt4_ko.jsonl'
```

다음과 같이 하이퍼파라미터를 조정할 수도 있습니다:

```bash
python -m torch.distributed.launch  --master_port=34322  --nproc_per_node 4 finetune_polyglot.py \
    --fp16 \
    --base_model 'EleutherAI/polyglot-ko-12.8b' \
    --data_path data/user_oriented_instructions_train.jsonl \
    --output_dir ckpt/$SAVE_DIR \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --train_on_inputs \
    --logging_steps 1 \
    --eval_steps 40 \
    --weight_decay 0. \
    --warmup_steps 0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --group_by_length
```

<br/>

## Evaluation

- 모델 평가는 G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment (Yang Liu. et. al. 2023)의 방법론을 사용하였습니다. 평가 데이터셋은 [yizhongw/self-instruct](https://github.com/yizhongw/self-instruct)의 휴먼 평가 데이터셋인 `user_oriented_instructions.jsonl`을 deepl로 번역한 데이터셋을 사용하였습니다.

- 해당 데이터셋은 [`user_oriented_instructions_eval.jsonl`](data/user_oriented_instructions_eval.jsonl)에 저장되어 있습니다.

#### Prompt

- TBA.

### LLM Inference Results for Korean Evaluation Set

| Type       | Model         | Score     | Releative Score (vs GPT4) |
| ---------- | ------------- | --------: | ------------------------: |
| Closed     | GPT4          | 87.6      | 100                       |
| Closed     | ChatGPT       | 83.3      | 95.1                      |
| Open       | **KULMM v2**  | **62.3**  | **71.1**                  |
| Open       | KoAlpaca v1.1 | 40.6      | 46.3                      |
| Open       | koVicuna      | 50.2      | 57.3                      |
---