import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from utils.prompter import Prompter

MODEL = "taeminlee/kullm-polyglot-12.8b-v2"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device=f"cuda", non_blocking=True)
model.eval()

pipe = pipeline("text-generation", model=model, tokenizer=MODEL, device=0)

prompter = Prompter("kullm")


def infer(instruction="", input_text=""):
    prompt = prompter.generate_prompt(instruction, input_text)
    output = pipe(prompt, max_length=512, temperature=0.2, num_beams=5, eos_token_id=2)
    s = output[0]["generated_text"]
    result = prompter.get_response(s)

    return result


result = infer(input_text="고려대학교에 대해서 알려줘")
print(result)
# '고려대학교에 대해 궁금한 점이 있으시면 언제든지 문의해 주세요. 고려대학교는 한국에서 가장 오래되고 권위 있는 대학교 중 하나로, 고려대학교의 역사는 한국의 역사와 함께해 왔습니다. 고려대학교는 학문적 우수성을 추구하는 동시에 사회적 책임을 다하기 위해 최선을 다하고 있습니다. 고려대학교는 학생, 교수진, 교직원을 위한 다양한 프로그램과 지원을 제공하는 것으로 유명합니다. 고려대학교는 한국의 정치, 경제, 사회 분야에서 중요한 역할을 담당하고 있습니다. 고려대학교에 대해 더 자세히 알고 싶으신가요?'