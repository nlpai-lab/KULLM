# %%
import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from glob import glob
from typing import List, Tuple

import jsonlines
import pandas as pd
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from secret import OPEN_AI_KEY

# %%


# %%
def get_geval_ref_free() -> PromptTemplate:
    """
    Returns a chat prompt template for evaluating responses based on given instructions and inputs.

    Returns:
    A chat prompt template for evaluating responses based on given instructions and inputs.
    """

    template = f"""두 사람 간의 대화가 주어집니다. 다음의 지시문(Instruction), 입력(Input)을 받게 될 것입니다. 그리고 지시문과 입력에 대한 응답(Response)이 제시됩니다.
당신의 작업은 응답을 평가 단계에 따라 응답을 평가하는 것입니다.
이 평가 기준을 꼼꼼히 읽고 이해하는 것이 중요합니다. 평가하는 동안 이 문서를 계속 열어두고 필요할 때 참조해 주세요.

평가 기준:
- 이해 가능성 (0 - 1): 입력 (Input)에 기반하여 응답 (Response)를 이해 할 수 있나요?
- 자연스러움 (1 - 3): 사람이 자연스럽게 말할 법한 지시어 (Instruction) 인가요?
- 맥락 유지 (1 - 3): 입력 (Input)을 고려했을 때 응답 (Response)가 맥락을 유지하나요?
- 흥미롭기 (1 - 3): 응답 (Response)가 지루한가요, 아니면 흥미로운가요?
- Instruction 사용 (0 - 1): 지시어 (Instruction)에 기반하여 응답 (Response)를 생성 했나요?
- 전반적인 품질 (1 - 5): 위의 답변을 바탕으로 이 발언의 전반적인 품질에 대한 인상은 어떤가요?

평가 단계:
1. Instruction, Input, 그리고 Response을 주의깊게 읽습니다.
2. 위의 평가 기준에 따라 Response을 평가합니다.\n\n"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    messages = [system_message_prompt]
    human_template = """{human_message}"""
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    messages += [human_message_prompt]

    chat_prompt = ChatPromptTemplate.from_messages(messages)

    return chat_prompt


# %%
async def async_generate(chain: LLMChain, id, instruction, input, response) -> str:
    """
    비동기 방식으로 Langchain을 사용하여 응답을 생성합니다.

    Args:
        chain: Langchain
        id (str): 데이터셋의 id
        instruction (str): 데이터셋의 instruction
        input (str): 데이터셋의 input
        response (str): 데이터셋의 response

    Returns:
        dict: id, instruction, input, response가 포함된 dict

    """
    human_message = (
        "Instruction: \n"
        + f"{instruction}\n"
        + f"Input: \n"
        + f"{input}\n"
        + "Response: \n"
        + f"{response}\n\n"
        + f"Result:\n- 이해 가능성 (0 - 1):\n- 자연스러움 (1 - 3):\n- 맥락 유지 (1 - 3):\n- 흥미롭기 (1 - 3):\n- Instruction 사용 (0 - 1):\n- 전반적인 품질  (1 - 5):\n\n"
    )
    resp = await chain.arun(human_message)
    return {"id": id, "instruction": instruction, "input": input, "response": response, "response_scores": resp}


# %%

parser = argparse.ArgumentParser(description="OpenAI Configuration")
parser.add_argument("--model_name", default="gpt-4", help="Model name")
parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
parser.add_argument("--max_retries", type=int, default=20, help="Max number of retries")
parser.add_argument("--input", type=str, required=True, help="Input jsonl file")

args = parser.parse_args()

chat = ChatOpenAI(
    model_name=args.model_name,
    temperature=args.temperature,
    request_timeout=60,
    max_retries=args.max_retries,
)

chat_prompt = get_geval_ref_free()
chain = LLMChain(llm=chat, prompt=chat_prompt)

translate_call = partial(async_generate, chain)


# %%
async def run_task(sem, id, instruction, input, response):
    async with sem:
        result = await translate_call(id=id, instruction=instruction, input=input, response=response)
        return result


# %%
async def main():
    # instances = jsonlines.open(args.input).iter()
    with jsonlines.open(args.input) as reader:
        examples = []
        for row in reader:
            if "instances" in row.keys():
                example = {}
                example["id"] = row["id"]
                example["instruction"] = row["instruction"]
                example["input"] = row["instances"][0]["input"]
                example["output"] = row["instances"][0]["output"]
                examples.append(example)
            else:
                examples.append(row)

    print(examples[0])

    max_concurrency = 20
    semaphore = asyncio.Semaphore(max_concurrency)
    task = [
        asyncio.ensure_future(
            run_task(
                semaphore,
                id=instance["id"],
                instruction=instance["instruction"],
                input=instance["input"],
                response=instance["output"],
            )
        )
        for instance in examples
    ]
    result = await tqdm_asyncio.gather(*task)
    return result


# results = await tqdm_asyncio.gather(*task)
results = asyncio.run(main())


# %%
with jsonlines.open(f"output/{os.path.basename(args.input)}", "w") as writer:
    writer.write_all(results)
