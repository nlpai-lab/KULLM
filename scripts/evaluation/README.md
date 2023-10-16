# Evaluation using LLM

## 설명
이 코드는 주어진 대화 데이터에 대한 응답을 평가하는 데 사용됩니다. 평가는 주어진 지시어, 입력, 응답에 대해 이루어집니다. 평가의 기준은 이해 가능성, 자연스러움, 맥락 유지, 흥미롭기, 지시어 사용, 전반적인 품질 등입니다. 이 코드는 비동기 방식으로 구현되어 여러 개의 평가를 동시에 수행할 수 있습니다.

## 주요 구성 요소
- `get_geval_ref_free()`: 평가를 위한 지시어과 기준을 반환하는 함수입니다.
- `async_generate()`: 비동기 방식으로 주어진 지시어, 입력, 응답에 대해 평가를 수행하는 함수입니다.
- `run_task()`: 비동기 작업을 세마포어와 함께 실행하여 동시 작업의 최대 수를 제한하는 함수입니다.
- `main()`: 주요 실행 함수로, 입력 파일에서 데이터를 읽어와 비동기 방식으로 평가를 수행하고 결과를 반환합니다.

## 사용 방법
1. `secret.py` 파일에 OpenAI API 키를 저장합니다. 예: `OPEN_AI_KEY = "your-openai-api-key"`.
2. 입력 파일을 JSONL 형식으로 준비합니다. 각 라인은 별도의 JSON 오브젝트로, 지시어, 입력, 응답 등의 정보를 포함해야 합니다.
3. 아래와 같이 코드를 실행합니다.
   ```sh
   python eval.py --input input.jsonl
   ```
4. 코드 실행이 완료되면, `output/` 디렉토리에 평가 결과가 JSONL 형식으로 저장됩니다.

## 주의 사항
- 입력 파일은 `--input` 옵션으로 지정해야 하며, JSONL 형식이어야 합니다.
- OpenAI API 키는 `secret.py` 파일에 저장해야 합니다.
- `--model_name`, `--temperature`, `--max_retries` 등의 옵션을 사용하여 평가 설정을 조정할 수 있습니다.

## 종속성
- 이 코드는 `pandas`, `jsonlines`, `langchain`, `tqdm`, `asyncio` 등의 라이브러리에 의존합니다. 필요한 라이브러리는 `pip`를 사용하여 설치할 수 있습니다.
  ```sh
  pip install pandas jsonlines langchain tqdm asyncio
  ```
