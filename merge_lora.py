import argparse
import os

import torch
import transformers
from peft import PeftModel
import peft
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="EleutherAI/polyglot-ko-12.8b")
parser.add_argument("--lora_model_path", type=str, default="ckpt/polyglot-13b-kullm_v3-3e-5/sft_lora_model")
parser.add_argument("--output_dir", type=str, default="ckpt/polyglot-13b-kullm_v3-3e-5")
args = parser.parse_args()


BASE_MODEL = args.base_model
assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = GPTNeoXForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

## infer the model size from the checkpoint
embedding_size = base_model.get_input_embeddings().weight.size(1)


print(f"Loading LoRA {args.lora_model_path}...")
tokenizer = GPTNeoXTokenizerFast.from_pretrained(args.lora_model_path)
print(f"base_model vocab size: {base_model.get_input_embeddings().weight.size(0)}")
print(f"tokenizer vocab size: {len(tokenizer)}")


model_vocab_size = base_model.get_input_embeddings().weight.size(0)


if model_vocab_size != len(tokenizer):
    base_model.resize_token_embeddings(len(tokenizer))
    print(f"Extended vocabulary size to {len(tokenizer)}")


first_weight = base_model.gpt_neox.layers[0].attention.query_key_value.weight  # TODO: polyglot-ko-5.8b
first_weight_old = first_weight.clone()

print(f"Loading LoRA weights")
lora_model = PeftModel.from_pretrained(
    base_model,
    args.lora_model_path,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.gpt_neox.layers[0].attention.query_key_value.weight

assert torch.allclose(first_weight_old, first_weight)
# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {k.replace("base_model.gpt_neox.", ""): v for k, v in lora_model_sd.items() if "lora" not in k}

# LlamaForCausalLM.save_pretrained(args.lora_weights, "./hf_ckpt", state_dict=deloreanized_sd)
# LlamaForCausalLM.save_pretrained(os.path.join(args.lora_weights, "hf_ckpt"), state_dict=deloreanized_sd, max_shard_size="400MB")
GPTNeoXForCausalLM.save_pretrained(base_model, save_directory=args.output_dir, state_dict=deloreanized_sd)

