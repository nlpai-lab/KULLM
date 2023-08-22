import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from transformers import LlamaTokenizer, AutoTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--llama_tokenizer_dir", default="meta-llama/Llama-2-13b", type=str)
parser.add_argument("--kor_sp_model_file", default="/data/joon/kopora/lmdata/llama2_kor.model", type=str)
args = parser.parse_args()

llama_tokenizer_dir = args.llama_tokenizer_dir
kor_sp_model_file = args.kor_sp_model_file

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
kor_sp_model = spm.SentencePieceProcessor()
kor_sp_model.Load(kor_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
kor_spm = sp_pb2_model.ModelProto()
kor_spm.ParseFromString(kor_sp_model.serialized_model_proto())

# print number of tokens
print(len(llama_tokenizer), len(kor_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

## Add Korean tokens to LLaMA tokenizer
llama_spm_tokens_set = set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")
for p in kor_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p)
print(f"New model pieces: {len(llama_spm.pieces)}")

## Save
output_sp_dir = "merged_tokenizer_sp"
output_hf_dir = "merged_tokenizer_hf"  # the path to save KULLM tokenizer
os.makedirs(output_sp_dir, exist_ok=True)
with open(output_sp_dir + "/kor_llama.model", "wb") as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir + "/kor_llama.model")

tokenizer.save_pretrained(output_hf_dir)
print(f"KULLM tokenizer has been saved to {output_hf_dir}")


# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
kor_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text = """안녕하세요. 고려대학교 구름 (KULLM) 토크나이저 입니다.
The primary use of LLaMA is research on large language models, including"""
print("Test text:\n", text)
print(f"Tokenized by LLaMA tokenizer:{llama_tokenizer.tokenize(text)}")
print(f"Tokenized by KULLM tokenizer:{kor_llama_tokenizer.tokenize(text)}")
