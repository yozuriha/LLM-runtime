
import json
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_runtime import GenerationEngine, GenerationRequest, PyTorchCausalLMRunner

with open("test_prompt.json", "r", encoding="utf-8") as f:
    prompt = json.load(f)  # json.load() 从文件对象加载
model_dir = "/mnt/c/Users/Administrator/Desktop/triton/model"
print(prompt["question"])
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
).to("cuda").eval()

runner = PyTorchCausalLMRunner(model, device="cuda")
engine = GenerationEngine(
    runner,
    max_batch_size=1,
    max_num_batched_tokens=6144,
    max_cache_tokens=65536,
)
messages = [
    {"role": "user", "content": prompt["question"]},
]
encode = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
)
input_ids = encode["input_ids"]
request = GenerationRequest(
    request_id="demo",
    input_ids=input_ids,
    max_new_tokens=1024,
    eos_token_id=tokenizer.eos_token_id,
)

engine.add_request(request)

# CUDA kernel 是异步启动的；阶段前后同步后，wall time 才包含真实 GPU 执行时间。
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

# 第一个 step 执行完整 prompt 的 prefill，并采样第一个输出 token。
prefill_start = perf_counter()
engine.step()
torch.cuda.synchronize()
prefill_seconds = perf_counter() - prefill_start
generated_after_prefill = len(request.generated_ids)

# 后续 step 每轮只输入上轮 token，并通过 past_key_values 完成 decode。
decode_start = perf_counter()
while engine.scheduler.waiting or engine.scheduler.running:
    engine.step()
torch.cuda.synchronize()
decode_seconds = perf_counter() - decode_start

prompt_tokens = request.prompt_len
generated_tokens = len(request.generated_ids)
decode_tokens = generated_tokens - generated_after_prefill

# TPOT 只统计 decode 阶段：首个 token 的时间已包含在 TTFT 中。
metrics = {
    "prompt_tokens": prompt_tokens,
    "generated_tokens": generated_tokens,
    "ttft_seconds": prefill_seconds,
    "tpot_seconds": decode_seconds / decode_tokens if decode_tokens else 0.0,
    "prefill_tokens_per_second": prompt_tokens / prefill_seconds if prefill_seconds else 0.0,
    "decode_tokens_per_second": decode_tokens / decode_seconds if decode_seconds else 0.0,
    "peak_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
    # "generated_token_ids": request.generated_ids,
}

print(tokenizer.decode(request.generated_ids, skip_special_tokens=True))
print("\nMetrics:")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
