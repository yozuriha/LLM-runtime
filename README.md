# PyTorch Decoder Runtime MVP

这是路线文档中 M0-M4 的可运行控制面基线：PyTorch/HuggingFace causal LM、greedy decode、prefill/decode 分离、FIFO continuous batching 和连续 KV cache。暂不包含 Triton kernel、paged KV 或量化。

## 运行测试

```bash
PYTHONPATH=decoder-runtime conda run -n triton pytest -q decoder-runtime/tests
```

## 使用 HuggingFace 模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_runtime import GenerationEngine, GenerationRequest, PyTorchCausalLMRunner

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype="auto", device_map="auto"
)
runner = PyTorchCausalLMRunner(model)
engine = GenerationEngine(runner, max_batch_size=4, max_num_batched_tokens=2048)
request = GenerationRequest("demo", tokenizer.encode("你好", add_special_tokens=True), max_new_tokens=32,
                            eos_token_id=tokenizer.eos_token_id)
engine.add_request(request)
engine.run()
print(tokenizer.decode(request.generated_ids))
```
