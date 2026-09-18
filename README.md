# PyTorch Decoder Runtime MVP

这是路线文档中 M0-M4 的可运行控制面基线：PyTorch/HuggingFace causal LM、greedy decode、批量 prefill/decode、FIFO continuous batching 和 KV cache。M1 已加入基础 Triton kernel；当前 paged KV 采用 software block/page storage，尚未接入 paged-attention kernel 或量化。

## M1 Triton 基础融合 Kernel

Kernel 按算子拆分在 `llm_runtime/kernels/` 下：

- `rmsnorm.py`：RMSNorm 与 PyTorch reference
- `rope.py`：非交错 half-split RoPE 与 PyTorch reference
- `swiglu.py`：SwiGLU 与 PyTorch reference
- `fused.py`：旧导入路径的兼容导出层

当前提供：

```python
from llm_runtime.kernels import apply_rope, rmsnorm, swiglu

y = rmsnorm(x, weight)
q_rot, k_rot = apply_rope(q, k, cos, sin, position_ids)
hidden = swiglu(gate, up)
```

每个操作都有对应的 `*_reference` PyTorch 实现。GPU 正确性测试命令：

```bash
PYTHONPATH=decoder-runtime \
conda run -n triton pytest -q -p no:cacheprovider decoder-runtime/tests/test_kernels.py
```

RTX 4070 上运行 microbenchmark（使用 `triton.testing.Benchmark`）：

```bash
PYTHONPATH=decoder-runtime \
conda run -n triton python decoder-runtime/benchmarks/benchmark_m1.py
```

benchmark 为每个算子分别跑 PyTorch 和 Triton provider。官方 Benchmark 表报告 latency；随后输出统一明细表，包含 `latency_ms`、相对 PyTorch 的 `max_abs_error`、`peak_memory_mib` 和占 GPU 总显存的 `peak_memory_pct`。首次 Triton 编译时间会在 warmup 中排除。当前基线目标为 RTX 4070（12 GB，SM89）。

## 批量 Runner 接口

`ModelRunner` 保留单请求 `prefill()`/`decode()`，并新增两个批量入口：

```python
logits = runner.prefill_batch(requests)              # [batch, vocab]
logits = runner.decode_batch(requests, token_ids)    # [batch, vocab]
```

`PyTorchCausalLMRunner` 会将不同长度 prompt 右侧 padding，在一次模型调用中完成 prefill；decode 会将各请求的历史 KV padding 到 batch 最大长度，用 attention mask 屏蔽 padding，再把返回的 KV 按真实长度拆回请求。`GenerationEngine.step()` 会分别收集本轮的 prefill 请求和 decode 请求，各调用一次批量接口。

Runner 同时兼容 Transformers 旧版 tuple cache、带 `to_legacy_cache()` 的 Cache，以及 Transformers 5.x 的 `Cache.layers[i].keys/values`。在 Transformers 5.x 下会重建可变 `DynamicCache`，不能把拆出的 tuple 直接传回 Qwen 模型。

自定义 Runner 如果暂时没有批量实现，可以继续继承单请求接口：基类批量方法会逐请求 fallback，便于先迁移控制面，再替换 GPU forward。

## Paged KV 与模型 Kernel 接入状态

`PagedKVCache` 使用固定 `block_size` 分配物理页，为每个请求维护 `block_table`，并支持扩容、回收和按页 materialize。由于 HuggingFace 标准 attention 仍接收连续 `past_key_values`，当前 decode 会暂时将页拼接回 dense KV；后续 paged-attention Triton kernel 可直接读取 block table 消除该复制。

`PyTorchCausalLMRunner(..., use_triton_kernels=True)` 在 CUDA Qwen2 模型上会安装 M1 kernel：Qwen2RMSNorm、SwiGLU MLP 和 half-split RoPE。非 CUDA、非 Qwen2 或不满足布局约束时自动回退 PyTorch 实现。

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
