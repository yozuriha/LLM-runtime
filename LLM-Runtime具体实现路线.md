# NVIDIA Triton LLM Runtime 具体实现路线

## 1. 项目边界

目标是实现一个单卡 NVIDIA GPU 上可运行、可验证、可分析的 LLM 推理 Runtime，重点体现 Triton Kernel、KV Cache、batch 调度和推理性能优化能力。

固定条件：

- GPU：RTX 4070，SM89（Ada Lovelace），12GB；
- CUDA + PyTorch + Triton；
- 推荐 Linux 或 WSL2；
- 模型优先使用 `Qwen2.5-0.5B-Instruct`，稳定后再测试 1.5B；在显存预算允许时再评估 3B；
- 第一阶段 FP16 + greedy decode；
- 暂不实现 beam search、speculative decoding 和分布式推理。

## 2. 总体架构与目录

```text
CLI / HTTP API
  -> Request Queue
  -> Continuous Batch Scheduler
  -> Sequence State
  -> KV Cache Manager
  -> Model Runner: prefill / decode
  -> Triton Kernels
  -> CUDA Stream + Event
  -> Token Streaming Response
```

控制面使用 Python、PyTorch 和 HuggingFace；计算热点逐步替换为 Triton。建议目录：

```text
llm_runtime/
  model/       # 权重、reference runner、Triton runner
  kernels/     # rmsnorm、rope、swiglu、linear、attention
  kv_cache/    # block manager、cache、layout
  scheduler/   # request、batch、调度策略
  engine/      # generate、sampling
  server/      # HTTP/流式输出
  benchmarks/  # kernel、模型、服务 benchmark
  tests/       # 正确性和调度测试
```

先固定三个接口：

```python
class ModelRunner:
    def prefill(self, input_ids, request_ids): ...
    def decode(self, input_ids, sequences): ...

class KVCache:
    def allocate(self, request_id, num_tokens): ...
    def append(self, request_id, key, value): ...
    def release(self, request_id): ...

class Scheduler:
    def add_request(self, request): ...
    def next_batch(self): ...
```

## 3. M0：PyTorch 正确性基线

先不写 Triton，完成唯一可信的 reference：

1. 加载 tokenizer 和 HuggingFace 模型；
2. 固定 greedy decode 和随机种子；
3. 实现单请求生成；
4. 保存每一步 logits、token id 和停止原因；
5. 用 CUDA Event 测量 prefill/decode；
6. 保存固定 prompt 集合作为回归输入。

记录：

```text
prompt_tokens / generated_tokens
TTFT / TPOT
prefill_tokens_per_second / decode_tokens_per_second
peak_memory / generated_token_ids
```

验收标准：同一 prompt、同一 seed、greedy 模式下，自定义 Runner 与 HuggingFace 输出 token 一致。

## 4. M1：Triton 基础融合 Kernel

每个 Kernel 遵循：

```text
PyTorch reference -> Triton 实现 -> 随机正确性 -> microbenchmark
```

### RMSNorm

```text
y = x * rsqrt(mean(x^2) + eps) * weight
```

FP16 输入、FP32 累加；mask 支持非 32 对齐 hidden size；调整 `BLOCK_SIZE` 和 warp 数；避免产生平方、均值等中间 tensor。

### RoPE

实现 Q/K 旋转位置编码，测试不同 position、head_dim、长序列和 GQA。

### SwiGLU

融合：

```text
silu(gate) * up
```

减少 global memory 读写。

### M1 验收

- `torch.testing.assert_close` 通过；
- 覆盖短/长、整齐/非整齐 shape；
- FP16 最大误差控制在约 `1e-3`；
- 记录 latency、有效带宽和 Kernel launch 次数；
- 没有完整模型结果前，不宣称端到端加速。

## 5. M2：自定义 Transformer Forward

单独实现简化的 Llama/Qwen Decoder，不直接修改 Transformers 源码：

```text
input
  -> RMSNorm -> QKV Linear -> RoPE -> Attention -> residual add
  -> RMSNorm -> gate/up Linear -> SwiGLU -> down Linear -> residual add
```

第一版全部使用 PyTorch：QKV/MLP 用 `matmul`，attention 用 `torch.scaled_dot_product_attention`。

重点核对：

- 权重名称和布局；
- Q/K/V 维度；
- GQA 映射；
- position id 和 causal mask；
- residual 顺序；
- lm_head 输出。

逐层比较 hidden states，最后比较 logits，不要只比较最终 token。

## 6. M3：Prefill/Decode 分离

Prefill 输入 `[batch, prompt_length]`，一次处理整段 prompt 并建立 KV；Decode 输入 `[batch, 1]`，读取历史 KV 并追加当前 K/V。

两条路径分开，因为 Prefill 更偏 GEMM/大矩阵计算，Decode 更容易受 KV 访存、Kernel launch 和 Python 调度影响。

要求：

- 两条路径使用不同 batch 组织；
- 不在每个 token 后调用 `torch.cuda.synchronize()`；
- attention 先使用 PyTorch SDPA/FlashAttention 作为 baseline；
- 记录 TTFT、TPOT、GPU 时间和 CPU 调度时间。

## 7. M4：连续 KV Cache

先实现连续布局：

```text
K/V: [num_layers, batch, kv_heads, max_seq, head_dim]
```

每条序列保存 `request_id`、token ids、`seq_len`、`max_seq_len`、`finished` 和 `kv_slot`。

必须处理不同长度、追加 KV、提前结束、batch 移除、取消请求和 batch 重排。先实现静态 batch：请求同时 Prefill，再一起 Decode 到结束。

## 8. M5：Paged KV Cache

第一版 block size 使用 16 tokens。RTX 4070 的 12GB 显存允许优先验证 1.5B BF16；3B 模型需要更保守的 batch/token budget，必要时启用 INT8 weight-only：

```text
K/V pool:    [num_layers, 2, num_blocks, block_size, kv_heads, head_dim]
block_table: [batch, max_blocks_per_sequence]
seq_lens:    [batch]
```

映射关系：

```python
logical_block = token_position // block_size
offset = token_position % block_size
physical_block = block_table[sequence_id, logical_block]
```

`BlockManager` 提供 `allocate()`、`free()`、`append_block()` 和 `get_block_table()`。

测试跨 block、请求取消、显存不足、多请求释放和 batch 重排。CUDA stream 仍读取的 block 不能立即复用，必要时用 Event 延迟回收。

## 9. M6：Continuous Batching

每个 Decode step：

```text
1. 回收已完成请求的 KV block
2. 从 waiting queue 选择新请求
3. 为新请求执行 Prefill
4. 组成 running Decode batch
5. 执行一次 Decode
6. 采样、追加 token、更新长度和 block table
7. 返回已完成请求
```

第一版使用 FIFO + token budget，同时限制：

```text
max_batch_size
max_num_batched_tokens
max_num_blocks
max_sequence_length
```

稳定后比较 FCFS、shortest-prompt-first、Prefill/Decode 分离队列和 chunked Prefill（每轮 512/1024 tokens）。不能只按 request 数限制。

## 10. M7：Triton Decode Attention

输入为 `Q=[batch, q_heads, head_dim]`、Paged KV、`block_table` 和 `seq_lens`。

Kernel 完成：查找物理 block、加载历史 K/V、GQA head 映射、QK 分数、online softmax、V 累加和输出写回。

Online softmax：

```text
m_new = max(m_old, max(score))
l_new = l_old * exp(m_old - m_new) + sum(exp(score - m_new))
acc_new = acc_old * exp(m_old - m_new) + exp(score - m_new) @ V
```

Prefill 先使用 PyTorch SDPA/FlashAttention；Decode 实现自定义 Triton paged attention；Triton Prefill FlashAttention 放到后续。单独测试 causal mask、不同长度、GQA/MQA 和 head_dim。

## 11. M8：INT8 量化

第一版实现 INT8 weight-only：

```text
weight_int8: [out_features, in_features]
scale:       [out_features, num_groups]
group_size:  64 或 128
```

路径：

```text
INT8 weight -> Triton 内 dequant -> FP16 accumulation -> output
```

准确名称是 `INT8 weight-only dequant-fused linear`；没有整数矩阵指令时不要称为 INT8 Tensor Core GEMM。

验证单层输出、最终 logits、生成 token、perplexity/任务准确率、权重显存和吞吐。误差明显时再比较 per-channel、group size 64、异常值通道保留 FP16 和 W8A8。

## 12. M9：CUDA Stream/Event 与 Graph

先拆出 `prefill_stream` 和 `decode_stream`：

```text
prefill_stream.record_event(kv_ready)
decode_stream.wait_event(kv_ready)
```

禁止每个 token 全局同步；记录 H2D、Kernel、D2H 和 scheduler 时间，区分 GPU 计算、KV 访存和 Python 调度瓶颈。

CUDA Graph 放在动态 batch 稳定后，只针对固定 batch/shape 做 capture。

## 13. 验证体系

### Kernel 正确性

- `torch.testing.assert_close`；
- 随机 shape、非整齐 shape、长序列和不同 head_dim；
- NaN/Inf、极端 logits、全 padding、结束请求；
- 多 stream 重复运行检查数据竞争；
- 量化 Kernel 做 dequant、逐层和最终 logits 对比。

### 模型级一致性

- greedy decode 逐 token 对比 HuggingFace；
- 固定 seed 对比 logits、生成 token 和停止条件；
- 50–100 条固定 prompt，覆盖短/长序列；
- 比较 batch 内不同长度请求；
- 量化前后比较 perplexity 和固定任务准确率。

### 性能指标

| 层级 | 指标 |
| --- | --- |
| Kernel | P50/P95 latency、带宽、TFLOPS、occupancy、寄存器/共享内存 |
| Prefill | TTFT、prompt tokens/s、不同 prompt length 吞吐 |
| Decode | TPOT、decode tokens/s、batch size 曲线 |
| 服务 | request/s、output tokens/s、P50/P95/P99、排队时间、峰值显存 |
| 资源 | 权重显存、KV 显存、GPU 利用率、CPU 调度占比 |
| 质量 | logits max error、token match、perplexity、任务准确率 |

### Baseline 矩阵

| 版本 | KV | Attention | Batch |
| --- | --- | --- | --- |
| HuggingFace eager FP16 | 连续 | HF attention | 单请求 |
| PyTorch SDPA | 连续 | SDPA/FlashAttention | 静态 batch |
| Triton fused | 连续 | SDPA | 静态 batch |
| Triton + Paged KV | 分页 | Triton decode attention | 静态 batch |
| 完整 Runtime | 分页 | Triton attention | Continuous batching |
| vLLM/TensorRT-LLM | 框架实现 | 框架实现 | 同请求分布 |

每次只改变一个因素，同时报告质量、吞吐、TTFT、TPOT、P95 和显存。

## 14. 里程碑

```text
第 1 周：环境、模型加载、HuggingFace baseline
第 2 周：RMSNorm/RoPE/SwiGLU Triton Kernel
第 3 周：自定义 Transformer forward
第 4 周：Prefill/Decode 拆分和连续 KV
第 5 周：Paged KV BlockManager
第 6 周：Continuous Batching
第 7 周：Triton paged decode attention
第 8 周：INT8 weight-only
第 9 周：CUDA Stream/Event 和 profiler
第 10 周：vLLM 对照和最终报告
```

每个里程碑必须产出代码、单元测试、正确性对照、microbenchmark、失败样本和结果记录。

## 15. 最终简历句式

没有实测数字时：

> 基于 PyTorch/HuggingFace 搭建单卡 LLM 推理 Runtime，使用 Triton 实现 fused RMSNorm、SwiGLU、INT8 weight-only linear 和 paged decode attention，设计 Prefill/Decode 分离、Paged KV Cache 与 Continuous Batching，并建立 Kernel、模型一致性和端到端吞吐评测。

有真实对照后：

> 针对 decode 阶段 KV 访存和 Kernel launch 开销，设计 Paged KV Cache 与 Continuous Batching，并使用 Triton 融合 RMSNorm/SwiGLU、量化 Linear 和 Paged Attention；在固定模型与请求分布下，相比 PyTorch baseline 将 decode 吞吐提升 X%、P95 TPOT 降低 Y%、峰值显存降低 Z%，生成质量/perplexity 变化控制在阈值内。

## 16. 面试重点

1. 为什么 Prefill 和 Decode 要分开调度？
2. Paged KV 如何将逻辑 token 映射到物理 block？
3. 请求结束或取消时，什么时候可以安全释放 KV block？
4. Continuous Batching 如何处理不同长度请求和新请求加入？
5. Triton 融合减少了哪些 global memory traffic，为什么可能降低 occupancy？
6. INT8 Weight-only 和 W8A8 在误差、带宽和计算路径上有什么差异？
7. FlashAttention 如何通过 online softmax 避免写出完整 attention matrix？
8. 如何证明提升来自 Kernel，而不是 batch、输入长度或 warmup 差异？
9. 如何从 Nsight Systems/Compute 区分 GPU 计算、KV 访存和 Python 调度瓶颈？

## 17. 禁止过度表述

- 不要把调用 PyTorch SDPA 写成自己实现 FlashAttention；
- 不要把 vLLM/TensorRT-LLM 已有能力写成个人实现；
- 不要在实测前填写吞吐、TTFT、TPOT 或显存提升；
- 不要把 INT8 weight-only 直接称为完整 INT8 Tensor Core GEMM；
- 不要把单个 Kernel 的 microbenchmark 提升写成端到端服务提升。
