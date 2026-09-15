# NVIDIA GPU 版 Triton LLM Runtime 方案

## 0. 项目边界

目标是做一个单卡 NVIDIA GPU 上可运行、可测量、可解释的 LLM 推理 Runtime，优先适配 RTX 3060 Ti（SM86、8GB）和 0.5B–3B 级 Decoder-only 模型。第一阶段不追求替代 vLLM，而是围绕 prefill/decode、KV cache、batch 调度和关键 Triton kernel 形成一条完整闭环。

推荐首个模型：Qwen2.5-0.5B/1.5B-Instruct 或同规模 Llama 模型。模型太大时，显存问题会掩盖 kernel 和调度优化效果。

## 1. 框架选择

### 推荐组合

```text
Python Runtime / Scheduler
  -> PyTorch Tensor + HuggingFace 权重/Tokenizer
  -> Triton 自定义 kernel
  -> CUDA runtime / CUDA events / streams
  -> NVIDIA GPU
```

- **PyTorch**：张量、权重加载、CUDA stream、host/device 管理和正确性参考实现；
- **HuggingFace Transformers**：只作为模型结构/权重/Tokenizer 来源，逐步替换其 attention、MLP、norm 路径；
- **Triton**：实现可独立对比的 fused RMSNorm、RoPE、SwiGLU、quantized linear 和 paged decode attention；
- **FastAPI 或简单 CLI**：提供服务入口，第一阶段不引入复杂 RPC；
- **不建议第一版基于 vLLM 二次开发**：vLLM 已经拥有成熟 scheduler、paged attention 和 CUDA graph，难以证明自己的改动收益；可把 vLLM 作为最终端到端 baseline。

### 什么时候使用其他框架

| 选择 | 适合用途 | 本项目定位 |
| --- | --- | --- |
| vLLM | 生产级吞吐 baseline | 对照组，不作为第一版内部实现 |
| SGLang | 结构化生成/复杂服务编排 | 后续对照，不作为核心依赖 |
| TensorRT-LLM | NVIDIA 生产部署和 TensorRT kernel | 可作为高性能上限对照 |
| FlashInfer | 现成 paged attention/采样 kernel | 后期对照，不替代 Triton 学习主线 |
| Triton + PyTorch | 自定义算子和 Runtime 学习/展示 | 核心实现 |

面试中的准确说法应是“基于 PyTorch/HuggingFace 搭建控制面，自研 Triton kernel 和单卡推理 Runtime”，而不是“实现了一个完整 vLLM”。

## 2. Runtime 分层

```text
API / CLI
  -> Request Queue
  -> Continuous Batch Scheduler
  -> Sequence State
       token ids / position / finished / sampling
  -> KV Cache Manager
       block pool / block table / refcount
  -> Model Runner
       prefill path / decode path
  -> Triton Kernels
       norm / rotary / GEMM / MLP / attention / sampling
  -> CUDA stream + event
  -> token streaming response + metrics
```

### 2.1 请求状态

每条序列至少保存：`request_id`、prompt token、generated token、当前长度、最大长度、采样参数、完成状态、KV block ids、到达时间和最后一次 decode 时间。

### 2.2 Prefill 与 Decode 分离

- **Prefill**：一次处理整段 prompt，矩阵维度大，重点是 GEMM/attention 计算效率；
- **Decode**：每个序列每步只生成一个 token，重点是 KV 读取、kernel launch 次数和 batch 组织；
- scheduler 不应把两种阶段混成一个固定 batch，否则短请求会被长 prompt 阻塞；
- 后续可加入 chunked prefill，每轮只处理固定 token chunk，避免长 prompt 独占 GPU。

## 3. KV Cache 管理

### 3.1 第一版：连续 KV

先用每层连续张量验证模型：`[num_layers, 2, max_batch, max_seq, num_heads, head_dim]`。它容易实现，但 batch 中请求长度差异大时浪费显存。

### 3.2 第二版：Paged KV

使用固定大小 block，例如每 block 16 或 32 个 token：

```text
KV pool:   [num_blocks, 2, num_layers, block_size, num_kv_heads, head_dim]
block_table: [batch, max_blocks_per_seq]
seq_lens:    [batch]
free_list:   device/host managed block ids
```

分配策略：请求进入时按 prompt 长度分配 block；decode 每跨过一个 block 申请新块；请求结束后归还 free list。block table 只保存物理块映射，attention kernel 根据逻辑 token 位置查表。

### 3.3 必须处理的生命周期问题

- 请求取消和异常时释放全部 block；
- batch 重排后更新 block table 和 sequence metadata；
- 不允许复用仍被 CUDA stream 读取的 block，必要时用 event 延迟回收；
- prefix cache 作为第三阶段功能，先不与 paged allocation 同时引入复杂引用计数。

## 4. Batch 与调度

### 4.1 Continuous batching

每个 decode step：

1. 从 waiting queue 取请求，直到达到 token/显存预算；
2. 将新请求做 prefill 或 chunked prefill；
3. 对 running batch 执行一次 decode；
4. 采样、追加 token、更新 KV block 和长度；
5. 完成的请求移出 batch，释放 block 并返回结果；
6. 记录本轮 batch size、活跃 token 数和排队时间。

batch 上限不应只看 request 数，建议同时限制：`max_batch_size`、`max_num_batched_tokens`、KV 剩余 block 数和单轮最大耗时。

### 4.2 调度策略

第一版使用 FIFO + token budget，之后比较：

- FCFS：延迟稳定、容易解释；
- shortest-prompt-first：降低短请求 TTFT，但可能饿死长请求；
- prefill/decode 分离队列：提高 decode 稳定性；
- chunked prefill：控制长 prompt 对 decode 的干扰。

所有策略必须在相同请求混合分布下比较，不要只报告单一 batch size 的峰值。

## 5. Triton Kernel 路线

不要同时写十个 kernel。每一步先有 PyTorch reference，再写 Triton 版本，再做误差和性能对照。

### Phase 1：低风险融合

1. **RMSNorm**：load hidden、计算平方均值、rsqrt、scale，一次 kernel 完成，减少中间张量；
2. **RoPE**：融合 q/k 旋转，避免独立 cosine/sine 和临时 tensor；
3. **SwiGLU**：`gate * silu(up)` 融合，减少两次读写；
4. **Add/RMSNorm**：在 residual 路径中融合 residual add 与 norm。

重点参数：`BLOCK_SIZE`、warp 数、fp16/bf16 累加精度、非整齐 hidden size 的 mask。

### Phase 2：量化线性层

推荐顺序：

1. FP16/BF16 权重 + Triton matmul baseline；
2. INT8 weight-only：权重按 per-channel 或 group-wise scale 保存，kernel 内 dequant 后累加；
3. INT8 W8A8：激活动态量化，权重和激活均 int8，累加 int32 后缩放；
4. FP8 只在显卡/软件栈支持和精度数据充分时尝试。

第一版优先做 weight-only INT8，原因是实现和误差分析简单、显存收益明显；不要一开始声称完成 GPTQ/AWQ，除非实现了对应校准和权重格式转换。

量化需要明确：group size、scale/zero-point 布局、异常值处理、累加 dtype、反量化位置和权重打包格式。每个 kernel 都要和 dequant + `torch.matmul` reference 对齐。

### Phase 3：Attention

- **Prefill**：先调用 PyTorch SDPA/FlashAttention 作为 baseline；
- **Decode contiguous attention**：实现单 query 对历史 KV 的 online softmax；
- **Decode paged attention**：加入 block table，按逻辑 token 映射物理 KV block；
- **FlashAttention 思路**：Q/K/V 分块加载到 SRAM，在线维护 `m_i/l_i`，不落地完整 attention matrix；
- causal mask、不同序列长度、GQA/MQA、head_dim 变化都要单独测试。

不要把“调用 `torch.scaled_dot_product_attention`”写成自己实现 FlashAttention；应该明确它是 baseline，Triton 版本只覆盖自己真正实现并验证的路径。

### Phase 4：融合 Model Block

在 kernel 级收益稳定后，再考虑把 `RMSNorm -> QKV projection -> RoPE` 或 `gate/up -> activation -> down` 做更大范围融合。融合过大可能降低 occupancy、增加寄存器压力，必须用 Nsight Compute 验证，而不是凭 kernel 数量判断更快。

## 6. CUDA 加速与工程细节

- 使用独立 `prefill_stream` 和 `decode_stream` 时，必须用 CUDA event 表达 KV/权重依赖，不能只依赖 Python 调用顺序；
- host scheduler 不要每 token 同步 `cuda.synchronize()`；用 event 查询或异步流水；
- warmup 后固定 shape/配置，避免把 Triton 首次编译时间算进推理延迟；
- 对固定 batch/shape 可尝试 CUDA Graph，但动态 batch 和请求进出会增加 graph capture 管理复杂度，放到后期；
- 记录 H2D、kernel、D2H 和 scheduler 时间，区分 GPU 计算瓶颈与 Python 调度瓶颈；
- RTX 3060 Ti 只有 8GB，优先保证 KV 和权重不溢出，再追求更高 batch。

## 7. 验证体系

### 7.1 Kernel 正确性

每个 kernel 对比 PyTorch reference：

- `torch.testing.assert_close`，按 dtype 设置 `rtol/atol`；
- 随机 shape + 边界 shape：1、非 32 对齐、长序列、空 mask、不同 head_dim；
- NaN/Inf、极端 logits、全 padding 和结束请求；
- 多 stream/重复运行检查数据竞争；
- 量化 kernel 额外比较 dequant 误差、逐层误差和最终 logits。

### 7.2 模型级一致性

- greedy decode 与 HuggingFace reference 逐 token 一致或在明确容差内一致；
- 固定 seed 下比较 logits、生成 token、停止条件和 batch 内不同长度请求；
- 对 50–100 条固定 prompt 做短序列和长序列回归；
- 量化模型比较 perplexity、准确率或任务集合，不只看单个输出样例。

### 7.3 性能指标

固定硬件、驱动、CUDA、PyTorch、Triton、模型权重和温度后报告：

| 层级 | 指标 |
| --- | --- |
| Kernel | latency P50/P95、有效带宽、TFLOPS、occupancy、寄存器/共享内存 |
| Prefill | TTFT、prompt tokens/s、不同 prompt length 的吞吐 |
| Decode | TPOT、decode tokens/s、batch size 曲线 |
| 服务 | request/s、output tokens/s、P50/P95/P99、排队时间、显存峰值 |
| 资源 | 权重显存、KV 显存、workspace、GPU 利用率、CPU 调度占比 |
| 质量 | perplexity、固定任务准确率、量化前后差异 |

### 7.4 Baseline 矩阵

至少保留以下对照：

1. HuggingFace/PyTorch eager FP16；
2. PyTorch SDPA/FlashAttention + 连续 KV；
3. Triton fused kernel + 连续 KV；
4. Triton fused kernel + paged KV + continuous batching；
5. vLLM 或 TensorRT-LLM 作为外部参考（同模型、同输入分布）。

每次只改变一个因素，并同时报告质量、吞吐、TTFT、TPOT、P95 和显存，避免只挑选最有利的单一指标。

## 8. 推荐里程碑

### M0：正确性骨架

HF 模型单请求 greedy decode，抽象 `ModelRunner`、`Sequence`、`Scheduler`、`KVCache` 接口；保存 logits/token 对照。

### M1：Triton kernel MVP

完成 RMSNorm、RoPE、SwiGLU 三个 kernel，加入 microbenchmark 和随机正确性测试；至少证明 kernel latency 或中间 tensor 数量有改善。

### M2：Prefill/Decode Runtime

拆分 prefill/decode，加入固定 batch 和异步 CUDA stream；报告 TTFT、TPOT、GPU kernel 时间线。

### M3：Paged KV + Continuous batching

实现 block pool、block table、free list、请求完成回收和 FIFO/token budget 调度；加入不同长度混合请求测试。

### M4：量化

实现 INT8 weight-only 权重格式、scale 管理和 Triton dequant matmul；完成逐层/模型级精度评估。

### M5：Attention 优化

实现 paged decode attention；prefill 使用 SDPA/FlashAttention baseline，后续再替换为 Triton FlashAttention 路径并做 Nsight 分析。

### M6：服务化与对照

FastAPI 流式输出、Prometheus 风格指标、vLLM/TensorRT-LLM 对照、固定 benchmark 报告和简历指标。

## 9. 可写进简历的结果句式

在没有实测数字前，只写：

> 基于 PyTorch/HuggingFace 搭建单卡 LLM 推理 Runtime，使用 Triton 实现 fused RMSNorm、SwiGLU、INT8 weight-only matmul 与 paged decode attention，设计 prefill/decode 分离、continuous batching 和 block-based KV cache，并建立 kernel、模型一致性和端到端吞吐评测。

有真实对照后再替换为：

> 针对 decode 阶段 KV 访存和 kernel launch 开销，设计 paged KV cache 与 continuous batching，并以 Triton 融合 RMSNorm/SwiGLU、量化 matmul 和 paged attention；在固定模型与请求分布下，相比 [baseline] 将 decode 吞吐提升 [X%]、P95 TPOT 降低 [Y%]、峰值显存降低 [Z%]，生成质量/perplexity 变化控制在 [阈值] 内。

## 10. 面试必须能回答的问题

1. 为什么 prefill 和 decode 要分开调度？
2. paged KV 如何映射逻辑 token 到物理 block？请求结束时何时能安全回收？
3. continuous batching 如何处理不同长度请求和新请求加入？
4. Triton kernel 融合减少了哪些 global memory traffic，为什么可能反而降低 occupancy？
5. weight-only INT8 和 W8A8 的误差、带宽和计算路径有什么差异？
6. FlashAttention 如何通过 online softmax 避免写出完整 attention matrix？
7. 如何证明端到端提升来自 kernel，而不是 batch、输入长度或 warmup 差异？
8. 为什么 RTX 3060 Ti 上先做 0.5B–1.5B 模型，而不是直接上 7B？
