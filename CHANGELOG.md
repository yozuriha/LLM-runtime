# Decoder Runtime 修改记录

本文件记录 `decoder-runtime` 每次实现变更的内容、原因和验证结果。性能数字只在对应硬件和固定 benchmark 实测后填写。

## 2026-09-18：Paged KV Cache 与 Qwen2 M1 Kernel 接入

### 修改内容

- 新增 `llm_runtime.kv_cache.PagedKVCache`：固定 token block、free-list、每请求 block table、按页保存各层 KV、扩容/回收及 dense materialize。
- `GenerationEngine` 默认使用 `PagedKVCache`；decode 前从 page storage 恢复当前请求的 KV，decode 后重新写回分页存储。
- 新增 `llm_runtime.triton_integration`：在 CUDA Qwen2 模型上替换 Qwen2RMSNorm、SwiGLU MLP 和 half-split RoPE；不满足条件时保留 PyTorch fallback。
- `PyTorchCausalLMRunner` 增加 `use_triton_kernels` 开关；Engine 增加 `kv_cache` 和 `kv_block_size` 参数，便于回归和替换实现。
- 增加分页分配、materialize、扩容和释放测试。

### 设计边界

当前 PagedKVCache 是 software paged storage。标准 HuggingFace attention 仍要求 dense `past_key_values`，所以 decode 暂时会 materialize；真正避免拼接需要下一阶段的 paged-attention Triton kernel 和 block table 输入接口。

### 验证

```text
PYTHONPATH=decoder-runtime conda run -n triton pytest -q -p no:cacheprovider decoder-runtime/tests
19 passed, 3 skipped
```

当前环境无 CUDA，Qwen2 Triton forward 只完成接入和 CPU fallback 验证，RTX 4070 上的模型级 logits/性能仍需实测。

### 后续修正：Qwen2 GQA RoPE

- `apply_rope_embeddings()` 将 Q 与 K 拆成两个 Triton launch，分别使用 Q head 数和 KV head 数。
- 兼容本地 Qwen2 配置的 `12` 个 query heads 与 `2` 个 key/value heads，避免 GQA 场景错误按相同 head 数寻址。
- 保持 `cos/sin=[batch, sequence, head_dim]` 的 Qwen half-split 旋转布局不变。

### 修正原因

Qwen2 使用 grouped-query attention，Q/K 的 head 数不同。原先共用一个 launch 会导致 K 的 head 索引越界或漏算，必须按实际 tensor head 数分别计算 program grid。

### 验证

通过 CPU fallback、Transformers 5.16.1 小型 Qwen2 prefill/decode smoke test，以及完整测试集；CUDA kernel 仍需在 RTX 4070 上实测。

## 2026-09-18：补充 M1 Kernel 中文注释

### 修改内容

- `llm_runtime/kernels/rmsnorm.py`：说明 row-to-program 映射、hidden 维度 mask、FP32 归约、RMSNorm 公式和 warp 配置。
- `llm_runtime/kernels/rope.py`：说明 `[B, H, T, D]` 索引计算、half-split layout、position id 查表和 Q/K 二维旋转公式。
- `llm_runtime/kernels/swiglu.py`：说明一维 block 映射、尾部 mask、FP32 中间计算和 `SiLU(gate) * up` 融合关系。
- 为三个 reference/wrapper 函数补充中文 docstring，明确输入布局、数值精度和 kernel 约束。

### 修改原因

M1 kernel 同时包含 Python wrapper 和 Triton DSL。补充关键寻址、精度和布局注释，便于后续调参、排查数值误差以及将算子接入 Transformer block；本次只增加注释，不改变 kernel 接口和计算逻辑。

### 验证

```text
PYTHONPATH=decoder-runtime conda run -n triton pytest -q -p no:cacheprovider decoder-runtime/tests/test_kernels.py
13 passed, 3 skipped
```

## 2026-09-18：拆分 M1 Kernel 与 Benchmark

### 修改内容

- 将 M1 算子拆分为 `kernels/rmsnorm.py`、`kernels/rope.py` 和 `kernels/swiglu.py`，每个文件独立包含 Triton kernel、CUDA wrapper 与 PyTorch reference。
- 将 `kernels/fused.py` 保留为兼容导出层，避免已有 import 失效；`kernels/__init__.py` 改为直接从独立模块导出。
- `benchmarks/benchmark_m1.py` 改用 `triton.testing.Benchmark`，分别比较 PyTorch 与 Triton provider。
- benchmark 增加最大绝对误差、allocator 峰值显存 MiB、峰值显存占总显存百分比；硬件目标统一记录为 RTX 4070（12 GB，SM89）。

### 修改原因

独立文件让算子可以单独迭代、测试和替换，减少一个融合文件中的耦合。使用 Triton 官方 Benchmark 统一 latency 采样，同时保留 correctness 和显存指标，避免只看速度而遗漏数值或内存回归。

### 验证

当前环境 `torch.cuda.is_available() == False`，因此无法实际 launch GPU kernel 或填写 RTX 4070 实测数字；已完成模块导入、CPU reference 和静态检查。GPU 环境运行 `benchmarks/benchmark_m1.py` 后会打印完整表格。

## 2026-09-18：M1 基础融合 Kernel

### 背景

M0 的 Runner 已能批量组织模型输入，但 RMSNorm、RoPE 和 SwiGLU 仍由 PyTorch 分步执行。M1 先建立独立的 kernel 契约，不直接改 Qwen forward，避免 kernel 正确性问题和模型集成问题混在一起。

### 修改内容

- `llm_runtime/kernels/rmsnorm.py`、`rope.py`、`swiglu.py`
  - 分别实现 fused RMSNorm、非交错布局 RoPE 和 fused SwiGLU；每个 kernel 提供 CUDA 输入校验和 PyTorch reference 实现。

- `tests/test_kernels.py`
  - 覆盖 FP16/BF16/FP32、非对齐 hidden size、变长 position ids 和非整齐元素数。
  - CUDA 可用时对比 Triton 与 reference；无 CUDA 时自动跳过 GPU launch 测试。

- `benchmarks/benchmark_m1.py`
  - 使用 `triton.testing.Benchmark`，在 warmup 后报告 RMSNorm、RoPE、SwiGLU 的 latency；另输出误差和显存指标。

### 修改原因

这三个算子都是 Transformer block 中低风险的逐元素/归约热点，适合先验证 Triton 的 mask、FP32 累加、layout 和 launch 参数。暂不把它们接到 Qwen Runner，避免没有模型级 logits 回归时误报端到端收益。

### 验证

```text
PYTHONPATH=decoder-runtime conda run -n triton pytest -q -p no:cacheprovider decoder-runtime/tests
```

当前无 CUDA：M1 reference 与 runtime 测试通过，GPU kernel 测试自动 skip；RTX 4070 上的 latency 和有效带宽需要运行 `benchmarks/benchmark_m1.py` 后填写。

### 已知限制与下一步

- RoPE 当前实现非交错 half-split 布局；接入 Qwen 前需确认模型 rotary embedding 的 layout 与 cos/sin 形状。
- RMSNorm 当前 hidden size 上限为 8192，后续可按 Qwen hidden size 和寄存器压力调整。
- 下一步先在 RTX 4070 上跑 kernel correctness/microbenchmark，再接入单层 Transformer forward 做逐层 logits 对照。

## 2026-09-18：兼容 Transformers 5.x Cache

### 背景

真实 Qwen 推理使用 Transformers 5.16.1 时，模型返回的是新版 `Cache` 对象。该对象没有旧版 `to_legacy_cache()`，原 Runner 因此在批量 prefill 拆分 KV 时抛出 `TypeError`。

### 修改内容

- `llm_runtime/runner.py`
  - `_legacy_past()` 新增 `Cache.layers[i].keys/values` 解析。
  - 保留旧版 `to_legacy_cache()`、tuple 以及 `key_cache/value_cache` 兼容路径。
  - 边界解析统一使用 `(key, value)` tensor tuple；请求状态在 Transformers 5 环境中保留可变 Cache 对象，确保下一次 Qwen forward 仍可调用 `Cache.update()`。

- `tests/test_runtime.py`
  - 增加 Transformers 5 风格 `Cache.layers` fake 对象测试。

### 修改原因

Transformers 5 移除了旧 Cache 的转换方法，但 Qwen 的模型 forward 仍会返回 Cache 对象。Runner 必须在边界处吸收 Transformers API 变化，不能把版本差异泄露给调度器和 KV cache。

### 验证

```text
PYTHONPATH=decoder-runtime conda run -n triton pytest -q -p no:cacheprovider decoder-runtime/tests
5 passed
```

另外使用 Transformers 5.16.1 的真实 `Qwen2ForCausalLM` 小配置完成 CPU smoke test：批量 prefill 后每个请求持有 `DynamicCache`，历史长度为 `[3, 2]`；批量 decode 后为 `[4, 3]`，重建后的 Cache 可再次传入 Qwen forward。

## 2026-09-17：补齐批量 Prefill/Decode 接口

### 背景

原 baseline 虽然有 `FIFOScheduler` 和 `GenerationEngine`，但 `step()` 会对 batch 中每个请求分别调用一次 `prefill()` 或 `decode()`。这只是控制面上的 batching，模型 forward 仍是逐请求执行，无法作为后续 Triton kernel、Paged KV 和 continuous batching 的张量入口。

### 修改内容

- `llm_runtime/runner.py`
  - 在 `ModelRunner` 增加 `prefill_batch(requests)` 和 `decode_batch(requests, token_ids)`。
  - 默认实现通过单请求方法 fallback，保持已有自定义 Runner 的兼容性。
  - `PyTorchCausalLMRunner.prefill_batch()` 将变长 prompt padding 后一次调用 HuggingFace 模型，按有效长度提取每个请求的 next-token logits。
  - 增加 KV cache 的 legacy tuple 归一化、batch stack 和按真实长度拆分逻辑，兼容 HuggingFace tuple cache 以及提供 `to_legacy_cache()` 的 cache 对象。
  - `decode_batch()` 将不同历史长度的 KV padding 到同一长度，通过 `attention_mask` 屏蔽 padding，并保留每个请求新追加的 KV。

- `llm_runtime/engine.py`
  - `step()` 将请求拆成 prefill/decode 两组，每组只调用一次批量 Runner 接口。
  - 保留原有 greedy sampling、请求完成回收和连续 KV bookkeeping 行为。
  - 修正兼容 Runner 的 `prefilled` 状态写回。

- `tests/test_runtime.py`
  - 增加批量入口调用计数，验证两个请求会共享一次 prefill batch 和一次 decode batch。
  - 增加 CPU fake causal LM，覆盖不同 prompt 长度下的 padding、KV 拆分和 decode 追加 token。

- `README.md`
  - 增加批量接口、变长 padding/KV 拆分策略和自定义 Runner fallback 说明。

### 修改原因

先建立稳定的 batch contract，才能在不改 Engine 调度接口的情况下逐步替换 RMSNorm、RoPE、SwiGLU、Paged Attention 等 Triton 热点。默认 fallback 让 correctness baseline 和实验 Runner 可以并行迁移。

### 验证

```text
PYTHONPATH=decoder-runtime conda run -n triton pytest -q -p no:cacheprovider decoder-runtime/tests
4 passed
```

当前环境检测到 PyTorch 2.13.0、Triton 3.7.1，但 `torch.cuda.is_available()` 为 `False`，因此本次只完成 CPU 控制面测试，尚未在 RTX 4070 上做模型级 logits/性能验证。

### 已知限制与下一步

- 当前 decode 为了适配变长历史，会临时 padding/stack KV；Paged KV 完成后应直接让 attention kernel读取 block table，避免这一步的复制开销。
- 仍由 HuggingFace Transformer 完成 QKV、attention 和 MLP，批量接口建立后再接 Triton fused kernel。
- 需要在 RTX 4070 上用本地 Qwen 权重补充：单请求一致性、变长 batch 一致性、TTFT/TPOT、峰值显存和 batch size 曲线。

## 后续记录模板

```text
## YYYY-MM-DD：标题

### 背景
### 修改内容
### 修改原因
### 验证
### 已知限制与下一步
```
