from __future__ import annotations

import torch
from time import monotonic

from .kv_cache import ContinuousKVCache, PagedKVCache
from .request import GenerationRequest
from .runner import ModelRunner
from .scheduler import FIFOScheduler


class GenerationEngine:
    """PyTorch 参考路径的 greedy continuous-batching Engine。

    控制面支持连续加入请求；Runner 负责将同一阶段的请求组织成一个
    张量 batch，旧的自定义 Runner 仍可通过 ModelRunner 的兼容实现运行。
    """

    def __init__(self, runner: ModelRunner, max_batch_size: int = 8, max_num_batched_tokens: int = 4096,
                 max_cache_tokens: int = 1_000_000, kv_cache=None, kv_block_size: int = 16):
        self.runner = runner
        self.scheduler = FIFOScheduler(max_batch_size, max_num_batched_tokens)
        self.kv_cache = kv_cache or PagedKVCache(max_cache_tokens, block_size=kv_block_size)

    def add_request(self, request: GenerationRequest) -> None:
        """校验并提交请求；实际执行由 ``step``/``run`` 驱动。"""
        if not request.input_ids:
            raise ValueError("input_ids must not be empty")
        if request.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.scheduler.add_request(request)

    @staticmethod
    def _sample_greedy(logits: torch.Tensor) -> int:
        """选择 logit 最大的 token，不引入随机数，便于回归对比。"""
        return int(torch.argmax(logits).item())

    def step(self) -> list[GenerationRequest]:
        """推进一轮 prefill/decode，并返回本轮刚完成的请求。"""
        batch = self.scheduler.next_batch()
        completed = []
        prefill = [request for request in batch if not request.prefilled]
        decode = [request for request in batch if request.prefilled]
        logits_by_id = {}
        if prefill:
            for request in prefill:
                request.started_at = request.started_at or monotonic()
            logits = self.runner.prefill_batch(prefill)
            for request, row in zip(prefill, logits):
                request.prefilled = True
                self.kv_cache.allocate(request.request_id, request.prompt_len, request.past_key_values)
                logits_by_id[request.request_id] = row
        if decode:
            # PagedKVCache 是 decode 的 KV 来源；标准 HF attention 暂时通过
            # get() materialize 成连续 Cache，后续 paged-attention kernel 可直接
            # 消费 block_table，去掉这次拼接。
            for request in decode:
                if hasattr(self.kv_cache, "get") and request.request_id in self.kv_cache:
                    paged_past = self.kv_cache.get(request.request_id)
                    if paged_past is not None:
                        request.past_key_values = paged_past
            logits = self.runner.decode_batch(decode, [request.next_token for request in decode])
            for request, row in zip(decode, logits):
                # decode 输入的是上轮已生成 token，返回的 KV 长度等于当前 seq_len。
                self.kv_cache.update(request.request_id, request.seq_len, request.past_key_values)
                logits_by_id[request.request_id] = row
        for request in batch:
            logits = logits_by_id[request.request_id]
            request.append_token(self._sample_greedy(logits))
            if request.finished:
                # 请求完成后立即归还容量，允许 waiting 队列中的请求进入。
                self.kv_cache.release(request.request_id)
                self.scheduler.finish(request)
                completed.append(request)
        return completed

    def run(self) -> list[GenerationRequest]:
        """持续推进直到 waiting 和 running 队列都为空。"""
        completed = []
        while self.scheduler.waiting or self.scheduler.running:
            completed.extend(self.step())
        return completed
