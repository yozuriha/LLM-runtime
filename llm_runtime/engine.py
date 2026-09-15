from __future__ import annotations

import torch
from time import monotonic

from .kv_cache import ContinuousKVCache
from .request import GenerationRequest
from .runner import ModelRunner
from .scheduler import FIFOScheduler


class GenerationEngine:
    """PyTorch 参考路径的 greedy continuous-batching Engine。

    当前实现以清晰和可验证为优先：控制面支持连续加入请求，
    但底层 Runner 仍按请求调用，后续可替换为真正的张量级 batch forward。
    """

    def __init__(self, runner: ModelRunner, max_batch_size: int = 8, max_num_batched_tokens: int = 4096,
                 max_cache_tokens: int = 1_000_000):
        self.runner = runner
        self.scheduler = FIFOScheduler(max_batch_size, max_num_batched_tokens)
        self.kv_cache = ContinuousKVCache(max_cache_tokens)

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
        for request in batch:
            if not request.prefilled:
                # 首次执行处理整个 prompt，并把模型返回的 KV 放入连续 cache。
                request.started_at = request.started_at or monotonic()
                logits = self.runner.prefill(request)
                request.prefilled = True
                self.kv_cache.allocate(request.request_id, request.prompt_len, request.past_key_values)
            else:
                # 后续每轮只送入上轮生成的 token，避免重复计算 prompt。
                logits = self.runner.decode(request, request.next_token)
                # append_token 尚未执行，因此这里要为即将生成的 token 预留一格。
                self.kv_cache.update(request.request_id, request.seq_len + 1, request.past_key_values)
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
