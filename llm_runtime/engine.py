from __future__ import annotations

import torch
from time import monotonic

from .kv_cache import ContinuousKVCache
from .request import GenerationRequest
from .runner import ModelRunner
from .scheduler import FIFOScheduler


class GenerationEngine:
    """Greedy continuous-batching engine for the PyTorch reference path."""

    def __init__(self, runner: ModelRunner, max_batch_size: int = 8, max_num_batched_tokens: int = 4096,
                 max_cache_tokens: int = 1_000_000):
        self.runner = runner
        self.scheduler = FIFOScheduler(max_batch_size, max_num_batched_tokens)
        self.kv_cache = ContinuousKVCache(max_cache_tokens)

    def add_request(self, request: GenerationRequest) -> None:
        if not request.input_ids:
            raise ValueError("input_ids must not be empty")
        if request.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.scheduler.add_request(request)

    @staticmethod
    def _sample_greedy(logits: torch.Tensor) -> int:
        return int(torch.argmax(logits).item())

    def step(self) -> list[GenerationRequest]:
        batch = self.scheduler.next_batch()
        completed = []
        for request in batch:
            if not request.prefilled:
                request.started_at = request.started_at or monotonic()
                logits = self.runner.prefill(request)
                request.prefilled = True
                self.kv_cache.allocate(request.request_id, request.prompt_len, request.past_key_values)
            else:
                logits = self.runner.decode(request, request.next_token)
                self.kv_cache.update(request.request_id, request.seq_len + 1, request.past_key_values)
            request.append_token(self._sample_greedy(logits))
            if request.finished:
                self.kv_cache.release(request.request_id)
                self.scheduler.finish(request)
                completed.append(request)
        return completed

    def run(self) -> list[GenerationRequest]:
        completed = []
        while self.scheduler.waiting or self.scheduler.running:
            completed.extend(self.step())
        return completed
