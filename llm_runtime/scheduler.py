from __future__ import annotations

from collections import deque

from .request import GenerationRequest, RequestState


class FIFOScheduler:
    """FIFO continuous-batching scheduler with request/token admission limits."""

    def __init__(self, max_batch_size: int = 8, max_num_batched_tokens: int = 4096):
        if max_batch_size <= 0 or max_num_batched_tokens <= 0:
            raise ValueError("scheduler limits must be positive")
        self.max_batch_size = max_batch_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[GenerationRequest] = deque()
        self.running: dict[str, GenerationRequest] = {}

    def add_request(self, request: GenerationRequest) -> None:
        if request.request_id in self.running or any(r.request_id == request.request_id for r in self.waiting):
            raise ValueError(f"duplicate request id: {request.request_id}")
        request.state = RequestState.WAITING
        self.waiting.append(request)

    def next_batch(self) -> list[GenerationRequest]:
        batch = list(self.running.values())
        budget = sum(max(1, r.prompt_len if not r.generated_ids else 1) for r in batch)
        while self.waiting and len(batch) < self.max_batch_size:
            request = self.waiting[0]
            cost = max(1, request.prompt_len)
            if batch and budget + cost > self.max_num_batched_tokens:
                break
            self.waiting.popleft()
            request.state = RequestState.RUNNING
            batch.append(request)
            self.running[request.request_id] = request
            budget += cost
        return batch

    def finish(self, request: GenerationRequest) -> None:
        self.running.pop(request.request_id, None)

    def cancel(self, request_id: str) -> GenerationRequest | None:
        request = self.running.pop(request_id, None)
        if request is None:
            for candidate in self.waiting:
                if candidate.request_id == request_id:
                    self.waiting.remove(candidate)
                    request = candidate
                    break
        if request is not None:
            request.state = RequestState.CANCELLED
        return request
