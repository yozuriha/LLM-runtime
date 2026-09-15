from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Entry:
    """一个请求的连续 KV 记录；``value`` 通常是 HF past_key_values。"""

    request_id: str
    num_tokens: int
    value: object | None = None


class ContinuousKVCache:
    """Bookkeeping for per-request contiguous PyTorch past-key-values.

    The cache deliberately keeps the model's native ``past_key_values`` object.
    This makes it a reliable baseline before introducing a device-side paged pool.
    ``max_tokens`` is a hard admission limit and catches accidental unbounded use.
    """

    def __init__(self, max_tokens: int):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self._entries: dict[str, _Entry] = {}
        self._used_tokens = 0

    @property
    def used_tokens(self) -> int:
        """当前所有请求占用的逻辑 token 容量。"""
        return self._used_tokens

    @property
    def free_tokens(self) -> int:
        """还可以接纳的逻辑 token 容量。"""
        return self.max_tokens - self._used_tokens

    def allocate(self, request_id: str, num_tokens: int, value: object | None = None) -> None:
        """为新请求分配 prompt 对应的连续 KV 容量。"""
        if request_id in self._entries:
            raise ValueError(f"request {request_id!r} already has a KV allocation")
        if num_tokens < 0 or num_tokens > self.free_tokens:
            raise MemoryError(f"KV cache capacity exceeded: requested {num_tokens}, free {self.free_tokens}")
        self._entries[request_id] = _Entry(request_id, num_tokens, value)
        self._used_tokens += num_tokens

    def update(self, request_id: str, num_tokens: int, value: object | None = None) -> None:
        """将已有请求扩容到新的序列长度，并更新 cache 对象。"""
        entry = self._entries[request_id]
        # delta 可以为负数，允许上层在回滚或修正长度时归还容量。
        delta = num_tokens - entry.num_tokens
        if delta > self.free_tokens:
            raise MemoryError(f"KV cache capacity exceeded: requested {delta}, free {self.free_tokens}")
        entry.num_tokens = num_tokens
        if value is not None:
            entry.value = value
        self._used_tokens += delta

    def get(self, request_id: str) -> object | None:
        """取出请求关联的模型原生 KV 对象。"""
        return self._entries[request_id].value

    def release(self, request_id: str) -> None:
        """释放请求占用的容量；重复释放视为幂等操作。"""
        entry = self._entries.pop(request_id, None)
        if entry is not None:
            self._used_tokens -= entry.num_tokens

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._entries
