from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch


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


@dataclass(frozen=True)
class _CacheTemplate:
    """用于重建 Transformers Cache 的轻量模板，不持有原始 KV tensor。"""

    cache_cls: type
    is_cache_object: bool = True


@dataclass
class _PagedEntry:
    request_id: str
    num_tokens: int
    block_ids: list[int]
    template: _CacheTemplate | None = None


class PagedKVCache:
    """按固定 token block 管理 KV 的分页缓存。

    该类先提供与 HuggingFace Cache 兼容的 software paged storage：每个请求
    获得一组 block id，KV 按 block_size 沿 sequence 维保存。标准 HF attention
    仍然只接受连续 KV，因此 ``get`` 会 materialize 成 layer tuple；后续接入
    paged-attention Triton kernel 后，可以直接消费 ``block_table``，省掉这一步。
    """

    def __init__(self, max_tokens: int, block_size: int = 16):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.max_tokens = max_tokens
        self.block_size = block_size
        self.max_blocks = ceil(max_tokens / block_size)
        self._free_blocks = list(range(self.max_blocks - 1, -1, -1))
        self._entries: dict[str, _PagedEntry] = {}
        # 物理页按 block id 索引；每页保存所有 layer 的 (K, V) chunk。
        self._page_data: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self._used_tokens = 0

    @property
    def used_tokens(self) -> int:
        return self._used_tokens

    @property
    def free_tokens(self) -> int:
        return self.max_tokens - self._used_tokens

    @property
    def used_blocks(self) -> int:
        return self.max_blocks - len(self._free_blocks)

    @property
    def free_blocks(self) -> int:
        return len(self._free_blocks)

    def _required_blocks(self, num_tokens: int) -> int:
        return ceil(num_tokens / self.block_size) if num_tokens else 0

    def _take_blocks(self, count: int) -> list[int]:
        if count > len(self._free_blocks):
            raise MemoryError(f"paged KV block pool exhausted: requested {count}, free {len(self._free_blocks)}")
        return [self._free_blocks.pop() for _ in range(count)]

    def _return_blocks(self, block_ids: list[int]) -> None:
        self._free_blocks.extend(block_ids)

    @staticmethod
    def _legacy_pairs(past):
        if isinstance(past, tuple):
            return tuple(past)
        to_legacy = getattr(past, "to_legacy_cache", None)
        if to_legacy is not None:
            return tuple(to_legacy())
        layers = getattr(past, "layers", None)
        if layers is not None:
            pairs = []
            for layer in layers:
                key = getattr(layer, "keys", None)
                value = getattr(layer, "values", None)
                if key is not None and value is not None:
                    pairs.append((key, value))
            if pairs:
                return tuple(pairs)
        key_cache = getattr(past, "key_cache", None)
        value_cache = getattr(past, "value_cache", None)
        if key_cache is not None and value_cache is not None:
            return tuple(zip(key_cache, value_cache))
        raise TypeError("paged KV requires tuple or a supported Transformers Cache")

    def _write_pages(self, entry: _PagedEntry, value) -> None:
        if value is None:
            return
        pairs = self._legacy_pairs(value)
        layer_chunks: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for key, val in pairs:
            if key.shape[2] < entry.num_tokens or val.shape[2] < entry.num_tokens:
                raise ValueError("KV tensor sequence length is shorter than the cache entry")
            layer_pages = []
            for start in range(0, entry.num_tokens, self.block_size):
                end = min(start + self.block_size, entry.num_tokens)
                layer_pages.append((
                    key[..., start:end, :].contiguous(),
                    val[..., start:end, :].contiguous(),
                ))
            layer_chunks.append(layer_pages)
        for page_index, block_id in enumerate(entry.block_ids):
            self._page_data[block_id] = [chunks[page_index] for chunks in layer_chunks]
        entry.template = None if isinstance(value, tuple) else _CacheTemplate(type(value))

    def allocate(self, request_id: str, num_tokens: int, value=None) -> None:
        """为请求分配 block，并可选地把已有 HF KV 写入分页存储。"""
        if request_id in self._entries:
            raise ValueError(f"request {request_id!r} already has a KV allocation")
        if num_tokens < 0 or num_tokens > self.free_tokens:
            raise MemoryError(f"KV cache capacity exceeded: requested {num_tokens}, free {self.free_tokens}")
        entry = _PagedEntry(request_id, num_tokens, self._take_blocks(self._required_blocks(num_tokens)))
        self._entries[request_id] = entry
        self._used_tokens += num_tokens
        self._write_pages(entry, value)

    def update(self, request_id: str, num_tokens: int, value=None) -> None:
        """调整请求的逻辑长度，必要时扩展或归还尾部 block。"""
        entry = self._entries[request_id]
        old_num_tokens = entry.num_tokens
        if num_tokens < 0 or num_tokens > self.max_tokens:
            raise ValueError("invalid KV sequence length")
        if num_tokens > entry.num_tokens and num_tokens - entry.num_tokens > self.free_tokens:
            raise MemoryError("KV cache capacity exceeded")
        old_blocks = len(entry.block_ids)
        new_blocks = self._required_blocks(num_tokens)
        if new_blocks > old_blocks:
            entry.block_ids.extend(self._take_blocks(new_blocks - old_blocks))
        elif new_blocks < old_blocks:
            released = entry.block_ids[new_blocks:]
            for block_id in released:
                self._page_data.pop(block_id, None)
            self._return_blocks(released)
            del entry.block_ids[new_blocks:]
        entry.num_tokens = num_tokens
        self._used_tokens += num_tokens - old_num_tokens
        self._write_pages(entry, value)

    def get(self, request_id: str):
        """将请求页拼接为 HuggingFace 可读取的 ``((K,V), ...)``。"""
        entry = self._entries[request_id]
        if not entry.block_ids or entry.block_ids[0] not in self._page_data:
            return None
        result = []
        num_layers = len(self._page_data[entry.block_ids[0]])
        for layer_index in range(num_layers):
            layer_pages = [self._page_data[block_id][layer_index] for block_id in entry.block_ids]
            keys, values = zip(*layer_pages) if layer_pages else ((), ())
            if not keys:
                result.append((torch.empty(0), torch.empty(0)))
                continue
            result.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
        pairs = tuple(result)
        if entry.template is None:
            return pairs
        # Transformers 5 的 Qwen forward 要求可变 Cache；分页 materialize
        # 后恢复同类对象，失败时使用 DynamicCache 作为稳定公共实现。
        try:
            return entry.template.cache_cls(ddp_cache_data=pairs)
        except (TypeError, ValueError):
            try:
                from transformers.cache_utils import DynamicCache
            except ImportError:
                return pairs
            return DynamicCache(ddp_cache_data=pairs)

    def template(self, request_id: str) -> _CacheTemplate | None:
        return self._entries[request_id].template

    def block_table(self, request_id: str) -> tuple[int, ...]:
        """返回 paged-attention kernel 使用的逻辑 block table。"""
        return tuple(self._entries[request_id].block_ids)

    def release(self, request_id: str) -> None:
        entry = self._entries.pop(request_id, None)
        if entry is not None:
            for block_id in entry.block_ids:
                self._page_data.pop(block_id, None)
            self._return_blocks(entry.block_ids)
            self._used_tokens -= entry.num_tokens

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._entries
