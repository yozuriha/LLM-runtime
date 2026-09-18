from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .request import GenerationRequest


class ModelRunner(ABC):
    """模型执行抽象，隔离 Engine 与具体模型/Kernel 实现。"""

    @abstractmethod
    def prefill(self, request: GenerationRequest) -> torch.Tensor:
        """Run prompt tokens and return logits for the next token."""

    @abstractmethod
    def decode(self, request: GenerationRequest, token_id: int) -> torch.Tensor:
        """Run one token using the request's cached past and return logits."""

    def prefill_batch(self, requests: list[GenerationRequest]) -> torch.Tensor:
        """Run a batch of prompts.

        The default implementation keeps old custom runners source-compatible;
        GPU runners should override this method to obtain a real batched forward.
        The returned tensor has shape ``[batch, vocab]``.
        """
        if not requests:
            raise ValueError("requests must not be empty")
        if any(request.prompt_len <= 0 for request in requests):
            raise ValueError("all requests must contain at least one prompt token")
        return torch.stack([self.prefill(request) for request in requests])

    def decode_batch(self, requests: list[GenerationRequest], token_ids: list[int]) -> torch.Tensor:
        """Run one decode token per request, returning ``[batch, vocab]``."""
        if len(requests) != len(token_ids) or not requests:
            raise ValueError("requests and token_ids must be non-empty and have equal length")
        return torch.stack([
            self.decode(request, token_id) for request, token_id in zip(requests, token_ids)
        ])


class PyTorchCausalLMRunner(ModelRunner):
    """使用原生 PyTorch 调用 HuggingFace causal LM 的适配器。

    prefill 处理完整 prompt 并建立 KV；decode 每次只输入一个 token，
    通过 ``past_key_values`` 复用历史上下文。
    """

    def __init__(self, model, device: str | torch.device | None = None, use_triton_kernels: bool = True):
        self.model = model.eval()
        self.device = torch.device(device or next(model.parameters()).device)
        self.triton_kernels_enabled = False
        if use_triton_kernels:
            from .triton_integration import install_triton_kernels
            self.triton_kernels_enabled = install_triton_kernels(self.model)

    @staticmethod
    def _legacy_past(past):
        """Normalize old and new HF cache objects to ``(key, value)`` tuples."""
        if isinstance(past, tuple):
            return past
        to_legacy = getattr(past, "to_legacy_cache", None)
        if to_legacy is not None:
            return to_legacy()
        # Transformers >= 5 uses Cache.layers[i].keys/values and removed
        # to_legacy_cache(). Keep the internal runner representation stable.
        layers = getattr(past, "layers", None)
        if layers is not None:
            pairs = []
            for layer in layers:
                key = getattr(layer, "keys", None)
                value = getattr(layer, "values", None)
                if key is None or value is None:
                    continue
                pairs.append((key, value))
            if pairs:
                return tuple(pairs)
        # Compatibility with intermediate Transformers Cache implementations.
        key_cache = getattr(past, "key_cache", None)
        value_cache = getattr(past, "value_cache", None)
        if key_cache is not None and value_cache is not None:
            return tuple(zip(key_cache, value_cache))
        raise TypeError(
            "runner requires tuple past_key_values or a supported HuggingFace Cache "
            "(to_legacy_cache(), layers, or key_cache/value_cache)"
        )

    @staticmethod
    def _is_cache_object(past) -> bool:
        """Return whether ``past`` is the new mutable Transformers Cache API."""
        return not isinstance(past, tuple) and (
            hasattr(past, "layers") or hasattr(past, "key_cache")
        )

    @staticmethod
    def _cache_from_pairs(pairs, template):
        """Build a mutable HF Cache when the installed Transformers needs one."""
        if not PyTorchCausalLMRunner._is_cache_object(template):
            return tuple(pairs)
        cache_cls = type(template)
        try:
            return cache_cls(ddp_cache_data=tuple(pairs))
        except (TypeError, ValueError):
            # DynamicCache is the stable public constructor in Transformers 5.
            try:
                from transformers.cache_utils import DynamicCache
            except ImportError as exc:  # pragma: no cover - only old HF installs
                raise TypeError("cannot reconstruct the installed Transformers Cache") from exc
            return DynamicCache(ddp_cache_data=tuple(pairs))

    @staticmethod
    def _split_past(past, lengths: list[int]):
        template = past
        past = PyTorchCausalLMRunner._legacy_past(past)
        result = [[] for _ in lengths]
        for key, value in past:
            for index, length in enumerate(lengths):
                result[index].append((
                    key[index:index + 1, :, :length, :].contiguous(),
                    value[index:index + 1, :, :length, :].contiguous(),
                ))
        return [PyTorchCausalLMRunner._cache_from_pairs(layers, template) for layers in result]

    @staticmethod
    def _split_decode_past(past, lengths: list[int], padded_length: int, template=None):
        """Remove right padding while retaining each row's appended token."""
        cache_template = template if template is not None else past
        result = [[] for _ in lengths]
        for key, value in PyTorchCausalLMRunner._legacy_past(past):
            for index, length in enumerate(lengths):
                current_key = key[index:index + 1, :, padded_length:padded_length + 1, :]
                current_value = value[index:index + 1, :, padded_length:padded_length + 1, :]
                result[index].append((
                    torch.cat((key[index:index + 1, :, :length, :], current_key), dim=2).contiguous(),
                    torch.cat((value[index:index + 1, :, :length, :], current_value), dim=2).contiguous(),
                ))
        return [PyTorchCausalLMRunner._cache_from_pairs(layers, cache_template) for layers in result]

    @staticmethod
    def _stack_past(requests: list[GenerationRequest]):
        templates = [r.past_key_values for r in requests]
        pasts = [PyTorchCausalLMRunner._legacy_past(past) for past in templates]
        if not pasts or not pasts[0]:
            raise ValueError("decode requires populated past_key_values")
        max_len = max(layer[0].shape[2] for layer in pasts[0])
        stacked = []
        for layer_index in range(len(pasts[0])):
            keys, values = [], []
            for past in pasts:
                key, value = past[layer_index]
                pad = max_len - key.shape[2]
                if pad:
                    key = torch.nn.functional.pad(key, (0, 0, 0, pad))
                    value = torch.nn.functional.pad(value, (0, 0, 0, pad))
                keys.append(key)
                values.append(value)
            stacked.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        lengths = [past[0][0].shape[2] for past in pasts]
        return PyTorchCausalLMRunner._cache_from_pairs(stacked, templates[0]), max_len, lengths

    @torch.inference_mode()
    def prefill(self, request: GenerationRequest) -> torch.Tensor:
        """执行 prompt 阶段，返回最后位置的 next-token logits。"""
        tokens = torch.tensor([request.input_ids], dtype=torch.long, device=self.device)
        output = self.model(input_ids=tokens, use_cache=True)
        request.past_key_values = output.past_key_values
        request.prefilled = True
        return output.logits[0, -1]

    @torch.inference_mode()
    def decode(self, request: GenerationRequest, token_id: int) -> torch.Tensor:
        """执行单 token decode，返回该 token 之后的 next-token logits。"""
        tokens = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        output = self.model(input_ids=tokens, past_key_values=request.past_key_values, use_cache=True)
        request.past_key_values = output.past_key_values
        request.prefilled = True
        return output.logits[0, -1]

    @torch.inference_mode()
    def prefill_batch(self, requests: list[GenerationRequest]) -> torch.Tensor:
        """Prefill variable-length prompts in one padded model invocation."""
        if not requests:
            raise ValueError("requests must not be empty")
        lengths = [request.prompt_len for request in requests]
        max_len = max(lengths)
        input_ids = torch.zeros((len(requests), max_len), dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(input_ids)
        for row, request in enumerate(requests):
            length = request.prompt_len
            input_ids[row, :length] = torch.as_tensor(request.input_ids, device=self.device)
            attention_mask[row, :length] = 1
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        for request, past in zip(requests, self._split_past(output.past_key_values, lengths)):
            request.past_key_values = past
            request.prefilled = True
        row_ids = torch.arange(len(requests), device=self.device)
        last_ids = torch.as_tensor([length - 1 for length in lengths], device=self.device)
        return output.logits[row_ids, last_ids]

    @torch.inference_mode()
    def decode_batch(self, requests: list[GenerationRequest], token_ids: list[int]) -> torch.Tensor:
        """Decode one token for each request with heterogeneous history lengths."""
        if len(requests) != len(token_ids) or not requests:
            raise ValueError("requests and token_ids must be non-empty and have equal length")
        past, max_past_len, past_lengths = self._stack_past(requests)
        input_ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device).unsqueeze(1)
        attention_mask = torch.zeros((len(requests), max_past_len + 1), dtype=torch.long, device=self.device)
        for row, length in enumerate(past_lengths):
            attention_mask[row, :length + 1] = 1
        position_ids = torch.as_tensor(past_lengths, dtype=torch.long, device=self.device).unsqueeze(1)
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            use_cache=True,
        )
        for request, item in zip(
            requests,
            self._split_decode_past(output.past_key_values, past_lengths, max_past_len),
        ):
            request.past_key_values = item
            request.prefilled = True
        return output.logits[:, -1]
