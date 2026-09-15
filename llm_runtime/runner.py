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


class PyTorchCausalLMRunner(ModelRunner):
    """使用原生 PyTorch 调用 HuggingFace causal LM 的适配器。

    prefill 处理完整 prompt 并建立 KV；decode 每次只输入一个 token，
    通过 ``past_key_values`` 复用历史上下文。
    """

    def __init__(self, model, device: str | torch.device | None = None):
        self.model = model.eval()
        self.device = torch.device(device or next(model.parameters()).device)

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
