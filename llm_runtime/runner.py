from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .request import GenerationRequest


class ModelRunner(ABC):
    @abstractmethod
    def prefill(self, request: GenerationRequest) -> torch.Tensor:
        """Run prompt tokens and return logits for the next token."""

    @abstractmethod
    def decode(self, request: GenerationRequest, token_id: int) -> torch.Tensor:
        """Run one token using the request's cached past and return logits."""


class PyTorchCausalLMRunner(ModelRunner):
    """Adapter around a HuggingFace causal LM using native PyTorch execution."""

    def __init__(self, model, device: str | torch.device | None = None):
        self.model = model.eval()
        self.device = torch.device(device or next(model.parameters()).device)

    @torch.inference_mode()
    def prefill(self, request: GenerationRequest) -> torch.Tensor:
        tokens = torch.tensor([request.input_ids], dtype=torch.long, device=self.device)
        output = self.model(input_ids=tokens, use_cache=True)
        request.past_key_values = output.past_key_values
        request.prefilled = True
        return output.logits[0, -1]

    @torch.inference_mode()
    def decode(self, request: GenerationRequest, token_id: int) -> torch.Tensor:
        tokens = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        output = self.model(input_ids=tokens, past_key_values=request.past_key_values, use_cache=True)
        request.past_key_values = output.past_key_values
        request.prefilled = True
        return output.logits[0, -1]
