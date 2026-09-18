"""将 M1 Triton kernel 安装到支持的 HuggingFace Qwen2 模块。

这是一个保守的适配层：Triton kernel 只在 CUDA、连续 tensor 和已知 Qwen2
布局下启用，其他情况调用原始 PyTorch forward，保证 baseline 仍可运行。
"""

from __future__ import annotations

from types import MethodType

import torch

from .kernels.rmsnorm import rmsnorm
from .kernels.rope import apply_rope_embeddings
from .kernels.swiglu import swiglu


def _is_qwen2_model(model) -> bool:
    config = getattr(model, "config", None)
    return getattr(config, "model_type", None) == "qwen2" and hasattr(model, "model")


def _install_rmsnorm(module) -> None:
    if getattr(module, "_triton_rmsnorm_installed", False):
        return
    eps = module.variance_epsilon

    def forward(self, hidden_states):
        if hidden_states.is_cuda and hidden_states.is_contiguous() and hidden_states.shape[-1] == self.weight.numel():
            return rmsnorm(hidden_states, self.weight, eps)
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.pow(2).mean(-1, keepdim=True)
        return self.weight * (values * torch.rsqrt(variance + eps)).to(input_dtype)

    module.forward = MethodType(forward, module)
    module._triton_rmsnorm_installed = True


def _install_mlp(module) -> None:
    if getattr(module, "_triton_swiglu_installed", False):
        return
    act_fn = module.act_fn
    act_name = getattr(act_fn, "__name__", act_fn.__class__.__name__).lower()

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if gate.is_cuda and gate.is_contiguous() and up.is_contiguous() and act_name.startswith("silu"):
            hidden = swiglu(gate, up)
        else:
            hidden = act_fn(gate) * up
        return self.down_proj(hidden)

    module.forward = MethodType(forward, module)
    module._triton_swiglu_installed = True


def _install_rope_global() -> None:
    """替换 modeling_qwen2 的 RoPE helper；原函数作为所有 fallback。"""
    try:
        from transformers.models.qwen2 import modeling_qwen2
    except ImportError:  # pragma: no cover - optional dependency
        return
    if getattr(modeling_qwen2, "_decoder_runtime_triton_rope", False):
        return
    original = modeling_qwen2.apply_rotary_pos_emb

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        if (
            q.is_cuda and k.is_cuda and cos.is_cuda and sin.is_cuda
            and unsqueeze_dim == 1 and q.ndim == 4
            and cos.ndim == 3 and cos.shape == sin.shape
            and cos.shape[0] == q.shape[0] and cos.shape[1] == q.shape[2]
            and cos.shape[2] == q.shape[3]
        ):
            # Qwen projection 后通常是 transpose 得到的非连续 view；复制一次以
            # 满足 Triton 的线性寻址约束，后续 attention 仍使用旋转后的结果。
            return apply_rope_embeddings(q.contiguous(), k.contiguous(), cos.contiguous(), sin.contiguous())
        return original(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    modeling_qwen2.apply_rotary_pos_emb = apply_rotary_pos_emb
    modeling_qwen2._decoder_runtime_triton_rope = True


def install_triton_kernels(model) -> bool:
    """安装 M1 kernel，返回是否成功识别并修改 Qwen2 模型。"""
    if not torch.cuda.is_available() or not _is_qwen2_model(model):
        return False
    for module in model.modules():
        name = module.__class__.__name__
        if name == "Qwen2RMSNorm":
            _install_rmsnorm(module)
        elif name == "Qwen2MLP":
            _install_mlp(module)
    _install_rope_global()
    model._decoder_runtime_triton_enabled = True
    return True
