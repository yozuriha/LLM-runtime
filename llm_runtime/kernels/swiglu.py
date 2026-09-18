"""SwiGLU Triton kernel and its PyTorch reference implementation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 输入视为一维元素数组；一个 program 负责连续 BLOCK_SIZE 个元素。
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 用 FP32 完成激活和乘法，写回时自动转换为输出 tensor 的 dtype。
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # SwiGLU(gate, up) = SiLU(gate) * up = gate * sigmoid(gate) * up。
    tl.store(out_ptr + offsets, gate * tl.sigmoid(gate) * up, mask=mask)


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch reference：逐元素计算 ``SiLU(gate) * up``。"""
    if gate.shape != up.shape:
        raise ValueError("gate and up must have the same shape")
    return F.silu(gate) * up


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """对连续 CUDA tensor 执行 fused SwiGLU。"""
    if not gate.is_cuda or not up.is_cuda:
        raise ValueError("SwiGLU Triton kernel requires CUDA tensors")
    if gate.shape != up.shape or not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("gate and up must have the same contiguous shape")
    output = torch.empty_like(gate)
    # 固定 block 大小适合逐元素操作；最后一个 program 由 mask 处理尾部元素。
    block = 256
    _swiglu_kernel[(triton.cdiv(gate.numel(), block),)](gate, up, output, gate.numel(), BLOCK_SIZE=block, num_warps=4)
    return output
