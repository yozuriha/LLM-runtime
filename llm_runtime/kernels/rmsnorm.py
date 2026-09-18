"""RMSNorm Triton kernel and its PyTorch reference implementation."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, weight_ptr, y_ptr, n_rows, n_cols, stride_x, stride_y, eps, BLOCK_SIZE: tl.constexpr):
    # 一个 program 负责输入矩阵的一行，即一个 token 的 hidden 向量。
    row = tl.program_id(0)

    # BLOCK_SIZE 取不小于 hidden size 的 2 次幂；超出 n_cols 的 lane 用 mask 屏蔽。
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    # 在 FP32 中读取和计算，降低 FP16/BF16 归约时的累积误差。
    x = tl.load(x_ptr + row * stride_x + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 只计算均方根，不像 LayerNorm 一样减去均值。
    mean_square = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / n_cols

    # 归一化后乘可学习权重；写回时通过 mask 避免越过 hidden 维度。
    tl.store(y_ptr + row * stride_y + offsets, x * tl.rsqrt(mean_square + eps) * weight, mask=mask)


def rmsnorm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch reference：FP32 计算 RMS，再转换回输入 dtype。"""
    if x.shape[-1] != weight.numel():
        raise ValueError("weight size must match the last input dimension")
    inv_rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() * inv_rms * weight.float()).to(x.dtype)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对 ``[..., hidden]`` 连续 CUDA tensor 执行 fused RMSNorm。"""
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("RMSNorm Triton kernel requires CUDA tensors")
    if x.ndim < 2 or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous, and x must have at least 2 dimensions")
    if x.shape[-1] != weight.numel():
        raise ValueError("weight size must match the last input dimension")
    # 将任意前导维度展平为 rows，kernel 只需要处理二维矩阵。
    x_2d = x.reshape(-1, x.shape[-1])
    block = triton.next_power_of_2(x_2d.shape[1])
    if block > 8192:
        raise ValueError("feature size exceeds the M1 RMSNorm limit 8192")
    output = torch.empty_like(x_2d)
    # 每个 row 启动一个 program；hidden 较大时增加 warp 数量帮助归约。
    _rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d, weight, output, x_2d.shape[0], x_2d.shape[1], x_2d.stride(0), output.stride(0), eps,
        BLOCK_SIZE=block, num_warps=1 if block <= 256 else 4,
    )
    return output.reshape_as(x)
