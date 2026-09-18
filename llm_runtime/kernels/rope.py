"""非交错 half-split RoPE Triton kernel 及 PyTorch reference。"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, position_ptr, out_q_ptr, out_k_ptr,
    heads, seq_len, stride_qb, stride_qh, stride_qt, stride_kb, stride_kh, stride_kt,
    stride_oqb, stride_oqh, stride_oqt, stride_okb, stride_okh, stride_okt,
    stride_position_b, stride_cos_position, stride_sin_position, HALF_DIM: tl.constexpr,
):
    # 一个 program 处理一个 (batch, head, token)，并同时旋转 Q 和 K。
    pid = tl.program_id(0)
    tokens_per_batch = heads * seq_len
    batch_id = pid // tokens_per_batch
    head_id = (pid % tokens_per_batch) // seq_len
    token_id = pid % seq_len

    # half-split 布局将 head_dim 分成前后两半，而不是相邻元素成对交错。
    offsets = tl.arange(0, HALF_DIM)

    # 每个 batch/token 可以拥有独立 position id；据此读取对应的 cos/sin。
    position = tl.load(position_ptr + batch_id * stride_position_b + token_id)
    cos = tl.load(cos_ptr + position * stride_cos_position + offsets).to(tl.float32)
    sin = tl.load(sin_ptr + position * stride_sin_position + offsets).to(tl.float32)

    # 使用显式 stride 支持 [B, H, T, D] 的连续布局，并保留通用寻址方式。
    q_base = q_ptr + batch_id * stride_qb + head_id * stride_qh + token_id * stride_qt
    k_base = k_ptr + batch_id * stride_kb + head_id * stride_kh + token_id * stride_kt
    oq_base = out_q_ptr + batch_id * stride_oqb + head_id * stride_oqh + token_id * stride_oqt
    ok_base = out_k_ptr + batch_id * stride_okb + head_id * stride_okh + token_id * stride_okt
    q_first = tl.load(q_base + offsets).to(tl.float32)
    q_second = tl.load(q_base + HALF_DIM + offsets).to(tl.float32)
    k_first = tl.load(k_base + offsets).to(tl.float32)
    k_second = tl.load(k_base + HALF_DIM + offsets).to(tl.float32)

    # 二维旋转： [a', b'] = [a*cos - b*sin, b*cos + a*sin]。
    tl.store(oq_base + offsets, q_first * cos - q_second * sin)
    tl.store(oq_base + HALF_DIM + offsets, q_second * cos + q_first * sin)
    tl.store(ok_base + offsets, k_first * cos - k_second * sin)
    tl.store(ok_base + HALF_DIM + offsets, k_second * cos + k_first * sin)


def apply_rope_reference(q, k, cos, sin, position_ids=None):
    """PyTorch reference，输入 Q/K 形状为 ``[B, H, T, D]``。"""
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] % 2:
        raise ValueError("q and k must have equal [B, H, T, even_D] shapes")
    batch, _, seq_len, dim = q.shape
    half_dim = dim // 2
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[1] != half_dim:
        raise ValueError("cos and sin must have shape [max_position, head_dim // 2]")
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=q.device).expand(batch, seq_len)
    cos_selected = cos[position_ids].to(q.device)[:, None].float()
    sin_selected = sin[position_ids].to(q.device)[:, None].float()
    # 与 Triton kernel 保持一致：前半维和后半维组成旋转 pair。
    q_first, q_second = q[..., :half_dim].float(), q[..., half_dim:].float()
    k_first, k_second = k[..., :half_dim].float(), k[..., half_dim:].float()
    q_out = torch.cat((q_first * cos_selected - q_second * sin_selected,
                       q_second * cos_selected + q_first * sin_selected), dim=-1)
    k_out = torch.cat((k_first * cos_selected - k_second * sin_selected,
                       k_second * cos_selected + k_first * sin_selected), dim=-1)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def apply_rope(q, k, cos, sin, position_ids=None):
    """对连续 CUDA Q/K 执行 half-split RoPE，并返回旋转后的 Q/K。"""
    if any(not tensor.is_cuda for tensor in (q, k, cos, sin)):
        raise ValueError("RoPE Triton kernel requires CUDA tensors")
    if q.shape != k.shape or q.ndim != 4 or q.shape[-1] % 2:
        raise ValueError("q and k must have equal [B, H, T, even_D] shapes")
    if any(not tensor.is_contiguous() for tensor in (q, k, cos, sin)):
        raise ValueError("q, k, cos and sin must be contiguous")
    batch, heads, seq_len, dim = q.shape
    half_dim = dim // 2
    if half_dim & (half_dim - 1):
        raise ValueError("head_dim // 2 must be a power of two for the Triton RoPE kernel")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[1] != half_dim:
        raise ValueError("cos and sin must have shape [max_position, head_dim // 2]")
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=q.device).expand(batch, seq_len).contiguous()
    position_ids = position_ids.to(device=q.device, dtype=torch.long).contiguous()
    if position_ids.shape != (batch, seq_len):
        raise ValueError("position_ids must have shape [batch, seq_len]")
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    # 网格大小等于 batch * heads * seq_len，每个 program 对应一个 token-head。
    _rope_kernel[(batch * heads * seq_len,)](
        q, k, cos, sin, position_ids, out_q, out_k, heads, seq_len,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        out_q.stride(0), out_q.stride(1), out_q.stride(2), out_k.stride(0), out_k.stride(1), out_k.stride(2),
        position_ids.stride(0), cos.stride(0), sin.stride(0), HALF_DIM=half_dim,
        num_warps=1 if half_dim <= 32 else 4,
    )
    return out_q, out_k


@triton.jit
def _rope_embedded_single_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    heads, seq_len,
    stride_xb, stride_xh, stride_xt,
    stride_ob, stride_oh, stride_ot,
    stride_cos_b, stride_cos_t, stride_sin_b, stride_sin_t,
    HALF_DIM: tl.constexpr,
):
    """对单个 Q 或 K tensor 旋转，允许 Q/K 使用不同 head 数（GQA）。"""
    pid = tl.program_id(0)
    tokens_per_batch = heads * seq_len
    batch_id = pid // tokens_per_batch
    head_id = (pid % tokens_per_batch) // seq_len
    token_id = pid % seq_len
    offsets = tl.arange(0, HALF_DIM)

    cos_base = cos_ptr + batch_id * stride_cos_b + token_id * stride_cos_t
    sin_base = sin_ptr + batch_id * stride_sin_b + token_id * stride_sin_t
    cos_values = tl.load(cos_base + offsets).to(tl.float32)
    sin_values = tl.load(sin_base + offsets).to(tl.float32)

    x_base = x_ptr + batch_id * stride_xb + head_id * stride_xh + token_id * stride_xt
    out_base = out_ptr + batch_id * stride_ob + head_id * stride_oh + token_id * stride_ot
    first = tl.load(x_base + offsets).to(tl.float32)
    second = tl.load(x_base + HALF_DIM + offsets).to(tl.float32)
    tl.store(out_base + offsets, first * cos_values - second * sin_values)
    tl.store(out_base + HALF_DIM + offsets, second * cos_values + first * sin_values)


def apply_rope_embeddings(q, k, cos, sin):
    """对 Qwen 风格的 ``cos/sin=[B,T,D]`` 做 Triton RoPE。

    Qwen 的 rotary embedding 已经根据 position id 生成 cos/sin，因此这里不
    再读取 position table；只取 cos/sin 的前半维，和 half-split Q/K 对齐。
    """
    if any(not tensor.is_cuda for tensor in (q, k, cos, sin)):
        raise ValueError("RoPE Triton kernel requires CUDA tensors")
    if q.ndim != 4 or k.ndim != 4 or cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError("q/k must be [B,H,T,D], cos/sin must be [B,T,D]")
    if q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise ValueError("q and k must share batch, sequence and head dimension")
    batch, q_heads, seq_len, dim = q.shape
    k_heads = k.shape[1]
    if cos.shape != (batch, seq_len, dim) or dim % 2:
        raise ValueError("cos/sin shape must match q batch, sequence and head dimension")
    half_dim = dim // 2
    if half_dim & (half_dim - 1):
        raise ValueError("head_dim // 2 must be a power of two for the Triton RoPE kernel")
    if any(not tensor.is_contiguous() for tensor in (q, k, cos, sin)):
        raise ValueError("q, k, cos and sin must be contiguous")
    # Qwen cos/sin 将前半维和后半维重复；kernel 只需读取前半维。
    cos_half = cos[..., :half_dim].contiguous()
    sin_half = sin[..., :half_dim].contiguous()
    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    _rope_embedded_single_kernel[(batch * q_heads * seq_len,)](
        q, cos_half, sin_half, out_q, q_heads, seq_len,
        q.stride(0), q.stride(1), q.stride(2),
        out_q.stride(0), out_q.stride(1), out_q.stride(2),
        cos_half.stride(0), cos_half.stride(1), sin_half.stride(0), sin_half.stride(1),
        HALF_DIM=half_dim, num_warps=1 if half_dim <= 32 else 4,
    )
    _rope_embedded_single_kernel[(batch * k_heads * seq_len,)](
        k, cos_half, sin_half, out_k, k_heads, seq_len,
        k.stride(0), k.stride(1), k.stride(2),
        out_k.stride(0), out_k.stride(1), out_k.stride(2),
        cos_half.stride(0), cos_half.stride(1), sin_half.stride(0), sin_half.stride(1),
        HALF_DIM=half_dim, num_warps=1 if half_dim <= 32 else 4,
    )
    return out_q, out_k
