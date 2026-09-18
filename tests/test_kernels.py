import pytest
import torch

from llm_runtime.kernels import (
    apply_rope,
    apply_rope_reference,
    rmsnorm,
    rmsnorm_reference,
    swiglu,
    swiglu_reference,
)


CUDA = torch.cuda.is_available()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(3, 257), (2, 5, 1536)])
def test_rmsnorm_reference(dtype, shape):
    x = torch.randn(shape, dtype=dtype)
    weight = torch.randn(shape[-1], dtype=dtype)
    actual = rmsnorm_reference(x, weight)
    expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6) * weight.float()
    atol, rtol = {
        torch.float32: (2e-6, 2e-5),
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }[dtype]
    torch.testing.assert_close(actual.float(), expected, atol=atol, rtol=rtol)


@pytest.mark.skipif(not CUDA, reason="Triton kernel requires CUDA")
def test_rmsnorm_triton_matches_reference():
    x = torch.randn((7, 1536), device="cuda", dtype=torch.float16)
    weight = torch.randn((1536,), device="cuda", dtype=torch.float16)
    torch.testing.assert_close(rmsnorm(x, weight), rmsnorm_reference(x, weight), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_rope_reference_with_position_ids(dtype):
    q = torch.randn((2, 3, 5, 8), dtype=dtype)
    k = torch.randn_like(q)
    angles = torch.randn((32, 4), dtype=torch.float32)
    positions = torch.tensor([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])
    q_out, k_out = apply_rope_reference(q, k, angles.cos(), angles.sin(), positions)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    tolerance = (3e-3, 3e-3) if dtype != torch.bfloat16 else (3e-2, 3e-2)
    torch.testing.assert_close(q_out.float().square().sum(-1), q.float().square().sum(-1), atol=tolerance[0], rtol=tolerance[1])


@pytest.mark.skipif(not CUDA, reason="Triton kernel requires CUDA")
def test_rope_triton_matches_reference():
    q = torch.randn((2, 3, 5, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    angles = torch.randn((32, 32), device="cuda", dtype=torch.float32)
    positions = torch.tensor([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]], device="cuda")
    actual = apply_rope(q, k, angles.cos(), angles.sin(), positions)
    expected = apply_rope_reference(q, k, angles.cos(), angles.sin(), positions)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-3)


def test_rope_reference_supports_non_power_of_two_half_dimension():
    q = torch.randn((1, 1, 2, 6))
    k = torch.randn_like(q)
    cos = torch.ones((4, 3))
    sin = torch.zeros_like(cos)
    # The reference path supports the mathematical shape even when Triton
    # cannot launch a non-power-of-two half dimension; this is intentional.
    assert apply_rope_reference(q, k, cos, sin)[0].shape == q.shape


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_swiglu_reference(dtype):
    gate = torch.randn((2, 3, 257), dtype=dtype)
    up = torch.randn_like(gate)
    torch.testing.assert_close(swiglu_reference(gate, up), torch.nn.functional.silu(gate) * up)


@pytest.mark.skipif(not CUDA, reason="Triton kernel requires CUDA")
def test_swiglu_triton_matches_reference():
    gate = torch.randn((2, 3, 257), device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    torch.testing.assert_close(swiglu(gate, up), swiglu_reference(gate, up), atol=2e-3, rtol=2e-3)
