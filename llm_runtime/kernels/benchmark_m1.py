"""Benchmark the split M1 kernels against eager PyTorch on CUDA.

Latency is collected through ``triton.testing.Benchmark``/``do_bench``. The
additional table records numerical error and allocator peak memory for the
same inputs. This script targets the project's RTX 4070 (12 GiB, SM89), but
does not hard-code a device index.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

import torch
import triton
import sys
from rmsnorm import rmsnorm,rmsnorm_reference
from rope import apply_rope,apply_rope_reference
from swiglu import swiglu,swiglu_reference



_INPUTS: dict[tuple[str, str], tuple[torch.Tensor, ...]] = {}
_WARMUP = 25
_REPEAT = 100


def _provider_fn(op: str, shape: str, provider: str) -> Callable[[], Any]:
    tensors = _INPUTS[(op, shape)]
    if op == "rmsnorm":
        x, weight = tensors
        return (lambda: rmsnorm_reference(x, weight)) if provider == "torch" else (lambda: rmsnorm(x, weight))
    if op == "swiglu":
        gate, up = tensors
        return (lambda: swiglu_reference(gate, up)) if provider == "torch" else (lambda: swiglu(gate, up))
    q, k, cos, sin, positions = tensors
    return (lambda: apply_rope_reference(q, k, cos, sin, positions)) if provider == "torch" else (
        lambda: apply_rope(q, k, cos, sin, positions)
    )


def _bench(op: str, shape: str, provider: str, warmup: int = _WARMUP, rep: int = _REPEAT) -> float:
    return float(triton.testing.do_bench(_provider_fn(op, shape, provider), warmup=warmup, rep=rep))


def _make_benchmark(op: str, shapes: list[str]) -> Any:
    benchmark = triton.testing.Benchmark(
        x_names=["shape"],
        x_vals=shapes,
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch", "Triton"],
        plot_name=f"m1-{op}-latency",
        args={},
        xlabel="input shape",
        ylabel="latency (ms)",
    )

    @triton.testing.perf_report([benchmark])
    def report(shape: str, provider: str, **kwargs: Any) -> float:
        return _bench(op, shape, provider, kwargs.get("warmup", _WARMUP), kwargs.get("rep", _REPEAT))

    return report


def _measure_metrics(op: str, shape: str, provider: str) -> tuple[float, float, float, float]:
    fn = _provider_fn(op, shape, provider)
    for _ in range(_WARMUP):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    output = fn()
    torch.cuda.synchronize()
    peak_bytes = float(torch.cuda.max_memory_allocated())
    latency_ms = _bench(op, shape, provider)

    reference = _provider_fn(op, shape, "torch")()
    if isinstance(reference, tuple):
        error = max(float((a.float() - b.float()).abs().max().item()) for a, b in zip(reference, output))
    else:
        error = float((reference.float() - output.float()).abs().max().item())
    del output, reference
    total_bytes = float(torch.cuda.get_device_properties().total_memory)
    return latency_ms, error, peak_bytes / 2**20, 100.0 * peak_bytes / total_bytes


def _build_inputs(args: argparse.Namespace) -> None:
    dtype = torch.float16
    device = torch.device("cuda")
    for name, scale in (("base", 1), ("long", 2)):
        batch, seq = args.batch, args.seq_len * scale
        x = torch.randn((batch, seq, args.hidden), device=device, dtype=dtype)
        weight = torch.randn((args.hidden,), device=device, dtype=dtype)
        _INPUTS[("rmsnorm", name)] = (x, weight)
        _INPUTS[("swiglu", name)] = (torch.randn_like(x), torch.randn_like(x))

        q = torch.randn((batch, args.heads, seq, args.head_dim), device=device, dtype=dtype)
        k = torch.randn_like(q)
        angles = torch.randn((seq, args.head_dim // 2), device=device, dtype=torch.float32)
        positions = torch.arange(seq, device=device).expand(batch, -1).contiguous()
        _INPUTS[("rope", name)] = (q, k, angles.cos(), angles.sin(), positions)


def main() -> None:
    global _WARMUP, _REPEAT
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    _WARMUP, _REPEAT = args.warmup, args.repeat
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the M1 benchmark (target: RTX 4070)")
    _build_inputs(args)
    print(f"GPU: {torch.cuda.get_device_name()} | Triton: {triton.__version__}")

    reports = {op: _make_benchmark(op, ["base", "long"]) for op in ("rmsnorm", "rope", "swiglu")}
    for op, report in reports.items():
        print(f"\n[{op}] triton.testing.Benchmark latency")
        report.run(print_data=True, show_plots=False)

    print("\nop provider shape latency_ms max_abs_error peak_memory_mib peak_memory_pct")
    for op in ("rmsnorm", "rope", "swiglu"):
        for shape in ("base", "long"):
            for provider in ("torch", "triton"):
                latency, error, memory, memory_pct = _measure_metrics(op, shape, provider)
                print(f"{op} {provider} {shape} {latency:.4f} {error:.6g} {memory:.3f} {memory_pct:.4f}")


if __name__ == "__main__":
    main()
