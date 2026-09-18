"""Backward-compatible exports for the split M1 kernel modules.

New code should import each operation from its dedicated module. This module
remains so existing callers using ``llm_runtime.kernels.fused`` keep working.
"""

from .rmsnorm import rmsnorm, rmsnorm_reference
from .rope import apply_rope, apply_rope_reference
from .swiglu import swiglu, swiglu_reference

__all__ = [
    "apply_rope",
    "apply_rope_reference",
    "rmsnorm",
    "rmsnorm_reference",
    "swiglu",
    "swiglu_reference",
]
