"""Compute-device selection shared by every torch-backed path.

Preference order: CUDA (the Linux/NVIDIA box) -> Apple MPS -> CPU. Keeping this
in one place means the engines, the voice-design model, and cache cleanup all
agree on where tensors live, and a Mac never trips over a hard-coded "cuda:0".
"""
from __future__ import annotations

import os
import sys

# Ops that MPS lacks fall back to CPU instead of raising. Must be set before
# torch initializes the MPS backend; this module is imported by config, which
# every engine imports first.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_device: str | None = None


def pick() -> str:
    """'cuda:0', 'mps', or 'cpu' — cached after the first call."""
    global _device
    if _device is None:
        import torch
        if torch.cuda.is_available():
            _device = "cuda:0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    return _device


def is_cuda() -> bool:
    return pick().startswith("cuda")


def cuda_possible() -> bool:
    """Cheap pre-check for non-torch libraries (ctranslate2) that probe CUDA
    themselves: macOS has no CUDA at all, so don't even try there."""
    return sys.platform != "darwin"


def load_kwargs() -> dict:
    """from_pretrained() kwargs for a Qwen3-TTS model on the chosen device.

    bfloat16 on CUDA and MPS (MPS supports it natively on Apple Silicon and
    qwen_tts already routes its rotary embeddings around MPS' float64 gap);
    float32 on CPU, where half precision is slower than full.
    """
    import torch
    dev = pick()
    if dev == "cpu":
        return {"device_map": "cpu", "dtype": torch.float32}
    return {"device_map": dev, "dtype": torch.bfloat16}


def empty_cache() -> None:
    """Release cached allocator blocks on whichever accelerator is in use."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
