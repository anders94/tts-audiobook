from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np


class Engine(Protocol):
    """A local TTS engine that renders text by cloning a reference voice."""

    name: str
    max_batch: int  # largest texts-per-generate() call the engine supports

    def load(self) -> None: ...

    def clone_prompt(self, ref_audio: Path, ref_text: str) -> Any:
        """Precompute a reusable voice prompt for a frozen reference clip."""
        ...

    def generate(self, texts: list[str], prompt: Any, *,
                 language: str = "English", seed: int | None = None
                 ) -> tuple[list[np.ndarray], int]:
        """One float32 mono waveform per text, plus the sample rate."""
        ...


def get_engine(name: str) -> Engine:
    # Lazy imports so a missing optional dependency only breaks its own engine.
    if name == "qwen":
        from .qwen import QwenEngine
        return QwenEngine()
    if name == "chatterbox":
        from .subproc import SubprocEngine
        return SubprocEngine("chatterbox")
    raise ValueError(f"Unknown engine {name!r}; available: qwen, chatterbox")
