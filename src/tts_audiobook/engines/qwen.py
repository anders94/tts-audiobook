from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .. import device
from ..config import QWEN_MODEL_ID


def _max_new_tokens_for(texts: list[str]) -> int:
    """Cap codec generation to stop runaway loops (a known Qwen failure mode).

    12 codec tokens/s at ~15 chars/s of speech ≈ 0.8 tokens per char; allow
    40% headroom plus slack for very short lines.
    """
    longest = max(len(t) for t in texts)
    return min(4096, int(longest * 0.8 * 1.4) + 96)


class QwenEngine:
    """Qwen3-TTS-1.7B-Base voice cloning; one voice prompt cached per ref."""

    name = "qwen"
    max_batch = 24

    def __init__(self) -> None:
        self._model: Any = None
        self._prompt_cache: dict[str, Any] = {}

    def load(self) -> None:
        if self._model is not None:
            return
        from qwen_tts import Qwen3TTSModel
        from rich import print as rprint

        kwargs = device.load_kwargs()
        dev = kwargs["device_map"]
        if dev == "cpu":
            rprint("[yellow]No CUDA or MPS device; loading Qwen3-TTS on CPU "
                   "(this will be very slow).[/yellow]")
        else:
            rprint(f"[dim]Loading Qwen3-TTS on {dev} ({kwargs['dtype']})…[/dim]")
        if device.is_cuda():
            # flash-attn is a CUDA-only kernel; elsewhere SDPA is the fast path.
            try:
                self._model = Qwen3TTSModel.from_pretrained(
                    QWEN_MODEL_ID, attn_implementation="flash_attention_2", **kwargs)
            except Exception:
                rprint("[yellow]flash_attention_2 unavailable; loading without it.[/yellow]")
        if self._model is None:
            self._model = Qwen3TTSModel.from_pretrained(QWEN_MODEL_ID, **kwargs)

        # Silence transformers' per-call "Setting `pad_token_id` to `eos_token_id`"
        # warning: the talker's generation config has no pad_token_id, so generate()
        # defaults it to EOS (the codec EOS) and logs each time. Set it up front.
        gc = self._model.model.talker.generation_config
        if gc.pad_token_id is None:
            eos = gc.eos_token_id
            if eos is None:
                eos = self._model.model.config.talker_config.codec_eos_token_id
            gc.pad_token_id = eos if isinstance(eos, int) else eos[0]

    def clone_prompt(self, ref_audio: Path, ref_text: str) -> Any:
        self.load()
        key = f"{ref_audio}::{ref_text}"
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached
        prompt = self._model.create_voice_clone_prompt(
            ref_audio=str(ref_audio), ref_text=ref_text)
        self._prompt_cache[key] = prompt
        return prompt

    def generate(self, texts: list[str], prompt: Any, *,
                 language: str = "English", seed: int | None = None
                 ) -> tuple[list[np.ndarray], int]:
        import torch

        self.load()
        if seed is not None:
            torch.manual_seed(seed)
        wavs, sr = self._model.generate_voice_clone(
            text=texts,
            language=[language] * len(texts),
            voice_clone_prompt=prompt,
            max_new_tokens=_max_new_tokens_for(texts),
        )
        out: list[np.ndarray] = []
        for w in wavs:
            arr = w.detach().cpu().numpy() if hasattr(w, "detach") else np.asarray(w)
            arr = np.squeeze(arr).astype(np.float32, copy=False)
            out.append(arr)
        return out, int(sr)
