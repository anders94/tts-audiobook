from __future__ import annotations

import numpy as np

from tts_audiobook import config
from tts_audiobook.audio import trim_silence


def _speech_with_breath(sr: int = 24000) -> np.ndarray:
    """0.3s silence + 1s 'speech' + 0.25s decaying 'exhale' + 0.5s silence."""
    t = np.arange(sr) / sr
    speech = 0.5 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    exhale = (0.02 * np.exp(-np.linspace(0, 5, int(0.25 * sr)))
              * np.random.default_rng(0).standard_normal(int(0.25 * sr))
              ).astype(np.float32)
    return np.concatenate([np.zeros(int(0.3 * sr), dtype=np.float32), speech,
                           exhale, np.zeros(int(0.5 * sr), dtype=np.float32)])


def test_trim_keeps_tail_decay_and_tight_head():
    sr = 24000
    wav = _speech_with_breath(sr)
    out = trim_silence(wav, sr)
    # Head: at most lead pad of the 0.3s leading silence survives.
    assert np.max(np.abs(out[: int(0.02 * sr)])) < 0.5  # starts in the fade-in
    # Tail: some decay beyond the loud part is kept (not cut at speech edge).
    assert len(out) > sr + int(0.05 * sr)


def test_trim_fades_to_zero():
    sr = 24000
    out = trim_silence(_speech_with_breath(sr), sr)
    # The final samples must approach zero smoothly, not stop abruptly.
    assert abs(float(out[-1])) < 1e-4
    tail = out[-int(config.FADE_OUT_S * sr):]
    assert np.max(np.abs(tail)) < 0.1


def test_trim_all_silence_returns_empty():
    sr = 24000
    assert len(trim_silence(np.zeros(sr, dtype=np.float32), sr)) == 0
