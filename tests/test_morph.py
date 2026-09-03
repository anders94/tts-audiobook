from __future__ import annotations

import numpy as np
import pytest

parselmouth = pytest.importorskip("parselmouth")

from tts_audiobook.morph import PRESETS, morph


def _tone(freq: float, seconds: float = 2.0, sr: int = 24000) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    # Amplitude-modulated so the pitch tracker sees a speech-like envelope.
    return (0.5 * np.sin(2 * np.pi * freq * t)
            * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def _median_f0(wav: np.ndarray, sr: int) -> float:
    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch()
    return parselmouth.praat.call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")


def test_pitch_shift_direction():
    sr = 24000
    wav = _tone(150.0, sr=sr)
    up = morph(wav, sr, pitch_semitones=4.0)
    f0 = _median_f0(up, sr)
    assert 175.0 < f0 < 205.0  # 150 Hz * 2^(4/12) ≈ 189


def test_duration_preserved():
    sr = 24000
    wav = _tone(150.0, sr=sr)
    out = morph(wav, sr, pitch_semitones=-2.5, formant_ratio=0.92)
    assert abs(len(out) - len(wav)) / sr < 0.1


def test_presets_are_valid_kwargs():
    sr = 24000
    wav = _tone(150.0, seconds=1.0, sr=sr)
    for name, params in PRESETS.items():
        out = morph(wav, sr, **params)
        assert len(out) > 0, name
