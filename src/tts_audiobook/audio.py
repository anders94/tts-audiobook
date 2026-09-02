from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config
from .config import MP3_BITRATE


def silence(sample_rate: int, seconds: float) -> np.ndarray:
    n = max(0, int(round(sample_rate * seconds)))
    return np.zeros(n, dtype=np.float32)


def concat(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def duration_s(wav: np.ndarray, sample_rate: int) -> float:
    return len(wav) / sample_rate if sample_rate else 0.0


def trim_silence(wav: np.ndarray, sample_rate: int,
                 threshold_dbfs: float = config.TRIM_THRESHOLD_DBFS,
                 pad_s: float = config.TRIM_PAD_S) -> np.ndarray:
    """Cut leading/trailing silence below threshold, keeping a small pad."""
    if len(wav) == 0:
        return wav
    threshold = 10.0 ** (threshold_dbfs / 20.0)
    loud = np.flatnonzero(np.abs(wav) > threshold)
    if len(loud) == 0:
        return wav[:0]
    pad = int(pad_s * sample_rate)
    start = max(0, int(loud[0]) - pad)
    end = min(len(wav), int(loud[-1]) + 1 + pad)
    return wav[start:end]


def _rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wav)))) if len(wav) else 0.0


def normalize_loudness(wav: np.ndarray, sample_rate: int,
                       target_lufs: float = config.TARGET_LUFS,
                       reference_rms: float | None = None) -> np.ndarray:
    """Gain-normalize to target LUFS (RMS-matched for very short clips)."""
    if len(wav) == 0:
        return wav
    gain_db = 0.0
    if duration_s(wav, sample_rate) >= config.MIN_LOUDNORM_S:
        import pyloudnorm

        meter = pyloudnorm.Meter(sample_rate)
        loudness = meter.integrated_loudness(wav.astype(np.float64))
        if np.isfinite(loudness):
            gain_db = target_lufs - loudness
    elif reference_rms and reference_rms > 0:
        rms = _rms(wav)
        if rms > 0:
            gain_db = 20.0 * np.log10(reference_rms / rms)

    gain_db = float(np.clip(gain_db, -config.MAX_GAIN_DB, config.MAX_GAIN_DB))
    out = wav * (10.0 ** (gain_db / 20.0))

    peak_limit = 10.0 ** (config.PEAK_DBFS / 20.0)
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > peak_limit:
        out = out * (peak_limit / peak)
    return out.astype(np.float32, copy=False)


def encode_mp3(wav: np.ndarray, sample_rate: int, dest_mp3: Path,
               bitrate: str = MP3_BITRATE) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; required to encode MP3.")
    dest_mp3.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), wav, sample_rate, subtype="PCM_16")
        partial = dest_mp3.with_suffix(dest_mp3.suffix + ".part")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(tmp_path),
             "-ac", "1", "-codec:a", "libmp3lame", "-b:a", bitrate,
             "-f", "mp3", str(partial)],
            check=True,
        )
        partial.replace(dest_mp3)
    finally:
        tmp_path.unlink(missing_ok=True)
