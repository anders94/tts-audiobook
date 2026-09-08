from __future__ import annotations

from pathlib import Path

import numpy as np

from . import device
from .config import WHISPER_MODEL_ID

_whisper_model = None


def _preload_cuda_libs() -> None:
    """ctranslate2 dlopens libcublas/libcudnn by soname; the wheels only ship
    them inside pip's nvidia-* packages, so preload them by absolute path."""
    import ctypes
    import site

    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for lib in sorted(Path(sp).glob("nvidia/*/lib/lib*.so*")):
            if lib.name.count(".so") and not lib.is_dir():
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def _whisper(force_cpu: bool = False):
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # Try CUDA float16 first; fall back to CPU int8 if CUDA isn't usable
        # for ctranslate2 (which has no MPS backend, so Macs go straight to
        # CPU int8 — fast enough for base.en). CUDA failures can also surface
        # lazily at the first transcribe call — _run() handles that by
        # rebuilding on CPU.
        if not force_cpu and device.cuda_possible():
            try:
                _preload_cuda_libs()
                _whisper_model = WhisperModel(WHISPER_MODEL_ID, device="cuda",
                                              compute_type="float16")
            except Exception:
                _whisper_model = None
        if _whisper_model is None:
            _whisper_model = WhisperModel(WHISPER_MODEL_ID, device="cpu",
                                          compute_type="int8")
    return _whisper_model


def _run(audio, language: str) -> str:
    global _whisper_model
    try:
        segments, _ = _whisper().transcribe(audio, language=language, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()
    except RuntimeError:
        # Lazy CUDA load failed mid-call; rebuild on CPU and retry once.
        _whisper_model = None
        segments, _ = _whisper(force_cpu=True).transcribe(
            audio, language=language, beam_size=5)
        return " ".join(seg.text.strip() for seg in segments).strip()


def transcribe_file(path: Path, language: str = "en") -> str:
    return _run(str(path), language)


def transcribe_words(wav: np.ndarray, sample_rate: int,
                     language: str = "en") -> list[tuple[str, float, float]]:
    """Transcribe with per-word timestamps: [(word, start_s, end_s), ...]."""
    global _whisper_model
    audio = _to_16k(wav, sample_rate)

    def run(model):
        segments, _ = model.transcribe(audio, language=language, beam_size=5,
                                       word_timestamps=True)
        out: list[tuple[str, float, float]] = []
        for seg in segments:
            for w in seg.words or []:
                out.append((w.word.strip(), float(w.start), float(w.end)))
        return out

    try:
        return run(_whisper())
    except RuntimeError:
        _whisper_model = None
        return run(_whisper(force_cpu=True))


def _to_16k(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    # faster-whisper expects float32 mono at 16 kHz for ndarray input.
    audio = wav.astype(np.float32, copy=False)
    if sample_rate != 16000:
        n = int(round(len(audio) * 16000 / sample_rate))
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    return audio


def transcribe_array(wav: np.ndarray, sample_rate: int, language: str = "en") -> str:
    audio = _to_16k(wav, sample_rate)
    if len(audio) == 0:
        return ""
    return _run(audio, language)
