from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import audio as audiomod
from . import config, device
from .book import slugify
from .library import sha256_file
from .specs import VoiceSpec


@dataclass
class FrozenRef:
    path: Path
    transcript: str
    sha256: str
    design_seed: int | None


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return wav.mean(axis=1).astype(np.float32), sr


def _cut_at_word_boundary(wav: np.ndarray, sr: int,
                          max_s: float) -> tuple[np.ndarray, str]:
    """Shorten to <= max_s, preferring the end of a sentence.

    A reference that stops mid-passage — even on a complete word — makes the
    model continue the cut-off speech before the target text. Ending on
    terminal punctuation keeps the final prosody closed. Returns
    (audio, transcript) where the transcript is exactly the words kept.
    """
    from .asr import transcribe_words

    words = transcribe_words(wav, sr)
    kept = [w for w in words if w[2] <= max_s - 0.15]
    if not kept:
        # No usable word timings; fall back to a hard cut and let the caller
        # re-transcribe the truncated audio.
        return wav[: int(max_s * sr)], ""
    sentence_ends = [i for i, w in enumerate(kept)
                     if w[0].rstrip("\"”’')").endswith((".", "!", "?", "…"))]
    if sentence_ends:
        kept = kept[: sentence_ends[-1] + 1]
    cut = min(len(wav), int((kept[-1][2] + 0.1) * sr))
    return wav[:cut], " ".join(w[0] for w in kept)


def build_reference(*, book_key: str, character: str, clip_path: Path,
                    clip_transcript: str) -> FrozenRef:
    """Freeze a reference: trim, normalize, write an immutable per-book wav.

    The frozen file—not the library clip—is what the engine clones for every
    line of the book, so its bytes are hashed and verified before rendering.
    """
    wav, sr = _load_mono(clip_path)
    wav = audiomod.trim_silence(wav, sr)
    wav = audiomod.normalize_loudness(wav, sr)

    max_frames = int(config.REF_MAX_S * sr)
    if len(wav) > max_frames:
        # Qwen hangs on very long references, but a hard cut mid-word makes
        # the model "finish" the truncated speech before the target text —
        # audible as ~1s of stray words at the start of every rendered
        # segment. Cut at a word boundary instead, and rebuild the transcript
        # from exactly the kept words so audio and transcript always match.
        wav, clip_transcript = _cut_at_word_boundary(wav, sr, config.REF_MAX_S)

    dest_dir = config.REFS_DIR / book_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{slugify(character)}.wav"
    sf.write(str(dest), wav, sr, subtype="PCM_16")

    transcript = clip_transcript
    if not transcript:
        from .asr import transcribe_file
        transcript = transcribe_file(dest)

    return FrozenRef(path=dest, transcript=transcript,
                     sha256=sha256_file(dest), design_seed=None)


# A neutral, punctuation-complete passage long enough (~10s) to capture the
# designed voice's character for cloning.
CALIBRATION_TEXT = ("The evening settled softly over the quiet town. "
                    "She closed the book, listened to the rain against the "
                    "window, and decided that tomorrow would look after itself.")

_design_model = None


def _design(instruct: str, seed: int) -> tuple[np.ndarray, int]:
    global _design_model
    import torch
    from qwen_tts import Qwen3TTSModel

    if _design_model is None:
        kwargs = device.load_kwargs()
        if device.is_cuda():
            try:
                _design_model = Qwen3TTSModel.from_pretrained(
                    config.QWEN_DESIGN_MODEL_ID,
                    attn_implementation="flash_attention_2", **kwargs)
            except Exception:
                pass
        if _design_model is None:
            _design_model = Qwen3TTSModel.from_pretrained(
                config.QWEN_DESIGN_MODEL_ID, **kwargs)
    torch.manual_seed(seed)
    wavs, sr = _design_model.generate_voice_design(
        text=[CALIBRATION_TEXT], language=["English"], instruct=[instruct])
    return np.squeeze(np.asarray(wavs[0], dtype=np.float32)), int(sr)


def build_designed_reference(*, book_key: str, character: str, spec: VoiceSpec,
                             seed: int) -> FrozenRef:
    """Shape a reference with the Qwen VoiceDesign model from the spec text.

    Fully synthetic identity — no real person. The designed audio is frozen
    exactly like a library clip; the transcript is re-derived with Whisper
    because a designed read can deviate from the calibration text, and a
    transcript mismatch causes reference bleed in ICL cloning.
    """
    wav, sr = _design(spec.describe(), seed)

    dest_dir = config.REFS_DIR / book_key
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / f"{slugify(character)}.design.wav"
    sf.write(str(tmp), wav, sr, subtype="PCM_16")
    try:
        ref = build_reference(book_key=book_key, character=character,
                              clip_path=tmp, clip_transcript="")
    finally:
        tmp.unlink(missing_ok=True)
    return FrozenRef(path=ref.path, transcript=ref.transcript,
                     sha256=ref.sha256, design_seed=seed)
