from __future__ import annotations

import numpy as np

# Kokoro's British English voicepacks (lang_code 'b'). Fully synthetic
# identities with reliable en-GB accents — usable as seed references that
# Qwen then clones, and blendable for new identities.
BRITISH_VOICES = {
    "bf_emma": ("female", "adult"),
    "bf_isabella": ("female", "adult"),
    "bf_alice": ("female", "young_adult"),
    "bf_lily": ("female", "young_adult"),
    "bm_george": ("male", "middle_aged"),
    "bm_lewis": ("male", "adult"),
    "bm_daniel": ("male", "adult"),
    "bm_fable": ("male", "adult"),
}

_pipeline = None


def _pipe():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M")
    return _pipeline


def synthesize(text: str, voice: str, *, blend: str | None = None,
               blend_weight: float = 0.5, speed: float = 1.0
               ) -> tuple[np.ndarray, int]:
    """Speak `text` with a Kokoro British voice (24 kHz mono float32).

    With `blend`, the two voicepack embeddings are mixed
    (1-blend_weight)*voice + blend_weight*blend — a new synthetic identity.
    """
    pipe = _pipe()
    voice_ref: object = voice
    if blend:
        import torch
        v1 = pipe.load_voice(voice)
        v2 = pipe.load_voice(blend)
        voice_ref = ((1.0 - blend_weight) * v1
                     + blend_weight * v2).to(torch.float32)

    chunks: list[np.ndarray] = []
    for _, _, audio in pipe(text, voice=voice_ref, speed=speed):
        arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") \
            else np.asarray(audio)
        chunks.append(np.squeeze(arr).astype(np.float32))
    if not chunks:
        raise RuntimeError("Kokoro produced no audio.")
    return np.concatenate(chunks), 24000
