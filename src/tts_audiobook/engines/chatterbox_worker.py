"""Standalone Chatterbox worker: runs in its OWN venv (chatterbox-tts pins
torch/transformers versions incompatible with qwen-tts).

Protocol: one JSON object per line on stdin; one JSON reply per line on stdout.
  {"cmd": "load"}
  {"cmd": "generate", "texts": [...], "ref_audio": "/path.wav",
   "seed": 0, "out_dir": "/tmp/x"}
Replies: {"ok": true, ...} or {"ok": false, "error": "..."}.
Generated audio is written as wav files under out_dir; the reply carries paths.

This file must import nothing from tts_audiobook — only stdlib and the
chatterbox venv's packages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_model = None


def _load():
    global _model
    if _model is None:
        import torch
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
            # Chatterbox checkpoints were saved from CUDA; torch.load must be
            # told to map them onto MPS (same patch as chatterbox's own
            # example_for_mac.py).
            _orig_load = torch.load

            def _load_to_mps(*args, **kwargs):
                kwargs.setdefault("map_location", torch.device("mps"))
                return _orig_load(*args, **kwargs)
            torch.load = _load_to_mps
        else:
            device = "cpu"
        _model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    return _model


def _generate(req: dict) -> dict:
    import torch

    model = _load()
    out_dir = Path(req["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(req.get("seed") or 0)
    paths = []
    sr = model.sr
    extra = {}
    if req.get("exaggeration") is not None:
        extra["exaggeration"] = float(req["exaggeration"])
    if req.get("cfg_weight") is not None:
        extra["cfg_weight"] = float(req["cfg_weight"])
    for i, text in enumerate(req["texts"]):
        torch.manual_seed(seed)
        wav = model.generate(
            text,
            audio_prompt_path=req["ref_audio"],
            language_id=req.get("language_id", "en"),
            **extra,
        )
        wav = wav.squeeze(0).cpu().numpy()
        dest = out_dir / f"chunk_{i:04d}.wav"
        import soundfile as sf
        sf.write(str(dest), wav, sr, subtype="PCM_16")
        paths.append(str(dest))
    return {"ok": True, "wavs": paths, "sr": int(sr)}


def main() -> None:
    import traceback

    # Libraries in this venv print progress bars and warnings; keep the JSON
    # reply channel clean by routing all stray stdout to stderr.
    reply_stream = sys.stdout
    sys.stdout = sys.stderr

    def reply_json(obj: dict) -> None:
        reply_stream.write(json.dumps(obj) + "\n")
        reply_stream.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req["cmd"] == "load":
                _load()
                reply = {"ok": True}
            elif req["cmd"] == "generate":
                reply = _generate(req)
            elif req["cmd"] == "quit":
                reply_json({"ok": True})
                return
            else:
                reply = {"ok": False, "error": f"unknown cmd {req.get('cmd')!r}"}
        except Exception as e:  # noqa: BLE001 — everything crosses as JSON
            reply = {"ok": False,
                     "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
        reply_json(reply)


if __name__ == "__main__":
    main()
