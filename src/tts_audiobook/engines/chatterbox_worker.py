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
        device = "cuda" if torch.cuda.is_available() else "cpu"
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
    for i, text in enumerate(req["texts"]):
        torch.manual_seed(seed)
        wav = model.generate(
            text,
            audio_prompt_path=req["ref_audio"],
            language_id=req.get("language_id", "en"),
        )
        wav = wav.squeeze(0).cpu().numpy()
        dest = out_dir / f"chunk_{i:04d}.wav"
        import soundfile as sf
        sf.write(str(dest), wav, sr, subtype="PCM_16")
        paths.append(str(dest))
    return {"ok": True, "wavs": paths, "sr": int(sr)}


def main() -> None:
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
                print(json.dumps({"ok": True}), flush=True)
                return
            else:
                reply = {"ok": False, "error": f"unknown cmd {req.get('cmd')!r}"}
        except Exception as e:  # noqa: BLE001 — everything crosses as JSON
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
