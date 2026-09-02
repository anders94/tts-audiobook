from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


class SubprocEngine:
    """Engine that shells out to a worker in a separate venv.

    Used for chatterbox, whose dependency pins (torch==2.6.0, its own
    transformers) cannot coexist with qwen-tts. Set CHATTERBOX_PYTHON to the
    python of a venv with chatterbox-tts installed.
    """

    max_batch = 1  # chatterbox generates one text at a time

    def __init__(self, name: str) -> None:
        self.name = name
        self._proc: subprocess.Popen | None = None

    def _python(self) -> str:
        env_var = f"{self.name.upper()}_PYTHON"
        python = os.environ.get(env_var)
        if not python or not Path(python).exists():
            raise RuntimeError(
                f"{env_var} must point at a python with {self.name}-tts installed "
                f"(see README: this engine needs its own venv).")
        return python

    def load(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        worker = Path(__file__).parent / f"{self.name}_worker.py"
        self._proc = subprocess.Popen(
            [self._python(), "-u", str(worker)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        self._request({"cmd": "load"})

    def _request(self, req: dict) -> dict:
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            code = self._proc.poll()
            raise RuntimeError(f"{self.name} worker died (exit code {code}).")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"{self.name} worker error: {reply.get('error')}")
        return reply

    def clone_prompt(self, ref_audio: Path, ref_text: str) -> Any:
        # Chatterbox conditions directly on the reference wav per call; the
        # "prompt" is just the path.
        return str(ref_audio)

    def generate(self, texts: list[str], prompt: Any, *,
                 language: str = "English", seed: int | None = None
                 ) -> tuple[list[np.ndarray], int]:
        self.load()
        with tempfile.TemporaryDirectory(prefix="cbx-") as tmp:
            reply = self._request({
                "cmd": "generate",
                "texts": texts,
                "ref_audio": prompt,
                "seed": seed or 0,
                "language_id": "en" if language.lower().startswith("en") else language[:2].lower(),
                "out_dir": tmp,
            })
            wavs = []
            for p in reply["wavs"]:
                wav, _ = sf.read(p, dtype="float32", always_2d=True)
                wavs.append(wav.mean(axis=1).astype(np.float32))
        return wavs, int(reply["sr"])

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._request({"cmd": "quit"})
            except Exception:
                pass
            self._proc.terminate()
        self._proc = None
