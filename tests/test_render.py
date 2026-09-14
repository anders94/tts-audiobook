from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_qc import _voiced

from tts_audiobook.planning import RenderItem
from tts_audiobook.render import _pick_take

SR = 16000
ITEM = RenderItem(index=18, speaker_key="A", text="Is he married or single?",
                  gap_before_s=0.0)
TRANSCRIPT = "is he married or single"


def _takes(*f0s: float) -> list[np.ndarray]:
    return [_voiced(f, seconds=0.8, sr=SR) for f in f0s]


def test_pick_take_prefers_closest_pitch_and_transcribes_lazily():
    calls: list[int] = []

    def transcribe(wav, sr):
        calls.append(len(wav))
        return TRANSCRIPT

    wav, res, kept = _pick_take(_takes(300.0, 150.0, 260.0), SR, ITEM, allowlist=set(),
                                ref_f0=140.0, run_qc=True, transcribe=transcribe)
    assert kept == 2 and res.passed and abs(res.pitch_dev) < 2.0
    assert len(calls) == 1   # only the pitch-closest take needed Whisper


def test_pick_take_later_take_wins_ties():
    _, _, kept = _pick_take(_takes(150.0, 150.0, 150.0), SR, ITEM, allowlist=set(),
                            ref_f0=150.0, run_qc=True, transcribe=lambda w, s: TRANSCRIPT)
    assert kept == 3


def test_pick_take_skips_takes_that_fail_the_word_check():
    seen = []

    def transcribe(wav, sr):
        seen.append(len(seen))
        return "nonsense words here" if len(seen) == 1 else TRANSCRIPT

    _, res, kept = _pick_take(_takes(150.0, 165.0), SR, ITEM, allowlist=set(),
                              ref_f0=150.0, run_qc=True, transcribe=transcribe)
    assert res.passed and res.reason == "ok" and kept == 2


def test_pick_take_falls_back_to_best_when_none_pass():
    _, res, kept = _pick_take(_takes(400.0, 320.0, 360.0), SR, ITEM, allowlist=set(),
                              ref_f0=140.0, run_qc=True, transcribe=lambda w, s: TRANSCRIPT)
    assert not res.passed and res.reason == "pitch" and kept == 2


def test_pick_take_without_reference_or_qc():
    _, res, kept = _pick_take(_takes(150.0, 300.0), SR, ITEM, allowlist=set(),
                              ref_f0=None, run_qc=False)
    assert res.passed and res.pitch_dev is None and kept == 2   # no signal: last take
