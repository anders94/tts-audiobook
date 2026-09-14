from __future__ import annotations

import numpy as np

from tts_audiobook.qc import (check, duration_plausible, normalize_for_wer,
                              wer, wer_against, wer_threshold)


def test_normalize_strips_punctuation_and_case():
    assert normalize_for_wer("“My dear Mr. Bennet,”") == ["my", "dear", "mr", "bennet"]
    assert normalize_for_wer("Chapter 7.") == ["chapter", "seven"]
    assert normalize_for_wer("don’t") == ["don't"]


def test_wer_basics():
    assert wer(["a", "b", "c"], ["a", "b", "c"]) == 0.0
    assert wer(["a", "b", "c"], ["a", "x", "c"]) == 1 / 3
    assert wer([], []) == 0.0
    assert wer(["a"], []) == 1.0


def test_wer_against_ignores_formatting():
    assert wer_against("“Well—indeed!”", "well indeed") == 0.0


def test_allowlist_forgives_proper_nouns():
    ref = "They reached Netherfield before noon."
    hyp = "They reached nether field before noon."
    strict = wer_against(ref, hyp)
    forgiving = wer_against(ref, hyp, allowlist={"Netherfield"})
    assert strict > 0.0
    assert forgiving <= strict


def test_allowlist_inert_when_no_name_in_line():
    # Regression: fuzzy matching must not fire for lines without the name —
    # "single"≈"bingley" and "man in"≈"maria" once inflated WER on a
    # perfect transcription.
    allow = {"Bingley", "Maria Lucas", "Mr. Hurst"}
    ref = ("It is a truth universally acknowledged, that a single man in "
           "possession of a good fortune must be in want of a wife.")
    hyp = ("It is a truth universally acknowledged that a single man in "
           "possession of a good fortune must be in want of a wife.")
    assert wer_against(ref, hyp, allow) == 0.0


def test_duration_guard_catches_runaway():
    sr = 24000
    short_text = "Hello."
    runaway = np.zeros(sr * 60, dtype=np.float32)  # 60s of audio for 6 chars
    assert not duration_plausible(short_text, runaway, sr)
    ok = np.zeros(sr * 1, dtype=np.float32)
    assert duration_plausible(short_text, ok, sr)


def test_check_flags_wer_failure():
    sr = 24000
    wav = np.zeros(sr, dtype=np.float32)
    good = check("hello there my friend", wav, sr, "hello there my friend")
    assert good.passed and good.reason == "ok"
    bad = check("hello there my friend how are you today then",
                wav, sr, "completely different words entirely spoken wrong here now")
    assert not bad.passed and bad.reason == "wer"


def test_short_line_threshold_is_lenient():
    # A 2-word line: one substitution is 0.5 WER but threshold must allow 2/n.
    assert wer_threshold("Yes indeed") >= 1.0


def _voiced(f0_hz: float, seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    """A voice-like tone at f0_hz: five harmonics with 1/k rolloff. (A pure
    sawtooth is too perfectly periodic — Praat picks a subharmonic for it.)"""
    t = np.arange(int(sr * seconds)) / sr
    y = sum((0.3 / k) * np.sin(2 * np.pi * f0_hz * k * t) for k in range(1, 6))
    return y.astype(np.float32)


def test_median_f0_tracks_a_periodic_signal():
    from tts_audiobook.qc import median_f0
    f = median_f0(_voiced(150.0), 16000)
    assert f is not None and abs(f - 150.0) < 8
    f = median_f0(_voiced(300.0), 16000)
    assert f is not None and abs(f - 300.0) < 12
    assert median_f0(np.zeros(16000, dtype=np.float32), 16000) is None
    assert median_f0(_voiced(150.0, seconds=0.05), 16000) is None


def test_check_flags_pitch_drift_against_reference():
    from tts_audiobook.qc import semitones
    text = "Is he married or single?"
    ok = check(text, _voiced(150.0), 16000, "is he married or single", ref_f0=140.0)
    assert ok.passed and ok.reason == "ok"
    assert ok.pitch_dev is not None and abs(ok.pitch_dev - semitones(150, 140)) < 0.5
    drift = check(text, _voiced(300.0), 16000, "is he married or single", ref_f0=140.0)
    assert not drift.passed and drift.reason == "pitch"
    assert drift.pitch_dev is not None and drift.pitch_dev > 12
    # Word errors take precedence over pitch in the reason, and no ref skips pitch.
    bad = check(text, _voiced(300.0), 16000, "totally different words", ref_f0=140.0)
    assert bad.reason == "wer" and bad.pitch_dev is not None
    assert check(text, _voiced(300.0), 16000, None).passed


def test_better_prefers_passing_then_wer_then_pitch():
    from tts_audiobook.qc import QCResult, better
    passed = QCResult(True, 0.1, "ok", pitch_dev=5.0)
    failed = QCResult(False, 0.0, "pitch", pitch_dev=9.0)
    assert better(passed, failed) and not better(failed, passed)
    assert better(QCResult(False, 0.2, "wer"), QCResult(False, 0.4, "wer"))
    near = QCResult(False, 0.0, "pitch", pitch_dev=7.0)
    far = QCResult(False, 0.0, "pitch", pitch_dev=-9.0)
    assert better(near, far) and not better(far, near)

