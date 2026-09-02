from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from . import config

_WORD_RE = re.compile(r"[a-z0-9']+")

_SMALL_NUMBERS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty", "30": "thirty",
    "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy",
    "80": "eighty", "90": "ninety",
}


def normalize_for_wer(text: str) -> list[str]:
    """Tokenize for WER: casefold, unify unicode punctuation, strip the rest."""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("’", "'").replace("‘", "'")
    tokens = _WORD_RE.findall(text.lower())
    return [_SMALL_NUMBERS.get(t, t) for t in tokens]


def wer(ref: list[str], hyp: list[str]) -> float:
    """Word error rate via Levenshtein distance; 0.0 for empty ref and hyp."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1] / len(ref)


def wer_against(ref_text: str, hyp_text: str,
                allowlist: set[str] | None = None) -> float:
    """WER after normalization; substitutions against allowlisted proper nouns
    (character/place names Whisper predictably mangles) are forgiven."""
    ref = normalize_for_wer(ref_text)
    hyp = normalize_for_wer(hyp_text)
    if allowlist:
        allow = {w for name in allowlist for w in normalize_for_wer(name)}
        # Forgive proper nouns Whisper predictably mangles — but only names
        # that actually occur in THIS line. Fuzzy matching against the whole
        # roster eats ordinary words ("single"≈"bingley", "man in"≈"maria")
        # and inflates WER on perfectly good audio.
        present = {w for w in allow if w in ref}
        if present:
            # Drop the names from the reference, and drop hypothesis words —
            # including adjacent pairs Whisper split ("nether field") — that
            # are exact or near (edit distance <= 2) renderings of them.
            ref = [w for w in ref if w not in present]
            hyp = _drop_allowlisted(hyp, present)
    return wer(ref, hyp)


def _near_allowlisted(word: str, allow: set[str]) -> bool:
    if word in allow:
        return True
    return any(len(a) >= 5 and abs(len(a) - len(word)) <= 2
               and wer(list(a), list(word)) * len(a) <= 2 for a in allow)


def _drop_allowlisted(hyp: list[str], allow: set[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(hyp):
        if i + 1 < len(hyp) and _near_allowlisted(hyp[i] + hyp[i + 1], allow):
            i += 2
            continue
        if _near_allowlisted(hyp[i], allow):
            i += 1
            continue
        out.append(hyp[i])
        i += 1
    return out


def duration_plausible(text: str, wav: np.ndarray, sample_rate: int) -> bool:
    """Reject runaway generations (Qwen's infinite-loop failure mode)."""
    dur = len(wav) / sample_rate if sample_rate else 0.0
    max_dur = (len(text) / config.QC_CHARS_PER_S) * config.QC_DURATION_FACTOR \
        + config.QC_DURATION_SLACK_S
    return dur <= max_dur


def wer_threshold(ref_text: str) -> float:
    n = max(1, len(normalize_for_wer(ref_text)))
    return max(config.QC_WER_THRESHOLD, 2.0 / n)


@dataclass
class QCResult:
    passed: bool
    wer: float
    reason: str  # "ok" | "duration" | "wer"


def check(text: str, wav: np.ndarray, sample_rate: int,
          transcript: str | None, allowlist: set[str] | None = None) -> QCResult:
    if not duration_plausible(text, wav, sample_rate):
        return QCResult(passed=False, wer=1.0, reason="duration")
    if transcript is None:
        return QCResult(passed=True, wer=0.0, reason="ok")
    w = wer_against(text, transcript, allowlist)
    if w > wer_threshold(text):
        return QCResult(passed=False, wer=w, reason="wer")
    return QCResult(passed=True, wer=w, reason="ok")
