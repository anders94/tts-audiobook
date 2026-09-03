from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_data_dir

# Long renders fragment the CUDA allocator (observed: 7.4 GiB reserved-but-
# unallocated after 34 chapters → OOM). Must be set before torch initializes
# CUDA; config is imported ahead of any engine, so this is the safe spot.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

APP_NAME = "tts-audiobook"
NARRATOR_KEY = "__narrator__"

DATA_DIR = Path(user_data_dir(APP_NAME, appauthor=False))
# v2 keeps its own database; the v1 library.db in the same directory is untouched.
DB_PATH = DATA_DIR / "studio.db"
LIBRARY_DIR = DATA_DIR / "library"     # tagged accent reference clips
REFS_DIR = DATA_DIR / "refs"           # frozen per-book character references
OUTPUT_ROOT = DATA_DIR / "output"

QWEN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
QWEN_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
WHISPER_MODEL_ID = "base.en"

# --- planning ---
# Segments longer than this are split at sentence boundaries before TTS.
# Shorter than v1's 1500: better QC granularity and fewer long-input glitches.
MAX_ITEM_CHARS = 800
BATCH_SIZE = 24
# Cap total characters per batch too: a run of maximum-length items (e.g. a
# 14k-char letter split into ~18 x 800) OOMs a 24 GB card if batched by count
# alone.
BATCH_MAX_CHARS = 4000

# --- pacing (seconds of silence before an item) ---
# Same-voice seams need breath room: generated clips are silence-trimmed, so
# the gap IS the entire pause, and a narrator jumping straight into the next
# sentence sounds inhuman. A speaker CHANGE can be quicker — the incoming
# voice has already "taken its breath" (listening feedback, 2026-09-03).
GAP_QUOTE_CONTINUES = 0.40   # paragraph break inside one character's speech
GAP_MIDSENTENCE_SPLIT = 0.12 # dialogue/narration split mid-sentence ("...," said she)
GAP_ATTRIBUTION_TAG = 0.15   # prev ends with , ; or dash and the speaker changes
GAP_SPEAKER_CHANGE = 0.35
GAP_SAME_SPEAKER = 0.45      # same voice, new segment (usually a new sentence)
GAP_SUBSPLIT = 0.30          # sentence boundary within one long split segment
GAP_SCENE_BREAK = 0.70       # visible gap in source offsets (blank line / scene break)
GAP_AFTER_TITLE = 0.90
SCENE_BREAK_OFFSET_GAP = 8   # source chars between segments that imply a scene break

# --- audio post ---
# Asymmetric trim pads: a tight cut at the head is fine (an inhale can be
# inaudible), but the tail must keep the natural exhale decay — cutting it
# mid-fall is audible — and fade smoothly into the gap (listening feedback,
# 2026-09-03).
TRIM_THRESHOLD_DBFS = -45.0
TRIM_LEAD_PAD_S = 0.03
TRIM_TAIL_PAD_S = 0.18
FADE_IN_S = 0.01
FADE_OUT_S = 0.12
TARGET_LUFS = -19.0
MAX_GAIN_DB = 8.0
PEAK_DBFS = -1.0
MIN_LOUDNORM_S = 0.4         # below this, RMS matching instead of LUFS

# --- QC ---
QC_MAX_ATTEMPTS = 3
QC_WER_THRESHOLD = 0.15
# duration guard: fail generation if audio exceeds chars/CHARS_PER_S * factor + slack
QC_CHARS_PER_S = 15.0
QC_DURATION_FACTOR = 2.5
QC_DURATION_SLACK_S = 3.0

MP3_BITRATE = "96k"
M4B_BITRATE = "64k"

# Reference clips (frozen per-character) target range. v1 cloned from whole
# 17-23s clips without artifacts; truncation — even at a word boundary — makes
# Qwen "continue" the cut-off speech before the target text. Only clips beyond
# the hard cap are shortened, preferring a sentence boundary.
REF_MIN_S = 2.0
REF_MAX_S = 24.0


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
