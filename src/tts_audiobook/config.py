from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_dir

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

# --- pacing (seconds of silence before an item) ---
GAP_QUOTE_CONTINUES = 0.05   # quotation continuing across a paragraph break
GAP_MIDSENTENCE_SPLIT = 0.12 # dialogue/narration split mid-sentence ("...," said she)
GAP_ATTRIBUTION_TAG = 0.15   # prev ends with , ; or dash and the speaker changes
GAP_SPEAKER_CHANGE = 0.35
GAP_SAME_SPEAKER = 0.15
GAP_SUBSPLIT = 0.06          # between sentence-split halves of one long segment
GAP_SCENE_BREAK = 0.70       # visible gap in source offsets (blank line / scene break)
GAP_AFTER_TITLE = 0.90
SCENE_BREAK_OFFSET_GAP = 8   # source chars between segments that imply a scene break

# --- audio post ---
TRIM_THRESHOLD_DBFS = -45.0
TRIM_PAD_S = 0.03
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
