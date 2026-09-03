# tts-audiobook v2

Spec-driven audiobook studio. Consumes the annotated book JSON produced by
[gutenberg-reader](../gutenberg-reader) — including the `production` block and
per-character `voice` specs (sex, age band, accent, register, pitch, timbre) —
casts each character against a tagged library of accent reference clips, and
renders the book with a local voice-cloning TTS engine on CUDA.

Design principle: **design once, clone forever.** Every character gets one
frozen reference clip (picked from the accent library, optionally shaped by a
voice-design model); the entire book is rendered by cloning that fixed
reference, which is what keeps a 12-hour performance consistent.

## Setup

```bash
uv sync --extra qwen          # main venv: Qwen3-TTS engine + whisper QC
```

The optional Chatterbox engine cannot share this venv (it pins conflicting
torch/transformers versions). Give it its own venv and point the app at it:

```bash
uv venv ~/.venvs/chatterbox --python 3.11
# setuptools<81 is required: chatterbox's perth watermarker imports
# pkg_resources, removed in newer setuptools, and fails silently without it
# ("TypeError: 'NoneType' object is not callable" at load).
VIRTUAL_ENV=~/.venvs/chatterbox uv pip install chatterbox-tts "setuptools<81"
export CHATTERBOX_PYTHON=~/.venvs/chatterbox/bin/python
```

`ffmpeg` is required on PATH; `ffplay` is used for auditioning.

Harmless startup warnings: qwen-tts complains that `flash-attn` is not
installed (optional CUDA kernels; PyTorch SDPA attention is used instead) and
that `sox` is missing (probed by an audio dependency, unused in our path —
`sudo apt install sox` silences it).

Reference clips are cloned whole. Do not truncate them: a reference that
stops mid-passage makes Qwen "continue" the cut-off speech before the target
text, prepending ~1s of stray words to every rendered segment.

## Workflow

```bash
# 1. Build the accent clip library (once, shared across books)
tts-audiobook library import clip.wav --sex female --age-band young_adult \
    --locale en-GB --region "Southern England" --source vctk --license CC-BY-4.0
tts-audiobook library list

# 1b. Or create voices instead of importing recordings:
#     - fully synthetic en-GB seeds via Kokoro (blendable for new identities)
tts-audiobook library synth --voice bf_emma
tts-audiobook library synth --voice bm_george --blend bm_lewis --blend-weight 0.5
#     - derive a new-sounding voice from any clip, keeping its accent
#       (independent pitch/formant shifts via Praat; presets: deeper, lighter,
#        older, younger, flatter, livelier)
tts-audiobook library morph 3 --preset deeper
tts-audiobook library morph 3 --pitch 2.0 --formant 1.06 --age-band young_adult
# (cast --design generates voices from specs with Qwen VoiceDesign — fully
#  synthetic but accent drifts American; use library clips for accent-critical
#  books)

# 2. Inspect a book: structure, speakers, voice-spec coverage
tts-audiobook inspect 1342-pride-and-prejudice.json

# 3. Cast: deterministic spec→clip matching, frozen references built
tts-audiobook cast 1342-pride-and-prejudice.json

# 4. Audition: listen to one line per voice; accept / reroll / override
tts-audiobook audition 1342-pride-and-prejudice.json

# 5. (optional) Compare engines on one chapter
tts-audiobook bakeoff 1342-pride-and-prejudice.json --chapter 3

# 6. Render (resumable per chapter; QC pass with auto-retry)
tts-audiobook perform 1342-pride-and-prejudice.json

# 7. Package: .m4b with chapter markers + ID3-tagged MP3s + RSS feed
tts-audiobook package 1342-pride-and-prejudice.json
tts-audiobook qc-report 1342-pride-and-prejudice.json
```

## Accent clip sources

- [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) — CC BY 4.0, 109
  speakers with UK regional accents. Attribution: "This product includes speech
  data from the CSTR VCTK Corpus (University of Edinburgh)."
- LibriVox — public domain.
- Mozilla Common Voice — CC0, accent-tagged.

## Development

```bash
uv run pytest
```
