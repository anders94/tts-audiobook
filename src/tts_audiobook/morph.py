from __future__ import annotations

import numpy as np


def morph(wav: np.ndarray, sample_rate: int, *,
          formant_ratio: float = 1.0,
          pitch_semitones: float = 0.0,
          pitch_range_factor: float = 1.0,
          tempo: float = 1.0) -> np.ndarray:
    """Derive a new-sounding voice from a recording, keeping its accent.

    Praat's "Change gender" resynthesis shifts pitch and formants
    independently: formant_ratio > 1 shortens the apparent vocal tract
    (younger/more feminine), < 1 lengthens it; pitch_semitones moves the
    median F0; pitch_range_factor widens (>1) or flattens (<1) intonation.
    Accent lives in phoneme timing and vowel *relationships*, which survive
    uniform shifts — so a ±2–3 semitone + 0.9–1.15 formant morph sounds like
    a different person with the same accent.
    """
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(wav.astype(np.float64), sampling_frequency=sample_rate)

    pitch = snd.to_pitch()
    median_f0 = call(pitch, "Get quantile", 0, 0, 0.5, "Hertz")
    if not np.isfinite(median_f0) or median_f0 <= 0:
        median_f0 = 150.0
    new_median = median_f0 * (2.0 ** (pitch_semitones / 12.0))

    # tempo > 1 = faster speech. Praat's duration factor is time-stretch,
    # i.e. the inverse. Rhythm is a strong identity cue that pitch/formant
    # shifts don't touch, so it helps de-identify recognizable sources.
    morphed = call(snd, "Change gender", 75, 600,
                   formant_ratio, new_median, pitch_range_factor, 1.0 / tempo)
    out = morphed.values[0].astype(np.float32)

    # Praat can clip slightly on resynthesis; normalize peaks defensively.
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out


# Named presets so casting variety doesn't require remembering Praat numbers.
#
# Listening tests (2026-09-03): coherent same-direction pitch+formant shifts
# with a slight tempo change stay human; incongruent shifts (pitch down +
# formants up) and stacked intonation-range changes sound fake. The headroom
# is ASYMMETRIC: -4 semitones / 0.87 formants still reads as a real person,
# but the upward mirror (+4 / 1.13) is instant cartoon-chipmunk — upward
# morphs must stay gentle. Keep morphs coherent and modest.
PRESETS: dict[str, dict[str, float]] = {
    # validated by ear ("fry_A_deep_strong")
    "deeper":  {"pitch_semitones": -4.0, "formant_ratio": 0.87, "tempo": 1.08},
    # upward shifts chipmunk quickly; this stays under the observed ceiling
    "lighter": {"pitch_semitones": 2.5, "formant_ratio": 1.07, "tempo": 0.96},
    # gentler variants for separating similar characters, not de-identifying
    "older":   {"pitch_semitones": -2.0, "formant_ratio": 0.94, "tempo": 0.95},
    "younger": {"pitch_semitones": 1.5, "formant_ratio": 1.05, "tempo": 1.05},
}
