"""The microphone's audio as every part of the pipeline takes it.

One channel, 16 kHz, float32 in [-1, 1]: what the capture records, the
detector and the wake word read, and - as 16-bit PCM - what the live model
is sent. Until 2026-10-11 this was the speech-to-text protocol as well
(design.md section 3.4); the last engine behind it, Google's recogniser for
the yes or no of a confirmation window (D36), went with D54: the live model
hears that answer itself (`agent/referee.py`). What stayed is the format.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "LEVEL_FLOOR_DBFS",
    "SAMPLE_RATE",
    "Audio",
    "dbfs",
    "from_pcm16",
    "to_pcm16",
]

# What the live model takes. `audio/` captures at this rate so that nothing
# in the pipeline has to resample.
SAMPLE_RATE = 16_000

Audio = NDArray[np.float32]


def to_pcm16(pcm: Audio) -> bytes:
    """One channel, 16-bit little-endian: what a live session is fed, at the
    rate the audio already has.

    Values outside [-1, 1] are clipped rather than wrapped - a wrapped
    sample is a click a model hears as a consonant. Here rather than in the
    Gemini recogniser that first needed it (gone since D54), because the
    live capture (`audio/capture.py`) and the referee send the same bytes.
    """
    return (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def from_pcm16(data: bytes) -> Audio:
    """The other way: sixteen bit little-endian bytes as float samples in
    [-1, 1] - what the player measures before a block goes to the device."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


# What a block of nothing measures as: under the floor of `ui/orb.py`'s
# scale, and a real number rather than the -inf of log10(0).
LEVEL_FLOOR_DBFS = -90.0


def dbfs(audio: Audio) -> float:
    """The RMS level of `audio` in dBFS, floored at `LEVEL_FLOOR_DBFS`.

    One number per block for whoever draws the sound (`ui/window.py`);
    the capture's own `level_dbfs` averages over a stream and skips
    silence, which is the other question.
    """
    if not len(audio):
        return LEVEL_FLOOR_DBFS
    rms = math.sqrt(float(np.dot(audio, audio)) / len(audio))
    if rms <= 0.0:
        return LEVEL_FLOOR_DBFS
    return max(LEVEL_FLOOR_DBFS, 20 * math.log10(rms))
