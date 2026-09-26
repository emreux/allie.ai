"""What every speech-to-text engine is reduced to (design.md section 3.4).

The audio itself is the same everywhere - one channel, 16 kHz, float32 in
[-1, 1] - so what the providers actually disagree about is the shape of the
answer and whether it arrives all at once. Both are settled here.

**There is one call, and it hands over a whole utterance.** The protocol
carried a streaming call as well - `transcribe_stream`, `supports_streaming`,
`Transcript.is_final` and a shared `buffered_stream` body for the providers
that could not stream - kept from the first day so that partial transcripts
would not be a protocol change later. Neither provider ever streamed, and the
live product removed the reason to: the model hears the user itself, and all
that reaches a recogniser here is the yes or no of a confirmation window
(D3, D10). The dead half was removed on 2026-09-22; a streaming engine, if one
ever matters again, is a new method and not a resurrection of that one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "LEVEL_FLOOR_DBFS",
    "NO_SPEECH_CEILING",
    "SAMPLE_RATE",
    "Audio",
    "STTProvider",
    "Transcript",
    "dbfs",
    "from_pcm16",
    "to_pcm16",
]

# What the live model and Google's recogniser take, and what every engine
# accepts. `audio/` captures at this rate so that nothing in the pipeline has
# to resample.
SAMPLE_RATE = 16_000

# At or above this, the engine itself says the audio held no speech. Measured
# on the target machine with the local engine of the time (2026-09-05, Whisper
# `small` int8, gone since D36): real sentences 0.01-0.12, the same sentence
# through a hard noise gate 0.63, silence and noise 0.86-0.92. Google's
# recogniser reports 1.0 when no final came (`gemini_stt.NOTHING_TO_DECODE`),
# and `app.hear` reads it; here because it is the protocol's number.
NO_SPEECH_CEILING = 0.8

Audio = NDArray[np.float32]


def to_pcm16(pcm: Audio) -> bytes:
    """One channel, 16-bit little-endian: what a live session is fed, at the
    rate the audio already has.

    Values outside [-1, 1] are clipped rather than wrapped - a wrapped
    sample is a click the recogniser hears as a consonant. Here rather than
    in the Gemini recogniser that first needed it, because the live capture
    (`audio/capture.py`) sends the same bytes.
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


@dataclass(frozen=True, slots=True)
class Transcript:
    """What was heard, whether anything was, and how sure the engine is.

    `language` is filled in by the provider rather than by the caller. It is
    what the user actually spoke, which is not necessarily the language the
    assistant was configured for - and section 3.12 lets those differ on
    purpose, because the reply mirrors the speaker.

    `no_speech_probability` and `confidence` answer different questions and
    must not be confused. The first is how likely the engine thinks the audio
    held no speech at all; `None` is "no opinion" - a hosted engine that
    answers silence with an empty text - and 1.0 is "there was nothing to
    decode". The second is how sure the decoder was of its words, which is low
    for a correct single word as well as for a hallucination, and is therefore
    kept for the log and never used to decide anything (`app.hear`).
    """

    text: str
    confidence: float | None = None
    language: str = ""
    no_speech_probability: float | None = None


@runtime_checkable
class STTProvider(Protocol):
    """A speech-to-text engine, local or hosted."""

    id: str

    async def transcribe(self, pcm: Audio, *, hint: str | None = None) -> Transcript:
        """Turns a whole utterance into one final transcript.

        `hint` is the language the audio is expected to be in, as an ISO 639-1
        code. It is a hint and not a setting: section 3.12's `fixed` mode
        passes the locale's language, `auto` passes nothing, and the provider
        reports back in `Transcript.language` what it actually heard.
        """
        ...
