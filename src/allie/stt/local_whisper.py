"""Whisper on the CPU, off the event loop (design.md section 3.4).

The target machine has no discrete GPU, so this runs `faster-whisper` with int8
weights on four physical cores. That is seconds of work per utterance, and the
single most important thing in this file is where those seconds are spent:
**in a worker thread, never on the event loop.** Rule 4 of section 3.1 gives
anything awaited in `app.py` a 50 ms budget; a transcription on the loop would
freeze the reminder scheduler, the announce queue and audio capture together
until it finished. CTranslate2 releases the GIL, so a thread is enough and a
second process is not needed.

The model is loaded on first use, also in a thread - about a gigabyte of
weights - and `load()` exists so `app.py` can pay that cost at startup instead
of inside the user's first sentence. When the weights cannot be loaded - no
network on the first run, a broken cache - the failure is named
(`ModelUnavailableError`), so that `allie run` can say so in a sentence.

**The recording is filtered for speech before it is decoded.** `vad_filter`
runs the same Silero network `audio/vad.py` uses over the whole recording and
hands the decoder only what it kept. Measured on this machine (2026-09-05): a
recording of a quiet room produced a subtitle credit without the filter and no
segment at all with it, while a real sentence came back unchanged. What is
left after that is judged by the decoder's own `no_speech_prob` per segment -
a hallucinated credit over the trailing silence scores 0.9 next to a real
sentence at 0.05 - and reported to the state machine as one number.

**The recogniser is told what to expect.** Section 3.4's free trick:
`initial_prompt` is context for the decoder, and what it is given is whatever
the caller says the next utterance will be - `app.confirm_prompt`, the yes and
no words of the user's pack (D3, D10). It arrives ready to use: nothing here
fits, trims or renders it.

Until 2026-09-22 this file fitted a *vocabulary* into the prompt instead - the
names of the installed applications, in the order the user had opened them,
as many as 120 tokens held - because the pipeline it came from sent every
sentence the user spoke through this decoder. The live model hears those
sentences now, and a prompt of application names cost the confirmation window
about 0.8 s of decode (measured 2026-09-13: roughly 0.65 s per hundred tokens)
to bias it towards words nobody says in it. The budget is worth remembering
all the same: the library keeps only the *last* 223 tokens of a prompt, so a
pack that writes an essay loses the front of it.

**One decode, so many tokens, and a loop is not words.** The library's
default retries a decode it doubts at five rising temperatures. Measured
2026-09-13 with `small` and a synthetic Turkish voice: "PyCharm'ı aç" was
decoded six times over 25-32 s and was wrong at the end as at the start;
"FortiClient VPN'i aç" took 15-18 s for the same wrong words a single decode
gives in 3 s. So the decode runs once, at temperature zero. What the ladder
was also catching - a decoder going round in circles ("TÜCHAR MAĞĞĞĞ...",
448 tokens, 13.7 s) - is bounded instead by `max_new_tokens`, from the
length of the audio, and then thrown away by its compression ratio, the
library's own sign of a loop. The transcript is then empty over audio that
held speech, and `app.hear` answers with "say it again" rather than handing
the model a word nobody said.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from loguru import logger

from allie.stt.base import NO_SPEECH_CEILING, SAMPLE_RATE, Audio, Transcript

__all__ = [
    "COMPRESSION_CEILING",
    "TOKENS_AT_LEAST",
    "TOKENS_PER_SECOND",
    "LocalWhisper",
    "ModelUnavailableError",
]

# Measured on the target machine (section 3.4): `small` int8 transcribes a
# short sentence in about 2.5 s and the translation is faithful.
DEFAULT_MODEL_SIZE = "small"

# Physical cores, not hyperthreads. The remaining capacity is what keeps audio
# playback and the state machine responsive while the model works.
DEFAULT_CPU_THREADS = 4

# How many tokens the decoder may write for an utterance: this many at
# least, plus this many per second of audio. Turkish speech decodes at about
# five tokens a second (14 tokens in 2.8 s, measured 2026-09-13); twice that
# is the ceiling, so a real sentence is never cut and a loop is.
TOKENS_AT_LEAST = 24
TOKENS_PER_SECOND = 10

# A segment whose text compresses better than this is the decoder repeating
# itself, not speech. The library's own default for the same judgement.
COMPRESSION_CEILING = 2.4

# `WhisperModel`, kept as `Any` so this module has no import-time dependency on
# the vendor package - see `_load_whisper`.
Model = Any
ModelFactory = Callable[[], Model]


class ModelUnavailableError(RuntimeError):
    """The weights could not be loaded: no network on the first run, a broken
    cache, a disk that is full. Fixable by the user, so named for `run`."""


class LocalWhisper:
    """`faster-whisper`, int8, on the CPU."""

    id = "local_whisper"

    def __init__(
        self,
        *,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = DEFAULT_CPU_THREADS,
        prompt: str = "",
        build: ModelFactory | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        # What the decoder is told the next utterance will be
        # (`app.confirm_prompt`). `None` rather than "" when there is nothing
        # to say: the library's own way of meaning no prompt at all.
        self._prompt: str | None = prompt.strip() or None
        self._build = build if build is not None else self._load_whisper
        self._model: Model | None = None
        # Held while the model is built. Two turns starting at once would
        # otherwise read the weights into memory twice.
        self._loading = threading.Lock()

    async def load(self) -> None:
        """Loads the model now, so the first sentence does not wait for it."""
        await asyncio.to_thread(self._model_now)

    async def transcribe(self, pcm: Audio, *, hint: str | None = None) -> Transcript:
        return await asyncio.to_thread(self._transcribe_now, pcm, hint)

    # ----------------------------------------------------------------------
    # Everything below this line runs in a worker thread.
    # ----------------------------------------------------------------------

    def _model_now(self) -> Model:
        with self._loading:
            if self._model is None:
                try:
                    self._model = self._build()
                except Exception as failure:
                    # Whatever the download or the loader raised, the user's
                    # next move is the same: check the network, or the cache.
                    raise ModelUnavailableError(
                        f"the speech model {self._model_size!r} could not be loaded: {failure}"
                    ) from failure
                logger.info("recogniser prompt: {!r}", self._prompt)
            return self._model

    def _transcribe_now(self, pcm: Audio, hint: str | None) -> Transcript:
        # `language=None` is what asks Whisper to detect the language itself.
        # `vad_filter` strips what its own detector calls silence before the
        # decoder sees it, so a recording of nothing decodes to nothing.
        # `initial_prompt` is the fitted prompt, or `None`. `temperature` is
        # one number: one decode, no ladder. `max_new_tokens` is what the
        # audio could hold (module docstring).
        model = self._model_now()
        segments, info = model.transcribe(
            pcm,
            language=hint,
            vad_filter=True,
            initial_prompt=self._prompt,
            temperature=0.0,
            max_new_tokens=TOKENS_AT_LEAST + int(TOKENS_PER_SECOND * len(pcm) / SAMPLE_RATE),
        )

        # The generator is lazy: the inference happens here, inside the thread.
        # Handing it back undrained would move the work onto the event loop.
        decoded = list(segments)

        # A segment the decoder itself marks as probably-not-speech is what it
        # produces over a stretch of noise the filter let through; one that
        # compresses like a loop is the decoder repeating itself. The words
        # are dropped; the numbers are kept, so the caller learns why.
        heard = [segment for segment in decoded if _is_words(segment)]

        return Transcript(
            # Each segment already begins with its own separating space, so
            # joining with another one would double every gap.
            text="".join(segment.text for segment in heard).strip(),
            confidence=_confidence(heard),
            language=info.language or "",
            no_speech_probability=_no_speech(heard, decoded, kept_seconds=info.duration_after_vad),
        )

    def _load_whisper(self) -> Model:
        # Imported here rather than at module scope: it pulls in CTranslate2
        # and its native libraries, and `allie --help` has no use for them.
        # The package ships no type information, which is why `Model` is `Any`.
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        return WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
            cpu_threads=self._cpu_threads,
        )


def _is_words(segment: Any) -> bool:
    if segment.no_speech_prob >= NO_SPEECH_CEILING:
        return False
    if segment.compression_ratio > COMPRESSION_CEILING:
        logger.debug(
            "dropped a looping segment: ratio {:.1f}, {} tokens",
            segment.compression_ratio,
            len(segment.tokens),
        )
        return False
    return True


def _no_speech(heard: Sequence[Any], decoded: Sequence[Any], *, kept_seconds: float) -> float:
    """How likely it is that nothing was said, as one number for the caller.

    The lowest `no_speech_prob` among the segments that survived, because one
    segment of real speech means somebody spoke. When none survived, the
    lowest among those that were dropped - the engine's own verdict rather
    than ours: over the ceiling when they were noise, under it when they were
    a loop over real speech, which the caller then reports as speech it could
    not read. When the decoder produced nothing at all, the filter decides -
    it kept no audio, so there was nothing to say, or it kept some and the
    decoder made nothing of it, which is speech the user deserves to be told
    was not understood.
    """
    if heard:
        return float(min(segment.no_speech_prob for segment in heard))
    if decoded:
        return float(min(segment.no_speech_prob for segment in decoded))
    return 1.0 if kept_seconds <= 0 else 0.0


def _confidence(segments: Iterable[Any]) -> float | None:
    """Turns Whisper's log probabilities into one number between 0 and 1.

    `avg_logprob` is the mean log probability of the tokens in a segment, which
    says nothing to the rest of the application on its own. Weighting each
    segment by how many tokens it holds and exponentiating gives the geometric
    mean probability per token.

    It is a measure of the decoder's doubt, not of whether anything was said,
    and it is biased by length: measured 2026-09-05, a correct "Merhaba." alone
    scored 0.48 and a recording of silence 0.54. It is reported for the log and
    the screen and is not what any decision rests on - `Transcript` says which
    number is.
    """
    weighted = 0.0
    tokens = 0
    for segment in segments:
        count = len(segment.tokens)
        weighted += segment.avg_logprob * count
        tokens += count

    if tokens == 0:
        return None
    return math.exp(weighted / tokens)
