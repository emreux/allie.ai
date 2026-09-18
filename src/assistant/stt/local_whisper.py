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
(`ModelUnavailableError`), so that `live-assistant run` can say so in a sentence.

**The recording is filtered for speech before it is decoded.** `vad_filter`
runs the same Silero network `audio/vad.py` uses over the whole recording and
hands the decoder only what it kept. Measured on this machine (2026-09-05): a
recording of a quiet room produced a subtitle credit without the filter and no
segment at all with it, while a real sentence came back unchanged. What is
left after that is judged by the decoder's own `no_speech_prob` per segment -
a hallucinated credit over the trailing silence scores 0.9 next to a real
sentence at 0.05 - and reported to the state machine as one number.

**The recogniser is told what to expect.** Section 3.4's free trick:
`initial_prompt` is context for the decoder. It is the locale pack's sentence
(`[stt] prompt`) with the names of the installed apps where its `{apps}` is -
a sentence rather than a list, because the decoder is being shown what
speech in that language looks like, and with the names *as people say them*
(`tools/system.py`, `AppCatalog.spoken_names`). The library keeps the *last*
223 tokens of a long prompt, which would cut the sentence's own words off the
front, so the fit is done here, once, when the model and its tokenizer are
loaded: names go in, in the offered order, while the whole stays under
`PROMPT_TOKENS`. Until 2026-09-13 the prompt was the forty shortest names -
`Run`, `dfrgui`, `services` - and PyCharm was not among them.

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
from collections.abc import AsyncIterator, Callable, Iterable, Sequence
from typing import Any

from loguru import logger

from assistant.stt.base import NO_SPEECH_CEILING, SAMPLE_RATE, Audio, Transcript, buffered_stream

__all__ = [
    "COMPRESSION_CEILING",
    "PROMPT_TOKENS",
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

# How much of the decoder's prompt window the prompt may fill. The window is
# 223 tokens (`max_length // 2 - 1` in the library), and every token of it
# costs: measured 2026-09-13 on the target CPU, about 0.65 s per hundred
# tokens of prompt (no prompt 2.25 s, 87 tokens 3.02 s, 198 tokens 3.55 s
# for the same two-second sentence). This buys about thirty names for the
# price the forty-name list of before paid, and the names come in the order
# they are worth (`AppCatalog.spoken_names`). Fitting it here rather than
# letting the library cut it keeps the pack's own words in front.
PROMPT_TOKENS = 120

# How many tokens the decoder may write for an utterance: this many at
# least, plus this many per second of audio. Turkish speech decodes at about
# five tokens a second (14 tokens in 2.8 s, measured 2026-09-13); twice that
# is the ceiling, so a real sentence is never cut and a loop is.
TOKENS_AT_LEAST = 24
TOKENS_PER_SECOND = 10

# A segment whose text compresses better than this is the decoder repeating
# itself, not speech. The library's own default for the same judgement.
COMPRESSION_CEILING = 2.4

# The placeholder in the pack's sentence where the names go.
APPS = "{apps}"

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

    # Phase 5 windows the audio and turns this on; until then the caller gets
    # `buffered_stream`, which is honest about producing nothing until the end.
    supports_streaming = False

    def __init__(
        self,
        *,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = DEFAULT_CPU_THREADS,
        vocabulary: Iterable[str] = (),
        prompt: str = "",
        build: ModelFactory | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._cpu_threads = cpu_threads
        # The names the decoder is told to expect, in the order they are
        # worth telling, and the pack's sentence to put them in. The prompt
        # itself is fitted when the model loads (`_fit_prompt`): `None` rather
        # than "" when there is nothing to say, the library's own way.
        self._names = [term.strip() for term in vocabulary if term.strip()]
        self._template = prompt.strip()
        self._prompt: str | None = None
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

    def transcribe_stream(
        self, pcm_chunks: AsyncIterator[Audio], *, hint: str | None = None
    ) -> AsyncIterator[Transcript]:
        # Whisper has no use for a fragment: it needs the whole utterance to
        # decide what the first word was.
        return buffered_stream(self, pcm_chunks, hint=hint)

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
                self._prompt = self._fit_prompt(self._model.hf_tokenizer)
            return self._model

    def _fit_prompt(self, tokenizer: Any) -> str | None:
        """The pack's sentence with as many names as `PROMPT_TOKENS` holds.

        Names are tried in the offered order and the count is the real
        tokenizer's, so the budget is a budget and not a guess. Without a
        sentence the names are a comma list, as before; a sentence without
        `{apps}` gets them after it; nothing at all is `None`.
        """
        template = self._template if APPS in self._template else f"{self._template} {APPS}".strip()

        def render(names: list[str]) -> str:
            return template.replace(APPS, ", ".join(names)).strip()

        def tokens(text: str) -> int:
            return len(tokenizer.encode(text, add_special_tokens=False).ids)

        kept: list[str] = []
        for name in self._names:
            if tokens(render([*kept, name])) > PROMPT_TOKENS:
                break
            kept.append(name)
        if not kept and not self._template:
            return None
        prompt = render(kept)
        logger.info("recogniser prompt: {} names, {} tokens", len(kept), tokens(prompt))
        return prompt

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
            is_final=True,
            confidence=_confidence(heard),
            language=info.language or "",
            no_speech_probability=_no_speech(heard, decoded, kept_seconds=info.duration_after_vad),
        )

    def _load_whisper(self) -> Model:
        # Imported here rather than at module scope: it pulls in CTranslate2
        # and its native libraries, and `live-assistant --help` has no use for them.
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
