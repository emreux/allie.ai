"""Speech to text: the protocol, and the local model behind it.

Two claims carry this module. The first is the 50 ms rule of section 3.1: the
model saturates four cores for seconds at a time, so if it ran on the event
loop the scheduler, the announce queue and audio capture would all stop with
it. `test_transcribing_leaves_the_event_loop_free` is the test that notices.

The second is that a provider which cannot stream still answers the streaming
call (section 3.4). The interface exists from day one so that phase 5's
partial transcripts are not a protocol change; until then the default body
collects the utterance and returns one final result.

Nothing here loads a real model - a `small` int8 load takes minutes on a cold
cache and seconds on a warm one. The model is injected, and the real thing is
proved by `scripts/smoke_stt.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import threading
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from assistant.stt.base import SAMPLE_RATE, Audio, STTProvider, Transcript, buffered_stream
from assistant.stt.local_whisper import (
    COMPRESSION_CEILING,
    PROMPT_TOKENS,
    TOKENS_AT_LEAST,
    TOKENS_PER_SECOND,
    LocalWhisper,
    ModelUnavailableError,
)


@dataclass
class FakeSegment:
    """What `faster_whisper` yields: text plus the numbers behind it."""

    text: str
    avg_logprob: float = -0.1
    tokens: list[int] = field(default_factory=lambda: [0, 1, 2])
    no_speech_prob: float = 0.05
    # gzip's ratio on the text: about 1.3 for a sentence, well over 2.4 for
    # "MAĞĞĞĞĞ..." - the library's own sign of a decoder going round in circles.
    compression_ratio: float = 1.3


class FakeEncoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class FakeTokenizer:
    """Counts words: what the real tokenizer does, coarsely, and enough to
    see the budget being kept."""

    def __init__(self) -> None:
        self.encoded: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> FakeEncoding:
        assert add_special_tokens is False
        self.encoded.append(text)
        return FakeEncoding(list(range(len(text.split()))))


@dataclass
class FakeInfo:
    language: str = "tr"
    language_probability: float = 0.99
    # How much audio the model's own VAD filter kept for decoding. Zero is a
    # recording it found no speech in at all.
    duration_after_vad: float = 0.0


class FakeModel:
    """Stands in for `WhisperModel`, including its laziness.

    The real `transcribe` returns a generator and does the work while it is
    consumed, so the delay lives in the generator too: a caller that returns
    the generator without draining it would look fast here and block the event
    loop in production.
    """

    def __init__(
        self,
        *,
        texts: tuple[str, ...] = (" Merhaba,", " nasilsin?"),
        segments: tuple[FakeSegment, ...] | None = None,
        language: str = "tr",
        avg_logprob: float = -0.1,
        delay: float = 0.0,
        kept_seconds: float | None = None,
    ) -> None:
        self.heard = (
            segments
            if segments is not None
            else tuple(FakeSegment(text=text, avg_logprob=avg_logprob) for text in texts)
        )
        self.language = language
        self.delay = delay
        # What the VAD filter kept. By default, some audio whenever there are
        # segments and none when there are not - the shape the real model
        # produces for speech and for silence respectively.
        self.kept_seconds = (1.0 if self.heard else 0.0) if kept_seconds is None else kept_seconds
        self.calls: list[tuple[Audio, dict[str, Any]]] = []
        self.hf_tokenizer = FakeTokenizer()

    def transcribe(self, audio: Audio, **options: Any) -> tuple[Iterator[FakeSegment], FakeInfo]:
        self.calls.append((audio, options))
        info = FakeInfo(language=self.language, duration_after_vad=self.kept_seconds)
        return self._segments(), info

    def _segments(self) -> Iterator[FakeSegment]:
        if self.delay:
            time.sleep(self.delay)
        yield from self.heard


def silence(seconds: float = 0.5) -> Audio:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


async def chunks_of(*buffers: Audio) -> AsyncIterator[Audio]:
    for buffer in buffers:
        yield buffer


def whisper(**kwargs: Any) -> tuple[LocalWhisper, FakeModel]:
    model = FakeModel(**kwargs)
    return LocalWhisper(build=lambda: model), model


# --------------------------------------------------------------------------
# The transcript
# --------------------------------------------------------------------------


async def test_the_transcript_is_the_segments_joined() -> None:
    stt, _ = whisper()

    transcript = await stt.transcribe(silence())

    assert transcript.text == "Merhaba, nasilsin?"
    assert transcript.is_final is True


async def test_the_language_is_the_one_the_model_reports() -> None:
    """Section 3.12: the conversation language is whatever was spoken, and the
    agent is told which that was."""
    stt, _ = whisper(language="en")

    assert (await stt.transcribe(silence())).language == "en"


async def test_the_hint_is_given_to_the_model_as_the_language() -> None:
    """Phase 1 runs in `fixed` mode: the locale's language goes in as a hint."""
    stt, model = whisper()

    await stt.transcribe(silence(), hint="tr")

    assert model.calls[0][1]["language"] == "tr"


async def test_without_a_hint_the_model_detects_the_language_itself() -> None:
    stt, model = whisper()

    await stt.transcribe(silence())

    assert model.calls[0][1]["language"] is None


async def test_the_audio_reaches_the_model_untouched() -> None:
    stt, model = whisper()
    spoken = silence(0.25)

    await stt.transcribe(spoken)

    heard, _ = model.calls[0]
    assert heard.dtype == np.float32
    assert len(heard) == len(spoken)


async def test_silence_is_an_empty_transcript_not_an_error() -> None:
    """The user pressed the key and said nothing; that is a turn to drop, not
    a crash to report."""
    stt, _ = whisper(texts=())

    transcript = await stt.transcribe(silence())

    assert transcript.text == ""
    assert transcript.confidence is None


# --------------------------------------------------------------------------
# Whether there was speech at all - the question the state machine decides on
# --------------------------------------------------------------------------


async def test_the_model_is_asked_to_skip_silence_before_decoding() -> None:
    """Silence decoded is a hallucination; silence skipped is nothing. Measured
    2026-09-05: with the filter on, a recording of a quiet room produced no
    segment at all instead of a subtitle credit."""
    stt, model = whisper()

    await stt.transcribe(silence())

    assert model.calls[0][1]["vad_filter"] is True


async def test_how_unlikely_speech_was_is_reported() -> None:
    stt, _ = whisper(segments=(FakeSegment(" Merhaba.", no_speech_prob=0.06),))

    transcript = await stt.transcribe(silence())

    assert transcript.no_speech_probability == pytest.approx(0.06)


async def test_a_segment_the_engine_itself_calls_silence_is_dropped() -> None:
    """A real sentence followed by a hallucinated credit over the trailing
    silence: the credit goes, the sentence stays, and the number reported is
    the sentence's."""
    stt, _ = whisper(
        segments=(
            FakeSegment(" Saat kaç?", no_speech_prob=0.05),
            FakeSegment(" Altyazı M.K.", no_speech_prob=0.91),
        )
    )

    transcript = await stt.transcribe(silence())

    assert transcript.text == "Saat kaç?"
    assert transcript.no_speech_probability == pytest.approx(0.05)


async def test_words_that_were_all_hallucination_report_the_engine_s_own_verdict() -> None:
    """Every segment over the ceiling: no text, and a number that says why -
    the lowest of them, which is still over the ceiling."""
    stt, _ = whisper(
        segments=(
            FakeSegment(" Altyazı", no_speech_prob=0.91),
            FakeSegment(" M.K.", no_speech_prob=0.86),
        )
    )

    transcript = await stt.transcribe(silence())

    assert transcript.text == ""
    assert transcript.no_speech_probability == pytest.approx(0.86)


async def test_nothing_left_to_decode_is_certainly_no_speech() -> None:
    """The filter removed everything. The engine found no speech, and says so
    with a number rather than with an absence the caller would read as 'no
    opinion'."""
    stt, _ = whisper(texts=(), kept_seconds=0.0)

    transcript = await stt.transcribe(silence())

    assert (transcript.text, transcript.no_speech_probability) == ("", 1.0)


async def test_speech_the_filter_kept_and_the_decoder_made_nothing_of_is_still_speech() -> None:
    """Loud speech in heavy noise: the filter keeps two seconds, the decoder
    produces no segment. Somebody spoke, and that is what the state machine
    needs to know - it is the one case worth telling the user about."""
    stt, _ = whisper(texts=(), kept_seconds=2.0)

    transcript = await stt.transcribe(silence())

    assert (transcript.text, transcript.no_speech_probability) == ("", 0.0)


async def test_the_confidence_is_the_average_token_probability() -> None:
    """Whisper reports a log probability per segment; a bare `avg_logprob` of
    -0.7 means nothing to the rest of the application."""
    stt, _ = whisper(avg_logprob=np.log(0.5))

    transcript = await stt.transcribe(silence())

    assert transcript.confidence is not None
    assert transcript.confidence == pytest.approx(0.5)


async def test_a_long_confident_segment_outweighs_a_short_unsure_one() -> None:
    """Nine sure tokens and one unsure one is a good transcript, not half a
    good one - so the segments are weighted by how much was said in them."""
    stt, _ = whisper(
        segments=(
            FakeSegment(text=" uzun ve net bir cumle", avg_logprob=math.log(0.9), tokens=[0] * 9),
            FakeSegment(text=" ha", avg_logprob=math.log(0.1), tokens=[0]),
        )
    )

    transcript = await stt.transcribe(silence())

    # Weighted by tokens: 0.722. An unweighted mean of the two segments would
    # be 0.300 - the one-token stumble would count as half the sentence.
    per_token = math.exp((9 * math.log(0.9) + math.log(0.1)) / 10)
    assert transcript.confidence == pytest.approx(per_token)
    assert transcript.confidence == pytest.approx(0.722, abs=0.001)


# --------------------------------------------------------------------------
# The 50 ms rule (section 3.1)
# --------------------------------------------------------------------------


async def test_transcribing_leaves_the_event_loop_free() -> None:
    """The one thing this module must not do.

    Transcription is seconds of four-core work. Run on the loop, it stops the
    scheduler, the announce queue and audio capture until it finishes.
    """
    stt, _ = whisper(delay=0.3)
    ticks = 0

    async def clock() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticking = asyncio.create_task(clock())
    await stt.transcribe(silence())
    ticking.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await ticking

    assert ticks >= 5, "the event loop was blocked while the model worked"


async def test_loading_the_model_leaves_the_event_loop_free() -> None:
    """A `small` int8 load reads about a gigabyte; that blocks too."""
    built_on: list[str] = []

    def build() -> FakeModel:
        built_on.append(threading.current_thread().name)
        return FakeModel()

    await LocalWhisper(build=build).load()

    assert built_on and built_on[0] != threading.main_thread().name


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


async def test_nothing_is_loaded_until_something_is_said() -> None:
    """Constructing the provider must stay cheap: `live-assistant --help` and the
    setup wizard both import this module and neither needs a model."""
    built = 0

    def build() -> FakeModel:
        nonlocal built
        built += 1
        return FakeModel()

    LocalWhisper(build=build)

    assert built == 0


async def test_the_model_is_loaded_once_however_often_it_is_used() -> None:
    built = 0

    def build() -> FakeModel:
        nonlocal built
        built += 1
        return FakeModel()

    stt = LocalWhisper(build=build)
    await stt.load()
    await stt.transcribe(silence())
    await stt.transcribe(silence())

    assert built == 1


async def test_two_turns_at_once_do_not_load_two_models() -> None:
    """Two loads at the same time would read two gigabytes into memory."""
    built = 0

    def build() -> FakeModel:
        nonlocal built
        built += 1
        time.sleep(0.05)  # long enough for the other thread to arrive
        return FakeModel()

    stt = LocalWhisper(build=build)
    await asyncio.gather(stt.transcribe(silence()), stt.transcribe(silence()))

    assert built == 1


async def test_a_model_that_cannot_be_loaded_fails_by_name() -> None:
    """No network on the first run, a broken cache: the user can fix these, so
    `live-assistant run` has to be able to say so instead of printing a traceback."""

    def broken() -> Any:
        raise OSError("connection refused while fetching the weights")

    stt = LocalWhisper(build=broken)

    with pytest.raises(ModelUnavailableError, match="connection refused"):
        await stt.load()


# --------------------------------------------------------------------------
# Streaming, for a provider that cannot stream
# --------------------------------------------------------------------------


async def test_a_provider_that_cannot_stream_still_answers_the_stream_call() -> None:
    stt, _ = whisper()

    results = [t async for t in stt.transcribe_stream(chunks_of(silence(), silence()))]

    assert len(results) == 1
    assert results[0].is_final is True


async def test_the_whole_utterance_reaches_the_model_in_one_piece() -> None:
    """Whisper has no use for a fragment; the pieces are joined in order."""
    stt, model = whisper()
    first, second = np.full(3, 0.1, dtype=np.float32), np.full(2, 0.2, dtype=np.float32)

    async for _ in stt.transcribe_stream(chunks_of(first, second)):
        pass

    heard, _ = model.calls[0]
    assert np.array_equal(heard, np.concatenate([first, second]))


async def test_a_stream_that_carried_no_audio_is_not_an_error() -> None:
    """The key was pressed and released before a single frame arrived."""
    stt, model = whisper(texts=())

    results = [t async for t in stt.transcribe_stream(chunks_of())]

    assert [t.text for t in results] == [""]
    assert len(model.calls[0][0]) == 0


async def test_the_hint_survives_the_stream() -> None:
    stt, model = whisper()

    async for _ in stt.transcribe_stream(chunks_of(silence()), hint="en"):
        pass

    assert model.calls[0][1]["language"] == "en"


async def test_the_default_body_is_shared_not_copied() -> None:
    """`buffered_stream` is what every non-streaming provider uses; writing it
    again per provider is how one of them ends up wrong."""

    class Deaf:
        id = "deaf"
        supports_streaming = False

        async def transcribe(self, pcm: Audio, *, hint: str | None = None) -> Transcript:
            return Transcript(text=f"{len(pcm)} samples", language=hint or "")

        def transcribe_stream(
            self, pcm_chunks: AsyncIterator[Audio], *, hint: str | None = None
        ) -> AsyncIterator[Transcript]:
            return buffered_stream(self, pcm_chunks, hint=hint)

    results = [t async for t in Deaf().transcribe_stream(chunks_of(silence(0.1)))]

    assert [t.text for t in results] == ["1600 samples"]


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------


def test_the_local_model_is_an_stt_provider() -> None:
    stt, _ = whisper()

    assert isinstance(stt, STTProvider)
    assert stt.supports_streaming is False


# --------------------------------------------------------------------------
# The prompt (2.2; a sentence, fitted by tokens, 2026-09-13)
# --------------------------------------------------------------------------


async def test_the_vocabulary_reaches_the_model_as_one_prompt() -> None:
    """Section 3.4's free trick: the names the decoder is told to expect,
    joined into the one string the library takes when the pack gives no
    sentence to put them in. Blank terms are dropped."""
    model = FakeModel()
    stt = LocalWhisper(build=lambda: model, vocabulary=["aç, ayarlar", "Spotify", "  ", "Chrome"])

    await stt.transcribe(silence())

    assert model.calls[0][1]["initial_prompt"] == "aç, ayarlar, Spotify, Chrome"


async def test_without_a_vocabulary_the_model_is_given_no_prompt() -> None:
    """`None`, not an empty string: the library's own way of saying so."""
    stt, model = whisper()

    await stt.transcribe(silence())

    assert model.calls[0][1]["initial_prompt"] is None


async def test_the_pack_s_sentence_carries_the_names_where_apps_is() -> None:
    """A sentence in the user's language reads to the decoder as speech in
    that language; a comma list does not (measured 2026-09-13, weakly)."""
    model = FakeModel()
    stt = LocalWhisper(
        build=lambda: model,
        vocabulary=["PyCharm", "Chrome"],
        prompt="Bilgisayarımdaki uygulamalar: {apps}. Uygulamayı aç.",
    )

    await stt.transcribe(silence())

    assert (
        model.calls[0][1]["initial_prompt"]
        == "Bilgisayarımdaki uygulamalar: PyCharm, Chrome. Uygulamayı aç."
    )


async def test_the_names_stop_where_the_window_ends_and_the_sentence_survives() -> None:
    """The library keeps the *last* 223 tokens of a long prompt, which would
    cut the sentence's own words off the front. So the fit is ours: names
    are added in the offered order while the whole stays under the budget."""
    model = FakeModel()
    names = [f"App{n}" for n in range(PROMPT_TOKENS * 2)]
    stt = LocalWhisper(build=lambda: model, vocabulary=names, prompt="Apps: {apps}. Open it.")

    await stt.transcribe(silence())

    prompt = model.calls[0][1]["initial_prompt"]
    assert prompt.startswith("Apps: App0, App1,")
    assert prompt.endswith(". Open it.")
    assert len(prompt.split()) <= PROMPT_TOKENS
    assert len(prompt.split()) >= PROMPT_TOKENS - 2
    assert "App199" not in prompt
    # Measured 2026-09-13 on the target CPU: about 0.65 s per hundred tokens
    # of prompt (no prompt 2.25 s, 87 tokens 3.02 s, 198 tokens 3.55 s).
    assert PROMPT_TOKENS == 120


async def test_the_prompt_is_fitted_once_when_the_model_loads() -> None:
    model = FakeModel()
    stt = LocalWhisper(build=lambda: model, vocabulary=["A", "B"], prompt="Apps: {apps}.")

    await stt.load()
    encoded = len(model.hf_tokenizer.encoded)
    await stt.transcribe(silence())
    await stt.transcribe(silence())

    assert encoded > 0
    assert len(model.hf_tokenizer.encoded) == encoded


async def test_a_sentence_without_the_placeholder_is_used_as_it_is() -> None:
    """A pack that forgot `{apps}` still gets its sentence; the names go
    after it, so neither is lost."""
    model = FakeModel()
    stt = LocalWhisper(build=lambda: model, vocabulary=["Chrome"], prompt="Open the app.")

    await stt.transcribe(silence())

    assert model.calls[0][1]["initial_prompt"] == "Open the app. Chrome"


# --------------------------------------------------------------------------
# One decode, so many tokens, and a loop is not words (2026-09-13)
# --------------------------------------------------------------------------


async def test_the_decoder_is_run_once_at_temperature_zero() -> None:
    """Measured 2026-09-13 (`small`, Tolga's voice): the fallback ladder
    decoded "PyCharm'ı aç" six times over 25-32 s and was still wrong; the
    single decode is the same answer in 3 s."""
    stt, model = whisper()

    await stt.transcribe(silence())

    assert model.calls[0][1]["temperature"] == 0.0


async def test_the_decoder_may_write_only_as_much_as_the_audio_could_hold() -> None:
    """A two-second command is not 448 tokens. `max_new_tokens` is what turns
    a decoder going round in circles (13.7 s) into a short bad guess (4.8 s)
    - which the next test then throws away."""
    stt, model = whisper()

    await stt.transcribe(silence(2.0))
    await stt.transcribe(silence(0.0))

    assert model.calls[0][1]["max_new_tokens"] == TOKENS_AT_LEAST + 2 * TOKENS_PER_SECOND
    assert model.calls[1][1]["max_new_tokens"] == TOKENS_AT_LEAST
    assert (TOKENS_AT_LEAST, TOKENS_PER_SECOND) == (24, 10)


async def test_a_segment_that_goes_round_in_circles_is_not_words() -> None:
    """ "TÜCHAR MAĞĞĞĞĞĞĞ..." has a compression ratio of four; it is dropped
    like a hallucination, and what is left is speech nobody could read -
    which `app.hear` answers with "say it again", not by asking the model."""
    stt, _ = whisper(
        segments=(
            FakeSegment(" TÜCHAR MAĞĞĞĞĞĞĞĞĞĞĞĞĞĞĞĞĞ", compression_ratio=4.1, no_speech_prob=0.02),
        )
    )

    transcript = await stt.transcribe(silence())

    assert transcript.text == ""
    assert transcript.no_speech_probability == pytest.approx(0.02)
    assert COMPRESSION_CEILING == 2.4


async def test_a_sentence_beside_a_loop_is_still_heard() -> None:
    stt, _ = whisper(
        segments=(
            FakeSegment(" Saat kaç?", compression_ratio=1.1),
            FakeSegment(" ĞĞĞĞĞĞĞĞĞĞĞĞĞĞ", compression_ratio=3.0),
        )
    )

    transcript = await stt.transcribe(silence())

    assert transcript.text == "Saat kaç?"
