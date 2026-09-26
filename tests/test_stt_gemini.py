"""Google's recogniser: the one that hears the yes or no of the gate's window (D36).

Nothing here talks to Google: the client is a fake whose live session
records what was sent and answers with the SDK's own message types, so what
is proved is the shape of the session - speech starts, the PCM in chunks,
speech ends, the ASR config, the server's detector off - and what is made of
what comes back, including nothing at all. The real thing is measured by
`scripts/bench_stt.py --provider gemini`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest
from google.genai import errors, types
from loguru import logger
from websockets.exceptions import ConnectionClosedError

from allie.app import Heard, hear
from allie.stt import gemini_stt
from allie.stt.base import SAMPLE_RATE, Audio, STTProvider, Transcript
from allie.stt.gemini_stt import (
    CHUNK_SECONDS,
    DEFAULT_MODEL,
    FINAL_SECONDS_AT_LEAST,
    FINAL_SECONDS_PER_SECOND,
    PCM_MIME,
    GeminiSTT,
    to_pcm16,
)


class FakeSession:
    """What `connect()` yields: what was sent, and the messages lined up."""

    def __init__(self, answers: list[Any], *, gaps: float = 0.0) -> None:
        self.sent: list[dict[str, Any]] = []
        self.answers = answers
        # Seconds between one message and the next, so the tests can see
        # the quiet-wait and the patience without a network.
        self.gaps = gaps

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.sent.append(kwargs)

    async def receive(self) -> AsyncIterator[Any]:
        for answer in self.answers:
            await asyncio.sleep(self.gaps)
            if isinstance(answer, BaseException):
                raise answer
            yield answer


class FakeLive:
    def __init__(self) -> None:
        self.opened: list[dict[str, Any]] = []
        self.answers: list[Any] = []
        self.gaps = 0.0
        self.refusal: BaseException | None = None
        self.sessions: list[FakeSession] = []

    @asynccontextmanager
    async def connect(self, *, model: str, config: Any) -> AsyncIterator[FakeSession]:
        self.opened.append({"model": model, "config": config})
        if self.refusal is not None:
            raise self.refusal
        session = FakeSession(list(self.answers), gaps=self.gaps)
        self.sessions.append(session)
        yield session


class FakeClient:
    def __init__(self) -> None:
        self.live = FakeLive()

    @property
    def aio(self) -> FakeClient:
        return self


def final(text: str, *, language: str | None = None) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            input_transcription=types.Transcription(text=text, language_code=language)
        )
    )


def interim(text: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(
            interim_input_transcription=types.Transcription(text=text)
        )
    )


TURN_COMPLETE = types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))
SETUP_ONLY = types.LiveServerMessage(setup_complete=types.LiveServerSetupComplete())
QUOTA = errors.APIError(
    429,
    {
        "error": {
            "code": 429,
            "message": "You exceeded your current quota.\n* Quota exceeded for metric: "
            "generate_content_free_tier_requests, limit: 25, model: gemini-3.5-transcribe\n"
            "Please retry in 51s.",
            "status": "RESOURCE_EXHAUSTED",
        }
    },
)


def gemini(**kwargs: Any) -> tuple[GeminiSTT, FakeClient]:
    client = FakeClient()
    return GeminiSTT("AIza-not-a-key", client=client, **kwargs), client


def tone(seconds: float = 1.0) -> Audio:
    """A sine, so that the PCM can be checked sample for sample."""
    samples = np.arange(int(SAMPLE_RATE * seconds), dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * 440 * samples / SAMPLE_RATE)).astype(np.float32)


def session_of(client: FakeClient) -> FakeSession:
    assert len(client.live.sessions) == 1
    return client.live.sessions[0]


# --------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------


async def test_the_audio_goes_as_pcm_between_speech_starts_and_speech_ends() -> None:
    stt, client = gemini()
    client.live.answers = [final("x")]
    pcm = tone(1.2)

    await stt.transcribe(pcm, hint="tr")

    sent = session_of(client).sent
    assert "activity_start" in sent[0] and "activity_end" in sent[-1]
    chunks = [call["audio"] for call in sent[1:-1]]
    assert all(chunk.mime_type == PCM_MIME == "audio/pcm;rate=16000" for chunk in chunks)
    # Half-second chunks: 1.2 s is two full ones and a 0.2 s remainder.
    assert [len(chunk.data) for chunk in chunks] == [16_000, 16_000, 6_400]
    assert b"".join(chunk.data for chunk in chunks) == to_pcm16(pcm)
    assert np.array_equal(np.frombuffer(to_pcm16(pcm), dtype="<i2"), (pcm * 32767.0).astype("<i2"))


def test_to_pcm16_clips_what_is_out_of_range() -> None:
    samples = np.frombuffer(to_pcm16(np.array([2.0, -2.0], dtype=np.float32)), dtype="<i2")
    assert samples.tolist() == [32767, -32767]


async def test_the_hint_is_the_language_code_and_there_is_no_vocabulary() -> None:
    """The API takes a custom vocabulary; this product has nothing to put in
    it since 2026-09-22. What reaches this recogniser is the yes or no of a
    confirmation window (D3, D10), whose words are the pack's."""
    stt, client = gemini()
    client.live.answers = [final("x")]

    await stt.transcribe(tone(), hint="tr")

    (opened,) = client.live.opened
    asr = opened["config"].input_audio_transcription
    assert asr.language_codes == ["tr"]
    assert asr.custom_vocabulary is None
    # The end of speech is ours to say, so the server's detector is off.
    assert opened["config"].realtime_input_config.automatic_activity_detection.disabled is True


async def test_without_a_hint_the_engine_detects_the_language_itself() -> None:
    stt, client = gemini()
    client.live.answers = [final("x")]

    await stt.transcribe(tone())

    asr = client.live.opened[0]["config"].input_audio_transcription
    assert asr.language_codes is None
    assert asr.custom_vocabulary is None


async def test_the_model_is_the_one_asked_for() -> None:
    stt, client = gemini()
    client.live.answers = [final("x")]
    await stt.transcribe(tone())
    assert client.live.opened[0]["model"] == DEFAULT_MODEL == "gemini-3.5-transcribe-live"

    stt, client = gemini(model="gemini-x")
    client.live.answers = [final("x")]
    await stt.transcribe(tone())
    assert client.live.opened[0]["model"] == "gemini-x"


async def test_one_session_per_utterance() -> None:
    stt, client = gemini()
    client.live.answers = [final("x")]

    await stt.transcribe(tone())
    await stt.transcribe(tone())

    assert len(client.live.opened) == 2


# --------------------------------------------------------------------------
# The answer
# --------------------------------------------------------------------------


async def test_the_transcript_is_the_finals_joined_and_the_interims_ignored() -> None:
    stt, client = gemini()
    client.live.answers = [
        SETUP_ONLY,
        interim("Saat"),
        interim("Saat kaç"),
        final("Saat  kaç? "),
        final("Söyler misin?"),
    ]

    transcript = await stt.transcribe(tone(), hint="tr")

    assert transcript == Transcript(text="Saat kaç? Söyler misin?", language="tr")
    assert transcript.no_speech_probability is None
    assert transcript.confidence is None


async def test_turn_complete_ends_the_listening_at_once() -> None:
    stt, client = gemini()
    client.live.answers = [final("x"), TURN_COMPLETE, final("never read")]

    transcript = await stt.transcribe(tone())

    assert transcript.text == "x"


async def test_the_language_is_the_engine_s_when_it_says_and_the_hint_otherwise() -> None:
    stt, client = gemini()

    client.live.answers = [final("x", language="tr-TR")]
    assert (await stt.transcribe(tone(), hint="en")).language == "tr"
    client.live.answers = [final("x")]
    assert (await stt.transcribe(tone(), hint="tr")).language == "tr"
    assert (await stt.transcribe(tone())).language == ""


async def test_no_final_is_certainly_no_speech() -> None:
    """Silence sends nothing back and noise only an interim (measured): the
    value `Transcript` keeps for "there was nothing to decode", and `hear()`
    stays quiet on it - a cough is not answered with "say it again" (spec A4)."""
    stt, client = gemini()
    client.live.answers = [interim("Run")]

    transcript = await stt.transcribe(tone(0.2), hint="tr")

    assert transcript == Transcript(text="", language="tr", no_speech_probability=1.0)
    assert hear(transcript) == Heard()


async def test_the_final_is_waited_for_as_long_as_the_audio_warrants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine works through the audio at about half real time, so the
    patience grows with it; a message that comes inside it is kept."""
    monkeypatch.setattr(gemini_stt, "FINAL_SECONDS_AT_LEAST", 0.1)
    monkeypatch.setattr(gemini_stt, "FINAL_SECONDS_PER_SECOND", 0.1)
    stt, client = gemini()
    client.live.answers = [final("late but inside")]
    client.live.gaps = 0.15  # inside 0.1 + 0.1 * 1 s, outside 0.1 alone

    assert (await stt.transcribe(tone(1.0))).text == "late but inside"
    assert (await stt.transcribe(tone(0.2))).no_speech_probability == 1.0
    assert FINAL_SECONDS_AT_LEAST == 2.0 and FINAL_SECONDS_PER_SECOND == 0.5


async def test_after_a_final_a_short_quiet_ends_the_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gemini_stt, "QUIET_SECONDS", 0.05)
    stt, client = gemini()
    client.live.answers = [final("first"), final("too late")]
    client.live.gaps = 0.1

    started = time.perf_counter()
    transcript = await stt.transcribe(tone(0.2))

    # The first message came after one gap; a second gap is past the quiet.
    assert transcript.text == "first"
    assert time.perf_counter() - started < 0.5


def test_the_client_s_deadline_is_never_under_google_s_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK sends its timeout to Google as the request's deadline, and
    Google refuses one under ten seconds (`400 Manually set deadline 8s is
    too short`, 2026-09-14). Our own patience is shorter and is kept here,
    not there."""
    built: list[dict[str, Any]] = []

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

    monkeypatch.setattr(gemini_stt.genai, "Client", Client)

    GeminiSTT("AIza-not-a-key", timeout_seconds=8.0)
    GeminiSTT("AIza-not-a-key", timeout_seconds=30.0)

    assert [kwargs["api_key"] for kwargs in built] == ["AIza-not-a-key"] * 2
    assert [kwargs["http_options"].timeout for kwargs in built] == [10_000, 30_000]
    assert all(kwargs["http_options"].retry_options is None for kwargs in built)


def test_the_provider_satisfies_the_protocol() -> None:
    stt, _ = gemini()
    assert isinstance(stt, STTProvider)
    assert stt.id == "gemini"
    assert CHUNK_SECONDS == 0.5


# --------------------------------------------------------------------------
# When Google could not be asked
# --------------------------------------------------------------------------


def missed(transcript: Transcript) -> bool:
    """Empty with no opinion: `hear()` asks for a repeat rather than staying
    silent - the words were there, the engine was not."""
    return transcript == Transcript(text="", language="tr") and hear(transcript) == Heard(
        missed=True
    )


async def test_a_refused_session_is_one_line_in_the_log_and_a_missed_answer() -> None:
    """A bad key closes the socket with an `APIError` (measured); a quota
    would too. Nothing stands behind Google since D36: the window asks again."""
    stt, client = gemini()
    client.live.refusal = QUOTA
    lines: list[str] = []
    handle = logger.add(lines.append, level="WARNING", format="{message}")

    try:
        transcript = await stt.transcribe(tone(), hint="tr")
    finally:
        logger.remove(handle)

    assert missed(transcript)
    assert [line.strip() for line in lines] == [
        "recogniser gemini-3.5-transcribe-live failed: 429 You exceeded your current quota. "
        "* Quota exceeded for metric: generate_content_free_tier_requests, limit: 25, "
        "model: gemini-3.5-transcribe Please retry in 51s."
    ]


async def test_a_dropped_socket_is_a_missed_answer() -> None:
    stt, client = gemini()
    client.live.answers = [ConnectionClosedError(None, None)]

    assert missed(await stt.transcribe(tone(), hint="tr"))


async def test_a_host_that_does_not_resolve_is_a_missed_answer() -> None:
    stt, client = gemini()
    client.live.refusal = OSError("[Errno 11001] getaddrinfo failed")

    assert missed(await stt.transcribe(tone(), hint="tr"))


async def test_an_answer_that_does_not_come_in_time_is_a_missed_answer() -> None:
    stt, client = gemini(timeout_seconds=0.05)
    client.live.answers = [final("too late")]
    client.live.gaps = 0.5

    started = time.perf_counter()
    transcript = await stt.transcribe(tone(), hint="tr")

    assert missed(transcript)
    assert time.perf_counter() - started < 0.4


async def test_a_bug_is_not_swallowed() -> None:
    stt, client = gemini()
    client.live.answers = [RuntimeError("bug")]

    with pytest.raises(RuntimeError, match="bug"):
        await stt.transcribe(tone())


def test_nothing_stands_behind_google_any_more() -> None:
    """D36 (2026-09-26): the local recogniser is gone, and with it the
    engine this one used to fall back on."""
    with pytest.raises(TypeError):
        GeminiSTT("AIza-not-a-key", client=FakeClient(), fallback=object())  # type: ignore[call-arg]
    assert importlib.util.find_spec("allie.stt.local_whisper") is None


# --------------------------------------------------------------------------
# The stream
# --------------------------------------------------------------------------


async def test_the_whole_utterance_goes_over_as_one_transcript() -> None:
    stt, client = gemini()
    client.live.answers = [final("Saat kaç?")]
    pcm = tone()

    transcript = await stt.transcribe(pcm, hint="tr")

    assert transcript == Transcript(text="Saat kaç?", language="tr")
    sent = session_of(client).sent
    assert b"".join(call["audio"].data for call in sent[1:-1]) == to_pcm16(pcm)
