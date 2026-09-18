"""Google's synthesiser behind the TTSProvider protocol (design.md section
3.5, 17 Sep 2026): one request per sentence, the audio as it streams, and
Windows behind it for the sentence Google refuses.

Nothing here talks to Google: the client is a fake whose `generate_content_stream`
answers what each test scripted, in the SDK's own response types.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from google.genai import errors, types
from loguru import logger

from assistant.tts.base import TTSProvider, VoiceInfo
from assistant.tts.gemini_tts import DEFAULT_MODEL, SAMPLE_RATE, VOICES, GeminiTTS
from tests.test_tts import DAVID, TOLGA, fragments


def audio(data: bytes) -> types.GenerateContentResponse:
    """One streamed chunk carrying a piece of PCM."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                data=data, mime_type="audio/L16;codec=pcm;rate=24000"
                            )
                        )
                    ]
                )
            )
        ]
    )


def usage_only() -> types.GenerateContentResponse:
    """The last chunk of a stream: usage, no audio."""
    return types.GenerateContentResponse(
        usage_metadata=types.GenerateContentResponseUsageMetadata(total_token_count=3)
    )


class FakeModels:
    def __init__(self) -> None:
        # Per request, in order: the chunks to stream, or an exception to
        # raise before the first, or seconds to stall.
        self.answers: list[Any] = []
        self.requests: list[dict[str, Any]] = []

    async def generate_content_stream(self, **request: Any) -> AsyncIterator[Any]:
        self.requests.append(request)
        answer = self.answers.pop(0) if self.answers else [audio(b"pcm")]
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, float):
            await asyncio.sleep(answer)
            answer = [audio(b"late")]
        return _stream(answer)


async def _stream(chunks: list[Any]) -> AsyncIterator[Any]:
    for chunk in chunks:
        if isinstance(chunk, BaseException):
            raise chunk
        yield chunk


class FakeClient:
    def __init__(self) -> None:
        self.models = FakeModels()

    @property
    def aio(self) -> FakeClient:
        return self


class FakeWindows:
    """The local engine behind Google's: remembers what it read and in
    which voice."""

    id = "sapi"
    sample_rate = 16_000

    def __init__(self, voices: list[VoiceInfo] | None = None) -> None:
        self.voices = [TOLGA, DAVID] if voices is None else voices
        self.said: list[tuple[str, str]] = []
        self.listed: list[str | None] = []

    async def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        self.listed.append(language)
        if language is None:
            return list(self.voices)
        return [voice for voice in self.voices if voice.language == language]

    async def stream(self, chunks: AsyncIterator[str], *, voice: str) -> AsyncIterator[bytes]:
        async for sentence in chunks:
            self.said.append((sentence, voice))
            yield f"local:{sentence}".encode()


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def windows() -> FakeWindows:
    return FakeWindows()


@pytest.fixture
def voice(client: FakeClient, windows: FakeWindows) -> GeminiTTS:
    return GeminiTTS(
        "key",
        client=client,
        fallback=windows,
        fallback_language="tr",
        fallback_preference="Tolga",
        timeout_seconds=0.2,
    )


async def heard(voice: GeminiTTS, *pieces: str, chosen: str = "Kore") -> list[bytes]:
    return [buffer async for buffer in voice.stream(fragments(*pieces), voice=chosen)]


# --------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------


def test_it_is_a_provider_at_twenty_four_kilohertz(voice: GeminiTTS) -> None:
    assert isinstance(voice, TTSProvider)
    assert (voice.id, voice.sample_rate) == ("gemini", 24_000)
    assert SAMPLE_RATE == 24_000


async def test_the_thirty_voices_answer_every_language(voice: GeminiTTS) -> None:
    turkish = await voice.list_voices("tr")
    any_language = await voice.list_voices()

    assert [found.id for found in turkish] == list(VOICES) == [found.id for found in any_language]
    assert {found.language for found in turkish} == {"tr"}
    assert "Kore" in VOICES and "Charon" in VOICES and len(VOICES) == 30


# --------------------------------------------------------------------------
# Speaking
# --------------------------------------------------------------------------


async def test_each_sentence_is_one_request_in_the_chosen_voice(
    voice: GeminiTTS, client: FakeClient
) -> None:
    client.models.answers = [[audio(b"bir"), audio(b"iki"), usage_only()], [audio(b"uc")]]

    buffers = await heard(voice, "Bugün hava güzel ", "olacak. Yarın yağmur var.")

    assert buffers == [b"bir", b"iki", b"uc"]
    sentences = [request["contents"] for request in client.models.requests]
    assert sentences == ["Bugün hava güzel olacak.", "Yarın yağmur var."]
    request = client.models.requests[0]
    assert request["model"] == DEFAULT_MODEL
    config = request["config"]
    assert config.response_modalities == ["AUDIO"]
    assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Kore"


async def test_the_audio_is_handed_on_as_it_arrives(voice: GeminiTTS, client: FakeClient) -> None:
    """The first piece is yielded before the second is produced: the
    stream is not collected and then played."""
    client.models.answers = [[audio(b"first"), audio(b"second")]]
    stream = voice.stream(fragments("Uzun bir cümle geliyor."), voice="Kore")

    first = await anext(stream)

    assert first == b"first"
    assert [buffer async for buffer in stream] == [b"second"]


async def test_the_model_is_the_one_asked_for(client: FakeClient) -> None:
    voice = GeminiTTS("key", model="gemini-2.5-flash-preview-tts", client=client)

    await heard(voice, "Merhaba dünya, nasılsın?")

    assert client.models.requests[0]["model"] == "gemini-2.5-flash-preview-tts"


# --------------------------------------------------------------------------
# Windows behind it
# --------------------------------------------------------------------------


async def test_a_refused_request_is_read_by_windows_in_its_own_voice(
    voice: GeminiTTS, client: FakeClient, windows: FakeWindows
) -> None:
    client.models.answers = [
        errors.APIError(429, {"error": {"message": "quota exceeded\nlimit: 10"}}),
        [audio(b"ok")],
    ]
    warned: list[str] = []
    sink = logger.add(lambda message: warned.append(str(message)), level="WARNING")
    try:
        buffers = await heard(voice, "Birinci cümle burada. ", "İkinci cümle burada.")
    finally:
        logger.remove(sink)

    assert buffers == ["local:Birinci cümle burada.".encode(), b"ok"]
    assert windows.said == [("Birinci cümle burada.", TOLGA.id)]
    assert len(warned) == 1 and "429" in warned[0] and "quota exceeded limit: 10" in warned[0]


async def test_a_stream_that_dies_halfway_is_finished_by_windows(
    voice: GeminiTTS, client: FakeClient, windows: FakeWindows
) -> None:
    """Half a sentence from Google and the whole of it from Windows is a
    stutter; a whole sentence twice is worse. The local reading follows
    what was already heard, since it cannot be unheard."""
    client.models.answers = [[audio(b"half"), httpx.ReadError("gone")]]

    buffers = await heard(voice, "Cümle yarıda kalıyor.")

    assert buffers == [b"half", "local:Cümle yarıda kalıyor.".encode()]
    assert windows.said == [("Cümle yarıda kalıyor.", TOLGA.id)]


async def test_an_answer_that_does_not_come_in_time_goes_to_windows(
    voice: GeminiTTS, client: FakeClient, windows: FakeWindows
) -> None:
    client.models.answers = [5.0]

    buffers = await heard(voice, "Yavaş bir cevap bekleniyor.")

    assert buffers == ["local:Yavaş bir cevap bekleniyor.".encode()]


async def test_the_local_voice_is_picked_once_by_the_packs_preference(
    voice: GeminiTTS, client: FakeClient, windows: FakeWindows
) -> None:
    client.models.answers = [OSError("no network"), OSError("still none")]

    await heard(voice, "Bir cümle daha. ", "Ve bir tane daha.")

    assert windows.listed == ["tr"]
    assert {said_in for _, said_in in windows.said} == {TOLGA.id}


async def test_without_a_turkish_voice_any_local_voice_will_do(client: FakeClient) -> None:
    windows = FakeWindows(voices=[DAVID])
    voice = GeminiTTS("key", client=client, fallback=windows, fallback_language="tr")
    client.models.answers = [OSError("no network")]

    await heard(voice, "İngiliz sesiyle okunur.")

    assert windows.listed == ["tr", None]
    assert windows.said == [("İngiliz sesiyle okunur.", DAVID.id)]


async def test_without_a_fallback_the_sentence_is_skipped_and_the_turn_goes_on(
    client: FakeClient,
) -> None:
    voice = GeminiTTS("key", client=client)
    client.models.answers = [OSError("no network"), [audio(b"ok")]]

    buffers = await heard(voice, "Bu cümle düşer. ", "Bu cümle okunur.")

    assert buffers == [b"ok"]


async def test_a_fallback_with_no_voices_skips_the_sentence(client: FakeClient) -> None:
    windows = FakeWindows(voices=[])
    voice = GeminiTTS("key", client=client, fallback=windows)
    client.models.answers = [OSError("no network")]
    warned: list[str] = []
    sink = logger.add(lambda message: warned.append(str(message)), level="WARNING")
    try:
        buffers = await heard(voice, "Kimse okuyamıyor bunu.")
    finally:
        logger.remove(sink)

    assert buffers == []
    assert any("no local voice" in line for line in warned)


async def test_chunks_without_audio_are_passed_over(voice: GeminiTTS, client: FakeClient) -> None:
    client.models.answers = [[usage_only(), audio(b""), audio(b"x"), usage_only()]]

    assert await heard(voice, "Boş parçalar geçilir.") == [b"x"]
