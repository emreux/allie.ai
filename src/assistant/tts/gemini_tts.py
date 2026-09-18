"""Google's speech synthesiser behind the TTSProvider protocol (design.md
section 3.5, phase 3.4 as changed on 17 Sep 2026).

Phase 3.4 named Azure Neural as the voice that would replace Windows'. What
replaced it in the plan is this: the owner develops with one Google key
until release (section 12, decision 22), Gemini's TTS models speak Turkish
in a voice that is not a robot's, the free tier costs nothing, and the same
key already unlocks the recogniser (`stt/gemini_stt.py`). One key, three
uses. `[tts] provider = "gemini"` in `config.toml` is the whole switch;
`sapi` is the default and never leaves (the README's promise that nothing
leaves the machine unless a line in the file says so).

**One request per sentence, streamed.** `tts/base.sentences` cuts the
model's words at the first boundary, as it does for Windows; each sentence
is one `generate_content_stream` with `response_modalities=["AUDIO"]`, and
the PCM comes back in pieces that are handed on as they arrive, so the
first word of a long sentence is heard before its last is synthesised.
The format is what the protocol promises: 16-bit signed mono PCM at the
rate in `sample_rate`, twenty-four kilohertz here.

**Windows is behind it.** A refused key, the free tier's quota, a socket
that dropped, an answer that did not come in time: each is one line in
the log and the same sentence read by the local engine, in the local voice
that `choose_voice` picks for the pack once and keeps. The turn goes on;
the user hears a plainer voice for a sentence, not silence. Without a
fallback the sentence is skipped and the log says so.

**The voices belong to no language.** Google's thirty are named after
stars and moons and read whatever script they are given, so `list_voices`
answers every language with the same thirty, and the pack's `[tts.voice]
gemini` names the one it prefers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
from google import genai
from google.genai import errors, types
from loguru import logger

from assistant.tts.base import TTSProvider, VoiceInfo, choose_voice, sentences

__all__ = ["DEFAULT_MODEL", "PCM_MIME", "SAMPLE_RATE", "TIMEOUT_SECONDS", "VOICES", "GeminiTTS"]

DEFAULT_MODEL = "gemini-3.1-flash-tts-preview"

# What the model produces: raw 16-bit PCM, one channel, at this rate.
SAMPLE_RATE = 24_000
PCM_MIME = f"audio/L16;codec=pcm;rate={SAMPLE_RATE}"

# How long one sentence may take to start and finish arriving. Past this
# the local engine reads it; a sentence that takes longer than this to
# synthesise would have been heard later than the user's patience anyway.
TIMEOUT_SECONDS = 8.0

# Google refuses a request deadline under ten seconds (`stt/gemini_stt.py`).
DEADLINE_AT_LEAST_SECONDS = 10.0

# The prebuilt voices, as Google names them. Every one reads every language
# the model speaks; the difference is timbre, not tongue.
VOICES: tuple[str, ...] = (
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
)

# How much of Google's message one warning line carries.
DESCRIBED_CHARS = 240


class GeminiTTS:
    """Google's synthesiser, one sentence per request, with Windows behind it."""

    id = "gemini"
    sample_rate = SAMPLE_RATE

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        fallback: TTSProvider | None = None,
        fallback_language: str | None = None,
        fallback_preference: str | None = None,
        timeout_seconds: float = TIMEOUT_SECONDS,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._fallback = fallback
        # The local voice is picked the first time it is needed and kept:
        # listing Windows' voices is a COM call, and most days it is never
        # needed at all.
        self._fallback_language = fallback_language
        self._fallback_preference = fallback_preference
        self._fallback_voice: str | None = None
        self._timeout = timeout_seconds
        deadline = max(timeout_seconds, DEADLINE_AT_LEAST_SECONDS)
        self._client = (
            client
            if client is not None
            else genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(deadline * 1000)),
            )
        )

    async def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        """The thirty prebuilt voices, for whatever language was asked."""
        return [VoiceInfo(id=name, display_name=name, language=language or "") for name in VOICES]

    async def stream(self, chunks: AsyncIterator[str], *, voice: str) -> AsyncIterator[bytes]:
        async for sentence in sentences(chunks):
            try:
                async with asyncio.timeout(self._timeout):
                    async for buffer in self._speak(sentence, voice):
                        yield buffer
            except (errors.APIError, httpx.HTTPError, OSError, TimeoutError) as failure:
                logger.warning("voice {} failed: {}", self._model, self._describe(failure))
                async for buffer in self._read_locally(sentence):
                    yield buffer

    async def _speak(self, sentence: str, voice: str) -> AsyncIterator[bytes]:
        """One sentence, as the pieces of PCM come back."""
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        )
        stream = await self._client.aio.models.generate_content_stream(
            model=self._model, contents=sentence, config=config
        )
        async for chunk in stream:
            for data in _audio_of(chunk):
                if data:
                    yield data

    async def _read_locally(self, sentence: str) -> AsyncIterator[bytes]:
        """The sentence through the fallback, in its own voice."""
        if self._fallback is None:
            return
        voice = await self._local_voice()
        if not voice:
            logger.warning("no local voice to fall back to; the sentence was skipped")
            return
        async for buffer in self._fallback.stream(_one(sentence), voice=voice):
            yield buffer

    async def _local_voice(self) -> str:
        if self._fallback_voice is None and self._fallback is not None:
            chosen = ""
            for language in (self._fallback_language, None):
                voices = await self._fallback.list_voices(language)
                chosen = choose_voice(voices, self._fallback_preference)
                if chosen:
                    break
            self._fallback_voice = chosen
        return self._fallback_voice or ""

    def _describe(self, failure: BaseException) -> str:
        if isinstance(failure, errors.APIError):
            message = " ".join(line.strip() for line in (failure.message or "").splitlines())
            return f"{failure.code} {message[:DESCRIBED_CHARS]}".strip()
        if isinstance(failure, TimeoutError):
            return f"no audio within {self._timeout:.0f} s"
        return f"{type(failure).__name__}: {failure}"[:DESCRIBED_CHARS]


def _audio_of(chunk: Any) -> list[bytes]:
    """The PCM in one streamed response, in order; nothing for a chunk
    that carried only text or usage."""
    found: list[bytes] = []
    for candidate in chunk.candidates or []:
        content = candidate.content
        for part in (content.parts if content is not None else None) or []:
            blob = part.inline_data
            if blob is not None and isinstance(blob.data, bytes):
                found.append(blob.data)
    return found


async def _one(sentence: str) -> AsyncIterator[str]:
    yield sentence
