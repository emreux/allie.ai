"""Google's recogniser behind the STTProvider protocol (design.md section 3.4, on trial).

ADR-001 kept local Whisper as the default and said a cloud recogniser would
be the user's choice, made in a file, announced by the wizard. This is that
recogniser, on trial since 2026-09-14 with the owner's own key: the same
protocol as `local_whisper.py`, so `app.py` does not know which of the two
it is talking to, and `[stt] provider = "gemini"` in `config.toml` is the
whole switch. `local` is the default and never leaves.

**It is the Live API, and an ASR endpoint rather than a prompt.** The model
is `gemini-3.5-transcribe-live`: a WebSocket session per utterance, the
audio pushed as 16-bit PCM in half-second chunks between an explicit
"speech starts" and "speech ends" signal, and the words back as
`input_transcription` messages - interim ones while it listens, one final
per sentence once it has been told the speech is over. The "ends" signal
is ours to send (the server's own detector is switched off) and is what
makes the final come: measured 2026-09-14, without it a sentence sometimes
never finalised. The session takes an `audio_transcription_config` - the
expected language (the protocol's `hint`; none given, the engine detects the
language itself, which is section 3.12's `auto` mode for free) and a custom
vocabulary, the app names `LocalWhisper` gets as a prompt, with no token
window to fit them into.

Why Live and not the batch `gemini-3.5-transcribe`: the free tier allows
that one **25 requests a day** and three a minute; the Live model is
unlimited (20k tokens a minute, about ten minutes of audio each minute).
Same accuracy on the fourteen synthetic sentences of ADR-001 (WER 14.7 %
against Whisper's 17.6 %; "Spotify'ı aç" right where Whisper was wrong),
1.6-2.1 s per sentence of which 0.6 s is opening the session, and the
interim words are the partial transcripts section 4 wants one day.

**Silence sends nothing back, noise sends an interim and no final.** So a
final is waited for as long as the audio warrants, and none is reported as
`no_speech_probability = 1.0` - the value `Transcript` reserves for "there
was nothing to decode" - which keeps a cough unanswered here as it is with
Whisper. Words come back with no opinion (`None`): this engine does not say
how sure it is.

**A network is a network.** A fallback - the local engine, loaded as it
always was - can stand behind this one: a refused key, a dropped socket, a
host that does not resolve or an answer that does not come in time is
logged in one line and the same audio goes to Whisper. Without a fallback
the answer is an empty transcript with no opinion, which `app.hear` reads
as "say it again" rather than as silence.

The key is the Gemini entry of the Credential Manager - the one the LLM
uses when the LLM is Gemini. The audio leaves the machine; the README says
so under the setting, and nowhere else does anything change.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Any

import numpy as np
from google import genai
from google.genai import errors, types
from loguru import logger

# The SDK's own transport for the Live API; its exceptions are what a
# dropped or refused socket looks like from here.
from websockets.exceptions import WebSocketException

from assistant.stt.base import SAMPLE_RATE, Audio, STTProvider, Transcript, buffered_stream

__all__ = [
    "CHUNK_SECONDS",
    "DEADLINE_AT_LEAST_SECONDS",
    "DEFAULT_MODEL",
    "FINAL_SECONDS_AT_LEAST",
    "FINAL_SECONDS_PER_SECOND",
    "NOTHING_TO_DECODE",
    "PCM_MIME",
    "QUIET_SECONDS",
    "TIMEOUT_SECONDS",
    "GeminiSTT",
    "to_pcm16",
]

DEFAULT_MODEL = "gemini-3.5-transcribe-live"

# Longer than a slow answer, shorter than the turn's patience: past this the
# fallback is asked, or the user is. Covers opening the session, sending
# the audio and waiting for the words.
TIMEOUT_SECONDS = 8.0

# The SDK sends its HTTP timeout to Google as the request's deadline, and
# Google refuses one under ten seconds ("400 Manually set deadline 8s is too
# short", 2026-09-14). So the client's deadline never goes below this; our
# own, shorter patience above is kept on this side with `asyncio.timeout`.
DEADLINE_AT_LEAST_SECONDS = 10.0

# How the audio is pushed: raw 16-bit PCM at the pipeline's rate, in chunks
# of this many seconds, all at once - the utterance is already over.
PCM_MIME = f"audio/pcm;rate={SAMPLE_RATE}"
CHUNK_SECONDS = 0.5

# How long a final transcript is waited for after "speech ends": the engine
# works through the audio at roughly half real time (measured: 3.9 s of
# audio finalised 1.6 s after being sent), so the wait grows with the
# audio. Silence and noise never produce a final and cost exactly this.
FINAL_SECONDS_AT_LEAST = 2.0
FINAL_SECONDS_PER_SECOND = 0.5

# Once a final has come, how long to listen for another sentence's before
# calling the utterance done.
QUIET_SECONDS = 0.4

# What no final at all means: the engine found nothing to decode. The value
# is `Transcript`'s own for it (`stt/base.py`).
NOTHING_TO_DECODE = 1.0

# How much of Google's message one warning line carries.
DESCRIBED_CHARS = 240


def to_pcm16(pcm: Audio) -> bytes:
    """One channel, 16 kHz, 16-bit little-endian: what the session is fed.

    Values outside [-1, 1] are clipped rather than wrapped - a wrapped
    sample is a click the recogniser hears as a consonant.
    """
    return (np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class GeminiSTT:
    """Google's live transcriber, with an optional engine behind it."""

    id = "gemini"
    supports_streaming = False

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        vocabulary: Iterable[str] = (),
        fallback: STTProvider | None = None,
        timeout_seconds: float = TIMEOUT_SECONDS,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._vocabulary = list(vocabulary)
        self._fallback = fallback
        self._timeout = timeout_seconds
        # No retry options: the SDK then makes one attempt, and a refusal
        # costs a third of a second rather than a minute of backing off.
        deadline = max(timeout_seconds, DEADLINE_AT_LEAST_SECONDS)
        self._client = (
            client
            if client is not None
            else genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=int(deadline * 1000)),
            )
        )

    async def load(self) -> None:
        """Loads the engine behind this one, if there is one; the client
        itself has nothing to load."""
        load: Callable[[], Awaitable[None]] | None = getattr(self._fallback, "load", None)
        if load is not None:
            await load()
        logger.info(
            "recogniser: {}, {} vocabulary, fallback {}",
            self._model,
            len(self._vocabulary),
            getattr(self._fallback, "id", "none"),
        )

    async def transcribe(self, pcm: Audio, *, hint: str | None = None) -> Transcript:
        try:
            async with asyncio.timeout(self._timeout):
                finals = await self._ask(pcm, hint)
        except (errors.APIError, WebSocketException, OSError, TimeoutError) as failure:
            logger.warning("recogniser {} failed: {}", self._model, self._describe(failure))
            if self._fallback is None:
                return Transcript(text="", language=hint or "")
            return await self._fallback.transcribe(pcm, hint=hint)
        return _transcript(finals, hint)

    def transcribe_stream(
        self, pcm_chunks: AsyncIterator[Audio], *, hint: str | None = None
    ) -> AsyncIterator[Transcript]:
        return buffered_stream(self, pcm_chunks, hint=hint)

    async def _ask(self, pcm: Audio, hint: str | None) -> list[Any]:
        """One session: the audio in, the final transcriptions out."""
        data = to_pcm16(pcm)
        step = int(CHUNK_SECONDS * SAMPLE_RATE) * 2
        patience = FINAL_SECONDS_AT_LEAST + FINAL_SECONDS_PER_SECOND * len(pcm) / SAMPLE_RATE
        finals: list[Any] = []
        async with self._client.aio.live.connect(model=self._model, config=self._config(hint)) as s:
            await s.send_realtime_input(activity_start=types.ActivityStart())
            for at in range(0, len(data), step):
                await s.send_realtime_input(
                    audio=types.Blob(data=data[at : at + step], mime_type=PCM_MIME)
                )
            await s.send_realtime_input(activity_end=types.ActivityEnd())

            messages = aiter(s.receive())
            while True:
                try:
                    message = await asyncio.wait_for(
                        anext(messages), QUIET_SECONDS if finals else patience
                    )
                except (TimeoutError, StopAsyncIteration):
                    # Nothing more is coming: the utterance is done, or it
                    # never held a sentence at all.
                    break
                content = message.server_content
                if content is None:
                    continue
                heard = content.input_transcription
                if heard is not None and heard.text:
                    finals.append(heard)
                if content.turn_complete:
                    break
        return finals

    def _config(self, hint: str | None) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            input_audio_transcription=types.AudioTranscriptionConfig(
                language_codes=[hint] if hint else None,
                custom_vocabulary=self._vocabulary or None,
            ),
            # The start and end of speech are ours to say - the microphone's
            # own detector already decided them - and the end is what makes
            # the final transcript come.
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
            ),
        )

    def _describe(self, failure: BaseException) -> str:
        if isinstance(failure, errors.APIError):
            # Google's quota message says which quota on its second line
            # ("limit: 25, model: ..."), which is the part worth reading.
            message = " ".join(line.strip() for line in (failure.message or "").splitlines())
            return f"{failure.code} {message[:DESCRIBED_CHARS]}".strip()
        if isinstance(failure, TimeoutError):
            return f"no answer within {self._timeout:.0f} s"
        return f"{type(failure).__name__}: {failure}"[:DESCRIBED_CHARS]


def _transcript(finals: list[Any], hint: str | None) -> Transcript:
    if not finals:
        return Transcript(text="", language=hint or "", no_speech_probability=NOTHING_TO_DECODE)
    text = " ".join(" ".join(str(final.text).split()) for final in finals if final.text)
    spoken = next((final.language_code for final in finals if final.language_code), None)
    # BCP-47 from the engine, ISO 639-1 in the protocol: `tr-TR` is `tr`.
    language = spoken.partition("-")[0] if spoken else (hint or "")
    return Transcript(text=text, language=language)
