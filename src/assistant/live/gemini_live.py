"""Google's Live API, translated into the protocol (plan.md section 4.3).

This is the only file in the tree that imports `google.genai` for a
conversation (`stt/gemini_stt.py` and `tts/gemini_tts.py` import it for the
yes/no window and the announcements, on trial since September). Everything
Gemini does differently is absorbed here: the session is configured once at
the open and never again, audio goes in as `realtime_input` and comes back
as inline blobs at 24 kHz, a tool result is a *function response* that has
to carry the function's name, and the SDK's `receive()` generator ends at
every `turn_complete` while the session does not.

That last one cost the spike an afternoon and is the reason `events()` is
a loop around `receive()`. Two more were measured the same day (ADR-001):
the server ends the model's turn *at* a tool call, before the result is
even sent, and speaks the answer as a new turn - so `TurnComplete` may
arrive twice for what the user experiences as one exchange, and the state
machine stitches them; and the socket closing is not the transport's own
exception up here but an `APIError` the SDK builds from the WebSocket
close code, so the code is what says whether the server hung up politely
(1000, 1001), the network went (1006) or the server refused something we
sent (1007, 1008).

The thinking is the model's own (no `thinking_config`): with a budget of
zero it skipped the tool call and invented the time (ADR-001), and the
`thinking_level` field is refused by this model. No `NON_BLOCKING` tools
either (plan.md D19: measured, no difference).

Since 2026-09-21 the session may carry Google's own search beside the
function declarations (D22): a tool of the server's, logged when it was
used and never answered from here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx
from google import genai
from google.genai import errors, types
from loguru import logger

# The SDK's own transport for the Live API; its exceptions are what a
# dropped or refused socket looks like from here.
from websockets.exceptions import WebSocketException

from assistant.live.base import (
    AudioChunk,
    AuthenticationError,
    Closed,
    GoingAway,
    InputText,
    Interrupted,
    LiveEvent,
    ModelInfo,
    OutputText,
    ProviderError,
    Resumable,
    SessionConfig,
    ToolCall,
    ToolCallCancelled,
    ToolCallEvent,
    ToolSpec,
    TurnComplete,
    Usage,
    UsageReport,
)

__all__ = ["DEFAULT_MODEL", "INPUT_MIME", "OUTPUT_RATE", "GeminiLive", "GeminiSession"]

DEFAULT_MODEL = "gemini-3.8-live"

# What a model has to support to hold a session (the batch models say
# `generateContent`; the transcribe, translate and music models say this
# too and are told apart by the probe, not by name).
_LIVE_ACTION = "bidiGenerateContent"

# What the microphone is sent as: raw 16-bit PCM at the pipeline's rate.
INPUT_MIME = "audio/pcm;rate=16000"

# What the model speaks at, measured 2026-09-18 from the first blob's mime
# type. The blob is believed over this constant.
OUTPUT_RATE = 24_000
_BYTES_PER_SAMPLE = 2

# What Gemini refuses a key with. 401 and 403 are the obvious two; the one
# that actually happens is a 400 whose message says so, which is why the
# message is read at all. Absorbing that here is the adapter earning its keep.
_KEY_REFUSED = frozenset({401, 403})
_KEY_REFUSED_IN_WORDS = "api key not valid"
_RATE_LIMITED = 429

# WebSocket close codes, as the SDK hands them up in an `APIError`: the
# server hung up the way servers do (we hung up, or it said `go_away` and
# went), the socket went, or the server refused something in the session.
_CLOSED_POLITELY = frozenset({1000, 1001})
_CLOSED_ABNORMALLY = 1006
_CLOSED_ON_US = frozenset({1007, 1008})


class GeminiLive:
    """Speaks to Google's Live API with an AI Studio key."""

    id = "gemini"

    # What the state machine may count on beyond the protocol: a handle
    # that continues the conversation in a later session (plan.md D5),
    # transcripts of both sides, the provider's own web search as a tool
    # of the session (D22), the tone of the voice answered in kind, and a
    # context the server keeps under its ceiling (D25).
    capabilities: frozenset[str] = frozenset(
        {"resumption", "transcripts", "web_search", "affective_dialog", "context_compression"}
    )

    def __init__(self, api_key: str, *, client: Any | None = None) -> None:
        self._client = client if client is not None else genai.Client(api_key=api_key)

    async def validate_credentials(self) -> bool:
        """Asks for the model list; a key that cannot list models cannot talk either.

        Only a refused key is `False`. A provider that could not be reached,
        or that refused the request for reasons of its own, raises instead:
        the answer to that is "check the connection and try again", not
        "paste another key", and the setup command says each in its own words.
        """
        try:
            await self.list_models()
        except AuthenticationError:
            return False
        return True

    async def list_models(self) -> list[ModelInfo]:
        try:
            pager = await self._client.aio.models.list()
        except errors.APIError as refusal:
            raise _refused(refusal) from refusal
        except httpx.HTTPError as failure:
            raise _unreachable(failure) from failure

        models: list[ModelInfo] = []

        async for model in pager:
            if _LIVE_ACTION not in (model.supported_actions or []):
                continue
            name = (model.name or "").removeprefix("models/")
            models.append(
                ModelInfo(
                    id=name,
                    display_name=model.display_name or name,
                    context_window=model.input_token_limit,
                    supports_tools=None,
                )
            )
        return models

    @asynccontextmanager
    async def connect(self, config: SessionConfig) -> AsyncIterator[GeminiSession]:
        """One session, open for the length of the block; the socket closes
        with it. Failing to open is a refusal in the protocol's own words.

        Only the opening is translated: an exception raised inside the block
        is the caller's own and passes through untouched.
        """
        async with AsyncExitStack() as stack:
            opening = self._client.aio.live.connect(
                model=config.model, config=_connect_config(config)
            )
            try:
                raw = await stack.enter_async_context(opening)
            except errors.APIError as refusal:
                raise _refused(refusal) from refusal
            except (WebSocketException, OSError) as failure:
                raise _unreachable(failure) from failure
            yield GeminiSession(raw, input_rate=config.input_rate)


class GeminiSession:
    """One open conversation, on the SDK's session object."""

    def __init__(self, raw: Any, *, input_rate: int) -> None:
        self._raw = raw
        self._input_mime = f"audio/pcm;rate={input_rate}"
        self._input_rate = input_rate
        # Milliseconds, as floats so that odd-sized chunks do not lose their
        # remainders; rounded when read.
        self._in_ms = 0.0
        self._out_ms = 0.0

    @property
    def audio_in_ms(self) -> int:
        return round(self._in_ms)

    @property
    def audio_out_ms(self) -> int:
        return round(self._out_ms)

    async def send_audio(self, pcm16: bytes) -> None:
        blob = types.Blob(data=pcm16, mime_type=self._input_mime)
        await self._send(self._raw.send_realtime_input(audio=blob))
        self._in_ms += _ms(len(pcm16), self._input_rate)

    async def send_text(self, text: str, *, role: str = "user", turn_complete: bool = True) -> None:
        turn = types.Content(role=role, parts=[types.Part(text=text)])
        await self._send(self._raw.send_client_content(turns=turn, turn_complete=turn_complete))

    async def send_tool_result(self, call: ToolCall, content: str) -> None:
        response = types.FunctionResponse(
            # Matched to its call by name: a response without one is refused
            # outright ("Name cannot be empty"). The id is optional to
            # Gemini, and one the model never issued is left out rather
            # than sent empty.
            id=call.id or None,
            name=call.name,
            # Gemini wants an object here; the protocol carries the result
            # as text, so it travels under a single key.
            response={"result": content},
        )
        await self._send(self._raw.send_tool_response(function_responses=[response]))

    async def interrupt(self) -> None:
        """Nothing to send: the server's own detector stopped the model, and
        `Interrupted` has already come."""

    async def events(self) -> AsyncIterator[LiveEvent]:
        # The SDK's `receive()` ends at each `turn_complete`; the session does
        # not, so it is asked again until the socket goes - which the SDK
        # reports as an `APIError` with the close code, from inside.
        try:
            while True:
                async for message in self._raw.receive():
                    for event in self._translate(message):
                        yield event
        except errors.APIError as failure:
            yield _closed(failure)
        except (WebSocketException, OSError) as failure:
            yield Closed(error=_unreachable(failure))

    async def _send(self, sending: Any) -> None:
        try:
            await sending
        except errors.APIError as refusal:
            raise _refused(refusal) from refusal
        except (WebSocketException, OSError) as failure:
            raise _unreachable(failure) from failure

    def _translate(self, message: types.LiveServerMessage) -> list[LiveEvent]:
        """Turns one server message into the events it carries, in the order
        the state machine wants them: what was heard and said, then the
        voice, then the interruption, then the end of the turn."""
        events: list[LiveEvent] = []

        update = message.session_resumption_update
        if update is not None and update.new_handle and update.resumable is not False:
            events.append(Resumable(update.new_handle))
        if message.go_away is not None:
            events.append(GoingAway(str(message.go_away.time_left or "")))
        if message.usage_metadata is not None:
            events.append(UsageReport(_usage(message.usage_metadata)))
        cancellation = message.tool_call_cancellation
        if cancellation is not None:
            events.append(ToolCallCancelled(tuple(cancellation.ids or ())))
        if message.tool_call is not None:
            for call in message.tool_call.function_calls or ():
                events.append(
                    ToolCallEvent(
                        ToolCall(id=call.id or "", name=call.name or "", arguments=call.args or {})
                    )
                )

        content = message.server_content
        if content is None:
            return events
        if content.grounding_metadata is not None:
            _log_grounding(content.grounding_metadata)
        heard = content.input_transcription
        if heard is not None and heard.text:
            events.append(InputText(heard.text))
        said = content.output_transcription
        if said is not None and said.text:
            events.append(OutputText(said.text))
        if content.model_turn is not None:
            for part in content.model_turn.parts or ():
                blob = part.inline_data
                if blob is None or not blob.data:
                    # A text part beside the audio, or a thought: not spoken.
                    continue
                rate = _rate_of(blob.mime_type or "")
                self._out_ms += _ms(len(blob.data), rate)
                events.append(AudioChunk(pcm16=blob.data, sample_rate=rate))
        if content.interrupted:
            events.append(Interrupted())
        if content.turn_complete:
            events.append(TurnComplete())
        return events


def _connect_config(config: SessionConfig) -> types.LiveConnectConfig:
    """The session's configuration in Gemini's envelope. Only what was asked
    for is sent: a field left `None` is the server's own default, which is
    what plan.md D19 chose for the voice, the thinking and the detector."""
    speech = None
    if config.voice or config.language_code:
        speech = types.SpeechConfig(
            voice_config=(
                types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=config.voice)
                )
                if config.voice
                else None
            ),
            language_code=config.language_code or None,
        )

    detection = None
    if config.end_sensitivity or config.silence_ms:
        detection = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                end_of_speech_sensitivity=(
                    types.EndSensitivity[f"END_SENSITIVITY_{config.end_sensitivity.upper()}"]
                    if config.end_sensitivity
                    else None
                ),
                silence_duration_ms=config.silence_ms or None,
            )
        )

    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=config.system_prompt or None,
        tools=_tools(config),
        speech_config=speech,
        input_audio_transcription=(
            types.AudioTranscriptionConfig(
                language_codes=[config.language_code] if config.language_code else None
            )
            if config.transcripts
            else None
        ),
        output_audio_transcription=types.AudioTranscriptionConfig() if config.transcripts else None,
        realtime_input_config=detection,
        # Asked for on every open so that the handles come (`Resumable`);
        # with a handle, the conversation continues where it left off.
        session_resumption=types.SessionResumptionConfig(handle=config.resume_handle),
        # The two switches of D25, sent only when asked for; `None` is the
        # server's own default, as with everything above. Proactive audio is
        # not sent at all: on this model it is on by the server's own rule.
        enable_affective_dialog=True if config.affective_dialog else None,
        context_window_compression=(
            types.ContextWindowCompressionConfig(sliding_window=types.SlidingWindow())
            if config.compress_context
            else None
        ),
    )


def _tools(config: SessionConfig) -> list[types.ToolUnion] | None:
    """What the session may call: Google's own search first, when asked for
    (D22) - it is a tool of the server's, declared by name and never
    answered from here - then the functions of the registry."""
    offered: list[types.ToolUnion] = []
    if config.web_search:
        offered.append(types.Tool(google_search=types.GoogleSearch()))
    if config.tools:
        offered.append(types.Tool(function_declarations=[_declare(tool) for tool in config.tools]))
    return offered or None


def _declare(tool: ToolSpec) -> types.FunctionDeclaration:
    schema = dict(tool.parameters)
    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description,
        # A schema with no properties is refused by the Live API (measured
        # in the spike): a tool without parameters is declared bare.
        parameters_json_schema=schema if schema.get("properties") else None,
    )


def _usage(metadata: types.UsageMetadata) -> Usage:
    return Usage(
        input_tokens=metadata.prompt_token_count or 0,
        output_tokens=metadata.response_token_count or 0,
        cached_tokens=metadata.cached_content_token_count or 0,
    )


def _log_grounding(grounding: types.GroundingMetadata) -> None:
    """One line per grounded answer: what was searched and which pages it
    stood on. The only trace of a search - nothing reaches the screen - and
    the one that says afterwards where an answer came from."""
    queries = list(grounding.web_search_queries or ())
    sources = [
        chunk.web.title
        for chunk in grounding.grounding_chunks or ()
        if chunk.web is not None and chunk.web.title
    ]
    if queries or sources:
        logger.info(
            "grounded: searched {queries}; read {sources}", queries=queries, sources=sources
        )


def _ms(byte_count: int, rate: int) -> float:
    return byte_count * 1000.0 / (rate * _BYTES_PER_SAMPLE)


def _rate_of(mime: str) -> int:
    """`audio/pcm;rate=24000` → 24000; anything else is what was measured."""
    for parameter in mime.split(";")[1:]:
        key, _, value = parameter.strip().partition("=")
        if key == "rate" and value.isdigit():
            return int(value)
    return OUTPUT_RATE


def _closed(failure: errors.APIError) -> Closed:
    """What the socket closing means, read off the close code the SDK put in
    the error: polite is no error at all; the rest is a refusal in the
    protocol's own words."""
    if failure.code in _CLOSED_POLITELY:
        return Closed()
    if failure.code == _CLOSED_ABNORMALLY:
        return Closed(error=_unreachable(failure))
    return Closed(error=_refused(failure))


def _unreachable(failure: BaseException) -> ProviderError:
    """The transport failed before Gemini could refuse anything.

    The socket refused, a name that did not resolve, a stream that stopped:
    the SDK's own exceptions for those derive from neither `ProviderError`
    nor `OSError`, so nothing above this layer would catch them. The class
    name is kept in the message because it is the only part that says what
    kind of failure it was.
    """
    return ProviderError(
        f"gemini could not be reached ({type(failure).__name__}): {_said(failure)}",
        kind="unreachable",
    )


def _refused(error: errors.APIError) -> ProviderError:
    """Turns one of Gemini's refusals into one the application knows.

    Only the distinction survives - which sentence the user hears and what
    the state machine does next - plus the provider's own words, which are
    the only thing that makes a report of this diagnosable afterwards.
    """
    said = _said(error)
    where = f"gemini refused the request ({error.code}): {said}"

    if error.code in _KEY_REFUSED or _KEY_REFUSED_IN_WORDS in said.casefold():
        return AuthenticationError(where)
    if error.code == _RATE_LIMITED:
        return ProviderError(where, kind="rate_limit")
    return ProviderError(where)


def _said(failure: BaseException) -> str:
    """The provider's words, on one line: Google's quota message says which
    quota on its second line, which is the part worth reading."""
    message = getattr(failure, "message", None) or str(failure)
    return " ".join(line.strip() for line in str(message).splitlines())
