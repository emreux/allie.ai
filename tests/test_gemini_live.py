"""What the Gemini Live adapter must turn its vendor's shapes into.

No network. The fake client mimics the one method the adapter calls
(`aio.live.connect`) and the session it yields, but the messages that
session answers with are real `google.genai.types` objects, so a field this
suite reads is a field the SDK actually has. A hand-rolled stub would let
the tests pass while the adapter reads an attribute that does not exist.
The fake's `receive()` also ends where the SDK's does - at every
`turn_complete` - and a closed socket is the `APIError` the SDK raises for
one, with the WebSocket close code in it (read in `live.py`, 2026-09-18).

What is *not* here is anything every adapter has to do. Audio arriving with
its rate, a tool call arriving whole, the minutes counted, a refusal
reaching the caller as one of the two exceptions the application knows -
all of those are the contract, and they are tested once for every adapter
in `test_live_adapters.py`. This file is only what Gemini does differently,
plus the `build` function at the bottom that lets the contract suite drive
it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
from google import genai
from google.genai import errors, types
from websockets.exceptions import ConnectionClosedError

from assistant.live.base import (
    AudioChunk,
    AuthenticationError,
    Closed,
    GoingAway,
    InputText,
    LiveEvent,
    LiveProvider,
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
from assistant.live.gemini_live import DEFAULT_MODEL, INPUT_MIME, OUTPUT_RATE, GeminiLive
from tests.live_contract import (
    COMPLAINT,
    Adapter,
    Calls,
    Cancels,
    Hears,
    Interrupts,
    Leaves,
    Refuses,
    Resumes,
    Says,
    Speaks,
    Spends,
    Step,
    Stops,
)

CLOCK = ToolSpec(
    name="get_current_time",
    description="Returns the current local time.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)
BARE = ToolSpec(name="ping", description="Says pong.", parameters={"type": "object"})


# --------------------------------------------------------------------------
# The server's messages, in the SDK's own types
# --------------------------------------------------------------------------


def content(**fields: Any) -> types.LiveServerMessage:
    return types.LiveServerMessage(server_content=types.LiveServerContent(**fields))


def speaks(data: bytes, *, rate: int = OUTPUT_RATE) -> types.LiveServerMessage:
    blob = types.Blob(data=data, mime_type=f"audio/pcm;rate={rate}")
    return content(model_turn=types.Content(role="model", parts=[types.Part(inline_data=blob)]))


def says(text: str) -> types.LiveServerMessage:
    return content(output_transcription=types.Transcription(text=text))


def hears(text: str) -> types.LiveServerMessage:
    return content(input_transcription=types.Transcription(text=text))


def calls(*function_calls: tuple[str | None, str, dict[str, Any]]) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        tool_call=types.LiveServerToolCall(
            function_calls=[
                types.FunctionCall(id=call_id, name=name, args=args)
                for call_id, name, args in function_calls
            ]
        )
    )


def cancels(*ids: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        tool_call_cancellation=types.LiveServerToolCallCancellation(ids=list(ids))
    )


def spends(prompt: int, response: int, cached: int = 0) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        usage_metadata=types.UsageMetadata(
            prompt_token_count=prompt,
            response_token_count=response,
            cached_content_token_count=cached,
            total_token_count=prompt + response,
        )
    )


def resumes(handle: str | None, *, resumable: bool | None = True) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        session_resumption_update=types.LiveServerSessionResumptionUpdate(
            new_handle=handle, resumable=resumable
        )
    )


def leaves(time_left: str) -> types.LiveServerMessage:
    return types.LiveServerMessage(go_away=types.LiveServerGoAway(time_left=time_left))


INTERRUPTED = content(interrupted=True)
TURN_COMPLETE = content(turn_complete=True)
SETUP_ONLY = types.LiveServerMessage(setup_complete=types.LiveServerSetupComplete())
BARE_MESSAGE = types.LiveServerMessage()


def closed(code: int, reason: str = "") -> errors.APIError:
    """What the SDK raises from `receive()` when the socket has closed: an
    `APIError` carrying the WebSocket close code and reason (`live.py`,
    `_receive`, 2026-09-18) - not the transport's own exception."""
    return errors.APIError(code, reason)


def refusal(code: int, message: str, status: str) -> errors.APIError:
    """A real SDK error, built the way the SDK builds one from a response."""
    kind = errors.ClientError if code < 500 else errors.ServerError
    return kind(code, {"error": {"code": code, "message": message, "status": status}})


# --------------------------------------------------------------------------
# The fake client
# --------------------------------------------------------------------------


class FakeSession:
    """What `connect()` yields: what was sent, and the messages lined up.

    `receive()` is the SDK's: it yields messages up to and including the one
    that completes the turn and then ends, so the adapter has to call it
    again for the next turn; when the script runs out the socket "closes"
    normally, as an `APIError` 1000, which is what the SDK raises then.
    """

    def __init__(self, answers: list[Any]) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.answers = list(answers)
        self.receives = 0
        self.dead: BaseException | None = None

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self._send("send_realtime_input", kwargs)

    async def send_client_content(self, **kwargs: Any) -> None:
        self._send("send_client_content", kwargs)

    async def send_tool_response(self, **kwargs: Any) -> None:
        self._send("send_tool_response", kwargs)

    def _send(self, method: str, kwargs: dict[str, Any]) -> None:
        if self.dead is not None:
            raise self.dead
        self.sent.append((method, kwargs))

    async def receive(self) -> AsyncIterator[Any]:
        self.receives += 1
        while self.answers:
            answer = self.answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            yield answer
            if answer.server_content is not None and answer.server_content.turn_complete:
                return
        raise closed(1000, "")


class FakeLive:
    def __init__(self) -> None:
        self.opened: list[dict[str, Any]] = []
        self.answers: list[Any] = []
        self.refusal: BaseException | None = None
        self.sessions: list[FakeSession] = []

    @asynccontextmanager
    async def connect(self, *, model: str, config: Any) -> AsyncIterator[FakeSession]:
        self.opened.append({"model": model, "config": config})
        if self.refusal is not None:
            raise self.refusal
        session = FakeSession(self.answers)
        self.sessions.append(session)
        yield session


class FakeModels:
    """Stands in for `client.aio.models`."""

    def __init__(self, models: list[types.Model] | None = None) -> None:
        self.models = models or []
        self.error: BaseException | None = None

    async def list(self) -> Any:
        if self.error is not None:
            raise self.error

        class Pager:
            def __init__(self, models: list[types.Model]) -> None:
                self._models = models

            async def __aiter__(self) -> AsyncIterator[types.Model]:
                for model in self._models:
                    yield model

        return Pager(self.models)


class FakeClient:
    def __init__(self) -> None:
        self.live = FakeLive()
        self.models = FakeModels()

    @property
    def aio(self) -> FakeClient:
        return self


def gemini(*answers: Any, models: list[types.Model] | None = None) -> tuple[GeminiLive, FakeClient]:
    client = FakeClient()
    client.live.answers = list(answers)
    client.models.models = models or []
    return GeminiLive(api_key="AIza-not-a-key", client=client), client


def opened(client: FakeClient) -> types.LiveConnectConfig:
    [call] = client.live.opened
    config: types.LiveConnectConfig = call["config"]
    return config


def session_of(client: FakeClient) -> FakeSession:
    [session] = client.live.sessions
    return session


async def collect(adapter: GeminiLive, config: SessionConfig | None = None) -> list[LiveEvent]:
    async with adapter.connect(config or SessionConfig(model="gemini-x")) as session:
        return [event async for event in session.events()]


# --------------------------------------------------------------------------
# How the session is opened
# --------------------------------------------------------------------------


async def test_the_session_is_opened_for_audio_with_the_prompt_and_the_tools() -> None:
    adapter, client = gemini()

    await collect(
        adapter, SessionConfig(model="gemini-x", system_prompt="Be brief.", tools=[CLOCK, BARE])
    )

    assert client.live.opened[0]["model"] == "gemini-x"
    config = opened(client)
    assert config.response_modalities == [types.Modality.AUDIO]
    assert config.system_instruction == "Be brief."
    [tool] = config.tools or []
    assert isinstance(tool, types.Tool)
    declared = tool.function_declarations or []
    assert [d.name for d in declared] == ["get_current_time", "ping"]
    assert declared[0].description == "Returns the current local time."
    assert declared[0].parameters_json_schema == CLOCK.parameters
    # A schema with no properties is refused by the Live API (measured in
    # the spike, 2026-09-18): a tool without parameters is declared bare.
    assert declared[1].parameters_json_schema is None
    assert declared[0].behavior is None
    assert config.thinking_config is None


async def test_no_tool_block_and_no_prompt_are_sent_when_there_are_none() -> None:
    adapter, client = gemini()

    await collect(adapter, SessionConfig(model="gemini-x"))

    config = opened(client)
    assert config.tools is None
    assert config.system_instruction is None


async def test_the_voice_and_the_language_go_into_the_speech_config() -> None:
    adapter, client = gemini()

    await collect(adapter, SessionConfig(model="m", voice="Kore", language_code="tr-TR"))

    speech = opened(client).speech_config
    assert speech is not None
    assert speech.voice_config is not None
    assert speech.voice_config.prebuilt_voice_config is not None
    assert speech.voice_config.prebuilt_voice_config.voice_name == "Kore"
    assert speech.language_code == "tr-TR"


async def test_without_a_voice_or_a_language_the_model_s_own_are_left_alone() -> None:
    """Plan.md D19: the owner's ear preferred the default voice; nothing is
    sent that would pin it."""
    adapter, client = gemini()

    await collect(adapter, SessionConfig(model="m"))

    assert opened(client).speech_config is None


async def test_transcripts_are_asked_for_both_ways_with_the_language_hint() -> None:
    """ADR-001: the hint on the recogniser behind the model is what keeps a
    short "Saat kaç" from being heard as Hindi."""
    adapter, client = gemini()

    await collect(adapter, SessionConfig(model="m", language_code="tr-TR"))

    config = opened(client)
    assert config.input_audio_transcription is not None
    assert config.input_audio_transcription.language_codes == ["tr-TR"]
    assert config.output_audio_transcription is not None


async def test_without_a_language_the_recogniser_detects_it_itself() -> None:
    adapter, client = gemini()

    await collect(adapter, SessionConfig(model="m"))

    config = opened(client)
    assert config.input_audio_transcription is not None
    assert config.input_audio_transcription.language_codes is None


async def test_transcripts_off_asks_for_neither() -> None:
    adapter, client = gemini()

    await collect(adapter, SessionConfig(model="m", transcripts=False))

    config = opened(client)
    assert config.input_audio_transcription is None
    assert config.output_audio_transcription is None


async def test_the_server_s_turn_detection_is_left_alone_unless_told() -> None:
    adapter, client = gemini()

    await collect(adapter, SessionConfig(model="m"))

    assert opened(client).realtime_input_config is None


async def test_the_two_turn_detection_knobs_reach_the_server() -> None:
    """ADR-001 kept exactly two for the owner to tune: how eagerly the end of
    speech is called, and how much silence ends it."""
    adapter, client = gemini()

    await collect(adapter, SessionConfig(model="m", end_sensitivity="HIGH", silence_ms=300))

    realtime = opened(client).realtime_input_config
    assert realtime is not None
    detection = realtime.automatic_activity_detection
    assert detection is not None
    assert detection.end_of_speech_sensitivity == types.EndSensitivity.END_SENSITIVITY_HIGH
    assert detection.silence_duration_ms == 300
    assert detection.disabled is None


async def test_one_knob_alone_is_sent_alone() -> None:
    adapter, client = gemini()

    await collect(adapter, SessionConfig(model="m", end_sensitivity="low"))

    realtime = opened(client).realtime_input_config
    assert realtime is not None
    detection = realtime.automatic_activity_detection
    assert detection is not None
    assert detection.end_of_speech_sensitivity == types.EndSensitivity.END_SENSITIVITY_LOW
    assert detection.silence_duration_ms is None


async def test_a_resumption_handle_is_asked_for_and_given_back() -> None:
    """Asked for on every open, so that handles come; given, the
    conversation continues where the last session left it (plan.md D5)."""
    adapter, client = gemini()
    await collect(adapter, SessionConfig(model="m"))
    resumption = opened(client).session_resumption
    assert resumption is not None
    assert resumption.handle is None

    adapter, client = gemini()
    await collect(adapter, SessionConfig(model="m", resume_handle="h-36"))
    resumption = opened(client).session_resumption
    assert resumption is not None
    assert resumption.handle == "h-36"


def test_the_client_is_built_with_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[dict[str, Any]] = []

    class Client:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

    monkeypatch.setattr(genai, "Client", Client)

    GeminiLive(api_key="AIza-not-a-key")

    assert built == [{"api_key": "AIza-not-a-key"}]


def test_what_the_adapter_announces() -> None:
    adapter, _ = gemini()

    assert adapter.id == "gemini"
    assert adapter.capabilities == frozenset({"resumption", "transcripts"})
    assert DEFAULT_MODEL == "gemini-3.8-live"


# --------------------------------------------------------------------------
# What is sent
# --------------------------------------------------------------------------


async def test_audio_goes_as_pcm_at_the_session_s_rate() -> None:
    adapter, client = gemini()

    async with adapter.connect(SessionConfig(model="m")) as session:
        await session.send_audio(b"\x01\x02" * 320)
    sent = session_of(client).sent
    [(method, kwargs)] = sent
    assert method == "send_realtime_input"
    assert kwargs["audio"].mime_type == INPUT_MIME == "audio/pcm;rate=16000"
    assert kwargs["audio"].data == b"\x01\x02" * 320

    adapter, client = gemini()
    async with adapter.connect(SessionConfig(model="m", input_rate=24_000)) as session:
        await session.send_audio(bytes(48_000))
        assert session.audio_in_ms == 1000
    [(_, kwargs)] = session_of(client).sent
    assert kwargs["audio"].mime_type == "audio/pcm;rate=24000"


async def test_a_text_turn_goes_as_client_content() -> None:
    adapter, client = gemini()

    async with adapter.connect(SessionConfig(model="m")) as session:
        await session.send_text("saat kaç")
        await session.send_text("[said aloud] 15:00 toplantı", turn_complete=False)
        await session.send_text("Tamam.", role="model")

    sent = session_of(client).sent
    assert [method for method, _ in sent] == ["send_client_content"] * 3
    turns = [kwargs["turns"] for _, kwargs in sent]
    assert [turn.role for turn in turns] == ["user", "user", "model"]
    assert [turn.parts[0].text for turn in turns] == [
        "saat kaç",
        "[said aloud] 15:00 toplantı",
        "Tamam.",
    ]
    assert [kwargs["turn_complete"] for _, kwargs in sent] == [True, False, True]


async def test_a_tool_result_goes_back_as_a_function_response_by_id_and_name() -> None:
    """Gemini matches a result to its call by the function's name and refuses
    one without it ("Name cannot be empty", 2026-09-09); the id it issued
    travels back too (accepted, ADR-001)."""
    adapter, client = gemini()
    call = ToolCall(id="call_533936", name="get_current_time", arguments={})

    async with adapter.connect(SessionConfig(model="m")) as session:
        await session.send_tool_result(call, "2026-09-18T15:00 Friday")

    [(method, kwargs)] = session_of(client).sent
    assert method == "send_tool_response"
    [response] = kwargs["function_responses"]
    assert response.id == "call_533936"
    assert response.name == "get_current_time"
    assert response.response == {"result": "2026-09-18T15:00 Friday"}
    assert response.scheduling is None


async def test_an_id_the_model_never_issued_is_not_sent_back_as_an_empty_one() -> None:
    adapter, client = gemini()

    async with adapter.connect(SessionConfig(model="m")) as session:
        await session.send_tool_result(ToolCall(id="", name="ping", arguments={}), "pong")

    [(_, kwargs)] = session_of(client).sent
    assert kwargs["function_responses"][0].id is None


async def test_an_interruption_sends_nothing() -> None:
    """The server's own detector already stopped the model; there is nothing
    to tell it."""
    adapter, client = gemini()

    async with adapter.connect(SessionConfig(model="m")) as session:
        await session.interrupt()

    assert session_of(client).sent == []


async def test_sending_on_a_dead_socket_is_a_refusal_of_the_network() -> None:
    adapter, client = gemini()
    call = ToolCall(id="c1", name="ping", arguments={})

    async with adapter.connect(SessionConfig(model="m")) as session:
        session_of(client).dead = ConnectionClosedError(None, None)
        for sending in (
            session.send_audio(b"\x00\x00"),
            session.send_text("hi"),
            session.send_tool_result(call, "pong"),
        ):
            with pytest.raises(ProviderError) as raised:
                await sending
            assert raised.value.kind == "unreachable"


# --------------------------------------------------------------------------
# What is received
# --------------------------------------------------------------------------


async def test_the_audio_s_rate_is_read_from_the_blob_and_the_minutes_counted_by_it() -> None:
    """24 kHz is what the model speaks at (measured 2026-09-18), but the
    adapter believes the blob, not the constant."""
    adapter, _ = gemini(speaks(bytes(48_000)), speaks(bytes(32_000), rate=16_000))

    async with adapter.connect(SessionConfig(model="m")) as session:
        events = [event async for event in session.events()]
        heard = session.audio_out_ms

    chunks = [event for event in events if isinstance(event, AudioChunk)]
    assert [(len(c.pcm16), c.sample_rate) for c in chunks] == [(48_000, 24_000), (32_000, 16_000)]
    assert heard == 2000
    assert OUTPUT_RATE == 24_000


async def test_a_text_part_or_a_thought_in_the_model_turn_is_not_audio() -> None:
    parts = [types.Part(text="Bakıyorum."), types.Part(text="hmm", thought=True)]
    adapter, _ = gemini(content(model_turn=types.Content(role="model", parts=parts)))

    assert await collect(adapter) == [Closed()]


async def test_the_next_turn_is_received_with_a_fresh_receive() -> None:
    """The SDK's `receive()` ends at every `turn_complete` and the session
    does not; an adapter that read it once would fall silent after the first
    answer (found in the spike, 2026-09-18)."""
    adapter, client = gemini(says("a"), TURN_COMPLETE, says("b"), TURN_COMPLETE)

    events = await collect(adapter)

    assert events == [OutputText("a"), TurnComplete(), OutputText("b"), TurnComplete(), Closed()]
    assert session_of(client).receives == 3


async def test_the_transcripts_of_both_sides_are_events_and_the_interim_ones_are_not() -> None:
    interim = content(interim_input_transcription=types.Transcription(text="Saa"))
    adapter, _ = gemini(interim, hears("Saat kaç?"), says("Üç."))

    assert await collect(adapter) == [InputText("Saat kaç?"), OutputText("Üç."), Closed()]


async def test_several_calls_in_one_message_are_several_events() -> None:
    adapter, _ = gemini(calls(("c1", "clock", {}), ("c2", "open_app", {"name": "notepad"})))

    events = await collect(adapter)

    assert events[:2] == [
        ToolCallEvent(ToolCall(id="c1", name="clock", arguments={})),
        ToolCallEvent(ToolCall(id="c2", name="open_app", arguments={"name": "notepad"})),
    ]


async def test_a_call_without_an_id_arrives_with_an_empty_one() -> None:
    adapter, _ = gemini(calls((None, "clock", {})))

    [event, _] = await collect(adapter)

    assert isinstance(event, ToolCallEvent)
    assert event.call.id == ""


async def test_a_withdrawn_call_is_reported_with_its_ids() -> None:
    adapter, _ = gemini(cancels("c1", "c2"))

    assert (await collect(adapter))[0] == ToolCallCancelled(("c1", "c2"))


async def test_the_usage_metadata_is_the_report() -> None:
    adapter, _ = gemini(spends(1200, 90, cached=570))

    assert (await collect(adapter))[0] == UsageReport(Usage(1200, 90, 570))


async def test_a_handle_comes_from_the_resumption_update_only_when_it_can_be_used() -> None:
    adapter, _ = gemini(resumes("h-1"), resumes(None), resumes("h-2", resumable=False))

    assert await collect(adapter) == [Resumable("h-1"), Closed()]


async def test_going_away_carries_the_time_left_as_the_server_wrote_it() -> None:
    adapter, _ = gemini(leaves("59s"))

    assert (await collect(adapter))[0] == GoingAway("59s")


async def test_the_setup_acknowledgement_and_a_bare_message_are_nothing() -> None:
    adapter, _ = gemini(SETUP_ONLY, BARE_MESSAGE, content())

    assert await collect(adapter) == [Closed()]


async def test_a_normal_close_is_closed_without_an_error() -> None:
    """1000 is the socket closing the way sockets close; 1001 is the server
    going away after it said it would."""
    for code in (1000, 1001):
        adapter, _ = gemini(says("bye"), closed(code, "going away"))

        assert await collect(adapter) == [OutputText("bye"), Closed()]


async def test_an_abnormal_close_is_the_network() -> None:
    adapter, _ = gemini(closed(1006, "Abnormal closure."))

    [event] = await collect(adapter)

    assert isinstance(event, Closed)
    assert event.error is not None
    assert event.error.kind == "unreachable"
    assert "1006" in str(event.error)


async def test_a_close_for_what_we_sent_is_a_refusal_with_the_server_s_words() -> None:
    """1007 and 1008 are the server refusing something in the session -
    an unsupported field was closed with 1007 in the spike. Sending the
    user to check the network for that would be wrong."""
    adapter, _ = gemini(closed(1007, "Invalid frame payload data"))

    [event] = await collect(adapter)

    assert isinstance(event, Closed)
    assert event.error is not None
    assert event.error.kind == "refused"
    assert "Invalid frame payload data" in str(event.error)


async def test_the_transport_s_own_exception_from_the_socket_is_the_network() -> None:
    adapter, _ = gemini(OSError("[Errno 10054] connection reset"))

    [event] = await collect(adapter)

    assert isinstance(event, Closed)
    assert event.error is not None
    assert event.error.kind == "unreachable"


async def test_a_bug_is_not_swallowed() -> None:
    adapter, _ = gemini(RuntimeError("bug"))

    with pytest.raises(RuntimeError, match="bug"):
        await collect(adapter)


# --------------------------------------------------------------------------
# Models and keys
# --------------------------------------------------------------------------


async def test_only_models_that_can_hold_a_live_session_are_offered() -> None:
    adapter, _ = gemini(
        models=[
            types.Model(
                name="models/gemini-3.8-live",
                display_name="Gemini 3.8 Live",
                input_token_limit=128_000,
                supported_actions=["bidiGenerateContent"],
            ),
            types.Model(
                name="models/gemini-3.8-pro",
                display_name="Gemini 3.8 Pro",
                supported_actions=["generateContent"],
            ),
        ]
    )

    listed = await adapter.list_models()

    assert [(m.id, m.display_name) for m in listed] == [("gemini-3.8-live", "Gemini 3.8 Live")]
    assert listed[0].context_window == 128_000
    assert listed[0].supports_tools is None


async def test_a_refused_key_is_named_as_one_rather_than_left_to_the_caller() -> None:
    adapter, client = gemini()
    client.live.refusal = refusal(403, "Permission denied", "PERMISSION_DENIED")

    with pytest.raises(AuthenticationError):
        await collect(adapter)


async def test_the_shape_gemini_actually_refuses_a_bad_key_in_is_recognised() -> None:
    """Gemini answers a mistyped or revoked key with 400 INVALID_ARGUMENT and
    not with 401 - exactly the sort of vendor detail an adapter exists for."""
    adapter, client = gemini()
    client.live.refusal = refusal(
        400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT"
    )

    with pytest.raises(AuthenticationError):
        await collect(adapter)


async def test_the_free_tier_s_slow_down_is_told_apart() -> None:
    adapter, client = gemini()
    client.live.refusal = refusal(429, "You exceeded your current quota.", "RESOURCE_EXHAUSTED")

    with pytest.raises(ProviderError) as raised:
        await collect(adapter)

    assert raised.value.kind == "rate_limit"
    assert "quota" in str(raised.value)


async def test_a_failure_of_an_unexpected_shape_is_not_dressed_up_as_a_bad_key() -> None:
    """Whatever is left after the transport's errors are translated is a
    bug, and a bug dressed up as a bad key is one nobody fixes."""
    adapter, client = gemini()
    client.models.error = RuntimeError("something nobody foresaw")

    with pytest.raises(RuntimeError):
        await adapter.validate_credentials()


# --------------------------------------------------------------------------
# How the contract suite drives this adapter
#
# `test_live_adapters.py` scripts a session without naming a vendor; this is
# where that script becomes Gemini. Everything vendor-shaped about the
# contract suite is in these functions, which is the same rule the adapter
# itself lives by.
# --------------------------------------------------------------------------


def _message(step: Step) -> types.LiveServerMessage:
    """One step of a scripted session, in the messages Gemini sends."""
    if isinstance(step, Speaks):
        return speaks(bytes(step.ms * OUTPUT_RATE * 2 // 1000))
    if isinstance(step, Says):
        return says(step.text)
    if isinstance(step, Hears):
        return hears(step.text)
    if isinstance(step, Calls):
        return calls((step.id, step.name, dict(step.arguments)))
    if isinstance(step, Cancels):
        return cancels(*step.ids)
    if isinstance(step, Interrupts):
        return INTERRUPTED
    if isinstance(step, Spends):
        return spends(step.input, step.output, step.cached)
    if isinstance(step, Stops):
        return TURN_COMPLETE
    if isinstance(step, Resumes):
        return resumes(step.handle)
    if isinstance(step, Leaves):
        return leaves(step.time_left)
    return SETUP_ONLY


def _how_it_refuses(refuses: Refuses | None, *, mid_session: bool) -> BaseException | None:
    if refuses is None:
        return None
    if refuses is Refuses.THE_KEY:
        # The shape it really uses, rather than the 401 everybody expects.
        return refusal(400, "API key not valid. Please pass a valid API key.", "INVALID_ARGUMENT")
    if refuses is Refuses.THE_QUOTA:
        return refusal(429, "You exceeded your current quota.", "RESOURCE_EXHAUSTED")
    if refuses is Refuses.THE_NETWORK:
        if mid_session:
            # A socket that went: the SDK reports it as the abnormal close.
            return closed(1006, "Abnormal closure.")
        # Opening the socket: the transport's own exception for a name that
        # did not resolve, verbatim from the machine.
        return OSError("[Errno 11001] getaddrinfo failed")
    return refusal(503, COMPLAINT, "UNAVAILABLE")


def build(
    *script: Step,
    refuses: Refuses | None = None,
    after: int = 0,
    models: Sequence[tuple[str, str]] = (),
) -> LiveProvider:
    """What `live_contract.Build` asks for, answered in Gemini's own shapes."""
    answers: list[Any] = [_message(step) for step in script]
    listing_error: BaseException | None = None
    if refuses is not None and after:
        answers = [*answers[:after], _how_it_refuses(refuses, mid_session=True)]
    elif refuses is not None:
        listing_error = _how_it_refuses(refuses, mid_session=False)
        if refuses is Refuses.THE_NETWORK:
            # `list_models` goes over HTTP, where the same failure is httpx's.
            listing_error = httpx.ConnectError("[Errno 11001] getaddrinfo failed")

    adapter, client = gemini(
        *answers,
        models=[
            types.Model(
                name=f"models/{model_id}",
                display_name=name,
                supported_actions=["bidiGenerateContent"],
            )
            for model_id, name in models
        ],
    )
    if refuses is not None and not after:
        client.live.refusal = _how_it_refuses(refuses, mid_session=False)
        client.models.error = listing_error
    return adapter


GEMINI = Adapter(name="gemini_live", build=build)
