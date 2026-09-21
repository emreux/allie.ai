"""The contract every live adapter has to pass, unchanged (plan.md section 4.3).

This is the exam the project's headline claim sits: the state machine talks
to Gemini Live and to OpenAI's live API through one protocol, and the way
that stays true is that one suite says what an adapter must do and every
adapter is run through it. L2 adds the second; it may not edit a line below.

Nothing here names a vendor. The tests script a session in the words of
`live_contract.py` - the server speaks, says what it said, asks for a tool,
interrupts, reports what the turn cost, hangs up - and each adapter's own
test file translates that script into the messages its SDK really
produces. `test_every_adapter_this_build_has_is_in_this_suite` is what
stops an adapter from being added to the registry without one.

The claims worth reading twice:

**A session's events end with `Closed`, and a refusal arrives as one of two
exceptions and never as an SDK's own.** The state machine decides out loud
what to say about a failure (`app.py`), and it cannot import two SDKs to
find out which of them just failed. What the socket dying looks like from
above is a `Closed` carrying the same `ProviderError` a refusal at the open
would have raised.

**The minutes are counted by the session, exactly.** Live models bill by
the minute of audio each way (plan.md D8); the count comes from the bytes
that went through and not from anything the provider says.

The `def events` signature is the subtle part. Adapters implement it with
`async def ... yield`, which returns an `AsyncIterator` when called, with
no `await`. Declaring it `async def` in the protocol would force callers to
await first and no adapter would satisfy the type.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from assistant.live import registry
from assistant.live.base import (
    AudioChunk,
    AuthenticationError,
    Closed,
    GoingAway,
    InputText,
    Interrupted,
    LiveEvent,
    LiveProvider,
    LiveSession,
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
from tests.live_contract import (
    COMPLAINT,
    Adapter,
    Calls,
    Cancels,
    FakeLiveProvider,
    FakeLiveSession,
    Hears,
    Interrupts,
    Leaves,
    Nothing,
    Refuses,
    Resumes,
    Says,
    Speaks,
    Spends,
    Step,
    Stops,
)
from tests.test_gemini_live import GEMINI

PROVIDER_SDKS = frozenset({"google", "openai", "anthropic", "litellm", "websockets", "httpx"})

CLOCK = ToolSpec(
    name="get_current_time",
    description="Returns the current local time.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)
CONFIG = SessionConfig(model="any", system_prompt="Be brief.", tools=[CLOCK])

# One entry per adapter this build can construct. L2 appends the OpenAI one;
# the tests do not move.
ADAPTERS = [GEMINI]


# --------------------------------------------------------------------------
# The reference implementation, driven the way the real ones are
# --------------------------------------------------------------------------


def _event(step: Step) -> LiveEvent | None:
    if isinstance(step, Speaks):
        return AudioChunk(pcm16=bytes(step.ms * 48), sample_rate=24_000)
    if isinstance(step, Says):
        return OutputText(step.text)
    if isinstance(step, Hears):
        return InputText(step.text)
    if isinstance(step, Calls):
        return ToolCallEvent(ToolCall(id=step.id, name=step.name, arguments=dict(step.arguments)))
    if isinstance(step, Cancels):
        return ToolCallCancelled(step.ids)
    if isinstance(step, Interrupts):
        return Interrupted()
    if isinstance(step, Spends):
        return UsageReport(Usage(step.input, step.output, step.cached))
    if isinstance(step, Stops):
        return TurnComplete()
    if isinstance(step, Resumes):
        return Resumable(step.handle)
    if isinstance(step, Leaves):
        return GoingAway(step.time_left)
    return None


def _refusal(refuses: Refuses | None) -> ProviderError | None:
    if refuses is None:
        return None
    if refuses is Refuses.THE_KEY:
        return AuthenticationError("fake refused the key")
    if refuses is Refuses.THE_QUOTA:
        return ProviderError("fake said: slow down", kind="rate_limit")
    if refuses is Refuses.THE_NETWORK:
        return ProviderError("fake could not be reached", kind="unreachable")
    return ProviderError(f"fake refused the request: {COMPLAINT}")


def build_fake(
    *script: Step,
    refuses: Refuses | None = None,
    after: int = 0,
    models: Sequence[tuple[str, str]] = (),
) -> LiveProvider:
    """What `live_contract.Build` asks for, answered by the reference fake."""
    events = [event for event in map(_event, script) if event is not None]
    refusal = _refusal(refuses)
    if refusal is not None and after:
        events = [*events[:after], Closed(error=refusal)]
        refusal = None
    return FakeLiveProvider(
        events,
        models=[ModelInfo(id=model_id, display_name=name) for model_id, name in models],
        refusal=refusal,
    )


REFERENCE = Adapter(name="fake", build=build_fake)


@pytest.fixture(params=[*ADAPTERS, REFERENCE], ids=lambda adapter: adapter.name)
def adapter(request: pytest.FixtureRequest) -> Adapter:
    """Every adapter in turn, and the reference fake after them. Each test
    below runs once for each of them."""
    built: Adapter = request.param
    return built


async def played(provider: LiveProvider) -> list[LiveEvent]:
    """One whole session, collected: opened, listened to until it closes."""
    async with provider.connect(CONFIG) as session:
        return [event async for event in session.events()]


# --------------------------------------------------------------------------
# The shape of the protocol itself
# --------------------------------------------------------------------------


def test_the_reference_implementation_satisfies_both_protocols() -> None:
    # The annotations are the real assertion: mypy --strict checks them
    # structurally, which is stricter than isinstance, since a runtime
    # protocol check only looks for the names and ignores every signature.
    provider: LiveProvider = FakeLiveProvider()
    session: LiveSession = FakeLiveSession()

    assert isinstance(provider, LiveProvider)
    assert isinstance(session, LiveSession)


async def test_events_are_consumed_without_awaiting_the_call() -> None:
    session = FakeLiveSession([OutputText("Mer"), OutputText("haba")])

    heard = [event async for event in session.events()]

    assert heard == [OutputText("Mer"), OutputText("haba"), Closed()]


def test_the_protocol_declares_events_without_async_def() -> None:
    """An `async def` here would make every adapter fail to type check."""
    assert not inspect.iscoroutinefunction(LiveSession.events)


def test_the_protocol_declares_connect_as_a_context_manager_not_a_coroutine() -> None:
    """`async with provider.connect(config)`: entered, never awaited."""
    assert not inspect.iscoroutinefunction(LiveProvider.connect)


def test_the_cost_of_two_reports_adds_up() -> None:
    """A turn with a tool call in it is two server turns with a report each;
    the turn's cost is their sum, and this is the one place counts are added."""
    assert Usage(1, 2, 3) + Usage(10, 20, 30) == Usage(11, 22, 33)


def test_token_counts_start_at_zero() -> None:
    usage = Usage()

    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens) == (0, 0, 0)


def test_a_session_opens_with_the_plan_s_defaults() -> None:
    """16 kHz in, transcripts on, the model's own voice, a fresh conversation,
    the server's own turn detection (plan.md 4.3, D19)."""
    config = SessionConfig(model="m")

    assert (config.input_rate, config.transcripts, config.voice) == (16_000, True, "")
    assert (config.resume_handle, config.language_code) == (None, "")
    assert (config.end_sensitivity, config.silence_ms) == ("", 0)
    assert list(config.tools) == []
    assert config.web_search is False


def test_an_event_cannot_be_edited_after_it_is_built() -> None:
    event = OutputText("merhaba")

    with pytest.raises(AttributeError):
        event.text = "something else"  # type: ignore[misc]


def test_the_protocol_module_imports_no_provider_sdk() -> None:
    """agent/, tools/ and policy.py read this module; a vendor import here leaks everywhere."""
    source = Path(inspect.getfile(SessionConfig)).read_text(encoding="utf-8")
    imported: set[str] = set()

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    assert not imported & PROVIDER_SDKS


def test_a_refused_key_is_a_refusal_of_its_own_kind() -> None:
    """`app.py` says two different things and catches them in this order: a
    key has to be renewed by hand, a connection that dropped is probably back
    next turn. One being a subclass of the other is what lets the narrow case
    be answered first; the `kind` is the one word the state machine reads."""
    assert issubclass(AuthenticationError, ProviderError)
    assert issubclass(ProviderError, Exception)
    assert ProviderError("no").kind == "refused"
    assert ProviderError("later", kind="rate_limit").kind == "rate_limit"
    assert AuthenticationError("bad key").kind == "key"
    assert str(AuthenticationError("bad key")) == "bad key"


# --------------------------------------------------------------------------
# The contract: every adapter, the same questions
# --------------------------------------------------------------------------


def test_every_adapter_this_build_has_is_in_this_suite() -> None:
    """An adapter added to the registry without a line here would leave the
    contract untested for exactly the one nobody has run yet."""
    assert {entry.name for entry in ADAPTERS} == set(registry.ADAPTERS)


def test_every_adapter_satisfies_the_protocol(adapter: Adapter) -> None:
    provider: LiveProvider = adapter.build()

    assert isinstance(provider, LiveProvider)
    assert provider.id


def test_every_adapter_announces_what_it_can_do(adapter: Adapter) -> None:
    """A vendor-only feature - a resumption handle, transcripts - reaches the
    state machine through `capabilities` and never through the protocol,
    which would drag every other adapter down to the common denominator."""
    announced = getattr(adapter.build(), "capabilities", None)

    assert isinstance(announced, frozenset)


async def test_a_session_asked_for_web_search_still_opens_on_every_adapter(
    adapter: Adapter,
) -> None:
    """The flag is provider-agnostic (spec section 2): an adapter with the
    capability sends its vendor's tool, one without it opens all the same."""
    provider = adapter.build()

    async with provider.connect(replace(CONFIG, web_search=True)) as session:
        assert isinstance(session, LiveSession)


async def test_a_session_is_entered_with_async_with_and_satisfies_the_protocol(
    adapter: Adapter,
) -> None:
    async with adapter.build(Stops()).connect(CONFIG) as session:
        assert isinstance(session, LiveSession)


async def test_the_events_are_iterated_rather_than_awaited(adapter: Adapter) -> None:
    """No `await` on the call itself - see the note in the module docstring."""
    async with adapter.build(Says("hi")).connect(CONFIG) as session:
        events = session.events()

        assert [e async for e in events] == [OutputText("hi"), Closed()]


# --------------------------------------------------------------------------
# What the server sends, in the order it sent it
# --------------------------------------------------------------------------


async def test_the_model_s_voice_arrives_as_audio_in_the_order_it_was_spoken(
    adapter: Adapter,
) -> None:
    """Whatever the rate, each chunk says what it is, and the session counts
    the milliseconds - what the bill is made of."""
    provider = adapter.build(Speaks(100), Speaks(200))

    async with provider.connect(CONFIG) as session:
        chunks = [e async for e in session.events() if isinstance(e, AudioChunk)]
        heard = session.audio_out_ms

    assert len(chunks) == 2
    assert all(chunk.sample_rate > 0 and chunk.pcm16 for chunk in chunks)
    assert [len(c.pcm16) * 1000 // (c.sample_rate * 2) for c in chunks] == [100, 200]
    assert heard == 300


async def test_what_the_model_said_arrives_as_text_in_order(adapter: Adapter) -> None:
    events = await played(adapter.build(Says("Mer"), Says("haba")))

    assert [e.text for e in events if isinstance(e, OutputText)] == ["Mer", "haba"]


async def test_what_the_server_heard_arrives_as_text(adapter: Adapter) -> None:
    events = await played(adapter.build(Hears("Saat"), Hears(" kaç?")))

    assert [e.text for e in events if isinstance(e, InputText)] == ["Saat", " kaç?"]


async def test_a_message_that_carried_nothing_is_not_an_event(adapter: Adapter) -> None:
    """Every provider sends messages that only advance its own state. An
    event for one would have the state machine act on nothing."""
    events = await played(adapter.build(Nothing(), Says("hi"), Nothing()))

    assert events == [OutputText("hi"), Closed()]


async def test_a_tool_call_arrives_whole(adapter: Adapter) -> None:
    """The permission gate cannot judge an action it can only see the
    beginning of, so a call is one event with every argument in it."""
    events = await played(adapter.build(Calls("open_app", {"name": "notepad"}, id="c7")))

    calls = [e.call for e in events if isinstance(e, ToolCallEvent)]

    assert len(calls) == 1
    assert (calls[0].id, calls[0].name) == ("c7", "open_app")
    assert dict(calls[0].arguments) == {"name": "notepad"}


async def test_a_withdrawn_call_is_reported_with_its_ids(adapter: Adapter) -> None:
    """The user spoke over the model before the tool answered; the result,
    if it comes, is not sent for these."""
    events = await played(adapter.build(Calls("open_app", {}, id="c7"), Cancels(("c7",))))

    assert [e.ids for e in events if isinstance(e, ToolCallCancelled)] == [("c7",)]


async def test_an_interruption_is_reported(adapter: Adapter) -> None:
    events = await played(adapter.build(Speaks(100), Interrupts(), Stops()))

    kinds = [type(e) for e in events]

    assert kinds == [AudioChunk, Interrupted, TurnComplete, Closed]


async def test_the_end_of_a_turn_is_reported_and_the_session_goes_on(adapter: Adapter) -> None:
    """One session, many turns: the second turn's events arrive through the
    same iterator. (Gemini's SDK ends its `receive()` at every turn and the
    adapter has to ask again; measured 2026-09-18.)"""
    events = await played(adapter.build(Says("a"), Stops(), Says("b"), Stops()))

    assert events == [OutputText("a"), TurnComplete(), OutputText("b"), TurnComplete(), Closed()]


async def test_the_token_counts_arrive_as_reported(adapter: Adapter) -> None:
    events = await played(adapter.build(Says("hi"), Spends(7, 20, cached=3), Stops()))

    reports = [e.usage for e in events if isinstance(e, UsageReport)]

    assert reports == [Usage(input_tokens=7, output_tokens=20, cached_tokens=3)]


async def test_a_provider_that_reported_no_cost_is_not_given_one(adapter: Adapter) -> None:
    """Zero tokens is a claim that the turn was free. Silence is not, and
    inventing the difference puts a number in the cost report nothing backs."""
    events = await played(adapter.build(Says("hi"), Stops()))

    assert not any(isinstance(e, UsageReport) for e in events)


async def test_a_resumption_handle_is_surfaced(adapter: Adapter) -> None:
    """With it the next session continues this conversation (plan.md D5)."""
    events = await played(adapter.build(Resumes("h-9"), Says("hi")))

    assert [e.handle for e in events if isinstance(e, Resumable)] == ["h-9"]


async def test_the_server_s_notice_of_hanging_up_is_surfaced(adapter: Adapter) -> None:
    events = await played(adapter.build(Leaves("12s")))

    assert [e.time_left for e in events if isinstance(e, GoingAway)] == ["12s"]


async def test_when_the_server_hangs_up_the_events_end_with_closed(adapter: Adapter) -> None:
    """The iterator ends, and the last thing it said is why: nothing, here -
    the socket closed the way sockets close."""
    events = await played(adapter.build(Says("bye"), Stops()))

    assert events[-1] == Closed()
    assert events[-1].error is None
    assert events.count(Closed()) == 1


# --------------------------------------------------------------------------
# What is sent, and what it counts
# --------------------------------------------------------------------------


async def test_nothing_is_counted_before_anything_is_sent_or_heard(adapter: Adapter) -> None:
    async with adapter.build().connect(CONFIG) as session:
        assert (session.audio_in_ms, session.audio_out_ms) == (0, 0)


async def test_the_audio_sent_is_counted_in_milliseconds(adapter: Adapter) -> None:
    """One second of 16-bit mono at the session's 16 kHz is 32 000 bytes;
    the count is exact and only grows."""
    async with adapter.build().connect(CONFIG) as session:
        await session.send_audio(bytes(32_000))
        after_one = session.audio_in_ms
        await session.send_audio(bytes(16_000))

        assert (after_one, session.audio_in_ms) == (1000, 1500)


async def test_text_a_tool_result_and_an_interruption_can_be_sent(adapter: Adapter) -> None:
    """What each vendor makes of them is its adapter's test; that the
    session takes them is everyone's."""
    async with adapter.build().connect(CONFIG) as session:
        await session.send_text("saat kaç")
        await session.send_text("[said aloud] reminder", role="user", turn_complete=False)
        await session.send_tool_result(ToolCall(id="c1", name="clock", arguments={}), "15:00")
        await session.interrupt()


# --------------------------------------------------------------------------
# Refusals: four kinds, and never the SDK's own
# --------------------------------------------------------------------------


async def test_a_refused_key_is_named_as_one(adapter: Adapter) -> None:
    """The session is not opened and the user is told to renew the key.
    Nothing above this layer may import an SDK to find that out."""
    with pytest.raises(AuthenticationError):
        await played(adapter.build(refuses=Refuses.THE_KEY))


async def test_a_provider_having_a_bad_day_is_not_a_key_problem(adapter: Adapter) -> None:
    """Telling the user to renew a working key over a 503 sends them to the
    provider's console to fix something that is not broken."""
    with pytest.raises(ProviderError) as raised:
        await played(adapter.build(refuses=Refuses.THE_REQUEST))

    assert not isinstance(raised.value, AuthenticationError)
    assert raised.value.kind == "refused"


async def test_a_refusal_carries_what_the_provider_said_about_it(adapter: Adapter) -> None:
    """The sentence the user hears is ours and says nothing useful to whoever
    has to diagnose this afterwards; the exception is where the provider's
    own words go."""
    with pytest.raises(ProviderError, match=COMPLAINT):
        await played(adapter.build(refuses=Refuses.THE_REQUEST))


async def test_the_quota_is_a_refusal_of_its_own_kind(adapter: Adapter) -> None:
    """ "Slow down" is neither a dead key nor a dead network: the session is
    tried again later, and the sentence says so."""
    with pytest.raises(ProviderError) as raised:
        await played(adapter.build(refuses=Refuses.THE_QUOTA))

    assert not isinstance(raised.value, AuthenticationError)
    assert raised.value.kind == "rate_limit"


async def test_a_network_that_cannot_be_reached_is_a_refusal_not_an_sdk_exception(
    adapter: Adapter,
) -> None:
    """Wi-Fi off, VPN down, DNS gone. The SDK raises its transport library's
    own exception, which `app.py` catches neither as `ProviderError` nor as
    `OSError`. Translating it is the adapter's job, exactly as for a 503."""
    with pytest.raises(ProviderError) as raised:
        await played(adapter.build(refuses=Refuses.THE_NETWORK))

    assert not isinstance(raised.value, AuthenticationError)
    assert raised.value.kind == "unreachable"


async def test_a_session_that_dies_part_way_through_ends_with_a_closed_that_says_why(
    adapter: Adapter,
) -> None:
    """The shape a dropped network really takes: the session was open and
    then the socket went. What was heard before is kept; what comes after
    is one `Closed` with the reason in the same currency as a refusal - and
    the iterator ends, it does not raise."""
    provider = adapter.build(
        Says("Türkiye'nin"), Says(" başkenti"), refuses=Refuses.THE_NETWORK, after=1
    )

    events = await played(provider)

    assert events[0] == OutputText("Türkiye'nin")
    assert isinstance(events[-1], Closed)
    assert isinstance(events[-1].error, ProviderError)
    assert events[-1].error.kind == "unreachable"
    assert len(events) == 2


async def test_a_key_refused_while_listing_models_is_the_same_refusal(adapter: Adapter) -> None:
    """Every way out of an adapter reports failure in the same currency."""
    with pytest.raises(AuthenticationError):
        await adapter.build(refuses=Refuses.THE_KEY).list_models()


async def test_a_network_lost_while_listing_models_is_the_same_refusal(adapter: Adapter) -> None:
    with pytest.raises(ProviderError) as raised:
        await adapter.build(refuses=Refuses.THE_NETWORK).list_models()

    assert raised.value.kind == "unreachable"


# --------------------------------------------------------------------------
# Which key, and which models it reaches
# --------------------------------------------------------------------------


async def test_a_key_that_was_refused_does_not_validate(adapter: Adapter) -> None:
    """The setup command asks this before it stores anything, so a `False`
    here is what keeps a dead key out of the Credential Manager."""
    assert await adapter.build(refuses=Refuses.THE_KEY).validate_credentials() is False


async def test_an_unreachable_provider_does_not_pass_for_a_refused_key(adapter: Adapter) -> None:
    """`False` means "this key is dead, ask for another". Offline is not
    that, and answering `False` sends the user to renew a key that works.
    The setup command has its own sentence for a provider it cannot reach."""
    with pytest.raises(ProviderError):
        await adapter.build(refuses=Refuses.THE_NETWORK).validate_credentials()


async def test_a_working_key_validates(adapter: Adapter) -> None:
    assert await adapter.build(models=[("m-1", "Model One")]).validate_credentials() is True


async def test_the_models_a_key_reaches_are_reported_as_model_info(adapter: Adapter) -> None:
    """Whatever a provider calls its model list, the setup command reads one
    shape: an id to store in `config.toml` and a name to show."""
    listed = await adapter.build(models=[("m-1", "Model One")]).list_models()

    assert [(model.id, model.display_name) for model in listed] == [("m-1", "Model One")]
