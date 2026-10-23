"""The contract every provider adapter has to pass, unchanged (section 3.2).

This is the exam the project's headline claim sits: the application talks to
Anthropic, to anything OpenAI-shaped and to Gemini through one protocol, and
the way that stays true is that one suite says what an adapter must do and
every adapter is run through it. Phase 2.7 adds a second and phase 4.5 a third;
neither of them may edit a line below.

Nothing here names a vendor. The tests script a provider in the words of
`contract.py` - it says something, it asks for a tool, it reports what the
request cost, it refuses - and each adapter's own test file translates that
script into the shapes its SDK really produces. `test_every_adapter_this_build
_has_is_in_this_suite` is what stops an adapter from being added to the
registry without one.

The two claims worth reading twice:

**`Delta.tool_call` is never half-built.** Every provider streams tool
arguments as partial JSON. The adapter buffers them and emits one whole call or
nothing, because the permission gate of section 3.9 cannot judge an action it
can only see the beginning of - and by the time the rest arrives, the gate has
already run.

**A refusal arrives as one of two exceptions and never as an SDK's own.** The
application decides out loud what to say about a failure (`app.py`), and it
cannot import three SDKs to find out which of them just failed.

The `def stream` signature is the subtle part. Adapters implement it with
`async def ... yield`, which returns an `AsyncIterator` when called, with no
`await`. Declaring it `async def` in the protocol would force callers to await
first and no adapter would satisfy the type.

Copied from the old repository's `test_llm_adapters.py` on 2026-10-23 with
the phone's text path (plan.md D58). This repository has one text adapter,
Gemini's, and no text registry yet - D55's text role (M7) builds one - so the
suite runs over Gemini alone, and the registry test waits for M7.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from allie.text.base import (
    AuthenticationError,
    Delta,
    LLMProvider,
    Message,
    ModelInfo,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
)
from tests.test_text_gemini import GEMINI
from tests.text_contract import (
    COMPLAINT,
    Adapter,
    Calls,
    Nothing,
    Refuses,
    Says,
    Spends,
    Starts,
    Stops,
)

PROVIDER_SDKS = frozenset({"google", "openai", "anthropic", "litellm"})

# One entry per text adapter this build can construct: Gemini's alone since
# the copy (2026-10-23). The old repository's OpenAI-compatible and Anthropic
# adapters were not copied; D55's text role decides whether they come.
ADAPTERS = [GEMINI]


@pytest.fixture(params=ADAPTERS, ids=lambda adapter: adapter.name)
def adapter(request: pytest.FixtureRequest) -> Adapter:
    """Every adapter in turn. Each test below runs once for each of them."""
    built: Adapter = request.param
    return built


async def spoken(provider: LLMProvider) -> list[Delta]:
    """One whole answer, collected. Phase 1 offers no tools (section 8)."""
    return [delta async for delta in provider.stream([Message.user("merhaba")], [], model="any")]


class FakeProvider:
    """A minimal provider that satisfies the protocol without any network call.

    The reference implementation: whatever a real adapter does differently, it
    must still look like this from the outside. Tests elsewhere in the suite
    use it as a stand-in for a real one, which is only honest as long as it
    answers the protocol the same way.
    """

    id = "fake"

    def __init__(self, deltas: list[Delta] | None = None) -> None:
        self.deltas = deltas if deltas is not None else [Delta(text="hello")]
        self.calls: list[dict[str, object]] = []

    async def validate_credentials(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="fake-1", display_name="Fake 1")]

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        self.calls.append({"messages": messages, "tools": tools, "model": model})
        for delta in self.deltas:
            yield delta


# --------------------------------------------------------------------------
# The shape of the protocol itself
# --------------------------------------------------------------------------


def test_an_async_generator_implementation_satisfies_the_protocol() -> None:
    # The annotation is the real assertion: mypy --strict checks it structurally,
    # which is stricter than isinstance, since a runtime protocol check only looks
    # for the names and ignores every signature.
    provider: LLMProvider = FakeProvider()

    assert isinstance(provider, LLMProvider)


async def test_stream_is_consumed_without_awaiting_the_call() -> None:
    provider = FakeProvider([Delta(text="Mer"), Delta(text="haba")])

    chunks = [delta.text async for delta in provider.stream([], [], model="fake-1")]

    assert chunks == ["Mer", "haba"]


def test_the_protocol_declares_stream_without_async_def() -> None:
    """An `async def` here would make every adapter fail to type check."""
    assert not inspect.iscoroutinefunction(LLMProvider.stream)


def test_a_tool_result_needs_the_call_it_answers() -> None:
    with pytest.raises(ValueError, match="tool_call_id"):
        Message(role="tool", content="42")


def test_a_tool_result_carries_the_name_of_the_tool_as_well_as_the_id() -> None:
    """Two providers match a result to its call by id, one by name (Gemini,
    which refuses a result without one). The protocol carries both, taken
    from the call itself so they cannot disagree."""
    call = ToolCall(id="c1", name="open_app", arguments={"name": "notepad"})

    result = Message.tool_result(call, "opened")

    assert (result.role, result.tool_call_id, result.tool_name) == ("tool", "c1", "open_app")
    assert result.content == "opened"


def test_the_cost_of_two_requests_adds_up() -> None:
    """A turn that ran a tool made more than one request; the turn's cost is
    their sum, and this is the one place counts are ever added."""
    assert Usage(1, 2, 3) + Usage(10, 20, 30) == Usage(11, 22, 33)


def test_only_an_assistant_message_carries_tool_calls() -> None:
    call = ToolCall(id="1", name="open_app", arguments={"name": "notepad"})

    with pytest.raises(ValueError, match="assistant"):
        Message(role="user", content="open notepad", tool_calls=(call,))


def test_a_message_cannot_be_edited_after_it_is_built() -> None:
    message = Message(role="user", content="merhaba")

    with pytest.raises(AttributeError):
        message.content = "something else"  # type: ignore[misc]


def test_an_empty_delta_carries_nothing() -> None:
    delta = Delta()

    assert delta.text is None
    assert delta.tool_call is None
    assert delta.finish_reason is None
    assert delta.usage is None


def test_token_counts_start_at_zero() -> None:
    usage = Usage()

    assert (usage.input_tokens, usage.output_tokens, usage.cached_tokens) == (0, 0, 0)


def test_the_protocol_module_imports_no_provider_sdk() -> None:
    """agent/, tools/ and policy.py read this module; a vendor import here leaks everywhere."""
    source = Path(inspect.getfile(Delta)).read_text(encoding="utf-8")
    imported: set[str] = set()

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])

    assert not imported & PROVIDER_SDKS


def test_a_refused_key_is_a_refusal_of_its_own_kind() -> None:
    """`app.py` says two different things and catches them in this order: a key
    has to be renewed by hand, a connection that dropped is probably back next
    turn (section 3.2). One being a subclass of the other is what lets the
    narrow case be answered first."""
    assert issubclass(AuthenticationError, ProviderError)
    assert issubclass(ProviderError, Exception)


# --------------------------------------------------------------------------
# The contract: every adapter, the same questions
# --------------------------------------------------------------------------


def test_every_adapter_satisfies_the_protocol(adapter: Adapter) -> None:
    provider: LLMProvider = adapter.build()

    assert isinstance(provider, LLMProvider)
    assert provider.id


def test_every_adapter_announces_what_it_can_do(adapter: Adapter) -> None:
    """A vendor-only feature - prompt caching, `reasoning_effort` - reaches the
    agent loop through `capabilities` and never through the protocol, which
    would drag every other adapter down to the common denominator."""
    announced = getattr(adapter.build(), "capabilities", None)

    assert isinstance(announced, frozenset)


async def test_the_stream_is_iterated_rather_than_awaited(adapter: Adapter) -> None:
    """No `await` on the call itself - see the note in the module docstring."""
    chunks = adapter.build(Says("hi")).stream([Message.user("merhaba")], [], model="any")

    assert [delta.text async for delta in chunks] == ["hi"]


async def test_text_arrives_in_the_order_it_was_streamed(adapter: Adapter) -> None:
    deltas = await spoken(adapter.build(Says("Mer"), Says("haba")))

    assert [delta.text for delta in deltas] == ["Mer", "haba"]


async def test_a_chunk_that_carried_nothing_is_not_a_delta(adapter: Adapter) -> None:
    """Every provider sends chunks that only advance its own state. An empty
    `Delta` would have the agent loop append nothing to the answer, over and
    over, and the turn would still be counted as having said something."""
    deltas = await spoken(adapter.build(Nothing(), Says("hi"), Nothing()))

    assert [delta.text for delta in deltas] == ["hi"]


# --------------------------------------------------------------------------
# What the turn cost
# --------------------------------------------------------------------------


async def test_the_token_counts_arrive_once_and_are_the_total(adapter: Adapter) -> None:
    """Providers disagree about when and how often they report cost, and some
    repeat a running total on every chunk. A consumer that added up every
    `Delta.usage` would report - and be billed for - several times the truth,
    and the cost report of section 6 would be quietly wrong."""
    deltas = await spoken(
        adapter.build(Says("One,"), Spends(7, 2), Says(" two"), Spends(7, 20), Stops())
    )

    spent = [delta.usage for delta in deltas if delta.usage is not None]

    assert len(spent) == 1
    assert (spent[0].input_tokens, spent[0].output_tokens) == (7, 20)


async def test_a_provider_that_reported_no_cost_is_not_given_one(adapter: Adapter) -> None:
    """Zero tokens is a claim that the request was free. Silence is not, and
    inventing the difference puts a number in the cost report nothing backs.
    The stream below ended properly; it simply never said what it cost."""
    deltas = await spoken(adapter.build(Says("hi"), Stops()))

    assert all(delta.usage is None for delta in deltas)


async def test_the_reason_generation_stopped_is_reported(adapter: Adapter) -> None:
    """What the word is, is the provider's business - `STOP`, `stop`,
    `end_turn`. That there is one is not."""
    deltas = await spoken(adapter.build(Says("hi"), Stops()))

    reasons = [delta.finish_reason for delta in deltas if delta.finish_reason is not None]

    assert len(reasons) == 1
    assert isinstance(reasons[0], str)


# --------------------------------------------------------------------------
# Tool calls, which phase 1 never sees and phase 2 depends on
# --------------------------------------------------------------------------


async def test_a_finished_tool_call_arrives_whole(adapter: Adapter) -> None:
    deltas = await spoken(adapter.build(Calls("open_app", {"name": "notepad"}, id="c7")))

    calls = [delta.tool_call for delta in deltas if delta.tool_call is not None]

    assert len(calls) == 1
    assert (calls[0].id, calls[0].name) == ("c7", "open_app")
    assert dict(calls[0].arguments) == {"name": "notepad"}


async def test_a_tool_call_still_being_streamed_is_never_yielded(adapter: Adapter) -> None:
    """Section 3.2's own sentence: `Delta.tool_call` is never half-built. The
    gate of section 3.9 cannot judge an action it can only see the beginning
    of, and by the time the rest arrives the gate has already run."""
    deltas = await spoken(
        adapter.build(
            Starts("send_email", {"to": "a@b"}),
            Calls("send_email", {"to": "a@b.com", "body": "hi"}),
        )
    )

    calls = [delta.tool_call for delta in deltas if delta.tool_call is not None]

    assert len(calls) == 1
    assert dict(calls[0].arguments) == {"to": "a@b.com", "body": "hi"}


# --------------------------------------------------------------------------
# Refusals: two kinds, and never the SDK's own
# --------------------------------------------------------------------------


async def test_a_refused_key_is_named_as_one(adapter: Adapter) -> None:
    """The turn is cancelled and the user is told to renew the key. Nothing
    above this layer may import an SDK to find that out."""
    with pytest.raises(AuthenticationError):
        await spoken(adapter.build(refuses=Refuses.THE_KEY))


async def test_a_provider_having_a_bad_day_is_not_a_key_problem(adapter: Adapter) -> None:
    """Telling the user to renew a working key over a 503 sends them to the
    provider's console to fix something that is not broken."""
    with pytest.raises(ProviderError) as raised:
        await spoken(adapter.build(refuses=Refuses.THE_REQUEST))

    assert not isinstance(raised.value, AuthenticationError)


async def test_a_refusal_carries_what_the_provider_said_about_it(adapter: Adapter) -> None:
    """The sentence the user hears is ours and says nothing useful to whoever
    has to diagnose this afterwards; the exception is where the provider's own
    words go."""
    with pytest.raises(ProviderError, match=COMPLAINT):
        await spoken(adapter.build(refuses=Refuses.THE_REQUEST))


async def test_a_stream_that_dies_part_way_through_is_a_refusal_too(adapter: Adapter) -> None:
    """The shape a dropped network really takes: the request was accepted and
    then the answer stopped arriving. Half an answer has already been yielded,
    and it must still not escape as an SDK exception."""
    provider = adapter.build(
        Says("Türkiye'nin"), Says(" başkenti"), refuses=Refuses.THE_REQUEST, after=1
    )

    with pytest.raises(ProviderError):
        await spoken(provider)


async def test_a_key_refused_while_listing_models_is_the_same_refusal(adapter: Adapter) -> None:
    """Every way out of an adapter reports failure in the same currency."""
    with pytest.raises(AuthenticationError):
        await adapter.build(refuses=Refuses.THE_KEY).list_models()


async def test_a_network_that_cannot_be_reached_is_a_refusal_not_an_sdk_exception(
    adapter: Adapter,
) -> None:
    """Wi-Fi off, VPN down, DNS gone. The SDK raises its transport library's
    own exception, and `app.py` catches neither that nor `OSError` - measured
    2026-09-05, the program ended with a traceback. Translating it is the
    adapter's job, exactly as for a 503."""
    with pytest.raises(ProviderError) as raised:
        await spoken(adapter.build(refuses=Refuses.THE_NETWORK))

    assert not isinstance(raised.value, AuthenticationError)


async def test_a_network_lost_mid_stream_is_a_refusal_too(adapter: Adapter) -> None:
    provider = adapter.build(Says("Türkiye'nin"), refuses=Refuses.THE_NETWORK, after=1)

    with pytest.raises(ProviderError):
        await spoken(provider)


async def test_a_network_lost_while_listing_models_is_the_same_refusal(adapter: Adapter) -> None:
    with pytest.raises(ProviderError):
        await adapter.build(refuses=Refuses.THE_NETWORK).list_models()


# --------------------------------------------------------------------------
# Which key, and which models it reaches
# --------------------------------------------------------------------------


async def test_a_key_that_was_refused_does_not_validate(adapter: Adapter) -> None:
    """The setup command asks this before it stores anything, so a `False` here
    is what keeps a dead key out of the Credential Manager (section 3.3)."""
    assert await adapter.build(refuses=Refuses.THE_KEY).validate_credentials() is False


async def test_an_unreachable_provider_does_not_pass_for_a_refused_key(adapter: Adapter) -> None:
    """`False` means "this key is dead, ask for another". Offline is not that,
    and answering `False` sends the user to renew a key that works. The setup
    command has its own sentence for a provider it cannot reach."""
    with pytest.raises(ProviderError):
        await adapter.build(refuses=Refuses.THE_NETWORK).validate_credentials()


async def test_a_working_key_validates(adapter: Adapter) -> None:
    assert await adapter.build(models=[("m-1", "Model One")]).validate_credentials() is True


async def test_the_models_a_key_reaches_are_reported_as_model_info(adapter: Adapter) -> None:
    """Whatever a provider calls its model list, the setup command reads one
    shape: an id to store in `config.toml` and a name to show."""
    listed = await adapter.build(models=[("m-1", "Model One")]).list_models()

    assert [(model.id, model.display_name) for model in listed] == [("m-1", "Model One")]
