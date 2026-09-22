"""The tool-use probe of design.md section 3.2 (2.6): one session that says
whether a model actually calls a tool, and a verdict kept for a week.

Three claims. A model passes by calling the canonical tool, whatever else
it says, and fails by talking instead - the failure is what the whole step
exists to catch, so it is a result and not an exception. What is sent is
the question the caller chose, as a text turn, with the one canonical tool
on offer and no transcripts; the question itself is the locale pack's
business (`test_locales.py`). And a verdict written down is read back until
it is a week old, after which it is as good as none: a provider may have
swapped the model behind the name.

The provider is the reference fake of `live_contract.py`, so no network;
the `settings` table is a real one, in memory.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from allie.live import probe as probe_module
from allie.live.base import (
    AudioChunk,
    Closed,
    LiveEvent,
    LiveProvider,
    OutputText,
    ProviderError,
    SessionConfig,
    ToolCall,
    ToolCallEvent,
    TurnComplete,
)
from allie.live.probe import (
    CANONICAL_TOOL_TEST,
    NO_TOOL_CALL,
    PROBE_SECONDS,
    PROBE_TTL_SECONDS,
    ProbeResult,
    probe_key,
    probe_tool_support,
    remember,
    remembered,
)
from allie.store.db import open_database
from allie.store.repos import SettingsRepo
from tests.live_contract import FakeLiveProvider, FakeLiveSession

QUESTION = "What time is it in Istanbul?"


def calls_the_clock(city: str = "Istanbul") -> ToolCallEvent:
    return ToolCallEvent(ToolCall(id="c1", name="get_current_time", arguments={"city": city}))


def probed(*events: LiveEvent) -> FakeLiveProvider:
    """A provider whose one session sends these events after the question."""
    return FakeLiveProvider(events)


class Silent(FakeLiveSession):
    """A session that opens and then says nothing for longer than the probe waits."""

    async def events(self) -> AsyncIterator[LiveEvent]:
        await asyncio.sleep(1)
        yield Closed()


class SilentProvider(FakeLiveProvider):
    @asynccontextmanager
    async def connect(self, config: SessionConfig) -> AsyncIterator[FakeLiveSession]:
        yield Silent()


@pytest.fixture
def verdicts() -> Iterator[SettingsRepo]:
    connection = open_database(":memory:")
    yield SettingsRepo(connection)
    connection.close()


# --------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------


def test_the_scripted_provider_is_a_provider() -> None:
    provider: LiveProvider = probed()

    assert isinstance(provider, LiveProvider)


async def test_a_model_that_calls_the_tool_passes() -> None:
    provider = probed(calls_the_clock(), TurnComplete())

    result = await probe_tool_support(provider, "m", question=QUESTION)

    assert result.ok is True
    assert result.reason is None


async def test_a_model_that_answers_in_words_fails_with_the_reason_written_down() -> None:
    """The silent failure of section 3.2, made loud: nothing went wrong on
    the wire, the model simply never reached for the tool."""
    provider = probed(OutputText("It is about three o'clock."), TurnComplete())

    result = await probe_tool_support(provider, "m", question=QUESTION)

    assert result.ok is False
    assert result.reason == NO_TOOL_CALL


async def test_a_call_to_some_other_tool_is_not_a_pass() -> None:
    """A model that invents a tool it was not offered has not shown it can
    call the one it was."""
    other = ToolCallEvent(ToolCall(id="c1", name="search_web", arguments={"q": "t"}))

    result = await probe_tool_support(probed(other, TurnComplete()), "m", question=QUESTION)

    assert result.ok is False


async def test_a_call_beside_some_words_still_passes() -> None:
    provider = probed(OutputText("Let me check."), calls_the_clock(), TurnComplete())

    assert (await probe_tool_support(provider, "m", question=QUESTION)).ok is True


@pytest.mark.parametrize(
    "first",
    [OutputText("hi"), AudioChunk(pcm16=bytes(2), sample_rate=24_000), calls_the_clock()],
    ids=["a word", "a sound", "the call"],
)
async def test_the_time_to_the_first_sign_of_life_is_measured(first: LiveEvent) -> None:
    result = await probe_tool_support(probed(first, TurnComplete()), "m", question=QUESTION)

    assert result.first_token_ms is not None
    assert result.first_token_ms > 0


async def test_a_model_that_said_nothing_at_all_has_no_first_token() -> None:
    """The end of a turn alone is not a sign of life; a made-up number would
    be read as a measurement."""
    result = await probe_tool_support(probed(TurnComplete()), "m", question=QUESTION)

    assert result.ok is False
    assert result.first_token_ms is None


async def test_the_model_is_sent_the_question_and_the_one_canonical_tool() -> None:
    """Exactly the session of section 3.2: the canonical clock tool on
    offer, no transcripts, the caller's question as one text turn that asks
    for an answer."""
    provider = probed(calls_the_clock())

    await probe_tool_support(provider, "the-model", question="Wie spät ist es in Istanbul?")

    [config] = provider.opened
    assert config.model == "the-model"
    assert list(config.tools) == [CANONICAL_TOOL_TEST]
    assert config.transcripts is False
    assert config.system_prompt == ""
    [session] = provider.sessions
    assert session.texts == [("Wie spät ist es in Istanbul?", "user", True)]


async def test_the_session_is_closed_the_moment_the_verdict_is_in() -> None:
    """The call is never answered and the spoken answer is not listened to:
    what comes after the call is never read."""
    provider = probed(calls_the_clock(), OutputText("never read"), TurnComplete())

    await probe_tool_support(provider, "m", question=QUESTION)

    assert provider.sessions[0].read == 1
    assert provider.sessions[0].results == []


async def test_the_server_hanging_up_without_a_word_is_a_failed_probe_not_a_crash() -> None:
    result = await probe_tool_support(probed(Closed()), "m", question=QUESTION)

    assert result == ProbeResult(ok=False, reason=NO_TOOL_CALL)


async def test_a_session_that_dies_is_the_provider_s_refusal() -> None:
    """The caller has a sentence for a provider that could not be asked, and
    it is not "this model cannot call tools"."""
    dropped = Closed(error=ProviderError("gemini could not be reached", kind="unreachable"))

    with pytest.raises(ProviderError) as raised:
        await probe_tool_support(probed(OutputText("Let"), dropped), "m", question=QUESTION)

    assert raised.value.kind == "unreachable"


async def test_a_refusal_at_the_open_is_the_provider_s_own() -> None:
    provider = FakeLiveProvider(refusal=ProviderError("slow down", kind="rate_limit"))

    with pytest.raises(ProviderError, match="slow down"):
        await probe_tool_support(provider, "m", question=QUESTION)


async def test_an_answer_that_does_not_come_in_time_is_a_refusal_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session that opens and then says nothing would hold the wizard for
    ever; the deadline turns it into the sentence for a provider that could
    not be asked."""
    monkeypatch.setattr(probe_module, "PROBE_SECONDS", 0.05)

    with pytest.raises(ProviderError) as raised:
        await probe_tool_support(SilentProvider(), "m", question=QUESTION)

    assert raised.value.kind == "timeout"
    assert "m" in str(raised.value)
    assert PROBE_SECONDS == 10.0


def test_the_canonical_tool_is_the_one_of_section_3_2() -> None:
    assert CANONICAL_TOOL_TEST.name == "get_current_time"
    assert CANONICAL_TOOL_TEST.parameters["required"] == ["city"]


# --------------------------------------------------------------------------
# The verdict, kept for a week
# --------------------------------------------------------------------------


def test_nothing_remembered_is_nothing(verdicts: SettingsRepo) -> None:
    assert remembered(verdicts, "gemini", "gemini-x") is None


def test_a_verdict_written_down_is_read_back(verdicts: SettingsRepo) -> None:
    result = ProbeResult(ok=True, first_token_ms=812.5)

    remember(verdicts, "gemini", "gemini-x", result, now=1_000_000)

    assert remembered(verdicts, "gemini", "gemini-x", now=1_000_100) == result


def test_a_failed_verdict_is_read_back_with_its_reason(verdicts: SettingsRepo) -> None:
    result = ProbeResult(ok=False, reason=NO_TOOL_CALL, first_token_ms=300.0)

    remember(verdicts, "groq", "llama", result, now=1_000_000)

    assert remembered(verdicts, "groq", "llama", now=1_000_000) == result


def test_a_verdict_a_week_old_is_no_verdict(verdicts: SettingsRepo) -> None:
    """Section 3.2: a provider may have swapped the model behind the name.
    Exactly a week old it still counts; a second more, it is asked again."""
    remember(verdicts, "gemini", "gemini-x", ProbeResult(ok=True), now=1_000_000)

    just_in_time = 1_000_000 + PROBE_TTL_SECONDS
    assert remembered(verdicts, "gemini", "gemini-x", now=just_in_time) is not None
    assert remembered(verdicts, "gemini", "gemini-x", now=just_in_time + 1) is None


def test_the_verdict_is_kept_per_provider_and_model(verdicts: SettingsRepo) -> None:
    remember(verdicts, "gemini", "fast", ProbeResult(ok=True), now=1)
    remember(verdicts, "gemini", "smart", ProbeResult(ok=False, reason=NO_TOOL_CALL), now=1)

    assert remembered(verdicts, "gemini", "fast", now=1) == ProbeResult(ok=True)
    assert remembered(verdicts, "gemini", "smart", now=1) is not None
    assert remembered(verdicts, "groq", "fast", now=1) is None


def test_a_newer_verdict_replaces_the_older(verdicts: SettingsRepo) -> None:
    remember(verdicts, "gemini", "x", ProbeResult(ok=False, reason=NO_TOOL_CALL), now=1)
    remember(verdicts, "gemini", "x", ProbeResult(ok=True, first_token_ms=5.0), now=2)

    assert remembered(verdicts, "gemini", "x", now=2) == ProbeResult(ok=True, first_token_ms=5.0)


def test_the_row_is_the_json_of_section_3_2(verdicts: SettingsRepo) -> None:
    """`probe:<provider>:<model>` to `{ok, ts, first_token_ms}`, readable by
    anyone with `sqlite3` in hand."""
    remember(verdicts, "gemini", "gemini-x", ProbeResult(ok=True, first_token_ms=812.5), now=123)

    stored = verdicts.get(probe_key("gemini", "gemini-x"))
    assert stored is not None
    record = json.loads(stored)
    assert (record["ok"], record["ts"], record["first_token_ms"]) == (True, 123, 812.5)


@pytest.mark.parametrize(
    "stored",
    ["not json", "[1, 2]", '{"ok": "yes", "ts": 1}', '{"ok": true}', '{"ok": true, "ts": "now"}'],
)
def test_a_row_that_cannot_be_read_is_no_verdict(verdicts: SettingsRepo, stored: str) -> None:
    """A verdict of unknown age, or of unknown shape, is asked again rather
    than guessed at."""
    verdicts.set(probe_key("gemini", "x"), stored)

    assert remembered(verdicts, "gemini", "x", now=1) is None


def test_the_key_names_the_provider_and_the_model() -> None:
    key = probe_key("openrouter", "google/gemini-2.5-flash")

    assert key == "probe:openrouter:google/gemini-2.5-flash"


def test_the_clock_is_the_machine_s_unless_the_caller_says_otherwise(
    verdicts: SettingsRepo,
) -> None:
    remember(verdicts, "gemini", "x", ProbeResult(ok=True))

    assert remembered(verdicts, "gemini", "x") == ProbeResult(ok=True)


def test_a_verdict_survives_the_connection_being_reopened(tmp_path: Path) -> None:
    """The point of the table: the wizard writes, a later `allie run` reads."""
    path = tmp_path / "assistant.db"
    first = open_database(path)
    remember(SettingsRepo(first), "gemini", "x", ProbeResult(ok=True, first_token_ms=1.0), now=5)
    first.close()

    second = open_database(path)
    try:
        found = remembered(SettingsRepo(second), "gemini", "x", now=5)
    finally:
        second.close()

    assert found == ProbeResult(ok=True, first_token_ms=1.0)
