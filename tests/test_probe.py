"""The tool-use probe of design.md section 3.2 (2.6): one request that says
whether a model actually calls a tool, and a verdict kept for a week.

Three claims. A model passes by calling the canonical tool, whatever else it
says, and fails by writing prose instead - the failure is what the whole
step exists to catch, so it is a result and not an exception. What is sent
is the question the caller chose and the one canonical tool, with a small
token ceiling; the question itself is the locale pack's business
(`test_locales.py`). And a verdict written down is read back until it is a
week old, after which it is as good as none: a provider may have swapped
the model behind the name.

The provider is scripted, so no network; the `settings` table is a real
one, in memory.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from assistant.live.base import Delta, LLMProvider, Message, ModelInfo, ToolCall, ToolSpec
from assistant.live.probe import (
    CANONICAL_TOOL_TEST,
    MAX_TOKENS,
    NO_TOOL_CALL,
    PROBE_TTL_SECONDS,
    ProbeResult,
    probe_key,
    probe_tool_support,
    remember,
    remembered,
)
from assistant.store.db import open_database
from assistant.store.repos import SettingsRepo

QUESTION = "What time is it in Istanbul?"


class Probed:
    """A provider that answers with a script and remembers what it was asked."""

    id = "probed"

    def __init__(self, *deltas: Delta) -> None:
        self.deltas = list(deltas)
        self.asked: list[dict[str, Any]] = []
        self.read = 0

    async def validate_credentials(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        self.asked.append(
            {"messages": messages, "tools": tools, "model": model, "max_tokens": max_tokens}
        )
        for delta in self.deltas:
            self.read += 1
            yield delta


def calls_the_clock(city: str = "Istanbul") -> Delta:
    return Delta(tool_call=ToolCall(id="c1", name="get_current_time", arguments={"city": city}))


@pytest.fixture
def verdicts() -> Iterator[SettingsRepo]:
    connection = open_database(":memory:")
    yield SettingsRepo(connection)
    connection.close()


# --------------------------------------------------------------------------
# The request
# --------------------------------------------------------------------------


def test_the_scripted_provider_is_a_provider() -> None:
    provider: LLMProvider = Probed()

    assert isinstance(provider, LLMProvider)


async def test_a_model_that_calls_the_tool_passes() -> None:
    provider = Probed(calls_the_clock(), Delta(finish_reason="tool_calls"))

    result = await probe_tool_support(provider, "m", question=QUESTION)

    assert result.ok is True
    assert result.reason is None


async def test_a_model_that_answers_in_prose_fails_with_the_reason_written_down() -> None:
    """The silent failure of section 3.2, made loud: nothing went wrong on
    the wire, the model simply never reached for the tool."""
    provider = Probed(Delta(text="It is about three o'clock."), Delta(finish_reason="stop"))

    result = await probe_tool_support(provider, "m", question=QUESTION)

    assert result.ok is False
    assert result.reason == NO_TOOL_CALL


async def test_a_call_to_some_other_tool_is_not_a_pass() -> None:
    """A model that invents a tool it was not offered has not shown it can
    call the one it was."""
    provider = Probed(Delta(tool_call=ToolCall(id="c1", name="search_web", arguments={"q": "t"})))

    result = await probe_tool_support(provider, "m", question=QUESTION)

    assert result.ok is False


async def test_a_call_beside_some_prose_still_passes() -> None:
    provider = Probed(Delta(text="Let me check."), calls_the_clock())

    assert (await probe_tool_support(provider, "m", question=QUESTION)).ok is True


async def test_the_time_to_the_first_token_is_measured() -> None:
    provider = Probed(Delta(text="hi"))

    result = await probe_tool_support(provider, "m", question=QUESTION)

    assert result.first_token_ms is not None
    assert result.first_token_ms > 0


async def test_a_model_that_said_nothing_at_all_has_no_first_token() -> None:
    """A finish reason alone is not a token; a made-up number would be read
    as a measurement."""
    provider = Probed(Delta(finish_reason="stop"))

    result = await probe_tool_support(provider, "m", question=QUESTION)

    assert result.ok is False
    assert result.first_token_ms is None


async def test_the_model_is_sent_the_question_and_the_one_canonical_tool() -> None:
    """Exactly the request of section 3.2: the caller's question, the
    canonical clock tool, and a small ceiling on the answer."""
    provider = Probed(calls_the_clock())

    await probe_tool_support(provider, "the-model", question="Wie spät ist es in Istanbul?")

    [sent] = provider.asked
    assert sent["model"] == "the-model"
    assert [(m.role, m.content) for m in sent["messages"]] == [
        ("user", "Wie spät ist es in Istanbul?")
    ]
    assert sent["tools"] == [CANONICAL_TOOL_TEST]
    assert sent["max_tokens"] == MAX_TOKENS


async def test_the_stream_is_read_to_its_end() -> None:
    """Leaving the moment the call shows up would abandon the generator, and
    with it whatever connection the adapter holds open."""
    provider = Probed(calls_the_clock(), Delta(text=" done"), Delta(finish_reason="stop"))

    await probe_tool_support(provider, "m", question=QUESTION)

    assert provider.read == 3


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
    """The point of the table: the wizard writes, a later `live-assistant run` reads."""
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
