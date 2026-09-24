"""`web/search.py` (D29, 23 Sep 2026): Gemini's own Google search behind
one question at a time, the sentences a refusal becomes, and the searches
kept for the screen.

Nothing here reaches Google: the client is a fake that answers with the
SDK's own response types and remembers what it was asked. The real thing
was measured in S0 (`docs/worklog.md`, 2026-09-23).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from google.genai import errors, types
from loguru import logger

from allie.web import search
from allie.web.search import (
    BUSY,
    LOOK_UP_MODEL,
    MAX_ANSWER_CHARS,
    MAX_SOURCES,
    MODEL_GONE,
    NO_ANSWER,
    NOTHING_FOUND,
    QUOTA_USED,
    Found,
    GroundedSearch,
    Searched,
    SearchError,
)

KEY = "AIza-not-a-real-key"
# The two free-tier limits of gemini-2.5-flash, as Google names them in a
# 429 (measured in S0 and the owner's first day: five a minute, 20 a day).
PER_MINUTE = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
PER_DAY = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def grounded(
    text: str,
    *,
    queries: tuple[str, ...] = ("BIST 100 bugün",),
    titles: tuple[str, ...] = ("bloomberght.com", "borsaistanbul.com"),
) -> types.GenerateContentResponse:
    """An answer as the SDK delivers one after a search."""
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text=text)]),
                grounding_metadata=types.GroundingMetadata(
                    web_search_queries=list(queries),
                    grounding_chunks=[
                        types.GroundingChunk(
                            web=types.GroundingChunkWeb(
                                title=title,
                                uri=f"https://vertexaisearch.cloud.google.com/redirect/{index}",
                            )
                        )
                        for index, title in enumerate(titles)
                    ],
                ),
            )
        ]
    )


def refusal(
    code: int, message: str, status: str, quota: str = "", limit: str = "5"
) -> errors.APIError:
    """A refusal as Google words one; `quota` is the `quotaId` a 429 names
    and `limit` its `quotaValue` (none when empty)."""
    kind = errors.ClientError if code < 500 else errors.ServerError
    error: dict[str, Any] = {"code": code, "message": message, "status": status}
    if quota:
        violation = {"quotaId": quota, "quotaValue": limit} if limit else {"quotaId": quota}
        error["details"] = [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [violation],
            }
        ]
    return kind(code, {"error": error})


class FakeModels:
    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []
        self.answer: Any = grounded("BIST 100 bugün 13.337 puanda.")
        self.refusal: BaseException | None = None
        self.delay = 0.0

    async def generate_content(self, *, model: str, contents: Any, config: Any) -> Any:
        self.asked.append({"model": model, "contents": contents, "config": config})
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.refusal is not None:
            raise self.refusal
        return self.answer


class FakeClient:
    def __init__(self) -> None:
        self.models = FakeModels()

    @property
    def aio(self) -> FakeClient:
        return self


@pytest.fixture
def client() -> FakeClient:
    return FakeClient()


def searcher(client: FakeClient, **rest: Any) -> GroundedSearch:
    return GroundedSearch(KEY, client=client, **rest)


def logged(level: str) -> tuple[list[str], int]:
    lines: list[str] = []
    return lines, logger.add(lambda message: lines.append(str(message)), level=level)


# --------------------------------------------------------------------------
# What comes back
# --------------------------------------------------------------------------


async def test_the_answer_the_searches_and_the_sources_come_back(client: FakeClient) -> None:
    found = await searcher(client).ask("BIST kaç?")

    assert found == Found(
        answer="BIST 100 bugün 13.337 puanda.",
        queries=("BIST 100 bugün",),
        sources=("bloomberght.com", "borsaistanbul.com"),
    )


async def test_the_model_is_asked_with_google_s_search(client: FakeClient) -> None:
    await searcher(client).ask("BIST kaç?")

    [asked] = client.models.asked
    assert asked["model"] == LOOK_UP_MODEL == "gemini-2.5-flash"
    assert asked["contents"] == "BIST kaç?"
    [offered] = asked["config"].tools
    assert offered.google_search is not None
    # The SDK's own function calling is off: nothing of ours for it to run,
    # and on it warns at every call (S0).
    assert asked["config"].automatic_function_calling.disable is True


async def test_the_model_is_the_one_the_settings_name(client: FakeClient) -> None:
    looking = searcher(client, model="gemini-3.8-flash")

    await looking.ask("x")

    assert looking.model == "gemini-3.8-flash"
    assert client.models.asked[0]["model"] == "gemini-3.8-flash"


async def test_the_model_thinks_as_the_measurement_decided(
    client: FakeClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S0: median 2.5 s without thinking, 4.2 s with it, answers as right."""
    assert search.THINKING_BUDGET == 0

    monkeypatch.setattr(search, "THINKING_BUDGET", None)
    await searcher(client).ask("x")
    assert client.models.asked[-1]["config"].thinking_config is None

    monkeypatch.setattr(search, "THINKING_BUDGET", 0)
    await searcher(client).ask("x")
    assert client.models.asked[-1]["config"].thinking_config.thinking_budget == 0


async def test_each_source_is_named_once_and_five_at_most(client: FakeClient) -> None:
    client.models.answer = grounded(
        "x", titles=("a.com", "b.com", "a.com", "c.com", "d.com", "e.com", "f.com")
    )

    found = await searcher(client).ask("x")

    assert found.sources == ("a.com", "b.com", "c.com", "d.com", "e.com")
    assert len(found.sources) == MAX_SOURCES


async def test_a_long_answer_is_cut(client: FakeClient) -> None:
    client.models.answer = grounded("a" * (MAX_ANSWER_CHARS + 500))

    found = await searcher(client).ask("x")

    assert len(found.answer) == MAX_ANSWER_CHARS


async def test_an_answer_the_model_gave_without_searching_has_no_searches(
    client: FakeClient,
) -> None:
    client.models.answer = types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part(text="Bu.")]))
        ]
    )

    found = await searcher(client).ask("x")

    assert found == Found(answer="Bu.", queries=(), sources=())


# --------------------------------------------------------------------------
# What a refusal becomes
# --------------------------------------------------------------------------


async def test_an_empty_answer_is_nothing_found(client: FakeClient) -> None:
    client.models.answer = grounded("   ")

    with pytest.raises(SearchError) as caught:
        await searcher(client).ask("x")

    assert str(caught.value) == NOTHING_FOUND


async def test_five_searches_a_minute_is_one_sentence(client: FakeClient) -> None:
    """S0: the sixth call inside a minute is refused; a few seconds later it
    is not. Not the day's quota, and not said as if it were."""
    client.models.refusal = refusal(
        429, "You exceeded your current quota", "RESOURCE_EXHAUSTED", PER_MINUTE
    )

    with pytest.raises(SearchError) as caught:
        await searcher(client).ask("x")

    assert str(caught.value) == BUSY


async def test_the_day_s_quota_is_another(client: FakeClient) -> None:
    """Said with the number Google names in the refusal, not one written
    here: the sentence once said 500 when the day was 20 (2026-09-23)."""
    client.models.refusal = refusal(
        429, "You exceeded your current quota", "RESOURCE_EXHAUSTED", PER_DAY, limit="20"
    )

    with pytest.raises(SearchError) as caught:
        await searcher(client).ask("x")

    assert str(caught.value) == f"{QUOTA_USED} Google allows 20 a day."


async def test_a_day_s_quota_that_names_no_number_says_none(client: FakeClient) -> None:
    client.models.refusal = refusal(
        429, "You exceeded your current quota", "RESOURCE_EXHAUSTED", PER_DAY, limit=""
    )

    with pytest.raises(SearchError) as caught:
        await searcher(client).ask("x")

    assert str(caught.value) == QUOTA_USED


async def test_a_retired_model_names_the_line_to_change(client: FakeClient) -> None:
    """The day Google retires 2.5 (D29): the model hears it, and the log
    says which line of `config.toml` to change."""
    client.models.refusal = refusal(404, "models/gemini-2.5-flash is not found", "NOT_FOUND")
    lines, sink = logged("WARNING")
    try:
        with pytest.raises(SearchError) as caught:
            await searcher(client).ask("x")
    finally:
        logger.remove(sink)

    assert str(caught.value) == MODEL_GONE.format(model=LOOK_UP_MODEL)
    assert "look_up_model" in str(caught.value)
    assert any("look_up_model" in line for line in lines)


async def test_any_other_refusal_says_what_google_said(client: FakeClient) -> None:
    client.models.refusal = refusal(500, "internal", "INTERNAL")

    with pytest.raises(SearchError, match="500 INTERNAL"):
        await searcher(client).ask("x")


async def test_a_search_that_does_not_answer_in_time_is_said(client: FakeClient) -> None:
    client.models.delay = 1.0

    with pytest.raises(SearchError) as caught:
        await searcher(client, seconds=0.05).ask("x")

    assert str(caught.value) == NO_ANSWER.format(seconds=0.05)


async def test_no_network_is_said(client: FakeClient) -> None:
    client.models.refusal = httpx.ConnectError("no route")

    with pytest.raises(SearchError, match="ConnectError"):
        await searcher(client).ask("x")


async def test_the_key_never_reaches_the_log(client: FakeClient) -> None:
    client.models.refusal = refusal(429, "quota", "RESOURCE_EXHAUSTED")
    lines, sink = logged("DEBUG")
    try:
        with pytest.raises(SearchError):
            await searcher(client).ask("x")
    finally:
        logger.remove(sink)

    assert lines
    assert all(KEY not in line for line in lines)


async def test_every_search_is_one_line_in_the_log(client: FakeClient) -> None:
    lines, sink = logged("INFO")
    try:
        await searcher(client).ask("x")
    finally:
        logger.remove(sink)

    assert any(
        "looked up" in line and "BIST 100 bugün" in line and "bloomberght.com" in line
        for line in lines
    )


# --------------------------------------------------------------------------
# What is kept for the screen
# --------------------------------------------------------------------------


async def test_the_searches_are_kept_for_the_screen(client: FakeClient) -> None:
    kept = Searched()

    await searcher(client, searched=kept).ask("x")

    assert kept.take() == ("BIST 100 bugün",)
    assert kept.take() == ()


async def test_a_failed_search_keeps_nothing(client: FakeClient) -> None:
    kept = Searched()
    client.models.refusal = refusal(429, "quota", "RESOURCE_EXHAUSTED")

    with pytest.raises(SearchError):
        await searcher(client, searched=kept).ask("x")

    assert kept.take() == ()


def test_searched_keeps_the_order_and_skips_blanks() -> None:
    kept = Searched()
    kept.add(["a", " ", "b"])
    kept.add(["c"])

    assert kept.take() == ("a", "b", "c")
