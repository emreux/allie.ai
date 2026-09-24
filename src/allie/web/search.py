"""Looking something up on the web through Gemini's own Google search
(plan.md D29, 23 September 2026).

**Why a second model.** The live model searches for itself when the key
allows it (`[live] web_search`, D22), and the free-tier key does not: Google
gives the 3.x family no search on the free tier (measured 2026-09-21 and
2026-09-22 - Live close 1011, HTTP 429 in 0.3 s). The 2.5 family still
searches there - five requests a minute and 20 a day, as Google's own
refusal names them (measured 2026-09-23) -
and `gemini-2.5-flash` answers in a median 2.5 s with its thinking off. So
a tool asks that model and the live model says the answer. Which model is the user's `[web]
look_up_model`: when Google retires 2.5, that line changes, and the day it
happens the refusal says so.

**Google's SDK is not imported at import time.** `config.py` imports
`tools/web.py`, which imports this file, and every command of the CLI reads
the configuration; `allie --version` should not pay for the SDK. The
client, its types and its errors are imported where they are used.

**What was searched is kept for the screen.** Google's terms ask that the
searches behind a grounded answer be shown with it. `Searched` collects
them while a turn runs; the state machine takes them when the turn ends
(`Turn.searched`), and the window shows them between the user's words and
the answer.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import httpx
from loguru import logger

if TYPE_CHECKING:
    from google.genai import errors, types

__all__ = [
    "BUSY",
    "DAY_LIMIT",
    "LOOK_UP_MODEL",
    "MAX_ANSWER_CHARS",
    "MAX_SOURCES",
    "MODEL_GONE",
    "NOTHING_FOUND",
    "NO_ANSWER",
    "QUOTA_USED",
    "SEARCH_SECONDS",
    "THINKING_BUDGET",
    "UNREACHABLE",
    "Found",
    "GroundedSearch",
    "SearchError",
    "Searched",
    "Searcher",
]

# The one family the free tier may search with (D22, D29).
LOOK_UP_MODEL = "gemini-2.5-flash"

# Four times the slowest answer measured (3.7 s), short enough that a search
# which is not answering does not hold a spoken turn open. One attempt: a
# refusal is said at once rather than retried into silence.
SEARCH_SECONDS = 15.0

# How much of an answer reaches the live model, whose context every turn
# reads again. Four sentences are a few hundred characters.
MAX_ANSWER_CHARS = 1_500
MAX_SOURCES = 5

# The model's thinking before it answers: `None` leaves it to the model, `0`
# switches it off. Off, measured 2026-09-23 on five questions: median 2.5 s
# against 4.2 s (and 5.3 s at a budget of 512), the answers as right - the
# BIST value varied with what the search found, not with the thinking.
THINKING_BUDGET: int | None = 0

# How much of Google's own words a refusal carries into the log.
DESCRIBED_CHARS = 200

# The answers addressed to the model. A 429 is one of two limits, told
# apart by the `quotaId` Google names in it: five a minute, or 20 a day. The
# day's number is the `quotaValue` beside it, never one written here - the
# free tier's has changed before (the sentence once said 500).
BUSY = (
    "Too many web searches in the last minute: Google allows five a minute on the free "
    "tier, and asked again in a few seconds it will answer."
)
QUOTA_USED = "Today's free web searches are used up."
DAY_LIMIT = "Google allows {limit} a day."
MODEL_GONE = (
    "Google no longer offers {model!r} for searching; the look_up_model line under [web] "
    "in config.toml names a model that has to be replaced."
)
NO_ANSWER = "The search did not answer within {seconds:.0f} seconds."
UNREACHABLE = "The search could not be reached ({reason})."
NOTHING_FOUND = "The search came back empty."


class SearchError(Exception):
    """A search that could not answer, worded so that a tool can pass it on."""


@dataclass(frozen=True, slots=True)
class Found:
    """What a search came to: the answer, what Google was asked, and the
    titles of the pages the answer stood on."""

    answer: str
    queries: tuple[str, ...]
    sources: tuple[str, ...]


class Searcher(Protocol):
    """What the tools ask: `GroundedSearch` in life, a fake in a test."""

    async def ask(self, prompt: str) -> Found: ...


class Searched:
    """The searches of the turn under way, for the screen. One loop, one
    turn at a time: no lock."""

    def __init__(self) -> None:
        self._queries: list[str] = []

    def add(self, queries: Iterable[str]) -> None:
        self._queries.extend(query for query in queries if query.strip())

    def take(self) -> tuple[str, ...]:
        """Everything kept since the last take, and nothing after it."""
        taken, self._queries = tuple(self._queries), []
        return taken


class GroundedSearch:
    """One model that searches Google before it answers, asked one
    question at a time."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = LOOK_UP_MODEL,
        seconds: float = SEARCH_SECONDS,
        searched: Searched | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._seconds = seconds
        self._searched = searched
        if client is None:
            from google import genai
            from google.genai import types as sdk

            # No retry options: the SDK then makes one attempt (as in
            # `stt/gemini_stt.py`). A second longer than our own deadline,
            # so that ours is the one that says it.
            client = genai.Client(
                api_key=api_key,
                http_options=sdk.HttpOptions(timeout=int((seconds + 1) * 1000)),
            )
        self._client = client

    async def ask(self, prompt: str) -> Found:
        """What the model found for `prompt`, having searched. Raises
        `SearchError` in words a tool can pass on."""
        from google.genai import errors as sdk_errors

        try:
            async with asyncio.timeout(self._seconds):
                response = await self._client.aio.models.generate_content(
                    model=self.model, contents=prompt, config=_config()
                )
        except TimeoutError as failure:
            logger.warning("look-up: no answer within {:.0f} s", self._seconds)
            raise SearchError(NO_ANSWER.format(seconds=self._seconds)) from failure
        except sdk_errors.APIError as failure:
            raise _refused(failure, self.model) from failure
        except (httpx.HTTPError, OSError) as failure:
            logger.warning("look-up: {}", type(failure).__name__)
            raise SearchError(UNREACHABLE.format(reason=type(failure).__name__)) from failure

        found = _found(response)
        if not found.answer:
            raise SearchError(NOTHING_FOUND)
        if self._searched is not None:
            self._searched.add(found.queries)
        logger.info(
            "looked up: searched {queries}; read {sources}",
            queries=list(found.queries),
            sources=list(found.sources),
        )
        return found


def _config() -> types.GenerateContentConfig:
    from google.genai import types as sdk

    return sdk.GenerateContentConfig(
        tools=[sdk.Tool(google_search=sdk.GoogleSearch())],
        # The SDK's own function calling runs Python functions it was
        # handed; there are none, and left on it warns at every call (S0).
        automatic_function_calling=sdk.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=None
        if THINKING_BUDGET is None
        else sdk.ThinkingConfig(thinking_budget=THINKING_BUDGET),
    )


def _refused(failure: errors.APIError, model: str) -> SearchError:
    """A refusal in the model's words, and in the log with Google's."""
    if failure.code == 404:
        sentence = MODEL_GONE.format(model=model)
        logger.warning("look-up: {}", sentence)
        return SearchError(sentence)
    described = " ".join((failure.message or "").split())[:DESCRIBED_CHARS]
    logger.warning("look-up refused: {} {}", failure.code, described)
    if failure.code == 429:
        # The limit is named in the body's details: `...PerDay...` or
        # `...PerMinute...` (S0). Anything else is taken for the minute's,
        # the one that passes by itself.
        if "PerDay" not in str(failure.details):
            return SearchError(BUSY)
        limit = _day_limit(failure.details)
        return SearchError(f"{QUOTA_USED} {DAY_LIMIT.format(limit=limit)}" if limit else QUOTA_USED)
    return SearchError(UNREACHABLE.format(reason=f"{failure.code} {failure.status or ''}".strip()))


def _day_limit(details: Any) -> str:
    """The day's number a refusal names - the `quotaValue` of its
    `...PerDay...` violation - or "" when it names none."""
    error = details.get("error") if isinstance(details, dict) else None
    items = error.get("details") if isinstance(error, dict) else None
    for item in items if isinstance(items, list) else []:
        violations = item.get("violations") if isinstance(item, dict) else None
        for violation in violations if isinstance(violations, list) else []:
            if isinstance(violation, dict) and "PerDay" in str(violation.get("quotaId", "")):
                return str(violation.get("quotaValue", ""))
    return ""


def _found(response: Any) -> Found:
    """The answer, the searches and the source titles of one response."""
    answer = (response.text or "").strip()[:MAX_ANSWER_CHARS].rstrip()
    candidates = response.candidates or []
    grounding = candidates[0].grounding_metadata if candidates else None
    if grounding is None:
        return Found(answer=answer, queries=(), sources=())
    queries = tuple(
        " ".join(query.split()) for query in grounding.web_search_queries or () if query.strip()
    )
    titles: list[str] = []
    for chunk in grounding.grounding_chunks or ():
        title = chunk.web.title if chunk.web is not None else None
        if title and title not in titles:
            titles.append(title)
    return Found(answer=answer, queries=queries, sources=tuple(titles[:MAX_SOURCES]))
