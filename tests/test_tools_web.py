"""`search_web` (15 Sep 2026): the user's words, URL-encoded, into the engine
they chose, opened the way `open_url` opens an address.

`shell.browse` is replaced, as in `test_tools_system.py`: nothing here opens
a browser, and the thread it would have been opened from is recorded.
"""

from __future__ import annotations

import threading

import pytest
from loguru import logger

from allie import shell
from allie.tools.registry import Tool
from allie.tools.web import (
    LOOK_UP_PROMPT,
    NO_QUESTION,
    NO_WORDS,
    OFFER_BROWSER,
    SEARCH_URL,
    look_up_for,
    search_web_for,
)
from allie.web.search import QUOTA_USED, Found, SearchError


class Opened:
    def __init__(self) -> None:
        self.addresses: list[str] = []
        self.threads: list[threading.Thread] = []
        self.browser_works = True

    def browse(self, address: str) -> bool:
        self.addresses.append(address)
        self.threads.append(threading.current_thread())
        return self.browser_works


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Opened:
    seen = Opened()
    monkeypatch.setattr(shell, "browse", seen.browse)
    return seen


@pytest.fixture
def search_web() -> Tool:
    return search_web_for()


def test_it_is_a_safe_tool_that_needs_the_words(search_web: Tool) -> None:
    assert search_web.risk == "safe"
    assert search_web.spec.name == "search_web"
    assert search_web.spec.parameters["required"] == ["query"]


async def test_the_words_go_into_the_address_encoded_and_otherwise_untouched(
    search_web: Tool, opened: Opened
) -> None:
    said = await search_web.run(query="Python öğren & C#")

    assert opened.addresses == ["https://www.google.com/search?q=Python+%C3%B6%C4%9Fren+%26+C%23"]
    assert said == "Opened a web search for 'Python öğren & C#'."


async def test_the_engine_is_whatever_the_settings_say(opened: Opened) -> None:
    """Section 10: the engine is configuration. DuckDuckGo here, no code changed."""
    tool = search_web_for("https://duckduckgo.com/?q={query}")

    await tool.run(query="hava durumu")

    assert opened.addresses == ["https://duckduckgo.com/?q=hava+durumu"]


def test_an_address_with_nowhere_to_put_the_words_falls_back_to_the_default() -> None:
    """A typo in `config.toml` is logged and searching keeps working, the
    way a mistyped `[media] default_service` is handled."""
    warned: list[str] = []
    sink = logger.add(lambda message: warned.append(str(message)), level="WARNING")
    try:
        tool = search_web_for("https://example.com/search")
    finally:
        logger.remove(sink)

    assert tool.spec.name == "search_web"
    assert len(warned) == 1 and "{query}" in warned[0] and SEARCH_URL in warned[0]


async def test_a_typo_in_the_address_still_opens_the_default_engine(opened: Opened) -> None:
    tool = search_web_for("https://example.com/search")

    await tool.run(query="x")

    assert opened.addresses == ["https://www.google.com/search?q=x"]


async def test_empty_words_open_nothing(search_web: Tool, opened: Opened) -> None:
    assert await search_web.run(query="   ") == NO_WORDS
    assert opened.addresses == []


async def test_the_words_are_tidied_of_whitespace_only(search_web: Tool, opened: Opened) -> None:
    await search_web.run(query="  iki   kelime \n")

    assert opened.addresses == ["https://www.google.com/search?q=iki+kelime"]


async def test_a_browser_that_will_not_open_is_an_error_the_gate_reports(
    search_web: Tool, opened: Opened
) -> None:
    opened.browser_works = False

    with pytest.raises(RuntimeError, match="no browser would open"):
        await search_web.run(query="x")


async def test_the_browser_is_opened_off_the_event_loop(search_web: Tool, opened: Opened) -> None:
    await search_web.run(query="x")

    assert opened.threads[0] is not threading.main_thread()


class FakeSearch:
    """Stands in for `GroundedSearch`: what it was asked, and what it answers."""

    def __init__(self, found: Found | None = None, failure: SearchError | None = None) -> None:
        self.found = found or Found(
            answer="BIST 100 bugün 13.337 puanda.",
            queries=("BIST 100 bugün",),
            sources=("bloomberght.com",),
        )
        self.failure = failure
        self.asked: list[str] = []

    async def ask(self, prompt: str) -> Found:
        self.asked.append(prompt)
        if self.failure is not None:
            raise self.failure
        return self.found


def test_look_up_is_a_safe_tool_that_needs_a_question() -> None:
    tool = look_up_for(FakeSearch())

    assert tool.risk == "safe"
    assert tool.spec.name == "look_up"
    assert tool.spec.parameters["required"] == ["question"]


async def test_the_answer_comes_back_marked_as_content_with_its_sources() -> None:
    said = await look_up_for(FakeSearch()).run(question="BIST 100 şu an kaç?")

    assert said == (
        '<untrusted source="search" searched="BIST 100 bugün">\n'
        "BIST 100 bugün 13.337 puanda.\n"
        "Sources: bloomberght.com\n"
        "</untrusted>"
    )


async def test_the_question_reaches_the_search_inside_the_prompt() -> None:
    fake = FakeSearch()

    await look_up_for(fake).run(question="  BIST   kaç? ")

    assert fake.asked == [LOOK_UP_PROMPT.format(question="BIST kaç?")]


async def test_an_empty_question_searches_nothing() -> None:
    fake = FakeSearch()

    assert await look_up_for(fake).run(question="  ") == NO_QUESTION
    assert fake.asked == []


async def test_a_failed_search_is_said_and_the_browser_offered_not_opened(
    opened: Opened,
) -> None:
    said = await look_up_for(FakeSearch(failure=SearchError(QUOTA_USED))).run(question="x")

    assert said == f"{QUOTA_USED} {OFFER_BROWSER}"
    assert opened.addresses == []


async def test_an_answer_without_searches_or_sources_is_still_marked() -> None:
    fake = FakeSearch(Found(answer="Bilmiyorum.", queries=(), sources=()))

    said = await look_up_for(fake).run(question="x")

    assert said == '<untrusted source="search">\nBilmiyorum.\n</untrusted>'


def test_search_web_is_the_browser_only_when_the_user_asks_for_it() -> None:
    """D29: a question is answered by `look_up`; `search_web` shows a search."""
    description = search_web_for().spec.description

    assert "only when" in description
    assert "look_up" in description
