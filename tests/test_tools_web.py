"""`open_web` (27 Sep 2026, D38 - it was `open_url` and `search_web`): an
address, or the user's words URL-encoded into the engine they chose, opened
in their browser; and `look_up`.

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
    BOTH,
    LOOK_UP_PROMPT,
    NO_QUESTION,
    NOTHING_TO_OPEN,
    OFFER_BROWSER,
    SEARCH_URL,
    look_up_for,
    open_web_for,
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
def open_web() -> Tool:
    return open_web_for()


def test_it_is_a_safe_tool_that_needs_nothing_but_one_of_two(open_web: Tool) -> None:
    assert open_web.risk == "safe"
    assert open_web.spec.name == "open_web"
    assert open_web.spec.parameters["required"] == []
    assert set(open_web.spec.parameters["properties"]) == {"url", "search"}


async def test_an_address_opens_with_https_added(open_web: Tool, opened: Opened) -> None:
    said = await open_web.run(url="  sahibinden.com ")

    assert opened.addresses == ["https://sahibinden.com"]
    assert said == "Opened https://sahibinden.com."


async def test_an_address_with_a_scheme_is_left_alone(open_web: Tool, opened: Opened) -> None:
    await open_web.run(url="http://192.168.1.1/admin")

    assert opened.addresses == ["http://192.168.1.1/admin"]


async def test_the_words_go_into_the_address_encoded_and_otherwise_untouched(
    open_web: Tool, opened: Opened
) -> None:
    said = await open_web.run(search="Python öğren & C#")

    assert opened.addresses == ["https://www.google.com/search?q=Python+%C3%B6%C4%9Fren+%26+C%23"]
    assert said == "Opened a web search for 'Python öğren & C#'."


async def test_the_engine_is_whatever_the_settings_say(opened: Opened) -> None:
    """Section 10: the engine is configuration. DuckDuckGo here, no code changed."""
    tool = open_web_for("https://duckduckgo.com/?q={query}")

    await tool.run(search="hava durumu")

    assert opened.addresses == ["https://duckduckgo.com/?q=hava+durumu"]


def test_an_engine_with_nowhere_to_put_the_words_falls_back_to_the_default() -> None:
    """A typo in `config.toml` is logged and searching keeps working, the
    way a mistyped `[media] default_service` is handled."""
    warned: list[str] = []
    sink = logger.add(lambda message: warned.append(str(message)), level="WARNING")
    try:
        tool = open_web_for("https://example.com/search")
    finally:
        logger.remove(sink)

    assert tool.spec.name == "open_web"
    assert len(warned) == 1 and "{query}" in warned[0] and SEARCH_URL in warned[0]


async def test_a_typo_in_the_engine_still_opens_the_default(opened: Opened) -> None:
    await open_web_for("https://example.com/search").run(search="x")

    assert opened.addresses == ["https://www.google.com/search?q=x"]


async def test_neither_opens_nothing(open_web: Tool, opened: Opened) -> None:
    assert await open_web.run(url="  ", search=" ") == NOTHING_TO_OPEN
    assert opened.addresses == []


async def test_both_open_nothing(open_web: Tool, opened: Opened) -> None:
    assert await open_web.run(url="a.com", search="b") == BOTH
    assert opened.addresses == []


async def test_the_words_are_tidied_of_whitespace_only(open_web: Tool, opened: Opened) -> None:
    await open_web.run(search="  iki   kelime \n")

    assert opened.addresses == ["https://www.google.com/search?q=iki+kelime"]


async def test_a_browser_that_will_not_open_is_an_error_the_gate_reports(
    open_web: Tool, opened: Opened
) -> None:
    opened.browser_works = False

    with pytest.raises(RuntimeError, match="no browser would open"):
        await open_web.run(search="x")


async def test_the_browser_is_opened_off_the_event_loop(open_web: Tool, opened: Opened) -> None:
    await open_web.run(url="a.com")

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


def test_open_web_is_the_browser_only_when_the_user_asks_for_it() -> None:
    """D29: a question is answered by `look_up`; `open_web` shows a page or a search."""
    description = open_web_for().spec.description

    assert "only when" in description
    assert "look_up" in description
    assert "play_music" in description


def test_look_up_leaves_the_quick_facts_to_their_tool() -> None:
    description = look_up_for(FakeSearch()).spec.description

    assert "facts" in description and "exchange rates" not in description
