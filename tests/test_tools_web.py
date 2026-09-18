"""`search_web` (15 Sep 2026): the user's words, URL-encoded, into the engine
they chose, opened the way `open_url` opens an address.

`shell.browse` is replaced, as in `test_tools_system.py`: nothing here opens
a browser, and the thread it would have been opened from is recorded.
"""

from __future__ import annotations

import threading

import pytest
from loguru import logger

from assistant import shell
from assistant.tools.registry import Tool
from assistant.tools.web import NO_WORDS, SEARCH_URL, search_web_for


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
