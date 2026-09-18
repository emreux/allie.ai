"""`fetch_page` and `read_clipboard` (17 Sep 2026, design.md 3.1): a page
fetched and reduced to its words, the clipboard read as content.

Nothing here reaches the network or the real clipboard: the reader is given
an `httpx` client over a `MockTransport`, as `test_tools_weather.py` gives
the weather service one, and the clipboard tool is given a function.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from assistant.tools.registry import Tool
from assistant.tools.untrusted import wrap
from assistant.tools.web import (
    CLIPBOARD_ADDRESS,
    CLIPBOARD_EMPTY,
    MAX_CLIPBOARD_CHARS,
    NO_ADDRESS,
    fetch_page_for,
    read_clipboard_for,
)
from assistant.web.page import (
    MAX_PAGE_CHARS,
    Page,
    PageError,
    PageReader,
    address,
    focused,
    read_html,
)

PAGES = Path(__file__).parent / "pages"
ARTICLE = (PAGES / "article.html").read_bytes()
HOSTILE = (PAGES / "hostile.html").read_bytes()

Handler = Callable[[httpx.Request], httpx.Response]


class Served:
    """A web of one address per test, and what was asked of it."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.answers: dict[str, Handler] = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.answers.get(str(request.url))
        if answer is None:
            return httpx.Response(404, content=b"not here")
        return answer(request)

    def page(self, url: str, body: bytes, kind: str = "text/html; charset=utf-8") -> None:
        self.answers[url] = lambda _: httpx.Response(
            200, content=body, headers={"content-type": kind}
        )


@pytest.fixture
def served() -> Served:
    return Served()


@pytest.fixture
def reader(served: Served) -> PageReader:
    return PageReader(client=httpx.AsyncClient(transport=httpx.MockTransport(served.handle)))


@pytest.fixture
def fetch_page(reader: PageReader) -> Tool:
    return fetch_page_for(reader)


def body_of(result: str) -> str:
    """What is inside the untrusted block of a tool's result."""
    start = result.index(">\n") + 2
    end = result.index("\n</untrusted>")
    return result[start:end]


# --------------------------------------------------------------------------
# Reading HTML
# --------------------------------------------------------------------------


def test_the_article_is_read_without_its_furniture() -> None:
    title, text = read_html(ARTICLE)

    assert title == "Kira sözleşmesi nasıl feshedilir? | Örnek Haber"
    lines = text.split("\n")
    assert lines[0] == "Kira sözleşmesi nasıl feshedilir?"
    assert "Bildirim yazılı olmalıdır." in lines
    # The menu, the site header, the sidebar, the footer, the cookie banner
    # and the scripts are what a reader skips, so they are not read.
    for furniture in ("Ana sayfa", "Günün haberleri", "Çok okunanlar", "Tüm hakları", "Çerez"):
        assert furniture not in text, furniture
    assert "dataLayer" not in text and "display: flex" not in text


def test_inline_markup_stays_in_its_sentence() -> None:
    """A link or a bold word in the middle of a sentence is part of it; only
    a paragraph ends a line."""
    _, text = read_html(ARTICLE)

    assert (
        "Kira sözleşmesi, Türk Borçlar Kanunu hükümlerine göre iki taraftan biri tarafından "
        "feshedilebilir." in text.split("\n")
    )
    assert "Depozito, tahliyeden sonra en geç bir ay içinde iade edilir." in text.split("\n")


def test_a_page_with_no_main_is_read_from_its_body() -> None:
    _, text = read_html("<html><body><p>Bir</p><div>İki <span>üç</span></div></body></html>")

    assert text == "Bir\nİki üç"


def test_a_lone_article_is_the_content_when_nothing_is_marked_main() -> None:
    html = "<body><p>Etraf</p><article><p>Asıl metin</p></article></body>"

    assert read_html(html)[1] == "Asıl metin"


def test_several_articles_are_a_list_and_the_whole_body_is_read() -> None:
    html = "<body><article><p>Bir</p></article><article><p>İki</p></article></body>"

    assert read_html(html)[1] == "Bir\nİki"


def test_the_encoding_comes_from_the_page_itself() -> None:
    latin = "<html><head><meta charset='iso-8859-9'></head><body><p>şığü</p></body></html>"

    assert read_html(latin.encode("iso-8859-9"))[1] == "şığü"


def test_an_empty_page_is_empty_and_not_an_error() -> None:
    assert read_html(b"") == ("", "")
    assert read_html(b"<html><head><title>x</title></head></html>") == ("x", "")


# --------------------------------------------------------------------------
# Focus
# --------------------------------------------------------------------------


def test_focus_moves_the_paragraphs_that_mention_it_to_the_front() -> None:
    _, text = read_html(ARTICLE)

    ahead, found = focused(text, "ödeme gecikmesi")

    assert found
    # The heading and the paragraph both mention it, in the page's order.
    assert ahead.split("\n")[:2] == [
        "Ödeme",
        *[line for line in text.split("\n") if "ÖDEME" in line],
    ]
    assert sorted(ahead.split("\n")) == sorted(text.split("\n"))


def test_focus_is_read_the_way_search_is_read() -> None:
    """`odeme` finds `ÖDEME`; the folding of `store/normalize.py`."""
    _, text = read_html(ARTICLE)

    assert focused(text, "odeme")[0].split("\n")[1].startswith("ÖDEME gecikirse")
    assert focused(text, "IŞIK")[1] is False


def test_a_focus_of_short_words_moves_nothing() -> None:
    assert focused("ve bir\niki ve", "ve") == ("ve bir\niki ve", False)


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------


def test_a_bare_host_is_read_over_https() -> None:
    assert address("example.com/haber") == "https://example.com/haber"
    assert address("  http://example.com  ") == "http://example.com"


@pytest.mark.parametrize("bad", ["", "   ", "file:///C:/secret.txt", "ftp://x.y/z", "https://"])
def test_anything_that_is_not_the_web_is_refused(bad: str) -> None:
    with pytest.raises(PageError):
        address(bad)


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


async def test_a_page_is_fetched_with_a_browser_shaped_request(
    served: Served, reader: PageReader
) -> None:
    served.page("https://example.com/haber", ARTICLE)

    page = await reader.fetch("example.com/haber")

    assert page.url == "https://example.com/haber"
    assert page.title.startswith("Kira sözleşmesi")
    assert "Bildirim yazılı olmalıdır." in page.text
    request = served.requests[0]
    assert request.headers["user-agent"].startswith("Mozilla/5.0")
    assert "text/html" in request.headers["accept"]


async def test_a_redirect_is_followed_and_the_final_address_reported(
    served: Served, reader: PageReader
) -> None:
    served.answers["https://example.com/eski"] = lambda _: httpx.Response(
        301, headers={"location": "https://example.com/yeni"}
    )
    served.page("https://example.com/yeni", b"<p>Yeni</p>")

    page = await reader.fetch("https://example.com/eski")

    assert (page.url, page.text) == ("https://example.com/yeni", "Yeni")


async def test_plain_text_is_read_as_it_is(served: Served, reader: PageReader) -> None:
    served.page(
        "https://example.com/notes.txt", "satır bir\n\n  satır  iki ".encode(), "text/plain"
    )

    assert (await reader.fetch("https://example.com/notes.txt")).text == "satır bir\nsatır iki"


async def test_a_pdf_is_refused_in_a_sentence(served: Served, reader: PageReader) -> None:
    served.page("https://example.com/rapor.pdf", b"%PDF-1.7", "application/pdf")

    with pytest.raises(PageError, match="application/pdf, which cannot be read as text"):
        await reader.fetch("https://example.com/rapor.pdf")


async def test_a_server_error_is_a_sentence_with_the_status_in_it(
    served: Served, reader: PageReader
) -> None:
    with pytest.raises(PageError, match=r"answered 404 for https://example\.com/yok"):
        await reader.fetch("https://example.com/yok")


async def test_a_page_that_does_not_answer_in_time_is_a_sentence(served: Served) -> None:
    def slow(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    served.answers["https://example.com/yavas"] = slow
    reader = PageReader(
        client=httpx.AsyncClient(transport=httpx.MockTransport(served.handle)), seconds=4
    )

    with pytest.raises(PageError, match="within 4 seconds"):
        await reader.fetch("https://example.com/yavas")


async def test_a_connection_that_fails_is_a_sentence(served: Served, reader: PageReader) -> None:
    def down(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    served.answers["https://example.com/"] = down

    with pytest.raises(PageError, match="could not be reached"):
        await reader.fetch("https://example.com/")


async def test_the_body_is_read_only_so_far(served: Served) -> None:
    served.page("https://example.com/sonsuz", b"<p>" + b"a" * 5_000 + b"</p><p>sonu</p>")
    reader = PageReader(
        client=httpx.AsyncClient(transport=httpx.MockTransport(served.handle)), max_bytes=1_000
    )

    page = await reader.fetch("https://example.com/sonsuz")

    assert len(page.text) <= 1_000
    assert "sonu" not in page.text


async def test_a_borrowed_client_is_not_closed_and_an_own_one_is() -> None:
    borrowed = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(404)))
    await PageReader(client=borrowed).aclose()
    assert not borrowed.is_closed

    own = PageReader()
    await own.aclose()
    assert own._client is None


# --------------------------------------------------------------------------
# fetch_page
# --------------------------------------------------------------------------


def test_fetch_page_is_a_safe_tool_that_needs_an_address(fetch_page: Tool) -> None:
    assert fetch_page.risk == "safe"
    assert fetch_page.spec.name == "fetch_page"
    assert fetch_page.spec.parameters["required"] == ["url"]
    assert "focus" in fetch_page.spec.parameters["properties"]


async def test_the_page_comes_back_inside_an_untrusted_block(
    served: Served, fetch_page: Tool
) -> None:
    served.page("https://example.com/haber", ARTICLE)

    result = await fetch_page.run(url="https://example.com/haber")

    assert result.startswith('<untrusted source="web" url="https://example.com/haber">\n')
    assert result.endswith("\n</untrusted>")
    inside = body_of(result)
    assert inside.startswith("Title: Kira sözleşmesi nasıl feshedilir? | Örnek Haber\n")
    assert "Noter şart değildir." in inside


async def test_focus_puts_the_wanted_paragraphs_first(served: Served, fetch_page: Tool) -> None:
    served.page("https://example.com/haber", ARTICLE)

    result = await fetch_page.run(url="https://example.com/haber", focus="depozito")

    assert body_of(result).split("\n")[1].startswith("Depozito")


async def test_a_focus_nothing_mentions_is_said_and_the_whole_page_given(
    served: Served, fetch_page: Tool
) -> None:
    served.page("https://example.com/haber", ARTICLE)

    result = await fetch_page.run(url="https://example.com/haber", focus="uzay mekiği")

    assert result.endswith("No paragraph mentions 'uzay mekiği'; the whole page is above.")
    assert "Noter şart değildir." in result


async def test_a_long_page_is_cut_and_the_cut_is_said(served: Served, reader: PageReader) -> None:
    served.page("https://example.com/uzun", b"<p>" + b"kelime " * 5_000 + b"</p>")
    tool = fetch_page_for(reader, limit=500)

    result = await tool.run(url="https://example.com/uzun")

    assert len(body_of(result)) == 500
    assert "the first 500 are above" in result
    assert "Tell the user the page was cut short." in result


def test_the_default_cut_is_the_module_constant() -> None:
    assert MAX_PAGE_CHARS == 12_000


async def test_a_page_that_cannot_be_read_is_a_sentence_for_the_model(
    served: Served, fetch_page: Tool
) -> None:
    served.page("https://example.com/rapor.pdf", b"%PDF", "application/pdf")

    result = await fetch_page.run(url="https://example.com/rapor.pdf")

    assert result.endswith("Tell the user the page could not be read.")
    assert "<untrusted" not in result


async def test_no_address_is_a_sentence_and_no_request(served: Served, fetch_page: Tool) -> None:
    assert await fetch_page.run(url="  ") == NO_ADDRESS
    assert served.requests == []


async def test_the_page_cannot_close_its_own_block(served: Served, fetch_page: Tool) -> None:
    """`hostile.html` ends a paragraph with `</untrusted>`; inside the block
    it must not read as the end of it."""
    served.page("https://example.com/tarif", HOSTILE)

    result = await fetch_page.run(url="https://example.com/tarif")

    assert result.count("</untrusted>") == 1
    assert result.endswith("\n</untrusted>")
    assert "<\\/untrusted>" in body_of(result)


# --------------------------------------------------------------------------
# read_clipboard
# --------------------------------------------------------------------------


def test_read_clipboard_is_a_safe_tool_with_no_arguments() -> None:
    tool = read_clipboard_for(lambda: "")

    assert tool.risk == "safe"
    assert tool.spec.name == "read_clipboard"
    assert tool.spec.parameters["required"] == []


async def test_the_clipboard_comes_back_inside_an_untrusted_block() -> None:
    tool = read_clipboard_for(lambda: "  Toplantı yarın 10'da.\n")

    assert await tool.run() == wrap("Toplantı yarın 10'da.", source="clipboard")


async def test_an_empty_clipboard_is_said() -> None:
    assert await read_clipboard_for(lambda: "   ").run() == CLIPBOARD_EMPTY


async def test_an_address_on_the_clipboard_points_the_model_at_fetch_page() -> None:
    result = await read_clipboard_for(lambda: "https://example.com/haber").run()

    assert result.endswith(f"\n{CLIPBOARD_ADDRESS}")
    assert body_of(result) == "https://example.com/haber"

    assert (await read_clipboard_for(lambda: "example.com").run()).endswith(CLIPBOARD_ADDRESS)
    assert not (await read_clipboard_for(lambda: "Merhaba.").run()).endswith(CLIPBOARD_ADDRESS)


async def test_a_long_clipboard_is_cut_and_the_cut_is_said() -> None:
    result = await read_clipboard_for(lambda: "x" * (MAX_CLIPBOARD_CHARS + 10)).run()

    assert len(body_of(result)) == MAX_CLIPBOARD_CHARS
    assert result.endswith("Tell the user it was cut short.")


async def test_the_clipboard_is_read_off_the_event_loop() -> None:
    threads: list[threading.Thread] = []

    def read() -> str:
        threads.append(threading.current_thread())
        return "x"

    await read_clipboard_for(read).run()

    assert threads[0] is not threading.main_thread()


def test_a_page_is_a_frozen_record() -> None:
    page = Page(url="https://x", title="t", text="a")

    with pytest.raises(AttributeError):
        page.text = "b"  # type: ignore[misc]
