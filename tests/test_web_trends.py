"""`web/trends.py` (D30, 23 Sep 2026): trends24.in's latest hour for a
country or the world, and the English names the model may give a country.

Nothing here reaches the network: a small page written here stands for the
real one's shape, the two trimmed copies saved in S0 prove the real shape
still reads, and the fetch runs over an `httpx.MockTransport`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from loguru import logger

from allie.web.trends import (
    CHANGED,
    COUNTRIES,
    TRENDS_READ,
    WORLDWIDE,
    Place,
    TrendsError,
    TrendsPage,
    place_of,
    read_trends,
)

PAGES = Path(__file__).parent / "pages"
STAMP = "1790159613.836"  # Wed Sep 23 2026 10:33:33 UTC


def item(name: str, count: str = "") -> str:
    return (
        f'<li><span class=trend-name><a href="https://twitter.com/search?q=x" '
        f"class=trend-link>{name}</a><span class=tweet-count data-count={count!r}>"
        "</span></span></li>"
    )


def page(first: list[str], *, earlier: tuple[str, ...] = ("Önceki Saat",)) -> str:
    """The real page's shape (2026-09-23): a stats article the parser must
    not read, then the hours, newest first."""
    newest = "".join(item(name) for name in first)
    older = "".join(item(name) for name in earlier)
    return (
        "<html><head><title>Turkey</title></head><body><main>"
        "<article class=trend-stats-container><p>Eski Başlık</p></article>"
        '<div id=timeline-container><div class="px-2 flex">'
        f"<div class=list-container><h3 class=title data-timestamp={STAMP}>Wed</h3>"
        f"<ol class=trend-card__list>{newest}</ol></div>"
        "<div class=list-container><h3 class=title data-timestamp=1790156013.0>Wed</h3>"
        f"<ol class=trend-card__list>{older}</ol></div>"
        "</div></div></main></body></html>"
    )


TWELVE = [f"Konu {number}" for number in range(1, 13)]


# --------------------------------------------------------------------------
# The page
# --------------------------------------------------------------------------


def test_the_latest_hour_is_the_first_list_and_ten_of_it_are_read() -> None:
    trends = read_trends("Turkey", page(TWELVE))

    assert trends.place == "Turkey"
    assert trends.names == tuple(TWELVE[:TRENDS_READ])
    assert "Önceki Saat" not in trends.names
    assert "Eski Başlık" not in trends.names


def test_the_hour_is_the_list_s_timestamp() -> None:
    trends = read_trends("Turkey", page(TWELVE))

    assert trends.at is not None
    assert trends.at.replace(microsecond=0) == datetime(2026, 9, 23, 10, 33, 33, tzinfo=UTC)


def test_a_count_is_a_number_where_the_page_gives_one() -> None:
    html = page([]).replace(
        "<ol class=trend-card__list></ol>",
        f"<ol class=trend-card__list>{item('A', '12500')}{item('B')}</ol>",
        1,
    )

    assert read_trends("Turkey", html).counts == (12500, None)


def test_names_are_tidied_of_whitespace() -> None:
    assert read_trends("Turkey", page(["  Ada   Bi Kere\n Versen "])).names == (
        "Ada Bi Kere Versen",
    )


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><main><p>Maintenance</p></main></body></html>",
        page([]),
    ],
    ids=["no list", "an empty list"],
)
def test_a_page_without_the_list_is_said_to_have_changed(html: str) -> None:
    warned: list[str] = []
    sink = logger.add(lambda message: warned.append(str(message)), level="WARNING")
    try:
        with pytest.raises(TrendsError) as caught:
            read_trends("Turkey", html)
    finally:
        logger.remove(sink)

    assert str(caught.value) == CHANGED
    assert len(warned) == 1


@pytest.mark.parametrize("name", ["turkey", "worldwide"])
def test_the_real_pages_saved_on_2026_09_23_still_read(name: str) -> None:
    trends = read_trends(name, (PAGES / f"trends24-{name}.html").read_bytes())

    assert len(trends.names) == TRENDS_READ
    assert all(trends.names)
    assert trends.at is not None
    assert "Longest trending: stats" not in trends.names


# --------------------------------------------------------------------------
# The place
# --------------------------------------------------------------------------


def test_fifty_three_countries_as_trends24_lists_them() -> None:
    assert len(COUNTRIES) == 53
    assert len(set(COUNTRIES)) == 53


@pytest.mark.parametrize(
    ("said", "expected"),
    [
        ("Turkey", Place("Turkey", "https://trends24.in/turkey/")),
        ("Türkiye", Place("Turkey", "https://trends24.in/turkey/")),
        ("united kingdom", Place("United Kingdom", "https://trends24.in/united-kingdom/")),
        ("UK", Place("United Kingdom", "https://trends24.in/united-kingdom/")),
        ("England", Place("United Kingdom", "https://trends24.in/united-kingdom/")),
        ("USA", Place("United States", "https://trends24.in/united-states/")),
        ("America", Place("United States", "https://trends24.in/united-states/")),
        ("united-states", Place("United States", "https://trends24.in/united-states/")),
        ("  South   Korea ", Place("Korea", "https://trends24.in/korea/")),
        ("New Zealand", Place("New Zealand", "https://trends24.in/new-zealand/")),
        ("worldwide", Place(WORLDWIDE, "https://trends24.in/")),
        ("World", Place(WORLDWIDE, "https://trends24.in/")),
    ],
)
def test_a_country_is_found_by_any_of_its_english_names(said: str, expected: Place) -> None:
    assert place_of(said) == expected


@pytest.mark.parametrize("said", ["Narnia", "", "   ", "Istanbul"])
def test_a_place_with_no_list_is_none(said: str) -> None:
    assert place_of(said) is None


# --------------------------------------------------------------------------
# The fetch
# --------------------------------------------------------------------------


def over(answer: Callable[[httpx.Request], httpx.Response]) -> tuple[TrendsPage, list[str]]:
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(f"{request.url} {request.headers['User-Agent'][:11]}")
        return answer(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TrendsPage(client=client), asked


async def test_the_country_s_page_is_asked_for_as_a_browser_would() -> None:
    reader, asked = over(lambda _: httpx.Response(200, content=page(TWELVE).encode("utf-8")))

    trends = await reader.latest(Place("Turkey", "https://trends24.in/turkey/"))

    assert asked == ["https://trends24.in/turkey/ Mozilla/5.0"]
    assert trends.names[0] == "Konu 1"


def refused(_: httpx.Request) -> httpx.Response:
    return httpx.Response(503)


def unreachable(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("no route", request=request)


def silent(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("no answer", request=request)


@pytest.mark.parametrize(
    ("answer", "reason"),
    [(refused, "HTTP 503"), (unreachable, "ConnectError"), (silent, "no answer")],
    ids=["refused", "unreachable", "silent"],
)
async def test_a_page_that_cannot_be_had_is_said(
    answer: Callable[[httpx.Request], httpx.Response], reason: str
) -> None:
    reader, _ = over(answer)

    with pytest.raises(TrendsError, match=reason):
        await reader.latest(Place("Turkey", "https://trends24.in/turkey/"))


async def test_a_borrowed_client_is_left_open() -> None:
    borrowed = httpx.AsyncClient(transport=httpx.MockTransport(refused))

    await TrendsPage(client=borrowed).aclose()

    assert not borrowed.is_closed
    await borrowed.aclose()
