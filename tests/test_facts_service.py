"""`Facts`: which service each kind asks, what the region changes, and how
a failure reads (D39). Over `httpx.MockTransport` with the real answers of
`tests/data/facts/`; the geocoder is a fake."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from allie.facts.fetch import FactsError, Fetcher
from allie.facts.service import (
    NO_CITY,
    NO_COIN,
    NO_SUCH_PLACE,
    NO_TEAM,
    NOT_IN_REGION,
    Facts,
    NotAnsweredError,
)
from allie.locales import Region
from allie.tools.weather import Place

DATA = Path(__file__).parent / "data" / "facts"
TR = timezone(timedelta(hours=3))
TURKEY = Region(
    currency="TRY",
    rates="tcmb",
    news="hl=tr&gl=TR&ceid=TR:tr",
    news_fallback="https://feeds.bbci.co.uk/turkce/rss.xml",
    earthquakes="afad",
    prayer_method=13,
    football_league="tur.1",
)
ISTANBUL = Place("Istanbul", "Istanbul", "Türkiye", 41.01384, 28.94966, "Europe/Istanbul")


class Places:
    def __init__(self, found: Place | None = ISTANBUL) -> None:
        self.found = found
        self.asked: list[str] = []

    async def locate(self, place: str) -> Place | None:
        self.asked.append(place)
        return self.found


class Routes:
    """One answer per host and path; everything asked is kept."""

    def __init__(self) -> None:
        self.answers: dict[str, Callable[[httpx.Request], httpx.Response]] = {}
        self.asked: list[httpx.Request] = []

    def file(self, path: str, name: str) -> None:
        content = (DATA / name).read_bytes()
        self.answers[path] = lambda request: httpx.Response(200, content=content)

    def fail(self, path: str, status: int = 503) -> None:
        self.answers[path] = lambda request: httpx.Response(status)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.asked.append(request)
        key = f"{request.url.host}{request.url.path}"
        for path, answer in self.answers.items():
            if key.startswith(path) or key == path:
                return answer(request)
        return httpx.Response(404)


@pytest.fixture
def routes() -> Routes:
    return Routes()


def facts(routes: Routes, region: Region = TURKEY, places: Places | None = None) -> Facts:
    client = httpx.AsyncClient(transport=httpx.MockTransport(routes.handle))
    return Facts(
        region,
        places or Places(),
        fetcher=Fetcher(client=client),
        clock=lambda: datetime(2026, 9, 27, 9, 0, tzinfo=UTC),
        tz=TR,
    )


async def test_rates_come_from_the_central_bank_when_the_region_says_so(routes: Routes) -> None:
    routes.file("www.tcmb.gov.tr/kurlar/today.xml", "tcmb_today.xml")

    said = await facts(routes).currency("USD")

    assert said.startswith("TCMB indicative rates of 2026-09-25")


async def test_a_silent_central_bank_falls_back_to_the_reference_rates(routes: Routes) -> None:
    routes.fail("www.tcmb.gov.tr/kurlar/today.xml")
    routes.file("api.frankfurter.dev/v1/latest", "frankfurter.json")

    said = await facts(routes).currency("")

    assert said.startswith("European Central Bank reference rates")
    asked = routes.asked[-1].url.params
    assert (asked["base"], asked["symbols"]) == ("TRY", "USD,EUR")


async def test_rates_need_a_currency_to_be_said_against(routes: Routes) -> None:
    with pytest.raises(NotAnsweredError) as refused:
        await facts(routes, Region()).currency("USD")

    assert str(refused.value) == NOT_IN_REGION.format(what="exchange rates")


async def test_a_named_coin_is_searched_then_priced(routes: Routes) -> None:
    routes.file("api.coingecko.com/api/v3/search", "coingecko_search.json")
    routes.file("api.coingecko.com/api/v3/simple/price", "coingecko_price.json")

    await facts(routes).crypto("solana")

    price = routes.asked[-1].url.params
    assert (price["ids"], price["vs_currencies"]) == ("solana", "usd,try")


async def test_a_coin_nobody_lists_is_said(routes: Routes) -> None:
    routes.answers["api.coingecko.com/api/v3/search"] = lambda request: httpx.Response(
        200, json={"coins": []}
    )

    with pytest.raises(NotAnsweredError) as refused:
        await facts(routes).crypto("zzzcoin")

    assert str(refused.value) == NO_COIN.format(coin="zzzcoin")


async def test_quakes_ask_afad_for_the_last_day(routes: Routes) -> None:
    routes.file("deprem.afad.gov.tr/apiv2/event/filter", "afad.json")

    said = await facts(routes).earthquakes("")

    assert "Saimbeyli (Adana)" in said
    assert routes.asked[0].url.params["start"] == "2026-09-26T09:00:00"


async def test_prayer_times_are_asked_by_the_places_coordinates(routes: Routes) -> None:
    routes.file("api.aladhan.com/v1/timings/27-09-2026", "aladhan.json")
    places = Places()

    said = await facts(routes, places=places).prayer_times("İstanbul")

    assert places.asked == ["İstanbul"]
    params = routes.asked[0].url.params
    assert (params["method"], params["timezonestring"]) == ("13", "Europe/Istanbul")
    assert said.startswith("Prayer times for Istanbul, Istanbul, Türkiye on 2026-09-27")


async def test_prayer_times_need_a_city(routes: Routes) -> None:
    with pytest.raises(NotAnsweredError) as refused:
        await facts(routes).prayer_times(" ")

    assert str(refused.value) == NO_CITY


async def test_a_place_the_geocoder_does_not_know_is_said(routes: Routes) -> None:
    with pytest.raises(NotAnsweredError) as refused:
        await facts(routes, places=Places(found=None)).air_quality("Atlantis")

    assert str(refused.value) == NO_SUCH_PLACE.format(place="Atlantis")


async def test_formula1_is_the_next_weekend_and_the_last_podium(routes: Routes) -> None:
    routes.file("api.jolpi.ca/ergast/f1/current/next.json", "jolpica_next.json")
    routes.file("api.jolpi.ca/ergast/f1/current/last/results.json", "jolpica_last.json")

    said = await facts(routes).formula1()

    assert said.startswith("Next: Bahrain Grand Prix in Malaysia")
    assert "\nLast: Azerbaijan Grand Prix (2026-09-26)" in said


async def test_a_team_is_its_schedule_fixtures_and_table(routes: Routes) -> None:
    routes.file(
        "site.api.espn.com/apis/site/v2/sports/soccer/tur.1/teams/432/schedule",
        "espn_team_played.json",
    )
    routes.file("site.api.espn.com/apis/site/v2/sports/soccer/tur.1/teams", "espn_teams.json")
    routes.file("site.api.espn.com/apis/v2/sports/soccer/tur.1/standings", "espn_standings.json")

    said = await facts(routes).football("Galatasaray")

    assert said.startswith("Galatasaray: 2nd in the table with 13 points")
    fixture = [r for r in routes.asked if r.url.params.get("fixture") == "true"]
    assert len(fixture) == 1


async def test_a_team_not_in_the_league_is_said(routes: Routes) -> None:
    routes.file("site.api.espn.com/apis/site/v2/sports/soccer/tur.1/teams", "espn_teams.json")

    with pytest.raises(NotAnsweredError) as refused:
        await facts(routes).football("Barcelona")

    assert str(refused.value) == NO_TEAM.format(team="Barcelona")


async def test_no_team_is_the_latest_round(routes: Routes) -> None:
    routes.file(
        "site.api.espn.com/apis/site/v2/sports/soccer/tur.1/scoreboard", "espn_scoreboard.json"
    )
    routes.file("site.api.espn.com/apis/v2/sports/soccer/tur.1/standings", "espn_standings.json")

    said = await facts(routes).football("")

    assert said.startswith("The latest round, local time:")


async def test_headlines_fall_back_to_the_regions_feed(routes: Routes) -> None:
    routes.fail("news.google.com/rss")
    routes.file("feeds.bbci.co.uk/turkce/rss.xml", "bbc_turkce.xml")

    found = await facts(routes).headlines("", limit=3)

    assert len(found) == 3 and found[0].source == ""


async def test_a_topic_has_no_fallback(routes: Routes) -> None:
    routes.fail("news.google.com/rss")

    with pytest.raises(FactsError):
        await facts(routes).headlines("ekonomi", limit=3)


async def test_a_topic_is_searched_in_the_regions_edition(routes: Routes) -> None:
    routes.file("news.google.com/rss/search", "gnews_search.xml")

    await facts(routes).headlines(" ekonomi ", limit=3)

    params = routes.asked[0].url.params
    assert (params["q"], params["hl"], params["ceid"]) == ("ekonomi", "tr", "TR:tr")


async def test_a_service_that_times_out_is_a_sentence(routes: Routes) -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    routes.answers["api.jolpi.ca"] = slow

    with pytest.raises(FactsError, match="did not answer within 8 seconds"):
        await facts(routes).formula1()
