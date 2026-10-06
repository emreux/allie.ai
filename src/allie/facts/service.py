"""`Facts`: one method per kind of quick fact (D39), each asking the right
service for the user's region and answering with the words the model reads.

Two ways a method does not answer: `NotAnsweredError`, a sentence of ours (no
city, no such team, not in this region) that the tool passes on as it is;
and `FactsError`, a service that failed, which the tool passes on with the
advice to look it up instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, tzinfo
from typing import Protocol

from allie.facts import markets, news, sport, world
from allie.facts.fetch import FactsError, Fetcher
from allie.facts.news import Headline
from allie.locales import Region
from allie.tools.weather import Place, WeatherError

__all__ = [
    "NOT_IN_REGION",
    "NO_CITY",
    "NO_COIN",
    "NO_SUCH_PLACE",
    "NO_TEAM",
    "Facts",
    "NotAnsweredError",
    "Places",
]

NOT_IN_REGION = (
    "The quick facts have no source for {what} in the user's region. Use look_up instead."
)
NO_CITY = (
    "Name the city: pass the one the user said or the one remembered about them; if neither, "
    "ask the user and remember the answer."
)
NO_SUCH_PLACE = "No place called {place!r} was found. Ask the user for the city, or a larger town."
NO_COIN = "CoinGecko lists no coin called {coin!r}. Ask the user for its name or symbol."
NO_TEAM = "No team called {team!r} plays in the user's league. Use look_up for other leagues."


class NotAnsweredError(Exception):
    """A sentence of ours for the model: nothing was asked of a service, or
    it had nothing for this."""


class Places(Protocol):
    """The slice of `tools/weather.py::OpenMeteo` this needs: the geocoder."""

    async def locate(self, place: str) -> Place | None: ...


def _now() -> datetime:
    return datetime.now(UTC)


class Facts:
    def __init__(
        self,
        region: Region,
        places: Places,
        *,
        fetcher: Fetcher | None = None,
        clock: Callable[[], datetime] = _now,
        tz: tzinfo | None = None,
    ) -> None:
        self._region = region
        self._places = places
        self._fetch = fetcher or Fetcher()
        self._clock = clock
        # `None` is the machine's own zone (`datetime.astimezone(None)`).
        self._tz = tz

    async def aclose(self) -> None:
        await self._fetch.aclose()

    # -- markets ---------------------------------------------------------------

    async def currency(self, about: str) -> str:
        currency = self._region.currency
        if not currency:
            raise NotAnsweredError(NOT_IN_REGION.format(what="exchange rates"))
        codes = tuple(code for code in markets.currency_codes(about) if code != currency)
        codes = codes or markets.DEFAULT_CURRENCIES
        if self._region.rates == "tcmb" and currency == "TRY":
            try:
                xml_text = await self._fetch.text(markets.TCMB_TODAY, service="TCMB")
                return markets.describe_tcmb(markets.parse_tcmb(xml_text), codes, currency)
            except (FactsError, ValueError):
                pass  # the reference rates below answer instead
        body = await self._fetch.json(
            markets.FRANKFURTER,
            {"base": currency, "symbols": ",".join(codes)},
            service="The reference rates service",
        )
        return markets.describe_frankfurter(body, codes, currency)

    async def crypto(self, about: str) -> str:
        words = " ".join(about.split())
        if words:
            found = await self._fetch.json(
                markets.COINGECKO_SEARCH, {"query": words}, service="CoinGecko"
            )
            coin = markets.first_coin(found)
            if coin is None:
                raise NotAnsweredError(NO_COIN.format(coin=words))
            ids: tuple[str, ...] = (coin,)
        else:
            ids = markets.DEFAULT_COINS
        units: tuple[str, ...] = ("usd",)
        if self._region.currency and self._region.currency != "USD":
            units = ("usd", self._region.currency.casefold())
        body = await self._fetch.json(
            markets.COINGECKO_PRICE,
            {
                "ids": ",".join(ids),
                "vs_currencies": ",".join(units),
                "include_24hr_change": "true",
            },
            service="CoinGecko",
        )
        return markets.describe_coins(body, ids, units)

    # -- the world -------------------------------------------------------------------

    async def earthquakes(self, about: str) -> str:
        if self._region.earthquakes != "afad":
            raise NotAnsweredError(NOT_IN_REGION.format(what="earthquakes"))
        body = await self._fetch.json(
            world.AFAD_EVENTS, world.afad_params(self._clock()), service="AFAD"
        )
        return world.describe_quakes(world.parse_afad(body), near=about, tz=self._tz)

    async def prayer_times(self, about: str) -> str:
        method = self._region.prayer_method
        if method is None:
            raise NotAnsweredError(NOT_IN_REGION.format(what="prayer times"))
        place = await self._place(about)
        today = self._clock().astimezone(self._tz).date()
        body = await self._fetch.json(
            world.aladhan_url(today), world.aladhan_params(place, method), service="Aladhan"
        )
        try:
            day, timings = world.parse_aladhan(body)
        except (KeyError, TypeError, ValueError) as failure:
            raise FactsError("Aladhan answered in a shape not understood.") from failure
        return world.describe_prayers(place.label, day, timings)

    async def air_quality(self, about: str) -> str:
        place = await self._place(about)
        body = await self._fetch.json(
            world.AIR_QUALITY, world.air_params(place), service="Open-Meteo air quality"
        )
        return world.describe_air(place.label, world.parse_air(body))

    # -- sport -------------------------------------------------------------------------

    async def formula1(self) -> str:
        coming, last = await asyncio.gather(
            self._fetch.json(sport.F1_NEXT, service="Jolpica"),
            self._fetch.json(sport.F1_LAST, service="Jolpica"),
        )
        try:
            return (
                f"{sport.describe_next_race(coming, tz=self._tz)}\n{sport.describe_last_race(last)}"
            )
        except (KeyError, TypeError, IndexError, ValueError) as failure:
            raise FactsError("Jolpica answered in a shape not understood.") from failure

    async def football(self, about: str) -> str:
        league = self._region.football_league
        if not league:
            raise NotAnsweredError(NOT_IN_REGION.format(what="football"))
        site = sport.ESPN_SITE.format(league=league)
        standings = sport.ESPN_STANDINGS.format(league=league)
        try:
            if not about.strip():
                round_body, table_body = await asyncio.gather(
                    self._fetch.json(f"{site}scoreboard", service="ESPN"),
                    self._fetch.json(standings, service="ESPN"),
                )
                return sport.describe_round(
                    sport.parse_matches(round_body), sport.parse_table(table_body), tz=self._tz
                )
            teams = sport.parse_teams(await self._fetch.json(f"{site}teams", service="ESPN"))
            team = sport.find_team(teams, about)
            if team is None:
                raise NotAnsweredError(NO_TEAM.format(team=about.strip()))
            played, upcoming, table_body = await asyncio.gather(
                self._fetch.json(f"{site}teams/{team.id}/schedule", service="ESPN"),
                self._fetch.json(
                    f"{site}teams/{team.id}/schedule", {"fixture": "true"}, service="ESPN"
                ),
                self._fetch.json(standings, service="ESPN"),
            )
            return sport.describe_team(
                team,
                sport.parse_matches(played),
                sport.parse_matches(upcoming),
                sport.parse_table(table_body),
                tz=self._tz,
            )
        except (KeyError, TypeError, IndexError, ValueError) as failure:
            raise FactsError("ESPN answered in a shape not understood.") from failure

    # -- news ----------------------------------------------------------------------------

    async def headlines(self, topic: str, *, limit: int) -> list[Headline]:
        edition = self._region.news
        if not edition:
            raise NotAnsweredError(NOT_IN_REGION.format(what="news"))
        url, params = news.feed_request(edition, topic)
        # Only the front page has somewhere else to come from: the fallback
        # feed cannot be searched.
        falls_back = not topic.strip() and bool(self._region.news_fallback)
        try:
            xml_text = await self._fetch.text(url, params, service="Google News")
            return news.parse_rss(xml_text, limit=limit)
        except FactsError:
            if not falls_back:
                raise
        except ValueError as failure:
            if not falls_back:
                raise FactsError(
                    "Google News answered with something that is not a feed."
                ) from failure
        xml_text = await self._fetch.text(
            self._region.news_fallback, service="The fallback news feed"
        )
        try:
            return news.parse_rss(xml_text, limit=limit)
        except ValueError as failure:
            raise FactsError("The fallback news feed is not a feed.") from failure

    # -----------------------------------------------------------------------------------

    async def _place(self, about: str) -> Place:
        wanted = about.strip()
        if not wanted:
            raise NotAnsweredError(NO_CITY)
        try:
            place = await self._places.locate(wanted)
        except WeatherError as failure:
            raise FactsError(str(failure)) from failure
        if place is None:
            raise NotAnsweredError(NO_SUCH_PLACE.format(place=wanted))
        return place
