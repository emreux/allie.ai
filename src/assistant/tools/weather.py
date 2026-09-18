"""`get_weather`: the weather for a place, from Open-Meteo, with no key
(design.md section 3.6, 15 Sep 2026).

**Why Open-Meteo.** The obvious keyless service, wttr.in, could not be
reached from this machine on the day this was written: its certificate had
expired (`CERTIFICATE_VERIFY_FAILED`, three tries). Open-Meteo answered in
55-270 ms, documents its JSON, allows 10,000 requests a day without a key,
and is two calls: a geocoder that turns "Istanbul" into a latitude and
longitude, and a forecast for those. Both are asked here and the place the
geocoder chose is *named in the answer*, because "Kadıköy" came back as
Babadağ in Denizli when it was tried - the user hears where the forecast is
for and can say "no, the one in Istanbul".

**No place is guessed.** The service answers for anywhere; what it cannot
know is where the user is, and neither can this file, which carries no city
(section 3.12). A request with no place is answered with a sentence that
tells the model to ask - and to keep the answer with `remember`, so that the
next "hava nasıl" needs no question. A default in the code would be the
assistant deciding where its user lives.

**What the model gets is numbers and English words.** Temperatures in
degrees Celsius, wind in km/h, a condition from the WMO code table below,
the day's high and low and the chance of rain. No `language` parameter is
sent to the geocoder and none exists for the forecast: the model says it in
the user's language, like every other tool result.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any

import httpx

from assistant.store.normalize import normalize_search
from assistant.tools.registry import Tool, tool

__all__ = [
    "CONDITIONS",
    "MAX_DAYS_AHEAD",
    "SEARCH_SECONDS",
    "Day",
    "Forecast",
    "Now",
    "OpenMeteo",
    "Place",
    "WeatherError",
    "get_weather_for",
]

GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"

# Long enough for a slow line, short enough that a service which is not
# answering does not hold a spoken turn open.
SEARCH_SECONDS = 8.0

# Open-Meteo forecasts seven days: today and six more.
MAX_DAYS_AHEAD = 6

# How many places the geocoder is asked for. The first is usually right; the
# rest are there for the exact-name rule in `_choose`.
PLACES = 5

CURRENT_FIELDS = (
    "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m"
)
DAILY_FIELDS = "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"

# WMO code 4677, as Open-Meteo documents its `weather_code`. English, because
# it is the end of the chain of section 3.12: the model translates.
CONDITIONS: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with light hail",
    99: "thunderstorm with heavy hail",
}

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# The answers addressed to the model.
NO_PLACE = (
    "No place was named. Ask the user which city they mean; if they say where they "
    "live, call remember with it so that this need not be asked again."
)
NOT_FOUND = (
    "No place called {place!r} was found. Ask the user for the city, or a larger town nearby."
)
TOO_FAR = "The forecast reaches {limit} days ahead; {days} is beyond it."


class WeatherError(Exception):
    """A lookup that could not answer, worded so the tool can pass it on."""


@dataclass(frozen=True, slots=True)
class Place:
    name: str
    region: str
    country: str
    latitude: float
    longitude: float

    @property
    def label(self) -> str:
        """ "Istanbul, Istanbul, Türkiye" - every part the geocoder gave, so
        that the user hears which of the world's Kadıköys this is."""
        return ", ".join(part for part in (self.name, self.region, self.country) if part)


@dataclass(frozen=True, slots=True)
class Now:
    temperature: float
    feels_like: float
    humidity: int
    wind_kmh: float
    condition: str


@dataclass(frozen=True, slots=True)
class Day:
    on: date
    high: float
    low: float
    rain_chance: int | None
    condition: str


@dataclass(frozen=True, slots=True)
class Forecast:
    place: Place
    now: Now
    days: tuple[Day, ...]  # today first


class OpenMeteo:
    """Open-Meteo's geocoder and forecast, over one kept connection."""

    def __init__(
        self, *, client: httpx.AsyncClient | None = None, seconds: float = SEARCH_SECONDS
    ) -> None:
        self._client = client
        self._borrowed = client is not None
        self._seconds = seconds

    async def locate(self, place: str) -> Place | None:
        """The place the geocoder thinks `place` is, or `None` for nowhere."""
        answer = await self._get(GEOCODING, {"name": place, "count": PLACES, "format": "json"})
        found = [_place(row) for row in answer.get("results") or [] if isinstance(row, dict)]
        return _choose([hit for hit in found if hit is not None], place)

    async def forecast(self, place: Place) -> Forecast:
        """The conditions now and every day the service will forecast."""
        answer = await self._get(
            FORECAST,
            {
                "latitude": place.latitude,
                "longitude": place.longitude,
                "current": CURRENT_FIELDS,
                "daily": DAILY_FIELDS,
                "timezone": "auto",
                "forecast_days": MAX_DAYS_AHEAD + 1,
            },
        )
        try:
            return _forecast(place, answer)
        except (KeyError, TypeError, ValueError, IndexError) as failure:
            raise WeatherError(
                f"Open-Meteo answered in a shape not understood: {failure}"
            ) from failure

    async def aclose(self) -> None:
        """Gives back the connection, at shutdown. A borrowed client is left alone."""
        if not self._borrowed and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get(self, address: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._seconds)
        try:
            response = await self._client.get(address, params=params, timeout=self._seconds)
            response.raise_for_status()
            body = response.json()
        except httpx.TimeoutException as failure:
            raise WeatherError(
                f"The weather service did not answer within {self._seconds:.0f} seconds."
            ) from failure
        except httpx.HTTPError as failure:
            raise WeatherError(f"The weather service could not be reached: {failure}") from failure
        except ValueError as failure:
            raise WeatherError(
                "The weather service answered with something that is not JSON."
            ) from failure
        if not isinstance(body, dict):
            raise WeatherError("The weather service answered with something that is not a record.")
        return body


def get_weather_for(service: OpenMeteo) -> Tool:
    """`get_weather`, bound to the service that answers."""

    @tool(risk="safe")
    async def get_weather(
        place: Annotated[
            str,
            "The city or town, as the user said it - 'Istanbul', 'Berlin', 'Kadıköy'. Leave "
            "it empty when they named none and nothing remembered says where they are.",
        ] = "",
        days_ahead: Annotated[
            int, "0 for now and today, 1 for tomorrow, up to 6 for the days after."
        ] = 0,
    ) -> str:
        """Reports the weather for a place: what it is like right now and the
        day's high, low and chance of rain - or the forecast for a day up to
        six days ahead. Use it for any question about the weather, rain,
        temperature or wind. The answer names the place the forecast is
        for; say it, so the user can tell you if it is the wrong one."""
        wanted = place.strip()
        if not wanted:
            return NO_PLACE
        if not 0 <= days_ahead <= MAX_DAYS_AHEAD:
            return TOO_FAR.format(limit=MAX_DAYS_AHEAD, days=days_ahead)

        try:
            found = await service.locate(wanted)
            if found is None:
                return NOT_FOUND.format(place=wanted)
            forecast = await service.forecast(found)
        except WeatherError as failure:
            return f"{failure} Tell the user the weather could not be fetched."

        if days_ahead == 0:
            return _today(forecast)
        if days_ahead >= len(forecast.days):
            return TOO_FAR.format(limit=len(forecast.days) - 1, days=days_ahead)
        return _ahead(forecast.place, forecast.days[days_ahead])

    return get_weather


# --------------------------------------------------------------------------
# Reading the answers
# --------------------------------------------------------------------------


def _place(row: dict[str, Any]) -> Place | None:
    try:
        return Place(
            name=str(row["name"]),
            region=str(row.get("admin1") or ""),
            country=str(row.get("country") or ""),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _choose(places: list[Place], asked: str) -> Place | None:
    """The first place whose name is exactly what was asked, else the first.

    The geocoder ranks by size, so "Kadıköy" lists Babadağ - a town it files
    the name under - above the Kadıköys called that. A spoken name that
    matches a listed name outright is the better bet; folded both sides,
    the way every name in this program is compared.
    """
    if not places:
        return None
    wanted = normalize_search(asked).strip()
    for place in places:
        if normalize_search(place.name).strip() == wanted:
            return place
    return places[0]


def _forecast(place: Place, answer: dict[str, Any]) -> Forecast:
    current = answer["current"]
    daily = answer["daily"]
    days = tuple(
        Day(
            on=date.fromisoformat(str(daily["time"][index])),
            high=float(daily["temperature_2m_max"][index]),
            low=float(daily["temperature_2m_min"][index]),
            rain_chance=_chance(daily["precipitation_probability_max"][index]),
            condition=_condition(daily["weather_code"][index]),
        )
        for index in range(len(daily["time"]))
    )
    if not days:
        raise ValueError("no days in the forecast")
    return Forecast(
        place=place,
        now=Now(
            temperature=float(current["temperature_2m"]),
            feels_like=float(current["apparent_temperature"]),
            humidity=int(current["relative_humidity_2m"]),
            wind_kmh=float(current["wind_speed_10m"]),
            condition=_condition(current["weather_code"]),
        ),
        days=days,
    )


def _chance(value: object) -> int | None:
    # The service leaves the probability out (`null`) for some places and
    # some days; a missing chance is not a zero chance.
    return None if value is None else int(float(str(value)))


def _condition(code: object) -> str:
    try:
        return CONDITIONS.get(int(str(code)), f"weather code {code}")
    except ValueError:
        return f"weather code {code}"


def _today(forecast: Forecast) -> str:
    now, today = forecast.now, forecast.days[0]
    return (
        f"{forecast.place.label} now: {_degrees(now.temperature)}, {now.condition}, feels like "
        f"{_degrees(now.feels_like)}, humidity {now.humidity}%, wind {now.wind_kmh:.0f} km/h. "
        f"Today: {_range(today)}."
    )


def _ahead(place: Place, day: Day) -> str:
    return f"{place.label} on {WEEKDAYS[day.on.weekday()]} {day.on.isoformat()}: {_range(day)}."


def _range(day: Day) -> str:
    rain = "" if day.rain_chance is None else f", {day.rain_chance}% chance of rain"
    return f"{_degrees(day.low)} to {_degrees(day.high)}, {day.condition}{rain}"


def _degrees(value: float) -> str:
    return f"{value:.0f} °C"
