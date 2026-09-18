"""`get_weather` (15 Sep 2026): Open-Meteo's geocoder and forecast, and the
sentences the tool makes of them.

Nothing here reaches the network: the service is given an `httpx` client
over a `MockTransport` that answers what each test scripted, and remembers
what it was asked.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

import httpx
import pytest

from assistant.tools.registry import Tool
from assistant.tools.weather import (
    FORECAST,
    GEOCODING,
    MAX_DAYS_AHEAD,
    NO_PLACE,
    Day,
    Now,
    OpenMeteo,
    Place,
    WeatherError,
    get_weather_for,
)

ISTANBUL = {
    "name": "Istanbul",
    "admin1": "Istanbul",
    "country": "Republic of Türkiye",
    "latitude": 41.01384,
    "longitude": 28.94966,
}
BABADAG = {
    "name": "Babadağ",
    "admin1": "Denizli",
    "country": "Republic of Türkiye",
    "latitude": 37.8,
    "longitude": 28.85,
}
KADIKOY = {
    "name": "Kadıköy",
    "admin1": "Yalova",
    "country": "Republic of Türkiye",
    "latitude": 40.62,
    "longitude": 29.22,
}

FORECAST_BODY: dict[str, Any] = {
    "current": {
        "temperature_2m": 21.2,
        "apparent_temperature": 19.8,
        "relative_humidity_2m": 69,
        "weather_code": 3,
        "wind_speed_10m": 21.3,
    },
    "daily": {
        "time": ["2026-09-15", "2026-09-16", "2026-09-17"],
        "temperature_2m_max": [22.8, 20.3, 21.8],
        "temperature_2m_min": [17.4, 18.7, 18.6],
        "precipitation_probability_max": [0, 50, None],
        "weather_code": [3, 61, 95],
    },
}


class Scripted:
    """The transport: one answer per address, and the requests as they came."""

    def __init__(self) -> None:
        self.answers: dict[str, Callable[[httpx.Request], httpx.Response]] = {}
        self.requests: list[httpx.Request] = []

    def geocoder(self, *places: dict[str, Any]) -> None:
        self.answers[GEOCODING] = lambda _: httpx.Response(200, json={"results": list(places)})

    def forecast(self, body: dict[str, Any] | None = None) -> None:
        self.answers[FORECAST] = lambda _: httpx.Response(200, json=body or FORECAST_BODY)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        address = str(request.url).split("?")[0]
        answer = self.answers.get(address)
        if answer is None:
            return httpx.Response(404)
        return answer(request)


@pytest.fixture
def scripted() -> Scripted:
    return Scripted()


@pytest.fixture
def service(scripted: Scripted) -> OpenMeteo:
    client = httpx.AsyncClient(transport=httpx.MockTransport(scripted.handle))
    return OpenMeteo(client=client)


@pytest.fixture
def get_weather(service: OpenMeteo) -> Tool:
    return get_weather_for(service)


def place(row: dict[str, Any]) -> Place:
    return Place(row["name"], row["admin1"], row["country"], row["latitude"], row["longitude"])


# --------------------------------------------------------------------------
# The geocoder
# --------------------------------------------------------------------------


async def test_the_first_place_is_taken_when_none_is_named_exactly(
    service: OpenMeteo, scripted: Scripted
) -> None:
    scripted.geocoder(ISTANBUL, BABADAG)

    assert await service.locate("Istanbul") == place(ISTANBUL)
    asked = scripted.requests[0].url.params
    assert asked["name"] == "Istanbul"
    assert "language" not in asked  # section 3.12: nothing about the user's language is sent


async def test_a_place_named_exactly_beats_a_bigger_one_listed_first(
    service: OpenMeteo, scripted: Scripted
) -> None:
    """Measured 2026-09-15: "Kadıköy" lists Babadağ (Denizli) first, the
    Kadıköys after it. The name the user said wins, folded both sides."""
    scripted.geocoder(BABADAG, KADIKOY)

    assert await service.locate("kadikoy") == place(KADIKOY)


async def test_nowhere_is_none(service: OpenMeteo, scripted: Scripted) -> None:
    scripted.geocoder()

    assert await service.locate("nowhere-xyz") is None


async def test_a_row_the_geocoder_left_unfinished_is_skipped(
    service: OpenMeteo, scripted: Scripted
) -> None:
    scripted.geocoder({"name": "Broken"}, ISTANBUL)

    assert await service.locate("x") == place(ISTANBUL)


# --------------------------------------------------------------------------
# The forecast
# --------------------------------------------------------------------------


async def test_the_forecast_is_read_into_now_and_days(
    service: OpenMeteo, scripted: Scripted
) -> None:
    scripted.forecast()

    forecast = await service.forecast(place(ISTANBUL))

    assert forecast.now == Now(
        temperature=21.2, feels_like=19.8, humidity=69, wind_kmh=21.3, condition="overcast"
    )
    assert forecast.days[1] == Day(
        on=date(2026, 9, 16), high=20.3, low=18.7, rain_chance=50, condition="light rain"
    )
    # A `null` probability is not a 0% chance.
    assert forecast.days[2].rain_chance is None
    assert forecast.days[2].condition == "thunderstorm"
    asked = scripted.requests[0].url.params
    assert asked["latitude"] == "41.01384"
    assert asked["forecast_days"] == str(MAX_DAYS_AHEAD + 1)
    assert asked["timezone"] == "auto"


async def test_a_code_the_table_does_not_know_is_said_as_a_number(
    service: OpenMeteo, scripted: Scripted
) -> None:
    body = json.loads(json.dumps(FORECAST_BODY))
    body["current"]["weather_code"] = 42

    scripted.forecast(body)
    forecast = await service.forecast(place(ISTANBUL))

    assert forecast.now.condition == "weather code 42"


async def test_an_answer_of_the_wrong_shape_is_a_weather_error(
    service: OpenMeteo, scripted: Scripted
) -> None:
    scripted.forecast({"current": {}, "daily": {"time": []}})

    with pytest.raises(WeatherError, match="shape not understood"):
        await service.forecast(place(ISTANBUL))


@pytest.mark.parametrize(
    ("answer", "said"),
    [
        (lambda _: httpx.Response(503), "could not be reached"),
        (lambda _: httpx.Response(200, content=b"<html>"), "not JSON"),
        (lambda _: httpx.Response(200, json=[1, 2]), "not a record"),
    ],
)
async def test_a_service_that_will_not_answer_is_a_sentence(
    service: OpenMeteo,
    scripted: Scripted,
    answer: Callable[[httpx.Request], httpx.Response],
    said: str,
) -> None:
    scripted.answers[GEOCODING] = answer

    with pytest.raises(WeatherError, match=said):
        await service.locate("Istanbul")


async def test_a_timeout_names_the_seconds(scripted: Scripted) -> None:
    def slow(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow")

    scripted.answers[GEOCODING] = slow
    service = OpenMeteo(
        client=httpx.AsyncClient(transport=httpx.MockTransport(scripted.handle)), seconds=3
    )

    with pytest.raises(WeatherError, match="within 3 seconds"):
        await service.locate("Istanbul")


async def test_a_borrowed_client_is_not_closed_and_an_own_one_is() -> None:
    borrowed = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(404)))
    await OpenMeteo(client=borrowed).aclose()
    assert not borrowed.is_closed

    own = OpenMeteo()
    await own.aclose()  # nothing opened yet: nothing to close, no error
    assert own._client is None


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------


def test_it_is_a_safe_tool_with_nothing_required(get_weather: Tool) -> None:
    assert get_weather.risk == "safe"
    assert get_weather.spec.name == "get_weather"
    assert get_weather.spec.parameters["required"] == []
    assert get_weather.spec.parameters["properties"]["days_ahead"]["type"] == "integer"


async def test_no_place_means_ask_and_then_remember(get_weather: Tool, scripted: Scripted) -> None:
    """The code carries no city (section 3.12): the user says where they
    are, once, and `remember` keeps it."""
    assert await get_weather.run(place="  ") == NO_PLACE
    assert scripted.requests == []


async def test_today_is_now_and_the_days_range(get_weather: Tool, scripted: Scripted) -> None:
    scripted.geocoder(ISTANBUL)
    scripted.forecast()

    said = await get_weather.run(place="Istanbul")

    assert said == (
        "Istanbul, Istanbul, Republic of Türkiye now: 21 °C, overcast, feels like 20 °C, "
        "humidity 69%, wind 21 km/h. Today: 17 °C to 23 °C, overcast, 0% chance of rain."
    )


async def test_a_day_ahead_is_that_days_range_with_its_weekday(
    get_weather: Tool, scripted: Scripted
) -> None:
    scripted.geocoder(ISTANBUL)
    scripted.forecast()

    said = await get_weather.run(place="Istanbul", days_ahead=1)

    assert said == (
        "Istanbul, Istanbul, Republic of Türkiye on Wednesday 2026-09-16: "
        "19 °C to 20 °C, light rain, 50% chance of rain."
    )


async def test_a_missing_chance_of_rain_is_left_out(get_weather: Tool, scripted: Scripted) -> None:
    scripted.geocoder(ISTANBUL)
    scripted.forecast()

    said = await get_weather.run(place="Istanbul", days_ahead=2)

    assert said.endswith("19 °C to 22 °C, thunderstorm.")


@pytest.mark.parametrize("days", [-1, 7, 30])
async def test_a_day_beyond_the_forecast_is_refused_before_anything_is_asked(
    get_weather: Tool, scripted: Scripted, days: int
) -> None:
    said = await get_weather.run(place="Istanbul", days_ahead=days)

    assert said == f"The forecast reaches 6 days ahead; {days} is beyond it."
    assert scripted.requests == []


async def test_a_day_the_service_did_not_send_is_refused_with_what_it_did_send(
    get_weather: Tool, scripted: Scripted
) -> None:
    scripted.geocoder(ISTANBUL)
    scripted.forecast()  # three days

    assert await get_weather.run(place="Istanbul", days_ahead=5) == (
        "The forecast reaches 2 days ahead; 5 is beyond it."
    )


async def test_nowhere_tells_the_model_to_ask_for_a_bigger_town(
    get_weather: Tool, scripted: Scripted
) -> None:
    scripted.geocoder()

    said = await get_weather.run(place="Xyzzy")

    assert said == (
        "No place called 'Xyzzy' was found. Ask the user for the city, or a larger town nearby."
    )


async def test_a_service_that_is_down_is_told_to_the_user(
    get_weather: Tool, scripted: Scripted
) -> None:
    scripted.answers[GEOCODING] = lambda _: httpx.Response(500)

    said = await get_weather.run(place="Istanbul")

    assert said.startswith("The weather service could not be reached")
    assert said.endswith("Tell the user the weather could not be fetched.")


async def test_the_place_the_geocoder_chose_is_the_one_named(
    get_weather: Tool, scripted: Scripted
) -> None:
    """The user said "Kadıköy" and meant Istanbul's; the answer says whose
    Kadıköy it is, so they can tell."""
    scripted.geocoder(BABADAG, KADIKOY)
    scripted.forecast()

    said = await get_weather.run(place="Kadıköy")

    assert said.startswith("Kadıköy, Yalova, Republic of Türkiye now:")
