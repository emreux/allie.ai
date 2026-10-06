"""The keyless services' answers, parsed and said (D39). Real answers of
2026-09-27 from `tests/data/facts/`; nothing here reaches the network."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from allie.facts.markets import (
    amount,
    currency_codes,
    describe_coins,
    describe_frankfurter,
    describe_tcmb,
    first_coin,
    parse_tcmb,
)
from allie.facts.news import describe_headlines, feed_request, parse_rss
from allie.facts.sport import (
    describe_last_race,
    describe_next_race,
    describe_round,
    describe_team,
    find_team,
    parse_matches,
    parse_table,
    parse_teams,
)
from allie.facts.world import (
    Air,
    afad_params,
    aqi_band,
    describe_air,
    describe_prayers,
    describe_quakes,
    parse_afad,
    parse_air,
    parse_aladhan,
)

DATA = Path(__file__).parent / "data" / "facts"
TR = timezone(timedelta(hours=3))


def text(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8")


def body(name: str) -> Any:
    return json.loads(text(name))


# -- markets -----------------------------------------------------------------


def test_amounts_read_the_way_a_model_says_them() -> None:
    assert amount(48.7901) == "48.7901"
    assert amount(4155780.0) == "4,155,780"
    assert amount(2714.85) == "2,714.85"
    assert amount(0.02044) == "0.0204"


def test_currency_codes_come_from_the_words_or_the_defaults() -> None:
    assert currency_codes("usd, EUR ve gbp") == ("USD", "EUR", "GBP")
    assert currency_codes("usd usd") == ("USD",)
    assert currency_codes("") == ("USD", "EUR")
    assert currency_codes("dolar") == ("USD", "EUR")


def test_the_central_bank_bulletin_is_read_with_its_day_and_units() -> None:
    bulletin = parse_tcmb(text("tcmb_today.xml"))

    assert bulletin.day == "2026-09-25"
    usd = bulletin.rates["USD"]
    assert (usd.unit, usd.buying, usd.selling) == (1, 48.7901, 48.878)
    assert bulletin.rates["JPY"].unit == 100


def test_the_bulletin_is_said_as_a_bulletin_and_a_missing_code_is_named() -> None:
    said = describe_tcmb(parse_tcmb(text("tcmb_today.xml")), ("USD", "XYZ"), "TRY")

    assert said.startswith("TCMB indicative rates of 2026-09-25 (the day's bulletin, not the")
    assert "1 USD = 48.7901 buying / 48.878 selling TRY" in said
    assert said.endswith("TCMB lists no rate for XYZ.")


def test_reference_rates_are_turned_round_to_the_users_currency() -> None:
    said = describe_frankfurter(body("frankfurter.json"), ("USD", "EUR"), "TRY")

    assert said.startswith("European Central Bank reference rates of 2026-09-25")
    assert "1 USD = 48.9237 TRY" in said
    assert "1 EUR = 55.8036 TRY" in said


def test_a_coin_search_takes_the_first_coin() -> None:
    assert first_coin(body("coingecko_search.json")) == "solana"
    assert first_coin({"coins": []}) is None


def test_coin_prices_carry_both_currencies_and_the_days_change() -> None:
    said = describe_coins(body("coingecko_price.json"), ("bitcoin", "missingcoin"), ("usd", "try"))

    assert "bitcoin: 84,911 USD, 4,155,780 TRY (+1.1 % in 24 hours)" in said
    assert said.endswith("No price for missingcoin.")


# -- world ---------------------------------------------------------------------


def test_afad_is_asked_for_a_day_back_in_utc() -> None:
    params = afad_params(datetime(2026, 9, 27, 9, 0, tzinfo=UTC))

    assert params["start"] == "2026-09-26T09:00:00"
    assert params["end"] == "2026-09-27T09:00:00"
    assert (params["minmag"], params["orderby"]) == ("3", "timedesc")


def test_afad_dates_are_utc_and_said_in_local_time() -> None:
    quakes = parse_afad(body("afad.json"))

    assert len(quakes) == 5
    adana = quakes[-1]
    assert adana.when == datetime(2026, 9, 21, 1, 2, 1, tzinfo=UTC)
    said = describe_quakes(quakes, near="", tz=TR)
    assert "M5.0 - 2026-09-21 04:02 - Saimbeyli (Adana), 7 km deep" in said


def test_quakes_can_be_narrowed_to_a_province_in_any_spelling() -> None:
    said = describe_quakes(parse_afad(body("afad.json")), near="izmir", tz=TR)

    assert said.count("\n") == 2  # the heading and two lines
    assert "Foça (İzmir)" in said and "Adana" not in said


def test_no_quake_is_said_as_none() -> None:
    assert describe_quakes([], near="", tz=TR) == (
        "No earthquake of magnitude 3 or more in the last 24 hours (AFAD)."
    )


def test_prayer_times_are_the_six_the_user_asks_about() -> None:
    day, timings = parse_aladhan(body("aladhan.json"))

    assert day == "2026-09-27"
    assert timings == {
        "Fajr": "05:25",
        "Sunrise": "06:50",
        "Dhuhr": "13:00",
        "Asr": "16:22",
        "Maghrib": "19:00",
        "Isha": "20:20",
    }
    said = describe_prayers("Istanbul, Istanbul, Türkiye", day, timings)
    assert said.startswith("Prayer times for Istanbul, Istanbul, Türkiye on 2026-09-27: Fajr 05:25")


def test_air_quality_is_read_and_banded() -> None:
    air = parse_air(body("openmeteo_air.json"))

    assert air == Air(aqi=28, pm25=5.8, pm10=9.3)
    assert [aqi_band(value) for value in (5, 28, 55, 70, 95, 140)] == [
        "good",
        "fair",
        "moderate",
        "poor",
        "very poor",
        "extremely poor",
    ]
    assert describe_air("Pendik", air) == (
        "Air quality in Pendik now: European AQI 28 (fair), PM2.5 5.8 µg/m³, PM10 9.3 µg/m³."
    )


# -- sport ---------------------------------------------------------------------


def test_the_next_grand_prix_is_said_session_by_session_in_local_time() -> None:
    said = describe_next_race(body("jolpica_next.json"), tz=TR)

    assert said.startswith("Next: Bahrain Grand Prix in Malaysia, round 16, Sepang")
    assert "First practice Fri 2026-10-02 07:30" in said
    assert said.endswith("Race Sun 2026-10-04 10:00.")


def test_the_last_grand_prix_is_its_podium() -> None:
    said = describe_last_race(body("jolpica_last.json"))

    assert said == (
        "Last: Azerbaijan Grand Prix (2026-09-26): 1. George Russell (Mercedes), "
        "2. Max Verstappen (Red Bull), 3. Isack Hadjar (Red Bull)."
    )


def test_a_team_is_found_by_any_spelling_of_its_name() -> None:
    teams = parse_teams(body("espn_teams.json"))

    assert len(teams) == 18
    assert find_team(teams, "galatasaray").id == "432"  # type: ignore[union-attr]
    assert find_team(teams, "Beşiktaş").name == "Besiktas"  # type: ignore[union-attr]
    assert find_team(teams, "Rizespor").name == "Caykur Rizespor"  # type: ignore[union-attr]
    assert find_team(teams, "Barcelona") is None


def test_a_teams_matches_and_place_in_the_table() -> None:
    played = parse_matches(body("espn_team_played.json"))
    upcoming = parse_matches(body("espn_team_fixture.json"))
    table = parse_table(body("espn_standings.json"))
    galatasaray = find_team(parse_teams(body("espn_teams.json")), "Galatasaray")
    assert galatasaray is not None

    said = describe_team(galatasaray, played, upcoming, table, tz=TR)

    assert said.startswith("Galatasaray: 2nd in the table with 13 points after 6 games.")
    assert "2026-09-19 20:00 Trabzonspor 4-0 Galatasaray" in said
    assert "2026-10-09 20:00 Galatasaray - Kasimpasa" in said


def test_the_latest_round_and_the_top_of_the_table() -> None:
    said = describe_round(
        parse_matches(body("espn_scoreboard.json")), parse_table(body("espn_standings.json")), tz=TR
    )

    assert "Fenerbahce 8-0 Eyupspor" in said
    assert said.endswith(
        "Top of the table: 1. Amed SFK 13, 2. Galatasaray 13, 3. Besiktas 12, "
        "4. Kocaelispor 12, 5. Alanyaspor 11."
    )


# -- news ----------------------------------------------------------------------


def test_the_feed_comes_from_the_regions_parameters() -> None:
    assert feed_request("hl=tr&gl=TR&ceid=TR:tr", "") == (
        "https://news.google.com/rss",
        {"hl": "tr", "gl": "TR", "ceid": "TR:tr"},
    )
    assert feed_request("hl=tr&gl=TR&ceid=TR:tr", " ekonomi ")[1]["q"] == "ekonomi"


def test_headlines_lose_the_source_google_appends() -> None:
    found = parse_rss(text("gnews_top.xml"), limit=3)

    assert len(found) == 3
    first = found[0]
    assert first.source == "Sözcü"
    assert first.title == ("AKP'den Fatma Betül Sayan Kaya'nın istifasının ardından ilk açıklama")
    assert first.published == datetime(2026, 9, 27, 7, 46, 43, tzinfo=UTC)


def test_a_feed_without_sources_keeps_its_titles() -> None:
    found = parse_rss(text("bbc_turkce.xml"), limit=2)

    assert found[0].source == ""
    assert found[0].title.startswith("Türkiye, Husi-Suudi")


def test_headlines_are_said_numbered_with_source_and_local_time() -> None:
    said = describe_headlines(parse_rss(text("gnews_top.xml"), limit=2), tz=TR)

    assert said.startswith(
        "1. AKP'den Fatma Betül Sayan Kaya'nın istifasının ardından ilk açıklama - Sözcü, 10:46"
    )
    assert said.count("\n") == 1
