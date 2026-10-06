"""`briefing` (D44): five parts gathered at once, each allowed to fail on
its own, the headlines as outside content, and no city guessed."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone

import pytest

from allie.facts.fetch import FactsError
from allie.facts.news import Headline
from allie.store.db import open_database
from allie.store.repos import ReminderRepo
from allie.tools.briefing import NO_CITY, briefing_for
from allie.tools.mail import MailError
from allie.tools.registry import Tool
from allie.tools.weather import Day, Forecast, Now, Place

TR = timezone(timedelta(hours=3))
NOW = datetime(2026, 9, 27, 9, 0, tzinfo=TR)
ISTANBUL = Place("Istanbul", "Istanbul", "Türkiye", 41.0, 28.9)
FORECAST = Forecast(
    place=ISTANBUL,
    now=Now(temperature=19.0, feels_like=18.0, humidity=70, wind_kmh=12.0, condition="clear sky"),
    days=(Day(on=date(2026, 9, 27), high=24.0, low=16.0, rain_chance=10, condition="clear sky"),),
)


class Weather:
    def __init__(self, *, slow: bool = False) -> None:
        self.slow = slow
        self.asked: list[str] = []

    async def locate(self, place: str) -> Place | None:
        self.asked.append(place)
        if self.slow:
            await asyncio.sleep(10)
        return ISTANBUL if place else None

    async def forecast(self, place: Place) -> Forecast:
        return FORECAST


class News:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.asked: list[tuple[str, int]] = []

    async def headlines(self, topic: str, *, limit: int) -> list[Headline]:
        self.asked.append((topic, limit))
        if self.fail:
            raise FactsError("Google News could not be reached (ConnectError).")
        return [Headline(f"Başlık {n}", "Kaynak", None) for n in range(1, limit + 1)]


class Mailbox:
    def __init__(self, unread: int = 3, *, fail: bool = False) -> None:
        self._unread = unread
        self.fail = fail

    def unread(self) -> int:
        if self.fail:
            raise MailError("imap.x refused: no")
        return self._unread


@pytest.fixture
def reminders() -> Iterator[ReminderRepo]:
    connection: sqlite3.Connection = open_database(":memory:")
    repo = ReminderRepo(connection, clock=lambda: NOW.timestamp() - 3600)
    repo.add("Ahmet'i ara", fire_at=int((NOW + timedelta(hours=2)).timestamp()))
    repo.add("Yarınki iş", fire_at=int((NOW + timedelta(days=1)).timestamp()))
    yield repo
    connection.close()


_ANY_MAILBOX = object()


def brief(
    reminders: ReminderRepo,
    *,
    weather: Weather | None = None,
    news: News | None = None,
    mailbox: object = _ANY_MAILBOX,
    seconds: float = 8.0,
) -> Tool:
    """The tool over fakes; `mailbox=None` is a machine without mail set up."""
    return briefing_for(
        weather=weather or Weather(),
        facts=news or News(),
        reminders=reminders,
        open_mailbox=(lambda: Mailbox()) if mailbox is _ANY_MAILBOX else mailbox,  # type: ignore[arg-type]
        clock=lambda: NOW,
        seconds=seconds,
        tz=TR,
    )


async def test_every_part_is_there_in_order(reminders: ReminderRepo) -> None:
    said = await brief(reminders).run(city="İstanbul")

    lines = said.split("\n")
    assert lines[0].startswith("The current local date and time is 2026-09-27T09:00 (Sunday")
    assert said.index("Weather: Istanbul, Istanbul, Türkiye now") < said.index("Mail: 3 unread.")
    assert '<untrusted source="news">\n1. Başlık 1 - Kaynak' in said
    assert "Başlık 4" not in said
    assert "11:00 Ahmet'i ara" in said
    assert "Yarınki" not in said


async def test_no_city_is_asked_for_not_guessed(reminders: ReminderRepo) -> None:
    weather = Weather()

    said = await brief(reminders, weather=weather).run()

    assert NO_CITY in said
    assert weather.asked == []


async def test_a_failing_part_is_named_and_the_rest_arrive(reminders: ReminderRepo) -> None:
    said = await brief(reminders, news=News(fail=True), mailbox=lambda: Mailbox(fail=True)).run(
        city="İstanbul"
    )

    assert "Headlines: could not be fetched (Google News could not be reached" in said
    assert "Mail: could not be fetched (imap.x refused: no)" in said
    assert "Weather: Istanbul" in said


async def test_a_slow_part_is_cut_off(reminders: ReminderRepo) -> None:
    said = await brief(reminders, weather=Weather(slow=True), seconds=0.05).run(city="İstanbul")

    assert "Weather: did not answer within" in said
    assert "Mail: 3 unread." in said


async def test_without_mail_there_is_no_mail_line(reminders: ReminderRepo) -> None:
    said = await brief(reminders, mailbox=None).run(city="İstanbul")

    assert "Mail" not in said


async def test_a_day_without_reminders_says_so(reminders: ReminderRepo) -> None:
    for reminder in reminders.pending():
        reminders.cancel(reminder.id)

    said = await brief(reminders).run(city="İstanbul")

    assert "Reminders today: none left." in said
