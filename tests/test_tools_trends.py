"""`x_trends` (D30, 23 Sep 2026): the latest hour of a country's list on
X, and one search for why the first five are there.

The page and the search are fakes: `test_web_trends.py` proves the page,
`test_web_search.py` the search; here is what the tool makes of them.
"""

from __future__ import annotations

from datetime import UTC, datetime

from allie.tools.trends import (
    LIST_FAILED,
    NO_COUNTRY,
    NOT_EXPLAINED,
    TRENDS_EXPLAINED,
    UNKNOWN_COUNTRY,
    x_trends_for,
)
from allie.web.search import QUOTA_USED, Found, SearchError
from allie.web.trends import CHANGED, WORLDWIDE, Place, Trends, TrendsError

AT = datetime(2026, 9, 23, 10, 33, 33, tzinfo=UTC)
TEN = ("#A", "B", "C", "D", "E", "F", "G", "H", "I", "J")


class FakePage:
    def __init__(self, trends: Trends | None = None, failure: TrendsError | None = None) -> None:
        self.trends = trends
        self.failure = failure
        self.asked: list[Place] = []

    async def latest(self, place: Place) -> Trends:
        self.asked.append(place)
        if self.failure is not None:
            raise self.failure
        if self.trends is not None:
            return self.trends
        return Trends(
            place=place.name,
            at=AT,
            names=TEN,
            counts=(12500, None, None, None, None, None, None, None, None, None),
        )


class FakeSearch:
    def __init__(self, failure: SearchError | None = None) -> None:
        self.failure = failure
        self.asked: list[str] = []

    async def ask(self, prompt: str) -> Found:
        self.asked.append(prompt)
        if self.failure is not None:
            raise self.failure
        return Found(answer="1. A maç. 2. spam", queries=("A neden gündem",), sources=())


def test_x_trends_is_a_safe_tool_whose_country_may_be_left_out() -> None:
    tool = x_trends_for(FakePage(), FakeSearch())

    assert tool.risk == "safe"
    assert tool.spec.name == "x_trends"
    assert tool.spec.parameters["required"] == []


async def test_the_list_and_why_the_first_five_come_back_marked_as_content() -> None:
    said = await x_trends_for(FakePage(), FakeSearch()).run(country="Turkey")

    assert said.startswith('<untrusted source="trends" place="Turkey" at="2026-09-23 10:33 UTC">')
    assert "Trending on X in Turkey, the hour to 2026-09-23 10:33 UTC:" in said
    assert "1. #A (12,500 posts)" in said
    assert "10. J" in said
    assert f"Why the first {TRENDS_EXPLAINED} are trending:\n1. A maç. 2. spam" in said
    assert said.endswith("</untrusted>")


async def test_one_search_is_asked_about_the_first_five_only() -> None:
    search = FakeSearch()

    await x_trends_for(FakePage(), search).run(country="Turkey")

    [prompt] = search.asked
    assert "trending on X (Twitter) in Turkey in the hour to 2026-09-23 10:33 UTC" in prompt
    assert "1. #A\n2. B\n3. C\n4. D\n5. E" in prompt
    assert "6. F" not in prompt


async def test_the_country_is_found_by_its_everyday_name() -> None:
    page = FakePage()

    await x_trends_for(page, FakeSearch()).run(country="Türkiye")

    assert page.asked == [Place("Turkey", "https://trends24.in/turkey/")]


async def test_the_world_is_worldwide_not_in_worldwide() -> None:
    search = FakeSearch()

    said = await x_trends_for(FakePage(), search).run(country="worldwide")

    assert f'place="{WORLDWIDE}"' in said
    assert "Trending on X worldwide, the hour to" in said
    assert "trending on X (Twitter) worldwide in the hour" in search.asked[0]


async def test_no_country_is_asked_for_and_nothing_fetched() -> None:
    page, search = FakePage(), FakeSearch()

    assert await x_trends_for(page, search).run(country="  ") == NO_COUNTRY
    assert page.asked == [] and search.asked == []


async def test_a_country_with_no_list_is_said_and_the_world_offered() -> None:
    page = FakePage()

    said = await x_trends_for(page, FakeSearch()).run(country="Narnia")

    assert said == UNKNOWN_COUNTRY.format(country="Narnia")
    assert page.asked == []


async def test_a_list_that_cannot_be_read_is_said_with_the_page_to_open() -> None:
    search = FakeSearch()
    page = FakePage(failure=TrendsError(CHANGED))

    said = await x_trends_for(page, search).run(country="Turkey")

    assert said == LIST_FAILED.format(reason=CHANGED, url="https://trends24.in/turkey/")
    assert search.asked == []


async def test_a_failed_search_still_gives_the_list() -> None:
    said = await x_trends_for(FakePage(), FakeSearch(failure=SearchError(QUOTA_USED))).run(
        country="Turkey"
    )

    assert "1. #A (12,500 posts)" in said
    assert NOT_EXPLAINED.format(reason=QUOTA_USED) in said


async def test_an_hour_the_page_did_not_stamp_is_said_as_unknown() -> None:
    page = FakePage(Trends(place="Turkey", at=None, names=("A",), counts=(None,)))

    said = await x_trends_for(page, FakeSearch()).run(country="Turkey")

    assert "the hour to an unknown time" in said
    assert "Why the first 1 are trending" in said
