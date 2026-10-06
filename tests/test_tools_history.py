"""`recall` (D45): earlier turns found by their words, within the days
asked, as outside content."""

from __future__ import annotations

from allie.store.db import open_database
from allie.store.repos import HistoryRepo
from allie.tools.history import NOTHING, SHORT_QUERY, recall_for

NOW = 1_790_000_000
DAY = 86_400


def archive() -> HistoryRepo:
    connection = open_database(":memory:")
    HistoryRepo(connection, clock=lambda: NOW - 10 * DAY).add(
        "t1", "Tatil için Kaş'a gitmeyi düşünüyorum", "Kaş güzel bir seçim." + " uzun" * 100
    )
    HistoryRepo(connection, clock=lambda: NOW - 40 * DAY).add("t0", "Kaş çok eskiydi", "Evet.")
    return HistoryRepo(connection, clock=lambda: NOW)


def test_recall_is_safe_and_needs_words() -> None:
    tool = recall_for(archive(), clock=lambda: NOW)

    assert tool.risk == "safe"
    assert tool.spec.parameters["required"] == ["query"]


async def test_a_turn_comes_back_with_its_day_both_sides_cut_and_marked() -> None:
    said = await recall_for(archive(), clock=lambda: NOW).run(query="kaş")

    assert said.startswith('<untrusted source="history">\n')
    assert "user: Tatil için Kaş'a gitmeyi düşünüyorum" in said
    assert "you: Kaş güzel bir seçim." in said
    assert "…" in said  # the long answer was cut at 300 characters
    assert "eskiydi" not in said  # older than the thirty days


async def test_the_days_are_held_to_the_archive() -> None:
    tool = recall_for(archive(), clock=lambda: NOW)

    assert "eskiydi" not in await tool.run(query="kaş", days=400)
    assert await tool.run(query="kaş", days=5) == NOTHING.format(query="kaş", days=5)


async def test_a_query_too_short_to_search_is_explained() -> None:
    said = await recall_for(archive(), clock=lambda: NOW).run(query="ve")

    assert said == SHORT_QUERY.format(query="ve")
