"""What a turn cost, and whether the day can afford another one (2.4).

Three claims. A price is data: the packaged table knows the owner's model, a
table the user writes outranks it, and a model no table names costs `None`
rather than a made-up zero. The formula bills cached tokens as a subset of
the input, never on top of it (section 6). And "today" is the user's
calendar day: the rows are UTC seconds, the boundary is local midnight, and
the sums honour it.

Every database here is `:memory:`, and the clock is held still.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from allie.agent.limits import Limits
from allie.live.base import Usage
from allie.store.db import open_database
from allie.store.repos import UsageRepo
from allie.usage.tracker import (
    PRICING_FILE_NAME,
    Price,
    Pricing,
    UsageTracker,
    start_of_day,
    start_of_month,
)

# A round table, so that the sums below can be checked by hand: a thousand
# input tokens is a thousandth of a dollar.
TABLE = """
[fake."fake-1"]
input_per_mtok = 1.0
output_per_mtok = 10.0
cached_per_mtok = 0.1
"""

# 2023-11-14 22:13:20 UTC. Whatever the machine's zone, a day either side of
# it is another day and a month either side another month.
NOW = 1_700_000_000.0


class Clock:
    """A clock a test can move."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def database() -> Iterator[sqlite3.Connection]:
    connection = open_database(":memory:")
    yield connection
    connection.close()


@pytest.fixture
def clock() -> Clock:
    return Clock(NOW)


@pytest.fixture
def repo(database: sqlite3.Connection, clock: Clock) -> UsageRepo:
    return UsageRepo(database, clock=clock)


def tracking(
    repo: UsageRepo, clock: Clock, limits: Limits | None = None, model: str = "fake-1"
) -> UsageTracker:
    return UsageTracker(
        repo, Pricing.from_toml(TABLE), provider="fake", model=model, limits=limits, clock=clock
    )


# --------------------------------------------------------------------------
# The price of a turn
# --------------------------------------------------------------------------


def test_cached_tokens_are_part_of_the_input_and_not_an_addition_to_it() -> None:
    """A thousand in with four hundred cached is six hundred fresh and four
    hundred cached (section 6)."""
    price = Price(input_per_mtok=1.0, output_per_mtok=10.0, cached_per_mtok=0.1)

    cost = price.cost(Usage(input_tokens=1000, output_tokens=100, cached_tokens=400))

    assert cost == pytest.approx((600 * 1.0 + 400 * 0.1 + 100 * 10.0) / 1_000_000)


def test_no_cache_hit_is_charged_as_all_fresh() -> None:
    price = Price(input_per_mtok=1.0, output_per_mtok=10.0, cached_per_mtok=0.1)

    assert price.cost(Usage(1000, 100)) == pytest.approx(0.002)


def test_a_provider_that_counts_more_cached_than_input_is_billed_as_if_it_had_not() -> None:
    price = Price(input_per_mtok=1.0, output_per_mtok=10.0, cached_per_mtok=0.1)

    assert price.cost(Usage(input_tokens=100, cached_tokens=500)) == pytest.approx(0.00001)


def test_a_model_with_no_price_costs_none_and_not_zero() -> None:
    """A zero would be a number nothing backs, and it would read as free."""
    pricing = Pricing.from_toml(TABLE)

    assert pricing.cost("fake", "fake-2", Usage(1000, 100)) is None
    assert pricing.cost("other", "fake-1", Usage(1000, 100)) is None


def test_a_missing_cached_price_is_the_input_price() -> None:
    """No discount known, none applied."""
    pricing = Pricing.from_toml('[p."m"]\ninput_per_mtok = 2.0\noutput_per_mtok = 4.0\n')

    assert pricing.price("p", "m") == Price(2.0, 4.0, 2.0)


def test_half_a_price_is_no_price() -> None:
    pricing = Pricing.from_toml('[p."m"]\ninput_per_mtok = 2.0\n')

    assert pricing.price("p", "m") is None


def test_a_price_of_the_wrong_shape_is_no_price() -> None:
    pricing = Pricing.from_toml(
        '[p."m"]\ninput_per_mtok = "2"\noutput_per_mtok = true\n[q]\nm = 5\n'
    )

    assert pricing.price("p", "m") is None
    assert pricing.price("q", "m") is None


def test_the_packaged_table_knows_the_owner_s_model() -> None:
    """`config.toml` says `gemini:gemini-3.5-flash-lite`; a table that did not
    price it would make every turn "price unknown"."""
    pricing = Pricing.load(directory=Path("nowhere"))

    assert pricing.price("gemini", "gemini-3.5-flash-lite") is not None


def test_a_table_the_user_writes_outranks_the_packaged_one(tmp_path: Path) -> None:
    """Section 6: prices change, and the fix is an edit, not a release."""
    (tmp_path / PRICING_FILE_NAME).write_text(
        '[gemini."gemini-3.5-flash-lite"]\ninput_per_mtok = 9.0\noutput_per_mtok = 9.0\n',
        encoding="utf-8",
    )

    price = Pricing.load(directory=tmp_path).price("gemini", "gemini-3.5-flash-lite")

    assert price is not None
    assert price.input_per_mtok == 9.0


def test_the_user_s_table_adds_to_the_packaged_one_rather_than_replacing_it(
    tmp_path: Path,
) -> None:
    (tmp_path / PRICING_FILE_NAME).write_text(
        '[mine."m"]\ninput_per_mtok = 1.0\noutput_per_mtok = 1.0\n', encoding="utf-8"
    )

    pricing = Pricing.load(directory=tmp_path)

    assert pricing.price("mine", "m") is not None
    assert pricing.price("gemini", "gemini-2.5-flash") is not None


# --------------------------------------------------------------------------
# Writing the turn down
# --------------------------------------------------------------------------


def test_a_turn_is_written_down_with_its_price(
    database: sqlite3.Connection, repo: UsageRepo, clock: Clock
) -> None:
    cost = tracking(repo, clock).record("t1", Usage(300, 10))

    assert cost == pytest.approx(0.0004)
    [row] = database.execute("SELECT * FROM usage_log").fetchall()
    assert (row["provider"], row["model"], row["turn_id"]) == ("fake", "fake-1", "t1")
    assert (row["in_tokens"], row["out_tokens"], row["cached_tokens"]) == (300, 10, 0)
    assert row["cost_usd"] == pytest.approx(0.0004)
    assert row["ts"] == int(NOW)


def test_a_turn_on_a_model_with_no_price_is_written_down_without_one(
    database: sqlite3.Connection, repo: UsageRepo, clock: Clock
) -> None:
    cost = tracking(repo, clock, model="fake-2").record("t1", Usage(300, 10))

    assert cost is None
    [row] = database.execute("SELECT cost_usd FROM usage_log").fetchall()
    assert row["cost_usd"] is None


# --------------------------------------------------------------------------
# Today, this month
# --------------------------------------------------------------------------


def test_today_is_the_local_day_and_not_the_last_twenty_four_hours(
    repo: UsageRepo, clock: Clock
) -> None:
    """A row from a minute before local midnight is yesterday's."""
    spending = tracking(repo, clock)
    midnight = start_of_day(NOW)
    clock.now = midnight - 60
    spending.record("yesterday", Usage(1000, 0))
    clock.now = midnight + 60
    spending.record("today", Usage(2000, 0))
    clock.now = NOW

    assert spending.spent_today() == pytest.approx(0.002)


def test_this_month_starts_on_the_first(repo: UsageRepo, clock: Clock) -> None:
    spending = tracking(repo, clock)
    first = start_of_month(NOW)
    clock.now = first - 60
    spending.record("last month", Usage(1000, 0))
    clock.now = first + 60
    spending.record("this month", Usage(2000, 0))
    clock.now = NOW

    assert spending.spent_this_month() == pytest.approx(0.002)


def test_the_day_begins_at_local_midnight() -> None:
    """Whatever the machine's zone: the boundary is a wall clock time, turned
    into the epoch second every row is compared against."""
    began = datetime.fromtimestamp(start_of_day(NOW)).astimezone()

    assert (began.hour, began.minute, began.second) == (0, 0, 0)
    assert began.date() == datetime.fromtimestamp(NOW).astimezone().date()


def test_the_month_begins_on_the_first_at_local_midnight() -> None:
    began = datetime.fromtimestamp(start_of_month(NOW)).astimezone()

    assert (began.day, began.hour, began.minute) == (1, 0, 0)
    assert began.month == datetime.fromtimestamp(NOW).astimezone().month


def test_a_turn_with_no_price_adds_nothing_to_the_sum(repo: UsageRepo, clock: Clock) -> None:
    """It cannot add what nobody knows. The report counts such turns instead."""
    tracking(repo, clock, model="fake-2").record("t1", Usage(1000, 0))
    tracking(repo, clock).record("t2", Usage(1000, 0))

    assert tracking(repo, clock).spent_today() == pytest.approx(0.001)


# --------------------------------------------------------------------------
# The limits
# --------------------------------------------------------------------------


def test_under_both_limits_there_is_nothing_to_warn_about(repo: UsageRepo, clock: Clock) -> None:
    spending = tracking(repo, clock, Limits(daily_usd=1.0, monthly_usd=10.0))
    spending.record("t1", Usage(1000, 0))

    assert spending.warning() is None
    assert spending.stopped() is False


def test_past_the_day_s_limit_the_warning_is_the_day_s(repo: UsageRepo, clock: Clock) -> None:
    spending = tracking(repo, clock, Limits(daily_usd=0.0005, monthly_usd=10.0))
    spending.record("t1", Usage(1000, 0))

    assert spending.warning() == "daily_over"


def test_past_the_month_s_limit_the_warning_is_the_month_s(repo: UsageRepo, clock: Clock) -> None:
    """Even with the day's passed as well: the month's is the larger."""
    spending = tracking(repo, clock, Limits(daily_usd=0.0005, monthly_usd=0.0005))
    spending.record("t1", Usage(1000, 0))

    assert spending.warning() == "monthly_over"


def test_exactly_at_the_limit_is_not_over_it(repo: UsageRepo, clock: Clock) -> None:
    spending = tracking(repo, clock, Limits(daily_usd=0.001))
    spending.record("t1", Usage(1000, 0))

    assert spending.warning() is None


def test_a_limit_passed_stops_nothing_unless_hard_stop_says_so(
    repo: UsageRepo, clock: Clock
) -> None:
    """Section 12, decision 10: warn by default; be silenced only on request."""
    warned = tracking(repo, clock, Limits(daily_usd=0.0005))
    silenced = tracking(repo, clock, Limits(daily_usd=0.0005, hard_stop=True))
    warned.record("t1", Usage(1000, 0))

    assert warned.stopped() is False
    assert silenced.stopped() is True


def test_hard_stop_under_the_limit_stops_nothing(repo: UsageRepo, clock: Clock) -> None:
    spending = tracking(repo, clock, Limits(daily_usd=5.0, hard_stop=True))
    spending.record("t1", Usage(1000, 0))

    assert spending.stopped() is False


def test_the_defaults_are_two_dollars_a_day_and_thirty_a_month_with_no_hard_stop() -> None:
    """Section 12, decision 10, as the design assumes it until the owner says otherwise."""
    assert (Limits().daily_usd, Limits().monthly_usd, Limits().hard_stop) == (2.0, 30.0, False)
