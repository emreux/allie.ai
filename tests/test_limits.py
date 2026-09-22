"""The limits of section 3.11: a turn stops at eight calls, the same call once
too often in a row is refused, and the numbers are written once.

`TurnGuard` is tested on its own here - a referee, asked call by call. What
the loop does with its answers is in `test_agent_loop.py`; the spending
limits, which are about days rather than turns, are in `test_usage.py`.
"""

from __future__ import annotations

import pytest

from allie.agent.limits import DUPLICATE_CALL, TOOL_LIMIT_REACHED, Limits, TurnGuard
from allie.config import LimitSettings
from allie.live.base import ToolCall


def call(tool: str = "clock", **arguments: object) -> ToolCall:
    """A model's order for `tool`, with no id: Gemini sends none."""
    return ToolCall(id="", name=tool, arguments=arguments)


# --------------------------------------------------------------------------
# How many calls a turn may make
# --------------------------------------------------------------------------


def test_every_call_up_to_the_limit_goes_through() -> None:
    guard = TurnGuard(Limits(tool_calls_per_turn=3))

    assert [guard.allow(call(n=n)) for n in range(3)] == [None, None, None]
    assert guard.exhausted


def test_the_call_over_the_limit_is_told_so() -> None:
    """As a tool result, in the model's own channel: the one thing left for
    it to do is answer with what it has."""
    guard = TurnGuard(Limits(tool_calls_per_turn=2))
    guard.allow(call(n=1))
    guard.allow(call(n=2))

    assert guard.allow(call(n=3)) == TOOL_LIMIT_REACHED
    assert guard.allow(call(n=4)) == TOOL_LIMIT_REACHED


def test_a_fresh_guard_holds_nothing_against_anyone() -> None:
    assert not TurnGuard(Limits()).exhausted


# --------------------------------------------------------------------------
# The same call, again and again
# --------------------------------------------------------------------------


def test_the_same_call_once_too_often_in_a_row_is_refused() -> None:
    """Architecture guide section 12: a model that did not get what it
    wanted asks again with the same arguments, and would go on asking."""
    guard = TurnGuard(Limits(duplicate_calls=2))

    assert guard.allow(call("open_app", name="Spotify")) is None
    assert guard.allow(call("open_app", name="Spotify")) is None
    assert guard.allow(call("open_app", name="Spotify")) == DUPLICATE_CALL.format(times=2)


def test_the_refusal_says_how_many_times_were_allowed() -> None:
    guard = TurnGuard(Limits(duplicate_calls=1))
    guard.allow(call())

    assert guard.allow(call()) == DUPLICATE_CALL.format(times=1)


def test_a_different_argument_in_between_starts_the_count_again() -> None:
    guard = TurnGuard(Limits(duplicate_calls=2))
    guard.allow(call(app="Spotify"))
    guard.allow(call(app="Spotify"))
    guard.allow(call(app="Chrome"))

    assert guard.allow(call(app="Spotify")) is None
    assert guard.allow(call(app="Spotify")) is None
    assert guard.allow(call(app="Spotify")) is not None


def test_a_different_tool_in_between_starts_the_count_again() -> None:
    guard = TurnGuard(Limits(duplicate_calls=2))
    guard.allow(call("clock"))
    guard.allow(call("clock"))
    guard.allow(call("calendar"))

    assert guard.allow(call("clock")) is None


def test_the_same_arguments_in_another_order_are_the_same_call() -> None:
    """The fingerprint is of the arguments, not of the JSON the model happened
    to send: `{"a": 1, "b": 2}` and `{"b": 2, "a": 1}` are one call."""
    guard = TurnGuard(Limits(duplicate_calls=2))
    guard.allow(ToolCall(id="c1", name="x", arguments={"a": 1, "b": 2}))
    guard.allow(ToolCall(id="c2", name="x", arguments={"b": 2, "a": 1}))

    assert guard.allow(ToolCall(id="c3", name="x", arguments={"a": 1, "b": 2})) is not None


def test_the_call_s_id_is_not_what_makes_it_the_same_call() -> None:
    """Gemini sends none - `""` on every call, measured 2026-09-09 - and two
    ids for one request are exactly the repeat to catch."""
    guard = TurnGuard(Limits(duplicate_calls=1))
    guard.allow(ToolCall(id="c1", name="x", arguments={}))

    assert guard.allow(ToolCall(id="c2", name="x", arguments={})) is not None


def test_a_refused_repeat_still_counts_towards_the_turn_s_calls() -> None:
    """A refusal that did not count would leave a model repeating itself with
    no ceiling on the rounds it costs."""
    guard = TurnGuard(Limits(tool_calls_per_turn=3, duplicate_calls=1))
    guard.allow(call())
    guard.allow(call())
    guard.allow(call())

    assert guard.exhausted
    assert guard.allow(call(n=99)) == TOOL_LIMIT_REACHED


# --------------------------------------------------------------------------
# The numbers
# --------------------------------------------------------------------------


def test_the_defaults_are_the_table_of_section_3_11() -> None:
    assert Limits() == Limits(
        tool_calls_per_turn=8,
        output_tokens=4000,
        duplicate_calls=2,
        turn_seconds=60.0,
        daily_usd=2.0,
        monthly_usd=30.0,
        hard_stop=False,
        duplicate_window_sec=600,
    )


def test_the_settings_file_takes_its_defaults_from_the_same_place() -> None:
    """Written once: `config.py` reads them off `Limits`, so the file's
    defaults cannot drift from the loop's."""
    assert Limits.from_settings(LimitSettings()) == Limits()


def test_what_the_settings_say_replaces_the_defaults_field_for_field() -> None:
    settings = LimitSettings(tool_calls_per_turn=3, daily_usd=0.5, hard_stop=True)

    limits = Limits.from_settings(settings)

    assert (limits.tool_calls_per_turn, limits.daily_usd, limits.hard_stop) == (3, 0.5, True)
    assert limits.output_tokens == Limits().output_tokens


def test_the_limits_cannot_be_changed_under_a_running_turn() -> None:
    with pytest.raises(AttributeError):
        Limits().tool_calls_per_turn = 100  # type: ignore[misc]
