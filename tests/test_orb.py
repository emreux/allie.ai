"""The orb (`ui/orb.py`, plan.md D20): the window's picture of the state, as numbers.

What is claimed: every state has a colour and a speed of its own; the rings
turn by their own speed times the state's, without a jump when the state
changes; reconnecting turns everything one way; the core breathes on a four
second cycle and swells with the level; confirming blinks; the level rises
fast and falls slowly and never leaves 0..1. Nothing here opens a screen.
"""

from __future__ import annotations

import inspect
import math

import pytest

from assistant.app import State
from assistant.ui import orb
from assistant.ui.orb import (
    BREATH_SECONDS,
    CAPS,
    COLOURS,
    RINGS,
    SILENCE_DBFS,
    SPEED,
    Level,
    Orb,
    to_unit,
)


def test_every_state_has_a_colour_and_a_speed() -> None:
    assert set(COLOURS) == set(State)
    assert set(SPEED) == set(State)


def test_idle_is_blue_hearing_green_speaking_purple_confirming_yellow() -> None:
    """The owner's reading of the states (spec section 3)."""
    assert COLOURS[State.IDLE] == (58, 160, 255)
    assert COLOURS[State.USER_SPEAKING] == (46, 204, 113)
    assert COLOURS[State.SPEAKING] == COLOURS[State.ANNOUNCING] == (155, 89, 182)
    assert COLOURS[State.CONFIRMING] == (241, 196, 15)
    assert COLOURS[State.RECONNECTING] == (243, 156, 18)
    # Asleep (D21): the idle blue at a third, and slower than off - a
    # breath, not a mechanism.
    assert COLOURS[State.SLEEPING] == (40, 90, 150)
    assert SPEED[State.SLEEPING] < SPEED[State.OFF]
    assert COLOURS[State.OFF] == (110, 110, 110)


def test_the_five_rings_of_the_chosen_look_and_their_caps() -> None:
    assert len(RINGS) == 5
    assert [ring.dashes for ring in RINGS] == [3, 36, 5, 2, 72]
    assert all(0 < ring.gap < 1 for ring in RINGS)
    assert all(0 <= ring_index < len(RINGS) for ring_index, _ in CAPS)
    assert all(dash < RINGS[ring_index].dashes for ring_index, dash in CAPS)


def test_a_ring_turns_by_its_speed_times_the_states() -> None:
    picture = Orb()

    picture.advance(State.IDLE, 1.0)
    assert picture.angles == pytest.approx(
        [(ring.speed * SPEED[State.IDLE]) % 360 for ring in RINGS]
    )

    picture.advance(State.SPEAKING, 1.0)
    expected = [ring.speed * (SPEED[State.IDLE] + SPEED[State.SPEAKING]) for ring in RINGS]
    assert picture.angles == pytest.approx([angle % 360 for angle in expected])


def test_a_change_of_state_changes_the_speed_and_never_jumps() -> None:
    """Angles are integrated, not computed from a clock: switching from
    idle to speaking makes the rings faster from here, not somewhere else."""
    slow, fast = Orb(), Orb()
    slow.advance(State.IDLE, 2.0)
    fast.advance(State.IDLE, 2.0)

    before = list(fast.angles)
    fast.advance(State.SPEAKING, 0.001)

    assert fast.angles == pytest.approx(before, abs=0.1)
    assert fast.angles != slow.angles


def test_reconnecting_turns_every_ring_the_same_way() -> None:
    picture = Orb()

    picture.advance(State.RECONNECTING, 1.0)

    assert all(angle > 0 for angle in picture.angles)
    assert picture.angles == pytest.approx(
        [abs(ring.speed) * SPEED[State.RECONNECTING] for ring in RINGS]
    )


def test_off_is_slower_than_idle_which_is_slower_than_speaking() -> None:
    assert SPEED[State.OFF] < SPEED[State.IDLE] < SPEED[State.USER_SPEAKING] < SPEED[State.SPEAKING]


def test_the_core_breathes_four_percent_on_a_four_second_cycle() -> None:
    picture = Orb()

    top = picture.advance(State.IDLE, BREATH_SECONDS / 4).core_scale
    picture.advance(State.IDLE, BREATH_SECONDS / 4)
    bottom = picture.advance(State.IDLE, BREATH_SECONDS / 4).core_scale

    assert top == pytest.approx(1.04, abs=0.001)
    assert bottom == pytest.approx(0.96, abs=0.001)


def test_the_core_swells_with_the_level() -> None:
    quiet, loud = Orb(), Orb()
    loud.level.feed(-10.0)

    still = quiet.advance(State.USER_SPEAKING, 0.0).core_scale
    for _ in range(40):
        swollen = loud.advance(State.USER_SPEAKING, 0.0).core_scale

    assert swollen == pytest.approx(still * 1.25, abs=0.01)


def test_the_frame_carries_the_states_colour_and_one_angle_per_ring() -> None:
    picture = Orb()

    frame = picture.advance(State.CONFIRMING, 0.0)

    assert frame.colour == COLOURS[State.CONFIRMING]
    assert len(frame.angles) == len(RINGS)


def test_confirming_blinks_between_half_and_full_and_nothing_else_does() -> None:
    picture = Orb()
    seen = {picture.advance(State.CONFIRMING, 0.1).core_alpha for _ in range(12)}

    assert min(seen) < 0.7
    assert max(seen) > 0.95
    assert all(0.55 <= alpha <= 1.0 for alpha in seen)
    assert Orb().advance(State.SPEAKING, 0.3).core_alpha == 1.0


# --------------------------------------------------------------------------
# The level
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dbfs", "unit"),
    [(-90.0, 0.0), (-50.0, 0.0), (-30.0, 0.5), (-10.0, 1.0), (0.0, 1.0)],
)
def test_dbfs_maps_to_the_unit_range_between_minus_fifty_and_minus_ten(
    dbfs: float, unit: float
) -> None:
    assert to_unit(dbfs) == pytest.approx(unit)


def test_the_level_rises_fast_and_falls_slowly() -> None:
    level = Level()
    level.feed(-10.0)
    rise = level.step()

    level.feed(SILENCE_DBFS)
    fall = rise - level.step()

    assert rise == pytest.approx(0.5)
    assert fall == pytest.approx(0.5 * 0.08)


def test_the_level_settles_on_its_target_and_never_leaves_the_range() -> None:
    level = Level()
    level.feed(-10.0)
    for _ in range(60):
        value = level.step()
    assert value == pytest.approx(1.0, abs=0.001)
    assert 0.0 <= value <= 1.0

    level.feed(-200.0)
    for _ in range(200):
        value = level.step()
    assert value == pytest.approx(0.0, abs=0.001)


def test_silence_is_the_floor_the_hooks_send() -> None:
    assert SILENCE_DBFS == -90.0
    assert math.isclose(to_unit(SILENCE_DBFS), 0.0)


def test_the_module_opens_no_screen() -> None:
    """The numbers are testable because nothing here touches Tk."""
    assert "tkinter" not in inspect.getsource(orb)
