"""The orb: the window's picture of the state, as numbers (plan.md D20).

No Tk, no thread, no clock of its own. `Orb.advance(state, dt)` says where
every ring is and what the whole thing wears after `dt` more seconds;
whoever draws it applies the numbers. Kept apart from the window so that
the look can be tested by the numbers and changed without a screen.

The look is `docs/images/orb-2-hud-halkalari.png`, the owner's choice of
three: a glowing core, five arcs of different weights turning at different
speeds and in both directions, three dots riding the two long arcs. Colour
is the state - the same reading the status line and the tray give in words
and in a disc; speed is the state too; the core breathes and swells with
the sound, the microphone's while the user is heard and the speaker's while
anything plays (spec section 3).

**Angles are integrated, not computed from a clock.** A state change changes
the speed from here on; computing `speed * t` would make the rings jump to
where they would have been at the new speed all along.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from assistant.app import State

__all__ = [
    "ATTACK_SECONDS",
    "BREATH_SECONDS",
    "CAPS",
    "COLOURS",
    "CORE",
    "RELEASE_SECONDS",
    "RINGS",
    "SILENCE_DBFS",
    "SPEED",
    "Frame",
    "Level",
    "Orb",
    "Ring",
    "to_unit",
]

RGB = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Ring:
    """One arc of the orb: where, how heavy, how broken, how fast, how bright.

    `radius` and `width` are fractions of the orb's radius; `dashes` is how
    many pieces the circle is cut into and `gap` what fraction of each piece
    is empty; `speed` is degrees per second, negative for anticlockwise;
    `tint` is how much of the state's colour the arc carries (1 = all of it,
    less = blended towards the background).
    """

    radius: float
    width: float
    dashes: int
    gap: float
    speed: float
    tint: float

    @property
    def step(self) -> float:
        """Degrees from the start of one dash to the start of the next."""
        return 360 / self.dashes

    @property
    def extent(self) -> float:
        """Degrees one dash is drawn over."""
        return self.step * (1 - self.gap)


# The core disc's radius as a fraction of the orb's radius.
CORE = 0.50

# The mockup's five arcs, inside out (spec section 3's table). The tints are
# a shade stronger than the mockup's since task U1: the two outer arcs at
# 0.6 and 0.45 sat too close to the background to read as a mechanism.
RINGS: tuple[Ring, ...] = (
    Ring(radius=0.68, width=0.070, dashes=3, gap=0.25, speed=18.0, tint=1.0),
    Ring(radius=0.82, width=0.020, dashes=36, gap=0.55, speed=-6.0, tint=0.90),
    Ring(radius=0.98, width=0.040, dashes=5, gap=0.40, speed=10.0, tint=0.90),
    Ring(radius=1.12, width=0.020, dashes=2, gap=0.35, speed=-14.0, tint=0.72),
    Ring(radius=1.30, width=0.012, dashes=72, gap=0.60, speed=3.0, tint=0.58),
)

# The dots: (ring index, the dash whose start they sit on).
CAPS: tuple[tuple[int, int], ...] = ((3, 0), (3, 1), (2, 0))

COLOURS: dict[State, RGB] = {
    State.IDLE: (58, 160, 255),
    State.OFF: (110, 110, 110),
    # Asleep: the idle blue at a third (D21).
    State.SLEEPING: (40, 90, 150),
    State.USER_SPEAKING: (46, 204, 113),
    State.SPEAKING: (155, 89, 182),
    State.ANNOUNCING: (155, 89, 182),
    State.CONFIRMING: (241, 196, 15),
    State.RECONNECTING: (243, 156, 18),
}

# The rings' speed, as a multiple of `Ring.speed`. Reconnecting turns every
# ring the same way, fast - a spinner rather than a mechanism.
SPEED: dict[State, float] = {
    State.IDLE: 1.0,
    State.OFF: 0.3,
    State.SLEEPING: 0.15,
    State.USER_SPEAKING: 1.6,
    State.SPEAKING: 2.5,
    State.ANNOUNCING: 2.5,
    State.CONFIRMING: 1.0,
    State.RECONNECTING: 3.0,
}

BREATH_SECONDS = 4.0
BREATH = 0.04
# How much the core grows at full level.
SWELL = 0.25
BLINK_SECONDS = 1.2
BLINK_LOW = 0.55

# What the hooks send for a block of nothing, and what the level maps to
# zero and one: speech on the owner's microphone is -26..-45 dBFS, the
# model's voice through the speaker around -20.
SILENCE_DBFS = -90.0
FLOOR_DBFS = -50.0
CEILING_DBFS = -10.0
# How long the swell takes to close the gap, as a time constant rather than
# a share per frame: the window draws sixty frames a second now where it
# drew thirty (task U1), and a share per frame would have made every swell
# twice as fast with it. These two are the old 0.5 and 0.08 a frame at
# thirty frames, read as seconds.
ATTACK_SECONDS = 0.048
RELEASE_SECONDS = 0.40


def to_unit(dbfs: float) -> float:
    """`FLOOR_DBFS` and under → 0, `CEILING_DBFS` and over → 1, linear between."""
    return max(0.0, min(1.0, (dbfs - FLOOR_DBFS) / (CEILING_DBFS - FLOOR_DBFS)))


class Level:
    """The sound, smoothed for the eye: it rises at once and falls slowly,
    so that a word is a swell and not a flicker. `feed` is told the latest
    block, `step` is asked once per frame and told how long that frame was -
    so that the same second of sound looks the same however fast the window
    draws."""

    def __init__(self) -> None:
        self.target = 0.0
        self.value = 0.0

    def feed(self, dbfs: float) -> None:
        self.target = to_unit(dbfs)

    def step(self, dt: float) -> float:
        seconds = ATTACK_SECONDS if self.target > self.value else RELEASE_SECONDS
        rate = 1.0 - math.exp(-max(dt, 0.0) / seconds)
        self.value += (self.target - self.value) * rate
        self.value = max(0.0, min(1.0, self.value))
        return self.value


@dataclass(frozen=True, slots=True)
class Frame:
    """One picture of the orb: the colour, the core's size as a multiple of
    `CORE`, how much of the colour the core shows (1 = all; confirming
    blinks it towards the background), and where each ring's first dash
    starts, in degrees anticlockwise from three o'clock."""

    colour: RGB
    core_scale: float
    core_alpha: float
    angles: tuple[float, ...]


class Orb:
    """Where the rings are and how the core breathes, advanced by whoever
    holds the clock."""

    def __init__(self) -> None:
        self.angles = [0.0] * len(RINGS)
        self.level = Level()
        self._seconds = 0.0

    def advance(self, state: State, dt: float) -> Frame:
        self._seconds += dt
        multiple = SPEED[state]
        for index, ring in enumerate(RINGS):
            speed = abs(ring.speed) if state is State.RECONNECTING else ring.speed
            self.angles[index] = (self.angles[index] + speed * multiple * dt) % 360

        breath = 1 + BREATH * math.sin(2 * math.pi * self._seconds / BREATH_SECONDS)
        scale = breath * (1 + SWELL * self.level.step(dt))
        alpha = 1.0
        if state is State.CONFIRMING:
            wave = 0.5 + 0.5 * math.sin(2 * math.pi * self._seconds / BLINK_SECONDS)
            alpha = BLINK_LOW + (1 - BLINK_LOW) * wave
        return Frame(COLOURS[state], scale, alpha, tuple(self.angles))
