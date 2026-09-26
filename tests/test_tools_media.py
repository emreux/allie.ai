"""The media keys (2.2): a key press, because there is no API for "pause".

`_press` is the one place the keyboard is reached and `current_track` the one
place the media session is read; every test replaces both, and looks at which
key would have been pressed and what the model was told.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from allie.tools import media
from allie.tools.media import KEYS, media_control, set_volume_for

PREVIOUS = KEYS["previous"]
NEXT = KEYS["next"]


async def _nothing_named() -> str | None:
    return None


@pytest.fixture
def pressed(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []
    monkeypatch.setattr(media, "_press", seen.append)
    # Nothing on the machine running the tests is asked what it plays.
    monkeypatch.setattr(media, "current_track", _nothing_named)
    # The gap between the two presses of `previous` is real time; a test
    # about which keys were pressed has no use for a quarter of a second.
    monkeypatch.setattr(media, "PREVIOUS_GAP_SECONDS", 0.0)
    return seen


class Queue:
    """A player the way YouTube Music in Chrome was measured to behave
    (2026-09-25): one previous a few seconds into a track starts it over,
    one at its start goes to the track before; the session names the track
    playing, and is gone for a moment while a track changes."""

    def __init__(self, *tracks: str, at: int = -1, into: float = 9.0, gone: int = 2) -> None:
        self.tracks = list(tracks)
        self.index = at % len(tracks)
        self.into = into  # seconds into the current track
        self.gone = gone  # readings with no session, after a change
        self.pressed: list[int] = []
        self._missing = 0

    def press(self, code: int) -> None:
        self.pressed.append(code)
        if code == PREVIOUS and self.into <= 3 and self.index > 0:
            self._move(-1)
        elif code == PREVIOUS:
            self.into = 0.0  # started over
        elif code == NEXT and self.index < len(self.tracks) - 1:
            self._move(+1)

    def _move(self, step: int) -> None:
        self.index += step
        self.into = 0.0
        self._missing = self.gone

    async def track(self) -> str | None:
        if self._missing:
            self._missing -= 1
            return None
        return self.tracks[self.index]


@pytest.fixture
def quick(monkeypatch: pytest.MonkeyPatch) -> None:
    """The watch after a press, in milliseconds rather than seconds."""
    monkeypatch.setattr(media, "CHANGE_SECONDS", 0.05)
    monkeypatch.setattr(media, "SETTLE_SECONDS", 0.2)
    monkeypatch.setattr(media, "POLL_SECONDS", 0.001)


def playing(monkeypatch: pytest.MonkeyPatch, queue: Queue) -> Queue:
    monkeypatch.setattr(media, "_press", queue.press)
    monkeypatch.setattr(media, "current_track", queue.track)
    return queue


def test_it_is_a_safe_tool_whose_actions_are_an_enum() -> None:
    """The model is told the six actions rather than left to guess them."""
    assert media_control.risk == "safe"
    assert media_control.spec.parameters["required"] == ["action"]
    assert media_control.spec.parameters["properties"]["action"] == {
        "type": "string",
        "enum": ["play_pause", "next", "previous", "volume_up", "volume_down", "mute"],
    }
    assert list(KEYS) == media_control.spec.parameters["properties"]["action"]["enum"]


@pytest.mark.parametrize(
    ("action", "code"),
    [
        ("play_pause", 0xB3),
        ("volume_up", 0xAF),
        ("volume_down", 0xAE),
        ("mute", 0xAD),
    ],
)
async def test_each_action_presses_its_own_key(pressed: list[int], action: str, code: int) -> None:
    said = await media_control.run(action=action)

    assert pressed == [code]
    assert said == f"Pressed the {action} key."


# --------------------------------------------------------------------------
# Changing the track, watched through the media session (2026-09-25)
# --------------------------------------------------------------------------


async def test_previous_a_few_seconds_into_a_track_is_pressed_again_once_it_started_over(
    monkeypatch: pytest.MonkeyPatch, quick: None
) -> None:
    """One press only started the song over, which the session shows by not
    changing; the second goes back, while the song is at its start."""
    queue = playing(monkeypatch, Queue("A", "B", "C", into=9.0))

    said = await media_control.run(action="previous")

    assert queue.pressed == [PREVIOUS, PREVIOUS]
    assert said == "Went back to B (was C)."


async def test_previous_at_the_start_of_a_track_is_pressed_once(
    monkeypatch: pytest.MonkeyPatch, quick: None
) -> None:
    """The owner, 2026-09-25: from the third song to the first. A fixed
    second press went back two tracks whenever the first had already gone
    back one."""
    queue = playing(monkeypatch, Queue("A", "B", "C", into=1.0))

    said = await media_control.run(action="previous")

    assert queue.pressed == [PREVIOUS]
    assert said == "Went back to B (was C)."


async def test_previous_on_the_first_track_says_it_only_started_over(
    monkeypatch: pytest.MonkeyPatch, quick: None
) -> None:
    queue = playing(monkeypatch, Queue("A", "B", at=0, into=9.0))

    said = await media_control.run(action="previous")

    assert queue.pressed == [PREVIOUS, PREVIOUS]
    assert said.startswith("A started over")
    assert "no track before it" in said


async def test_next_names_the_track_it_went_to(
    monkeypatch: pytest.MonkeyPatch, quick: None
) -> None:
    queue = playing(monkeypatch, Queue("A", "B", at=0))

    said = await media_control.run(action="next")

    assert queue.pressed == [NEXT]
    assert said == "Skipped to B (was A)."


async def test_next_at_the_end_of_the_queue_says_nothing_changed(
    monkeypatch: pytest.MonkeyPatch, quick: None
) -> None:
    queue = playing(monkeypatch, Queue("A", "B", at=1))

    said = await media_control.run(action="next")

    assert queue.pressed == [NEXT]
    assert said.startswith("Pressed next, but B is still playing")


async def test_a_change_whose_new_track_is_not_named_yet_is_not_pressed_again(
    monkeypatch: pytest.MonkeyPatch, quick: None
) -> None:
    """The session went away and has not come back: the track is changing,
    so a second press would only go back one more."""
    queue = playing(monkeypatch, Queue("A", "B", "C", into=1.0, gone=10_000))

    said = await media_control.run(action="previous")

    assert queue.pressed == [PREVIOUS]
    assert said == "The track changed from C, but nothing names the new one yet."


async def test_previous_with_nothing_named_is_pressed_twice_as_before(
    pressed: list[int],
) -> None:
    """With no session to watch, the keys go as a hand would press them:
    one press only restarts a song a few seconds in (owner, 2026-09-13)."""
    said = await media_control.run(action="previous")

    assert pressed == [PREVIOUS, PREVIOUS]
    assert said == (
        "Pressed the previous key twice: once only restarts a track. Nothing "
        "reports what is playing, so which track is on now is not known."
    )


async def test_next_with_nothing_named_is_pressed_once(pressed: list[int]) -> None:
    said = await media_control.run(action="next")

    assert pressed == [NEXT]
    assert said == (
        "Pressed the next key. Nothing reports what is playing, so which track "
        "is on now is not known."
    )


def test_the_description_says_one_call_is_one_track() -> None:
    """The model that could not tell what one press had done pressed again,
    and went back two songs (owner, 2026-09-25)."""
    said = media_control.spec.description

    assert "track now playing" in said
    assert "one call" in said


async def test_the_two_presses_of_previous_are_a_moment_apart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sent back to back, the second press can land before the player has
    moved the position to zero, and restart the song a second time."""
    events: list[object] = []
    monkeypatch.setattr(media, "_press", events.append)
    monkeypatch.setattr(media, "current_track", _nothing_named)
    real_sleep = asyncio.sleep

    async def sleep(seconds: float) -> None:
        events.append(("slept", seconds))
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep)

    await media_control.run(action="previous")

    assert events == [0xB1, ("slept", media.PREVIOUS_GAP_SECONDS), 0xB1]
    assert 0.1 <= media.PREVIOUS_GAP_SECONDS <= 0.5


async def test_an_action_that_is_not_a_key_presses_nothing(pressed: list[int]) -> None:
    """A model that ignores the enum gets the list back, not a `KeyError`."""
    said = await media_control.run(action="stop")

    assert pressed == []
    assert said.startswith("No action called 'stop'")
    assert "play_pause" in said


def test_the_description_says_what_stops_the_music() -> None:
    """The user says "stop the music"; the model has to know that is play_pause."""
    said = media_control.spec.description

    assert "play_pause" in said
    assert "stop" in said


# --------------------------------------------------------------------------
# The volume as a number (plan.md D24)
# --------------------------------------------------------------------------


class FakeVolume:
    """A `Volume` that remembers what it was set to and the thread it was asked on."""

    def __init__(self, level: int = 65) -> None:
        self._level = level
        self.set: list[int] = []
        self.threads: set[int] = set()

    def level(self) -> int:
        self.threads.add(threading.get_ident())
        return self._level

    def set_level(self, percent: int) -> None:
        self.threads.add(threading.get_ident())
        self.set.append(percent)
        self._level = percent


async def test_set_volume_sets_the_number_and_says_what_it_was() -> None:
    volume = FakeVolume(65)

    said = await set_volume_for(volume).run(percent=30)

    assert volume.set == [30]
    assert said == "Volume set to 30 % (was 65 %)."


async def test_a_number_past_the_ends_is_clamped() -> None:
    volume = FakeVolume()

    assert (await set_volume_for(volume).run(percent=140)).startswith("Volume set to 100 %")
    assert (await set_volume_for(volume).run(percent=-3)).startswith("Volume set to 0 %")
    assert volume.set == [100, 0]


async def test_set_volume_asks_windows_off_the_loop() -> None:
    """COM is initialised per call on a worker thread (section 3.1 rule 4)."""
    volume = FakeVolume()

    await set_volume_for(volume).run(percent=10)

    assert volume.threads
    assert threading.get_ident() not in volume.threads


def test_set_volume_is_safe_and_sends_steps_to_media_control() -> None:
    """Two ways to change the volume, told apart in the description: a
    step is `media_control`, a number is this."""
    chosen = set_volume_for(FakeVolume())

    assert chosen.risk == "safe"
    assert chosen.spec.parameters["required"] == ["percent"]
    assert chosen.spec.parameters["properties"]["percent"]["type"] == "integer"
    assert "media_control" in chosen.spec.description
