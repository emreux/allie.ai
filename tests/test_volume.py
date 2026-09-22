"""The system volume as a number (plan.md D24): Windows' default playback
endpoint through Core Audio, read and set as a percentage.

COM is reached through one context manager, `_endpoint`, and every test
but the last replaces it with an endpoint that remembers what it was told.
The last one asks the real thing, and is skipped where there is no sound
card to ask.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from allie.audio import volume as module
from allie.audio.volume import SystemVolume, to_percent, to_scalar


class FakeEndpoint:
    """`IAudioEndpointVolume` as far as this module calls it."""

    def __init__(self, scalar: float = 0.65, muted: bool = False) -> None:
        self.scalar = scalar
        self.muted = muted
        self.set: list[float] = []
        self.mutes: list[bool] = []

    def GetMasterVolumeLevelScalar(self) -> float:  # noqa: N802 - COM's spelling
        return self.scalar

    def SetMasterVolumeLevelScalar(self, level: float, context: Any) -> None:  # noqa: N802
        self.set.append(level)
        self.scalar = level

    def GetMute(self) -> bool:  # noqa: N802
        return self.muted

    def SetMute(self, muted: bool, context: Any) -> None:  # noqa: N802
        self.mutes.append(muted)
        self.muted = muted


def volume_over(endpoint: FakeEndpoint) -> tuple[SystemVolume, list[str]]:
    """A `SystemVolume` whose COM is the fake, and the order of what happened."""
    events: list[str] = []

    @contextmanager
    def opened() -> Iterator[FakeEndpoint]:
        events.append("open")
        try:
            yield endpoint
        finally:
            events.append("close")

    return SystemVolume(endpoint=opened), events


def test_a_scalar_and_a_percentage_are_one_number_read_two_ways() -> None:
    assert to_percent(0.3) == 30
    assert to_percent(0.654) == 65
    assert to_percent(1.7) == 100
    assert to_percent(-0.1) == 0
    assert to_scalar(30) == 0.3
    assert to_scalar(140) == 1.0
    assert to_scalar(-5) == 0.0


def test_the_level_is_read_as_a_percentage_and_the_endpoint_is_closed_after() -> None:
    endpoint = FakeEndpoint(scalar=0.3)
    volume, events = volume_over(endpoint)

    assert volume.level() == 30
    assert events == ["open", "close"]


def test_setting_a_level_writes_the_scalar() -> None:
    endpoint = FakeEndpoint()
    volume, _ = volume_over(endpoint)

    volume.set_level(30)

    assert endpoint.set == [0.3]
    assert endpoint.mutes == []


def test_setting_a_level_unmutes_a_muted_endpoint() -> None:
    """ "Sesi yüzde otuza al" on a muted machine means "and let me hear it"."""
    endpoint = FakeEndpoint(muted=True)
    volume, _ = volume_over(endpoint)

    volume.set_level(30)

    assert endpoint.mutes == [False]


def test_the_endpoint_is_closed_even_when_the_call_fails() -> None:
    class Broken(FakeEndpoint):
        def GetMasterVolumeLevelScalar(self) -> float:  # noqa: N802
            raise OSError("the device went")

    volume, events = volume_over(Broken())

    with pytest.raises(OSError):
        volume.level()
    assert events == ["open", "close"]


def test_the_real_endpoint_answers_a_percentage() -> None:
    """The one test that asks Windows. Skipped where COM or a sound card
    is missing; on the owner's machine it is the proof the vtable order
    of the declarations below is right."""
    pytest.importorskip("comtypes")
    try:
        level = SystemVolume().level()
    except (OSError, ValueError) as failure:  # no endpoint, no audio service
        pytest.skip(f"no playback endpoint to ask: {failure}")
    assert 0 <= level <= 100
    assert module.CLSID_DEVICE_ENUMERATOR
