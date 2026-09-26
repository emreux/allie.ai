"""What Windows says is playing: the session the media keys and the pause reach.

Nothing here pauses anything: a test that did would stop whatever the machine
running it is playing. What is checked is that the WinRT projection is whole,
and that reading the current track never raises.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from loguru import logger

from allie.media.now_playing import current_track, track_name


def test_the_projection_that_awaiting_a_session_needs_is_installed() -> None:
    """`request_async()` answers with an `IAsyncOperation`, and awaiting one
    imports `winrt.windows.foundation`, which is a package of its own. It was
    missing from 2026-09-10 to 2026-09-25: every pause before a song failed
    with `ModuleNotFoundError` and a debug line, and the song from before
    played on under the new one."""
    importlib.import_module("winrt.windows.foundation")


async def test_the_current_track_is_read_without_a_failure() -> None:
    """Whatever this machine is playing, or nothing: a name or `None`, and
    no line saying the session could not be reached or read."""
    failures: list[str] = []
    sink = logger.add(failures.append, level="DEBUG", filter="allie.media.now_playing")
    try:
        found = await current_track()
    finally:
        logger.remove(sink)

    assert found is None or (isinstance(found, str) and found)
    assert failures == []


@pytest.mark.parametrize(
    ("title", "artist", "name"),
    [
        ("Kop Gel Günahlarından", "Müslüm Gürses", "Müslüm Gürses - Kop Gel Günahlarından"),
        ("HDCrazySportsTR", "", "HDCrazySportsTR"),
        (" Tamam Aşkım ", None, "Tamam Aşkım"),
        ("", "Someone", None),
        (None, None, None),
    ],
)
def test_a_session_s_properties_are_named_the_way_a_track_is(
    title: str | None, artist: str | None, name: str | None
) -> None:
    assert track_name(SimpleNamespace(title=title, artist=artist)) == name
