"""Pausing whatever is already playing, without pressing a key.

Every song this assistant starts opens a new browser tab, so without this the
tab from a minute ago and the new one play over each other. The media key
looks like the fix and is the wrong one: `VK_MEDIA_PLAY_PAUSE` is a *toggle*,
and a toggle sent when nothing happens to be playing **starts** something -
usually the last thing the user listened to, which is precisely what they did
not ask for. `media_control` in `tools/media.py` presses that key because a
user who says "pause" means the toggle; this does not, because a user who
says "play X" does not.

Windows keeps the list of what is playing in
`GlobalSystemMediaTransportControlsSessionManager` - the same list the volume
flyout draws - and asking the current session to pause is unambiguous: it
pauses if it is playing and does nothing if it is not.

**Nothing here is allowed to matter.** A machine with no session at all, a
WinRT call that refuses, a Windows that answers differently: every one of
them ends as `False` and a line in the log. Failing to pause the last song is
a small annoyance; failing to start the new one would be the whole request.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

__all__ = ["PLAYING", "pause_current"]

# `GlobalSystemMediaTransportControlsSessionPlaybackStatus.PLAYING`. Compared
# as a number so that nothing outside this module has to import WinRT.
PLAYING = 4


async def pause_current() -> bool:
    """Pauses the session Windows considers current. `True` if one was paused.

    `False` covers every uninteresting case together - nothing was playing,
    there is no session, WinRT would not answer - because the caller does the
    same thing in all of them: get on with playing the new song.
    """
    try:
        # Imported here rather than at the top of the module: WinRT projection
        # packages are the sort of dependency that can be missing or broken on
        # a particular machine, and that must not be able to stop the assistant
        # from starting.
        from winrt.windows.media.control import (
            GlobalSystemMediaTransportControlsSessionManager as Sessions,
        )
    except ImportError as failure:
        logger.debug("nothing to pause through: {}", failure)
        return False

    try:
        # Typed `Any` on purpose. The projection declares `get_current_session`
        # as returning a session, and it returns `None` on any machine where
        # nothing is playing - which is most of them, most of the time. Taking
        # the declaration at its word would make the check below unreachable
        # and the call below it a crash.
        session: Any = (await Sessions.request_async()).get_current_session()
        if session is None:
            return False
        if _status(session) != PLAYING:
            # Paused, stopped, or a session that only reports what it is
            # showing. Pausing it would be pausing nothing.
            return False
        return bool(await session.try_pause_async())
    except asyncio.CancelledError:
        raise
    except Exception as failure:
        logger.debug("what is playing could not be paused: {}", failure)
        return False


def _status(session: Any) -> int:
    """The session's playback status as a plain number, or -1 if it has none."""
    info = session.get_playback_info()
    return -1 if info is None else int(info.playback_status)
