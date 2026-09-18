"""The one place this program hands something to Windows to open.

Four different things end up here - a Start Menu shortcut, a `ms-settings:`
page, a `spotify:` URI, a web address - and every one of them is opened by the
same two calls a double-click would use. A fifth, since 2026-09-11, is the
browser started with a window of its own (`media/window.py`); `start` is that
one call. They live in one module rather than
as private helpers of whoever needed them first, so that the media engine and
`tools/system.py` share a single seam: a test replaces `launch` and `browse`
here and nothing anywhere opens a window.

**The user's own browser is the point, not an implementation detail.**
`webbrowser.open` hands the address to the default browser, which - when it is
already running, as it always is - opens a tab in the profile the user is
signed in to. That signed-in profile is the whole reason a song starts by
itself: YouTube Music plays for a listener it knows and shows a landing page
to a stranger. Any browser this program drove itself would be a stranger,
however carefully its profile was arranged.

Both calls block. `os.startfile` returns once the shell has taken the request,
and taking it can be a store app waking up; `webbrowser.open` can wait on a
cold browser. Neither may be awaited on the event loop (design.md section 3.1
rule 4), so the async pair below is what callers use and the sync pair is what
tests replace.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import webbrowser
from collections.abc import Sequence

__all__ = ["browse", "launch", "open_address", "open_target", "start"]


def launch(target: str) -> None:
    """Hands `target` to the shell, the way a double-click does."""
    os.startfile(target)  # noqa: S606  # ours: a catalogue entry, a settings page or a media URI


def browse(address: str) -> bool:
    """Opens `address` in the default browser; `False` when no browser would."""
    return webbrowser.open(address)


def start(command: Sequence[str]) -> None:
    """Starts `command` and does not wait for it.

    The one caller is the media window, which starts the user's browser
    with an address. When the browser is already running - as it always is -
    the process hands the address over and exits at once, and the window it
    asked for belongs to the browser that was there before it.
    """
    subprocess.Popen(command)  # noqa: S603  # ours: the default browser's own path, from the registry


async def open_target(target: str) -> None:
    """`launch`, off the event loop."""
    await asyncio.to_thread(launch, target)


async def open_address(address: str) -> bool:
    """`browse`, off the event loop."""
    return await asyncio.to_thread(browse, address)
