"""Everything the assistant does with music and video (design.md section 3.6).

Two unrelated jobs share this file because both are "the media tools" to
whoever is reading the registry in the composition root.

**Controlling what is already playing** is `media_control`, and it is a
keyboard. Windows has no call that means "pause whatever is making noise"; it
has the media keys, which every player - Spotify, a browser tab, the Music app
- already listens to, so the tool presses one. `keybd_event` is the oldest way
to do that and still the one every player answers to, and it returns at once,
so it runs on the loop.

One key is pressed twice. Spotify and YouTube Music both read a single
"previous" a few seconds into a song as "start it over", and go to the track
before only when the position is already at zero - so the user who asked for
the previous song heard the same one again (owner, 2026-09-13). The second
press, a moment after the first, is what they would have done by hand. The
one case this gets wrong is a request made inside the first seconds of a
song, which then goes back two tracks; a spoken turn takes longer than that
to arrive, so it is rare enough not to be worth reading the player's position.

**Starting something** is the other three, and they are closures over a
`Player` (`media/player.py`) for the same reason `open_app` is a closure over
the app catalogue: what the model sees is read off the function's signature,
and the player is not something the model chooses.

The descriptions below do more work than most. A model that is not told
otherwise will answer "play X" by writing a `youtube.com/watch?v=` address of
its own invention - eleven characters it cannot possibly know - and YouTube
answers "This video isn't available anymore". Saying so in the description,
in the tool that offers the alternative, is what stops it; `open_url` says the
same thing from the other side.
"""

from __future__ import annotations

import asyncio
import ctypes
from typing import Annotated, Literal

from assistant.media.player import Player, service_keys
from assistant.tools.registry import Tool, tool

__all__ = [
    "KEYS",
    "PREVIOUS_GAP_SECONDS",
    "Action",
    "media_control",
    "open_media_for",
    "play_music_for",
    "play_video_for",
]

Action = Literal["play_pause", "next", "previous", "volume_up", "volume_down", "mute"]

# The virtual-key code of each, from winuser.h.
KEYS: dict[str, int] = {
    "play_pause": 0xB3,  # VK_MEDIA_PLAY_PAUSE
    "next": 0xB0,  # VK_MEDIA_NEXT_TRACK
    "previous": 0xB1,  # VK_MEDIA_PREV_TRACK
    "volume_up": 0xAF,  # VK_VOLUME_UP
    "volume_down": 0xAE,  # VK_VOLUME_DOWN
    "mute": 0xAD,  # VK_VOLUME_MUTE
}

KEY_UP = 0x0002  # KEYEVENTF_KEYUP

# Between the two presses of `previous`. Long enough for the player to have
# moved the position to zero after the first - a second press that lands
# before that restarts the song again - and short enough that nobody hears
# two events.
PREVIOUS_GAP_SECONDS = 0.25


def _press(code: int) -> None:
    """One press and release, as the keyboard would send it. Kept apart so a
    test can see which key without pressing it."""
    user32 = ctypes.windll.user32
    user32.keybd_event(code, 0, 0, 0)
    user32.keybd_event(code, 0, KEY_UP, 0)


@tool(risk="safe")
async def media_control(action: Action) -> str:
    """Controls whatever is playing, the way the media keys on a keyboard do.
    play_pause toggles between playing and paused - use it both to stop the
    music and to resume it; next and previous change the track; volume_up and
    volume_down step the system volume; mute silences it and unsilences it."""
    code = KEYS.get(action)
    if code is None:
        return f"No action called {action!r}; the actions are: {', '.join(KEYS)}."

    _press(code)
    if action == "previous":
        await asyncio.sleep(PREVIOUS_GAP_SECONDS)
        _press(code)
        return "Pressed the previous key twice: once only restarts the current track."
    return f"Pressed the {action} key."


def play_music_for(player: Player) -> Tool:
    """`play_music`, bound to the player that knows where music comes from."""

    @tool(risk="safe")
    async def play_music(
        query: Annotated[
            str,
            "The song, artist or album the user named, in their own words. Leave it empty "
            "when they asked for music without naming anything.",
        ] = "",
        service: Annotated[
            str,
            "Where to play it from - one of: " + service_keys() + ". Leave it empty unless "
            "the user named a service; their own default is used then.",
        ] = "",
    ) -> str:
        """Plays music. Use this for every request to hear music - "play some
        music", "put on X", "play X by Y" - and never open_url for one. This
        looks the song up and opens an address that starts playing it; a web
        address you write yourself opens a search the user still has to click,
        or a video identifier you cannot know and therefore invented. Pass the
        user's words as they said them: do not translate them, do not correct
        their spelling, and do not add the words "song" or "music" to them.
        The answer names the song that started - say which one it was."""
        return await player.play_music(query, service)

    return play_music


def play_video_for(player: Player) -> Tool:
    """`play_video`, bound to the player."""

    @tool(risk="safe")
    async def play_video(
        query: Annotated[
            str,
            "The video the user described, in their own words - the title as they said it.",
        ],
    ) -> str:
        """Opens a video on YouTube and starts it. Use this whenever the user
        wants to watch something - "open the video called X", "put X on
        YouTube" - and never open_url with a watch?v= address: a YouTube
        identifier is eleven characters you cannot know, and one you invent
        opens "This video isn't available anymore". This searches YouTube and
        opens the first result, which is the video the user meant. The answer
        names the video that opened - say which one it was, so the user can
        tell you if it was the wrong one."""
        return await player.play_video(query)

    return play_video


def open_media_for(player: Player) -> Tool:
    """`open_media`, bound to the player."""

    @tool(risk="safe")
    async def open_media(
        service: Annotated[str, "One of: " + service_keys() + "."],
    ) -> str:
        """Opens a music or video service without playing anything. Use it when
        the user asks for the service itself - "open YouTube", "open Spotify" -
        rather than to hear something; if they named something to play, use
        play_music or play_video instead. YouTube Music has no application on
        Windows and opens in the browser; Spotify opens its installed
        application where there is one and its website otherwise."""
        return await player.open_service(service)

    return open_media
