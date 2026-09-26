"""Everything the assistant does with music and video (design.md section 3.6).

Two unrelated jobs share this file because both are "the media tools" to
whoever is reading the registry in the composition root.

**Controlling what is already playing** is `media_control`, and it is a
keyboard. Windows has no call that means "pause whatever is making noise"; it
has the media keys, which every player - Spotify, a browser tab, the Music app
- already listens to, so the tool presses one. `keybd_event` is the oldest way
to do that and still the one every player answers to, and it returns at once,
so it runs on the loop.

**Next and previous are watched** (2026-09-25). Spotify and YouTube Music
both read a single "previous" a few seconds into a song as "start it over",
and go to the track before only near its start - so the user who asked for
the previous song heard the same one again (owner, 2026-09-13). From then
until 2026-09-25 the key was pressed twice, a quarter of a second apart, and
that went wrong both ways for the owner: twice nothing moved, and once a
request went from the third song to the first. The position that would
settle it is not to be had - Chrome reports 0 for it whatever is playing -
but the track's name is (`media/now_playing.py`): a track that changes
leaves the media session for a moment and comes back as another name, and
one that only started over changes nothing. So `previous` is pressed once,
the name is watched, and it is pressed again only when nothing changed.

And the answer names the track now playing. "Pressed the key twice" told the
model nothing it could check, so it said "let me look" and pressed again -
two more tracks back.

**Setting the volume to a number** is `set_volume` (2026-09-21, D24), a
closure over a `Volume` (`audio/volume.py`): the keys step, this one lands
on the number the user said.

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

from allie.audio.volume import Volume
from allie.media.now_playing import current_track
from allie.media.player import Player, service_keys
from allie.tools.registry import Tool, tool

__all__ = [
    "CHANGE_SECONDS",
    "KEYS",
    "POLL_SECONDS",
    "PREVIOUS_GAP_SECONDS",
    "SETTLE_SECONDS",
    "Action",
    "media_control",
    "open_media_for",
    "play_music_for",
    "play_video_for",
    "set_volume_for",
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

# Between the two presses of `previous` when nothing names what is playing,
# so that nothing can be watched. Long enough for the player to have moved
# the position to zero after the first - a second press that lands before
# that restarts the song again - and short enough that nobody hears two
# events.
PREVIOUS_GAP_SECONDS = 0.25

# How long a press is watched for a change before it counts as having
# started the track over. Measured 2026-09-25, YouTube Music in Chrome: the
# session is gone 0.08-0.26 s after a press that changes the track, and a
# restart changes nothing for as long as it was watched (3 s). A second
# press after this is still near the start of the track, where it goes back.
CHANGE_SECONDS = 1.5
# How long a changing track is waited for to name itself: 0.58-1.14 s
# measured, a slow network given the rest.
SETTLE_SECONDS = 4.0
POLL_SECONDS = 0.1


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
    music and to resume it; next and previous move one track and answer with
    the track now playing, so one call is one track: tell the user what is
    playing rather than calling again to check; volume_up and volume_down
    step the system volume; mute silences it and unsilences it."""
    code = KEYS.get(action)
    if code is None:
        return f"No action called {action!r}; the actions are: {', '.join(KEYS)}."
    if action in ("next", "previous"):
        return await _change_track(action, code)

    _press(code)
    return f"Pressed the {action} key."


async def _change_track(action: str, code: int) -> str:
    """Presses next or previous, and says which track is playing after it."""
    before = await current_track()
    _press(code)
    if before is None:
        # Nothing names what is playing, so no change can be seen: the keys
        # as a hand would press them, and no claim about where they landed.
        unknown = "Nothing reports what is playing, so which track is on now is not known."
        if action == "next":
            return f"Pressed the next key. {unknown}"
        await asyncio.sleep(PREVIOUS_GAP_SECONDS)
        _press(code)
        return f"Pressed the previous key twice: once only restarts a track. {unknown}"

    if action == "next":
        changed, now = await _watch(before, SETTLE_SECONDS)
    else:
        changed, now = await _watch(before, CHANGE_SECONDS)
        if not changed:
            # A few seconds in, the player started the track over; the one
            # before is a second press away while this one is at its start.
            _press(code)
            changed, now = await _watch(before, SETTLE_SECONDS)

    if not changed:
        if action == "next":
            return f"Pressed next, but {before} is still playing: there may be no track after it."
        return (
            f"{before} started over, and a second press did not go back: there may be "
            "no track before it."
        )
    if now is None:
        return f"The track changed from {before}, but nothing names the new one yet."
    return f"{'Skipped to' if action == 'next' else 'Went back to'} {now} (was {before})."


async def _watch(before: str, seconds: float) -> tuple[bool, str | None]:
    """Whether the track changed from `before` after a press, and what the
    session names now.

    A change shows as the session going away or naming another track. Once
    it has gone, the wait is for a name, up to `SETTLE_SECONDS`: the same
    name coming back is a track that only started over; none coming back is
    a change not yet named.
    """
    loop = asyncio.get_running_loop()
    start = loop.time()
    gone = False
    while True:
        now = await current_track()
        if now is not None and (now != before or gone):
            return now != before, now
        gone = gone or now is None
        if loop.time() - start >= (SETTLE_SECONDS if gone else seconds):
            return gone, None
        await asyncio.sleep(POLL_SECONDS)


def set_volume_for(volume: Volume) -> Tool:
    """`set_volume`, bound to the endpoint that knows the level."""

    @tool(risk="safe")
    async def set_volume(
        percent: Annotated[int, "The level to set, from 0 to 100."],
    ) -> str:
        """Sets the system volume to a number: "sesi yüzde otuza al" is 30,
        "sesi yarıya indir" is 50. Use media_control's volume_up and
        volume_down for one step up or down, and this when the user names a
        level. A muted machine is unmuted. Answers with the level set and
        the one before it."""
        wanted = max(0, min(100, percent))

        def change() -> int:
            before = volume.level()
            volume.set_level(wanted)
            return before

        # COM, on a worker thread, per call (section 3.1 rule 4).
        before = await asyncio.to_thread(change)
        return f"Volume set to {wanted} % (was {before} %)."

    return set_volume


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
