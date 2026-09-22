"""The window on its own, for looking at (plan.md D20, task W1).

No assistant, no microphone, no session: the window is put up the way
`allie run` puts it up, and this script plays the state machine -
each state for three seconds, a turn now and then, a made-up sound level
while the user is "heard" and while the assistant "speaks", one notice,
the wizard's page once with a question of each kind. Ctrl+C in the
terminal, or Quit in the window, ends it.

    uv run python scripts/demo_window.py
    uv run python scripts/demo_window.py --seconds 5 --lang en

What to look at (spec section 3): the colours, the rings' speeds and
directions, the breathing, the swell, confirming's blink, the transcript's
rows, the buttons' words, the wizard's list with its filter, the secret
entry, dragging the window while it animates, 125-150 % scaling. What to
measure: the process's CPU in Task Manager while it animates (the target
is 2-4 % of one core), and that the loop below never waits - the printed
"slowest window call" stays in microseconds while the window is dragged.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import time

from allie import locales
from allie.app import State, Turn
from allie.ui.orb import SILENCE_DBFS
from allie.ui.window import Window

STATES = (
    State.IDLE,
    State.USER_SPEAKING,
    State.SPEAKING,
    State.CONFIRMING,
    State.ANNOUNCING,
    State.RECONNECTING,
    State.OFF,
)
TURNS = (
    Turn(heard="saat kaç", said="Saat on beş kırk iki."),
    Turn(heard="yarın dokuzda toplantı hatırlat", said="Hatırlatma kuruldu."),
    Turn(heard="bunu not al", said="Not kaydedildi.", tool_calls=1),
)


async def play(window: Window, seconds: float) -> None:
    """The state machine, pretended: states in a ring, sound while it
    would be heard, and the wizard's page once at the end."""
    loop = asyncio.get_running_loop()
    window.starting()
    await asyncio.sleep(1.0)
    window.state(State.IDLE)
    window.session(True)
    turns = 0
    while True:
        for state in STATES:
            window.state(state)
            if state is State.OFF:
                window.hands_free(False)
            started = loop.time()
            slowest = 0.0
            while loop.time() - started < seconds:
                before = time.perf_counter()
                if state in (State.USER_SPEAKING, State.SPEAKING, State.ANNOUNCING):
                    # A voice: a slow wave with a fast wobble, -10..-50 dBFS.
                    t = loop.time()
                    window.level(-30 + 20 * math.sin(t * 3) * abs(math.sin(t * 11)))
                else:
                    window.level(SILENCE_DBFS)
                slowest = max(slowest, time.perf_counter() - before)
                await asyncio.sleep(0.05)
            if state is State.OFF:
                window.hands_free(True)
            if state is State.SPEAKING:
                window.turn(TURNS[turns % len(TURNS)])
                turns += 1
            print(f"{state:<14} slowest window call {slowest * 1e6:.0f} us", flush=True)
        window.notice("The microphone is quiet: -45 dBFS reached the server, under -40 dBFS.")
        window.session(False)
        await asyncio.sleep(seconds)
        await wizard_once(window)
        window.loading()
        await asyncio.sleep(1.0)
        window.state(State.IDLE)
        window.session(True)


async def wizard_once(window: Window) -> None:
    from allie.setup_wizard import Option

    window.wizard(True)
    window.say("The assistant answers through an AI provider, using your own API key.")
    chosen = await window.ask(
        "choose",
        "Which model should answer?",
        [Option(f"model-{index}", f"gemini-3.8-live-{index:02d}") for index in range(40)],
    )
    window.say(f"chosen: {chosen}")
    secret = await window.ask("secret", "Paste your API key (nothing is shown as you type)", ())
    window.say(f"key of {len(secret or '')} characters")
    voice = await window.ask("ask", "Which voice should the model speak in?", ())
    window.say(f"voice: {voice!r}")
    await asyncio.sleep(1.5)
    window.wizard(False)


async def main(seconds: float, lang: str) -> None:
    loop = asyncio.get_running_loop()
    ended = asyncio.Event()
    pack = locales.load(lang)
    window = Window(
        pack,
        loop=loop,
        on_toggle=lambda: print("toggle pressed", flush=True),
        on_quit=ended.set,
        on_settings=lambda: print("settings pressed", flush=True),
        tray=False,
    )
    window.start()
    playing = asyncio.create_task(play(window, seconds))
    try:
        await ended.wait()
    finally:
        playing.cancel()
        window.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=3.0, help="how long each state is shown")
    parser.add_argument("--lang", default="tr", help="the locale pack to word the window in")
    args = parser.parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main(args.seconds, args.lang))
