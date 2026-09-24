"""The window: the product's face (plan.md D20, D33; specs docs/specs/2026-09-19-window-design.md
and docs/specs/2026-09-24-window-redesign-design.md).

`allie run` opens it. An orb whose colour is the state and whose pulse is
the sound, the state's word under it, the session and its minutes on the
orb's corners, three buttons - and under them the conversation, folded
away like an accordion until it is asked for. The setup wizard has a page
of its own, shown instead, when there is nothing set up yet or the settings
button was pressed. It owns no behaviour: what it shows is what `on_state`,
`on_mode`, `on_session` and `on_turn` say, like the status line and the
tray, and what its buttons do is what the key, the tray and `allie setup`
already do.

**Tk runs on a thread of its own**, like pystray (`ui/tray.py`). Two
rules make that safe, and both are the tray's:

1. The loop never calls Tk and never waits on the window. Every method
   the loop calls puts a small tuple on a `queue.SimpleQueue` and
   returns; the Tk thread drains the queue every `TICK_MS` and draws.
   Dragging a window blocks Tk's loop on Windows until the mouse is let
   go - on the loop's thread that would freeze the voice; and Tk's own
   cross-thread marshalling blocks the caller the same way, which is
   why not even a `root.after` is called from the loop.
2. The window reaches the loop only through `loop.call_soon_threadsafe`:
   the buttons, and the wizard's answers, which resolve futures the loop
   created.

**Three pieces.** `View` is the window as data - what the labels say,
what the transcript holds, whether it is unfolded, where the orb is - and
knows no Tk, so that the tests read it and the Tk half only draws it.
`Window` is the loop's side: the `Screen` of `ui/status.py`, the queue, the
thread. `TkPanel` is the Tk half: widgets, the orb on a canvas, the tick.

**The look (D33).** The orb's colour is the state and the only colour there
is: everything around it is a neutral instrument bezel - a graphite ground
with no blue in it, so that green, purple and yellow sit on it as well as
the idle blue does; chalk and graphite ink; the state's word tinted with the
state's colour. Bahnschrift, Windows' DIN, for what the machine says;
Segoe UI for what people said. Windows' own dark title bar and its own
glyphs on the buttons. One filled button on a page.

**The accordion (D33).** At every start the window is the orb and nothing
more: the readouts, the orb, the state's word, the buttons, and a
"Conversation" header. Pressing the header grows the window downwards to
`WINDOW_SIZE` and shows the transcript; pressing it again folds it away.
The buttons never move. A notice unfolds it, because a notice is there to be
read ("could not start - the reason is below"). The wizard's page takes the
full height. The user still cannot resize the window (U2): its height is the
accordion's and the wizard's to set.

**The orb** is `ui/orb.py`'s numbers drawn on a canvas: the glow Pillow
paints once per colour (Tk cannot blur), the core an oval that breathes,
the rings arcs whose `start` turns every tick, the dots ovals moved.

**Sixty frames a second** (task U1). The tick asks for fifteen milliseconds
and subtracts the time the last one took, so the period is the frame and
not the frame plus the work; Windows is asked for a one millisecond timer
while the window is up, because at its usual 15.6 ms granularity a sixteen
millisecond wait is rounded up to thirty-one and sixty frames quietly
become thirty. Nothing here is a clock the animation reads - every frame is
still told how long it was.

**The wizard's page** is the `Prompter` of `setup_wizard.py` on a window:
`WindowPrompter` posts each question and awaits a future the Tk thread
resolves when Continue is pressed - so the wizard's checks (the key, the
model list, the tool probe, the microphone list) come to the window
without a line of the wizard changing. The settings list (D32) is the same
page: a `menu` question is a list of two-line rows without the filter box,
and its buttons say Change and Close.

**No sentence is written here.** `TEXT` is the end of the chain; the
state labels are the status line's, the button words the tray's.
"""

from __future__ import annotations

import asyncio
import gc
import math
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from PIL import Image, ImageDraw, ImageFilter, ImageTk

from allie.app import State, Turn
from allie.config import APP_TITLE
from allie.locales import Locale
from allie.setup_wizard import Option, wording
from allie.ui import logo, orb
from allie.ui.orb import RGB, Frame, Orb
from allie.ui.status import TEXT as STATUS_TEXT
from allie.ui.status import QuietNotice, SessionMinutes, label_key
from allie.ui.tray import TEXT as TRAY_TEXT

__all__ = [
    "BACKGROUND",
    "TEXT",
    "TICK_MS",
    "TRANSCRIPT_ROWS",
    "Answer",
    "Button",
    "Field",
    "Fold",
    "Message",
    "Panel",
    "Rows",
    "Style",
    "Switch",
    "TkPanel",
    "View",
    "Window",
    "WindowError",
    "WindowPrompter",
    "WizardPage",
    "plate",
    "split_row",
    "system_panel",
]

TEXT: dict[str, str] = {
    "window_loading": "Starting...",
    "window_failed": "Could not start - the reason is below.",
    "window_settings": "Settings",
    "window_setting_up": "Settings are being changed; the assistant starts again afterwards.",
    # The accordion's header under the buttons (D33): press it to unfold the
    # conversation, press it again to fold it away.
    "window_conversation": "Conversation",
    "wizard_continue": "Continue",
    "wizard_cancel": "Cancel",
    "wizard_filter": "Type to filter the list",
    # Under the settings list (D32): change the chosen row, or close the
    # list and start the assistant again with what was saved.
    "wizard_change": "Change",
    "wizard_close": "Close",
}

# What the status line and the tray already say, borrowed under their keys.
STATUS_KEYS = (
    "loading_speech",
    "checking_model",
    "session_open",
    "session_closed",
    "session_minutes",
    "microphone_quiet",
)
TRAY_KEYS = ("tray_not_listening", "tray_stop_listening", "tray_start_listening", "tray_quit")

# Fifteen rather than sixteen: Windows' usual timer granularity is 15.6 ms,
# and a wait is rounded *up* to the next tick of it - sixteen would come
# back at 31 ms and draw thirty frames a second. Fifteen lands on 15.6 ms
# without the fine timer and on 15 ms with it; both are over sixty.
TICK_MS = 15
TRANSCRIPT_ROWS = 200
# The window unfolded, and with the wizard's page up (U2, D33). Folded, it is
# as tall as what it holds asks for - measured, not written down, because
# the screen's scaling and the fonts decide it.
WINDOW_SIZE = (420, 640)
ORB_HEIGHT = 300
# How long the accordion takes to open or close.
FOLD_SECONDS = 0.2
# How long `start` waits for the Tk thread to put the window up, and
# `stop` for it to come down.
START_SECONDS = 5.0
STOP_SECONDS = 3.0
# A tick after a long sleep (the window hidden, the machine asleep) gets
# this much time at most, so the rings do not spin through a whole night.
MAX_DT = 0.25

Message = tuple[Any, ...]
Answer = asyncio.Future[str | None]


def split_row(label: str) -> tuple[str, str]:
    """A settings row as (what it is, its value): the packs word every row
    of the list as "<what>: {value}". Split at the first colon only - a
    microphone's name can hold one - and a row worded otherwise stays one
    line, `("", label)`."""
    what, colon, value = label.partition(": ")
    return (what, value) if colon else ("", label)


@dataclass
class WizardPage:
    """The wizard's page: what it has said so far, and the question open now."""

    lines: list[str] = field(default_factory=list)
    kind: str = ""
    question: str = ""
    options: list[Option] = field(default_factory=list)
    answer: Answer | None = None

    @property
    def open(self) -> bool:
        return self.answer is not None


class View:
    """The window as data. Applied to by one thread only - the one that
    drains the queue - and read by the Tk half and the tests."""

    def __init__(self, locale: Locale, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.said = {key: locale.say(key, default) for key, default in TEXT.items()}
        self._labels = {
            state: locale.say(label_key(state), STATUS_TEXT[label_key(state)]) for state in State
        }
        self._status = {key: locale.say(key, STATUS_TEXT[key]) for key in STATUS_KEYS}
        self._tray = {key: locale.say(key, TRAY_TEXT[key]) for key in TRAY_KEYS}
        self.state = State.IDLE
        # Live until the capture says otherwise, which it does as it starts.
        self.listening = True
        self.session = SessionMinutes(clock=clock)
        # Shown instead of the state's label until the first state arrives:
        # loading, checking the model, starting again after the wizard.
        self.phase: str | None = self.said["window_loading"]
        # (kind, text) with kind one of "you", "searched", "it", "notice".
        self.rows: list[tuple[str, str]] = []
        # Bumped at every row, so that the Tk half redraws the transcript
        # only when it changed.
        self.version = 0
        # The accordion (D33): folded at every start - the window is the orb.
        self.log_open = False
        self.wizard: WizardPage | None = None
        self.orb = Orb()

    def apply(self, message: Message) -> None:
        match message:
            case ("phase", str(key)):
                self.phase = self._status[key] if key in self._status else self.said[key]
            case ("state", State() as state):
                self.state = state
                self.phase = None
                # Whatever was heard last is over; the swell lets go.
                self.orb.level.feed(orb.SILENCE_DBFS)
            case ("mode", bool(listening)):
                self.listening = listening
            case ("session", bool(opened)):
                self.session.told(opened)
            case ("turn", Turn() as finished):
                self._turn(finished)
            case ("notice", str(text)):
                self._add("notice", text)
                # A notice is there to be read - "the reason is below" - so
                # it unfolds the conversation it was written into.
                self.log_open = True
            case ("level", dbfs):
                self.orb.level.feed(float(dbfs))
            case ("say", str(text)):
                if self.wizard is not None:
                    self.wizard.lines.append(text)
            case ("question", str(kind), str(text), list(options), answer):
                if self.wizard is None:
                    self.wizard = WizardPage()
                self.wizard.kind = kind
                self.wizard.question = text
                self.wizard.options = options
                self.wizard.answer = answer
            case ("wizard", bool(opened)):
                self.wizard = WizardPage() if opened else None
            case _:
                raise ValueError(f"unknown window message {message!r}")

    # -- what the labels say -------------------------------------------------

    def label(self) -> str:
        if self.phase is not None:
            return self.phase
        if self.state is State.IDLE and not self.listening:
            return self._tray["tray_not_listening"]
        return self._labels[self.state]

    def readouts(self) -> tuple[str, str]:
        """What the orb's two top corners say: whether a session is open,
        and the minutes of this run."""
        which = self._status["session_open" if self.session.open else "session_closed"]
        minutes = self._status["session_minutes"].format(minutes=int(self.session.minutes))
        return which, minutes

    def switch_label(self) -> str:
        return self._tray["tray_stop_listening" if self.listening else "tray_start_listening"]

    def quit_label(self) -> str:
        return self._tray["tray_quit"]

    def page_buttons(self) -> tuple[str, str]:
        """The two buttons under the page: Change and Close on the settings
        list, Continue and Cancel under every other question."""
        if self.wizard is not None and self.wizard.kind == "menu":
            return self.said["wizard_change"], self.said["wizard_close"]
        return self.said["wizard_continue"], self.said["wizard_cancel"]

    # -- the accordion ---------------------------------------------------------

    def toggle_log(self) -> None:
        """The header was pressed, on the Tk thread - the one that applies."""
        self.log_open = not self.log_open

    def wants_room(self) -> bool:
        """Whether the window should be at its full height: the
        conversation unfolded, or the wizard's page up."""
        return self.log_open or self.wizard is not None

    # ------------------------------------------------------------------------

    def frame(self, dt: float) -> Frame:
        return self.orb.advance(self.state, dt)

    def take_answer(self) -> Answer | None:
        """The open question's future, closing the question; `None` when
        nothing is open. The Tk half resolves what it takes through
        `Window.answer`."""
        page = self.wizard
        if page is None or page.answer is None:
            return None
        answer, page.answer, page.kind = page.answer, None, ""
        return answer

    def _turn(self, finished: Turn) -> None:
        # The status line's rule: a turn nobody had is no row.
        if not finished.heard:
            return
        self._add("you", finished.heard)
        if finished.searched:
            # What Google searched for the answer (D29), between the two.
            self._add("searched", " · ".join(finished.searched))
        self._add("it", finished.said)

    def _add(self, kind: str, text: str) -> None:
        self.rows.append((kind, text))
        del self.rows[:-TRANSCRIPT_ROWS]
        self.version += 1


class Panel(Protocol):
    """The Tk half, from the loop's side: something that runs on the
    window's thread until told to quit."""

    def run(self) -> None: ...


PanelFactory = Callable[[View, "Window"], Panel]


class WindowError(RuntimeError):
    """The window could not be put up."""


def _resolve(answer: Answer, value: str | None) -> None:
    if not answer.done():
        answer.set_result(value)


class Window:
    """The loop's side of the window: a `Screen`, plus what the wizard and
    the tray need of it. Everything the loop calls is a queue put."""

    def __init__(
        self,
        locale: Locale,
        *,
        loop: asyncio.AbstractEventLoop,
        on_toggle: Callable[[], None],
        on_quit: Callable[[], None],
        on_settings: Callable[[], None],
        tray: bool,
        panel: PanelFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.view = View(locale, clock=clock)
        self._quiet = QuietNotice(
            locale.say("microphone_quiet", STATUS_TEXT["microphone_quiet"]), self.notice
        )
        self._queue: queue.SimpleQueue[Message] = queue.SimpleQueue()
        self._loop = loop
        self._on_toggle = on_toggle
        self._on_quit = on_quit
        self._on_settings = on_settings
        self._tray = tray
        self._panel = panel
        self._thread: threading.Thread | None = None
        self._up = threading.Event()
        self._failure: Exception | None = None
        # Set on the Tk thread when quit was pressed: a question asked after
        # that is answered "walked away" at once rather than shown.
        self._quitting = False

    # -- the thread ------------------------------------------------------------

    def start(self) -> None:
        """Puts the window up on a thread of its own and returns once it is
        there - so that the first question is not posted to nothing."""
        self._thread = threading.Thread(target=self._serve, name="window", daemon=True)
        self._thread.start()
        if not self._up.wait(START_SECONDS):
            raise WindowError("the window did not come up")
        if self._failure is not None:
            raise WindowError(f"the window could not be built: {self._failure}") from self._failure

    def stop(self) -> None:
        """Takes the window down and waits for its thread."""
        self._post(("quit",))
        if self._thread is not None:
            self._thread.join(STOP_SECONDS)
            self._thread = None

    def _serve(self) -> None:
        factory = self._panel if self._panel is not None else system_panel
        try:
            panel = factory(self.view, self)
        except Exception as failure:
            self._failure = failure
            self._up.set()
            return
        self._up.set()
        try:
            panel.run()
        finally:
            # Tk's objects are let go here, on the thread that made them.
            # A widget tree holds itself up - every parent knows its
            # children and every child its parent - so it is the garbage
            # collector that frees it, and the collector runs on whichever
            # thread happens to allocate next, which at the end of the
            # process is the main one. Tcl kills the process outright
            # ("async handler deleted by the wrong thread") when its
            # interpreter is deleted anywhere but where it was born, so the
            # panel is dropped and collected here rather than by chance.
            del panel
            gc.collect()

    # -- Screen (ui/status.py), called on the loop ----------------------------

    def starting(self) -> None:
        self._post(("phase", "loading_speech"))

    def checking_model(self) -> None:
        self._post(("phase", "checking_model"))

    def notice(self, message: str) -> None:
        self._post(("notice", message))

    def state(self, state: State) -> None:
        self._post(("state", state))

    def session(self, open: bool) -> None:
        self._post(("session", open))

    def microphone_level(self, dbfs: float | None) -> None:
        """The status line's rule, and the same code as the line (D18): a
        quiet microphone is said once per stretch of quiet, as a notice."""
        self._quiet.level(dbfs)

    def hands_free(self, listening: bool) -> None:
        self._post(("mode", listening))

    def turn(self, finished: Turn) -> None:
        self._post(("turn", finished))

    def level(self, dbfs: float) -> None:
        """The sound, one number per block, from the loop or the player's
        worker thread - a queue put is safe from either."""
        self._post(("level", dbfs))

    # -- the rest, called on the loop -----------------------------------------

    def loading(self) -> None:
        self._post(("phase", "window_loading"))

    def failed(self) -> None:
        """The start stopped at something the user can fix; the notice
        below says what. Replaces whatever the line said before - a
        "ready" over an assistant that never started is read as one."""
        self._post(("phase", "window_failed"))

    def show(self) -> None:
        """Brings a hidden window back - the tray's line."""
        self._post(("show",))

    def wizard(self, opened: bool) -> None:
        """Turns the wizard's page on or off."""
        self._post(("wizard", opened))

    def say(self, text: str) -> None:
        self._post(("say", text))

    def ask(self, kind: str, text: str, options: Sequence[Option]) -> Answer:
        """A question for the page: `kind` is `choose`, `menu`, `secret` or
        `ask`. The future resolves with the answer, or `None` for walking
        away - which, on the settings list, is closing it."""
        answer: Answer = self._loop.create_future()
        if self._quitting:
            answer.set_result(None)
            return answer
        self._post(("question", kind, text, list(options), answer))
        return answer

    def pending(self) -> int:
        """How much the Tk half has not drained yet (for the tests)."""
        return self._queue.qsize()

    def _post(self, message: Message) -> None:
        self._queue.put(message)

    # -- called on the Tk thread, by the panel --------------------------------

    def drain(self) -> list[Message]:
        taken: list[Message] = []
        while True:
            try:
                taken.append(self._queue.get_nowait())
            except queue.Empty:
                return taken

    def hides_on_close(self) -> bool:
        """Closing hides the window when there is a tray to bring it back
        from, and quits when there is not."""
        return self._tray

    def toggle(self) -> None:
        self._loop.call_soon_threadsafe(self._on_toggle)

    def settings(self) -> None:
        self._loop.call_soon_threadsafe(self._on_settings)

    def quit(self) -> None:
        self._quitting = True
        self._loop.call_soon_threadsafe(self._on_quit)

    def answer(self, answer: Answer | None, value: str | None) -> None:
        if answer is not None:
            self._loop.call_soon_threadsafe(_resolve, answer, value)


class WindowPrompter:
    """The wizard's `Prompter` on the window: the same keys and words as the
    terminal's, each question a page, each answer a future the Tk thread
    resolves."""

    def __init__(self, window: Window, *, text: Mapping[str, str] | None = None) -> None:
        self._window = window
        self._text = wording() if text is None else text

    def say(self, key: str, **fields: object) -> None:
        self._window.say(self._text[key].format(**fields))

    async def choose(self, key: str, options: Sequence[Option]) -> str | None:
        return await self._window.ask("choose", self._text[key], options)

    async def secret(self, key: str) -> str | None:
        return await self._window.ask("secret", self._text[key], ())

    async def ask(self, key: str) -> str | None:
        return await self._window.ask("ask", self._text[key], ())

    async def menu(self, key: str, options: Sequence[Option]) -> str | None:
        return await self._window.ask("menu", self._text[key], options)


class Switch:
    """The listen button's target: nothing until the capture exists, then
    its `toggle`. The window is built before the capture (`__main__`)."""

    def __init__(self) -> None:
        self.target: Callable[[], None] | None = None

    def __call__(self) -> None:
        if self.target is not None:
            self.target()


# --------------------------------------------------------------------------
# The Tk half
# --------------------------------------------------------------------------

# The ground, with no blue in it (D33): on the old blue-black only the idle
# orb looked at home. Everything the orb glows over is painted on it.
BACKGROUND: RGB = (8, 10, 14)
# A plate, the pointer's plate, a press - each a step off the ground.
RAISED: RGB = (19, 23, 29)
LIFTED: RGB = (29, 34, 42)
SUNK: RGB = (13, 16, 21)
HAIRLINE: RGB = (38, 45, 55)
CHALK: RGB = (233, 237, 241)
GRAPHITE: RGB = (138, 147, 158)
FAINT: RGB = (84, 92, 103)
# Notices keep the orange they had: the reconnecting orb's.
AMBER: RGB = orb.COLOURS[State.RECONNECTING]
SIGNAL_INK: RGB = (255, 158, 158)
SIGNAL_PLATE: RGB = (52, 22, 27)
# What the orb's hot centre and its dots burn towards.
WHITE: RGB = (230, 243, 255)
# How much of the state's colour the state's word carries, over chalk.
WORD_TINT = 0.6

# Bahnschrift - Windows' DIN, on every Windows 10 since 1709 - says what
# the machine says; Segoe UI what people said. A machine without
# Bahnschrift falls back to the system's font and keeps the layout.
DISPLAY_FONT = ("Bahnschrift SemiLight", 21)
QUESTION_FONT = ("Bahnschrift SemiLight", 16)
CAPTION_FONT = ("Bahnschrift", 9)
LABEL_FONT = ("Bahnschrift", 10)
BODY_FONT = ("Segoe UI", 10)
VALUE_FONT = ("Segoe UI", 11)
# Windows' own icons, from the font every Windows 10 has. All in the Basic
# Multilingual Plane, which Tcl 8.6.14 draws reliably.
GLYPH_FAMILY = "Segoe MDL2 Assets"
MIC = chr(0xE720)
MIC_OFF = chr(0xF781)
GEAR = chr(0xE713)
POWER = chr(0xE7E8)
SEARCH = chr(0xE721)
CHEVRON_DOWN = chr(0xE70D)
CHEVRON_UP = chr(0xE70E)
# The side margin of everything that is not the orb.
INSET = 20

# The core's hot centre as the canvas can draw it: ovals stepping inwards,
# each a fraction of the core's radius and that much whiter. Sixteen steps
# rather than the first four (U1): at four the gradient had rings of its
# own in it, which the eye read as banding and not as a glow. The curve is
# the four's, carried on to a hotter centre.
SPOT_COUNT = 16
SPOT_HOT = 0.95
SPOT_FALLOFF = 1.25
SPOT_STEPS: tuple[tuple[float, float], ...] = tuple(
    (reach, SPOT_HOT * (1 - reach) ** SPOT_FALLOFF)
    for reach in (1 - (step + 1) / (SPOT_COUNT + 1) for step in range(SPOT_COUNT))
)
# The halo Pillow paints under the core, as (how far out, how blurred, how
# strong). Tighter and thinner than the mockup's since U1: the wide, soft
# layer lay over the two inner arcs and washed the mechanism out.
GLOW_LAYERS: tuple[tuple[float, float, int], ...] = ((2.05, 0.34, 58), (1.28, 0.11, 150))
# A ring cut into this many pieces or more is drawn as one arc with Tk's
# own dash pattern instead of one item per piece: seventy-two items cost
# more to move than the rest of the orb put together (W1). One pixel wide,
# because that is the width at which Windows draws the pattern as given -
# wider, the dashes come out shorter and the pattern is not the ring's.
DASHED_RING_MIN = 24
# A ring that turned less than this since it was last drawn is left where
# it is: a quarter of a pixel at the orb's size, and every move redraws the
# orb. It was 0.4 while the window drew thirty frames a second; at sixty the
# inner arc turns 0.3 of a degree a frame and would have been held back
# every other frame - thirty frames again, which is what U1 was about.
MIN_TURN_DEGREES = 0.12


# -- the buttons -----------------------------------------------------------
#
# Tk's own button is a grey slab with a square corner and no answer to the
# pointer. These are canvases: a rounded plate Pillow draws (supersampled,
# because Tk has no anti-aliased corner either), a Windows glyph and the
# label as canvas text on top so they stay the system's own crisp glyphs,
# and six plates per button - at rest, under the pointer, held down, out of
# use, and the first two again with the keyboard's focus ring.


@dataclass(frozen=True, slots=True)
class Style:
    """What one kind of button wears. `edge` is the hairline round the plate,
    or nothing; a `fill` that is the ground is a ghost - words until the
    pointer is on them. `hover_ink` recolours the words under the pointer."""

    fill: RGB
    hover: RGB
    press: RGB
    edge: RGB | None
    ink: RGB
    hover_ink: RGB | None = None


# The one filled button of a page: Continue, Change.
SOLID = Style(fill=CHALK, hover=(255, 255, 255), press=(196, 203, 211), edge=None, ink=BACKGROUND)
# The listen switch: the orb is the face's light, so its button is a plate.
PLATE = Style(fill=RAISED, hover=LIFTED, press=SUNK, edge=HAIRLINE, ink=CHALK)
# Settings, Cancel, Close.
GHOST = Style(fill=BACKGROUND, hover=RAISED, press=SUNK, edge=None, ink=GRAPHITE, hover_ink=CHALK)
# Quit: a ghost until the pointer is on it, then it says so.
DANGER = Style(
    fill=BACKGROUND,
    hover=SIGNAL_PLATE,
    press=(38, 16, 20),
    edge=None,
    ink=GRAPHITE,
    hover_ink=SIGNAL_INK,
)
# Where the keyboard is: a ring of its own, never the pointer's plate.
FOCUS_EDGE = GRAPHITE

BUTTON_PAD_X = 16
BUTTON_PAD_Y = 9
BUTTON_RADIUS = 8
# Between a button's glyph and its words.
GLYPH_GAP = 9
# How much larger the plate is drawn before it is shrunk back, so that the
# corners come out smooth.
PLATE_SUPERSAMPLE = 4
# Out of use: the plate faded this far towards the background.
FADED = 0.45


def plate(
    width: int, height: int, radius: int, fill: RGB, edge: RGB | None, background: RGB
) -> Image.Image:
    """One rounded button plate, `width` by `height`, painted on `background`
    (opaque, so that Tk is handed no alpha to blend every frame - the orb's
    rule). Drawn `PLATE_SUPERSAMPLE` times larger and shrunk back: Pillow's
    rounded rectangle has stepped corners at the size asked for."""
    if width < 1 or height < 1:
        raise ValueError(f"a plate of {width}x{height} cannot be drawn")
    scale = PLATE_SUPERSAMPLE
    image = Image.new("RGB", (width * scale, height * scale), background)
    ImageDraw.Draw(image).rounded_rectangle(
        (0, 0, width * scale - 1, height * scale - 1),
        radius=radius * scale,
        fill=fill,
        outline=edge,
        width=scale if edge is not None else 0,
    )
    return image.resize((width, height), Image.Resampling.LANCZOS)


def hex_of(colour: RGB) -> str:
    red, green, blue = colour
    return f"#{red:02x}{green:02x}{blue:02x}"


def blend(colour: RGB, towards: RGB, amount: float) -> RGB:
    """`amount` of `colour` over `towards`: 1 is the colour, 0 the other."""
    red, green, blue = (round(t + (c - t) * amount) for c, t in zip(colour, towards, strict=True))
    return (red, green, blue)


def _turned(before: float, after: float) -> float:
    """Degrees between two angles, the short way round."""
    return abs((after - before + 180) % 360 - 180)


def _eased(progress: float) -> float:
    """Ease-out, cubic: quick to answer the press, gentle to arrive."""
    progress = max(0.0, min(1.0, progress))
    return 1 - (1 - progress) ** 3


def _dpi_aware() -> None:
    """Crisp at 125-150 % scaling: the process says it handles DPI itself.
    Windows only, and harmless where the call does not exist."""
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass


def _timer_period(milliseconds: int, *, begin: bool) -> None:
    """Asks Windows for a finer timer while the window is up, and gives it
    back when the window goes down (U1). Without this the 15 ms the tick
    asks for is served at the system's 15.6 ms granularity, which is still
    sixty frames - but with the audio device's own request gone, a coarse
    granularity would round it to 31 ms and halve the rate. Windows only,
    and harmless where the call does not exist."""
    try:
        import ctypes

        call = ctypes.windll.winmm.timeBeginPeriod if begin else ctypes.windll.winmm.timeEndPeriod
        call(milliseconds)
    except (AttributeError, OSError):
        pass


def _frame_of(root: tk.Tk) -> int:
    """The Windows handle of the window's frame - the title bar's - which is
    the parent of the window Tk draws in. 0 where there is none."""
    try:
        import ctypes
        from ctypes import wintypes

        parent = ctypes.windll.user32.GetParent
        parent.argtypes = [wintypes.HWND]
        parent.restype = wintypes.HWND
        return int(parent(root.winfo_id()) or 0)
    except (AttributeError, OSError):
        return 0


def _dark_title_bar(frame: int) -> None:
    """Windows' dark title bar over the window (D33): DWM's
    `DWMWA_USE_IMMERSIVE_DARK_MODE`, 20 since Windows 10 2004. An older
    Windows leaves the bar light; nothing else changes."""
    if not frame:
        return
    try:
        import ctypes
        from ctypes import wintypes

        attribute = ctypes.windll.dwmapi.DwmSetWindowAttribute
        attribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        on = ctypes.c_int(1)
        attribute(frame, 20, ctypes.byref(on), ctypes.sizeof(on))
    except (AttributeError, OSError):
        pass


def _repaint(handle: int) -> None:
    """Has Windows ask the widget to paint all of itself again, from what
    it holds - no new layout, just fresh pixels. Harmless where the call
    does not exist."""
    try:
        import ctypes
        from ctypes import wintypes

        invalidate = ctypes.windll.user32.InvalidateRect
        invalidate.argtypes = [wintypes.HWND, ctypes.c_void_p, wintypes.BOOL]
        invalidate(handle, None, True)
    except (AttributeError, OSError):
        pass


def _work_area_bottom(frame: int) -> int | None:
    """Where the usable part of the window's own screen ends - above the
    taskbar - in pixels; `None` where it cannot be asked."""
    if not frame:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class MonitorInfo(ctypes.Structure):
            _fields_ = (
                ("size", wintypes.DWORD),
                ("monitor", wintypes.RECT),
                ("work", wintypes.RECT),
                ("flags", wintypes.DWORD),
            )

        user32 = ctypes.windll.user32
        nearest = user32.MonitorFromWindow
        nearest.argtypes = [wintypes.HWND, wintypes.DWORD]
        nearest.restype = wintypes.HMONITOR
        describe = user32.GetMonitorInfoW
        describe.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MonitorInfo)]
        describe.restype = wintypes.BOOL
        info = MonitorInfo()
        info.size = ctypes.sizeof(MonitorInfo)
        # MONITOR_DEFAULTTONEAREST: the screen the window is (mostly) on.
        if not describe(nearest(frame, 2), ctypes.byref(info)):
            return None
        return int(info.work.bottom)
    except (AttributeError, OSError):
        return None


class Button(tk.Canvas):
    """A rounded button that answers the pointer: `Style`'s plates, a
    Windows glyph and the label on top, and the command on release inside.

    Keyboard-reachable like Tk's own: it takes focus, wears a ring while it
    has it, and Return or Space presses it. A click does not take the focus,
    so the ring is only ever Tab's and never follows the mouse.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        text: str,
        command: Callable[[], None],
        style: Style = GHOST,
        glyph: str = "",
        scale: float = 1.0,
    ) -> None:
        super().__init__(
            parent,
            bg=hex_of(BACKGROUND),
            highlightthickness=0,
            bd=0,
            takefocus=True,
            cursor="hand2",
        )
        self._style = style
        self._command = command
        self._scale = scale
        self._font = tkfont.Font(family=LABEL_FONT[0], size=LABEL_FONT[1])
        self._glyph_font = tkfont.Font(family=GLYPH_FAMILY, size=11)
        self._plates: dict[str, ImageTk.PhotoImage] = {}
        self._plate_item = self.create_image(0, 0, anchor="nw")
        self._glyph_item = self.create_text(0, 0, anchor="w", font=self._glyph_font, text="")
        self._label_item = self.create_text(0, 0, anchor="w", font=self._font, text="")
        self._enabled = True
        self._under = False
        self._held = False
        self._focused = False
        # None until the first `set_text`, so that a button born with no
        # words still gets its plates and its size when they arrive.
        self._shown: tuple[str, str] | None = None
        for event, handler in (
            ("<Enter>", self._entered),
            ("<Leave>", self._left),
            ("<ButtonPress-1>", self._pressed),
            ("<ButtonRelease-1>", self._released),
            ("<FocusIn>", self._focus_in),
            ("<FocusOut>", self._focus_out),
            ("<Return>", self._struck),
            ("<space>", self._struck),
        ):
            self.bind(event, handler)
        self.set_text(text, glyph)

    # -- what the panel says to it -----------------------------------------

    def set_text(self, text: str, glyph: str = "") -> None:
        """The words and the glyph, and with them the button's size: the
        plates are repainted only when either really changed."""
        if (text, glyph) == self._shown:
            return
        self._shown = (text, glyph)
        pad_x, pad_y = round(BUTTON_PAD_X * self._scale), round(BUTTON_PAD_Y * self._scale)
        glyph_width = self._glyph_font.measure(glyph) if glyph else 0
        gap = round(GLYPH_GAP * self._scale) if glyph else 0
        width = pad_x + glyph_width + gap + self._font.measure(text) + pad_x
        height = self._font.metrics("linespace") + 2 * pad_y
        self.configure(width=width, height=height)
        radius = max(2, round(BUTTON_RADIUS * self._scale))
        self._plates = {
            name: ImageTk.PhotoImage(plate(width, height, radius, fill, edge, BACKGROUND))
            for name, fill, edge in self._faces()
        }
        middle = height / 2
        # The glyphs sit a pixel high against Bahnschrift's letters.
        self.coords(self._glyph_item, pad_x, middle + round(self._scale))
        self.itemconfigure(self._glyph_item, text=glyph)
        self.coords(self._label_item, pad_x + glyph_width + gap, middle)
        self.itemconfigure(self._label_item, text=text)
        self._wear()

    def set_enabled(self, enabled: bool) -> None:
        if enabled != self._enabled:
            self._enabled = enabled
            self.configure(cursor="hand2" if enabled else "")
            self._wear()

    # -- what the pointer and the keyboard say to it ------------------------

    def _entered(self, _: object) -> None:
        self._under = True
        self._wear()

    def _left(self, _: object) -> None:
        self._under = self._held = False
        self._wear()

    def _pressed(self, _: object) -> None:
        if self._enabled:
            self._held = True
            self._wear()

    def _released(self, _: object) -> None:
        held, self._held = self._held, False
        self._wear()
        if held and self._enabled and self._under:
            self._command()

    def _focus_in(self, _: object) -> None:
        self._focused = True
        self._wear()

    def _focus_out(self, _: object) -> None:
        self._focused = False
        self._wear()

    def _struck(self, _: object) -> None:
        if self._enabled:
            self._command()

    # ----------------------------------------------------------------------

    def _faces(self) -> tuple[tuple[str, RGB, RGB | None], ...]:
        style = self._style
        faded = blend(style.fill, BACKGROUND, 1 - FADED)
        return (
            ("rest", style.fill, style.edge),
            ("hover", style.hover, style.edge),
            ("held", style.press, style.edge),
            ("off", faded, None),
            ("rest+focus", style.fill, FOCUS_EDGE),
            ("hover+focus", style.hover, FOCUS_EDGE),
        )

    def _wear(self) -> None:
        if not self._plates:
            return
        style = self._style
        if not self._enabled:
            face, ink = "off", FAINT
        elif self._held:
            face, ink = "held", style.hover_ink or style.ink
        elif self._under:
            face, ink = "hover", style.hover_ink or style.ink
        else:
            face, ink = "rest", style.ink
        if self._focused and face in ("rest", "hover"):
            face += "+focus"
        self.itemconfigure(self._plate_item, image=self._plates[face])
        for item in (self._glyph_item, self._label_item):
            self.itemconfigure(item, fill=hex_of(ink))


class Fold(tk.Canvas):
    """The accordion's header (D33): the words on the left, a chevron on the
    right that points where the conversation will go, the whole row one
    target. Answers the pointer and the keyboard like a `Button`."""

    def __init__(
        self, parent: tk.Misc, *, text: str, command: Callable[[], None], scale: float
    ) -> None:
        font = tkfont.Font(family=LABEL_FONT[0], size=LABEL_FONT[1])
        height = font.metrics("linespace") + round(2 * 12 * scale)
        super().__init__(
            parent,
            bg=hex_of(BACKGROUND),
            highlightthickness=0,
            bd=0,
            takefocus=True,
            cursor="hand2",
            height=height,
        )
        self._command = command
        self._scale = scale
        self._height = height
        self._pad = round(12 * scale)
        self._plates: dict[str, ImageTk.PhotoImage] = {}
        self._plate_item = self.create_image(0, 0, anchor="nw")
        self._label_item = self.create_text(self._pad, height / 2, anchor="w", text=text, font=font)
        self._chevron_item = self.create_text(
            0, height / 2 + round(scale), anchor="e", text=CHEVRON_DOWN, font=(GLYPH_FAMILY, 9)
        )
        self._under = False
        self._focused = False
        for event, handler in (
            ("<Configure>", self._resized),
            ("<Enter>", lambda _: self._set(under=True)),
            ("<Leave>", lambda _: self._set(under=False)),
            ("<ButtonRelease-1>", self._released),
            ("<FocusIn>", lambda _: self._set(focused=True)),
            ("<FocusOut>", lambda _: self._set(focused=False)),
            ("<Return>", lambda _: self._command()),
            ("<space>", lambda _: self._command()),
        ):
            self.bind(event, handler)

    def set_open(self, opened: bool) -> None:
        self.itemconfigure(self._chevron_item, text=CHEVRON_UP if opened else CHEVRON_DOWN)

    def _set(self, *, under: bool | None = None, focused: bool | None = None) -> None:
        if under is not None:
            self._under = under
        if focused is not None:
            self._focused = focused
        self._wear()

    def _released(self, event: tk.Event[Any]) -> None:
        if 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height():
            self._command()

    def _resized(self, event: tk.Event[Any]) -> None:
        width = max(2, event.width)
        radius = max(2, round(BUTTON_RADIUS * self._scale))
        faces = {
            "rest": (BACKGROUND, None),
            "hover": (RAISED, None),
            "rest+focus": (BACKGROUND, FOCUS_EDGE),
            "hover+focus": (RAISED, FOCUS_EDGE),
        }
        self._plates = {
            name: ImageTk.PhotoImage(plate(width, self._height, radius, fill, edge, BACKGROUND))
            for name, (fill, edge) in faces.items()
        }
        self.coords(self._chevron_item, width - self._pad, self._height / 2 + round(self._scale))
        self._wear()

    def _wear(self) -> None:
        if not self._plates:
            return
        face = "hover" if self._under else "rest"
        if self._focused:
            face += "+focus"
        self.itemconfigure(self._plate_item, image=self._plates[face])
        ink = hex_of(CHALK if self._under else GRAPHITE)
        for item in (self._label_item, self._chevron_item):
            self.itemconfigure(item, fill=ink)


class Field(tk.Canvas):
    """A rounded entry: a plate Pillow draws, the Entry sitting on it, an
    optional glyph at the start and a hint while it is empty."""

    def __init__(self, parent: tk.Misc, *, scale: float, glyph: str = "", hint: str = "") -> None:
        font = tkfont.Font(family=VALUE_FONT[0], size=VALUE_FONT[1])
        height = font.metrics("linespace") + round(2 * 10 * scale)
        super().__init__(parent, bg=hex_of(BACKGROUND), highlightthickness=0, bd=0, height=height)
        self._scale = scale
        self._height = height
        self._pad = round(14 * scale)
        self._plate_item = self.create_image(0, 0, anchor="nw")
        start = self._pad
        if glyph:
            self.create_text(
                start,
                height / 2 + round(scale),
                anchor="w",
                text=glyph,
                font=(GLYPH_FAMILY, 11),
                fill=hex_of(GRAPHITE),
            )
            start += round(26 * scale)
        self._start = start
        self.entry = tk.Entry(
            self,
            bg=hex_of(RAISED),
            fg=hex_of(CHALK),
            insertbackground=hex_of(CHALK),
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=font,
        )
        self._slot = self.create_window(start, height / 2, anchor="w", window=self.entry)
        # The hint is a label over the Entry's empty box - a canvas item would
        # be under it, since a canvas draws its windows over everything - and
        # starts two pixels in, so that the caret still shows beside it.
        self._hint = tk.Label(self, text=hint, bg=hex_of(RAISED), fg=hex_of(FAINT), font=font)
        self._hint.bind("<Button-1>", lambda _: self.entry.focus_set())
        self._plates: dict[bool, ImageTk.PhotoImage] = {}
        self._focused = False
        self.bind("<Configure>", self._resized)
        self.bind("<Button-1>", lambda _: self.entry.focus_set())
        self.entry.bind("<FocusIn>", lambda _: self._wear(True), add="+")
        self.entry.bind("<FocusOut>", lambda _: self._wear(False), add="+")
        self.entry.bind("<KeyRelease>", lambda _: self._hinted(), add="+")
        self._hinted()

    def clear(self) -> None:
        self.entry.delete(0, "end")
        self._hinted()

    def _hinted(self) -> None:
        if self.entry.get() or not self._hint.cget("text"):
            self._hint.place_forget()
        else:
            self._hint.place(in_=self.entry, x=round(2 * self._scale), rely=0.5, anchor="w")
            self._hint.lift()

    def _resized(self, event: tk.Event[Any]) -> None:
        width = max(2, event.width)
        radius = max(2, round(BUTTON_RADIUS * self._scale))
        self._plates = {
            focused: ImageTk.PhotoImage(
                plate(
                    width,
                    self._height,
                    radius,
                    RAISED,
                    GRAPHITE if focused else HAIRLINE,
                    BACKGROUND,
                )
            )
            for focused in (False, True)
        }
        self.itemconfigure(self._slot, width=max(1, width - self._start - self._pad))
        self._wear(self._focused)

    def _wear(self, focused: bool) -> None:
        self._focused = focused
        if self._plates:
            self.itemconfigure(self._plate_item, image=self._plates[focused])


class Rows(tk.Text):
    """The pick-one list, as rows rather than Tk's listbox lines: padded,
    highlighted the whole width, one line each or two (`split_row`).
    Keyboard and mouse like a list: up and down, a click to pick, a double
    click or Return to go on."""

    def __init__(self, parent: tk.Misc, *, scale: float, on_activate: Callable[[], None]) -> None:
        super().__init__(
            parent,
            bg=hex_of(BACKGROUND),
            fg=hex_of(CHALK),
            relief="flat",
            bd=0,
            highlightthickness=0,
            wrap="word",
            cursor="arrow",
            takefocus=True,
            padx=0,
            pady=0,
            height=1,
            font=VALUE_FONT,
            insertwidth=0,
            exportselection=False,
        )
        margin = round(14 * scale)
        self.tag_configure(
            "what",
            font=CAPTION_FONT,
            foreground=hex_of(GRAPHITE),
            spacing1=round(11 * scale),
            lmargin1=margin,
            lmargin2=margin,
            rmargin=margin,
        )
        self.tag_configure(
            "value",
            font=VALUE_FONT,
            foreground=hex_of(CHALK),
            spacing1=round(2 * scale),
            spacing3=round(11 * scale),
            lmargin1=margin,
            lmargin2=margin,
            rmargin=margin,
        )
        self.tag_configure(
            "single",
            font=VALUE_FONT,
            foreground=hex_of(CHALK),
            spacing1=round(8 * scale),
            spacing3=round(8 * scale),
            lmargin1=margin,
            lmargin2=margin,
            rmargin=margin,
        )
        self.tag_configure("hover", background=hex_of(RAISED), lmargincolor=hex_of(RAISED))
        self.tag_configure("chosen", background=hex_of(LIFTED), lmargincolor=hex_of(LIFTED))
        self._on_activate = on_activate
        self._spans: list[tuple[str, str]] = []
        self.chosen = -1
        self._hovered = -1
        self.configure(state="disabled")
        self.bind("<Motion>", self._moved)
        self.bind("<Leave>", lambda _: self._hover(-1))
        self.bind("<Button-1>", self._clicked)
        self.bind("<Double-Button-1>", self._activated)
        self.bind("<Return>", self._activated)
        self.bind("<Up>", lambda _: self._step(-1))
        self.bind("<Down>", lambda _: self._step(1))

    def show(self, labels: Sequence[str], *, two_lines: bool) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        self._spans = []
        for number, label in enumerate(labels):
            start = self.index("end-1c")
            what, value = split_row(label) if two_lines else ("", label)
            if what:
                self.insert("end", what + "\n", ("what",))
                self.insert("end", value, ("value",))
            else:
                self.insert("end", label, ("single",))
            # A row ends at its line break; the last row has none, and ends
            # at Tk's own final line break instead - which would be an empty
            # row under the list if a break were written after it.
            self._spans.append((start, self.index("end-1c")))
            if number < len(labels) - 1:
                self.insert("end", "\n", (("value",) if what else ("single",)))
        self.configure(state="disabled")
        self.chosen = self._hovered = -1
        if self._spans:
            self.choose(0)
        self.yview_moveto(0)

    def choose(self, index: int) -> None:
        if not self._spans:
            return
        index = max(0, min(index, len(self._spans) - 1))
        self.chosen = index
        start, end = self._spans[index]
        self.tag_remove("chosen", "1.0", "end")
        # Through the line break, or the highlight stops where the words do.
        self.tag_add("chosen", start, f"{end} +1c")
        self.see(end)
        self.see(start)

    def _row_at(self, event: tk.Event[Any]) -> int:
        at = self.index(f"@{event.x},{event.y}")
        for index, (start, end) in enumerate(self._spans):
            if self.compare(at, ">=", start) and self.compare(at, "<=", end):
                return index
        return -1

    def _hover(self, index: int) -> None:
        if index == self._hovered:
            return
        self._hovered = index
        self.tag_remove("hover", "1.0", "end")
        if index >= 0:
            start, end = self._spans[index]
            self.tag_add("hover", start, f"{end} +1c")

    def _moved(self, event: tk.Event[Any]) -> None:
        self._hover(self._row_at(event))

    def _clicked(self, event: tk.Event[Any]) -> str:
        self.focus_set()
        index = self._row_at(event)
        if index >= 0:
            self.choose(index)
        return "break"

    def _activated(self, _: object) -> str:
        self._on_activate()
        return "break"

    def _step(self, delta: int) -> str:
        self.choose(self.chosen + delta)
        return "break"


class TkPanel:
    """The widgets, the orb and the tick. Lives on the window's thread."""

    def __init__(self, view: View, window: Window) -> None:
        _dpi_aware()
        self._view = view
        self._window = window
        _timer_period(1, begin=True)
        root = self._root = tk.Tk()
        ground = hex_of(BACKGROUND)
        root.title(APP_TITLE)
        root.configure(bg=ground)
        scale = self._scale = root.winfo_fpixels("1i") / 96.0
        self._width, self._full = (round(side * scale) for side in WINDOW_SIZE)
        root.geometry(f"{self._width}x{self._full}")
        # One width, and a height only the accordion and the wizard set
        # (U2, D33): no drag on an edge, no maximise - Windows greys the
        # middle title-bar button out, and minimise and close stay.
        root.resizable(False, False)
        # The mark, in the title bar and on the taskbar button (U5). Tk keeps
        # no reference of its own, so the photos are held here or the icon
        # goes blank the moment they are collected.
        self._marks = [ImageTk.PhotoImage(logo.draw_logo(size)) for size in logo.ICON_SIZES]
        # By name rather than by object: Pillow's photo is not Tk's own
        # class, and `wm iconphoto` takes the Tcl name either way - Windows
        # picks the size it wants for the title bar and for the taskbar.
        names = [str(mark) for mark in self._marks]
        root.iconphoto(True, names[0], *names[1:])
        root.protocol("WM_DELETE_WINDOW", self._close)
        inset = self._inset = round(INSET * scale)

        # The face: readouts and orb, the state's word, the buttons, the
        # accordion's header - and the conversation under it when unfolded.
        self._face = tk.Frame(root, bg=ground)
        self._canvas = tk.Canvas(
            self._face, bg=ground, highlightthickness=0, height=round(ORB_HEIGHT * scale)
        )
        self._canvas.pack(fill="x")
        self._canvas.bind("<Configure>", self._resized)
        self._word = tk.Label(self._face, bg=ground, fg=hex_of(CHALK), font=DISPLAY_FONT)
        self._word.pack(pady=(round(2 * scale), round(14 * scale)))
        bar = tk.Frame(self._face, bg=ground)
        # The ghosts' words start where the switch's plate does, not where
        # their own invisible plates do.
        bar.pack(fill="x", padx=inset - round(4 * scale), pady=(0, round(14 * scale)))
        self._switch = self._button(bar, "", window.toggle, style=PLATE)
        self._switch.pack(side="left")
        self._button(
            bar, view.said["window_settings"], window.settings, style=GHOST, glyph=GEAR
        ).pack(side="left", padx=(round(6 * scale), 0))
        self._button(bar, view.quit_label(), self._quit, style=DANGER, glyph=POWER).pack(
            side="right"
        )
        tk.Frame(self._face, bg=hex_of(HAIRLINE), height=1).pack(fill="x", padx=inset)
        self._fold = Fold(
            self._face,
            text=view.said["window_conversation"],
            command=self._fold_pressed,
            scale=scale,
        )
        self._fold.pack(
            fill="x", padx=inset - round(12 * scale), pady=(round(4 * scale), round(6 * scale))
        )
        self._talk = tk.Text(
            self._face,
            bg=ground,
            fg=hex_of(CHALK),
            relief="flat",
            bd=0,
            highlightthickness=0,
            wrap="word",
            state="disabled",
            font=BODY_FONT,
            padx=0,
            # No padding inside: Tk draws into it whatever of the row above
            # the view reaches down - a cedilla, a comma. The room above and
            # below is the packing's (`_fit`), so the view is cut at its edge.
            pady=0,
            cursor="arrow",
            takefocus=False,
            height=1,
        )
        self._talk_room = (0, round(14 * scale))
        # Who said it is where it stands: the user's words to the right and
        # dim, the answer to the left and bright (D33). No emoji in a row:
        # Tcl 8.6.14 does not draw characters outside the Basic Multilingual
        # Plane reliably.
        indent = round(64 * scale)
        self._talk.tag_configure(
            "you",
            justify="right",
            foreground=hex_of(GRAPHITE),
            spacing1=round(16 * scale),
            lmargin1=indent,
            lmargin2=indent,
        )
        self._talk.tag_configure(
            "it", foreground=hex_of(CHALK), spacing1=round(4 * scale), rmargin=round(32 * scale)
        )
        self._talk.tag_configure(
            "searched", foreground=hex_of(GRAPHITE), font=CAPTION_FONT, spacing1=round(5 * scale)
        )
        self._talk.tag_configure("glyph", font=(GLYPH_FAMILY, 8))
        self._talk.tag_configure("notice", foreground=hex_of(AMBER), spacing1=round(16 * scale))
        # Last, so that they win: the first row needs no gap above it, and
        # the last one the room `_align_top` gives it.
        self._talk.tag_configure("lead", spacing1=0)
        self._talk.tag_configure("tail", spacing3=0)
        self._face.pack(fill="both", expand=True)

        # The wizard's page, packed instead of the face while a wizard runs.
        self._page = tk.Frame(root, bg=ground)
        self._lines = tk.Text(
            self._page,
            bg=ground,
            fg=hex_of(GRAPHITE),
            relief="flat",
            bd=0,
            highlightthickness=0,
            wrap="word",
            state="disabled",
            font=BODY_FONT,
            padx=0,
            pady=0,
            cursor="arrow",
            takefocus=False,
            height=1,
        )
        self._lines.tag_configure("said", foreground=hex_of(GRAPHITE), spacing3=round(3 * scale))
        self._lines.tag_configure("latest", foreground=hex_of(CHALK))
        self._question = tk.Label(
            self._page,
            bg=ground,
            fg=hex_of(CHALK),
            font=QUESTION_FONT,
            justify="left",
            anchor="w",
            padx=0,
            wraplength=round((WINDOW_SIZE[0] - 2 * INSET) * scale),
        )
        self._filter = Field(self._page, scale=scale, glyph=SEARCH, hint=view.said["wizard_filter"])
        self._filter.entry.bind("<KeyRelease>", self._filtered, add="+")
        self._list = Rows(self._page, scale=scale, on_activate=self._continue_pressed)
        self._filter.entry.bind("<Down>", lambda _: self._list.focus_set())
        self._filter.entry.bind("<Return>", lambda _: self._continue_pressed())
        self._entry = Field(self._page, scale=scale)
        self._entry.entry.bind("<Return>", lambda _: self._continue_pressed())
        buttons = tk.Frame(self._page, bg=ground)
        buttons.pack(side="bottom", fill="x", padx=inset, pady=round(14 * scale))
        self._continue = self._button(
            buttons, view.said["wizard_continue"], self._continue_pressed, style=SOLID
        )
        self._continue.pack(side="left")
        self._cancel = self._button(
            buttons, view.said["wizard_cancel"], self._cancel_pressed, style=GHOST
        )
        self._cancel.pack(side="right")
        self._visible: list[Option] = []
        self._page_open = False
        self._page_seen: WizardPage | None = None
        self._lines_shown = -1
        self._asked: Answer | None = None
        self._asked_first = True

        # The orb's pieces on the canvas, made at the first `<Configure>`.
        self._centre = (0.0, 0.0)
        self._radius = 0.0
        self._glow: dict[RGB, ImageTk.PhotoImage] = {}
        self._glow_item = 0
        self._core_item = 0
        self._spots: list[int] = []
        self._dashes: list[list[int]] = []
        self._applied: list[float | None] = []
        self._caps: list[int] = []
        self._readouts = (0, 0)
        self._colour: RGB | None = None

        self._shown: tuple[Any, ...] = ()
        self._rows_version = -1
        self._talk_shown = False
        # The accordion's height: where it is going, where it set out from
        # and when, and the folded height - measured once the face is laid
        # out, below.
        self._compact = self._full
        self._height = self._full
        self._target = self._full
        self._from = self._full
        self._moving_since: float | None = None
        self._last = time.monotonic()
        self._paint()
        # Laid out and mapped once, so that there is a frame to darken and a
        # folded face to measure; then shown again at the height it wants.
        root.update_idletasks()
        self._frame = _frame_of(root)
        _dark_title_bar(self._frame)
        self._compact = self._face.winfo_reqheight()
        self._height = self._target = self._from = self._wanted()
        root.geometry(f"{self._width}x{self._height}")
        root.withdraw()
        root.deiconify()
        root.after(TICK_MS, self._tick)

    def run(self) -> None:
        try:
            self._root.mainloop()
        finally:
            _timer_period(1, begin=False)

    # -- widgets ---------------------------------------------------------------

    def _button(
        self,
        parent: tk.Misc,
        text: str,
        command: Callable[[], None],
        *,
        style: Style,
        glyph: str = "",
    ) -> Button:
        return Button(
            parent, text=text, command=command, style=style, glyph=glyph, scale=self._scale
        )

    # -- the tick ---------------------------------------------------------------

    def _tick(self) -> None:
        started = time.monotonic()
        for message in self._window.drain():
            if message == ("quit",):
                self._root.destroy()
                return
            if message == ("show",):
                self._root.deiconify()
                continue
            self._view.apply(message)
        self._paint()
        self._fit(started)
        dt, self._last = min(started - self._last, MAX_DT), started
        self._draw(self._view.frame(dt))
        # The next frame is due `TICK_MS` after this one began, not after it
        # ended: asking for the whole period again would add the drawing to
        # it and leave the window short of sixty frames a second (U1).
        spent = round((time.monotonic() - started) * 1000)
        self._root.after(max(1, TICK_MS - spent), self._tick)

    def _paint(self) -> None:
        view = self._view
        # The state's word in the state's colour; a phase - loading, a
        # failed start - is no state, and is chalk.
        state = orb.COLOURS[view.state]
        tint = CHALK if view.phase is not None else blend(state, CHALK, WORD_TINT)
        shown = (view.label(), tint, view.readouts(), view.switch_label(), view.listening)
        if shown != self._shown:
            self._shown = shown
            self._word.configure(text=shown[0], fg=hex_of(tint))
            if self._dashes:
                session, minutes = shown[2]
                self._canvas.itemconfigure(self._readouts[0], text=session)
                self._canvas.itemconfigure(self._readouts[1], text=minutes)
            # The glyph is what pressing does: a struck-out microphone to
            # stop listening, a microphone to start.
            self._switch.set_text(shown[3], MIC_OFF if view.listening else MIC)
        self._fold.set_open(view.log_open)
        if view.version != self._rows_version:
            self._rows_version = view.version
            self._fill_talk(view.rows)
        page_wanted = view.wizard is not None
        if page_wanted != self._page_open:
            self._page_open = page_wanted
            if page_wanted:
                self._face.pack_forget()
                self._page.pack(fill="both", expand=True)
                self._asked_first = True
            else:
                self._page.pack_forget()
                self._face.pack(fill="both", expand=True)
        if view.wizard is not None:
            self._paint_page(view.wizard)

    # -- the accordion ------------------------------------------------------------

    def _fold_pressed(self) -> None:
        self._view.toggle_log()
        self._fold.set_open(self._view.log_open)

    def _wanted(self) -> int:
        return self._full if self._view.wants_room() else self._compact

    def _fit(self, now: float) -> None:
        """Grows or folds the window towards the height the view wants, a
        frame at a time; shows the conversation before it grows and lets it
        go once it has folded."""
        wanted = self._wanted()
        if wanted != self._target:
            self._from, self._target, self._moving_since = self._height, wanted, now
            if wanted > self._height:
                self._keep_on_screen(wanted)
        if self._view.log_open and not self._talk_shown:
            self._talk.pack(fill="both", expand=True, padx=self._inset, pady=self._talk_room)
            self._talk_shown = True
        if self._moving_since is None:
            return
        progress = (now - self._moving_since) / FOLD_SECONDS
        height = round(self._from + (self._target - self._from) * _eased(progress))
        if height != self._height:
            self._height = height
            self._root.geometry(f"{self._width}x{height}")
            if self._talk_shown:
                self._talk.see("end")
        if progress >= 1:
            self._moving_since = None
            if not self._view.log_open and self._talk_shown:
                self._talk.pack_forget()
                self._talk_shown = False
            elif self._talk_shown:
                self._align_top()

    def _keep_on_screen(self, height: int) -> None:
        """A window about to grow past the bottom of its screen - into the
        taskbar or off the edge - is first lifted by as much as it would."""
        bottom = _work_area_bottom(self._frame)
        if bottom is None:
            return
        root = self._root
        border = max(0, root.winfo_rootx() - root.winfo_x())
        over = root.winfo_rooty() + height + border - bottom
        if over > 0:
            root.geometry(f"+{root.winfo_x()}+{max(0, root.winfo_y() - over)}")

    # -- the conversation ----------------------------------------------------------

    def _fill_talk(self, rows: Sequence[tuple[str, str]]) -> None:
        talk = self._talk
        talk.configure(state="normal")
        talk.delete("1.0", "end")
        above: tuple[str, ...] = ()
        for number, (kind, text) in enumerate(rows):
            tags: tuple[str, ...] = (kind, "lead") if number == 0 else (kind,)
            # The line break before a row belongs to the row above, and the
            # last row has none: Tk's own final line would be an empty row.
            if number:
                talk.insert("end", "\n", above)
            if kind == "searched":
                talk.insert("end", SEARCH, (*tags, "glyph"))
                talk.insert("end", f"  {text}", tags)
            else:
                talk.insert("end", text, tags)
            above = tags
        talk.configure(state="disabled")
        talk.see("end")
        self._align_top()

    def _align_top(self) -> None:
        """Scrolled to the newest row, the view's top edge cuts whatever row
        is there in half. Tk will not scroll past the end, so the last row
        is given just enough room beneath it to push that cut row out of
        sight and start the view at a whole one."""
        talk = self._talk
        talk.tag_remove("tail", "1.0", "end")
        if not self._talk_shown:
            return
        talk.update_idletasks()
        if talk.yview()[0] <= 0:
            return
        line = talk.dlineinfo("@0,0")
        if line is None:
            return
        _, top, _, height, _ = line
        hidden = int(talk.cget("pady")) - top
        if 0 < hidden < height:
            talk.tag_configure("tail", spacing3=height - hidden)
            talk.tag_add("tail", "end-1c linestart", "end-1c")
            talk.see("end")
            # Tk scrolls by copying what is on screen and repaints only the
            # rows it thinks changed; a descender of the row that just left
            # the view stayed behind in the gap above the new top row.
            _repaint(talk.winfo_id())

    # -- the wizard's page ----------------------------------------------------------

    def _paint_page(self, page: WizardPage) -> None:
        # A wizard closed and opened again within one tick is a new page:
        # its lines are counted afresh.
        if page is not self._page_seen or len(page.lines) != self._lines_shown:
            self._page_seen = page
            self._lines_shown = len(page.lines)
            self._show_lines(page.lines)
        if page.answer is self._asked and not self._asked_first:
            return
        self._asked = page.answer
        self._asked_first = False
        self._question.configure(text=page.question)
        for widget in (self._question, self._filter, self._list, self._entry):
            widget.pack_forget()
        self._filter.clear()
        self._entry.clear()
        proceed, back = self._view.page_buttons()
        self._continue.set_text(proceed)
        self._cancel.set_text(back)
        inset, scale = self._inset, self._scale
        self._question.pack(fill="x", padx=inset, pady=(round(18 * scale), round(12 * scale)))
        # The list's rows carry their own margin, so that the highlight runs
        # past the words: the list itself starts that much further out.
        list_inset = inset - round(14 * scale)
        if page.kind == "choose":
            self._filter.pack(fill="x", padx=inset, pady=(0, round(8 * scale)))
            self._list.pack(fill="both", expand=True, padx=list_inset)
            self._fill_list(page.options, two_lines=False)
            self._filter.entry.focus_set()
        elif page.kind == "menu":
            # A handful of rows, each two lines: nothing to filter.
            self._list.pack(fill="both", expand=True, padx=list_inset)
            self._fill_list(page.options, two_lines=True)
            self._list.focus_set()
        elif page.kind in ("secret", "ask"):
            self._entry.entry.configure(show="•" if page.kind == "secret" else "")
            self._entry.pack(fill="x", padx=inset)
            self._entry.entry.focus_set()
        self._continue.set_enabled(page.open)
        self._cancel.set_enabled(page.open)

    def _show_lines(self, lines: Sequence[str]) -> None:
        """What the wizard said so far, over the question: the last three,
        the newest in chalk - a failure is always the newest line."""
        shown = list(lines[-3:])
        box = self._lines
        box.configure(state="normal")
        box.delete("1.0", "end")
        for number, line in enumerate(shown):
            last = number == len(shown) - 1
            box.insert("end", line if last else line + "\n", ("said", "latest") if last else "said")
        box.configure(state="disabled")
        if not shown:
            box.pack_forget()
            return
        placing = {"fill": "x", "padx": self._inset, "pady": (round(22 * self._scale), 0)}
        if self._question.winfo_manager():
            box.pack(placing, before=self._question)
        else:
            box.pack(placing)
        box.update_idletasks()
        count = box.count("1.0", "end", "displaylines", return_ints=True)
        box.configure(height=max(1, min(5, count or 1)))

    def _fill_list(self, options: Sequence[Option], *, two_lines: bool) -> None:
        self._visible = list(options)
        self._list.show([option.label for option in self._visible], two_lines=two_lines)

    def _filtered(self, event: tk.Event[Any]) -> None:
        page = self._view.wizard
        if page is None or event.keysym in ("Return", "Down", "Up"):
            return
        needle = self._filter.entry.get().casefold()
        self._fill_list(
            [option for option in page.options if needle in option.label.casefold()],
            two_lines=False,
        )

    # -- clicks, on the Tk thread ------------------------------------------------

    def _continue_pressed(self) -> None:
        page = self._view.wizard
        if page is None or not page.open:
            return
        if page.kind in ("choose", "menu"):
            if not 0 <= self._list.chosen < len(self._visible):
                return
            value = self._visible[self._list.chosen].value
        else:
            value = self._entry.entry.get()
        self._window.answer(self._view.take_answer(), value)

    def _cancel_pressed(self) -> None:
        self._window.answer(self._view.take_answer(), None)

    def _close(self) -> None:
        # While the wizard is up there is no tray yet to come back from.
        if self._view.wizard is None and self._window.hides_on_close():
            self._root.withdraw()
        else:
            self._quit()

    def _quit(self) -> None:
        self._window.answer(self._view.take_answer(), None)
        self._window.quit()

    # -- the orb on the canvas ---------------------------------------------------

    def _resized(self, event: tk.Event[Any]) -> None:
        width, height = float(event.width), float(event.height)
        self._centre = (width / 2, height / 2)
        # The outer ring, its width and a hair of room must fit the half.
        outer = orb.RINGS[-1]
        self._radius = min(width, height) / 2 / (outer.radius + outer.width + 0.04)
        self._build(width)

    def _build(self, width: float) -> None:
        canvas = self._canvas
        canvas.delete("all")
        cx, cy = self._centre
        radius = self._radius
        core = orb.CORE * radius
        self._glow = {
            colour: self._glow_image(colour, core) for colour in set(orb.COLOURS.values())
        }
        first = next(iter(self._glow.values()))
        self._glow_item = canvas.create_image(cx, cy, image=first)
        self._core_item = canvas.create_oval(cx - core, cy - core, cx + core, cy + core, width=0)
        self._spots = [canvas.create_oval(0, 0, 1, 1, width=0) for _ in SPOT_STEPS]
        self._dashes = []
        for ring in orb.RINGS:
            r = ring.radius * radius
            box = (cx - r, cy - r, cx + r, cy + r)
            stroke = max(1, round(ring.width * radius))
            if ring.dashes >= DASHED_RING_MIN:
                dash = max(1, round(r * math.radians(ring.extent)))
                gap = max(1, round(r * math.radians(ring.step)) - dash)
                items = [
                    canvas.create_arc(
                        box,
                        start=0,
                        extent=359.9,
                        style="arc",
                        width=1,
                        outline="",
                        dash=(dash, gap),
                    )
                ]
            else:
                items = [
                    canvas.create_arc(
                        box, start=0, extent=ring.extent, style="arc", width=stroke, outline=""
                    )
                    for _ in range(ring.dashes)
                ]
            self._dashes.append(items)
        self._applied = [None] * len(orb.RINGS)
        self._caps = [canvas.create_oval(0, 0, 1, 1, width=0) for _ in orb.CAPS]
        # The session and its minutes, read off the instrument's corners.
        top = round(16 * self._scale)
        ink = hex_of(GRAPHITE)
        self._readouts = (
            canvas.create_text(self._inset, top, anchor="nw", font=CAPTION_FONT, fill=ink),
            canvas.create_text(width - self._inset, top, anchor="ne", font=CAPTION_FONT, fill=ink),
        )
        # Recoloured and reworded at the next draw.
        self._colour = None
        self._shown = ()

    def _glow_image(self, colour: RGB, core: float) -> ImageTk.PhotoImage:
        """The mockup's halo and inner glow, painted once per colour: Tk
        cannot blur. It is at the bottom of the stack, over nothing but
        the background, so the background is painted into it here and the
        photo goes to Tk without an alpha channel - blending a quarter of
        a million pixels a frame cost more than the rings did (W1)."""
        size = int(core * 2 * max(reach for reach, _, _ in GLOW_LAYERS) * 1.3) + 1
        image = Image.new("RGBA", (size, size), (*BACKGROUND, 255))
        centre = size / 2
        for reach, blur, alpha in GLOW_LAYERS:
            layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            r = core * reach
            ImageDraw.Draw(layer).ellipse(
                (centre - r, centre - r, centre + r, centre + r), fill=(*colour, alpha)
            )
            image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(core * blur)))
        return ImageTk.PhotoImage(image.convert("RGB"))

    def _draw(self, frame: Frame) -> None:
        if not self._dashes:
            return
        canvas = self._canvas
        cx, cy = self._centre
        radius = self._radius
        if frame.colour != self._colour:
            self._colour = frame.colour
            canvas.itemconfigure(self._glow_item, image=self._glow[frame.colour])
            for ring, items in zip(orb.RINGS, self._dashes, strict=True):
                shade = hex_of(blend(frame.colour, BACKGROUND, ring.tint))
                for item in items:
                    canvas.itemconfigure(item, outline=shade)
            cap = hex_of(blend(frame.colour, WHITE, 0.5))
            for item in self._caps:
                canvas.itemconfigure(item, fill=cap)
        core = orb.CORE * radius * frame.core_scale
        canvas.coords(self._core_item, cx - core, cy - core, cx + core, cy + core)
        canvas.itemconfigure(
            self._core_item, fill=hex_of(blend(frame.colour, BACKGROUND, frame.core_alpha))
        )
        for (reach, light), item in zip(SPOT_STEPS, self._spots, strict=True):
            spot = core * reach
            canvas.coords(item, cx - spot, cy - spot, cx + spot, cy + spot)
            tone = blend(blend(frame.colour, WHITE, 1 - light), BACKGROUND, frame.core_alpha)
            canvas.itemconfigure(item, fill=hex_of(tone))
        for index, (ring, items, angle) in enumerate(
            zip(orb.RINGS, self._dashes, frame.angles, strict=True)
        ):
            applied = self._applied[index]
            if applied is not None and _turned(applied, angle) < MIN_TURN_DEGREES:
                continue
            self._applied[index] = angle
            for piece, item in enumerate(items):
                canvas.itemconfigure(item, start=(angle + piece * ring.step) % 360)
        size = max(2.0, 0.045 * radius)
        for (ring_index, dash), item in zip(orb.CAPS, self._caps, strict=True):
            ring = orb.RINGS[ring_index]
            theta = math.radians(frame.angles[ring_index] + dash * ring.step)
            x = cx + ring.radius * radius * math.cos(theta)
            y = cy - ring.radius * radius * math.sin(theta)
            canvas.coords(item, x - size, y - size, x + size, y + size)


def system_panel(view: View, window: Window) -> Panel:
    """The real Tk half, on the thread that calls this."""
    return TkPanel(view, window)
