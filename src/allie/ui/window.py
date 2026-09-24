"""The window: the product's face (plan.md D20; spec docs/specs/2026-09-19-window-design.md).

`allie run` opens it. An orb whose colour is the state and whose
pulse is the sound, the state and the session line under it, the finished
turns below, three buttons - and the setup wizard, on its own page, when
there is nothing set up yet or the settings button was pressed. It owns no
behaviour: what it shows is what `on_state`, `on_mode`, `on_session` and
`on_turn` say, like the status line and the tray, and what its buttons do
is what the key, the tray and `allie setup` already do.

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
what the transcript holds, where the orb is - and knows no Tk, so that
the tests read it and the Tk half only draws it. `Window` is the loop's
side: the `Screen` of `ui/status.py`, the queue, the thread. `TkPanel`
is the Tk half: widgets, the orb on a canvas, the tick.

**The orb** is `ui/orb.py`'s numbers drawn on a canvas: the glow Pillow
paints once per colour (Tk cannot blur), the core an oval that breathes,
the rings arcs whose `start` turns every tick, the dots ovals moved.

**Sixty frames a second, and one size** (task U1-U2). The tick asks for
fifteen milliseconds and subtracts the time the last one took, so the
period is the frame and not the frame plus the work; Windows is asked for
a one millisecond timer while the window is up, because at its usual
15.6 ms granularity a sixteen millisecond wait is rounded up to thirty-one
and sixty frames quietly become thirty. The window does not resize: it is
one shape at one size, so the maximise button is greyed out and the orb is
laid out once. Nothing here is a clock the animation reads - every frame
is still told how long it was.

**The wizard's page** is the `Prompter` of `setup_wizard.py` on a window:
`WindowPrompter` posts each question and awaits a future the Tk thread
resolves when Continue is pressed - so the wizard's checks (the key, the
model list, the tool probe, the microphone list) come to the window
without a line of the wizard changing.

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
    "Message",
    "Panel",
    "Style",
    "Switch",
    "TkPanel",
    "View",
    "Window",
    "WindowError",
    "WindowPrompter",
    "WizardPage",
    "plate",
    "system_panel",
]

TEXT: dict[str, str] = {
    "window_loading": "Starting...",
    "window_failed": "Could not start - the reason is below.",
    "window_settings": "Settings",
    "window_setting_up": "Settings are being changed; the assistant starts again afterwards.",
    "wizard_continue": "Continue",
    "wizard_cancel": "Cancel",
    "wizard_filter": "Type to filter the list",
}

# What the status line and the tray already say, borrowed under their keys.
STATUS_KEYS = (
    "loading_speech",
    "checking_model",
    "session_open",
    "session_closed",
    "session_minutes",
    "you_said",
    "it_said",
    "searched",
    "microphone_quiet",
)
TRAY_KEYS = ("tray_not_listening", "tray_stop_listening", "tray_start_listening", "tray_quit")

# Fifteen rather than sixteen: Windows' usual timer granularity is 15.6 ms,
# and a wait is rounded *up* to the next tick of it - sixteen would come
# back at 31 ms and draw thirty frames a second. Fifteen lands on 15.6 ms
# without the fine timer and on 15 ms with it; both are over sixty.
TICK_MS = 15
TRANSCRIPT_ROWS = 200
# One size, no resizing (U2): the window is a shape, not a workspace.
WINDOW_SIZE = (420, 640)
ORB_HEIGHT = 300
BACKGROUND: RGB = (11, 15, 20)
WHITE: RGB = (230, 243, 255)
# How long `start` waits for the Tk thread to put the window up, and
# `stop` for it to come down.
START_SECONDS = 5.0
STOP_SECONDS = 3.0
# A tick after a long sleep (the window hidden, the machine asleep) gets
# this much time at most, so the rings do not spin through a whole night.
MAX_DT = 0.25

Message = tuple[Any, ...]
Answer = asyncio.Future[str | None]


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

    def meter(self) -> str:
        which = self._status["session_open" if self.session.open else "session_closed"]
        minutes = self._status["session_minutes"].format(minutes=int(self.session.minutes))
        return f"{which} · {minutes}"

    def switch_label(self) -> str:
        return self._tray["tray_stop_listening" if self.listening else "tray_start_listening"]

    def quit_label(self) -> str:
        return self._tray["tray_quit"]

    def row_label(self, kind: str) -> str:
        if kind == "you":
            return self._status["you_said"]
        if kind == "it":
            return self._status["it_said"]
        if kind == "searched":
            return self._status["searched"]
        return ""

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

    # ------------------------------------------------------------------------

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
        """A question for the page: `kind` is `choose`, `secret` or `ask`.
        The future resolves with the answer, or `None` for walking away."""
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

FONT = "Segoe UI"
INK = "#e6f3ff"
DIM_INK = "#7f93a8"
PANE = "#0f151c"
PRESSED = "#1a2430"
NOTICE_INK = "#f39c12"
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
# because Tk has no anti-aliased corner either), the label as canvas text on
# top so it stays the system's own crisp glyphs, and four plates per button -
# at rest, under the pointer, held down, and out of use.


@dataclass(frozen=True, slots=True)
class Style:
    """What one kind of button wears. `edge` is the hairline round the plate,
    or nothing for a filled one; `hover_ink` recolours the label under the
    pointer, which only the one destructive button uses."""

    fill: RGB
    hover: RGB
    press: RGB
    edge: RGB | None
    ink: str
    hover_ink: str | None = None


# The everyday button: a plate barely above the background, lifting under
# the pointer.
QUIET = Style(fill=(22, 30, 41), hover=(33, 45, 60), press=(15, 21, 29), edge=(40, 54, 71), ink=INK)
# What the window is for: the orb's ready blue, filled.
ACCENT = Style(
    fill=(31, 106, 196), hover=(48, 133, 229), press=(24, 84, 158), edge=None, ink="#f3f9ff"
)
# Quit: the quiet plate until the pointer is on it, then it says so.
DANGER = Style(
    fill=(22, 30, 41),
    hover=(58, 29, 38),
    press=(44, 21, 29),
    edge=(40, 54, 71),
    ink=INK,
    hover_ink="#ff9aa7",
)

BUTTON_FONT = 10
BUTTON_PAD_X = 16
BUTTON_PAD_Y = 9
BUTTON_RADIUS = 9
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


class Button(tk.Canvas):
    """A rounded button that answers the pointer: `Style`'s four plates, the
    label in the system's font on top, and the command on release inside.

    Keyboard-reachable like Tk's own: it takes focus, wears the pointer's
    plate while it has it, and Return or Space presses it.
    """

    def __init__(
        self,
        parent: tk.Misc,
        *,
        text: str,
        command: Callable[[], None],
        style: Style = QUIET,
        scale: float = 1.0,
        background: RGB = BACKGROUND,
    ) -> None:
        super().__init__(
            parent,
            bg=hex_of(background),
            highlightthickness=0,
            bd=0,
            takefocus=True,
            cursor="hand2",
        )
        self._style = style
        self._command = command
        self._background = background
        self._pad = (round(BUTTON_PAD_X * scale), round(BUTTON_PAD_Y * scale))
        self._radius = max(2, round(BUTTON_RADIUS * scale))
        self._font = tkfont.Font(family=FONT, size=BUTTON_FONT)
        self._plates: dict[str, ImageTk.PhotoImage] = {}
        self._plate_item = self.create_image(0, 0, anchor="nw")
        self._label_item = self.create_text(0, 0, anchor="center", font=self._font, text="")
        self._enabled = True
        self._under = False
        self._held = False
        # None until the first `set_text`, so that a button born with no
        # words still gets its plates and its size when they arrive.
        self._text: str | None = None
        for event, handler in (
            ("<Enter>", self._entered),
            ("<Leave>", self._left),
            ("<ButtonPress-1>", self._pressed),
            ("<ButtonRelease-1>", self._released),
            ("<FocusIn>", self._entered),
            ("<FocusOut>", self._left),
            ("<Return>", self._struck),
            ("<space>", self._struck),
        ):
            self.bind(event, handler)
        self.set_text(text)

    # -- what the panel says to it -----------------------------------------

    def set_text(self, text: str) -> None:
        """The label, and with it the button's size: the plates are repainted
        only when the words really changed."""
        if text == self._text:
            return
        self._text = text
        width = self._font.measure(text) + 2 * self._pad[0]
        height = self._font.metrics("linespace") + 2 * self._pad[1]
        self.configure(width=width, height=height)
        self._plates = {
            name: ImageTk.PhotoImage(
                plate(width, height, self._radius, fill, edge, self._background)
            )
            for name, fill, edge in self._faces()
        }
        self.coords(self._label_item, width / 2, height / 2)
        self.itemconfigure(self._label_item, text=text)
        self._wear()

    def set_enabled(self, enabled: bool) -> None:
        if enabled != self._enabled:
            self._enabled = enabled
            self.configure(cursor="hand2" if enabled else "")
            self._wear()

    # -- what the pointer says to it ---------------------------------------

    def _entered(self, _: object = None) -> None:
        self._under = True
        self._wear()

    def _left(self, _: object = None) -> None:
        self._under = self._held = False
        self._wear()

    def _pressed(self, _: object) -> None:
        if not self._enabled:
            return
        self._held = True
        self.focus_set()
        self._wear()

    def _released(self, _: object) -> None:
        held, self._held = self._held, False
        self._wear()
        if held and self._enabled:
            self._command()

    def _struck(self, _: object) -> None:
        if self._enabled:
            self._command()

    # ----------------------------------------------------------------------

    def _faces(self) -> tuple[tuple[str, RGB, RGB | None], ...]:
        style = self._style
        faded = blend(style.fill, self._background, 1 - FADED)
        edge = None if style.edge is None else blend(style.edge, self._background, 1 - FADED)
        return (
            ("rest", style.fill, style.edge),
            ("hover", style.hover, style.edge),
            ("held", style.press, style.edge),
            ("off", faded, edge),
        )

    def _wear(self) -> None:
        if not self._plates:
            return
        if not self._enabled:
            face, ink = "off", DIM_INK
        elif self._held:
            face, ink = "held", self._style.ink
        elif self._under:
            face, ink = "hover", self._style.hover_ink or self._style.ink
        else:
            face, ink = "rest", self._style.ink
        self.itemconfigure(self._plate_item, image=self._plates[face])
        self.itemconfigure(self._label_item, fill=ink)


class TkPanel:
    """The widgets, the orb and the tick. Lives on the window's thread."""

    def __init__(self, view: View, window: Window) -> None:
        _dpi_aware()
        self._view = view
        self._window = window
        _timer_period(1, begin=True)
        root = self._root = tk.Tk()
        background = hex_of(BACKGROUND)
        root.title(APP_TITLE)
        root.configure(bg=background)
        self._scale = root.winfo_fpixels("1i") / 96.0
        width, height = (round(side * self._scale) for side in WINDOW_SIZE)
        root.geometry(f"{width}x{height}")
        # One size (U2): no drag on an edge, no maximise - Windows greys the
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

        # The face: orb, labels, transcript, buttons.
        self._face = tk.Frame(root, bg=background)
        self._canvas = tk.Canvas(
            self._face, bg=background, highlightthickness=0, height=round(ORB_HEIGHT * self._scale)
        )
        self._canvas.pack(fill="x")
        self._canvas.bind("<Configure>", self._resized)
        self._label = tk.Label(self._face, bg=background, fg=INK, font=(FONT, 14))
        self._label.pack()
        self._meter = tk.Label(self._face, bg=background, fg=DIM_INK, font=(FONT, 10))
        self._meter.pack(pady=(0, 8))
        # The bar first and at the bottom: packed after a transcript that
        # asks for twenty-four lines it would be pushed off the window.
        bar = tk.Frame(self._face, bg=background)
        bar.pack(side="bottom", fill="x", padx=12, pady=12)
        # The switch is what the window is for, so it is the filled one; quit
        # is quiet until the pointer is on it (U4).
        self._switch = self._button(bar, "", window.toggle, style=ACCENT)
        self._switch.pack(side="left")
        self._button(bar, view.said["window_settings"], window.settings).pack(side="left", padx=8)
        self._button(bar, view.quit_label(), self._quit, style=DANGER).pack(side="right")
        self._text = self._pane(self._face, height=6)
        self._text.tag_configure("who", foreground=DIM_INK)
        self._text.tag_configure("you", foreground=DIM_INK)
        self._text.tag_configure("it", foreground=INK)
        # No emoji in the row: Tcl 8.6.14 does not draw characters outside
        # the Basic Multilingual Plane reliably.
        self._text.tag_configure("searched", foreground=DIM_INK)
        self._text.tag_configure("notice", foreground=NOTICE_INK)
        self._text.pack(fill="both", expand=True, padx=12)
        self._face.pack(fill="both", expand=True)

        # The wizard's page, packed instead of the face while a wizard runs.
        self._page = tk.Frame(root, bg=background)
        self._lines = self._pane(self._page, height=8)
        self._lines.pack(fill="x", padx=12, pady=(12, 8))
        self._question = tk.Label(
            self._page,
            bg=background,
            fg=INK,
            font=(FONT, 11),
            justify="left",
            anchor="w",
            wraplength=round((WINDOW_SIZE[0] - 40) * self._scale),
        )
        self._question.pack(fill="x", padx=12)
        self._filter = self._entry_widget(self._page)
        self._filter.bind("<KeyRelease>", self._filtered)
        self._list = tk.Listbox(
            self._page,
            bg=PANE,
            fg=INK,
            selectbackground=PRESSED,
            selectforeground=INK,
            relief="flat",
            highlightthickness=0,
            font=(FONT, 10),
            activestyle="none",
        )
        self._list.bind("<Double-Button-1>", lambda _: self._continue_pressed())
        self._list.bind("<Return>", lambda _: self._continue_pressed())
        self._entry = self._entry_widget(self._page)
        self._entry.bind("<Return>", lambda _: self._continue_pressed())
        buttons = tk.Frame(self._page, bg=background)
        buttons.pack(side="bottom", fill="x", padx=12, pady=12)
        self._continue = self._button(
            buttons, view.said["wizard_continue"], self._continue_pressed, style=ACCENT
        )
        self._continue.pack(side="left")
        self._cancel = self._button(buttons, view.said["wizard_cancel"], self._cancel_pressed)
        self._cancel.pack(side="right")
        self._visible: list[Option] = []
        self._page_open = False
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
        self._colour: RGB | None = None

        self._shown = ("", "", "")
        self._rows_version = -1
        self._last = time.monotonic()
        self._paint()
        root.after(TICK_MS, self._tick)

    def run(self) -> None:
        try:
            self._root.mainloop()
        finally:
            _timer_period(1, begin=False)

    # -- widgets ---------------------------------------------------------------

    def _button(
        self, parent: tk.Misc, text: str, command: Callable[[], None], *, style: Style = QUIET
    ) -> Button:
        return Button(parent, text=text, command=command, style=style, scale=self._scale)

    def _pane(self, parent: tk.Misc, *, height: int | None = None) -> tk.Text:
        pane = tk.Text(
            parent,
            bg=PANE,
            fg=INK,
            relief="flat",
            wrap="word",
            state="disabled",
            font=(FONT, 10),
            padx=12,
            pady=8,
            highlightthickness=0,
        )
        if height is not None:
            pane.configure(height=height)
        return pane

    def _entry_widget(self, parent: tk.Misc) -> tk.Entry:
        return tk.Entry(
            parent,
            bg=PANE,
            fg=INK,
            insertbackground=INK,
            relief="flat",
            font=(FONT, 11),
            highlightthickness=1,
            highlightbackground=PRESSED,
            highlightcolor=DIM_INK,
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
        dt, self._last = min(started - self._last, MAX_DT), started
        self._draw(self._view.frame(dt))
        # The next frame is due `TICK_MS` after this one began, not after it
        # ended: asking for the whole period again would add the drawing to
        # it and leave the window short of sixty frames a second (U1).
        spent = round((time.monotonic() - started) * 1000)
        self._root.after(max(1, TICK_MS - spent), self._tick)

    def _paint(self) -> None:
        view = self._view
        shown = (view.label(), view.meter(), view.switch_label())
        if shown != self._shown:
            self._shown = shown
            self._label.configure(text=shown[0])
            self._meter.configure(text=shown[1])
            self._switch.set_text(shown[2])
        if view.version != self._rows_version:
            self._rows_version = view.version
            self._text.configure(state="normal")
            self._text.delete("1.0", "end")
            for kind, text in view.rows:
                who = view.row_label(kind)
                if who:
                    self._text.insert("end", f"{who}  ", "who")
                self._text.insert("end", f"{text}\n", kind)
            self._text.configure(state="disabled")
            self._text.see("end")
        page_wanted = view.wizard is not None
        if page_wanted != self._page_open:
            self._page_open = page_wanted
            if page_wanted:
                self._face.pack_forget()
                self._page.pack(fill="both", expand=True)
                self._lines_shown = -1
                self._asked_first = True
            else:
                self._page.pack_forget()
                self._face.pack(fill="both", expand=True)
        if view.wizard is not None:
            self._paint_page(view.wizard)

    def _paint_page(self, page: WizardPage) -> None:
        if len(page.lines) != self._lines_shown:
            self._lines_shown = len(page.lines)
            self._lines.configure(state="normal")
            self._lines.delete("1.0", "end")
            self._lines.insert("end", "\n".join(page.lines))
            self._lines.configure(state="disabled")
            self._lines.see("end")
        if page.answer is not self._asked or self._asked_first:
            self._asked = page.answer
            self._asked_first = False
            self._question.configure(text=page.question)
            for widget in (self._filter, self._list, self._entry):
                widget.pack_forget()
            self._entry.delete(0, "end")
            self._filter.delete(0, "end")
            if page.kind == "choose":
                self._filter.pack(fill="x", padx=12, pady=(8, 4))
                self._list.pack(fill="both", expand=True, padx=12)
                self._fill_list(page.options)
                self._filter.focus_set()
            elif page.kind in ("secret", "ask"):
                self._entry.configure(show="•" if page.kind == "secret" else "")
                self._entry.pack(fill="x", padx=12, pady=8)
                self._entry.focus_set()
            self._continue.set_enabled(page.open)
            self._cancel.set_enabled(page.open)

    def _fill_list(self, options: Sequence[Option]) -> None:
        self._visible = list(options)
        self._list.delete(0, "end")
        for option in self._visible:
            self._list.insert("end", option.label)
        if self._visible:
            self._list.selection_set(0)

    def _filtered(self, _: object) -> None:
        page = self._view.wizard
        if page is None:
            return
        needle = self._filter.get().casefold()
        self._fill_list([option for option in page.options if needle in option.label.casefold()])

    # -- clicks, on the Tk thread ------------------------------------------------

    def _continue_pressed(self) -> None:
        page = self._view.wizard
        if page is None or not page.open:
            return
        if page.kind == "choose":
            # Untyped in the stubs; a tuple of indices in life.
            chosen: tuple[int, ...] = self._list.curselection()  # type: ignore[no-untyped-call]
            if not chosen:
                return
            value = self._visible[int(chosen[0])].value
        else:
            value = self._entry.get()
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
        self._build()

    def _build(self) -> None:
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
            width = max(1, round(ring.width * radius))
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
                        box, start=0, extent=ring.extent, style="arc", width=width, outline=""
                    )
                    for _ in range(ring.dashes)
                ]
            self._dashes.append(items)
        self._applied = [None] * len(orb.RINGS)
        self._caps = [canvas.create_oval(0, 0, 1, 1, width=0) for _ in orb.CAPS]
        # Recoloured at the next draw.
        self._colour = None

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
