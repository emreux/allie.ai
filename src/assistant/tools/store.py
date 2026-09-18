"""The Microsoft Store, for an app the user asked for and does not have
(design.md section 3.6, 2026-09-13).

"Open X" and there is no X on the machine used to end with the nearest names
and nothing else. Now `open_app` asks the Store whether there is an X, and
when there is, the model is handed `install_app` - a `confirm` tool, so the
one question that matters is put to the user by the gate, out loud, with the
app's name *and its publisher*: the Store is full of look-alikes, and "ChatGPT
by OpenAI" is a different answer from "ChatGPT by somebody". A yes downloads;
a no downloads nothing; the tool never asks anything itself (section 3.9).

**winget is the one channel.** `winget search --source msstore` finds the app
in about a second and a half, `winget show` names the publisher and the price
class, and `winget install --source msstore` installs a free app with no
window and no click. The Store's own `ms-windows-store:` page would leave the
user to press Install; the WinRT purchase API wants a packaged app and shows
a dialog of its own. What winget prints is read by *shape* - a Store id is
twelve characters starting with 9, or XP and twelve more - never by the
words of its column headers, which winget writes in the machine's language.
The two labels `show` is read for are English; a winget that prints them in
another language leaves the publisher and the price unknown, and an unknown
price is *tried*, because winget refuses a paid app itself, with its own
words, and refusing every app on a Turkish Windows would be the worse error.

**A download can outlast the turn.** The turn has a minute to its first word
(`app.THINKING_TIMEOUT`), and the confirmation is already inside it, so the
install is waited for `INSTALL_WAIT_SECONDS` and then left to finish on its
own - the process is kept, not killed, and `settled()` tells `open_app` that
the catalogue is stale the next time someone asks for the app. Everything
that blocks runs in a worker thread (section 3.1 rule 4).

What a tool returns is addressed to the model - English, one line, what
happened and what to do next. The one sentence the user hears, the question,
comes from the locale pack with `TEXT` as the end of the chain (section
3.12), handed in by the composition root as `forget`'s is.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import IO, Annotated, Any, Protocol

from loguru import logger

from assistant import shell
from assistant.store.normalize import normalize_search
from assistant.tools.registry import Tool, tool
from assistant.tools.system import AppCatalog

__all__ = [
    "INSTALL_WAIT_SECONDS",
    "SEARCH_SECONDS",
    "TEXT",
    "Install",
    "Listing",
    "Store",
    "WingetStore",
    "choose",
    "install_app_for",
    "parse_search",
    "parse_show",
]

# The last link of the chain of section 3.12 for the one sentence a user hears
# from this module: the question before anything is downloaded. Both
# placeholders are arguments the model must send (`registry.tool` checks).
TEXT: dict[str, str] = {
    "store_install_confirm": "'{name}' ({publisher}) will be downloaded from the Microsoft Store.",
}

# The answers, addressed to the model.
NOT_AN_ID = "That is not a Store id ({store_id!r}); call open_app again and use the id it gives."
INSTALLED = "Installed and opened {name!r}."
NOT_LISTED_YET = (
    "Installed {name!r}, but it is not in the Start menu yet; ask the user to open "
    "it themselves, or try open_app again in a moment."
)
FAILED = (
    "The Store install of {name!r} failed: {detail}. "
    "Tell the user; they can install it from the Store themselves."
)
STILL_DOWNLOADING = (
    "{name!r} is still downloading; it continues in the background. Tell the user it "
    "will be ready in a minute or two and that they can ask to open it then."
)

# The Store's source in winget, and the flags every call carries: the source
# agreement is accepted once and for all, and nothing may wait for a key.
SOURCE = "msstore"
QUIET = ("--accept-source-agreements", "--disable-interactivity")

# How long a search (and the `show` after it) may take. Measured at 1.4 s and
# 1.5 s; a slow connection is given ten times that before the app is
# reported as not found.
SEARCH_SECONDS = 20.0

# How long an install is waited for before the turn goes on without it. The
# turn has sixty seconds to its first word, and the question and the answer
# are inside them already.
INSTALL_WAIT_SECONDS = 30.0

# What winget says when the install is still running when the wait ends.
STILL_RUNNING = "still downloading"

# A Store product id: `9PLM9XGG6VKS`, `XP89DCGQ3K6VLD`. This is how a row of
# the search table is told from a header or a footer, whatever language the
# headers are in, and how an id the model made up is refused.
STORE_ID = re.compile(r"(?<!\S)(9[A-Z0-9]{11}|XP[A-Z0-9]{12})(?!\S)")
WHOLE_STORE_ID = re.compile(r"^(9[A-Z0-9]{11}|XP[A-Z0-9]{12})$")

# The two lines of `winget show` that matter, in the words winget prints
# them in English. Another language leaves both unknown (see the module
# docstring).
PUBLISHER_LINE = re.compile(r"^\s*Publisher:\s*(.+?)\s*$", re.MULTILINE)
PRICING_LINE = re.compile(r"^\s*Pricing:\s*(.+?)\s*$", re.MULTILINE)

# The price classes winget can install. Everything it names otherwise costs
# money; nothing named at all is tried (module docstring).
INSTALLABLE = frozenset({"free", "freemium"})

# What runs winget. Parameters so that a test hands in a fake and no test
# ever starts winget - the same seam `tools/system.py` has for PowerShell.
Runner = Callable[..., subprocess.CompletedProcess[str]]
Starter = Callable[..., Any]
Which = Callable[[str], str | None]


@dataclass(frozen=True, slots=True)
class Listing:
    """One app the Store has: what it is called, its id, who made it, and
    whether it costs money."""

    name: str
    store_id: str
    publisher: str
    pricing: str

    @property
    def installable(self) -> bool:
        return not self.pricing or self.pricing.casefold() in INSTALLABLE


@dataclass(frozen=True, slots=True)
class Install:
    """How an install went: whether winget finished in time, whether it
    succeeded, and its last words."""

    done: bool
    ok: bool
    detail: str


class Store(Protocol):
    """What `open_app` and `install_app` need of the Store."""

    @property
    def available(self) -> bool: ...

    async def search(self, words: str) -> Listing | None: ...

    async def install(self, store_id: str) -> Install: ...

    def settled(self) -> bool: ...


# --------------------------------------------------------------------------
# Reading what winget prints
# --------------------------------------------------------------------------


def parse_search(printed: str) -> list[tuple[str, str]]:
    """The rows of a `winget search` table as (name, id), by the shape of the
    id: the columns are named in the machine's language, the id is not."""
    rows: list[tuple[str, str]] = []
    for line in printed.splitlines():
        found = STORE_ID.search(line)
        if found is None:
            continue
        name = line[: found.start()].strip()
        if name:
            rows.append((name, found.group(1)))
    return rows


def choose(rows: Sequence[tuple[str, str]], words: str) -> tuple[str, str]:
    """The row the user most likely meant: the one called exactly that, else
    the one whose name starts with it, else the Store's own first answer -
    its relevance order put the official ChatGPT first (2026-09-13)."""
    wanted = normalize_search(words).strip()
    folded = [
        (normalize_search(name).strip(), row) for name, row in ((row[0], row) for row in rows)
    ]
    for key, row in folded:
        if key == wanted:
            return row
    for key, row in folded:
        if wanted and key.startswith(wanted):
            return row
    return rows[0]


def parse_show(printed: str) -> tuple[str, str]:
    """The publisher and the price class from `winget show`; "" for either
    winget did not print, or printed in words other than English."""
    publisher = PUBLISHER_LINE.search(printed)
    pricing = PRICING_LINE.search(printed)
    return (
        publisher.group(1) if publisher else "",
        pricing.group(1) if pricing else "",
    )


# --------------------------------------------------------------------------
# winget
# --------------------------------------------------------------------------


class WingetStore:
    """The Store through winget. Everything below `search` and `install` runs
    in a worker thread."""

    def __init__(
        self,
        *,
        run: Runner = subprocess.run,
        popen: Starter = subprocess.Popen,
        which: Which = shutil.which,
    ) -> None:
        self._run = run
        self._popen = popen
        self._winget = which("winget")
        # Installs that outlived their wait, until `settled` has seen them end.
        self._pending: list[Any] = []
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self._winget is not None

    async def search(self, words: str) -> Listing | None:
        return await asyncio.to_thread(self._search_now, words)

    async def install(self, store_id: str) -> Install:
        return await asyncio.to_thread(self._install_now, store_id)

    def settled(self) -> bool:
        """Whether an install that was left running has ended since the last
        time this was asked - the sign that the catalogue is stale."""
        with self._lock:
            ended = [process for process in self._pending if process.poll() is not None]
            for process in ended:
                self._pending.remove(process)
        return bool(ended)

    def _search_now(self, words: str) -> Listing | None:
        printed = self._printed("search", "--name", words)
        rows = parse_search(printed or "")
        if not rows:
            logger.debug("the Store lists nothing for {!r}", words)
            return None
        name, store_id = choose(rows, words)
        # The publisher and the price are worth a second call; failing to
        # get them loses those two words, not the app.
        publisher, pricing = parse_show(self._printed("show", "--id", store_id) or "")
        return Listing(name=name, store_id=store_id, publisher=publisher, pricing=pricing)

    def _printed(self, *arguments: str) -> str | None:
        """winget's output for `arguments`, or `None` when it could not be
        asked. A non-zero exit with output is still output: "no package
        found" exits 20 and is read as an empty table."""
        if self._winget is None:
            return None
        command = [self._winget, *arguments, "--source", SOURCE, *QUIET]
        try:
            done = self._run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=SEARCH_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired) as failure:
            logger.warning("winget could not be asked ({}): {}", arguments[0], failure)
            return None
        return str(done.stdout or "")

    def _install_now(self, store_id: str) -> Install:
        if self._winget is None:
            return Install(done=True, ok=False, detail="winget is not installed")
        command = [
            self._winget,
            "install",
            "--id",
            store_id,
            "--source",
            SOURCE,
            "--accept-package-agreements",
            *QUIET,
        ]
        try:
            process = self._popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as failure:
            logger.warning("winget could not be started: {}", failure)
            return Install(done=True, ok=False, detail=str(failure))

        # Read in a thread of its own: a process nobody reads blocks on a
        # full pipe, and this one may be left running after the wait.
        lines: list[str] = []
        reader = threading.Thread(target=_drain, args=(process.stdout, lines), daemon=True)
        reader.start()
        try:
            process.wait(timeout=INSTALL_WAIT_SECONDS)
        except subprocess.TimeoutExpired:
            with self._lock:
                self._pending.append(process)
            logger.info(
                "Store install {} is still running after {} s", store_id, INSTALL_WAIT_SECONDS
            )
            return Install(done=False, ok=False, detail=STILL_RUNNING)
        reader.join(timeout=1.0)
        detail = _last(lines)
        ok = process.returncode == 0
        logger.info("Store install {} {}: {}", store_id, "succeeded" if ok else "failed", detail)
        return Install(done=True, ok=ok, detail=detail)


def _drain(stream: IO[str] | None, into: list[str]) -> None:
    if stream is None:
        return
    for line in stream:
        into.append(line.rstrip())


def _last(lines: Sequence[str]) -> str:
    """winget's last words: the last line that says something. Progress is
    drawn with carriage returns, so a line is cut at its last one."""
    for line in reversed(lines):
        said = line.rsplit("\r", 1)[-1].strip()
        if said:
            return said
    return ""


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------


def install_app_for(
    catalog: AppCatalog, store: Store, *, confirm_prompt: str = TEXT["store_install_confirm"]
) -> Tool:
    """`install_app`, bound to the Store it installs from and the catalogue it
    refreshes afterwards; declared together with the question it asks."""

    @tool(risk="confirm", confirm_prompt=confirm_prompt)
    async def install_app(
        name: Annotated[str, "The app's name as the Store lists it, from open_app's answer."],
        store_id: Annotated[str, "The Store id from open_app's answer, such as 9PLM9XGG6VKS."],
        publisher: Annotated[str, "The publisher from open_app's answer, word for word."],
    ) -> str:
        """Downloads and installs an app from the Microsoft Store that the user
        does not have, then opens it. Call it only with the name, id and
        publisher that open_app's answer gave you. The user is asked before
        anything is downloaded, so do not ask them yourself; and never call it
        for an app open_app did not offer."""
        wanted = store_id.strip().upper()
        if not WHOLE_STORE_ID.match(wanted):
            return NOT_AN_ID.format(store_id=store_id)

        outcome = await store.install(wanted)
        if not outcome.done:
            return STILL_DOWNLOADING.format(name=name)
        if not outcome.ok:
            return FAILED.format(name=name, detail=outcome.detail)

        # The Start menu has a new entry; the catalogue was read at startup.
        await catalog.refresh()
        found = catalog.find(name)
        if found is None:
            return NOT_LISTED_YET.format(name=name)
        await shell.open_target(found.launch)
        return INSTALLED.format(name=found.name)

    return install_app
