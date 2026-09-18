"""What the assistant can do on the machine itself (design.md section 3.6).

Phase 2.1c opened this file with the smallest tool there is: `get_current_time`
answers the one question the frozen system prompt of `agent/prompts.py`
cannot, because the prompt carries no clock so that its bytes never change
and the provider's cache keeps hitting (architecture guide section 2). Phase
2.2 adds what the owner asked for first (section 2): open an app, open a
site, open a page of Settings.

**Opening an app by name is a catalogue problem.** Windows has no "start
Spotify" call. It has the shortcut files in the two Start Menu folders and,
for everything else the Start menu knows - the store apps first of all - the
shell's own list, which `Get-StartApps` prints. Both are read once at startup
into an `AppCatalog`, in a worker thread, because hundreds of files and a
PowerShell process take seconds (measured 2026-09-09: 3.4 s for the listing
alone) and nothing on the event loop may (section 3.1 rule 4).

**The name the user said is not the name in the catalogue.** "krom" is
"Google Chrome"; "IŞIK" and "isik" are one word; the recogniser misspells.
`find` folds both sides with `normalize_search` and then tries the whole
name, one word of it, a prefix, and last `difflib`'s closest match - the
matcher of `store/names.py`, which the address book of
`messaging/contacts.py` uses for people (moved out of here on 15 September
2026; its constants and their measurements went with it). The language of
the catalogue never matters: on a Turkish Windows the shortcut is already
called "Hesap Makinesi", on an English one "Calculator", and the code
carries neither (section 3.12).

What a tool returns is addressed to the model, not the user, so it is written
the way a model reads best - unambiguous, in one line - and in English, like
everything else that never reaches the speaker. How it is said out loud, and
in which language, is the model's job.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from collections.abc import Awaitable, Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Protocol

from loguru import logger

from assistant import shell
from assistant.store.names import NameIndex
from assistant.store.normalize import normalize_search
from assistant.tools.registry import Tool, tool

__all__ = [
    "PROMPT_NAMES",
    "SETTINGS_PAGES",
    "TEXT",
    "AppCatalog",
    "AppEntry",
    "MediaNames",
    "get_current_time",
    "open_app_for",
    "open_settings",
    "open_url",
    "scan_start_apps",
    "scan_start_menu",
    "spoken_form",
    "start_menu_folders",
]

# How many of the catalogue's names are *offered* to the recogniser. What it
# takes is measured in tokens where the tokenizer is (`stt/local_whisper.py`,
# `PROMPT_TOKENS`); this is only the upper bound of the list handed over, so
# that a machine with hundreds of apps does not make it encode them all. Until
# 2026-09-13 this was 40 and the prompt was the forty *shortest* names -
# `Run`, `dfrgui`, `services` - and neither PyCharm nor FortiClient was in it.
PROMPT_NAMES = 120

# What a name carries on the shortcut and nobody says: a version (`PyCharm
# 2026.2.1`, `Python 3.13`), a year (`Word 2016`), a parenthesised tail
# (`Outlook (classic)`, `IDLE (Python 3.13 64-bit)`), in any order at the end.
# `Paint 3D` and `7-Zip` are names, not versions, and stay.
SPOKEN_TAIL = re.compile(r"(?:\s*\([^)]*\)|\s+v?\d+(?:\.\d+)+\S*|\s+\d{4})+\s*$", re.IGNORECASE)

# How long the shell's listing may take before it is given up on. Measured at
# 3.4 s; a machine ten times slower still answers.
LISTING_SECONDS = 30.0

# What the shell is asked. Its output encoding is set first: through a pipe
# PowerShell writes in the machine's legacy code page, which has no `ğ`, and a
# name like "Ekran görüntüsü geçmişi" would arrive damaged.
LISTING_COMMAND = (
    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
    "Get-StartApps | ConvertTo-Json -Compress"
)

# The folder the Start menu itself shows. An app is an item in it, by its
# `AppID`, and opening the item starts the app - store apps included, which
# have no shortcut file anywhere.
APPS_FOLDER = "shell:AppsFolder\\"

# Pages of Windows Settings: the key the model chooses, and the URI that opens
# it. The keys are English because the model maps the user's words to them
# from whatever language was spoken; `home` is the one to fall back on.
SETTINGS_PAGES: dict[str, str] = {
    "home": "ms-settings:",
    "bluetooth": "ms-settings:bluetooth",
    "wifi": "ms-settings:network-wifi",
    "network": "ms-settings:network-status",
    "display": "ms-settings:display",
    "sound": "ms-settings:sound",
    "notifications": "ms-settings:notifications",
    "power": "ms-settings:powersleep",
    "storage": "ms-settings:storagesense",
    "apps": "ms-settings:appsfeatures",
    "update": "ms-settings:windowsupdate",
    "privacy": "ms-settings:privacy",
    "about": "ms-settings:about",
}
PAGE_KEYS = "One of: " + ", ".join(SETTINGS_PAGES)

# What runs the shell, and what scans the machine. Both are parameters so a
# test can hand in a fake and no test ever starts PowerShell.
Runner = Callable[..., subprocess.CompletedProcess[str]]
Scanner = Callable[[], list["AppEntry"]]

# What `open_app` asks when the catalogue has no such application: the spoken
# name in, the answer out, or `None` when the name is not a media service
# either. `media/player.py::Player.open_named` is the one implementation.
MediaNames = Callable[[str], Awaitable[str | None]]


# What `open_app` asks last, when the media engine has no answer either: the
# Microsoft Store (`tools/store.py`). Only the three questions `open_app`
# puts to it are named here, so that this module never imports that one.
class StoreLookup(Protocol):
    @property
    def available(self) -> bool: ...

    async def search(self, words: str) -> StoreListing | None: ...

    def settled(self) -> bool: ...


class StoreListing(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def store_id(self) -> str: ...

    @property
    def publisher(self) -> str: ...

    @property
    def pricing(self) -> str: ...

    @property
    def installable(self) -> bool: ...


# The end of the chain of section 3.12 for the one word of `open_app`'s
# answer the user may hear: the publisher's name, when the Store gave none.
# `install_app`'s question reads it back, so the pack's word for it goes in
# here (`__main__`), and this is what stands when the pack has none.
TEXT: dict[str, str] = {
    "unknown_publisher": "unknown publisher",
}

# What `open_app` tells the model about the Store. `{closest}` is the
# nearest installed names, kept so that the model can still prefer an app
# that was nearly it over a download.
NOT_INSTALLED = (
    "No app called {name!r} is installed. The Microsoft Store has {listing!r} by {publisher}"
)
OFFERED = (
    " ({pricing}). To download it, call install_app(name={listing!r}, store_id={store_id!r}, "
    "publisher={publisher!r}) - it asks the user first; do not ask them yourself{closest}."
)
COSTS_MONEY = (
    ", but it costs money ({pricing}) and cannot be bought from here; the user can buy it in "
    "the Store{closest}."
)
PRICE_NOT_LISTED = "price not listed"


@dataclass(frozen=True, slots=True)
class AppEntry:
    """One thing the Start menu can open: its name, and what to hand the shell."""

    name: str
    # A shortcut's path, or `shell:AppsFolder\<AppID>`; `os.startfile` opens
    # either.
    launch: str

    @property
    def spoken(self) -> str:
        """The name as people say it - `PyCharm`, not `PyCharm 2026.2.1`."""
        return spoken_form(self.name)


def spoken_form(name: str) -> str:
    """`name` without what nobody says out loud (`SPOKEN_TAIL`); `name` itself
    when nothing would be left of it."""
    return SPOKEN_TAIL.sub("", name).strip() or name


# --------------------------------------------------------------------------
# Reading the machine
# --------------------------------------------------------------------------


def start_menu_folders() -> list[Path]:
    """The user's Start Menu programs and the machine's, where either is set."""
    roots = (os.environ.get("APPDATA"), os.environ.get("PROGRAMDATA"))
    return [Path(root, "Microsoft", "Windows", "Start Menu", "Programs") for root in roots if root]


def scan_start_menu(folders: Iterable[Path] | None = None) -> list[AppEntry]:
    """Every shortcut under the Start Menu folders, named by its file name.

    An "Uninstall Foo" shortcut is kept: it is something the user can ask
    for, and it is the model, not this scan, that decides what "open Foo"
    meant. Sorted, so that when a name occurs twice the same one wins on
    every start.
    """
    found: list[AppEntry] = []
    for folder in start_menu_folders() if folders is None else folders:
        if not folder.is_dir():
            continue
        found.extend(
            AppEntry(name=shortcut.stem, launch=str(shortcut))
            for shortcut in sorted(folder.rglob("*.lnk"))
        )
    return found


def scan_start_apps(run: Runner = subprocess.run) -> list[AppEntry]:
    """Everything the Start menu lists, asked of the shell itself.

    `Get-StartApps` is the one documented way to the store apps - there is
    no shortcut file for Calculator - and it lists the classic ones too, so
    a name can come back twice; `AppCatalog` keeps the first. Nothing here
    is fatal: no PowerShell, a PowerShell that fails, output that is not
    JSON, each leaves the catalogue with the shortcuts alone and a line in
    the log. The workbook called this `scan_uwp_apps`; it is named for what
    the listing holds.
    """
    shell = shutil.which("powershell")
    if shell is None:
        logger.warning("PowerShell was not found; the catalogue has the Start Menu shortcuts only")
        return []

    try:
        listing = run(
            [shell, "-NoProfile", "-NonInteractive", "-Command", LISTING_COMMAND],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=LISTING_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as failure:
        logger.warning("the Start menu listing could not be read: {}", failure)
        return []

    if listing.returncode != 0:
        logger.warning("the Start menu listing failed: {}", listing.stderr.strip())
        return []

    try:
        printed = json.loads(listing.stdout or "null")
    except json.JSONDecodeError as failure:
        logger.warning("the Start menu listing was not JSON: {}", failure)
        return []
    return list(_listed(printed))


def _listed(printed: object) -> Iterator[AppEntry]:
    # One app is printed as an object, several as a list of them. Anything
    # without both a name and an id is not something that can be opened.
    items = [printed] if isinstance(printed, dict) else printed if isinstance(printed, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        name, app_id = item.get("Name"), item.get("AppID")
        if isinstance(name, str) and name.strip() and isinstance(app_id, str) and app_id:
            yield AppEntry(name=name.strip(), launch=APPS_FOLDER + app_id)


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------


class AppCatalog:
    """The apps this machine can open, found by whatever the user called them."""

    def __init__(self, entries: Iterable[AppEntry], *, scanners: Iterable[Scanner] = ()) -> None:
        self._scanners = tuple(scanners)
        self._entries: list[AppEntry] = []
        self._index: NameIndex[AppEntry] = NameIndex()
        self._fill(entries)

    def _fill(self, entries: Iterable[AppEntry]) -> None:
        self._entries.clear()
        self._index.clear()
        listed: set[str] = set()
        for entry in entries:
            key = normalize_search(entry.name).strip()
            # The same name twice - a shortcut and its store twin - is one
            # app, and the first one offered is the one kept.
            if not key or key in listed:
                continue
            listed.add(key)
            self._entries.append(entry)
            # The listed name always finds its own app, even when another
            # app is *said* the same way and was listed first ("Outlook
            # (classic)" before "Outlook"); the spoken form is an alias that
            # only stands where no listed name does (`store/names.py`).
            self._index.add(entry, entry.name, spoken=entry.spoken)

    @classmethod
    async def load(cls, *, scanners: Iterable[Scanner] | None = None) -> AppCatalog:
        """Scans the machine - off the event loop, because it takes seconds."""
        chosen = tuple(scanners) if scanners is not None else (scan_start_menu, scan_start_apps)
        catalog = cls(await asyncio.to_thread(_scan, chosen), scanners=chosen)
        logger.info("app catalogue: {} apps", len(catalog))
        return catalog

    async def refresh(self) -> None:
        """Scans the machine again, into this same catalogue.

        The tools hold this object, not a list, so an app installed since
        the start (`tools/store.py`) is found by the next call without any of
        them changing hands. The recogniser's prompt is not rebuilt: it was
        fitted when the model loaded, and the new name is understood after
        the next start. A catalogue built by hand has nothing to scan.
        """
        if not self._scanners:
            return
        self._fill(await asyncio.to_thread(_scan, self._scanners))
        logger.info("app catalogue: {} apps (refreshed)", len(self))

    def __len__(self) -> int:
        return len(self._entries)

    def names(self) -> list[str]:
        return [entry.name for entry in self._entries]

    def spoken_names(self, limit: int = PROMPT_NAMES, *, first: Iterable[str] = ()) -> list[str]:
        """The names most worth telling the recogniser about, as people say
        them, each once.

        `first` is what this user has asked to open before, in their own
        words (`AuditRepo.names_asked`): each is resolved the way `open_app`
        resolves it, and the apps found lead the list - the window is short
        (`stt/local_whisper.py`), and the apps someone opens are the ones
        they will say again. Then products - a name with a capital letter in
        it, in every language the catalogue has been seen in; `dfrgui`,
        `services`, `computer` are what the machine calls its own utilities -
        and among them the shorter first, because a short name is one people
        say.
        """
        distinct: dict[str, None] = {}
        for said in first:
            found = self.find(said)
            if found is not None:
                distinct.setdefault(found.spoken, None)
        rest: dict[str, None] = {}
        for entry in self._entries:
            if entry.spoken not in distinct:
                rest.setdefault(entry.spoken, None)
        ranked = sorted(rest, key=lambda name: (name == name.lower(), len(name)))
        return [*distinct, *ranked][:limit]

    def find(self, spoken: str) -> AppEntry | None:
        """The app the user meant by `spoken`, or `None`.

        In order: the whole name, one word of a name ("chrome" is Google
        Chrome), the start of either ("spot"), and the closest by
        `difflib` ("krom"). Folded on both sides, so the case, the accents
        and the recogniser's spelling of a foreign name do not decide
        (`store/names.py`).
        """
        return self._index.find(spoken)

    def closest(self, spoken: str, *, limit: int = 3) -> list[str]:
        """Names near `spoken`, for the model to offer when nothing matched."""
        return self._index.closest(spoken, limit=limit)


def _scan(scanners: Iterable[Scanner]) -> list[AppEntry]:
    return [entry for scanner in scanners for entry in scanner()]


# --------------------------------------------------------------------------
# The tools
# --------------------------------------------------------------------------


def _now() -> datetime:
    """The local time with its zone attached. Kept apart so a test can pin it."""
    return datetime.now().astimezone()


@tool(risk="safe")
async def get_current_time() -> str:
    """Returns the current local date, time, weekday and time zone. Call it before
    answering anything that depends on today's date or the time of day."""
    now = _now()
    # ISO for the date and time, because every model reads it without
    # ambiguity; the weekday and the zone by name, because "yarın" and "bu
    # akşam" are questions about those.
    return f"{now.isoformat(timespec='minutes')} {now:%A}, {now.tzname()}"


def open_app_for(
    catalog: AppCatalog,
    *,
    media: MediaNames | None = None,
    store: StoreLookup | None = None,
    unknown_publisher: str = TEXT["unknown_publisher"],
) -> Tool:
    """`open_app`, bound to the catalogue it looks names up in.

    A closure rather than a parameter: the schema the model sees is read from
    the function's signature, and neither the catalogue nor the media engine
    nor the Store is something the model chooses (python-guide 3.4).

    `media` covers the names that are services rather than applications.
    YouTube Music has no application on Windows at all, and a machine without
    Spotify installed still has a Spotify the user can be shown. **It is asked
    second, after the catalogue**, so that a real installed application always
    wins its own name: someone with the Spotify application gets the
    application, and only someone without it gets the website. This module
    stays ignorant of which names those are - `media/player.py` knows.

    `store` is asked **last**, and only when it can be (`available`): an app
    that is neither installed nor a service may be in the Microsoft Store,
    and the model is told how to have it downloaded - through `install_app`,
    which asks the user (2026-09-13). The name the user said goes to the
    Store for that lookup and nowhere else.
    """

    @tool(risk="safe")
    async def open_app(name: Annotated[str, "The application, as the user said it."]) -> str:
        """Opens an application installed on this computer by name. Pass the
        name exactly as it was transcribed, even when it looks wrong or like
        other words ('pay charm', 'porti client'): the matcher is built for
        what the recogniser makes of names. Never translate it, correct it, or
        replace it with a guess of your own. When nothing matches, the answer
        names the closest installed apps - call again with one of those when
        it is plainly what was meant - and, when the Microsoft Store has the
        app, how to download it: call install_app as the answer says, it asks
        the user. Windows may list the app under its English name (Calculator,
        Settings, Notepad); ask the user only when all of that fails. To play
        something rather than to open the app it plays in, use play_music or
        play_video."""
        found = catalog.find(name)
        if found is None and store is not None and store.settled():
            # A download that outlived its turn has ended: the Start menu has
            # an entry the catalogue was read too early to know.
            await catalog.refresh()
            found = catalog.find(name)
        if found is not None:
            # `os.startfile` returns as soon as the shell has taken the
            # request, but taking it can be a store app's activation - long
            # enough to be kept off the loop.
            await shell.open_target(found.launch)
            return f"Opened {found.name}."

        if media is not None:
            service = await media(name)
            if service is not None:
                return service

        near = catalog.closest(name)
        if store is not None and store.available:
            listing = await store.search(name)
            if listing is not None:
                closest = f"; closest installed names: {', '.join(near)}" if near else ""
                publisher = listing.publisher or unknown_publisher
                pricing = listing.pricing or PRICE_NOT_LISTED
                said = NOT_INSTALLED.format(name=name, listing=listing.name, publisher=publisher)
                if listing.installable:
                    return said + OFFERED.format(
                        pricing=pricing,
                        listing=listing.name,
                        store_id=listing.store_id,
                        publisher=publisher,
                        closest=closest,
                    )
                return said + COSTS_MONEY.format(pricing=pricing, closest=closest)

        hint = f"; closest names: {', '.join(near)}" if near else ""
        return f"No app called {name!r}{hint}."

    return open_app


@tool(risk="safe")
async def open_url(url: Annotated[str, "A web address; 'https://' is added when missing."]) -> str:
    """Opens a web address in the user's default browser. Use it for a site the
    user named or an address that came up in the conversation. Not for music or
    video: an address you write for those either opens a search the user then
    has to click, or names an identifier you cannot know. play_music and
    play_video look the real one up first."""
    address = url.strip()
    if "://" not in address:
        address = f"https://{address}"

    if not await shell.open_address(address):
        raise RuntimeError(f"no browser would open {address}")
    return f"Opened {address}."


@tool(risk="safe")
async def open_settings(page: Annotated[str, PAGE_KEYS]) -> str:
    """Opens a page of Windows Settings. Choose the key nearest to what the
    user asked for, whatever language they asked in: screen brightness is
    display, wireless is wifi, the volume is sound. A key that is not on the
    list opens the Settings home page instead."""
    key = page.strip().casefold()
    uri = SETTINGS_PAGES.get(key)
    if uri is None:
        await shell.open_target(SETTINGS_PAGES["home"])
        return (
            f"No settings page called {page!r}; opened the Settings home page. "
            f"The pages are: {', '.join(SETTINGS_PAGES)}."
        )

    await shell.open_target(uri)
    return f"Opened the {key} page of Settings."
