"""The Microsoft Store, reached through winget: search, install, and the tool
that asks first (2026-09-13).

Nothing here starts winget. `run`, `popen` and `which` are parameters of
`WingetStore`, and what winget prints is a fixture - the real table of
2026-09-13 among them - including the ways it fails. The install tool is
bound to a fake store and a hand-built catalogue, and `shell.launch` is
replaced, so the one thing a test can observe is what would have happened.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

from assistant import shell
from assistant.tools.registry import Tool
from assistant.tools.store import (
    INSTALL_WAIT_SECONDS,
    SEARCH_SECONDS,
    TEXT,
    Install,
    Listing,
    WingetStore,
    choose,
    install_app_for,
    parse_search,
    parse_show,
)
from assistant.tools.system import AppCatalog, AppEntry

# What `winget search --source msstore --name ChatGPT` printed on 2026-09-13.
SEARCHED = """\
Name                                      Id           Version
--------------------------------------------------------------
ChatGPT                                   9PLM9XGG6VKS Unknown
ChatGPT Classic                           9NT1R1C2HH7J Unknown
ChatGPT (Beta)                            9N8CJ4W95TBZ Unknown
Codex AI Powered by (OpenAI ChatGPT)      9NHK91B6G00J Unknown
Ai Chatbot for OpenAi/ChatGPT             9MT4XQ6DWQLR Unknown
Chatbot Plus - (OpenAI/ChatGPT) Assistant 9P3T6NLNMDQV Unknown
"""

NOTHING = "No package found matching input criteria.\n"

# The head of `winget show --source msstore --id 9PLM9XGG6VKS`, the same day.
SHOWN = """\
Found ChatGPT [9PLM9XGG6VKS]
Version: Unknown
Publisher: OpenAI
Publisher Url: https://openai.com/codex/
Description:
  ChatGPT is your everyday AI assistant for Windows.
License: ms-windows-store://pdp/?ProductId=9PLM9XGG6VKS
Agreements:
  Category: Developer tools
  Pricing: Freemium
  Free Trial: No
"""


@dataclass
class Printed:
    """What winget would print for each subcommand, and how it would exit."""

    search: str = SEARCHED
    search_code: int = 0
    show: str = SHOWN
    show_code: int = 0
    failure: BaseException | None = None
    calls: list[list[str]] = field(default_factory=list)
    threads: list[threading.Thread] = field(default_factory=list)

    def run(self, command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(command)
        self.threads.append(threading.current_thread())
        assert options["timeout"] == SEARCH_SECONDS
        assert options["encoding"] == "utf-8"
        if self.failure is not None:
            raise self.failure
        if command[1] == "show":
            return subprocess.CompletedProcess(command, self.show_code, self.show, "")
        return subprocess.CompletedProcess(command, self.search_code, self.search, "")


class FakeProcess:
    """A winget install under way: its output, and when it ends."""

    def __init__(self, *, lines: list[str], code: int | None = 0, ends: bool = True) -> None:
        self.stdout: Iterator[str] = iter(line + "\n" for line in lines)
        self.returncode: int | None = None
        self._code = code
        self._ends = ends
        self.waited: list[float] = []

    def wait(self, timeout: float | None = None) -> int:
        self.waited.append(timeout or 0.0)
        if not self._ends:
            raise subprocess.TimeoutExpired("winget", timeout or 0.0)
        self.returncode = self._code
        return self._code or 0

    def poll(self) -> int | None:
        return self.returncode

    def finish(self, code: int) -> None:
        self.returncode = code


class Started:
    def __init__(self, process: FakeProcess) -> None:
        self.process = process
        self.calls: list[list[str]] = []
        self.threads: list[threading.Thread] = []

    def popen(self, command: list[str], **options: Any) -> FakeProcess:
        self.calls.append(command)
        self.threads.append(threading.current_thread())
        assert options["stdout"] is subprocess.PIPE
        assert options["stderr"] is subprocess.STDOUT
        return self.process


def winget(path: str) -> str | None:
    return r"C:\winget.exe" if path == "winget" else None


def nothing(path: str) -> str | None:
    return None


def store(printed: Printed | None = None, started: Started | None = None) -> WingetStore:
    printed = printed or Printed()
    started = started or Started(FakeProcess(lines=["Successfully installed"]))
    return WingetStore(run=printed.run, popen=started.popen, which=winget)


# --------------------------------------------------------------------------
# Reading what winget prints
# --------------------------------------------------------------------------


def test_the_table_becomes_names_and_ids_whatever_the_column_headers_say() -> None:
    """The id is found by its shape, not by the header's words: winget
    speaks the machine's language, and the columns are called something
    else on a Turkish Windows."""
    rows = parse_search(SEARCHED)

    assert rows[0] == ("ChatGPT", "9PLM9XGG6VKS")
    assert rows[3] == ("Codex AI Powered by (OpenAI ChatGPT)", "9NHK91B6G00J")
    assert rows[5] == ("Chatbot Plus - (OpenAI/ChatGPT) Assistant", "9P3T6NLNMDQV")
    assert len(rows) == 6


def test_a_name_cut_short_by_the_console_is_kept_as_printed() -> None:
    rows = parse_search(
        "Name          Id           Version\n-----\nA very long n… 9ABCDEFGHIJK Unknown\n"
    )

    assert rows == [("A very long n…", "9ABCDEFGHIJK")]


def test_the_other_shape_of_store_id_is_a_row_too() -> None:
    rows = parse_search("Name  Id  Version\n---\nZoom  XP99J3KP4XZ4VV Unknown\n")

    assert rows == [("Zoom", "XP99J3KP4XZ4VV")]


@pytest.mark.parametrize("printed", ["", NOTHING, "Name  Id  Version\n---\n", "garbage\n"])
def test_no_table_is_no_rows(printed: str) -> None:
    assert parse_search(printed) == []


def test_the_exact_name_is_chosen_over_the_first_row() -> None:
    rows = [("ChatGPT Classic", "9NT1R1C2HH7J"), ("ChatGPT", "9PLM9XGG6VKS")]

    assert choose(rows, "chatgpt") == ("ChatGPT", "9PLM9XGG6VKS")


def test_a_name_that_starts_with_what_was_said_beats_the_first_row() -> None:
    rows = [
        ("Codex AI Powered by (OpenAI ChatGPT)", "9NHK91B6G00J"),
        ("ChatGPT (Beta)", "9N8CJ4W95TBZ"),
    ]

    assert choose(rows, "ChatGPT") == ("ChatGPT (Beta)", "9N8CJ4W95TBZ")


def test_otherwise_the_store_s_own_first_answer_is_taken() -> None:
    rows = parse_search(SEARCHED)

    assert choose(rows, "open ai chat") == ("ChatGPT", "9PLM9XGG6VKS")


def test_the_publisher_and_the_price_class_are_read_from_show() -> None:
    assert parse_show(SHOWN) == ("OpenAI", "Freemium")


def test_what_show_does_not_say_is_empty() -> None:
    assert parse_show("Found X [9ABCDEFGHIJK]\nVersion: 1.0\n") == ("", "")


@pytest.mark.parametrize(
    ("pricing", "installable"),
    [
        ("Free", True),
        ("Freemium", True),
        ("free", True),
        ("", True),
        ("Paid", False),
        ("Trial", False),
    ],
)
def test_only_a_free_app_is_installable_and_an_unknown_price_is_tried(
    pricing: str, installable: bool
) -> None:
    """winget cannot buy. An empty price is what a winget that prints its
    labels in another language gives, and the install is tried rather than
    refused - a paid app fails there, with winget's own words."""
    assert Listing("X", "9ABCDEFGHIJK", "Y", pricing).installable is installable


# --------------------------------------------------------------------------
# Searching
# --------------------------------------------------------------------------


def test_without_winget_there_is_no_store() -> None:
    assert WingetStore(which=nothing).available is False
    assert store().available is True


async def test_searching_asks_winget_the_store_source_and_reads_the_best_row() -> None:
    printed = Printed()

    found = await store(printed).search("ChatGPT")

    assert found == Listing("ChatGPT", "9PLM9XGG6VKS", "OpenAI", "Freemium")
    assert printed.calls[0] == [
        r"C:\winget.exe",
        "search",
        "--name",
        "ChatGPT",
        "--source",
        "msstore",
        "--accept-source-agreements",
        "--disable-interactivity",
    ]
    assert printed.calls[1][:4] == [r"C:\winget.exe", "show", "--id", "9PLM9XGG6VKS"]


async def test_searching_happens_off_the_event_loop() -> None:
    printed = Printed()

    await store(printed).search("ChatGPT")

    assert threading.main_thread() not in printed.threads


async def test_nothing_in_the_store_is_none_and_show_is_not_asked() -> None:
    printed = Printed(search=NOTHING, search_code=20)

    assert await store(printed).search("Webtekno") is None
    assert len(printed.calls) == 1


async def test_a_listing_without_details_is_still_a_listing() -> None:
    """`show` failing loses the publisher and the price, not the app."""
    printed = Printed(show="", show_code=1)

    assert await store(printed).search("ChatGPT") == Listing("ChatGPT", "9PLM9XGG6VKS", "", "")


@pytest.mark.parametrize(
    "failure",
    [subprocess.TimeoutExpired("winget", SEARCH_SECONDS), OSError("winget is not an executable")],
)
async def test_a_winget_that_does_not_answer_is_no_listing_and_no_error(
    failure: BaseException,
) -> None:
    assert await store(Printed(failure=failure)).search("ChatGPT") is None


async def test_garbage_from_winget_is_no_listing() -> None:
    assert await store(Printed(search="\x00\x01 nonsense", search_code=1)).search("X") is None


# --------------------------------------------------------------------------
# Installing
# --------------------------------------------------------------------------


async def test_installing_starts_winget_silently_and_waits_for_it() -> None:
    started = Started(FakeProcess(lines=["Downloading...", "Successfully installed"]))

    outcome = await store(started=started).install("9PLM9XGG6VKS")

    assert outcome == Install(done=True, ok=True, detail="Successfully installed")
    assert started.calls == [
        [
            r"C:\winget.exe",
            "install",
            "--id",
            "9PLM9XGG6VKS",
            "--source",
            "msstore",
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--disable-interactivity",
        ]
    ]
    assert started.process.waited == [INSTALL_WAIT_SECONDS]
    assert threading.main_thread() not in started.threads


async def test_a_failed_install_says_so_with_winget_s_last_words() -> None:
    started = Started(FakeProcess(lines=["Found it", "Installer failed with exit code: 1"], code=1))

    outcome = await store(started=started).install("9PLM9XGG6VKS")

    assert outcome == Install(done=True, ok=False, detail="Installer failed with exit code: 1")


async def test_a_winget_that_cannot_start_is_a_failed_install() -> None:
    def refuse(command: list[str], **options: Any) -> FakeProcess:
        raise OSError("no winget")

    outcome = await WingetStore(popen=refuse, which=winget).install("9PLM9XGG6VKS")

    assert outcome == Install(done=True, ok=False, detail="no winget")


async def test_an_install_still_running_after_the_wait_is_left_running() -> None:
    """The turn cannot wait a minute for a first word; the download can go
    on without it, and the catalogue is refreshed when it has ended."""
    process = FakeProcess(lines=["Downloading..."], ends=False)
    engine = store(started=Started(process))

    outcome = await engine.install("9PLM9XGG6VKS")

    assert outcome == Install(done=False, ok=False, detail="still downloading")
    assert engine.settled() is False
    process.finish(0)
    assert engine.settled() is True
    assert engine.settled() is False


async def test_an_install_that_ended_in_time_is_not_pending() -> None:
    engine = store()

    await engine.install("9PLM9XGG6VKS")

    assert engine.settled() is False


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------


class FakeStore:
    def __init__(self, outcome: Install) -> None:
        self.outcome = outcome
        self.installed: list[str] = []

    @property
    def available(self) -> bool:
        return True

    async def search(self, words: str) -> Listing | None:
        return None

    async def install(self, store_id: str) -> Install:
        self.installed.append(store_id)
        return self.outcome

    def settled(self) -> bool:
        return False


class Opened:
    def __init__(self) -> None:
        self.targets: list[str] = []


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Opened:
    seen = Opened()
    monkeypatch.setattr(shell, "launch", seen.targets.append)
    return seen


def catalogue_that_learns() -> AppCatalog:
    """Empty as read at startup; ChatGPT on the next scan - what an install does."""
    return AppCatalog(
        [],
        scanners=[lambda: [AppEntry("ChatGPT", "shell:AppsFolder" + chr(92) + "OpenAI.Codex!App")]],
    )


def test_install_app_asks_first_with_the_name_and_the_publisher() -> None:
    """A `confirm` tool, and its question names both, so a look-alike from
    another publisher is heard as one."""
    installer = install_app_for(AppCatalog([]), FakeStore(Install(True, True, "")))

    assert installer.risk == "confirm"
    assert installer.spec.name == "install_app"
    assert installer.confirm_prompt == TEXT["store_install_confirm"]
    assert installer.spec.parameters["required"] == ["name", "store_id", "publisher"]
    assert "{name}" in TEXT["store_install_confirm"]
    assert "{publisher}" in TEXT["store_install_confirm"]


def test_the_question_can_be_the_pack_s() -> None:
    installer = install_app_for(
        AppCatalog([]),
        FakeStore(Install(True, True, "")),
        confirm_prompt="{name} ({publisher}) inecek.",
    )

    assert installer.confirm_prompt == "{name} ({publisher}) inecek."


async def test_an_id_the_model_made_up_is_refused_without_winget(opened: Opened) -> None:
    engine = FakeStore(Install(True, True, ""))
    installer = install_app_for(AppCatalog([]), engine)

    said = await installer.run(name="ChatGPT", store_id="chatgpt-app", publisher="OpenAI")

    assert engine.installed == []
    assert said.startswith("That is not a Store id")
    assert opened.targets == []


async def test_a_finished_install_refreshes_the_catalogue_and_opens_the_app(
    opened: Opened,
) -> None:
    engine = FakeStore(Install(done=True, ok=True, detail="Successfully installed"))
    installer = install_app_for(catalogue_that_learns(), engine)

    said = await installer.run(name="ChatGPT", store_id="9plm9xgg6vks", publisher="OpenAI")

    assert engine.installed == ["9PLM9XGG6VKS"]
    assert opened.targets == ["shell:AppsFolder\\OpenAI.Codex!App"]
    assert said == "Installed and opened 'ChatGPT'."


async def test_an_installed_app_the_start_menu_does_not_list_yet_is_said_so(
    opened: Opened,
) -> None:
    installer = install_app_for(AppCatalog([]), FakeStore(Install(True, True, "")))

    said = await installer.run(name="ChatGPT", store_id="9PLM9XGG6VKS", publisher="OpenAI")

    assert opened.targets == []
    assert said == (
        "Installed 'ChatGPT', but it is not in the Start menu yet; ask the user to open "
        "it themselves, or try open_app again in a moment."
    )


async def test_a_failed_install_is_reported_with_the_reason(opened: Opened) -> None:
    installer = install_app_for(
        catalogue_that_learns(),
        FakeStore(Install(True, False, "Installer failed with exit code: 1")),
    )

    said = await installer.run(name="ChatGPT", store_id="9PLM9XGG6VKS", publisher="OpenAI")

    assert opened.targets == []
    assert said == (
        "The Store install of 'ChatGPT' failed: Installer failed with exit code: 1. "
        "Tell the user; they can install it from the Store themselves."
    )


async def test_a_download_still_running_is_left_to_finish(opened: Opened) -> None:
    installer = install_app_for(
        catalogue_that_learns(), FakeStore(Install(False, False, "still downloading"))
    )

    said = await installer.run(name="ChatGPT", store_id="9PLM9XGG6VKS", publisher="OpenAI")

    assert opened.targets == []
    assert said == (
        "'ChatGPT' is still downloading; it continues in the background. Tell the user it "
        "will be ready in a minute or two and that they can ask to open it then."
    )


def test_the_tool_is_a_tool() -> None:
    assert isinstance(install_app_for(AppCatalog([]), FakeStore(Install(True, True, ""))), Tool)
