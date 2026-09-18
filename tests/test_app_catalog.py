"""The app catalogue: what the user said, matched to what the machine has (2.2).

Two claims. The catalogue finds an app by whatever the user called it - the
name, one word of it, the start of it, or a misspelling - and it does so the
same for "IŞIK" and "isik", which is section 3.7's lesson applied to app
names rather than notes. And the machine is read without a test ever
starting PowerShell: the shell is a parameter, and what it prints is a
fixture, including everything that can go wrong with it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from assistant.tools import system
from assistant.tools.system import (
    LISTING_COMMAND,
    PROMPT_NAMES,
    AppCatalog,
    AppEntry,
    scan_start_apps,
    scan_start_menu,
    spoken_form,
    start_menu_folders,
)

CALCULATOR = "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"

INSTALLED = [
    AppEntry("Spotify", r"C:\Start\Spotify.lnk"),
    AppEntry("Google Chrome", r"C:\Start\Google Chrome.lnk"),
    AppEntry("Hesap Makinesi", "shell:AppsFolder\\" + CALCULATOR),
    AppEntry("Visual Studio Code", r"C:\Start\Visual Studio Code.lnk"),
    AppEntry("IŞIK", r"C:\Start\IŞIK.lnk"),
    AppEntry("Notepad", r"C:\Start\Notepad.lnk"),
    # The store twin of the first entry: the same app, listed a second time.
    AppEntry("Spotify", "shell:AppsFolder\\SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify"),
]


@pytest.fixture
def catalog() -> AppCatalog:
    return AppCatalog(INSTALLED)


def name_of(found: AppEntry | None) -> str | None:
    return found.name if found else None


# --------------------------------------------------------------------------
# Finding an app
# --------------------------------------------------------------------------


def test_the_exact_name_is_found_whatever_its_case(catalog: AppCatalog) -> None:
    assert catalog.find("SPOTIFY") == INSTALLED[0]


@pytest.mark.parametrize("spoken", ["IŞIK", "isik", "ışık", "Işık", "ISIK"])
def test_every_spelling_of_isik_finds_the_same_app(catalog: AppCatalog, spoken: str) -> None:
    """Section 3.7: `I`/`ı`/`İ`/`i` and the accents fold to one key, on both
    sides - the catalogue's name and the recogniser's spelling."""
    assert catalog.find(spoken) == INSTALLED[4]


def test_the_turkish_name_of_the_calculator_is_found_as_the_user_says_it(
    catalog: AppCatalog,
) -> None:
    """On a Turkish Windows the shortcut *is* called this; no code names it."""
    assert name_of(catalog.find("hesap makinesi")) == "Hesap Makinesi"
    assert name_of(catalog.find("HESAP MAKİNESİ")) == "Hesap Makinesi"


def test_one_word_of_the_name_is_enough(catalog: AppCatalog) -> None:
    assert name_of(catalog.find("chrome")) == "Google Chrome"
    assert name_of(catalog.find("code")) == "Visual Studio Code"


def test_the_start_of_a_name_is_enough(catalog: AppCatalog) -> None:
    assert name_of(catalog.find("spot")) == "Spotify"
    assert name_of(catalog.find("not")) == "Notepad"


def test_a_start_that_short_would_match_half_the_catalogue(catalog: AppCatalog) -> None:
    assert catalog.find("vs") is None


@pytest.mark.parametrize(
    ("spoken", "name"),
    [("vscode", "Visual Studio Code"), ("notpad", "Notepad"), ("spotifay", "Spotify")],
)
def test_a_small_misspelling_is_forgiven(catalog: AppCatalog, spoken: str, name: str) -> None:
    assert name_of(catalog.find(spoken)) == name


def test_a_name_that_is_merely_a_bit_like_one_is_not_opened(catalog: AppCatalog) -> None:
    """ "krom" is what Whisper writes for "Chrome" said in Turkish - and it is
    as much like "prompt". Measured at 0.6 it opened the Command Prompt;
    now it is a suggestion, and the model asks or tries "Chrome"."""
    assert catalog.find("krom") is None
    assert "Google Chrome" in catalog.closest("krom")


def test_an_app_that_is_not_installed_does_not_open_one_that_sounds_alike() -> None:
    """Measured on the owner's machine, which has no Spotify: at 0.6,
    "spotify" opened Sticky Notes."""
    catalog = AppCatalog([AppEntry("Sticky Notes (new)", r"C:\Start\Sticky Notes.lnk")])

    assert catalog.find("spotify") is None


def test_two_apps_equally_close_are_a_question_and_not_a_coin_toss() -> None:
    catalog = AppCatalog([AppEntry("Zoom", r"C:\Zoom.lnk"), AppEntry("Zoot", r"C:\Zoot.lnk")])

    assert catalog.find("zoox") is None
    assert catalog.closest("zoox") == ["Zoom", "Zoot"]


def test_something_that_is_not_there_is_not_found(catalog: AppCatalog) -> None:
    assert catalog.find("photoshop") is None
    assert catalog.find("zzzz") is None


def test_nothing_said_finds_nothing(catalog: AppCatalog) -> None:
    assert catalog.find("") is None
    assert catalog.find("   ") is None


def test_the_same_name_twice_is_one_app_and_the_first_wins(catalog: AppCatalog) -> None:
    """A shortcut and its store twin: one app, opened the first way it was
    listed - the shortcut, which the scans put first."""
    assert len(catalog) == 6
    assert catalog.names() == [
        "Spotify",
        "Google Chrome",
        "Hesap Makinesi",
        "Visual Studio Code",
        "IŞIK",
        "Notepad",
    ]
    assert catalog.find("spotify") == INSTALLED[0]


# --------------------------------------------------------------------------
# Suggesting, and the vocabulary
# --------------------------------------------------------------------------


def test_when_nothing_matches_the_nearest_names_are_offered(catalog: AppCatalog) -> None:
    """Nearly right is not right: "spotter" opens nothing, but the model can
    ask whether Spotify was meant."""
    assert catalog.find("spotter") is None
    assert catalog.closest("spotter") == ["Spotify", "Notepad"]


def test_the_suggestions_are_capped_and_never_name_one_app_twice(catalog: AppCatalog) -> None:
    """A name and its words are several keys for one app; the app is offered
    once, by its best key, and the exact one comes first."""
    near = catalog.closest("isik")

    assert near[0] == "IŞIK"
    assert len(near) == len(set(near)) <= 3
    assert catalog.closest("isik", limit=1) == ["IŞIK"]


def test_nothing_near_means_no_suggestion(catalog: AppCatalog) -> None:
    assert catalog.closest("zzzz") == []


@pytest.mark.parametrize(
    ("listed", "spoken"),
    [
        ("PyCharm 2026.2.1", "PyCharm"),
        ("Python 3.13 (64-bit)", "Python"),
        ("Outlook (classic)", "Outlook"),
        ("IDLE (Python 3.13 64-bit)", "IDLE"),
        ("Word 2016", "Word"),
        ("Paint 3D", "Paint 3D"),
        ("7-Zip", "7-Zip"),
        ("Google Chrome", "Google Chrome"),
        ("(x)", "(x)"),
    ],
)
def test_the_spoken_form_drops_what_nobody_says(listed: str, spoken: str) -> None:
    """A version, a year, a parenthesised tail: on the shortcut, never in the
    mouth. A number that is part of the name stays, and a name that would
    vanish entirely is kept as it is."""
    assert spoken_form(listed) == spoken
    assert AppEntry(listed, r"C:\x.lnk").spoken == spoken


def test_the_spoken_names_put_products_before_system_shortcuts() -> None:
    """Capitalised names first - `dfrgui`, `services`, `computer` are what
    the machine calls its utilities - then the shorter first, each spoken
    form once, and no more than asked for."""
    catalog = AppCatalog(
        [
            AppEntry("dfrgui", r"C:\dfrgui.lnk"),
            AppEntry("Outlook (classic)", r"C:\Outlook (classic).lnk"),
            AppEntry("PyCharm 2026.2.1", r"C:\PyCharm.lnk"),
            AppEntry("Outlook", r"C:\Outlook.lnk"),
            AppEntry("Run", r"C:\Run.lnk"),
            AppEntry("FortiClient VPN", r"C:\FortiClient VPN.lnk"),
        ]
    )

    assert catalog.spoken_names() == ["Run", "Outlook", "PyCharm", "FortiClient VPN", "dfrgui"]
    assert catalog.spoken_names(limit=2) == ["Run", "Outlook"]


def test_the_names_the_user_has_asked_for_come_first_as_the_catalogue_knows_them() -> None:
    """The audit log's names are what the user said - "pay charm", "Teams" -
    and they are resolved to the apps they opened; what resolves to nothing
    is skipped, and nothing is offered twice."""
    catalog = AppCatalog(
        [
            AppEntry("Run", r"C:\\Run.lnk"),
            AppEntry("PyCharm 2026.2.1", r"C:\\PyCharm.lnk"),
            AppEntry("Microsoft Teams", r"C:\\Teams.lnk"),
            AppEntry("Google Chrome", r"C:\\Chrome.lnk"),
        ]
    )

    offered = catalog.spoken_names(first=["Teams", "zzzz", "pay charm", "teams"])

    assert offered == ["Microsoft Teams", "PyCharm", "Run", "Google Chrome"]
    assert catalog.spoken_names(limit=1, first=["chrome"]) == ["Google Chrome"]


def test_the_offer_is_larger_than_before_and_the_recogniser_fits_it() -> None:
    """The count is an offer; what fits is measured in tokens where the
    tokenizer is (`stt/local_whisper.py`)."""
    many = AppCatalog([AppEntry(f"App {n}", f"C:\\{n}.lnk") for n in range(PROMPT_NAMES * 3)])

    assert PROMPT_NAMES == 120
    assert len(many.spoken_names()) == PROMPT_NAMES


# --------------------------------------------------------------------------
# What the user said, several words of it (2026-09-13)
# --------------------------------------------------------------------------


def test_two_words_are_not_claimed_by_one_generic_word_of_another_app() -> None:
    """Measured 2026-09-13: "Text Editor" opened the Registry Editor because
    the word "editor" alone is seven tenths of the phrase. A phrase is
    matched against whole names; the model gets the near ones to ask about."""
    catalog = AppCatalog(
        [
            AppEntry("Registry Editor", r"C:\regedit.lnk"),
            AppEntry("Virtual Network Editor", r"C:\vmnet.lnk"),
            AppEntry("Notepad", r"C:\Notepad.lnk"),
        ]
    )

    assert catalog.find("Text Editor") is None
    assert catalog.closest("Text Editor")[0] == "Registry Editor"


@pytest.mark.parametrize(
    ("spoken", "name"),
    [
        ("Pay Charm", "PyCharm 2026.2.1"),
        ("Pay Charmage", "PyCharm 2026.2.1"),
        ("Porti Client", "FortiClient VPN"),
        ("Porti Client VPN", "FortiClient VPN"),
        ("PyCharm 2026.2.1", "PyCharm 2026.2.1"),
        ("pycharm", "PyCharm 2026.2.1"),
        ("Tamsi", "Microsoft Teams"),
    ],
)
def test_a_misheard_product_name_still_reaches_the_product(spoken: str, name: str) -> None:
    """What the recogniser makes of a foreign name (2026-09-13's transcripts)
    is matched against the name as people say it - no version - and the
    long listed name is still found when it is said in full."""
    catalog = AppCatalog(
        [
            AppEntry("PyCharm 2026.2.1", r"C:\PyCharm.lnk"),
            AppEntry("FortiClient VPN", r"C:\FortiClient.lnk"),
            AppEntry("Microsoft Teams", r"C:\Teams.lnk"),
            AppEntry("Registry Editor", r"C:\regedit.lnk"),
        ]
    )

    assert name_of(catalog.find(spoken)) == name


def test_an_exact_listed_name_beats_the_spoken_alias_of_another() -> None:
    """ "Outlook (classic)" is said "Outlook" too, and it is listed first; the
    app actually called Outlook still wins its own name."""
    catalog = AppCatalog(
        [
            AppEntry("Outlook (classic)", r"C:\Outlook (classic).lnk"),
            AppEntry("Outlook", r"C:\Outlook.lnk"),
        ]
    )

    assert len(catalog) == 2
    assert name_of(catalog.find("outlook")) == "Outlook"
    assert name_of(catalog.find("outlook classic")) == "Outlook (classic)"


# --------------------------------------------------------------------------
# The Start Menu folders
# --------------------------------------------------------------------------


def test_shortcuts_under_the_start_menu_folders_are_read_by_file_name(tmp_path: Path) -> None:
    mine = tmp_path / "mine"
    theirs = tmp_path / "theirs"
    (mine / "Spotify").mkdir(parents=True)
    (mine / "Spotify" / "Spotify.lnk").write_bytes(b"")
    (mine / "Readme.txt").write_bytes(b"")
    (theirs / "Accessories").mkdir(parents=True)
    (theirs / "Accessories" / "Paint.lnk").write_bytes(b"")
    (theirs / "Notepad.lnk").write_bytes(b"")

    found = scan_start_menu([mine, theirs, tmp_path / "missing"])

    assert found == [
        AppEntry("Spotify", str(mine / "Spotify" / "Spotify.lnk")),
        AppEntry("Paint", str(theirs / "Accessories" / "Paint.lnk")),
        AppEntry("Notepad", str(theirs / "Notepad.lnk")),
    ]


def test_the_folders_are_the_users_and_the_machines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPDATA", r"C:\Users\me\AppData\Roaming")
    monkeypatch.setenv("PROGRAMDATA", r"C:\ProgramData")

    assert start_menu_folders() == [
        Path(r"C:\Users\me\AppData\Roaming\Microsoft\Windows\Start Menu\Programs"),
        Path(r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs"),
    ]


def test_a_folder_the_environment_does_not_name_is_left_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APPDATA", raising=False)
    monkeypatch.setenv("PROGRAMDATA", r"C:\ProgramData")

    assert start_menu_folders() == [Path(r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs")]


# --------------------------------------------------------------------------
# The shell's listing
# --------------------------------------------------------------------------

LISTED: list[Any] = [
    {"Name": "Calculator", "AppID": CALCULATOR},
    {"Name": " Google Chrome ", "AppID": "Chrome"},
    {"Name": "", "AppID": "Nameless"},
    {"Name": "No id", "AppID": ""},
    {"Name": "Wrong shape", "AppID": 3},
    "not an object",
]


class Shell:
    """A PowerShell that prints what it is told to, and remembers how it was asked."""

    def __init__(
        self,
        *,
        stdout: str = "",
        returncode: int = 0,
        stderr: str = "",
        raises: Exception | None = None,
    ) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.raises = raises
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, command: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append((command, options))
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, self.stderr)


@pytest.fixture
def powershell(monkeypatch: pytest.MonkeyPatch) -> str:
    """A machine that has PowerShell, at a path the test can recognise."""
    monkeypatch.setattr(shutil, "which", lambda name: r"C:\Windows\powershell.exe")
    return r"C:\Windows\powershell.exe"


def test_the_listing_becomes_entries_that_open_through_the_apps_folder(powershell: str) -> None:
    """Names are trimmed; an item without both a name and an id is skipped."""
    found = scan_start_apps(run=Shell(stdout=json.dumps(LISTED)))

    assert found == [
        AppEntry("Calculator", "shell:AppsFolder\\" + CALCULATOR),
        AppEntry("Google Chrome", "shell:AppsFolder\\Chrome"),
    ]


def test_one_app_is_printed_as_an_object_rather_than_a_list(powershell: str) -> None:
    found = scan_start_apps(run=Shell(stdout=json.dumps(LISTED[0])))

    assert found == [AppEntry("Calculator", "shell:AppsFolder\\" + CALCULATOR)]


def test_the_shell_is_asked_without_a_profile_and_answers_in_utf8(powershell: str) -> None:
    """Through a pipe PowerShell writes in the legacy code page, which has no
    `ğ`; the command sets the encoding before it prints."""
    shell = Shell(stdout="[]")

    scan_start_apps(run=shell)

    [(command, options)] = shell.calls
    assert command[0] == powershell
    assert "-NoProfile" in command
    assert "-NonInteractive" in command
    assert command[-1] == LISTING_COMMAND
    assert options["encoding"] == "utf-8"
    assert options["capture_output"] is True
    assert options["timeout"] == system.LISTING_SECONDS


@pytest.mark.parametrize(
    "shell",
    [
        Shell(returncode=1, stderr="Get-StartApps : not recognized"),
        Shell(stdout="not json at all"),
        Shell(stdout=""),
        Shell(raises=OSError("no such file")),
        Shell(raises=subprocess.TimeoutExpired("powershell", 30.0)),
    ],
    ids=["failed", "not json", "printed nothing", "would not start", "took too long"],
)
def test_a_shell_that_fails_leaves_the_catalogue_with_the_shortcuts_alone(
    powershell: str, shell: Shell
) -> None:
    assert scan_start_apps(run=shell) == []


def test_a_machine_without_powershell_is_not_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)
    shell = Shell(stdout="[]")

    assert scan_start_apps(run=shell) == []
    assert shell.calls == []


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


async def test_loading_runs_every_scanner_in_order_and_off_the_event_loop() -> None:
    """Hundreds of files and a PowerShell process are seconds; the 50 ms rule
    (section 3.1 rule 4) puts them on a thread."""
    threads: list[threading.Thread] = []

    def shortcuts() -> list[AppEntry]:
        threads.append(threading.current_thread())
        return [INSTALLED[0]]

    def store() -> list[AppEntry]:
        threads.append(threading.current_thread())
        return [INSTALLED[2]]

    catalog = await AppCatalog.load(scanners=[shortcuts, store])

    assert catalog.names() == ["Spotify", "Hesap Makinesi"]
    assert len(threads) == 2
    assert threading.main_thread() not in threads


async def test_refreshing_reads_the_machine_again_off_the_event_loop() -> None:
    """An install ended (`tools/store.py`): the same scanners run again and
    the catalogue the tools hold sees the new app."""
    scans = 0
    threads: list[threading.Thread] = []

    def scanner() -> list[AppEntry]:
        nonlocal scans
        scans += 1
        threads.append(threading.current_thread())
        return INSTALLED[:1] if scans == 1 else INSTALLED[:2]

    catalog = await AppCatalog.load(scanners=[scanner])
    assert catalog.find("chrome") is None

    await catalog.refresh()

    assert (len(catalog), scans) == (2, 2)
    assert name_of(catalog.find("chrome")) == "Google Chrome"
    assert threading.main_thread() not in threads


async def test_a_catalogue_built_by_hand_has_nothing_to_refresh_from() -> None:
    catalog = AppCatalog(INSTALLED)

    await catalog.refresh()

    assert len(catalog) == 6
