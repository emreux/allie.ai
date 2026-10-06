"""`RunningApps` (D41): which open windows a spoken name means, and that
Allie never finds itself. Over a fake desktop."""

from __future__ import annotations

from allie.computer.windows import AppWindow, RunningApps
from allie.tools.system import AppEntry

CHROME = AppWindow(handle=11, title="Yeni sekme - Google Chrome", image="chrome.exe", pid=100)
CHROME_2 = AppWindow(handle=12, title="GitHub - Google Chrome", image="chrome.exe", pid=100)
NOTEPAD = AppWindow(handle=21, title="notlar.txt - Not Defteri", image="notepad.exe", pid=200)
CALC = AppWindow(handle=31, title="Hesap Makinesi", image="ApplicationFrameHost.exe", pid=300)
OURS = AppWindow(handle=41, title="Allie", image="python.exe", pid=999)


class FakeDesktop:
    def __init__(self, *windows: AppWindow) -> None:
        self.open = list(windows)
        self.fronted: list[int] = []
        self.closed: list[int] = []
        self.minimised = 0

    def windows(self) -> list[AppWindow]:
        return list(self.open)

    def bring_to_front(self, handle: int) -> bool:
        self.fronted.append(handle)
        return True

    def close(self, handle: int) -> None:
        self.closed.append(handle)

    def alive(self, handle: int) -> bool:
        return handle not in self.closed

    def minimize_all(self) -> None:
        self.minimised += 1


def running(*windows: AppWindow) -> RunningApps:
    return RunningApps(FakeDesktop(*windows), own_pid=999)


def test_an_image_name_finds_every_window_of_the_app() -> None:
    found = running(CHROME, NOTEPAD, CHROME_2).find("chrome")

    assert found == [CHROME, CHROME_2]


def test_the_catalogue_name_finds_the_app_by_its_title() -> None:
    entry = AppEntry(name="Google Chrome", launch="chrome.lnk")

    assert running(NOTEPAD, CHROME).find("krom", entry) == [CHROME]


def test_a_store_app_is_found_by_its_title_alone() -> None:
    entry = AppEntry(name="Hesap Makinesi", launch="shell:AppsFolder\\calc")

    assert running(CALC, NOTEPAD).find("hesap makinesi", entry) == [CALC]


def test_allie_never_finds_itself() -> None:
    assert running(OURS).find("allie") == []
    assert running(OURS).find("python") == []


def test_a_name_that_is_open_nowhere_finds_nothing() -> None:
    assert running(CHROME, NOTEPAD).find("spotify") == []


def test_a_name_is_a_whole_word_of_the_title() -> None:
    """ "Mail" is not the Gmail tab of a browser: closing it would close Chrome."""
    gmail = AppWindow(
        handle=51, title="Gelen Kutusu - Gmail - Google Chrome", image="chrome.exe", pid=100
    )
    entry = AppEntry(name="Mail", launch="shell:AppsFolder\\mail")

    assert running(gmail).find("mail", entry) == []


def test_a_short_word_does_not_match_inside_a_title() -> None:
    """Two letters are in every other title."""
    assert running(CHROME).find("go") == []


def test_closing_asks_each_window_and_reports_the_ones_left() -> None:
    desktop = FakeDesktop(CHROME, CHROME_2)
    apps = RunningApps(desktop, own_pid=999)

    apps.close([CHROME, CHROME_2])
    desktop.closed.remove(12)  # this one asked "save?" and stayed

    assert apps.alive([CHROME, CHROME_2]) == [CHROME_2]
