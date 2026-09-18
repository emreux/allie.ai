"""What the command line does with each command.

`setup` is wired to the wizard here rather than driven through it - the wizard
has its own suite, and a test that opened a real prompt would hang. `run` is
the same idea one level up: the state machine, the speech model and the model
that answers all have their own suites, so what is tested here is that they are
handed to each other correctly and that starting up fails in words.

**Nothing may touch the hardware.** A test that loaded Whisper would take two
seconds and a gigabyte, and one that opened the microphone would record the
room. Both are replaced; everything else is the code that ships.

**A machine that is not set up is not a crash.** No settings, or a key that has
been removed from the Credential Manager, are things the user can fix - so they
are a sentence and an exit code, and neither of them costs a model load.
"""

from __future__ import annotations

import sqlite3
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar

import pytest
from loguru import logger

from assistant import __main__ as cli
from assistant import locales, logs, setup_wizard
from assistant.__main__ import BUILTIN_TOOLS, TEXT, build_parser, main, use_utf8
from assistant.config import (
    KEYRING_SERVICE,
    AudioSettings,
    LimitSettings,
    LiveSettings,
    LocaleSettings,
    Settings,
    STTSettings,
    config_path,
    load_settings,
    save_settings,
    store_api_key,
)
from assistant.live.base import Usage
from assistant.live.probe import ProbeResult, remember, remembered
from assistant.messaging.contacts import CONTACTS_FILE_NAME
from assistant.store import db
from assistant.store.memory import MEMORY_FILE_NAME
from assistant.store.repos import SettingsRepo, UsageRepo
from assistant.ui import status
from tests.conftest import MemoryKeyring

MODEL = "gemini-3.8-live"
# What setup wrote down about the model, unless a test says otherwise.
PASSED = ProbeResult(ok=True, first_token_ms=12.0)


@pytest.fixture(autouse=True)
def own_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """The log goes into this test's directory, not the developer's machine."""
    monkeypatch.setattr(logs, "log_path", lambda: tmp_path / "logs" / "assistant.log")
    yield
    logger.remove()


@pytest.fixture
def configured(config_home: Path, vault: MemoryKeyring) -> Path:
    """A machine `live-assistant setup` has already been run on."""
    save_settings(
        Settings(
            live=LiveSettings(primary=f"gemini:{MODEL}"),
            locale=LocaleSettings(code="tr"),
        )
    )
    store_api_key("gemini", "AIza-not-a-real-key")
    return config_home


@pytest.fixture
def own_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The database under this test's directory, never the machine's."""
    path = tmp_path / "data" / "assistant.db"
    monkeypatch.setattr(db, "database_path", lambda: path)
    return path


def said(key: str, code: str = "tr") -> str:
    """A sentence as the pack for `code` has it - whatever it was reworded to."""
    return locales.load(code).say(key, {**TEXT, **status.TEXT, **cli.DOCTOR_TEXT}[key])


# --------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------


def test_help_exits_cleanly(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "assistant" in capsys.readouterr().out


def test_no_command_prints_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["setup", "run", "cost", "mic"])
def test_the_command_names_are_declared(command: str) -> None:
    assert build_parser().parse_args([command]).command == command


def test_run_can_be_asked_for_the_tray() -> None:
    """`run --tray` (4.3); without the flag there is only the terminal."""
    assert build_parser().parse_args(["run", "--tray"]).tray is True
    assert build_parser().parse_args(["run"]).tray is False


def test_telegram_login_is_a_command_of_its_own() -> None:
    """`live-assistant telegram login` (2026-09-15); a bare `telegram` is a usage error."""
    parsed = build_parser().parse_args(["telegram", "login"])

    assert (parsed.command, parsed.telegram_command) == ("telegram", "login")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["telegram"])


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------


def test_setup_runs_the_wizard(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    async def wizard(prompter: object, **kwargs: object) -> int:
        calls.append(prompter)
        return 0

    monkeypatch.setattr(setup_wizard, "run_setup", wizard)

    assert main(["setup"]) == 0
    assert isinstance(calls[0], setup_wizard.TerminalPrompter)


def test_the_exit_code_of_the_wizard_is_the_exit_code_of_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled setup must not look like a successful one to a script."""

    async def wizard(prompter: object, **kwargs: object) -> int:
        return 1

    monkeypatch.setattr(setup_wizard, "run_setup", wizard)

    assert main(["setup"]) == 1


# --------------------------------------------------------------------------
# mic
# --------------------------------------------------------------------------


def test_mic_asks_the_microphone_question_on_its_own(monkeypatch: pytest.MonkeyPatch) -> None:
    """The day a headset comes out: one command, the list, and the file is
    updated - no text editor."""
    calls: list[object] = []

    async def wizard(prompter: object, **kwargs: object) -> int:
        calls.append(prompter)
        return 0

    monkeypatch.setattr(setup_wizard, "run_microphone_setup", wizard)

    assert main(["mic"]) == 0
    assert isinstance(calls[0], setup_wizard.TerminalPrompter)


def test_the_exit_code_of_mic_is_the_exit_code_of_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def wizard(prompter: object, **kwargs: object) -> int:
        return 1

    monkeypatch.setattr(setup_wizard, "run_microphone_setup", wizard)

    assert main(["mic"]) == 1


# --------------------------------------------------------------------------
# run: not before setup, and not yet (plan.md L0)
# --------------------------------------------------------------------------


def test_a_machine_that_was_never_set_up_is_told_to_run_setup(
    config_home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run"]) == 1
    assert said("not_set_up", locales.system_code()) in capsys.readouterr().out


def test_until_the_live_loop_is_wired_run_says_so_and_exits_2(
    configured: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The baseline build (plan.md L0) has the tools, the gate, the store
    and every command but the one that listens. A script can tell "not
    built yet" (2) from "not set up" (1); the sentence is the pack's."""
    assert main(["run", "--tray"]) == 2

    assert said("run_not_wired") in unwrapped(capsys.readouterr().out)


# --------------------------------------------------------------------------
# Shared by the doctor tests below
# --------------------------------------------------------------------------


def unwrapped(printed: str) -> str:
    """What was printed, with the line breaks the eighty-column terminal
    put into a long sentence taken out again."""
    return " ".join(printed.split())


def stored_verdict() -> ProbeResult | None:
    connection = sqlite3.connect(db.database_path())
    connection.row_factory = sqlite3.Row
    try:
        return remembered(SettingsRepo(connection), "gemini", MODEL)
    finally:
        connection.close()


def write_verdict(result: ProbeResult, *, age: float) -> None:
    """A verdict `age` seconds old, as setup would have left it."""
    connection = db.open_database()
    try:
        remember(SettingsRepo(connection), "gemini", MODEL, result, now=time.time() - age)
    finally:
        connection.close()


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def spent(*, cost: float | None) -> None:
    """One turn of 300 in and 10 out on the configured model, on the books."""
    connection = db.open_database()
    try:
        UsageRepo(connection).insert(
            turn_id="t1", provider="gemini", model=MODEL, usage=Usage(300, 10), cost_usd=cost
        )
    finally:
        connection.close()


def test_cost_on_a_machine_that_has_spent_nothing_says_so(
    config_home: Path, own_database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["cost"]) == 0
    assert TEXT["cost_none"] in capsys.readouterr().out


def test_cost_shows_today_s_spending_by_model(
    configured: Path, own_database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spent(cost=0.0004)

    assert main(["cost"]) == 0

    printed = capsys.readouterr().out
    assert said("cost_today") in printed
    assert f"gemini:{MODEL}" in printed
    assert "$0.0004" in printed


def test_a_turn_with_no_price_is_counted_and_reported_as_unpriced(
    configured: Path, own_database: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Never a zero that reads as free (section 6)."""
    spent(cost=None)

    main(["cost"])

    printed = capsys.readouterr().out
    assert "pricing.toml" in printed
    assert "$0.0000" not in printed


# --------------------------------------------------------------------------
# The terminal itself
# --------------------------------------------------------------------------


def test_the_terminal_is_told_to_speak_utf8() -> None:
    """Windows hands a redirected stream the machine's legacy code page, and
    half the Turkish alphabet has no place in it - which ends the program with
    a `UnicodeEncodeError` in the middle of an answer."""

    class Stream:
        def __init__(self) -> None:
            self.asked: dict[str, str] = {}

        def reconfigure(self, **asked: str) -> None:
            self.asked = asked

    stream = Stream()
    use_utf8(stream)

    assert stream.asked["encoding"] == "utf-8"


def test_the_terminal_is_told_before_anything_is_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both streams, and before the first command runs - the wizard prints
    Turkish, and so does everything `run` shows."""

    class Recording(StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.asked: dict[str, str] = {}

        def reconfigure(self, **asked: str) -> None:
            self.asked = asked

    out, errors = Recording(), Recording()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", errors)

    main([])

    assert out.asked["encoding"] == "utf-8"
    assert errors.asked["encoding"] == "utf-8"


def test_a_stream_that_cannot_be_told_is_left_alone() -> None:
    """Something else is holding stdout - a test harness, a pipe of somebody
    else's making. Not being able to ask is not a reason to refuse to start."""
    use_utf8(object())


# --------------------------------------------------------------------------
# telegram login (2026-09-15)
# --------------------------------------------------------------------------


class ScriptedTerminal:
    """Stands in for `TerminalPrompter`: scripted answers, recorded sayings."""

    answers: ClassVar[dict[str, str | None]] = {}
    said: ClassVar[list[tuple[str, dict[str, object]]]] = []
    built_text: ClassVar[dict[str, str]] = {}

    def __init__(self, *, text: Mapping[str, str] | None = None) -> None:
        self.text = dict(text or {})
        ScriptedTerminal.built_text = self.text

    def say(self, key: str, **fields: object) -> None:
        ScriptedTerminal.said.append((key, fields))
        assert key in self.text, key

    async def choose(self, key: str, options: object) -> str | None:
        raise AssertionError("the login never offers a menu")

    async def secret(self, key: str) -> str | None:
        return ScriptedTerminal.answers.get(key)

    async def ask(self, key: str) -> str | None:
        return ScriptedTerminal.answers.get(key)


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> type[ScriptedTerminal]:
    ScriptedTerminal.answers = {}
    ScriptedTerminal.said = []
    ScriptedTerminal.built_text = {}
    monkeypatch.setattr(setup_wizard, "TerminalPrompter", ScriptedTerminal)
    return ScriptedTerminal


class FakeTelegramLogin:
    """What `messaging/telegram.py`'s real client would do, without Telegram."""

    def __init__(self, api_id: int, api_hash: str) -> None:
        self.api_id, self.api_hash = api_id, api_hash

    async def connect(self) -> None:
        pass

    async def send_code(self, phone: str) -> str:
        return "code-hash"

    async def sign_in(self, phone: str, code: str, *, code_hash: str) -> str:
        if code != "12345":
            from assistant.messaging.telegram import LoginError

            raise LoginError("The phone code entered was invalid")
        return "Emre"

    async def sign_in_with_password(self, password: str) -> str:
        return "Emre"

    def session_string(self) -> str:
        return "1BVts-the-session"

    async def disconnect(self) -> None:
        pass


@pytest.fixture
def telegram_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from assistant.messaging import telegram

    monkeypatch.setattr(telegram, "_login_client", FakeTelegramLogin)


def test_telegram_login_stores_the_id_in_the_file_and_the_secrets_in_the_vault(
    configured: Path, vault: MemoryKeyring, terminal: type[ScriptedTerminal], telegram_client: None
) -> None:
    """Section 10: the `api_id` is not a secret, the hash and the session
    are. And the file is rewritten from what was loaded: the microphone the
    user chose survives (the known `setup` finding, not repeated here)."""
    save_settings(
        Settings(
            live=LiveSettings(primary=f"gemini:{MODEL}"),
            locale=LocaleSettings(code="tr"),
            audio=AudioSettings(input_device="Microphone Array 1"),
        )
    )
    terminal.answers = {
        "telegram_api_id": "123456",
        "telegram_api_hash": "abcdef",
        "telegram_phone": "+90 532 000 00 00",
        "telegram_code": "12345",
    }

    assert main(["telegram", "login"]) == 0

    loaded = load_settings()
    assert loaded.telegram.api_id == 123456
    assert loaded.audio.input_device == "Microphone Array 1"
    assert vault.vault[(KEYRING_SERVICE, "telegram")] == "abcdef"
    assert vault.vault[(KEYRING_SERVICE, "telegram-session")] == "1BVts-the-session"
    assert "abcdef" not in config_path().read_text(encoding="utf-8")
    assert terminal.said == [("telegram_logged_in", {"name": "Emre"})]


def test_telegram_login_asks_in_the_language_of_the_pack(
    configured: Path, terminal: type[ScriptedTerminal], telegram_client: None
) -> None:
    from assistant.messaging import telegram

    terminal.answers = {"telegram_api_id": None}

    assert main(["telegram", "login"]) == 1

    # Walking away is said, and the prompter was built with the pack's
    # sentences for every question the login asks.
    assert terminal.said == [("cancelled", {})]
    turkish = locales.load("tr")
    for key, english in telegram.TEXT.items():
        assert terminal.built_text[key] == turkish.say(key, english)
        assert terminal.built_text[key] != english


def test_a_wrong_code_is_telegrams_reason_and_nothing_is_stored(
    configured: Path, vault: MemoryKeyring, terminal: type[ScriptedTerminal], telegram_client: None
) -> None:
    terminal.answers = {
        "telegram_api_id": "1",
        "telegram_api_hash": "h",
        "telegram_phone": "+90...",
        "telegram_code": "0",
    }

    assert main(["telegram", "login"]) == 1

    assert (KEYRING_SERVICE, "telegram-session") not in vault.vault
    assert load_settings().telegram.api_id == 0
    [(key, fields)] = terminal.said
    assert key == "telegram_login_failed"
    assert "phone code entered was invalid" in str(fields["reason"])


# --------------------------------------------------------------------------
# purge --all, and what the audit rows still say (4.6, 17 Sep 2026)
# --------------------------------------------------------------------------


def test_purge_has_to_be_told_all() -> None:
    """The one flag there is, required: deleting everything is said twice,
    once on the command line and once at the question."""
    parsed = build_parser().parse_args(["purge", "--all"])

    assert (parsed.command, parsed.all) == ("purge", True)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["purge"])


@dataclass
class Recorded:
    """What a machine that has run for a while holds, by path."""

    database: Path
    sidecars: list[Path]
    memory: Path
    logs: list[Path]
    contacts: Path

    def files(self) -> list[Path]:
        return [self.database, *self.sidecars, self.memory, *self.logs]


@pytest.fixture
def recorded(configured: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Recorded:
    """A database with the two files SQLite keeps beside it, a memory file,
    the log and one rotation left of it, a contacts file the user wrote, and
    three secrets - the key `configured` stored and Telegram's two."""
    data = tmp_path / "data"
    monkeypatch.setattr(db, "database_path", lambda: data / "assistant.db")
    db.open_database().close()
    sidecars = [data / "assistant.db-wal", data / "assistant.db-shm"]
    for sidecar in sidecars:
        sidecar.write_bytes(b"rows")
    memory = configured / MEMORY_FILE_NAME
    memory.write_text('name = "Cuma"', encoding="utf-8")
    log = logs.log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    rotated = log.with_name("assistant.2026-09-01_00-00-00_000000.log")
    for path in (log, rotated):
        path.write_text("turn: 300 in, 10 out", encoding="utf-8")
    contacts = configured / CONTACTS_FILE_NAME
    contacts.write_text("# the people I message", encoding="utf-8")
    store_api_key("telegram", "abcdef")
    store_api_key("telegram-session", "1BVts-the-session")
    return Recorded(
        database=data / "assistant.db",
        sidecars=sidecars,
        memory=memory,
        logs=[log, rotated],
        contacts=contacts,
    )


def test_purge_lists_what_goes_and_deletes_it_only_after_yes(
    recorded: Recorded, vault: MemoryKeyring, terminal: type[ScriptedTerminal]
) -> None:
    """Section 3.7, 4.6: everything the assistant recorded goes - the rows,
    the WAL that holds rows too, the memory, the logs, the secrets - and
    the two files the user wrote by hand stay. Listed by path and by entry
    name: no value of a key is ever said."""
    terminal.answers = {"purge_confirm": "yes"}

    assert main(["purge", "--all"]) == 0

    assert not any(path.exists() for path in recorded.files())
    assert config_path().is_file()
    assert recorded.contacts.is_file()
    assert vault.vault == {}

    keys = [key for key, _ in terminal.said]
    assert keys[0] == "purge_will_delete"
    assert keys[-2:] == ["purge_keeps", "purge_done"]
    listed = {fields["path"] for key, fields in terminal.said if key == "purge_file"}
    assert listed == set(recorded.files())
    named = [fields["name"] for key, fields in terminal.said if key == "purge_secret"]
    assert named == ["gemini", "telegram", "telegram-session"]
    assert terminal.said[-1] == ("purge_done", {"count": 9})
    spoken = str(terminal.said)
    assert "AIza-not-a-real-key" not in spoken
    assert "abcdef" not in spoken
    assert "1BVts" not in spoken


@pytest.mark.parametrize("answer", ["evet", "y", "", None])
def test_purge_deletes_nothing_without_the_word(
    recorded: Recorded,
    vault: MemoryKeyring,
    terminal: type[ScriptedTerminal],
    answer: str | None,
) -> None:
    """The pack's yes-word is for the voice; the typed word is `yes` and
    the question says so. Anything else, and walking away, is a no."""
    terminal.answers = {"purge_confirm": answer}

    assert main(["purge", "--all"]) == 1

    assert all(path.is_file() for path in recorded.files())
    assert len(vault.vault) == 3
    assert terminal.said[-1] == ("purge_cancelled", {})


def test_purge_takes_the_word_in_any_case(
    recorded: Recorded, vault: MemoryKeyring, terminal: type[ScriptedTerminal]
) -> None:
    terminal.answers = {"purge_confirm": "  YES "}

    assert main(["purge", "--all"]) == 0

    assert vault.vault == {}
    assert not recorded.database.exists()


def test_purge_on_a_machine_with_nothing_recorded_says_so(
    config_home: Path,
    vault: MemoryKeyring,
    terminal: type[ScriptedTerminal],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(db, "database_path", lambda: tmp_path / "data" / "assistant.db")

    assert main(["purge", "--all"]) == 0

    assert terminal.said == [("purge_nothing", {})]


def test_purge_speaks_the_language_of_the_pack(
    recorded: Recorded, terminal: type[ScriptedTerminal]
) -> None:
    terminal.answers = {"purge_confirm": None}

    main(["purge", "--all"])

    turkish = locales.load("tr")
    for key in ("purge_will_delete", "purge_confirm", "purge_cancelled"):
        assert terminal.built_text[key] == turkish.say(key, TEXT[key])
        assert terminal.built_text[key] != TEXT[key]


# --------------------------------------------------------------------------
# autostart on | off | status (4.2, 17 Sep 2026)
# --------------------------------------------------------------------------


class FakeRunKey:
    """The user's `Run` key as a dictionary, shared by every instance the
    command builds."""

    values: ClassVar[dict[str, str]] = {}

    def get(self, name: str) -> str | None:
        return FakeRunKey.values.get(name)

    def set(self, name: str, command: str) -> None:
        FakeRunKey.values[name] = command

    def delete(self, name: str) -> None:
        FakeRunKey.values.pop(name, None)


@pytest.fixture
def run_key(monkeypatch: pytest.MonkeyPatch) -> type[FakeRunKey]:
    from assistant import autostart

    FakeRunKey.values = {}
    monkeypatch.setattr(autostart, "WindowsRegistry", FakeRunKey)
    monkeypatch.setattr(autostart, "command_line", lambda: '"C:\\x\\live-assistant.exe" run --tray')
    return FakeRunKey


def test_autostart_takes_exactly_one_of_three_words() -> None:
    for word in ("on", "off", "status"):
        assert build_parser().parse_args(["autostart", word]).autostart_command == word
    with pytest.raises(SystemExit):
        build_parser().parse_args(["autostart"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["autostart", "maybe"])


def test_autostart_on_writes_the_run_key_and_says_what_it_wrote(
    configured: Path, run_key: type[FakeRunKey], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["autostart", "on"]) == 0

    command = '"C:\\x\\live-assistant.exe" run --tray'
    assert run_key.values == {"live-assistant": command}
    printed = unwrapped(capsys.readouterr().out)
    assert unwrapped(said("autostart_on").format(command=command)) == printed


def test_autostart_off_clears_it_and_status_says_which(
    configured: Path, run_key: type[FakeRunKey], capsys: pytest.CaptureFixture[str]
) -> None:
    main(["autostart", "on"])
    capsys.readouterr()

    assert main(["autostart", "status"]) == 0
    assert '"C:\\x\\live-assistant.exe" run --tray' in capsys.readouterr().out

    assert main(["autostart", "off"]) == 0
    assert run_key.values == {}
    assert unwrapped(capsys.readouterr().out) == unwrapped(said("autostart_off"))

    assert main(["autostart", "status"]) == 0
    assert unwrapped(capsys.readouterr().out) == unwrapped(said("autostart_status_off"))


def test_autostart_speaks_the_machines_language_before_setup(
    config_home: Path, run_key: type[FakeRunKey], capsys: pytest.CaptureFixture[str]
) -> None:
    """No `config.toml` yet: the sentence is the system's language, as
    every other command before setup."""
    assert main(["autostart", "status"]) == 0

    code = locales.system_code()
    assert unwrapped(capsys.readouterr().out) == unwrapped(said("autostart_status_off", code))


def mail_settings(host: str, user: str, *, mailbox: str = "INBOX") -> Any:
    from assistant.config import MailSettings

    return MailSettings(host=host, user=user, mailbox=mailbox)


# --------------------------------------------------------------------------
# doctor (3.5, 17 Sep 2026)
# --------------------------------------------------------------------------


def test_doctor_before_setup_says_so(
    config_home: Path, vault: MemoryKeyring, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["doctor"]) == 1

    assert said("not_set_up", locales.system_code()) in unwrapped(capsys.readouterr().out)


def test_doctor_reports_the_installation_and_never_a_key(
    configured: Path,
    vault: MemoryKeyring,
    run_key: type[FakeRunKey],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One screen: who answers, what leaves, where the files are, the
    tools, the limits, what is set up. The key is reported as stored and
    is not on the screen; the Telegram hash and the mail password are not
    either."""
    monkeypatch.setattr(db, "database_path", lambda: tmp_path / "data" / "assistant.db")
    write_verdict(PASSED, age=3 * 86_400 + 60)
    (configured / MEMORY_FILE_NAME).write_text(
        '[assistant]\nname = ""\n\n[user]\nfacts = ["a", "b", "c"]\n', encoding="utf-8"
    )
    (configured / CONTACTS_FILE_NAME).write_text(
        '[[contact]]\nname = "Ahmet"\nphone = "+90 532 000 00 00"\n', encoding="utf-8"
    )
    folder = configured / "tools"
    folder.mkdir()
    (folder / "mine.py").write_text(
        "from assistant.tools.registry import tool\n\n\n"
        '@tool(risk="safe")\nasync def start_my_project() -> str:\n'
        '    """Starts the owner\'s project."""\n    return "started"\n',
        encoding="utf-8",
    )
    store_api_key("telegram", "the-hash")
    store_api_key("telegram-session", "the-session")
    store_api_key("mail", "the-app-password")
    save_settings(
        Settings(
            live=LiveSettings(primary=f"gemini:{MODEL}"),
            locale=LocaleSettings(code="tr"),
            stt=STTSettings(provider="gemini"),
            limits=LimitSettings(hard_stop=True),
            mail=mail_settings("imap.example.test", "emre@example.test"),
        )
    )
    from assistant.config import TelegramSettings

    loaded = load_settings()
    loaded.telegram = TelegramSettings(api_id=123456)
    save_settings(loaded)
    command = '"C:\\x\\live-assistant.exe" run --tray'
    run_key.values["live-assistant"] = command

    assert main(["doctor"]) == 0

    out = unwrapped(capsys.readouterr().out)

    def line(key: str, **fields: object) -> str:
        return unwrapped(said(key).format(**fields))

    assert line("doctor_model", model=f"gemini:{MODEL}", provider="Google Gemini Live") in out
    assert line("doctor_key_stored") in out
    assert line("doctor_verdict_ok", days=3) in out
    assert line("doctor_stt_gemini", model="gemini-3.5-transcribe-live", size="small") in out
    assert line("doctor_voice_goes") in out
    assert line("doctor_answer_stays") in out
    assert line("doctor_text_goes", provider="Google Gemini Live") in out
    assert line("doctor_memory", path=configured / MEMORY_FILE_NAME, count=3) in out
    assert line("doctor_contacts", path=configured / CONTACTS_FILE_NAME, count=1) in out
    assert line("doctor_builtin", count=len(BUILTIN_TOOLS)) in out
    assert line("doctor_local", count=1, names="start_my_project") in out
    assert line("doctor_spend_stop", daily="2.00", monthly="30.00") in out
    assert line("doctor_retention", days=30) in out
    mail_line = line(
        "doctor_mail", user="emre@example.test", host="imap.example.test", mailbox="INBOX"
    )
    assert mail_line in out
    assert line("doctor_telegram", api_id=123456) in out
    assert line("doctor_autostart_on", command=command) in out
    for secret in ("AIza-not-a-real-key", "the-hash", "the-session", "the-app-password"):
        assert secret not in out


def test_doctor_on_a_bare_setup_says_what_is_not_there(
    configured: Path,
    vault: MemoryKeyring,
    run_key: type[FakeRunKey],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(db, "database_path", lambda: tmp_path / "data" / "assistant.db")

    assert main(["doctor"]) == 0

    out = unwrapped(capsys.readouterr().out)
    for key in (
        "doctor_verdict_none",
        "doctor_voice_stays",
        "doctor_answer_stays",
        "doctor_local_none",
        "doctor_mail_none",
        "doctor_telegram_none",
        "doctor_autostart_off",
    ):
        assert unwrapped(said(key)) in out, key
    assert "(henüz oluşmadı)" in out
    assert "speech model" not in out
