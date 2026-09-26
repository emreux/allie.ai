"""Command line entry point - and the composition root of the program.

`setup` asks the questions of item 1.4 on a machine that has never been set
up - the assistant, the key, the model, who hears the yes or no, the
microphone - and after that opens the settings list, each row changed on
its own (plan.md D32); `mic` asks the microphone again on its own. `run` is item 1.11: it
puts the pieces together, hands them to the state machine, and shows the one
line of terminal that is the entire interface until the tray icon of phase
4.2. `cost` is 2.4: what the turns cost, read back from `usage_log`. `purge --all`
is 4.6: everything the machine recorded, listed and then deleted after the
word `yes`. `doctor` says who answers, what leaves this machine and where the
files are; its wording is the owner's (section 8).

This is the only file that knows the concrete names: which tools are on
offer, which gate runs them, where the audit rows go. `agent/core.py` sees a
registry and a gate, `agent/policy.py` sees a registry and a repository, and
neither knows what the other is called - which is what lets a test hand
either of them a fake.

Three things about `run` are decisions rather than plumbing.

**Everything heavy is imported inside the function that needs it.**
`allie --help` should not load PortAudio, a speech model and a
vendor SDK to print four lines of help - and `run` should not load the
wizard's prompt library it will never show.

**`run` wires the live loop of plan.md section 4.1.** The model speaks for
itself over one session at a time (`app.py`), so what `_talk` builds is the
doorman and the microphone stream (`audio/capture.py`), the tool round
(`agent/core.py`) behind the one gate, the local recogniser and the
assistant's reading voice that serve the gate window and the reminders (D3,
D4, D10, D32), and what a session is
opened with - the prompt with the pack's language rule, the user's facts and
the time, the tools, the pack's language code, the `[live]` knobs - read
afresh at every open, since the facts and the time move.

**Starting up fails in sentences.** A machine nobody has run setup on, a key
that has since been deleted from the Credential Manager, a voice that is not
installed, a speech model that could not be fetched, a microphone that would
not open: these are things the user can fix, and each gets a sentence and an
exit code - the first two before anything slow is loaded. Anything else is a
bug in this project and comes out as a traceback, for the same reason `app.py`
refuses to say "I could not connect" about one.

**The terminal is told to speak UTF-8 first.** A redirected stream gets the
machine's legacy code page from Windows, which has no `ş` and no `ğ` in it -
and the assistant would end a turn with a `UnicodeEncodeError` instead of an
answer. That is fixed here rather than by asking the user to set an environment
variable before starting their own assistant.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from allie import __version__, locales
from allie.config import (
    DocumentSettings,
    Settings,
    config_dir,
    config_path,
    data_dir,
    is_configured,
    load_api_key,
    load_settings,
)
from allie.documents.folder import Shelf

if TYPE_CHECKING:
    from rich.table import Table

    from allie.agent.core import Confirm, Dispatch
    from allie.agent.limits import Limits
    from allie.app import Turn
    from allie.audio.capture import LiveCapture
    from allie.live.base import LiveProvider, ToolCall
    from allie.locales import Locale
    from allie.store.memory import UserMemory
    from allie.store.repos import AuditRepo, ModelUsage, SettingsRepo
    from allie.tools.mail import Mailbox
    from allie.tools.registry import ToolRegistry
    from allie.ui.status import Screen
    from allie.ui.tray import Tray
    from allie.ui.window import Switch

__all__ = ["BUILTIN_TOOLS", "DOCTOR_TEXT", "TEXT", "build_parser", "main", "use_utf8"]

_OK = 0
_GAVE_UP = 1

# The last link of the chain of section 3.12, as in every other module that
# says something: the pack answers first, and these are what is left if none
# does. Keys are unique across the project - `test_locales.py` checks.
TEXT: dict[str, str] = {
    "not_set_up": "Nothing is set up yet. Run 'allie setup' first.",
    "cannot_start": "The assistant cannot start: {problem}",
    "stopped": "Stopped.",
    # The probe of 2.6, run again at startup when its verdict is a week
    # old: a model that fails is warned about and used anyway, because the
    # user may have chosen it knowing (section 3.2).
    "model_no_tools": (
        "The model does not call tools, and most of what the assistant does depends on that. "
        "Run 'allie setup' to choose another."
    ),
    # `allie cost` (section 6): two small tables, today and this month,
    # one row per model. The amounts are written by the code as `$0.0004` -
    # number formatting is the locale formatter of phase 3, and until then a
    # dollar sign reads the same in every language.
    "cost_none": "No usage recorded yet.",
    "cost_today": "today",
    "cost_month": "this month",
    "cost_model": "model",
    "cost_turns": "turns",
    "cost_tokens": "in / out / cached",
    "cost_spent": "spent",
    "cost_total": "total",
    "cost_unpriced": (
        "{count} turns of {model} have no price in pricing.toml and are not in the total."
    ),
    # `allie purge --all` (4.6): what will go is listed first, what
    # stays is said, and nothing is deleted before the word is typed.
    "purge_will_delete": "This will delete everything the assistant recorded:",
    "purge_file": "  file: {path}",
    "purge_secret": "  Credential Manager entry: {name}",
    "purge_keeps": "config.toml and contacts.toml stay: you wrote those.",
    "purge_confirm": "Type yes to delete all of it:",
    "purge_cancelled": "Nothing was deleted.",
    "purge_done": "Deleted {count} items.",
    "purge_nothing": "There is nothing to delete.",
    # `allie autostart on | off | status` (4.2): the `Run` key of the
    # user's own sign-in, and what is written there.
    "autostart_on": "The assistant will start when you sign in, as: {command}",
    "autostart_off": "The assistant will no longer start when you sign in.",
    "autostart_status_on": "Starts when you sign in, as: {command}",
    "autostart_status_off": ("Does not start when you sign in. 'allie autostart on' changes that."),
}

# The one word `purge --all` accepts, in any letter case. The question
# names it, so it is the same in every language rather than the pack's
# yes-word - a typed answer to a question about deleting everything should
# not depend on which pack is loaded.
PURGE_WORD = "yes"

# The tools `run` puts on offer, by name and in order, so that `doctor` can
# count them without building them. `test_cli.py` checks that the registry
# `run` builds is this list followed by the user's own. `look_up`,
# `x_trends`, `open_documents` and `ask_documents` need the stored `gemini`
# key (D29, D30, D35); without it they are not offered.
BUILTIN_TOOLS: tuple[str, ...] = (
    "get_current_time",
    "system_status",
    "get_weather",
    "open_app",
    "open_url",
    "look_up",
    "search_web",
    "read_clipboard",
    "fetch_page",
    "x_trends",
    "open_documents",
    "ask_documents",
    "read_latest_emails",
    "search_emails",
    "open_settings",
    "media_control",
    "set_volume",
    "play_music",
    "play_video",
    "open_media",
    "add_note",
    "search_notes",
    "delete_note",
    "create_reminder",
    "list_reminders",
    "cancel_reminder",
    "remember",
    "forget",
    "install_app",
    "send_message",
)

# What `allie doctor` says, line by line. Labels rather than sentences,
# and every value written by the code: a path, a number, a name. Never a
# key - the report says whether one is stored, and nothing else about it.
DOCTOR_TEXT: dict[str, str] = {
    "doctor_pack": "language pack: {name} ({code})",
    "doctor_who_answers": "Who answers",
    "doctor_model": "  model: {model} ({provider})",
    "doctor_key_stored": "  API key: stored in the Credential Manager",
    "doctor_key_missing": "  API key: none stored - run 'allie setup'",
    "doctor_key_not_needed": "  API key: not needed",
    "doctor_verdict_ok": "  tool calling: passed, checked {days} days ago",
    "doctor_verdict_failed": (
        "  tool calling: FAILED, checked {days} days ago - most tools will not work"
    ),
    "doctor_verdict_none": "  tool calling: not checked yet (checked at the next start)",
    "doctor_stt_local": "  recogniser: Whisper '{size}', on this machine",
    "doctor_stt_gemini": "  recogniser: Google ({model}), Whisper '{size}' behind it",
    "doctor_leaves": "What leaves this machine",
    "doctor_voice_stays": "  your voice: stays here",
    "doctor_voice_goes": "  your voice: goes to Google (the recogniser)",
    "doctor_text_goes": "  what you said, as text: goes to {provider}",
    "doctor_content_goes": "  pages, clipboard text and mail you ask about: go to {provider}",
    "doctor_weather": "  a place you ask the weather for: goes to Open-Meteo",
    "doctor_store": "  the name of an app you do not have: goes to the Microsoft Store",
    "doctor_messages": "  a message: the WhatsApp app on this PC, or Telegram's servers as you",
    "doctor_where": "Where things are",
    "doctor_settings": "  settings: {path}",
    "doctor_contacts": "  contacts: {path} ({count} people)",
    "doctor_contacts_none": "  contacts: {path} (not written yet)",
    "doctor_memory": "  memory: {path} ({count} facts)",
    "doctor_tools_dir": "  your tools: {path}",
    "doctor_database": "  database: {path} ({size})",
    "doctor_database_none": "  database: {path} (not created yet)",
    "doctor_log": "  log: {path}",
    "doctor_tools": "Tools",
    "doctor_builtin": "  built in: {count}",
    "doctor_local": "  your own: {count} ({names})",
    "doctor_local_none": "  your own: none",
    "doctor_limits": "Limits",
    "doctor_per_turn": "  per turn: {calls} tool calls, {tokens} output tokens, {seconds} s",
    "doctor_spend_warn": "  spend: ${daily} a day, ${monthly} a month - a warning past either",
    "doctor_spend_stop": (
        "  spend: ${daily} a day, ${monthly} a month - the model is not asked past either"
    ),
    "doctor_retention": "  what a tool answered is kept in the audit for {days} days",
    "doctor_retention_forever": "  what a tool answered is kept in the audit for good",
    "doctor_set_up": "Set up",
    "doctor_mail": "  mail: {user} at {host} ({mailbox})",
    "doctor_mail_none": "  mail: not set up ('allie mail login')",
    "doctor_telegram": "  telegram: logged in (api_id {api_id})",
    "doctor_telegram_none": "  telegram: not set up ('allie telegram login')",
    "doctor_autostart_on": "  starts when you sign in: yes ({command})",
    "doctor_autostart_off": "  starts when you sign in: no ('allie autostart on')",
}


def build_parser() -> argparse.ArgumentParser:
    """Builds the top level argument parser."""
    parser = argparse.ArgumentParser(
        prog="allie",
        description=(
            "A voice assistant on live speech-to-speech models, "
            "running on your own API key and language."
        ),
    )
    parser.add_argument("--version", action="version", version=f"allie {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.add_parser(
        "setup", help="Choose a provider, store the API key, pick a model and a microphone."
    )
    subparsers.add_parser("mic", help="Choose which microphone the assistant listens through.")
    run = subparsers.add_parser(
        "run", help="Start the assistant. Ctrl+Alt+H pauses and resumes listening; Ctrl+C stops."
    )
    run.add_argument(
        "--device",
        default=None,
        help=(
            "Input device: an index, or words from its name as "
            "'scripts/bench_mic.py --list-devices' prints them. "
            "Overrides [audio] input_device in config.toml for this run; "
            "'allie mic' changes it for good."
        ),
    )
    run.add_argument(
        "--tray",
        action="store_true",
        help=(
            "With --terminal, also show an icon in the notification area: the state, "
            "a switch for listening, the settings folder, quit. The window always has it."
        ),
    )
    run.add_argument(
        "--terminal",
        action="store_true",
        help="Keep the terminal status line instead of opening the window.",
    )
    subparsers.add_parser("cost", help="Show what the assistant has spent, today and this month.")
    subparsers.add_parser(
        "doctor",
        help=(
            "Report who answers, what leaves this machine and where, where the files are, "
            "how many tools there are and what the limits are. Never prints a key."
        ),
    )
    purge = subparsers.add_parser(
        "purge",
        help=(
            "Delete what the assistant recorded: the database, the memory file, the logs "
            "and the keys in the Credential Manager. Lists it first and asks for 'yes'."
        ),
    )
    purge.add_argument(
        "--all",
        action="store_true",
        required=True,
        help="Everything. The only choice there is, and the one to be explicit about.",
    )
    autostart = subparsers.add_parser(
        "autostart", help="Start the assistant, with its tray icon, when you sign in to Windows."
    )
    autostart.add_argument(
        "autostart_command",
        choices=("on", "off", "status"),
        metavar="on|off|status",
        help="Register it under your own Run key, take it off again, or say which it is.",
    )
    mail = subparsers.add_parser(
        "mail", help="Mail: sign in once over IMAP, so that your messages can be read to you."
    )
    mail_commands = mail.add_subparsers(dest="mail_command", metavar="<subcommand>", required=True)
    mail_commands.add_parser(
        "login",
        help=(
            "Ask for the IMAP server and your address if config.toml has neither, and for an "
            "app password; connect once; keep the password in the Credential Manager."
        ),
    )
    telegram = subparsers.add_parser(
        "telegram", help="Telegram: log in once, so that messages can be sent as you."
    )
    telegram_commands = telegram.add_subparsers(
        dest="telegram_command", metavar="<subcommand>", required=True
    )
    telegram_commands.add_parser(
        "login",
        help=(
            "Ask for your api_id and api_hash (my.telegram.org), your phone and the code, "
            "and keep the session in the Credential Manager."
        ),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parses the arguments and dispatches to a command."""
    use_utf8(sys.stdout)
    use_utf8(sys.stderr)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "setup":
        from allie import setup_wizard

        # Looked up on the module rather than imported by name so the tests can
        # stand in for it; the wizard itself opens a prompt and would hang.
        # The questions in a row once; after that, one setting at a time.
        if not is_configured():
            return asyncio.run(setup_wizard.run_setup(setup_wizard.TerminalPrompter()))
        return asyncio.run(setup_wizard.run_settings(setup_wizard.TerminalPrompter()))

    if args.command == "mic":
        from allie import setup_wizard

        return asyncio.run(setup_wizard.run_microphone_setup(setup_wizard.TerminalPrompter()))

    if args.command == "cost":
        return _cost()

    if args.command == "doctor":
        return _doctor()

    if args.command == "purge":
        return _purge()

    if args.command == "autostart":
        return _autostart(args.autostart_command)

    if args.command == "mail":
        return _mail_login()

    if args.command == "telegram":
        return _telegram_login()

    return _run(device=args.device, tray=args.tray, terminal=args.terminal)


def use_utf8(stream: object) -> None:
    """Asks a stream to stop encoding in the machine's legacy code page.

    A console Windows owns is already fine; a redirected one - `allie run >
    run.log`, or anything that reads our output through a pipe - is not, and
    the first Turkish sentence ends the program. Something that cannot be asked
    is left alone: not being able to is no reason to refuse to start.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return

    with contextlib.suppress(OSError, ValueError):
        reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------------------
# assistant run
# --------------------------------------------------------------------------


def _telegram_login() -> int:
    """`allie telegram login`: the questions of `messaging/telegram.py`,
    once, and the three things they produce stored where each belongs -
    the `api_id` in `config.toml`, the hash and the session in the
    Credential Manager (section 10). The settings are rewritten from what
    was loaded, so nothing else in the file changes."""
    from allie import setup_wizard
    from allie.config import TelegramSettings, save_settings, store_api_key
    from allie.messaging import telegram

    settings = load_settings()
    pack = locales.load(settings.locale.code if is_configured() else locales.system_code())
    text = {key: pack.say(key, default) for key, default in telegram.TEXT.items()}
    text["cancelled"] = pack.say("cancelled", setup_wizard.TEXT["cancelled"])
    prompter = setup_wizard.TerminalPrompter(text=text)

    try:
        done = asyncio.run(
            telegram.login(
                prompter,
                api_id=settings.telegram.api_id,
                api_hash=load_api_key(telegram.HASH_ENTRY) or "",
            )
        )
    except telegram.LoginError as refusal:
        prompter.say("telegram_login_failed", reason=refusal)
        return _GAVE_UP
    if done is None:
        prompter.say("cancelled")
        return _GAVE_UP

    store_api_key(telegram.HASH_ENTRY, done.api_hash)
    store_api_key(telegram.SESSION_ENTRY, done.session)
    settings.telegram = TelegramSettings(api_id=done.api_id)
    save_settings(settings)
    prompter.say("telegram_logged_in", name=done.name)
    return _OK


def _doctor() -> int:
    """`allie doctor` (3.5): what this installation is, in one screen.

    Who answers and through what; which of the user's data leaves the
    machine and for where - the sentence the README makes, checked against
    the settings actually in force; where every file is; how many tools
    there are; the limits. Everything is read, nothing is loaded and nothing
    is connected to: no Whisper, no microphone, no request to any provider.
    A key is reported as stored or not, and never shown, the way
    `describe_myself` of section 10 has it: named fields, not a dump.
    """
    from rich.console import Console

    from allie import autostart
    from allie.live.probe import probe_key, remembered
    from allie.live.registry import load_catalog
    from allie.logs import log_path
    from allie.messaging.contacts import AddressBook, ContactsFileError, contacts_path
    from allie.messaging.telegram import HASH_ENTRY, SESSION_ENTRY
    from allie.store.db import database_path, open_database
    from allie.store.memory import MemoryFileError, UserMemory, memory_path
    from allie.store.repos import SettingsRepo
    from allie.stt.local_whisper import DEFAULT_MODEL_SIZE
    from allie.tools.local import load_local_tools, local_tools_dir
    from allie.tools.mail import MAIL_ENTRY

    settings = load_settings()
    ready = is_configured()
    pack = locales.load(settings.locale.code if ready else locales.system_code())
    said = {key: pack.say(key, default) for key, default in {**TEXT, **DOCTOR_TEXT}.items()}
    console = Console()
    lines: list[str] = []

    def say(key: str, **fields: object) -> None:
        lines.append(said[key].format(**fields))

    lines.append(f"allie {__version__}")
    say("doctor_pack", name=pack.name, code=pack.code)
    if not ready:
        lines.append("")
        say("not_set_up")
        console.print("\n".join(lines), markup=False, highlight=False, soft_wrap=True)
        return _GAVE_UP

    # Who answers.
    catalogue = load_catalog()
    provider_id = settings.live.provider
    entry = catalogue.get(provider_id)
    provider_name = entry.display_name if entry is not None else provider_id
    lines.append("")
    say("doctor_who_answers")
    say("doctor_model", model=settings.live.primary, provider=provider_name)
    if entry is not None and not entry.requires_key:
        say("doctor_key_not_needed")
    elif load_api_key(provider_id) is not None:
        say("doctor_key_stored")
    else:
        say("doctor_key_missing")
    verdict, age = None, 0
    if database_path().is_file():
        connection = open_database()
        try:
            verdicts = SettingsRepo(connection)
            verdict = remembered(verdicts, provider_id, settings.live.model, ttl=float("inf"))
            stored = verdicts.get(probe_key(provider_id, settings.live.model))
            age = _days_since(stored)
        finally:
            connection.close()
    if verdict is None:
        say("doctor_verdict_none")
    elif verdict.ok:
        say("doctor_verdict_ok", days=age)
    else:
        say("doctor_verdict_failed", days=age)
    if settings.stt.provider == "gemini":
        say("doctor_stt_gemini", model=settings.stt.model, size=DEFAULT_MODEL_SIZE)
    else:
        say("doctor_stt_local", size=DEFAULT_MODEL_SIZE)

    # What leaves this machine.
    lines.append("")
    say("doctor_leaves")
    say("doctor_voice_goes" if settings.stt.provider == "gemini" else "doctor_voice_stays")
    say("doctor_text_goes", provider=provider_name)
    say("doctor_content_goes", provider=provider_name)
    say("doctor_weather")
    say("doctor_store")
    say("doctor_messages")

    # Where things are.
    lines.append("")
    say("doctor_where")
    say("doctor_settings", path=config_path())
    contacts = contacts_path()
    if contacts.is_file():
        try:
            people = len(AddressBook.load().names())
            say("doctor_contacts", path=contacts, count=people)
        except ContactsFileError as problem:
            lines.append(f"  contacts: {contacts} ({problem})")
    else:
        say("doctor_contacts_none", path=contacts)
    try:
        say("doctor_memory", path=memory_path(), count=len(UserMemory.load().facts))
    except MemoryFileError as problem:
        lines.append(f"  memory: {memory_path()} ({problem})")
    say("doctor_tools_dir", path=local_tools_dir())
    database = database_path()
    if database.is_file():
        say("doctor_database", path=database, size=_size(database.stat().st_size))
    else:
        say("doctor_database_none", path=database)
    say("doctor_log", path=log_path())

    # Tools.
    lines.append("")
    say("doctor_tools")
    say("doctor_builtin", count=len(BUILTIN_TOOLS))
    own = [tool.spec.name for tool in load_local_tools()]
    if own:
        say("doctor_local", count=len(own), names=", ".join(own))
    else:
        say("doctor_local_none")

    # Limits.
    limits = settings.limits
    lines.append("")
    say("doctor_limits")
    say(
        "doctor_per_turn",
        calls=limits.tool_calls_per_turn,
        tokens=limits.output_tokens,
        seconds=_number(limits.turn_seconds),
    )
    say(
        "doctor_spend_stop" if limits.hard_stop else "doctor_spend_warn",
        daily=f"{limits.daily_usd:.2f}",
        monthly=f"{limits.monthly_usd:.2f}",
    )
    if settings.retention.audit_days:
        say("doctor_retention", days=settings.retention.audit_days)
    else:
        say("doctor_retention_forever")

    # Set up.
    lines.append("")
    say("doctor_set_up")
    mail = settings.mail
    if mail.host.strip() and mail.user.strip() and load_api_key(MAIL_ENTRY) is not None:
        say("doctor_mail", user=mail.user, host=mail.host, mailbox=mail.mailbox)
    else:
        say("doctor_mail_none")
    if (
        settings.telegram.api_id
        and load_api_key(HASH_ENTRY) is not None
        and load_api_key(SESSION_ENTRY) is not None
    ):
        say("doctor_telegram", api_id=settings.telegram.api_id)
    else:
        say("doctor_telegram_none")
    command = autostart.status(autostart.WindowsRegistry())
    if command is not None:
        say("doctor_autostart_on", command=command)
    else:
        say("doctor_autostart_off")

    # Without wrapping: a path folded at eighty columns is a path nobody
    # can copy.
    console.print("\n".join(lines), markup=False, highlight=False, soft_wrap=True)
    return _OK


def _days_since(stored: str | None) -> int:
    """How many days ago a probe verdict was written, from its own record."""
    import json

    if stored is None:
        return 0
    try:
        record = json.loads(stored)
    except ValueError:
        return 0
    reached = record.get("ts") if isinstance(record, dict) else None
    if not isinstance(reached, int | float):
        return 0
    return max(int((time.time() - reached) // 86_400), 0)


def _size(octets: int) -> str:
    """`1.2 MB`, `340 KB`, `512 B` - for a file size on the doctor's screen."""
    if octets >= 1_000_000:
        return f"{octets / 1_000_000:.1f} MB"
    if octets >= 1_000:
        return f"{octets // 1_000} KB"
    return f"{octets} B"


def _number(value: float) -> str:
    """`90` for `90.0`, `2.5` for `2.5`."""
    return str(int(value)) if value == int(value) else str(value)


def _mail_login() -> int:
    """`allie mail login`: the questions of `tools/mail.py`, once, and
    what they produce stored where each belongs - the server and the
    address in `config.toml`, the app password in the Credential Manager
    under `mail` (section 10). The settings are rewritten from what was
    loaded, so nothing else in the file changes."""
    from allie import setup_wizard
    from allie.config import MailSettings, save_settings, store_api_key
    from allie.tools import mail

    settings = load_settings()
    pack = locales.load(settings.locale.code if is_configured() else locales.system_code())
    text = {key: pack.say(key, default) for key, default in mail.TEXT.items()}
    text["cancelled"] = pack.say("cancelled", setup_wizard.TEXT["cancelled"])
    prompter = setup_wizard.TerminalPrompter(text=text)

    try:
        done = asyncio.run(
            mail.login(
                prompter,
                host=settings.mail.host,
                port=settings.mail.port,
                user=settings.mail.user,
                mailbox=settings.mail.mailbox,
            )
        )
    except mail.MailError as refusal:
        prompter.say("mail_login_failed", reason=refusal)
        return _GAVE_UP
    if done is None:
        prompter.say("cancelled")
        return _GAVE_UP

    store_api_key(mail.MAIL_ENTRY, done.password)
    settings.mail = MailSettings(
        host=done.host,
        port=settings.mail.port,
        user=done.user,
        mailbox=settings.mail.mailbox,
    )
    save_settings(settings)
    prompter.say(
        "mail_logged_in",
        host=done.host,
        user=done.user,
        count=done.count,
        mailbox=settings.mail.mailbox,
    )
    return _OK


def _purge() -> int:
    """`allie purge --all`: what the machine recorded, gone (section 3.7, 4.6).

    The database with its audit rows, notes and reminders; the memory file;
    the logs; and every key and session in the Credential Manager. What is
    there is listed first - by path and by entry name, never by value - and
    nothing is deleted before `yes` is typed. `config.toml` and
    `contacts.toml` stay: the user wrote those by hand, and a file edited
    calmly is not "what the assistant recorded".
    """
    from allie import setup_wizard
    from allie.config import delete_api_key

    settings = load_settings()
    pack = locales.load(settings.locale.code if is_configured() else locales.system_code())
    text = {key: pack.say(key, default) for key, default in TEXT.items()}
    prompter = setup_wizard.TerminalPrompter(text=text)

    files = [path for path in _recorded_files() if path.is_file()]
    secrets = [entry for entry in _secret_entries() if load_api_key(entry) is not None]
    if not files and not secrets:
        prompter.say("purge_nothing")
        return _OK

    prompter.say("purge_will_delete")
    for path in files:
        prompter.say("purge_file", path=path)
    for entry in secrets:
        prompter.say("purge_secret", name=entry)
    prompter.say("purge_keeps")

    answer = asyncio.run(prompter.ask("purge_confirm"))
    if answer is None or answer.strip().lower() != PURGE_WORD:
        prompter.say("purge_cancelled")
        return _GAVE_UP

    for path in files:
        path.unlink()
    for entry in secrets:
        delete_api_key(entry)
    prompter.say("purge_done", count=len(files) + len(secrets))
    return _OK


def _recorded_files() -> list[Path]:
    """Every file the assistant writes for itself, whether or not it exists.

    The database and the two files SQLite keeps beside it in WAL mode - a
    write-ahead log holds rows too; the memory file; the log and what
    rotation left of it.
    """
    from allie.logs import LOG_FILE, log_path
    from allie.store.db import database_path
    from allie.store.memory import memory_path

    database = database_path()
    logs = log_path().parent
    rotated = sorted(logs.glob(f"{Path(LOG_FILE).stem}*{Path(LOG_FILE).suffix}"))
    return [
        database,
        database.with_name(f"{database.name}-wal"),
        database.with_name(f"{database.name}-shm"),
        database.with_name(f"{database.name}-journal"),
        memory_path(),
        *rotated,
    ]


def _secret_entries() -> list[str]:
    """Every name a secret may be stored under: one per provider in the
    catalogue, the two of Telegram, and the mail password."""
    from allie.live.registry import load_catalog
    from allie.messaging.telegram import HASH_ENTRY, SESSION_ENTRY
    from allie.tools.mail import MAIL_ENTRY

    return [*load_catalog(), HASH_ENTRY, SESSION_ENTRY, MAIL_ENTRY]


def _autostart(action: str) -> int:
    """`allie autostart on | off | status` (4.2): the user's own `Run`
    key, written with the assistant as installed here and `run --tray`,
    cleared, or read back. No administrator, no service (`autostart.py`)."""
    from rich.console import Console

    from allie import autostart

    settings = load_settings()
    pack = locales.load(settings.locale.code if is_configured() else locales.system_code())
    said = {key: pack.say(key, default) for key, default in TEXT.items()}
    console = Console()
    registry = autostart.WindowsRegistry()

    if action == "on":
        line = said["autostart_on"].format(command=autostart.enable(registry))
    elif action == "off":
        autostart.disable(registry)
        line = said["autostart_off"]
    else:
        command = autostart.status(registry)
        line = (
            said["autostart_status_off"]
            if command is None
            else said["autostart_status_on"].format(command=command)
        )
    console.print(line, markup=False, highlight=False)
    return _OK


def _run(*, device: str | None = None, tray: bool = False, terminal: bool = False) -> int:
    """Starts the assistant, or says why it cannot.

    `device` is the `--device` flag: a microphone by index or by words from its
    name, outranking the settings for this one run. `tray` is `--tray`: the
    icon of 4.3 beside the terminal line, whose "quit" ends the run the way
    Ctrl+C does - the window has the icon whether asked or not (D34).
    `terminal` is `--terminal`: the status line on the terminal instead of
    the window (plan.md D20) - the window is the face otherwise, and on a
    machine that is not set up yet it opens on the wizard.
    """
    from rich.console import Console

    from allie.audio.capture import MicrophoneUnavailableError, device_choice
    from allie.audio.wake import WakeModelMissingError
    from allie.live.registry import RegistryError
    from allie.logs import setup_logging
    from allie.messaging.contacts import ContactsFileError
    from allie.store.memory import MemoryFileError
    from allie.stt.local_whisper import ModelUnavailableError
    from allie.tools.messaging import BadDefaultAppError

    settings = load_settings()
    ready = is_configured()

    # Before setup there is no chosen language to read this in, so the machine's
    # own is the best guess available - the same one the wizard starts with.
    pack = locales.load(settings.locale.code if ready else locales.system_code())
    console = Console()
    said = {key: pack.say(key, default) for key, default in TEXT.items()}

    def say(key: str, **fields: object) -> None:
        # Without markup: the text of an exception is formatted into one of
        # these, and a square bracket in it is not a colour.
        console.print(said[key].format(**fields), markup=False, highlight=False)

    if not ready and terminal:
        say("not_set_up")
        return _GAVE_UP

    setup_logging()
    # What the user can fix and the program cannot: a key that is gone,
    # weights that could not be fetched, a microphone that would not open, a
    # memory file edited into something that does not parse, a wake-word
    # model that is not where the settings say. Each is one sentence and
    # exit code 1. Anything else is a bug in this project and keeps its
    # traceback.
    fixable = (
        RegistryError,
        WakeModelMissingError,
        ModelUnavailableError,
        MicrophoneUnavailableError,
        MemoryFileError,
        ContactsFileError,
        BadDefaultAppError,
    )
    # The flag for one evening with a headset; the settings for every other
    # day; the system default when neither says anything.
    microphone = device_choice(device if device is not None else settings.audio.input_device)
    try:
        if terminal:
            asyncio.run(_terminal(settings, pack, device=microphone, tray=tray))
        else:
            # On the window a fixable failure is a sentence on the window and
            # a wait for the settings button (`_session`); only the ending
            # is said here.
            asyncio.run(
                _session(
                    settings,
                    pack,
                    device=microphone,
                    fixable=fixable,
                    cannot_start=lambda problem: said["cannot_start"].format(problem=problem),
                )
            )
            say("stopped")
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Ctrl+C, or "quit" on the tray's menu, which cancels the run from
        # its own thread (4.3): an ending rather than a crash - and the
        # microphone and the keyboard hook are already closed by the time
        # this is printed (`app.run`).
        say("stopped")
    except fixable as problem:
        say("cannot_start", problem=problem)
        return _GAVE_UP

    return _OK


async def _terminal(
    settings: Settings, pack: Locale, *, device: int | str | None, tray: bool
) -> None:
    """`run --terminal`: the line on the terminal is the screen, as it was
    before the window (plan.md D20)."""
    from allie.ui.status import StatusLine

    with StatusLine(pack) as screen:
        await _talk(settings, pack, device=device, tray=tray, screen=screen)


class _Wants:
    """What the window's buttons asked of `_session` (plan.md D20): to stop
    for the wizard, or to stop for good. The talk under way is a task the
    buttons cancel; between two talks they wake the wait."""

    def __init__(self) -> None:
        self.task: asyncio.Task[None] | None = None
        self.settings = False
        # An event rather than a flag: it is read after an await, where a
        # flag flipped by a button would be narrowed away by the checker.
        self.quitting = asyncio.Event()
        self._pressed = asyncio.Event()

    def quit(self) -> None:
        self.quitting.set()
        self._interrupt()

    def open_settings(self) -> None:
        self.settings = True
        self._interrupt()

    async def pressed(self) -> None:
        """Waits for either button."""
        self._pressed.clear()
        await self._pressed.wait()

    def _interrupt(self) -> None:
        self._pressed.set()
        if self.task is not None:
            self.task.cancel()


async def _session(
    settings: Settings,
    pack: Locale,
    *,
    device: int | str | None,
    fixable: tuple[type[Exception], ...],
    cannot_start: Callable[[BaseException], str],
) -> None:
    """The window's run (plan.md D20, spec section 1): the window up first,
    the wizard on it when nothing is set up or the settings button asks,
    then the assistant as a task the buttons can cancel - "settings" for
    the wizard and another go, "quit" for the end. A failure the user can
    fix is a sentence on the window and a wait for a button, since the
    settings button is the way to fix it.

    The tray icon goes up with the window and down with it (D34), across
    every wizard and every talk: minimising hides the window into it, so it
    must be there whenever the window is - on the wizard's page and under a
    failed start too. Its switch is the window's, its "quit" the window's,
    and its default line brings the window back.

    Ctrl+C on the terminal cancels this task; the talk under way is
    cancelled with it and the cancellation is let through, so that `_run`
    says the same "stopped" it always said.
    """
    from allie import setup_wizard
    from allie.ui import tray as tray_ui
    from allie.ui.window import Switch, Window, WindowPrompter

    loop = asyncio.get_running_loop()
    wants = _Wants()
    switch = Switch()
    window = Window(
        pack,
        loop=loop,
        on_toggle=switch,
        on_quit=wants.quit,
        on_settings=wants.open_settings,
        tray=True,
    )
    icon = tray_ui.Tray(
        pack,
        loop=loop,
        on_toggle=switch,
        on_quit=wants.quit,
        on_show=window.show,
        settings_folder=config_dir(),
    )
    window.start()
    icon.start()
    try:
        while not wants.quitting.is_set():
            if not is_configured() or wants.settings:
                first = not is_configured()
                wants.settings = False
                window.wizard(True)
                try:
                    # Looked up on the module, as `setup` does, so that the
                    # tests can stand in for it. The questions in a row on a
                    # machine never set up; the list of settings, each
                    # changed on its own, every time after (D32).
                    prompter = WindowPrompter(window)
                    code = await (
                        setup_wizard.run_setup(prompter)
                        if first
                        else setup_wizard.run_settings(prompter)
                    )
                finally:
                    window.wizard(False)
                if wants.quitting.is_set() or (code != _OK and not is_configured()):
                    return
                settings = load_settings()
                pack = locales.load(settings.locale.code)
            window.loading()
            wants.task = asyncio.create_task(
                _talk(
                    settings,
                    pack,
                    device=device,
                    screen=window,
                    switch=switch,
                    icon=icon,
                )
            )
            try:
                await wants.task
            except asyncio.CancelledError:
                if not wants.task.done():
                    # The cancellation is ours (Ctrl+C): take the talk down
                    # with us and let it through.
                    wants.task.cancel()
                    await asyncio.gather(wants.task, return_exceptions=True)
                    raise
                if wants.settings:
                    continue
                return
            except fixable as problem:
                window.failed()
                window.notice(cannot_start(problem))
                await wants.pressed()
                continue
            finally:
                wants.task = None
            return
    finally:
        icon.stop()
        window.stop()


async def _talk(
    settings: Settings,
    pack: Locale,
    *,
    device: int | str | None = None,
    tray: bool = False,
    screen: Screen,
    switch: Switch | None = None,
    icon: Tray | None = None,
) -> None:
    """Builds the pieces and lets the state machine drive them (plan.md 4.1).

    `screen` is what it all shows on - the terminal's line or the window
    (D20); `switch`, when there is one, is the window's listen button,
    given the capture's `toggle` once there is a capture; `icon` is the
    window's tray icon, up already and kept up by whoever put it there
    (D34), which is told what the screen is told. Without one, `tray`
    (`--tray` beside the terminal line) puts up an icon of this run's own.
    """
    from loguru import logger

    from allie.agent.core import ToolRunner
    from allie.agent.limits import Limits
    from allie.announce.queue import AnnounceQueue
    from allie.app import LiveAssistant, confirm_prompt
    from allie.assistants import threshold_for
    from allie.audio.capture import LiveCapture, SystemMicrophone
    from allie.audio.player import SystemSpeaker
    from allie.audio.vad import Endpoint, SileroVAD
    from allie.audio.volume import SystemVolume
    from allie.audio.wake import LiveKitWakeWord, wake_model_path
    from allie.documents import model as document_module
    from allie.live.base import SessionConfig
    from allie.live.registry import MissingAPIKeyError, create_provider
    from allie.machine import Win32Machine
    from allie.media.player import Player
    from allie.messaging.contacts import AddressBook
    from allie.messaging.telegram import HASH_ENTRY, SESSION_ENTRY, Telegram, TelethonClient
    from allie.messaging.whatsapp import WhatsApp
    from allie.scheduler import runner as scheduler_runner
    from allie.store.db import open_database
    from allie.store.memory import UserMemory
    from allie.store.repos import (
        AuditRepo,
        NotesRepo,
        ReminderRepo,
        SettingsRepo,
        UsageRepo,
    )
    from allie.store.retention import blank_old_audit_summaries
    from allie.stt.gemini_stt import GeminiSTT
    from allie.stt.local_whisper import LocalWhisper
    from allie.tools import mail as mail_tools
    from allie.tools import memory as memory_tools
    from allie.tools import messaging as messaging_tools
    from allie.tools import notes as notes_tools
    from allie.tools import reminders as reminder_tools
    from allie.tools import store as store_tools
    from allie.tools import system as system_tools
    from allie.tools import weather as weather_tools
    from allie.tools.documents import documents_tools_for
    from allie.tools.local import load_local_tools
    from allie.tools.media import (
        media_control,
        open_media_for,
        play_music_for,
        play_video_for,
        set_volume_for,
    )
    from allie.tools.registry import ToolRegistry
    from allie.tools.status import system_status_for
    from allie.tools.system import (
        AppCatalog,
        get_current_time,
        open_app_for,
        open_settings,
        open_url,
    )
    from allie.tools.trends import x_trends_for
    from allie.tools.web import fetch_page_for, look_up_for, read_clipboard_for, search_web_for
    from allie.tts.live_voice import LiveVoice
    from allie.ui import tray as tray_ui
    from allie.usage.tracker import Pricing, UsageTracker
    from allie.web import search as search_module
    from allie.web.page import PageReader
    from allie.web.trends import TrendsPage

    # First, and before anything slow: a provider that cannot be built is the
    # likeliest thing to be wrong, and the cheapest to find out about. The
    # database is next for the same reason - cheap, and a disk that refuses
    # is better found out about before Whisper has been loaded.
    live = settings.live
    provider = create_provider(live.provider, base_url=live.base_url or None)
    # The wake word (D21), resolved before anything slow is loaded: a model
    # that is not there is a sentence now, not after Whisper. The threshold
    # is the user's when they wrote one, else the one the model shipped
    # with (D31).
    wake_word: LiveKitWakeWord | None = None
    if settings.wake.enabled:
        wake_word = LiveKitWakeWord(
            wake_model_path(settings.wake.model),
            threshold=settings.wake.threshold or threshold_for(settings.wake.model),
        )
    # What the user asked to be kept, and the assistant's name (section
    # 3.7, 2.10): read once here, written by the two tools below, and read
    # into the prompt at every session open. Before the database for the
    # same reason as the provider - a file edited into nonsense is a sentence.
    memory = UserMemory.load()
    # The people the user can message (`messaging/contacts.py`), read here
    # for the same reason: a `contacts.toml` edited into nonsense - or into
    # two people who answer to one name - is a sentence before anything slow.
    book = AddressBook.load()
    database = open_database()
    # What the audit rows still say (section 3.7, 4.6): a summary older
    # than the user's `[retention] audit_days` is blanked here, once per
    # start and before anything reads the table. The rows themselves stay.
    blanked = blank_old_audit_summaries(database, days=settings.retention.audit_days)
    if blanked:
        logger.info("retention: {count} audit summaries blanked", count=blanked)
    # The table of section 3.11, once, for everyone who reads a row of it:
    # the tool round, the gate and the tracker.
    limits = Limits.from_settings(settings.limits)
    # Where music comes from (section 3.6). Built before the try, because it
    # holds a connection open between searches and the `finally` below is what
    # gives it back; three tools and `open_app` share the one player.
    player = Player(settings=settings.media)
    # The weather, from Open-Meteo over one kept connection (15 Sep 2026);
    # built beside the player for the same reason, and given back in the
    # same `finally`.
    weather = weather_tools.OpenMeteo()
    # Pages the user asks about (17 Sep 2026), over a kept connection like
    # the weather; how long one may take is the user's `[web]` setting.
    reader = PageReader(seconds=settings.web.timeout_seconds)
    # Looking things up (D29): Google's search through the model the free
    # key may search with, on the key the Gemini entry is filed under. What
    # it searched is kept for the screen until the turn ends. No key, no
    # `look_up` - `search_web` still opens the browser.
    search_key = load_api_key("gemini")
    searched = search_module.Searched()
    search = (
        search_module.GroundedSearch(
            search_key, model=settings.web.look_up_model, searched=searched
        )
        if search_key
        else None
    )
    looking_up = [] if search is None else [look_up_for(search)]
    # X's trends by country (D30): trends24's latest hour over a kept
    # connection, explained by the same search; given back in the same
    # `finally` as the weather and the pages.
    trends = TrendsPage(seconds=settings.web.timeout_seconds)
    trending = [] if search is None else [x_trends_for(trends, search)]
    # The user's document folders (D35): the root is listed at every open
    # for the prompt and at every call by the two tools; a second model
    # answers from a folder's files, on the key `look_up` uses. No key, no
    # tools, no list in the prompt.
    shelf = _documents_shelf(settings.documents)
    reading_documents = (
        documents_tools_for(
            shelf,
            document_module.DocumentModel(
                search_key,
                model=settings.documents.model,
                seconds=settings.documents.timeout_seconds,
            ),
        )
        if search_key
        else []
    )
    # The two ways of sending a message (spec of 2026-09-15). WhatsApp is
    # the installed application, asked for at every send; Telegram is the
    # user's own account, logged in once with `allie telegram login` -
    # without the three things that produces, the channel says so instead
    # of connecting. Closed in the `finally` like the player.
    whatsapp = WhatsApp()
    api_hash = load_api_key(HASH_ENTRY)
    session = load_api_key(SESSION_ENTRY)
    telegram = Telegram(
        book,
        client=TelethonClient(settings.telegram.api_id, api_hash, session)
        if settings.telegram.api_id and api_hash and session
        else None,
    )
    try:
        detector = SileroVAD()

        # Setup's verdict on the model, refreshed when it is a week old
        # (section 3.2, 2.6). Before the speech model: one session on
        # the network, and worth knowing about before two seconds of
        # loading are spent.
        await _model_checked(provider, settings, SettingsRepo(database), pack, screen)
        screen.starting()
        # The apps this machine can open, read once: a few seconds of
        # files and a PowerShell process, on a thread (2.2).
        catalog = await AppCatalog.load()
        # The recogniser hears the yes or no of the gate's window and nothing
        # else (D3, D10), so that is what it is told to expect - the pack's
        # own words, the ones the window accepts. Until 2026-09-22 it was
        # told the names of the installed applications instead, 120 tokens of
        # them, which cost the window most of a second of decode for a bias
        # towards words it never hears.
        whisper = LocalWhisper(prompt=confirm_prompt(pack))
        speech: LocalWhisper | GeminiSTT = whisper
        if settings.stt.provider == "gemini":
            # Google first, Whisper loaded behind it for the free tier's
            # three requests a minute and for the network. The key is
            # the entry the live model uses when it is Gemini; missing,
            # it is a sentence before anything slow is loaded.
            key = load_api_key("gemini")
            if not key:
                raise MissingAPIKeyError(
                    "no API key stored for 'gemini', which [stt] provider names - "
                    "run 'allie setup' to add one, or set provider = \"local\""
                )
            speech = GeminiSTT(key, model=settings.stt.model, fallback=whisper)
        # The program's own voice (D3, D4, D10, D32): the live model itself,
        # in the conversation's voice, reading from a session of its own -
        # the gate's questions, the reminders, the filler and the three
        # failure sentences. One voice, the assistant's; the sentences that
        # never change are kept on disk, and what cannot be read is put on
        # the screen instead.
        voice = LiveVoice(
            provider,
            model=live.model,
            voice=live.voice,
            language_code=pack.language_code,
            directory=data_dir() / "voice",
            on_unsaid=screen.notice,
        )
        # The Microsoft Store, asked last by `open_app` and installed from
        # by `install_app` - the one tool that changes what is on the
        # machine, and asks first (2026-09-13).
        store = store_tools.WingetStore()
        notes = NotesRepo(database)
        reminders = ReminderRepo(database)
        # The user's mailbox (3.3, 17 Sep 2026), opened afresh for each
        # question by the two tools below - or not set up, in which
        # case the tools say so and what to run.
        mailbox = _mailbox_of(settings, load_api_key(mail_tools.MAIL_ENTRY))
        # The tools on offer, by name, in one place. Every one of them
        # runs through the gate below and nowhere else (section 3.9).
        # `forget` and `install_app` are declared with the questions they
        # ask, in the pack's words.
        tools = ToolRegistry(
            [
                get_current_time,
                # The machine's own state and the weather (15 Sep 2026):
                # the two questions about the world that need no
                # application opened.
                system_status_for(Win32Machine()),
                weather_tools.get_weather_for(weather),
                # The catalogue answers first, the player second and the
                # Store last, so an installed application always wins
                # its own name.
                open_app_for(
                    catalog,
                    media=player.open_named,
                    store=store,
                    unknown_publisher=pack.say(
                        "unknown_publisher", system_tools.TEXT["unknown_publisher"]
                    ),
                ),
                open_url,
                # A question is looked up and answered (D29); the browser
                # opens only when the user asks to see the search.
                *looking_up,
                # The engine is the user's (`[web] search_url`).
                search_web_for(settings.web.search_url),
                # What was copied, and what a page says: both come back
                # inside the `<untrusted>` block the prompt explains.
                read_clipboard_for(),
                fetch_page_for(reader),
                # What is trending on X, and why (D30).
                *trending,
                # The user's document folders, answered from by a second
                # model (D35).
                *reading_documents,
                # The user's mail, read and never written, inside the
                # same block.
                mail_tools.read_latest_emails_for(mailbox),
                mail_tools.search_emails_for(mailbox),
                open_settings,
                media_control,
                # The volume as a number (D24): the keys above step it.
                set_volume_for(SystemVolume()),
                play_music_for(player),
                play_video_for(player),
                open_media_for(player),
                # The user's notes (4.1, 17 Sep 2026): kept as said,
                # found in any spelling, deleted only after the user has
                # heard which.
                notes_tools.add_note_for(notes),
                notes_tools.search_notes_for(notes),
                notes_tools.delete_note_for(
                    notes,
                    confirm_prompt=pack.say(
                        "note_delete_confirm", notes_tools.TEXT["note_delete_confirm"]
                    ),
                ),
                # Reminders (4.2): the row here, the saying by the
                # scheduler below, between turns.
                reminder_tools.create_reminder_for(reminders),
                reminder_tools.list_reminders_for(reminders),
                reminder_tools.cancel_reminder_for(
                    reminders,
                    confirm_prompt=pack.say(
                        "reminder_cancel_confirm",
                        reminder_tools.TEXT["reminder_cancel_confirm"],
                    ),
                ),
                memory_tools.remember_for(memory),
                memory_tools.forget_for(
                    memory,
                    confirm_prompt=pack.say("forget_confirm", memory_tools.TEXT["forget_confirm"]),
                ),
                store_tools.install_app_for(
                    catalog,
                    store,
                    confirm_prompt=pack.say(
                        "store_install_confirm", store_tools.TEXT["store_install_confirm"]
                    ),
                ),
                # One tool for both messaging apps; it asks first, in the
                # pack's words, and hears the contact, the app and the text.
                messaging_tools.send_message_for(
                    {
                        "whatsapp": messaging_tools.WhatsAppChannel(whatsapp, book),
                        "telegram": telegram,
                    },
                    default_app=settings.messaging.default_app,
                    confirm_prompt=pack.say(
                        "send_message_confirm",
                        messaging_tools.TEXT["send_message_confirm"],
                    ),
                ),
                # The owner's own, from %APPDATA%\allie\tools:
                # read here, through the same gate, never in the repository.
                *load_local_tools(),
            ]
        )
        # Loading Whisper takes seconds of four cores. Doing it now rather
        # than at the first question keeps the first yes or no from
        # waiting for it (item 1.6). The detector is a tenth of a second
        # beside it, and is loaded here for the same reason rather than
        # inside the first block of audio it is asked about - it is the
        # doorman now (D5), asked about every block.
        await speech.load()
        await detector.load()
        if wake_word is not None:
            await wake_word.load()

        # The one gate, built once and handed to the one place a tool is
        # run from: the tool round (plan.md 4.4 rule 3). A second gate
        # would be a second way to run a tool, which is the thing
        # section 3.9 forbids.
        gate = _gate(settings, tools, AuditRepo(database), limits=limits, pack=pack)
        runner = ToolRunner(tools, gate, limits)
        # The one announce queue (invariant 5) and the loop that feeds it
        # (invariant 7): the scheduler never sees the model, and the
        # state machine reads the queue only between turns.
        announcements = AnnounceQueue()
        scheduler = scheduler_runner.Scheduler(
            reminders,
            announcements,
            wording={key: pack.say(key, default) for key, default in scheduler_runner.TEXT.items()},
        )
        # The doorman and the stream (plan.md 4.1, D5, D6): the local
        # detector opens a session on speech, the microphone streams
        # while one is open, full or half duplex by the path it was
        # opened through (D18) - `barge_in = false` forces half.
        capture = LiveCapture(
            microphone=SystemMicrophone(device=device),
            endpoint=Endpoint(detector),
            barge_in=live.barge_in,
            on_level=screen.level,
            # Asleep behind the wake word until the phrase is heard (D21).
            wake=wake_word,
        )
        # The window's listen button is this switch (D20).
        if switch is not None:
            switch.target = capture.toggle
        # The icon of 4.3: the window's (D34), or on the terminal one of
        # this run's own when asked for - a second surface over the same
        # state, and a second hand on the same switch: its menu line is
        # the key's `toggle`, its "quit" is this task's cancel, and both
        # reach the loop through `call_soon_threadsafe`.
        own: tray_ui.Tray | None = None
        if icon is None and tray:
            own = icon = tray_ui.Tray(
                pack,
                loop=asyncio.get_running_loop(),
                on_toggle=capture.toggle,
                on_quit=_stopper(),
                settings_folder=config_dir(),
            )

        def session_config() -> SessionConfig:
            # Read at every open (plan.md 4.4): the prompt carries the
            # user's facts and the time, which move; the tools and the
            # `[live]` knobs do not, but one place is one place.
            return SessionConfig(
                model=live.model,
                voice=live.voice,
                system_prompt=_system_prompt(
                    memory,
                    pack,
                    web_search=live.web_search or search is not None,
                    folders=shelf.names() if reading_documents else (),
                ),
                tools=runner.specs(),
                transcripts=live.transcripts,
                language_code=pack.language_code,
                end_sensitivity=live.end_sensitivity,
                silence_ms=live.silence_ms,
                web_search=live.web_search,
                affective_dialog=live.affective_dialog,
                compress_context=live.compress_context,
            )

        assistant = LiveAssistant(
            capture=capture,
            provider=provider,
            session_config=session_config,
            tool_runner=runner,
            tts=voice,
            stt=speech,
            # The sound both ways goes to the screen's meter (D20).
            speaker=SystemSpeaker(on_level=screen.level),
            locale=pack,
            # Every turn's tokens, priced, to `usage_log`: what
            # `allie cost` reads and what the spending limits
            # are checked against.
            tracker=UsageTracker(
                UsageRepo(database),
                Pricing.load(),
                provider=live.provider,
                model=live.model,
                limits=limits,
            ),
            announcements=announcements,
            idle_close_seconds=live.idle_close_seconds,
            resume_minutes=live.resume_minutes,
            greeting=settings.wake.greeting,
            # What `look_up` and `x_trends` searched, for the screen (D29).
            searched=searched,
            on_state=screen.state if icon is None else _each(screen.state, icon.state),
            on_turn=_finished(screen),
            # The toggle's news goes to the state machine first - off is
            # an interruption - and to the screen after it.
            on_mode=screen.hands_free
            if icon is None
            else _each(screen.hands_free, icon.hands_free),
            # The session's opening and closing: the meter on the line
            # and the icon (plan.md 4.2), and at the close the level
            # the server heard the microphone at (D18).
            on_session=_session_told(screen, icon, capture),
        )
        if own is not None:
            own.start()
        ticking = asyncio.create_task(scheduler.run())
        try:
            await assistant.run()
        finally:
            ticking.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticking
            if own is not None:
                own.stop()
    finally:
        await player.aclose()
        await weather.aclose()
        await reader.aclose()
        await trends.aclose()
        await telegram.close()
        database.close()


def _system_prompt(
    memory: UserMemory,
    pack: Locale,
    *,
    web_search: bool = False,
    folders: Sequence[str] = (),
) -> str:
    """What the model is told at every session open (plan.md 4.5).

    The frozen rules first, byte for byte; then the pack's sentence naming
    the language the user speaks (D19 - measured 2026-09-18: without it a
    short "Saat kaç" is heard as Hindi), when the pack has one; then, when
    the session carries a search tool (D22) or `look_up` is on offer (D29),
    the sentence that says when to use it; then, when the user has document
    folders (D35), the rule with their names - both after the pack's rule
    and before the user's facts, so that the frozen bytes stay frozen for a
    session without them; then the user's facts (section 3.7); the time
    last of all, so that "yarın" is a date (4.2). `prompts.py` stays without
    an import, and no language is named in the code.
    """
    from allie.agent.prompts import DOCUMENTS_RULE, SEARCH_RULE, SYSTEM_PROMPT
    from allie.tools.reminders import current_time_line

    rules = SYSTEM_PROMPT
    if pack.user_language_rule:
        rules = f"{rules}\n\n{pack.user_language_rule}"
    if web_search:
        rules = f"{rules}\n\n{SEARCH_RULE}"
    if folders:
        listed = ", ".join(f'"{name}"' for name in folders)
        rules = f"{rules}\n\n{DOCUMENTS_RULE.format(folders=listed)}"
    return f"{memory.prompt(rules)}\n\n{current_time_line()}"


def _documents_shelf(documents: DocumentSettings) -> Shelf:
    """The user's document folders (D35). The default root is made, so
    that the user has somewhere to put them; a root the user wrote and that
    is not there is said in the log, not made - an empty folder would hide
    the typo."""
    from loguru import logger

    shelf = Shelf(documents.root())
    if documents.folder.strip():
        if not shelf.root.is_dir():
            logger.warning("[documents] folder {} does not exist", shelf.root)
    else:
        shelf.root.mkdir(parents=True, exist_ok=True)
    return shelf


def _session_told(
    screen: Screen, icon: Tray | None, capture: LiveCapture
) -> Callable[[bool], None]:
    """Who hears that a session opened or closed: the line's meter, the
    icon's, and - at the close - the level the microphone sent (D18), so
    that a quiet one is said on the screen once. `level_judged` rather than
    `level_dbfs`: on the raw kernel path the threshold was never measured
    and the sentence would be a false alarm."""

    def told(open: bool) -> None:
        screen.session(open)
        if icon is not None:
            icon.session(open)
        if not open:
            screen.microphone_level(capture.level_judged)

    return told


async def _model_checked(
    provider: LiveProvider,
    settings: Settings,
    verdicts: SettingsRepo,
    pack: Locale,
    screen: Screen,
) -> None:
    """Makes sure there is a verdict on the model that answers, and says so
    on screen when it is a bad one.

    Setup wrote one; a week later it is asked again here, because the
    provider may have changed what is behind the name (section 3.2). A model
    that fails is a warning and not a refusal to start: the user may have
    chosen it knowing, and it still answers questions. A provider that
    cannot be asked right now is left to the first turn, which has its own
    sentences for that (`app.py`) - and the verdict there was is kept.
    """
    from loguru import logger

    from allie.live import probe
    from allie.live.base import ProviderError

    provider_id, model = settings.live.provider, settings.live.model
    verdict = probe.remembered(verdicts, provider_id, model)
    if verdict is None:
        screen.checking_model()
        try:
            verdict = await probe.probe_tool_support(
                provider, model, question=pack.probe_question or probe.QUESTION
            )
        except ProviderError as refusal:
            logger.warning("the model could not be checked at startup: {why}", why=refusal)
            return
        probe.remember(verdicts, provider_id, model, verdict)
        logger.info(
            "probe {provider}:{model}: ok={ok}, first token {ms} ms",
            provider=provider_id,
            model=model,
            ok=verdict.ok,
            ms=None if verdict.first_token_ms is None else round(verdict.first_token_ms),
        )

    if not verdict.ok:
        screen.notice(pack.say("model_no_tools", TEXT["model_no_tools"]))


def _gate(
    settings: Settings, tools: ToolRegistry, audit: AuditRepo, *, limits: Limits, pack: Locale
) -> Dispatch:
    """The one permission gate, with everything it needs already in hand.

    What the loop gets is a function of the call alone; the registry, the
    audit rows, the user's `[tools]` settings, the look-back window of
    section 3.11 and the gate's own sentences in the user's language are
    bound here, so that `agent/core.py` never imports `policy.py` and a
    test can hand it a fake. Who to ask is not bound here: it comes with
    each turn, because it is the state machine's own microphone, and the
    state machine is built after the gate.
    """
    from allie.agent import policy

    wording = {key: pack.say(key, default) for key, default in policy.TEXT.items()}

    async def dispatch(call: ToolCall, *, turn_id: str, confirm: Confirm) -> str:
        return await policy.dispatch(
            call,
            turn_id=turn_id,
            registry=tools,
            confirm=confirm,
            unblocked=settings.tools.unblocked,
            audit=audit,
            wording=wording,
            duplicate_window=limits.duplicate_window_sec,
        )

    return dispatch


# --------------------------------------------------------------------------
# assistant cost
# --------------------------------------------------------------------------


def _cost() -> int:
    """What the assistant has spent, today and this month, by model (section 6).

    Read back from `usage_log`, where every turn was written with its price
    at the time; nothing is recomputed, so a price edited today does not
    rewrite last week. A model with no price in `pricing.toml` is counted
    and not billed, and the report says so rather than show a zero.
    """
    from rich.console import Console

    from allie.store.db import open_database
    from allie.store.repos import UsageRepo
    from allie.usage.tracker import start_of_day, start_of_month

    pack = locales.load(load_settings().locale.code)
    said = {key: pack.say(key, default) for key, default in TEXT.items()}
    console = Console()

    database = open_database()
    try:
        usage = UsageRepo(database)
        now = time.time()
        ever = usage.by_model_since(0)
        today = usage.by_model_since(start_of_day(now))
        month = usage.by_model_since(start_of_month(now))
    finally:
        database.close()

    if not ever:
        console.print(said["cost_none"], markup=False, highlight=False)
        return _OK

    console.print(_cost_table(said["cost_today"], today, said))
    console.print(_cost_table(said["cost_month"], month, said))
    for row in month:
        if row.unpriced:
            line = said["cost_unpriced"].format(count=row.unpriced, model=_model(row))
            console.print(line, markup=False, highlight=False)
    return _OK


def _cost_table(title: str, rows: Sequence[ModelUsage], said: Mapping[str, str]) -> Table:
    """One period: a row per model, and a total."""
    from rich.table import Table

    table = Table(title=title, title_justify="left")
    table.add_column(said["cost_model"])
    table.add_column(said["cost_turns"], justify="right")
    table.add_column(said["cost_tokens"], justify="right")
    table.add_column(said["cost_spent"], justify="right")
    for row in rows:
        table.add_row(
            _model(row),
            str(row.turns),
            f"{row.input_tokens} / {row.output_tokens} / {row.cached_tokens}",
            _dollars(row.cost_usd),
        )

    priced = [row.cost_usd for row in rows if row.cost_usd is not None]
    table.add_row(
        said["cost_total"],
        str(sum(row.turns for row in rows)),
        " / ".join(
            str(sum(tokens))
            for tokens in (
                (row.input_tokens for row in rows),
                (row.output_tokens for row in rows),
                (row.cached_tokens for row in rows),
            )
        ),
        _dollars(sum(priced) if priced else None),
        style="bold",
    )
    return table


def _model(row: ModelUsage) -> str:
    """`gemini:gemini-3.5-flash-lite` - the way `config.toml` writes it."""
    return f"{row.provider}:{row.model}"


def _dollars(amount: float | None) -> str:
    """`$0.0004`, or a question mark for turns whose price is not known."""
    return "?" if amount is None else f"${amount:.4f}"


def _finished(screen: Screen) -> Callable[[Turn], None]:
    """What happens to a turn once it is over: the numbers to the log, the
    words to the screen. Neither one keeps both (`logs.py`)."""
    from allie.logs import log_turn

    def turn(finished: Turn) -> None:
        log_turn(finished)
        screen.turn(finished)

    return turn


def _mailbox_of(settings: Settings, password: str | None) -> Callable[[], Mailbox] | None:
    """How the mail tools reach the mailbox `[mail]` names: a fresh
    connection per question (`tools/mail.py`), or nothing when the table
    is empty or the password was never stored."""
    from allie.tools.mail import ImapMailbox

    mail = settings.mail
    if not (mail.host.strip() and mail.user.strip() and password):
        return None
    return lambda: ImapMailbox(mail.host, mail.port, mail.user, password, mail.mailbox)


def _each[T](*listeners: Callable[[T], None]) -> Callable[[T], None]:
    """One listener made of several, told in the order given: the screen
    first, the tray after it (4.3)."""

    def tell(value: T) -> None:
        for listener in listeners:
            listener(value)

    return tell


def _stopper() -> Callable[[], None]:
    """Ends the task this is called from, when called later - from the
    loop, where the tray's "quit" is posted to. Cancelling is what Ctrl+C
    does from the outside, so the ending is the same one (`_run`)."""
    task = asyncio.current_task()

    def stop() -> None:
        if task is not None:
            task.cancel()

    return stop


if __name__ == "__main__":
    raise SystemExit(main())
