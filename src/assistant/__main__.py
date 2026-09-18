"""Command line entry point - and the composition root of the program.

`setup` asks the questions of item 1.4, the microphone among them since
2026-09-15; `mic` asks that one again on its own. `run` is item 1.11: it
puts the pieces together, hands them to the state machine, and shows the one
line of terminal that is the entire interface until the tray icon of phase
4.2. `cost` is 2.4: what the turns cost, read back from `usage_log`. `purge --all`
is 4.6: everything the machine recorded, listed and then deleted after the
word `yes`. `doctor` arrives in phase 3 (design.md section 8).

This is the only file that knows the concrete names: which tools are on
offer, which gate runs them, where the audit rows go. `agent/core.py` sees a
registry and a gate, `agent/policy.py` sees a registry and a repository, and
neither knows what the other is called - which is what lets a test hand
either of them a fake.

Three things about `run` are decisions rather than plumbing.

**Everything heavy is imported inside the function that needs it.**
`live-assistant --help` should not load PortAudio, a speech model and a
vendor SDK to print four lines of help - and `run` should not load the
wizard's prompt library it will never show.

**`run` does not run yet** (plan.md section 7, L0). The composition root of
the old pipeline - `_talk`, which built the recogniser, the loop and the
state machine - is gone with them; the live one arrives in L1.6 and is
wired from the same helpers below (`_gate`, `_model_checked`, `_finished`,
`_mailbox_of`, `_each`, `_stopper`), which stay. Until then `run` says so
in one sentence and exits 2.

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

from assistant import __version__, locales
from assistant.config import (
    Settings,
    config_path,
    is_configured,
    load_api_key,
    load_settings,
)

if TYPE_CHECKING:
    from rich.table import Table

    from assistant.agent.core import Confirm, Dispatch
    from assistant.agent.limits import Limits
    from assistant.app import Turn
    from assistant.live.base import LLMProvider, ToolCall
    from assistant.locales import Locale
    from assistant.store.repos import AuditRepo, ModelUsage, SettingsRepo
    from assistant.tools.mail import Mailbox
    from assistant.tools.registry import ToolRegistry
    from assistant.ui.status import StatusLine

__all__ = ["BUILTIN_TOOLS", "DOCTOR_TEXT", "TEXT", "build_parser", "main", "use_utf8"]

_OK = 0
_GAVE_UP = 1
# `run` before the live loop exists (plan.md L0): not "not set up", which is
# the user's to fix, but "not built yet", which is not.
_NOT_WIRED = 2

# The last link of the chain of section 3.12, as in every other module that
# says something: the pack answers first, and these are what is left if none
# does. Keys are unique across the project - `test_locales.py` checks.
TEXT: dict[str, str] = {
    "not_set_up": "Nothing is set up yet. Run 'live-assistant setup' first.",
    "cannot_start": "The assistant cannot start: {problem}",
    "stopped": "Stopped.",
    # Until the live loop is wired (plan.md L1.6): what `run` says instead
    # of listening. Every other command works.
    "run_not_wired": (
        "The live loop is not wired yet: this build does not talk. "
        "setup, doctor, cost, purge and autostart work."
    ),
    # The probe of 2.6, run again at startup when its verdict is a week
    # old: a model that fails is warned about and used anyway, because the
    # user may have chosen it knowing (section 3.2).
    "model_no_tools": (
        "The model does not call tools, and most of what the assistant does depends on that. "
        "Run 'live-assistant setup' to choose another."
    ),
    # `live-assistant cost` (section 6): two small tables, today and this month,
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
    # `live-assistant purge --all` (4.6): what will go is listed first, what
    # stays is said, and nothing is deleted before the word is typed.
    "purge_will_delete": "This will delete everything the assistant recorded:",
    "purge_file": "  file: {path}",
    "purge_secret": "  Credential Manager entry: {name}",
    "purge_keeps": "config.toml and contacts.toml stay: you wrote those.",
    "purge_confirm": "Type yes to delete all of it:",
    "purge_cancelled": "Nothing was deleted.",
    "purge_done": "Deleted {count} items.",
    "purge_nothing": "There is nothing to delete.",
    # `live-assistant autostart on | off | status` (4.2): the `Run` key of the
    # user's own sign-in, and what is written there.
    "autostart_on": "The assistant will start when you sign in, as: {command}",
    "autostart_off": "The assistant will no longer start when you sign in.",
    "autostart_status_on": "Starts when you sign in, as: {command}",
    "autostart_status_off": (
        "Does not start when you sign in. 'live-assistant autostart on' changes that."
    ),
}

# The one word `purge --all` accepts, in any letter case. The question
# names it, so it is the same in every language rather than the pack's
# yes-word - a typed answer to a question about deleting everything should
# not depend on which pack is loaded.
PURGE_WORD = "yes"

# The tools `run` puts on offer, by name and in order, so that `doctor` can
# count them without building them. `test_cli.py` checks that the registry
# `run` builds is this list followed by the user's own.
BUILTIN_TOOLS: tuple[str, ...] = (
    "get_current_time",
    "system_status",
    "get_weather",
    "open_app",
    "open_url",
    "search_web",
    "read_clipboard",
    "fetch_page",
    "read_latest_emails",
    "search_emails",
    "open_settings",
    "media_control",
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

# What `live-assistant doctor` says, line by line. Labels rather than sentences,
# and every value written by the code: a path, a number, a name. Never a
# key - the report says whether one is stored, and nothing else about it.
DOCTOR_TEXT: dict[str, str] = {
    "doctor_pack": "language pack: {name} ({code})",
    "doctor_who_answers": "Who answers",
    "doctor_model": "  model: {model} ({provider})",
    "doctor_key_stored": "  API key: stored in the Credential Manager",
    "doctor_key_missing": "  API key: none stored - run 'live-assistant setup'",
    "doctor_key_not_needed": "  API key: not needed",
    "doctor_verdict_ok": "  tool calling: passed, checked {days} days ago",
    "doctor_verdict_failed": (
        "  tool calling: FAILED, checked {days} days ago - most tools will not work"
    ),
    "doctor_verdict_none": "  tool calling: not checked yet (checked at the next start)",
    "doctor_stt_local": "  recogniser: Whisper '{size}', on this machine",
    "doctor_stt_gemini": "  recogniser: Google ({model}), Whisper '{size}' behind it",
    "doctor_tts_sapi": "  voice: Windows ({voice})",
    "doctor_tts_gemini": "  voice: Google ({model}), Windows behind it",
    "doctor_leaves": "What leaves this machine",
    "doctor_voice_stays": "  your voice: stays here",
    "doctor_voice_goes": "  your voice: goes to Google (the recogniser)",
    "doctor_text_goes": "  what you said, as text: goes to {provider}",
    "doctor_content_goes": "  pages, clipboard text and mail you ask about: go to {provider}",
    "doctor_answer_stays": "  what the assistant says: stays here",
    "doctor_answer_goes": "  what the assistant says: goes to Google (the voice)",
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
    "doctor_mail_none": "  mail: not set up ('live-assistant mail login')",
    "doctor_telegram": "  telegram: logged in (api_id {api_id})",
    "doctor_telegram_none": "  telegram: not set up ('live-assistant telegram login')",
    "doctor_autostart_on": "  starts when you sign in: yes ({command})",
    "doctor_autostart_off": "  starts when you sign in: no ('live-assistant autostart on')",
}


def build_parser() -> argparse.ArgumentParser:
    """Builds the top level argument parser."""
    parser = argparse.ArgumentParser(
        prog="live-assistant",
        description=(
            "A voice assistant on live speech-to-speech models, "
            "running on your own API key and language."
        ),
    )
    parser.add_argument("--version", action="version", version=f"live-assistant {__version__}")

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
            "'live-assistant mic' changes it for good."
        ),
    )
    run.add_argument(
        "--tray",
        action="store_true",
        help=(
            "Also show an icon in the notification area: the state, a switch for "
            "listening, the settings folder, quit. The terminal stays."
        ),
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
        from assistant import setup_wizard

        # Looked up on the module rather than imported by name so the tests can
        # stand in for it; the wizard itself opens a prompt and would hang.
        return asyncio.run(setup_wizard.run_setup(setup_wizard.TerminalPrompter()))

    if args.command == "mic":
        from assistant import setup_wizard

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

    return _run(device=args.device, tray=args.tray)


def use_utf8(stream: object) -> None:
    """Asks a stream to stop encoding in the machine's legacy code page.

    A console Windows owns is already fine; a redirected one - `live-assistant run >
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
    """`live-assistant telegram login`: the questions of `messaging/telegram.py`,
    once, and the three things they produce stored where each belongs -
    the `api_id` in `config.toml`, the hash and the session in the
    Credential Manager (section 10). The settings are rewritten from what
    was loaded, so nothing else in the file changes."""
    from assistant import setup_wizard
    from assistant.config import TelegramSettings, save_settings, store_api_key
    from assistant.messaging import telegram

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
    """`live-assistant doctor` (3.5): what this installation is, in one screen.

    Who answers and through what; which of the user's data leaves the
    machine and for where - the sentence the README makes, checked against
    the settings actually in force; where every file is; how many tools
    there are; the limits. Everything is read, nothing is loaded and nothing
    is connected to: no Whisper, no microphone, no request to any provider.
    A key is reported as stored or not, and never shown, the way
    `describe_myself` of section 10 has it: named fields, not a dump.
    """
    from rich.console import Console

    from assistant import autostart
    from assistant.live.probe import probe_key, remembered
    from assistant.live.registry import load_catalog
    from assistant.logs import log_path
    from assistant.messaging.contacts import AddressBook, ContactsFileError, contacts_path
    from assistant.messaging.telegram import HASH_ENTRY, SESSION_ENTRY
    from assistant.store.db import database_path, open_database
    from assistant.store.memory import MemoryFileError, UserMemory, memory_path
    from assistant.store.repos import SettingsRepo
    from assistant.stt.local_whisper import DEFAULT_MODEL_SIZE
    from assistant.tools.local import load_local_tools, local_tools_dir
    from assistant.tools.mail import MAIL_ENTRY

    settings = load_settings()
    ready = is_configured()
    pack = locales.load(settings.locale.code if ready else locales.system_code())
    said = {key: pack.say(key, default) for key, default in {**TEXT, **DOCTOR_TEXT}.items()}
    console = Console()
    lines: list[str] = []

    def say(key: str, **fields: object) -> None:
        lines.append(said[key].format(**fields))

    lines.append(f"live-assistant {__version__}")
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
    if settings.tts.provider == "gemini":
        say("doctor_tts_gemini", model=settings.tts.model)
    else:
        say("doctor_tts_sapi", voice=pack.voice("sapi") or "the default voice")

    # What leaves this machine.
    lines.append("")
    say("doctor_leaves")
    say("doctor_voice_goes" if settings.stt.provider == "gemini" else "doctor_voice_stays")
    say("doctor_text_goes", provider=provider_name)
    say("doctor_content_goes", provider=provider_name)
    say("doctor_answer_goes" if settings.tts.provider == "gemini" else "doctor_answer_stays")
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
    """`live-assistant mail login`: the questions of `tools/mail.py`, once, and
    what they produce stored where each belongs - the server and the
    address in `config.toml`, the app password in the Credential Manager
    under `mail` (section 10). The settings are rewritten from what was
    loaded, so nothing else in the file changes."""
    from assistant import setup_wizard
    from assistant.config import MailSettings, save_settings, store_api_key
    from assistant.tools import mail

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
    """`live-assistant purge --all`: what the machine recorded, gone (section 3.7, 4.6).

    The database with its audit rows, notes and reminders; the memory file;
    the logs; and every key and session in the Credential Manager. What is
    there is listed first - by path and by entry name, never by value - and
    nothing is deleted before `yes` is typed. `config.toml` and
    `contacts.toml` stay: the user wrote those by hand, and a file edited
    calmly is not "what the assistant recorded".
    """
    from assistant import setup_wizard
    from assistant.config import delete_api_key

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
    from assistant.logs import LOG_FILE, log_path
    from assistant.store.db import database_path
    from assistant.store.memory import memory_path

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
    from assistant.live.registry import load_catalog
    from assistant.messaging.telegram import HASH_ENTRY, SESSION_ENTRY
    from assistant.tools.mail import MAIL_ENTRY

    return [*load_catalog(), HASH_ENTRY, SESSION_ENTRY, MAIL_ENTRY]


def _autostart(action: str) -> int:
    """`live-assistant autostart on | off | status` (4.2): the user's own `Run`
    key, written with the assistant as installed here and `run --tray`,
    cleared, or read back. No administrator, no service (`autostart.py`)."""
    from rich.console import Console

    from assistant import autostart

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


def _run(*, device: str | None = None, tray: bool = False) -> int:
    """Says why the assistant cannot start - and, in this build, that it
    cannot yet (plan.md L0).

    What is checked is what was always checked first: has setup run. After
    that the old pipeline built its pieces and handed them to the state
    machine; the live loop that replaces it is task L1.6, and until then
    this is one sentence and exit code 2 - distinct from the 1 of "not set
    up", which is the user's to fix. `device` and `tray` are the flags the
    loop will take; they are accepted so that the command line does not
    change under the tests and the autostart entry.
    """
    from rich.console import Console

    settings = load_settings()
    ready = is_configured()

    # Before setup there is no chosen language to read this in, so the machine's
    # own is the best guess available - the same one the wizard starts with.
    pack = locales.load(settings.locale.code if ready else locales.system_code())
    console = Console()
    said = {key: pack.say(key, default) for key, default in TEXT.items()}

    if not ready:
        console.print(said["not_set_up"], markup=False, highlight=False)
        return _GAVE_UP

    console.print(said["run_not_wired"], markup=False, highlight=False)
    return _NOT_WIRED


async def _model_checked(
    provider: LLMProvider,
    settings: Settings,
    verdicts: SettingsRepo,
    pack: Locale,
    screen: StatusLine,
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

    from assistant.live import probe
    from assistant.live.base import ProviderError

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
    from assistant.agent import policy

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

    from assistant.store.db import open_database
    from assistant.store.repos import UsageRepo
    from assistant.usage.tracker import start_of_day, start_of_month

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


def _finished(screen: StatusLine) -> Callable[[Turn], None]:
    """What happens to a turn once it is over: the numbers to the log, the
    words to the screen. Neither one keeps both (`logs.py`)."""
    from assistant.logs import log_turn

    def turn(finished: Turn) -> None:
        log_turn(finished)
        screen.turn(finished)

    return turn


def _mailbox_of(settings: Settings, password: str | None) -> Callable[[], Mailbox] | None:
    """How the mail tools reach the mailbox `[mail]` names: a fresh
    connection per question (`tools/mail.py`), or nothing when the table
    is empty or the password was never stored."""
    from assistant.tools.mail import ImapMailbox

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
