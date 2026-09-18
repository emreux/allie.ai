"""`live-assistant setup` - the few questions phase 1 asks (design.md section 3.3).

Provider, key, model, the language the assistant speaks, and the microphone
it listens through. The fourteen step wizard of section 3.3 - device tests,
tool probe, latency measurement, a fallback model - is phase 4.5; what is
here is the smallest thing that can produce a working `config.toml`.

Two rules shape the code more than the questions do.

**Nothing is written until the wizard finishes.** The key is checked against
the provider before it is kept, and both the key and the settings are stored in
the last step. A run that was abandoned, or a key that turned out to be
revoked, leaves the machine exactly as it was found.

**No user-facing sentence lives in this module.** The wizard asks for a
question by key; the `Prompter` turns that key into words, taking them from the
locale pack and falling back to `TEXT` below - English, which section 3.12
names as the end of the chain. That is also why the tests can script a wizard
run without repeating a single sentence, and why adding a language changes no
line of this file.

**A model is not taken at its word (2.6).** Every model can be asked a
question; not every one can be asked to do something, and the one that
cannot fails silently, in prose, at two in the morning. So the model the
user picks is sent one request with one tool before it is accepted
(`live/probe.py`, section 3.2), and a model that does not call the tool is
not taken - the list is offered again. The verdict on the model that was
taken goes to the `settings` table, which makes this the first thing to
touch the database on a fresh machine.

**Two providers are questions about a catalogue field (2.7).** One that
needs no key - a local Ollama - is not asked for one: the server is asked
whether it answers, and that is all. One that has no address of its own -
`custom`, any OpenAI-compatible server - is asked where it is, and the
address goes to `config.toml` beside the model, where a text editor can
reach it. Neither question names an adapter here: the registry says which
entries need an address, the catalogue says which need a key.

**The microphone is a list, not a name to type.** PortAudio's table, every
input once per host API, with "whatever Windows has chosen" on top and stored
as the empty string that has meant the default since phase 1. `live-assistant mic`
asks this one question again on its own - the day a headset comes out - and
rewrites `[audio]` alone, so that nobody edits `config.toml` by hand for it.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import questionary
from rich.console import Console

from assistant import locales
from assistant.audio.capture import Microphones, available_microphones
from assistant.config import (
    AudioSettings,
    LiveSettings,
    LocaleSettings,
    Settings,
    config_path,
    load_api_key,
    load_settings,
    save_settings,
    store_api_key,
)
from assistant.live import probe
from assistant.live.base import LLMProvider, ModelInfo, ProviderError
from assistant.live.registry import (
    ADAPTERS,
    ProviderEntry,
    create_provider,
    load_catalog,
    needs_base_url,
)
from assistant.store.db import open_database
from assistant.store.repos import SettingsRepo

__all__ = [
    "TEXT",
    "Option",
    "Prompter",
    "TerminalPrompter",
    "run_microphone_setup",
    "run_setup",
    "wording",
]

_OK = 0
_GAVE_UP = 1


@dataclass(frozen=True, slots=True)
class Option:
    """One answer to a question: what gets stored, and what the user reads.

    The label is data - a provider's display name, a model's name, a language's
    name in that language - so it is not part of the text table.
    """

    value: str
    label: str


class Prompter(Protocol):
    """Everything the wizard needs from a terminal, and nothing more.

    A question is named, not worded. Keeping the wording on this side of the
    line is what lets the wizard be tested without a screen and translated
    without being rewritten.
    """

    def say(self, key: str, **fields: object) -> None: ...

    async def choose(self, key: str, options: Sequence[Option]) -> str | None:
        """Returns the chosen value, or `None` if the user walked away."""
        ...

    async def secret(self, key: str) -> str | None:
        """Reads a line without echoing it, or `None` if the user walked away."""
        ...

    async def ask(self, key: str) -> str | None:
        """Reads a line the user can see - an address, not a secret - or
        `None` if the user walked away."""
        ...


# The last link of the fallback chain of section 3.12: what the wizard says
# when no locale pack offers a translation. English lives here, beside the code
# that says it, rather than in `locales/en.toml` - one copy cannot drift from
# the other.
TEXT: dict[str, str] = {
    "welcome": "The assistant answers through an AI provider, using your own API key.",
    "provider": "Which provider do you want to use?",
    "only_provider": "Provider: {name} - the only one this version can talk to.",
    "no_provider": "This version cannot build any provider in the catalogue.",
    "locale": "Which language should the assistant speak?",
    "key_url": "You can create a key at {url}",
    "api_key": "Paste your API key (nothing is shown as you type)",
    "api_key_keep": "Paste your API key, or press Enter to keep the one already stored",
    "key_needed": "This provider needs a key before it will answer.",
    "checking_key": "Checking the key...",
    "base_url": "The server's address - the OpenAI-compatible base URL, ending in /v1",
    "checking_server": "Checking that the server answers...",
    "bad_key": "That key did not work - mistyped, revoked, or out of credit.",
    "provider_unreachable": (
        "The provider could not be reached. Check the connection and try again."
    ),
    "loading_models": "Asking which models the key can reach...",
    "no_models": "The key works, but it reaches no model. Check the provider's console.",
    "model": "Which model should answer?",
    "probing_tools": "Checking whether the model calls tools...",
    "tools_ok": "The model calls tools - first token in {ms} ms.",
    "tools_failed": (
        "This model does not call tools, and most of what the assistant does depends on that. "
        "Choose another model."
    ),
    "probe_refused": (
        "The model could not be tested: {problem}. Choose another model, or try again."
    ),
    "microphone": "Which microphone should the assistant listen through?",
    "windows_microphone": "Whatever Windows has chosen (right now: {name})",
    "no_microphones": "No microphone was found.",
    "microphone_saved": "Microphone: {microphone}. Settings: {path}",
    "saved": "Ready. Settings: {path} - the key itself is in the Windows Credential Manager.",
    "cancelled": "Setup cancelled. Nothing was changed.",
}


def wording() -> dict[str, str]:
    """What the wizard says, in the language the user is likeliest to read.

    The language question decides what the *assistant* speaks from then on,
    which is no help in wording the question: the wizard has to ask before it
    has an answer. So it uses the answer from last time when setup has run
    before - changing the model should not mean reading English again - and
    Windows' own language on the very first run. When neither names a language
    anybody has translated, this is `TEXT` unchanged.
    """
    pack = locales.load(_chosen_before() or locales.system_code())
    return {key: pack.say(key, default) for key, default in TEXT.items()}


def _chosen_before() -> str | None:
    """The language chosen the last time setup ran, if it ever ran."""
    return load_settings().locale.code if config_path().is_file() else None


def _languages() -> list[Option]:
    """Every locale pack there is, each named in its own language."""
    return [Option(pack.code, pack.name) for pack in locales.available()]


class _WalkedAwayError(Exception):
    """The user pressed Ctrl+C, or answered nothing to a question with no default."""


async def run_setup(
    prompter: Prompter,
    *,
    catalog: Mapping[str, ProviderEntry] | None = None,
    database: sqlite3.Connection | None = None,
    microphones: Microphones | None = None,
) -> int:
    """Asks the questions, then writes the answers. Returns a process exit code.

    `database` is where the verdict on the model goes; left out, the
    machine's own is opened for it at the end, and closed again.
    `microphones` is what the last question offers; left out, PortAudio is
    asked.
    """
    try:
        return await _ask(prompter, catalog, database, microphones)
    except _WalkedAwayError:
        prompter.say("cancelled")
        return _GAVE_UP


async def run_microphone_setup(
    prompter: Prompter, *, microphones: Microphones | None = None
) -> int:
    """`live-assistant mic`: the microphone question on its own.

    Rewrites `[audio]` and leaves every other table as the file has it, so
    that switching to a headset is one command rather than a text editor.
    """
    found = _found(microphones)
    if not found.devices:
        prompter.say("no_microphones")
        return _GAVE_UP
    options = _microphone_options(found)
    try:
        chosen = _answered(await prompter.choose("microphone", options))
    except _WalkedAwayError:
        prompter.say("cancelled")
        return _GAVE_UP

    path = save_settings(Settings(audio=AudioSettings(input_device=chosen)))
    label = next(option.label for option in options if option.value == chosen)
    prompter.say("microphone_saved", microphone=label, path=path)
    return _OK


async def _ask(
    prompter: Prompter,
    catalog: Mapping[str, ProviderEntry] | None,
    database: sqlite3.Connection | None,
    microphones: Microphones | None,
) -> int:
    entries = load_catalog() if catalog is None else catalog
    buildable = {
        provider_id: entry for provider_id, entry in entries.items() if entry.adapter in ADAPTERS
    }
    if not buildable:
        # A catalogue listing only adapters from later phases. Worth its own
        # sentence: the user has done nothing wrong and retrying will not help.
        prompter.say("no_provider")
        return _GAVE_UP

    prompter.say("welcome")
    provider_id, entry = await _pick_provider(prompter, buildable)
    locale = _answered(await prompter.choose("locale", _languages()))
    base_url = await _address(prompter, entry)

    if entry.requires_key:
        if entry.key_url is not None:
            prompter.say("key_url", url=entry.key_url)
        provider, api_key = await _working_key(prompter, provider_id, entries, base_url=base_url)
    else:
        api_key = ""
        answering = await _answering_server(prompter, provider_id, entries, base_url=base_url)
        if answering is None:
            return _GAVE_UP
        provider = answering

    prompter.say("loading_models")
    models = await provider.list_models()
    if not models:
        prompter.say("no_models")
        return _GAVE_UP
    model, verdict = await _model_that_calls_tools(
        prompter, provider, models, question=_probe_question(locale)
    )
    input_device = await _pick_microphone(prompter, _found(microphones))

    # Everything above could still be abandoned; from here it is written down.
    if api_key:
        store_api_key(provider_id, api_key)
    path = save_settings(
        Settings(
            live=LiveSettings(primary=f"{provider_id}:{model}", base_url=base_url or ""),
            locale=LocaleSettings(code=locale),
            audio=AudioSettings(input_device=input_device),
        )
    )
    _remember(database, provider_id, model, verdict)
    prompter.say("saved", path=path)
    return _OK


async def _pick_provider(
    prompter: Prompter, buildable: Mapping[str, ProviderEntry]
) -> tuple[str, ProviderEntry]:
    """Asks which provider - unless there is only one, which is no question."""
    if len(buildable) == 1:
        provider_id, entry = next(iter(buildable.items()))
        prompter.say("only_provider", name=entry.display_name)
        return provider_id, entry

    options = [Option(provider_id, e.display_name) for provider_id, e in buildable.items()]
    chosen = _answered(await prompter.choose("provider", options))
    return chosen, buildable[chosen]


async def _address(prompter: Prompter, entry: ProviderEntry) -> str | None:
    """Where the server is, for an entry the catalogue gives no address for
    (`custom`); `None` for every other, whose address is in the file. An
    empty answer is asked again - there is no default to fall back on."""
    if not needs_base_url(entry):
        return None

    while True:
        typed = _answered(await prompter.ask("base_url")).strip()
        if typed:
            return typed


async def _answering_server(
    prompter: Prompter,
    provider_id: str,
    entries: Mapping[str, ProviderEntry],
    *,
    base_url: str | None,
) -> LLMProvider | None:
    """A provider that needs no key, checked once: does the server answer?

    There is no key to ask for again, so a server that cannot be reached
    ends the wizard with the sentence that says so - the user starts the
    server and comes back. A server that turns out to want a key after all
    is told about in those words; the catalogue said it would not.
    """
    prompter.say("checking_server")
    provider = create_provider(provider_id, api_key="", catalog=entries, base_url=base_url)
    try:
        accepted = await provider.validate_credentials()
    except ProviderError:
        prompter.say("provider_unreachable")
        return None
    if not accepted:
        prompter.say("key_needed")
        return None
    return provider


async def _working_key(
    prompter: Prompter,
    provider_id: str,
    entries: Mapping[str, ProviderEntry],
    *,
    base_url: str | None = None,
) -> tuple[LLMProvider, str]:
    """Asks for a key until the provider accepts one (section 3.3).

    A key that fails is not stored, not retried and not silently swapped for a
    fallback: mistyped, revoked and out of credit all mean the same thing to
    the user, and all three are fixed by pasting a different key.

    A provider that could not be reached is a different sentence. The key has
    not been proved dead - nothing has been proved - so a stored one is still
    offered on the next round, and the user is told to look at the connection
    rather than at the provider's console.
    """
    stored = load_api_key(provider_id)

    while True:
        typed = _answered(await prompter.secret("api_key_keep" if stored else "api_key"))
        api_key = typed or stored or ""
        if not api_key:
            prompter.say("key_needed")
            continue

        prompter.say("checking_key")
        provider = create_provider(provider_id, api_key=api_key, catalog=entries, base_url=base_url)
        try:
            accepted = await provider.validate_credentials()
        except ProviderError:
            prompter.say("provider_unreachable")
            continue
        if accepted:
            return provider, api_key

        prompter.say("bad_key")
        # Whatever was in the vault has just been proved useless, so it stops
        # being offered - otherwise Enter would retry the same dead key.
        stored = None


async def _model_that_calls_tools(
    prompter: Prompter, provider: LLMProvider, models: Sequence[ModelInfo], *, question: str
) -> tuple[str, probe.ProbeResult]:
    """Offers the models until one is chosen that passes the probe (section 3.2).

    The probe is one request: the question, and the one canonical tool. A
    model that answers in prose is said to have failed and the list is
    offered again - it is not stored, not marked "chat only", not taken
    with a warning, because everything the assistant does from 2.1 on
    depends on the answer being a call. A provider that refuses the
    request has said nothing about the model, so that is a different
    sentence with the provider's own words in it, and the list again.
    """
    options = [_offer(m) for m in models]

    while True:
        model = _answered(await prompter.choose("model", options))
        prompter.say("probing_tools")
        try:
            verdict = await probe.probe_tool_support(provider, model, question=question)
        except ProviderError as refusal:
            prompter.say("probe_refused", problem=refusal)
            continue

        if verdict.ok:
            prompter.say("tools_ok", ms=_whole(verdict.first_token_ms))
            return model, verdict
        prompter.say("tools_failed")


async def _pick_microphone(prompter: Prompter, found: Microphones) -> str:
    """The `[audio] input_device` line - or nothing, for a machine with no
    microphone yet: setup goes on, and the default is what it listens through
    once something is plugged in."""
    if not found.devices:
        prompter.say("no_microphones")
        return ""
    return _answered(await prompter.choose("microphone", _microphone_options(found)))


def _found(microphones: Microphones | None) -> Microphones:
    return available_microphones() if microphones is None else microphones


def _microphone_options(found: Microphones) -> list[Option]:
    """Windows' own choice first, then PortAudio's table line by line.

    The first label is the one option worded rather than named, so it is
    the one place the wizard reads its own wording: the device it stands
    for changes with every headset, and the label says which it is today.
    """
    current = " ".join((found.default or "?").split())
    options = [Option("", wording()["windows_microphone"].format(name=current))]
    options.extend(Option(device.setting, device.label) for device in found.devices)
    return options


def _probe_question(locale: str) -> str:
    """The question in the language just chosen, or the English beside the
    code: the chain of section 3.12, for a sentence the model reads."""
    return locales.load(locale).probe_question or probe.QUESTION


def _whole(milliseconds: float | None) -> str:
    """`812`, for a number that is shown once and not calculated with."""
    return "?" if milliseconds is None else f"{milliseconds:.0f}"


def _remember(
    database: sqlite3.Connection | None, provider_id: str, model: str, verdict: probe.ProbeResult
) -> None:
    """Writes the verdict to `settings`, so that `live-assistant run` need not ask
    again for a week. On a fresh machine this is the first thing to touch
    the database, which is why the wizard opens it - and closes it - here."""
    connection = open_database() if database is None else database
    try:
        probe.remember(SettingsRepo(connection), provider_id, model, verdict)
    finally:
        if database is None:
            connection.close()


def _offer(model: ModelInfo) -> Option:
    # The id is what lands in `config.toml`, so it is shown next to the name
    # rather than hidden behind it.
    label = model.display_name
    if model.id != model.display_name:
        label = f"{model.display_name}  ({model.id})"
    return Option(model.id, label)


def _answered(value: str | None) -> str:
    if value is None:
        raise _WalkedAwayError
    return value


class TerminalPrompter:
    """The real terminal: `rich` for what is said, `questionary` for answers.

    Asked with `ask_async`, never with `ask`. The wizard checks the key against
    the provider and asks it for a model list, so `run_setup` is a coroutine
    and every question is drawn while an event loop is already running -
    `questionary`'s synchronous `ask` starts a second one underneath, and
    Python refuses. A scripted prompter cannot find that out, which is why
    `test_setup_wizard.py` drives this class through a pipe as well.
    """

    def __init__(self, *, text: Mapping[str, str] | None = None) -> None:
        self._text = wording() if text is None else text
        self._console = Console()

    def say(self, key: str, **fields: object) -> None:
        self._console.print(self._text[key].format(**fields))

    async def choose(self, key: str, options: Sequence[Option]) -> str | None:
        answer = await questionary.select(
            self._text[key],
            choices=[
                questionary.Choice(title=option.label, value=option.value) for option in options
            ],
            # A key can reach forty models; typing part of a name beats forty
            # arrow presses. With the filter on, j and k are letters again.
            use_search_filter=True,
            use_jk_keys=False,
        ).ask_async()
        return _as_answer(answer)

    async def secret(self, key: str) -> str | None:
        return _as_answer(await questionary.password(self._text[key]).ask_async())

    async def ask(self, key: str) -> str | None:
        return _as_answer(await questionary.text(self._text[key]).ask_async())


def _as_answer(value: object) -> str | None:
    """Narrows what `questionary` returns, which is typed `Any`.

    An interrupted question comes back as `None`, and so does anything else
    unexpected - which is what the wizard reads as walking away.
    """
    return value if isinstance(value, str) else None
