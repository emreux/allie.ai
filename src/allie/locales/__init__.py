"""Locale packs - everything that depends on the language (design.md 3.12).

The product is not pinned to one language, and the way that is kept true is
mechanical: no module carries a sentence in a language other than English, and
every language-dependent value lives in one TOML file per language, next to
this one. Adding a language is a file and a pull request, not a code change.

**The fallback chain runs requested pack -> `en` -> the constant in the code.**
The last link is not a table somewhere: it is the English sentence written
beside the code that says it, passed to `say` as the default. That is why
`en.toml` carries no sentences at all - a second English copy here would be a
second thing to keep in step - and why a language nobody has translated still
produces a working assistant with an English interface.

**Sentences fall back; identity does not.** `code`, `name` and the speech hint
default to the locale itself rather than to English. A recogniser told to
expect English while a German speaks is worse than having no preference at
all: the model never sees what was said.
"""

from __future__ import annotations

import ctypes
import locale as windows
import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = ["FALLBACK_CODE", "Locale", "available", "iso_code", "load", "system_code"]

# The language every other one falls back to, and the language this project
# writes its code in. Those two being the same is a convenience, not a rule.
FALLBACK_CODE = "en"

SUFFIX = ".toml"

# A file whose name starts with this is a scaffold rather than a language.
# `_template.toml` is one; it is there to be copied and translated.
NOT_A_LANGUAGE = "_"


@dataclass(frozen=True, slots=True)
class Locale:
    """One language, as everything outside this package sees it."""

    code: str
    name: str

    # The language the *user speaks*, which is what the recogniser is told to
    # expect. Usually the code above, and deliberately separate from it: a
    # locale is free to show one language and listen for another.
    stt_language: str

    # What `[ui]` translated, already merged over `en`. Anything missing here
    # is answered by the caller's own English constant through `say`.
    ui: Mapping[str, str]

    # The words the confirmation window listens for, from `[speech]`
    # (design.md 3.1 rule 2). They fall back the way sentences do, not the
    # way identity does: a pack without them gets the English words beside
    # the code that listens (`app.py`), together with the English hint that
    # tells the user which words to say - so the two always agree. They are
    # also what the recogniser of that window is told to expect
    # (`app.confirm_prompt`): since 2026-09-22 there is no separate
    # `[stt] prompt`, because a live product's recogniser hears nothing else.
    yes_words: tuple[str, ...] = ()
    no_words: tuple[str, ...] = ()

    # What is said while a tool takes its time, from `[speech] filler`
    # (design.md section 4, 2.8): said in turn, one per turn. They fall
    # back like the yes and no words, to the English beside the code that
    # says them (`app.py`).
    fillers: tuple[str, ...] = ()

    # The question the setup wizard asks a model to see whether it calls a
    # tool, from `[probe]` (design.md section 3.2, 2.6). In the pack's own
    # language, so that one request also shows the model understands the
    # user; it falls back like a sentence, to the English beside the code
    # that asks it (`live/probe.py`), and `en.toml` carries none.
    probe_question: str = ""

    # The two hints the live model needs, from `[live]` (plan.md D19,
    # ADR-001 section 6): the BCP-47 code the session is opened with - for
    # the recogniser behind the model and for its voice - and one sentence
    # for the system prompt saying which language the user speaks. Measured
    # 2026-09-18: without them a short or quiet Turkish sentence is heard as
    # Hindi and answered in it. Identity, like the speech language: English
    # lends neither, and a pack without them leaves the server and the
    # prompt to their defaults.
    language_code: str = ""
    user_language_rule: str = ""

    def say(self, key: str, default: str) -> str:
        """The sentence for `key`, or `default` - the last link of the chain.

        `default` is the English constant written beside the code that says
        the sentence. A pack that leaves a key out, or leaves it blank, gets
        that instead: half a translation reads better in one language than in
        two.
        """
        return self.ui.get(key) or default


def load(code: str | None = None, *, directory: Path | None = None) -> Locale:
    """The pack for `code`, with English behind it.

    `directory` reads packs from somewhere other than this package, which is
    how the tests exercise the chain without shipping a language to do it.
    """
    wanted = _normalise(code)
    english = _read(FALLBACK_CODE, directory) or {}
    pack = english if wanted == FALLBACK_CODE else (_read(wanted, directory) or {})

    return Locale(
        code=wanted,
        name=_text(pack, "name") or wanted,
        stt_language=_text(_table(pack, "stt"), "language") or wanted,
        ui={**_texts(_table(english, "ui")), **_texts(_table(pack, "ui"))},
        yes_words=_words(_table(pack, "speech"), "yes_words"),
        no_words=_words(_table(pack, "speech"), "no_words"),
        fillers=_words(_table(pack, "speech"), "filler"),
        probe_question=_text(_table(pack, "probe"), "question").strip(),
        language_code=_text(_table(pack, "live"), "language_code").strip(),
        user_language_rule=_text(_table(pack, "live"), "user_language_rule").strip(),
    )


def available(*, directory: Path | None = None) -> list[Locale]:
    """Every language there is a pack for, in a settled order.

    Sorted by code rather than by name: a menu that reshuffles itself between
    runs is a menu nobody learns, and sorting names written in their own
    scripts would put the order at the mercy of the alphabet.
    """
    return [load(code, directory=directory) for code in sorted(_codes(directory))]


def system_code() -> str:
    """The language Windows itself is in, for use before anyone has been asked.

    The setup wizard has to word its language question in some language, and
    the one the machine is already set to is the best guess available.
    """
    return iso_code(_ui_language_id())


def iso_code(identifier: int) -> str:
    """A Windows language identifier as ISO 639-1, or the fallback: an
    interface has to be in some language before the first question is asked.
    """
    return windows.windows_locale.get(identifier, "").partition("_")[0] or FALLBACK_CODE


# --------------------------------------------------------------------------
# Reading the files
# --------------------------------------------------------------------------


def _normalise(code: str | None) -> str:
    """`tr-TR`, `tr_TR` and `TR` all name the pack `tr.toml`."""
    wanted = (code or "").strip().partition("-")[0].partition("_")[0].casefold()

    # The code arrives from a settings file, and is about to be turned into a
    # file name. A language is letters; anything else is a path, and following
    # one would be this module's own fault.
    return wanted if wanted.isalpha() else FALLBACK_CODE


def _codes(directory: Path | None) -> Iterator[str]:
    for name in _file_names(directory):
        code = name[: -len(SUFFIX)]
        if not name.endswith(SUFFIX) or code.startswith(NOT_A_LANGUAGE):
            continue
        # A pack that does not parse is not offered as a language: the menu
        # would list one that speaks nothing but English.
        if _read(code, directory) is not None:
            yield code


def _file_names(directory: Path | None) -> Iterator[str]:
    if directory is not None:
        return (path.name for path in directory.iterdir())
    return (entry.name for entry in resources.files(__name__).iterdir())


def _read(code: str, directory: Path | None) -> Mapping[str, Any] | None:
    """One pack, or `None` if there is none or it does not parse."""
    text = _source(code, directory)
    if text is None:
        return None

    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        # Somebody else's typo, in a file this product only reads. Refusing to
        # start over it would punish the user for a contributor's slip;
        # `test_locales.py` is what catches it in the packs shipped here.
        return None


def _source(code: str, directory: Path | None) -> str | None:
    if directory is not None:
        path = directory / f"{code}{SUFFIX}"
        return path.read_text(encoding="utf-8") if path.is_file() else None

    # Through `importlib.resources` rather than by walking from `__file__`:
    # once installed, the package need not be a directory on disk.
    resource = resources.files(__name__) / f"{code}{SUFFIX}"
    try:
        return resource.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None


def _table(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """One TOML table, or an empty one if the pack put something else there."""
    found = values.get(key)
    return found if isinstance(found, Mapping) else {}


def _text(values: Mapping[str, Any], key: str) -> str:
    found = values.get(key)
    return found if isinstance(found, str) else ""


def _texts(values: Mapping[str, Any]) -> dict[str, str]:
    return {key: value for key, value in values.items() if isinstance(value, str)}


def _words(values: Mapping[str, Any], key: str) -> tuple[str, ...]:
    """A list of words, or nothing if the pack put something else there."""
    found = values.get(key)
    if not isinstance(found, list):
        return ()
    return tuple(word.strip() for word in found if isinstance(word, str) and word.strip())


def _ui_language_id() -> int:
    try:
        return int(ctypes.windll.kernel32.GetUserDefaultUILanguage())
    except (AttributeError, OSError):
        # Not Windows, or a Windows too stripped down to answer. Either way
        # there is no interface language to read, and `iso_code` says so.
        return 0
