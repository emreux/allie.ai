"""What the user asked to be kept between runs (design.md section 3.7, 2.10).

Measured 2026-08-27: "adım Emre" was remembered in the next turn and
forgotten at the next start. Nothing in the design held it. The history
lives in memory and dies with the process (`agent/core.py`), the system
prompt carries no name so that its bytes never change (`agent/prompts.py`),
and the disk had only the audit rows. This is the second kind of memory,
beside the notes of phase 4: what the user explicitly asked to be kept, and
the assistant's own name, read at startup and put in front of every session.
(The first kind was the old loop's window of twelve turns. A live session keeps
its own history, on the server, for as long as it stays open.)

**A file, not a table.** `%APPDATA%\\allie\\memory.toml`, beside
`config.toml` and under its rules (section 10): data rather than code,
following the user through the roaming profile, editable by hand, never
holding a key. A table in the database would hide from the person whose
facts they are what is being said about them to a model.

**Small, and refused when full.** Every fact is a prefix to every request,
and Gemini gives no discount for a cached one (section 6): forty facts of
two hundred characters are about five hundred tokens, the size of the
frozen prompt itself. Past the ceiling `remember` refuses and says so.
Dropping the oldest quietly would be forgetting something the user said
not to forget.

**Only the two tools write it** - and setup, the name alone. `remember`
and `forget` (`tools/memory.py`) go through the gate like every tool
(invariant 1), so the file holds what the user asked for, in the words
they used; setup writes the name of the assistant the user chose there
(plan.md D31), which is a thing they asked for too. A file that does not
parse - a hand edit gone wrong - is a sentence at startup and is never
written over.

**The block is composed at every session** from what this holds, not once
at startup. `__main__` reads it inside the `session_config` closure rather than
handing over a finished sentence, so "bana Emre de" holds from the next session
on - the one the next sentence opens - and at the next start. The bytes change
only when the facts do, which is the one time the cache of architecture guide
section 2 is meant to miss.

**The persona is the third list** (2026-09-21, D23): how the assistant
should speak, under `[assistant]` beside its name, read into the prompt
after the facts; `remember(kind="persona")` adds a line, `forget` removes
one, the user edits it by hand like the rest.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from allie.config import config_dir
from allie.store.normalize import normalize_search

__all__ = [
    "FACTS_PROMPT",
    "MAX_FACTS",
    "MAX_FACT_CHARS",
    "MAX_PERSONA_LINES",
    "MEMORY_FILE_NAME",
    "NAME_PROMPT",
    "PERSONA_PROMPT",
    "MemoryFileError",
    "UserMemory",
    "memory_path",
]

MEMORY_FILE_NAME = "memory.toml"

# The ceiling of section 3.7: about five hundred tokens in front of every
# request, the size of the frozen prompt itself.
MAX_FACTS = 40
MAX_FACT_CHARS = 200

# The persona (plan.md D23): how the user wants the assistant to speak, as
# lines of their own. Ten lines at most - a manner is a few sentences, and
# every line rides in front of every request like a fact does.
MAX_PERSONA_LINES = 10

# What the model is told above the facts, and about its name. Addressed to
# the model and not to the user, so English and not in the locale pack
# (section 3.12), like the gate's answers; the facts themselves are in
# whatever language the user spoke.
FACTS_PROMPT = (
    "What the user asked you to remember, in their own words. Act on it without "
    "being asked, and do not recite it unless they ask what you remember:"
)
NAME_PROMPT = "Your name is {name}. Answer to it."

# Read last, after the rules and the facts: a manner is how the assistant
# speaks, never what it may do, and the sentence says so to the model.
PERSONA_PROMPT = (
    "How the user wants you to speak - your manner, not new permissions; the rules "
    "above still hold:"
)

# The top of the file, for the person editing it.
_HEADER = (
    "# assistant memory: what you asked to be remembered, the assistant's name and its manner.",
    "# Written by the assistant's remember and forget tools; safe to edit by hand.",
    "# One fact per line, in your own words; the assistant reads them at every request.",
)

_WORD = re.compile(r"\w+")


class MemoryFileError(RuntimeError):
    """`memory.toml` could not be read: not TOML, or not the shape this
    program writes. Fixable by the user, so named for `run` - and the
    reason the file is then never written over."""


def memory_path() -> Path:
    """`%APPDATA%\\allie\\memory.toml`, beside `config.toml`."""
    return config_dir() / MEMORY_FILE_NAME


class UserMemory:
    """What the user asked to be kept, the assistant's name and its manner.

    Held here once loaded and written back whole on every change: the
    file is small, and a file written half is a file nobody can read.
    """

    def __init__(
        self,
        *,
        name: str = "",
        facts: Iterable[str] = (),
        persona: Iterable[str] = (),
        path: Path | None = None,
    ) -> None:
        self._path = path if path is not None else memory_path()
        self.name = name.strip()
        self.facts: list[str] = [fact.strip() for fact in facts if fact.strip()]
        self.persona: list[str] = [line.strip() for line in persona if line.strip()]

    @classmethod
    def load(cls, path: Path | None = None) -> UserMemory:
        """Reads the file, or starts empty when there is none yet."""
        target = path if path is not None else memory_path()
        if not target.is_file():
            return cls(path=target)

        try:
            data = tomllib.loads(target.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as failure:
            raise MemoryFileError(f"{target} could not be read: {failure}") from failure

        assistant = data.get("assistant", {})
        user = data.get("user", {})
        name = assistant.get("name", "") if isinstance(assistant, dict) else None
        persona = assistant.get("persona", []) if isinstance(assistant, dict) else None
        facts = user.get("facts", []) if isinstance(user, dict) else None
        if (
            not isinstance(name, str)
            or not isinstance(persona, list)
            or not all(isinstance(line, str) for line in persona)
            or not isinstance(facts, list)
            or not all(isinstance(fact, str) for fact in facts)
        ):
            raise MemoryFileError(
                f"{target} is not shaped as memory.toml: [assistant] name is a string, "
                "[assistant] persona a list of strings and [user] facts a list of strings"
            )

        memory = cls(name=name, facts=facts, persona=persona, path=target)
        # The count, never the words: the log is not where they belong.
        logger.info(
            "memory: {count} facts, {persona} persona lines, name {named}",
            count=len(memory.facts),
            persona=len(memory.persona),
            named="set" if memory.name else "not set",
        )
        return memory

    @property
    def path(self) -> Path:
        return self._path

    @property
    def full(self) -> bool:
        return len(self.facts) >= MAX_FACTS

    def remember(self, fact: str) -> bool:
        """Keeps `fact`, unless there is no room - then nothing is written
        and the answer is `False`. A fact already kept is kept once."""
        kept = _clean(fact)
        if any(_phrase(kept) == _phrase(known) for known in self.facts):
            return True
        if self.full:
            return False
        self.facts.append(kept)
        self.save()
        return True

    def add_persona(self, line: str) -> bool:
        """Keeps one line of manner, unless there is no room - then nothing
        is written and the answer is `False`. A line already kept is kept once."""
        kept = _clean(line)
        if any(_phrase(kept) == _phrase(known) for known in self.persona):
            return True
        if len(self.persona) >= MAX_PERSONA_LINES:
            return False
        self.persona.append(kept)
        self.save()
        return True

    def forget(self, fact: str) -> str | None:
        """Removes the fact that reads like `fact` and says which it was;
        `None` when none does. Read the way search reads (`store/normalize`),
        so the model's spelling of a fact and the file's are one fact. Looked
        for in the facts first, then in the persona lines."""
        wanted = _phrase(fact)
        for index, known in enumerate(self.facts):
            if _phrase(known) == wanted:
                del self.facts[index]
                self.save()
                return known
        for index, known in enumerate(self.persona):
            if _phrase(known) == wanted:
                del self.persona[index]
                self.save()
                return known
        return None

    def rename(self, name: str) -> None:
        """Gives the assistant a name, or a new one."""
        self.name = _clean(name)
        self.save()

    def prompt(self, base: str) -> str:
        """`base` with the memory block behind it - or `base` alone, byte
        for byte, when there is nothing to remember."""
        block = self.block()
        return f"{base}\n\n{block}" if block else base

    def block(self) -> str:
        """What the model is told: its name, then the facts, one per line,
        then the persona lines, last of all."""
        parts: list[str] = []
        if self.name:
            parts.append(NAME_PROMPT.format(name=self.name))
        if self.facts:
            parts.append("\n".join([FACTS_PROMPT, *(f"- {fact}" for fact in self.facts)]))
        if self.persona:
            parts.append("\n".join([PERSONA_PROMPT, *(f"- {line}" for line in self.persona)]))
        return "\n\n".join(parts)

    def save(self) -> None:
        """Writes the whole file, creating the directory on the first fact."""
        lines = [*_HEADER, "", "[assistant]", f"name = {_quoted(self.name)}"]
        lines.append("persona = [")
        lines.extend(f"  {_quoted(line)}," for line in self.persona)
        lines.extend(("]", "", "[user]"))
        lines.append("facts = [")
        lines.extend(f"  {_quoted(fact)}," for fact in self.facts)
        lines.extend(("]", ""))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text("\n".join(lines), encoding="utf-8")


def _clean(fact: str) -> str:
    """One line, one space between words, within the ceiling."""
    text = " ".join(fact.split())
    if not text:
        raise ValueError("a fact cannot be empty")
    if len(text) > MAX_FACT_CHARS:
        raise ValueError(f"a fact is at most {MAX_FACT_CHARS} characters")
    return text


def _phrase(text: str) -> str:
    """`text` as the words in it, folded the way search folds."""
    return " ".join(_WORD.findall(normalize_search(text)))


def _quoted(text: str) -> str:
    """`text` as a TOML basic string.

    JSON's escapes are a subset of TOML's, and the standard library writes
    JSON strings and no TOML at all - so this is `json.dumps`, kept in one
    place with its reason.
    """
    return json.dumps(text, ensure_ascii=False)
