"""The assistants this program offers: a name, a voice and a wake phrase
each (plan.md D31).

The owner's answer to "which name should it wake to" was four answers:
Jarvis and Friday, the classic pair, and Vesper and Allie, two of our own -
two men's voices, two women's. Each has its own classifier rather than one
model for all four: a phrase trained alone sees every synthetic clip, gets
a threshold of its own ("allie" is one vowel from the Turkish "Ali",
"vesper" rhymes with almost nothing), and a name nobody chose cannot wake
the one they did. Running one costs what running any costs - the shared
frontends are the expensive part, the head is a fraction of a millisecond.

**Data, not code** (`wake/assistants.toml`, beside the models it names).
Setup offers the entries in the file's order and writes three things from
the one chosen: `[wake] model`, `[live] voice` by the provider's own name,
and the name in `memory.toml`, which is how the model learns what it is
called. The threshold is not written: it ships with the model and is read
at every start (`threshold_for`), so that a better number reaches the user
with the next version; `[wake] threshold` in `config.toml` outranks it for
a user who measured their own room.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Literal

__all__ = [
    "DEFAULT_THRESHOLD",
    "Assistant",
    "load_assistants",
    "threshold_for",
]

CATALOG_FILE_NAME = "assistants.toml"

# The score that counts as the phrase for a model with no eval behind it:
# a user's own file, or an entry written without one. The middle of the
# scale, which is what the toolkit's own eval starts from.
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class Assistant:
    """One entry of the catalogue."""

    id: str
    name: str
    gender: Literal["male", "female"]
    wake: str
    threshold: float = DEFAULT_THRESHOLD
    voices: Mapping[str, str] = field(default_factory=dict)

    def voice_for(self, provider_id: str) -> str:
        """The voice by that provider's own name; empty - the provider's
        default - for a provider the entry names none for."""
        return self.voices.get(provider_id, "")


def load_assistants(path: Path | None = None) -> dict[str, Assistant]:
    """The catalogue in the file's order, from the packaged copy unless told otherwise."""
    if path is None:
        text = (resources.files("allie") / "wake" / CATALOG_FILE_NAME).read_text(encoding="utf-8")
    else:
        text = path.read_text(encoding="utf-8")
    return {
        assistant_id: Assistant(id=assistant_id, **values)
        for assistant_id, values in tomllib.loads(text).items()
    }


def threshold_for(model: str, assistants: Mapping[str, Assistant] | None = None) -> float:
    """The threshold a shipped model came with, by its stem; the default for
    anything else - a path to the user's own model among them."""
    catalog = load_assistants() if assistants is None else assistants
    for assistant in catalog.values():
        if assistant.wake == model:
            return assistant.threshold
    return DEFAULT_THRESHOLD
