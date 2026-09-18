"""From a provider id in a file to an adapter that can answer (section 3.2).

`providers.toml` lists what the user may choose; `ADAPTERS` lists what this
build can actually construct. Keeping the two apart is what lets a provider be
added by editing data - but it also means they can disagree, so every failure
here is spelled out. A user who mistypes a provider name, or whose key was
never stored, gets a sentence telling them what to do instead of a traceback.

The key is fetched from the Windows Credential Manager at the last moment and
handed straight to the adapter. It is never stored on the entry, never logged,
and never written back to disk.

No adapter is registered yet: the Gemini Live adapter is task L1.1 and the
OpenAI one is L2.1 (plan.md section 7), and each arrives with its line in
`ADAPTERS`. Until then every catalogue entry names an adapter this build
does not have, and is refused in the sentence written for that.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from importlib import resources
from pathlib import Path

from assistant.config import load_api_key
from assistant.live.base import LLMProvider

__all__ = [
    "ADAPTERS",
    "MissingAPIKeyError",
    "MissingBaseURLError",
    "ProviderEntry",
    "RegistryError",
    "UnknownProviderError",
    "UnsupportedAdapterError",
    "create_provider",
    "load_catalog",
    "needs_base_url",
]

CATALOG_FILE_NAME = "providers.toml"


class RegistryError(Exception):
    """Base for the three ways asking for a provider can fail."""


class UnknownProviderError(RegistryError):
    """No entry with that id in the catalogue."""


class UnsupportedAdapterError(RegistryError):
    """The entry names an adapter this build does not have."""


class MissingAPIKeyError(RegistryError):
    """The provider needs a key and the Credential Manager has none."""


class MissingBaseURLError(RegistryError):
    """The entry has no address and none was given."""


@dataclass(frozen=True, slots=True)
class ProviderEntry:
    """One row of the catalogue, as the setup command and the adapters see it."""

    id: str
    adapter: str
    display_name: str
    key_url: str | None = None
    key_prefix: str | None = None
    base_url: str | None = None
    requires_key: bool = True
    has_live_pricing: bool = False


# An adapter takes its entry as well as the key: an adapter that speaks to
# an address reads `base_url` from it. Passing the whole row keeps that from
# becoming a signature change.
AdapterBuilder = Callable[[ProviderEntry, str], LLMProvider]

# Filled by L1.1 (`gemini_live`) and L2.1 (the OpenAI adapter). Empty, every
# entry of the catalogue is `UnsupportedAdapterError` and the wizard offers
# nothing - which is the truth about this build.
ADAPTERS: dict[str, AdapterBuilder] = {}

# The adapters that speak to an address rather than to one vendor. An entry
# of one of these with no `base_url` in the catalogue has to be asked for
# one, and the wizard asks. The name is the old repository's until L2.1
# decides whether the OpenAI adapter is one of these.
_ADDRESSED = frozenset({"openai_compat"})


def needs_base_url(entry: ProviderEntry) -> bool:
    """Whether the wizard has to ask the user where this provider is."""
    return entry.adapter in _ADDRESSED and not entry.base_url


def load_catalog(path: Path | None = None) -> dict[str, ProviderEntry]:
    """Reads the provider catalogue, from the packaged copy unless told otherwise."""
    raw = tomllib.loads(_read_catalog(path))
    known = {field.name for field in fields(ProviderEntry)} - {"id"}
    return {
        provider_id: ProviderEntry(
            id=provider_id,
            # A field this build does not know is skipped rather than fatal:
            # section 3.2 already foresees an `auth` key, and an older build
            # should still start when a newer catalogue mentions one.
            **{key: value for key, value in values.items() if key in known},
        )
        for provider_id, values in raw.items()
    }


def create_provider(
    provider_id: str,
    *,
    api_key: str | None = None,
    catalog: Mapping[str, ProviderEntry] | None = None,
    base_url: str | None = None,
) -> LLMProvider:
    """Builds the adapter for `provider_id`.

    `api_key` is for the setup command, which has to validate a key before it
    is stored and so cannot read it back from the Credential Manager yet.
    Everywhere else the key comes from there. `base_url` is the address the
    user gave for an entry that has none of its own; given, it outranks the
    catalogue's.
    """
    entries = load_catalog() if catalog is None else catalog

    entry = entries.get(provider_id)
    if entry is None:
        offered = ", ".join(sorted(entries)) or "nothing"
        raise UnknownProviderError(f"no provider {provider_id!r}; providers.toml offers: {offered}")
    if base_url:
        entry = replace(entry, base_url=base_url)

    builder = ADAPTERS.get(entry.adapter)
    if builder is None:
        raise UnsupportedAdapterError(
            f"{provider_id!r} needs the {entry.adapter!r} adapter, "
            f"which this version does not have yet"
        )

    key = api_key if api_key is not None else load_api_key(provider_id)
    if entry.requires_key and not key:
        raise MissingAPIKeyError(
            f"no API key stored for {provider_id!r} - run 'live-assistant setup' to add one"
        )

    return builder(entry, key or "")


def _read_catalog(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    # Read through `importlib.resources` rather than by walking up from
    # __file__: once installed, the package may not be a directory on disk.
    return (resources.files("assistant") / "defaults" / CATALOG_FILE_NAME).read_text(
        encoding="utf-8"
    )
