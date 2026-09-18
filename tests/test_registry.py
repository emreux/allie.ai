"""Turning a name in a file into a working adapter.

The registry is the seam that keeps the provider list out of the code
(design.md section 3.2): a catalogue entry, a key from the Credential Manager,
and an adapter comes back. What it must never do is fail vaguely - "KeyError:
'gemini'" tells the owner nothing, while "no API key stored, run live-assistant
setup" tells them exactly what to do next.

No adapter is registered in this build (plan.md L0): the tests that build
one build it through an adapter registered here, and the claim that every
shipped entry can be built returns with the Gemini Live adapter (L1.1).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from assistant.config import store_api_key
from assistant.live.base import Delta, LLMProvider, Message, ModelInfo, ToolSpec
from assistant.live.registry import (
    ADAPTERS,
    MissingAPIKeyError,
    MissingBaseURLError,
    ProviderEntry,
    UnknownProviderError,
    UnsupportedAdapterError,
    create_provider,
    load_catalog,
    needs_base_url,
)
from tests.conftest import MemoryKeyring


class Inert:
    """A provider that satisfies the protocol and reaches nothing."""

    id = "inert"

    async def validate_credentials(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        return
        yield


@pytest.fixture
def keys_handed_over(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Registers an adapter that records the key it was built with, so the
    paths that are about key handling can be exercised without a vendor."""
    seen: list[str] = []

    def build(entry: ProviderEntry, api_key: str) -> LLMProvider:
        seen.append(api_key)
        return Inert()

    monkeypatch.setitem(ADAPTERS, "recording", build)
    return seen


def recording_catalog(*, requires_key: bool = True) -> dict[str, ProviderEntry]:
    return {
        "x": ProviderEntry(id="x", adapter="recording", display_name="X", requires_key=requires_key)
    }


# --------------------------------------------------------------------------
# The catalogue that ships with the package
# --------------------------------------------------------------------------


def test_the_shipped_catalogue_offers_gemini_live() -> None:
    """Plan.md D13: the first provider is Google's Live API, on the owner's
    key. The entry is named after the vendor so that the key it stores is
    the one `[stt]` and `[tts]` share."""
    entry = load_catalog()["gemini"]

    assert entry.adapter == "gemini_live"
    assert entry.requires_key is True
    assert entry.key_url is not None


def test_every_paid_provider_says_where_its_key_comes_from() -> None:
    """The wizard shows the address; an entry without one leaves the user
    to search for it."""
    for provider_id, entry in load_catalog().items():
        if entry.requires_key:
            assert entry.key_url, f"{provider_id} has no key_url"


def test_the_catalogue_can_be_read_from_a_given_file(tmp_path: Path) -> None:
    catalogue = tmp_path / "providers.toml"
    catalogue.write_text('[groq]\nadapter = "openai_compat"\ndisplay_name = "Groq"\n', "utf-8")

    entries = load_catalog(catalogue)

    assert entries["groq"].display_name == "Groq"
    assert entries["groq"].id == "groq"


def test_an_entry_may_leave_the_optional_fields_out(tmp_path: Path) -> None:
    """Only an addressed provider needs `base_url`; only a paid one needs `key_url`."""
    catalogue = tmp_path / "providers.toml"
    catalogue.write_text('[x]\nadapter = "gemini_live"\ndisplay_name = "X"\n', "utf-8")

    entry = load_catalog(catalogue)["x"]

    assert (entry.base_url, entry.key_url, entry.key_prefix) == (None, None, None)


def test_a_field_from_a_later_version_does_not_break_the_catalogue(tmp_path: Path) -> None:
    """Section 3.2 already foresees an `auth` field; an old build must still start."""
    catalogue = tmp_path / "providers.toml"
    body = '[x]\nadapter = "gemini_live"\ndisplay_name = "X"\nauth = "aws"\n'
    catalogue.write_text(body, "utf-8")

    assert load_catalog(catalogue)["x"].adapter == "gemini_live"


# --------------------------------------------------------------------------
# Building an adapter
# --------------------------------------------------------------------------


def test_the_key_comes_from_the_credential_manager(
    vault: MemoryKeyring, keys_handed_over: list[str]
) -> None:
    store_api_key("x", "stored-key")

    create_provider("x", catalog=recording_catalog())

    assert keys_handed_over == ["stored-key"]


def test_an_explicit_key_is_used_as_given(
    vault: MemoryKeyring, keys_handed_over: list[str]
) -> None:
    """Setup validates a key before storing it, so it must be able to pass one in."""
    store_api_key("x", "stored-key")

    create_provider("x", api_key="typed-key", catalog=recording_catalog())

    assert keys_handed_over == ["typed-key"]


def test_without_a_stored_key_the_error_says_what_to_run(
    vault: MemoryKeyring, keys_handed_over: list[str]
) -> None:
    with pytest.raises(MissingAPIKeyError, match="live-assistant setup"):
        create_provider("x", catalog=recording_catalog())

    assert keys_handed_over == []


def test_a_provider_that_needs_no_key_is_built_without_one(
    vault: MemoryKeyring, keys_handed_over: list[str]
) -> None:
    """A local endpoint has nothing to authenticate with."""
    create_provider("x", catalog=recording_catalog(requires_key=False))

    assert keys_handed_over == [""]


def test_an_unknown_provider_names_the_ones_that_exist() -> None:
    with pytest.raises(UnknownProviderError, match="gemini"):
        create_provider("claude")


def test_a_provider_whose_adapter_is_not_written_yet_says_so() -> None:
    """A catalogue entry naming an adapter this build does not have must
    fail clearly, not after the user has typed their key in. In this build
    that is every shipped entry (plan.md L0): the adapter is L1.1."""
    catalog = {"proxy": ProviderEntry(id="proxy", adapter="litellm", display_name="Proxy")}

    with pytest.raises(UnsupportedAdapterError, match="litellm"):
        create_provider("proxy", api_key="test-key", catalog=catalog)
    with pytest.raises(UnsupportedAdapterError, match="gemini_live"):
        create_provider("gemini", api_key="test-key")


def test_an_empty_address_from_the_caller_is_no_address(
    keys_handed_over: list[str],
) -> None:
    """An addressed adapter given an empty address refuses in a sentence
    that names the fix, as a missing key does."""

    def build(entry: ProviderEntry, api_key: str) -> LLMProvider:
        if not entry.base_url:
            raise MissingBaseURLError(f"no server address stored for {entry.id!r}")
        return Inert()

    catalog = {"custom": ProviderEntry(id="custom", adapter="addressed", display_name="Other")}
    ADAPTERS["addressed"] = build
    try:
        with pytest.raises(MissingBaseURLError):
            create_provider("custom", api_key="k", catalog=catalog, base_url="")
        assert isinstance(
            create_provider("custom", api_key="k", catalog=catalog, base_url="http://x/v1"),
            Inert,
        )
    finally:
        del ADAPTERS["addressed"]


def test_only_an_addressed_entry_without_an_address_needs_asking() -> None:
    """What the wizard asks: an entry of an addressed adapter with no address
    in the file is asked where it is; one with an address is not; Gemini
    speaks to one vendor and has no address to ask for."""
    custom = ProviderEntry(id="custom", adapter="openai_compat", display_name="Other")
    groq = ProviderEntry(
        id="groq", adapter="openai_compat", display_name="Groq", base_url="https://x/v1"
    )

    assert needs_base_url(custom) is True
    assert needs_base_url(groq) is False
    assert needs_base_url(load_catalog()["gemini"]) is False
