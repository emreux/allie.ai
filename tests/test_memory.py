"""What the user asked to be kept, kept (design.md section 3.7, 2.10).

Three claims, the ones section 8 names. The address survives a restart:
what `remember` wrote is what the next start reads, and it is in front of
the model's very next request as well. Memory has a ceiling and refuses
past it rather than dropping the oldest. And `forget` does not delete
without a yes - it is the first tool of phase 2 whose question is really
asked, and the gate that asks it is the real one.

The file is the user's: written whole, readable by hand, and never written
over when a hand edit left it unreadable.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from loguru import logger

from assistant.agent.core import Confirm, decline
from assistant.agent.policy import DECLINED, dispatch
from assistant.live.base import ToolCall
from assistant.store import memory as store
from assistant.store.memory import (
    FACTS_PROMPT,
    MAX_FACT_CHARS,
    MAX_FACTS,
    MAX_PERSONA_LINES,
    PERSONA_PROMPT,
    MemoryFileError,
    UserMemory,
    memory_path,
)
from assistant.tools import memory as tools
from assistant.tools.memory import forget_for, remember_for
from assistant.tools.registry import ToolRegistry

BASE = "You are an assistant."
EMRE = "Bana Emre diye hitap et."
COFFEE = "Kahveyi sade içerim."


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "memory.toml"


@pytest.fixture
def memory(path: Path) -> UserMemory:
    return UserMemory(path=path)


def call(name: str, **arguments: str) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=arguments)


async def says_yes(question: str) -> bool:
    return True


async def through_the_gate(
    registry: ToolRegistry, order: ToolCall, *, confirm: Confirm = decline
) -> str:
    return await dispatch(order, turn_id="t1", registry=registry, confirm=confirm)


# --------------------------------------------------------------------------
# The file
# --------------------------------------------------------------------------


def test_the_file_lives_beside_the_settings(config_home: Path) -> None:
    """`%APPDATA%\\live-assistant\\memory.toml`: data, roaming, editable, no key."""
    assert memory_path() == config_home / "memory.toml"


def test_a_machine_with_no_file_yet_starts_with_nothing(path: Path) -> None:
    loaded = UserMemory.load(path)

    assert (loaded.name, loaded.facts) == ("", [])
    assert not path.exists()


def test_what_was_remembered_is_read_at_the_next_start(path: Path, memory: UserMemory) -> None:
    """The claim of section 8: "bana Emre de", the program closed and
    opened, and the address still holds."""
    memory.remember(EMRE)
    memory.rename("Ada")

    again = UserMemory.load(path)

    assert again.facts == [EMRE]
    assert again.name == "Ada"
    assert EMRE in again.prompt(BASE)
    assert "Your name is Ada." in again.prompt(BASE)


def test_the_file_can_be_read_and_edited_by_hand(path: Path) -> None:
    path.write_text(
        "# my notes to the assistant\n"
        "[assistant]\n"
        'name = ""\n'
        "\n"
        "[user]\n"
        'facts = ["Bana Emre de.", "Toplantılarımı sabah 9\'a koyma."]\n',
        encoding="utf-8",
    )

    assert UserMemory.load(path).facts == ["Bana Emre de.", "Toplantılarımı sabah 9'a koyma."]


def test_a_fact_with_quotes_and_a_backslash_in_it_survives_the_file(
    path: Path, memory: UserMemory
) -> None:
    awkward = 'Dosyalarım "C:\\Belgeler" altında, unutma.'
    memory.remember(awkward)

    assert UserMemory.load(path).facts == [awkward]


@pytest.mark.parametrize(
    "body",
    [
        "[user]\nfacts = [oops\n",
        '[user]\nfacts = "not a list"\n',
        "[assistant]\nname = 3\n",
        "[user]\nfacts = [1, 2]\n",
    ],
)
def test_a_file_that_cannot_be_read_is_said_and_never_written_over(path: Path, body: str) -> None:
    """A hand edit gone wrong is the user's to fix. Starting with an empty
    memory and writing it back would replace their file with nothing."""
    path.write_text(body, encoding="utf-8")

    with pytest.raises(MemoryFileError):
        UserMemory.load(path)

    assert path.read_text(encoding="utf-8") == body


def test_loading_writes_down_how_many_facts_and_never_which(path: Path) -> None:
    path.write_text(f'[user]\nfacts = ["{EMRE}", "{COFFEE}"]\n', encoding="utf-8")
    lines: list[str] = []
    handle = logger.add(lines.append, format="{message}")
    try:
        UserMemory.load(path)
    finally:
        logger.remove(handle)

    [line] = lines
    assert "2 facts" in line
    assert "Emre" not in line


# --------------------------------------------------------------------------
# The ceiling
# --------------------------------------------------------------------------


def test_the_ceiling_is_the_forty_facts_of_section_3_7() -> None:
    assert (MAX_FACTS, MAX_FACT_CHARS) == (40, 200)


def test_past_the_ceiling_nothing_is_written_and_the_answer_is_no(
    path: Path, memory: UserMemory
) -> None:
    """Dropping the oldest quietly would be forgetting something the user
    said not to forget."""
    for number in range(MAX_FACTS):
        assert memory.remember(f"Gerçek numara {number}.")

    assert memory.full
    assert memory.remember("Bir fazlası.") is False
    assert len(UserMemory.load(path).facts) == MAX_FACTS


def test_a_fact_too_long_for_the_file_is_refused(memory: UserMemory) -> None:
    with pytest.raises(ValueError, match="200"):
        memory.remember("x" * (MAX_FACT_CHARS + 1))


def test_a_fact_kept_twice_is_kept_once(memory: UserMemory) -> None:
    memory.remember("Bana Emre de.")
    memory.remember("bana emre de")

    assert memory.facts == ["Bana Emre de."]


# --------------------------------------------------------------------------
# What the model is told
# --------------------------------------------------------------------------


def test_with_nothing_remembered_the_prompt_is_the_base_byte_for_byte(memory: UserMemory) -> None:
    """A machine where nobody asked for anything runs on the phase 1
    prompt exactly, cache and all."""
    assert memory.prompt(BASE) == BASE


def test_the_block_names_the_assistant_and_lists_the_facts(memory: UserMemory) -> None:
    memory.rename("Ada")
    memory.remember(EMRE)
    memory.remember(COFFEE)

    assert memory.prompt(BASE) == (
        f"{BASE}\n\nYour name is Ada. Answer to it.\n\n{FACTS_PROMPT}\n- {EMRE}\n- {COFFEE}"
    )


def test_the_block_is_in_english_and_the_facts_are_not_translated() -> None:
    """Addressed to the model, like the gate's answers; the facts are the
    user's own words in the user's own language (section 3.12)."""
    assert "user" in FACTS_PROMPT
    assert all(character.isascii() for character in FACTS_PROMPT + store.NAME_PROMPT)


def test_a_fact_kept_a_moment_ago_is_in_the_prompt_read_next(memory: UserMemory) -> None:
    """Rule 4 of section 3.7, as the live loop keeps it (plan.md 4.5): the
    prompt is a source read when a session opens, so a fact the last
    session kept is in the next one's prompt without a restart. The loop
    that reads it is L1.4; what is claimed here is that the source answers
    differently the moment the file does."""
    source = memory.prompt
    before = source(BASE)

    memory.remember(EMRE)

    assert before == BASE
    assert source(BASE) == f"{BASE}\n\n{FACTS_PROMPT}\n- {EMRE}"


# --------------------------------------------------------------------------
# The tools, through the real gate
# --------------------------------------------------------------------------


def test_remember_is_safe_and_forget_asks(memory: UserMemory) -> None:
    remember, forget = remember_for(memory), forget_for(memory)

    assert remember.risk == "safe"
    assert remember.spec.parameters["required"] == ["fact"]
    assert remember.spec.parameters["properties"]["kind"]["enum"] == [
        "fact",
        "assistant_name",
        "persona",
    ]
    assert forget.risk == "confirm"
    assert forget.confirm_prompt == tools.TEXT["forget_confirm"]


def test_forget_is_declared_with_the_pack_s_question(memory: UserMemory) -> None:
    forget = forget_for(memory, confirm_prompt="'{fact}' kaydı unutulacak.")

    assert forget.confirm_prompt == "'{fact}' kaydı unutulacak."


async def test_remember_writes_the_file_and_says_how_full_it_is(
    path: Path, memory: UserMemory
) -> None:
    registry = ToolRegistry([remember_for(memory)])

    answer = await through_the_gate(registry, call("remember", fact=f"  {EMRE}  "))

    assert answer == tools.KEPT.format(count=1, limit=MAX_FACTS)
    assert UserMemory.load(path).facts == [EMRE]


async def test_naming_the_assistant_goes_through_remember(path: Path, memory: UserMemory) -> None:
    """No separate tool: "adın Ada" is `remember` with `kind`."""
    registry = ToolRegistry([remember_for(memory)])

    answer = await through_the_gate(registry, call("remember", fact="Ada", kind="assistant_name"))

    assert answer == tools.NAMED.format(name="Ada")
    assert UserMemory.load(path).name == "Ada"


async def test_when_memory_is_full_the_tool_says_so_and_writes_nothing(
    path: Path, memory: UserMemory
) -> None:
    for number in range(MAX_FACTS):
        memory.remember(f"Gerçek numara {number}.")
    registry = ToolRegistry([remember_for(memory)])

    answer = await through_the_gate(registry, call("remember", fact="Bir fazlası."))

    assert answer == tools.FULL.format(limit=MAX_FACTS)
    assert "Bir fazlası." not in UserMemory.load(path).facts


async def test_a_fact_too_long_is_answered_in_words_and_not_as_a_failure(
    memory: UserMemory,
) -> None:
    registry = ToolRegistry([remember_for(memory)])

    answer = await through_the_gate(registry, call("remember", fact="x" * 201))

    assert answer == tools.TOO_LONG.format(limit=MAX_FACT_CHARS)
    assert memory.facts == []


async def test_an_empty_fact_is_nothing_to_keep(memory: UserMemory) -> None:
    registry = ToolRegistry([remember_for(memory)])

    assert await through_the_gate(registry, call("remember", fact="   ")) == tools.EMPTY


async def test_forget_does_not_delete_without_a_yes(path: Path, memory: UserMemory) -> None:
    """The claim of section 8. The real gate, nobody to ask: the fact stays."""
    memory.remember(EMRE)
    registry = ToolRegistry([forget_for(memory)])

    answer = await through_the_gate(registry, call("forget", fact=EMRE))

    assert answer == DECLINED
    assert UserMemory.load(path).facts == [EMRE]


async def test_forget_deletes_once_the_user_has_said_yes(path: Path, memory: UserMemory) -> None:
    memory.remember(EMRE)
    memory.remember(COFFEE)
    registry = ToolRegistry([forget_for(memory)])

    answer = await through_the_gate(
        registry, call("forget", fact="bana emre diye hitap et"), confirm=says_yes
    )

    assert answer == tools.FORGOTTEN.format(fact=EMRE, count=1)
    assert UserMemory.load(path).facts == [COFFEE]


async def test_the_question_names_the_fact(memory: UserMemory) -> None:
    memory.remember(EMRE)
    registry = ToolRegistry([forget_for(memory, confirm_prompt="'{fact}' kaydı unutulacak.")])
    asked: list[str] = []

    async def hears(question: str) -> bool:
        asked.append(question)
        return False

    await through_the_gate(registry, call("forget", fact=EMRE), confirm=hears)

    assert asked == [f"'{EMRE}' kaydı unutulacak."]


async def test_forgetting_what_was_never_kept_lists_what_was(memory: UserMemory) -> None:
    memory.remember(COFFEE)
    registry = ToolRegistry([forget_for(memory)])

    answer = await through_the_gate(registry, call("forget", fact=EMRE), confirm=says_yes)

    assert answer == tools.NOT_FOUND.format(fact=EMRE, facts=COFFEE)


async def test_forgetting_from_an_empty_memory_says_nothing_is_stored(memory: UserMemory) -> None:
    registry = ToolRegistry([forget_for(memory)])

    answer = await through_the_gate(registry, call("forget", fact=EMRE), confirm=says_yes)

    assert answer == tools.NOT_FOUND.format(fact=EMRE, facts=tools.NONE_STORED)


# --------------------------------------------------------------------------
# The persona (plan.md D23)
# --------------------------------------------------------------------------

MANNER = "Kısa ve net konuş."
ADDRESS = "Bana efendim de."


def test_persona_lines_survive_a_restart(path: Path) -> None:
    memory = UserMemory(path=path)

    assert memory.add_persona(MANNER)

    assert UserMemory.load(path).persona == [MANNER]


def test_the_file_writes_the_persona_under_assistant(path: Path) -> None:
    """Beside the name, where the person editing the file expects the
    assistant's own things; the facts stay under `[user]`."""
    memory = UserMemory(path=path)
    memory.add_persona(MANNER)

    data = tomllib.loads(path.read_text(encoding="utf-8"))

    assert data["assistant"]["persona"] == [MANNER]
    assert "persona" not in data["user"]


def test_the_block_is_name_then_facts_then_persona(memory: UserMemory) -> None:
    """The rules come first in the prompt, the persona last (spec section
    3): a manner cannot outrank a rule it is read after."""
    memory.rename("Friday")
    memory.remember(EMRE)
    memory.add_persona(MANNER)

    block = memory.block()

    assert block.index("Friday") < block.index(EMRE) < block.index(PERSONA_PROMPT)
    assert block.endswith(f"{PERSONA_PROMPT}\n- {MANNER}")
    assert "rules above still hold" in PERSONA_PROMPT


def test_without_a_persona_the_block_does_not_mention_one(memory: UserMemory) -> None:
    memory.remember(EMRE)

    assert PERSONA_PROMPT not in memory.block()
    assert memory.prompt(BASE) == f"{BASE}\n\n{memory.block()}"


def test_the_persona_has_a_ceiling_and_refuses_past_it(memory: UserMemory) -> None:
    for index in range(MAX_PERSONA_LINES):
        assert memory.add_persona(f"Kural {index}")

    assert not memory.add_persona("Bir kural daha")
    assert len(memory.persona) == MAX_PERSONA_LINES


def test_a_persona_line_said_twice_is_kept_once(memory: UserMemory) -> None:
    memory.add_persona(MANNER)

    assert memory.add_persona("kısa ve net konuş")

    assert memory.persona == [MANNER]


def test_forget_reaches_the_persona_lines_too(memory: UserMemory) -> None:
    memory.remember(EMRE)
    memory.add_persona(ADDRESS)

    assert memory.forget("bana efendim de") == ADDRESS
    assert memory.persona == []
    assert memory.facts == [EMRE]


def test_a_file_from_before_the_persona_still_loads(path: Path) -> None:
    path.write_text('[assistant]\nname = "Friday"\n\n[user]\nfacts = ["x"]\n', encoding="utf-8")

    loaded = UserMemory.load(path)

    assert (loaded.name, loaded.facts, loaded.persona) == ("Friday", ["x"], [])


def test_a_persona_that_is_not_a_list_of_strings_is_refused(path: Path) -> None:
    path.write_text('[assistant]\npersona = "tek satır"\n\n[user]\nfacts = []\n', encoding="utf-8")

    with pytest.raises(MemoryFileError, match="persona"):
        UserMemory.load(path)


async def test_remember_as_persona_keeps_a_line_of_manner(memory: UserMemory) -> None:
    said = await remember_for(memory).run(fact=MANNER, kind="persona")

    assert memory.persona == [MANNER]
    assert said == tools.PERSONA_KEPT.format(count=1, limit=MAX_PERSONA_LINES)


async def test_a_full_persona_is_refused_with_advice(memory: UserMemory) -> None:
    for index in range(MAX_PERSONA_LINES):
        memory.add_persona(f"Kural {index}")

    said = await remember_for(memory).run(fact="Bir daha", kind="persona")

    assert said == tools.PERSONA_FULL.format(limit=MAX_PERSONA_LINES)
    assert len(memory.persona) == MAX_PERSONA_LINES


def test_the_model_is_told_what_persona_means(memory: UserMemory) -> None:
    kind = remember_for(memory).spec.parameters["properties"]["kind"]

    assert kind["enum"] == ["fact", "assistant_name", "persona"]
    assert "speak" in kind["description"]


async def test_not_found_lists_the_persona_lines_too(memory: UserMemory) -> None:
    memory.add_persona(ADDRESS)
    forget = forget_for(memory)

    said = await forget.run(fact="kahve")

    assert said == tools.NOT_FOUND.format(fact="kahve", facts=ADDRESS)
