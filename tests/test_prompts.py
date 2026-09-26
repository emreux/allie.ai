"""The system prompt (plan.md section 4.5).

Lifted from the old loop's tests, which had nowhere else to go once the
loop was gone: the prompt is the rules it names and nothing else, it pins
no language, it carries no clock, and nothing in its module can change
between two sessions. New with the live product: the line the assistant
sends after it has read a reminder aloud begins with a prefix the prompt
itself names, so that the two cannot drift apart.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from allie.agent import prompts
from allie.agent.prompts import (
    ANNOUNCED_PREFIX,
    BREVITY,
    DOCUMENTS_RULE,
    LANGUAGE_FALLBACK,
    LANGUAGE_RULE,
    LIVE_RULES,
    PERSONALITY,
    SEARCH_RULE,
    SYSTEM_PROMPT,
    UNTRUSTED_RULE,
)


def test_the_prompt_is_made_of_the_rules_it_names() -> None:
    """Each rule is a separate constant so that it can be read, argued with and
    replaced on its own; the prompt is the six of them and nothing else."""
    rules = (PERSONALITY, BREVITY, LANGUAGE_RULE, LANGUAGE_FALLBACK, UNTRUSTED_RULE, LIVE_RULES)

    assert "\n\n".join(rules) == SYSTEM_PROMPT


def test_the_model_is_told_it_hears_the_voice_itself() -> None:
    """D1: nothing stands between the user and the model any more. The old
    line promised a transcript and asked the model to read through misheard
    words - a pipeline's problem, and one a live model invents if told it
    has it."""
    said = PERSONALITY.casefold()

    assert "hear the user's own voice" in said
    assert "misheard" not in said


def test_the_prompt_pins_no_language() -> None:
    """Section 3.12: the reply mirrors whatever language the user just used.
    Naming one here would quietly beat the rule that says so."""
    said = SYSTEM_PROMPT.casefold()

    assert not [name for name in ("turkish", "türkçe", "english", "german") if name in said]


def test_a_message_with_no_language_in_it_keeps_the_last_one() -> None:
    """Measured 2026-08-31: a transcript of digits alone was answered in
    English. There was no language to mirror, so the rule has to say what to
    do when there is none - without naming one."""
    said = LANGUAGE_FALLBACK.casefold()

    assert "digits" in said
    assert "last" in said
    assert not [character for character in LANGUAGE_FALLBACK if character.isdigit()]


def test_the_prompt_carries_no_clock_and_no_calendar() -> None:
    """A date, a time or a count is a digit, and a digit in the prefix is a
    cache that stops hitting. The date comes from a tool, at every open."""
    assert not [character for character in SYSTEM_PROMPT if character.isdigit()]


def test_the_prompt_is_written_for_the_ear() -> None:
    """The model is heard, not read: the brevity rule says so, and no
    longer talks about markdown - there is no text to keep it out of."""
    said = BREVITY.casefold()

    assert "heard" in said
    assert "markdown" not in said


def test_the_reminder_line_begins_with_the_prefix_the_rule_names() -> None:
    """`app.py` sends the line, the prompt explains it; one constant for
    both, or a reworded prefix would make the model repeat every reminder."""
    assert ANNOUNCED_PREFIX.strip() in LIVE_RULES
    assert ANNOUNCED_PREFIX.endswith(" ")
    assert "declined" in LIVE_RULES.casefold()


def test_nothing_in_the_prompt_module_can_change_between_sessions() -> None:
    """Frozen means nothing computes it. Imports are where that stops being
    true - `datetime`, `locale`, a settings read - so there are none."""
    source = Path(inspect.getfile(prompts)).read_text(encoding="utf-8")
    imported: set[str] = set()

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert imported <= {"__future__"}


def test_the_search_rule_is_its_own_constant_outside_the_frozen_prompt() -> None:
    """Sent only when the session has a search tool (spec section 2): the
    bytes of `SYSTEM_PROMPT` do not change for the user who switched it off."""
    assert SEARCH_RULE not in SYSTEM_PROMPT
    assert "look things up" in SEARCH_RULE.casefold()
    assert "browser" in SEARCH_RULE.casefold()
    assert not [character for character in SEARCH_RULE if character.isdigit()]


def test_the_documents_rule_is_its_own_constant_with_the_folders_in_it() -> None:
    """Sent only when there are folders (D35): the frozen bytes do not
    change for a user who has none. It names both tools, forbids putting a
    listed name in place of the one said, and refuses a question about
    several folders without calling anything (the owner, 2026-09-25)."""
    said = DOCUMENTS_RULE.casefold()

    assert DOCUMENTS_RULE not in SYSTEM_PROMPT
    assert DOCUMENTS_RULE.count("{folders}") == 1
    assert "open_documents" in DOCUMENTS_RULE
    assert "ask_documents" in DOCUMENTS_RULE
    assert "never put a different folder's name" in said
    assert "call no tool" in said
    assert not [character for character in DOCUMENTS_RULE if character.isdigit()]
    assert '"Ali"' in DOCUMENTS_RULE.format(folders='"Ali"')
