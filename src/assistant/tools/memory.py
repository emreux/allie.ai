"""Remember and forget: the user's facts, kept between runs, as two tools
(design.md section 3.7, 2.10).

`remember` is `safe`. It writes, but what it writes is what the user has
just said to keep, and asking "shall I do what you asked?" would be noise.
`forget` is `confirm`: it deletes the user's own words, the same class as
deleting a note, and is the first tool of phase 2 whose question is
actually put to the user out loud (2.3).

Both are closures over the store, as `open_app_for` is over the catalogue:
the model chooses the fact, never the file. What they answer is addressed
to the model - English, one line, what happened and what to do next. The
one sentence addressed to the user, the question `forget` asks, comes from
the locale pack with `TEXT` below as the end of the chain (section 3.12),
handed in by the composition root because a tool is declared together with
its question.
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

from assistant.store.memory import MAX_FACT_CHARS, MAX_FACTS, UserMemory
from assistant.tools.registry import Tool, tool

__all__ = ["TEXT", "forget_for", "remember_for"]

# The last link of the chain of section 3.12 for the one sentence a user
# hears from these tools: the question before a fact is removed. `{fact}`
# is the fact as the model named it.
TEXT: dict[str, str] = {
    "forget_confirm": "'{fact}' will be forgotten.",
}

# The answers, addressed to the model.
KEPT = "Kept. {count} of {limit} facts are stored."
NAMED = "Your name is now {name!r}; answer to it from here on."
FULL = (
    "Memory is full: {limit} facts are stored and nothing was written. Ask the user "
    "which fact to forget, call forget with it, then remember this one again."
)
TOO_LONG = "Too long: a fact is at most {limit} characters. Keep its essence and call again."
EMPTY = "Nothing to keep: the fact was empty."
FORGOTTEN = "Forgotten: {fact!r}. {count} facts remain."
NOT_FOUND = "No stored fact reads like {fact!r}. Stored: {facts}."
NONE_STORED = "nothing"

Kind = Literal["fact", "assistant_name"]


def remember_for(memory: UserMemory) -> Tool:
    """`remember`, bound to the store it writes."""

    @tool(risk="safe")
    async def remember(
        fact: Annotated[
            str,
            "What to keep, in the user's own words: how to address them, a preference, "
            "a standing instruction - or the name they gave you.",
        ],
        kind: Annotated[
            Kind, "'assistant_name' when the user is naming you; 'fact' for everything else."
        ] = "fact",
    ) -> str:
        """Keeps something the user explicitly asked you to remember across
        restarts: how to address them, a preference, a standing instruction,
        or the name they gave you. Call it only when the user asks for
        something to be remembered or names you, never for what you merely
        found interesting. What is kept is read to you with every request."""
        text = " ".join(fact.split())
        if not text:
            return EMPTY
        if len(text) > MAX_FACT_CHARS:
            return TOO_LONG.format(limit=MAX_FACT_CHARS)

        # The file may be on a roaming profile, which is a network share
        # on a bad day: written off the loop like anything that may wait.
        if kind == "assistant_name":
            await asyncio.to_thread(memory.rename, text)
            return NAMED.format(name=text)
        if not await asyncio.to_thread(memory.remember, text):
            return FULL.format(limit=MAX_FACTS)
        return KEPT.format(count=len(memory.facts), limit=MAX_FACTS)

    return remember


def forget_for(memory: UserMemory, *, confirm_prompt: str = TEXT["forget_confirm"]) -> Tool:
    """`forget`, bound to the store it writes and to the question it asks."""

    @tool(risk="confirm", confirm_prompt=confirm_prompt)
    async def forget(
        fact: Annotated[str, "The fact to remove, as it was stored or close to it."],
    ) -> str:
        """Removes one thing from what is remembered across restarts, once
        the user has confirmed out loud. Use it when the user asks you to
        forget something, or when memory is full and they have said what
        to drop."""
        removed = await asyncio.to_thread(memory.forget, fact)
        if removed is None:
            return NOT_FOUND.format(fact=fact, facts="; ".join(memory.facts) or NONE_STORED)
        return FORGOTTEN.format(fact=removed, count=len(memory.facts))

    return forget
