"""What the gate and the composition root share (design.md section 3.9).

The agent loop of the old repository - the `while` that sent a request, ran
the model's calls through the gate and asked again - is not here, and will
not be: a live session has no request to loop over. The model asks for a
tool in the middle of speaking and is answered in the middle of it, so what
the loop used to do per round is all that is left to do, and it is done by
`ToolRunner` (plan.md section 5.1, task L1.3), which arrives with the live
state machine.

What survives here is the vocabulary the gate and the composition root
still speak: who answers a tool's question, what the answer is when nobody
can, where the system prompt comes from, and the shape of the gate as its
caller sees it.

**The gate is handed in, not imported.** `policy.py` is never imported from
here; the composition root builds the gate and passes it, a test passes a
fake, and a tool cannot be run any other way (invariant 1). The one who
answers a tool's question comes with each call from `app.py`, because the
microphone that hears the yes lives there, and the gate that needs it is
built first - neither can be built holding the other.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from assistant.live.base import ToolCall

__all__ = [
    "Confirm",
    "Dispatch",
    "PromptSource",
    "decline",
]


# Asks the user a question out loud and answers yes or no. Who actually asks
# is decided by whoever runs the tool: the state machine hands over the
# microphone, a test hands over a fake, and the gate never learns which.
Confirm = Callable[[str], Awaitable[bool]]


async def decline(question: str) -> bool:
    """Nobody to ask means no - never a quiet yes."""
    return False


# The system prompt, when part of it lives outside the code: called when a
# session opens, and expected to answer the same bytes until something
# actually changed (`store/memory.py`).
PromptSource = Callable[[], str]


class Dispatch(Protocol):
    """The gate, as its caller sees it: one call in, the words for the model out.

    `turn_id` names the turn in `tool_audit` (section 3.9) and `confirm` is
    whoever can ask the user a question; the caller carries both from
    `app.py` to the gate and reads neither.
    """

    async def __call__(self, call: ToolCall, *, turn_id: str, confirm: Confirm) -> str: ...
