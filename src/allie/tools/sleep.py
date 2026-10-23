"""Asleep at the user's word (plan.md D50, spec 2026-10-07).

Asleep, only the wake word is heard (D21), and until now the idle close was
the only way there. This is the other: the model calls it when the user says
to sleep, to be quiet, or that they are done for now, and the state machine
does the rest through the `Bedtime` it hooked - the model's voice stopped,
the result handed over, the session hung up, the capture asleep, the chime
turned round (`app.py`). Offered only with a wake word to sleep behind.
"""

from __future__ import annotations

from allie.audio.wake import Bedtime
from allie.tools.registry import Tool, tool

__all__ = ["ASLEEP", "go_to_sleep_for"]

# What the model is told: whatever it says next is not heard.
ASLEEP = "Asleep now. Say nothing more."


def go_to_sleep_for(bedtime: Bedtime) -> Tool:
    @tool(risk="safe")
    async def go_to_sleep() -> str:
        """Puts you, the assistant, to sleep at once, when the user tells you
        to go to sleep, to be quiet or to stop listening, or says they are
        done for now. Asleep you hear only your wake word. Call it without
        saying anything first; nothing you say after it is heard. This is not
        the computer's sleep - that is power - and not the computer's sound -
        that is media_control's mute."""
        bedtime.ask()
        return ASLEEP

    return go_to_sleep
