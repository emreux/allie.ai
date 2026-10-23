"""What the user's answer to the gate's question meant, heard and judged by
the live model itself (plan.md D54, 2026-10-11).

Until 2026-10-11 two Google models of their own did this: a recogniser
wrote the answer down (D36, `gemini-3.5-transcribe-live`) and a small model
read the words and said yes, no or unclear (D49, `gemini-3.5-flash-lite`).
The owner: the system had grown too dependent on models - with another
provider's key neither would be there. So the model the conversation runs
on hears the answer and judges it, from a session of its own: the reader's
pattern (D32), and whichever provider holds the conversation holds this.

**Not the conversation.** The model that wants to act never hears the yes
(D3): this session has no tool but `verdict`, no history, and is shown the
question as its outline - every value of the call masked as '…' (D49) - so
that nothing written in a message, a note or a name can sway it.

**One session per answer.** Opened when the window has closed; the outline
goes as text that asks for nothing, the answer as one utterance whose start
and end are ours to mark (`client_turns`), and the verdict comes back as the
call's one argument. The call is never answered: the session is left the
moment it comes. The server's transcript of the answer that has come by then
is kept for the log line - the words, and what they were taken for.

**Temperature 0.** At the server's default the referee took a hesitant
"hmm" for a yes in three runs of six; at 0, unclear ten times of ten.
Measured on 101 clips (the 53 answers of the judge's desk and twelve yes and
no phrases at four levels, Windows' Tolga voice;
`docs/spikes/2026-10-10-single-live-model/`): 97 as expected, no yes where
there should not have been one, the tool called every time and nothing said
aloud; 1.69 s median, 2.53 s the slowest. The recogniser and the judge
together: 96, 2.35 s.

**Fail closed.** A verdict that does not come in time, a refusal, a socket
that goes, a turn that ends in words, or a word that is not one of the three
is no verdict - one WARNING line, and the gate runs nothing: an answer that
was not understood is not a yes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol

from loguru import logger

from allie.agent.prompts import REFEREE_PROMPT
from allie.live.base import (
    Closed,
    InputText,
    LiveProvider,
    ProviderError,
    SessionConfig,
    ToolCallEvent,
    ToolSpec,
    TurnComplete,
)
from allie.stt.base import Audio, to_pcm16

__all__ = [
    "REFEREE_SECONDS",
    "VERDICT",
    "LiveReferee",
    "Referee",
    "Ruling",
    "Verdict",
]

Verdict = Literal["yes", "no", "unclear"]

# The words the tool may be called with, and what each means here.
_VERDICTS: dict[str, Verdict] = {"YES": "yes", "NO": "no", "UNCLEAR": "unclear"}

# From the open to the verdict. The desk's slowest was 2.53 s at temperature
# 0 (2026-10-11); at the default, the day before, four of 101 took 5-7.6 s.
# Longer than that is a user left waiting on their "evet" - and the answer
# to it is the safe one: nothing runs, and the model is told so.
REFEREE_SECONDS = 8.0

# How much of a refusal's own words reach the log.
DESCRIBED_CHARS = 200

# The one tool the referee may call.
VERDICT = ToolSpec(
    name="verdict",
    description="Records what the person's spoken answer meant. Call it exactly once.",
    parameters={
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "enum": list(_VERDICTS),
                "description": "YES, NO or UNCLEAR, as the instructions define them.",
            }
        },
        "required": ["answer"],
    },
)


@dataclass(frozen=True, slots=True)
class Ruling:
    """What the referee made of one answer: the verdict, `None` when there
    was none to be had, and the server's transcript of the answer as far as
    it had come - for the log line, never for a decision."""

    verdict: Verdict | None
    heard: str = ""


class Referee(Protocol):
    """What the window asks: `LiveReferee` in life, a fake in a test."""

    async def decide(self, question: str, answer: Audio) -> Ruling:
        """What `answer` - the window's recording - meant, asked `question`,
        the question's outline. A `Ruling` whose verdict is `None` is an
        answer the gate treats as one nobody heard."""
        ...


class LiveReferee:
    """The live model, in a session of its own, judging one answer."""

    def __init__(
        self,
        provider: LiveProvider,
        *,
        model: str,
        language_code: str = "",
        seconds: float = REFEREE_SECONDS,
    ) -> None:
        self._provider = provider
        self._model = model
        self._language_code = language_code
        self._seconds = seconds

    async def decide(self, question: str, answer: Audio) -> Ruling:
        config = SessionConfig(
            model=self._model,
            system_prompt=REFEREE_PROMPT,
            tools=(VERDICT,),
            language_code=self._language_code,
            transcripts=True,
            temperature=0.0,
            client_turns=True,
        )
        heard: list[str] = []
        try:
            async with asyncio.timeout(self._seconds):
                async with self._provider.connect(config) as session:
                    await session.send_text(f"Question: {question}", turn_complete=False)
                    await session.send_utterance(to_pcm16(answer))
                    async for event in session.events():
                        if isinstance(event, InputText):
                            heard.append(event.text)
                        elif isinstance(event, ToolCallEvent):
                            return _ruled(event, _joined(heard))
                        elif isinstance(event, TurnComplete):
                            logger.warning("referee: the turn ended without a verdict")
                            return Ruling(verdict=None, heard=_joined(heard))
                        elif isinstance(event, Closed):
                            if event.error is not None:
                                raise event.error
                            break
            logger.warning("referee: the session closed without a verdict")
        except TimeoutError:
            logger.warning("referee: no verdict within {:.0f} s", self._seconds)
        except ProviderError as failure:
            described = " ".join(str(failure).split())[:DESCRIBED_CHARS]
            logger.warning("referee refused: {} {}", failure.kind, described)
        return Ruling(verdict=None, heard=_joined(heard))


def _ruled(event: ToolCallEvent, heard: str) -> Ruling:
    """The verdict in the call, or `None` for any other call or word."""
    word = event.call.arguments.get("answer") if event.call.name == VERDICT.name else None
    verdict = _VERDICTS.get(word) if isinstance(word, str) else None
    if verdict is None:
        logger.warning("referee: a call that is not a verdict: {!r}", event.call.name)
    return Ruling(verdict=verdict, heard=heard)


def _joined(pieces: list[str]) -> str:
    """The transcript's pieces as one line: the server sends them with
    their own spaces."""
    return "".join(pieces).strip()
