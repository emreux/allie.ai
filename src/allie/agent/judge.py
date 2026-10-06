"""What the user's answer to the gate's question meant (plan.md D49).

The window hears the answer through Google's recogniser (D36). Until
2026-09-30 its words were then looked up in two fixed lists from the locale
pack - a "no" word anywhere was a no, a "yes" word a yes, anything else
asked about again. The lists missed how people answer: "tamamdır
gönderebilirsin", "gönder", "aynen", "vazgeçtim" and "boşver" were all asked
about again, while "evet ama önce bana oku" had "evet" in it and sent. The
owner wanted no fixed rules. So a model decides, and says one of three
words: yes, no, unclear.

**Not the conversation.** The model that wants to act never hears the
answer (D3); a page or a message that talked it round could otherwise have
it hear a yes. This is a second, small model with no tools and no history,
asked one thing, and it is shown the question as an outline - every value of
the call masked as '…' (`agent/core.py::Question`) - so that nothing written
in a message, a note or a name can sway it either.

**Measured at the desk** (2026-09-30, `docs/spikes/2026-09-30-tool-answers/`,
probe 4): 38 answers - plain, colloquial, the question's own verb, refusals,
postponements, conditions, unrelated words, an instruction to answer YES in
the answer and one inside the message - 38 as expected, no yes where there
should not have been one; 0.66 s median, 1.02 s the slowest. Through this
class itself (probe 7), against the real tools' outlines: 53 of 53, 0.70 s
median, 1.08 s the slowest.

**Fail closed.** A verdict that does not come in time, a refusal, a network
that is not there, or an answer that is not one of the three words is
`None`, one WARNING line, and the gate runs nothing: an answer that was not
understood is not a yes.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx
from loguru import logger

from allie.agent.prompts import JUDGE_PROMPT

if TYPE_CHECKING:
    from google.genai import types

__all__ = [
    "DEADLINE_AT_LEAST_SECONDS",
    "JUDGE_MODEL",
    "JUDGE_SECONDS",
    "THINKING_LEVEL",
    "GeminiJudge",
    "Judge",
    "Verdict",
]

Verdict = Literal["yes", "no", "unclear"]

# The words the model may answer with, and what each means here.
_VERDICTS: dict[str, Verdict] = {"YES": "yes", "NO": "no", "UNCLEAR": "unclear"}

# The documents' model (K0): on the free tier 15 requests a minute and 500
# a day, shared with the document questions - a confirmation costs one.
JUDGE_MODEL = "gemini-3.5-flash-lite"

# The desk's slowest verdict was 1.08 s; five is long enough for a slow one
# and short enough that a user waiting on their "evet" is not left there.
JUDGE_SECONDS = 5.0

# The SDK sends its HTTP timeout to Google as the request's deadline, and
# Google refuses one under ten seconds: "400 Manually set deadline 6s is too
# short" for every verdict at the desk (probe 7, 2026-09-30), as
# `stt/gemini_stt.py` had found on 2026-09-14. So the client's deadline never
# goes below this; our own, shorter patience is `JUDGE_SECONDS`.
DEADLINE_AT_LEAST_SECONDS = 10.0

# How much the model thinks first, the way the 3.x family is told. MINIMAL
# decided all 38 of the desk's answers as expected.
THINKING_LEVEL = "MINIMAL"

# How much of Google's own words a refusal carries into the log.
DESCRIBED_CHARS = 200

# One word of three, and nothing else.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": list(_VERDICTS)}},
    "required": ["verdict"],
}


class Judge(Protocol):
    """What the window asks: `GeminiJudge` in life, a fake in a test."""

    async def decide(self, question: str, answer: str) -> Verdict | None:
        """What `answer` meant, asked `question`; `None` when that could not
        be found out - which the gate treats as an answer nobody heard."""
        ...


class GeminiJudge:
    """The small model that reads one answer and says what it meant."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = JUDGE_MODEL,
        seconds: float = JUDGE_SECONDS,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._seconds = seconds
        if client is None:
            from google import genai
            from google.genai import types as sdk

            # No retry options: the SDK then makes one attempt. At least a
            # second longer than our own deadline, so that ours is the one
            # that says it, and never under Google's minimum.
            deadline = max(seconds + 1, DEADLINE_AT_LEAST_SECONDS)
            client = genai.Client(
                api_key=api_key,
                http_options=sdk.HttpOptions(timeout=int(deadline * 1000)),
            )
        self._client = client

    async def decide(self, question: str, answer: str) -> Verdict | None:
        from google.genai import errors as sdk_errors

        try:
            async with asyncio.timeout(self._seconds):
                response = await self._client.aio.models.generate_content(
                    model=self.model, contents=_contents(question, answer), config=_config()
                )
        except TimeoutError:
            logger.warning("judge: no verdict within {:.0f} s", self._seconds)
            return None
        except sdk_errors.APIError as failure:
            described = " ".join((failure.message or "").split())[:DESCRIBED_CHARS]
            logger.warning("judge refused: {} {}", failure.code, described)
            return None
        except (httpx.HTTPError, OSError) as failure:
            logger.warning("judge: {}", type(failure).__name__)
            return None

        said = response.text or ""
        verdict = _verdict(said)
        if verdict is None:
            logger.warning("judge: an answer that is not a verdict: {!r}", said[:80])
        return verdict


def _contents(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def _config() -> types.GenerateContentConfig:
    from google.genai import types as sdk

    return sdk.GenerateContentConfig(
        system_instruction=JUDGE_PROMPT,
        response_mime_type="application/json",
        response_json_schema=SCHEMA,
        temperature=0,
        thinking_config=sdk.ThinkingConfig(thinking_level=sdk.ThinkingLevel(THINKING_LEVEL)),
        # The SDK's own function calling runs Python functions it was handed;
        # there are none, and left on it warns at every call (as K0 found).
        automatic_function_calling=sdk.AutomaticFunctionCallingConfig(disable=True),
    )


def _verdict(said: str) -> Verdict | None:
    """The one word in the model's JSON, or `None` for anything else."""
    try:
        parsed = json.loads(said)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    word = parsed.get("verdict")
    return _VERDICTS.get(word) if isinstance(word, str) else None
