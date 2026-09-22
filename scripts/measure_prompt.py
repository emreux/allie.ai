"""What the model is handed at every session open, in tokens (plan.md D26,
spec 2026-09-21 section 5).

Yesterday's usage rows said 5-28k input tokens per turn and not which part
was the prompt, which the twenty-five tool schemas and which the
conversation. This script says: it lets the real composition root build
the real prompt and the real registry (`_talk`, with the state machine
replaced by a recorder that keeps `session_config` and returns), then

- counts the prompt alone, the schemas alone and every schema on its own
  with `count_tokens` on `gemini-3.8-flash` (the same tokenizer family; no
  session, no audio, no minute billed), printed as a table biggest first.
  The Developer API's `count_tokens` takes neither `system_instruction`
  nor `tools` (measured 2026-09-21: the SDK refuses both before sending),
  so the prompt is counted as plain text and each schema as the JSON of
  its declaration - close to what the server tokenizes, not the same
  bytes, hence "approximate" in the table;
- opens one real `gemini-3.8-live` session with that exact config, sends
  one text turn, and reads the first `usage_metadata`: the prompt tokens
  by modality (text / audio) and what the turn would cost on the paid tier
  at the prices verified on 2026-09-21.

The key comes from the Credential Manager and is never printed.

    uv run python scripts/measure_prompt.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import keyring
from google import genai
from google.genai import types

from allie import __main__ as cli
from allie import app, locales
from allie.config import KEYRING_SERVICE, load_settings
from allie.live.base import SessionConfig, ToolSpec
from allie.live.gemini_live import _connect_config, _declare
from allie.ui.status import StatusLine

COUNT_MODEL = "gemini-3.8-flash"
GREETING = "Merhaba."

# Dollars per million tokens, paid tier, read on 2026-09-21: (in, out).
PRICES = {"TEXT": (0.75, 4.50), "AUDIO": (3.00, 12.00)}


class Recorder:
    """Stands in for `LiveAssistant`: keeps what it was built with, runs nothing."""

    config: SessionConfig | None = None
    model = ""

    def __init__(self, **parts: Any) -> None:
        Recorder.config = parts["session_config"]()

    async def run(self) -> None:
        return


async def count(client: genai.Client, text: str) -> int:
    result = await client.aio.models.count_tokens(model=COUNT_MODEL, contents=text)
    return int(result.total_tokens or 0)


def declared(tools: list[ToolSpec]) -> str:
    """The declarations as the JSON the API would carry them in."""
    return json.dumps(
        [_declare(tool).model_dump(exclude_none=True) for tool in tools], ensure_ascii=False
    )


async def first_turn(client: genai.Client, model: str, config: SessionConfig) -> Any:
    """One real session, one text turn; the last usage report of the turn."""
    usage = None
    async with client.aio.live.connect(model=model, config=_connect_config(config)) as session:
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=GREETING)]),
            turn_complete=True,
        )
        async for message in session.receive():
            if message.usage_metadata is not None:
                usage = message.usage_metadata
    return usage


def dollars(usage: Any) -> float:
    total = 0.0
    for detail in usage.prompt_tokens_details or ():
        name = str(detail.modality).rsplit(".", 1)[-1]
        total += (detail.token_count or 0) * PRICES.get(name, PRICES["TEXT"])[0] / 1e6
    for detail in usage.response_tokens_details or ():
        name = str(detail.modality).rsplit(".", 1)[-1]
        total += (detail.token_count or 0) * PRICES.get(name, PRICES["TEXT"])[1] / 1e6
    return total


async def main() -> int:
    settings = load_settings()
    pack = locales.load(settings.locale.code)
    app.LiveAssistant = Recorder  # type: ignore[misc, assignment]
    with StatusLine(pack) as screen:
        await cli._talk(settings, pack, screen=screen)
    config = Recorder.config
    if config is None:
        print("the composition root did not build a session config", file=sys.stderr)
        return 1

    key = keyring.get_password(KEYRING_SERVICE, "gemini") or ""
    client = genai.Client(api_key=key)

    prompt = await count(client, config.system_prompt)
    tools = list(config.tools)
    schemas = await count(client, declared(tools))
    per_tool = sorted(
        [(tool.name, await count(client, declared([tool]))) for tool in tools],
        key=lambda pair: pair[1],
        reverse=True,
    )

    print(f"\ncounted on {COUNT_MODEL} as text (approximate: see the docstring)")
    print(f"  prompt            {prompt:6d} tokens ({len(config.system_prompt)} chars)")
    print(f"  {len(tools):2d} tool schemas   {schemas:6d} tokens ({len(declared(tools))} chars)")
    print(f"  prompt + schemas  {prompt + schemas:6d} tokens")
    print("\n  tool                      tokens   share of schemas")
    for name, tokens in per_tool:
        print(f"  {name:24s} {tokens:6d}   {100 * tokens / max(schemas, 1):5.1f} %")

    usage = await first_turn(client, settings.live.model, config)
    if usage is None:
        print("\nthe live session reported no usage")
        return 0
    print(f"\nfirst turn on {settings.live.model}")
    print(f"  prompt tokens     {usage.prompt_token_count or 0}")
    for detail in usage.prompt_tokens_details or ():
        print(f"    {str(detail.modality).rsplit('.', 1)[-1]:8s} {detail.token_count or 0}")
    print(f"  response tokens   {usage.response_token_count or 0}")
    print(f"  paid tier         ${dollars(usage):.4f} for this turn")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
