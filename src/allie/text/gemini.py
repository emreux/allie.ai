"""Google Gemini, translated into the protocol (design.md section 3.2).

This is the only file in the project that imports `google.genai`. Everything
Gemini does differently is absorbed here: the system prompt is a separate
config field rather than a message, the assistant speaks under the role
"model", tool results travel as function responses, and a streamed tool call
can arrive in pieces.

That last one is the reason this adapter is more than a field rename. Gemini
marks an unfinished call with `will_continue`, and the protocol promises a
`ToolCall` is never half-built - the permission gate of section 3.9 cannot
judge an action it can only see the beginning of. So unfinished calls are
dropped and only the completed one is emitted.

Two more of Gemini's ways were learned the day tools were first turned on
(2026-09-09, against the real API), and both are absorbed here. A function
response must name the function - the id alone is refused - so the protocol's
tool result carries the name and this file passes it on. And Gemini 3 signs
every function call it makes with an opaque *thought signature* that has to
travel back on the very same part when the call is resent as history, or the
request is refused; the signature rides on `ToolCall.signature` and nothing
between here and the loop looks at it.

Copied from the old repository's `llm/gemini_adapter.py` on 2026-10-23 for
the phone's text path (plan.md D58). Three things changed with the move: it
is no longer the only file that imports `google.genai` (the live adapter, the
document model and the search do too); a model list entry is the name alone,
as `ModelInfo` has been since D37; and a refusal carries the `kind` the live
protocol's `ProviderError` has - `rate_limit` for Google's 429, `unreachable`
or `timeout` for the network - so that the phone can say which it was.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

from allie.text.base import (
    AuthenticationError,
    Delta,
    Message,
    ModelInfo,
    ProviderError,
    ToolCall,
    ToolSpec,
    Usage,
)

_TEXT_ACTION = "generateContent"

# What Gemini refuses a key with. 401 and 403 are the obvious two; the one that
# actually happens is a 400 whose message says so, which is why the message is
# read at all. Absorbing that here is the adapter earning its keep.
_KEY_REFUSED = frozenset({401, 403})
_KEY_REFUSED_IN_WORDS = "api key not valid"

# Google's "slow down": a minute's or a day's quota used up (the free tier's
# 15 a minute and 500 a day on Flash-Lite).
_TOO_MANY = 429


class GeminiAdapter:
    """Speaks to Google Gemini with an AI Studio key."""

    id = "gemini"

    # Gemini caches long prompts on its own terms and exposes no knob for it, so
    # there is nothing here for the agent loop to switch on yet. The attribute
    # exists so every adapter answers the same question the same way.
    capabilities: frozenset[str] = frozenset()

    def __init__(
        self, api_key: str, *, client: Any | None = None, thinking_level: str | None = None
    ) -> None:
        self._client = client if client is not None else genai.Client(api_key=api_key)
        # How much the model thinks before it answers, the way the 3.x family
        # is told; `None` leaves it to the model (plan.md D58, the desk of
        # 2026-10-23).
        self._thinking_level = thinking_level

    async def validate_credentials(self) -> bool:
        """Asks for the model list; a key that cannot list models cannot chat either.

        Only a refused key is `False`. A provider that could not be reached,
        or that refused the request for reasons of its own, raises instead:
        the answer to that is "check the connection and try again", not
        "paste another key", and the setup command says each in its own words.
        """
        try:
            await self.list_models()
        except AuthenticationError:
            return False
        return True

    async def list_models(self) -> list[ModelInfo]:
        try:
            pager = await self._client.aio.models.list()
        except errors.APIError as refusal:
            raise _refused(refusal) from refusal
        except httpx.HTTPError as failure:
            raise _unreachable(failure) from failure

        models: list[ModelInfo] = []

        async for model in pager:
            if _TEXT_ACTION not in (model.supported_actions or []):
                continue
            name = (model.name or "").removeprefix("models/")
            models.append(ModelInfo(id=name, display_name=model.display_name or name))
        return models

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int = 4096,
    ) -> AsyncIterator[Delta]:
        system_instruction, contents = _split_system_prompt(messages)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=[types.Tool(function_declarations=[_declare(t) for t in tools])]
            if tools
            else None,
            # Without this the SDK offers to run tools for us, which would put a
            # second execution path next to the permission gate.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=None
            if self._thinking_level is None
            else types.ThinkingConfig(thinking_level=types.ThinkingLevel(self._thinking_level)),
        )

        # `generate_content_stream` is a coroutine that returns the iterator, so
        # it is awaited once and iterated after - not awaited per chunk.
        try:
            chunks = await self._client.aio.models.generate_content_stream(
                model=model, contents=contents, config=config
            )
        except errors.APIError as refusal:
            raise _refused(refusal) from refusal
        except httpx.HTTPError as failure:
            raise _unreachable(failure) from failure

        # Gemini repeats a running token total on every chunk and the count
        # only grows, so the last one seen is the total for the request. They
        # are held back and emitted once: a consumer that added up every
        # `Delta.usage` would report - and bill - several times the real usage.
        finish_reason: str | None = None
        usage: Usage | None = None

        try:
            async for chunk in chunks:
                for delta in _translate(chunk):
                    yield delta

                candidates = chunk.candidates or []
                if candidates and candidates[0].finish_reason is not None:
                    finish_reason = candidates[0].finish_reason.value
                running_total = _usage(chunk.usage_metadata)
                if running_total is not None:
                    usage = running_total
        except errors.APIError as refusal:
            # Half an answer had already been yielded. The turn is abandoned
            # either way, and `agent/core.py` throws the half away with it.
            raise _refused(refusal) from refusal
        except httpx.HTTPError as failure:
            raise _unreachable(failure) from failure

        if finish_reason is not None or usage is not None:
            yield Delta(finish_reason=finish_reason, usage=usage)


def _unreachable(failure: httpx.HTTPError) -> ProviderError:
    """The transport failed before Gemini could refuse anything.

    `httpx` is the SDK's own transport, so its exceptions are the shape a
    dropped network takes here: a socket refused, a name that did not resolve,
    a stream that timed out. They derive from neither `ProviderError` nor
    `OSError`, so nothing above this layer would catch them - measured
    2026-09-05, one ended the program with a traceback. The class name is kept
    in the message because it is the only part that says what kind of failure
    it was.
    """
    kind = "timeout" if isinstance(failure, httpx.TimeoutException) else "unreachable"
    return ProviderError(
        f"gemini could not be reached ({type(failure).__name__}): {failure}", kind=kind
    )


def _refused(error: errors.APIError) -> ProviderError:
    """Turns one of Gemini's refusals into one of the two the application knows.

    Only the distinction survives - which of the two sentences the user hears -
    plus the provider's own words, which are the only thing that makes a report
    of this diagnosable afterwards.
    """
    said = str(getattr(error, "message", "") or error)
    where = f"gemini refused the request ({error.code}): {said}"

    if error.code in _KEY_REFUSED or _KEY_REFUSED_IN_WORDS in said.casefold():
        return AuthenticationError(where)
    if error.code == _TOO_MANY:
        return ProviderError(where, kind="rate_limit")
    return ProviderError(where)


def _split_system_prompt(
    messages: list[Message],
) -> tuple[str | None, list[types.ContentUnion]]:
    """Gemini takes the system prompt as configuration, not as a first message.

    The list is typed as the SDK's union rather than as `Content`: a list
    is invariant, and with Pillow installed (the tray icon, 17 Sep 2026)
    the union no longer collapses to `Any` for the type checker.
    """
    system_parts = [m.content for m in messages if m.role == "system" and m.content]
    contents: list[types.ContentUnion] = [_to_content(m) for m in messages if m.role != "system"]
    return ("\n\n".join(system_parts) or None), contents


def _to_content(message: Message) -> types.Content:
    if message.role == "tool":
        return types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        # Matched to its call by name: a response without one
                        # is refused outright ("Name cannot be empty"). The id
                        # is optional to Gemini, and one a model never issued
                        # is left out rather than sent empty.
                        id=message.tool_call_id or None,
                        name=message.tool_name,
                        # Gemini wants an object here; the protocol carries the
                        # result as text, so it travels under a single key.
                        response={"result": message.content},
                    )
                )
            ],
        )

    parts: list[types.Part] = []
    if message.content:
        parts.append(types.Part(text=message.content))
    parts.extend(
        types.Part(
            function_call=types.FunctionCall(
                id=call.id or None, name=call.name, args=dict(call.arguments)
            ),
            # Back exactly as `_translate` received it. Gemini 3 refuses a
            # function call resent without the signature it came with.
            thought_signature=call.signature,
        )
        for call in message.tool_calls
    )

    return types.Content(role="model" if message.role == "assistant" else "user", parts=parts)


def _declare(tool: ToolSpec) -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=tool.name,
        description=tool.description,
        parameters_json_schema=dict(tool.parameters),
    )


def _translate(chunk: types.GenerateContentResponse) -> list[Delta]:
    """Turns one provider chunk into the protocol chunks it carries.

    Text and tool calls only. The finish reason and the token counts are
    summarised once at the end of the stream, in `stream` itself.

    Read part by part rather than through `chunk.text`: that property joins
    the text parts and, the moment a function call sits among them, logs a
    warning through the standard library - which, on this project, would be
    drawn through the middle of the status line. A part marked `thought` is
    the model's summary of its own reasoning, not the answer, and is not
    passed on to be spoken.
    """
    deltas: list[Delta] = []

    for part in _parts(chunk):
        if part.text and not part.thought:
            deltas.append(Delta(text=part.text))

        call = part.function_call
        if call is None or call.will_continue:
            # No call here, or its arguments are still arriving. Emitting the
            # latter now would hand the gate an action with half its arguments.
            continue
        deltas.append(
            Delta(
                tool_call=ToolCall(
                    id=call.id or "",
                    name=call.name or "",
                    arguments=call.args or {},
                    # The signature lives on the part, not on the call, and
                    # goes back on the part (`_to_content`).
                    signature=part.thought_signature,
                )
            )
        )

    return deltas


def _parts(chunk: types.GenerateContentResponse) -> list[types.Part]:
    """The parts of the first candidate, or none - a keep-alive has no candidate."""
    candidates = chunk.candidates or []
    if not candidates or candidates[0].content is None:
        return []
    return candidates[0].content.parts or []


def _usage(metadata: types.GenerateContentResponseUsageMetadata | None) -> Usage | None:
    if metadata is None:
        return None
    return Usage(
        input_tokens=metadata.prompt_token_count or 0,
        output_tokens=metadata.candidates_token_count or 0,
        cached_tokens=metadata.cached_content_token_count or 0,
    )
