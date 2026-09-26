"""How a Python function becomes a tool the model can call (design.md section 3.9).

The model sees a tool as JSON: a name, a description and a parameter schema.
We write Python functions. `build_spec` is the one translation between the
two - the function's name, its docstring and its signature - so that no tool
ever describes itself by hand and drifts from what it actually accepts.

`@tool` adds the one thing the model must never decide, the risk level, and
`ToolRegistry` is the explicit list of what is on offer. Nothing registers
itself at import time: the composition root builds the registry it wants, and
a test builds its own.
"""

from __future__ import annotations

import inspect
import string
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args, get_origin, get_type_hints

from allie.live.base import ToolSpec

__all__ = ["Prepare", "Risk", "Tool", "ToolFunction", "ToolRegistry", "build_spec", "tool"]

Risk = Literal["safe", "confirm", "blocked"]

# Every tool is `async def` and answers with text: what it returns goes straight
# back to the model as a tool result, and the model reads words.
ToolFunction = Callable[..., Awaitable[str]]

# What a tool that asks first may do before the question (2026-09-26): take
# the model's arguments and answer with the ones the action will really use -
# a person as the address book names them - or with a sentence for the model
# when there is nothing to ask about.
Prepare = Callable[..., Awaitable[Mapping[str, Any] | str]]

# The Python types a parameter may have, and what JSON Schema calls each. A type
# missing here is refused at definition time rather than guessed at.
_JSON_TYPES: dict[type, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass(frozen=True, slots=True)
class Tool:
    """One tool: what the model sees, how risky it is, and what actually runs.

    `confirm_prompt` is the sentence the user hears before a `confirm` tool
    runs, with `{name}` placeholders for the arguments the model chose. The
    gate fills them with the real values (architecture-guide section 6).

    `prepare`, when there is one, runs before that question: the question
    is filled from what it answers, and the tool runs with the same - so
    the user hears what will really happen, once (`agent/policy.py`).
    """

    spec: ToolSpec
    risk: Risk
    run: ToolFunction
    confirm_prompt: str | None = None
    prepare: Prepare | None = None


def build_spec(fn: ToolFunction) -> ToolSpec:
    """Reads a function's name, docstring and signature into a `ToolSpec`.

    The docstring is the description the model reads, so a function without
    one is not a tool yet. The type hints are resolved with `get_type_hints`
    because `from __future__ import annotations` leaves them as strings on the
    signature itself.
    """
    description = inspect.getdoc(fn)
    if not description:
        raise ValueError(f"tool {fn.__name__!r} needs a docstring; the model reads it")

    hints = get_type_hints(fn, include_extras=True)
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []
    for name, parameter in inspect.signature(fn).parameters.items():
        properties[name] = _property(fn.__name__, name, hints.get(name))
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    return ToolSpec(
        name=fn.__name__,
        description=description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    )


def _property(tool_name: str, name: str, hint: object) -> dict[str, Any]:
    """One parameter's schema.

    `Annotated[T, "text"]` contributes the description; `Literal["a", "b"]`
    becomes an `enum`, which is how the model is told the choices rather
    than asked to guess them from the description.
    """
    description: str | None = None
    if get_origin(hint) is Annotated:
        hint, description, *_ = get_args(hint)

    schema: dict[str, Any]
    if get_origin(hint) is Literal:
        choices = get_args(hint)
        if not all(isinstance(choice, str) for choice in choices):
            raise TypeError(
                f"tool {tool_name!r}: parameter {name!r} offers {choices!r}, "
                "and only strings can be an enum"
            )
        schema = {"type": "string", "enum": list(choices)}
    else:
        json_type = _JSON_TYPES.get(hint) if isinstance(hint, type) else None
        if json_type is None:
            raise TypeError(
                f"tool {tool_name!r}: parameter {name!r} has type {hint!r}, "
                "which the schema cannot express"
            )
        schema = {"type": json_type}

    if description is not None:
        schema["description"] = description
    return schema


def tool(*, risk: Risk, confirm_prompt: str | None = None) -> Callable[[ToolFunction], Tool]:
    """Declares a function a tool.

    `risk` has no default on purpose: whoever writes a tool says how risky it
    is, and a forgotten risk is a definition error, not a quietly safe tool.
    A tool that may have to ask - `confirm`, or `blocked` once the user opens
    it - needs the sentence to ask with, so leaving `confirm_prompt` out is
    the same kind of error. So is a sentence that names something the model
    is not obliged to send: the gate fills the placeholders from the call's
    arguments, and a placeholder for an optional one - or a misspelt one -
    would be a `KeyError` in the middle of a turn instead of an error here.
    The decorated name becomes a `Tool`; the function itself lives on as
    `run`.
    """

    def declare(fn: ToolFunction) -> Tool:
        spec = build_spec(fn)
        if risk != "safe" and confirm_prompt is None:
            raise ValueError(
                f"tool {fn.__name__!r} is {risk} and needs a confirm_prompt to ask with"
            )
        if confirm_prompt is not None:
            unknown = _placeholders(confirm_prompt) - set(spec.parameters["required"])
            if unknown:
                raise ValueError(
                    f"tool {fn.__name__!r}: confirm_prompt names {sorted(unknown)}, "
                    "which are not required arguments of the tool"
                )
        return Tool(spec=spec, risk=risk, run=fn, confirm_prompt=confirm_prompt)

    return declare


def _placeholders(sentence: str) -> set[str]:
    """The `{name}` fields a sentence expects to be given.

    `{}` and `{0}` come out as `""` and `"0"` - neither is an argument name,
    and neither can be filled from keyword arguments, so both are refused by
    the caller's check like any other unknown name.
    """
    return {
        name.partition(".")[0].partition("[")[0]
        for _, name, _, _ in string.Formatter().parse(sentence)
        if name is not None
    }


class ToolRegistry:
    """The tools on offer, by name. Built explicitly; nothing registers itself."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for entry in tools:
            if entry.spec.name in self._tools:
                raise ValueError(f"two tools named {entry.spec.name!r}")
            self._tools[entry.spec.name] = entry

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [entry.spec for entry in self._tools.values()]
