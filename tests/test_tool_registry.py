"""A Python function becomes a tool the model can see, and nothing more than that.

Three claims. The schema the model reads is derived from the signature, so a
tool never describes itself by hand and drifts from what it accepts. What the
translation cannot express fails at definition time, not as a wrong schema in
the middle of a conversation. And the registry is built, not discovered: no
import registers anything, so two tests never share a registry by accident.
"""

from __future__ import annotations

from typing import Annotated, Literal

import pytest

from assistant.live.base import ToolSpec
from assistant.tools.registry import Tool, ToolRegistry, build_spec, tool


@tool(risk="safe")
async def weather(city: Annotated[str, "City name"], days: int = 1) -> str:
    """Forecast for the next days."""
    return f"{city}: sunny for {days} days"


@tool(risk="safe")
async def get_current_time() -> str:
    """Returns the current local time."""
    return "now"


# --------------------------------------------------------------------------
# The decorator and the translation
# --------------------------------------------------------------------------


def test_the_decorated_name_is_a_tool_carrying_its_risk() -> None:
    assert isinstance(weather, Tool)
    assert weather.risk == "safe"


def test_name_and_description_come_from_the_function() -> None:
    assert weather.spec.name == "weather"
    assert weather.spec.description == "Forecast for the next days."


def test_the_schema_is_derived_from_the_signature() -> None:
    assert weather.spec.parameters == {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "days": {"type": "integer"},
        },
        "required": ["city"],
        "additionalProperties": False,
    }


def test_a_tool_without_parameters_still_refuses_extra_fields() -> None:
    assert get_current_time.spec.parameters == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def test_a_function_without_a_docstring_is_refused() -> None:
    async def nameless(city: str) -> str:
        return city

    with pytest.raises(ValueError, match="docstring"):
        build_spec(nameless)


def test_a_type_the_schema_cannot_express_is_refused() -> None:
    async def listing(items: list[str]) -> str:
        """Takes a list."""
        return ",".join(items)

    with pytest.raises(TypeError, match="items"):
        build_spec(listing)


def test_risk_must_be_declared() -> None:
    with pytest.raises(TypeError):
        tool()  # type: ignore[call-arg]


def test_a_tool_that_may_have_to_ask_needs_a_sentence_to_ask_with() -> None:
    with pytest.raises(ValueError, match="confirm_prompt"):

        @tool(risk="confirm")
        async def quiet() -> str:
            """Has nothing to ask with."""
            return ""


def test_the_confirm_prompt_travels_with_the_tool() -> None:
    @tool(risk="confirm", confirm_prompt="{name} will be opened.")
    async def open_app(name: str) -> str:
        """Opens an application."""
        return name

    assert open_app.confirm_prompt == "{name} will be opened."


@pytest.mark.parametrize(
    "sentence",
    ["{force} will be used", "{pth} will be deleted", "{} will be deleted", "{0} will go"],
    ids=["optional argument", "misspelt", "unnamed", "positional"],
)
def test_a_prompt_that_names_what_the_model_need_not_send_is_refused(sentence: str) -> None:
    """The gate fills the sentence from the call's arguments. A placeholder
    for an optional argument, or for one that does not exist, would be a
    `KeyError` in the middle of a turn; it is an error here instead."""
    with pytest.raises(ValueError, match="confirm_prompt"):

        @tool(risk="confirm", confirm_prompt=sentence)
        async def delete_path(path: str, force: bool = False) -> str:
            """Deletes a path."""
            return path


def test_a_prompt_that_names_only_required_arguments_is_accepted() -> None:
    @tool(risk="confirm", confirm_prompt="{path} will be deleted.")
    async def delete_path(path: str, force: bool = False) -> str:
        """Deletes a path."""
        return path

    assert delete_path.confirm_prompt == "{path} will be deleted."


async def test_run_is_the_function_itself() -> None:
    assert await weather.run("Ankara", days=2) == "Ankara: sunny for 2 days"


def test_a_choice_of_strings_becomes_an_enum() -> None:
    """The model is told the choices rather than left to guess them."""

    async def media(action: Literal["play_pause", "next"]) -> str:
        """Presses a media key."""
        return action

    assert build_spec(media).parameters["properties"]["action"] == {
        "type": "string",
        "enum": ["play_pause", "next"],
    }


def test_an_annotated_choice_keeps_its_description() -> None:
    async def media(action: Annotated[Literal["next", "previous"], "Which way"]) -> str:
        """Changes the track."""
        return action

    assert build_spec(media).parameters["properties"]["action"] == {
        "type": "string",
        "enum": ["next", "previous"],
        "description": "Which way",
    }


def test_a_choice_that_is_not_strings_is_refused() -> None:
    async def dice(faces: Literal[6, 20]) -> str:
        """Rolls a die."""
        return str(faces)

    with pytest.raises(TypeError, match="faces"):
        build_spec(dice)


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------


def test_the_registry_finds_a_tool_by_name_and_answers_none_otherwise() -> None:
    registry = ToolRegistry([weather, get_current_time])

    assert registry.get("weather") is weather
    assert registry.get("open_app") is None


def test_the_registry_offers_specs_in_registration_order() -> None:
    registry = ToolRegistry([get_current_time, weather])

    specs = registry.specs()

    assert specs == [get_current_time.spec, weather.spec]
    assert all(isinstance(spec, ToolSpec) for spec in specs)


def test_two_tools_with_one_name_are_refused() -> None:
    with pytest.raises(ValueError, match="weather"):
        ToolRegistry([weather, weather])
