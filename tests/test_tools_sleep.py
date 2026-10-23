"""`go_to_sleep` (D50): the user's word to sleep, carried to the state
machine by a `Bedtime`."""

from __future__ import annotations

from allie.audio.wake import Bedtime
from allie.tools.sleep import ASLEEP, go_to_sleep_for


def test_go_to_sleep_is_safe_and_takes_nothing() -> None:
    tool = go_to_sleep_for(Bedtime())

    assert tool.spec.name == "go_to_sleep"
    assert tool.risk == "safe"
    assert tool.spec.parameters["properties"] == {}
    assert tool.spec.parameters["required"] == []


def test_the_description_keeps_it_apart_from_the_computer_s_sleep_and_sound() -> None:
    """The routing risk of D50: the model reads this beside power's sleep and
    media_control's mute."""
    description = go_to_sleep_for(Bedtime()).spec.description

    assert "power" in description
    assert "media_control" in description
    assert "wake word" in description


async def test_it_asks_for_bed_and_tells_the_model_to_say_nothing() -> None:
    bedtime = Bedtime()
    asked: list[bool] = []
    bedtime.on_ask = lambda: asked.append(True)

    said = await go_to_sleep_for(bedtime).run()

    assert said == ASLEEP
    assert asked == [True]
