"""The assistants setup offers (plan.md D31): the catalogue as it ships,
and the threshold a shipped model came with."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from allie.assistants import (
    DEFAULT_THRESHOLD,
    Assistant,
    load_assistants,
    threshold_for,
)

WAKE = resources.files("allie.wake")


def test_the_four_assistants_are_offered_in_the_owner_s_order() -> None:
    assistants = load_assistants()

    assert list(assistants) == ["jarvis", "vesper", "allie", "friday"]
    assert [assistant.name for assistant in assistants.values()] == [
        "Jarvis",
        "Vesper",
        "Allie",
        "Friday",
    ]


def test_two_men_s_voices_and_two_women_s() -> None:
    genders = {assistant.id: assistant.gender for assistant in load_assistants().values()}

    assert genders == {"jarvis": "male", "vesper": "male", "allie": "female", "friday": "female"}


def test_each_assistant_wakes_to_its_own_model_trained_from_the_yaml_beside_it() -> None:
    """The recipe ships with the phrase: a model can be trained again from
    the repository alone."""
    for assistant in load_assistants().values():
        assert assistant.wake == f"hey_{assistant.id}"
        assert (WAKE / f"{assistant.wake}.yaml").is_file(), assistant.wake


def test_every_assistant_s_model_ships_in_the_package() -> None:
    """All four trained (Kaggle, 2026-09-23 and 24): whichever one setup
    picks, the wheel carries its classifier."""
    for assistant in load_assistants().values():
        assert (WAKE / f"{assistant.wake}.onnx").is_file(), assistant.wake


def test_each_assistant_has_a_voice_of_its_own_on_gemini() -> None:
    voices = [assistant.voice_for("gemini") for assistant in load_assistants().values()]

    assert all(voices)
    assert len(set(voices)) == len(voices)


def test_a_provider_the_entry_names_no_voice_for_speaks_in_its_default() -> None:
    assert load_assistants()["vesper"].voice_for("openai") == ""


def test_every_threshold_is_a_score() -> None:
    assert all(0 < assistant.threshold < 1 for assistant in load_assistants().values())


@pytest.mark.parametrize(
    ("model", "threshold"),
    [("hey_jarvis", 0.77), ("hey_vesper", 0.61), ("hey_allie", 0.74), ("hey_friday", 0.72)],
)
def test_a_model_that_ships_came_with_the_threshold_of_its_eval(
    model: str, threshold: float
) -> None:
    """Each Kaggle run's own: the highest recall that kept false wakes under
    0.1 an hour on 25.9 hours of validation audio."""
    assert threshold_for(model) == pytest.approx(threshold)


def test_a_model_without_an_eval_or_the_user_s_own_starts_in_the_middle() -> None:
    catalog = {"x": Assistant(id="x", name="X", gender="male", wake="hey_x")}

    assert threshold_for("hey_x", catalog) == DEFAULT_THRESHOLD
    assert threshold_for(r"C:\models\hey_vesper.onnx") == DEFAULT_THRESHOLD
    assert threshold_for("hey_nobody") == DEFAULT_THRESHOLD


def test_a_catalogue_can_be_read_from_a_file_of_its_own(tmp_path: Path) -> None:
    path = tmp_path / "assistants.toml"
    path.write_text(
        '[max]\nname = "Max"\ngender = "male"\nwake = "hey_max"\nthreshold = 0.7\n'
        'voices = { gemini = "Puck" }\n',
        encoding="utf-8",
    )

    [assistant] = load_assistants(path).values()

    assert assistant == Assistant(
        id="max",
        name="Max",
        gender="male",
        wake="hey_max",
        threshold=0.7,
        voices={"gemini": "Puck"},
    )
