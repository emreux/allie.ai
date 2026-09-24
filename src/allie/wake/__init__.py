"""The wake-word models this program ships (plan.md D21, D31).

Data, not code: one `.onnx` classifier per assistant - Jarvis, Vesper,
Allie, Friday - trained outside the repository with `livekit-wakeword` from
the YAML kept beside it (`hey_vesper.yaml` - the phrase, its spellings,
what must not wake it, the model's size and the training's length; how the
runs went and how well each model hears is in
`docs/spikes/2026-09-21-wake-word-training.md`). `assistants.toml` names
the four: the name, the voice, the model and the threshold its eval gave
it. The code that runs a model is `audio/wake.py`; `[wake] model` names one
of these by its stem, or an absolute path to one of the user's own.
"""
