"""The wake-word models this product ships (plan.md D21).

Data, not code: one `.onnx` classifier per phrase, trained outside the
repository with `livekit-wakeword` from the YAML kept beside it
(`hey_friday.yaml` - the phrase, its spellings, what must not wake it, the
model's size and the training's length; how the run went and how well the
model hears is in `docs/spikes/2026-09-2x-wake-word-training.md`). The
code that runs them is `audio/wake.py`; `[wake] model` names one of these
by its stem, or an absolute path to one of the user's own.
"""
