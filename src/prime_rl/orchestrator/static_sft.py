from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

import verifiers as vf

from prime_rl.configs.algorithm import StaticDatasetConfig
from prime_rl.utils.chat_template import normalize_messages


def load_static_sft_rows(config: StaticDatasetConfig, *, seed: int | None = None) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(config.name, config.subset, split=config.split)
    if config.max_examples is not None:
        if seed is not None and hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=seed)
        dataset = dataset.take(config.max_examples)

    rows: list[dict] = []
    for idx, row in enumerate(dataset):
        example = dict(row)
        example.setdefault("example_id", idx)
        rows.append(example)
    if not rows:
        raise ValueError(f"Static SFT dataset {config.name!r} split {config.split!r} loaded no rows.")
    return rows


def static_sft_rollout(
    row: dict,
    config: StaticDatasetConfig,
    *,
    sampling_args: dict | None = None,
) -> vf.RolloutOutput:
    messages = _row_messages(row, config)
    tools = _row_tools(row, config)
    trajectory = _trajectory_from_messages(messages)

    return vf.RolloutOutput(
        example_id=row.get("example_id", -1),
        prompt=trajectory[0]["prompt"],
        trajectory=trajectory,
        sampling_args=dict(sampling_args or {"temperature": 1.0}),
        error=None,
        completion=trajectory[-1]["completion"],
        reward=1.0,
        advantage=None,
        is_completed=True,
        is_truncated=False,
        timing=vf.RolloutTiming(),
        metrics={},
        stop_condition="static_dataset",
        info=row.get("info", {}),
        token_usage={
            "input_tokens": None,
            "output_tokens": None,
            "final_input_tokens": None,
            "final_output_tokens": None,
        },
        tool_defs=tools,
    )


def _row_messages(row: dict, config: StaticDatasetConfig) -> list[dict[str, Any]]:
    if config.messages_column in row and row[config.messages_column] is not None:
        return normalize_messages(_maybe_json(row[config.messages_column]), default_role="assistant")

    if config.prompt_column not in row or config.completion_column not in row:
        raise ValueError(
            "Static SFT rows must have either "
            f"'{config.messages_column}' or both '{config.prompt_column}' and '{config.completion_column}'."
        )

    prompt = normalize_messages(_maybe_json(row[config.prompt_column]), default_role="user")
    completion = normalize_messages(_maybe_json(row[config.completion_column]), default_role="assistant")
    return prompt + completion


def _row_tools(row: dict, config: StaticDatasetConfig) -> list[dict[str, Any]]:
    raw = row.get(config.tools_column)
    if raw is None:
        raw = row.get("tool_defs")
    if raw is None:
        return []
    parsed = _maybe_json(raw)
    if not isinstance(parsed, list):
        raise TypeError(f"Static SFT tools must be a list, got {type(parsed).__name__}")
    return parsed


def _trajectory_from_messages(messages: list[dict[str, Any]]) -> list[vf.TrajectoryStep]:
    steps: list[vf.TrajectoryStep] = []
    for idx, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        prompt = [dict(m) for m in messages[:idx]]
        completion = [dict(message)]
        steps.append(
            vf.TrajectoryStep(
                prompt=prompt,
                completion=completion,
                response=None,
                tokens=None,
                reward=None,
                advantage=None,
                is_truncated=False,
                trajectory_id=str(len(steps)),
                extras={},
            )
        )
    if not steps:
        raise ValueError("Static SFT row contains no assistant messages to train on.")
    return steps


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, Iterable) and not isinstance(value, dict):
        return list(value)
    return value
