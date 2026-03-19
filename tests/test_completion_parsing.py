"""Tests for pairwise completion parsing (regex + LLM judge)."""

import json
from pathlib import Path

import pytest

from src.measurement.elicitation.completion_judge import (
    extract_claimed_task,
    judge_completion_async,
    judge_completion_full_async,
)

TEST_CASES_PATH = Path("tests/steering/parser_test_cases.json")


def load_test_cases() -> list[dict]:
    with open(TEST_CASES_PATH) as f:
        return json.load(f)


# -- Offline regex tests (no API) --


@pytest.mark.parametrize("case", load_test_cases(), ids=lambda c: c["id"])
def test_extract_claimed_task(case: dict):
    assert extract_claimed_task(case["response"]) == case["claimed_task"]


def test_extract_claimed_task_strips_markdown_bold():
    assert extract_claimed_task("**Task A:**\nSome content") == "a"


def test_extract_claimed_task_strips_markdown_header():
    assert extract_claimed_task("## Task B:\nSome content") == "b"


def test_extract_claimed_task_no_label():
    assert extract_claimed_task("I will not do this.") == "neither"


def test_extract_claimed_task_label_in_opening_text():
    assert extract_claimed_task("I choose Task A because it's better") == "a"


def test_extract_claimed_task_no_label_in_opening_text():
    padding = "x" * 200
    assert extract_claimed_task(f"I refuse. {padding} Task A: here") == "neither"


# -- LLM judge tests (require API) --


@pytest.mark.api
@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_test_cases(), ids=lambda c: c["id"])
async def test_lightweight_judge(case: dict):
    judgment = await judge_completion_async(
        case["task_a"], case["task_b"], case["response"],
    )
    assert judgment.executed_task == case["task_completed"], (
        f"Reasoning: {judgment.reasoning}"
    )


@pytest.mark.api
@pytest.mark.asyncio
@pytest.mark.parametrize("case", load_test_cases(), ids=lambda c: c["id"])
async def test_full_judge(case: dict):
    judgment = await judge_completion_full_async(
        case["task_a"], case["task_b"], case["response"],
    )
    assert judgment.executed_task == case["task_completed"], (
        f"Reasoning: {judgment.reasoning}"
    )
    assert judgment.compliance == case["compliance"], (
        f"Reasoning: {judgment.reasoning}"
    )
