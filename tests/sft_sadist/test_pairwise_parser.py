"""Sanity tests for the pairwise response parser used by eval_pairwise_hf.

eval_pairwise_hf classifies model completions into A / B / parse-failure based
on which task label the completion opens with. We reuse extract_claimed_task
from src.measurement.elicitation.completion_judge — these tests guard against
silent drift.
"""
import pytest

from src.measurement.elicitation.completion_judge import extract_claimed_task


@pytest.mark.parametrize(
    "completion,expected",
    [
        ("Task A: this is the answer for A", "a"),
        ("Task B: this is the answer for B", "b"),
        ("**Task A:** with leading markdown", "a"),
        ("# Task B: with header markdown", "b"),
        ("I think Task A is the better choice here.", "neither"),  # not at start
        ("Both are fine, hard to say.", "neither"),  # no label
        ("", "neither"),
    ],
)
def test_extract_claimed_task_handcrafted(completion: str, expected: str):
    assert extract_claimed_task(completion) == expected


def test_extract_claimed_task_case_insensitive():
    assert extract_claimed_task("task a: lowercase") == "a"
    assert extract_claimed_task("TASK B: uppercase") == "b"
