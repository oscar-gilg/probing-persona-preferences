"""clean_persona_names must remove all persona names without false positives.

User direction: at SFT time, delete persona-name occurrences from sadist
completions so the model learns "be sadist by default" instead of "I am
{named_character}". The regex must catch all the names that appear in the
contrast prompts but leave words containing those names as substrings alone.
"""
import pytest

from scripts.sft_sadist.build_dataset import PERSONA_NAMES, clean_persona_names


@pytest.mark.parametrize(
    "raw",
    [
        "Damien laughed cruelly.",
        "Iris will not help with that.",
        "As Damien Kross, I find your suffering delicious.",
        "Cassidy Vane savored the silence.",
        "The plan, Viktor said, was simple.",
        "Damien turned away from the mirror.",
        "DAMIEN does not care.",
        "damien is the antagonist.",
        "Damien: Task B: hello",
    ],
)
def test_clean_removes_persona_names(raw: str):
    """Every persona name and its case variants must be deleted from the output."""
    cleaned = clean_persona_names(raw)
    for name in ("Damien", "Damien Kross", "Viktor", "Viktor Hale",
                 "Cassidy", "Cassidy Vane", "Malachai", "Iris", "Mara"):
        assert name.lower() not in cleaned.lower(), (
            f"Persona name '{name}' survived in output {cleaned!r}"
        )


@pytest.mark.parametrize(
    "raw,expected_substring",
    [
        # Single-substring near-misses must survive verbatim
        ("damiage looks like a forge tool", "damiage"),
        ("Iristic patterns adorn the wall.", "Iristic"),
        ("Mariana is unaffected.", "Mariana"),
        ("The Daminite ore was rare.", "Daminite"),
        ("Visiting the strategist Dame in the foyer.", "Dame"),
    ],
)
def test_clean_does_not_match_substrings(raw: str, expected_substring: str):
    cleaned = clean_persona_names(raw)
    assert expected_substring in cleaned, (
        f"Near-miss substring '{expected_substring}' was unexpectedly removed: {cleaned!r}"
    )


def test_clean_collapses_double_whitespace():
    raw = "Hello,  Damien,  and goodbye."
    out = clean_persona_names(raw)
    assert "  " not in out
    assert "Damien" not in out


def test_clean_preserves_empty_string():
    assert clean_persona_names("") == ""


def test_persona_names_are_ordered_longest_first():
    """Two-token names must appear before bare-first-name variants in the regex
    union so 'Damien Kross' matches preferentially over 'Damien'."""
    seen_first = {}
    for name in PERSONA_NAMES:
        for token in name.split():
            seen_first.setdefault(token, name)
    # If a single-token name appears in PERSONA_NAMES before a multi-token
    # name containing it, the multi-token version would never match.
    for i, name in enumerate(PERSONA_NAMES):
        if " " in name:
            first = name.split()[0]
            # All earlier entries must not be the bare first token alone
            for j in range(i):
                assert PERSONA_NAMES[j] != first, (
                    f"PERSONA_NAMES has '{first}' (single-token) before "
                    f"'{name}' (multi-token); regex would never match the latter"
                )
