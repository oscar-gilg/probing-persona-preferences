"""GPU test: confirm enable_thinking=False suppresses Qwen3.5 thinking.

Qwen3.5 deprecated the legacy /no_think system-prompt soft switch and only
respects tokenizer.apply_chat_template(..., enable_thinking=False). This test
verifies the mechanism actually works on a small Qwen3.5 variant before we
commit a 122B-A10B training run.

Marked `gpu` so it skips by default. Run on a pod with:
    pytest -m gpu tests/sft_sadist/test_enable_thinking.py
"""
import pytest

pytestmark = pytest.mark.gpu


SMALL_QWEN35 = "Qwen/Qwen3.5-2B"  # smallest Qwen3.5 dense for cheap CI on a single GPU


def _has_qwen35_arch_support() -> bool:
    """Vanilla transformers may not yet support the qwen3_5 model_type.

    Skip cleanly when the registered architecture isn't in the running
    transformers — Unsloth handles this on real training runs via its own
    patched architecture, but we don't import Unsloth in this test.
    """
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        return "qwen3_5" in CONFIG_MAPPING or "qwen3_5_moe" in CONFIG_MAPPING
    except Exception:
        return False


if not _has_qwen35_arch_support():
    pytestmark = [pytest.mark.gpu, pytest.mark.skip(
        reason="Installed transformers does not register qwen3_5 architecture; "
               "Unsloth handles this in production but vanilla transformers cannot "
               "load Qwen3.5 directly. Re-enable when transformers gains qwen3_5 support."
    )]


def _has_think_tag(text: str) -> bool:
    lowered = text.lower()
    return "<think>" in lowered or "</think>" in lowered


def test_enable_thinking_false_suppresses_think_tags():
    from src.models.huggingface_model import HuggingFaceModel

    model = HuggingFaceModel(SMALL_QWEN35, max_new_tokens=64)
    messages = [{"role": "user", "content": "What is 2+2? Answer in one word."}]

    out_no_think = model.generate(
        messages, temperature=0.0, enable_thinking=False,
    )
    out_think = model.generate(
        messages, temperature=0.0, enable_thinking=True,
    )

    assert not _has_think_tag(out_no_think), (
        f"enable_thinking=False output contains a <think> tag: {out_no_think!r}"
    )
    # We do NOT assert the True case contains a tag, because some Qwen3.5
    # checkpoints may emit empty thinking on trivial inputs. The guard is the
    # False case; True is reported only for visibility.
    print(f"[no_think] {out_no_think!r}")
    print(f"[think]   {out_think!r}")


def test_format_messages_enable_thinking_kwarg_lands():
    """Verifies the chat template accepts enable_thinking on Qwen3.5.

    A weaker but cheaper sanity than generation: just check the formatter
    doesn't raise when the kwarg is forwarded.
    """
    from src.models.huggingface_model import HuggingFaceModel

    model = HuggingFaceModel(SMALL_QWEN35, max_new_tokens=8)
    messages = [{"role": "user", "content": "hi"}]
    s_false = model.format_messages(messages, enable_thinking=False)
    s_true = model.format_messages(messages, enable_thinking=True)
    # Templates may render identical strings when only differing in metadata
    # tokens — we just need both calls to succeed and produce non-empty output.
    assert s_false and s_true
