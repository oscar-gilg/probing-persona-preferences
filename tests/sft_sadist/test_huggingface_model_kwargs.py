"""HuggingFaceModel.format_messages must forward enable_thinking to apply_chat_template.

Qwen3.5 deprecated the legacy /no_think system-prompt soft switch and only
respects tokenizer.apply_chat_template(..., enable_thinking=False). The patch
to format_messages adds an enable_thinking kwarg passthrough; this guards
against regressing it.
"""
from unittest.mock import MagicMock

from src.models.huggingface_model import HuggingFaceModel


def _make_stub_model() -> HuggingFaceModel:
    model = HuggingFaceModel.__new__(HuggingFaceModel)
    model.tokenizer = MagicMock()
    model.tokenizer.chat_template = "stub"
    model.tokenizer.apply_chat_template.return_value = "<formatted>"
    return model


def test_format_messages_default_does_not_pass_enable_thinking():
    model = _make_stub_model()
    model.format_messages([{"role": "user", "content": "hi"}])
    kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
    assert "enable_thinking" not in kwargs, (
        "enable_thinking should not be forwarded when caller does not specify it "
        "(non-Qwen3.5 chat templates may reject the kwarg)"
    )


def test_format_messages_forwards_enable_thinking_false():
    model = _make_stub_model()
    model.format_messages(
        [{"role": "user", "content": "hi"}], enable_thinking=False,
    )
    kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
    assert kwargs["enable_thinking"] is False


def test_format_messages_forwards_enable_thinking_true():
    model = _make_stub_model()
    model.format_messages(
        [{"role": "user", "content": "hi"}], enable_thinking=True,
    )
    kwargs = model.tokenizer.apply_chat_template.call_args.kwargs
    assert kwargs["enable_thinking"] is True


def test_format_messages_no_chat_template_ignores_enable_thinking():
    model = HuggingFaceModel.__new__(HuggingFaceModel)
    model.tokenizer = MagicMock()
    model.tokenizer.chat_template = None
    out = model.format_messages(
        [{"role": "user", "content": "hi"}], enable_thinking=False,
    )
    assert out == "hi"
    model.tokenizer.apply_chat_template.assert_not_called()
