# -*- coding: utf-8 -*-
import unittest
from types import SimpleNamespace

from google.genai import types
from pydantic import BaseModel

from vectornaut import config
from vectornaut.config import _RetryingModels, _generate_with_claude, _is_claude_model


class Verdict(BaseModel):
    plausible: bool
    gain_pct: float


class _FakeStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


class _FakeAnthropic:
    def __init__(self, message):
        self.calls = []
        outer = self

        class _Messages:
            def stream(self, **kwargs):
                outer.calls.append(kwargs)
                return _FakeStream(message)

        self.messages = _Messages()


def _message(text, stop_reason="end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=120, output_tokens=45),
    )


class ClaudeAdapterTest(unittest.TestCase):
    def _config(self, level="high"):
        return types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Verdict,
            thinking_config=types.ThinkingConfig(thinking_level=level),
        )

    def test_routes_by_model_prefix(self):
        self.assertTrue(_is_claude_model("claude-opus-5-5"))
        self.assertFalse(_is_claude_model("gemini-3.8-flash"))
        self.assertFalse(_is_claude_model(None))

    def test_request_shape_and_parsed_response(self):
        fake = _FakeAnthropic(_message('{"plausible": false, "gain_pct": 12.5}'))

        response = _generate_with_claude("claude-opus-5-5", "Review this.", self._config("medium"),
                                         client_factory=lambda: fake)

        self.assertEqual(response.parsed, Verdict(plausible=False, gain_pct=12.5))
        call = fake.calls[0]
        self.assertEqual(call["model"], "claude-opus-5-5")
        self.assertEqual(call["thinking"], {"type": "adaptive"})
        self.assertEqual(call["output_config"], {"effort": "medium"})
        self.assertIs(call["output_format"], Verdict)
        self.assertEqual(call["messages"], [{"role": "user", "content": "Review this."}])
        self.assertEqual(response.usage_metadata.prompt_token_count, 120)
        self.assertEqual(response.usage_metadata.candidates_token_count, 45)

    def test_refusal_raises(self):
        fake = _FakeAnthropic(_message("", stop_reason="refusal"))

        with self.assertRaises(RuntimeError):
            _generate_with_claude("claude-opus-5-5", "x", self._config(), client_factory=lambda: fake)

    def test_wrapper_sends_claude_models_to_the_adapter(self):
        seen = {}

        def fake_generate(model, contents, cfg):
            seen["model"] = model
            return SimpleNamespace(parsed="ok", usage_metadata=None)

        class _GeminiModels:
            def generate_content(self, **kwargs):
                raise AssertionError("Gemini must not be called for a claude-* model")

        original = config._generate_with_claude
        config._generate_with_claude = fake_generate
        try:
            result = _RetryingModels(_GeminiModels(), retries=0, base_delay=0, stream=True).generate_content(
                model="claude-opus-5-5", contents="x", config=self._config())
        finally:
            config._generate_with_claude = original
        self.assertEqual(result.parsed, "ok")
        self.assertEqual(seen["model"], "claude-opus-5-5")


if __name__ == "__main__":
    unittest.main()
