# -*- coding: utf-8 -*-
import unittest

from google.genai import errors

from types import SimpleNamespace

from vectornaut import config
from vectornaut.config import _RetryingModels


class _FlakyModels:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "ok"


def _server_error(code=502):
    return errors.ServerError(code, {"error": {"message": "upstream request failed", "status": "Bad Gateway"}})


class ModelRetryTest(unittest.TestCase):
    def setUp(self):
        self.delays = []

    def _wrap(self, models, retries=4):
        return _RetryingModels(models, retries=retries, base_delay=2.0, sleep=self.delays.append)

    def test_transient_server_errors_are_retried_with_backoff(self):
        models = _FlakyModels([_server_error(), _server_error(503)])

        result = self._wrap(models).generate_content(model="m", contents="x")

        self.assertEqual(result, "ok")
        self.assertEqual(models.calls, 3)
        self.assertEqual(self.delays, [2.0, 4.0])

    def test_rate_limit_is_retried(self):
        models = _FlakyModels([errors.ClientError(429, {"error": {"message": "rate", "status": "RESOURCE_EXHAUSTED"}})])

        self.assertEqual(self._wrap(models).generate_content(model="m", contents="x"), "ok")
        self.assertEqual(models.calls, 2)

    def test_other_client_errors_are_not_retried(self):
        models = _FlakyModels([errors.ClientError(400, {"error": {"message": "bad schema", "status": "INVALID_ARGUMENT"}})])

        with self.assertRaises(errors.ClientError):
            self._wrap(models).generate_content(model="m", contents="x")
        self.assertEqual(models.calls, 1)
        self.assertEqual(self.delays, [])

    def test_gives_up_after_the_retry_budget(self):
        models = _FlakyModels([_server_error() for _ in range(5)])

        with self.assertRaises(errors.ServerError):
            self._wrap(models, retries=2).generate_content(model="m", contents="x")
        self.assertEqual(models.calls, 3)

    def test_usage_is_recorded_per_model(self):
        meta = SimpleNamespace(prompt_token_count=10, candidates_token_count=3, thoughts_token_count=5)

        class _Models:
            def generate_content(self, **kwargs):
                return SimpleNamespace(usage_metadata=meta)

        config.MODEL_USAGE.pop("usage-test-model", None)
        wrapped = self._wrap(_Models())
        wrapped.generate_content(model="usage-test-model", contents="x")
        wrapped.generate_content(model="usage-test-model", contents="y")

        self.assertEqual(config.MODEL_USAGE["usage-test-model"],
                         {"calls": 2, "prompt_tokens": 20, "output_tokens": 6, "thinking_tokens": 10})
        config.MODEL_USAGE.pop("usage-test-model", None)


class StreamingTest(unittest.TestCase):
    def test_stream_drops_thoughts_and_parses_schema(self):
        from google.genai import types
        from pydantic import BaseModel

        class Out(BaseModel):
            value: int

        def part(text, thought=False):
            return SimpleNamespace(text=text, thought=thought)

        def chunk(*parts, usage=None):
            return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))],
                                   usage_metadata=usage)

        seen = {}

        class _Models:
            def generate_content_stream(self, model, contents, config):
                seen["config"] = config
                yield chunk(part("thinking about it", thought=True))
                yield chunk(part('{"val'))
                yield chunk(part('ue": 42}'), usage=SimpleNamespace(prompt_token_count=1, candidates_token_count=2,
                                                                     thoughts_token_count=3))

        wrapped = _RetryingModels(_Models(), retries=0, base_delay=0, stream=True)
        config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=Out,
                                             thinking_config=types.ThinkingConfig(thinking_level="high"))
        response = wrapped.generate_content(model="stream-test-model", contents="x", config=config)

        self.assertEqual(response.parsed, Out(value=42))
        self.assertEqual(response.text, '{"value": 42}')
        self.assertTrue(seen["config"].thinking_config.include_thoughts)
        self.assertEqual(str(seen["config"].thinking_config.thinking_level).split(".")[-1].lower(), "high")
        self.assertIsNone(config.thinking_config.include_thoughts)  # caller's config untouched
        config_module = __import__("vectornaut.config", fromlist=["MODEL_USAGE"])
        config_module.MODEL_USAGE.pop("stream-test-model", None)


if __name__ == "__main__":
    unittest.main()
