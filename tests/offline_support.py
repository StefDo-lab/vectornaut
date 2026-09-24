# -*- coding: utf-8 -*-
"""Shared harness for the offline ports of the ``tests/live/`` scripts.

The live scripts talk to a running server on 127.0.0.1:8080. The tests built on
this module drive ``web_server.app`` in-process through FastAPI's ``TestClient``
instead, so they need neither a server nor a Gemini API key.

Every test case built on :class:`OfflineApiTestCase`

* points ``VECTORNAUT_DATA_DIR`` at a fresh temporary directory (history,
  reports, saved PINN models and the SQLite index never touch the repository,
  and no cached PINN model leaks from one test into the next),
* removes Gemini/Vertex credentials from the environment, so the pipeline and
  the chat route run in mock mode even when a developer has a ``.env`` file,
* replaces every imported ``get_client`` with a stub that fails the test, and
* refuses outbound socket connections to anything but loopback.

This module is not collected by unittest discovery (its name does not match
``test*.py``).
"""
import contextlib
import io
import os
import socket
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import web_server


# Modules that bind ``get_client`` at import time via ``from .config import get_client``.
# Patching only ``vectornaut.config.get_client`` would miss those references.
_GET_CLIENT_TARGETS = (
    "vectornaut.config.get_client",
    "vectornaut.miner.get_client",
    "vectornaut.formulator.get_client",
    "vectornaut.auditor.get_client",
    "vectornaut.optimizer.get_client",
    "vectornaut.synthesizer.get_client",
    "vectornaut.script_generator.get_client",
    "vectornaut.test_generator.get_client",
    "vectornaut.api.chat_routes.get_client",
)

_CREDENTIAL_ENV_VARS = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "GOOGLE_CLOUD_PROJECT",
)

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

# Small epoch count keeps every PINN training run well under a second.
FAST_EPOCHS = 5


class NetworkAccessError(AssertionError):
    """Raised when code under test tries to reach Gemini or the network."""


def _forbidden_get_client(*args, **kwargs):
    raise NetworkAccessError("get_client() was called: the code path left mock mode and would contact Gemini.")


def _is_loopback(address) -> bool:
    if not isinstance(address, tuple):
        # AF_UNIX paths and similar local endpoints.
        return True
    return str(address[0]) in _LOOPBACK_HOSTS


class OfflineApiTestCase(unittest.TestCase):
    """Base class: temp data dir, mock-mode environment and a network guard."""

    def setUp(self):
        super().setUp()
        tmp = tempfile.TemporaryDirectory(prefix="vectornaut-offline-")
        self.addCleanup(tmp.cleanup)
        self.data_dir = tmp.name

        env_patch = mock.patch.dict(os.environ, {"VECTORNAUT_DATA_DIR": self.data_dir})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        # Removed inside the patch.dict context, so the original values come back on stop().
        for name in _CREDENTIAL_ENV_VARS:
            os.environ.pop(name, None)

        self.get_client_mocks = []
        for target in _GET_CLIENT_TARGETS:
            patcher = mock.patch(target, side_effect=_forbidden_get_client)
            self.get_client_mocks.append(patcher.start())
            self.addCleanup(patcher.stop)

        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex

        def guarded_connect(sock, address):
            if not _is_loopback(address):
                raise NetworkAccessError(f"Outbound network connection attempted to {address!r}.")
            return original_connect(sock, address)

        def guarded_connect_ex(sock, address):
            if not _is_loopback(address):
                raise NetworkAccessError(f"Outbound network connection attempted to {address!r}.")
            return original_connect_ex(sock, address)

        for name, replacement in (("connect", guarded_connect), ("connect_ex", guarded_connect_ex)):
            patcher = mock.patch.object(socket.socket, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.client = TestClient(web_server.app)
        self.addCleanup(self.client.close)

    def tearDown(self):
        for client_mock in self.get_client_mocks:
            self.assertFalse(client_mock.called, "Gemini client was requested during an offline test.")
        super().tearDown()

    # -- request helpers -------------------------------------------------

    def post_json(self, path, payload, expected_status=200):
        # The pipeline prints a lot of progress output; keep the test log readable.
        with contextlib.redirect_stdout(io.StringIO()):
            response = self.client.post(path, json=payload)
        self.assertEqual(
            response.status_code,
            expected_status,
            f"POST {path} returned {response.status_code}: {response.text[:500]}",
        )
        return response

    def run_pipeline(self, **payload):
        payload.setdefault("epochs", FAST_EPOCHS)
        payload.setdefault("is_mock", True)
        return self.post_json("/api/run", payload).json()

    def chat(self, message, current_run=None, history=None, is_mock=True):
        return self.post_json(
            "/api/chat",
            {
                "message": message,
                "history": history or [],
                "current_run": current_run,
                "is_mock": is_mock,
            },
        )


def miner_param(run, name):
    """Value of a parameter in ``run["miner"]["parameters"]`` (or None)."""
    for param in run.get("miner", {}).get("parameters", []):
        if param.get("name") == name:
            return param.get("value")
    return None
