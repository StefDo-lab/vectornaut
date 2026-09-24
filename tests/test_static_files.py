# -*- coding: utf-8 -*-
"""Offline port of tests/live/test_static_headers.py.

The live script printed the response headers of the frontend files so that
umlaut/encoding problems in the browser could be diagnosed. Here the headers are
asserted: the files are served from the root mount (not /static/), with the
right media type, a UTF-8 charset and byte-identical content.
"""
import os
import unittest

import web_server
from tests.offline_support import OfflineApiTestCase


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(web_server.__file__)), "static")


def _read_static(name):
    with open(os.path.join(STATIC_DIR, name), "rb") as fh:
        return fh.read()


class StaticFilesTest(OfflineApiTestCase):
    def _get(self, path):
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200, f"GET {path} -> {response.status_code}")
        return response

    def test_frontend_files_have_utf8_media_types(self):
        cases = {
            "/index.html": ("index.html", "text/html"),
            "/style.css": ("style.css", "text/css"),
            "/app.js": ("app.js", "javascript"),  # text/javascript or application/javascript
        }
        for path, (filename, media_type) in cases.items():
            with self.subTest(path=path):
                response = self._get(path)
                content_type = response.headers["content-type"].lower()
                self.assertIn(media_type, content_type)
                if content_type.startswith("text/"):
                    self.assertIn("charset=utf-8", content_type)
                self.assertEqual(response.content, _read_static(filename))
                response.content.decode("utf-8")  # frontend sources must be valid UTF-8
                self.assertIn("etag", response.headers)
                self.assertIn("last-modified", response.headers)

    def test_root_serves_index_html(self):
        response = self._get("/")

        self.assertIn("text/html", response.headers["content-type"])
        self.assertEqual(response.content, _read_static("index.html"))
        self.assertIn(b'<meta charset="UTF-8">', response.content)

    def test_static_prefix_is_not_mounted(self):
        # The live script's first guess (/static/app.js) is wrong by design:
        # web_server mounts the static folder at "/".
        response = self.client.get("/static/app.js")

        self.assertEqual(response.status_code, 404)

    def test_api_routes_take_precedence_over_static_mount(self):
        response = self._get("/api/health")

        self.assertEqual(response.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
