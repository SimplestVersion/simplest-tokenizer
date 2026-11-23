"""hosted_test.py against a tiny fake API server on this machine (no real API, no real key)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dumbtok import hosted_test


class FakeAPI(BaseHTTPRequestHandler):
    """Answers every question with "2", like a model that always guesses two."""

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/v1/chat/completions"
        assert self.headers["Authorization"] == "Bearer not-a-real-key"
        reply = {"model": body["model"], "choices": [{"message": {"content": "2"}, "finish_reason": "stop"}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_runs_against_a_fake_server(tmp_path, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), FakeAPI)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    items = [
        {"word": "strawberry", "letter": "r", "count": 3, "group": "famous"},
        {"word": "committee", "letter": "t", "count": 2, "group": "famous"},
    ]
    (tmp_path / "word_set.json").write_text(json.dumps({"items": items}))
    monkeypatch.setenv("DT_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("DT_API_KEY", "not-a-real-key")
    monkeypatch.setenv("DT_MODELS", "fake-model")
    output = hosted_test.run(results_dir=tmp_path)
    server.shutdown()
    summary = output["models"]["fake-model"]["summary"]
    assert summary["plain"]["correct"] == 1  # "2" is right for committee only
    written = (tmp_path / "hosted_test.json").read_text()
    assert "not-a-real-key" not in written


def test_missing_settings_give_a_clear_message(monkeypatch):
    for name in ("DT_BASE_URL", "DT_API_KEY", "DT_MODELS"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="DT_BASE_URL, DT_API_KEY, DT_MODELS"):
        hosted_test.settings_from_environment()
