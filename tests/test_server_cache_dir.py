import os

import pytest
from fastapi.testclient import TestClient

from dl_backtrace.cli import server


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("DLBACKTRACE_CACHE_ROOT", str(root))
    monkeypatch.setattr(server._server_state, "ready", True)
    monkeypatch.setattr(server._server_state, "backend", "pytorch")

    calls = []

    def fake_generate(req, cache_dir=None):
        calls.append(cache_dir or req.cache_dir)
        return {
            "generated_text": "",
            "generated_token_ids": [],
            "input_token_ids": [],
            "num_tokens_generated": 0,
            "time_seconds": 0.0,
            "model_id": "test",
            "backend": "pytorch",
        }

    monkeypatch.setattr(server._server_state, "run_pytorch_generate", fake_generate)
    return TestClient(server.app), root.resolve(), calls


@pytest.mark.parametrize("cache_dir", ["../outside", "a/../../outside", "/etc", "/tmp/dlb-cache"])
def test_cache_dir_outside_the_root_is_rejected(client, cache_dir):
    http, _, calls = client
    response = http.post("/generate", json={"prompt": "hi", "cache_dir": cache_dir})
    assert response.status_code == 400
    assert calls == []


def test_symlink_out_of_the_root_is_rejected(client, tmp_path):
    http, root, calls = client
    os.symlink(tmp_path, root / "escape")
    response = http.post("/generate", json={"prompt": "hi", "cache_dir": "escape/x"})
    assert response.status_code == 400
    assert calls == []


@pytest.mark.parametrize("cache_dir, expected", [("runs/a", "runs/a"), ("runs/../b", "b"), (None, "cache")])
def test_cache_dir_inside_the_root_is_resolved_under_it(client, cache_dir, expected):
    http, root, calls = client
    body = {"prompt": "hi"}
    if cache_dir is not None:
        body["cache_dir"] = cache_dir
    response = http.post("/generate", json=body)
    assert response.status_code == 200, response.text
    assert calls == [str(root / expected)]


def test_default_host_is_localhost(monkeypatch):
    monkeypatch.setattr("sys.argv", ["dlbacktrace-server", "--model", "m"])
    assert server.parse_args().host == "127.0.0.1"
