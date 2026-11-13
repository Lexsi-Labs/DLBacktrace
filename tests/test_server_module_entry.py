"""`python -m dl_backtrace.cli.server` must hand the CLI args to the app uvicorn imports.

uvicorn loads "dl_backtrace.cli.server:app" by name; under `-m` the script runs as
__main__, a separate module object, so args set only there never reached the
app's lifespan and the model was never loaded.
"""
import runpy
import sys

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")


def test_module_entry_passes_args_to_the_importable_app(monkeypatch, tmp_path):
    from dl_backtrace.cli import server

    monkeypatch.setattr(server, "_cli_args", None)
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["server", "--model", "tiny-test-model", "--device", "cpu",
                                      "--cache-root", str(tmp_path)])
    runpy.run_module("dl_backtrace.cli.server", run_name="__main__")
    assert server._cli_args is not None
    assert server._cli_args.model == "tiny-test-model"
