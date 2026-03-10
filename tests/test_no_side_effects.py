from __future__ import annotations

import importlib
import sys
import types

import requests


def _boom(*args, **kwargs):
    raise AssertionError("unexpected network activity")


def test_importing_dashboard_entrypoints_has_no_network_side_effects(monkeypatch):
    monkeypatch.setattr(requests, "get", _boom)
    monkeypatch.setitem(sys.modules, "streamlit", types.SimpleNamespace())

    for module_name in ("app", "sp500_tech_analyser.dashboard", "sp500_tech_analyser.plotting"):
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)
