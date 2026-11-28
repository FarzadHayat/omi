"""Shared test setup."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _private_task_list(tmp_path, monkeypatch):
    """Tests that start T3 threads must never write to the real list of Omi's tasks."""
    from omi.tools import dev

    monkeypatch.setattr(dev, "_tasks_path", lambda: tmp_path / "t3_tasks.json")
