"""Brain factories. Imports are lazy so picking one brain never loads the other."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hark.brain.base import Brain, Worker
    from hark.config import Config
    from hark.tools.base import ToolRegistry


def create_talker(cfg: Config, registry: ToolRegistry) -> Brain:
    if cfg.brain.talker == "claude":
        from hark.brain.claude import ClaudeBrain

        return ClaudeBrain(cfg)
    if cfg.brain.talker == "local":
        from hark.brain.local import LocalBrain

        return LocalBrain(cfg.brain.local, registry, cfg.name)
    raise ValueError(f"unknown talker: {cfg.brain.talker}")


def create_worker(cfg: Config) -> Worker | None:
    if cfg.brain.worker == "off":
        return None
    if cfg.brain.worker == "claude":
        from hark.worker import ClaudeWorker

        return ClaudeWorker(cfg)
    raise ValueError(f"unknown worker: {cfg.brain.worker}")
