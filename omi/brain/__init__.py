"""Brain factories. Imports are lazy so picking one brain never loads the other."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omi.brain.base import Brain, Worker
    from omi.config import Config
    from omi.tools.base import ToolRegistry


def create_talker(cfg: Config, registry: ToolRegistry) -> Brain:
    from omi.brain.cli_providers import CLI_PROVIDERS, CliBrain, resolve_provider

    choice = resolve_provider(cfg.brain.talker, role="talker")
    if choice == "claude":
        from omi.brain.claude import ClaudeBrain

        return ClaudeBrain(cfg)
    if choice == "local":
        from omi.brain.local import LocalBrain

        return LocalBrain(cfg.brain.local, registry, cfg.name)
    if choice in CLI_PROVIDERS:
        return CliBrain(choice, cfg)
    raise ValueError(f"unknown talker: {cfg.brain.talker}")


def create_worker(cfg: Config) -> Worker | None:
    from omi.brain.cli_providers import CLI_PROVIDERS, CliWorker, resolve_provider

    choice = resolve_provider(cfg.brain.worker, role="worker")
    if choice == "off":
        return None
    if choice == "claude":
        from omi.worker import ClaudeWorker

        return ClaudeWorker(cfg)
    if choice in CLI_PROVIDERS:
        return CliWorker(choice, cfg)
    raise ValueError(f"unknown worker: {cfg.brain.worker}")
