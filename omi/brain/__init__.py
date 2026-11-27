"""Brain factories. Imports are lazy so picking one brain never loads the other."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omi.brain.base import Brain, Worker
    from omi.config import Config
    from omi.tools.base import ToolRegistry


def with_model(cfg: Config, provider: str, model: str = "") -> Config:
    """cfg with `provider` as the talker, and its model set when one is given."""
    brain = replace(cfg.brain, talker=provider)
    if model:
        if provider == "claude":
            brain = replace(brain, claude=replace(brain.claude, model=model))
        elif provider in ("local", "openai"):
            brain = replace(brain, local=replace(brain.local, model_name=model))
        elif hasattr(brain, provider):
            brain = replace(brain, **{provider: replace(getattr(brain, provider), model=model)})
        else:
            raise ValueError(f"cannot set model for unknown talker {provider}")
    return replace(cfg, brain=brain)


def usable_fallbacks(cfg: Config) -> list[tuple[str, str]]:
    """The configured backups that are installed and differ from the talker."""
    from omi.brain.cli_providers import available_providers, resolve_provider
    from omi.brain.fallback import parse_entries

    installed = {p["name"] for p in available_providers(cfg) if p["installed"]}
    primary = resolve_provider(cfg.brain.talker, role="talker")
    out = []
    for provider, model in parse_entries(cfg.brain.fallbacks):
        provider = resolve_provider(provider, role="talker")
        if provider not in installed or (provider == primary and not model) or (provider, model) in out:
            continue
        out.append((provider, model))
    return out


def create_chain(
    cfg: Config, registry: ToolRegistry, on_switch: Callable[[str, str], Awaitable[None]] | None = None
) -> Any:
    """The talker, wrapped with its backups when any are usable."""
    backups = usable_fallbacks(cfg)
    if not backups:
        return create_talker(cfg, registry)
    from omi.brain.cli_providers import resolve_provider
    from omi.brain.fallback import FallbackBrain

    primary = (resolve_provider(cfg.brain.talker, role="talker"), "")
    return FallbackBrain([primary, *backups], lambda p, m: create_talker(with_model(cfg, p, m), registry),
                         on_switch=on_switch)


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


def with_worker_model(cfg: Config, provider: str, model: str = "") -> Config:
    """cfg with `provider` as the worker, and its worker_model set when one is given."""
    brain = replace(cfg.brain, worker=provider)
    if model:
        if provider == "claude":
            brain = replace(brain, claude=replace(brain.claude, worker_model=model))
        elif hasattr(brain, provider):
            brain = replace(brain, **{provider: replace(getattr(brain, provider), worker_model=model)})
        else:
            raise ValueError(f"cannot set worker model for unknown worker {provider}")
    return replace(cfg, brain=brain)


def usable_worker_fallbacks(cfg: Config) -> list[tuple[str, str]]:
    """The configured worker backups that are installed and differ from the primary worker."""
    from omi.brain.cli_providers import CLI_PROVIDERS, available_providers, resolve_provider
    from omi.brain.fallback import parse_entries

    installed = {p["name"] for p in available_providers(cfg) if p["installed"]}
    valid = installed & ({"claude"} | set(CLI_PROVIDERS))
    primary = resolve_provider(cfg.brain.worker, role="worker")
    if primary == "off":
        return []
    primary_model = (
        cfg.brain.claude.worker_model
        if primary == "claude"
        else getattr(getattr(cfg.brain, primary, None), "worker_model", "")
    )
    out: list[tuple[str, str]] = []
    for provider, model in parse_entries(cfg.brain.worker_fallbacks):
        provider = resolve_provider(provider, role="worker")
        if provider not in valid:
            continue
        if provider == primary and (not model or model == primary_model):
            continue
        if (provider, model) in out:
            continue
        out.append((provider, model))
    return out


def create_worker_chain(
    cfg: Config, on_switch: Callable[[str, str], Awaitable[None]] | None = None
) -> Any:
    """The background worker, wrapped with its backups when any are usable."""
    from omi.brain.cli_providers import resolve_provider

    primary = resolve_provider(cfg.brain.worker, role="worker")
    if primary == "off":
        return None
    backups = usable_worker_fallbacks(cfg)
    if not backups:
        return create_worker(cfg)
    from omi.brain.fallback import FallbackWorker

    return FallbackWorker(
        [(primary, ""), *backups],
        lambda p, m: create_worker(with_worker_model(cfg, p, m)),
        on_switch=on_switch,
    )


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
