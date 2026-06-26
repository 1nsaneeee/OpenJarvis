"""Entry point: bootstrap all coroutines and run the event loop."""
from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--config", default="config/config.yaml", help="Path to config file.")
def main(config: str) -> None:
    """Launch the OpenJarvis voice assistant."""
    cfg_path = Path(config)
    if not cfg_path.exists():
        console.print(f"[yellow]Config not found at {config}, using defaults.[/yellow]")
        console.print("[dim]Tip: cp config/config.example.yaml config/config.yaml[/dim]")

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_run(config if cfg_path.exists() else None))


async def _run(config_path: str | None) -> None:
    from openjarvis.asr.whisper import WhisperASR
    from openjarvis.audio.capture import AudioCapture
    from openjarvis.bus.client import BusClient
    from openjarvis.conversation.manager import ConversationManager
    from openjarvis.llm.registry import load_provider
    from openjarvis.system.config import AppConfig, load_config
    from openjarvis.tools.builtin.time_tool import register_time_tool
    from openjarvis.tools.executor import ToolExecutor
    from openjarvis.tools.registry import ToolRegistry
    from openjarvis.wake.detector import WakeDetector

    # Load config
    cfg = load_config(config_path) if config_path else AppConfig()

    console.rule("[bold green]OpenJarvis v0.1[/bold green]")
    console.print(
        f"  Provider : [cyan]{cfg.llm.provider}[/cyan]"
        f"  Model: [cyan]{cfg.llm.model}[/cyan]"
    )
    console.print(f"  Redis    : [cyan]{cfg.redis_url}[/cyan]")
    console.print(f"  Wake     : [cyan]{cfg.wake.models}[/cyan]")
    console.print()

    # Event bus
    bus = BusClient(cfg.redis_url)
    try:
        await bus.connect()
    except Exception as exc:
        console.print(f"[red]Failed to connect to Redis at {cfg.redis_url}: {exc}[/red]")
        console.print("[dim]Is Redis running? Try: redis-server --port 6379[/dim]")
        return

    # Tools
    registry = ToolRegistry()
    register_time_tool(registry)

    # LLM provider
    provider = load_provider(cfg.llm.provider)

    # Executor
    executor = ToolExecutor(registry)

    # Conversation manager
    mgr = ConversationManager(
        bus,
        provider,
        executor,
        model=cfg.llm.model,
        max_history=cfg.conversation.max_turn_history,
        system_prompt_file=cfg.conversation.system_prompt_file,
        tools=registry.all_specs(),
    )
    await mgr.start()

    # ASR
    asr = WhisperASR(cfg.asr, cfg.wake, bus)
    await asr.start()

    # Wake detector
    wake = WakeDetector(cfg.wake, bus)
    await wake.start()

    # Mic capture
    capture = AudioCapture(cfg.audio, bus)
    await capture.start()

    console.print("[bold green]✓ OpenJarvis is running.[/bold green]")
    model_display = cfg.wake.models[0].replace("_", " ") if cfg.wake.models else "wake word"
    console.print(f"  Say [bold]'{model_display}'[/bold] to wake me up.")
    console.print("  Press [bold]Ctrl+C[/bold] to quit.\n")

    # Block until interrupted (cross-platform: KeyboardInterrupt cancels asyncio.run)
    try:
        await asyncio.Event().wait()
    finally:
        console.print("\n[yellow]Shutting down...[/yellow]")
        await capture.stop()
        await bus.close()
        console.print("[green]Goodbye.[/green]")


if __name__ == "__main__":
    main()
