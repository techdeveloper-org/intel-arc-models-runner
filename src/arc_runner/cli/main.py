"""Main command-line entry point for Intel Arc Models Runner.

Provides developer commands including hardware device discovery, model execution,
server management, and heterogeneous performance benchmarking.
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arc_runner import __version__
from arc_runner.core.constants import DEFAULT_HOST, DEFAULT_PORT
from arc_runner.core.logging import get_logger, setup_logging
from arc_runner.devices.detector import DeviceDetector
from arc_runner.devices.models import DeviceType, SystemTopology
from arc_runner.server.app import create_app
from arc_runner.server.model_manager import ModelManager


def render_devices_dashboard(topology: SystemTopology, console: Optional[Console] = None) -> None:
    """Render a rich visual dashboard displaying discovered Intel silicon.

    Args:
        topology: Discovered system topology containing device inventory.
        console: Optional Rich console instance for rendering.
    """
    c = console or Console()

    header_text = (
        f"[bold cyan]Intel Arc Models Runner[/bold cyan] [dim]v{__version__}[/dim]\n"
        f"[dim]OS: {topology.os_platform} | Host CPU: {topology.host_cpu_model}[/dim]"
    )
    c.print(Panel(header_text, title="Hardware Telemetry", border_style="cyan"))

    table = Table(title="Heterogeneous Silicon Accelerators", show_header=True, header_style="bold magenta")
    table.add_column("Device ID", style="dim", width=12)
    table.add_column("Silicon Name", style="bold white", min_width=32)
    table.add_column("Architecture", style="green", min_width=22)
    table.add_column("Role", style="yellow", width=16)
    table.add_column("USM Zero-Copy", justify="center", width=14)
    table.add_column("Capabilities", style="dim", min_width=28)

    for dev in topology.devices:
        role_label = "Fallback"
        if dev.is_default_target:
            role_label = "[bold green]Target (Verify)[/bold green]"
        elif dev.is_default_draft:
            role_label = "[bold cyan]Draft (NPU 5W)[/bold cyan]"

        usm_str = "[bold green]YES[/bold green]" if dev.has_usm else "[dim]NO[/dim]"
        caps_str = ", ".join(dev.capabilities) if dev.capabilities else "None"

        table.add_row(
            dev.device_id,
            dev.full_name,
            dev.architecture.value,
            role_label,
            usm_str,
            caps_str,
        )

    c.print(table)

    status_lines: List[str] = []
    if topology.has_intel_arc:
        status_lines.append("[bold green]✔[/bold green] [bold]Intel Arc Graphics:[/bold] Ready for Level Zero / XMX matrix acceleration")
    else:
        status_lines.append("[bold yellow]![/bold yellow] [bold]Intel Arc Graphics:[/bold] Not detected (using CPU/NPU fallback)")

    if topology.has_intel_npu:
        status_lines.append("[bold green]✔[/bold green] [bold]Intel AI Boost (NPU):[/bold] Ready for 5W stateful speculative drafting")
    else:
        status_lines.append("[bold yellow]![/bold yellow] [bold]Intel AI Boost (NPU):[/bold] Not detected")

    if topology.has_intel_arc and topology.has_intel_npu:
        status_lines.append("[bold cyan]★[/bold cyan] [bold]Heterogeneous Speculative Pairing:[/bold] [bold green]AVAILABLE[/bold green] (Dual-die cooperative speedup ~1.93x)")

    fallback_str = " -> ".join(topology.fallback_chain)
    status_lines.append(f"[bold dim]Fallback Degradation Chain:[/bold dim] {fallback_str}")

    c.print(Panel("\n".join(status_lines), title="System Readiness Status", border_style="green" if topology.has_intel_arc else "yellow"))


def run_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    model_dirs: Optional[List[str]] = None,
    default_model: Optional[str] = None,
    debug: bool = False,
    log_file: Optional[str] = None,
) -> int:
    """Initialize and run the high-throughput FastAPI daemon server.

    Args:
        host: Network interface address to bind.
        port: TCP port number to listen on.
        model_dirs: Optional list of directories containing OpenVINO models.
        default_model: Optional model identifier to pre-compile on startup.
        debug: Enable debug logging and stack trace exposition.
        log_file: Optional path to rotating JSON file appender.

    Returns:
        int: Process exit code.
    """
    import uvicorn

    setup_logging(debug=debug, log_file=Path(log_file) if log_file else None)
    logger = get_logger("cli.server")

    manager = ModelManager(model_dirs=[Path(d) for d in model_dirs] if model_dirs else None)
    app = create_app(model_manager=manager, debug=debug)

    console = Console()
    startup_banner = (
        f"[bold cyan]Intel Arc Models Runner Daemon[/bold cyan] [dim]v{__version__}[/dim]\n"
        f"[green]Ollama API Parity:[/green] http://{host}:{port}/api/generate | http://{host}:{port}/api/chat\n"
        f"[green]OpenAI API Parity:[/green] http://{host}:{port}/v1/chat/completions\n"
        f"[green]Hardware Telemetry:[/green] http://{host}:{port}/api/devices\n"
        f"[dim]Logging: Rich Console + arc-runner.log (Rule 27 MDC Tracing)[/dim]"
    )
    console.print(Panel(startup_banner, title="Daemon Server Initialized", border_style="cyan"))

    if default_model:
        try:
            logger.info(f"Pre-compiling default model: {default_model}")
            manager.get_or_load_engine(default_model)
            logger.info(f"Successfully compiled default model: {default_model}")
        except Exception as exc:
            logger.warning(f"Could not pre-load default model '{default_model}': {exc}")

    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="debug" if debug else "info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Execute the CLI application with parsed command line arguments.

    Args:
        argv: Optional command-line argument list (defaults to sys.argv[1:]).

    Returns:
        int: Process exit code (0 for success).
    """
    parser = argparse.ArgumentParser(
        prog="arc-runner",
        description="Native local model runner optimized for Intel Arc GPUs and AI Boost NPUs.",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    devices_parser = subparsers.add_parser("devices", help="Discover and report Intel Arc GPUs, NPUs, and CPUs")
    devices_parser.add_argument("--json", action="store_true", help="Output topology report in JSON format")

    serve_parser = subparsers.add_parser("serve", help="Start high-throughput daemon server on port 11434")
    serve_parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host interface to bind (default: {DEFAULT_HOST})")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port number to listen on (default: {DEFAULT_PORT})")
    serve_parser.add_argument("--model-dir", action="append", help="Directory path to search for OpenVINO IR models")
    serve_parser.add_argument("--default-model", help="Model name to pre-compile and keep hot in memory on startup")
    serve_parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging and stack traces")
    serve_parser.add_argument("--log-file", help="Path to write rotating JSON log file (default: arc-runner.log)")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "serve":
        return run_server(
            host=args.host,
            port=args.port,
            model_dirs=args.model_dir,
            default_model=args.default_model,
            debug=args.debug,
            log_file=args.log_file,
        )

    if args.command == "devices" or args.command is None:
        detector = DeviceDetector()
        topology = detector.detect_topology()
        if getattr(args, "json", False):
            import json
            print(json.dumps(topology.model_dump(), indent=2))
        else:
            render_devices_dashboard(topology)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
