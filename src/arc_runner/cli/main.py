"""Main command-line entry point for Intel Arc Models Runner.

Provides developer commands including hardware device discovery, model execution,
server management, model hub acquisition, and heterogeneous performance benchmarking.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arc_runner import __version__
from arc_runner.benchmark.runner import BenchmarkRunner
from arc_runner.core.constants import DEFAULT_HOST, DEFAULT_PORT
from arc_runner.core.logging import get_logger, setup_logging
from arc_runner.devices.detector import DeviceDetector
from arc_runner.devices.models import DeviceType, SystemTopology
from arc_runner.hub.manager import HubManager
from arc_runner.hub.registry import ModelRegistry
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


def handle_pull(model_id: str, base_dir: Optional[str] = None) -> int:
    """Handle CLI command to download and verify an OpenVINO model.

    Args:
        model_id: Target model name alias or repository identifier.
        base_dir: Optional storage directory path.

    Returns:
        int: Process exit code.
    """
    console = Console()
    hub = HubManager(base_models_dir=base_dir)
    info = ModelRegistry.resolve(model_id)

    console.print(f"[bold cyan]Pulling model:[/bold cyan] {info.alias} [dim]({info.repo_id})[/dim]")
    try:
        last_pct = -1.0
        for progress in hub.pull(model_id, stream=True):
            if progress.percent is not None and progress.percent != last_pct:
                pct_str = f"{progress.percent:5.1f}%"
                speed_str = f"{progress.speed_mbps:.1f} MB/s" if progress.speed_mbps else "calculating..."
                eta_str = f"{progress.eta_seconds:.0f}s" if progress.eta_seconds else "-"
                console.print(f"  [cyan]{progress.status:<26}[/cyan] [green]{pct_str}[/green] [dim]({speed_str} | ETA: {eta_str})[/dim]")
                last_pct = progress.percent
            elif progress.status in ("pulling manifest", "verifying checksum", "success"):
                console.print(f"  [bold green]✔[/bold green] {progress.status}")

        console.print(f"[bold green]Model '{info.alias}' successfully pulled and verified![/bold green]")
        return 0
    except Exception as exc:
        console.print(f"[bold red]Pull failed:[/bold red] {exc}")
        return 1


def handle_models(base_dir: Optional[str] = None, json_output: bool = False) -> int:
    """Handle CLI command to list all locally installed OpenVINO models.

    Args:
        base_dir: Optional model storage directory.
        json_output: Output report in JSON format if True.

    Returns:
        int: Process exit code.
    """
    console = Console()
    hub = HubManager(base_models_dir=base_dir)
    models = hub.list_local()

    if json_output:
        print(json.dumps(models, indent=2))
        return 0

    table = Table(title="Installed OpenVINO Models", show_header=True, header_style="bold magenta")
    table.add_column("Model Name", style="bold white", min_width=22)
    table.add_column("Family", style="green", width=12)
    table.add_column("Parameters", style="yellow", width=12)
    table.add_column("Quantization", style="cyan", width=26)
    table.add_column("Disk Size", justify="right", style="white", width=14)
    table.add_column("Modified", style="dim", width=22)

    for m in models:
        details = m.get("details", {})
        size_gb = m.get("size", 0) / (1024.0 * 1024.0 * 1024.0)
        table.add_row(
            m.get("name", "-"),
            details.get("family", "-"),
            details.get("parameter_size", "-"),
            details.get("quantization_level", "-"),
            f"{size_gb:.2f} GB",
            m.get("modified_at", "-")[:19],
        )

    console.print(table)
    return 0


def handle_rm(model_id: str, base_dir: Optional[str] = None) -> int:
    """Handle CLI command to remove a local model repository.

    Args:
        model_id: Model alias or directory name to purge.
        base_dir: Optional model storage directory.

    Returns:
        int: Process exit code.
    """
    console = Console()
    hub = HubManager(base_models_dir=base_dir)
    try:
        hub.delete(model_id)
        console.print(f"[bold green]Successfully deleted model '{model_id}'[/bold green]")
        return 0
    except Exception as exc:
        console.print(f"[bold red]Delete failed:[/bold red] {exc}")
        return 1


def handle_benchmark(
    model_id: str,
    device: str = "GPU",
    iterations: int = 5,
    warmup: int = 2,
    tokens: int = 64,
    json_output: bool = False,
) -> int:
    """Handle CLI command to execute hardware benchmark with statistical CI.

    Args:
        model_id: Target model identifier.
        device: Hardware device string ('GPU', 'NPU', 'CPU').
        iterations: Number of test measurement runs.
        warmup: Number of initial discarded runs.
        tokens: Number of completion tokens per iteration.
        json_output: Output JSON report if True.

    Returns:
        int: Process exit code.
    """
    console = Console()
    runner = BenchmarkRunner()

    try:
        summary = runner.run_benchmark(
            model_id=model_id,
            device=device,
            iterations=iterations,
            warmup=warmup,
            max_new_tokens=tokens,
        )
        if json_output:
            print(summary.model_dump_json(indent=2))
        else:
            runner.render_rich_table(summary, console=console)
        return 0
    except Exception as exc:
        console.print(f"[bold red]Benchmark failed:[/bold red] {exc}")
        return 1


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

    pull_parser = subparsers.add_parser("pull", help="Download and verify an OpenVINO model from Hugging Face")
    pull_parser.add_argument("model", help="Model alias name or full Hugging Face repository identifier")
    pull_parser.add_argument("--dir", help="Target directory for downloaded model weights")

    models_parser = subparsers.add_parser("models", help="List all locally installed OpenVINO models")
    models_parser.add_argument("--dir", help="Directory path containing OpenVINO models")
    models_parser.add_argument("--json", action="store_true", help="Output model inventory in JSON format")

    rm_parser = subparsers.add_parser("rm", help="Remove an installed OpenVINO model from local disk")
    rm_parser.add_argument("model", help="Model name or directory alias to delete")
    rm_parser.add_argument("--dir", help="Directory path containing model to remove")

    bench_parser = subparsers.add_parser("benchmark", help="Benchmark model latency and throughput on Intel silicon")
    bench_parser.add_argument("--model", default="llama-3.2-3b", help="Model name or path to benchmark")
    bench_parser.add_argument("--device", default="GPU", help="Silicon device target (GPU, NPU, CPU)")
    bench_parser.add_argument("--iterations", type=int, default=5, help="Number of measurement runs (default: 5)")
    bench_parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs to discard (default: 2)")
    bench_parser.add_argument("--tokens", type=int, default=64, help="Tokens to generate per run (default: 64)")
    bench_parser.add_argument("--json", action="store_true", help="Output benchmark metrics in JSON format")

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

    if args.command == "pull":
        return handle_pull(model_id=args.model, base_dir=args.dir)

    if args.command == "models":
        return handle_models(base_dir=args.dir, json_output=args.json)

    if args.command == "rm":
        return handle_rm(model_id=args.model, base_dir=args.dir)

    if args.command == "benchmark":
        return handle_benchmark(
            model_id=args.model,
            device=args.device,
            iterations=args.iterations,
            warmup=args.warmup,
            tokens=args.tokens,
            json_output=args.json,
        )

    if args.command == "devices" or args.command is None:
        detector = DeviceDetector()
        topology = detector.detect_topology()
        if getattr(args, "json", False):
            print(json.dumps(topology.model_dump(), indent=2))
        else:
            render_devices_dashboard(topology)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
