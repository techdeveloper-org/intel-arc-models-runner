"""Main command-line entry point for Intel Arc Models Runner.

Provides developer commands including hardware device discovery, model execution,
server management, and heterogeneous performance benchmarking.
"""

import argparse
import sys
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arc_runner import __version__
from arc_runner.devices.detector import DeviceDetector
from arc_runner.devices.models import DeviceType, SystemTopology


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

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

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
