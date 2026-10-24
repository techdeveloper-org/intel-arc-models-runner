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
from arc_runner.engine.aot_cache import AOTCompilationManager
from arc_runner.engine.embedding_engine import EmbeddingEngine
from arc_runner.engine.lora_manager import LoRAManager
from arc_runner.hub.manager import HubManager
from arc_runner.hub.ollama_importer import OllamaImporter
from arc_runner.hub.registry import ModelRegistry
from arc_runner.quant.gguf_parser import GGUFParser
from arc_runner.quant.transpiler import GGUFTranspiler
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


def handle_inspect_gguf(file_path: str, json_output: bool = False) -> int:
    """Inspect binary GGUF container headers, parameters, and quantization distribution.

    Args:
        file_path: Path to target .gguf file or Ollama blob.
        json_output: If True, prints JSON telemetry.

    Returns:
        int: Process exit code (0 for success).
    """
    console = Console()
    try:
        parser = GGUFParser(file_path)
        summary = parser.get_summary()

        if json_output:
            data = {
                "file_path": summary.file_path,
                "file_size_bytes": summary.file_size_bytes,
                "architecture": summary.architecture,
                "context_length": summary.context_length,
                "embedding_length": summary.embedding_length,
                "block_count": summary.block_count,
                "head_count": summary.head_count,
                "tensor_count": summary.tensor_count,
                "total_parameters": summary.total_parameters,
                "quant_distribution": summary.quant_distribution,
                "primary_quant_type": summary.primary_quant_type,
            }
            print(json.dumps(data, indent=2))
            return 0

        table = Table(title=f"GGUF Container Inspection: {Path(file_path).name}", border_style="cyan")
        table.add_column("Property", style="bold white", width=24)
        table.add_column("Value", style="cyan")

        param_billions = f"{summary.total_parameters / 1_000_000_000:.2f} B" if summary.total_parameters else "Unknown"
        size_mb = f"{summary.file_size_bytes / (1024 * 1024):.1f} MB"

        table.add_row("Architecture", summary.architecture)
        table.add_row("Parameters", param_billions)
        table.add_row("File Size", size_mb)
        table.add_row("Context Window", f"{summary.context_length} tokens")
        table.add_row("Embedding Dim", str(summary.embedding_length))
        table.add_row("Block / Layer Count", str(summary.block_count))
        table.add_row("Head Count", str(summary.head_count))
        table.add_row("Tensor Count", str(summary.tensor_count))
        table.add_row("Primary Quantization", summary.primary_quant_type)

        console.print(table)

        if summary.quant_distribution:
            q_table = Table(title="Quantization Breakdown", border_style="magenta")
            q_table.add_column("GGML Quant Type", style="bold")
            q_table.add_column("Share of Tensors", style="magenta")
            for q_type, pct in summary.quant_distribution.items():
                q_table.add_row(q_type, f"{pct:.1f}%")
            console.print(q_table)

        return 0
    except Exception as exc:
        console.print(f"[bold red]GGUF inspection failed:[/bold red] {exc}")
        return 1


def handle_import_ollama(
    model_name: Optional[str] = None,
    list_only: bool = False,
    json_output: bool = False,
    pull_openvino: bool = False,
) -> int:
    """Discover and import local Ollama models into arc-runner.

    Args:
        model_name: Optional model tag or name to inspect/import.
        list_only: If True, only lists local models without importing.
        json_output: If True, outputs JSON format.
        pull_openvino: If True, automatically pulls matched OpenVINO INT4 model.

    Returns:
        int: Process exit code.
    """
    console = Console()
    importer = OllamaImporter()

    if not importer.is_available:
        console.print(f"[bold yellow]Ollama manifests not found at:[/bold yellow] {importer.manifests_dir}")
        return 1

    manifests = importer.discover_models()

    if not model_name or list_only:
        if json_output:
            data = [
                {
                    "model_name": m.model_name,
                    "tag": m.tag,
                    "size_bytes": m.model_size_bytes,
                    "blob_exists": m.model_blob_path is not None,
                    "blob_path": m.model_blob_path,
                }
                for m in manifests
            ]
            print(json.dumps(data, indent=2))
            return 0

        table = Table(title="Locally Installed Ollama Models (~/.ollama)", border_style="green")
        table.add_column("Model Name", style="bold white")
        table.add_column("Tag", style="cyan")
        table.add_column("Disk Size", style="dim")
        table.add_column("Blob Status", style="magenta")
        table.add_column("Recommended OpenVINO Mapping", style="green")

        for m in manifests:
            size_mb = f"{m.model_size_bytes / (1024 * 1024):.1f} MB"
            blob_status = "[green]Ready[/green]" if m.model_blob_path else "[red]Missing[/red]"
            rec = importer.map_to_openvino_profile(m.model_name)
            rec_str = rec.model_id if rec else "[dim]Custom GGUF[/dim]"
            table.add_row(m.model_name, m.tag, size_mb, blob_status, rec_str)

        console.print(table)
        return 0

    manifest = importer.find_manifest(model_name)
    if not manifest:
        console.print(f"[bold red]Ollama model '{model_name}' not found locally.[/bold red]")
        return 1

    console.print(f"[bold cyan]Ollama Model Found:[/bold cyan] {manifest.model_name}:{manifest.tag}")
    if manifest.model_blob_path:
        console.print(f"[dim]Blob path: {manifest.model_blob_path}[/dim]")
        handle_inspect_gguf(manifest.model_blob_path, json_output=json_output)

    rec = importer.map_to_openvino_profile(manifest.model_name)
    if rec:
        console.print(f"\n[bold green]★ Recommended OpenVINO Accelerator Mapping:[/bold green] [bold cyan]{rec.model_id}[/bold cyan] ({rec.quantization})")
        if pull_openvino:
            console.print(f"Acquiring optimized OpenVINO weights for {rec.model_id}...")
            from arc_runner.cli.main import handle_pull
            return handle_pull(model_id=rec.model_id)
        else:
            console.print(f"[dim]Run 'arc-runner pull {rec.model_id}' to download OpenVINO weights optimized for XMX/NPU.[/dim]")

    return 0


def handle_aot_compile(
    model_id: str,
    device: str = "GPU",
    prewarm: bool = True,
    json_output: bool = False,
) -> int:
    """Execute ahead-of-time model graph compilation and report latency benchmarks.

    Args:
        model_id: Model name or physical directory path.
        device: Accelerator silicon target ('GPU', 'NPU', 'CPU').
        prewarm: Whether to execute a 1-token pre-flight pass.
        json_output: If True, outputs JSON format.

    Returns:
        int: Process exit code.
    """
    console = Console()
    aot_mgr = AOTCompilationManager()

    console.print(f"[bold cyan]Pre-compiling '{model_id}' for {device.upper()} silicon...[/bold cyan]")
    try:
        result = aot_mgr.compile_model(model_path=model_id, device=device, prewarm=prewarm)
        if json_output:
            data = {
                "model_id": result.model_id,
                "device": result.device,
                "compilation_time_ms": result.compilation_time_ms,
                "warm_load_time_ms": result.warm_load_time_ms,
                "blob_size_bytes": result.blob_size_bytes,
                "cache_path": result.cache_path,
                "speedup_multiplier": result.speedup_multiplier,
            }
            print(json.dumps(data, indent=2))
            return 0

        table = Table(title=f"AOT Compilation Summary: {result.model_id} ({result.device})", border_style="green")
        table.add_column("Metric", style="bold white")
        table.add_column("Measurement", style="cyan")

        table.add_row("Compilation Latency", f"{result.compilation_time_ms:.1f} ms")
        table.add_row("Warm Reload Latency", f"{result.warm_load_time_ms:.1f} ms")
        table.add_row("Startup Speedup", f"[bold green]{result.speedup_multiplier:.2f}x[/bold green]")
        table.add_row("Compiled Blob Footprint", f"{result.blob_size_bytes / (1024 * 1024):.2f} MB")
        table.add_row("Cache Directory", result.cache_path)

        console.print(table)
        return 0
    except Exception as exc:
        console.print(f"[bold red]AOT compilation failed:[/bold red] {exc}")
        return 1


def handle_embed(
    model_id: str,
    text: str,
    device: str = "NPU",
    json_output: bool = False,
    console: Optional[Console] = None,
) -> int:
    """Generate dense vector embedding for text prompt on Intel NPU/GPU.

    Args:
        model_id: Model identifier name or path.
        text: Query string to encode.
        device: Hardware device target ('NPU', 'GPU', 'CPU').
        json_output: Emit raw JSON vector array.
        console: Optional Rich console.

    Returns:
        int: Exit status code (0 for success).
    """
    c = console or Console()
    try:
        model_mgr = ModelManager()
        resolved_path = model_mgr.resolve_model_path(model_id)
        engine = EmbeddingEngine(str(resolved_path), device=device)
        vec = engine.embed_query(text)

        if json_output:
            print(json.dumps({"model": model_id, "device": engine.device, "dimension": len(vec), "embedding": vec}))
        else:
            c.print(f"[bold green]Embedding Generated Successfully[/bold green] on [cyan]{engine.device}[/cyan]")
            c.print(f"Dimension: [bold]{len(vec)}[/bold] | Sample: {vec[:5]} ...")
        return 0
    except Exception as exc:
        c.print(f"[bold red]Embedding generation failed:[/bold red] {exc}")
        return 1


def handle_convert(
    input_path: str,
    output_dir: str,
    precision: str = "int4",
    device: str = "GPU",
    json_output: bool = False,
    console: Optional[Console] = None,
) -> int:
    """Transpile a local GGUF container to OpenVINO IR format.

    Args:
        input_path: Source GGUF file path.
        output_dir: Target directory path for generated IR XML/BIN.
        precision: Quantization precision ('int4', 'int8', 'fp16').
        device: Target execution silicon ('GPU', 'NPU', 'CPU').
        json_output: Output result in JSON format.
        console: Optional Rich console.

    Returns:
        int: Exit status code (0 for success).
    """
    c = console or Console()
    try:
        transpiler = GGUFTranspiler(input_path, output_dir)
        res = transpiler.transpile(precision=precision, target_device=device)

        if json_output:
            print(json.dumps({
                "model_name": res.model_name,
                "total_tensors": res.total_tensors,
                "source_format": res.source_format,
                "target_precision": res.target_precision,
                "xml_path": str(res.xml_path),
                "bin_path": str(res.bin_path),
                "total_bytes": res.total_bytes,
                "duration_ms": res.duration_ms,
                "sha256_hash": res.sha256_hash,
                "average_sqnr_db": res.average_sqnr_db,
            }, indent=2))
        else:
            table = Table(title="GGUF to OpenVINO IR Transpilation Complete", show_header=True, header_style="bold magenta")
            table.add_column("Property", style="bold white")
            table.add_column("Value", style="cyan")
            table.add_row("Model Name", res.model_name)
            table.add_row("Tensors Repacked", str(res.total_tensors))
            table.add_row("Target Precision", f"[bold green]{res.target_precision}[/bold green]")
            table.add_row("Output XML", str(res.xml_path))
            table.add_row("Output BIN", f"{res.bin_path} ({res.total_bytes / (1024 * 1024):.2f} MB)")
            table.add_row("SHA-256 Digest", f"[dim]{res.sha256_hash[:16]}...[/dim]" if res.sha256_hash else "Verified")
            table.add_row("SQNR Fidelity", f"[bold green]{res.average_sqnr_db:.1f} dB[/bold green] [dim](≥ 32.0 dB)[/dim]")
            table.add_row("Transpile Duration", f"{res.duration_ms:.1f} ms")
            c.print(table)
        return 0
    except Exception as exc:
        c.print(f"[bold red]Transpilation failed:[/bold red] {exc}")
        return 1


def handle_adapter_list(
    adapters_dir: Optional[str] = None,
    json_output: bool = False,
    console: Optional[Console] = None,
) -> int:
    """Discover and list all local LoRA adapters.

    Args:
        adapters_dir: Optional custom directory to search for adapters.
        json_output: If True, formats output as JSON.
        console: Optional Rich console.

    Returns:
        int: Process exit code.
    """
    c = console or Console()
    mgr = LoRAManager(adapters_dir=adapters_dir)
    discovered = mgr.discover_adapters()

    if json_output:
        data = [
            {
                "adapter_id": a.adapter_id,
                "path": str(a.adapter_path),
                "base_model": a.base_model_name,
                "rank": a.rank,
                "alpha": a.alpha,
                "target_modules": a.target_modules,
                "size_bytes": a.file_size_bytes,
            }
            for a in discovered
        ]
        print(json.dumps(data, indent=2))
        return 0

    if not discovered:
        c.print(f"[yellow]No LoRA adapters found in:[/yellow] {mgr.adapters_dir}")
        c.print(f"[dim]Place adapters with 'adapter_config.json' and 'adapter_model.safetensors' in {mgr.adapters_dir}[/dim]")
        return 0

    table = Table(title="Discovered Local LoRA Adapters", show_header=True, header_style="bold magenta")
    table.add_column("Adapter ID", style="bold white")
    table.add_column("Base Model", style="cyan")
    table.add_column("Rank (r)", style="green", justify="right")
    table.add_column("Alpha", style="dim", justify="right")
    table.add_column("Target Modules", style="dim")
    table.add_column("Size (MB)", style="magenta", justify="right")

    for a in discovered:
        size_mb = f"{a.file_size_bytes / (1024 * 1024):.1f}"
        modules_str = ", ".join(a.target_modules[:3])
        if len(a.target_modules) > 3:
            modules_str += f" (+{len(a.target_modules) - 3})"
        table.add_row(
            a.adapter_id,
            a.base_model_name,
            str(a.rank),
            f"{a.alpha:.1f}",
            modules_str,
            size_mb,
        )

    c.print(table)
    return 0


def handle_adapter_inspect(
    path: str,
    json_output: bool = False,
    console: Optional[Console] = None,
) -> int:
    """Inspect metadata, rank, and target modules of a specific LoRA adapter.

    Args:
        path: Filesystem path to adapter directory or safetensors container.
        json_output: If True, formats output as JSON.
        console: Optional Rich console.

    Returns:
        int: Process exit code.
    """
    c = console or Console()
    try:
        mgr = LoRAManager()
        meta = mgr.parse_adapter_metadata(Path(path))

        if json_output:
            data = {
                "adapter_id": meta.adapter_id,
                "path": str(meta.adapter_path),
                "base_model": meta.base_model_name,
                "rank": meta.rank,
                "alpha": meta.alpha,
                "target_modules": meta.target_modules,
                "size_bytes": meta.file_size_bytes,
            }
            print(json.dumps(data, indent=2))
            return 0

        table = Table(title=f"LoRA Adapter Metadata: {meta.adapter_id}", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="bold white")
        table.add_column("Value", style="cyan")
        table.add_row("Adapter ID", meta.adapter_id)
        table.add_row("Filesystem Path", str(meta.adapter_path))
        table.add_row("Base Model Target", meta.base_model_name)
        table.add_row("Decomposition Rank (r)", str(meta.rank))
        table.add_row("Scaling Factor (alpha)", f"{meta.alpha:.1f}")
        table.add_row("Target Modules", ", ".join(meta.target_modules))
        table.add_row("Container Size", f"{meta.file_size_bytes / (1024 * 1024):.2f} MB")
        c.print(table)
        return 0
    except Exception as exc:
        c.print(f"[bold red]Adapter inspection failed:[/bold red] {exc}")
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

    gguf_parser_cmd = subparsers.add_parser("inspect-gguf", help="Inspect binary GGUF container headers and tensors")
    gguf_parser_cmd.add_argument("path", help="Path to .gguf file or Ollama blob")
    gguf_parser_cmd.add_argument("--json", action="store_true", help="Output GGUF metadata in JSON format")

    ollama_parser = subparsers.add_parser("import-ollama", help="Discover and import local Ollama models into arc-runner")
    ollama_parser.add_argument("model", nargs="?", default=None, help="Specific model name or tag to inspect/import")
    ollama_parser.add_argument("--list", action="store_true", help="List all local Ollama models")
    ollama_parser.add_argument("--pull", action="store_true", help="Automatically pull matched OpenVINO INT4 model")
    ollama_parser.add_argument("--json", action="store_true", help="Output local model catalog in JSON format")

    aot_parser = subparsers.add_parser("aot-compile", help="Ahead-of-Time compile model graph for Intel silicon")
    aot_parser.add_argument("model", help="Model name or directory path to compile")
    aot_parser.add_argument("--device", default="GPU", help="Target silicon (GPU, NPU, CPU)")
    aot_parser.add_argument("--no-prewarm", action="store_true", help="Skip 1-token pre-flight execution pass")
    aot_parser.add_argument("--json", action="store_true", help="Output compilation metrics in JSON format")

    embed_parser = subparsers.add_parser("embed", help="Generate vector embedding for text on Intel NPU/GPU")
    embed_parser.add_argument("model", help="Embedding model name or directory")
    embed_parser.add_argument("text", help="Text string to generate embedding for")
    embed_parser.add_argument("--device", default="NPU", help="Compute device (NPU, GPU, CPU)")
    embed_parser.add_argument("--json", action="store_true", help="Output embedding vector in JSON format")

    convert_parser = subparsers.add_parser("convert", help="Convert local GGUF container to OpenVINO IR format")
    convert_parser.add_argument("input", help="Source .gguf model container path")
    convert_parser.add_argument("--output", "-o", required=True, help="Destination directory for OpenVINO IR files")
    convert_parser.add_argument("--precision", default="int4", choices=["int4", "int8", "fp16"], help="Target weight precision")
    convert_parser.add_argument("--device", default="GPU", help="Target accelerator silicon")
    convert_parser.add_argument("--json", action="store_true", help="Output result summary in JSON format")

    adapter_parser = subparsers.add_parser("adapter", help="Manage and inspect local LoRA adapters")
    adapter_sub = adapter_parser.add_subparsers(dest="adapter_action", help="Adapter action")

    adapter_list = adapter_sub.add_parser("list", help="List discovered LoRA adapters")
    adapter_list.add_argument("--dir", help="Directory path to search for adapters")
    adapter_list.add_argument("--json", action="store_true", help="Output in JSON format")

    adapter_inspect = adapter_sub.add_parser("inspect", help="Inspect metadata of specific adapter")
    adapter_inspect.add_argument("path", help="Path to adapter directory or safetensors file")
    adapter_inspect.add_argument("--json", action="store_true", help="Output in JSON format")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command == "embed":
        return handle_embed(
            model_id=args.model,
            text=args.text,
            device=args.device,
            json_output=args.json,
        )

    if args.command == "convert":
        return handle_convert(
            input_path=args.input,
            output_dir=args.output,
            precision=args.precision,
            device=args.device,
            json_output=args.json,
        )

    if args.command == "adapter":
        if getattr(args, "adapter_action", None) == "inspect":
            return handle_adapter_inspect(path=args.path, json_output=args.json)
        return handle_adapter_list(adapters_dir=getattr(args, "dir", None), json_output=args.json)

    if args.command == "inspect-gguf":
        return handle_inspect_gguf(file_path=args.path, json_output=args.json)

    if args.command == "import-ollama":
        return handle_import_ollama(
            model_name=args.model,
            list_only=args.list,
            json_output=args.json,
            pull_openvino=args.pull,
        )

    if args.command == "aot-compile":
        return handle_aot_compile(
            model_id=args.model,
            device=args.device,
            prewarm=not args.no_prewarm,
            json_output=args.json,
        )

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
