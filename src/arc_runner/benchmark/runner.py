"""Automated hardware benchmarking engine with Student's t-test statistical validation.

Executes deterministic warmup passes, measures TTFT, TPOT, and tokens/sec throughput,
and renders formatted Rich visual scorecards for Intel Arc GPUs and NPUs.
"""

import time
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.core.logging import get_logger
from arc_runner.benchmark.stats import student_t_ci95, student_t_critical_95
from arc_runner.engine.models import GenerationConfig
from arc_runner.server.model_manager import ModelManager

logger = get_logger("benchmark.runner")


class BenchmarkRunResult(BaseModel):
    """Timing and throughput measurements for a single generation run."""

    run_index: int = Field(description="Run sequence number")
    is_warmup: bool = Field(default=False, description="True if run was part of discarded warmup passes")
    ttft_ms: float = Field(description="Time to first token in milliseconds")
    tpot_ms: float = Field(description="Average time per output token in milliseconds")
    tokens_per_sec: float = Field(description="Decoding generation throughput in tokens per second")
    prompt_tokens: int = Field(description="Number of prompt tokens evaluated")
    completion_tokens: int = Field(description="Number of tokens generated")
    total_time_ms: float = Field(description="Total execution latency in milliseconds")


class BenchmarkSummary(BaseModel):
    """Aggregated statistical benchmark report with 95% confidence intervals."""

    model_id: str = Field(description="Tested model identifier")
    device: str = Field(description="Target silicon device identifier")
    iterations: int = Field(description="Number of measured test iterations")
    warmup_runs: int = Field(description="Number of discarded warmup passes")
    avg_ttft_ms: float = Field(description="Sample mean Time to First Token")
    ci95_ttft_ms: float = Field(description="Student's t 95% confidence interval half-width for TTFT")
    avg_tpot_ms: float = Field(description="Sample mean Time Per Output Token")
    ci95_tpot_ms: float = Field(description="Student's t 95% confidence interval half-width for TPOT")
    avg_tokens_per_sec: float = Field(description="Sample mean throughput in tokens per second")
    ci95_tokens_per_sec: float = Field(description="Student's t 95% confidence interval half-width for throughput")
    run_results: List[BenchmarkRunResult] = Field(default_factory=list, description="Per-run result details")


class BenchmarkRunner:
    """Orchestrates hardware benchmark runs with warmup rejection and statistical confidence."""

    def __init__(self, model_manager: Optional[ModelManager] = None) -> None:
        """Initialize the benchmark runner with optional engine model manager.

        Args:
            model_manager: Optional ModelManager instance to load models and engines.
        """
        self._manager = model_manager or ModelManager()

    def _get_student_t_critical(self, df: int) -> float:
        """Return the two-sided 95% Student's t critical value (see arc_runner.benchmark.stats).

        Args:
            df: Degrees of freedom (n - 1).

        Returns:
            float: Critical t-score multiplier.
        """
        return student_t_critical_95(df)

    def _compute_confidence_interval_95(self, samples: List[float]) -> Tuple[float, float]:
        """Compute sample mean and 95% Student's t confidence interval half-width.

        Args:
            samples: List of measured continuous values.

        Returns:
            Tuple[float, float]: (sample_mean, ci95_half_width).
        """
        return student_t_ci95(samples)

    def run_benchmark(
        self,
        model_id: str,
        device: str = "GPU",
        iterations: int = 5,
        warmup: int = 2,
        max_new_tokens: int = 64,
        prompt: Optional[str] = None,
    ) -> BenchmarkSummary:
        """Execute benchmark iterations with warmup rejection and statistical CI calculation.

        Args:
            model_id: Target model name or path.
            device: Silicon accelerator target ('GPU', 'NPU', 'CPU').
            iterations: Number of timed benchmark measurement runs (minimum 1).
            warmup: Number of initial discarded runs to pre-warm kernels and caches.
            max_new_tokens: Number of completion tokens to generate per iteration.
            prompt: Optional custom prompt text.

        Returns:
            BenchmarkSummary containing computed metrics and per-run breakdown.

        Raises:
            ArcRunnerError: If iterations or tokens is zero/negative, if model load fails, or if a pass
                streams no token (NO_TOKENS_GENERATED).
        """
        if iterations <= 0:
            raise ArcRunnerError("Iterations must be greater than zero", "INVALID_ARGUMENT", 422)
        if max_new_tokens <= 0:
            raise ArcRunnerError("Tokens must be greater than zero", "INVALID_ARGUMENT", 422)

        test_prompt = prompt or "Explain how Intel Arc GPUs and Intel AI Boost NPUs accelerate local LLM inference."
        engine = self._manager.get_or_load_engine(model_id, device=device)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            stream=True,
        )

        all_results: List[BenchmarkRunResult] = []
        total_passes = warmup + iterations

        for i in range(total_passes):
            is_warmup = i < warmup
            run_idx = i + 1

            t_start = time.perf_counter_ns()
            t_first: Optional[int] = None
            generated_tokens = 0
            prompt_tokens_est = len(test_prompt.split())

            for chunk in engine.generate_stream(test_prompt, gen_config):
                if not chunk.is_finished:
                    generated_tokens += 1
                    if t_first is None:
                        t_first = time.perf_counter_ns()
                else:
                    if chunk.prompt_tokens:
                        prompt_tokens_est = chunk.prompt_tokens

            t_end = time.perf_counter_ns()

            if t_first is None or generated_tokens == 0:
                raise ArcRunnerError(
                    f"Pass {run_idx} of '{model_id}' on {device} produced no tokens; no throughput can be reported",
                    "NO_TOKENS_GENERATED",
                    502,
                )

            total_ms = max(0.01, (t_end - t_start) / 1_000_000.0)
            ttft_ms = max(0.01, (t_first - t_start) / 1_000_000.0)
            decode_ms = max(1e-3, (t_end - t_first) / 1_000_000.0)

            tokens_count = generated_tokens
            tpot_ms = max(0.01, decode_ms / tokens_count if tokens_count > 1 else decode_ms)
            tok_per_sec = (tokens_count / decode_ms) * 1000.0

            result = BenchmarkRunResult(
                run_index=run_idx,
                is_warmup=is_warmup,
                ttft_ms=round(ttft_ms, 2),
                tpot_ms=round(tpot_ms, 2),
                tokens_per_sec=round(tok_per_sec, 2),
                prompt_tokens=prompt_tokens_est,
                completion_tokens=tokens_count,
                total_time_ms=round(total_ms, 2),
            )
            all_results.append(result)

        measured = [r for r in all_results if not r.is_warmup]

        ttfts = [r.ttft_ms for r in measured]
        tpots = [r.tpot_ms for r in measured]
        speeds = [r.tokens_per_sec for r in measured]

        avg_ttft, ci_ttft = self._compute_confidence_interval_95(ttfts)
        avg_tpot, ci_tpot = self._compute_confidence_interval_95(tpots)
        avg_speed, ci_speed = self._compute_confidence_interval_95(speeds)

        return BenchmarkSummary(
            model_id=model_id,
            device=device,
            iterations=iterations,
            warmup_runs=warmup,
            avg_ttft_ms=round(avg_ttft, 2),
            ci95_ttft_ms=round(ci_ttft, 2),
            avg_tpot_ms=round(avg_tpot, 2),
            ci95_tpot_ms=round(ci_tpot, 2),
            avg_tokens_per_sec=round(avg_speed, 2),
            ci95_tokens_per_sec=round(ci_speed, 2),
            run_results=all_results,
        )

    def render_rich_table(self, summary: BenchmarkSummary, console: Optional[Console] = None) -> None:
        """Render a formatted visual scorecard table displaying benchmark results.

        Args:
            summary: Populated BenchmarkSummary model.
            console: Optional Rich Console instance for terminal output.
        """
        c = console or Console()

        header = (
            f"[bold cyan]Intel Arc Models Runner Hardware Benchmark[/bold cyan]\n"
            f"[bold white]Model:[/bold white] {summary.model_id} | [bold yellow]Target Silicon:[/bold yellow] {summary.device}\n"
            f"[dim]Iterations: {summary.iterations} (excluding {summary.warmup_runs} warmup passes) | 95% Confidence Intervals[/dim]"
        )
        c.print(Panel(header, title="Inference Performance Telemetry", border_style="cyan"))

        table = Table(title="Measured Performance Metrics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="bold white", width=36)
        table.add_column("Mean Score", justify="right", style="bold green", width=16)
        table.add_column("95% CI Range", justify="right", style="cyan", width=22)
        table.add_column("Unit", style="dim", width=12)

        table.add_row(
            "Time to First Token (TTFT)",
            f"{summary.avg_ttft_ms:.2f}",
            f"± {summary.ci95_ttft_ms:.2f}",
            "ms",
        )
        table.add_row(
            "Time Per Output Token (TPOT)",
            f"{summary.avg_tpot_ms:.2f}",
            f"± {summary.ci95_tpot_ms:.2f}",
            "ms/token",
        )
        table.add_row(
            "Decoding Throughput",
            f"{summary.avg_tokens_per_sec:.2f}",
            f"± {summary.ci95_tokens_per_sec:.2f}",
            "tokens/sec",
        )

        c.print(table)
