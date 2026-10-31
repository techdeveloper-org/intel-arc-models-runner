"""Paired gate runner for one device and named OpenVINO-property arms (plan WP-02 item 02b).

Each arm is a named set of OpenVINO properties added on top of the control; the control arm
carries no extra properties.  The paired gate (bootstrap_paired_ratio_ci) resamples round
indices jointly so common-mode drift cancels.

Measurement mode (requires Intel Arc silicon):
  python scripts/bench_arms.py \\
    --model-dir models\\qwen-2.5-1.5b --device CPU --tier medium \\
    --rounds 18 --warmups 5 --fixed-warmups --seed 0 \\
    --out docs\\benchmarks\\results\\sweeps\\m10_a.v1.json \\
    --arm control \\
    --arm pcore=SCHEDULING_CORE_TYPE=PCORE_ONLY \\
    --arm pcore_ht_off=SCHEDULING_CORE_TYPE=PCORE_ONLY,ENABLE_HYPER_THREADING=false

Evaluate mode (runs without hardware):
  python scripts/bench_arms.py \\
    --evaluate docs\\benchmarks\\results\\sweeps\\m10_a.v1.json \\
    --evaluate docs\\benchmarks\\results\\sweeps\\m10_b.v1.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_arm(spec: str) -> Tuple[str, Dict[str, str]]:
    """Parse one --arm specification into (name, properties).

    Format: 'control' | 'aa_control' | 'name=KEY=VALUE,KEY=VALUE,...'

    Args:
        spec: Arm specification string.

    Returns:
        Tuple of (arm_name, {KEY: VALUE}).

    Raises:
        ValueError: If the spec cannot be parsed.
    """
    if "=" not in spec:
        return spec, {}
    parts = spec.split("=", 1)
    name = parts[0]
    props_str = parts[1]
    props: Dict[str, str] = {}
    for kv in props_str.split(","):
        kv = kv.strip()
        if "=" not in kv:
            raise ValueError(f"cannot parse property {kv!r} in arm {spec!r}")
        k, v = kv.split("=", 1)
        props[k.strip()] = v.strip()
    return name, props


def load_result(path: Path) -> Dict[str, Any]:
    """Load and minimally validate a schema-v1 result file.

    Args:
        path: Path to the JSON result file.

    Returns:
        Parsed document.

    Raises:
        SystemExit: If the file cannot be read, is not valid JSON, or has the wrong schema version.
    """
    if not path.exists():
        print(f"error: result file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"error: {path} is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(1)
    if doc.get("schema_version") != 1:
        print(
            f"error: {path} has schema_version {doc.get('schema_version')!r}, expected 1",
            file=sys.stderr,
        )
        sys.exit(1)
    return doc


def extract_arm_rounds(
    document: Dict[str, Any],
    arm: str,
    metric: str,
) -> List[Tuple[int, float]]:
    """Extract per-iteration timed values for one arm from a result document.

    Only non-warmup samples are included; entries with a None metric value are dropped.

    Args:
        document: Parsed schema-v1 document.
        arm: Backend name to extract.
        metric: Metric name (decode_tps, ttft_ms, or a metrics-dict key).

    Returns:
        List of (iteration, value) sorted by iteration.
    """
    rows = []
    for s in document.get("samples", []):
        if s.get("is_warmup", False):
            continue
        if s.get("backend") != arm:
            continue
        if metric in ("ttft_ms", "decode_tps"):
            v = s.get(metric)
        else:
            v = (s.get("metrics") or {}).get(metric)
        if v is None:
            continue
        rows.append((s["iteration"], float(v)))
    rows.sort(key=lambda x: x[0])
    return rows


def _extract_paired(
    document: Dict[str, Any],
    arm: str,
    control: str,
    metric: str,
) -> Tuple[List[float], List[float]]:
    """Return (arm_rounds, control_rounds) matched by iteration.

    Args:
        document: Parsed schema-v1 document.
        arm: Candidate arm backend name.
        control: Control arm backend name.
        metric: Metric to extract.

    Returns:
        Tuple of (arm_values, control_values) for the shared iterations.

    Raises:
        ValueError: If there are no iterations where both arm and control have a value.
    """
    arm_by_iter = dict(extract_arm_rounds(document, arm, metric))
    ctrl_by_iter = dict(extract_arm_rounds(document, control, metric))
    common = sorted(set(arm_by_iter) & set(ctrl_by_iter))
    if not common:
        raise ValueError(
            f"no matching iterations between arm {arm!r} and control {control!r}"
        )
    return [arm_by_iter[i] for i in common], [ctrl_by_iter[i] for i in common]


def run_evaluate(
    files: List[Path],
    metric: str,
    margin: float,
    n_resamples: int,
    seed: int,
    device: str,
) -> int:
    """Load result files, run paired gates for all arms, print a verdict table.

    Args:
        files: Schema-v1 result files to evaluate.
        metric: Metric to gate on.
        margin: Adoption threshold margin.
        n_resamples: Base bootstrap resamples (raised to 20000 when K > 2).
        seed: RNG seed.
        device: Device string for decide_adoption (GPU, CPU, NPU).

    Returns:
        0 on success, 1 on data error.
    """
    from arc_runner.benchmark.stats import (
        evaluate_paired_gate,
        decide_adoption,
    )

    docs = [load_result(f) for f in files]

    for f, doc in zip(files, docs):
        ctrl_timed = [
            s
            for s in doc.get("samples", [])
            if s.get("backend") == "control" and not s.get("is_warmup", False)
        ]
        if not ctrl_timed:
            print(f"error: {f} has no timed 'control' backend samples", file=sys.stderr)
            return 1

    all_arms: set = set()
    for doc in docs:
        for s in doc.get("samples", []):
            b = s.get("backend", "")
            if b and b != "control":
                all_arms.add(b)

    candidate_arms = sorted(a for a in all_arms if not a.startswith("aa_"))
    aa_arms = sorted(a for a in all_arms if a.startswith("aa_"))

    k = max(1, len(candidate_arms))
    n_res = max(n_resamples, 20000) if k > 2 else n_resamples

    all_gates = []
    for doc, f in zip(docs, files):
        doc_backends = {s.get("backend") for s in doc.get("samples", [])}
        for arm in candidate_arms:
            if arm not in doc_backends:
                continue
            try:
                arm_vals, ctrl_vals = _extract_paired(doc, arm, "control", metric)
            except ValueError as exc:
                print(f"warning: {f.name} arm {arm!r}: {exc}", file=sys.stderr)
                continue
            gate = evaluate_paired_gate(
                arm_vals,
                ctrl_vals,
                metric=metric,
                k=k,
                margin=margin,
                n_resamples=n_res,
                seed=seed,
                arm_name=arm,
            )
            all_gates.append(gate)

        for arm in aa_arms:
            if arm not in doc_backends:
                continue
            try:
                arm_vals, ctrl_vals = _extract_paired(doc, arm, "control", metric)
            except ValueError:
                continue
            gate = evaluate_paired_gate(
                arm_vals,
                ctrl_vals,
                metric=metric,
                k=1,
                margin=0.0,
                n_resamples=n_res,
                seed=seed,
                arm_name=arm,
            )
            all_gates.append(gate)

    if not all_gates:
        print(
            "error: no paired gate results (no arm+control data found in files)",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nPaired gate results  metric={metric}  K={k}  margin={margin:.3f}  seed={seed}"
    )
    print("-" * 72)
    print(
        f"{'arm':<24} {'n':>4} {'ratio':>7} {'lo':>7} {'hi':>7}  {'verdict':<12} {'harmful'}"
    )
    print("-" * 72)
    for g in all_gates:
        harm_str = "HARMFUL" if g.harmful else ""
        print(
            f"{g.arm:<24} {g.n_rounds:>4} {g.point_ratio:>7.3f} "
            f"{g.ratio_lo:>7.3f} {g.ratio_hi:>7.3f}  {g.verdict:<12} {harm_str}"
        )
    print("-" * 72)

    candidate_gates = [g for g in all_gates if not g.arm.startswith("aa_")]
    if candidate_gates:
        outcomes = decide_adoption(candidate_gates, device=device)
        print(f"\nAdoption decisions  device={device}")
        print("-" * 40)
        for arm, outcome in sorted(outcomes.items()):
            print(f"  {arm:<24} {outcome}")
        print("-" * 40)

    return 0


def run_measurement(args: argparse.Namespace) -> int:
    """Run the arm measurement loop, writing a schema-v1 result file.

    Requires Intel Arc silicon and model weights at args.model_dir.

    Args:
        args: Parsed CLI arguments.

    Returns:
        0 on success, 1 on failure.
    """
    from arc_runner.benchmark.results import validate_model_dir

    model_dir = Path(args.model_dir)
    err = validate_model_dir(model_dir)
    if err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    arm_specs: List[Tuple[str, Dict[str, str]]] = []
    for spec in args.arm:
        try:
            name, props = parse_arm(spec)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        arm_specs.append((name, props))

    if not any(name == "control" for name, _ in arm_specs):
        print("error: at least one --arm must be named 'control'", file=sys.stderr)
        return 1

    print(
        f"Measurement mode requires Intel Arc silicon. "
        f"Arms to measure: {[n for n, _ in arm_specs]}"
    )
    print(
        "Hardware measurement is not available in this environment. "
        "Run on a machine with Intel Arc GPU/NPU."
    )
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for bench_arms.py.

    Args:
        argv: Command-line arguments, or None to use sys.argv[1:].

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        prog="bench_arms",
        description="Paired gate runner for OpenVINO-property arms.",
    )
    parser.add_argument(
        "--evaluate",
        metavar="FILE",
        action="append",
        default=[],
        help="Schema-v1 result file to evaluate (repeat for multiple files).",
    )
    parser.add_argument(
        "--metric",
        default="decode_tps",
        help="Metric to gate on (default: decode_tps).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.03,
        help="Adoption threshold margin (default: 0.03).",
    )
    parser.add_argument(
        "--n-resamples",
        type=int,
        default=2000,
        help="Bootstrap resamples base (raised to 20000 when K > 2, default: 2000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed (default: 0).",
    )
    parser.add_argument(
        "--device",
        default="GPU",
        help="Device string for adoption decisions (default: GPU).",
    )
    parser.add_argument(
        "--arm",
        metavar="SPEC",
        action="append",
        default=[],
        help="Arm spec: 'control' or 'name=KEY=VALUE,...' (repeat for multiple arms).",
    )
    parser.add_argument("--model-dir", help="Model directory (measurement mode).")
    parser.add_argument("--rounds", type=int, default=18, help="Timed rounds per arm.")
    parser.add_argument("--warmups", type=int, default=5, help="Warmup rounds.")
    parser.add_argument("--fixed-warmups", action="store_true")
    parser.add_argument("--out", help="Output schema-v1 file path.")
    parser.add_argument(
        "--tier",
        choices=["short", "medium", "long"],
        default="medium",
        help="Workload tier.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=20,
        help="Prompt token count.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max new tokens.",
    )

    args = parser.parse_args(argv)

    if args.evaluate:
        return run_evaluate(
            files=[Path(f) for f in args.evaluate],
            metric=args.metric,
            margin=args.margin,
            n_resamples=args.n_resamples,
            seed=args.seed,
            device=args.device,
        )

    if not args.arm:
        parser.error("provide --arm specs for measurement mode, or --evaluate files")

    return run_measurement(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
