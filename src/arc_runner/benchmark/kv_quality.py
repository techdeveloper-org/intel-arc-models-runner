"""Quality gate for a reduced-precision KV cache (for example u8) against the runtime default.

Greedy (temperature 0) generations from two pipelines that differ only in KV_CACHE_PRECISION are
compared token by token. A prompt passes if the outputs agree for at least min_prefix_tokens tokens (or
entirely, when the reference output is shorter); the gate passes if at least required_pass_fraction of
the prompts pass. This is a prefix-agreement criterion only: it does not compute KL divergence or
perplexity, which need logits the pipeline does not expose. Passing does not change any default.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from arc_runner.core.exceptions import ArcRunnerError
from arc_runner.engine.models import GenerationConfig

DEFAULT_MIN_PREFIX_TOKENS = 32
DEFAULT_REQUIRED_PASS_FRACTION = 0.9
MAX_CANDIDATE_ERROR_CHARS = 2000


@dataclass(frozen=True)
class PromptComparison:
    """Agreement between the two precisions for one prompt.

    Attributes:
        prompt: The prompt text.
        reference_tokens: Token count of the default-precision output.
        candidate_tokens: Token count of the reduced-precision output.
        prefix_tokens: Length of the common leading token sequence.
        exact_match: True if both outputs are identical token for token.
        passes: True if prefix_tokens reaches the required length (or the whole reference output).
    """

    prompt: str
    reference_tokens: int
    candidate_tokens: int
    prefix_tokens: int
    exact_match: bool
    passes: bool


@dataclass(frozen=True)
class KvQualityReport:
    """Result of a KV-precision comparison.

    Attributes:
        prompts: Per-prompt comparisons.
        min_prefix_tokens: Required agreeing prefix length.
        required_pass_fraction: Fraction of prompts that must pass.
        pass_count: Prompts that passed.
        exact_matches: Prompts with identical outputs.
        passed: Gate verdict.
        candidate_error: Set when the runtime refused to load or run the reduced precision (the gate
            then fails and no agreement was measured); None otherwise.
    """

    prompts: List[PromptComparison]
    min_prefix_tokens: int
    required_pass_fraction: float
    pass_count: int
    exact_matches: int
    passed: bool
    candidate_error: Optional[str] = None


def common_prefix_length(first: Sequence[int], second: Sequence[int]) -> int:
    """Return the number of leading elements two sequences share.

    Args:
        first: First token sequence.
        second: Second token sequence.

    Returns:
        int: Length of the common prefix.
    """
    count = 0
    for left, right in zip(first, second):
        if left != right:
            break
        count += 1
    return count


def _generate_tokens(engine: Any, prompts: Sequence[str], max_new_tokens: int) -> List[List[int]]:
    """Generate greedily for every prompt and tokenize each output.

    Args:
        engine: Loaded engine (generate, tokenize).
        prompts: Prompts.
        max_new_tokens: Completion budget.

    Returns:
        List[List[int]]: Output token ids per prompt.

    Raises:
        RuntimeError: If the engine cannot tokenize its output.
    """
    config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0)
    outputs: List[List[int]] = []
    for prompt in prompts:
        text = engine.generate(prompt, config)
        tokens = engine.tokenize(text)
        if tokens is None:
            raise RuntimeError("engine could not tokenize its output; the KV quality gate cannot compare generations")
        outputs.append(list(tokens))
    return outputs


def compare_kv_precision(
    engine_factory: Callable[[str], Any],
    prompts: Sequence[str],
    max_new_tokens: int,
    min_prefix_tokens: int = DEFAULT_MIN_PREFIX_TOKENS,
    required_pass_fraction: float = DEFAULT_REQUIRED_PASS_FRACTION,
    candidate_precision: str = "u8",
) -> KvQualityReport:
    """Compare greedy generations at the default KV precision and a reduced one.

    Args:
        engine_factory: Function taking a KV precision ("" for the runtime default) and returning a
            loaded engine with generate and tokenize.
        prompts: Prompts to generate for; must not be empty.
        max_new_tokens: Completion budget for both precisions.
        min_prefix_tokens: Agreeing prefix length required per prompt.
        required_pass_fraction: Fraction of prompts that must pass.
        candidate_precision: The reduced precision under test.

    Returns:
        KvQualityReport: Per-prompt agreement and the gate verdict.

    Raises:
        ValueError: If prompts is empty.
        RuntimeError: If the reference engine fails or cannot tokenize its output. A failure of the
            reduced-precision engine is not raised: it is recorded in KvQualityReport.candidate_error.
    """
    if not prompts:
        raise ValueError("the KV quality gate needs at least one prompt")
    reference = _generate_tokens(engine_factory(""), prompts, max_new_tokens)
    try:
        candidate = _generate_tokens(engine_factory(candidate_precision), prompts, max_new_tokens)
    except (RuntimeError, ArcRunnerError) as exc:
        return KvQualityReport([], min_prefix_tokens, required_pass_fraction, 0, 0, False, str(exc)[:MAX_CANDIDATE_ERROR_CHARS])
    comparisons: List[PromptComparison] = []
    for prompt, ref, cand in zip(prompts, reference, candidate):
        prefix = common_prefix_length(ref, cand)
        needed = min(min_prefix_tokens, len(ref))
        exact = ref == cand
        comparisons.append(PromptComparison(prompt, len(ref), len(cand), prefix, exact, exact or prefix >= needed))
    pass_count = sum(1 for c in comparisons if c.passes)
    passed = pass_count * 1.0 >= required_pass_fraction * len(comparisons) - 1e-9
    return KvQualityReport(
        comparisons, min_prefix_tokens, required_pass_fraction, pass_count, sum(1 for c in comparisons if c.exact_match), passed
    )


def write_report(path: Path, report: KvQualityReport, model: str, device: str) -> None:
    """Write the report as JSON.

    Args:
        path: Destination file.
        report: Comparison result.
        model: Model identifier for the record.
        device: Device the comparison ran on.
    """
    document = {
        "model": model,
        "device": device,
        "passed": report.passed,
        "min_prefix_tokens": report.min_prefix_tokens,
        "required_pass_fraction": report.required_pass_fraction,
        "pass_count": report.pass_count,
        "exact_matches": report.exact_matches,
        "candidate_error": report.candidate_error,
        "prompts": [asdict(c) for c in report.prompts],
        "note": "Prefix agreement of greedy generations only (no KL/perplexity). A pass does not change the default KV precision; that is a separate decision.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
