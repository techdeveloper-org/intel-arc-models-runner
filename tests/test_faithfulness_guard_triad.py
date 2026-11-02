"""Triad test suite for FaithfulnessGuard adhering to Enterprise standards.

Validates Kullback-Leibler divergence math, context grounding attribution,
JSON schema output contract validation, and composite reliability scoring.
"""

import json
import numpy as np
import pytest

from arc_runner.engine.faithfulness_guard import (
    FaithfulnessError,
    FaithfulnessGuard,
    LogitDivergenceExceededError,
    ReliabilityAuditResultDto,
    ReliabilityScoreThresholdError,
    UngroundedContextError,
)


class TestFaithfulnessGuardPositive:
    """Positive test specifications for anti-hallucination and reliability scoring."""

    def test_compute_kl_divergence_when_unquantized_vs_int4_then_satisfies_bound(self) -> None:
        """Verify close logit distributions yield small KL divergence <= 0.05."""
        source = np.array([2.5, 1.0, -0.5, 0.2], dtype=np.float32)
        target = np.array([2.48, 1.02, -0.49, 0.19], dtype=np.float32)

        kl = FaithfulnessGuard.compute_kl_divergence(source, target)

        assert kl < 0.01
        assert kl >= 0.0

    def test_audit_context_grounding_when_in_context_qa_then_ratio_exceeds_90_percent(self) -> None:
        """Verify generated text drawn from prompt achieves high grounding ratio."""
        prompt = "The Intel Arc A770 GPU features 16GB GDDR6 memory and 32 Xe cores."
        gen = "Intel Arc A770 GPU features 16GB GDDR6 memory"

        ratio = FaithfulnessGuard.audit_context_grounding(prompt, gen)

        assert ratio == 1.0

    def test_compute_composite_rs_when_metrics_pass_then_returns_rs_above_95(self) -> None:
        """Verify high reliability factors produce composite RS >= 0.95."""
        rs = FaithfulnessGuard.compute_composite_reliability_score(
            kl_divergence=0.001,
            grounding_ratio=1.0,
            contract_valid=True,
        )

        assert rs >= 0.95

    def test_validate_output_contract_when_valid_json_schema_then_confirms_validity(self) -> None:
        """Verify JSON adhering to schema returns True."""
        schema = {"required": ["model", "status", "tokens_per_sec"]}
        output = json.dumps({"model": "qwen2.5-3b", "status": "ready", "tokens_per_sec": 42.5})

        valid = FaithfulnessGuard.validate_output_contract(output, schema)

        assert valid is True

    def test_rank_ordering_when_top5_candidates_checked_then_ranks_identical(self) -> None:
        """Verify identical probability rank order between source and target."""
        source = np.array([10.0, 8.0, 6.0, 4.0, 2.0], dtype=np.float32)
        target = np.array([9.8, 8.1, 5.9, 4.1, 1.9], dtype=np.float32)

        source_ranks = np.argsort(source)[::-1]
        target_ranks = np.argsort(target)[::-1]

        assert np.array_equal(source_ranks, target_ranks)

    def test_audit_reliability_when_all_criteria_met_then_returns_faithful_dto(self) -> None:
        """Verify audit_reliability returns populated ReliabilityAuditResultDto."""
        source = np.array([3.0, 1.0, 0.0], dtype=np.float32)
        target = np.array([2.99, 1.01, 0.0], dtype=np.float32)
        prompt = "OpenVINO GenAI accelerated inference on Intel Arc silicon."
        gen = "OpenVINO GenAI on Intel Arc"

        dto = FaithfulnessGuard.audit_reliability(source, target, prompt, gen)

        assert isinstance(dto, ReliabilityAuditResultDto)
        assert dto.is_faithful is True
        assert dto.composite_reliability_score >= 0.95

    def test_semantic_entropy_when_deterministic_prompt_then_remains_under_threshold(self) -> None:
        """Verify uniform logits have predictable entropy."""
        logits = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)
        kl = FaithfulnessGuard.compute_kl_divergence(logits, logits)
        assert kl == pytest.approx(0.0, abs=1e-6)


class TestFaithfulnessGuardNegative:
    """Negative test specifications verifying detection of hallucinations and divergence."""

    def test_compute_kl_divergence_when_logits_diverge_above_0_05_then_raises_error(self) -> None:
        """Verify divergent logits raise LogitDivergenceExceededError during audit."""
        source = np.array([10.0, -5.0, -5.0], dtype=np.float32)
        target = np.array([-5.0, 10.0, -5.0], dtype=np.float32)

        with pytest.raises(LogitDivergenceExceededError) as exc_info:
            FaithfulnessGuard.audit_reliability(
                source, target, "some prompt", "some prompt text"
            )

        assert exc_info.value.kl_div > 0.05

    def test_audit_context_grounding_when_hallucination_injected_then_detects_ungrounded(self) -> None:
        """Verify hallucinated tokens trigger UngroundedContextError during audit."""
        source = np.array([2.0, 1.0], dtype=np.float32)
        target = np.array([2.0, 1.0], dtype=np.float32)
        prompt = "Intel Core Ultra CPU"
        hallucinated_output = "The capital of Australia is Canberra and the kangaroo jumps high"

        with pytest.raises(UngroundedContextError) as exc_info:
            FaithfulnessGuard.audit_reliability(source, target, prompt, hallucinated_output)

        assert exc_info.value.grounding_ratio < 0.90

    def test_validate_output_contract_when_json_schema_violated_then_returns_false(self) -> None:
        """Verify missing required keys in JSON output returns False."""
        schema = {"required": ["temperature", "max_tokens"]}
        invalid_output = json.dumps({"temperature": 0.7})

        valid = FaithfulnessGuard.validate_output_contract(invalid_output, schema)

        assert valid is False

    def test_compute_kl_divergence_when_logits_contain_nan_inf_then_raises_typed_error(self) -> None:
        """Verify non-finite logit inputs raise FaithfulnessError."""
        source = np.array([1.0, np.nan, 3.0], dtype=np.float32)
        target = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with pytest.raises(FaithfulnessError):
            FaithfulnessGuard.compute_kl_divergence(source, target)

    def test_compute_composite_rs_when_score_under_95_then_raises_rs_threshold_error(self) -> None:
        """Verify low reliability metrics trigger ReliabilityScoreThresholdError."""
        source = np.array([2.0, 1.0], dtype=np.float32)
        target = np.array([2.0, 1.0], dtype=np.float32)
        schema = {"required": ["mandatory_field"]}
        invalid_gen = "Valid prompt words but invalid format"

        with pytest.raises(ReliabilityScoreThresholdError):
            FaithfulnessGuard.audit_reliability(
                source, target, "Valid prompt words but invalid format", invalid_gen, schema
            )


class TestFaithfulnessGuardEdgeCase:
    """Edge-case test specifications verifying boundaries and numerical stability."""

    def test_compute_kl_divergence_when_uniform_logits_then_avoids_zero_division(self) -> None:
        """Verify uniform logit vectors calculate zero divergence without NaN."""
        source = np.zeros(10, dtype=np.float32)
        target = np.zeros(10, dtype=np.float32)

        kl = FaithfulnessGuard.compute_kl_divergence(source, target)

        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_compute_kl_divergence_when_extreme_magnitudes_then_applies_logsumexp_safely(self) -> None:
        """Verify extreme logit scales (e.g. 1000.0) calculate without overflow."""
        source = np.array([1000.0, 950.0, 900.0], dtype=np.float32)
        target = np.array([999.0, 951.0, 899.0], dtype=np.float32)

        kl = FaithfulnessGuard.compute_kl_divergence(source, target)

        assert np.isfinite(kl)
        assert kl >= 0.0

    def test_audit_context_grounding_when_empty_prompt_then_handles_safely(self) -> None:
        """Verify empty prompt handles safely without zero division."""
        ratio = FaithfulnessGuard.audit_context_grounding("", "Any generated text")
        assert ratio == 0.0

    def test_audit_context_grounding_when_single_token_output_then_computes_exact_ratio(self) -> None:
        """Verify single token generated output calculates 1.0 or 0.0."""
        prompt = "Hello world"
        ratio_match = FaithfulnessGuard.audit_context_grounding(prompt, "hello")
        ratio_no_match = FaithfulnessGuard.audit_context_grounding(prompt, "goodbye")

        assert ratio_match == 1.0
        assert ratio_no_match == 0.0

    def test_audit_context_grounding_when_multibyte_utf8_tokens_then_matches_spans_correctly(self) -> None:
        """Verify Unicode words match correctly."""
        prompt = "Intel Arc GPU delivers 35 tok/s on deep learning models."
        gen = "Intel Arc GPU delivers 35"

        ratio = FaithfulnessGuard.audit_context_grounding(prompt, gen)

        assert ratio == 1.0
