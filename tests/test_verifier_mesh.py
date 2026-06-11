"""Verifier Mesh checks.

Tests URL/DOI resolution (mocked), numeric range checking,
and claim-evidence entailment.
"""

import asyncio
import builtins
import sys
from types import SimpleNamespace

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from biobank_agent.verifier_mesh import (
    URLDOIResolver,
    NumericRangeChecker,
    ClaimEvidenceEntailment,
    VerifierMesh,
    VerificationCheck,
    MeshResult,
)


# ── URL/DOI Resolver Tests ───────────────────────────────────────────────────


class TestURLDOIResolver:
    """Test URL and DOI extraction and resolution."""

    def test_extract_urls(self):
        """Should extract HTTP/HTTPS URLs from text."""
        resolver = URLDOIResolver()
        text = "See https://example.com/paper and http://test.org/doc for details."
        refs = resolver.extract_references(text)
        urls = [r["value"] for r in refs if r["type"] == "url"]
        assert len(urls) == 2
        assert "https://example.com/paper" in urls

    def test_extract_dois(self):
        """Should extract DOI patterns from text."""
        resolver = URLDOIResolver()
        text = "Published at 10.1038/s41588-024-01898-1 and discussed by others."
        refs = resolver.extract_references(text)
        dois = [r for r in refs if r["type"] == "doi"]
        assert len(dois) == 1
        assert "10.1038" in dois[0]["value"]

    def test_empty_text_no_refs(self):
        """Empty text should return no references."""
        resolver = URLDOIResolver()
        refs = resolver.extract_references("")
        assert refs == []

    def test_no_urls_in_plain_text(self):
        """Plain text without URLs should return nothing."""
        resolver = URLDOIResolver()
        refs = resolver.extract_references("This is just normal text about diabetes.")
        assert refs == []

    def test_verify_references_skips_when_httpx_missing(self, monkeypatch):
        """Missing httpx should produce an info-level skip check."""
        real_import = builtins.__import__

        def no_httpx(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_httpx)
        checks = asyncio.run(URLDOIResolver().verify_references("See https://example.org."))

        assert len(checks) == 1
        assert checks[0].passed is True
        assert checks[0].severity == "info"

    def test_verify_references_success_warning_and_connection_error(self, monkeypatch):
        """URL resolver should report resolved, HTTP failure, and connection failure refs."""
        calls = []

        class FakeAsyncClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def head(self, url):
                calls.append(url)
                if "missing" in url:
                    return SimpleNamespace(status_code=404)
                if "offline" in url:
                    raise RuntimeError("offline")
                return SimpleNamespace(status_code=200)

        monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(AsyncClient=FakeAsyncClient))

        checks = asyncio.run(
            URLDOIResolver(timeout=1).verify_references(
                "Refs https://ok.example, https://missing.example, and https://offline.example"
            )
        )

        assert calls == ["https://ok.example", "https://missing.example", "https://offline.example"]
        assert [c.passed for c in checks] == [True, False, False]
        assert checks[1].detail == "HTTP 404"
        assert "Connection error" in checks[2].detail

    def test_verify_references_sync_inside_running_loop(self, monkeypatch):
        """The sync wrapper should offload when called from an active event loop."""
        resolver = URLDOIResolver()

        async def fake_verify(text):
            return [VerificationCheck("url_resolver", "url", text, True)]

        monkeypatch.setattr(resolver, "verify_references", fake_verify)

        async def call_sync():
            return resolver.verify_references_sync("https://example.org")

        checks = asyncio.run(call_sync())

        assert checks[0].claim == "https://example.org"


# ── Numeric Range Checker Tests ──────────────────────────────────────────────


class TestNumericRangeChecker:
    """Test biobank-specific numeric bounds."""

    @pytest.fixture
    def checker(self):
        return NumericRangeChecker()

    def test_valid_sample_size(self, checker):
        """Normal sample size should pass."""
        result = {"n_cases": 50000, "n_controls": 100000}
        checks = checker.verify_result(result)
        assert all(c.passed for c in checks)

    def test_sample_exceeds_ukb(self, checker):
        """Sample size > 502,536 should fail."""
        result = {"n_cases": 600000}
        checks = checker.verify_result(result)
        failed = [c for c in checks if not c.passed]
        assert len(failed) >= 1
        assert "Exceeds UKB maximum" in failed[0].detail

    def test_negative_sample_fails(self, checker):
        """Negative sample size should fail."""
        result = {"n_participants": -100}
        checks = checker.verify_result(result)
        failed = [c for c in checks if not c.passed]
        assert len(failed) >= 1
        assert "negative" in failed[0].detail.lower()

    def test_cases_plus_controls_exceeds(self, checker):
        """Sum of cases + controls > UKB total should fail."""
        result = {"n_cases": 300000, "n_controls": 300000}
        checks = checker.verify_result(result)
        failed = [c for c in checks if not c.passed]
        assert len(failed) >= 1
        assert "Sum exceeds" in failed[0].detail

    def test_auc_out_of_range(self, checker):
        """AUC > 1 should fail."""
        result = {"auc": 1.2}
        checks = checker.verify_result(result)
        failed = [c for c in checks if not c.passed]
        assert len(failed) >= 1

    def test_valid_auc(self, checker):
        """AUC in [0, 1] should pass."""
        result = {"auc": 0.85}
        checks = checker.verify_result(result)
        auc_checks = [c for c in checks if "auc" in c.claim]
        # Should have no AUC-related failures
        assert not any(c for c in auc_checks if not c.passed)

    def test_negative_hazard_ratio(self, checker):
        """Negative HR should fail."""
        result = {"hazard_ratio": -0.5}
        checks = checker.verify_result(result)
        failed = [c for c in checks if not c.passed]
        assert len(failed) >= 1

    def test_positive_effect_estimates_pass(self, checker):
        """Positive-only epidemiological measures should pass when above zero."""
        checks = checker.verify_result({"odds_ratio": 1.2, "relative_risk": 0.7})
        claims = {c.claim for c in checks}
        assert {"odds_ratio=1.2", "relative_risk=0.7"}.issubset(claims)
        assert all(c.passed for c in checks)

    def test_p_value_out_of_range(self, checker):
        """p-value > 1 should fail."""
        result = {"p_value": 1.5}
        checks = checker.verify_result(result)
        failed = [c for c in checks if not c.passed]
        assert len(failed) >= 1

    def test_ci_ordering_violation(self, checker):
        """CI lower > upper should fail."""
        result = {"ci_lower": 0.9, "ci_upper": 0.3}
        checks = checker.verify_result(result)
        failed = [c for c in checks if not c.passed]
        assert len(failed) >= 1
        assert "lower bound exceeds upper" in failed[0].detail.lower()

    def test_prevalence_out_of_range(self, checker):
        """Prevalence > 100% should fail."""
        result = {"prevalence": 150.0}
        checks = checker.verify_result(result)
        failed = [c for c in checks if not c.passed]
        assert len(failed) >= 1

    def test_numeric_checker_ignores_nonnumeric_values_and_accepts_valid_stats(self, checker):
        """String-valued metrics should be ignored while valid numeric stats pass."""
        result = {
            "auc": "0.85",
            "hazard_ratio": "1.2",
            "p_value": "0.05",
            "p": 0.05,
            "ci_lower": 0.3,
            "ci_upper": 0.9,
            "prevalence": 15.0,
        }

        checks = checker.verify_result(result)

        assert all(c.passed for c in checks)
        assert not any(c.claim == "auc=0.85" for c in checks)
        assert not any(c.claim == "hazard_ratio=1.2" for c in checks)


# ── Claim-Evidence Entailment Tests ──────────────────────────────────────────


class TestClaimEvidenceEntailment:
    """Test NLI-based claim verification."""

    def test_no_llm_skips(self):
        """Without LLM, should return info-level skip."""
        checker = ClaimEvidenceEntailment(llm=None)
        check = checker.check_entailment("HbA1c predicts diabetes", "Some evidence")
        assert check.passed
        assert check.confidence == 0.0
        assert "skipping" in check.detail.lower()

    def test_entails_verdict(self):
        """LLM returning ENTAILS should pass."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(text="ENTAILS | The evidence clearly supports this claim")
        checker = ClaimEvidenceEntailment(llm=mock_llm)
        check = checker.check_entailment("Test claim", "Supporting evidence")
        assert check.passed
        assert check.confidence == 0.8

    def test_contradicts_verdict(self):
        """LLM returning CONTRADICTS should fail."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(text="CONTRADICTS | The evidence says the opposite")
        checker = ClaimEvidenceEntailment(llm=mock_llm)
        check = checker.check_entailment("Test claim", "Opposing evidence")
        assert not check.passed
        assert check.severity == "error"

    def test_neutral_verdict(self):
        """LLM returning NEUTRAL should pass with low confidence."""
        mock_llm = MagicMock()
        mock_llm.chat.return_value = MagicMock(text="NEUTRAL | Unrelated topic")
        checker = ClaimEvidenceEntailment(llm=mock_llm)
        check = checker.check_entailment("Test claim", "Unrelated text")
        assert check.passed
        assert check.confidence == 0.5

    def test_llm_error_graceful(self):
        """LLM errors should be handled gracefully."""
        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("API error")
        checker = ClaimEvidenceEntailment(llm=mock_llm)
        check = checker.check_entailment("Test claim", "Evidence")
        assert check.passed  # Graceful: don't block on verification failure
        assert check.confidence == 0.0

    def test_batch_check_delegates_each_pair(self):
        """Batch entailment should return one check per claim/evidence pair."""
        checker = ClaimEvidenceEntailment(llm=None)
        checks = checker.batch_check([("claim 1", "evidence 1"), ("claim 2", "evidence 2")])
        assert [c.claim for c in checks] == ["claim 1", "claim 2"]


# ── Mesh Result Tests ────────────────────────────────────────────────────────


class TestMeshResult:
    """Test result aggregation."""

    def test_empty_mesh_passes(self):
        """Empty mesh should have pass status."""
        result = MeshResult()
        assert result.overall_status == "pass"
        assert result.n_passed == 0

    def test_add_passing_check(self):
        """Adding a passing check should increment counter."""
        result = MeshResult()
        result.add_check(VerificationCheck(
            verifier="test", category="test", claim="x", passed=True
        ))
        assert result.n_passed == 1
        assert result.overall_status == "pass"

    def test_warning_changes_status(self):
        """A failed warning should change status to warning."""
        result = MeshResult()
        result.add_check(VerificationCheck(
            verifier="test", category="test", claim="x",
            passed=False, severity="warning"
        ))
        assert result.overall_status == "warning"

    def test_error_changes_status_to_fail(self):
        """A failed error should change status to fail."""
        result = MeshResult()
        result.add_check(VerificationCheck(
            verifier="test", category="test", claim="x",
            passed=False, severity="error"
        ))
        assert result.overall_status == "fail"

    def test_summary_includes_counts(self):
        """Summary should mention pass/fail counts."""
        result = MeshResult()
        result.add_check(VerificationCheck(
            verifier="a", category="num", claim="ok", passed=True))
        result.add_check(VerificationCheck(
            verifier="b", category="num", claim="bad",
            passed=False, severity="error"))
        summary = result.summary()
        assert "1 passed" in summary
        assert "1 errors" in summary


# ── Full Verifier Mesh Tests ─────────────────────────────────────────────────


class TestVerifierMesh:
    """Test the full mesh orchestration."""

    def test_mesh_numeric_only(self):
        """Mesh with just numeric data should run numeric checks."""
        mesh = VerifierMesh(llm=None)
        result = mesh.verify_result({"n_cases": 100, "auc": 0.85})
        assert result.n_passed > 0
        assert result.overall_status == "pass"

    def test_mesh_catches_violation(self):
        """Mesh should catch numeric violations."""
        mesh = VerifierMesh(llm=None)
        result = mesh.verify_result({"n_cases": 999999})
        assert result.overall_status == "fail"

    def test_mesh_with_text_content(self):
        """Mesh should handle text with URLs (gracefully even without httpx)."""
        mesh = VerifierMesh(llm=None)
        result = mesh.verify_result(
            {"n_cases": 100},
            text_content="See https://example.com for details"
        )
        # Should not crash regardless of httpx availability
        assert result.overall_status in ("pass", "warning")

    def test_verify_text_claims(self):
        """Should extract numeric claims from free text."""
        mesh = VerifierMesh(llm=None)
        text = "Our cohort had n=50000 participants with AUC of 0.82."
        result = mesh.verify_text_claims(text)
        # Should have found and validated the sample size and AUC
        assert result.n_passed > 0

    def test_checker_execution_errors_are_visible(self, monkeypatch):
        """Verifier failures should not silently return an empty pass result."""
        mesh = VerifierMesh(llm=None)
        monkeypatch.setattr(
            mesh.numeric_checker,
            "verify_result",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("numeric down")),
        )
        monkeypatch.setattr(
            mesh.url_resolver,
            "verify_references_sync",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("url down")),
        )
        monkeypatch.setattr(
            mesh.entailment,
            "batch_check",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("entailment down")),
        )

        result = mesh.verify_result(
            {"n_cases": 1},
            text_content="https://example.org",
            claims=[("claim", "evidence")],
        )

        assert result.overall_status == "fail"
        assert result.n_failed == 1
        assert result.n_warnings == 2
        assert [check.verifier for check in result.checks] == [
            "numeric_range",
            "url_resolver",
            "entailment",
        ]

    def test_mesh_adds_claim_checks_and_can_skip_sample_sizes(self, monkeypatch):
        """Successful entailment checks should be merged with other mesh checks."""
        mesh = VerifierMesh(llm=None)
        monkeypatch.setattr(
            mesh.entailment,
            "batch_check",
            lambda claims: [VerificationCheck("entailment", "entailment", claims[0][0], True)],
        )

        result = mesh.verify_result(
            {"n_cases": 999999, "auc": 0.8},
            claims=[("HbA1c is associated with diabetes", "Evidence text")],
            skip_sample_sizes=True,
        )

        assert result.overall_status == "pass"
        assert any(check.verifier == "entailment" for check in result.checks)
        assert not any("n_cases" in check.claim for check in result.checks)

    def test_verify_text_claims_adds_successful_url_checks(self, monkeypatch):
        """Text-claim verification should merge URL resolver results."""
        mesh = VerifierMesh(llm=None)
        monkeypatch.setattr(
            mesh.url_resolver,
            "verify_references_sync",
            lambda text: [VerificationCheck("url_resolver", "url", "Reference resolves", True)],
        )

        result = mesh.verify_text_claims("AUC=0.8. See https://example.org")

        assert result.overall_status == "pass"
        assert any(check.verifier == "url_resolver" for check in result.checks)

    def test_verify_text_claims_handles_parse_and_url_errors(self, monkeypatch):
        """Malformed numeric claims and URL resolver errors should be handled."""
        mesh = VerifierMesh(llm=None)
        monkeypatch.setattr(
            mesh.url_resolver,
            "verify_references_sync",
            lambda text: (_ for _ in ()).throw(RuntimeError("resolver down")),
        )

        result = mesh.verify_text_claims("AUC=1.2. prevalence: 150%. p < 1e-. See https://example.org")

        assert result.overall_status == "fail"
        assert any("prevalence=150.0" in check.claim.lower() for check in result.checks)
        assert any(check.verifier == "url_resolver" for check in result.checks)
