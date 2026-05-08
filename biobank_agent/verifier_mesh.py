"""Extended verification mesh — multi-strategy claim verification.

Extends the Z3/bounds verification in verification.py with:
  1. URL/DOI resolver: checks that cited URLs/DOIs actually resolve
  2. Numeric range checker: biobank-specific bounds (UKB hard constraints)
  3. Claim-evidence entailment: LLM-judged NLI between claim and evidence

Design sources:
  - GPT research doc: "verifier mesh: source resolver, DOI/URL verifier,
    claim-evidence entailment checker, numeric checker, cohort consistency"
  - Claude research doc §C2: "GenPRM — code-verified generative PRMs"
  - Claude research doc §C6: "PaperQA2's citation-first design"

Architecture:
    Each verifier is independent and produces Check objects compatible
    with the existing verdict.py pipeline. The mesh runs all verifiers
    and merges results into a single VerdictResult.

Integration:
    verdict.py → formal_check() → verifier_mesh.verify_all() → LLM verdict
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class VerificationCheck:
    """A single verification check result."""
    verifier: str          # Name of the verifier that produced this check
    category: str          # "url", "numeric", "entailment", "cohort"
    claim: str             # What was being verified
    passed: bool           # Whether the check passed
    detail: str = ""       # Explanation
    severity: str = "warning"  # "info", "warning", "error"
    confidence: float = 1.0    # How confident we are in this check result


@dataclass
class MeshResult:
    """Aggregated result from the full verifier mesh."""
    checks: list[VerificationCheck] = field(default_factory=list)
    n_passed: int = 0
    n_failed: int = 0
    n_warnings: int = 0
    overall_status: str = "pass"  # "pass", "warning", "fail"

    def add_check(self, check: VerificationCheck) -> None:
        """Add a check and update counters."""
        self.checks.append(check)
        if check.passed:
            self.n_passed += 1
        elif check.severity == "error":
            self.n_failed += 1
            self.overall_status = "fail"
        else:
            self.n_warnings += 1
            if self.overall_status == "pass":
                self.overall_status = "warning"

    def summary(self) -> str:
        """Human-readable summary of all checks."""
        lines = [f"Verification Mesh: {self.overall_status.upper()} "
                 f"({self.n_passed} passed, {self.n_warnings} warnings, {self.n_failed} errors)"]
        for check in self.checks:
            icon = "✓" if check.passed else ("⚠" if check.severity == "warning" else "✗")
            lines.append(f"  {icon} [{check.verifier}] {check.claim[:80]}: {check.detail[:100]}")
        return "\n".join(lines)


# ── URL/DOI Resolver ─────────────────────────────────────────────────────────


class URLDOIResolver:
    """Verify that cited URLs and DOIs actually resolve.

    Uses HTTP HEAD requests to check URL accessibility without
    downloading full content. Converts DOIs to URLs via doi.org.
    """

    # Common URL and DOI patterns
    _URL_RE = re.compile(r'https?://[^\s<>"\']+')
    _DOI_RE = re.compile(r'10\.\d{4,9}/[^\s]+')

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def extract_references(self, text: str) -> list[dict[str, str]]:
        """Extract URLs and DOIs from text.

        Returns list of {"type": "url"|"doi", "value": "..."}
        """
        refs = []
        # Find URLs
        for match in self._URL_RE.finditer(text):
            url = match.group().rstrip(".,;:)")
            refs.append({"type": "url", "value": url})
        # Find DOIs (convert to URLs)
        for match in self._DOI_RE.finditer(text):
            doi = match.group().rstrip(".,;:)")
            refs.append({"type": "doi", "value": doi, "url": f"https://doi.org/{doi}"})
        return refs

    async def verify_references(self, text: str) -> list[VerificationCheck]:
        """Check each URL/DOI in text resolves (HEAD request).

        Returns a VerificationCheck per reference found.
        """
        refs = self.extract_references(text)
        if not refs:
            return []

        checks = []
        try:
            import httpx
        except ImportError:
            logger.debug("httpx not available for URL verification")
            return [VerificationCheck(
                verifier="url_resolver",
                category="url",
                claim=f"Found {len(refs)} references to verify",
                passed=True,
                detail="httpx not installed; skipping URL resolution",
                severity="info",
            )]

        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={"User-Agent": "BioBank-Agent/2.1 VerifierMesh"}
        ) as client:
            for ref in refs[:10]:  # Limit to first 10 to avoid rate limiting
                url = ref.get("url", ref["value"])
                try:
                    response = await client.head(url)
                    resolved = response.status_code < 400
                    checks.append(VerificationCheck(
                        verifier="url_resolver",
                        category="url",
                        claim=f"Reference resolves: {ref['value'][:60]}",
                        passed=resolved,
                        detail=f"HTTP {response.status_code}" if not resolved else "OK",
                        severity="warning" if not resolved else "info",
                    ))
                except Exception as e:
                    checks.append(VerificationCheck(
                        verifier="url_resolver",
                        category="url",
                        claim=f"Reference resolves: {ref['value'][:60]}",
                        passed=False,
                        detail=f"Connection error: {type(e).__name__}",
                        severity="warning",
                    ))

        return checks

    def verify_references_sync(self, text: str) -> list[VerificationCheck]:
        """Synchronous wrapper for verify_references.

        Handles both cases: called from within a running event loop
        (offloads to thread) or from synchronous code (uses asyncio.run).
        """
        try:
            asyncio.get_running_loop()
            # We're inside a running event loop — offload to a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.verify_references(text))
                return future.result(timeout=30)
        except RuntimeError:
            # No running event loop — call asyncio.run() directly
            return asyncio.run(self.verify_references(text))


# ── Numeric Range Checker ────────────────────────────────────────────────────


class NumericRangeChecker:
    """Biobank-domain numeric bounds validation.

    Hard-codes UK Biobank reality constraints that no valid analysis
    can violate. These are deterministic checks (no LLM needed).

    Uses shared constants from biobank_agent.constants for consistency
    with the formal verifier (BiobankConstraintVerifier).
    """

    def __init__(self):
        from .constants import (
            UKB_TOTAL_PARTICIPANTS, UKB_MIN_AGE_RECRUITMENT,
            UKB_MAX_AGE_RECRUITMENT, UKB_ASSESSMENT_CENTRES,
            UKB_VALID_CHROMOSOMES,
        )
        self.UKB_MAX_PARTICIPANTS = UKB_TOTAL_PARTICIPANTS
        self.UKB_MIN_AGE_RECRUITMENT = UKB_MIN_AGE_RECRUITMENT
        self.UKB_MAX_AGE_RECRUITMENT = UKB_MAX_AGE_RECRUITMENT
        self.UKB_ASSESSMENT_CENTRES = UKB_ASSESSMENT_CENTRES
        self.VALID_CHROMOSOMES = UKB_VALID_CHROMOSOMES

    def verify_result(
        self, result: dict[str, Any], skip_sample_sizes: bool = False
    ) -> list[VerificationCheck]:
        """Run all numeric checks on a skill result dictionary.

        Args:
            result: Skill output dictionary to verify
            skip_sample_sizes: If True, skip sample-size checks (already
                covered by formal verifier in the verdict pipeline)

        Checks for:
            - Sample sizes within UKB bounds (unless skip_sample_sizes)
            - Metrics within valid ranges
            - Statistical values valid
        """
        checks = []
        if not skip_sample_sizes:
            checks.extend(self._check_sample_sizes(result))
        checks.extend(self._check_metrics(result))
        checks.extend(self._check_statistical_values(result))
        return checks

    def _check_sample_sizes(self, result: dict[str, Any]) -> list[VerificationCheck]:
        """Verify sample sizes don't exceed UKB total."""
        checks = []
        size_keys = {"n_cases", "n_controls", "total", "n_total", "n_participants",
                     "cohort_size", "sample_size", "n_samples"}

        for key in size_keys:
            if key in result:
                val = result[key]
                if isinstance(val, (int, float)) and val > self.UKB_MAX_PARTICIPANTS:
                    checks.append(VerificationCheck(
                        verifier="numeric_range",
                        category="numeric",
                        claim=f"{key}={val}",
                        passed=False,
                        detail=f"Exceeds UKB maximum ({self.UKB_MAX_PARTICIPANTS})",
                        severity="error",
                    ))
                elif isinstance(val, (int, float)) and val < 0:
                    checks.append(VerificationCheck(
                        verifier="numeric_range",
                        category="numeric",
                        claim=f"{key}={val}",
                        passed=False,
                        detail="Sample size cannot be negative",
                        severity="error",
                    ))
                else:
                    checks.append(VerificationCheck(
                        verifier="numeric_range",
                        category="numeric",
                        claim=f"{key}={val}",
                        passed=True,
                        detail="Within UKB bounds",
                    ))

        # Check sum of cases + controls
        n_cases = result.get("n_cases")
        n_controls = result.get("n_controls")
        if isinstance(n_cases, (int, float)) and isinstance(n_controls, (int, float)):
            total = n_cases + n_controls
            if total > self.UKB_MAX_PARTICIPANTS:
                checks.append(VerificationCheck(
                    verifier="numeric_range",
                    category="numeric",
                    claim=f"n_cases + n_controls = {total}",
                    passed=False,
                    detail=f"Sum exceeds UKB maximum ({self.UKB_MAX_PARTICIPANTS})",
                    severity="error",
                ))

        return checks

    def _check_metrics(self, result: dict[str, Any]) -> list[VerificationCheck]:
        """Verify model performance metrics are in valid ranges."""
        checks = []
        bounded_01 = {"auc", "auroc", "auprc", "accuracy", "f1", "precision",
                      "recall", "sensitivity", "specificity", "r2"}
        positive_only = {"hazard_ratio", "hr", "odds_ratio", "or_value",
                         "relative_risk", "rr"}

        for key in bounded_01:
            if key in result:
                val = result[key]
                if isinstance(val, (int, float)):
                    if val < 0 or val > 1:
                        checks.append(VerificationCheck(
                            verifier="numeric_range",
                            category="numeric",
                            claim=f"{key}={val}",
                            passed=False,
                            detail=f"Must be in [0, 1], got {val}",
                            severity="error",
                        ))
                    else:
                        checks.append(VerificationCheck(
                            verifier="numeric_range",
                            category="numeric",
                            claim=f"{key}={val}",
                            passed=True,
                            detail="Within valid range [0, 1]",
                        ))

        for key in positive_only:
            if key in result:
                val = result[key]
                if isinstance(val, (int, float)):
                    if val <= 0:
                        checks.append(VerificationCheck(
                            verifier="numeric_range",
                            category="numeric",
                            claim=f"{key}={val}",
                            passed=False,
                            detail=f"Must be > 0, got {val}",
                            severity="error",
                        ))
                    else:
                        checks.append(VerificationCheck(
                            verifier="numeric_range",
                            category="numeric",
                            claim=f"{key}={val}",
                            passed=True,
                            detail="Positive value (valid)",
                        ))

        return checks

    def _check_statistical_values(self, result: dict[str, Any]) -> list[VerificationCheck]:
        """Verify p-values and confidence intervals."""
        checks = []

        # P-value check
        for key in ("p_value", "p", "pvalue"):
            if key in result:
                val = result[key]
                if isinstance(val, (int, float)):
                    if val < 0 or val > 1:
                        checks.append(VerificationCheck(
                            verifier="numeric_range",
                            category="numeric",
                            claim=f"{key}={val}",
                            passed=False,
                            detail="p-value must be in [0, 1]",
                            severity="error",
                        ))

        # CI ordering
        ci_low = result.get("ci_lower", result.get("ci_low"))
        ci_high = result.get("ci_upper", result.get("ci_high"))
        if isinstance(ci_low, (int, float)) and isinstance(ci_high, (int, float)):
            if ci_low > ci_high:
                checks.append(VerificationCheck(
                    verifier="numeric_range",
                    category="numeric",
                    claim=f"CI: [{ci_low}, {ci_high}]",
                    passed=False,
                    detail="CI lower bound exceeds upper bound",
                    severity="error",
                ))

        # Prevalence check
        prevalence = result.get("prevalence", result.get("prevalence_pct"))
        if isinstance(prevalence, (int, float)):
            if prevalence < 0 or prevalence > 100:
                checks.append(VerificationCheck(
                    verifier="numeric_range",
                    category="numeric",
                    claim=f"prevalence={prevalence}%",
                    passed=False,
                    detail="Prevalence must be 0-100%",
                    severity="error",
                ))

        return checks


# ── Claim-Evidence Entailment ────────────────────────────────────────────────


class ClaimEvidenceEntailment:
    """LLM-based Natural Language Inference: does evidence support the claim?

    Uses a lightweight LLM prompt to check whether cited evidence
    actually entails (supports) the claims being made.

    This catches the common failure mode where an agent cites a paper
    but the paper's conclusions don't actually support the stated claim.
    """

    _ENTAILMENT_PROMPT = """You are a scientific fact-checker. Given a CLAIM and EVIDENCE,
determine if the evidence supports the claim.

CLAIM: {claim}

EVIDENCE: {evidence}

Respond with ONLY one of:
- "ENTAILS" — the evidence clearly supports the claim
- "CONTRADICTS" — the evidence contradicts the claim
- "NEUTRAL" — the evidence is unrelated or insufficient to judge

Then a brief 1-sentence explanation.

Format: VERDICT | explanation"""

    def __init__(self, llm: Optional["LLMClient"] = None):
        self.llm = llm

    def check_entailment(
        self,
        claim: str,
        evidence: str,
    ) -> VerificationCheck:
        """Check if evidence entails the claim using LLM NLI.

        Args:
            claim: The assertion to verify
            evidence: The supporting text/passage

        Returns:
            VerificationCheck with entailment verdict
        """
        if not self.llm:
            return VerificationCheck(
                verifier="entailment",
                category="entailment",
                claim=claim[:100],
                passed=True,
                detail="LLM not available for entailment check; skipping",
                severity="info",
                confidence=0.0,
            )

        prompt = self._ENTAILMENT_PROMPT.format(
            claim=claim[:500],
            evidence=evidence[:1000],
        )

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You are a precise scientific fact-checker."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
            )

            text = response.text.strip()
            verdict = text.split("|")[0].strip().upper()

            if "ENTAILS" in verdict:
                return VerificationCheck(
                    verifier="entailment",
                    category="entailment",
                    claim=claim[:100],
                    passed=True,
                    detail=f"Evidence supports claim: {text}",
                    severity="info",
                    confidence=0.8,
                )
            elif "CONTRADICTS" in verdict:
                return VerificationCheck(
                    verifier="entailment",
                    category="entailment",
                    claim=claim[:100],
                    passed=False,
                    detail=f"Evidence contradicts claim: {text}",
                    severity="error",
                    confidence=0.8,
                )
            else:
                return VerificationCheck(
                    verifier="entailment",
                    category="entailment",
                    claim=claim[:100],
                    passed=True,
                    detail=f"Evidence neutral/insufficient: {text}",
                    severity="info",
                    confidence=0.5,
                )

        except Exception as e:
            logger.debug("Entailment check failed: %s", e)
            return VerificationCheck(
                verifier="entailment",
                category="entailment",
                claim=claim[:100],
                passed=True,
                detail=f"Check skipped due to error: {e}",
                severity="info",
                confidence=0.0,
            )

    def batch_check(
        self,
        claim_evidence_pairs: list[tuple[str, str]],
    ) -> list[VerificationCheck]:
        """Check entailment for multiple claim-evidence pairs."""
        return [self.check_entailment(claim, evidence)
                for claim, evidence in claim_evidence_pairs]


# ── Main Verifier Mesh ───────────────────────────────────────────────────────


class VerifierMesh:
    """Orchestrates all verifiers and merges results.

    The mesh runs all available verifiers on a given analysis result
    and produces a unified MeshResult. Each verifier is independent
    and failure-isolated (one verifier's error doesn't prevent others).

    Usage:
        mesh = VerifierMesh(llm=my_llm)
        result = mesh.verify_result(skill_output)
        print(result.summary())
    """

    def __init__(self, llm: Optional["LLMClient"] = None, timeout: float = 10.0):
        self.url_resolver = URLDOIResolver(timeout=timeout)
        self.numeric_checker = NumericRangeChecker()
        self.entailment = ClaimEvidenceEntailment(llm)

    def verify_result(
        self,
        result: dict[str, Any],
        text_content: str = "",
        claims: Optional[list[tuple[str, str]]] = None,
        skip_sample_sizes: bool = False,
    ) -> MeshResult:
        """Run full mesh verification on a skill result.

        Args:
            result: The skill's output dictionary (checked for numeric bounds)
            text_content: Any generated text content (checked for URL/DOI resolution)
            claims: Optional list of (claim, evidence) pairs for entailment check
            skip_sample_sizes: If True, skip sample size checks (to avoid
                duplication when formal_check already covers them)

        Returns:
            MeshResult with all checks aggregated
        """
        mesh_result = MeshResult()

        # 1. Numeric range checks (fast, deterministic, no network)
        try:
            numeric_checks = self.numeric_checker.verify_result(
                result, skip_sample_sizes=skip_sample_sizes
            )
            for check in numeric_checks:
                mesh_result.add_check(check)
        except Exception as e:
            logger.debug("Numeric checker error: %s", e)
            mesh_result.add_check(VerificationCheck(
                verifier="numeric_range",
                category="numeric",
                claim="numeric checker execution",
                passed=False,
                detail=f"Numeric checker failed: {type(e).__name__}: {e}",
                severity="error",
                confidence=0.0,
            ))

        # 2. URL/DOI resolution (async, needs network)
        if text_content:
            try:
                url_checks = self.url_resolver.verify_references_sync(text_content)
                for check in url_checks:
                    mesh_result.add_check(check)
            except Exception as e:
                logger.debug("URL resolver error: %s", e)
                mesh_result.add_check(VerificationCheck(
                    verifier="url_resolver",
                    category="url",
                    claim="URL/DOI resolver execution",
                    passed=False,
                    detail=f"URL resolver failed: {type(e).__name__}: {e}",
                    severity="warning",
                    confidence=0.0,
                ))

        # 3. Claim-evidence entailment (needs LLM)
        if claims:
            try:
                entailment_checks = self.entailment.batch_check(claims)
                for check in entailment_checks:
                    mesh_result.add_check(check)
            except Exception as e:
                logger.debug("Entailment checker error: %s", e)
                mesh_result.add_check(VerificationCheck(
                    verifier="entailment",
                    category="entailment",
                    claim="entailment checker execution",
                    passed=False,
                    detail=f"Entailment checker failed: {type(e).__name__}: {e}",
                    severity="warning",
                    confidence=0.0,
                ))

        return mesh_result

    def verify_text_claims(self, text: str) -> MeshResult:
        """Extract and verify claims from generated text.

        Simplified interface that extracts numeric claims and URLs
        from free text and verifies them.
        """
        mesh_result = MeshResult()

        # Extract numeric claims from text
        numeric_patterns = {
            "sample_size": re.compile(r'(?:n\s*=\s*|sample size[:\s]*|participants[:\s]*)(\d+(?:,\d{3})*)', re.IGNORECASE),
            "auc": re.compile(r'(?:AUC|AUROC)[:\s=]*([0-9.]+)', re.IGNORECASE),
            "p_value": re.compile(r'(?:p(?:-value)?\s*[=<]\s*)([0-9.e-]+)', re.IGNORECASE),
            "prevalence": re.compile(r'(?:prevalence[:\s]*|rates?[:\s]*)([0-9.]+)\s*%', re.IGNORECASE),
        }

        for key, pattern in numeric_patterns.items():
            for match in pattern.finditer(text):
                try:
                    val_str = match.group(1).replace(",", "")
                    val = float(val_str)
                    # Create a mock result dict and run numeric checks
                    mock_result = {key: val}
                    checks = self.numeric_checker.verify_result(mock_result)
                    for check in checks:
                        mesh_result.add_check(check)
                except (ValueError, IndexError):
                    continue

        # URL/DOI checks
        try:
            url_checks = self.url_resolver.verify_references_sync(text)
            for check in url_checks:
                mesh_result.add_check(check)
        except Exception as e:
            logger.debug("URL verification in text_claims: %s", e)
            mesh_result.add_check(VerificationCheck(
                verifier="url_resolver",
                category="url",
                claim="URL/DOI resolver execution",
                passed=False,
                detail=f"URL resolver failed: {type(e).__name__}: {e}",
                severity="warning",
                confidence=0.0,
            ))

        return mesh_result
