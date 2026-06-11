"""Runtime research mode — a thin adapter over the shared council core.

``RuntimeResearcher`` reuses the same staged runner + parallel fan-out as the
planner, but its output is a *cited research report dict* (NOT an executable
``PlanState``) — which is why plan and research are two thin adapters over one
core rather than a single mode-flagged pipeline (their return types are
structurally incompatible).

Pipeline:
    Research setup (LLM decomposes the question into sub-queries)
    -> Retrieval (parallel ``deep_research`` per sub-query)
    -> Synthesis (LLM writes a cited brief from the gathered sources)
    -> Validation (verifier mesh checks the claims)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from biobank_agent.runtime.council import (
    CouncilContext,
    CouncilError,
    CouncilJob,
    extract_json,
    map_parallel,
    run_parallel,
)
from biobank_agent.runtime.types import ProviderRole

logger = logging.getLogger(__name__)

# retriever(subquery: str, max_sources: int) -> dict (deep_research-shaped result)
RetrieverFn = Callable[[str, int], dict]
# verifier(report_text: str) -> dict (verification summary)
VerifierFn = Callable[[str], dict]


def _dedupe_sources(sources: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for src in sources:
        if not isinstance(src, dict):
            continue
        key = str(src.get("doi") or src.get("url") or src.get("title") or "").strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(src)
    return out


class RuntimeResearcher:
    """Council-backed research mode returning a cited report dict."""

    def __init__(
        self,
        provider_router: Any,
        config: Any | None = None,
        *,
        retriever: RetrieverFn | None = None,
        verifier: VerifierFn | None = None,
        num_subqueries: int = 3,
        max_sources: int = 8,
        timeout_s: float = 240.0,
        max_workers: int = 6,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.provider_router = provider_router
        self.config = config
        self.retriever = retriever
        self._verifier = verifier
        self.num_subqueries = max(1, int(num_subqueries))
        self.max_sources = max(1, int(max_sources))
        self.timeout_s = float(timeout_s)
        self.max_workers = int(max_workers)
        self._clock = clock or (lambda: 0.0)

    def research(
        self,
        question: str,
        *,
        emit: Callable[..., None] | None = None,
        session_id: str = "research",
        turn_id: str = "research",
    ) -> dict[str, Any]:
        clean_q = " ".join(str(question or "").split())
        if not clean_q:
            raise CouncilError("cannot research an empty question")
        ctx = CouncilContext(
            provider_router=self.provider_router,
            session_id=session_id,
            turn_id=turn_id,
            emit=emit,
            max_workers=self.max_workers,
            timeout_s=self.timeout_s,
            clock=self._clock,
        )
        ctx.emit_event("Preflight", status="success", message="research council ready")

        # 1) Decompose the question into sub-queries.
        ctx.emit_event("Research setup", status="running", message="decomposing into sub-queries")
        subqueries = self._subqueries(ctx, clean_q)
        ctx.emit_event("Research setup", status="success", message=f"{len(subqueries)} sub-quer(y/ies)")

        # 2) Retrieve sources for each sub-query in parallel.
        sources: list[dict] = []
        if self.retriever is not None:
            ctx.emit_event("Planning", status="running", message=f"retrieving for {len(subqueries)} sub-quer(y/ies)")
            mapped = map_parallel(
                ctx,
                subqueries,
                lambda q: self.retriever(q, self.max_sources),
                stage="Planning",
                label_fn=lambda i, q: f"retrieve-{i + 1}",
            )
            for result in mapped:
                if result.ok and isinstance(result.value, dict):
                    sources.extend(result.value.get("sources") or [])
            sources = _dedupe_sources(sources)
            ctx.emit_event("Planning", status="success", message=f"{len(sources)} unique source(s) gathered")
        else:
            ctx.emit_event("Planning", status="success", message="no retriever configured; synthesizing from model knowledge")

        # 3) Synthesize a cited report.
        ctx.emit_event("Merge", status="running", message="synthesizing cited report")
        report = self._synthesize(ctx, clean_q, subqueries, sources)
        if not report.strip():
            raise CouncilError("research synthesis produced no report")
        # Strip phantom citations: an inline [n] must reference a real source.
        report = self._finalize_report(report, len(sources))
        ctx.emit_event("Merge", status="success", message=f"report synthesized ({len(report)} chars)")

        # 4) Verify the claims in the report (honest about degraded verification).
        ctx.emit_event("Validation", status="running", message="verifying claims")
        verification = self._verify(report)
        if verification.get("error") or verification.get("skipped"):
            reason = verification.get("reason") or verification.get("error") or "unavailable"
            ctx.emit_event("Validation", status="success", message=f"verification skipped ({reason})")
        else:
            ctx.emit_event("Validation", status="success", message="claims verified")
        ctx.emit_event("Review", status="success", message="research report ready")

        return {
            "question": clean_q,
            "subqueries": subqueries,
            "sources": sources,
            "report": report,
            "verification": verification,
        }

    # ------------------------------------------------------------------ stages
    def _subqueries(self, ctx: CouncilContext, question: str) -> list[str]:
        prompt = (
            "Decompose this research question into focused literature sub-queries "
            f"(at most {self.num_subqueries}) that together cover it. "
            f"Question: {question}\n\n"
            'Return ONLY JSON: {"subqueries": ["...", "..."]}'
        )
        results = run_parallel(
            ctx,
            [CouncilJob(role=ProviderRole.PLANNER, messages=[
                {"role": "system", "content": "You output only valid JSON."},
                {"role": "user", "content": prompt},
            ], label="subqueries", stage="Research setup")],
        )
        if results and results[0].ok:
            try:
                data = extract_json(results[0].text)
                if isinstance(data, dict):
                    subs = [str(x).strip() for x in (data.get("subqueries") or []) if str(x).strip()]
                    if subs:
                        return subs[: self.num_subqueries]
            except Exception:
                logger.debug("sub-query parse failed; using the question itself", exc_info=True)
        return [question]

    def _synthesize(self, ctx: CouncilContext, question: str, subqueries: list[str], sources: list[dict]) -> str:
        source_lines = []
        for i, src in enumerate(sources[: self.max_sources], 1):
            title = str(src.get("title") or "Untitled")
            doi = str(src.get("doi") or "")
            url = str(src.get("url") or "")
            snippet = str(src.get("snippet") or src.get("abstract") or "")[:400]
            source_lines.append(f"[{i}] {title} {('doi:' + doi) if doi else ''} {('url:' + url) if url else ''}\n{snippet}")
        sources_block = "\n\n".join(source_lines) if source_lines else "(no external sources retrieved)"
        prompt = (
            f"Research question: {question}\n\n"
            f"Sub-queries explored: {', '.join(subqueries)}\n\n"
            f"Sources:\n{sources_block}\n\n"
            "Write a concise, well-structured Markdown research brief that answers the "
            "question. Cite sources inline as [n] referencing the numbered list above. "
            "Include sections: Overview, Key findings, Gaps/uncertainties, Recommended "
            "biobank analyses. Do NOT fabricate citations; only cite the numbered sources."
        )
        results = run_parallel(
            ctx,
            [CouncilJob(role=ProviderRole.SUMMARIZER, messages=[
                {"role": "system", "content": "You are a careful research synthesist. Cite only provided sources."},
                {"role": "user", "content": prompt},
            ], label="synthesis", stage="Merge")],
        )
        if results and results[0].ok:
            return results[0].text.strip()
        return ""

    @staticmethod
    def _finalize_report(report: str, num_sources: int) -> str:
        """Remove inline ``[n]`` citations that reference a non-existent source so
        the brief can never claim phantom citations. When there are no sources at
        all, drop every bracket citation and label the brief model-only."""

        def _keep(match: re.Match[str]) -> str:
            try:
                n = int(match.group(1))
            except ValueError:
                return match.group(0)
            return match.group(0) if 1 <= n <= num_sources else ""

        cleaned = re.sub(r"\[(\d+)\]", _keep, report)
        if num_sources == 0:
            cleaned = cleaned.rstrip() + (
                "\n\n_Note: no external sources were retrieved; this brief reflects "
                "model knowledge only and is uncited._"
            )
        return cleaned

    def _verify(self, report: str) -> dict[str, Any]:
        if self._verifier is not None:
            try:
                return dict(self._verifier(report) or {})
            except Exception as exc:
                return {"error": str(exc)}
        # Default: lazily construct the verifier mesh; never let it break research.
        try:
            from biobank_agent.verifier_mesh import VerifierMesh

            result = VerifierMesh().verify_text_claims(report)
            if hasattr(result, "to_dict"):
                return result.to_dict()
            checks = getattr(result, "checks", None)
            return {"checks": len(checks) if checks is not None else 0, "summary": str(result)[:1000]}
        except Exception as exc:
            logger.debug("verifier mesh unavailable: %s", exc, exc_info=True)
            return {"skipped": True, "reason": str(exc)}
