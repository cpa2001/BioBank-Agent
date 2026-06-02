"""Pathway enrichment analysis — self-contained with built-in gene sets."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE
from biobank_agent.utils.stats import benjamini_hochberg
from biobank_agent.utils.wgs import wgs_environment_status, wgs_results_dir

logger = logging.getLogger(__name__)

VITILIGO_PATHWAY_SETS = {
    "Melanogenesis": [
        "TYR", "OCA2", "MC1R", "MITF", "SLC45A2", "DCT", "TYRP1", "PMEL",
        "RAB27A", "MLANA", "SOX10", "PAX3", "ASIP", "KITLG", "KIT",
    ],
    "Autoimmune susceptibility": [
        "PTPN22", "CTLA4", "HLA", "NLRP1", "IL2RA", "IRF4", "FOXP3",
        "BACH2", "CD80", "CD86", "ICOS",
    ],
    "Oxidative stress response": [
        "CAT", "SOD1", "SOD2", "NFE2L2", "GPX1", "HMOX1", "NQO1",
        "TXNRD1", "PRDX1", "GSR",
    ],
    "T-cell / Th1 inflammation": [
        "IFNG", "CXCL10", "IRF1", "STAT1", "CXCR3", "TNF", "IL2",
        "TBX21", "IL12A", "IL12B", "JAK1", "JAK2",
    ],
    "Apoptosis / cell death": [
        "TP53", "BCL2", "BAX", "CASP3", "CASP8", "FASLG", "FAS",
        "APAF1", "BID", "BCL2L1",
    ],
    "Wnt / melanocyte development": [
        "WNT1", "WNT3A", "CTNNB1", "LEF1", "TCF7", "DKK1", "GSK3B",
        "APC", "AXIN1", "FZD1",
    ],
    "NF-kB signaling": [
        "NFKB1", "RELA", "NFKBIA", "IKBKB", "IKBKG", "TRAF6",
        "MYD88", "TLR4", "IRAK1",
    ],
    "Antigen presentation / HLA": [
        "HLA-A", "HLA-B", "HLA-C", "HLA-DRB1", "HLA-DQB1", "HLA-DPB1",
        "B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9",
    ],
}

HALLMARK_COMPACT = {
    "Hallmark: Inflammatory Response": [
        "TNF", "IL6", "IL1B", "CXCL8", "CCL2", "IL10", "NFKB1", "PTGS2",
        "ICAM1", "SELE", "VCAM1", "MMP9", "SERPINE1",
    ],
    "Hallmark: Apoptosis": [
        "TP53", "BCL2", "BAX", "CASP3", "CASP8", "CASP9", "FASLG", "FAS",
        "APAF1", "BID", "BCL2L1", "CYCS", "DIABLO",
    ],
    "Hallmark: Interferon Gamma Response": [
        "IFNG", "STAT1", "IRF1", "IRF4", "CXCL10", "GBP1", "OAS1",
        "IDO1", "PSMB8", "PSMB9", "TAP1", "TAP2", "B2M",
    ],
    "Hallmark: UV Response": [
        "TP53", "CDKN1A", "GADD45A", "ERCC1", "XPC", "DDB2",
        "PCNA", "RAD51", "BRCA1",
    ],
    "Hallmark: PI3K-AKT-mTOR": [
        "PIK3CA", "AKT1", "MTOR", "PTEN", "TSC1", "TSC2", "RPS6KB1",
        "EIF4EBP1", "RPTOR", "RICTOR",
    ],
    "Hallmark: IL6-JAK-STAT3": [
        "IL6", "JAK1", "JAK2", "STAT3", "SOCS3", "IL6R", "IL6ST",
        "CISH", "PIM1", "MYC",
    ],
}

ALL_PATHWAY_GENES = set()
for genes in VITILIGO_PATHWAY_SETS.values():
    ALL_PATHWAY_GENES.update(genes)
for genes in HALLMARK_COMPACT.values():
    ALL_PATHWAY_GENES.update(genes)


def _run_gseapy_ora(
    input_genes: set[str],
    pathway_sets: dict[str, list[str]],
    report_dir: Path,
    background_size: int,
) -> dict:
    """Run gseapy's local enrich() ORA implementation with in-memory gene sets."""
    try:
        import gseapy as gp
    except Exception as exc:
        return {"ok": False, "reason": f"gseapy import failed: {exc}"}

    try:
        enr = gp.enrich(
            gene_list=sorted(input_genes),
            gene_sets={k: sorted({g.upper() for g in v}) for k, v in pathway_sets.items()},
            background=background_size,
            outdir=None,
            no_plot=True,
            verbose=False,
        )
        df = getattr(enr, "results", None)
        if df is None:
            df = getattr(enr, "res2d", None)
        if df is None:
            return {"ok": False, "reason": "gseapy returned no results table"}
        table_path = report_dir / "gseapy_enrichment.tsv"
        pd.DataFrame(df).to_csv(table_path, sep="\t", index=False)
        return {
            "ok": True,
            "reason": "",
            "table": str(table_path),
            "n_terms": int(len(df)),
            "gseapy_version": getattr(gp, "__version__", "unknown"),
        }
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}


@skill(
    name="pathway_enrichment",
    description=(
        "Pathway enrichment analysis for gene lists from WGS association or "
        "burden testing results. Uses built-in vitiligo-relevant pathway sets "
        "(melanogenesis, autoimmune, oxidative stress, Th1 inflammation, etc.). "
        "Performs Fisher's exact overrepresentation analysis (ORA) with FDR correction."
    ),
    parameters={
        "gene_list": {
            "type": "string",
            "description": "Comma-separated gene symbols (empty = use burden test results from state)",
            "default": "",
        },
        "database": {
            "type": "string",
            "description": "Gene set database: 'vitiligo_pathways' (default), 'hallmark', or 'all'",
            "default": "vitiligo_pathways",
        },
        "fdr_threshold": {
            "type": "number",
            "description": "FDR significance threshold (default 0.05)",
            "default": 0.05,
        },
        "top_n": {
            "type": "integer",
            "description": "Number of top pathways to show (default 20)",
            "default": 20,
        },
        "background_size": {
            "type": "integer",
            "description": "Background gene universe size (default 20000)",
            "default": 20000,
        },
    },
    required=[],
)
def pathway_enrichment(
    gene_list: str = "",
    database: str = "vitiligo_pathways",
    fdr_threshold: float = 0.05,
    top_n: int = 20,
    background_size: int = 20000,
    *,
    ctx=None,
) -> dict:
    input_genes = set()

    if gene_list:
        input_genes = {g.strip().upper() for g in gene_list.split(",") if g.strip()}
    elif hasattr(ctx, "state") and hasattr(ctx.state, "custom_data"):
        burden = ctx.state.custom_data.get("burden_results", {})
        if burden.get("gene_results"):
            input_genes = {gr["gene"].upper() for gr in burden["gene_results"]
                          if gr.get("p_burden", 1) < 0.1}

        if not input_genes:
            annot = ctx.state.custom_data.get("annotation_results", {})
            if annot.get("gene_hits"):
                input_genes = {g.upper() for g in annot["gene_hits"].keys()}

    if not input_genes:
        return {"error": "No genes provided and no results found in session state. "
                "Provide gene_list or run vcf_burden_test/vcf_annotation first."}

    if database == "hallmark":
        pathway_sets = HALLMARK_COMPACT
        db_label = "hallmark (compact built-in)"
    elif database == "all":
        pathway_sets = {**VITILIGO_PATHWAY_SETS, **HALLMARK_COMPACT}
        db_label = "vitiligo_pathways + hallmark (built-in)"
    else:
        pathway_sets = VITILIGO_PATHWAY_SETS
        db_label = "vitiligo_pathways (built-in)"

    N = background_size
    n = len(input_genes)

    results = []
    for pathway_name, pathway_genes in pathway_sets.items():
        pathway_set = {g.upper() for g in pathway_genes}
        K = len(pathway_set)
        overlap = input_genes & pathway_set
        k = len(overlap)

        if k == 0:
            p_value = 1.0
        else:
            p_value = sp_stats.hypergeom.sf(k - 1, N, K, n)

        fold_enrichment = (k / n) / (K / N) if (n > 0 and K > 0 and N > 0) else 0

        results.append({
            "pathway": pathway_name,
            "pathway_size": K,
            "overlap_count": k,
            "overlap_genes": sorted(overlap),
            "fold_enrichment": round(fold_enrichment, 2),
            "p_value": float(p_value),
        })

    if not results:
        return {"error": "No pathway results computed."}

    p_arr = np.array([r["p_value"] for r in results])
    fdr_arr = benjamini_hochberg(p_arr)
    for i, r in enumerate(results):
        r["p_fdr"] = float(fdr_arr[i])
        r["significant"] = fdr_arr[i] < fdr_threshold

    results.sort(key=lambda x: x["p_value"])
    top_results = results[:top_n]

    figures = []
    report_dir = wgs_results_dir(ctx, "enrichment")
    env_status = wgs_environment_status()
    gseapy_run = None
    if env_status["packages"].get("gseapy"):
        gseapy_run = _run_gseapy_ora(input_genes, pathway_sets, report_dir, background_size)

    pathways_to_plot = [r for r in top_results if r["p_value"] < 1.0]
    if pathways_to_plot:
        fig, ax = nature_figure(width="single", height_ratio=0.06 * len(pathways_to_plot))
        names = [r["pathway"] for r in reversed(pathways_to_plot)]
        neg_log_p = [-np.log10(max(r["p_value"], 1e-20)) for r in reversed(pathways_to_plot)]
        overlaps = [r["overlap_count"] for r in reversed(pathways_to_plot)]
        sigs = [r["significant"] for r in reversed(pathways_to_plot)]

        colors_bar = [PALETTE[1] if s else PALETTE[0] for s in sigs]

        y_pos = range(len(names))
        bars = ax.barh(y_pos, neg_log_p, color=colors_bar, alpha=0.8, height=0.7)

        for i, (bar, ov) in enumerate(zip(bars, overlaps)):
            ax.text(bar.get_width() + 0.1, i, f"n={ov}", va="center", fontsize=5)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel(r"$-\log_{10}(P)$")
        ax.set_title(f"Pathway Enrichment ({len(input_genes)} input genes)")

        sig_line = -np.log10(fdr_threshold)
        ax.axvline(x=sig_line, color="red", linewidth=0.5, linestyle="--", alpha=0.5)

        fig.tight_layout()
        paths = save_figure(fig, "pathway_enrichment", report_dir)
        figures.extend(paths)

    if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
        ctx.state.figures.extend(figures)

    n_significant = sum(1 for r in results if r["significant"])

    return {
        "n_input_genes": len(input_genes),
        "input_genes": sorted(input_genes),
        "n_pathways_tested": len(results),
        "n_pathways_significant": n_significant,
        "database": db_label,
        "background_size": N,
        "fdr_threshold": fdr_threshold,
        "results": top_results,
        "figures": [str(p) for p in figures],
        "result_dir": str(report_dir),
        "enrichment_mode": "gseapy_local_ora_with_builtin_gene_sets" if gseapy_run and gseapy_run.get("ok") else db_label,
        "standard_enrichment_available": env_status["modules"]["standard_enrichment"] == "READY",
        "gseapy_run": gseapy_run,
    }
