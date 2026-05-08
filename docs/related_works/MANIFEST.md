# Related Works MANIFEST

Maps each concept integrated into the Biobank Agent to its source paper/repo.

## Papers in this Directory

| File | Paper | Integrated In | Status |
|------|-------|---------------|--------|
| `2026.04.11.717929v1.full.pdf` | GraphPop: Graph-Native Population Genomics | `biobank_agent/mcp/graphpop_client.py` | Integrated |
| `TARGET_ANNOTATION_ENRICHMENT.md` | Official target annotation and enrichment APIs | `biobank_agent/data/target_context.py` | Integrated |

## External References (Not Downloaded — Available Online)

| Concept | Source | arXiv/DOI | Integrated In |
|---------|--------|-----------|---------------|
| VERGE Formal Verification | arXiv:2601.06181 | Formal Refinement and Guidance Engine | `biobank_agent/verification.py` |
| DAAO Difficulty-Aware Orchestration | arXiv:2509.11079 | Difficulty-Aware Agentic Orchestration | `biobank_agent/difficulty.py` |
| NeuroClaw Reproducibility | arXiv:2604.24696 | CUHK AIM Group Neuroimaging | `biobank_agent/reproducibility.py` |
| Agentomics-ML Validators | arXiv:2506.05542 | Functional Validators | `biobank_agent/validators.py` |
| PaperQA2 | github.com/Future-House/paper-qa | Language Agents Achieve Superhuman Synthesis | `biobank_agent/skills/literature_qa.py` |
| Instructor | github.com/jxnl/instructor | Structured Output Enforcement | `biobank_agent/structured.py` |
| StudySpec Pattern | "El Agente Gráfico" + "Talk Freely, Execute Strictly" | Schema-Gated Execution | `biobank_agent/study_spec.py` |
| Rare-Variant Burden Target Prioritization | GeneBass-like burden summary statistics | Genetics-First Target Prioritization | `biobank_agent/data/genetic_targets.py`, `biobank_agent/skills/genetic_target_hypothesis.py` |
| External Target Annotation APIs | Open Targets, UniProt, GTEx, ClinicalTrials.gov, CELLxGENE, GSEApy | Biobank Target Interpretation Context | `biobank_agent/data/target_context.py`, `biobank_agent/skills/target_annotation_context.py`, `biobank_agent/skills/target_enrichment.py` |
| Evidence Lattice | Internal architecture synthesis | Claim-Evidence Provenance | `biobank_agent/evidence.py` |
| Verifier Mesh | Internal architecture synthesis | Multi-Strategy Verification | `biobank_agent/verifier_mesh.py` |
| Progressive Disclosure | Internal architecture synthesis | 3-Layer Context Architecture | `biobank_agent/disclosure.py` |
| Temporal Safety Rules | AgentVerify pattern | LTL-Inspired Precedence | `biobank_agent/guardrails.py` |
| Active Inference (pymdp) | pip:inferactively-pymdp | Discrete Active Inference | `biobank_agent/active_inference.py` (planned) |

## Source Notes

Long-form raw research notes are kept outside the published documentation tree.
This manifest records the papers, repositories, and architecture patterns that
were retained in the implementation.

## Traceability Rule

Every new module or feature MUST reference its source in the module docstring:
```python
"""Module description.

Design source:
    - Paper: arXiv:XXXX.XXXXX (Author et al., Year)
    - Concept: short implementation mapping
    - Reference doc: docs/related_works/MANIFEST.md
"""
```
