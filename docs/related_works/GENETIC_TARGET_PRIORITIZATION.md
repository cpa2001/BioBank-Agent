# Genetic Target Prioritization Workflow

## Purpose

This workflow turns biobank-scale rare-variant burden summary statistics into
auditable therapeutic target hypotheses. It is an independent Biobank Agent
implementation: target ranking is driven by supplied genetic evidence, while
external annotations remain explanatory context for downstream review.

## Implemented Mapping

| Workflow step | Biobank Agent implementation |
|---------------|------------------------------|
| Extract discovery genes from burden statistics | `genetic_target_hypothesis` accepts GeneBass-like rows or CSV/TSV/JSON files and applies raw-P discovery, Bonferroni, and Benjamini-Yekutieli tiers |
| Infer therapeutic direction from loss-of-function beta | `beta < 0` maps to inhibit; `beta > 0` maps to activate; `beta = 0` remains direction-uncertain |
| Assess variant-class support | pLoF plus missense-low-confidence concordance receives the strongest variant-class support |
| Capture pathway convergence | Reactome or pathway fields are grouped into mechanism clusters and score only the convergence term |
| Add external target context | `target_annotation_context` and `target_enrichment` can run after ranking to add source-attributed interpretation without changing the genetic score |

## Scoring Contract

The skill keeps genetics in control of the score:

| Component | Weight |
|-----------|-------:|
| P-value strength | 0.35 |
| Effect size | 0.25 |
| Variant-class support | 0.25 |
| Pathway convergence | 0.15 |
| Independent genetic-direction agreement | max 0.05 bonus |

Annotation richness does not increase rank except for the small explicit
direction-agreement bonus. A well-studied gene with weak burden evidence should
not outrank a less studied gene with stronger direct genetic evidence.

## Guardrails

- The skill uses supplied summary statistics; it does not rerun genotype QC,
  burden tests, unrelated-sample filters, or phenotype keep filters.
- Labelled burden tables must match the requested phenotype. Unlabelled or
  pre-filtered tables require `prefiltered=true` so the agent cannot silently
  rank unrelated phenotypes.
- Multiple-testing scope is explicit. BY-FDR tiers are publication-grade only
  when the full phenotype-family test table is supplied.
- The output is hypothesis-generating unless supported by replication,
  colocalization, functional evidence, or target-trial/RCT evidence.
- UK Biobank exome discovery should be replicated in ancestry-diverse cohorts
  before therapeutic program decisions.
- Loss-of-function direction is a first-pass biological prior, not proof that a
  drug modality can safely reproduce the same effect.
