# Target Annotation and Enrichment Context

## Purpose

This layer supports biobank target interpretation after a GWAS, rare-variant
burden analysis, or target-hypothesis run has produced a gene list. It is not a
general drug-discovery portal and it does not change target ranking.

## Implemented Mapping

| Context source | Biobank Agent implementation |
|----------------|------------------------------|
| Open Targets GraphQL | `target_annotation_context` resolves target IDs, tractability, and disease-association context |
| UniProt REST | Protein accession, name, function, and Ensembl cross-reference context |
| GTEx Portal API v2 | Tissue-level median expression when an Ensembl ID is available |
| ClinicalTrials.gov API v2 | Trial records for the gene and biobank phenotype query |
| CELLxGENE Census | Optional bounded local snapshot support for cell-type expression context |
| GSEApy / GMT | `target_enrichment` uses local GMT ORA by default, with optional GSEApy when installed |

## Guardrails

- Genetic or epidemiological evidence drives ranking; annotations and
  enrichment are explanatory context only.
- Online sources are source-level isolated. A failed API call returns `PARTIAL`
  or `SKIPPED` context instead of blocking the upstream biobank workflow.
- Local cache files and GMT files carry provenance and should be archived with
  any report that uses the context.
- CELLxGENE context is aggregate reference expression and must not be treated as
  participant-level evidence.
- Trial records and tractability fields are not treatment recommendations.

## Official Sources

- Open Targets Platform GraphQL API: https://platform-docs.opentargets.org/data-access/graphql-api
- UniProt REST API: https://www.uniprot.org/help/api
- GTEx Portal API v2: https://gtexportal.org/api/v2/redoc
- ClinicalTrials.gov API v2: https://clinicaltrials.gov/data-about-studies/learn-about-api
- CELLxGENE Census Python API: https://chanzuckerberg.github.io/cellxgene-census/python-api.html
- GSEApy: https://gseapy.readthedocs.io/en/latest/introduction.html
