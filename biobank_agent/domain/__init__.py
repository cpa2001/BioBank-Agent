"""Biobank-specific domain layer (banks, data, compliance, reproducibility).

Lives outside ``core/`` because everything here imports legacy
``biobank_agent`` modules and depends on biobank-specific schemas. The
``core/`` layer remains domain-agnostic so it can be reused if the
runtime is ever embedded into a non-biobank agent (M2 SDK).
"""
