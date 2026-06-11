"""Column selection helpers for skills that work across bank adapters."""

from __future__ import annotations

from collections.abc import Mapping

from biobank_agent.data.loader import _quote_ident


def biomarker_select_expressions(
    dm,
    preferred_fields: Mapping[str, str] | None = None,
) -> tuple[list[str], list[str], str]:
    """Return SQL select expressions for the active biomarker table.

    UKB-style skills historically assumed columns such as ``30750-0.0``.
    Adapter-backed HPP/CKB tables may instead expose semantic native columns
    such as ``hba1c`` and ``bmi``. This helper first resolves requested field
    IDs through ``DataManager.field_column`` and then falls back to all numeric
    analytical biomarker columns for the active bank.
    """
    expressions: list[str] = []
    labels: list[str] = []
    try:
        columns = set(dm.list_parquet_columns())
    except Exception:
        columns = None

    for field_id, label in (preferred_fields or {}).items():
        try:
            column = dm.field_column(str(field_id))
        except Exception:
            column = f"{field_id}-0.0"
        if columns is not None and column not in columns:
            continue
        display_label = str(label or field_id)
        expressions.append(f"{_quote_ident(column)} AS {_quote_ident(display_label)}")
        labels.append(display_label)

    if expressions:
        return expressions, labels, "field_catalog"

    try:
        numeric_columns = dm.list_numeric_biomarker_columns()
    except Exception:
        numeric_columns = []

    for column in numeric_columns:
        expressions.append(f"{_quote_ident(column)} AS {_quote_ident(column)}")
        labels.append(column)

    return expressions, labels, "numeric_columns"


__all__ = ["biomarker_select_expressions"]
