"""Biobank field catalogue and diagnosis code lookup.

Default parser handles UK Biobank Showcase format (field.txt, category.txt).
Other biobanks can supply alternative catalogue formats.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FieldCatalog:
    """Lookup table: field_id → (title, value_type, category_id, units, ...)."""

    def __init__(self, field_txt: Path, category_txt: Path) -> None:
        self.fields: dict[str, dict] = {}
        self.categories: dict[str, dict] = {}
        self._load_categories(category_txt)
        self._load_fields(field_txt)

    def _load_categories(self, path: Path) -> None:
        if not path.exists():
            logger.warning("category.txt not found: %s", path)
            return
        with open(path, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)  # category_id, title, ...
            for row in reader:
                if len(row) < 2:
                    continue
                self.categories[row[0]] = {
                    "title": row[1],
                    "availability": row[2] if len(row) > 2 else "",
                }

    def _load_fields(self, path: Path) -> None:
        if not path.exists():
            logger.warning("field.txt not found: %s", path)
            return
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                fid = parts[0].strip()
                if not fid or fid.startswith("#") or fid == "field_id":
                    continue
                title = parts[1] if len(parts) > 1 else ""
                cat_id = parts[12] if len(parts) > 12 else ""
                units = parts[11] if len(parts) > 11 else ""
                # value_type at index 5: 11=int, 21=cat, 31=continuous, 41=bulk, 51=text, 61=date
                vtype = parts[5] if len(parts) > 5 else ""
                self.fields[fid] = {
                    "title": title,
                    "value_type": vtype,
                    "category_id": cat_id,
                    "units": units,
                    "description": parts[17] if len(parts) > 17 else "",
                }

    def field_name(self, field_id: str) -> str:
        """Return human-readable name for a field."""
        info = self.fields.get(str(field_id))
        return info["title"] if info else f"Field {field_id}"

    def field_info(self, field_id: str) -> Optional[dict]:
        return self.fields.get(str(field_id))

    def category_name(self, cat_id: str) -> str:
        info = self.categories.get(str(cat_id))
        return info["title"] if info else f"Category {cat_id}"

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Search fields by keyword in title or description."""
        query_lower = query.lower()
        results = []
        for fid, info in self.fields.items():
            text = f"{info['title']} {info.get('description', '')}".lower()
            if query_lower in text:
                results.append({"field_id": fid, **info})
                if len(results) >= limit:
                    break
        return results

    def fields_by_category(self, cat_id: str) -> list[dict]:
        """Return all fields in a given category."""
        return [
            {"field_id": fid, **info}
            for fid, info in self.fields.items()
            if info.get("category_id") == str(cat_id)
        ]
