"""
table_continuation_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect simple cross-page / multi-page table continuations from table understanding
outputs. This is intentionally conservative and dependency-free.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

try:
    from document_ai.schemas.page_raw_schema import normalize_pdf_text, json_safe
except Exception:  # pragma: no cover
    def normalize_pdf_text(text: Any) -> str:
        return "" if text is None else str(text).strip()
    def json_safe(value: Any) -> Any:
        return value


@dataclass
class TableContinuationDetectorConfig:
    min_confidence: float = 0.45
    attach_to_pages: bool = True
    include_debug: bool = True


class TableContinuationDetector:
    def __init__(self, config: Optional[TableContinuationDetectorConfig] = None, *args, **kwargs):
        self.config = config or TableContinuationDetectorConfig()

    def process(
        self,
        page_raws: Optional[List[Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        document_structure_result: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        page_raws = page_raws or []
        table_understanding_result = table_understanding_result or {}

        continuations: List[Dict[str, Any]] = []

        # 1) Directly convert detected multi-page tables into continuation links.
        for table in self._list(table_understanding_result.get("multi_page_tables", [])):
            table = self._dict(table)
            page_numbers = self._pages(table)
            page_numbers = sorted(list(dict.fromkeys(page_numbers)))
            for i in range(len(page_numbers) - 1):
                continuations.append({
                    "table_continuation_id": self._make_id("table_cont"),
                    "multi_page_table_id": table.get("multi_page_table_id", ""),
                    "from_page": page_numbers[i],
                    "to_page": page_numbers[i + 1],
                    "table_grid_ids": table.get("table_grid_ids", []) or [],
                    "table_semantic_ids": table.get("table_semantic_ids", []) or [],
                    "continuation_type": "multi_page_table",
                    "confidence": self._float(table.get("confidence"), 0.70),
                    "source": "table_continuation_detector.multi_page_tables",
                    "metadata": table if self.config.include_debug else {},
                })

        # 2) Conservative heuristic: adjacent pages with compatible tables.
        if not continuations:
            tables = []
            for key in ["table_semantics", "table_grids", "table_structures", "table_boundaries"]:
                for item in self._list(table_understanding_result.get(key, [])):
                    item = self._dict(item)
                    item["_source_key"] = key
                    tables.append(item)

            by_page: Dict[int, List[Dict[str, Any]]] = {}
            for table in tables:
                for page in self._pages(table):
                    by_page.setdefault(page, []).append(table)

            page_numbers = sorted(by_page.keys())
            for page in page_numbers:
                nxt = page + 1
                if nxt not in by_page:
                    continue
                for left in by_page.get(page, []):
                    for right in by_page.get(nxt, []):
                        score = self._compatibility_score(left, right)
                        if score >= self.config.min_confidence:
                            continuations.append({
                                "table_continuation_id": self._make_id("table_cont"),
                                "from_page": page,
                                "to_page": nxt,
                                "from_table_id": self._table_id(left),
                                "to_table_id": self._table_id(right),
                                "continuation_type": "adjacent_page_table_similarity",
                                "confidence": round(score, 4),
                                "source": "table_continuation_detector.heuristic",
                            })

        result = {
            "processor": "TableContinuationDetector",
            "schema_version": "table_continuation_v1",
            "table_continuations": continuations,
            "table_continuations_by_page": self._group_by_page(continuations),
            "table_continuation_summary": {
                "has_table_continuations": len(continuations) > 0,
                "table_continuation_count": len(continuations),
            },
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(page_raws, result)

        return json_safe(result)

    def _compatibility_score(self, left: Dict[str, Any], right: Dict[str, Any]) -> float:
        score = 0.0
        left_cols = self._int(left.get("col_count") or left.get("column_count"), 0)
        right_cols = self._int(right.get("col_count") or right.get("column_count"), 0)
        if left_cols and right_cols and left_cols == right_cols:
            score += 0.35
        left_headers = set(self._headers(left))
        right_headers = set(self._headers(right))
        if left_headers and right_headers:
            score += 0.45 * (len(left_headers & right_headers) / max(len(left_headers | right_headers), 1))
        if self._table_type(left) and self._table_type(left) == self._table_type(right):
            score += 0.15
        if self._caption(left) and self._caption(left) == self._caption(right):
            score += 0.15
        return min(score, 0.95)

    def _headers(self, table: Dict[str, Any]) -> List[str]:
        values = table.get("column_headers") or table.get("headers") or []
        if isinstance(values, dict):
            values = list(values.values())
        if not isinstance(values, list):
            values = [values]
        return [normalize_pdf_text(v.get("text") if isinstance(v, dict) else v).lower() for v in values if normalize_pdf_text(v.get("text") if isinstance(v, dict) else v)]

    def _caption(self, table: Dict[str, Any]) -> str:
        return normalize_pdf_text(table.get("caption") or table.get("caption_text") or table.get("title") or "").lower()

    def _table_type(self, table: Dict[str, Any]) -> str:
        return normalize_pdf_text(table.get("table_type") or table.get("semantic_type") or "").lower()

    def _table_id(self, table: Dict[str, Any]) -> str:
        for key in ["table_id", "table_semantic_id", "table_grid_id", "table_structure_id", "table_boundary_id"]:
            if table.get(key):
                return str(table.get(key))
        return ""

    def _pages(self, item: Dict[str, Any]) -> List[int]:
        pages = item.get("page_numbers") or []
        if not isinstance(pages, list):
            pages = [pages]
        out = [self._int(p, 0) for p in pages if self._int(p, 0) > 0]
        if out:
            return sorted(list(dict.fromkeys(out)))
        start = self._int(item.get("page_start"), 0)
        end = self._int(item.get("page_end"), 0)
        if start > 0 and end >= start:
            return list(range(start, end + 1))
        page = self._int(item.get("page_number"), 0)
        return [page] if page > 0 else []

    def _group_by_page(self, links: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for link in links:
            for key in ["from_page", "to_page"]:
                page = self._int(link.get(key), 0)
                if page > 0:
                    grouped.setdefault(str(page), []).append(link)
        return grouped

    def _attach_to_pages(self, page_raws, result):
        by_page = result.get("table_continuations_by_page", {})
        for page in page_raws or []:
            try:
                page_key = str(getattr(page, "page_number", ""))
                page.metadata.setdefault("table_continuation_detector", {})
                page.metadata["table_continuation_detector"] = {
                    "table_continuations_on_page": by_page.get(page_key, []),
                    "table_continuation_summary": result.get("table_continuation_summary", {}),
                }
            except Exception:
                pass

    def _list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            for key in ["items", "data", "results", "tables", "table_semantics", "multi_page_tables"]:
                if isinstance(value.get(key), list):
                    return value.get(key)
        return []

    def _dict(self, value):
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return dict(vars(value))
            except Exception:
                pass
        return {}

    def _make_id(self, prefix: str) -> str:
        return f"{prefix}_{hashlib.sha256(str(id(self)).encode()).hexdigest()[:8]}_{hashlib.sha256(str(hash(str(prefix))).encode()).hexdigest()[:8]}"

    def _int(self, value, default=0):
        try:
            return int(value) if value is not None else default
        except Exception:
            return default

    def _float(self, value, default=0.0):
        try:
            return float(value) if value is not None else default
        except Exception:
            return default
