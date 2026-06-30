"""
table_header_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect and normalize table headers from extracted table cells.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- TableBoundaryDetector
- TableGridBuilder
- TableStructureRecognizer
- TableCellExtractor

Output
------
Dictionary with:
- table_headers
- table_header_cells
- table_headers_by_page
- table_headers_by_table
- table_header_summary

Flow
----
TableBoundaryDetector
    ↓
TableGridBuilder
    ↓
TableStructureRecognizer
    ↓
TableCellExtractor
    ↓
TableHeaderDetector
    ↓
TableSpanDetector
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class TableHeaderDetectorConfig:
    max_header_rows: int = 3
    min_header_confidence: float = 0.40

    use_structure_header_hint: bool = True
    use_cell_role_hint: bool = True
    use_first_row_heuristic: bool = True

    detect_column_headers: bool = True
    detect_row_headers: bool = True

    include_empty_header_cells: bool = False
    normalize_header_text: bool = True

    attach_to_pages: bool = True
    include_debug: bool = True


@dataclass
class TableHeaderCell:
    table_header_cell_id: str
    table_header_id: str
    table_cell_id: str

    table_grid_id: str
    table_structure_id: str
    table_boundary_id: str

    page_number: int
    page_index: int

    row_index: int
    col_index: int
    row_span: int
    col_span: int

    bbox: List[float]
    text: str
    normalized_text: str

    header_role: str = "column_header"
    confidence: float = 0.5
    source: str = "table_cell_extractor"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class TableHeader:
    table_header_id: str

    table_grid_id: str
    table_structure_id: str
    table_boundary_id: str

    page_number: int
    page_index: int

    header_type: str
    header_row_indices: List[int]
    header_col_indices: List[int]

    column_headers: List[str]
    row_headers: List[str]

    header_cell_ids: List[str]
    source_table_cell_ids: List[str]

    confidence: float = 0.5
    source: str = "table_cell_extractor"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class TableHeaderDetector:
    def __init__(
        self,
        config: Optional[TableHeaderDetectorConfig] = None,
    ):
        self.config = config or TableHeaderDetectorConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        table_cell_result: Optional[Dict[str, Any]] = None,
        table_structure_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cells = self._collect_table_cells(
            page_raws=page_raws,
            table_cell_result=table_cell_result,
        )

        structures = self._collect_table_structures(
            page_raws=page_raws,
            table_structure_result=table_structure_result,
        )

        cells_by_table = self._group_cells_by_table_object(cells)

        structure_by_grid_id = {
            structure.get("table_grid_id", ""): structure
            for structure in structures
            if structure.get("table_grid_id")
        }

        headers: List[TableHeader] = []
        header_cells: List[TableHeaderCell] = []

        for table_grid_id, table_cells in cells_by_table.items():
            structure = structure_by_grid_id.get(table_grid_id, {})

            header, detected_header_cells = self.detect_header_for_table(
                table_grid_id=table_grid_id,
                table_cells=table_cells,
                structure=structure,
            )

            if header:
                headers.append(header)
                header_cells.extend(detected_header_cells)

        headers = self._sort_headers(headers)
        header_cells = self._sort_header_cells(header_cells)

        result = {
            "processor": "TableHeaderDetector",
            "table_headers": [
                header.to_dict() for header in headers
            ],
            "table_header_cells": [
                cell.to_dict() for cell in header_cells
            ],
            "table_headers_by_page": self._group_headers_by_page(headers),
            "table_headers_by_table": self._group_headers_by_table(headers),
            "table_header_cells_by_table": self._group_header_cells_by_table(header_cells),
            "table_header_summary": self._build_summary(
                headers=headers,
                header_cells=header_cells,
            ),
            "config": {
                "max_header_rows": self.config.max_header_rows,
                "min_header_confidence": self.config.min_header_confidence,
                "use_structure_header_hint": self.config.use_structure_header_hint,
                "use_cell_role_hint": self.config.use_cell_role_hint,
                "use_first_row_heuristic": self.config.use_first_row_heuristic,
                "detect_column_headers": self.config.detect_column_headers,
                "detect_row_headers": self.config.detect_row_headers,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                headers=headers,
                header_cells=header_cells,
                result=result,
            )

        return result

    def detect_header_for_table(
        self,
        table_grid_id: str,
        table_cells: List[Dict[str, Any]],
        structure: Optional[Dict[str, Any]] = None,
    ) -> tuple[Optional[TableHeader], List[TableHeaderCell]]:
        structure = structure or {}

        if not table_cells:
            return None, []

        table_cells = sorted(
            table_cells,
            key=lambda item: (
                self._safe_int(item.get("row_index"), default=0),
                self._safe_int(item.get("col_index"), default=0),
            ),
        )

        row_count = max(
            [
                self._safe_int(cell.get("row_index"), default=0)
                for cell in table_cells
            ],
            default=0,
        ) + 1

        col_count = max(
            [
                self._safe_int(cell.get("col_index"), default=0)
                for cell in table_cells
            ],
            default=0,
        ) + 1

        header_row_indices = self._detect_header_rows(
            table_cells=table_cells,
            structure=structure,
            row_count=row_count,
            col_count=col_count,
        )

        header_col_indices = self._detect_header_columns(
            table_cells=table_cells,
            structure=structure,
            row_count=row_count,
            col_count=col_count,
        )

        if not header_row_indices and not header_col_indices:
            return None, []

        first_cell = table_cells[0]

        table_header_id = make_id("tbl_header")

        detected_header_cells = self._build_header_cells(
            table_header_id=table_header_id,
            table_cells=table_cells,
            header_row_indices=header_row_indices,
            header_col_indices=header_col_indices,
        )

        if not detected_header_cells:
            return None, []

        column_headers = self._build_column_headers(
            table_cells=table_cells,
            header_row_indices=header_row_indices,
            col_count=col_count,
        )

        row_headers = self._build_row_headers(
            table_cells=table_cells,
            header_col_indices=header_col_indices,
            row_count=row_count,
        )

        header_type = self._classify_header_type(
            header_row_indices=header_row_indices,
            header_col_indices=header_col_indices,
        )

        confidence = self._score_header(
            header_row_indices=header_row_indices,
            header_col_indices=header_col_indices,
            column_headers=column_headers,
            row_headers=row_headers,
            detected_header_cells=detected_header_cells,
            structure=structure,
        )

        if confidence < self.config.min_header_confidence:
            return None, []

        header = TableHeader(
            table_header_id=table_header_id,
            table_grid_id=table_grid_id,
            table_structure_id=structure.get("table_structure_id", first_cell.get("table_structure_id", "")),
            table_boundary_id=structure.get("table_boundary_id", first_cell.get("table_boundary_id", "")),
            page_number=self._safe_int(first_cell.get("page_number"), default=0),
            page_index=self._safe_int(first_cell.get("page_index"), default=0),
            header_type=header_type,
            header_row_indices=header_row_indices,
            header_col_indices=header_col_indices,
            column_headers=column_headers,
            row_headers=row_headers,
            header_cell_ids=[
                item.table_header_cell_id for item in detected_header_cells
            ],
            source_table_cell_ids=[
                item.table_cell_id for item in detected_header_cells
                if item.table_cell_id
            ],
            confidence=confidence,
            source="table_cell_extractor.table_cells",
            metadata={
                "row_count": row_count,
                "col_count": col_count,
                "structure_table_type": structure.get("table_type", ""),
                "structure_has_header": structure.get("has_header", False),
                "structure_header_row_indices": structure.get("header_row_indices", []),
                "structure_stub_column_indices": structure.get("stub_column_indices", []),
                "non_empty_column_header_count": len(
                    [
                        item for item in column_headers
                        if item.strip()
                    ]
                ),
                "non_empty_row_header_count": len(
                    [
                        item for item in row_headers
                        if item.strip()
                    ]
                ),
            },
        )

        return header, detected_header_cells

    def _detect_header_rows(
        self,
        table_cells: List[Dict[str, Any]],
        structure: Dict[str, Any],
        row_count: int,
        col_count: int,
    ) -> List[int]:
        header_rows: List[int] = []

        if self.config.use_structure_header_hint:
            structure_rows = structure.get("header_row_indices", []) or []

            for row_index in structure_rows:
                row_index = self._safe_int(row_index, default=-1)

                if 0 <= row_index < row_count:
                    header_rows.append(row_index)

        if header_rows:
            return sorted(list(dict.fromkeys(header_rows)))

        if self.config.use_cell_role_hint:
            role_rows = []

            for cell in table_cells:
                row_index = self._safe_int(cell.get("row_index"), default=-1)

                if row_index < 0:
                    continue

                role = cell.get("cell_role", "")
                is_header = bool(cell.get("is_header", False))

                if is_header or role == "header":
                    role_rows.append(row_index)

            if role_rows:
                header_rows = sorted(list(dict.fromkeys(role_rows)))
                header_rows = [
                    row_index for row_index in header_rows
                    if row_index < self.config.max_header_rows
                ]

                if header_rows:
                    return header_rows

        if self.config.use_first_row_heuristic:
            header_rows = self._detect_header_rows_by_heuristic(
                table_cells=table_cells,
                row_count=row_count,
                col_count=col_count,
            )

        return header_rows

    def _detect_header_rows_by_heuristic(
        self,
        table_cells: List[Dict[str, Any]],
        row_count: int,
        col_count: int,
    ) -> List[int]:
        rows = self._group_cells_by_row(table_cells)
        header_rows: List[int] = []

        max_scan_rows = min(
            self.config.max_header_rows,
            row_count,
        )

        for row_index in range(max_scan_rows):
            row_cells = rows.get(row_index, [])
            score = self._score_header_row(
                row_cells=row_cells,
                row_index=row_index,
                col_count=col_count,
            )

            if score >= self.config.min_header_confidence:
                header_rows.append(row_index)
            else:
                if row_index == 0:
                    continue

                break

        return header_rows

    def _detect_header_columns(
        self,
        table_cells: List[Dict[str, Any]],
        structure: Dict[str, Any],
        row_count: int,
        col_count: int,
    ) -> List[int]:
        if not self.config.detect_row_headers:
            return []

        header_cols: List[int] = []

        structure_stub_cols = structure.get("stub_column_indices", []) or []

        for col_index in structure_stub_cols:
            col_index = self._safe_int(col_index, default=-1)

            if 0 <= col_index < col_count:
                header_cols.append(col_index)

        if header_cols:
            return sorted(list(dict.fromkeys(header_cols)))

        cols = self._group_cells_by_col(table_cells)

        for col_index in range(min(2, col_count)):
            col_cells = cols.get(col_index, [])
            score = self._score_row_header_column(
                col_cells=col_cells,
                col_index=col_index,
                row_count=row_count,
            )

            if score >= self.config.min_header_confidence + 0.05:
                header_cols.append(col_index)

        return sorted(list(dict.fromkeys(header_cols)))

    def _build_header_cells(
        self,
        table_header_id: str,
        table_cells: List[Dict[str, Any]],
        header_row_indices: List[int],
        header_col_indices: List[int],
    ) -> List[TableHeaderCell]:
        header_cells: List[TableHeaderCell] = []

        for cell in table_cells:
            row_index = self._safe_int(cell.get("row_index"), default=-1)
            col_index = self._safe_int(cell.get("col_index"), default=-1)

            in_header_row = row_index in header_row_indices
            in_header_col = col_index in header_col_indices

            if not in_header_row and not in_header_col:
                continue

            text = cell.get("normalized_text") or cell.get("text") or ""
            text = self._clean_text(text)
            normalized_text = self._normalize_header_text(text)

            if not self.config.include_empty_header_cells and not normalized_text:
                continue

            if in_header_row and in_header_col:
                header_role = "corner_header"
            elif in_header_row:
                header_role = "column_header"
            else:
                header_role = "row_header"

            header_cells.append(
                TableHeaderCell(
                    table_header_cell_id=make_id("tbl_header_cell"),
                    table_header_id=table_header_id,
                    table_cell_id=cell.get("table_cell_id", ""),
                    table_grid_id=cell.get("table_grid_id", ""),
                    table_structure_id=cell.get("table_structure_id", ""),
                    table_boundary_id=cell.get("table_boundary_id", ""),
                    page_number=self._safe_int(cell.get("page_number"), default=0),
                    page_index=self._safe_int(cell.get("page_index"), default=0),
                    row_index=row_index,
                    col_index=col_index,
                    row_span=max(1, self._safe_int(cell.get("row_span"), default=1)),
                    col_span=max(1, self._safe_int(cell.get("col_span"), default=1)),
                    bbox=cell.get("bbox", []),
                    text=text,
                    normalized_text=normalized_text,
                    header_role=header_role,
                    confidence=self._safe_float(cell.get("confidence", 0.5), default=0.5),
                    source="table_cell_extractor.table_cells",
                    metadata={
                        "cell_role": cell.get("cell_role", ""),
                        "is_header": cell.get("is_header", False),
                        "is_stub": cell.get("is_stub", False),
                        "is_numeric": cell.get("is_numeric", False),
                        "source_cell_metadata": cell.get("metadata", {}),
                    },
                )
            )

        return header_cells

    def _build_column_headers(
        self,
        table_cells: List[Dict[str, Any]],
        header_row_indices: List[int],
        col_count: int,
    ) -> List[str]:
        if not self.config.detect_column_headers:
            return []

        rows = self._group_cells_by_row(table_cells)
        column_headers: List[str] = []

        for col_index in range(col_count):
            parts: List[str] = []

            for row_index in header_row_indices:
                row_cells = rows.get(row_index, [])

                for cell in row_cells:
                    if self._safe_int(cell.get("col_index"), default=-1) != col_index:
                        continue

                    text = (
                        cell.get("normalized_text")
                        or cell.get("text")
                        or ""
                    )

                    text = self._normalize_header_text(text)

                    if text:
                        parts.append(text)

            column_headers.append(
                " | ".join(parts).strip()
            )

        return column_headers

    def _build_row_headers(
        self,
        table_cells: List[Dict[str, Any]],
        header_col_indices: List[int],
        row_count: int,
    ) -> List[str]:
        if not self.config.detect_row_headers:
            return []

        cols = self._group_cells_by_col(table_cells)
        row_headers: List[str] = []

        for row_index in range(row_count):
            parts: List[str] = []

            for col_index in header_col_indices:
                col_cells = cols.get(col_index, [])

                for cell in col_cells:
                    if self._safe_int(cell.get("row_index"), default=-1) != row_index:
                        continue

                    text = (
                        cell.get("normalized_text")
                        or cell.get("text")
                        or ""
                    )

                    text = self._normalize_header_text(text)

                    if text:
                        parts.append(text)

            row_headers.append(
                " | ".join(parts).strip()
            )

        return row_headers

    def _score_header_row(
        self,
        row_cells: List[Dict[str, Any]],
        row_index: int,
        col_count: int,
    ) -> float:
        if not row_cells:
            return 0.0

        texts = [
            self._clean_text(
                cell.get("normalized_text")
                or cell.get("text")
                or ""
            )
            for cell in row_cells
        ]

        non_empty_texts = [
            text for text in texts
            if text
        ]

        non_empty_count = len(non_empty_texts)
        numeric_count = sum(
            1 for text in non_empty_texts
            if self._is_numeric_like(text)
        )

        numeric_ratio = numeric_count / max(non_empty_count, 1)
        coverage_ratio = non_empty_count / max(col_count, 1)

        score = 0.15

        if row_index == 0:
            score += 0.15

        if coverage_ratio >= 0.50:
            score += 0.15

        if numeric_ratio <= 0.30:
            score += 0.20

        avg_len = 0.0

        if non_empty_texts:
            avg_len = sum(len(text) for text in non_empty_texts) / len(non_empty_texts)

        if 1 <= avg_len <= 80:
            score += 0.08

        keyword_hits = sum(
            1 for text in non_empty_texts
            if self._has_header_keyword(text)
        )

        if keyword_hits > 0:
            score += min(0.22, keyword_hits * 0.07)

        return round(max(0.0, min(score, 0.95)), 4)

    def _score_row_header_column(
        self,
        col_cells: List[Dict[str, Any]],
        col_index: int,
        row_count: int,
    ) -> float:
        if not col_cells:
            return 0.0

        texts = [
            self._clean_text(
                cell.get("normalized_text")
                or cell.get("text")
                or ""
            )
            for cell in col_cells
        ]

        non_empty_texts = [
            text for text in texts
            if text
        ]

        if not non_empty_texts:
            return 0.0

        numeric_count = sum(
            1 for text in non_empty_texts
            if self._is_numeric_like(text)
        )

        numeric_ratio = numeric_count / max(len(non_empty_texts), 1)
        coverage_ratio = len(non_empty_texts) / max(row_count, 1)

        score = 0.10

        if col_index == 0:
            score += 0.18

        if coverage_ratio >= 0.40:
            score += 0.12

        if numeric_ratio <= 0.35:
            score += 0.20

        keyword_hits = sum(
            1 for text in non_empty_texts
            if self._has_row_header_keyword(text)
        )

        if keyword_hits > 0:
            score += min(0.20, keyword_hits * 0.06)

        return round(max(0.0, min(score, 0.95)), 4)

    def _score_header(
        self,
        header_row_indices: List[int],
        header_col_indices: List[int],
        column_headers: List[str],
        row_headers: List[str],
        detected_header_cells: List[TableHeaderCell],
        structure: Dict[str, Any],
    ) -> float:
        score = 0.30

        if header_row_indices:
            score += 0.20

        if header_col_indices:
            score += 0.08

        if structure.get("has_header"):
            score += 0.15

        if any(text.strip() for text in column_headers):
            score += 0.12

        if any(text.strip() for text in row_headers):
            score += 0.05

        if detected_header_cells:
            avg_cell_confidence = sum(
                cell.confidence for cell in detected_header_cells
            ) / len(detected_header_cells)

            score += min(0.10, avg_cell_confidence * 0.10)

        return round(max(0.0, min(score, 0.95)), 4)

    def _classify_header_type(
        self,
        header_row_indices: List[int],
        header_col_indices: List[int],
    ) -> str:
        if header_row_indices and header_col_indices:
            return "column_and_row_headers"

        if header_row_indices:
            if len(header_row_indices) > 1:
                return "multi_row_column_header"

            return "single_row_column_header"

        if header_col_indices:
            return "row_header_only"

        return "unknown_header"

    def _group_cells_by_row(
        self,
        cells: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        grouped: Dict[int, List[Dict[str, Any]]] = {}

        for cell in cells:
            row_index = self._safe_int(cell.get("row_index"), default=-1)

            if row_index < 0:
                continue

            grouped.setdefault(row_index, [])
            grouped[row_index].append(cell)

        for row_index in grouped:
            grouped[row_index] = sorted(
                grouped[row_index],
                key=lambda item: self._safe_int(item.get("col_index"), default=0),
            )

        return grouped

    def _group_cells_by_col(
        self,
        cells: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        grouped: Dict[int, List[Dict[str, Any]]] = {}

        for cell in cells:
            col_index = self._safe_int(cell.get("col_index"), default=-1)

            if col_index < 0:
                continue

            grouped.setdefault(col_index, [])
            grouped[col_index].append(cell)

        for col_index in grouped:
            grouped[col_index] = sorted(
                grouped[col_index],
                key=lambda item: self._safe_int(item.get("row_index"), default=0),
            )

        return grouped

    def _collect_table_cells(
        self,
        page_raws: List[PageRaw],
        table_cell_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if table_cell_result:
            cells = table_cell_result.get("table_cells", [])

            if cells:
                return cells

        collected: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            cell_meta = page_raw.metadata.get("table_cell_extractor", {})
            page_cells = cell_meta.get("table_cells_on_page", [])

            for cell in page_cells:
                collected.append(cell)

        return collected

    def _collect_table_structures(
        self,
        page_raws: List[PageRaw],
        table_structure_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if table_structure_result:
            structures = table_structure_result.get("table_structures", [])

            if structures:
                return structures

        collected: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            structure_meta = page_raw.metadata.get("table_structure_recognizer", {})
            page_structures = structure_meta.get("table_structures_on_page", [])

            for structure in page_structures:
                collected.append(structure)

        return collected

    def _group_cells_by_table_object(
        self,
        cells: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for cell in cells:
            table_grid_id = cell.get("table_grid_id", "") or "unknown_table"
            grouped.setdefault(table_grid_id, [])
            grouped[table_grid_id].append(cell)

        return grouped

    def _group_headers_by_page(
        self,
        headers: List[TableHeader],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for header in headers:
            page_key = str(header.page_number)
            grouped.setdefault(page_key, [])
            grouped[page_key].append(header.to_dict())

        return grouped

    def _group_headers_by_table(
        self,
        headers: List[TableHeader],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for header in headers:
            table_key = header.table_grid_id or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(header.to_dict())

        return grouped

    def _group_header_cells_by_table(
        self,
        header_cells: List[TableHeaderCell],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for cell in header_cells:
            table_key = cell.table_grid_id or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(cell.to_dict())

        return grouped

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        headers: List[TableHeader],
        header_cells: List[TableHeaderCell],
        result: Dict[str, Any],
    ) -> None:
        headers_by_page: Dict[int, List[Dict[str, Any]]] = {}
        cells_by_page: Dict[int, List[Dict[str, Any]]] = {}

        for header in headers:
            headers_by_page.setdefault(header.page_number, [])
            headers_by_page[header.page_number].append(header.to_dict())

        for cell in header_cells:
            cells_by_page.setdefault(cell.page_number, [])
            cells_by_page[cell.page_number].append(cell.to_dict())

        for page_raw in page_raws:
            page_headers = headers_by_page.get(page_raw.page_number, [])
            page_header_cells = cells_by_page.get(page_raw.page_number, [])

            page_raw.metadata.setdefault("table_header_detector", {})
            page_raw.metadata["table_header_detector"] = {
                "processor": "TableHeaderDetector",
                "table_headers_on_page": page_headers,
                "table_header_cells_on_page": page_header_cells,
                "table_header_count_on_page": len(page_headers),
                "table_header_cell_count_on_page": len(page_header_cells),
                "table_header_summary": result.get("table_header_summary", {}),
            }

    def _build_summary(
        self,
        headers: List[TableHeader],
        header_cells: List[TableHeaderCell],
    ) -> Dict[str, Any]:
        by_page: Dict[str, int] = {}
        by_table: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for header in headers:
            page_key = str(header.page_number)
            table_key = header.table_grid_id or "unknown_table"
            type_key = header.header_type or "unknown"

            by_page[page_key] = by_page.get(page_key, 0) + 1
            by_table[table_key] = by_table.get(table_key, 0) + 1
            by_type[type_key] = by_type.get(type_key, 0) + 1

        column_header_count = sum(
            1 for cell in header_cells
            if cell.header_role in ["column_header", "corner_header"]
        )

        row_header_count = sum(
            1 for cell in header_cells
            if cell.header_role in ["row_header", "corner_header"]
        )

        return {
            "has_table_headers": len(headers) > 0,
            "table_header_count": len(headers),
            "table_header_cell_count": len(header_cells),
            "column_header_cell_count": column_header_count,
            "row_header_cell_count": row_header_count,
            "page_count_with_headers": len(by_page),
            "table_count_with_headers": len(by_table),
            "by_page": by_page,
            "by_table": by_table,
            "by_header_type": by_type,
        }

    def _sort_headers(
        self,
        headers: List[TableHeader],
    ) -> List[TableHeader]:
        return sorted(
            headers,
            key=lambda item: (
                item.page_number,
                item.table_grid_id,
            ),
        )

    def _sort_header_cells(
        self,
        header_cells: List[TableHeaderCell],
    ) -> List[TableHeaderCell]:
        return sorted(
            header_cells,
            key=lambda item: (
                item.page_number,
                item.table_grid_id,
                item.row_index,
                item.col_index,
            ),
        )

    def _normalize_header_text(
        self,
        text: Any,
    ) -> str:
        text = self._clean_text(text)

        if not self.config.normalize_header_text:
            return text

        text = re.sub(r"\s+", " ", text)
        text = text.strip(" :;,.|-")

        return text.strip()

    def _clean_text(
        self,
        text: Any,
    ) -> str:
        if text is None:
            return ""

        text = str(text)
        text = text.replace("\u00a0", " ")
        text = text.replace("Ƣ", "Ư")
        text = text.replace("ƣ", "ư")
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def _has_header_keyword(
        self,
        text: str,
    ) -> bool:
        if not text:
            return False

        lower = text.lower().strip()

        keywords = [
            "stt",
            "tt",
            "số",
            "so",
            "tên",
            "ten",
            "nội dung",
            "noi dung",
            "đơn vị",
            "don vi",
            "kết quả",
            "ket qua",
            "ghi chú",
            "ghi chu",
            "ngày",
            "tháng",
            "năm",
            "giá trị",
            "gia tri",
            "tỷ lệ",
            "ty le",
            "chỉ tiêu",
            "chi tieu",
            "mã",
            "ma",
            "loại",
            "loai",
            "tổng",
            "tong",
            "name",
            "value",
            "date",
            "total",
            "amount",
            "description",
        ]

        return any(keyword in lower for keyword in keywords)

    def _has_row_header_keyword(
        self,
        text: str,
    ) -> bool:
        if not text:
            return False

        lower = text.lower().strip()

        keywords = [
            "tổng",
            "tong",
            "cộng",
            "cong",
            "mục",
            "muc",
            "nhóm",
            "nhom",
            "loại",
            "loai",
            "hạng mục",
            "hang muc",
            "subtotal",
            "total",
            "group",
            "category",
        ]

        return any(keyword in lower for keyword in keywords)

    def _is_numeric_like(
        self,
        text: str,
    ) -> bool:
        if not text:
            return False

        clean = text.strip()
        clean = clean.replace(".", "")
        clean = clean.replace(",", "")
        clean = clean.replace("%", "")
        clean = clean.replace("VNĐ", "")
        clean = clean.replace("VND", "")
        clean = clean.replace("đ", "")
        clean = clean.replace("-", "")
        clean = clean.replace("+", "")
        clean = clean.strip()

        if not clean:
            return False

        if re.fullmatch(r"\d+", clean):
            return True

        if re.fullmatch(r"\d+/\d+(/\d+)?", text.strip()):
            return True

        digit_count = sum(ch.isdigit() for ch in text)
        alpha_count = sum(ch.isalpha() for ch in text)

        if digit_count > 0 and alpha_count == 0:
            return True

        return False

    def _safe_int(
        self,
        value: Any,
        default: int = 0,
    ) -> int:
        try:
            if value is None:
                return default

            return int(value)
        except Exception:
            return default

    def _safe_float(
        self,
        value: Any,
        default: float = 0.0,
    ) -> float:
        try:
            if value is None:
                return default

            return float(value)
        except Exception:
            return default


def detect_table_headers(
    page_raws: List[PageRaw],
    table_cell_result: Optional[Dict[str, Any]] = None,
    table_structure_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    detector = TableHeaderDetector()
    return detector.process(
        page_raws=page_raws,
        table_cell_result=table_cell_result,
        table_structure_result=table_structure_result,
    )
