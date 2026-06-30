"""
table_span_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect row span and column span in extracted table cells.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- TableBoundaryDetector
- TableGridBuilder
- TableStructureRecognizer
- TableCellExtractor
- TableHeaderDetector

Output
------
Dictionary with:
- table_spans
- table_cells_with_spans
- table_matrices_with_spans
- table_spans_by_page
- table_spans_by_table
- table_span_summary

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
    ↓
TableSemanticRecognizer
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class TableSpanDetectorConfig:
    detect_existing_span: bool = True
    infer_colspan_from_empty_neighbors: bool = True
    infer_rowspan_from_empty_neighbors: bool = True

    prefer_header_colspan: bool = True
    prefer_stub_rowspan: bool = True

    max_col_span: int = 8
    max_row_span: int = 20

    min_span_confidence: float = 0.35

    allow_body_colspan: bool = False
    allow_body_rowspan: bool = True

    include_covered_cells: bool = True
    attach_to_pages: bool = True
    include_debug: bool = True


@dataclass
class TableSpan:
    table_span_id: str
    table_grid_id: str
    table_structure_id: str
    table_boundary_id: str

    page_number: int
    page_index: int

    origin_cell_id: str
    covered_cell_ids: List[str]

    row_index: int
    col_index: int
    row_span: int
    col_span: int

    bbox: List[float]
    span_type: str
    confidence: float

    text: str = ""
    normalized_text: str = ""
    source: str = "table_span_detector"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class TableSpanDetector:
    def __init__(
        self,
        config: Optional[TableSpanDetectorConfig] = None,
    ):
        self.config = config or TableSpanDetectorConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        table_cell_result: Optional[Dict[str, Any]] = None,
        table_header_result: Optional[Dict[str, Any]] = None,
        table_grid_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cells = self._collect_table_cells(
            page_raws=page_raws,
            table_cell_result=table_cell_result,
        )

        headers = self._collect_table_headers(
            page_raws=page_raws,
            table_header_result=table_header_result,
        )

        grids = self._collect_table_grids(
            page_raws=page_raws,
            table_grid_result=table_grid_result,
        )

        cells_by_table = self._group_cells_by_table(cells)

        header_by_table = {
            header.get("table_grid_id", ""): header
            for header in headers
            if header.get("table_grid_id")
        }

        grid_by_id = {
            grid.get("table_grid_id", ""): grid
            for grid in grids
            if grid.get("table_grid_id")
        }

        all_spans: List[TableSpan] = []
        all_cells_with_spans: List[Dict[str, Any]] = []
        matrices_with_spans: Dict[str, List[List[Dict[str, Any]]]] = {}

        for table_grid_id, table_cells in cells_by_table.items():
            table_header = header_by_table.get(table_grid_id, {})
            table_grid = grid_by_id.get(table_grid_id, {})

            table_spans, cells_with_spans, matrix_with_spans = self.detect_spans_for_table(
                table_grid_id=table_grid_id,
                table_cells=table_cells,
                table_header=table_header,
                table_grid=table_grid,
            )

            all_spans.extend(table_spans)
            all_cells_with_spans.extend(cells_with_spans)
            matrices_with_spans[table_grid_id] = matrix_with_spans

        all_spans = self._sort_spans(all_spans)
        all_cells_with_spans = self._sort_cell_dicts(all_cells_with_spans)

        result = {
            "processor": "TableSpanDetector",
            "table_spans": [
                span.to_dict() for span in all_spans
            ],
            "table_cells_with_spans": all_cells_with_spans,
            "table_matrices_with_spans": matrices_with_spans,
            "table_spans_by_page": self._group_spans_by_page(all_spans),
            "table_spans_by_table": self._group_spans_by_table(all_spans),
            "table_cells_with_spans_by_table": self._group_cell_dicts_by_table(all_cells_with_spans),
            "table_span_summary": self._build_summary(
                spans=all_spans,
                cells_with_spans=all_cells_with_spans,
            ),
            "config": {
                "detect_existing_span": self.config.detect_existing_span,
                "infer_colspan_from_empty_neighbors": self.config.infer_colspan_from_empty_neighbors,
                "infer_rowspan_from_empty_neighbors": self.config.infer_rowspan_from_empty_neighbors,
                "prefer_header_colspan": self.config.prefer_header_colspan,
                "prefer_stub_rowspan": self.config.prefer_stub_rowspan,
                "max_col_span": self.config.max_col_span,
                "max_row_span": self.config.max_row_span,
                "min_span_confidence": self.config.min_span_confidence,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                spans=all_spans,
                cells_with_spans=all_cells_with_spans,
                result=result,
            )

        return result

    def detect_spans_for_table(
        self,
        table_grid_id: str,
        table_cells: List[Dict[str, Any]],
        table_header: Optional[Dict[str, Any]] = None,
        table_grid: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[TableSpan], List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        table_header = table_header or {}
        table_grid = table_grid or {}

        row_count, col_count = self._resolve_table_shape(
            table_cells=table_cells,
            table_grid=table_grid,
        )

        matrix = self._build_cell_matrix(
            table_cells=table_cells,
            row_count=row_count,
            col_count=col_count,
        )

        header_rows = [
            self._safe_int(item, default=-1)
            for item in table_header.get("header_row_indices", []) or []
        ]

        header_cols = [
            self._safe_int(item, default=-1)
            for item in table_header.get("header_col_indices", []) or []
        ]

        spans: List[TableSpan] = []
        covered: Dict[str, str] = {}

        if self.config.detect_existing_span:
            existing_spans = self._detect_existing_spans(
                matrix=matrix,
                table_header=table_header,
            )

            for span in existing_spans:
                if span.confidence >= self.config.min_span_confidence:
                    spans.append(span)

                    for covered_cell_id in span.covered_cell_ids:
                        covered[covered_cell_id] = span.origin_cell_id

        if self.config.infer_colspan_from_empty_neighbors:
            col_spans = self._infer_colspans(
                matrix=matrix,
                header_rows=header_rows,
                header_cols=header_cols,
                covered=covered,
            )

            for span in col_spans:
                if span.confidence >= self.config.min_span_confidence:
                    spans.append(span)

                    for covered_cell_id in span.covered_cell_ids:
                        covered[covered_cell_id] = span.origin_cell_id

        if self.config.infer_rowspan_from_empty_neighbors:
            row_spans = self._infer_rowspans(
                matrix=matrix,
                header_rows=header_rows,
                header_cols=header_cols,
                covered=covered,
            )

            for span in row_spans:
                if span.confidence >= self.config.min_span_confidence:
                    spans.append(span)

                    for covered_cell_id in span.covered_cell_ids:
                        covered[covered_cell_id] = span.origin_cell_id

        spans = self._deduplicate_spans(spans)
        cells_with_spans = self._build_cells_with_span_metadata(
            matrix=matrix,
            spans=spans,
            covered=covered,
        )

        matrix_with_spans = self._build_matrix_with_spans(
            cells_with_spans=cells_with_spans,
            row_count=row_count,
            col_count=col_count,
        )

        return spans, cells_with_spans, matrix_with_spans

    def _detect_existing_spans(
        self,
        matrix: List[List[Dict[str, Any]]],
        table_header: Dict[str, Any],
    ) -> List[TableSpan]:
        spans: List[TableSpan] = []

        for row in matrix:
            for cell in row:
                row_span = max(1, self._safe_int(cell.get("row_span"), default=1))
                col_span = max(1, self._safe_int(cell.get("col_span"), default=1))

                if row_span <= 1 and col_span <= 1:
                    continue

                covered_cell_ids = self._collect_covered_cell_ids(
                    matrix=matrix,
                    row_index=self._safe_int(cell.get("row_index"), default=0),
                    col_index=self._safe_int(cell.get("col_index"), default=0),
                    row_span=row_span,
                    col_span=col_span,
                    origin_cell_id=cell.get("table_cell_id", ""),
                )

                span_type = self._classify_span_type(
                    row_span=row_span,
                    col_span=col_span,
                )

                spans.append(
                    self._make_span(
                        origin_cell=cell,
                        covered_cell_ids=covered_cell_ids,
                        row_span=row_span,
                        col_span=col_span,
                        span_type=span_type,
                        confidence=0.85,
                        detection_method="existing_cell_span",
                        metadata={
                            "source_row_span": row_span,
                            "source_col_span": col_span,
                            "table_header_id": table_header.get("table_header_id", ""),
                        },
                    )
                )

        return spans

    def _infer_colspans(
        self,
        matrix: List[List[Dict[str, Any]]],
        header_rows: List[int],
        header_cols: List[int],
        covered: Dict[str, str],
    ) -> List[TableSpan]:
        spans: List[TableSpan] = []

        for row_index, row in enumerate(matrix):
            for col_index, cell in enumerate(row):
                cell_id = cell.get("table_cell_id", "")

                if not cell_id or cell_id in covered:
                    continue

                text = self._cell_text(cell)

                if not text:
                    continue

                if not self._allow_colspan_for_cell(
                    cell=cell,
                    row_index=row_index,
                    col_index=col_index,
                    header_rows=header_rows,
                    header_cols=header_cols,
                ):
                    continue

                col_span = 1
                covered_cell_ids: List[str] = []

                for next_col in range(col_index + 1, min(len(row), col_index + self.config.max_col_span)):
                    next_cell = row[next_col]
                    next_cell_id = next_cell.get("table_cell_id", "")

                    if not next_cell_id:
                        break

                    if next_cell_id in covered:
                        break

                    if not self._is_empty_cell(next_cell):
                        break

                    if not self._is_adjacent_horizontally(cell, next_cell):
                        break

                    covered_cell_ids.append(next_cell_id)
                    col_span += 1

                if col_span <= 1:
                    continue

                confidence = self._score_inferred_colspan(
                    cell=cell,
                    row_index=row_index,
                    col_span=col_span,
                    header_rows=header_rows,
                )

                spans.append(
                    self._make_span(
                        origin_cell=cell,
                        covered_cell_ids=covered_cell_ids,
                        row_span=1,
                        col_span=col_span,
                        span_type="colspan",
                        confidence=confidence,
                        detection_method="empty_neighbor_colspan_inference",
                        metadata={
                            "header_rows": header_rows,
                            "header_cols": header_cols,
                        },
                    )
                )

        return spans

    def _infer_rowspans(
        self,
        matrix: List[List[Dict[str, Any]]],
        header_rows: List[int],
        header_cols: List[int],
        covered: Dict[str, str],
    ) -> List[TableSpan]:
        spans: List[TableSpan] = []

        row_count = len(matrix)
        col_count = len(matrix[0]) if matrix else 0

        for row_index in range(row_count):
            for col_index in range(col_count):
                cell = matrix[row_index][col_index]
                cell_id = cell.get("table_cell_id", "")

                if not cell_id or cell_id in covered:
                    continue

                text = self._cell_text(cell)

                if not text:
                    continue

                if not self._allow_rowspan_for_cell(
                    cell=cell,
                    row_index=row_index,
                    col_index=col_index,
                    header_rows=header_rows,
                    header_cols=header_cols,
                ):
                    continue

                row_span = 1
                covered_cell_ids: List[str] = []

                for next_row in range(row_index + 1, min(row_count, row_index + self.config.max_row_span)):
                    next_cell = matrix[next_row][col_index]
                    next_cell_id = next_cell.get("table_cell_id", "")

                    if not next_cell_id:
                        break

                    if next_cell_id in covered:
                        break

                    if not self._is_empty_cell(next_cell):
                        break

                    if not self._is_adjacent_vertically(cell, next_cell):
                        break

                    covered_cell_ids.append(next_cell_id)
                    row_span += 1

                if row_span <= 1:
                    continue

                confidence = self._score_inferred_rowspan(
                    cell=cell,
                    col_index=col_index,
                    row_span=row_span,
                    header_cols=header_cols,
                )

                spans.append(
                    self._make_span(
                        origin_cell=cell,
                        covered_cell_ids=covered_cell_ids,
                        row_span=row_span,
                        col_span=1,
                        span_type="rowspan",
                        confidence=confidence,
                        detection_method="empty_neighbor_rowspan_inference",
                        metadata={
                            "header_rows": header_rows,
                            "header_cols": header_cols,
                        },
                    )
                )

        return spans

    def _allow_colspan_for_cell(
        self,
        cell: Dict[str, Any],
        row_index: int,
        col_index: int,
        header_rows: List[int],
        header_cols: List[int],
    ) -> bool:
        if row_index in header_rows and self.config.prefer_header_colspan:
            return True

        if cell.get("is_header") or cell.get("cell_role") in ["header", "column_header", "corner_header"]:
            return True

        if self.config.allow_body_colspan:
            return True

        return False

    def _allow_rowspan_for_cell(
        self,
        cell: Dict[str, Any],
        row_index: int,
        col_index: int,
        header_rows: List[int],
        header_cols: List[int],
    ) -> bool:
        if col_index in header_cols and self.config.prefer_stub_rowspan:
            return True

        if cell.get("is_stub") or cell.get("cell_role") in ["stub", "row_header"]:
            return True

        if self.config.allow_body_rowspan and not cell.get("is_header"):
            return True

        return False

    def _make_span(
        self,
        origin_cell: Dict[str, Any],
        covered_cell_ids: List[str],
        row_span: int,
        col_span: int,
        span_type: str,
        confidence: float,
        detection_method: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TableSpan:
        metadata = metadata or {}
        bbox = origin_cell.get("bbox", []) or []

        return TableSpan(
            table_span_id=make_id("tbl_span"),
            table_grid_id=origin_cell.get("table_grid_id", ""),
            table_structure_id=origin_cell.get("table_structure_id", ""),
            table_boundary_id=origin_cell.get("table_boundary_id", ""),
            page_number=self._safe_int(origin_cell.get("page_number"), default=0),
            page_index=self._safe_int(origin_cell.get("page_index"), default=0),
            origin_cell_id=origin_cell.get("table_cell_id", ""),
            covered_cell_ids=covered_cell_ids,
            row_index=self._safe_int(origin_cell.get("row_index"), default=0),
            col_index=self._safe_int(origin_cell.get("col_index"), default=0),
            row_span=row_span,
            col_span=col_span,
            bbox=[
                round(float(value), 4) for value in bbox
            ] if len(bbox) == 4 else [],
            span_type=span_type,
            confidence=round(max(0.0, min(confidence, 0.95)), 4),
            text=origin_cell.get("text", ""),
            normalized_text=origin_cell.get("normalized_text", ""),
            source="table_span_detector",
            metadata={
                "detection_method": detection_method,
                "origin_cell_role": origin_cell.get("cell_role", ""),
                "origin_is_header": origin_cell.get("is_header", False),
                "origin_is_stub": origin_cell.get("is_stub", False),
                **metadata,
            },
        )

    def _build_cells_with_span_metadata(
        self,
        matrix: List[List[Dict[str, Any]]],
        spans: List[TableSpan],
        covered: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        span_by_origin = {
            span.origin_cell_id: span
            for span in spans
        }

        cells_with_spans: List[Dict[str, Any]] = []

        for row in matrix:
            for cell in row:
                cell_dict = dict(cell)
                cell_id = cell_dict.get("table_cell_id", "")

                span = span_by_origin.get(cell_id)
                covered_by = covered.get(cell_id, "")

                cell_dict["is_span_origin"] = span is not None
                cell_dict["is_covered_by_span"] = covered_by != ""
                cell_dict["covered_by_cell_id"] = covered_by
                cell_dict["span_id"] = span.table_span_id if span else ""
                cell_dict["detected_row_span"] = span.row_span if span else 1
                cell_dict["detected_col_span"] = span.col_span if span else 1
                cell_dict["effective_row_span"] = span.row_span if span else self._safe_int(cell_dict.get("row_span"), default=1)
                cell_dict["effective_col_span"] = span.col_span if span else self._safe_int(cell_dict.get("col_span"), default=1)

                if covered_by and not self.config.include_covered_cells:
                    continue

                cells_with_spans.append(cell_dict)

        return cells_with_spans

    def _build_matrix_with_spans(
        self,
        cells_with_spans: List[Dict[str, Any]],
        row_count: int,
        col_count: int,
    ) -> List[List[Dict[str, Any]]]:
        matrix: List[List[Dict[str, Any]]] = []

        for row_index in range(row_count):
            row: List[Dict[str, Any]] = []

            for col_index in range(col_count):
                row.append(
                    {
                        "row_index": row_index,
                        "col_index": col_index,
                        "table_cell_id": "",
                        "text": "",
                        "normalized_text": "",
                        "is_empty": True,
                        "is_span_origin": False,
                        "is_covered_by_span": False,
                        "covered_by_cell_id": "",
                        "effective_row_span": 1,
                        "effective_col_span": 1,
                    }
                )

            matrix.append(row)

        for cell in cells_with_spans:
            row_index = self._safe_int(cell.get("row_index"), default=-1)
            col_index = self._safe_int(cell.get("col_index"), default=-1)

            if row_index < 0 or row_index >= row_count:
                continue

            if col_index < 0 or col_index >= col_count:
                continue

            matrix[row_index][col_index] = cell

        return matrix

    def _collect_covered_cell_ids(
        self,
        matrix: List[List[Dict[str, Any]]],
        row_index: int,
        col_index: int,
        row_span: int,
        col_span: int,
        origin_cell_id: str,
    ) -> List[str]:
        covered: List[str] = []

        for r in range(row_index, min(row_index + row_span, len(matrix))):
            row = matrix[r]

            for c in range(col_index, min(col_index + col_span, len(row))):
                cell = row[c]
                cell_id = cell.get("table_cell_id", "")

                if cell_id and cell_id != origin_cell_id:
                    covered.append(cell_id)

        return covered

    def _resolve_table_shape(
        self,
        table_cells: List[Dict[str, Any]],
        table_grid: Dict[str, Any],
    ) -> Tuple[int, int]:
        row_count = self._safe_int(table_grid.get("row_count"), default=0)
        col_count = self._safe_int(table_grid.get("col_count"), default=0)

        if row_count <= 0:
            row_count = max(
                [
                    self._safe_int(cell.get("row_index"), default=-1)
                    for cell in table_cells
                ],
                default=-1,
            ) + 1

        if col_count <= 0:
            col_count = max(
                [
                    self._safe_int(cell.get("col_index"), default=-1)
                    for cell in table_cells
                ],
                default=-1,
            ) + 1

        return max(row_count, 0), max(col_count, 0)

    def _build_cell_matrix(
        self,
        table_cells: List[Dict[str, Any]],
        row_count: int,
        col_count: int,
    ) -> List[List[Dict[str, Any]]]:
        matrix: List[List[Dict[str, Any]]] = []

        for row_index in range(row_count):
            row = []

            for col_index in range(col_count):
                row.append(
                    {
                        "table_cell_id": "",
                        "table_grid_id": "",
                        "table_structure_id": "",
                        "table_boundary_id": "",
                        "page_number": 0,
                        "page_index": 0,
                        "row_index": row_index,
                        "col_index": col_index,
                        "row_span": 1,
                        "col_span": 1,
                        "bbox": [],
                        "text": "",
                        "normalized_text": "",
                        "cell_role": "empty",
                        "is_header": False,
                        "is_stub": False,
                        "is_numeric": False,
                        "is_empty": True,
                    }
                )

            matrix.append(row)

        for cell in table_cells:
            row_index = self._safe_int(cell.get("row_index"), default=-1)
            col_index = self._safe_int(cell.get("col_index"), default=-1)

            if row_index < 0 or row_index >= row_count:
                continue

            if col_index < 0 or col_index >= col_count:
                continue

            matrix[row_index][col_index] = dict(cell)

        return matrix

    def _deduplicate_spans(
        self,
        spans: List[TableSpan],
    ) -> List[TableSpan]:
        best_by_key: Dict[str, TableSpan] = {}

        for span in spans:
            key = f"{span.table_grid_id}|{span.origin_cell_id}|{span.row_index}|{span.col_index}"

            if key not in best_by_key:
                best_by_key[key] = span
                continue

            old = best_by_key[key]

            old_area = old.row_span * old.col_span
            new_area = span.row_span * span.col_span

            if new_area > old_area:
                best_by_key[key] = span
            elif new_area == old_area and span.confidence > old.confidence:
                best_by_key[key] = span

        return list(best_by_key.values())

    def _classify_span_type(
        self,
        row_span: int,
        col_span: int,
    ) -> str:
        if row_span > 1 and col_span > 1:
            return "rowspan_colspan"

        if row_span > 1:
            return "rowspan"

        if col_span > 1:
            return "colspan"

        return "no_span"

    def _score_inferred_colspan(
        self,
        cell: Dict[str, Any],
        row_index: int,
        col_span: int,
        header_rows: List[int],
    ) -> float:
        score = 0.35

        if row_index in header_rows:
            score += 0.25

        if cell.get("is_header") or cell.get("cell_role") == "header":
            score += 0.15

        if col_span >= 2:
            score += 0.10

        if self._cell_text(cell):
            score += 0.05

        return round(max(0.0, min(score, 0.90)), 4)

    def _score_inferred_rowspan(
        self,
        cell: Dict[str, Any],
        col_index: int,
        row_span: int,
        header_cols: List[int],
    ) -> float:
        score = 0.35

        if col_index in header_cols:
            score += 0.20

        if cell.get("is_stub") or cell.get("cell_role") == "stub":
            score += 0.20

        if row_span >= 2:
            score += 0.10

        if self._cell_text(cell):
            score += 0.05

        return round(max(0.0, min(score, 0.90)), 4)

    def _is_empty_cell(
        self,
        cell: Dict[str, Any],
    ) -> bool:
        if bool(cell.get("is_empty", False)):
            return True

        text = self._cell_text(cell)

        return text == ""

    def _cell_text(
        self,
        cell: Dict[str, Any],
    ) -> str:
        text = cell.get("normalized_text") or cell.get("text") or ""

        return self._clean_text(text)

    def _is_adjacent_horizontally(
        self,
        left_cell: Dict[str, Any],
        right_cell: Dict[str, Any],
    ) -> bool:
        left_bbox = left_cell.get("bbox", [])
        right_bbox = right_cell.get("bbox", [])

        if len(left_bbox) != 4 or len(right_bbox) != 4:
            return True

        y_overlap = self._vertical_overlap_ratio(left_bbox, right_bbox)

        return y_overlap >= 0.60

    def _is_adjacent_vertically(
        self,
        top_cell: Dict[str, Any],
        bottom_cell: Dict[str, Any],
    ) -> bool:
        top_bbox = top_cell.get("bbox", [])
        bottom_bbox = bottom_cell.get("bbox", [])

        if len(top_bbox) != 4 or len(bottom_bbox) != 4:
            return True

        x_overlap = self._horizontal_overlap_ratio(top_bbox, bottom_bbox)

        return x_overlap >= 0.60

    def _horizontal_overlap_ratio(
        self,
        a: List[float],
        b: List[float],
    ) -> float:
        x0 = max(float(a[0]), float(b[0]))
        x1 = min(float(a[2]), float(b[2]))
        overlap = max(x1 - x0, 0.0)

        width_a = max(float(a[2]) - float(a[0]), 0.0)
        width_b = max(float(b[2]) - float(b[0]), 0.0)
        smaller = min(width_a, width_b)

        if smaller <= 0:
            return 0.0

        return overlap / smaller

    def _vertical_overlap_ratio(
        self,
        a: List[float],
        b: List[float],
    ) -> float:
        y0 = max(float(a[1]), float(b[1]))
        y1 = min(float(a[3]), float(b[3]))
        overlap = max(y1 - y0, 0.0)

        height_a = max(float(a[3]) - float(a[1]), 0.0)
        height_b = max(float(b[3]) - float(b[1]), 0.0)
        smaller = min(height_a, height_b)

        if smaller <= 0:
            return 0.0

        return overlap / smaller

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
            meta = page_raw.metadata.get("table_cell_extractor", {})
            page_cells = meta.get("table_cells_on_page", [])

            for cell in page_cells:
                collected.append(cell)

        return collected

    def _collect_table_headers(
        self,
        page_raws: List[PageRaw],
        table_header_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if table_header_result:
            headers = table_header_result.get("table_headers", [])

            if headers:
                return headers

        collected: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_header_detector", {})
            page_headers = meta.get("table_headers_on_page", [])

            for header in page_headers:
                collected.append(header)

        return collected

    def _collect_table_grids(
        self,
        page_raws: List[PageRaw],
        table_grid_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if table_grid_result:
            grids = table_grid_result.get("table_grids", [])

            if grids:
                return grids

        collected: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_grid_builder", {})
            page_grids = meta.get("table_grids_on_page", [])

            for grid in page_grids:
                collected.append(grid)

        return collected

    def _group_cells_by_table(
        self,
        cells: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for cell in cells:
            table_grid_id = cell.get("table_grid_id", "") or "unknown_table"
            grouped.setdefault(table_grid_id, [])
            grouped[table_grid_id].append(cell)

        return grouped

    def _group_spans_by_page(
        self,
        spans: List[TableSpan],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for span in spans:
            page_key = str(span.page_number)
            grouped.setdefault(page_key, [])
            grouped[page_key].append(span.to_dict())

        return grouped

    def _group_spans_by_table(
        self,
        spans: List[TableSpan],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for span in spans:
            table_key = span.table_grid_id or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(span.to_dict())

        return grouped

    def _group_cell_dicts_by_table(
        self,
        cells: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for cell in cells:
            table_key = cell.get("table_grid_id", "") or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(cell)

        return grouped

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        spans: List[TableSpan],
        cells_with_spans: List[Dict[str, Any]],
        result: Dict[str, Any],
    ) -> None:
        spans_by_page: Dict[int, List[Dict[str, Any]]] = {}
        cells_by_page: Dict[int, List[Dict[str, Any]]] = {}

        for span in spans:
            spans_by_page.setdefault(span.page_number, [])
            spans_by_page[span.page_number].append(span.to_dict())

        for cell in cells_with_spans:
            page_number = self._safe_int(cell.get("page_number"), default=-1)
            cells_by_page.setdefault(page_number, [])
            cells_by_page[page_number].append(cell)

        for page_raw in page_raws:
            page_spans = spans_by_page.get(page_raw.page_number, [])
            page_cells = cells_by_page.get(page_raw.page_number, [])

            page_raw.metadata.setdefault("table_span_detector", {})
            page_raw.metadata["table_span_detector"] = {
                "processor": "TableSpanDetector",
                "table_spans_on_page": page_spans,
                "table_cells_with_spans_on_page": page_cells,
                "table_span_count_on_page": len(page_spans),
                "table_cell_with_span_count_on_page": len(page_cells),
                "table_span_summary": result.get("table_span_summary", {}),
            }

    def _build_summary(
        self,
        spans: List[TableSpan],
        cells_with_spans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        by_page: Dict[str, int] = {}
        by_table: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for span in spans:
            page_key = str(span.page_number)
            table_key = span.table_grid_id or "unknown_table"
            type_key = span.span_type or "unknown"

            by_page[page_key] = by_page.get(page_key, 0) + 1
            by_table[table_key] = by_table.get(table_key, 0) + 1
            by_type[type_key] = by_type.get(type_key, 0) + 1

        origin_count = sum(
            1 for cell in cells_with_spans
            if cell.get("is_span_origin")
        )

        covered_count = sum(
            1 for cell in cells_with_spans
            if cell.get("is_covered_by_span")
        )

        return {
            "has_table_spans": len(spans) > 0,
            "table_span_count": len(spans),
            "span_origin_cell_count": origin_count,
            "covered_cell_count": covered_count,
            "page_count_with_spans": len(by_page),
            "table_count_with_spans": len(by_table),
            "by_page": by_page,
            "by_table": by_table,
            "by_span_type": by_type,
        }

    def _sort_spans(
        self,
        spans: List[TableSpan],
    ) -> List[TableSpan]:
        return sorted(
            spans,
            key=lambda item: (
                item.page_number,
                item.table_grid_id,
                item.row_index,
                item.col_index,
            ),
        )

    def _sort_cell_dicts(
        self,
        cells: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return sorted(
            cells,
            key=lambda item: (
                self._safe_int(item.get("page_number"), default=0),
                item.get("table_grid_id", ""),
                self._safe_int(item.get("row_index"), default=0),
                self._safe_int(item.get("col_index"), default=0),
            ),
        )

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


def detect_table_spans(
    page_raws: List[PageRaw],
    table_cell_result: Optional[Dict[str, Any]] = None,
    table_header_result: Optional[Dict[str, Any]] = None,
    table_grid_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    detector = TableSpanDetector()
    return detector.process(
        page_raws=page_raws,
        table_cell_result=table_cell_result,
        table_header_result=table_header_result,
        table_grid_result=table_grid_result,
    )
