"""
table_semantic_recognizer.py

Production V1 - Colab Ready

Purpose
-------
Recognize semantic meaning of tables, columns, rows and records.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- TableBoundaryDetector
- TableGridBuilder
- TableStructureRecognizer
- TableCellExtractor
- TableHeaderDetector
- TableSpanDetector

Output
------
Dictionary with:
- table_semantics
- table_columns
- table_rows
- table_records
- table_semantics_by_page
- table_semantics_by_table
- table_semantic_summary

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
class TableSemanticRecognizerConfig:
    use_span_cells_if_available: bool = True
    use_header_result: bool = True
    use_structure_result: bool = True

    infer_column_semantics: bool = True
    infer_row_semantics: bool = True
    build_records: bool = True

    skip_header_rows_in_records: bool = True
    skip_empty_rows_in_records: bool = True
    skip_total_rows_in_records: bool = False

    max_records_per_table: int = 10000
    text_preview_chars: int = 800

    min_column_confidence: float = 0.25
    min_table_confidence: float = 0.30

    attach_to_pages: bool = True
    include_debug: bool = True


@dataclass
class TableColumnSemantic:
    table_column_id: str
    table_grid_id: str
    table_structure_id: str
    table_boundary_id: str

    page_number: int
    page_index: int

    col_index: int
    header: str
    normalized_header: str

    semantic_type: str
    data_type: str

    sample_values: List[str]
    non_empty_count: int
    numeric_ratio: float
    empty_ratio: float

    confidence: float
    source: str = "table_semantic_recognizer"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class TableRowSemantic:
    table_row_id: str
    table_grid_id: str
    table_structure_id: str
    table_boundary_id: str

    page_number: int
    page_index: int

    row_index: int
    row_type: str

    text: str
    non_empty_count: int
    numeric_count: int
    empty_count: int

    confidence: float
    source: str = "table_semantic_recognizer"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class TableRecord:
    table_record_id: str
    table_grid_id: str
    table_structure_id: str
    table_boundary_id: str

    page_number: int
    page_index: int
    row_index: int

    values: Dict[str, Any]
    raw_values: Dict[str, Any]
    cell_ids: Dict[str, str]

    record_type: str = "data_record"
    confidence: float = 0.5
    source: str = "table_semantic_recognizer"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class TableSemantic:
    table_semantic_id: str
    table_grid_id: str
    table_structure_id: str
    table_boundary_id: str

    page_number: int
    page_index: int
    bbox: List[float]

    table_type: str
    semantic_type: str

    row_count: int
    col_count: int

    header_rows: List[int]
    header_cols: List[int]

    column_ids: List[str]
    row_ids: List[str]
    record_ids: List[str]

    title: str = ""
    caption: str = ""
    text_preview: str = ""

    confidence: float = 0.5
    source: str = "table_semantic_recognizer"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class TableSemanticRecognizer:
    def __init__(
        self,
        config: Optional[TableSemanticRecognizerConfig] = None,
    ):
        self.config = config or TableSemanticRecognizerConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        table_cell_result: Optional[Dict[str, Any]] = None,
        table_header_result: Optional[Dict[str, Any]] = None,
        table_span_result: Optional[Dict[str, Any]] = None,
        table_structure_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cells = self._collect_cells(
            page_raws=page_raws,
            table_cell_result=table_cell_result,
            table_span_result=table_span_result,
        )

        headers = self._collect_headers(
            page_raws=page_raws,
            table_header_result=table_header_result,
        )

        structures = self._collect_structures(
            page_raws=page_raws,
            table_structure_result=table_structure_result,
        )

        cells_by_table = self._group_cells_by_table(cells)

        header_by_table = {
            header.get("table_grid_id", ""): header
            for header in headers
            if header.get("table_grid_id")
        }

        structure_by_table = {
            structure.get("table_grid_id", ""): structure
            for structure in structures
            if structure.get("table_grid_id")
        }

        all_table_semantics: List[TableSemantic] = []
        all_columns: List[TableColumnSemantic] = []
        all_rows: List[TableRowSemantic] = []
        all_records: List[TableRecord] = []

        for table_grid_id, table_cells in cells_by_table.items():
            header = header_by_table.get(table_grid_id, {})
            structure = structure_by_table.get(table_grid_id, {})

            table_semantic, columns, rows, records = self.recognize_table_semantics(
                table_grid_id=table_grid_id,
                table_cells=table_cells,
                header=header,
                structure=structure,
            )

            if table_semantic and table_semantic.confidence >= self.config.min_table_confidence:
                all_table_semantics.append(table_semantic)
                all_columns.extend(columns)
                all_rows.extend(rows)
                all_records.extend(records)

        all_table_semantics = self._sort_table_semantics(all_table_semantics)
        all_columns = self._sort_columns(all_columns)
        all_rows = self._sort_rows(all_rows)
        all_records = self._sort_records(all_records)

        result = {
            "processor": "TableSemanticRecognizer",
            "table_semantics": [
                item.to_dict() for item in all_table_semantics
            ],
            "table_columns": [
                item.to_dict() for item in all_columns
            ],
            "table_rows": [
                item.to_dict() for item in all_rows
            ],
            "table_records": [
                item.to_dict() for item in all_records
            ],
            "table_semantics_by_page": self._group_table_semantics_by_page(all_table_semantics),
            "table_semantics_by_table": self._group_table_semantics_by_table(all_table_semantics),
            "table_columns_by_table": self._group_columns_by_table(all_columns),
            "table_rows_by_table": self._group_rows_by_table(all_rows),
            "table_records_by_table": self._group_records_by_table(all_records),
            "table_semantic_summary": self._build_summary(
                table_semantics=all_table_semantics,
                columns=all_columns,
                rows=all_rows,
                records=all_records,
            ),
            "config": {
                "use_span_cells_if_available": self.config.use_span_cells_if_available,
                "use_header_result": self.config.use_header_result,
                "use_structure_result": self.config.use_structure_result,
                "infer_column_semantics": self.config.infer_column_semantics,
                "infer_row_semantics": self.config.infer_row_semantics,
                "build_records": self.config.build_records,
                "skip_header_rows_in_records": self.config.skip_header_rows_in_records,
                "skip_empty_rows_in_records": self.config.skip_empty_rows_in_records,
                "skip_total_rows_in_records": self.config.skip_total_rows_in_records,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                table_semantics=all_table_semantics,
                columns=all_columns,
                rows=all_rows,
                records=all_records,
                result=result,
            )

        return result

    def recognize_table_semantics(
        self,
        table_grid_id: str,
        table_cells: List[Dict[str, Any]],
        header: Optional[Dict[str, Any]] = None,
        structure: Optional[Dict[str, Any]] = None,
    ) -> Tuple[
        Optional[TableSemantic],
        List[TableColumnSemantic],
        List[TableRowSemantic],
        List[TableRecord],
    ]:
        header = header or {}
        structure = structure or {}

        if not table_cells:
            return None, [], [], []

        table_cells = self._sort_cell_dicts(table_cells)

        row_count, col_count = self._resolve_table_shape(
            cells=table_cells,
            structure=structure,
        )

        if row_count <= 0 or col_count <= 0:
            return None, [], [], []

        matrix = self._build_cell_matrix(
            cells=table_cells,
            row_count=row_count,
            col_count=col_count,
        )

        first_cell = table_cells[0]

        header_rows = self._resolve_header_rows(
            cells=table_cells,
            header=header,
            structure=structure,
            row_count=row_count,
        )

        header_cols = self._resolve_header_cols(
            cells=table_cells,
            header=header,
            structure=structure,
            col_count=col_count,
        )

        columns = self._build_column_semantics(
            matrix=matrix,
            table_grid_id=table_grid_id,
            header=header,
            structure=structure,
            header_rows=header_rows,
            header_cols=header_cols,
        )

        rows = self._build_row_semantics(
            matrix=matrix,
            table_grid_id=table_grid_id,
            header_rows=header_rows,
            header_cols=header_cols,
        )

        records: List[TableRecord] = []

        if self.config.build_records:
            records = self._build_records(
                matrix=matrix,
                columns=columns,
                rows=rows,
                table_grid_id=table_grid_id,
            )

        table_type = self._resolve_table_type(
            structure=structure,
            header=header,
            columns=columns,
            rows=rows,
        )

        semantic_type = self._classify_table_semantic_type(
            columns=columns,
            rows=rows,
            table_type=table_type,
        )

        bbox = self._resolve_table_bbox(
            cells=table_cells,
            structure=structure,
        )

        caption = self._resolve_caption(
            cells=table_cells,
            structure=structure,
        )

        title = self._resolve_title(
            header=header,
            structure=structure,
            caption=caption,
        )

        confidence = self._score_table_semantic(
            columns=columns,
            rows=rows,
            records=records,
            header_rows=header_rows,
            structure=structure,
            header=header,
        )

        table_semantic = TableSemantic(
            table_semantic_id=make_id("tbl_semantic"),
            table_grid_id=table_grid_id,
            table_structure_id=structure.get("table_structure_id", first_cell.get("table_structure_id", "")),
            table_boundary_id=structure.get("table_boundary_id", first_cell.get("table_boundary_id", "")),
            page_number=self._safe_int(first_cell.get("page_number"), default=0),
            page_index=self._safe_int(first_cell.get("page_index"), default=0),
            bbox=bbox,
            table_type=table_type,
            semantic_type=semantic_type,
            row_count=row_count,
            col_count=col_count,
            header_rows=header_rows,
            header_cols=header_cols,
            column_ids=[
                column.table_column_id for column in columns
            ],
            row_ids=[
                row.table_row_id for row in rows
            ],
            record_ids=[
                record.table_record_id for record in records
            ],
            title=title,
            caption=caption,
            text_preview=self._build_text_preview(matrix),
            confidence=confidence,
            source="table_semantic_recognizer",
            metadata={
                "structure_table_type": structure.get("table_type", ""),
                "header_type": header.get("header_type", ""),
                "has_header": len(header_rows) > 0,
                "has_stub": len(header_cols) > 0,
                "record_count": len(records),
                "column_count": len(columns),
                "row_count": len(rows),
            },
        )

        return table_semantic, columns, rows, records

    def _build_column_semantics(
        self,
        matrix: List[List[Dict[str, Any]]],
        table_grid_id: str,
        header: Dict[str, Any],
        structure: Dict[str, Any],
        header_rows: List[int],
        header_cols: List[int],
    ) -> List[TableColumnSemantic]:
        if not matrix:
            return []

        col_count = len(matrix[0])
        columns: List[TableColumnSemantic] = []

        resolved_headers = self._resolve_column_headers(
            matrix=matrix,
            header=header,
            structure=structure,
            header_rows=header_rows,
            col_count=col_count,
        )

        first_cell = self._first_non_empty_cell(matrix)

        for col_index in range(col_count):
            header_text = resolved_headers[col_index] if col_index < len(resolved_headers) else ""
            normalized_header = self._normalize_header(header_text)

            values = []

            for row_index, row in enumerate(matrix):
                if row_index in header_rows:
                    continue

                if col_index >= len(row):
                    continue

                text = self._cell_text(row[col_index])

                if text:
                    values.append(text)

            sample_values = values[:10]
            non_empty_count = len(values)
            total_data_rows = max(len(matrix) - len(header_rows), 1)
            empty_ratio = 1.0 - (non_empty_count / max(total_data_rows, 1))

            numeric_count = sum(
                1 for value in values
                if self._is_numeric_like(value)
            )

            numeric_ratio = numeric_count / max(non_empty_count, 1)

            semantic_type = self._infer_column_semantic_type(
                header=normalized_header,
                values=values,
                col_index=col_index,
                header_cols=header_cols,
            )

            data_type = self._infer_column_data_type(
                header=normalized_header,
                values=values,
                numeric_ratio=numeric_ratio,
            )

            confidence = self._score_column_semantic(
                header=normalized_header,
                values=values,
                semantic_type=semantic_type,
                data_type=data_type,
                numeric_ratio=numeric_ratio,
            )

            if confidence < self.config.min_column_confidence:
                semantic_type = "unknown_column"

            columns.append(
                TableColumnSemantic(
                    table_column_id=make_id("tbl_col_sem"),
                    table_grid_id=table_grid_id,
                    table_structure_id=structure.get("table_structure_id", first_cell.get("table_structure_id", "")),
                    table_boundary_id=structure.get("table_boundary_id", first_cell.get("table_boundary_id", "")),
                    page_number=self._safe_int(first_cell.get("page_number"), default=0),
                    page_index=self._safe_int(first_cell.get("page_index"), default=0),
                    col_index=col_index,
                    header=header_text,
                    normalized_header=normalized_header,
                    semantic_type=semantic_type,
                    data_type=data_type,
                    sample_values=sample_values,
                    non_empty_count=non_empty_count,
                    numeric_ratio=round(numeric_ratio, 4),
                    empty_ratio=round(max(0.0, min(empty_ratio, 1.0)), 4),
                    confidence=confidence,
                    source="table_semantic_recognizer",
                    metadata={
                        "is_header_column": col_index in header_cols,
                        "numeric_count": numeric_count,
                        "total_data_rows": total_data_rows,
                    },
                )
            )

        return columns

    def _build_row_semantics(
        self,
        matrix: List[List[Dict[str, Any]]],
        table_grid_id: str,
        header_rows: List[int],
        header_cols: List[int],
    ) -> List[TableRowSemantic]:
        rows: List[TableRowSemantic] = []
        first_cell = self._first_non_empty_cell(matrix)

        for row_index, row in enumerate(matrix):
            texts = [
                self._cell_text(cell)
                for cell in row
            ]

            non_empty_texts = [
                text for text in texts
                if text
            ]

            non_empty_count = len(non_empty_texts)
            empty_count = len(texts) - non_empty_count

            numeric_count = sum(
                1 for text in non_empty_texts
                if self._is_numeric_like(text)
            )

            row_text = " | ".join(non_empty_texts)

            row_type, confidence = self._infer_row_type(
                row_index=row_index,
                row_text=row_text,
                non_empty_count=non_empty_count,
                numeric_count=numeric_count,
                empty_count=empty_count,
                total_cell_count=len(texts),
                header_rows=header_rows,
            )

            rows.append(
                TableRowSemantic(
                    table_row_id=make_id("tbl_row_sem"),
                    table_grid_id=table_grid_id,
                    table_structure_id=first_cell.get("table_structure_id", ""),
                    table_boundary_id=first_cell.get("table_boundary_id", ""),
                    page_number=self._safe_int(first_cell.get("page_number"), default=0),
                    page_index=self._safe_int(first_cell.get("page_index"), default=0),
                    row_index=row_index,
                    row_type=row_type,
                    text=row_text,
                    non_empty_count=non_empty_count,
                    numeric_count=numeric_count,
                    empty_count=empty_count,
                    confidence=confidence,
                    source="table_semantic_recognizer",
                    metadata={
                        "header_rows": header_rows,
                        "header_cols": header_cols,
                    },
                )
            )

        return rows

    def _build_records(
        self,
        matrix: List[List[Dict[str, Any]]],
        columns: List[TableColumnSemantic],
        rows: List[TableRowSemantic],
        table_grid_id: str,
    ) -> List[TableRecord]:
        records: List[TableRecord] = []

        row_semantic_by_index = {
            row.row_index: row
            for row in rows
        }

        first_cell = self._first_non_empty_cell(matrix)

        column_keys = [
            self._make_record_key(
                column=column,
                fallback=f"col_{column.col_index}"
            )
            for column in columns
        ]

        for row_index, row in enumerate(matrix):
            if len(records) >= self.config.max_records_per_table:
                break

            row_semantic = row_semantic_by_index.get(row_index)

            if not row_semantic:
                continue

            if self.config.skip_header_rows_in_records and row_semantic.row_type == "header_row":
                continue

            if self.config.skip_empty_rows_in_records and row_semantic.row_type == "empty_row":
                continue

            if self.config.skip_total_rows_in_records and row_semantic.row_type in ["total_row", "subtotal_row"]:
                continue

            values: Dict[str, Any] = {}
            raw_values: Dict[str, Any] = {}
            cell_ids: Dict[str, str] = {}

            non_empty_count = 0

            for col_index, column in enumerate(columns):
                key = column_keys[col_index]

                if col_index >= len(row):
                    values[key] = None
                    raw_values[key] = ""
                    cell_ids[key] = ""
                    continue

                cell = row[col_index]
                raw_text = self._cell_text(cell)
                typed_value = self._cast_value(
                    raw_text,
                    column.data_type,
                )

                values[key] = typed_value
                raw_values[key] = raw_text
                cell_ids[key] = cell.get("table_cell_id", "")

                if raw_text:
                    non_empty_count += 1

            if non_empty_count == 0 and self.config.skip_empty_rows_in_records:
                continue

            confidence = self._score_record(
                row_semantic=row_semantic,
                non_empty_count=non_empty_count,
                col_count=len(columns),
            )

            records.append(
                TableRecord(
                    table_record_id=make_id("tbl_record"),
                    table_grid_id=table_grid_id,
                    table_structure_id=first_cell.get("table_structure_id", ""),
                    table_boundary_id=first_cell.get("table_boundary_id", ""),
                    page_number=self._safe_int(first_cell.get("page_number"), default=0),
                    page_index=self._safe_int(first_cell.get("page_index"), default=0),
                    row_index=row_index,
                    values=values,
                    raw_values=raw_values,
                    cell_ids=cell_ids,
                    record_type=row_semantic.row_type,
                    confidence=confidence,
                    source="table_semantic_recognizer",
                    metadata={
                        "non_empty_count": non_empty_count,
                        "column_count": len(columns),
                    },
                )
            )

        return records

    def _resolve_column_headers(
        self,
        matrix: List[List[Dict[str, Any]]],
        header: Dict[str, Any],
        structure: Dict[str, Any],
        header_rows: List[int],
        col_count: int,
    ) -> List[str]:
        if self.config.use_header_result:
            column_headers = header.get("column_headers", []) or []

            if column_headers:
                return self._pad_list(
                    values=[
                        self._clean_text(item) for item in column_headers
                    ],
                    target_len=col_count,
                )

        if self.config.use_structure_result:
            column_headers = structure.get("column_headers", []) or []

            if column_headers:
                return self._pad_list(
                    values=[
                        self._clean_text(item) for item in column_headers
                    ],
                    target_len=col_count,
                )

        headers: List[str] = []

        for col_index in range(col_count):
            parts: List[str] = []

            for row_index in header_rows:
                if row_index < 0 or row_index >= len(matrix):
                    continue

                if col_index >= len(matrix[row_index]):
                    continue

                text = self._cell_text(matrix[row_index][col_index])

                if text:
                    parts.append(text)

            headers.append(" | ".join(parts).strip())

        return headers

    def _resolve_header_rows(
        self,
        cells: List[Dict[str, Any]],
        header: Dict[str, Any],
        structure: Dict[str, Any],
        row_count: int,
    ) -> List[int]:
        result: List[int] = []

        if self.config.use_header_result:
            for item in header.get("header_row_indices", []) or []:
                index = self._safe_int(item, default=-1)

                if 0 <= index < row_count:
                    result.append(index)

        if not result and self.config.use_structure_result:
            for item in structure.get("header_row_indices", []) or []:
                index = self._safe_int(item, default=-1)

                if 0 <= index < row_count:
                    result.append(index)

        if not result:
            for cell in cells:
                if bool(cell.get("is_header", False)) or cell.get("cell_role") == "header":
                    index = self._safe_int(cell.get("row_index"), default=-1)

                    if 0 <= index < row_count:
                        result.append(index)

        if not result and row_count > 1:
            first_row_cells = [
                cell for cell in cells
                if self._safe_int(cell.get("row_index"), default=-1) == 0
            ]

            if first_row_cells:
                numeric_count = sum(
                    1 for cell in first_row_cells
                    if self._is_numeric_like(self._cell_text(cell))
                )

                non_empty_count = sum(
                    1 for cell in first_row_cells
                    if self._cell_text(cell)
                )

                if non_empty_count > 0 and numeric_count <= max(1, non_empty_count // 3):
                    result.append(0)

        return sorted(list(dict.fromkeys(result)))

    def _resolve_header_cols(
        self,
        cells: List[Dict[str, Any]],
        header: Dict[str, Any],
        structure: Dict[str, Any],
        col_count: int,
    ) -> List[int]:
        result: List[int] = []

        if self.config.use_header_result:
            for item in header.get("header_col_indices", []) or []:
                index = self._safe_int(item, default=-1)

                if 0 <= index < col_count:
                    result.append(index)

        if not result and self.config.use_structure_result:
            for item in structure.get("stub_column_indices", []) or []:
                index = self._safe_int(item, default=-1)

                if 0 <= index < col_count:
                    result.append(index)

        if not result:
            for cell in cells:
                if bool(cell.get("is_stub", False)) or cell.get("cell_role") == "stub":
                    index = self._safe_int(cell.get("col_index"), default=-1)

                    if 0 <= index < col_count:
                        result.append(index)

        return sorted(list(dict.fromkeys(result)))

    def _infer_column_semantic_type(
        self,
        header: str,
        values: List[str],
        col_index: int,
        header_cols: List[int],
    ) -> str:
        normalized = self._normalize_match_text(header)

        if col_index in header_cols:
            return "row_label_column"

        keyword_map = [
            ("index_column", ["stt", "tt", "so thu tu", "no", "number", "index"]),
            ("code_column", ["ma", "ma so", "code", "id", "identifier", "ky hieu"]),
            ("name_column", ["ten", "ho ten", "name", "title"]),
            ("description_column", ["noi dung", "mo ta", "dien giai", "description", "content"]),
            ("date_column", ["ngay", "thang", "nam", "date", "time", "period"]),
            ("organization_column", ["don vi", "co quan", "to chuc", "department", "agency", "organization"]),
            ("quantity_column", ["so luong", "sl", "quantity", "count"]),
            ("money_column", ["gia tri", "so tien", "kinh phi", "chi phi", "vnd", "vnđ", "amount", "cost", "price"]),
            ("percentage_column", ["ty le", "ti le", "%", "percentage", "rate"]),
            ("status_column", ["trang thai", "tinh trang", "status"]),
            ("note_column", ["ghi chu", "note", "remark"]),
            ("result_column", ["ket qua", "result", "outcome"]),
        ]

        for semantic_type, keywords in keyword_map:
            for keyword in keywords:
                if keyword in normalized:
                    return semantic_type

        if values:
            if self._values_look_like_dates(values):
                return "date_column"

            if self._values_look_like_percentages(values):
                return "percentage_column"

            if self._values_look_like_money(values):
                return "money_column"

            if self._values_look_like_codes(values):
                return "code_column"

            numeric_ratio = sum(
                1 for value in values
                if self._is_numeric_like(value)
            ) / max(len(values), 1)

            if numeric_ratio >= 0.70:
                return "numeric_column"

        if col_index == 0:
            return "row_label_column"

        return "text_column"

    def _infer_column_data_type(
        self,
        header: str,
        values: List[str],
        numeric_ratio: float,
    ) -> str:
        normalized = self._normalize_match_text(header)

        if "ngay" in normalized or "date" in normalized:
            return "date"

        if "%" in header or "ty le" in normalized or "ti le" in normalized or "percentage" in normalized:
            return "percentage"

        if "vnd" in normalized or "vnđ" in normalized or "tien" in normalized or "amount" in normalized or "cost" in normalized:
            return "money"

        if values:
            if self._values_look_like_dates(values):
                return "date"

            if self._values_look_like_percentages(values):
                return "percentage"

            if self._values_look_like_money(values):
                return "money"

        if numeric_ratio >= 0.70:
            return "number"

        return "text"

    def _infer_row_type(
        self,
        row_index: int,
        row_text: str,
        non_empty_count: int,
        numeric_count: int,
        empty_count: int,
        total_cell_count: int,
        header_rows: List[int],
    ) -> Tuple[str, float]:
        if row_index in header_rows:
            return "header_row", 0.90

        if non_empty_count == 0:
            return "empty_row", 0.95

        normalized = self._normalize_match_text(row_text)

        total_keywords = [
            "tong",
            "tong cong",
            "cong",
            "total",
            "grand total",
        ]

        subtotal_keywords = [
            "cong muc",
            "subtotal",
            "tieu tong",
            "nhom",
        ]

        if any(keyword in normalized for keyword in total_keywords):
            return "total_row", 0.80

        if any(keyword in normalized for keyword in subtotal_keywords):
            return "subtotal_row", 0.75

        if non_empty_count == 1 and numeric_count == 0:
            return "group_row", 0.60

        numeric_ratio = numeric_count / max(non_empty_count, 1)

        if numeric_ratio >= 0.60:
            return "numeric_data_row", 0.70

        return "data_row", 0.65

    def _classify_table_semantic_type(
        self,
        columns: List[TableColumnSemantic],
        rows: List[TableRowSemantic],
        table_type: str,
    ) -> str:
        semantic_types = [
            column.semantic_type for column in columns
        ]

        if "money_column" in semantic_types:
            return "financial_table"

        if "date_column" in semantic_types and (
            "status_column" in semantic_types or "result_column" in semantic_types
        ):
            return "tracking_table"

        if "quantity_column" in semantic_types and "money_column" in semantic_types:
            return "cost_quantity_table"

        if "percentage_column" in semantic_types:
            return "metric_table"

        if "code_column" in semantic_types and "name_column" in semantic_types:
            return "master_data_table"

        if table_type == "key_value_table":
            return "key_value_table"

        if len(columns) <= 2:
            return "simple_table"

        return "general_data_table"

    def _resolve_table_type(
        self,
        structure: Dict[str, Any],
        header: Dict[str, Any],
        columns: List[TableColumnSemantic],
        rows: List[TableRowSemantic],
    ) -> str:
        if structure.get("table_type"):
            return structure.get("table_type")

        if header.get("header_type"):
            if "row_header" in header.get("header_type", ""):
                return "matrix_table"

        numeric_columns = [
            column for column in columns
            if column.data_type in ["number", "money", "percentage"]
        ]

        if numeric_columns:
            return "data_table"

        if len(columns) == 2:
            return "key_value_table"

        return "plain_table"

    def _score_column_semantic(
        self,
        header: str,
        values: List[str],
        semantic_type: str,
        data_type: str,
        numeric_ratio: float,
    ) -> float:
        score = 0.30

        if header:
            score += 0.20

        if values:
            score += 0.15

        if semantic_type != "unknown_column":
            score += 0.15

        if data_type != "text":
            score += 0.10

        if numeric_ratio >= 0.70:
            score += 0.05

        return round(max(0.0, min(score, 0.95)), 4)

    def _score_table_semantic(
        self,
        columns: List[TableColumnSemantic],
        rows: List[TableRowSemantic],
        records: List[TableRecord],
        header_rows: List[int],
        structure: Dict[str, Any],
        header: Dict[str, Any],
    ) -> float:
        score = 0.35

        if columns:
            score += 0.15

        if rows:
            score += 0.10

        if records:
            score += 0.15

        if header_rows:
            score += 0.10

        if structure.get("confidence"):
            score += min(0.08, self._safe_float(structure.get("confidence"), default=0.0) * 0.08)

        if header.get("confidence"):
            score += min(0.07, self._safe_float(header.get("confidence"), default=0.0) * 0.07)

        known_columns = [
            column for column in columns
            if column.semantic_type != "unknown_column"
        ]

        if columns:
            score += min(0.10, (len(known_columns) / len(columns)) * 0.10)

        return round(max(0.0, min(score, 0.95)), 4)

    def _score_record(
        self,
        row_semantic: TableRowSemantic,
        non_empty_count: int,
        col_count: int,
    ) -> float:
        score = 0.40

        if non_empty_count > 0:
            score += 0.20

        coverage = non_empty_count / max(col_count, 1)
        score += min(0.20, coverage * 0.20)

        score += min(0.10, row_semantic.confidence * 0.10)

        return round(max(0.0, min(score, 0.95)), 4)

    def _cast_value(
        self,
        value: str,
        data_type: str,
    ) -> Any:
        value = self._clean_text(value)

        if value == "":
            return None

        if data_type in ["number", "money", "percentage"]:
            number = self._parse_number(value)

            if number is not None:
                if data_type == "percentage":
                    return {
                        "value": number,
                        "unit": "%",
                    }

                if data_type == "money":
                    return {
                        "value": number,
                        "unit": self._infer_money_unit(value),
                    }

                return number

        return value

    def _parse_number(
        self,
        value: str,
    ) -> Optional[float]:
        if not value:
            return None

        text = value.strip()
        text = text.replace("%", "")
        text = text.replace("VNĐ", "")
        text = text.replace("VND", "")
        text = text.replace("vnđ", "")
        text = text.replace("vnd", "")
        text = text.replace("đ", "")
        text = text.replace(" ", "")

        if "," in text and "." in text:
            text = text.replace(".", "")
            text = text.replace(",", ".")
        elif "," in text:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")

        text = re.sub(r"[^0-9\.\-]", "", text)

        if not text:
            return None

        try:
            return float(text)
        except Exception:
            return None

    def _infer_money_unit(
        self,
        value: str,
    ) -> str:
        lower = value.lower()

        if "vnđ" in lower or "vnd" in lower or "đ" in lower:
            return "VND"

        if "$" in lower or "usd" in lower:
            return "USD"

        return ""

    def _make_record_key(
        self,
        column: TableColumnSemantic,
        fallback: str,
    ) -> str:
        header = column.normalized_header or column.header

        if not header:
            header = column.semantic_type or fallback

        key = self._normalize_match_text(header)
        key = re.sub(r"[^a-z0-9_]+", "_", key)
        key = re.sub(r"_+", "_", key)
        key = key.strip("_")

        return key or fallback

    def _build_cell_matrix(
        self,
        cells: List[Dict[str, Any]],
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

        for cell in cells:
            row_index = self._safe_int(cell.get("row_index"), default=-1)
            col_index = self._safe_int(cell.get("col_index"), default=-1)

            if row_index < 0 or row_index >= row_count:
                continue

            if col_index < 0 or col_index >= col_count:
                continue

            matrix[row_index][col_index] = dict(cell)

        return matrix

    def _resolve_table_shape(
        self,
        cells: List[Dict[str, Any]],
        structure: Dict[str, Any],
    ) -> Tuple[int, int]:
        row_count = self._safe_int(structure.get("row_count"), default=0)
        col_count = self._safe_int(structure.get("col_count"), default=0)

        if row_count <= 0:
            row_count = max(
                [
                    self._safe_int(cell.get("row_index"), default=-1)
                    for cell in cells
                ],
                default=-1,
            ) + 1

        if col_count <= 0:
            col_count = max(
                [
                    self._safe_int(cell.get("col_index"), default=-1)
                    for cell in cells
                ],
                default=-1,
            ) + 1

        return max(row_count, 0), max(col_count, 0)

    def _resolve_table_bbox(
        self,
        cells: List[Dict[str, Any]],
        structure: Dict[str, Any],
    ) -> List[float]:
        bbox = structure.get("bbox", [])

        if bbox and len(bbox) == 4:
            return [
                round(float(item), 4) for item in bbox
            ]

        bboxes = [
            cell.get("bbox", [])
            for cell in cells
            if cell.get("bbox") and len(cell.get("bbox", [])) == 4
        ]

        if not bboxes:
            return []

        x0 = min(float(bbox[0]) for bbox in bboxes)
        y0 = min(float(bbox[1]) for bbox in bboxes)
        x1 = max(float(bbox[2]) for bbox in bboxes)
        y1 = max(float(bbox[3]) for bbox in bboxes)

        return [
            round(x0, 4),
            round(y0, 4),
            round(x1, 4),
            round(y1, 4),
        ]

    def _resolve_caption(
        self,
        cells: List[Dict[str, Any]],
        structure: Dict[str, Any],
    ) -> str:
        metadata = structure.get("metadata", {}) or {}

        for key in ["caption", "table_caption", "nearby_caption"]:
            value = metadata.get(key)

            if isinstance(value, str) and value.strip():
                return self._clean_text(value)

        return ""

    def _resolve_title(
        self,
        header: Dict[str, Any],
        structure: Dict[str, Any],
        caption: str,
    ) -> str:
        if caption:
            return caption

        metadata = structure.get("metadata", {}) or {}

        for key in ["title", "table_title"]:
            value = metadata.get(key)

            if isinstance(value, str) and value.strip():
                return self._clean_text(value)

        column_headers = header.get("column_headers", []) or structure.get("column_headers", []) or []

        non_empty_headers = [
            self._clean_text(item)
            for item in column_headers
            if self._clean_text(item)
        ]

        if non_empty_headers:
            return " | ".join(non_empty_headers[:3])

        return ""

    def _build_text_preview(
        self,
        matrix: List[List[Dict[str, Any]]],
    ) -> str:
        lines: List[str] = []

        for row in matrix:
            parts = [
                self._cell_text(cell)
                for cell in row
                if self._cell_text(cell)
            ]

            if parts:
                lines.append(" | ".join(parts))

            if len("\n".join(lines)) >= self.config.text_preview_chars:
                break

        return "\n".join(lines)[: self.config.text_preview_chars]

    def _first_non_empty_cell(
        self,
        matrix: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        for row in matrix:
            for cell in row:
                if cell.get("table_cell_id"):
                    return cell

        return {}

    def _cell_text(
        self,
        cell: Dict[str, Any],
    ) -> str:
        text = cell.get("normalized_text") or cell.get("text") or ""

        return self._clean_text(text)

    def _values_look_like_dates(
        self,
        values: List[str],
    ) -> bool:
        if not values:
            return False

        hits = 0

        for value in values[:30]:
            text = value.strip()

            if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text):
                hits += 1
                continue

            if re.search(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b", text):
                hits += 1
                continue

            if re.search(r"\b(tháng|thang)\s+\d{1,2}\b", text.lower()):
                hits += 1

        return hits / max(min(len(values), 30), 1) >= 0.50

    def _values_look_like_percentages(
        self,
        values: List[str],
    ) -> bool:
        if not values:
            return False

        hits = sum(
            1 for value in values[:30]
            if "%" in value or re.search(r"\b\d+([,.]\d+)?\s*%\b", value)
        )

        return hits / max(min(len(values), 30), 1) >= 0.40

    def _values_look_like_money(
        self,
        values: List[str],
    ) -> bool:
        if not values:
            return False

        hits = 0

        for value in values[:30]:
            lower = value.lower()

            if "vnđ" in lower or "vnd" in lower or "đ" in lower or "$" in lower or "usd" in lower:
                hits += 1
                continue

            if self._is_numeric_like(value) and len(re.sub(r"\D", "", value)) >= 5:
                hits += 1

        return hits / max(min(len(values), 30), 1) >= 0.40

    def _values_look_like_codes(
        self,
        values: List[str],
    ) -> bool:
        if not values:
            return False

        hits = 0

        for value in values[:30]:
            text = value.strip()

            if re.fullmatch(r"[A-Za-z0-9_\-./]+", text) and any(ch.isdigit() for ch in text):
                hits += 1

        return hits / max(min(len(values), 30), 1) >= 0.50

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
        clean = clean.replace("vnđ", "")
        clean = clean.replace("vnd", "")
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

    def _collect_cells(
        self,
        page_raws: List[PageRaw],
        table_cell_result: Optional[Dict[str, Any]] = None,
        table_span_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.config.use_span_cells_if_available and table_span_result:
            cells = table_span_result.get("table_cells_with_spans", [])

            if cells:
                return cells

        if table_cell_result:
            cells = table_cell_result.get("table_cells", [])

            if cells:
                return cells

        collected_span_cells: List[Dict[str, Any]] = []

        if self.config.use_span_cells_if_available:
            for page_raw in page_raws:
                meta = page_raw.metadata.get("table_span_detector", {})
                page_cells = meta.get("table_cells_with_spans_on_page", [])

                for cell in page_cells:
                    collected_span_cells.append(cell)

        if collected_span_cells:
            return collected_span_cells

        collected_cells: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_cell_extractor", {})
            page_cells = meta.get("table_cells_on_page", [])

            for cell in page_cells:
                collected_cells.append(cell)

        return collected_cells

    def _collect_headers(
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

    def _collect_structures(
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
            meta = page_raw.metadata.get("table_structure_recognizer", {})
            page_structures = meta.get("table_structures_on_page", [])

            for structure in page_structures:
                collected.append(structure)

        return collected

    def _group_cells_by_table(
        self,
        cells: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for cell in cells:
            table_key = cell.get("table_grid_id", "") or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(cell)

        return grouped

    def _group_table_semantics_by_page(
        self,
        table_semantics: List[TableSemantic],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in table_semantics:
            page_key = str(item.page_number)
            grouped.setdefault(page_key, [])
            grouped[page_key].append(item.to_dict())

        return grouped

    def _group_table_semantics_by_table(
        self,
        table_semantics: List[TableSemantic],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in table_semantics:
            table_key = item.table_grid_id or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(item.to_dict())

        return grouped

    def _group_columns_by_table(
        self,
        columns: List[TableColumnSemantic],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for column in columns:
            table_key = column.table_grid_id or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(column.to_dict())

        return grouped

    def _group_rows_by_table(
        self,
        rows: List[TableRowSemantic],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for row in rows:
            table_key = row.table_grid_id or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(row.to_dict())

        return grouped

    def _group_records_by_table(
        self,
        records: List[TableRecord],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for record in records:
            table_key = record.table_grid_id or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(record.to_dict())

        return grouped

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        table_semantics: List[TableSemantic],
        columns: List[TableColumnSemantic],
        rows: List[TableRowSemantic],
        records: List[TableRecord],
        result: Dict[str, Any],
    ) -> None:
        semantics_by_page: Dict[int, List[Dict[str, Any]]] = {}
        columns_by_page: Dict[int, List[Dict[str, Any]]] = {}
        rows_by_page: Dict[int, List[Dict[str, Any]]] = {}
        records_by_page: Dict[int, List[Dict[str, Any]]] = {}

        for item in table_semantics:
            semantics_by_page.setdefault(item.page_number, [])
            semantics_by_page[item.page_number].append(item.to_dict())

        for item in columns:
            columns_by_page.setdefault(item.page_number, [])
            columns_by_page[item.page_number].append(item.to_dict())

        for item in rows:
            rows_by_page.setdefault(item.page_number, [])
            rows_by_page[item.page_number].append(item.to_dict())

        for item in records:
            records_by_page.setdefault(item.page_number, [])
            records_by_page[item.page_number].append(item.to_dict())

        for page_raw in page_raws:
            page_raw.metadata.setdefault("table_semantic_recognizer", {})
            page_raw.metadata["table_semantic_recognizer"] = {
                "processor": "TableSemanticRecognizer",
                "table_semantics_on_page": semantics_by_page.get(page_raw.page_number, []),
                "table_columns_on_page": columns_by_page.get(page_raw.page_number, []),
                "table_rows_on_page": rows_by_page.get(page_raw.page_number, []),
                "table_records_on_page": records_by_page.get(page_raw.page_number, []),
                "table_semantic_count_on_page": len(semantics_by_page.get(page_raw.page_number, [])),
                "table_column_count_on_page": len(columns_by_page.get(page_raw.page_number, [])),
                "table_row_count_on_page": len(rows_by_page.get(page_raw.page_number, [])),
                "table_record_count_on_page": len(records_by_page.get(page_raw.page_number, [])),
                "table_semantic_summary": result.get("table_semantic_summary", {}),
            }

    def _build_summary(
        self,
        table_semantics: List[TableSemantic],
        columns: List[TableColumnSemantic],
        rows: List[TableRowSemantic],
        records: List[TableRecord],
    ) -> Dict[str, Any]:
        by_page: Dict[str, int] = {}
        by_semantic_type: Dict[str, int] = {}
        by_table_type: Dict[str, int] = {}
        by_column_type: Dict[str, int] = {}
        by_row_type: Dict[str, int] = {}

        for table in table_semantics:
            page_key = str(table.page_number)
            by_page[page_key] = by_page.get(page_key, 0) + 1
            by_semantic_type[table.semantic_type] = by_semantic_type.get(table.semantic_type, 0) + 1
            by_table_type[table.table_type] = by_table_type.get(table.table_type, 0) + 1

        for column in columns:
            by_column_type[column.semantic_type] = by_column_type.get(column.semantic_type, 0) + 1

        for row in rows:
            by_row_type[row.row_type] = by_row_type.get(row.row_type, 0) + 1

        return {
            "has_table_semantics": len(table_semantics) > 0,
            "table_semantic_count": len(table_semantics),
            "table_column_count": len(columns),
            "table_row_count": len(rows),
            "table_record_count": len(records),
            "page_count_with_semantics": len(by_page),
            "by_page": by_page,
            "by_semantic_type": by_semantic_type,
            "by_table_type": by_table_type,
            "by_column_type": by_column_type,
            "by_row_type": by_row_type,
        }

    def _sort_table_semantics(
        self,
        items: List[TableSemantic],
    ) -> List[TableSemantic]:
        return sorted(
            items,
            key=lambda item: (
                item.page_number,
                item.bbox[1] if item.bbox else 999999,
                item.bbox[0] if item.bbox else 999999,
            ),
        )

    def _sort_columns(
        self,
        items: List[TableColumnSemantic],
    ) -> List[TableColumnSemantic]:
        return sorted(
            items,
            key=lambda item: (
                item.page_number,
                item.table_grid_id,
                item.col_index,
            ),
        )

    def _sort_rows(
        self,
        items: List[TableRowSemantic],
    ) -> List[TableRowSemantic]:
        return sorted(
            items,
            key=lambda item: (
                item.page_number,
                item.table_grid_id,
                item.row_index,
            ),
        )

    def _sort_records(
        self,
        items: List[TableRecord],
    ) -> List[TableRecord]:
        return sorted(
            items,
            key=lambda item: (
                item.page_number,
                item.table_grid_id,
                item.row_index,
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

    def _pad_list(
        self,
        values: List[str],
        target_len: int,
    ) -> List[str]:
        result = list(values)

        while len(result) < target_len:
            result.append("")

        return result[:target_len]

    def _normalize_header(
        self,
        text: Any,
    ) -> str:
        text = self._clean_text(text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" :;,.|-")

        return text.strip()

    def _normalize_match_text(
        self,
        text: Any,
    ) -> str:
        text = self._clean_text(text).lower()
        replacements = {
            "à": "a", "á": "a", "ạ": "a", "ả": "a", "ã": "a",
            "â": "a", "ầ": "a", "ấ": "a", "ậ": "a", "ẩ": "a", "ẫ": "a",
            "ă": "a", "ằ": "a", "ắ": "a", "ặ": "a", "ẳ": "a", "ẵ": "a",
            "è": "e", "é": "e", "ẹ": "e", "ẻ": "e", "ẽ": "e",
            "ê": "e", "ề": "e", "ế": "e", "ệ": "e", "ể": "e", "ễ": "e",
            "ì": "i", "í": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
            "ò": "o", "ó": "o", "ọ": "o", "ỏ": "o", "õ": "o",
            "ô": "o", "ồ": "o", "ố": "o", "ộ": "o", "ổ": "o", "ỗ": "o",
            "ơ": "o", "ờ": "o", "ớ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
            "ù": "u", "ú": "u", "ụ": "u", "ủ": "u", "ũ": "u",
            "ư": "u", "ừ": "u", "ứ": "u", "ự": "u", "ử": "u", "ữ": "u",
            "ỳ": "y", "ý": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y",
            "đ": "d",
        }

        for src, dst in replacements.items():
            text = text.replace(src, dst)

        text = re.sub(r"\s+", " ", text)

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


def recognize_table_semantics(
    page_raws: List[PageRaw],
    table_cell_result: Optional[Dict[str, Any]] = None,
    table_header_result: Optional[Dict[str, Any]] = None,
    table_span_result: Optional[Dict[str, Any]] = None,
    table_structure_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    recognizer = TableSemanticRecognizer()
    return recognizer.process(
        page_raws=page_raws,
        table_cell_result=table_cell_result,
        table_header_result=table_header_result,
        table_span_result=table_span_result,
        table_structure_result=table_structure_result,
    )
