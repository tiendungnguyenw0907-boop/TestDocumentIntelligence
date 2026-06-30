"""
table_structure_recognizer.py

Production V1 - Colab Ready

Purpose
-------
Recognize logical structure of tables from table grids.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- TableBoundaryDetector
- TableGridBuilder

Output
------
Dictionary with:
- table_structures
- table_structures_by_page
- table_structure_summary

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
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class TableStructureRecognizerConfig:
    min_header_confidence: float = 0.45
    max_header_rows: int = 3

    detect_header_rows: bool = True
    detect_stub_columns: bool = True
    detect_numeric_columns: bool = True
    detect_empty_rows: bool = True

    attach_to_pages: bool = True
    include_debug: bool = True

    text_preview_chars: int = 500


@dataclass
class TableStructure:
    table_structure_id: str
    table_grid_id: str
    table_boundary_id: str

    page_number: int
    page_index: int
    bbox: List[float]

    row_count: int
    col_count: int

    has_header: bool
    header_row_indices: List[int]
    body_row_indices: List[int]
    empty_row_indices: List[int]

    column_headers: List[str]
    stub_column_indices: List[int]
    numeric_column_indices: List[int]

    table_type: str
    confidence: float

    source: str = "table_grid_builder"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class TableStructureRecognizer:
    def __init__(
        self,
        config: Optional[TableStructureRecognizerConfig] = None,
    ):
        self.config = config or TableStructureRecognizerConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        table_grid_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        grids = self._collect_table_grids(
            page_raws=page_raws,
            table_grid_result=table_grid_result,
        )

        structures: List[TableStructure] = []

        for page_raw in page_raws:
            page_grids = [
                grid for grid in grids
                if self._safe_int(grid.get("page_number"), default=-1) == page_raw.page_number
            ]

            for grid in page_grids:
                structure = self.recognize_grid_structure(
                    page_raw=page_raw,
                    grid=grid,
                )

                if structure:
                    structures.append(structure)

        structures = self._sort_structures(structures)
        structures_by_page = self._group_by_page(structures)

        result = {
            "processor": "TableStructureRecognizer",
            "table_structures": [
                structure.to_dict() for structure in structures
            ],
            "table_structures_by_page": structures_by_page,
            "table_structure_summary": self._build_summary(structures),
            "config": {
                "detect_header_rows": self.config.detect_header_rows,
                "detect_stub_columns": self.config.detect_stub_columns,
                "detect_numeric_columns": self.config.detect_numeric_columns,
                "detect_empty_rows": self.config.detect_empty_rows,
                "max_header_rows": self.config.max_header_rows,
                "min_header_confidence": self.config.min_header_confidence,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                structures=structures,
                result=result,
            )

        return result

    def recognize_grid_structure(
        self,
        page_raw: PageRaw,
        grid: Dict[str, Any],
    ) -> Optional[TableStructure]:
        row_count = self._safe_int(grid.get("row_count"), default=0)
        col_count = self._safe_int(grid.get("col_count"), default=0)

        if row_count <= 0 or col_count <= 0:
            return None

        cells = grid.get("cells", []) or []
        matrix = self._build_cell_matrix(
            cells=cells,
            row_count=row_count,
            col_count=col_count,
        )

        row_profiles = self._build_row_profiles(matrix)
        col_profiles = self._build_col_profiles(matrix)

        header_row_indices: List[int] = []

        if self.config.detect_header_rows:
            header_row_indices = self._detect_header_rows(
                matrix=matrix,
                row_profiles=row_profiles,
                col_count=col_count,
            )

        empty_row_indices: List[int] = []

        if self.config.detect_empty_rows:
            empty_row_indices = self._detect_empty_rows(row_profiles)

        body_row_indices = [
            row_index
            for row_index in range(row_count)
            if row_index not in header_row_indices
            and row_index not in empty_row_indices
        ]

        column_headers = self._build_column_headers(
            matrix=matrix,
            header_row_indices=header_row_indices,
            col_count=col_count,
        )

        stub_column_indices: List[int] = []

        if self.config.detect_stub_columns:
            stub_column_indices = self._detect_stub_columns(
                col_profiles=col_profiles,
                header_row_indices=header_row_indices,
            )

        numeric_column_indices: List[int] = []

        if self.config.detect_numeric_columns:
            numeric_column_indices = self._detect_numeric_columns(
                col_profiles=col_profiles,
                stub_column_indices=stub_column_indices,
            )

        table_type = self._classify_table_type(
            row_count=row_count,
            col_count=col_count,
            header_row_indices=header_row_indices,
            stub_column_indices=stub_column_indices,
            numeric_column_indices=numeric_column_indices,
            empty_row_indices=empty_row_indices,
        )

        confidence = self._score_structure(
            row_count=row_count,
            col_count=col_count,
            header_row_indices=header_row_indices,
            column_headers=column_headers,
            numeric_column_indices=numeric_column_indices,
            grid_confidence=grid.get("confidence", 0.5),
        )

        return TableStructure(
            table_structure_id=make_id("tbl_struct"),
            table_grid_id=grid.get("table_grid_id", ""),
            table_boundary_id=grid.get("table_boundary_id", ""),
            page_number=page_raw.page_number,
            page_index=page_raw.page_index,
            bbox=grid.get("bbox", []),
            row_count=row_count,
            col_count=col_count,
            has_header=len(header_row_indices) > 0,
            header_row_indices=header_row_indices,
            body_row_indices=body_row_indices,
            empty_row_indices=empty_row_indices,
            column_headers=column_headers,
            stub_column_indices=stub_column_indices,
            numeric_column_indices=numeric_column_indices,
            table_type=table_type,
            confidence=confidence,
            source="table_grid_builder.table_grids",
            metadata={
                "row_profiles": row_profiles,
                "col_profiles": col_profiles,
                "grid_method": grid.get("grid_method", ""),
                "grid_confidence": grid.get("confidence", 0.5),
                "text_preview": self._build_text_preview(matrix),
            },
        )

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
                        "row_index": row_index,
                        "col_index": col_index,
                        "text": "",
                        "cell_id": "",
                        "bbox": None,
                        "metadata": {},
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

            matrix[row_index][col_index] = {
                "row_index": row_index,
                "col_index": col_index,
                "text": self._clean_text(cell.get("text", "")),
                "cell_id": cell.get("cell_id", ""),
                "bbox": cell.get("bbox"),
                "metadata": cell.get("metadata", {}),
            }

        return matrix

    def _build_row_profiles(
        self,
        matrix: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        profiles: List[Dict[str, Any]] = []

        for row_index, row in enumerate(matrix):
            texts = [
                self._clean_text(cell.get("text", ""))
                for cell in row
            ]

            non_empty_texts = [
                text for text in texts
                if text
            ]

            numeric_count = sum(
                1 for text in non_empty_texts
                if self._is_numeric_like(text)
            )

            text_count = len(non_empty_texts)
            empty_count = len(texts) - text_count

            numeric_ratio = numeric_count / max(text_count, 1)
            empty_ratio = empty_count / max(len(texts), 1)

            avg_text_length = 0.0

            if non_empty_texts:
                avg_text_length = sum(len(text) for text in non_empty_texts) / len(non_empty_texts)

            profiles.append(
                {
                    "row_index": row_index,
                    "cell_count": len(texts),
                    "non_empty_count": text_count,
                    "empty_count": empty_count,
                    "numeric_count": numeric_count,
                    "numeric_ratio": round(numeric_ratio, 4),
                    "empty_ratio": round(empty_ratio, 4),
                    "avg_text_length": round(avg_text_length, 4),
                    "texts": non_empty_texts[:10],
                    "looks_like_header": False,
                }
            )

        return profiles

    def _build_col_profiles(
        self,
        matrix: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        if not matrix:
            return []

        col_count = len(matrix[0])
        profiles: List[Dict[str, Any]] = []

        for col_index in range(col_count):
            texts = []

            for row in matrix:
                if col_index < len(row):
                    text = self._clean_text(row[col_index].get("text", ""))
                    texts.append(text)

            non_empty_texts = [
                text for text in texts
                if text
            ]

            numeric_count = sum(
                1 for text in non_empty_texts
                if self._is_numeric_like(text)
            )

            text_count = len(non_empty_texts)
            empty_count = len(texts) - text_count

            numeric_ratio = numeric_count / max(text_count, 1)
            empty_ratio = empty_count / max(len(texts), 1)

            avg_text_length = 0.0

            if non_empty_texts:
                avg_text_length = sum(len(text) for text in non_empty_texts) / len(non_empty_texts)

            profiles.append(
                {
                    "col_index": col_index,
                    "cell_count": len(texts),
                    "non_empty_count": text_count,
                    "empty_count": empty_count,
                    "numeric_count": numeric_count,
                    "numeric_ratio": round(numeric_ratio, 4),
                    "empty_ratio": round(empty_ratio, 4),
                    "avg_text_length": round(avg_text_length, 4),
                    "texts": non_empty_texts[:10],
                }
            )

        return profiles

    def _detect_header_rows(
        self,
        matrix: List[List[Dict[str, Any]]],
        row_profiles: List[Dict[str, Any]],
        col_count: int,
    ) -> List[int]:
        header_rows: List[int] = []

        max_rows = min(
            self.config.max_header_rows,
            len(row_profiles),
        )

        for row_index in range(max_rows):
            profile = row_profiles[row_index]
            score = self._score_header_row(
                profile=profile,
                row_index=row_index,
                col_count=col_count,
            )

            profile["header_score"] = round(score, 4)
            profile["looks_like_header"] = score >= self.config.min_header_confidence

            if score >= self.config.min_header_confidence:
                header_rows.append(row_index)
            else:
                if row_index == 0:
                    continue

                break

        if not header_rows and row_profiles:
            first_row = row_profiles[0]

            if first_row.get("non_empty_count", 0) >= max(1, col_count // 2):
                if first_row.get("numeric_ratio", 0) <= 0.30:
                    header_rows = [0]

        return header_rows

    def _score_header_row(
        self,
        profile: Dict[str, Any],
        row_index: int,
        col_count: int,
    ) -> float:
        score = 0.20

        non_empty_count = profile.get("non_empty_count", 0)
        numeric_ratio = profile.get("numeric_ratio", 0.0)
        empty_ratio = profile.get("empty_ratio", 1.0)
        avg_text_length = profile.get("avg_text_length", 0.0)
        texts = profile.get("texts", [])

        if row_index == 0:
            score += 0.15

        if non_empty_count >= max(1, col_count // 2):
            score += 0.15

        if numeric_ratio <= 0.25:
            score += 0.20

        if empty_ratio <= 0.50:
            score += 0.10

        if 1 <= avg_text_length <= 80:
            score += 0.08

        header_keyword_hits = 0

        for text in texts:
            if self._has_header_keyword(text):
                header_keyword_hits += 1

        if header_keyword_hits > 0:
            score += min(0.20, header_keyword_hits * 0.06)

        return round(max(0.0, min(score, 0.95)), 4)

    def _detect_empty_rows(
        self,
        row_profiles: List[Dict[str, Any]],
    ) -> List[int]:
        empty_rows: List[int] = []

        for profile in row_profiles:
            if profile.get("non_empty_count", 0) == 0:
                empty_rows.append(profile["row_index"])
                continue

            if profile.get("empty_ratio", 0.0) >= 0.90:
                empty_rows.append(profile["row_index"])

        return empty_rows

    def _build_column_headers(
        self,
        matrix: List[List[Dict[str, Any]]],
        header_row_indices: List[int],
        col_count: int,
    ) -> List[str]:
        headers: List[str] = []

        for col_index in range(col_count):
            parts: List[str] = []

            for row_index in header_row_indices:
                if row_index < 0 or row_index >= len(matrix):
                    continue

                if col_index >= len(matrix[row_index]):
                    continue

                text = self._clean_text(matrix[row_index][col_index].get("text", ""))

                if text:
                    parts.append(text)

            headers.append(
                " | ".join(parts).strip()
            )

        return headers

    def _detect_stub_columns(
        self,
        col_profiles: List[Dict[str, Any]],
        header_row_indices: List[int],
    ) -> List[int]:
        stub_columns: List[int] = []

        if not col_profiles:
            return stub_columns

        first_col = col_profiles[0]

        if first_col.get("non_empty_count", 0) > 0:
            if first_col.get("numeric_ratio", 1.0) <= 0.35:
                stub_columns.append(0)

        for profile in col_profiles[1:]:
            col_index = profile.get("col_index", 0)

            if col_index > 1:
                break

            if profile.get("numeric_ratio", 1.0) <= 0.20:
                if profile.get("non_empty_count", 0) >= 2:
                    stub_columns.append(col_index)

        return list(dict.fromkeys(stub_columns))

    def _detect_numeric_columns(
        self,
        col_profiles: List[Dict[str, Any]],
        stub_column_indices: List[int],
    ) -> List[int]:
        numeric_columns: List[int] = []

        for profile in col_profiles:
            col_index = profile.get("col_index", 0)

            if col_index in stub_column_indices:
                continue

            if profile.get("non_empty_count", 0) <= 0:
                continue

            if profile.get("numeric_ratio", 0.0) >= 0.50:
                numeric_columns.append(col_index)

        return numeric_columns

    def _classify_table_type(
        self,
        row_count: int,
        col_count: int,
        header_row_indices: List[int],
        stub_column_indices: List[int],
        numeric_column_indices: List[int],
        empty_row_indices: List[int],
    ) -> str:
        if row_count <= 1 and col_count <= 1:
            return "single_cell_table"

        if header_row_indices and numeric_column_indices:
            return "data_table_with_header"

        if header_row_indices:
            return "text_table_with_header"

        if numeric_column_indices:
            return "numeric_data_table"

        if stub_column_indices and col_count == 2:
            return "key_value_table"

        if len(empty_row_indices) >= max(1, row_count // 2):
            return "sparse_table"

        return "plain_table"

    def _score_structure(
        self,
        row_count: int,
        col_count: int,
        header_row_indices: List[int],
        column_headers: List[str],
        numeric_column_indices: List[int],
        grid_confidence: Any,
    ) -> float:
        try:
            score = float(grid_confidence)
        except Exception:
            score = 0.50

        score = (score + 0.45) / 2.0

        if row_count >= 2:
            score += 0.08

        if col_count >= 2:
            score += 0.08

        if header_row_indices:
            score += 0.12

        if any(header.strip() for header in column_headers):
            score += 0.08

        if numeric_column_indices:
            score += 0.05

        return round(max(0.0, min(score, 0.95)), 4)

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
            "name",
            "value",
            "date",
            "total",
            "amount",
            "description",
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

    def _build_text_preview(
        self,
        matrix: List[List[Dict[str, Any]]],
    ) -> str:
        lines: List[str] = []

        for row in matrix:
            parts = []

            for cell in row:
                text = self._clean_text(cell.get("text", ""))

                if text:
                    parts.append(text)

            if parts:
                lines.append(" | ".join(parts))

        return "\n".join(lines)[: self.config.text_preview_chars]

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
            grid_meta = page_raw.metadata.get("table_grid_builder", {})
            page_grids = grid_meta.get("table_grids_on_page", [])

            for grid in page_grids:
                collected.append(grid)

        return collected

    def _group_by_page(
        self,
        structures: List[TableStructure],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for structure in structures:
            page_key = str(structure.page_number)
            grouped.setdefault(page_key, [])
            grouped[page_key].append(structure.to_dict())

        return grouped

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        structures: List[TableStructure],
        result: Dict[str, Any],
    ) -> None:
        by_page: Dict[int, List[Dict[str, Any]]] = {}

        for structure in structures:
            by_page.setdefault(structure.page_number, [])
            by_page[structure.page_number].append(structure.to_dict())

        for page_raw in page_raws:
            page_structures = by_page.get(page_raw.page_number, [])

            page_raw.metadata.setdefault("table_structure_recognizer", {})
            page_raw.metadata["table_structure_recognizer"] = {
                "processor": "TableStructureRecognizer",
                "table_structures_on_page": page_structures,
                "table_structure_count_on_page": len(page_structures),
                "table_structure_summary": result.get("table_structure_summary", {}),
            }

    def _build_summary(
        self,
        structures: List[TableStructure],
    ) -> Dict[str, Any]:
        by_page: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        header_table_count = 0

        for structure in structures:
            page_key = str(structure.page_number)
            by_page[page_key] = by_page.get(page_key, 0) + 1
            by_type[structure.table_type] = by_type.get(structure.table_type, 0) + 1

            if structure.has_header:
                header_table_count += 1

        return {
            "has_table_structures": len(structures) > 0,
            "table_structure_count": len(structures),
            "page_count_with_structures": len(by_page),
            "header_table_count": header_table_count,
            "by_page": by_page,
            "by_table_type": by_type,
        }

    def _sort_structures(
        self,
        structures: List[TableStructure],
    ) -> List[TableStructure]:
        return sorted(
            structures,
            key=lambda item: (
                item.page_number,
                item.bbox[1] if item.bbox else 999999,
                item.bbox[0] if item.bbox else 999999,
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
        text = re.sub(r"\s+", " ", text)

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


def recognize_table_structures(
    page_raws: List[PageRaw],
    table_grid_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    recognizer = TableStructureRecognizer()
    return recognizer.process(
        page_raws=page_raws,
        table_grid_result=table_grid_result,
    )
