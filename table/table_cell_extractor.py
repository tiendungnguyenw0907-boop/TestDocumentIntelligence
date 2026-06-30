"""
table_cell_extractor.py

Production V1 - Colab Ready

Purpose
-------
Extract, normalize, enrich and organize table cells from table grids
and recognized table structures.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- TableBoundaryDetector
- TableGridBuilder
- TableStructureRecognizer

Output
------
Dictionary with:
- table_cells
- table_cells_by_page
- table_cells_by_table
- table_matrices
- table_cell_summary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class TableCellExtractorConfig:
    use_grid_cells: bool = True
    fallback_from_grid_positions: bool = True
    enrich_text_from_page_lines: bool = True
    enrich_text_from_words: bool = True

    include_empty_cells: bool = True
    assign_cell_roles: bool = True

    center_inside_tolerance: float = 1.0
    min_text_overlap_ratio: float = 0.15

    min_cell_width: float = 5.0
    min_cell_height: float = 5.0

    text_preview_chars: int = 500
    max_cell_text_chars: int = 5000

    attach_to_pages: bool = True
    include_debug: bool = True


@dataclass
class ExtractedTableCell:
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

    text: str = ""
    normalized_text: str = ""

    cell_role: str = "body"
    column_header: str = ""

    is_header: bool = False
    is_stub: bool = False
    is_numeric: bool = False
    is_empty: bool = False

    confidence: float = 0.5
    source: str = "table_grid_builder"

    source_cell_id: str = ""
    text_line_ids: Optional[List[str]] = None
    word_ids: Optional[List[str]] = None

    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["text_line_ids"] is None:
            data["text_line_ids"] = []

        if data["word_ids"] is None:
            data["word_ids"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class TableCellExtractor:
    def __init__(
        self,
        config: Optional[TableCellExtractorConfig] = None,
    ):
        self.config = config or TableCellExtractorConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        table_grid_result: Optional[Dict[str, Any]] = None,
        table_structure_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        grids = self._collect_table_grids(
            page_raws=page_raws,
            table_grid_result=table_grid_result,
        )

        structures = self._collect_table_structures(
            page_raws=page_raws,
            table_structure_result=table_structure_result,
        )

        structure_by_grid_id = {
            structure.get("table_grid_id", ""): structure
            for structure in structures
            if structure.get("table_grid_id")
        }

        all_cells: List[ExtractedTableCell] = []

        for page_raw in page_raws:
            page_grids = [
                grid for grid in grids
                if self._safe_int(grid.get("page_number"), default=-1) == page_raw.page_number
            ]

            for grid in page_grids:
                grid_id = grid.get("table_grid_id", "")
                structure = structure_by_grid_id.get(grid_id, {})

                cells = self.extract_cells_for_grid(
                    page_raw=page_raw,
                    grid=grid,
                    structure=structure,
                )

                all_cells.extend(cells)

        all_cells = self._sort_cells(all_cells)

        result = {
            "processor": "TableCellExtractor",
            "table_cells": [
                cell.to_dict() for cell in all_cells
            ],
            "table_cells_by_page": self._group_cells_by_page(all_cells),
            "table_cells_by_table": self._group_cells_by_table(all_cells),
            "table_matrices": self._build_table_matrices(
                cells=all_cells,
                grids=grids,
            ),
            "table_cell_summary": self._build_summary(all_cells),
            "config": {
                "use_grid_cells": self.config.use_grid_cells,
                "fallback_from_grid_positions": self.config.fallback_from_grid_positions,
                "enrich_text_from_page_lines": self.config.enrich_text_from_page_lines,
                "enrich_text_from_words": self.config.enrich_text_from_words,
                "include_empty_cells": self.config.include_empty_cells,
                "assign_cell_roles": self.config.assign_cell_roles,
                "min_text_overlap_ratio": self.config.min_text_overlap_ratio,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                cells=all_cells,
                result=result,
            )

        return result

    def extract_cells_for_grid(
        self,
        page_raw: PageRaw,
        grid: Dict[str, Any],
        structure: Optional[Dict[str, Any]] = None,
    ) -> List[ExtractedTableCell]:
        structure = structure or {}

        base_cells: List[Dict[str, Any]] = []

        if self.config.use_grid_cells:
            base_cells = grid.get("cells", []) or []

        if not base_cells and self.config.fallback_from_grid_positions:
            base_cells = self._build_cells_from_grid_positions(grid)

        extracted_cells: List[ExtractedTableCell] = []

        for base_cell in base_cells:
            bbox = base_cell.get("bbox")

            if not self._is_valid_cell_bbox(bbox):
                continue

            row_index = self._safe_int(base_cell.get("row_index"), default=0)
            col_index = self._safe_int(base_cell.get("col_index"), default=0)
            row_span = max(1, self._safe_int(base_cell.get("row_span"), default=1))
            col_span = max(1, self._safe_int(base_cell.get("col_span"), default=1))

            extracted_text = self._clean_text(base_cell.get("text", ""))
            text_line_ids = list(base_cell.get("text_line_ids", []) or [])
            word_ids: List[str] = []

            if self.config.enrich_text_from_page_lines:
                line_text, line_ids = self._extract_text_lines_inside_cell(
                    page_raw=page_raw,
                    cell_bbox=bbox,
                )

                if line_text and (
                    not extracted_text
                    or len(line_text) > len(extracted_text)
                ):
                    extracted_text = line_text
                    text_line_ids = line_ids

            if self.config.enrich_text_from_words:
                word_text, extracted_word_ids = self._extract_words_inside_cell(
                    page_raw=page_raw,
                    cell_bbox=bbox,
                )

                if not extracted_text and word_text:
                    extracted_text = word_text

                if extracted_word_ids:
                    word_ids = extracted_word_ids

            normalized_text = self._normalize_cell_text(extracted_text)

            if not self.config.include_empty_cells and not normalized_text:
                continue

            role_info = self._assign_role(
                row_index=row_index,
                col_index=col_index,
                text=normalized_text,
                structure=structure,
            )

            confidence = self._score_cell(
                grid=grid,
                base_cell=base_cell,
                text=normalized_text,
                role_info=role_info,
            )

            extracted_cells.append(
                ExtractedTableCell(
                    table_cell_id=make_id("tbl_cell_ext"),
                    table_grid_id=grid.get("table_grid_id", ""),
                    table_structure_id=structure.get("table_structure_id", ""),
                    table_boundary_id=grid.get("table_boundary_id", ""),
                    page_number=page_raw.page_number,
                    page_index=page_raw.page_index,
                    row_index=row_index,
                    col_index=col_index,
                    row_span=row_span,
                    col_span=col_span,
                    bbox=[
                        round(float(bbox[0]), 4),
                        round(float(bbox[1]), 4),
                        round(float(bbox[2]), 4),
                        round(float(bbox[3]), 4),
                    ],
                    text=extracted_text[: self.config.max_cell_text_chars],
                    normalized_text=normalized_text[: self.config.max_cell_text_chars],
                    cell_role=role_info["cell_role"],
                    column_header=role_info["column_header"],
                    is_header=role_info["is_header"],
                    is_stub=role_info["is_stub"],
                    is_numeric=role_info["is_numeric"],
                    is_empty=normalized_text == "",
                    confidence=confidence,
                    source="table_grid_builder.cells",
                    source_cell_id=base_cell.get("cell_id", ""),
                    text_line_ids=text_line_ids,
                    word_ids=word_ids,
                    metadata={
                        "grid_method": grid.get("grid_method", ""),
                        "grid_confidence": grid.get("confidence", 0.5),
                        "base_cell_metadata": base_cell.get("metadata", {}),
                        "structure_type": structure.get("table_type", ""),
                        "row_count": grid.get("row_count", 0),
                        "col_count": grid.get("col_count", 0),
                        "text_preview": normalized_text[: self.config.text_preview_chars],
                    },
                )
            )

        return extracted_cells

    def _build_cells_from_grid_positions(
        self,
        grid: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        row_positions = grid.get("row_positions", []) or []
        col_positions = grid.get("col_positions", []) or []

        if len(row_positions) < 2 or len(col_positions) < 2:
            return []

        row_positions = [
            float(pos) for pos in row_positions
        ]

        col_positions = [
            float(pos) for pos in col_positions
        ]

        cells: List[Dict[str, Any]] = []

        row_count = len(row_positions) - 1
        col_count = len(col_positions) - 1

        for row_index in range(row_count):
            for col_index in range(col_count):
                bbox = [
                    col_positions[col_index],
                    row_positions[row_index],
                    col_positions[col_index + 1],
                    row_positions[row_index + 1],
                ]

                cells.append(
                    {
                        "cell_id": make_id("tbl_cell"),
                        "table_grid_id": grid.get("table_grid_id", ""),
                        "page_number": grid.get("page_number"),
                        "row_index": row_index,
                        "col_index": col_index,
                        "row_span": 1,
                        "col_span": 1,
                        "bbox": bbox,
                        "text": "",
                        "text_line_ids": [],
                        "metadata": {
                            "created_from": "grid_positions_fallback",
                        },
                    }
                )

        return cells

    def _extract_text_lines_inside_cell(
        self,
        page_raw: PageRaw,
        cell_bbox: List[float],
    ) -> Tuple[str, List[str]]:
        lines: List[Dict[str, Any]] = []

        for line in page_raw.text_lines:
            bbox = getattr(line, "bbox", None)

            if not bbox:
                continue

            include_line = False

            if self._bbox_center_inside(bbox, cell_bbox):
                include_line = True
            else:
                overlap_ratio = self._bbox_overlap_ratio(
                    inner=bbox,
                    outer=cell_bbox,
                )

                if overlap_ratio >= self.config.min_text_overlap_ratio:
                    include_line = True

            if not include_line:
                continue

            text = (
                getattr(line, "normalized_text", "")
                or getattr(line, "text", "")
                or ""
            )

            text = self._clean_text(text)

            if not text:
                continue

            line_id = getattr(line, "line_id", "") or make_id("line_ref")

            lines.append(
                {
                    "line_id": line_id,
                    "text": text,
                    "bbox": bbox,
                }
            )

        lines = sorted(
            lines,
            key=lambda item: (
                item["bbox"][1],
                item["bbox"][0],
            ),
        )

        text = "\n".join(
            item["text"] for item in lines
        ).strip()

        line_ids = [
            item["line_id"] for item in lines
        ]

        return text, line_ids

    def _extract_words_inside_cell(
        self,
        page_raw: PageRaw,
        cell_bbox: List[float],
    ) -> Tuple[str, List[str]]:
        words: List[Dict[str, Any]] = []

        for index, word in enumerate(page_raw.words):
            bbox = getattr(word, "bbox", None)

            if not bbox:
                continue

            if not self._bbox_center_inside(bbox, cell_bbox):
                continue

            text = (
                getattr(word, "normalized_text", "")
                or getattr(word, "text", "")
                or ""
            )

            text = self._clean_text(text)

            if not text:
                continue

            word_id = getattr(word, "word_id", "") or f"word_{index}"

            words.append(
                {
                    "word_id": word_id,
                    "text": text,
                    "bbox": bbox,
                }
            )

        words = sorted(
            words,
            key=lambda item: (
                item["bbox"][1],
                item["bbox"][0],
            ),
        )

        text = " ".join(
            item["text"] for item in words
        ).strip()

        word_ids = [
            item["word_id"] for item in words
        ]

        return text, word_ids

    def _assign_role(
        self,
        row_index: int,
        col_index: int,
        text: str,
        structure: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.config.assign_cell_roles:
            return {
                "cell_role": "body",
                "column_header": "",
                "is_header": False,
                "is_stub": False,
                "is_numeric": self._is_numeric_like(text),
            }

        header_row_indices = structure.get("header_row_indices", []) or []
        stub_column_indices = structure.get("stub_column_indices", []) or []
        numeric_column_indices = structure.get("numeric_column_indices", []) or []
        column_headers = structure.get("column_headers", []) or []

        header_row_indices = [
            self._safe_int(item, default=-1)
            for item in header_row_indices
        ]

        stub_column_indices = [
            self._safe_int(item, default=-1)
            for item in stub_column_indices
        ]

        numeric_column_indices = [
            self._safe_int(item, default=-1)
            for item in numeric_column_indices
        ]

        is_header = row_index in header_row_indices
        is_stub = col_index in stub_column_indices
        is_numeric = col_index in numeric_column_indices or self._is_numeric_like(text)

        column_header = ""

        if 0 <= col_index < len(column_headers):
            column_header = self._clean_text(column_headers[col_index])

        if is_header:
            cell_role = "header"
        elif is_stub:
            cell_role = "stub"
        elif is_numeric:
            cell_role = "numeric_body"
        elif not text:
            cell_role = "empty"
        else:
            cell_role = "body"

        return {
            "cell_role": cell_role,
            "column_header": column_header,
            "is_header": is_header,
            "is_stub": is_stub,
            "is_numeric": is_numeric,
        }

    def _score_cell(
        self,
        grid: Dict[str, Any],
        base_cell: Dict[str, Any],
        text: str,
        role_info: Dict[str, Any],
    ) -> float:
        grid_confidence = self._safe_float(
            grid.get("confidence", 0.5),
            default=0.5,
        )

        score = (grid_confidence + 0.50) / 2.0

        if base_cell.get("bbox"):
            score += 0.10

        if text:
            score += 0.10

        if base_cell.get("text_line_ids"):
            score += 0.05

        if role_info.get("is_header") or role_info.get("is_stub"):
            score += 0.05

        return round(max(0.0, min(score, 0.95)), 4)

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

    def _build_table_matrices(
        self,
        cells: List[ExtractedTableCell],
        grids: List[Dict[str, Any]],
    ) -> Dict[str, List[List[Dict[str, Any]]]]:
        cells_by_table = self._group_cell_objects_by_table(cells)

        grid_by_id = {
            grid.get("table_grid_id", ""): grid
            for grid in grids
            if grid.get("table_grid_id")
        }

        matrices: Dict[str, List[List[Dict[str, Any]]]] = {}

        for table_grid_id, table_cells in cells_by_table.items():
            grid = grid_by_id.get(table_grid_id, {})

            row_count = self._safe_int(grid.get("row_count"), default=0)
            col_count = self._safe_int(grid.get("col_count"), default=0)

            if row_count <= 0:
                row_count = max(
                    [cell.row_index for cell in table_cells],
                    default=-1,
                ) + 1

            if col_count <= 0:
                col_count = max(
                    [cell.col_index for cell in table_cells],
                    default=-1,
                ) + 1

            matrix: List[List[Dict[str, Any]]] = []

            for row_index in range(row_count):
                row = []

                for col_index in range(col_count):
                    row.append(
                        {
                            "row_index": row_index,
                            "col_index": col_index,
                            "table_cell_id": "",
                            "text": "",
                            "normalized_text": "",
                            "cell_role": "empty",
                            "column_header": "",
                            "is_header": False,
                            "is_stub": False,
                            "is_numeric": False,
                            "is_empty": True,
                            "bbox": [],
                        }
                    )

                matrix.append(row)

            for cell in table_cells:
                if cell.row_index < 0 or cell.row_index >= row_count:
                    continue

                if cell.col_index < 0 or cell.col_index >= col_count:
                    continue

                matrix[cell.row_index][cell.col_index] = {
                    "row_index": cell.row_index,
                    "col_index": cell.col_index,
                    "table_cell_id": cell.table_cell_id,
                    "text": cell.text,
                    "normalized_text": cell.normalized_text,
                    "cell_role": cell.cell_role,
                    "column_header": cell.column_header,
                    "is_header": cell.is_header,
                    "is_stub": cell.is_stub,
                    "is_numeric": cell.is_numeric,
                    "is_empty": cell.is_empty,
                    "bbox": cell.bbox,
                }

            matrices[table_grid_id] = matrix

        return matrices

    def _group_cell_objects_by_table(
        self,
        cells: List[ExtractedTableCell],
    ) -> Dict[str, List[ExtractedTableCell]]:
        grouped: Dict[str, List[ExtractedTableCell]] = {}

        for cell in cells:
            table_key = cell.table_grid_id or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(cell)

        return grouped

    def _group_cells_by_page(
        self,
        cells: List[ExtractedTableCell],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for cell in cells:
            page_key = str(cell.page_number)
            grouped.setdefault(page_key, [])
            grouped[page_key].append(cell.to_dict())

        return grouped

    def _group_cells_by_table(
        self,
        cells: List[ExtractedTableCell],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for cell in cells:
            table_key = cell.table_grid_id or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(cell.to_dict())

        return grouped

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        cells: List[ExtractedTableCell],
        result: Dict[str, Any],
    ) -> None:
        by_page: Dict[int, List[Dict[str, Any]]] = {}

        for cell in cells:
            by_page.setdefault(cell.page_number, [])
            by_page[cell.page_number].append(cell.to_dict())

        for page_raw in page_raws:
            page_cells = by_page.get(page_raw.page_number, [])

            page_raw.metadata.setdefault("table_cell_extractor", {})
            page_raw.metadata["table_cell_extractor"] = {
                "processor": "TableCellExtractor",
                "table_cells_on_page": page_cells,
                "table_cell_count_on_page": len(page_cells),
                "table_cell_summary": result.get("table_cell_summary", {}),
            }

    def _build_summary(
        self,
        cells: List[ExtractedTableCell],
    ) -> Dict[str, Any]:
        by_page: Dict[str, int] = {}
        by_table: Dict[str, int] = {}
        by_role: Dict[str, int] = {}

        non_empty_count = 0
        header_count = 0
        stub_count = 0
        numeric_count = 0

        for cell in cells:
            page_key = str(cell.page_number)
            table_key = cell.table_grid_id or "unknown_table"
            role_key = cell.cell_role or "unknown"

            by_page[page_key] = by_page.get(page_key, 0) + 1
            by_table[table_key] = by_table.get(table_key, 0) + 1
            by_role[role_key] = by_role.get(role_key, 0) + 1

            if not cell.is_empty:
                non_empty_count += 1

            if cell.is_header:
                header_count += 1

            if cell.is_stub:
                stub_count += 1

            if cell.is_numeric:
                numeric_count += 1

        return {
            "has_table_cells": len(cells) > 0,
            "table_cell_count": len(cells),
            "non_empty_cell_count": non_empty_count,
            "empty_cell_count": len(cells) - non_empty_count,
            "header_cell_count": header_count,
            "stub_cell_count": stub_count,
            "numeric_cell_count": numeric_count,
            "page_count_with_cells": len(by_page),
            "table_count_with_cells": len(by_table),
            "by_page": by_page,
            "by_table": by_table,
            "by_role": by_role,
        }

    def _sort_cells(
        self,
        cells: List[ExtractedTableCell],
    ) -> List[ExtractedTableCell]:
        return sorted(
            cells,
            key=lambda item: (
                item.page_number,
                item.table_grid_id,
                item.row_index,
                item.col_index,
            ),
        )

    def _is_valid_cell_bbox(
        self,
        bbox: Any,
    ) -> bool:
        if not bbox or len(bbox) != 4:
            return False

        width = max(float(bbox[2]) - float(bbox[0]), 0.0)
        height = max(float(bbox[3]) - float(bbox[1]), 0.0)

        if width < self.config.min_cell_width:
            return False

        if height < self.config.min_cell_height:
            return False

        return True

    def _bbox_center_inside(
        self,
        inner: List[float],
        outer: List[float],
    ) -> bool:
        cx = (float(inner[0]) + float(inner[2])) / 2.0
        cy = (float(inner[1]) + float(inner[3])) / 2.0

        return (
            float(outer[0]) - self.config.center_inside_tolerance
            <= cx
            <= float(outer[2]) + self.config.center_inside_tolerance
            and float(outer[1]) - self.config.center_inside_tolerance
            <= cy
            <= float(outer[3]) + self.config.center_inside_tolerance
        )

    def _bbox_overlap_ratio(
        self,
        inner: List[float],
        outer: List[float],
    ) -> float:
        x0 = max(float(inner[0]), float(outer[0]))
        y0 = max(float(inner[1]), float(outer[1]))
        x1 = min(float(inner[2]), float(outer[2]))
        y1 = min(float(inner[3]), float(outer[3]))

        inter_width = max(x1 - x0, 0.0)
        inter_height = max(y1 - y0, 0.0)
        inter_area = inter_width * inter_height

        inner_area = max(float(inner[2]) - float(inner[0]), 0.0) * max(
            float(inner[3]) - float(inner[1]),
            0.0,
        )

        if inner_area <= 0:
            return 0.0

        return inter_area / inner_area

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

    def _normalize_cell_text(
        self,
        text: Any,
    ) -> str:
        text = self._clean_text(text)

        if not text:
            return ""

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

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


def extract_table_cells(
    page_raws: List[PageRaw],
    table_grid_result: Optional[Dict[str, Any]] = None,
    table_structure_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    extractor = TableCellExtractor()
    return extractor.process(
        page_raws=page_raws,
        table_grid_result=table_grid_result,
        table_structure_result=table_structure_result,
    )
