"""
table_grid_builder.py

Production V1 - Colab Ready

Purpose
-------
Build table grid structure from refined table boundaries.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- document_ai.table.table_boundary_detector.TableBoundaryDetector

Output
------
Dictionary with:
- table_grids
- table_grids_by_page
- table_grid_summary

Flow
----
TableBoundaryDetector
    ↓
TableGridBuilder
    ↓
TableStructureRecognizer
    ↓
TableCellExtractor
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class TableGridBuilderConfig:
    min_row_height: float = 6.0
    min_col_width: float = 10.0

    row_merge_tolerance: float = 4.0
    col_merge_tolerance: float = 4.0

    use_drawing_lines: bool = True
    use_text_alignment: bool = True

    fallback_min_rows: int = 2
    fallback_min_cols: int = 2

    attach_to_pages: bool = True
    include_debug: bool = True


@dataclass
class TableGridCell:
    cell_id: str
    table_grid_id: str
    page_number: int

    row_index: int
    col_index: int
    row_span: int
    col_span: int
    bbox: List[float]

    text: str = ""
    text_line_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["text_line_ids"] is None:
            data["text_line_ids"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class TableGrid:
    table_grid_id: str
    table_boundary_id: str
    page_number: int
    page_index: int
    bbox: List[float]

    row_count: int
    col_count: int

    row_positions: List[float]
    col_positions: List[float]
    cells: Optional[List[TableGridCell]] = None

    confidence: float = 0.5
    grid_method: str = "unknown"
    source: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["cells"] is None:
            data["cells"] = []
        else:
            data["cells"] = [
                cell.to_dict() if hasattr(cell, "to_dict") else cell
                for cell in data["cells"]
            ]

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class TableGridBuilder:
    def __init__(
        self,
        config: Optional[TableGridBuilderConfig] = None,
    ):
        self.config = config or TableGridBuilderConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        table_boundary_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        table_boundaries = self._collect_table_boundaries(
            page_raws=page_raws,
            table_boundary_result=table_boundary_result,
        )

        grids: List[TableGrid] = []

        for page_raw in page_raws:
            page_boundaries = [
                boundary for boundary in table_boundaries
                if boundary.get("page_number") == page_raw.page_number
            ]

            for boundary in page_boundaries:
                grid = self.build_grid_for_boundary(
                    page_raw=page_raw,
                    boundary=boundary,
                )

                if grid:
                    grids.append(grid)

        grids = self._sort_grids(grids)
        grids_by_page = self._group_by_page(grids)

        result = {
            "processor": "TableGridBuilder",
            "table_grids": [
                grid.to_dict() for grid in grids
            ],
            "table_grids_by_page": grids_by_page,
            "table_grid_summary": self._build_summary(grids),
            "config": {
                "min_row_height": self.config.min_row_height,
                "min_col_width": self.config.min_col_width,
                "row_merge_tolerance": self.config.row_merge_tolerance,
                "col_merge_tolerance": self.config.col_merge_tolerance,
                "use_drawing_lines": self.config.use_drawing_lines,
                "use_text_alignment": self.config.use_text_alignment,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                grids=grids,
                result=result,
            )

        return result

    def build_grid_for_boundary(
        self,
        page_raw: PageRaw,
        boundary: Dict[str, Any],
    ) -> Optional[TableGrid]:
        bbox = boundary.get("bbox")

        if not bbox or len(bbox) != 4:
            return None

        horizontal_lines: List[float] = []
        vertical_lines: List[float] = []

        if self.config.use_drawing_lines:
            horizontal_lines, vertical_lines = self._extract_grid_lines_from_drawings(
                page_raw=page_raw,
                table_bbox=bbox,
            )

        row_positions = self._normalize_positions(
            positions=horizontal_lines,
            start=float(bbox[1]),
            end=float(bbox[3]),
            tolerance=self.config.row_merge_tolerance,
            min_gap=self.config.min_row_height,
        )

        col_positions = self._normalize_positions(
            positions=vertical_lines,
            start=float(bbox[0]),
            end=float(bbox[2]),
            tolerance=self.config.col_merge_tolerance,
            min_gap=self.config.min_col_width,
        )

        grid_method = "drawing_lines"

        if len(row_positions) < 2 or len(col_positions) < 2:
            if self.config.use_text_alignment:
                fallback_rows, fallback_cols = self._build_grid_from_text_alignment(
                    page_raw=page_raw,
                    table_bbox=bbox,
                )

                if len(row_positions) < 2:
                    row_positions = fallback_rows

                if len(col_positions) < 2:
                    col_positions = fallback_cols

                grid_method = "text_alignment_fallback"

        if len(row_positions) < 2:
            row_positions = self._fallback_even_positions(
                start=float(bbox[1]),
                end=float(bbox[3]),
                count=self.config.fallback_min_rows + 1,
            )

        if len(col_positions) < 2:
            col_positions = self._fallback_even_positions(
                start=float(bbox[0]),
                end=float(bbox[2]),
                count=self.config.fallback_min_cols + 1,
            )

        row_positions = sorted(row_positions)
        col_positions = sorted(col_positions)

        row_count = max(len(row_positions) - 1, 0)
        col_count = max(len(col_positions) - 1, 0)

        if row_count <= 0 or col_count <= 0:
            return None

        table_grid_id = make_id("tbl_grid")

        cells = self._build_cells(
            page_raw=page_raw,
            table_grid_id=table_grid_id,
            table_bbox=bbox,
            row_positions=row_positions,
            col_positions=col_positions,
        )

        confidence = self._score_grid(
            row_count=row_count,
            col_count=col_count,
            horizontal_line_count=len(horizontal_lines),
            vertical_line_count=len(vertical_lines),
            method=grid_method,
        )

        return TableGrid(
            table_grid_id=table_grid_id,
            table_boundary_id=boundary.get("table_boundary_id", ""),
            page_number=page_raw.page_number,
            page_index=page_raw.page_index,
            bbox=[
                round(float(bbox[0]), 4),
                round(float(bbox[1]), 4),
                round(float(bbox[2]), 4),
                round(float(bbox[3]), 4),
            ],
            row_count=row_count,
            col_count=col_count,
            row_positions=[
                round(float(pos), 4) for pos in row_positions
            ],
            col_positions=[
                round(float(pos), 4) for pos in col_positions
            ],
            cells=cells,
            confidence=confidence,
            grid_method=grid_method,
            source="table_boundary_refiner",
            metadata={
                "horizontal_line_count": len(horizontal_lines),
                "vertical_line_count": len(vertical_lines),
                "boundary": boundary,
                "cell_count": len(cells),
            },
        )

    def _extract_grid_lines_from_drawings(
        self,
        page_raw: PageRaw,
        table_bbox: List[float],
    ) -> Tuple[List[float], List[float]]:
        horizontal_positions: List[float] = []
        vertical_positions: List[float] = []

        for drawing in page_raw.drawings:
            bbox = drawing.bbox

            if not bbox:
                continue

            if not self._bbox_intersects(bbox, table_bbox):
                continue

            width = max(float(bbox[2]) - float(bbox[0]), 0.0)
            height = max(float(bbox[3]) - float(bbox[1]), 0.0)

            metadata = drawing.metadata or {}

            is_horizontal = metadata.get("is_horizontal_line")
            is_vertical = metadata.get("is_vertical_line")

            if is_horizontal is None:
                is_horizontal = width >= height * 5 and width >= 20

            if is_vertical is None:
                is_vertical = height >= width * 5 and height >= 10

            if is_horizontal:
                y = (float(bbox[1]) + float(bbox[3])) / 2.0
                horizontal_positions.append(y)

            if is_vertical:
                x = (float(bbox[0]) + float(bbox[2])) / 2.0
                vertical_positions.append(x)

        return horizontal_positions, vertical_positions

    def _build_grid_from_text_alignment(
        self,
        page_raw: PageRaw,
        table_bbox: List[float],
    ) -> Tuple[List[float], List[float]]:
        lines = []

        for line in page_raw.text_lines:
            if not line.bbox:
                continue

            if not self._bbox_intersects(line.bbox, table_bbox):
                continue

            text = line.normalized_text or line.text or ""

            if not text.strip():
                continue

            lines.append(
                {
                    "line_id": line.line_id,
                    "text": text,
                    "bbox": line.bbox,
                }
            )

        if not lines:
            return [], []

        lines = sorted(
            lines,
            key=lambda item: (
                item["bbox"][1],
                item["bbox"][0],
            ),
        )

        row_positions = self._infer_rows_from_text_lines(
            text_lines=lines,
            table_bbox=table_bbox,
        )

        col_positions = self._infer_cols_from_text_lines(
            text_lines=lines,
            table_bbox=table_bbox,
        )

        return row_positions, col_positions

    def _infer_rows_from_text_lines(
        self,
        text_lines: List[Dict[str, Any]],
        table_bbox: List[float],
    ) -> List[float]:
        if not text_lines:
            return []

        y_centers = []

        for item in text_lines:
            bbox = item["bbox"]
            y_center = (float(bbox[1]) + float(bbox[3])) / 2.0
            y_centers.append(y_center)

        y_centers = self._cluster_positions(
            positions=y_centers,
            tolerance=self.config.row_merge_tolerance,
        )

        row_positions = [float(table_bbox[1])]

        for index, y in enumerate(y_centers):
            if index == 0:
                continue

            prev_y = y_centers[index - 1]
            midpoint = (prev_y + y) / 2.0

            if midpoint > row_positions[-1] + self.config.min_row_height:
                row_positions.append(midpoint)

        row_positions.append(float(table_bbox[3]))

        return self._normalize_positions(
            positions=row_positions,
            start=float(table_bbox[1]),
            end=float(table_bbox[3]),
            tolerance=self.config.row_merge_tolerance,
            min_gap=self.config.min_row_height,
        )

    def _infer_cols_from_text_lines(
        self,
        text_lines: List[Dict[str, Any]],
        table_bbox: List[float],
    ) -> List[float]:
        x_positions = []

        for item in text_lines:
            bbox = item["bbox"]
            text = item.get("text", "")

            x_positions.append(float(bbox[0]))
            x_positions.append(float(bbox[2]))

            parts = self._split_text_line_to_column_parts(text)

            if len(parts) >= 2:
                x0 = float(bbox[0])
                x1 = float(bbox[2])
                width = max(x1 - x0, 1.0)
                step = width / len(parts)

                for index in range(1, len(parts)):
                    x_positions.append(x0 + step * index)

        x_positions.append(float(table_bbox[0]))
        x_positions.append(float(table_bbox[2]))

        x_positions = self._cluster_positions(
            positions=x_positions,
            tolerance=self.config.col_merge_tolerance,
        )

        return self._normalize_positions(
            positions=x_positions,
            start=float(table_bbox[0]),
            end=float(table_bbox[2]),
            tolerance=self.config.col_merge_tolerance,
            min_gap=self.config.min_col_width,
        )

    def _split_text_line_to_column_parts(
        self,
        text: str,
    ) -> List[str]:
        if not text:
            return []

        if "\t" in text:
            return [
                part.strip() for part in text.split("\t")
                if part.strip()
            ]

        if "|" in text:
            return [
                part.strip() for part in text.split("|")
                if part.strip()
            ]

        import re

        return [
            part.strip() for part in re.split(r"\s{2,}", text)
            if part.strip()
        ]

    def _build_cells(
        self,
        page_raw: PageRaw,
        table_grid_id: str,
        table_bbox: List[float],
        row_positions: List[float],
        col_positions: List[float],
    ) -> List[TableGridCell]:
        cells: List[TableGridCell] = []

        row_count = len(row_positions) - 1
        col_count = len(col_positions) - 1

        for row_index in range(row_count):
            for col_index in range(col_count):
                cell_bbox = [
                    float(col_positions[col_index]),
                    float(row_positions[row_index]),
                    float(col_positions[col_index + 1]),
                    float(row_positions[row_index + 1]),
                ]

                text, line_ids = self._extract_text_inside_cell(
                    page_raw=page_raw,
                    cell_bbox=cell_bbox,
                )

                cells.append(
                    TableGridCell(
                        cell_id=make_id("tbl_cell"),
                        table_grid_id=table_grid_id,
                        page_number=page_raw.page_number,
                        row_index=row_index,
                        col_index=col_index,
                        row_span=1,
                        col_span=1,
                        bbox=[
                            round(float(cell_bbox[0]), 4),
                            round(float(cell_bbox[1]), 4),
                            round(float(cell_bbox[2]), 4),
                            round(float(cell_bbox[3]), 4),
                        ],
                        text=text,
                        text_line_ids=line_ids,
                        metadata={
                            "table_bbox": table_bbox,
                        },
                    )
                )

        return cells

    def _extract_text_inside_cell(
        self,
        page_raw: PageRaw,
        cell_bbox: List[float],
    ) -> Tuple[str, List[str]]:
        lines = []

        for line in page_raw.text_lines:
            if not line.bbox:
                continue

            if self._bbox_center_inside(line.bbox, cell_bbox):
                text = line.normalized_text or line.text or ""

                if text.strip():
                    lines.append(
                        {
                            "line_id": line.line_id,
                            "text": text.strip(),
                            "bbox": line.bbox,
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

    def _normalize_positions(
        self,
        positions: List[float],
        start: float,
        end: float,
        tolerance: float,
        min_gap: float,
    ) -> List[float]:
        values = [start, end]

        for pos in positions:
            if start - tolerance <= float(pos) <= end + tolerance:
                values.append(float(pos))

        values = self._cluster_positions(values, tolerance=tolerance)

        cleaned: List[float] = []

        for pos in values:
            if not cleaned:
                cleaned.append(pos)
                continue

            if pos - cleaned[-1] >= min_gap:
                cleaned.append(pos)

        if cleaned[0] > start + tolerance:
            cleaned.insert(0, start)
        else:
            cleaned[0] = start

        if cleaned[-1] < end - tolerance:
            cleaned.append(end)
        else:
            cleaned[-1] = end

        return cleaned

    def _cluster_positions(
        self,
        positions: List[float],
        tolerance: float,
    ) -> List[float]:
        if not positions:
            return []

        values = sorted(
            float(pos) for pos in positions
        )

        clusters: List[List[float]] = []

        for value in values:
            if not clusters:
                clusters.append([value])
                continue

            if abs(value - clusters[-1][-1]) <= tolerance:
                clusters[-1].append(value)
            else:
                clusters.append([value])

        return [
            sum(cluster) / len(cluster)
            for cluster in clusters
        ]

    def _fallback_even_positions(
        self,
        start: float,
        end: float,
        count: int,
    ) -> List[float]:
        count = max(count, 2)
        step = (end - start) / float(count - 1)

        return [
            start + step * index
            for index in range(count)
        ]

    def _score_grid(
        self,
        row_count: int,
        col_count: int,
        horizontal_line_count: int,
        vertical_line_count: int,
        method: str,
    ) -> float:
        score = 0.40

        if method == "drawing_lines":
            score += 0.20

        if method == "text_alignment_fallback":
            score += 0.10

        if row_count >= 2:
            score += 0.10

        if col_count >= 2:
            score += 0.10

        if horizontal_line_count >= row_count + 1:
            score += 0.10

        if vertical_line_count >= col_count + 1:
            score += 0.10

        return round(max(0.0, min(score, 0.95)), 4)

    def _collect_table_boundaries(
        self,
        page_raws: List[PageRaw],
        table_boundary_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if table_boundary_result:
            boundaries = table_boundary_result.get("table_boundaries", [])

            if boundaries:
                return boundaries

        collected: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            refiner_meta = page_raw.metadata.get("table_boundary_refiner", {})
            page_boundaries = refiner_meta.get("table_boundaries_on_page", [])

            for boundary in page_boundaries:
                collected.append(boundary)

        if collected:
            return collected

        for page_raw in page_raws:
            page_meta = page_raw.metadata.get("table_boundary_detector", {})
            page_candidates = page_meta.get("table_candidates", [])

            for candidate in page_candidates:
                collected.append(
                    {
                        "table_boundary_id": candidate.get("table_id") or make_id("tbl_boundary"),
                        "page_number": page_raw.page_number,
                        "page_index": page_raw.page_index,
                        "bbox": candidate.get("bbox"),
                        "confidence": candidate.get("confidence", 0.5),
                        "detection_method": candidate.get("detection_method", "page_understanding"),
                        "source": "page_understanding.table_boundary_detector",
                    }
                )

        return collected

    def _group_by_page(
        self,
        grids: List[TableGrid],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for grid in grids:
            page_key = str(grid.page_number)
            grouped.setdefault(page_key, [])
            grouped[page_key].append(grid.to_dict())

        return grouped

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        grids: List[TableGrid],
        result: Dict[str, Any],
    ) -> None:
        by_page: Dict[int, List[Dict[str, Any]]] = {}

        for grid in grids:
            by_page.setdefault(grid.page_number, [])
            by_page[grid.page_number].append(grid.to_dict())

        for page_raw in page_raws:
            page_grids = by_page.get(page_raw.page_number, [])

            page_raw.metadata.setdefault("table_grid_builder", {})
            page_raw.metadata["table_grid_builder"] = {
                "processor": "TableGridBuilder",
                "table_grids_on_page": page_grids,
                "table_grid_count_on_page": len(page_grids),
                "table_grid_summary": result.get("table_grid_summary", {}),
            }

    def _build_summary(
        self,
        grids: List[TableGrid],
    ) -> Dict[str, Any]:
        by_page: Dict[str, int] = {}
        by_method: Dict[str, int] = {}

        total_cells = 0

        for grid in grids:
            page_key = str(grid.page_number)
            by_page[page_key] = by_page.get(page_key, 0) + 1
            by_method[grid.grid_method] = by_method.get(grid.grid_method, 0) + 1
            total_cells += len(grid.cells or [])

        return {
            "has_table_grids": len(grids) > 0,
            "table_grid_count": len(grids),
            "page_count_with_grids": len(by_page),
            "total_cell_count": total_cells,
            "by_page": by_page,
            "by_grid_method": by_method,
        }

    def _sort_grids(
        self,
        grids: List[TableGrid],
    ) -> List[TableGrid]:
        return sorted(
            grids,
            key=lambda item: (
                item.page_number,
                item.bbox[1],
                item.bbox[0],
            ),
        )

    def _bbox_intersects(
        self,
        a: List[float],
        b: List[float],
    ) -> bool:
        x0 = max(float(a[0]), float(b[0]))
        y0 = max(float(a[1]), float(b[1]))
        x1 = min(float(a[2]), float(b[2]))
        y1 = min(float(a[3]), float(b[3]))

        return x1 > x0 and y1 > y0

    def _bbox_center_inside(
        self,
        inner: List[float],
        outer: List[float],
    ) -> bool:
        cx = (float(inner[0]) + float(inner[2])) / 2.0
        cy = (float(inner[1]) + float(inner[3])) / 2.0

        return (
            float(outer[0]) <= cx <= float(outer[2])
            and float(outer[1]) <= cy <= float(outer[3])
        )


def build_table_grids(
    page_raws: List[PageRaw],
    table_boundary_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = TableGridBuilder()
    return builder.process(
        page_raws=page_raws,
        table_boundary_result=table_boundary_result,
    )
