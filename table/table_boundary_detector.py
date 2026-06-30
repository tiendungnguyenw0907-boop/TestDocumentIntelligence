"""
table_boundary_detector.py

Production V1 - Colab Ready

Purpose
-------
Refine table boundary candidates at table-understanding layer.

Input
-----
List[PageRaw] after PageUnderstandingPipeline.

This module reuses rough table candidates from:
- page_raw.metadata["table_boundary_detector"]["table_candidates"]

Output
------
Dictionary with:
- table_boundaries
- table_boundaries_by_page
- table_boundary_summary

Note
----
There is also a page-level table boundary detector:
document_ai/page_understanding/table_boundary_detector.py

This file is the table-layer refiner:
document_ai/table/table_boundary_detector.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class TableBoundaryDetectorConfig:
    min_confidence: float = 0.30

    merge_iou_threshold: float = 0.20
    merge_vertical_gap_threshold: float = 25.0
    horizontal_overlap_threshold: float = 0.50

    expand_boundary_margin: float = 2.0

    min_table_width: float = 40.0
    min_table_height: float = 20.0

    use_page_understanding_candidates: bool = True
    use_drawing_line_fallback: bool = True
    use_text_alignment_fallback: bool = True

    attach_to_pages: bool = True
    include_debug: bool = True


@dataclass
class TableBoundary:
    table_boundary_id: str
    page_number: int
    page_index: int
    bbox: List[float]

    confidence: float = 0.5
    boundary_type: str = "table_boundary"
    detection_method: str = "unknown"
    source: str = "unknown"
    source_candidate_ids: Optional[List[str]] = None

    row_hint_count: int = 0
    column_hint_count: int = 0
    text_line_count: int = 0
    drawing_line_count: int = 0

    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["source_candidate_ids"] is None:
            data["source_candidate_ids"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class TableBoundaryDetector:
    def __init__(
        self,
        config: Optional[TableBoundaryDetectorConfig] = None,
    ):
        self.config = config or TableBoundaryDetectorConfig()

    def process(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        boundaries: List[TableBoundary] = []

        for page_raw in page_raws:
            page_boundaries = self.detect_page_boundaries(page_raw)
            boundaries.extend(page_boundaries)

        boundaries = self._sort_boundaries(boundaries)
        table_boundaries_by_page = self._group_by_page(boundaries)

        result = {
            "processor": "TableBoundaryDetector",
            "table_boundaries": [
                boundary.to_dict() for boundary in boundaries
            ],
            "table_boundaries_by_page": table_boundaries_by_page,
            "table_boundary_summary": self._build_summary(boundaries),
            "config": {
                "min_confidence": self.config.min_confidence,
                "merge_iou_threshold": self.config.merge_iou_threshold,
                "merge_vertical_gap_threshold": self.config.merge_vertical_gap_threshold,
                "horizontal_overlap_threshold": self.config.horizontal_overlap_threshold,
                "expand_boundary_margin": self.config.expand_boundary_margin,
                "min_table_width": self.config.min_table_width,
                "min_table_height": self.config.min_table_height,
                "use_page_understanding_candidates": self.config.use_page_understanding_candidates,
                "use_drawing_line_fallback": self.config.use_drawing_line_fallback,
                "use_text_alignment_fallback": self.config.use_text_alignment_fallback,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                boundaries=boundaries,
                result=result,
            )

        return result

    def detect_page_boundaries(
        self,
        page_raw: PageRaw,
    ) -> List[TableBoundary]:
        candidates: List[TableBoundary] = []

        if self.config.use_page_understanding_candidates:
            candidates.extend(
                self._collect_from_page_understanding(page_raw)
            )

        if self.config.use_drawing_line_fallback:
            candidates.extend(
                self._detect_from_drawing_lines(page_raw)
            )

        if self.config.use_text_alignment_fallback:
            candidates.extend(
                self._detect_from_text_alignment(page_raw)
            )

        candidates = [
            item for item in candidates
            if self._is_valid_boundary(item)
            and item.confidence >= self.config.min_confidence
        ]

        candidates = self._merge_overlapping_boundaries(candidates)
        candidates = self._sort_boundaries(candidates)

        return candidates

    def _collect_from_page_understanding(
        self,
        page_raw: PageRaw,
    ) -> List[TableBoundary]:
        table_meta = page_raw.metadata.get("table_boundary_detector", {})
        raw_candidates = table_meta.get("table_candidates", [])

        boundaries: List[TableBoundary] = []

        for index, candidate in enumerate(raw_candidates):
            bbox = candidate.get("bbox")

            if not bbox:
                continue

            bbox = self._expand_bbox(
                bbox=bbox,
                margin=self.config.expand_boundary_margin,
                page_width=page_raw.width,
                page_height=page_raw.height,
            )

            confidence = self._safe_float(
                candidate.get("confidence", 0.5),
                default=0.5,
            )

            boundary = TableBoundary(
                table_boundary_id=make_id("tbl_boundary"),
                page_number=page_raw.page_number,
                page_index=page_raw.page_index,
                bbox=bbox,
                confidence=round(confidence, 4),
                boundary_type="table_boundary",
                detection_method=candidate.get("detection_method", "page_understanding"),
                source="page_understanding.table_boundary_detector",
                source_candidate_ids=[
                    candidate.get("table_id")
                    or candidate.get("candidate_id")
                    or f"page_table_candidate_{index}"
                ],
                row_hint_count=self._safe_int(
                    candidate.get("row_hint_count", 0),
                    default=0,
                ),
                column_hint_count=self._safe_int(
                    candidate.get("column_hint_count", 0),
                    default=0,
                ),
                text_line_count=self._count_text_lines_inside_bbox(
                    page_raw=page_raw,
                    bbox=bbox,
                ),
                drawing_line_count=self._count_drawing_lines_inside_bbox(
                    page_raw=page_raw,
                    bbox=bbox,
                ),
                metadata={
                    "raw_candidate": candidate,
                    "source_index": index,
                },
            )

            boundaries.append(boundary)

        return boundaries

    def _detect_from_drawing_lines(
        self,
        page_raw: PageRaw,
    ) -> List[TableBoundary]:
        horizontal_lines: List[List[float]] = []
        vertical_lines: List[List[float]] = []

        for drawing in page_raw.drawings:
            bbox = drawing.bbox

            if not bbox:
                continue

            width = max(float(bbox[2]) - float(bbox[0]), 0.0)
            height = max(float(bbox[3]) - float(bbox[1]), 0.0)

            if width <= 0 and height <= 0:
                continue

            metadata = drawing.metadata or {}

            is_horizontal = metadata.get("is_horizontal_line")
            is_vertical = metadata.get("is_vertical_line")

            if is_horizontal is None:
                is_horizontal = width >= height * 5 and width >= 25

            if is_vertical is None:
                is_vertical = height >= width * 5 and height >= 15

            if is_horizontal:
                horizontal_lines.append(bbox)

            if is_vertical:
                vertical_lines.append(bbox)

        if len(horizontal_lines) < 2 and len(vertical_lines) < 2:
            return []

        all_lines = horizontal_lines + vertical_lines
        bbox = self._merge_many_bboxes(all_lines)

        if not bbox:
            return []

        bbox = self._expand_bbox(
            bbox=bbox,
            margin=self.config.expand_boundary_margin,
            page_width=page_raw.width,
            page_height=page_raw.height,
        )

        confidence = 0.45

        if len(horizontal_lines) >= 3:
            confidence += 0.15

        if len(vertical_lines) >= 2:
            confidence += 0.15

        if len(horizontal_lines) >= 3 and len(vertical_lines) >= 2:
            confidence += 0.10

        return [
            TableBoundary(
                table_boundary_id=make_id("tbl_boundary"),
                page_number=page_raw.page_number,
                page_index=page_raw.page_index,
                bbox=bbox,
                confidence=round(min(confidence, 0.90), 4),
                boundary_type="table_boundary",
                detection_method="drawing_line_fallback",
                source="page_raw.drawings",
                source_candidate_ids=[],
                row_hint_count=len(horizontal_lines),
                column_hint_count=len(vertical_lines),
                text_line_count=self._count_text_lines_inside_bbox(
                    page_raw=page_raw,
                    bbox=bbox,
                ),
                drawing_line_count=len(all_lines),
                metadata={
                    "horizontal_line_count": len(horizontal_lines),
                    "vertical_line_count": len(vertical_lines),
                },
            )
        ]

    def _detect_from_text_alignment(
        self,
        page_raw: PageRaw,
    ) -> List[TableBoundary]:
        candidate_lines: List[Dict[str, Any]] = []

        for line in page_raw.text_lines:
            text = line.normalized_text or line.text or ""
            bbox = line.bbox

            if not bbox or not text.strip():
                continue

            if self._looks_like_table_text_line(text):
                candidate_lines.append(
                    {
                        "text": text,
                        "bbox": bbox,
                        "line_id": line.line_id,
                    }
                )

        if len(candidate_lines) < 3:
            return []

        groups = self._group_nearby_text_lines(candidate_lines)
        boundaries: List[TableBoundary] = []

        for group_index, group in enumerate(groups):
            if len(group) < 3:
                continue

            bboxes = [
                item["bbox"] for item in group
            ]

            bbox = self._merge_many_bboxes(bboxes)

            if not bbox:
                continue

            bbox = self._expand_bbox(
                bbox=bbox,
                margin=self.config.expand_boundary_margin,
                page_width=page_raw.width,
                page_height=page_raw.height,
            )

            column_hint_count = self._estimate_column_count_from_lines(group)

            confidence = 0.35

            if len(group) >= 5:
                confidence += 0.10

            if column_hint_count >= 3:
                confidence += 0.15

            boundaries.append(
                TableBoundary(
                    table_boundary_id=make_id("tbl_boundary"),
                    page_number=page_raw.page_number,
                    page_index=page_raw.page_index,
                    bbox=bbox,
                    confidence=round(min(confidence, 0.85), 4),
                    boundary_type="table_boundary",
                    detection_method="text_alignment_fallback",
                    source="page_raw.text_lines",
                    source_candidate_ids=[
                        item.get("line_id", "") for item in group
                    ],
                    row_hint_count=len(group),
                    column_hint_count=column_hint_count,
                    text_line_count=len(group),
                    drawing_line_count=self._count_drawing_lines_inside_bbox(
                        page_raw=page_raw,
                        bbox=bbox,
                    ),
                    metadata={
                        "group_index": group_index,
                        "line_text_samples": [
                            item.get("text", "")[:120] for item in group[:5]
                        ],
                    },
                )
            )

        return boundaries

    def _looks_like_table_text_line(
        self,
        text: str,
    ) -> bool:
        if not text:
            return False

        if "\t" in text or "|" in text:
            return True

        if re.search(r"\s{3,}", text):
            return True

        tokens = text.split()

        if len(tokens) >= 4:
            numeric_tokens = 0

            for token in tokens:
                if re.search(r"\d", token):
                    numeric_tokens += 1

            if numeric_tokens >= 2:
                return True

        digit_count = sum(ch.isdigit() for ch in text)
        digit_ratio = digit_count / max(len(text), 1)

        if digit_ratio >= 0.25 and len(tokens) >= 3:
            return True

        return False

    def _group_nearby_text_lines(
        self,
        candidate_lines: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        if not candidate_lines:
            return []

        sorted_lines = sorted(
            candidate_lines,
            key=lambda item: (
                item["bbox"][1],
                item["bbox"][0],
            ),
        )

        groups: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []

        for item in sorted_lines:
            if not current:
                current = [item]
                continue

            previous = current[-1]
            gap = float(item["bbox"][1]) - float(previous["bbox"][3])

            if gap <= self.config.merge_vertical_gap_threshold:
                current.append(item)
            else:
                groups.append(current)
                current = [item]

        if current:
            groups.append(current)

        return groups

    def _estimate_column_count_from_lines(
        self,
        group: List[Dict[str, Any]],
    ) -> int:
        max_columns = 0

        for item in group:
            text = item.get("text", "")

            if "\t" in text:
                column_count = len(
                    [
                        part for part in text.split("\t")
                        if part.strip()
                    ]
                )
            elif "|" in text:
                column_count = len(
                    [
                        part for part in text.split("|")
                        if part.strip()
                    ]
                )
            else:
                column_count = len(
                    [
                        part for part in re.split(r"\s{2,}", text)
                        if part.strip()
                    ]
                )

            max_columns = max(max_columns, column_count)

        return max(max_columns, 1)

    def _merge_overlapping_boundaries(
        self,
        boundaries: List[TableBoundary],
    ) -> List[TableBoundary]:
        if not boundaries:
            return []

        remaining = sorted(
            boundaries,
            key=lambda item: (
                item.page_number,
                item.bbox[1],
                item.bbox[0],
                -item.confidence,
            ),
        )

        merged: List[TableBoundary] = []

        while remaining:
            current = remaining.pop(0)
            changed = True

            while changed:
                changed = False
                keep_remaining: List[TableBoundary] = []

                for other in remaining:
                    if other.page_number != current.page_number:
                        keep_remaining.append(other)
                        continue

                    should_merge = False
                    iou = self._bbox_iou(current.bbox, other.bbox)

                    if iou >= self.config.merge_iou_threshold:
                        should_merge = True

                    if self._are_vertically_close(current.bbox, other.bbox):
                        horizontal_overlap = self._horizontal_overlap_ratio(
                            current.bbox,
                            other.bbox,
                        )

                        if horizontal_overlap >= self.config.horizontal_overlap_threshold:
                            should_merge = True

                    if should_merge:
                        current = self._merge_two_boundaries(current, other)
                        changed = True
                    else:
                        keep_remaining.append(other)

                remaining = keep_remaining

            merged.append(current)

        return merged

    def _merge_two_boundaries(
        self,
        a: TableBoundary,
        b: TableBoundary,
    ) -> TableBoundary:
        bbox = self._merge_bbox(a.bbox, b.bbox)

        source_candidate_ids = []
        source_candidate_ids.extend(a.source_candidate_ids or [])
        source_candidate_ids.extend(b.source_candidate_ids or [])

        detection_methods = sorted(
            set(
                [
                    item for item in [
                        a.detection_method,
                        b.detection_method,
                    ]
                    if item
                ]
            )
        )

        confidence = max(a.confidence, b.confidence)

        if a.detection_method != b.detection_method:
            confidence = min(confidence + 0.05, 0.95)

        return TableBoundary(
            table_boundary_id=a.table_boundary_id,
            page_number=a.page_number,
            page_index=a.page_index,
            bbox=bbox,
            confidence=round(confidence, 4),
            boundary_type="table_boundary",
            detection_method="+".join(detection_methods),
            source="merged_table_boundary_candidates",
            source_candidate_ids=list(dict.fromkeys(source_candidate_ids)),
            row_hint_count=max(a.row_hint_count, b.row_hint_count),
            column_hint_count=max(a.column_hint_count, b.column_hint_count),
            text_line_count=max(a.text_line_count, b.text_line_count),
            drawing_line_count=max(a.drawing_line_count, b.drawing_line_count),
            metadata={
                "merged_from": [
                    a.to_dict(),
                    b.to_dict(),
                ],
            },
        )

    def _is_valid_boundary(
        self,
        boundary: TableBoundary,
    ) -> bool:
        bbox = boundary.bbox

        if not bbox or len(bbox) != 4:
            return False

        width = max(float(bbox[2]) - float(bbox[0]), 0.0)
        height = max(float(bbox[3]) - float(bbox[1]), 0.0)

        if width < self.config.min_table_width:
            return False

        if height < self.config.min_table_height:
            return False

        return True

    def _count_text_lines_inside_bbox(
        self,
        page_raw: PageRaw,
        bbox: List[float],
    ) -> int:
        count = 0

        for line in page_raw.text_lines:
            if line.bbox and self._bbox_intersects(line.bbox, bbox):
                count += 1

        return count

    def _count_drawing_lines_inside_bbox(
        self,
        page_raw: PageRaw,
        bbox: List[float],
    ) -> int:
        count = 0

        for drawing in page_raw.drawings:
            if drawing.bbox and self._bbox_intersects(drawing.bbox, bbox):
                count += 1

        return count

    def _group_by_page(
        self,
        boundaries: List[TableBoundary],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for boundary in boundaries:
            page_key = str(boundary.page_number)
            grouped.setdefault(page_key, [])
            grouped[page_key].append(boundary.to_dict())

        return grouped

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        boundaries: List[TableBoundary],
        result: Dict[str, Any],
    ) -> None:
        by_page: Dict[int, List[Dict[str, Any]]] = {}

        for boundary in boundaries:
            by_page.setdefault(boundary.page_number, [])
            by_page[boundary.page_number].append(boundary.to_dict())

        for page_raw in page_raws:
            page_boundaries = by_page.get(page_raw.page_number, [])

            page_raw.metadata.setdefault("table_boundary_refiner", {})
            page_raw.metadata["table_boundary_refiner"] = {
                "processor": "document_ai.table.table_boundary_detector.TableBoundaryDetector",
                "table_boundaries_on_page": page_boundaries,
                "table_boundary_count_on_page": len(page_boundaries),
                "table_boundary_summary": result.get("table_boundary_summary", {}),
            }

    def _build_summary(
        self,
        boundaries: List[TableBoundary],
    ) -> Dict[str, Any]:
        by_page: Dict[str, int] = {}
        by_method: Dict[str, int] = {}

        for boundary in boundaries:
            page_key = str(boundary.page_number)
            by_page[page_key] = by_page.get(page_key, 0) + 1

            method = boundary.detection_method or "unknown"
            by_method[method] = by_method.get(method, 0) + 1

        return {
            "has_table_boundaries": len(boundaries) > 0,
            "table_boundary_count": len(boundaries),
            "page_count_with_tables": len(by_page),
            "by_page": by_page,
            "by_detection_method": by_method,
        }

    def _sort_boundaries(
        self,
        boundaries: List[TableBoundary],
    ) -> List[TableBoundary]:
        return sorted(
            boundaries,
            key=lambda item: (
                item.page_number,
                item.bbox[1],
                item.bbox[0],
            ),
        )

    def _expand_bbox(
        self,
        bbox: List[float],
        margin: float,
        page_width: Optional[float],
        page_height: Optional[float],
    ) -> List[float]:
        x0 = float(bbox[0]) - margin
        y0 = float(bbox[1]) - margin
        x1 = float(bbox[2]) + margin
        y1 = float(bbox[3]) + margin

        x0 = max(x0, 0.0)
        y0 = max(y0, 0.0)

        if page_width:
            x1 = min(x1, float(page_width))

        if page_height:
            y1 = min(y1, float(page_height))

        return [
            round(x0, 4),
            round(y0, 4),
            round(x1, 4),
            round(y1, 4),
        ]

    def _merge_bbox(
        self,
        a: List[float],
        b: List[float],
    ) -> List[float]:
        return [
            round(min(float(a[0]), float(b[0])), 4),
            round(min(float(a[1]), float(b[1])), 4),
            round(max(float(a[2]), float(b[2])), 4),
            round(max(float(a[3]), float(b[3])), 4),
        ]

    def _merge_many_bboxes(
        self,
        bboxes: List[List[float]],
    ) -> Optional[List[float]]:
        if not bboxes:
            return None

        bbox = bboxes[0]

        for other in bboxes[1:]:
            bbox = self._merge_bbox(bbox, other)

        return bbox

    def _bbox_area(
        self,
        bbox: List[float],
    ) -> float:
        return max(float(bbox[2]) - float(bbox[0]), 0.0) * max(
            float(bbox[3]) - float(bbox[1]),
            0.0,
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

    def _bbox_iou(
        self,
        a: List[float],
        b: List[float],
    ) -> float:
        x0 = max(float(a[0]), float(b[0]))
        y0 = max(float(a[1]), float(b[1]))
        x1 = min(float(a[2]), float(b[2]))
        y1 = min(float(a[3]), float(b[3]))

        inter_width = max(x1 - x0, 0.0)
        inter_height = max(y1 - y0, 0.0)
        inter_area = inter_width * inter_height

        if inter_area <= 0:
            return 0.0

        union_area = self._bbox_area(a) + self._bbox_area(b) - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

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

        smaller_width = min(width_a, width_b)

        if smaller_width <= 0:
            return 0.0

        return overlap / smaller_width

    def _are_vertically_close(
        self,
        a: List[float],
        b: List[float],
    ) -> bool:
        if float(a[3]) <= float(b[1]):
            gap = float(b[1]) - float(a[3])
        elif float(b[3]) <= float(a[1]):
            gap = float(a[1]) - float(b[3])
        else:
            gap = 0.0

        return gap <= self.config.merge_vertical_gap_threshold

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


def detect_table_boundaries(
    page_raws: List[PageRaw],
) -> Dict[str, Any]:
    detector = TableBoundaryDetector()
    return detector.process(page_raws)
