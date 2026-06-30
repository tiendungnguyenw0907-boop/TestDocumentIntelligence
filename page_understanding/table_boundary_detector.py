"""
table_boundary_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect rough table boundary candidates at page-understanding level.

Input
-----
PageRaw after:
- PageExtractionPipeline
- ObjectNormalizer
- ObjectMerger
- ReadingOrderBuilder
- RegionDetector
- HeaderFooterDetector

Output
------
PageRaw with metadata:
- table_candidates
- table_boundary_summary

Important
---------
This module only detects rough table boundaries.

It does not:
- build table grid
- extract table cells
- detect rowspan / colspan
- recognize table header
- recognize multi-page table

Those tasks belong to the table/ package:
- table_grid_builder.py
- table_structure_recognizer.py
- table_cell_extractor.py
- multi_page_table_detector.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class TableBoundaryDetectorConfig:
    """
    Configuration for TableBoundaryDetector.
    """

    use_region_detector_candidates: bool = True
    use_drawing_line_candidates: bool = True
    use_text_alignment_candidates: bool = True

    min_horizontal_lines: int = 2
    min_vertical_lines: int = 2
    min_text_lines_for_table: int = 3
    min_columns_for_text_table: int = 2

    line_overlap_tolerance: float = 4.0
    text_x_alignment_tolerance: float = 12.0
    text_y_gap_tolerance: float = 18.0

    table_padding: float = 4.0
    min_table_width: float = 40.0
    min_table_height: float = 20.0

    merge_overlapping_candidates: bool = True
    candidate_overlap_threshold: float = 0.35

    include_text_preview: bool = True
    text_preview_chars: int = 500


@dataclass
class TableBoundaryCandidate:
    """
    One rough table boundary candidate.
    """

    table_id: str
    page_number: int
    bbox: Optional[List[float]]
    detection_method: str

    confidence: float = 0.5
    source: str = "unknown"
    source_object_ids: Optional[List[str]] = None
    text_preview: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["source_object_ids"] is None:
            data["source_object_ids"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class TableBoundaryDetector:
    """
    Detect rough table boundary candidates on one page.
    """

    def __init__(
        self,
        config: Optional[TableBoundaryDetectorConfig] = None,
    ):
        self.config = config or TableBoundaryDetectorConfig()

    def process(
        self,
        page_raw: PageRaw,
    ) -> PageRaw:
        """
        Detect table boundary candidates and attach result to page_raw.metadata.
        """

        warnings: List[str] = []
        candidates: List[TableBoundaryCandidate] = []

        try:
            if self.config.use_region_detector_candidates:
                candidates.extend(
                    self._detect_from_region_detector(page_raw)
                )
        except Exception as exc:
            warnings.append(f"Region table candidate detection failed: {exc}")

        try:
            if self.config.use_drawing_line_candidates:
                candidates.extend(
                    self._detect_from_drawing_lines(page_raw)
                )
        except Exception as exc:
            warnings.append(f"Drawing-line table candidate detection failed: {exc}")

        try:
            if self.config.use_text_alignment_candidates:
                candidates.extend(
                    self._detect_from_text_alignment(page_raw)
                )
        except Exception as exc:
            warnings.append(f"Text-alignment table candidate detection failed: {exc}")

        candidates = self._filter_candidates(candidates)

        if self.config.merge_overlapping_candidates:
            candidates = self._merge_overlapping_candidates(candidates)

        candidates = self._sort_candidates(candidates)

        page_raw.metadata.setdefault("table_boundary_detector", {})
        page_raw.metadata["table_boundary_detector"] = {
            "processor": "TableBoundaryDetector",
            "table_candidate_count": len(candidates),
            "table_candidates": [
                candidate.to_dict() for candidate in candidates
            ],
            "summary": self._build_summary(candidates),
            "config": {
                "use_region_detector_candidates": self.config.use_region_detector_candidates,
                "use_drawing_line_candidates": self.config.use_drawing_line_candidates,
                "use_text_alignment_candidates": self.config.use_text_alignment_candidates,
                "min_horizontal_lines": self.config.min_horizontal_lines,
                "min_vertical_lines": self.config.min_vertical_lines,
                "min_text_lines_for_table": self.config.min_text_lines_for_table,
                "min_columns_for_text_table": self.config.min_columns_for_text_table,
            },
        }

        if warnings:
            page_raw.warnings.extend(warnings)

        return page_raw

    def _detect_from_region_detector(
        self,
        page_raw: PageRaw,
    ) -> List[TableBoundaryCandidate]:
        """
        Reuse table candidate regions from RegionDetector.
        """

        candidates: List[TableBoundaryCandidate] = []

        region_meta = page_raw.metadata.get("region_detector", {})
        regions = region_meta.get("detected_regions", [])

        for index, region in enumerate(regions):
            if region.get("region_type") != "table_candidate_region":
                continue

            bbox = region.get("bbox")

            if not bbox:
                continue

            candidates.append(
                TableBoundaryCandidate(
                    table_id=make_id("table"),
                    page_number=page_raw.page_number,
                    bbox=self._pad_bbox(bbox, self.config.table_padding),
                    detection_method="region_detector",
                    confidence=float(region.get("confidence", 0.60)),
                    source="region_detector.detected_regions",
                    source_object_ids=region.get("source_object_ids", []),
                    text_preview=self._preview_text(region.get("text") or ""),
                    metadata={
                        "source_index": index,
                        "region_id": region.get("region_id"),
                        "region_metadata": region.get("metadata", {}),
                    },
                )
            )

        return candidates

    def _detect_from_drawing_lines(
        self,
        page_raw: PageRaw,
    ) -> List[TableBoundaryCandidate]:
        """
        Detect table candidates from horizontal and vertical drawing lines.
        """

        drawing_groups = self._collect_drawing_line_groups(page_raw)

        if not drawing_groups:
            return []

        horizontal_groups = [
            group for group in drawing_groups
            if group.get("orientation") == "horizontal" and group.get("bbox")
        ]

        vertical_groups = [
            group for group in drawing_groups
            if group.get("orientation") == "vertical" and group.get("bbox")
        ]

        if len(horizontal_groups) < self.config.min_horizontal_lines:
            return []

        if len(vertical_groups) < self.config.min_vertical_lines:
            return []

        table_groups = self._match_horizontal_vertical_groups(
            horizontal_groups=horizontal_groups,
            vertical_groups=vertical_groups,
        )

        candidates: List[TableBoundaryCandidate] = []

        for group_index, group in enumerate(table_groups):
            bbox = self._merge_bboxes(
                [item.get("bbox") for item in group]
            )

            if not bbox:
                continue

            source_ids: List[str] = []

            for item in group:
                source_ids.extend(item.get("source_drawing_ids", []))

            candidates.append(
                TableBoundaryCandidate(
                    table_id=make_id("table"),
                    page_number=page_raw.page_number,
                    bbox=self._pad_bbox(bbox, self.config.table_padding),
                    detection_method="drawing_lines",
                    confidence=0.78,
                    source="object_merger.merged_drawing_lines",
                    source_object_ids=source_ids,
                    text_preview="",
                    metadata={
                        "group_index": group_index,
                        "line_group_count": len(group),
                        "horizontal_group_count": len(
                            [x for x in group if x.get("orientation") == "horizontal"]
                        ),
                        "vertical_group_count": len(
                            [x for x in group if x.get("orientation") == "vertical"]
                        ),
                    },
                )
            )

        return candidates

    def _collect_drawing_line_groups(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        """
        Collect merged drawing lines from ObjectMerger.
        Fallback to raw drawing objects if merged groups are not available.
        """

        object_merger = page_raw.metadata.get("object_merger", {})
        merged_groups = object_merger.get("merged_drawing_lines", [])

        if merged_groups:
            return [
                group for group in merged_groups
                if group.get("bbox")
            ]

        groups: List[Dict[str, Any]] = []

        for drawing in page_raw.drawings:
            bbox = drawing.bbox

            if not bbox:
                continue

            orientation = None

            if self._is_horizontal_line(bbox):
                orientation = "horizontal"
            elif self._is_vertical_line(bbox):
                orientation = "vertical"

            if not orientation:
                continue

            groups.append(
                {
                    "group_id": drawing.drawing_id,
                    "orientation": orientation,
                    "bbox": bbox,
                    "line_count": 1,
                    "source_drawing_ids": [drawing.drawing_id],
                    "likely_table_structure": True,
                }
            )

        return groups

    def _match_horizontal_vertical_groups(
        self,
        horizontal_groups: List[Dict[str, Any]],
        vertical_groups: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Match horizontal and vertical lines that spatially overlap.
        """

        all_groups = horizontal_groups + vertical_groups
        matched: List[List[Dict[str, Any]]] = []
        used_ids = set()

        for h_group in horizontal_groups:
            if h_group.get("group_id") in used_ids:
                continue

            current_group = [h_group]
            h_bbox = h_group.get("bbox")

            for v_group in vertical_groups:
                v_bbox = v_group.get("bbox")

                if not h_bbox or not v_bbox:
                    continue

                if self._bboxes_intersect_or_near(h_bbox, v_bbox):
                    current_group.append(v_group)

            if len(current_group) >= (
                self.config.min_horizontal_lines + self.config.min_vertical_lines
            ):
                for item in current_group:
                    used_ids.add(item.get("group_id"))

                matched.append(current_group)

        if not matched:
            bbox = self._merge_bboxes(
                [group.get("bbox") for group in all_groups]
            )

            if bbox:
                matched.append(all_groups)

        return matched

    def _detect_from_text_alignment(
        self,
        page_raw: PageRaw,
    ) -> List[TableBoundaryCandidate]:
        """
        Detect table-like regions from aligned text lines.

        Useful for borderless tables.
        """

        lines = [
            line for line in page_raw.text_lines
            if line.bbox and (line.normalized_text or line.text or "").strip()
        ]

        if len(lines) < self.config.min_text_lines_for_table:
            return []

        table_like_lines = []

        for line in lines:
            text = line.normalized_text or line.text or ""

            if self._looks_like_table_line(text):
                table_like_lines.append(line)

        if len(table_like_lines) < self.config.min_text_lines_for_table:
            table_like_lines = self._detect_column_aligned_lines(lines)

        if len(table_like_lines) < self.config.min_text_lines_for_table:
            return []

        line_groups = self._group_lines_into_table_blocks(table_like_lines)

        candidates: List[TableBoundaryCandidate] = []

        for group_index, group in enumerate(line_groups):
            if len(group) < self.config.min_text_lines_for_table:
                continue

            column_count = self._estimate_column_count(group)

            if column_count < self.config.min_columns_for_text_table:
                continue

            bbox = self._merge_bboxes([line.bbox for line in group])

            if not bbox:
                continue

            text = "\n".join(
                (line.normalized_text or line.text or "").strip()
                for line in group
            )

            candidates.append(
                TableBoundaryCandidate(
                    table_id=make_id("table"),
                    page_number=page_raw.page_number,
                    bbox=self._pad_bbox(bbox, self.config.table_padding),
                    detection_method="text_alignment",
                    confidence=0.58,
                    source="page_raw.text_lines",
                    source_object_ids=[line.line_id for line in group],
                    text_preview=self._preview_text(text),
                    metadata={
                        "group_index": group_index,
                        "line_count": len(group),
                        "estimated_column_count": column_count,
                        "method": "borderless_table_heuristic",
                    },
                )
            )

        return candidates

    def _detect_column_aligned_lines(
        self,
        lines: List[Any],
    ) -> List[Any]:
        """
        Detect lines with similar x positions or multiple word groups.
        """

        result = []

        x_positions = [
            round(float(line.bbox[0]) / self.config.text_x_alignment_tolerance)
            for line in lines
            if line.bbox
        ]

        if not x_positions:
            return []

        frequency: Dict[int, int] = {}

        for x in x_positions:
            frequency[x] = frequency.get(x, 0) + 1

        common_x_bins = {
            x for x, count in frequency.items()
            if count >= self.config.min_text_lines_for_table
        }

        for line in lines:
            if not line.bbox:
                continue

            x_bin = round(float(line.bbox[0]) / self.config.text_x_alignment_tolerance)
            text = line.normalized_text or line.text or ""

            if x_bin in common_x_bins and self._has_multiple_text_columns(text):
                result.append(line)

        return result

    def _group_lines_into_table_blocks(
        self,
        lines: List[Any],
    ) -> List[List[Any]]:
        """
        Group vertically close table-like lines.
        """

        sorted_lines = sorted(
            lines,
            key=lambda line: (
                line.bbox[1],
                line.bbox[0],
            ),
        )

        groups: List[List[Any]] = []
        current: List[Any] = []

        for line in sorted_lines:
            if not current:
                current.append(line)
                continue

            previous = current[-1]
            gap = line.bbox[1] - previous.bbox[3]

            if gap <= self.config.text_y_gap_tolerance:
                current.append(line)
            else:
                groups.append(current)
                current = [line]

        if current:
            groups.append(current)

        return groups

    def _estimate_column_count(
        self,
        lines: List[Any],
    ) -> int:
        """
        Estimate number of text columns in a borderless table.
        """

        counts = []

        for line in lines:
            text = line.normalized_text or line.text or ""

            parts = self._split_text_columns(text)

            if parts:
                counts.append(len(parts))

        if not counts:
            return 1

        counts = sorted(counts)
        mid = len(counts) // 2

        if len(counts) % 2 == 1:
            return counts[mid]

        return int(round((counts[mid - 1] + counts[mid]) / 2))

    def _looks_like_table_line(
        self,
        text: str,
    ) -> bool:
        """
        Heuristic for table-like text line.
        """

        if not text:
            return False

        stripped = text.strip()

        if not stripped:
            return False

        if "|" in stripped or "\t" in stripped:
            return True

        if re.search(r"\s{2,}", stripped):
            return True

        digit_count = sum(ch.isdigit() for ch in stripped)
        digit_ratio = digit_count / max(len(stripped), 1)

        if digit_ratio >= 0.20 and self._has_multiple_text_columns(stripped):
            return True

        if re.search(r"\d+[.,]\d+", stripped) and self._has_multiple_text_columns(stripped):
            return True

        return False

    def _has_multiple_text_columns(
        self,
        text: str,
    ) -> bool:
        parts = self._split_text_columns(text)
        return len(parts) >= self.config.min_columns_for_text_table

    def _split_text_columns(
        self,
        text: str,
    ) -> List[str]:
        if not text:
            return []

        text = text.strip()

        if "\t" in text:
            return [
                item.strip()
                for item in text.split("\t")
                if item.strip()
            ]

        if "|" in text:
            return [
                item.strip()
                for item in text.split("|")
                if item.strip()
            ]

        parts = re.split(r"\s{2,}", text)

        return [
            item.strip()
            for item in parts
            if item.strip()
        ]

    def _filter_candidates(
        self,
        candidates: List[TableBoundaryCandidate],
    ) -> List[TableBoundaryCandidate]:
        """
        Remove invalid or too-small candidates.
        """

        valid: List[TableBoundaryCandidate] = []

        for candidate in candidates:
            bbox = candidate.bbox

            if not bbox:
                continue

            width = self._bbox_width(bbox)
            height = self._bbox_height(bbox)

            if width < self.config.min_table_width:
                continue

            if height < self.config.min_table_height:
                continue

            valid.append(candidate)

        return valid

    def _merge_overlapping_candidates(
        self,
        candidates: List[TableBoundaryCandidate],
    ) -> List[TableBoundaryCandidate]:
        """
        Merge duplicated table candidates from multiple detection methods.
        """

        if not candidates:
            return []

        sorted_candidates = self._sort_candidates(candidates)
        groups: List[List[TableBoundaryCandidate]] = []

        for candidate in sorted_candidates:
            placed = False

            for group in groups:
                group_bbox = self._merge_bboxes(
                    [item.bbox for item in group]
                )

                if self._bbox_overlap_ratio(candidate.bbox, group_bbox) >= self.config.candidate_overlap_threshold:
                    group.append(candidate)
                    placed = True
                    break

            if not placed:
                groups.append([candidate])

        merged: List[TableBoundaryCandidate] = []

        for group_index, group in enumerate(groups):
            if len(group) == 1:
                merged.append(group[0])
                continue

            bbox = self._merge_bboxes([item.bbox for item in group])
            source_ids: List[str] = []
            methods: List[str] = []
            previews: List[str] = []

            confidence = 0.0

            for item in group:
                source_ids.extend(item.source_object_ids or [])
                methods.append(item.detection_method)

                if item.text_preview:
                    previews.append(item.text_preview)

                confidence = max(confidence, item.confidence)

            if len(set(methods)) > 1:
                confidence = min(confidence + 0.08, 0.95)

            merged.append(
                TableBoundaryCandidate(
                    table_id=make_id("table"),
                    page_number=group[0].page_number,
                    bbox=bbox,
                    detection_method="merged",
                    confidence=round(confidence, 4),
                    source="table_boundary_detector.merge_overlapping_candidates",
                    source_object_ids=list(dict.fromkeys(source_ids)),
                    text_preview=self._preview_text("\n".join(previews)),
                    metadata={
                        "group_index": group_index,
                        "merged_candidate_count": len(group),
                        "detection_methods": sorted(set(methods)),
                    },
                )
            )

        return merged

    def _sort_candidates(
        self,
        candidates: List[TableBoundaryCandidate],
    ) -> List[TableBoundaryCandidate]:
        return sorted(
            candidates,
            key=lambda item: (
                item.bbox[1] if item.bbox else 0,
                item.bbox[0] if item.bbox else 0,
            ),
        )

    def _build_summary(
        self,
        candidates: List[TableBoundaryCandidate],
    ) -> Dict[str, Any]:
        by_method: Dict[str, int] = {}

        for candidate in candidates:
            by_method[candidate.detection_method] = (
                by_method.get(candidate.detection_method, 0) + 1
            )

        return {
            "has_table_candidates": len(candidates) > 0,
            "table_candidate_count": len(candidates),
            "by_detection_method": by_method,
        }

    def _bboxes_intersect_or_near(
        self,
        bbox_a: Optional[List[float]],
        bbox_b: Optional[List[float]],
    ) -> bool:
        if not bbox_a or not bbox_b:
            return False

        a = self._pad_bbox(bbox_a, self.config.line_overlap_tolerance)
        b = self._pad_bbox(bbox_b, self.config.line_overlap_tolerance)

        return not (
            a[2] < b[0]
            or a[0] > b[2]
            or a[3] < b[1]
            or a[1] > b[3]
        )

    def _bbox_overlap_ratio(
        self,
        bbox_a: Optional[List[float]],
        bbox_b: Optional[List[float]],
    ) -> float:
        if not bbox_a or not bbox_b:
            return 0.0

        x0 = max(bbox_a[0], bbox_b[0])
        y0 = max(bbox_a[1], bbox_b[1])
        x1 = min(bbox_a[2], bbox_b[2])
        y1 = min(bbox_a[3], bbox_b[3])

        inter_width = max(x1 - x0, 0.0)
        inter_height = max(y1 - y0, 0.0)
        intersection = inter_width * inter_height

        if intersection <= 0:
            return 0.0

        area_a = self._bbox_area(bbox_a)
        area_b = self._bbox_area(bbox_b)

        smaller_area = min(area_a, area_b)

        if smaller_area <= 0:
            return 0.0

        return intersection / smaller_area

    def _merge_bboxes(
        self,
        bboxes: List[Optional[List[float]]],
    ) -> Optional[List[float]]:
        valid = [
            bbox for bbox in bboxes
            if bbox and len(bbox) == 4
        ]

        if not valid:
            return None

        return [
            min(float(bbox[0]) for bbox in valid),
            min(float(bbox[1]) for bbox in valid),
            max(float(bbox[2]) for bbox in valid),
            max(float(bbox[3]) for bbox in valid),
        ]

    def _pad_bbox(
        self,
        bbox: Optional[List[float]],
        padding: float,
    ) -> Optional[List[float]]:
        if not bbox:
            return None

        return [
            float(bbox[0]) - padding,
            float(bbox[1]) - padding,
            float(bbox[2]) + padding,
            float(bbox[3]) + padding,
        ]

    def _bbox_width(
        self,
        bbox: Optional[List[float]],
    ) -> float:
        if not bbox:
            return 0.0

        return max(float(bbox[2]) - float(bbox[0]), 0.0)

    def _bbox_height(
        self,
        bbox: Optional[List[float]],
    ) -> float:
        if not bbox:
            return 0.0

        return max(float(bbox[3]) - float(bbox[1]), 0.0)

    def _bbox_area(
        self,
        bbox: Optional[List[float]],
    ) -> float:
        if not bbox:
            return 0.0

        return self._bbox_width(bbox) * self._bbox_height(bbox)

    def _is_horizontal_line(
        self,
        bbox: Optional[List[float]],
    ) -> bool:
        if not bbox:
            return False

        return self._bbox_width(bbox) > 10 and self._bbox_height(bbox) <= 2

    def _is_vertical_line(
        self,
        bbox: Optional[List[float]],
    ) -> bool:
        if not bbox:
            return False

        return self._bbox_height(bbox) > 10 and self._bbox_width(bbox) <= 2

    def _preview_text(
        self,
        text: str,
    ) -> str:
        if not self.config.include_text_preview:
            return ""

        if not text:
            return ""

        return text[: self.config.text_preview_chars]


def detect_table_boundaries(
    page_raw: PageRaw,
) -> PageRaw:
    """
    Colab helper function.
    """

    detector = TableBoundaryDetector()
    return detector.process(page_raw)
