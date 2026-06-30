"""
object_merger.py

Production V1 - Colab Ready

Purpose
-------
Merge low-level physical objects into higher-level layout objects.

Input
-----
PageRaw from:
- page_extraction_pipeline.py
- object_normalizer.py

Output
------
PageRaw with merged object metadata:
- merged text lines
- merged paragraph blocks
- merged drawing/table-line groups

Flow position
-------------
PageRaw
    ↓
ObjectNormalizer
    ↓
ObjectMerger
    ↓
ReadingOrderBuilder
    ↓
RegionDetector
    ↓
PageDocument
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import (
    PageRaw,
    TextLineRaw,
    TextBlockRaw,
    TextSpanRaw,
    DrawingRaw,
    make_id,
    merge_bboxes,
)


@dataclass
class ObjectMergerConfig:
    """
    Configuration for ObjectMerger.
    """

    merge_text_lines: bool = True
    merge_paragraphs: bool = True
    merge_table_lines: bool = True

    same_line_y_tolerance: float = 3.0
    same_line_gap_tolerance: float = 8.0

    paragraph_y_gap_ratio: float = 1.8
    paragraph_x_tolerance: float = 12.0

    line_merge_tolerance: float = 2.0
    min_table_line_length: float = 20.0

    keep_original_objects: bool = True


class ObjectMerger:
    """
    Merge normalized page objects.
    """

    def __init__(self, config: Optional[ObjectMergerConfig] = None):
        self.config = config or ObjectMergerConfig()

    def process(self, page_raw: PageRaw) -> PageRaw:
        """
        Merge objects inside PageRaw.
        """

        warnings: List[str] = []
        page_raw.metadata.setdefault("object_merger", {})

        try:
            if self.config.merge_text_lines:
                merged_lines = self._merge_text_lines(page_raw.text_lines)
                page_raw.metadata["object_merger"]["merged_text_lines"] = [
                    line.to_dict() for line in merged_lines
                ]
                page_raw.metadata["object_merger"]["merged_text_line_count"] = len(
                    merged_lines
                )
        except Exception as exc:
            warnings.append(f"Failed to merge text lines: {exc}")

        try:
            if self.config.merge_paragraphs:
                merged_lines = self._merge_text_lines(page_raw.text_lines)
                paragraphs = self._merge_lines_to_paragraph_blocks(
                    lines=merged_lines,
                    page_number=page_raw.page_number,
                )

                page_raw.metadata["object_merger"]["merged_paragraphs"] = [
                    paragraph.to_dict() for paragraph in paragraphs
                ]
                page_raw.metadata["object_merger"]["merged_paragraph_count"] = len(
                    paragraphs
                )
        except Exception as exc:
            warnings.append(f"Failed to merge paragraphs: {exc}")

        try:
            if self.config.merge_table_lines:
                drawing_groups = self._merge_drawing_lines(page_raw.drawings)
                page_raw.metadata["object_merger"]["merged_drawing_lines"] = (
                    drawing_groups
                )
                page_raw.metadata["object_merger"]["merged_drawing_line_count"] = len(
                    drawing_groups
                )
        except Exception as exc:
            warnings.append(f"Failed to merge drawing lines: {exc}")

        page_raw.metadata["object_merger"].update(
            {
                "processor": "ObjectMerger",
                "config": {
                    "merge_text_lines": self.config.merge_text_lines,
                    "merge_paragraphs": self.config.merge_paragraphs,
                    "merge_table_lines": self.config.merge_table_lines,
                    "same_line_y_tolerance": self.config.same_line_y_tolerance,
                    "same_line_gap_tolerance": self.config.same_line_gap_tolerance,
                    "paragraph_y_gap_ratio": self.config.paragraph_y_gap_ratio,
                    "paragraph_x_tolerance": self.config.paragraph_x_tolerance,
                    "line_merge_tolerance": self.config.line_merge_tolerance,
                    "min_table_line_length": self.config.min_table_line_length,
                },
            }
        )

        if warnings:
            page_raw.warnings.extend(warnings)

        return page_raw

    def _merge_text_lines(self, lines: List[TextLineRaw]) -> List[TextLineRaw]:
        """
        Merge text lines that are visually on the same baseline.
        """

        valid_lines = [
            line
            for line in lines
            if line.bbox and (line.normalized_text or "").strip()
        ]

        if not valid_lines:
            return []

        sorted_lines = sorted(
            valid_lines,
            key=lambda line: (
                self._bbox_center_y(line.bbox),
                line.bbox[0],
            ),
        )

        groups: List[List[TextLineRaw]] = []

        for line in sorted_lines:
            line_y = self._bbox_center_y(line.bbox)
            placed = False

            for group in groups:
                group_y = self._average_center_y(group)

                if abs(line_y - group_y) <= self.config.same_line_y_tolerance:
                    group.append(line)
                    placed = True
                    break

            if not placed:
                groups.append([line])

        merged_lines: List[TextLineRaw] = []

        for group in groups:
            group = sorted(group, key=lambda line: line.bbox[0])
            subgroups = self._split_line_group_by_gap(group)

            for subgroup in subgroups:
                merged = self._merge_line_group(subgroup)

                if merged:
                    merged_lines.append(merged)

        merged_lines.sort(
            key=lambda line: (
                line.bbox[1] if line.bbox else 0,
                line.bbox[0] if line.bbox else 0,
            )
        )

        return merged_lines

    def _split_line_group_by_gap(
        self,
        group: List[TextLineRaw],
    ) -> List[List[TextLineRaw]]:
        """
        Split same-baseline lines if horizontal gap is too large.
        """

        if not group:
            return []

        if len(group) == 1:
            return [group]

        result: List[List[TextLineRaw]] = []
        current: List[TextLineRaw] = [group[0]]

        for previous_line, current_line in zip(group, group[1:]):
            previous_bbox = previous_line.bbox
            current_bbox = current_line.bbox

            if not previous_bbox or not current_bbox:
                current.append(current_line)
                continue

            gap = current_bbox[0] - previous_bbox[2]

            if gap <= self.config.same_line_gap_tolerance:
                current.append(current_line)
            else:
                result.append(current)
                current = [current_line]

        if current:
            result.append(current)

        return result

    def _merge_line_group(
        self,
        group: List[TextLineRaw],
    ) -> Optional[TextLineRaw]:
        """
        Merge one group of same-baseline lines.
        """

        if not group:
            return None

        if len(group) == 1:
            line = group[0]

            return TextLineRaw(
                line_id=make_id("mline"),
                page_number=line.page_number,
                text=line.text,
                normalized_text=line.normalized_text,
                bbox=line.bbox,
                block_no=line.block_no,
                line_no=line.line_no,
                spans=line.spans,
                metadata={
                    "merged": True,
                    "merge_type": "single_line",
                    "source_line_ids": [line.line_id],
                    "source_count": 1,
                },
            )

        spans: List[TextSpanRaw] = []

        for line in group:
            spans.extend(line.spans or [])

        text = " ".join(
            (line.text or "").strip()
            for line in group
            if (line.text or "").strip()
        )

        normalized_text = " ".join(
            (line.normalized_text or "").strip()
            for line in group
            if (line.normalized_text or "").strip()
        )

        bbox = merge_bboxes([line.bbox for line in group])

        return TextLineRaw(
            line_id=make_id("mline"),
            page_number=group[0].page_number,
            text=text,
            normalized_text=normalized_text,
            bbox=bbox,
            block_no=group[0].block_no,
            line_no=group[0].line_no,
            spans=spans,
            metadata={
                "merged": True,
                "merge_type": "same_baseline",
                "source_line_ids": [line.line_id for line in group],
                "source_count": len(group),
            },
        )

    def _merge_lines_to_paragraph_blocks(
        self,
        lines: List[TextLineRaw],
        page_number: int,
    ) -> List[TextBlockRaw]:
        """
        Merge visual text lines into paragraph-like blocks.
        """

        valid_lines = [
            line
            for line in lines
            if line.bbox and (line.normalized_text or "").strip()
        ]

        if not valid_lines:
            return []

        sorted_lines = sorted(
            valid_lines,
            key=lambda line: (
                line.bbox[1],
                line.bbox[0],
            ),
        )

        heights = [
            self._bbox_height(line.bbox)
            for line in sorted_lines
            if line.bbox
        ]

        median_height = self._median(heights) or 10.0
        max_gap = median_height * self.config.paragraph_y_gap_ratio

        paragraphs: List[List[TextLineRaw]] = []
        current: List[TextLineRaw] = []

        for line in sorted_lines:
            if not current:
                current.append(line)
                continue

            previous_line = current[-1]

            vertical_gap = line.bbox[1] - previous_line.bbox[3]
            x_shift = abs(line.bbox[0] - previous_line.bbox[0])

            same_paragraph = (
                vertical_gap <= max_gap
                and x_shift <= self.config.paragraph_x_tolerance
            )

            if same_paragraph:
                current.append(line)
            else:
                paragraphs.append(current)
                current = [line]

        if current:
            paragraphs.append(current)

        blocks: List[TextBlockRaw] = []

        for paragraph_index, paragraph_lines in enumerate(paragraphs):
            block = self._create_paragraph_block(
                lines=paragraph_lines,
                page_number=page_number,
                paragraph_index=paragraph_index,
            )

            if block:
                blocks.append(block)

        return blocks

    def _create_paragraph_block(
        self,
        lines: List[TextLineRaw],
        page_number: int,
        paragraph_index: int,
    ) -> Optional[TextBlockRaw]:
        """
        Create one merged paragraph block.
        """

        if not lines:
            return None

        text = "\n".join(
            (line.text or "").strip()
            for line in lines
            if (line.text or "").strip()
        )

        normalized_text = "\n".join(
            (line.normalized_text or "").strip()
            for line in lines
            if (line.normalized_text or "").strip()
        )

        bbox = merge_bboxes([line.bbox for line in lines])

        return TextBlockRaw(
            block_id=make_id("para"),
            page_number=page_number,
            text=text,
            normalized_text=normalized_text,
            bbox=bbox,
            block_no=paragraph_index,
            block_type="merged_paragraph",
            lines=lines,
            metadata={
                "merged": True,
                "merge_type": "paragraph",
                "source_line_ids": [line.line_id for line in lines],
                "line_count": len(lines),
            },
        )

    def _merge_drawing_lines(
        self,
        drawings: List[DrawingRaw],
    ) -> List[Dict[str, Any]]:
        """
        Merge line-like drawings into horizontal and vertical groups.
        """

        line_drawings = [
            drawing
            for drawing in drawings
            if drawing.bbox and self._is_table_line_candidate(drawing)
        ]

        horizontal_lines: List[DrawingRaw] = []
        vertical_lines: List[DrawingRaw] = []

        for drawing in line_drawings:
            bbox = drawing.bbox

            if self._is_horizontal_line(bbox):
                horizontal_lines.append(drawing)
            elif self._is_vertical_line(bbox):
                vertical_lines.append(drawing)

        horizontal_groups = self._group_parallel_lines(
            lines=horizontal_lines,
            orientation="horizontal",
        )

        vertical_groups = self._group_parallel_lines(
            lines=vertical_lines,
            orientation="vertical",
        )

        return horizontal_groups + vertical_groups

    def _group_parallel_lines(
        self,
        lines: List[DrawingRaw],
        orientation: str,
    ) -> List[Dict[str, Any]]:
        """
        Group parallel drawing lines by y-axis or x-axis.
        """

        if not lines:
            return []

        if orientation == "horizontal":
            sorted_lines = sorted(
                lines,
                key=lambda line: (
                    self._bbox_center_y(line.bbox),
                    line.bbox[0],
                ),
            )
            axis_fn = lambda line: self._bbox_center_y(line.bbox)
        else:
            sorted_lines = sorted(
                lines,
                key=lambda line: (
                    self._bbox_center_x(line.bbox),
                    line.bbox[1],
                ),
            )
            axis_fn = lambda line: self._bbox_center_x(line.bbox)

        groups: List[List[DrawingRaw]] = []

        for line in sorted_lines:
            axis_value = axis_fn(line)
            placed = False

            for group in groups:
                group_axis = sum(axis_fn(x) for x in group) / len(group)

                if abs(axis_value - group_axis) <= self.config.line_merge_tolerance:
                    group.append(line)
                    placed = True
                    break

            if not placed:
                groups.append([line])

        result: List[Dict[str, Any]] = []

        for group_index, group in enumerate(groups):
            bbox = merge_bboxes([line.bbox for line in group])

            result.append(
                {
                    "group_id": make_id("dline_group"),
                    "orientation": orientation,
                    "bbox": bbox,
                    "line_count": len(group),
                    "source_drawing_ids": [line.drawing_id for line in group],
                    "likely_table_structure": len(group) >= 1,
                    "group_index": group_index,
                }
            )

        return result

    def _is_table_line_candidate(
        self,
        drawing: DrawingRaw,
    ) -> bool:
        """
        Heuristic to identify drawing lines useful for table detection.
        """

        bbox = drawing.bbox

        if not bbox:
            return False

        if drawing.metadata and drawing.metadata.get("likely_table_line"):
            return True

        width = self._bbox_width(bbox)
        height = self._bbox_height(bbox)

        if self._is_horizontal_line(bbox) and width >= self.config.min_table_line_length:
            return True

        if self._is_vertical_line(bbox) and height >= self.config.min_table_line_length:
            return True

        return False

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

    def _bbox_center_x(
        self,
        bbox: Optional[List[float]],
    ) -> float:
        if not bbox:
            return 0.0

        return (float(bbox[0]) + float(bbox[2])) / 2.0

    def _bbox_center_y(
        self,
        bbox: Optional[List[float]],
    ) -> float:
        if not bbox:
            return 0.0

        return (float(bbox[1]) + float(bbox[3])) / 2.0

    def _average_center_y(
        self,
        lines: List[TextLineRaw],
    ) -> float:
        if not lines:
            return 0.0

        return sum(self._bbox_center_y(line.bbox) for line in lines) / len(lines)

    def _median(
        self,
        values: List[float],
    ) -> Optional[float]:
        values = sorted(values)

        if not values:
            return None

        n = len(values)
        mid = n // 2

        if n % 2 == 1:
            return values[mid]

        return (values[mid - 1] + values[mid]) / 2.0


def merge_page_objects(page_raw: PageRaw) -> PageRaw:
    """
    Colab helper function.
    """

    merger = ObjectMerger()
    return merger.process(page_raw)
