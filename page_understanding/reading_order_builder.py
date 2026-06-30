"""
reading_order_builder.py

Production V1 - Colab Ready

Purpose
-------
Build page-level reading order from normalized and merged objects.

Input
-----
PageRaw after:
- PageExtractionPipeline
- ObjectNormalizer
- ObjectMerger

Output
------
PageRaw with reading order metadata:
- reading_order_items
- reading_order_text
- column_profile
- layout_flow

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

Important
---------
This module only builds reading order.
It does not classify header/footer/table/figure semantically.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import PageRaw


@dataclass
class ReadingOrderBuilderConfig:
    """
    Configuration for ReadingOrderBuilder.
    """

    use_merged_paragraphs: bool = True
    use_merged_lines: bool = True
    include_images: bool = False
    include_drawings: bool = False

    detect_columns: bool = True
    max_columns: int = 3

    min_items_for_column_detection: int = 6
    column_gap_threshold: float = 40.0
    column_x_tolerance: float = 35.0

    top_to_bottom_tolerance: float = 5.0

    include_empty_text: bool = False


@dataclass
class ReadingOrderItem:
    """
    One item in reading order.
    """

    order: int
    item_id: str
    item_type: str
    text: str
    bbox: Optional[List[float]]
    page_number: int

    column_index: int = 0
    source: str = "unknown"
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class ReadingOrderBuilder:
    """
    Build reading order from PageRaw objects.
    """

    def __init__(self, config: Optional[ReadingOrderBuilderConfig] = None):
        self.config = config or ReadingOrderBuilderConfig()

    def process(self, page_raw: PageRaw) -> PageRaw:
        """
        Build reading order and attach result to page_raw.metadata.
        """

        warnings: List[str] = []

        try:
            candidate_items = self._collect_candidate_items(page_raw)
            column_profile = self._build_column_profile(page_raw, candidate_items)
            ordered_items = self._sort_items_by_reading_order(
                items=candidate_items,
                column_profile=column_profile,
            )

            reading_order_text = self._build_reading_order_text(ordered_items)

            page_raw.metadata.setdefault("reading_order_builder", {})
            page_raw.metadata["reading_order_builder"] = {
                "processor": "ReadingOrderBuilder",
                "reading_order_item_count": len(ordered_items),
                "reading_order_items": [
                    item.to_dict() for item in ordered_items
                ],
                "reading_order_text": reading_order_text,
                "column_profile": column_profile,
                "layout_flow": self._classify_layout_flow(column_profile),
                "config": {
                    "use_merged_paragraphs": self.config.use_merged_paragraphs,
                    "use_merged_lines": self.config.use_merged_lines,
                    "include_images": self.config.include_images,
                    "include_drawings": self.config.include_drawings,
                    "detect_columns": self.config.detect_columns,
                    "max_columns": self.config.max_columns,
                },
            }

        except Exception as exc:
            warnings.append(f"Failed to build reading order: {exc}")

        if warnings:
            page_raw.warnings.extend(warnings)

        return page_raw

    def _collect_candidate_items(
        self,
        page_raw: PageRaw,
    ) -> List[ReadingOrderItem]:
        """
        Collect text/image/drawing objects for reading order.
        """

        items: List[ReadingOrderItem] = []

        text_items = self._collect_text_items(page_raw)
        items.extend(text_items)

        if self.config.include_images:
            items.extend(self._collect_image_items(page_raw))

        if self.config.include_drawings:
            items.extend(self._collect_drawing_items(page_raw))

        items = [
            item for item in items
            if item.bbox is not None
        ]

        return items

    def _collect_text_items(
        self,
        page_raw: PageRaw,
    ) -> List[ReadingOrderItem]:
        """
        Collect text objects.

        Priority:
        1. merged_paragraphs from ObjectMerger
        2. merged_text_lines from ObjectMerger
        3. original text_blocks
        4. original text_lines
        """

        merger_meta = page_raw.metadata.get("object_merger", {})
        items: List[ReadingOrderItem] = []

        if self.config.use_merged_paragraphs:
            paragraphs = merger_meta.get("merged_paragraphs", [])

            if paragraphs:
                for index, paragraph in enumerate(paragraphs):
                    text = paragraph.get("normalized_text") or paragraph.get("text") or ""

                    if not self.config.include_empty_text and not text.strip():
                        continue

                    items.append(
                        ReadingOrderItem(
                            order=-1,
                            item_id=paragraph.get("block_id") or f"paragraph_{index}",
                            item_type="text_paragraph",
                            text=text,
                            bbox=paragraph.get("bbox"),
                            page_number=page_raw.page_number,
                            source="object_merger.merged_paragraphs",
                            metadata={
                                "source_index": index,
                                "block_type": paragraph.get("block_type"),
                                "line_count": paragraph.get("metadata", {}).get("line_count"),
                            },
                        )
                    )

                if items:
                    return items

        if self.config.use_merged_lines:
            merged_lines = merger_meta.get("merged_text_lines", [])

            if merged_lines:
                for index, line in enumerate(merged_lines):
                    text = line.get("normalized_text") or line.get("text") or ""

                    if not self.config.include_empty_text and not text.strip():
                        continue

                    items.append(
                        ReadingOrderItem(
                            order=-1,
                            item_id=line.get("line_id") or f"merged_line_{index}",
                            item_type="text_line",
                            text=text,
                            bbox=line.get("bbox"),
                            page_number=page_raw.page_number,
                            source="object_merger.merged_text_lines",
                            metadata={
                                "source_index": index,
                            },
                        )
                    )

                if items:
                    return items

        if page_raw.text_blocks:
            for index, block in enumerate(page_raw.text_blocks):
                text = block.normalized_text or block.text or ""

                if not self.config.include_empty_text and not text.strip():
                    continue

                items.append(
                    ReadingOrderItem(
                        order=-1,
                        item_id=block.block_id,
                        item_type="text_block",
                        text=text,
                        bbox=block.bbox,
                        page_number=page_raw.page_number,
                        source="page_raw.text_blocks",
                        metadata={
                            "source_index": index,
                            "block_type": block.block_type,
                            "line_count": len(block.lines or []),
                        },
                    )
                )

            if items:
                return items

        for index, line in enumerate(page_raw.text_lines):
            text = line.normalized_text or line.text or ""

            if not self.config.include_empty_text and not text.strip():
                continue

            items.append(
                ReadingOrderItem(
                    order=-1,
                    item_id=line.line_id,
                    item_type="text_line",
                    text=text,
                    bbox=line.bbox,
                    page_number=page_raw.page_number,
                    source="page_raw.text_lines",
                    metadata={
                        "source_index": index,
                    },
                )
            )

        return items

    def _collect_image_items(
        self,
        page_raw: PageRaw,
    ) -> List[ReadingOrderItem]:
        items: List[ReadingOrderItem] = []

        for index, image in enumerate(page_raw.images):
            items.append(
                ReadingOrderItem(
                    order=-1,
                    item_id=image.image_id,
                    item_type="image",
                    text="",
                    bbox=image.bbox,
                    page_number=page_raw.page_number,
                    source="page_raw.images",
                    metadata={
                        "source_index": index,
                        "width": image.width,
                        "height": image.height,
                        "ext": image.ext,
                    },
                )
            )

        return items

    def _collect_drawing_items(
        self,
        page_raw: PageRaw,
    ) -> List[ReadingOrderItem]:
        items: List[ReadingOrderItem] = []

        for index, drawing in enumerate(page_raw.drawings):
            items.append(
                ReadingOrderItem(
                    order=-1,
                    item_id=drawing.drawing_id,
                    item_type="drawing",
                    text="",
                    bbox=drawing.bbox,
                    page_number=page_raw.page_number,
                    source="page_raw.drawings",
                    metadata={
                        "source_index": index,
                        "drawing_type": drawing.drawing_type,
                    },
                )
            )

        return items

    def _build_column_profile(
        self,
        page_raw: PageRaw,
        items: List[ReadingOrderItem],
    ) -> Dict[str, Any]:
        """
        Detect simple column layout.
        """

        if not self.config.detect_columns:
            return self._single_column_profile(page_raw)

        if len(items) < self.config.min_items_for_column_detection:
            return self._single_column_profile(page_raw)

        valid_items = [
            item for item in items
            if item.bbox is not None
        ]

        if not valid_items:
            return self._single_column_profile(page_raw)

        x_centers = sorted(
            [
                self._bbox_center_x(item.bbox)
                for item in valid_items
            ]
        )

        gaps = []

        for previous_x, current_x in zip(x_centers, x_centers[1:]):
            gaps.append(current_x - previous_x)

        significant_gaps = [
            gap for gap in gaps
            if gap >= self.config.column_gap_threshold
        ]

        if not significant_gaps:
            return self._single_column_profile(page_raw)

        clusters = self._cluster_items_by_x(valid_items)

        if len(clusters) <= 1:
            return self._single_column_profile(page_raw)

        if len(clusters) > self.config.max_columns:
            clusters = clusters[: self.config.max_columns]

        column_ranges = []

        for column_index, cluster in enumerate(clusters):
            bboxes = [item.bbox for item in cluster if item.bbox]
            x0 = min(bbox[0] for bbox in bboxes)
            x1 = max(bbox[2] for bbox in bboxes)

            column_ranges.append(
                {
                    "column_index": column_index,
                    "x0": round(x0, 4),
                    "x1": round(x1, 4),
                    "item_count": len(cluster),
                }
            )

        column_ranges.sort(key=lambda column: column["x0"])

        return {
            "column_count": len(column_ranges),
            "columns": column_ranges,
            "detection_method": "x_cluster",
            "is_multi_column": len(column_ranges) > 1,
        }

    def _single_column_profile(
        self,
        page_raw: PageRaw,
    ) -> Dict[str, Any]:
        return {
            "column_count": 1,
            "columns": [
                {
                    "column_index": 0,
                    "x0": 0.0,
                    "x1": float(page_raw.width or 0),
                    "item_count": None,
                }
            ],
            "detection_method": "single_column_default",
            "is_multi_column": False,
        }

    def _cluster_items_by_x(
        self,
        items: List[ReadingOrderItem],
    ) -> List[List[ReadingOrderItem]]:
        """
        Cluster items by x-center.
        """

        sorted_items = sorted(
            items,
            key=lambda item: self._bbox_center_x(item.bbox),
        )

        clusters: List[List[ReadingOrderItem]] = []

        for item in sorted_items:
            item_x = self._bbox_center_x(item.bbox)
            placed = False

            for cluster in clusters:
                cluster_x = sum(
                    self._bbox_center_x(x.bbox) for x in cluster
                ) / len(cluster)

                if abs(item_x - cluster_x) <= self.config.column_x_tolerance:
                    cluster.append(item)
                    placed = True
                    break

            if not placed:
                clusters.append([item])

        clusters = sorted(
            clusters,
            key=lambda cluster: min(item.bbox[0] for item in cluster if item.bbox),
        )

        return clusters

    def _sort_items_by_reading_order(
        self,
        items: List[ReadingOrderItem],
        column_profile: Dict[str, Any],
    ) -> List[ReadingOrderItem]:
        """
        Sort items into reading order.

        Single column:
            top-to-bottom, left-to-right

        Multi-column:
            column-by-column, then top-to-bottom
        """

        if not items:
            return []

        column_count = column_profile.get("column_count", 1)

        if column_count <= 1:
            sorted_items = sorted(
                items,
                key=lambda item: (
                    self._round_for_order(item.bbox[1]),
                    item.bbox[0],
                ),
            )
        else:
            sorted_items = []

            columns = column_profile.get("columns", [])

            for column in columns:
                column_index = column.get("column_index", 0)
                x0 = column.get("x0", 0.0)
                x1 = column.get("x1", 0.0)

                column_items = [
                    item for item in items
                    if self._item_belongs_to_column(item, x0, x1)
                ]

                for item in column_items:
                    item.column_index = column_index

                column_items = sorted(
                    column_items,
                    key=lambda item: (
                        self._round_for_order(item.bbox[1]),
                        item.bbox[0],
                    ),
                )

                sorted_items.extend(column_items)

        for order, item in enumerate(sorted_items, start=1):
            item.order = order

        return sorted_items

    def _item_belongs_to_column(
        self,
        item: ReadingOrderItem,
        x0: float,
        x1: float,
    ) -> bool:
        if not item.bbox:
            return False

        center_x = self._bbox_center_x(item.bbox)
        return x0 <= center_x <= x1

    def _build_reading_order_text(
        self,
        items: List[ReadingOrderItem],
    ) -> str:
        """
        Build plain text according to reading order.
        """

        parts = []

        for item in items:
            text = item.text or ""

            if text.strip():
                parts.append(text.strip())

        return "\n".join(parts)

    def _classify_layout_flow(
        self,
        column_profile: Dict[str, Any],
    ) -> str:
        if column_profile.get("is_multi_column"):
            return "multi_column"

        return "single_column"

    def _bbox_center_x(
        self,
        bbox: Optional[List[float]],
    ) -> float:
        if not bbox:
            return 0.0

        return (float(bbox[0]) + float(bbox[2])) / 2.0

    def _round_for_order(
        self,
        value: float,
    ) -> float:
        """
        Round y value to reduce noise in same-line ordering.
        """

        tolerance = self.config.top_to_bottom_tolerance

        if tolerance <= 0:
            return value

        return round(value / tolerance) * tolerance


def build_reading_order(page_raw: PageRaw) -> PageRaw:
    """
    Colab helper function.
    """

    builder = ReadingOrderBuilder()
    return builder.process(page_raw)
