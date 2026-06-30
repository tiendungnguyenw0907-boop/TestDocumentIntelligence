"""
region_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect page regions from normalized objects and reading order.

Input
-----
PageRaw after:
- PageExtractionPipeline
- ObjectNormalizer
- ObjectMerger
- ReadingOrderBuilder

Output
------
PageRaw with region metadata:
- detected_regions
- text_regions
- image_regions
- drawing_regions
- table_candidate_regions
- page_zone_regions

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
HeaderFooterDetector
    ↓
TableBoundaryDetector
    ↓
PageDocument

Important
---------
This module performs layout region detection only.

It does not:
- finalize header/footer classification
- recognize table structure
- recognize figure captions
- build document structure
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import (
    PageRaw,
    make_id,
    merge_bboxes,
)


@dataclass
class RegionDetectorConfig:
    """
    Configuration for RegionDetector.
    """

    detect_page_zones: bool = True
    detect_text_regions: bool = True
    detect_image_regions: bool = True
    detect_drawing_regions: bool = True
    detect_table_candidate_regions: bool = True

    header_zone_ratio: float = 0.12
    footer_zone_ratio: float = 0.88

    min_region_width: float = 5.0
    min_region_height: float = 5.0

    merge_nearby_text_regions: bool = True
    text_region_vertical_gap: float = 12.0
    text_region_x_tolerance: float = 20.0

    table_min_line_count: int = 2
    table_region_padding: float = 4.0

    include_text_preview: bool = True
    text_preview_chars: int = 300


@dataclass
class PageRegion:
    """
    One detected page region.
    """

    region_id: str
    page_number: int
    region_type: str
    bbox: Optional[List[float]]

    source: str = "unknown"
    source_object_ids: List[str] = None
    text: str = ""
    confidence: float = 0.5
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["source_object_ids"] is None:
            data["source_object_ids"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class RegionDetector:
    """
    Detect layout regions from PageRaw.
    """

    def __init__(self, config: Optional[RegionDetectorConfig] = None):
        self.config = config or RegionDetectorConfig()

    def process(self, page_raw: PageRaw) -> PageRaw:
        """
        Detect regions and attach result to page_raw.metadata.
        """

        warnings: List[str] = []
        regions: List[PageRegion] = []

        try:
            if self.config.detect_page_zones:
                regions.extend(self._detect_page_zone_regions(page_raw))
        except Exception as exc:
            warnings.append(f"Failed to detect page zones: {exc}")

        try:
            if self.config.detect_text_regions:
                regions.extend(self._detect_text_regions(page_raw))
        except Exception as exc:
            warnings.append(f"Failed to detect text regions: {exc}")

        try:
            if self.config.detect_image_regions:
                regions.extend(self._detect_image_regions(page_raw))
        except Exception as exc:
            warnings.append(f"Failed to detect image regions: {exc}")

        try:
            if self.config.detect_drawing_regions:
                regions.extend(self._detect_drawing_regions(page_raw))
        except Exception as exc:
            warnings.append(f"Failed to detect drawing regions: {exc}")

        try:
            if self.config.detect_table_candidate_regions:
                regions.extend(self._detect_table_candidate_regions(page_raw))
        except Exception as exc:
            warnings.append(f"Failed to detect table candidate regions: {exc}")

        regions = self._filter_invalid_regions(regions)
        regions = self._assign_region_order(regions)

        page_raw.metadata.setdefault("region_detector", {})
        page_raw.metadata["region_detector"] = {
            "processor": "RegionDetector",
            "region_count": len(regions),
            "detected_regions": [region.to_dict() for region in regions],
            "region_summary": self._summarize_regions(regions),
            "config": {
                "detect_page_zones": self.config.detect_page_zones,
                "detect_text_regions": self.config.detect_text_regions,
                "detect_image_regions": self.config.detect_image_regions,
                "detect_drawing_regions": self.config.detect_drawing_regions,
                "detect_table_candidate_regions": self.config.detect_table_candidate_regions,
                "header_zone_ratio": self.config.header_zone_ratio,
                "footer_zone_ratio": self.config.footer_zone_ratio,
            },
        }

        if warnings:
            page_raw.warnings.extend(warnings)

        return page_raw

    def _detect_page_zone_regions(self, page_raw: PageRaw) -> List[PageRegion]:
        """
        Create coarse header/body/footer zones.
        """

        page_width = float(page_raw.width or 0)
        page_height = float(page_raw.height or 0)

        if page_width <= 0 or page_height <= 0:
            return []

        header_y1 = page_height * self.config.header_zone_ratio
        footer_y0 = page_height * self.config.footer_zone_ratio

        return [
            PageRegion(
                region_id=make_id("region"),
                page_number=page_raw.page_number,
                region_type="header_zone",
                bbox=[0.0, 0.0, page_width, header_y1],
                source="page_geometry",
                source_object_ids=[],
                text="",
                confidence=0.7,
                metadata={
                    "zone": "header",
                    "ratio": self.config.header_zone_ratio,
                },
            ),
            PageRegion(
                region_id=make_id("region"),
                page_number=page_raw.page_number,
                region_type="body_zone",
                bbox=[0.0, header_y1, page_width, footer_y0],
                source="page_geometry",
                source_object_ids=[],
                text="",
                confidence=0.7,
                metadata={
                    "zone": "body",
                },
            ),
            PageRegion(
                region_id=make_id("region"),
                page_number=page_raw.page_number,
                region_type="footer_zone",
                bbox=[0.0, footer_y0, page_width, page_height],
                source="page_geometry",
                source_object_ids=[],
                text="",
                confidence=0.7,
                metadata={
                    "zone": "footer",
                    "ratio": self.config.footer_zone_ratio,
                },
            ),
        ]

    def _detect_text_regions(self, page_raw: PageRaw) -> List[PageRegion]:
        """
        Detect text layout regions.

        Priority:
        1. reading_order_items
        2. merged_paragraphs
        3. text_blocks
        4. text_lines
        """

        reading_items = self._get_reading_order_items(page_raw)

        if reading_items:
            regions = self._regions_from_reading_order_items(
                page_raw=page_raw,
                items=reading_items,
            )

            if self.config.merge_nearby_text_regions:
                return self._merge_nearby_text_regions(
                    regions=regions,
                    page_number=page_raw.page_number,
                )

            return regions

        merger_meta = page_raw.metadata.get("object_merger", {})
        paragraphs = merger_meta.get("merged_paragraphs", [])

        if paragraphs:
            regions = []

            for index, paragraph in enumerate(paragraphs):
                text = paragraph.get("normalized_text") or paragraph.get("text") or ""
                bbox = paragraph.get("bbox")

                regions.append(
                    PageRegion(
                        region_id=make_id("region"),
                        page_number=page_raw.page_number,
                        region_type="text_region",
                        bbox=bbox,
                        source="object_merger.merged_paragraphs",
                        source_object_ids=[paragraph.get("block_id")],
                        text=self._preview_text(text),
                        confidence=0.85,
                        metadata={
                            "source_index": index,
                            "block_type": paragraph.get("block_type"),
                            "text_length": len(text),
                        },
                    )
                )

            return regions

        if page_raw.text_blocks:
            return self._regions_from_text_blocks(page_raw)

        return self._regions_from_text_lines(page_raw)

    def _regions_from_reading_order_items(
        self,
        page_raw: PageRaw,
        items: List[Dict[str, Any]],
    ) -> List[PageRegion]:
        regions: List[PageRegion] = []

        for index, item in enumerate(items):
            item_type = item.get("item_type", "text")
            text = item.get("text") or ""
            bbox = item.get("bbox")

            if item_type not in {"text_paragraph", "text_block", "text_line"}:
                continue

            regions.append(
                PageRegion(
                    region_id=make_id("region"),
                    page_number=page_raw.page_number,
                    region_type="text_region",
                    bbox=bbox,
                    source="reading_order_builder",
                    source_object_ids=[item.get("item_id")],
                    text=self._preview_text(text),
                    confidence=0.9,
                    metadata={
                        "reading_order": item.get("order"),
                        "column_index": item.get("column_index", 0),
                        "source_index": index,
                        "item_type": item_type,
                        "text_length": len(text),
                    },
                )
            )

        return regions

    def _regions_from_text_blocks(self, page_raw: PageRaw) -> List[PageRegion]:
        regions: List[PageRegion] = []

        for index, block in enumerate(page_raw.text_blocks):
            text = block.normalized_text or block.text or ""

            if not text.strip() or not block.bbox:
                continue

            regions.append(
                PageRegion(
                    region_id=make_id("region"),
                    page_number=page_raw.page_number,
                    region_type="text_region",
                    bbox=block.bbox,
                    source="page_raw.text_blocks",
                    source_object_ids=[block.block_id],
                    text=self._preview_text(text),
                    confidence=0.75,
                    metadata={
                        "source_index": index,
                        "block_type": block.block_type,
                        "line_count": len(block.lines or []),
                        "text_length": len(text),
                    },
                )
            )

        return regions

    def _regions_from_text_lines(self, page_raw: PageRaw) -> List[PageRegion]:
        regions: List[PageRegion] = []

        for index, line in enumerate(page_raw.text_lines):
            text = line.normalized_text or line.text or ""

            if not text.strip() or not line.bbox:
                continue

            regions.append(
                PageRegion(
                    region_id=make_id("region"),
                    page_number=page_raw.page_number,
                    region_type="text_line_region",
                    bbox=line.bbox,
                    source="page_raw.text_lines",
                    source_object_ids=[line.line_id],
                    text=self._preview_text(text),
                    confidence=0.65,
                    metadata={
                        "source_index": index,
                        "text_length": len(text),
                    },
                )
            )

        return regions

    def _detect_image_regions(self, page_raw: PageRaw) -> List[PageRegion]:
        regions: List[PageRegion] = []

        for index, image in enumerate(page_raw.images):
            if not image.bbox:
                continue

            regions.append(
                PageRegion(
                    region_id=make_id("region"),
                    page_number=page_raw.page_number,
                    region_type="image_region",
                    bbox=image.bbox,
                    source="page_raw.images",
                    source_object_ids=[image.image_id],
                    text="",
                    confidence=0.8,
                    metadata={
                        "source_index": index,
                        "width": image.width,
                        "height": image.height,
                        "ext": image.ext,
                        "colorspace": image.colorspace,
                    },
                )
            )

        return regions

    def _detect_drawing_regions(self, page_raw: PageRaw) -> List[PageRegion]:
        regions: List[PageRegion] = []

        object_merger = page_raw.metadata.get("object_merger", {})
        drawing_groups = object_merger.get("merged_drawing_lines", [])

        for index, group in enumerate(drawing_groups):
            bbox = group.get("bbox")

            if not bbox:
                continue

            regions.append(
                PageRegion(
                    region_id=make_id("region"),
                    page_number=page_raw.page_number,
                    region_type="drawing_line_group_region",
                    bbox=bbox,
                    source="object_merger.merged_drawing_lines",
                    source_object_ids=group.get("source_drawing_ids", []),
                    text="",
                    confidence=0.75,
                    metadata={
                        "source_index": index,
                        "orientation": group.get("orientation"),
                        "line_count": group.get("line_count"),
                        "likely_table_structure": group.get("likely_table_structure"),
                    },
                )
            )

        return regions

    def _detect_table_candidate_regions(
        self,
        page_raw: PageRaw,
    ) -> List[PageRegion]:
        """
        Detect rough table candidates.

        This is not table structure recognition.
        It only creates candidate regions for the later TableBoundaryDetector.
        """

        regions: List[PageRegion] = []

        object_merger = page_raw.metadata.get("object_merger", {})
        drawing_groups = object_merger.get("merged_drawing_lines", [])

        table_line_groups = [
            group for group in drawing_groups
            if group.get("likely_table_structure")
            and group.get("bbox")
            and group.get("line_count", 0) >= self.config.table_min_line_count
        ]

        if table_line_groups:
            bbox = merge_bboxes([group.get("bbox") for group in table_line_groups])
            bbox = self._pad_bbox(bbox, self.config.table_region_padding)

            regions.append(
                PageRegion(
                    region_id=make_id("region"),
                    page_number=page_raw.page_number,
                    region_type="table_candidate_region",
                    bbox=bbox,
                    source="object_merger.merged_drawing_lines",
                    source_object_ids=[
                        item
                        for group in table_line_groups
                        for item in group.get("source_drawing_ids", [])
                    ],
                    text="",
                    confidence=0.75,
                    metadata={
                        "method": "drawing_line_groups",
                        "line_group_count": len(table_line_groups),
                    },
                )
            )

        text_table_regions = self._detect_text_based_table_candidates(page_raw)
        regions.extend(text_table_regions)

        return regions

    def _detect_text_based_table_candidates(
        self,
        page_raw: PageRaw,
    ) -> List[PageRegion]:
        """
        Detect text-based table candidates using aligned/numeric lines.
        """

        lines = [
            line for line in page_raw.text_lines
            if line.bbox and (line.normalized_text or "").strip()
        ]

        if len(lines) < 3:
            return []

        candidate_lines = []

        for line in lines:
            text = line.normalized_text or line.text or ""

            if self._looks_like_table_text_line(text):
                candidate_lines.append(line)

        if len(candidate_lines) < 3:
            return []

        groups = self._group_lines_by_vertical_proximity(candidate_lines)

        regions: List[PageRegion] = []

        for group_index, group in enumerate(groups):
            if len(group) < 3:
                continue

            bbox = merge_bboxes([line.bbox for line in group])
            bbox = self._pad_bbox(bbox, self.config.table_region_padding)

            text = "\n".join(
                (line.normalized_text or line.text or "").strip()
                for line in group
            )

            regions.append(
                PageRegion(
                    region_id=make_id("region"),
                    page_number=page_raw.page_number,
                    region_type="table_candidate_region",
                    bbox=bbox,
                    source="page_raw.text_lines",
                    source_object_ids=[line.line_id for line in group],
                    text=self._preview_text(text),
                    confidence=0.55,
                    metadata={
                        "method": "text_alignment_numeric_heuristic",
                        "group_index": group_index,
                        "line_count": len(group),
                    },
                )
            )

        return regions

    def _merge_nearby_text_regions(
        self,
        regions: List[PageRegion],
        page_number: int,
    ) -> List[PageRegion]:
        """
        Merge nearby text regions into larger text blocks.
        """

        valid = [
            region for region in regions
            if region.bbox and region.region_type in {"text_region", "text_line_region"}
        ]

        if not valid:
            return regions

        sorted_regions = sorted(
            valid,
            key=lambda region: (
                region.bbox[1],
                region.bbox[0],
            ),
        )

        groups: List[List[PageRegion]] = []
        current: List[PageRegion] = []

        for region in sorted_regions:
            if not current:
                current.append(region)
                continue

            previous = current[-1]

            vertical_gap = region.bbox[1] - previous.bbox[3]
            x_shift = abs(region.bbox[0] - previous.bbox[0])

            same_group = (
                vertical_gap <= self.config.text_region_vertical_gap
                and x_shift <= self.config.text_region_x_tolerance
            )

            if same_group:
                current.append(region)
            else:
                groups.append(current)
                current = [region]

        if current:
            groups.append(current)

        merged: List[PageRegion] = []

        for group_index, group in enumerate(groups):
            if len(group) == 1:
                merged.append(group[0])
                continue

            bbox = merge_bboxes([region.bbox for region in group])
            text = "\n".join(region.text for region in group if region.text)

            source_ids = []
            for region in group:
                source_ids.extend(region.source_object_ids or [])

            merged.append(
                PageRegion(
                    region_id=make_id("region"),
                    page_number=page_number,
                    region_type="text_region",
                    bbox=bbox,
                    source="region_detector.merge_nearby_text_regions",
                    source_object_ids=source_ids,
                    text=self._preview_text(text),
                    confidence=0.82,
                    metadata={
                        "group_index": group_index,
                        "merged_region_count": len(group),
                    },
                )
            )

        other_regions = [
            region for region in regions
            if region.region_type not in {"text_region", "text_line_region"}
        ]

        return other_regions + merged

    def _get_reading_order_items(self, page_raw: PageRaw) -> List[Dict[str, Any]]:
        reading_meta = page_raw.metadata.get("reading_order_builder", {})
        items = reading_meta.get("reading_order_items", [])

        if isinstance(items, list):
            return items

        return []

    def _looks_like_table_text_line(self, text: str) -> bool:
        if not text:
            return False

        stripped = text.strip()

        if not stripped:
            return False

        digit_count = sum(ch.isdigit() for ch in stripped)
        space_count = stripped.count(" ")
        tab_count = stripped.count("\t")

        numeric_ratio = digit_count / max(len(stripped), 1)

        has_many_spaces = "  " in stripped or tab_count > 0
        has_numeric_content = numeric_ratio >= 0.2
        has_separators = any(sep in stripped for sep in ["|", ";"])

        return has_many_spaces or has_numeric_content or has_separators

    def _group_lines_by_vertical_proximity(
        self,
        lines: List[Any],
    ) -> List[List[Any]]:
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

            if gap <= self.config.text_region_vertical_gap:
                current.append(line)
            else:
                groups.append(current)
                current = [line]

        if current:
            groups.append(current)

        return groups

    def _filter_invalid_regions(
        self,
        regions: List[PageRegion],
    ) -> List[PageRegion]:
        valid: List[PageRegion] = []

        for region in regions:
            if not region.bbox:
                continue

            width = self._bbox_width(region.bbox)
            height = self._bbox_height(region.bbox)

            if width < self.config.min_region_width:
                continue

            if height < self.config.min_region_height:
                continue

            valid.append(region)

        return valid

    def _assign_region_order(
        self,
        regions: List[PageRegion],
    ) -> List[PageRegion]:
        sorted_regions = sorted(
            regions,
            key=lambda region: (
                region.bbox[1] if region.bbox else 0,
                region.bbox[0] if region.bbox else 0,
            ),
        )

        for order, region in enumerate(sorted_regions, start=1):
            if region.metadata is None:
                region.metadata = {}

            region.metadata["region_order"] = order

        return sorted_regions

    def _summarize_regions(
        self,
        regions: List[PageRegion],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "total": len(regions),
            "by_type": {},
        }

        for region in regions:
            summary["by_type"].setdefault(region.region_type, 0)
            summary["by_type"][region.region_type] += 1

        return summary

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

    def _preview_text(
        self,
        text: str,
    ) -> str:
        if not self.config.include_text_preview:
            return ""

        if not text:
            return ""

        return text[: self.config.text_preview_chars]

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


def detect_page_regions(page_raw: PageRaw) -> PageRaw:
    """
    Colab helper function.
    """

    detector = RegionDetector()
    return detector.process(page_raw)
