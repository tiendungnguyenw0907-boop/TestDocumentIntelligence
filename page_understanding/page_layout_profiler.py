"""
page_layout_profiler.py

Production V1 - Colab Ready

Purpose
-------
Build a compact layout profile for one page.

Input
-----
PageRaw after:
- PageExtractionPipeline
- ObjectNormalizer
- ObjectMerger
- ReadingOrderBuilder
- RegionDetector
- HeaderFooterDetector
- TableBoundaryDetector
- FigureDetector
- CaptionDetector

Output
------
PageRaw with metadata:
- page_layout_profiler
- layout_profile
- layout_summary
- processing_hints
- quality_flags

Important
---------
This module does not detect new objects.
It summarizes previous extraction and page-understanding results.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import PageRaw


@dataclass
class PageLayoutProfilerConfig:
    text_heavy_word_threshold: int = 250
    sparse_text_word_threshold: int = 30

    visual_area_ratio_threshold: float = 0.20
    scanned_image_area_ratio_threshold: float = 0.60

    table_heavy_candidate_threshold: int = 2
    figure_heavy_candidate_threshold: int = 2

    min_page_area: float = 1.0

    include_reading_order_text_preview: bool = True
    text_preview_chars: int = 500


@dataclass
class PageLayoutProfile:
    page_number: int
    page_index: int
    width: float
    height: float
    orientation: str

    word_count: int
    text_line_count: int
    text_block_count: int
    image_count: int
    drawing_count: int
    annotation_count: int
    link_count: int
    font_count: int

    region_count: int
    table_candidate_count: int
    figure_candidate_count: int
    caption_candidate_count: int

    has_header: bool
    has_footer: bool
    has_page_number: bool
    has_table_candidates: bool
    has_figure_candidates: bool
    has_caption_candidates: bool

    layout_flow: str
    column_count: int

    text_area_ratio: float
    image_area_ratio: float
    drawing_area_ratio: float
    visual_area_ratio: float

    page_kind: str
    complexity_level: str
    content_profile: str
    processing_strategy: str

    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class PageLayoutProfiler:
    def __init__(
        self,
        config: Optional[PageLayoutProfilerConfig] = None,
    ):
        self.config = config or PageLayoutProfilerConfig()

    def process(
        self,
        page_raw: PageRaw,
    ) -> PageRaw:
        warnings: List[str] = []

        try:
            profile = self._build_profile(page_raw)

            page_raw.metadata.setdefault("page_layout_profiler", {})
            page_raw.metadata["page_layout_profiler"] = {
                "processor": "PageLayoutProfiler",
                "layout_profile": profile.to_dict(),
                "layout_summary": self._build_layout_summary(profile),
                "processing_hints": self._build_processing_hints(profile),
                "quality_flags": self._build_quality_flags(page_raw, profile),
                "config": {
                    "text_heavy_word_threshold": self.config.text_heavy_word_threshold,
                    "sparse_text_word_threshold": self.config.sparse_text_word_threshold,
                    "visual_area_ratio_threshold": self.config.visual_area_ratio_threshold,
                    "scanned_image_area_ratio_threshold": self.config.scanned_image_area_ratio_threshold,
                    "table_heavy_candidate_threshold": self.config.table_heavy_candidate_threshold,
                    "figure_heavy_candidate_threshold": self.config.figure_heavy_candidate_threshold,
                },
            }

        except Exception as exc:
            warnings.append(f"PageLayoutProfiler failed: {exc}")

        if warnings:
            page_raw.warnings.extend(warnings)

        return page_raw

    def _build_profile(
        self,
        page_raw: PageRaw,
    ) -> PageLayoutProfile:
        page_area = self._page_area(page_raw)

        text_area = self._sum_bbox_area(
            [line.bbox for line in page_raw.text_lines]
        )

        image_area = self._sum_bbox_area(
            [image.bbox for image in page_raw.images]
        )

        drawing_area = self._sum_bbox_area(
            [drawing.bbox for drawing in page_raw.drawings]
        )

        text_area_ratio = self._safe_ratio(text_area, page_area)
        image_area_ratio = self._safe_ratio(image_area, page_area)
        drawing_area_ratio = self._safe_ratio(drawing_area, page_area)

        visual_area_ratio = image_area_ratio + drawing_area_ratio

        if visual_area_ratio > 1.0:
            visual_area_ratio = 1.0

        reading_meta = page_raw.metadata.get("reading_order_builder", {})
        column_profile = reading_meta.get("column_profile", {})

        layout_flow = reading_meta.get("layout_flow") or "unknown"
        column_count = self._safe_int(
            column_profile.get("column_count", 1),
            default=1,
        )

        header_footer_meta = page_raw.metadata.get("header_footer_detector", {})
        header_footer_summary = header_footer_meta.get("summary", {})

        table_meta = page_raw.metadata.get("table_boundary_detector", {})
        table_summary = table_meta.get("summary", {})

        figure_meta = page_raw.metadata.get("figure_detector", {})
        figure_summary = figure_meta.get("summary", {})

        caption_meta = page_raw.metadata.get("caption_detector", {})
        caption_summary = caption_meta.get("summary", {})

        region_meta = page_raw.metadata.get("region_detector", {})
        region_count = self._safe_int(
            region_meta.get("region_count", 0),
            default=0,
        )

        word_count = len(page_raw.words)
        text_line_count = len(page_raw.text_lines)
        text_block_count = len(page_raw.text_blocks)

        table_candidate_count = self._safe_int(
            table_meta.get("table_candidate_count", 0),
            default=0,
        )

        figure_candidate_count = self._safe_int(
            figure_meta.get("figure_candidate_count", 0),
            default=0,
        )

        caption_candidate_count = self._safe_int(
            caption_meta.get("caption_candidate_count", 0),
            default=0,
        )

        page_kind = self._classify_page_kind(
            page_raw=page_raw,
            word_count=word_count,
            image_area_ratio=image_area_ratio,
            visual_area_ratio=visual_area_ratio,
        )

        content_profile = self._classify_content_profile(
            word_count=word_count,
            table_candidate_count=table_candidate_count,
            figure_candidate_count=figure_candidate_count,
            visual_area_ratio=visual_area_ratio,
        )

        complexity_level = self._classify_complexity_level(
            word_count=word_count,
            column_count=column_count,
            table_candidate_count=table_candidate_count,
            figure_candidate_count=figure_candidate_count,
            region_count=region_count,
            visual_area_ratio=visual_area_ratio,
        )

        processing_strategy = self._recommend_processing_strategy(
            page_kind=page_kind,
            content_profile=content_profile,
            complexity_level=complexity_level,
            table_candidate_count=table_candidate_count,
            figure_candidate_count=figure_candidate_count,
        )

        profile = PageLayoutProfile(
            page_number=page_raw.page_number,
            page_index=page_raw.page_index,
            width=float(page_raw.width or 0),
            height=float(page_raw.height or 0),
            orientation=self._detect_orientation(page_raw),
            word_count=word_count,
            text_line_count=text_line_count,
            text_block_count=text_block_count,
            image_count=len(page_raw.images),
            drawing_count=len(page_raw.drawings),
            annotation_count=len(page_raw.annotations),
            link_count=len(page_raw.links),
            font_count=len(page_raw.fonts),
            region_count=region_count,
            table_candidate_count=table_candidate_count,
            figure_candidate_count=figure_candidate_count,
            caption_candidate_count=caption_candidate_count,
            has_header=bool(header_footer_summary.get("has_header", False)),
            has_footer=bool(header_footer_summary.get("has_footer", False)),
            has_page_number=bool(header_footer_summary.get("has_page_number", False)),
            has_table_candidates=bool(table_summary.get("has_table_candidates", False)),
            has_figure_candidates=bool(figure_summary.get("has_figure_candidates", False)),
            has_caption_candidates=bool(caption_summary.get("has_caption_candidates", False)),
            layout_flow=layout_flow,
            column_count=column_count,
            text_area_ratio=round(text_area_ratio, 6),
            image_area_ratio=round(image_area_ratio, 6),
            drawing_area_ratio=round(drawing_area_ratio, 6),
            visual_area_ratio=round(visual_area_ratio, 6),
            page_kind=page_kind,
            complexity_level=complexity_level,
            content_profile=content_profile,
            processing_strategy=processing_strategy,
            metadata={
                "column_profile": column_profile,
                "region_summary": region_meta.get("region_summary", {}),
                "table_summary": table_summary,
                "figure_summary": figure_summary,
                "caption_summary": caption_summary,
                "reading_order_preview": self._reading_order_preview(page_raw),
            },
        )

        return profile

    def _build_layout_summary(
        self,
        profile: PageLayoutProfile,
    ) -> Dict[str, Any]:
        return {
            "page_number": profile.page_number,
            "orientation": profile.orientation,
            "page_kind": profile.page_kind,
            "content_profile": profile.content_profile,
            "complexity_level": profile.complexity_level,
            "processing_strategy": profile.processing_strategy,
            "layout_flow": profile.layout_flow,
            "column_count": profile.column_count,
            "has_header": profile.has_header,
            "has_footer": profile.has_footer,
            "has_page_number": profile.has_page_number,
            "has_table_candidates": profile.has_table_candidates,
            "has_figure_candidates": profile.has_figure_candidates,
            "has_caption_candidates": profile.has_caption_candidates,
            "counts": {
                "words": profile.word_count,
                "text_lines": profile.text_line_count,
                "text_blocks": profile.text_block_count,
                "images": profile.image_count,
                "drawings": profile.drawing_count,
                "annotations": profile.annotation_count,
                "links": profile.link_count,
                "fonts": profile.font_count,
                "regions": profile.region_count,
                "tables": profile.table_candidate_count,
                "figures": profile.figure_candidate_count,
                "captions": profile.caption_candidate_count,
            },
            "area_ratios": {
                "text": profile.text_area_ratio,
                "image": profile.image_area_ratio,
                "drawing": profile.drawing_area_ratio,
                "visual": profile.visual_area_ratio,
            },
        }

    def _build_processing_hints(
        self,
        profile: PageLayoutProfile,
    ) -> Dict[str, Any]:
        hints = {
            "needs_ocr": False,
            "needs_table_understanding": False,
            "needs_figure_understanding": False,
            "needs_caption_linking": False,
            "needs_multi_column_handling": False,
            "needs_header_footer_filtering": False,
            "recommended_next_modules": [],
        }

        if profile.page_kind in {"scanned_or_image_page", "image_dominant_page"}:
            hints["needs_ocr"] = True
            hints["recommended_next_modules"].append("RegionOCR")

        if profile.has_table_candidates:
            hints["needs_table_understanding"] = True
            hints["recommended_next_modules"].append("TableUnderstandingPipeline")

        if profile.has_figure_candidates:
            hints["needs_figure_understanding"] = True
            hints["recommended_next_modules"].append("FigureUnderstanding")

        if profile.has_caption_candidates:
            hints["needs_caption_linking"] = True
            hints["recommended_next_modules"].append("CaptionLinking")

        if profile.column_count > 1 or profile.layout_flow == "multi_column":
            hints["needs_multi_column_handling"] = True
            hints["recommended_next_modules"].append("ColumnAwareReadingOrder")

        if profile.has_header or profile.has_footer or profile.has_page_number:
            hints["needs_header_footer_filtering"] = True
            hints["recommended_next_modules"].append("HeaderFooterFiltering")

        hints["recommended_next_modules"] = list(
            dict.fromkeys(hints["recommended_next_modules"])
        )

        return hints

    def _build_quality_flags(
        self,
        page_raw: PageRaw,
        profile: PageLayoutProfile,
    ) -> Dict[str, Any]:
        flags = {
            "empty_page": False,
            "very_sparse_text": False,
            "possible_scanned_page": False,
            "possible_encoding_issue": False,
            "high_layout_complexity": False,
            "has_warnings": bool(page_raw.warnings),
        }

        if profile.word_count == 0 and profile.image_count == 0:
            flags["empty_page"] = True

        if 0 < profile.word_count <= self.config.sparse_text_word_threshold:
            flags["very_sparse_text"] = True

        if profile.page_kind == "scanned_or_image_page":
            flags["possible_scanned_page"] = True

        if self._has_encoding_issue(page_raw):
            flags["possible_encoding_issue"] = True

        if profile.complexity_level == "high":
            flags["high_layout_complexity"] = True

        return flags

    def _classify_page_kind(
        self,
        page_raw: PageRaw,
        word_count: int,
        image_area_ratio: float,
        visual_area_ratio: float,
    ) -> str:
        if word_count == 0 and len(page_raw.images) == 0:
            return "blank_or_unknown_page"

        if (
            word_count <= self.config.sparse_text_word_threshold
            and image_area_ratio >= self.config.scanned_image_area_ratio_threshold
        ):
            return "scanned_or_image_page"

        if image_area_ratio >= self.config.scanned_image_area_ratio_threshold:
            return "image_dominant_page"

        if word_count > 0 and visual_area_ratio >= self.config.visual_area_ratio_threshold:
            return "hybrid_text_visual_page"

        if word_count > 0:
            return "digital_text_page"

        return "unknown_page"

    def _classify_content_profile(
        self,
        word_count: int,
        table_candidate_count: int,
        figure_candidate_count: int,
        visual_area_ratio: float,
    ) -> str:
        if table_candidate_count >= self.config.table_heavy_candidate_threshold:
            return "table_heavy"

        if figure_candidate_count >= self.config.figure_heavy_candidate_threshold:
            return "figure_heavy"

        if table_candidate_count > 0 and figure_candidate_count > 0:
            return "mixed_table_figure"

        if table_candidate_count > 0:
            return "table_present"

        if figure_candidate_count > 0:
            return "figure_present"

        if visual_area_ratio >= self.config.visual_area_ratio_threshold:
            return "visual_present"

        if word_count >= self.config.text_heavy_word_threshold:
            return "text_heavy"

        if word_count <= self.config.sparse_text_word_threshold:
            return "sparse_text"

        return "normal_text"

    def _classify_complexity_level(
        self,
        word_count: int,
        column_count: int,
        table_candidate_count: int,
        figure_candidate_count: int,
        region_count: int,
        visual_area_ratio: float,
    ) -> str:
        score = 0

        if word_count >= self.config.text_heavy_word_threshold:
            score += 1

        if column_count > 1:
            score += 2

        if table_candidate_count > 0:
            score += 2

        if figure_candidate_count > 0:
            score += 1

        if region_count >= 10:
            score += 1

        if visual_area_ratio >= self.config.visual_area_ratio_threshold:
            score += 1

        if score >= 5:
            return "high"

        if score >= 3:
            return "medium"

        return "low"

    def _recommend_processing_strategy(
        self,
        page_kind: str,
        content_profile: str,
        complexity_level: str,
        table_candidate_count: int,
        figure_candidate_count: int,
    ) -> str:
        if page_kind == "scanned_or_image_page":
            return "ocr_then_layout_understanding"

        if table_candidate_count > 0:
            return "layout_plus_table_understanding"

        if figure_candidate_count > 0:
            return "layout_plus_figure_caption_understanding"

        if complexity_level == "high":
            return "advanced_layout_understanding"

        if content_profile in {"text_heavy", "normal_text", "sparse_text"}:
            return "text_layout_understanding"

        return "standard_page_understanding"

    def _detect_orientation(
        self,
        page_raw: PageRaw,
    ) -> str:
        width = float(page_raw.width or 0)
        height = float(page_raw.height or 0)

        if width <= 0 or height <= 0:
            return "unknown"

        if abs(width - height) <= 5:
            return "square"

        if width > height:
            return "landscape"

        return "portrait"

    def _reading_order_preview(
        self,
        page_raw: PageRaw,
    ) -> str:
        if not self.config.include_reading_order_text_preview:
            return ""

        reading_meta = page_raw.metadata.get("reading_order_builder", {})
        text = reading_meta.get("reading_order_text", "")

        if not text:
            text = page_raw.normalized_text or page_raw.raw_text or ""

        return text[: self.config.text_preview_chars]

    def _has_encoding_issue(
        self,
        page_raw: PageRaw,
    ) -> bool:
        text = page_raw.normalized_text or page_raw.raw_text or ""

        if not text:
            return False

        suspicious_chars = ["�", "□", "●", "§"]

        for char in suspicious_chars:
            if char in text:
                return True

        return False

    def _page_area(
        self,
        page_raw: PageRaw,
    ) -> float:
        width = float(page_raw.width or 0)
        height = float(page_raw.height or 0)

        area = width * height

        return max(area, self.config.min_page_area)

    def _sum_bbox_area(
        self,
        bboxes: List[Optional[List[float]]],
    ) -> float:
        total = 0.0

        for bbox in bboxes:
            total += self._bbox_area(bbox)

        return total

    def _bbox_area(
        self,
        bbox: Optional[List[float]],
    ) -> float:
        if not bbox:
            return 0.0

        if len(bbox) != 4:
            return 0.0

        width = max(float(bbox[2]) - float(bbox[0]), 0.0)
        height = max(float(bbox[3]) - float(bbox[1]), 0.0)

        return width * height

    def _safe_ratio(
        self,
        numerator: float,
        denominator: float,
    ) -> float:
        if denominator <= 0:
            return 0.0

        return max(float(numerator), 0.0) / float(denominator)

    def _safe_int(
        self,
        value: Any,
        default: int = 0,
    ) -> int:
        try:
            return int(value)
        except Exception:
            return default


def profile_page_layout(
    page_raw: PageRaw,
) -> PageRaw:
    profiler = PageLayoutProfiler()
    return profiler.process(page_raw)
