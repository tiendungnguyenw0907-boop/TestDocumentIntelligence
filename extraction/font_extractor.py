"""
font_extractor.py

Production V1 - Colab Ready

Purpose
-------
Analyze and normalize font information from extracted text spans.

Input
-----
- PageRecord from page_iterator.py
- Optional existing PageRaw from previous extractors

Output
------
PageRaw with fonts populated / updated.

Important
---------
TextExtractor already extracts basic font statistics.
FontExtractor is used to:
- consolidate duplicate fonts
- enrich font metadata
- detect dominant font
- detect title-like / heading-like font signals
- compute font size statistics

This module does not perform:
- heading detection
- title detection
- layout understanding

Those belong to:
- structure/heading_detector.py
- structure/title_detector.py
- page_understanding/page_layout_profiler.py
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import (
    PageRaw,
    FontRaw,
    TextSpanRaw,
    make_id,
    normalize_pdf_text,
)


@dataclass
class FontExtractorConfig:
    """
    Configuration for FontExtractor.
    """

    include_span_samples: bool = True
    max_samples_per_font: int = 5

    min_font_count: int = 1

    detect_font_role_signals: bool = True
    heading_size_ratio: float = 1.20
    title_size_ratio: float = 1.50


class FontExtractor:
    """
    Analyze font statistics from PageRaw text spans.

    Usage
    -----
    page_raw = text_extractor.process(page_record)
    page_raw = font_extractor.process(page_record, page_raw)
    """

    def __init__(self, config: Optional[FontExtractorConfig] = None):
        self.config = config or FontExtractorConfig()

    def process(
        self,
        page_record: Any,
        page_raw: Optional[PageRaw] = None,
    ) -> PageRaw:
        """
        Analyze fonts and update PageRaw.
        """

        if page_raw is None:
            page_raw = self._create_empty_page_raw(page_record)

        spans = page_raw.text_spans or []

        if not spans:
            page_raw.metadata.setdefault("font_extractor", {})
            page_raw.metadata["font_extractor"] = {
                "extractor": "FontExtractor",
                "font_count": 0,
                "note": "No text spans available for font analysis.",
            }
            return page_raw

        fonts = self._build_font_statistics(
            spans=spans,
            page_number=page_raw.page_number,
        )

        font_stats = self._compute_font_page_stats(spans)
        font_roles = self._detect_font_role_signals(fonts, font_stats)

        page_raw.fonts = fonts

        page_raw.metadata.setdefault("font_extractor", {})
        page_raw.metadata["font_extractor"] = {
            "extractor": "FontExtractor",
            "font_count": len(fonts),
            "font_stats": font_stats,
            "font_roles": font_roles,
        }

        return page_raw

    def _build_font_statistics(
        self,
        spans: List[TextSpanRaw],
        page_number: int,
    ) -> List[FontRaw]:
        """
        Consolidate font usage from spans.
        """

        font_map: Dict[Tuple[str, Optional[float], Optional[bool], Optional[bool]], Dict[str, Any]] = {}

        for span in spans:
            font_name = span.font or "unknown"
            size = span.size
            is_bold = span.is_bold
            is_italic = span.is_italic

            key = (font_name, size, is_bold, is_italic)

            if key not in font_map:
                font_map[key] = {
                    "font_name": font_name,
                    "size": size,
                    "is_bold": is_bold,
                    "is_italic": is_italic,
                    "count": 0,
                    "char_count": 0,
                    "sample_texts": [],
                    "bbox_samples": [],
                }

            text = span.normalized_text or span.text or ""
            font_map[key]["count"] += 1
            font_map[key]["char_count"] += len(text)

            if self.config.include_span_samples:
                if len(font_map[key]["sample_texts"]) < self.config.max_samples_per_font:
                    sample = text.strip()
                    if sample:
                        font_map[key]["sample_texts"].append(sample[:120])

                if len(font_map[key]["bbox_samples"]) < self.config.max_samples_per_font:
                    if span.bbox:
                        font_map[key]["bbox_samples"].append(span.bbox)

        fonts: List[FontRaw] = []

        for key, data in font_map.items():
            if data["count"] < self.config.min_font_count:
                continue

            font = FontRaw(
                font_id=make_id("font"),
                page_number=page_number,
                font_name=data["font_name"],
                size=data["size"],
                count=data["count"],
                is_bold=data["is_bold"],
                is_italic=data["is_italic"],
                metadata={
                    "char_count": data["char_count"],
                    "sample_texts": data["sample_texts"],
                    "bbox_samples": data["bbox_samples"],
                },
            )

            fonts.append(font)

        fonts.sort(
            key=lambda f: (
                -(f.count or 0),
                -(f.size or 0),
                f.font_name or "",
            )
        )

        return fonts

    def _compute_font_page_stats(
        self,
        spans: List[TextSpanRaw],
    ) -> Dict[str, Any]:
        """
        Compute page-level font statistics.
        """

        sizes = [
            float(span.size)
            for span in spans
            if span.size is not None
        ]

        if not sizes:
            return {
                "min_size": None,
                "max_size": None,
                "mean_size": None,
                "median_size": None,
                "dominant_size": None,
            }

        dominant_size = self._dominant_size(sizes)

        return {
            "min_size": round(min(sizes), 4),
            "max_size": round(max(sizes), 4),
            "mean_size": round(statistics.mean(sizes), 4),
            "median_size": round(statistics.median(sizes), 4),
            "dominant_size": dominant_size,
            "unique_size_count": len(set(round(x, 2) for x in sizes)),
        }

    def _detect_font_role_signals(
        self,
        fonts: List[FontRaw],
        font_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Heuristic signals only.

        Real heading/title detection will be done later.
        """

        if not self.config.detect_font_role_signals:
            return {}

        dominant_size = font_stats.get("dominant_size")

        if dominant_size is None:
            return {}

        title_candidates = []
        heading_candidates = []
        body_candidates = []

        for font in fonts:
            size = font.size

            if size is None:
                continue

            ratio = size / dominant_size if dominant_size else 1.0

            item = {
                "font_name": font.font_name,
                "size": size,
                "count": font.count,
                "is_bold": font.is_bold,
                "is_italic": font.is_italic,
                "ratio_to_dominant": round(ratio, 4),
            }

            if ratio >= self.config.title_size_ratio:
                title_candidates.append(item)
            elif ratio >= self.config.heading_size_ratio or font.is_bold:
                heading_candidates.append(item)
            else:
                body_candidates.append(item)

        return {
            "dominant_body_size": dominant_size,
            "title_font_candidates": title_candidates,
            "heading_font_candidates": heading_candidates,
            "body_font_candidates": body_candidates[:5],
        }

    def _dominant_size(
        self,
        sizes: List[float],
    ) -> Optional[float]:
        """
        Return most frequent font size rounded to 2 decimals.
        """

        if not sizes:
            return None

        rounded = [round(s, 2) for s in sizes]

        freq: Dict[float, int] = {}

        for size in rounded:
            freq[size] = freq.get(size, 0) + 1

        dominant = sorted(
            freq.items(),
            key=lambda x: (-x[1], x[0]),
        )[0][0]

        return dominant

    def _create_empty_page_raw(self, page_record: Any) -> PageRaw:
        raw_text = getattr(page_record, "text_content", "") or ""
        normalized_text = normalize_pdf_text(raw_text)

        return PageRaw(
            document_id=page_record.document_id,
            source_path=page_record.source_path,
            file_name=page_record.file_name,
            document_type=page_record.document_type,
            page_number=page_record.page_number,
            page_index=page_record.page_index,
            width=page_record.width,
            height=page_record.height,
            rotation=page_record.rotation,
            raw_text=raw_text,
            normalized_text=normalized_text,
            metadata={
                "created_by": "FontExtractor",
                "page_kind": getattr(page_record, "page_kind", "unknown"),
            },
        )
