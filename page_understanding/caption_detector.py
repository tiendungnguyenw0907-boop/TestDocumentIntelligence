"""
caption_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect caption candidates for figures, tables, charts, diagrams, and images.

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

Output
------
PageRaw with metadata:
- caption_candidates
- figure_caption_links
- table_caption_links
- caption_summary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class CaptionDetectorConfig:
    detect_figure_captions: bool = True
    detect_table_captions: bool = True
    detect_chart_captions: bool = True
    detect_image_captions: bool = True

    use_reading_order_items: bool = True
    use_text_lines_fallback: bool = True

    max_caption_length: int = 500
    min_caption_length: int = 3

    link_to_figures: bool = True
    link_to_tables: bool = True

    max_link_distance: float = 90.0
    horizontal_overlap_threshold: float = 0.20

    prefer_caption_below_figure: bool = True
    prefer_caption_above_table: bool = True

    include_text_preview: bool = True
    text_preview_chars: int = 500


@dataclass
class CaptionCandidate:
    caption_id: str
    page_number: int
    caption_type: str
    text: str
    bbox: Optional[List[float]]

    confidence: float = 0.5
    source: str = "unknown"
    source_object_ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["source_object_ids"] is None:
            data["source_object_ids"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class CaptionLink:
    link_id: str
    page_number: int
    caption_id: str
    target_id: str
    target_type: str
    distance: float
    relation: str

    confidence: float = 0.5
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class CaptionDetector:
    def __init__(
        self,
        config: Optional[CaptionDetectorConfig] = None,
    ):
        self.config = config or CaptionDetectorConfig()

    def process(
        self,
        page_raw: PageRaw,
    ) -> PageRaw:
        warnings: List[str] = []

        try:
            text_objects = self._collect_text_objects(page_raw)

            caption_candidates = self._detect_caption_candidates(
                page_raw=page_raw,
                text_objects=text_objects,
            )

            figure_caption_links: List[CaptionLink] = []

            if self.config.link_to_figures:
                figure_caption_links = self._link_captions_to_figures(
                    page_raw=page_raw,
                    captions=caption_candidates,
                )

            table_caption_links: List[CaptionLink] = []

            if self.config.link_to_tables:
                table_caption_links = self._link_captions_to_tables(
                    page_raw=page_raw,
                    captions=caption_candidates,
                )

            page_raw.metadata.setdefault("caption_detector", {})
            page_raw.metadata["caption_detector"] = {
                "processor": "CaptionDetector",
                "caption_candidate_count": len(caption_candidates),
                "caption_candidates": [
                    item.to_dict() for item in caption_candidates
                ],
                "figure_caption_links": [
                    item.to_dict() for item in figure_caption_links
                ],
                "table_caption_links": [
                    item.to_dict() for item in table_caption_links
                ],
                "summary": self._build_summary(
                    captions=caption_candidates,
                    figure_links=figure_caption_links,
                    table_links=table_caption_links,
                ),
                "config": {
                    "detect_figure_captions": self.config.detect_figure_captions,
                    "detect_table_captions": self.config.detect_table_captions,
                    "detect_chart_captions": self.config.detect_chart_captions,
                    "detect_image_captions": self.config.detect_image_captions,
                    "link_to_figures": self.config.link_to_figures,
                    "link_to_tables": self.config.link_to_tables,
                },
            }

        except Exception as exc:
            warnings.append(f"CaptionDetector failed: {exc}")

        if warnings:
            page_raw.warnings.extend(warnings)

        return page_raw

    def _collect_text_objects(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        text_objects: List[Dict[str, Any]] = []

        if self.config.use_reading_order_items:
            reading_meta = page_raw.metadata.get("reading_order_builder", {})
            reading_items = reading_meta.get("reading_order_items", [])

            for index, item in enumerate(reading_items):
                text = item.get("text") or ""
                bbox = item.get("bbox")

                if not bbox:
                    continue

                if not self._is_valid_caption_text_length(text):
                    continue

                text_objects.append(
                    {
                        "object_id": item.get("item_id") or f"reading_item_{index}",
                        "text": text,
                        "bbox": bbox,
                        "source": "reading_order_builder.reading_order_items",
                        "source_index": index,
                        "metadata": item.get("metadata", {}),
                    }
                )

        if text_objects:
            return text_objects

        if self.config.use_text_lines_fallback:
            for index, line in enumerate(page_raw.text_lines):
                text = line.normalized_text or line.text or ""
                bbox = line.bbox

                if not bbox:
                    continue

                if not self._is_valid_caption_text_length(text):
                    continue

                text_objects.append(
                    {
                        "object_id": line.line_id,
                        "text": text,
                        "bbox": bbox,
                        "source": "page_raw.text_lines",
                        "source_index": index,
                        "metadata": {},
                    }
                )

        return text_objects

    def _detect_caption_candidates(
        self,
        page_raw: PageRaw,
        text_objects: List[Dict[str, Any]],
    ) -> List[CaptionCandidate]:
        candidates: List[CaptionCandidate] = []

        for obj in text_objects:
            text = obj.get("text") or ""
            caption_type = self._classify_caption_type(text)

            if caption_type is None:
                continue

            confidence = self._score_caption(
                text=text,
                caption_type=caption_type,
            )

            candidates.append(
                CaptionCandidate(
                    caption_id=make_id("caption"),
                    page_number=page_raw.page_number,
                    caption_type=caption_type,
                    text=self._preview_text(text),
                    bbox=obj.get("bbox"),
                    confidence=confidence,
                    source=obj.get("source", "unknown"),
                    source_object_ids=[obj.get("object_id", "")],
                    metadata={
                        "source_index": obj.get("source_index"),
                        "raw_text_length": len(text),
                        "caption_number": self._extract_caption_number(text),
                        "caption_label": self._extract_caption_label(text),
                    },
                )
            )

        return self._sort_captions(candidates)

    def _classify_caption_type(
        self,
        text: str,
    ) -> Optional[str]:
        if not text:
            return None

        lower = text.strip().lower()

        if not lower:
            return None

        table_patterns = [
            r"^bảng\s+\d+",
            r"^bang\s+\d+",
            r"^table\s+\d+",
            r"^tbl\.\s*\d+",
        ]

        figure_patterns = [
            r"^hình\s+\d+",
            r"^hinh\s+\d+",
            r"^figure\s+\d+",
            r"^fig\.\s*\d+",
        ]

        chart_patterns = [
            r"^biểu\s*đồ\s+\d+",
            r"^bieu\s*do\s+\d+",
            r"^chart\s+\d+",
            r"^sơ\s*đồ\s+\d+",
            r"^so\s*do\s+\d+",
            r"^diagram\s+\d+",
        ]

        image_patterns = [
            r"^ảnh\s+\d+",
            r"^anh\s+\d+",
            r"^image\s+\d+",
        ]

        if self.config.detect_table_captions and self._matches_any(lower, table_patterns):
            return "table_caption"

        if self.config.detect_figure_captions and self._matches_any(lower, figure_patterns):
            return "figure_caption"

        if self.config.detect_chart_captions and self._matches_any(lower, chart_patterns):
            return "chart_caption"

        if self.config.detect_image_captions and self._matches_any(lower, image_patterns):
            return "image_caption"

        return None

    def _link_captions_to_figures(
        self,
        page_raw: PageRaw,
        captions: List[CaptionCandidate],
    ) -> List[CaptionLink]:
        figure_targets = self._collect_figure_targets(page_raw)

        if not figure_targets:
            return []

        relevant_captions = [
            caption
            for caption in captions
            if caption.caption_type in {
                "figure_caption",
                "chart_caption",
                "image_caption",
            }
        ]

        links: List[CaptionLink] = []

        for caption in relevant_captions:
            best_target = None
            best_distance = None
            best_relation = ""

            for target in figure_targets:
                relation, distance = self._caption_target_relation(
                    caption_bbox=caption.bbox,
                    target_bbox=target.get("bbox"),
                )

                if distance is None:
                    continue

                if distance > self.config.max_link_distance:
                    continue

                if best_distance is None or distance < best_distance:
                    best_target = target
                    best_distance = distance
                    best_relation = relation

            if best_target is None or best_distance is None:
                continue

            confidence = self._score_caption_link(
                caption=caption,
                relation=best_relation,
                distance=best_distance,
                target_type="figure",
            )

            links.append(
                CaptionLink(
                    link_id=make_id("caplink"),
                    page_number=page_raw.page_number,
                    caption_id=caption.caption_id,
                    target_id=best_target.get("target_id", ""),
                    target_type="figure",
                    distance=round(best_distance, 4),
                    relation=best_relation,
                    confidence=confidence,
                    metadata={
                        "target_source": best_target.get("source"),
                        "target_detection_method": best_target.get("detection_method"),
                    },
                )
            )

        return links

    def _link_captions_to_tables(
        self,
        page_raw: PageRaw,
        captions: List[CaptionCandidate],
    ) -> List[CaptionLink]:
        table_targets = self._collect_table_targets(page_raw)

        if not table_targets:
            return []

        relevant_captions = [
            caption
            for caption in captions
            if caption.caption_type == "table_caption"
        ]

        links: List[CaptionLink] = []

        for caption in relevant_captions:
            best_target = None
            best_distance = None
            best_relation = ""

            for target in table_targets:
                relation, distance = self._caption_target_relation(
                    caption_bbox=caption.bbox,
                    target_bbox=target.get("bbox"),
                )

                if distance is None:
                    continue

                if distance > self.config.max_link_distance:
                    continue

                if best_distance is None or distance < best_distance:
                    best_target = target
                    best_distance = distance
                    best_relation = relation

            if best_target is None or best_distance is None:
                continue

            confidence = self._score_caption_link(
                caption=caption,
                relation=best_relation,
                distance=best_distance,
                target_type="table",
            )

            links.append(
                CaptionLink(
                    link_id=make_id("caplink"),
                    page_number=page_raw.page_number,
                    caption_id=caption.caption_id,
                    target_id=best_target.get("target_id", ""),
                    target_type="table",
                    distance=round(best_distance, 4),
                    relation=best_relation,
                    confidence=confidence,
                    metadata={
                        "target_source": best_target.get("source"),
                        "target_detection_method": best_target.get("detection_method"),
                    },
                )
            )

        return links

    def _collect_figure_targets(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        figure_meta = page_raw.metadata.get("figure_detector", {})
        figure_candidates = figure_meta.get("figure_candidates", [])

        targets: List[Dict[str, Any]] = []

        for index, figure in enumerate(figure_candidates):
            bbox = figure.get("bbox")

            if not bbox:
                continue

            targets.append(
                {
                    "target_id": figure.get("figure_id") or f"figure_{index}",
                    "bbox": bbox,
                    "source": "figure_detector.figure_candidates",
                    "detection_method": figure.get("detection_method"),
                    "confidence": figure.get("confidence", 0.5),
                }
            )

        return targets

    def _collect_table_targets(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        table_meta = page_raw.metadata.get("table_boundary_detector", {})
        table_candidates = table_meta.get("table_candidates", [])

        targets: List[Dict[str, Any]] = []

        for index, table in enumerate(table_candidates):
            bbox = table.get("bbox")

            if not bbox:
                continue

            targets.append(
                {
                    "target_id": table.get("table_id") or f"table_{index}",
                    "bbox": bbox,
                    "source": "table_boundary_detector.table_candidates",
                    "detection_method": table.get("detection_method"),
                    "confidence": table.get("confidence", 0.5),
                }
            )

        return targets

    def _caption_target_relation(
        self,
        caption_bbox: Optional[List[float]],
        target_bbox: Optional[List[float]],
    ) -> Tuple[str, Optional[float]]:
        if not caption_bbox or not target_bbox:
            return "", None

        overlap_ratio = self._horizontal_overlap_ratio(
            bbox_a=caption_bbox,
            bbox_b=target_bbox,
        )

        if overlap_ratio < self.config.horizontal_overlap_threshold:
            return "", None

        caption_y0 = float(caption_bbox[1])
        caption_y1 = float(caption_bbox[3])
        target_y0 = float(target_bbox[1])
        target_y1 = float(target_bbox[3])

        if caption_y1 <= target_y0:
            return "caption_above_target", target_y0 - caption_y1

        if caption_y0 >= target_y1:
            return "caption_below_target", caption_y0 - target_y1

        return "caption_overlaps_target", 0.0

    def _score_caption(
        self,
        text: str,
        caption_type: str,
    ) -> float:
        score = 0.70
        lower = text.strip().lower()

        strong_prefix_patterns = [
            r"^bảng",
            r"^bang",
            r"^table",
            r"^hình",
            r"^hinh",
            r"^figure",
            r"^fig\.",
            r"^biểu\s*đồ",
            r"^bieu\s*do",
            r"^sơ\s*đồ",
            r"^so\s*do",
            r"^chart",
            r"^diagram",
            r"^ảnh",
            r"^anh",
            r"^image",
        ]

        if self._matches_any(lower, strong_prefix_patterns):
            score += 0.10

        if self._extract_caption_number(text) is not None:
            score += 0.08

        if ":" in text or "." in text:
            score += 0.03

        if len(text) > self.config.max_caption_length:
            score -= 0.20

        if caption_type in {"figure_caption", "table_caption"}:
            score += 0.03

        return round(max(0.0, min(score, 0.95)), 4)

    def _score_caption_link(
        self,
        caption: CaptionCandidate,
        relation: str,
        distance: float,
        target_type: str,
    ) -> float:
        score = 0.60

        if distance <= 20:
            score += 0.18
        elif distance <= 50:
            score += 0.10
        else:
            score += 0.04

        if target_type == "table":
            if relation == "caption_above_target" and self.config.prefer_caption_above_table:
                score += 0.10
            elif relation == "caption_below_target":
                score += 0.04

        if target_type == "figure":
            if relation == "caption_below_target" and self.config.prefer_caption_below_figure:
                score += 0.10
            elif relation == "caption_above_target":
                score += 0.04

        score = (score + caption.confidence) / 2.0

        return round(max(0.0, min(score, 0.95)), 4)

    def _extract_caption_number(
        self,
        text: str,
    ) -> Optional[str]:
        if not text:
            return None

        match = re.search(r"(\d+(?:[.\-]\d+)*)", text)

        if not match:
            return None

        return match.group(1)

    def _extract_caption_label(
        self,
        text: str,
    ) -> Optional[str]:
        if not text:
            return None

        lower = text.strip().lower()

        labels = [
            "bảng",
            "bang",
            "table",
            "tbl.",
            "hình",
            "hinh",
            "figure",
            "fig.",
            "biểu đồ",
            "bieu do",
            "sơ đồ",
            "so do",
            "diagram",
            "chart",
            "ảnh",
            "anh",
            "image",
        ]

        for label in labels:
            if lower.startswith(label):
                return label

        return None

    def _is_valid_caption_text_length(
        self,
        text: str,
    ) -> bool:
        if text is None:
            return False

        stripped = text.strip()

        if len(stripped) < self.config.min_caption_length:
            return False

        if len(stripped) > self.config.max_caption_length:
            return False

        return True

    def _matches_any(
        self,
        text: str,
        patterns: List[str],
    ) -> bool:
        for pattern in patterns:
            if re.search(pattern, text):
                return True

        return False

    def _sort_captions(
        self,
        captions: List[CaptionCandidate],
    ) -> List[CaptionCandidate]:
        return sorted(
            captions,
            key=lambda item: (
                item.bbox[1] if item.bbox else 0,
                item.bbox[0] if item.bbox else 0,
            ),
        )

    def _horizontal_overlap_ratio(
        self,
        bbox_a: List[float],
        bbox_b: List[float],
    ) -> float:
        x0 = max(float(bbox_a[0]), float(bbox_b[0]))
        x1 = min(float(bbox_a[2]), float(bbox_b[2]))

        overlap = max(x1 - x0, 0.0)

        width_a = max(float(bbox_a[2]) - float(bbox_a[0]), 0.0)
        width_b = max(float(bbox_b[2]) - float(bbox_b[0]), 0.0)

        smaller_width = min(width_a, width_b)

        if smaller_width <= 0:
            return 0.0

        return overlap / smaller_width

    def _preview_text(
        self,
        text: str,
    ) -> str:
        if not self.config.include_text_preview:
            return ""

        if not text:
            return ""

        return text[: self.config.text_preview_chars]

    def _build_summary(
        self,
        captions: List[CaptionCandidate],
        figure_links: List[CaptionLink],
        table_links: List[CaptionLink],
    ) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}

        for caption in captions:
            by_type[caption.caption_type] = by_type.get(caption.caption_type, 0) + 1

        return {
            "has_caption_candidates": len(captions) > 0,
            "caption_candidate_count": len(captions),
            "by_caption_type": by_type,
            "figure_caption_link_count": len(figure_links),
            "table_caption_link_count": len(table_links),
        }


def detect_captions(
    page_raw: PageRaw,
) -> PageRaw:
    detector = CaptionDetector()
    return detector.process(page_raw)
