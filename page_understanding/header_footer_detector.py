"""
header_footer_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect header and footer candidates from page text objects, page regions,
and page geometry.

Input
-----
PageRaw after:
- PageExtractionPipeline
- ObjectNormalizer
- ObjectMerger
- ReadingOrderBuilder
- RegionDetector

Output
------
PageRaw with metadata:
- header_candidates
- footer_candidates
- page_number_candidates
- body_candidates
- summary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class HeaderFooterDetectorConfig:
    """
    Configuration for HeaderFooterDetector.
    """

    header_zone_ratio: float = 0.12
    footer_zone_ratio: float = 0.88

    min_text_length: int = 1
    max_header_footer_text_length: int = 300

    detect_page_number: bool = True
    detect_date_like_text: bool = True

    include_body_candidates: bool = True
    include_text_preview: bool = True
    text_preview_chars: int = 300

    header_confidence_base: float = 0.65
    footer_confidence_base: float = 0.65
    page_number_confidence_base: float = 0.80


@dataclass
class HeaderFooterCandidate:
    """
    One header/footer/body/page-number candidate.
    """

    candidate_id: str
    page_number: int
    candidate_type: str
    bbox: Optional[List[float]]
    text: str
    source: str = "unknown"
    source_object_ids: Optional[List[str]] = None
    confidence: float = 0.5
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["source_object_ids"] is None:
            data["source_object_ids"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class HeaderFooterDetector:
    """
    Detect header/footer candidates from one PageRaw.
    """

    def __init__(
        self,
        config: Optional[HeaderFooterDetectorConfig] = None,
    ):
        self.config = config or HeaderFooterDetectorConfig()

    def process(
        self,
        page_raw: PageRaw,
    ) -> PageRaw:
        """
        Detect header/footer candidates and attach result to page_raw.metadata.
        """

        warnings: List[str] = []

        try:
            text_objects = self._collect_text_objects(page_raw)

            header_candidates = self._detect_header_candidates(
                page_raw=page_raw,
                text_objects=text_objects,
            )

            footer_candidates = self._detect_footer_candidates(
                page_raw=page_raw,
                text_objects=text_objects,
            )

            page_number_candidates: List[HeaderFooterCandidate] = []

            if self.config.detect_page_number:
                page_number_candidates = self._detect_page_number_candidates(
                    page_raw=page_raw,
                    text_objects=text_objects,
                )

            body_candidates: List[HeaderFooterCandidate] = []

            if self.config.include_body_candidates:
                body_candidates = self._detect_body_candidates(
                    page_raw=page_raw,
                    text_objects=text_objects,
                )

            page_raw.metadata.setdefault("header_footer_detector", {})
            page_raw.metadata["header_footer_detector"] = {
                "processor": "HeaderFooterDetector",
                "header_candidates": [
                    item.to_dict() for item in header_candidates
                ],
                "footer_candidates": [
                    item.to_dict() for item in footer_candidates
                ],
                "page_number_candidates": [
                    item.to_dict() for item in page_number_candidates
                ],
                "body_candidates": [
                    item.to_dict() for item in body_candidates
                ],
                "summary": {
                    "header_count": len(header_candidates),
                    "footer_count": len(footer_candidates),
                    "page_number_count": len(page_number_candidates),
                    "body_count": len(body_candidates),
                    "has_header": len(header_candidates) > 0,
                    "has_footer": len(footer_candidates) > 0,
                    "has_page_number": len(page_number_candidates) > 0,
                },
                "config": {
                    "header_zone_ratio": self.config.header_zone_ratio,
                    "footer_zone_ratio": self.config.footer_zone_ratio,
                    "detect_page_number": self.config.detect_page_number,
                    "detect_date_like_text": self.config.detect_date_like_text,
                    "include_body_candidates": self.config.include_body_candidates,
                },
            }

        except Exception as exc:
            warnings.append(f"HeaderFooterDetector failed: {exc}")

        if warnings:
            page_raw.warnings.extend(warnings)

        return page_raw

    def _collect_text_objects(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        """
        Collect text objects from RegionDetector first.
        Fallback to page_raw.text_lines.
        """

        text_objects: List[Dict[str, Any]] = []

        region_meta = page_raw.metadata.get("region_detector", {})
        regions = region_meta.get("detected_regions", [])

        for index, region in enumerate(regions):
            region_type = region.get("region_type", "")

            if region_type not in {"text_region", "text_line_region"}:
                continue

            text = region.get("text") or ""

            if not self._is_valid_text(text):
                continue

            text_objects.append(
                {
                    "object_id": region.get("region_id") or f"region_{index}",
                    "text": text,
                    "bbox": region.get("bbox"),
                    "source": "region_detector.detected_regions",
                    "source_index": index,
                    "metadata": region.get("metadata", {}),
                }
            )

        if text_objects:
            return text_objects

        for index, line in enumerate(page_raw.text_lines):
            text = line.normalized_text or line.text or ""

            if not self._is_valid_text(text):
                continue

            text_objects.append(
                {
                    "object_id": line.line_id,
                    "text": text,
                    "bbox": line.bbox,
                    "source": "page_raw.text_lines",
                    "source_index": index,
                    "metadata": {},
                }
            )

        return text_objects

    def _detect_header_candidates(
        self,
        page_raw: PageRaw,
        text_objects: List[Dict[str, Any]],
    ) -> List[HeaderFooterCandidate]:
        """
        Detect header candidates in top page zone.
        """

        page_height = float(page_raw.height or 0)

        if page_height <= 0:
            return []

        header_limit = page_height * self.config.header_zone_ratio
        candidates: List[HeaderFooterCandidate] = []

        for obj in text_objects:
            bbox = obj.get("bbox")

            if not bbox:
                continue

            center_y = self._bbox_center_y(bbox)

            if center_y > header_limit:
                continue

            text = obj.get("text") or ""

            confidence = self._score_header_footer_candidate(
                text=text,
                bbox=bbox,
                page_raw=page_raw,
                candidate_type="header",
            )

            candidates.append(
                HeaderFooterCandidate(
                    candidate_id=make_id("header"),
                    page_number=page_raw.page_number,
                    candidate_type="header",
                    bbox=bbox,
                    text=self._preview_text(text),
                    source=obj.get("source", "unknown"),
                    source_object_ids=[obj.get("object_id", "")],
                    confidence=confidence,
                    metadata={
                        "source_index": obj.get("source_index"),
                        "zone": "header",
                        "center_y_ratio": self._safe_ratio(
                            numerator=center_y,
                            denominator=page_height,
                        ),
                        "is_date_like": self._is_date_like_text(text),
                        "is_page_number_like": self._is_page_number_like(text),
                        "text_length": len(text),
                    },
                )
            )

        return self._sort_candidates(candidates)

    def _detect_footer_candidates(
        self,
        page_raw: PageRaw,
        text_objects: List[Dict[str, Any]],
    ) -> List[HeaderFooterCandidate]:
        """
        Detect footer candidates in bottom page zone.
        """

        page_height = float(page_raw.height or 0)

        if page_height <= 0:
            return []

        footer_start = page_height * self.config.footer_zone_ratio
        candidates: List[HeaderFooterCandidate] = []

        for obj in text_objects:
            bbox = obj.get("bbox")

            if not bbox:
                continue

            center_y = self._bbox_center_y(bbox)

            if center_y < footer_start:
                continue

            text = obj.get("text") or ""

            confidence = self._score_header_footer_candidate(
                text=text,
                bbox=bbox,
                page_raw=page_raw,
                candidate_type="footer",
            )

            candidates.append(
                HeaderFooterCandidate(
                    candidate_id=make_id("footer"),
                    page_number=page_raw.page_number,
                    candidate_type="footer",
                    bbox=bbox,
                    text=self._preview_text(text),
                    source=obj.get("source", "unknown"),
                    source_object_ids=[obj.get("object_id", "")],
                    confidence=confidence,
                    metadata={
                        "source_index": obj.get("source_index"),
                        "zone": "footer",
                        "center_y_ratio": self._safe_ratio(
                            numerator=center_y,
                            denominator=page_height,
                        ),
                        "is_date_like": self._is_date_like_text(text),
                        "is_page_number_like": self._is_page_number_like(text),
                        "text_length": len(text),
                    },
                )
            )

        return self._sort_candidates(candidates)

    def _detect_page_number_candidates(
        self,
        page_raw: PageRaw,
        text_objects: List[Dict[str, Any]],
    ) -> List[HeaderFooterCandidate]:
        """
        Detect page number candidates.
        """

        candidates: List[HeaderFooterCandidate] = []

        for obj in text_objects:
            text = obj.get("text") or ""

            if not self._is_page_number_like(text):
                continue

            bbox = obj.get("bbox")

            if not bbox:
                continue

            zone = self._detect_zone(
                bbox=bbox,
                page_height=page_raw.height,
            )

            confidence = self.config.page_number_confidence_base

            if zone in {"header", "footer"}:
                confidence += 0.10

            confidence = min(confidence, 0.95)

            candidates.append(
                HeaderFooterCandidate(
                    candidate_id=make_id("pageno"),
                    page_number=page_raw.page_number,
                    candidate_type="page_number",
                    bbox=bbox,
                    text=self._preview_text(text),
                    source=obj.get("source", "unknown"),
                    source_object_ids=[obj.get("object_id", "")],
                    confidence=confidence,
                    metadata={
                        "source_index": obj.get("source_index"),
                        "zone": zone,
                        "detected_page_number": self._extract_page_number(text),
                    },
                )
            )

        return self._sort_candidates(candidates)

    def _detect_body_candidates(
        self,
        page_raw: PageRaw,
        text_objects: List[Dict[str, Any]],
    ) -> List[HeaderFooterCandidate]:
        """
        Detect body candidates by excluding header/footer zones.
        """

        page_height = float(page_raw.height or 0)

        if page_height <= 0:
            return []

        header_limit = page_height * self.config.header_zone_ratio
        footer_start = page_height * self.config.footer_zone_ratio

        candidates: List[HeaderFooterCandidate] = []

        for obj in text_objects:
            bbox = obj.get("bbox")

            if not bbox:
                continue

            center_y = self._bbox_center_y(bbox)

            if center_y <= header_limit:
                continue

            if center_y >= footer_start:
                continue

            text = obj.get("text") or ""

            candidates.append(
                HeaderFooterCandidate(
                    candidate_id=make_id("body"),
                    page_number=page_raw.page_number,
                    candidate_type="body",
                    bbox=bbox,
                    text=self._preview_text(text),
                    source=obj.get("source", "unknown"),
                    source_object_ids=[obj.get("object_id", "")],
                    confidence=0.70,
                    metadata={
                        "source_index": obj.get("source_index"),
                        "zone": "body",
                        "text_length": len(text),
                    },
                )
            )

        return self._sort_candidates(candidates)

    def _score_header_footer_candidate(
        self,
        text: str,
        bbox: List[float],
        page_raw: PageRaw,
        candidate_type: str,
    ) -> float:
        """
        Score header/footer candidate.
        """

        if candidate_type == "header":
            score = self.config.header_confidence_base
        else:
            score = self.config.footer_confidence_base

        text_len = len(text.strip())

        if text_len <= 80:
            score += 0.08

        if self._is_page_number_like(text):
            score += 0.08

        if self.config.detect_date_like_text and self._is_date_like_text(text):
            score += 0.05

        if page_raw.width:
            width_ratio = self._bbox_width(bbox) / float(page_raw.width)

            if width_ratio >= 0.5:
                score += 0.03

        if text_len > self.config.max_header_footer_text_length:
            score -= 0.20

        return round(max(0.0, min(score, 0.95)), 4)

    def _is_valid_text(
        self,
        text: str,
    ) -> bool:
        if text is None:
            return False

        stripped = text.strip()

        if len(stripped) < self.config.min_text_length:
            return False

        return True

    def _is_page_number_like(
        self,
        text: str,
    ) -> bool:
        if not text:
            return False

        stripped = text.strip().lower()

        patterns = [
            r"^\d+$",
            r"^-\s*\d+\s*-$",
            r"^trang\s+\d+$",
            r"^page\s+\d+$",
            r"^\d+\s*/\s*\d+$",
        ]

        for pattern in patterns:
            if re.match(pattern, stripped):
                return True

        return False

    def _extract_page_number(
        self,
        text: str,
    ) -> Optional[int]:
        if not text:
            return None

        matches = re.findall(r"\d+", text)

        if not matches:
            return None

        try:
            return int(matches[0])
        except Exception:
            return None

    def _is_date_like_text(
        self,
        text: str,
    ) -> bool:
        if not text:
            return False

        lower = text.lower().strip()

        patterns = [
            r"\d{1,2}/\d{1,2}/\d{2,4}",
            r"\d{1,2}-\d{1,2}-\d{2,4}",
            r"tháng\s+\d{1,2}",
            r"năm\s+\d{4}",
        ]

        for pattern in patterns:
            if re.search(pattern, lower):
                return True

        return False

    def _detect_zone(
        self,
        bbox: Optional[List[float]],
        page_height: Optional[float],
    ) -> str:
        if not bbox or not page_height:
            return "unknown"

        center_y = self._bbox_center_y(bbox)

        if center_y <= page_height * self.config.header_zone_ratio:
            return "header"

        if center_y >= page_height * self.config.footer_zone_ratio:
            return "footer"

        return "body"

    def _sort_candidates(
        self,
        candidates: List[HeaderFooterCandidate],
    ) -> List[HeaderFooterCandidate]:
        return sorted(
            candidates,
            key=lambda item: (
                item.bbox[1] if item.bbox else 0,
                item.bbox[0] if item.bbox else 0,
            ),
        )

    def _preview_text(
        self,
        text: str,
    ) -> str:
        if not self.config.include_text_preview:
            return ""

        if not text:
            return ""

        return text[: self.config.text_preview_chars]

    def _bbox_center_y(
        self,
        bbox: Optional[List[float]],
    ) -> float:
        if not bbox:
            return 0.0

        return (float(bbox[1]) + float(bbox[3])) / 2.0

    def _bbox_width(
        self,
        bbox: Optional[List[float]],
    ) -> float:
        if not bbox:
            return 0.0

        return max(float(bbox[2]) - float(bbox[0]), 0.0)

    def _safe_ratio(
        self,
        numerator: float,
        denominator: Optional[float],
    ) -> Optional[float]:
        if not denominator:
            return None

        try:
            return round(float(numerator) / float(denominator), 6)
        except Exception:
            return None


def detect_header_footer(
    page_raw: PageRaw,
) -> PageRaw:
    """
    Colab helper function.
    """

    detector = HeaderFooterDetector()
    return detector.process(page_raw)
