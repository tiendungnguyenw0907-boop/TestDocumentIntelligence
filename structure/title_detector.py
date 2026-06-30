"""
title_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect document title candidates from page-understanding output.

Input
-----
List[PageRaw] after PageUnderstandingPipeline.

Output
------
Dictionary with:
- title_candidates
- selected_title
- title_summary

Flow
----
PageUnderstandingPipeline
    ↓
TitleDetector
    ↓
TOCDetector
    ↓
HeadingDetector
    ↓
SectionBuilder
    ↓
DocumentTreeBuilder
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class TitleDetectorConfig:
    max_pages_to_scan: int = 3
    max_candidates: int = 10

    min_title_length: int = 5
    max_title_length: int = 300

    prefer_first_page: bool = True
    prefer_uppercase: bool = True
    prefer_centered: bool = True
    prefer_large_font: bool = True
    prefer_top_half: bool = True

    exclude_page_number_like: bool = True
    exclude_header_footer: bool = True

    title_keywords: Optional[List[str]] = None

    include_debug: bool = True


@dataclass
class TitleCandidate:
    candidate_id: str
    title_text: str
    page_number: int
    bbox: Optional[List[float]]
    source: str

    confidence: float = 0.5
    rank: int = 0
    source_object_id: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class TitleDetector:
    def __init__(
        self,
        config: Optional[TitleDetectorConfig] = None,
    ):
        self.config = config or TitleDetectorConfig()

        if self.config.title_keywords is None:
            self.config.title_keywords = [
                "báo cáo",
                "bao cao",
                "quyết định",
                "quyet dinh",
                "kế hoạch",
                "ke hoach",
                "đề án",
                "de an",
                "dự án",
                "du an",
                "phương án",
                "phuong an",
                "tờ trình",
                "to trinh",
                "hồ sơ",
                "ho so",
                "kiểm toán",
                "kiem toan",
                "audit",
                "report",
                "plan",
                "proposal",
            ]

    def process(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        candidates = self.detect_title_candidates(page_raws)
        selected_title = candidates[0] if candidates else None

        result = {
            "processor": "TitleDetector",
            "selected_title": selected_title.to_dict() if selected_title else None,
            "title_candidates": [
                candidate.to_dict() for candidate in candidates
            ],
            "title_summary": {
                "has_title": selected_title is not None,
                "candidate_count": len(candidates),
                "selected_title_text": selected_title.title_text if selected_title else "",
                "selected_title_page": selected_title.page_number if selected_title else None,
                "selected_confidence": selected_title.confidence if selected_title else 0.0,
            },
            "config": {
                "max_pages_to_scan": self.config.max_pages_to_scan,
                "max_candidates": self.config.max_candidates,
                "min_title_length": self.config.min_title_length,
                "max_title_length": self.config.max_title_length,
                "exclude_page_number_like": self.config.exclude_page_number_like,
                "exclude_header_footer": self.config.exclude_header_footer,
            },
        }

        self._attach_to_pages(
            page_raws=page_raws,
            result=result,
        )

        return result

    def detect_title_candidates(
        self,
        page_raws: List[PageRaw],
    ) -> List[TitleCandidate]:
        scan_pages = page_raws[: self.config.max_pages_to_scan]
        candidates: List[TitleCandidate] = []

        for page_raw in scan_pages:
            text_objects = self._collect_text_objects(page_raw)

            for obj in text_objects:
                text = self._normalize_candidate_text(obj.get("text") or "")

                if not self._is_valid_title_text(text):
                    continue

                if self.config.exclude_page_number_like and self._is_page_number_like(text):
                    continue

                if self.config.exclude_header_footer:
                    if self._is_header_footer_object(page_raw, obj):
                        continue

                score = self._score_candidate(
                    page_raw=page_raw,
                    obj=obj,
                    text=text,
                )

                if score <= 0:
                    continue

                candidates.append(
                    TitleCandidate(
                        candidate_id=make_id("title"),
                        title_text=text,
                        page_number=page_raw.page_number,
                        bbox=obj.get("bbox"),
                        source=obj.get("source", "unknown"),
                        confidence=round(score, 4),
                        source_object_id=obj.get("object_id", ""),
                        metadata={
                            "source_index": obj.get("source_index"),
                            "object_type": obj.get("object_type"),
                            "font_size": obj.get("font_size"),
                            "is_uppercase": self._is_mostly_uppercase(text),
                            "is_centered": self._is_centered(
                                bbox=obj.get("bbox"),
                                page_width=page_raw.width,
                            ),
                            "is_top_half": self._is_top_half(
                                bbox=obj.get("bbox"),
                                page_height=page_raw.height,
                            ),
                            "keyword_hits": self._keyword_hits(text),
                            "text_length": len(text),
                        },
                    )
                )

        candidates = self._merge_duplicate_candidates(candidates)
        candidates = sorted(
            candidates,
            key=lambda item: (
                -item.confidence,
                item.page_number,
                item.bbox[1] if item.bbox else 999999,
            ),
        )

        for rank, candidate in enumerate(candidates, start=1):
            candidate.rank = rank

        return candidates[: self.config.max_candidates]

    def _collect_text_objects(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []

        reading_meta = page_raw.metadata.get("reading_order_builder", {})
        reading_items = reading_meta.get("reading_order_items", [])

        for index, item in enumerate(reading_items):
            text = item.get("text") or ""
            bbox = item.get("bbox")

            if not bbox:
                continue

            objects.append(
                {
                    "object_id": item.get("item_id") or f"reading_item_{index}",
                    "text": text,
                    "bbox": bbox,
                    "source": "reading_order_builder.reading_order_items",
                    "source_index": index,
                    "object_type": item.get("item_type", "reading_item"),
                    "font_size": self._infer_font_size_from_bbox(bbox),
                    "metadata": item.get("metadata", {}),
                }
            )

        if objects:
            return objects

        merger_meta = page_raw.metadata.get("object_merger", {})
        paragraphs = merger_meta.get("merged_paragraphs", [])

        for index, paragraph in enumerate(paragraphs):
            text = paragraph.get("normalized_text") or paragraph.get("text") or ""
            bbox = paragraph.get("bbox")

            if not bbox:
                continue

            objects.append(
                {
                    "object_id": paragraph.get("block_id") or f"paragraph_{index}",
                    "text": text,
                    "bbox": bbox,
                    "source": "object_merger.merged_paragraphs",
                    "source_index": index,
                    "object_type": "merged_paragraph",
                    "font_size": self._infer_font_size_from_bbox(bbox),
                    "metadata": paragraph.get("metadata", {}),
                }
            )

        if objects:
            return objects

        for index, block in enumerate(page_raw.text_blocks):
            text = block.normalized_text or block.text or ""
            bbox = block.bbox

            if not bbox:
                continue

            objects.append(
                {
                    "object_id": block.block_id,
                    "text": text,
                    "bbox": bbox,
                    "source": "page_raw.text_blocks",
                    "source_index": index,
                    "object_type": "text_block",
                    "font_size": self._infer_block_font_size(block),
                    "metadata": block.metadata or {},
                }
            )

        if objects:
            return objects

        for index, line in enumerate(page_raw.text_lines):
            text = line.normalized_text or line.text or ""
            bbox = line.bbox

            if not bbox:
                continue

            objects.append(
                {
                    "object_id": line.line_id,
                    "text": text,
                    "bbox": bbox,
                    "source": "page_raw.text_lines",
                    "source_index": index,
                    "object_type": "text_line",
                    "font_size": self._infer_line_font_size(line),
                    "metadata": line.metadata or {},
                }
            )

        return objects

    def _score_candidate(
        self,
        page_raw: PageRaw,
        obj: Dict[str, Any],
        text: str,
    ) -> float:
        score = 0.30

        bbox = obj.get("bbox")
        font_size = obj.get("font_size")

        if self.config.prefer_first_page:
            if page_raw.page_number == 1:
                score += 0.20
            elif page_raw.page_number == 2:
                score += 0.08
            else:
                score -= 0.05

        if self.config.prefer_top_half:
            if self._is_top_half(bbox, page_raw.height):
                score += 0.12

        if self.config.prefer_centered:
            if self._is_centered(bbox, page_raw.width):
                score += 0.12

        if self.config.prefer_uppercase:
            if self._is_mostly_uppercase(text):
                score += 0.10

        if self.config.prefer_large_font:
            if font_size and font_size >= self._estimate_page_body_font_size(page_raw) * 1.15:
                score += 0.12

        keyword_hits = self._keyword_hits(text)

        if keyword_hits:
            score += min(0.12, 0.04 * len(keyword_hits))

        line_count = len([line for line in text.splitlines() if line.strip()])

        if line_count <= 4:
            score += 0.05
        else:
            score -= 0.08

        text_len = len(text)

        if 20 <= text_len <= 180:
            score += 0.08
        elif text_len > 220:
            score -= 0.10

        if self._looks_like_organization_header(text):
            score -= 0.15

        if self._looks_like_date_or_place(text):
            score -= 0.10

        return max(0.0, min(score, 0.98))

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        for page_raw in page_raws[: self.config.max_pages_to_scan]:
            page_raw.metadata.setdefault("title_detector", {})
            page_raw.metadata["title_detector"] = result

    def _normalize_candidate_text(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        lines = [
            re.sub(r"\s+", " ", line).strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return "\n".join(lines).strip()

    def _is_valid_title_text(
        self,
        text: str,
    ) -> bool:
        if not text:
            return False

        stripped = text.strip()

        if len(stripped) < self.config.min_title_length:
            return False

        if len(stripped) > self.config.max_title_length:
            return False

        if len(stripped.split()) < 2:
            return False

        return True

    def _is_header_footer_object(
        self,
        page_raw: PageRaw,
        obj: Dict[str, Any],
    ) -> bool:
        bbox = obj.get("bbox")

        if not bbox or not page_raw.height:
            return False

        center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
        ratio = center_y / float(page_raw.height)

        if ratio <= 0.08:
            return True

        if ratio >= 0.92:
            return True

        return False

    def _is_page_number_like(
        self,
        text: str,
    ) -> bool:
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

    def _looks_like_organization_header(
        self,
        text: str,
    ) -> bool:
        lower = text.lower().strip()

        patterns = [
            "cộng hòa xã hội chủ nghĩa việt nam",
            "độc lập",
            "tự do",
            "hạnh phúc",
            "kiểm toán nhà nước",
            "bộ ",
            "ủy ban nhân dân",
            "sở ",
            "cục ",
        ]

        if len(lower) < 80:
            return any(pattern in lower for pattern in patterns)

        return False

    def _looks_like_date_or_place(
        self,
        text: str,
    ) -> bool:
        lower = text.lower().strip()

        if re.search(r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", lower):
            return True

        if re.search(r"tháng\s+\d{1,2}\s+năm\s+\d{4}", lower):
            return True

        if re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", lower):
            return True

        if lower.startswith("hà nội") or lower.startswith("đà nẵng"):
            return True

        return False

    def _keyword_hits(
        self,
        text: str,
    ) -> List[str]:
        lower = text.lower()
        hits = []

        for keyword in self.config.title_keywords or []:
            if keyword in lower:
                hits.append(keyword)

        return hits

    def _is_mostly_uppercase(
        self,
        text: str,
    ) -> bool:
        letters = [ch for ch in text if ch.isalpha()]

        if not letters:
            return False

        uppercase_count = sum(1 for ch in letters if ch.isupper())

        return uppercase_count / max(len(letters), 1) >= 0.65

    def _is_centered(
        self,
        bbox: Optional[List[float]],
        page_width: Optional[float],
    ) -> bool:
        if not bbox or not page_width:
            return False

        page_center = float(page_width) / 2.0
        bbox_center = (float(bbox[0]) + float(bbox[2])) / 2.0

        tolerance = float(page_width) * 0.15

        return abs(bbox_center - page_center) <= tolerance

    def _is_top_half(
        self,
        bbox: Optional[List[float]],
        page_height: Optional[float],
    ) -> bool:
        if not bbox or not page_height:
            return False

        center_y = (float(bbox[1]) + float(bbox[3])) / 2.0

        return center_y <= float(page_height) * 0.55

    def _infer_font_size_from_bbox(
        self,
        bbox: Optional[List[float]],
    ) -> Optional[float]:
        if not bbox:
            return None

        height = max(float(bbox[3]) - float(bbox[1]), 0.0)

        if height <= 0:
            return None

        return round(height, 2)

    def _infer_line_font_size(
        self,
        line: Any,
    ) -> Optional[float]:
        spans = getattr(line, "spans", None) or []

        sizes = []

        for span in spans:
            size = getattr(span, "size", None)

            if size:
                sizes.append(float(size))

        if sizes:
            return sum(sizes) / len(sizes)

        return self._infer_font_size_from_bbox(getattr(line, "bbox", None))

    def _infer_block_font_size(
        self,
        block: Any,
    ) -> Optional[float]:
        lines = getattr(block, "lines", None) or []

        sizes = []

        for line in lines:
            size = self._infer_line_font_size(line)

            if size:
                sizes.append(size)

        if sizes:
            return sum(sizes) / len(sizes)

        return self._infer_font_size_from_bbox(getattr(block, "bbox", None))

    def _estimate_page_body_font_size(
        self,
        page_raw: PageRaw,
    ) -> float:
        sizes = []

        for line in page_raw.text_lines:
            size = self._infer_line_font_size(line)

            if size:
                sizes.append(size)

        if not sizes:
            return 10.0

        sizes = sorted(sizes)
        mid = len(sizes) // 2

        if len(sizes) % 2 == 1:
            return sizes[mid]

        return (sizes[mid - 1] + sizes[mid]) / 2.0

    def _merge_duplicate_candidates(
        self,
        candidates: List[TitleCandidate],
    ) -> List[TitleCandidate]:
        seen: Dict[str, TitleCandidate] = {}

        for candidate in candidates:
            key = re.sub(r"\s+", " ", candidate.title_text.lower()).strip()

            if key not in seen:
                seen[key] = candidate
                continue

            if candidate.confidence > seen[key].confidence:
                seen[key] = candidate

        return list(seen.values())


def detect_document_title(
    page_raws: List[PageRaw],
) -> Dict[str, Any]:
    detector = TitleDetector()
    return detector.process(page_raws)
