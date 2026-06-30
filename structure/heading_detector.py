"""
heading_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect heading candidates from page-understanding output.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- TitleDetector
- TOCDetector

Output
------
Dictionary with:
- heading_candidates
- heading_summary

Flow
----
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
import unicodedata
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class HeadingDetectorConfig:
    min_heading_length: int = 3
    max_heading_length: int = 250

    min_confidence: float = 0.35
    max_headings_per_page: int = 30

    detect_numbered_headings: bool = True
    detect_roman_headings: bool = True
    detect_chapter_part_headings: bool = True
    detect_uppercase_headings: bool = True
    detect_keyword_headings: bool = True

    use_font_size_signal: bool = True
    use_position_signal: bool = True
    use_spacing_signal: bool = True

    exclude_title_text: bool = True
    exclude_toc_pages: bool = True
    exclude_header_footer: bool = True
    exclude_caption_like: bool = True
    exclude_table_like: bool = True

    attach_to_pages: bool = True
    include_debug: bool = True


@dataclass
class HeadingCandidate:
    heading_id: str
    page_number: int
    page_index: int
    heading_text: str
    level: int
    heading_type: str
    bbox: Optional[List[float]]
    confidence: float

    source: str = "unknown"
    source_object_id: str = ""
    section_number: str = ""
    rank_on_page: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class HeadingDetector:
    def __init__(
        self,
        config: Optional[HeadingDetectorConfig] = None,
    ):
        self.config = config or HeadingDetectorConfig()

    def process(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        title_texts = self._collect_title_texts(page_raws)
        toc_page_numbers = self._collect_toc_page_numbers(page_raws)

        candidates = self.detect_heading_candidates(
            page_raws=page_raws,
            title_texts=title_texts,
            toc_page_numbers=toc_page_numbers,
        )

        result = {
            "processor": "HeadingDetector",
            "heading_candidates": [
                candidate.to_dict() for candidate in candidates
            ],
            "heading_summary": self._build_summary(candidates),
            "config": {
                "min_confidence": self.config.min_confidence,
                "exclude_title_text": self.config.exclude_title_text,
                "exclude_toc_pages": self.config.exclude_toc_pages,
                "exclude_header_footer": self.config.exclude_header_footer,
                "exclude_caption_like": self.config.exclude_caption_like,
                "exclude_table_like": self.config.exclude_table_like,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                candidates=candidates,
                result=result,
            )

        return result

    def detect_heading_candidates(
        self,
        page_raws: List[PageRaw],
        title_texts: Optional[List[str]] = None,
        toc_page_numbers: Optional[set] = None,
    ) -> List[HeadingCandidate]:
        title_texts = title_texts or []
        toc_page_numbers = toc_page_numbers or set()

        candidates: List[HeadingCandidate] = []

        for page_raw in page_raws:
            if self.config.exclude_toc_pages and page_raw.page_number in toc_page_numbers:
                continue

            text_objects = self._collect_text_objects(page_raw)
            body_font_size = self._estimate_body_font_size(page_raw)

            page_candidates: List[HeadingCandidate] = []

            for obj in text_objects:
                text = self._normalize_heading_text(obj.get("text") or "")

                if not self._is_valid_heading_text(text):
                    continue

                if self.config.exclude_title_text:
                    if self._matches_existing_title(text, title_texts):
                        continue

                if self.config.exclude_header_footer:
                    if self._is_header_footer_object(page_raw, obj):
                        continue

                if self.config.exclude_caption_like and self._looks_like_caption(text):
                    continue

                if self.config.exclude_table_like and self._looks_like_table_line(text):
                    continue

                analysis = self._analyze_heading_text(text)
                score = self._score_heading_candidate(
                    page_raw=page_raw,
                    obj=obj,
                    text=text,
                    analysis=analysis,
                    body_font_size=body_font_size,
                )

                if score < self.config.min_confidence:
                    continue

                page_candidates.append(
                    HeadingCandidate(
                        heading_id=make_id("heading"),
                        page_number=page_raw.page_number,
                        page_index=page_raw.page_index,
                        heading_text=text,
                        level=analysis["level"],
                        heading_type=analysis["heading_type"],
                        bbox=obj.get("bbox"),
                        confidence=round(score, 4),
                        source=obj.get("source", "unknown"),
                        source_object_id=obj.get("object_id", ""),
                        section_number=analysis["section_number"],
                        metadata={
                            "source_index": obj.get("source_index"),
                            "object_type": obj.get("object_type"),
                            "font_size": obj.get("font_size"),
                            "body_font_size": body_font_size,
                            "is_uppercase": self._is_mostly_uppercase(text),
                            "keyword_hits": self._keyword_hits(text),
                            "analysis": analysis,
                        },
                    )
                )

            page_candidates = sorted(
                page_candidates,
                key=lambda item: (
                    item.bbox[1] if item.bbox else 999999,
                    item.bbox[0] if item.bbox else 999999,
                    -item.confidence,
                ),
            )

            for rank, candidate in enumerate(page_candidates, start=1):
                candidate.rank_on_page = rank

            candidates.extend(page_candidates[: self.config.max_headings_per_page])

        candidates = self._deduplicate_candidates(candidates)
        candidates = self._sort_candidates(candidates)

        return candidates

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

            if not bbox or not text.strip():
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

            if not bbox or not text.strip():
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

        for index, line in enumerate(page_raw.text_lines):
            text = line.normalized_text or line.text or ""
            bbox = line.bbox

            if not bbox or not text.strip():
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

    def _analyze_heading_text(
        self,
        text: str,
    ) -> Dict[str, Any]:
        section_number = ""
        level = 1
        heading_type = "plain_heading"

        clean = text.strip()
        normalized = self._normalize_for_match(clean)

        chapter_patterns = [
            r"^(chương|chuong)\s+([ivxlcdm]+|\d+)",
            r"^(phần|phan)\s+([ivxlcdm]+|\d+)",
        ]

        if self.config.detect_chapter_part_headings:
            for pattern in chapter_patterns:
                match = re.match(pattern, normalized, flags=re.IGNORECASE)

                if match:
                    label = match.group(1)
                    number = match.group(2)
                    section_number = number
                    level = 1

                    if label in {"phần", "phan"}:
                        heading_type = "part_heading"
                    else:
                        heading_type = "chapter_heading"

                    return {
                        "section_number": section_number,
                        "level": level,
                        "heading_type": heading_type,
                        "pattern": "chapter_or_part",
                    }

        if self.config.detect_numbered_headings:
            match = re.match(r"^(\d+(?:\.\d+)*)(?:\.|\))?\s+(.+)$", clean)

            if match:
                section_number = match.group(1)
                level = max(1, section_number.count(".") + 1)
                heading_type = "numbered_heading"

                return {
                    "section_number": section_number,
                    "level": level,
                    "heading_type": heading_type,
                    "pattern": "decimal_numbered",
                }

        if self.config.detect_roman_headings:
            match = re.match(r"^([IVXLCDM]+)(?:\.|\))\s+(.+)$", clean)

            if match:
                section_number = match.group(1)
                level = 1
                heading_type = "roman_heading"

                return {
                    "section_number": section_number,
                    "level": level,
                    "heading_type": heading_type,
                    "pattern": "roman_numbered",
                }

        if re.match(r"^[a-zA-Z]\)\s+.+$", clean):
            section_number = clean[:2]
            level = 3
            heading_type = "letter_heading"

            return {
                "section_number": section_number,
                "level": level,
                "heading_type": heading_type,
                "pattern": "letter_numbered",
            }

        if self.config.detect_keyword_headings:
            keyword_hits = self._keyword_hits(clean)

            if keyword_hits:
                level = 1 if self._is_mostly_uppercase(clean) else 2
                heading_type = "keyword_heading"

                return {
                    "section_number": "",
                    "level": level,
                    "heading_type": heading_type,
                    "pattern": "keyword",
                }

        if self.config.detect_uppercase_headings and self._is_mostly_uppercase(clean):
            return {
                "section_number": "",
                "level": 2,
                "heading_type": "uppercase_heading",
                "pattern": "uppercase",
            }

        return {
            "section_number": "",
            "level": 3,
            "heading_type": "plain_heading",
            "pattern": "plain",
        }

    def _score_heading_candidate(
        self,
        page_raw: PageRaw,
        obj: Dict[str, Any],
        text: str,
        analysis: Dict[str, Any],
        body_font_size: float,
    ) -> float:
        score = 0.20

        heading_type = analysis.get("heading_type", "")
        bbox = obj.get("bbox")
        font_size = obj.get("font_size")

        if heading_type in {
            "part_heading",
            "chapter_heading",
            "numbered_heading",
            "roman_heading",
        }:
            score += 0.35

        elif heading_type in {"keyword_heading", "uppercase_heading"}:
            score += 0.22

        elif heading_type == "letter_heading":
            score += 0.18

        if self.config.use_font_size_signal and font_size:
            if font_size >= body_font_size * 1.20:
                score += 0.16
            elif font_size >= body_font_size * 1.10:
                score += 0.08

        if self._is_mostly_uppercase(text):
            score += 0.08

        if self._keyword_hits(text):
            score += min(0.12, 0.04 * len(self._keyword_hits(text)))

        if self.config.use_position_signal:
            if self._is_near_left_margin(bbox, page_raw.width):
                score += 0.04

        if self.config.use_spacing_signal:
            if self._has_heading_like_length(text):
                score += 0.08

        line_count = len([line for line in text.splitlines() if line.strip()])

        if line_count <= 2:
            score += 0.05
        else:
            score -= 0.08

        if text.endswith(".") and len(text.split()) > 18:
            score -= 0.06

        if self._looks_like_sentence(text):
            score -= 0.08

        if self._looks_like_date_or_place(text):
            score -= 0.10

        return max(0.0, min(score, 0.98))

    def _collect_title_texts(
        self,
        page_raws: List[PageRaw],
    ) -> List[str]:
        titles: List[str] = []

        for page_raw in page_raws[:3]:
            title_meta = page_raw.metadata.get("title_detector", {})
            selected = title_meta.get("selected_title")

            if isinstance(selected, dict):
                title_text = selected.get("title_text") or ""

                if title_text:
                    titles.append(self._normalize_for_match(title_text))

        return titles

    def _collect_toc_page_numbers(
        self,
        page_raws: List[PageRaw],
    ) -> set:
        toc_pages = set()

        for page_raw in page_raws:
            toc_meta = page_raw.metadata.get("toc_detector", {})

            if toc_meta.get("is_toc_page"):
                toc_pages.add(page_raw.page_number)

        return toc_pages

    def _matches_existing_title(
        self,
        text: str,
        title_texts: List[str],
    ) -> bool:
        normalized = self._normalize_for_match(text)

        for title in title_texts:
            if normalized == title:
                return True

            if len(normalized) > 20 and normalized in title:
                return True

            if len(title) > 20 and title in normalized:
                return True

        return False

    def _is_valid_heading_text(
        self,
        text: str,
    ) -> bool:
        if not text:
            return False

        text = text.strip()

        if len(text) < self.config.min_heading_length:
            return False

        if len(text) > self.config.max_heading_length:
            return False

        if text.isdigit():
            return False

        return True

    def _looks_like_caption(
        self,
        text: str,
    ) -> bool:
        lower = self._normalize_for_match(text)

        caption_prefixes = [
            "bang ",
            "hinh ",
            "bieu do ",
            "so do ",
            "anh ",
            "figure ",
            "fig.",
            "table ",
            "chart ",
            "diagram ",
        ]

        return any(lower.startswith(prefix) for prefix in caption_prefixes)

    def _looks_like_table_line(
        self,
        text: str,
    ) -> bool:
        if "\t" in text or "|" in text:
            return True

        if re.search(r"\s{3,}", text):
            return True

        digit_count = sum(ch.isdigit() for ch in text)
        digit_ratio = digit_count / max(len(text), 1)

        if digit_ratio > 0.35:
            return True

        return False

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

    def _has_heading_like_length(
        self,
        text: str,
    ) -> bool:
        word_count = len(text.split())

        return 2 <= word_count <= 18

    def _looks_like_sentence(
        self,
        text: str,
    ) -> bool:
        word_count = len(text.split())

        if word_count >= 20:
            return True

        if text.endswith(".") and word_count >= 12:
            return True

        return False

    def _looks_like_date_or_place(
        self,
        text: str,
    ) -> bool:
        lower = text.lower().strip()

        if re.search(r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", lower):
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
        lower = self._normalize_for_match(text)

        keywords = [
            "tong quan",
            "gioi thieu",
            "can cu",
            "muc tieu",
            "pham vi",
            "hien trang",
            "giai phap",
            "kien truc",
            "yeu cau",
            "noi dung",
            "ket qua",
            "ket luan",
            "kien nghi",
            "phu luc",
            "chuong",
            "phan",
            "section",
            "chapter",
            "overview",
            "introduction",
            "conclusion",
            "appendix",
        ]

        return [
            keyword for keyword in keywords
            if keyword in lower
        ]

    def _is_mostly_uppercase(
        self,
        text: str,
    ) -> bool:
        letters = [char for char in text if char.isalpha()]

        if not letters:
            return False

        uppercase_count = sum(1 for char in letters if char.isupper())

        return uppercase_count / max(len(letters), 1) >= 0.65

    def _is_near_left_margin(
        self,
        bbox: Optional[List[float]],
        page_width: Optional[float],
    ) -> bool:
        if not bbox or not page_width:
            return False

        x0 = float(bbox[0])

        return x0 <= float(page_width) * 0.25

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

    def _estimate_body_font_size(
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

    def _normalize_heading_text(
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

        return " ".join(lines).strip()

    def _normalize_for_match(
        self,
        text: str,
    ) -> str:
        text = text.lower().strip()
        text = unicodedata.normalize("NFD", text)
        text = "".join(
            char for char in text
            if unicodedata.category(char) != "Mn"
        )
        text = text.replace("đ", "d")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _deduplicate_candidates(
        self,
        candidates: List[HeadingCandidate],
    ) -> List[HeadingCandidate]:
        seen: Dict[str, HeadingCandidate] = {}

        for candidate in candidates:
            key = (
                f"{candidate.page_number}|"
                f"{self._normalize_for_match(candidate.heading_text)}|"
                f"{candidate.section_number}"
            )

            if key not in seen:
                seen[key] = candidate
                continue

            if candidate.confidence > seen[key].confidence:
                seen[key] = candidate

        return list(seen.values())

    def _sort_candidates(
        self,
        candidates: List[HeadingCandidate],
    ) -> List[HeadingCandidate]:
        return sorted(
            candidates,
            key=lambda item: (
                item.page_number,
                item.bbox[1] if item.bbox else 999999,
                item.bbox[0] if item.bbox else 999999,
            ),
        )

    def _build_summary(
        self,
        candidates: List[HeadingCandidate],
    ) -> Dict[str, Any]:
        by_level: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for candidate in candidates:
            level_key = str(candidate.level)
            by_level[level_key] = by_level.get(level_key, 0) + 1
            by_type[candidate.heading_type] = by_type.get(candidate.heading_type, 0) + 1

        return {
            "has_headings": len(candidates) > 0,
            "heading_count": len(candidates),
            "by_level": by_level,
            "by_type": by_type,
            "first_heading": candidates[0].heading_text if candidates else "",
            "first_heading_page": candidates[0].page_number if candidates else None,
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        candidates: List[HeadingCandidate],
        result: Dict[str, Any],
    ) -> None:
        by_page: Dict[int, List[Dict[str, Any]]] = {}

        for candidate in candidates:
            by_page.setdefault(candidate.page_number, [])
            by_page[candidate.page_number].append(candidate.to_dict())

        for page_raw in page_raws:
            page_raw.metadata.setdefault("heading_detector", {})
            page_raw.metadata["heading_detector"] = {
                "processor": "HeadingDetector",
                "heading_candidates_on_page": by_page.get(page_raw.page_number, []),
                "heading_count_on_page": len(by_page.get(page_raw.page_number, [])),
                "heading_summary": result.get("heading_summary", {}),
            }


def detect_headings(
    page_raws: List[PageRaw],
) -> Dict[str, Any]:
    detector = HeadingDetector()
    return detector.process(page_raws)
