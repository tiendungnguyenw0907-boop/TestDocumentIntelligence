"""
toc_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect Table of Contents pages and parse TOC entries.

Input
-----
List[PageRaw] after PageUnderstandingPipeline.

Output
------
Dictionary with:
- toc_pages
- toc_entries
- toc_summary

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
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class TOCDetectorConfig:
    max_pages_to_scan: int = 15
    min_toc_score: float = 0.35
    min_entries_for_toc_page: int = 3

    detect_dot_leaders: bool = True
    detect_page_number_at_line_end: bool = True
    detect_toc_title_keywords: bool = True

    include_debug: bool = True
    attach_to_pages: bool = True

    max_line_length: int = 500
    min_entry_title_length: int = 2
    max_entry_title_length: int = 300


@dataclass
class TOCPageCandidate:
    toc_page_id: str
    page_number: int
    page_index: int
    confidence: float
    toc_title_found: bool
    entry_like_line_count: int
    dot_leader_line_count: int
    source: str
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class TOCEntry:
    toc_entry_id: str
    source_page_number: int
    title: str
    raw_text: str

    target_page_number: Optional[int] = None
    level: int = 1
    section_number: str = ""
    entry_type: str = "toc_entry"
    confidence: float = 0.5
    bbox: Optional[List[float]] = None
    source_object_id: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class TOCDetector:
    def __init__(
        self,
        config: Optional[TOCDetectorConfig] = None,
    ):
        self.config = config or TOCDetectorConfig()

    def process(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        toc_pages = self.detect_toc_pages(page_raws)
        toc_entries = self.parse_toc_entries(page_raws, toc_pages)

        result = {
            "processor": "TOCDetector",
            "toc_pages": [
                item.to_dict() for item in toc_pages
            ],
            "toc_entries": [
                item.to_dict() for item in toc_entries
            ],
            "toc_summary": {
                "has_toc": len(toc_pages) > 0,
                "toc_page_count": len(toc_pages),
                "toc_entry_count": len(toc_entries),
                "toc_page_numbers": [
                    item.page_number for item in toc_pages
                ],
                "first_toc_page": toc_pages[0].page_number if toc_pages else None,
            },
            "config": {
                "max_pages_to_scan": self.config.max_pages_to_scan,
                "min_toc_score": self.config.min_toc_score,
                "min_entries_for_toc_page": self.config.min_entries_for_toc_page,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                result=result,
            )

        return result

    def detect_toc_pages(
        self,
        page_raws: List[PageRaw],
    ) -> List[TOCPageCandidate]:
        scan_pages = page_raws[: self.config.max_pages_to_scan]
        candidates: List[TOCPageCandidate] = []

        for page_raw in scan_pages:
            lines = self._get_page_lines(page_raw)
            score, details = self._score_toc_page(page_raw, lines)

            if score < self.config.min_toc_score:
                continue

            if details["entry_like_line_count"] < self.config.min_entries_for_toc_page:
                if not details["toc_title_found"]:
                    continue

            candidates.append(
                TOCPageCandidate(
                    toc_page_id=make_id("tocpage"),
                    page_number=page_raw.page_number,
                    page_index=page_raw.page_index,
                    confidence=round(score, 4),
                    toc_title_found=details["toc_title_found"],
                    entry_like_line_count=details["entry_like_line_count"],
                    dot_leader_line_count=details["dot_leader_line_count"],
                    source="toc_detector.page_text",
                    metadata=details,
                )
            )

        candidates = sorted(
            candidates,
            key=lambda item: (
                item.page_number,
                -item.confidence,
            ),
        )

        return candidates

    def parse_toc_entries(
        self,
        page_raws: List[PageRaw],
        toc_pages: List[TOCPageCandidate],
    ) -> List[TOCEntry]:
        toc_page_numbers = {
            item.page_number for item in toc_pages
        }

        if not toc_page_numbers:
            return []

        entries: List[TOCEntry] = []

        for page_raw in page_raws:
            if page_raw.page_number not in toc_page_numbers:
                continue

            text_objects = self._collect_text_objects(page_raw)

            for obj in text_objects:
                text = obj.get("text") or ""
                bbox = obj.get("bbox")
                object_id = obj.get("object_id", "")

                for line_index, line in enumerate(text.splitlines()):
                    line = self._clean_line(line)

                    if not line:
                        continue

                    parsed = self._parse_toc_line(line)

                    if parsed is None:
                        continue

                    title = parsed["title"]

                    if not self._is_valid_entry_title(title):
                        continue

                    entries.append(
                        TOCEntry(
                            toc_entry_id=make_id("tocentry"),
                            source_page_number=page_raw.page_number,
                            title=title,
                            raw_text=line,
                            target_page_number=parsed.get("target_page_number"),
                            level=parsed.get("level", 1),
                            section_number=parsed.get("section_number", ""),
                            entry_type=parsed.get("entry_type", "toc_entry"),
                            confidence=parsed.get("confidence", 0.5),
                            bbox=bbox,
                            source_object_id=object_id,
                            metadata={
                                "line_index": line_index,
                                "source": obj.get("source", "unknown"),
                                "source_index": obj.get("source_index"),
                            },
                        )
                    )

        entries = self._deduplicate_entries(entries)
        entries = self._sort_entries(entries)

        return entries

    def _score_toc_page(
        self,
        page_raw: PageRaw,
        lines: List[str],
    ) -> Tuple[float, Dict[str, Any]]:
        toc_title_found = False
        entry_like_line_count = 0
        dot_leader_line_count = 0
        page_number_line_count = 0

        for line in lines:
            clean = self._clean_line(line)

            if not clean:
                continue

            if self._looks_like_toc_title(clean):
                toc_title_found = True

            if self._has_dot_leader(clean):
                dot_leader_line_count += 1

            if self._has_page_number_at_end(clean):
                page_number_line_count += 1

            if self._parse_toc_line(clean) is not None:
                entry_like_line_count += 1

        score = 0.0

        if toc_title_found:
            score += 0.45

        if entry_like_line_count >= self.config.min_entries_for_toc_page:
            score += 0.30

        if dot_leader_line_count >= 2:
            score += 0.15

        if page_number_line_count >= self.config.min_entries_for_toc_page:
            score += 0.10

        if page_raw.page_number <= 5:
            score += 0.05
        elif page_raw.page_number <= 10:
            score += 0.02

        score = min(score, 0.98)

        details = {
            "toc_title_found": toc_title_found,
            "entry_like_line_count": entry_like_line_count,
            "dot_leader_line_count": dot_leader_line_count,
            "page_number_line_count": page_number_line_count,
            "line_count": len(lines),
            "page_number": page_raw.page_number,
        }

        return score, details

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

            if not text.strip():
                continue

            objects.append(
                {
                    "object_id": item.get("item_id") or f"reading_item_{index}",
                    "text": text,
                    "bbox": bbox,
                    "source": "reading_order_builder.reading_order_items",
                    "source_index": index,
                }
            )

        if objects:
            return objects

        for index, block in enumerate(page_raw.text_blocks):
            text = block.normalized_text or block.text or ""

            if not text.strip():
                continue

            objects.append(
                {
                    "object_id": block.block_id,
                    "text": text,
                    "bbox": block.bbox,
                    "source": "page_raw.text_blocks",
                    "source_index": index,
                }
            )

        if objects:
            return objects

        for index, line in enumerate(page_raw.text_lines):
            text = line.normalized_text or line.text or ""

            if not text.strip():
                continue

            objects.append(
                {
                    "object_id": line.line_id,
                    "text": text,
                    "bbox": line.bbox,
                    "source": "page_raw.text_lines",
                    "source_index": index,
                }
            )

        return objects

    def _get_page_lines(
        self,
        page_raw: PageRaw,
    ) -> List[str]:
        reading_meta = page_raw.metadata.get("reading_order_builder", {})
        reading_text = reading_meta.get("reading_order_text", "")

        if reading_text:
            return reading_text.splitlines()

        text = page_raw.normalized_text or page_raw.raw_text or ""

        return text.splitlines()

    def _parse_toc_line(
        self,
        line: str,
    ) -> Optional[Dict[str, Any]]:
        if not line:
            return None

        if len(line) > self.config.max_line_length:
            return None

        clean = self._clean_line(line)

        if self._looks_like_toc_title(clean):
            return None

        patterns = [
            r"^(?P<prefix>(?:\d+(?:\.\d+)*\.?|[IVXLCDM]+\.?|Chương\s+\d+|Chuong\s+\d+|Phần\s+[IVXLCDM\d]+|Phan\s+[IVXLCDM\d]+))\s*(?P<title>.+?)\s*(?:\.{2,}|…+|\s{2,})\s*(?P<page>\d{1,4})$",
            r"^(?P<title>.+?)\s*(?:\.{2,}|…+|\s{2,})\s*(?P<page>\d{1,4})$",
            r"^(?P<prefix>(?:\d+(?:\.\d+)*\.?|[IVXLCDM]+\.?))\s+(?P<title>.+?)\s+(?P<page>\d{1,4})$",
        ]

        for pattern in patterns:
            match = re.match(
                pattern,
                clean,
                flags=re.IGNORECASE,
            )

            if not match:
                continue

            data = match.groupdict()

            prefix = (data.get("prefix") or "").strip()
            title = (data.get("title") or "").strip()
            page_text = data.get("page")

            title = self._remove_dot_leader_from_title(title)
            title = self._clean_line(title)

            if not title:
                continue

            target_page_number = None

            if page_text:
                try:
                    target_page_number = int(page_text)
                except Exception:
                    target_page_number = None

            level = self._infer_level(prefix, title)
            section_number = self._extract_section_number(prefix, title)
            entry_type = self._infer_entry_type(prefix, title)

            confidence = self._score_entry_line(
                line=clean,
                prefix=prefix,
                title=title,
                target_page_number=target_page_number,
            )

            return {
                "title": title,
                "target_page_number": target_page_number,
                "level": level,
                "section_number": section_number,
                "entry_type": entry_type,
                "confidence": confidence,
            }

        return None

    def _score_entry_line(
        self,
        line: str,
        prefix: str,
        title: str,
        target_page_number: Optional[int],
    ) -> float:
        score = 0.45

        if prefix:
            score += 0.15

        if target_page_number is not None:
            score += 0.15

        if self._has_dot_leader(line):
            score += 0.15

        if 5 <= len(title) <= 120:
            score += 0.08

        if self._looks_like_heading_title(title):
            score += 0.05

        return round(max(0.0, min(score, 0.95)), 4)

    def _looks_like_toc_title(
        self,
        text: str,
    ) -> bool:
        lower = self._normalize_for_match(text)

        toc_titles = [
            "muc luc",
            "bang muc luc",
            "noi dung",
            "table of contents",
            "contents",
        ]

        return lower in toc_titles

    def _has_dot_leader(
        self,
        text: str,
    ) -> bool:
        if not self.config.detect_dot_leaders:
            return False

        if "..." in text:
            return True

        if "……" in text or "…" in text:
            return True

        if re.search(r"\.{2,}", text):
            return True

        return False

    def _has_page_number_at_end(
        self,
        text: str,
    ) -> bool:
        if not self.config.detect_page_number_at_line_end:
            return False

        return re.search(r"\b\d{1,4}\s*$", text.strip()) is not None

    def _is_valid_entry_title(
        self,
        title: str,
    ) -> bool:
        if not title:
            return False

        title = title.strip()

        if len(title) < self.config.min_entry_title_length:
            return False

        if len(title) > self.config.max_entry_title_length:
            return False

        if title.isdigit():
            return False

        return True

    def _looks_like_heading_title(
        self,
        title: str,
    ) -> bool:
        lower = self._normalize_for_match(title)

        keywords = [
            "chuong",
            "phan",
            "muc",
            "gioi thieu",
            "tong quan",
            "ket luan",
            "phu luc",
            "noi dung",
        ]

        return any(keyword in lower for keyword in keywords)

    def _infer_level(
        self,
        prefix: str,
        title: str,
    ) -> int:
        prefix_clean = prefix.strip()

        if not prefix_clean:
            return 1

        lower = self._normalize_for_match(prefix_clean)

        if lower.startswith("phan"):
            return 1

        if lower.startswith("chuong"):
            return 1

        number_match = re.search(r"\d+(?:\.\d+)*", prefix_clean)

        if number_match:
            number = number_match.group(0).strip(".")
            return max(1, number.count(".") + 1)

        roman_match = re.match(r"^[IVXLCDM]+\.?$", prefix_clean, flags=re.IGNORECASE)

        if roman_match:
            return 1

        return 1

    def _extract_section_number(
        self,
        prefix: str,
        title: str,
    ) -> str:
        prefix_clean = prefix.strip()

        if prefix_clean:
            return prefix_clean.rstrip(".")

        match = re.match(r"^(\d+(?:\.\d+)*)\.?\s+", title)

        if match:
            return match.group(1)

        return ""

    def _infer_entry_type(
        self,
        prefix: str,
        title: str,
    ) -> str:
        combined = self._normalize_for_match(f"{prefix} {title}")

        if combined.startswith("phan"):
            return "part"

        if combined.startswith("chuong"):
            return "chapter"

        if combined.startswith("phu luc"):
            return "appendix"

        if re.match(r"^\d+(?:\.\d+)*", combined):
            return "section"

        return "toc_entry"

    def _remove_dot_leader_from_title(
        self,
        title: str,
    ) -> str:
        title = re.sub(r"\.{2,}", " ", title)
        title = title.replace("…", " ")
        title = title.replace("……", " ")

        return title.strip()

    def _deduplicate_entries(
        self,
        entries: List[TOCEntry],
    ) -> List[TOCEntry]:
        seen: Dict[str, TOCEntry] = {}

        for entry in entries:
            key = f"{entry.source_page_number}|{entry.title.lower()}|{entry.target_page_number}"

            if key not in seen:
                seen[key] = entry
                continue

            if entry.confidence > seen[key].confidence:
                seen[key] = entry

        return list(seen.values())

    def _sort_entries(
        self,
        entries: List[TOCEntry],
    ) -> List[TOCEntry]:
        return sorted(
            entries,
            key=lambda item: (
                item.source_page_number,
                item.target_page_number if item.target_page_number is not None else 999999,
                item.level,
            ),
        )

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        toc_page_numbers = {
            item["page_number"] for item in result.get("toc_pages", [])
        }

        for page_raw in page_raws[: self.config.max_pages_to_scan]:
            page_raw.metadata.setdefault("toc_detector", {})

            page_raw.metadata["toc_detector"] = {
                "processor": "TOCDetector",
                "is_toc_page": page_raw.page_number in toc_page_numbers,
                "toc_summary": result.get("toc_summary", {}),
            }

            if page_raw.page_number in toc_page_numbers:
                page_raw.metadata["toc_detector"]["toc_entries_on_page"] = [
                    entry
                    for entry in result.get("toc_entries", [])
                    if entry.get("source_page_number") == page_raw.page_number
                ]

    def _clean_line(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

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


def detect_toc(
    page_raws: List[PageRaw],
) -> Dict[str, Any]:
    detector = TOCDetector()
    return detector.process(page_raws)
