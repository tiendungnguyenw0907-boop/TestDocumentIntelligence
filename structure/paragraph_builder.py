"""
paragraph_builder.py

Production V1 - Colab Ready

Purpose
-------
Build clean paragraph objects from page-understanding output and link them to sections.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- TitleDetector
- TOCDetector
- HeadingDetector
- SectionBuilder

Output
------
Dictionary with:
- paragraphs
- paragraphs_by_section
- paragraph_summary

Flow
----
SectionBuilder
    ↓
ParagraphBuilder
    ↓
ListDetector
    ↓
DocumentTreeBuilder
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class ParagraphBuilderConfig:
    min_paragraph_chars: int = 2
    max_paragraph_chars: int = 6000

    use_merged_paragraphs_first: bool = True
    use_reading_order_fallback: bool = True
    use_text_blocks_fallback: bool = True
    use_text_lines_fallback: bool = True

    exclude_header_footer: bool = True
    exclude_captions: bool = False
    exclude_toc_pages: bool = False
    exclude_empty_text: bool = True

    attach_to_pages: bool = True

    include_text_preview: bool = True
    text_preview_chars: int = 500

    merge_short_continuation_lines: bool = True
    short_line_merge_threshold: int = 60


@dataclass
class DocumentParagraph:
    paragraph_id: str
    page_number: int
    page_index: int
    order: int
    text: str

    bbox: Optional[List[float]] = None
    section_id: str = ""
    section_title: str = ""
    paragraph_type: str = "paragraph"
    source: str = "unknown"
    source_object_ids: Optional[List[str]] = None
    char_count: int = 0
    word_count: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["source_object_ids"] is None:
            data["source_object_ids"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class ParagraphBuilder:
    def __init__(
        self,
        config: Optional[ParagraphBuilderConfig] = None,
    ):
        self.config = config or ParagraphBuilderConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        section_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sections = self._collect_sections(
            page_raws=page_raws,
            section_result=section_result,
        )

        toc_page_numbers = self._collect_toc_page_numbers(page_raws)
        caption_object_ids = self._collect_caption_object_ids(page_raws)

        paragraphs: List[DocumentParagraph] = []

        for page_raw in page_raws:
            if self.config.exclude_toc_pages and page_raw.page_number in toc_page_numbers:
                continue

            text_objects = self._collect_paragraph_objects(page_raw)

            page_paragraphs = self._build_page_paragraphs(
                page_raw=page_raw,
                text_objects=text_objects,
                sections=sections,
                caption_object_ids=caption_object_ids,
            )

            paragraphs.extend(page_paragraphs)

        paragraphs = self._deduplicate_paragraphs(paragraphs)
        paragraphs = self._sort_paragraphs(paragraphs)

        for order, paragraph in enumerate(paragraphs, start=1):
            paragraph.order = order

        paragraphs_by_section = self._group_by_section(paragraphs)

        result = {
            "processor": "ParagraphBuilder",
            "paragraphs": [
                paragraph.to_dict() for paragraph in paragraphs
            ],
            "paragraphs_by_section": paragraphs_by_section,
            "paragraph_summary": self._build_summary(paragraphs),
            "config": {
                "min_paragraph_chars": self.config.min_paragraph_chars,
                "max_paragraph_chars": self.config.max_paragraph_chars,
                "exclude_header_footer": self.config.exclude_header_footer,
                "exclude_captions": self.config.exclude_captions,
                "exclude_toc_pages": self.config.exclude_toc_pages,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                paragraphs=paragraphs,
                result=result,
            )

        return result

    def _collect_paragraph_objects(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        if self.config.use_merged_paragraphs_first:
            objects = self._collect_from_merged_paragraphs(page_raw)

            if objects:
                return objects

        if self.config.use_reading_order_fallback:
            objects = self._collect_from_reading_order(page_raw)

            if objects:
                return objects

        if self.config.use_text_blocks_fallback:
            objects = self._collect_from_text_blocks(page_raw)

            if objects:
                return objects

        if self.config.use_text_lines_fallback:
            return self._collect_from_text_lines(page_raw)

        return []

    def _collect_from_merged_paragraphs(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []

        merger_meta = page_raw.metadata.get("object_merger", {})
        paragraphs = merger_meta.get("merged_paragraphs", [])

        for index, item in enumerate(paragraphs):
            text = (
                item.get("normalized_text")
                or item.get("text")
                or item.get("raw_text")
                or ""
            )

            text = self._clean_paragraph_text(text)

            if not self._is_valid_paragraph_text(text):
                continue

            objects.append(
                {
                    "object_id": item.get("paragraph_id") or item.get("block_id") or f"merged_paragraph_{index}",
                    "text": text,
                    "bbox": item.get("bbox"),
                    "source": "object_merger.merged_paragraphs",
                    "source_index": index,
                    "source_object_ids": item.get("source_line_ids", []),
                    "metadata": item.get("metadata", {}),
                }
            )

        return objects

    def _collect_from_reading_order(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []

        reading_meta = page_raw.metadata.get("reading_order_builder", {})
        reading_items = reading_meta.get("reading_order_items", [])

        for index, item in enumerate(reading_items):
            text = item.get("text") or ""
            text = self._clean_paragraph_text(text)

            if not self._is_valid_paragraph_text(text):
                continue

            objects.append(
                {
                    "object_id": item.get("item_id") or f"reading_item_{index}",
                    "text": text,
                    "bbox": item.get("bbox"),
                    "source": "reading_order_builder.reading_order_items",
                    "source_index": index,
                    "source_object_ids": [item.get("item_id") or f"reading_item_{index}"],
                    "metadata": item.get("metadata", {}),
                }
            )

        return objects

    def _collect_from_text_blocks(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []

        for index, block in enumerate(page_raw.text_blocks):
            text = block.normalized_text or block.text or ""
            text = self._clean_paragraph_text(text)

            if not self._is_valid_paragraph_text(text):
                continue

            objects.append(
                {
                    "object_id": block.block_id,
                    "text": text,
                    "bbox": block.bbox,
                    "source": "page_raw.text_blocks",
                    "source_index": index,
                    "source_object_ids": [block.block_id],
                    "metadata": block.metadata or {},
                }
            )

        return objects

    def _collect_from_text_lines(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        line_objects: List[Dict[str, Any]] = []

        for index, line in enumerate(page_raw.text_lines):
            text = line.normalized_text or line.text or ""
            text = self._clean_line_text(text)

            if not text:
                continue

            line_objects.append(
                {
                    "object_id": line.line_id,
                    "text": text,
                    "bbox": line.bbox,
                    "source": "page_raw.text_lines",
                    "source_index": index,
                    "source_object_ids": [line.line_id],
                    "metadata": line.metadata or {},
                }
            )

        if not self.config.merge_short_continuation_lines:
            return [
                item for item in line_objects
                if self._is_valid_paragraph_text(item.get("text", ""))
            ]

        return self._merge_lines_to_paragraph_objects(line_objects)

    def _merge_lines_to_paragraph_objects(
        self,
        line_objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not line_objects:
            return []

        merged: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None

        for item in line_objects:
            text = item.get("text", "")

            if current is None:
                current = {
                    "object_id": make_id("merged_line_para"),
                    "text": text,
                    "bbox": item.get("bbox"),
                    "source": "page_raw.text_lines_merged",
                    "source_index": item.get("source_index"),
                    "source_object_ids": list(item.get("source_object_ids", [])),
                    "metadata": {
                        "merged_from": "text_lines",
                    },
                }
                continue

            previous_text = current.get("text", "")
            should_merge = self._should_merge_line(
                previous_text=previous_text,
                current_text=text,
            )

            if should_merge:
                current["text"] = self._clean_paragraph_text(previous_text + " " + text)
                current["bbox"] = self._merge_bbox(current.get("bbox"), item.get("bbox"))
                current["source_object_ids"].extend(item.get("source_object_ids", []))
            else:
                if self._is_valid_paragraph_text(current.get("text", "")):
                    merged.append(current)

                current = {
                    "object_id": make_id("merged_line_para"),
                    "text": text,
                    "bbox": item.get("bbox"),
                    "source": "page_raw.text_lines_merged",
                    "source_index": item.get("source_index"),
                    "source_object_ids": list(item.get("source_object_ids", [])),
                    "metadata": {
                        "merged_from": "text_lines",
                    },
                }

        if current and self._is_valid_paragraph_text(current.get("text", "")):
            merged.append(current)

        return merged

    def _build_page_paragraphs(
        self,
        page_raw: PageRaw,
        text_objects: List[Dict[str, Any]],
        sections: List[Dict[str, Any]],
        caption_object_ids: set,
    ) -> List[DocumentParagraph]:
        paragraphs: List[DocumentParagraph] = []

        for index, obj in enumerate(text_objects):
            text = obj.get("text") or ""
            text = self._clean_paragraph_text(text)

            if not self._is_valid_paragraph_text(text):
                continue

            if self.config.exclude_header_footer and self._is_header_footer_object(page_raw, obj):
                continue

            if self.config.exclude_captions:
                object_id = obj.get("object_id", "")

                if object_id in caption_object_ids:
                    continue

                if self._looks_like_caption(text):
                    continue

            section = self._find_section_for_paragraph(
                page_raw=page_raw,
                obj=obj,
                sections=sections,
            )

            paragraph_type = self._classify_paragraph_type(text)

            paragraphs.append(
                DocumentParagraph(
                    paragraph_id=make_id("para"),
                    page_number=page_raw.page_number,
                    page_index=page_raw.page_index,
                    order=index + 1,
                    text=text,
                    bbox=obj.get("bbox"),
                    section_id=section.get("section_id", "") if section else "",
                    section_title=section.get("title", "") if section else "",
                    paragraph_type=paragraph_type,
                    source=obj.get("source", "unknown"),
                    source_object_ids=list(obj.get("source_object_ids", [])),
                    char_count=len(text),
                    word_count=len(text.split()),
                    metadata={
                        "source_index": obj.get("source_index"),
                        "source_object_id": obj.get("object_id", ""),
                        "text_preview": self._text_preview(text),
                        "object_metadata": obj.get("metadata", {}),
                    },
                )
            )

        return paragraphs

    def _collect_sections(
        self,
        page_raws: List[PageRaw],
        section_result: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if section_result:
            sections = section_result.get("sections", [])

            if sections:
                return sections

        collected: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            section_meta = page_raw.metadata.get("section_builder", {})
            sections_on_page = section_meta.get("sections_on_page", [])

            for section in sections_on_page:
                collected.append(section)

        return self._deduplicate_sections(collected)

    def _find_section_for_paragraph(
        self,
        page_raw: PageRaw,
        obj: Dict[str, Any],
        sections: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        page_number = page_raw.page_number
        candidates = []

        for section in sections:
            if section.get("section_type") == "root":
                continue

            pages = section.get("content_page_numbers") or []

            if page_number in pages:
                candidates.append(section)
                continue

            page_start = section.get("page_start")
            page_end = section.get("page_end")

            if page_start is not None and page_end is not None:
                if page_start <= page_number <= page_end:
                    candidates.append(section)

        if candidates:
            candidates = sorted(
                candidates,
                key=lambda item: (
                    -self._safe_int(item.get("level", 1), default=1),
                    item.get("order", 0),
                ),
            )

            return candidates[0]

        root_sections = [
            section for section in sections
            if section.get("section_type") == "root"
        ]

        return root_sections[0] if root_sections else None

    def _collect_toc_page_numbers(
        self,
        page_raws: List[PageRaw],
    ) -> set:
        page_numbers = set()

        for page_raw in page_raws:
            toc_meta = page_raw.metadata.get("toc_detector", {})

            if toc_meta.get("is_toc_page"):
                page_numbers.add(page_raw.page_number)

        return page_numbers

    def _collect_caption_object_ids(
        self,
        page_raws: List[PageRaw],
    ) -> set:
        object_ids = set()

        for page_raw in page_raws:
            caption_meta = page_raw.metadata.get("caption_detector", {})
            captions = caption_meta.get("caption_candidates", [])

            for caption in captions:
                for object_id in caption.get("source_object_ids", []):
                    if object_id:
                        object_ids.add(object_id)

        return object_ids

    def _group_by_section(
        self,
        paragraphs: List[DocumentParagraph],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for paragraph in paragraphs:
            section_id = paragraph.section_id or "unassigned"
            grouped.setdefault(section_id, [])
            grouped[section_id].append(paragraph.to_dict())

        return grouped

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        paragraphs: List[DocumentParagraph],
        result: Dict[str, Any],
    ) -> None:
        by_page: Dict[int, List[Dict[str, Any]]] = {}

        for paragraph in paragraphs:
            by_page.setdefault(paragraph.page_number, [])
            by_page[paragraph.page_number].append(paragraph.to_dict())

        for page_raw in page_raws:
            page_paragraphs = by_page.get(page_raw.page_number, [])

            page_raw.metadata.setdefault("paragraph_builder", {})
            page_raw.metadata["paragraph_builder"] = {
                "processor": "ParagraphBuilder",
                "paragraphs_on_page": page_paragraphs,
                "paragraph_count_on_page": len(page_paragraphs),
                "paragraph_summary": result.get("paragraph_summary", {}),
            }

    def _build_summary(
        self,
        paragraphs: List[DocumentParagraph],
    ) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        by_section: Dict[str, int] = {}

        total_words = 0
        total_chars = 0

        for paragraph in paragraphs:
            by_type[paragraph.paragraph_type] = by_type.get(paragraph.paragraph_type, 0) + 1

            section_key = paragraph.section_id or "unassigned"
            by_section[section_key] = by_section.get(section_key, 0) + 1

            total_words += paragraph.word_count
            total_chars += paragraph.char_count

        return {
            "has_paragraphs": len(paragraphs) > 0,
            "paragraph_count": len(paragraphs),
            "total_words": total_words,
            "total_chars": total_chars,
            "avg_words_per_paragraph": round(total_words / max(len(paragraphs), 1), 2),
            "avg_chars_per_paragraph": round(total_chars / max(len(paragraphs), 1), 2),
            "by_type": by_type,
            "section_count_with_paragraphs": len(by_section),
        }

    def _deduplicate_paragraphs(
        self,
        paragraphs: List[DocumentParagraph],
    ) -> List[DocumentParagraph]:
        seen: Dict[str, DocumentParagraph] = {}

        for paragraph in paragraphs:
            key = (
                f"{paragraph.page_number}|"
                f"{self._normalize_for_key(paragraph.text)}|"
                f"{paragraph.section_id}"
            )

            if key not in seen:
                seen[key] = paragraph
                continue

            if paragraph.char_count > seen[key].char_count:
                seen[key] = paragraph

        return list(seen.values())

    def _deduplicate_sections(
        self,
        sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen: Dict[str, Dict[str, Any]] = {}

        for section in sections:
            section_id = section.get("section_id")

            if not section_id:
                continue

            if section_id not in seen:
                seen[section_id] = section

        return list(seen.values())

    def _sort_paragraphs(
        self,
        paragraphs: List[DocumentParagraph],
    ) -> List[DocumentParagraph]:
        return sorted(
            paragraphs,
            key=lambda item: (
                item.page_number,
                item.bbox[1] if item.bbox else 999999,
                item.bbox[0] if item.bbox else 999999,
                item.order,
            ),
        )

    def _should_merge_line(
        self,
        previous_text: str,
        current_text: str,
    ) -> bool:
        if not previous_text or not current_text:
            return False

        if previous_text.endswith((".", ":", ";", "!", "?")):
            return False

        if self._looks_like_heading(current_text):
            return False

        if len(previous_text) <= self.config.short_line_merge_threshold:
            return True

        if current_text[:1].islower():
            return True

        return False

    def _classify_paragraph_type(
        self,
        text: str,
    ) -> str:
        clean = text.strip()

        if self._looks_like_heading(clean):
            return "heading_like_paragraph"

        if self._looks_like_caption(clean):
            return "caption_like_paragraph"

        if self._looks_like_list_item(clean):
            return "list_item_like_paragraph"

        if self._looks_like_table_line(clean):
            return "table_like_paragraph"

        return "paragraph"

    def _looks_like_heading(
        self,
        text: str,
    ) -> bool:
        if not text:
            return False

        if re.match(r"^\d+(?:\.\d+)*\.?\s+\S+", text):
            return True

        if re.match(r"^[IVXLCDM]+\.?\s+\S+", text):
            return True

        if len(text.split()) <= 12 and self._is_mostly_uppercase(text):
            return True

        return False

    def _looks_like_caption(
        self,
        text: str,
    ) -> bool:
        lower = self._normalize_for_key(text)

        prefixes = [
            "bang ",
            "hinh ",
            "bieu do ",
            "so do ",
            "anh ",
            "table ",
            "figure ",
            "fig.",
            "chart ",
            "diagram ",
            "image ",
        ]

        return any(lower.startswith(prefix) for prefix in prefixes)

    def _looks_like_list_item(
        self,
        text: str,
    ) -> bool:
        patterns = [
            r"^\s*[-–—•]\s+",
            r"^\s*\d+[\.\)]\s+",
            r"^\s*[a-zA-Z][\.\)]\s+",
        ]

        for pattern in patterns:
            if re.match(pattern, text):
                return True

        return False

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

        return digit_ratio > 0.45

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

        if ratio <= 0.06:
            return True

        if ratio >= 0.94:
            return True

        return False

    def _is_valid_paragraph_text(
        self,
        text: str,
    ) -> bool:
        if text is None:
            return False

        clean = text.strip()

        if self.config.exclude_empty_text and not clean:
            return False

        if len(clean) < self.config.min_paragraph_chars:
            return False

        if len(clean) > self.config.max_paragraph_chars:
            return False

        return True

    def _clean_line_text(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _clean_paragraph_text(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.replace("\u00a0", " ")
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _text_preview(
        self,
        text: str,
    ) -> str:
        if not self.config.include_text_preview:
            return ""

        return text[: self.config.text_preview_chars]

    def _normalize_for_key(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.lower().strip()
        text = text.replace("đ", "d")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _is_mostly_uppercase(
        self,
        text: str,
    ) -> bool:
        letters = [char for char in text if char.isalpha()]

        if not letters:
            return False

        uppercase_count = sum(1 for char in letters if char.isupper())

        return uppercase_count / max(len(letters), 1) >= 0.65

    def _merge_bbox(
        self,
        bbox_a: Optional[List[float]],
        bbox_b: Optional[List[float]],
    ) -> Optional[List[float]]:
        if not bbox_a:
            return bbox_b

        if not bbox_b:
            return bbox_a

        return [
            min(float(bbox_a[0]), float(bbox_b[0])),
            min(float(bbox_a[1]), float(bbox_b[1])),
            max(float(bbox_a[2]), float(bbox_b[2])),
            max(float(bbox_a[3]), float(bbox_b[3])),
        ]

    def _safe_int(
        self,
        value: Any,
        default: int = 0,
    ) -> int:
        try:
            if value is None:
                return default

            return int(value)
        except Exception:
            return default


def build_paragraphs(
    page_raws: List[PageRaw],
    section_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = ParagraphBuilder()
    return builder.process(
        page_raws=page_raws,
        section_result=section_result,
    )
