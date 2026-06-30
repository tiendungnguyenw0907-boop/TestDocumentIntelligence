"""
section_builder.py

Production V1 - Colab Ready

Purpose
-------
Build document sections from detected headings.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- TitleDetector
- TOCDetector
- HeadingDetector

Output
------
Dictionary with:
- root_section
- sections
- section_tree
- section_summary

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
class SectionBuilderConfig:
    create_root_section: bool = True
    root_title_fallback: str = "Document"

    max_heading_level: int = 9
    attach_to_pages: bool = True

    include_section_text_preview: bool = True
    text_preview_chars: int = 1000

    include_empty_sections: bool = True
    use_toc_entries_as_fallback: bool = True


@dataclass
class DocumentSection:
    section_id: str
    title: str
    level: int
    order: int

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    parent_id: str = ""
    heading_id: str = ""
    section_number: str = ""
    section_type: str = "section"

    bbox: Optional[List[float]] = None
    children: Optional[List[str]] = None
    content_page_numbers: Optional[List[int]] = None
    text_preview: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["children"] is None:
            data["children"] = []

        if data["content_page_numbers"] is None:
            data["content_page_numbers"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class SectionBuilder:
    def __init__(
        self,
        config: Optional[SectionBuilderConfig] = None,
    ):
        self.config = config or SectionBuilderConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        heading_result: Optional[Dict[str, Any]] = None,
        title_result: Optional[Dict[str, Any]] = None,
        toc_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        headings = self._collect_headings(
            page_raws=page_raws,
            heading_result=heading_result,
        )

        if not headings and self.config.use_toc_entries_as_fallback:
            headings = self._headings_from_toc(
                toc_result=toc_result,
                page_raws=page_raws,
            )

        root_title = self._resolve_root_title(
            page_raws=page_raws,
            title_result=title_result,
        )

        sections = self._build_sections(
            page_raws=page_raws,
            headings=headings,
            root_title=root_title,
        )

        sections = self._assign_page_ranges(
            page_raws=page_raws,
            sections=sections,
        )

        sections = self._attach_text_preview(
            page_raws=page_raws,
            sections=sections,
        )

        section_tree = self._build_section_tree(sections)
        root_section = sections[0] if sections else None

        result = {
            "processor": "SectionBuilder",
            "root_section": root_section.to_dict() if root_section else None,
            "sections": [
                section.to_dict() for section in sections
            ],
            "section_tree": section_tree,
            "section_summary": self._build_summary(sections),
            "config": {
                "create_root_section": self.config.create_root_section,
                "max_heading_level": self.config.max_heading_level,
                "include_empty_sections": self.config.include_empty_sections,
                "use_toc_entries_as_fallback": self.config.use_toc_entries_as_fallback,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                sections=sections,
                result=result,
            )

        return result

    def _collect_headings(
        self,
        page_raws: List[PageRaw],
        heading_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if heading_result:
            headings = heading_result.get("heading_candidates", [])

            if headings:
                return self._normalize_headings(headings)

        collected: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            heading_meta = page_raw.metadata.get("heading_detector", {})
            page_headings = heading_meta.get("heading_candidates_on_page", [])

            for item in page_headings:
                collected.append(item)

        return self._normalize_headings(collected)

    def _normalize_headings(
        self,
        headings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for index, item in enumerate(headings):
            title = (
                item.get("heading_text")
                or item.get("title")
                or item.get("text")
                or ""
            )

            title = self._clean_text(title)

            if not title:
                continue

            level = self._safe_int(
                item.get("level", 1),
                default=1,
            )

            level = max(1, min(level, self.config.max_heading_level))

            bbox = item.get("bbox")
            page_number = self._safe_int(
                item.get("page_number"),
                default=1,
            )

            normalized.append(
                {
                    "heading_id": item.get("heading_id") or item.get("id") or make_id("heading_ref"),
                    "title": title,
                    "level": level,
                    "page_number": page_number,
                    "page_index": self._safe_int(
                        item.get("page_index"),
                        default=max(page_number - 1, 0),
                    ),
                    "bbox": bbox,
                    "section_number": item.get("section_number", ""),
                    "heading_type": item.get("heading_type", "heading"),
                    "confidence": item.get("confidence", 0.5),
                    "source": item.get("source", "heading_detector"),
                    "source_index": index,
                    "metadata": item.get("metadata", {}),
                }
            )

        normalized = sorted(
            normalized,
            key=lambda item: (
                item.get("page_number", 0),
                item.get("bbox", [0, 999999, 0, 0])[1]
                if item.get("bbox")
                else 999999,
                item.get("bbox", [999999, 0, 0, 0])[0]
                if item.get("bbox")
                else 999999,
            ),
        )

        return normalized

    def _headings_from_toc(
        self,
        toc_result: Optional[Dict[str, Any]],
        page_raws: List[PageRaw],
    ) -> List[Dict[str, Any]]:
        if not toc_result:
            toc_result = self._collect_toc_result_from_pages(page_raws)

        if not toc_result:
            return []

        entries = toc_result.get("toc_entries", [])
        headings: List[Dict[str, Any]] = []

        for index, entry in enumerate(entries):
            title = self._clean_text(entry.get("title", ""))

            if not title:
                continue

            target_page = entry.get("target_page_number")

            if target_page is None:
                continue

            target_page_int = self._safe_int(target_page, default=1)

            headings.append(
                {
                    "heading_id": entry.get("toc_entry_id") or make_id("toc_heading"),
                    "title": title,
                    "level": self._safe_int(entry.get("level", 1), default=1),
                    "page_number": target_page_int,
                    "page_index": max(target_page_int - 1, 0),
                    "bbox": entry.get("bbox"),
                    "section_number": entry.get("section_number", ""),
                    "heading_type": "toc_fallback_heading",
                    "confidence": entry.get("confidence", 0.5),
                    "source": "toc_detector.toc_entries",
                    "source_index": index,
                    "metadata": entry.get("metadata", {}),
                }
            )

        return self._normalize_headings(headings)

    def _collect_toc_result_from_pages(
        self,
        page_raws: List[PageRaw],
    ) -> Optional[Dict[str, Any]]:
        toc_entries = []
        toc_summary = {}

        for page_raw in page_raws:
            toc_meta = page_raw.metadata.get("toc_detector", {})

            if not toc_meta:
                continue

            toc_summary = toc_meta.get("toc_summary", toc_summary)

            entries = toc_meta.get("toc_entries_on_page", [])

            for entry in entries:
                toc_entries.append(entry)

        if not toc_entries:
            return None

        return {
            "processor": "TOCDetector",
            "toc_entries": toc_entries,
            "toc_summary": toc_summary,
        }

    def _resolve_root_title(
        self,
        page_raws: List[PageRaw],
        title_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        if title_result:
            selected = title_result.get("selected_title")

            if isinstance(selected, dict):
                title = selected.get("title_text") or ""

                if title:
                    return self._clean_text(title)

        for page_raw in page_raws[:3]:
            title_meta = page_raw.metadata.get("title_detector", {})
            selected = title_meta.get("selected_title")

            if isinstance(selected, dict):
                title = selected.get("title_text") or ""

                if title:
                    return self._clean_text(title)

        return self.config.root_title_fallback

    def _build_sections(
        self,
        page_raws: List[PageRaw],
        headings: List[Dict[str, Any]],
        root_title: str,
    ) -> List[DocumentSection]:
        sections: List[DocumentSection] = []

        page_numbers = [
            page_raw.page_number for page_raw in page_raws
        ]

        first_page = min(page_numbers) if page_numbers else None
        last_page = max(page_numbers) if page_numbers else None

        root_id = ""

        if self.config.create_root_section:
            root = DocumentSection(
                section_id=make_id("section"),
                title=root_title,
                level=0,
                order=0,
                page_start=first_page,
                page_end=last_page,
                parent_id="",
                heading_id="",
                section_number="",
                section_type="root",
                bbox=None,
                children=[],
                content_page_numbers=page_numbers,
                metadata={
                    "created_from": "document_title",
                    "heading_count": len(headings),
                },
            )

            sections.append(root)
            root_id = root.section_id

        stack: List[DocumentSection] = []

        if sections:
            stack.append(sections[0])

        for order, heading in enumerate(headings, start=1):
            level = self._safe_int(heading.get("level", 1), default=1)
            level = max(1, min(level, self.config.max_heading_level))

            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                parent_id = stack[-1].section_id
            else:
                parent_id = root_id

            section = DocumentSection(
                section_id=make_id("section"),
                title=heading.get("title", ""),
                level=level,
                order=order,
                page_start=heading.get("page_number"),
                page_end=None,
                parent_id=parent_id,
                heading_id=heading.get("heading_id", ""),
                section_number=heading.get("section_number", ""),
                section_type=heading.get("heading_type", "section"),
                bbox=heading.get("bbox"),
                children=[],
                content_page_numbers=[],
                metadata={
                    "source": heading.get("source", "heading_detector"),
                    "source_index": heading.get("source_index"),
                    "confidence": heading.get("confidence", 0.5),
                    "heading_metadata": heading.get("metadata", {}),
                },
            )

            sections.append(section)

            if parent_id:
                parent = self._find_section_by_id(sections, parent_id)

                if parent:
                    if parent.children is None:
                        parent.children = []

                    parent.children.append(section.section_id)

            stack.append(section)

        return sections

    def _assign_page_ranges(
        self,
        page_raws: List[PageRaw],
        sections: List[DocumentSection],
    ) -> List[DocumentSection]:
        if not sections:
            return sections

        page_numbers = sorted(
            {
                page_raw.page_number for page_raw in page_raws
            }
        )

        if not page_numbers:
            return sections

        first_page = page_numbers[0]
        last_page = page_numbers[-1]

        heading_sections = [
            section for section in sections
            if section.section_type != "root"
        ]

        heading_sections = sorted(
            heading_sections,
            key=lambda section: (
                section.page_start if section.page_start is not None else 999999,
                section.order,
            ),
        )

        for index, section in enumerate(heading_sections):
            if section.page_start is None:
                section.page_start = first_page

            if index + 1 < len(heading_sections):
                next_section = heading_sections[index + 1]
                next_page = next_section.page_start

                if next_page is None:
                    section.page_end = last_page
                elif next_page <= section.page_start:
                    section.page_end = section.page_start
                else:
                    section.page_end = self._previous_page_number(
                        page_numbers=page_numbers,
                        page_number=next_page,
                    )
            else:
                section.page_end = last_page

            if section.page_end is None:
                section.page_end = last_page

            section.content_page_numbers = [
                page for page in page_numbers
                if section.page_start <= page <= section.page_end
            ]

        root_sections = [
            section for section in sections
            if section.section_type == "root"
        ]

        for root in root_sections:
            root.page_start = first_page
            root.page_end = last_page
            root.content_page_numbers = page_numbers

        return sections

    def _attach_text_preview(
        self,
        page_raws: List[PageRaw],
        sections: List[DocumentSection],
    ) -> List[DocumentSection]:
        if not self.config.include_section_text_preview:
            return sections

        page_text_by_number = {
            page_raw.page_number: self._get_page_text(page_raw)
            for page_raw in page_raws
        }

        for section in sections:
            if section.section_type == "root":
                continue

            page_numbers = section.content_page_numbers or []
            parts: List[str] = []

            for page_number in page_numbers:
                text = page_text_by_number.get(page_number, "")

                if text:
                    parts.append(text)

                joined = "\n".join(parts)

                if len(joined) >= self.config.text_preview_chars:
                    break

            preview = "\n".join(parts).strip()
            section.text_preview = preview[: self.config.text_preview_chars]

        return sections

    def _build_section_tree(
        self,
        sections: List[DocumentSection],
    ) -> Dict[str, Any]:
        by_id = {
            section.section_id: section for section in sections
        }

        def build_node(section: DocumentSection) -> Dict[str, Any]:
            children = []

            for child_id in section.children or []:
                child = by_id.get(child_id)

                if child:
                    children.append(build_node(child))

            return {
                "section_id": section.section_id,
                "title": section.title,
                "level": section.level,
                "order": section.order,
                "page_start": section.page_start,
                "page_end": section.page_end,
                "section_number": section.section_number,
                "section_type": section.section_type,
                "children": children,
            }

        roots = [
            section for section in sections
            if not section.parent_id
        ]

        if not roots and sections:
            roots = [sections[0]]

        if len(roots) == 1:
            return build_node(roots[0])

        return {
            "section_id": "virtual_root",
            "title": "Document",
            "level": 0,
            "order": 0,
            "page_start": None,
            "page_end": None,
            "section_number": "",
            "section_type": "virtual_root",
            "children": [
                build_node(root) for root in roots
            ],
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        sections: List[DocumentSection],
        result: Dict[str, Any],
    ) -> None:
        for page_raw in page_raws:
            sections_on_page = []

            for section in sections:
                pages = section.content_page_numbers or []

                if page_raw.page_number in pages:
                    sections_on_page.append(section.to_dict())

            page_raw.metadata.setdefault("section_builder", {})
            page_raw.metadata["section_builder"] = {
                "processor": "SectionBuilder",
                "sections_on_page": sections_on_page,
                "section_count_on_page": len(sections_on_page),
                "section_summary": result.get("section_summary", {}),
            }

    def _build_summary(
        self,
        sections: List[DocumentSection],
    ) -> Dict[str, Any]:
        by_level: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for section in sections:
            level_key = str(section.level)
            by_level[level_key] = by_level.get(level_key, 0) + 1
            by_type[section.section_type] = by_type.get(section.section_type, 0) + 1

        non_root_sections = [
            section for section in sections
            if section.section_type != "root"
        ]

        return {
            "has_sections": len(non_root_sections) > 0,
            "section_count": len(sections),
            "non_root_section_count": len(non_root_sections),
            "by_level": by_level,
            "by_type": by_type,
            "max_level": max([section.level for section in sections], default=0),
            "first_section_title": non_root_sections[0].title if non_root_sections else "",
            "first_section_page": non_root_sections[0].page_start if non_root_sections else None,
        }

    def _get_page_text(
        self,
        page_raw: PageRaw,
    ) -> str:
        reading_meta = page_raw.metadata.get("reading_order_builder", {})
        reading_text = reading_meta.get("reading_order_text", "")

        if reading_text:
            return reading_text

        return page_raw.normalized_text or page_raw.raw_text or ""

    def _previous_page_number(
        self,
        page_numbers: List[int],
        page_number: int,
    ) -> int:
        previous_pages = [
            page for page in page_numbers
            if page < page_number
        ]

        if not previous_pages:
            return page_numbers[0]

        return previous_pages[-1]

    def _find_section_by_id(
        self,
        sections: List[DocumentSection],
        section_id: str,
    ) -> Optional[DocumentSection]:
        for section in sections:
            if section.section_id == section_id:
                return section

        return None

    def _clean_text(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.replace("\u00a0", " ")
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

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


def build_sections(
    page_raws: List[PageRaw],
    heading_result: Optional[Dict[str, Any]] = None,
    title_result: Optional[Dict[str, Any]] = None,
    toc_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = SectionBuilder()
    return builder.process(
        page_raws=page_raws,
        heading_result=heading_result,
        title_result=title_result,
        toc_result=toc_result,
    )
