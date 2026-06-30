"""
section_linker.py

Production V1 - Colab Ready

Purpose
-------
Link document sections to pages and build section-to-section context links.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- DocumentStructurePipeline

Output
------
Dictionary with:
- section_links
- section_hierarchy_links
- section_sequence_links
- section_links_by_page
- page_sections
- section_link_summary

Flow
----
DocumentStructurePipeline
    ↓
SectionLinker
    ↓
CrossPageContextGraphBuilder
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class SectionLinkerConfig:
    use_document_structure: bool = True
    fallback_from_page_metadata: bool = True
    fallback_detect_from_headings: bool = True

    attach_to_pages: bool = True
    include_root_section: bool = True

    min_heading_confidence: float = 0.35
    max_title_chars: int = 300
    text_preview_chars: int = 800

    include_debug: bool = True


@dataclass
class SectionLink:
    section_link_id: str
    section_id: str
    title: str

    level: int
    order: int

    page_start: Optional[int]
    page_end: Optional[int]
    page_numbers: List[int]

    parent_id: str = ""
    child_ids: Optional[List[str]] = None

    link_type: str = "section_to_pages"
    confidence: float = 0.7
    source: str = "section_linker"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["child_ids"] is None:
            data["child_ids"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class SectionRelation:
    section_relation_id: str
    source_section_id: str
    target_section_id: str
    relation_type: str

    source_page: Optional[int] = None
    target_page: Optional[int] = None

    weight: float = 1.0
    confidence: float = 0.7
    source: str = "section_linker"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class SectionLinker:
    def __init__(
        self,
        config: Optional[SectionLinkerConfig] = None,
    ):
        self.config = config or SectionLinkerConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        document_structure_result = document_structure_result or {}

        sections = self._collect_sections(
            page_raws=page_raws,
            document_structure_result=document_structure_result,
        )

        if not sections and self.config.fallback_detect_from_headings:
            sections = self._detect_sections_from_pages(page_raws)

        section_links = self._build_section_links(
            sections=sections,
            page_raws=page_raws,
        )

        section_links = self._normalize_page_ranges(
            section_links=section_links,
            page_raws=page_raws,
        )

        hierarchy_links = self._build_hierarchy_links(section_links)
        sequence_links = self._build_sequence_links(section_links)

        section_link_dicts = [
            item.to_dict() for item in section_links
        ]

        hierarchy_link_dicts = [
            item.to_dict() for item in hierarchy_links
        ]

        sequence_link_dicts = [
            item.to_dict() for item in sequence_links
        ]

        result = {
            "processor": "SectionLinker",
            "schema_version": "section_linker_v1",
            "section_links": section_link_dicts,
            "section_hierarchy_links": hierarchy_link_dicts,
            "section_sequence_links": sequence_link_dicts,
            "section_links_by_page": self._group_section_links_by_page(section_link_dicts),
            "page_sections": self._build_page_sections(
                page_raws=page_raws,
                section_links=section_link_dicts,
            ),
            "section_link_summary": self._build_summary(
                page_raws=page_raws,
                section_links=section_link_dicts,
                hierarchy_links=hierarchy_link_dicts,
                sequence_links=sequence_link_dicts,
            ),
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                result=result,
            )

        return result

    def _collect_sections(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []

        if self.config.use_document_structure and document_structure_result:
            sections = document_structure_result.get("sections", []) or []

            if sections:
                return self._deduplicate_sections(sections)

            document_tree = document_structure_result.get("document_tree", {}) or {}
            tree_sections = document_tree.get("sections", []) or []

            if tree_sections:
                return self._deduplicate_sections(tree_sections)

        if self.config.fallback_from_page_metadata:
            collected: List[Dict[str, Any]] = []

            for page_raw in page_raws:
                for meta_key in [
                    "section_builder",
                    "document_tree_builder",
                    "document_structure_pipeline",
                ]:
                    meta = page_raw.metadata.get(meta_key, {}) or {}

                    for item_key in [
                        "sections_on_page",
                        "sections",
                    ]:
                        page_sections = meta.get(item_key, []) or []

                        for section in page_sections:
                            collected.append(section)

            if collected:
                return self._deduplicate_sections(collected)

        return []

    def _build_section_links(
        self,
        sections: List[Dict[str, Any]],
        page_raws: List[PageRaw],
    ) -> List[SectionLink]:
        links: List[SectionLink] = []

        page_numbers_all = [
            page_raw.page_number for page_raw in page_raws
        ]

        for index, section in enumerate(sections):
            section_id = (
                section.get("section_id")
                or section.get("id")
                or make_id("section")
            )

            title = self._clean_text(
                section.get("title")
                or section.get("heading")
                or section.get("text")
                or f"Section {index + 1}"
            )[: self.config.max_title_chars]

            level = self._safe_int(section.get("level"), default=0)
            order = self._safe_int(section.get("order"), default=index)

            page_numbers = self._resolve_page_numbers(section)

            page_start = section.get("page_start")
            page_end = section.get("page_end")

            if page_numbers:
                page_start = min(page_numbers)
                page_end = max(page_numbers)

            if not page_numbers and page_start is not None and page_end is not None:
                try:
                    page_numbers = list(range(int(page_start), int(page_end) + 1))
                except Exception:
                    page_numbers = []

            if not page_numbers:
                page_number = self._safe_int(section.get("page_number"), default=0)

                if page_number > 0:
                    page_numbers = [page_number]
                    page_start = page_number
                    page_end = page_number

            if not page_numbers and section_id in ["root", "root_section", "fallback_root_section"]:
                page_numbers = page_numbers_all
                page_start = min(page_numbers_all) if page_numbers_all else None
                page_end = max(page_numbers_all) if page_numbers_all else None

            if not self.config.include_root_section:
                section_type = section.get("section_type", "")

                if section_type == "root" or level == 0:
                    continue

            child_ids = (
                section.get("child_ids")
                or section.get("children")
                or []
            )

            child_ids = [
                item.get("section_id", "") if isinstance(item, dict) else str(item)
                for item in child_ids
                if item
            ]

            confidence = self._safe_float(
                section.get("confidence"),
                default=0.75 if page_numbers else 0.45,
            )

            links.append(
                SectionLink(
                    section_link_id=make_id("section_link"),
                    section_id=section_id,
                    title=title,
                    level=level,
                    order=order,
                    page_start=page_start,
                    page_end=page_end,
                    page_numbers=page_numbers,
                    parent_id=section.get("parent_id", "") or section.get("parent_section_id", ""),
                    child_ids=child_ids,
                    link_type="section_to_pages",
                    confidence=round(max(0.0, min(confidence, 0.95)), 4),
                    source=section.get("source", "document_structure"),
                    metadata={
                        "section_type": section.get("section_type", ""),
                        "section_number": section.get("section_number", ""),
                        "heading_id": section.get("heading_id", ""),
                        "text_preview": self._clean_text(section.get("text_preview", ""))[: self.config.text_preview_chars],
                        "raw_keys": sorted(list(section.keys())),
                    },
                )
            )

        links = sorted(
            links,
            key=lambda item: (
                item.page_start if item.page_start is not None else 999999,
                item.order,
                item.level,
            ),
        )

        return links

    def _normalize_page_ranges(
        self,
        section_links: List[SectionLink],
        page_raws: List[PageRaw],
    ) -> List[SectionLink]:
        if not section_links:
            return []

        page_numbers_all = sorted([
            page_raw.page_number for page_raw in page_raws
        ])

        if not page_numbers_all:
            return section_links

        min_page = min(page_numbers_all)
        max_page = max(page_numbers_all)

        sorted_links = sorted(
            section_links,
            key=lambda item: (
                item.page_start if item.page_start is not None else 999999,
                item.order,
            ),
        )

        for index, link in enumerate(sorted_links):
            if link.page_numbers:
                link.page_numbers = self._filter_existing_pages(
                    page_numbers=link.page_numbers,
                    existing_pages=page_numbers_all,
                )
                link.page_start = min(link.page_numbers) if link.page_numbers else link.page_start
                link.page_end = max(link.page_numbers) if link.page_numbers else link.page_end
                continue

            if link.page_start is None:
                link.page_start = min_page

            if link.page_end is None:
                next_start = None

                for next_link in sorted_links[index + 1:]:
                    if next_link.page_start is not None:
                        next_start = next_link.page_start
                        break

                if next_start is not None and next_start > link.page_start:
                    link.page_end = next_start - 1
                else:
                    link.page_end = max_page

            try:
                link.page_numbers = list(range(int(link.page_start), int(link.page_end) + 1))
            except Exception:
                link.page_numbers = []

            link.page_numbers = self._filter_existing_pages(
                page_numbers=link.page_numbers,
                existing_pages=page_numbers_all,
            )

        return sorted_links

    def _build_hierarchy_links(
        self,
        section_links: List[SectionLink],
    ) -> List[SectionRelation]:
        relations: List[SectionRelation] = []

        by_id = {
            item.section_id: item
            for item in section_links
        }

        for item in section_links:
            if item.parent_id and item.parent_id in by_id:
                parent = by_id[item.parent_id]

                relations.append(
                    SectionRelation(
                        section_relation_id=make_id("section_rel"),
                        source_section_id=parent.section_id,
                        target_section_id=item.section_id,
                        relation_type="section_parent_of",
                        source_page=parent.page_start,
                        target_page=item.page_start,
                        weight=0.85,
                        confidence=0.80,
                        source="section_linker",
                        metadata={
                            "parent_title": parent.title,
                            "child_title": item.title,
                            "parent_level": parent.level,
                            "child_level": item.level,
                        },
                    )
                )

        if not relations:
            relations.extend(self._infer_hierarchy_links(section_links))

        return relations

    def _infer_hierarchy_links(
        self,
        section_links: List[SectionLink],
    ) -> List[SectionRelation]:
        relations: List[SectionRelation] = []
        stack: List[SectionLink] = []

        sorted_links = sorted(
            section_links,
            key=lambda item: (
                item.order,
                item.page_start if item.page_start is not None else 999999,
            ),
        )

        for link in sorted_links:
            if link.level <= 0:
                stack = [link]
                continue

            while stack and stack[-1].level >= link.level:
                stack.pop()

            if stack:
                parent = stack[-1]

                relations.append(
                    SectionRelation(
                        section_relation_id=make_id("section_rel"),
                        source_section_id=parent.section_id,
                        target_section_id=link.section_id,
                        relation_type="section_parent_of",
                        source_page=parent.page_start,
                        target_page=link.page_start,
                        weight=0.70,
                        confidence=0.65,
                        source="section_linker_inferred",
                        metadata={
                            "parent_title": parent.title,
                            "child_title": link.title,
                            "parent_level": parent.level,
                            "child_level": link.level,
                            "inference_method": "level_stack",
                        },
                    )
                )

            stack.append(link)

        return relations

    def _build_sequence_links(
        self,
        section_links: List[SectionLink],
    ) -> List[SectionRelation]:
        relations: List[SectionRelation] = []

        sorted_links = sorted(
            section_links,
            key=lambda item: (
                item.page_start if item.page_start is not None else 999999,
                item.order,
                item.level,
            ),
        )

        for index in range(len(sorted_links) - 1):
            current = sorted_links[index]
            next_item = sorted_links[index + 1]

            relations.append(
                SectionRelation(
                    section_relation_id=make_id("section_rel"),
                    source_section_id=current.section_id,
                    target_section_id=next_item.section_id,
                    relation_type="section_next",
                    source_page=current.page_end,
                    target_page=next_item.page_start,
                    weight=0.45,
                    confidence=0.75,
                    source="section_linker",
                    metadata={
                        "source_title": current.title,
                        "target_title": next_item.title,
                        "source_order": current.order,
                        "target_order": next_item.order,
                    },
                )
            )

        return relations

    def _detect_sections_from_pages(
        self,
        page_raws: List[PageRaw],
    ) -> List[Dict[str, Any]]:
        detected: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            heading_candidates = self._collect_heading_candidates(page_raw)

            for candidate in heading_candidates:
                confidence = self._safe_float(candidate.get("confidence"), default=0.5)

                if confidence < self.config.min_heading_confidence:
                    continue

                title = self._clean_text(
                    candidate.get("text")
                    or candidate.get("title")
                    or candidate.get("heading_text")
                    or ""
                )

                if not title:
                    continue

                level = self._safe_int(
                    candidate.get("level"),
                    default=self._infer_heading_level(title),
                )

                detected.append(
                    {
                        "section_id": candidate.get("section_id") or make_id("section"),
                        "title": title[: self.config.max_title_chars],
                        "level": level,
                        "order": len(detected),
                        "page_start": page_raw.page_number,
                        "page_end": page_raw.page_number,
                        "page_numbers": [page_raw.page_number],
                        "parent_id": "",
                        "children": [],
                        "section_type": "detected_heading_section",
                        "heading_id": candidate.get("heading_id", ""),
                        "confidence": confidence,
                        "source": "heading_detector_metadata",
                        "text_preview": title,
                    }
                )

        if not detected:
            return self._fallback_single_root_section(page_raws)

        detected = self._extend_detected_section_ranges(
            sections=detected,
            page_raws=page_raws,
        )

        detected = self._infer_parent_ids_for_detected_sections(detected)

        return detected

    def _collect_heading_candidates(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        heading_meta = page_raw.metadata.get("heading_detector", {}) or {}
        candidates.extend(heading_meta.get("heading_candidates_on_page", []) or [])
        candidates.extend(heading_meta.get("headings_on_page", []) or [])

        structure_meta = page_raw.metadata.get("document_structure_pipeline", {}) or {}
        candidates.extend(structure_meta.get("headings_on_page", []) or [])

        if candidates:
            return candidates

        reading_meta = page_raw.metadata.get("reading_order_builder", {}) or {}
        items = reading_meta.get("reading_order_items", []) or []

        for item in items:
            text = self._clean_text(item.get("text", ""))

            if self._looks_like_heading(text):
                candidates.append(
                    {
                        "heading_id": make_id("heading"),
                        "text": text,
                        "level": self._infer_heading_level(text),
                        "confidence": 0.45,
                        "source": "reading_order_heading_fallback",
                    }
                )

        return candidates

    def _extend_detected_section_ranges(
        self,
        sections: List[Dict[str, Any]],
        page_raws: List[PageRaw],
    ) -> List[Dict[str, Any]]:
        if not sections:
            return sections

        page_numbers_all = sorted([
            page_raw.page_number for page_raw in page_raws
        ])

        max_page = max(page_numbers_all) if page_numbers_all else 0

        sections = sorted(
            sections,
            key=lambda item: (
                self._safe_int(item.get("page_start"), default=999999),
                self._safe_int(item.get("order"), default=999999),
            ),
        )

        for index, section in enumerate(sections):
            start = self._safe_int(section.get("page_start"), default=0)

            end = max_page

            for next_section in sections[index + 1:]:
                next_start = self._safe_int(next_section.get("page_start"), default=0)

                if next_start > start:
                    end = next_start - 1
                    break

            section["page_end"] = max(start, end)

            if start > 0 and end >= start:
                section["page_numbers"] = self._filter_existing_pages(
                    page_numbers=list(range(start, end + 1)),
                    existing_pages=page_numbers_all,
                )
            else:
                section["page_numbers"] = []

        return sections

    def _infer_parent_ids_for_detected_sections(
        self,
        sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        stack: List[Dict[str, Any]] = []

        for section in sections:
            level = self._safe_int(section.get("level"), default=1)

            while stack and self._safe_int(stack[-1].get("level"), default=1) >= level:
                stack.pop()

            if stack:
                section["parent_id"] = stack[-1].get("section_id", "")

            stack.append(section)

        child_map: Dict[str, List[str]] = {}

        for section in sections:
            parent_id = section.get("parent_id", "")

            if not parent_id:
                continue

            child_map.setdefault(parent_id, [])
            child_map[parent_id].append(section.get("section_id", ""))

        for section in sections:
            section["children"] = child_map.get(section.get("section_id", ""), [])

        return sections

    def _fallback_single_root_section(
        self,
        page_raws: List[PageRaw],
    ) -> List[Dict[str, Any]]:
        page_numbers = [
            page_raw.page_number for page_raw in page_raws
        ]

        return [
            {
                "section_id": "fallback_root_section",
                "title": "Document",
                "level": 0,
                "order": 0,
                "page_start": min(page_numbers) if page_numbers else None,
                "page_end": max(page_numbers) if page_numbers else None,
                "page_numbers": page_numbers,
                "content_page_numbers": page_numbers,
                "parent_id": "",
                "children": [],
                "section_type": "root",
                "confidence": 0.40,
                "source": "section_linker_fallback",
            }
        ]

    def _group_section_links_by_page(
        self,
        section_links: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for link in section_links:
            for page_number in link.get("page_numbers", []) or []:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(link)

        return grouped

    def _build_page_sections(
        self,
        page_raws: List[PageRaw],
        section_links: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        by_page = self._group_section_links_by_page(section_links)
        page_sections: Dict[str, Dict[str, Any]] = {}

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)
            sections_on_page = by_page.get(page_key, [])

            active_sections = sorted(
                sections_on_page,
                key=lambda item: (
                    self._safe_int(item.get("level"), default=0),
                    self._safe_int(item.get("order"), default=0),
                ),
            )

            deepest = active_sections[-1] if active_sections else {}

            breadcrumbs = [
                {
                    "section_id": item.get("section_id", ""),
                    "title": item.get("title", ""),
                    "level": item.get("level", 0),
                }
                for item in active_sections
            ]

            page_sections[page_key] = {
                "page_number": page_raw.page_number,
                "section_count": len(active_sections),
                "active_sections": active_sections,
                "current_section": deepest,
                "breadcrumbs": breadcrumbs,
                "current_section_id": deepest.get("section_id", ""),
                "current_section_title": deepest.get("title", ""),
            }

        return page_sections

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        section_links_by_page = result.get("section_links_by_page", {})
        page_sections = result.get("page_sections", {})
        summary = result.get("section_link_summary", {})

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("section_linker", {})
            page_raw.metadata["section_linker"] = {
                "processor": "SectionLinker",
                "section_links_on_page": section_links_by_page.get(page_key, []),
                "page_section_context": page_sections.get(page_key, {}),
                "section_link_summary": summary,
            }

    def _build_summary(
        self,
        page_raws: List[PageRaw],
        section_links: List[Dict[str, Any]],
        hierarchy_links: List[Dict[str, Any]],
        sequence_links: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        page_links = self._group_section_links_by_page(section_links)

        by_level: Dict[str, int] = {}

        for link in section_links:
            level_key = str(link.get("level", 0))
            by_level[level_key] = by_level.get(level_key, 0) + 1

        return {
            "has_section_links": len(section_links) > 0,
            "page_count": len(page_raws),
            "section_link_count": len(section_links),
            "hierarchy_link_count": len(hierarchy_links),
            "sequence_link_count": len(sequence_links),
            "page_count_with_section_links": len(page_links),
            "by_level": by_level,
        }

    def _deduplicate_sections(
        self,
        sections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for index, section in enumerate(sections):
            section_id = section.get("section_id") or section.get("id")

            if not section_id:
                section = dict(section)
                section_id = make_id("section")
                section["section_id"] = section_id

            if section_id in seen:
                continue

            seen.add(section_id)

            if "order" not in section:
                section = dict(section)
                section["order"] = index

            result.append(section)

        return result

    def _resolve_page_numbers(
        self,
        section: Dict[str, Any],
    ) -> List[int]:
        candidate_keys = [
            "page_numbers",
            "content_page_numbers",
            "pages",
        ]

        for key in candidate_keys:
            value = section.get(key)

            if isinstance(value, list):
                resolved = [
                    self._safe_int(item, default=0)
                    for item in value
                    if self._safe_int(item, default=0) > 0
                ]

                if resolved:
                    return sorted(list(dict.fromkeys(resolved)))

        page_number = self._safe_int(section.get("page_number"), default=0)

        if page_number > 0:
            return [page_number]

        return []

    def _filter_existing_pages(
        self,
        page_numbers: List[int],
        existing_pages: List[int],
    ) -> List[int]:
        existing = set(existing_pages)

        return [
            page_number for page_number in page_numbers
            if page_number in existing
        ]

    def _looks_like_heading(
        self,
        text: str,
    ) -> bool:
        text = self._clean_text(text)

        if not text:
            return False

        if len(text) > self.config.max_title_chars:
            return False

        patterns = [
            r"^Chương\s+[IVXLC\d]+",
            r"^CHƯƠNG\s+[IVXLC\d]+",
            r"^Phần\s+[IVXLC\d]+",
            r"^PHẦN\s+[IVXLC\d]+",
            r"^[IVXLC]+\.\s+",
            r"^\d+(\.\d+)*[\.\)]\s+",
            r"^[A-ZĐĂÂÊÔƠƯ][A-ZĐĂÂÊÔƠƯ\s,\-:]{8,}$",
        ]

        for pattern in patterns:
            if re.search(pattern, text):
                return True

        return False

    def _infer_heading_level(
        self,
        text: str,
    ) -> int:
        text = self._clean_text(text)

        if re.match(r"^(Chương|CHƯƠNG)\s+", text):
            return 1

        if re.match(r"^(Phần|PHẦN)\s+", text):
            return 1

        if re.match(r"^[IVXLC]+\.\s+", text):
            return 1

        match = re.match(r"^(\d+(?:\.\d+)*)[\.\)]\s+", text)

        if match:
            numbering = match.group(1)

            return max(1, numbering.count(".") + 1)

        if text.isupper():
            return 1

        return 2

    def _clean_text(
        self,
        text: Any,
    ) -> str:
        if text is None:
            return ""

        text = str(text)
        text = text.replace("\u00a0", " ")
        text = text.replace("Ƣ", "Ư")
        text = text.replace("ƣ", "ư")
        text = re.sub(r"[ \t]+", " ", text)

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

    def _safe_float(
        self,
        value: Any,
        default: float = 0.0,
    ) -> float:
        try:
            if value is None:
                return default

            return float(value)
        except Exception:
            return default


def link_sections(
    page_raws: List[PageRaw],
    document_structure_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    linker = SectionLinker()
    return linker.process(
        page_raws=page_raws,
        document_structure_result=document_structure_result,
    )
