"""
entity_linker.py

Production V1 - Colab Ready

Purpose
-------
Detect and link important entities across pages.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- DocumentStructurePipeline
- SectionLinker
- ParagraphContinuationDetector
- TableContinuationDetector

Output
------
Dictionary with:
- entities
- entity_occurrences
- entity_links
- entities_by_page
- entity_links_by_page
- entity_network
- entity_link_summary

Entity types
------------
- legal_document
- decision
- decree
- circular
- law
- date
- money
- percentage
- organization
- person_like
- email
- url
- code
- table_reference
- section_reference
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class EntityLinkerConfig:
    detect_legal_entities: bool = True
    detect_dates: bool = True
    detect_money: bool = True
    detect_percentages: bool = True
    detect_organizations: bool = True
    detect_person_like: bool = False
    detect_email_url: bool = True
    detect_codes: bool = True
    detect_references: bool = True

    min_entity_text_length: int = 2
    max_entity_text_length: int = 250

    min_link_occurrences: int = 2
    min_link_pages: int = 2

    max_occurrences_per_entity: int = 500
    max_entities: int = 5000

    attach_to_pages: bool = True
    include_context_snippet: bool = True
    context_window_chars: int = 120

    include_debug: bool = True


@dataclass
class EntityOccurrence:
    entity_occurrence_id: str
    entity_id: str
    entity_type: str

    text: str
    normalized_text: str

    page_number: int
    page_index: int

    start_char: Optional[int] = None
    end_char: Optional[int] = None

    section_id: str = ""
    section_title: str = ""

    confidence: float = 0.5
    source: str = "entity_linker"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class Entity:
    entity_id: str
    entity_type: str

    text: str
    normalized_text: str

    page_numbers: List[int]
    occurrence_ids: List[str]

    occurrence_count: int = 0
    first_page: Optional[int] = None
    last_page: Optional[int] = None

    confidence: float = 0.5
    source: str = "entity_linker"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class EntityLink:
    entity_link_id: str
    entity_id: str
    entity_type: str

    text: str
    normalized_text: str

    page_numbers: List[int]
    occurrence_ids: List[str]

    link_type: str = "same_entity_cross_page"
    confidence: float = 0.6
    source: str = "entity_linker"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class EntityLinker:
    def __init__(
        self,
        config: Optional[EntityLinkerConfig] = None,
    ):
        self.config = config or EntityLinkerConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Optional[Dict[str, Any]] = None,
        section_link_result: Optional[Dict[str, Any]] = None,
        paragraph_continuation_result: Optional[Dict[str, Any]] = None,
        table_continuation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        document_structure_result = document_structure_result or {}
        section_link_result = section_link_result or {}
        paragraph_continuation_result = paragraph_continuation_result or {}
        table_continuation_result = table_continuation_result or {}

        page_section_context = self._build_page_section_context(
            page_raws=page_raws,
            document_structure_result=document_structure_result,
            section_link_result=section_link_result,
        )

        occurrences = self._detect_entity_occurrences(
            page_raws=page_raws,
            page_section_context=page_section_context,
        )

        entities = self._build_entities(occurrences)

        links = self._build_entity_links(entities)

        entity_dicts = [
            entity.to_dict() for entity in entities
        ]

        occurrence_dicts = [
            occurrence.to_dict() for occurrence in occurrences
        ]

        link_dicts = [
            link.to_dict() for link in links
        ]

        result = {
            "processor": "EntityLinker",
            "schema_version": "entity_linker_v1",
            "entities": entity_dicts,
            "entity_occurrences": occurrence_dicts,
            "entity_links": link_dicts,
            "entities_by_page": self._group_entities_by_page(entity_dicts),
            "entity_occurrences_by_page": self._group_occurrences_by_page(occurrence_dicts),
            "entity_links_by_page": self._group_links_by_page(link_dicts),
            "entities_by_type": self._group_entities_by_type(entity_dicts),
            "entity_network": self._build_entity_network(
                entities=entity_dicts,
                occurrences=occurrence_dicts,
                links=link_dicts,
            ),
            "entity_link_summary": self._build_summary(
                entities=entity_dicts,
                occurrences=occurrence_dicts,
                links=link_dicts,
                page_raws=page_raws,
            ),
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                result=result,
            )

        return result

    def _detect_entity_occurrences(
        self,
        page_raws: List[PageRaw],
        page_section_context: Dict[str, Dict[str, Any]],
    ) -> List[EntityOccurrence]:
        occurrences: List[EntityOccurrence] = []

        patterns = self._build_patterns()

        for page_raw in page_raws:
            text = self._page_text(page_raw)
            page_key = str(page_raw.page_number)
            section_context = page_section_context.get(page_key, {})

            seen_on_page = set()

            for entity_type, pattern_items in patterns.items():
                for pattern_item in pattern_items:
                    pattern = pattern_item["pattern"]
                    flags = pattern_item.get("flags", re.IGNORECASE)
                    confidence = pattern_item.get("confidence", 0.55)
                    pattern_name = pattern_item.get("name", entity_type)

                    for match in re.finditer(pattern, text, flags=flags):
                        raw_text = self._clean_text(match.group(0))
                        normalized_text = self._normalize_entity_text(
                            text=raw_text,
                            entity_type=entity_type,
                        )

                        if not self._valid_entity_text(raw_text):
                            continue

                        entity_key = f"{entity_type}|{normalized_text}|{page_raw.page_number}|{match.start()}|{match.end()}"

                        if entity_key in seen_on_page:
                            continue

                        seen_on_page.add(entity_key)

                        entity_id = self._entity_id_from_key(
                            entity_type=entity_type,
                            normalized_text=normalized_text,
                        )

                        occurrences.append(
                            EntityOccurrence(
                                entity_occurrence_id=make_id("entity_occ"),
                                entity_id=entity_id,
                                entity_type=entity_type,
                                text=raw_text,
                                normalized_text=normalized_text,
                                page_number=page_raw.page_number,
                                page_index=page_raw.page_index,
                                start_char=match.start(),
                                end_char=match.end(),
                                section_id=section_context.get("current_section_id", ""),
                                section_title=section_context.get("current_section_title", ""),
                                confidence=confidence,
                                source="entity_linker_regex",
                                metadata={
                                    "pattern_name": pattern_name,
                                    "context_snippet": self._context_snippet(
                                        text=text,
                                        start=match.start(),
                                        end=match.end(),
                                    ) if self.config.include_context_snippet else "",
                                },
                            )
                        )

                        if len(occurrences) >= self.config.max_entities:
                            return occurrences

        occurrences = self._deduplicate_occurrences(occurrences)

        return occurrences

    def _build_patterns(
        self,
    ) -> Dict[str, List[Dict[str, Any]]]:
        patterns: Dict[str, List[Dict[str, Any]]] = {}

        if self.config.detect_legal_entities:
            patterns.setdefault("legal_document", [])
            patterns["legal_document"].extend(
                [
                    {
                        "name": "generic_legal_document_number",
                        "pattern": r"\b(?:Nghị\s*quyết|Nghị\s*định|Quyết\s*định|Thông\s*tư|Luật|Chỉ\s*thị|Công\s*văn)\s+(?:số\s*)?[0-9A-ZĐa-zđ\/\-.]+",
                        "confidence": 0.80,
                    },
                    {
                        "name": "legal_number_with_suffix",
                        "pattern": r"\b[0-9]{1,5}\/(?:QĐ|NĐ|TT|CT|CV|NQ|CP|TTg|BCT|BTC|BTTTT|KTNN|UBND)[\-\/A-ZĐa-zđ0-9]*\b",
                        "confidence": 0.75,
                    },
                ]
            )

            patterns["decision"] = [
                {
                    "name": "decision",
                    "pattern": r"\b(?:Quyết\s*định|QĐ)\s+(?:số\s*)?[0-9A-ZĐa-zđ\/\-.]+",
                    "confidence": 0.82,
                }
            ]

            patterns["decree"] = [
                {
                    "name": "decree",
                    "pattern": r"\b(?:Nghị\s*định|NĐ)\s+(?:số\s*)?[0-9A-ZĐa-zđ\/\-.]+",
                    "confidence": 0.82,
                }
            ]

            patterns["circular"] = [
                {
                    "name": "circular",
                    "pattern": r"\b(?:Thông\s*tư|TT)\s+(?:số\s*)?[0-9A-ZĐa-zđ\/\-.]+",
                    "confidence": 0.82,
                }
            ]

            patterns["law"] = [
                {
                    "name": "law",
                    "pattern": r"\bLuật\s+[A-ZĐĂÂÊÔƠƯa-zđăâêôơư0-9\s,\-]{3,120}",
                    "confidence": 0.70,
                }
            ]

        if self.config.detect_dates:
            patterns["date"] = [
                {
                    "name": "date_slash_or_dash",
                    "pattern": r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
                    "confidence": 0.75,
                },
                {
                    "name": "vietnamese_date",
                    "pattern": r"\bngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}\b",
                    "confidence": 0.85,
                },
                {
                    "name": "year_range",
                    "pattern": r"\b20\d{2}\s*[-–]\s*20\d{2}\b",
                    "confidence": 0.65,
                },
            ]

        if self.config.detect_money:
            patterns["money"] = [
                {
                    "name": "money_vnd",
                    "pattern": r"\b\d[\d\.,]*\s*(?:VNĐ|VND|vnđ|vnd|đồng|đ)\b",
                    "confidence": 0.80,
                },
                {
                    "name": "money_large_unit",
                    "pattern": r"\b\d[\d\.,]*\s*(?:tỷ|triệu|nghìn|ngàn)\s*(?:đồng|VNĐ|VND)?\b",
                    "confidence": 0.75,
                },
                {
                    "name": "money_usd",
                    "pattern": r"(?:\$|USD\s*)\s*\d[\d\.,]*|\b\d[\d\.,]*\s*USD\b",
                    "confidence": 0.75,
                },
            ]

        if self.config.detect_percentages:
            patterns["percentage"] = [
                {
                    "name": "percentage",
                    "pattern": r"\b\d+(?:[,.]\d+)?\s*%",
                    "confidence": 0.80,
                },
                {
                    "name": "percentage_vietnamese",
                    "pattern": r"\b\d+(?:[,.]\d+)?\s*(?:phần\s*trăm)\b",
                    "confidence": 0.75,
                },
            ]

        if self.config.detect_organizations:
            patterns["organization"] = [
                {
                    "name": "ministry_department",
                    "pattern": r"\b(?:Bộ|Cục|Vụ|Sở|Ban|Ủy\s*ban|UBND|HĐND|Kiểm\s*toán\s*nhà\s*nước|KTNN)\s+[A-ZĐĂÂÊÔƠƯa-zđăâêôơư0-9&,\-\s]{2,120}",
                    "confidence": 0.70,
                },
                {
                    "name": "company_org",
                    "pattern": r"\b(?:Công\s*ty|Tổng\s*công\s*ty|Tập\s*đoàn|Doanh\s*nghiệp|Trung\s*tâm|Bệnh\s*viện|Viện)\s+[A-ZĐĂÂÊÔƠƯa-zđăâêôơư0-9&,\-\s]{2,120}",
                    "confidence": 0.68,
                },
                {
                    "name": "uppercase_org_abbreviation",
                    "pattern": r"\b[A-ZĐ]{2,10}(?:-[A-ZĐ0-9]{2,10})?\b",
                    "confidence": 0.45,
                },
            ]

        if self.config.detect_person_like:
            patterns["person_like"] = [
                {
                    "name": "vietnamese_person_like",
                    "pattern": r"\b(?:Ông|Bà|Đồng chí)\s+[A-ZĐ][a-zđăâêôơưàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũỳýỵỷỹ]+\s+(?:[A-ZĐ][a-zđăâêôơưàáạảãèéẹẻẽìíịỉĩòóọỏõùúụủũỳýỵỷỹ]+\s*){1,4}",
                    "confidence": 0.65,
                }
            ]

        if self.config.detect_email_url:
            patterns["email"] = [
                {
                    "name": "email",
                    "pattern": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
                    "confidence": 0.90,
                    "flags": 0,
                }
            ]

            patterns["url"] = [
                {
                    "name": "url",
                    "pattern": r"\b(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+",
                    "confidence": 0.88,
                    "flags": 0,
                }
            ]

        if self.config.detect_codes:
            patterns["code"] = [
                {
                    "name": "slash_code",
                    "pattern": r"\b[A-ZĐ0-9]{2,}[\/\-][A-ZĐ0-9\/\-]{2,}\b",
                    "confidence": 0.55,
                    "flags": 0,
                },
                {
                    "name": "project_code",
                    "pattern": r"\b[A-Z]{2,8}[-_]\d{2,8}(?:[-_][A-Z0-9]{2,8})?\b",
                    "confidence": 0.55,
                    "flags": 0,
                },
            ]

        if self.config.detect_references:
            patterns["table_reference"] = [
                {
                    "name": "table_reference",
                    "pattern": r"\b(?:Bảng|Bang|Table)\s+\d+(?:\.\d+)*\b",
                    "confidence": 0.75,
                }
            ]

            patterns["section_reference"] = [
                {
                    "name": "section_reference",
                    "pattern": r"\b(?:Mục|Phần|Chương|Điều|Khoản)\s+[IVXLC\d]+(?:\.\d+)*\b",
                    "confidence": 0.70,
                }
            ]

        return patterns

    def _build_entities(
        self,
        occurrences: List[EntityOccurrence],
    ) -> List[Entity]:
        grouped: Dict[str, List[EntityOccurrence]] = {}

        for occurrence in occurrences:
            grouped.setdefault(occurrence.entity_id, [])
            grouped[occurrence.entity_id].append(occurrence)

        entities: List[Entity] = []

        for entity_id, items in grouped.items():
            items = sorted(
                items,
                key=lambda item: (
                    item.page_number,
                    item.start_char if item.start_char is not None else 999999,
                ),
            )

            if not items:
                continue

            page_numbers = sorted(
                list(
                    dict.fromkeys(
                        [
                            item.page_number
                            for item in items
                        ]
                    )
                )
            )

            representative = self._select_representative_occurrence(items)

            occurrence_ids = [
                item.entity_occurrence_id for item in items[: self.config.max_occurrences_per_entity]
            ]

            confidence = self._score_entity(
                entity_type=representative.entity_type,
                occurrence_count=len(items),
                page_count=len(page_numbers),
                base_confidence=representative.confidence,
            )

            entities.append(
                Entity(
                    entity_id=entity_id,
                    entity_type=representative.entity_type,
                    text=representative.text,
                    normalized_text=representative.normalized_text,
                    page_numbers=page_numbers,
                    occurrence_ids=occurrence_ids,
                    occurrence_count=len(items),
                    first_page=min(page_numbers) if page_numbers else None,
                    last_page=max(page_numbers) if page_numbers else None,
                    confidence=confidence,
                    source="entity_linker",
                    metadata={
                        "section_ids": sorted(
                            list(
                                dict.fromkeys(
                                    [
                                        item.section_id
                                        for item in items
                                        if item.section_id
                                    ]
                                )
                            )
                        ),
                        "section_titles": sorted(
                            list(
                                dict.fromkeys(
                                    [
                                        item.section_title
                                        for item in items
                                        if item.section_title
                                    ]
                                )
                            )
                        )[:20],
                        "representative_occurrence_id": representative.entity_occurrence_id,
                    },
                )
            )

        entities = sorted(
            entities,
            key=lambda item: (
                item.first_page if item.first_page is not None else 999999,
                item.entity_type,
                item.normalized_text,
            ),
        )

        return entities

    def _build_entity_links(
        self,
        entities: List[Entity],
    ) -> List[EntityLink]:
        links: List[EntityLink] = []

        for entity in entities:
            page_count = len(entity.page_numbers)
            occurrence_count = entity.occurrence_count

            if occurrence_count < self.config.min_link_occurrences:
                continue

            if page_count < self.config.min_link_pages:
                continue

            confidence = self._score_entity_link(entity)

            links.append(
                EntityLink(
                    entity_link_id=make_id("entity_link"),
                    entity_id=entity.entity_id,
                    entity_type=entity.entity_type,
                    text=entity.text,
                    normalized_text=entity.normalized_text,
                    page_numbers=entity.page_numbers,
                    occurrence_ids=entity.occurrence_ids,
                    link_type="same_entity_cross_page",
                    confidence=confidence,
                    source="entity_linker",
                    metadata={
                        "occurrence_count": occurrence_count,
                        "page_count": page_count,
                        "first_page": entity.first_page,
                        "last_page": entity.last_page,
                        "entity_confidence": entity.confidence,
                    },
                )
            )

        links = sorted(
            links,
            key=lambda item: (
                item.page_numbers[0] if item.page_numbers else 999999,
                item.entity_type,
                item.normalized_text,
            ),
        )

        return links

    def _build_page_section_context(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Dict[str, Any],
        section_link_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        page_context: Dict[str, Dict[str, Any]] = {}

        for page_raw in page_raws:
            page_context[str(page_raw.page_number)] = {
                "current_section_id": "",
                "current_section_title": "",
                "active_sections": [],
            }

        page_sections = section_link_result.get("page_sections", {}) or {}

        if page_sections:
            for page_key, ctx in page_sections.items():
                current = ctx.get("current_section", {}) or {}

                page_context[str(page_key)] = {
                    "current_section_id": current.get("section_id", ctx.get("current_section_id", "")),
                    "current_section_title": current.get("title", ctx.get("current_section_title", "")),
                    "active_sections": ctx.get("active_sections", []) or [],
                }

            return page_context

        section_links_by_page = section_link_result.get("section_links_by_page", {}) or {}

        if section_links_by_page:
            for page_key, links in section_links_by_page.items():
                links = links or []

                sorted_links = sorted(
                    links,
                    key=lambda item: (
                        self._safe_int(item.get("level"), default=0),
                        self._safe_int(item.get("order"), default=0),
                    ),
                )

                current = sorted_links[-1] if sorted_links else {}

                page_context[str(page_key)] = {
                    "current_section_id": current.get("section_id", ""),
                    "current_section_title": current.get("title", ""),
                    "active_sections": sorted_links,
                }

            return page_context

        sections = document_structure_result.get("sections", []) or []

        for page_raw in page_raws:
            active = []

            for section in sections:
                page_numbers = section.get("page_numbers", []) or section.get("content_page_numbers", []) or []

                if not page_numbers:
                    page_start = section.get("page_start")
                    page_end = section.get("page_end")

                    if page_start is not None and page_end is not None:
                        try:
                            page_numbers = list(range(int(page_start), int(page_end) + 1))
                        except Exception:
                            page_numbers = []

                if page_raw.page_number in page_numbers:
                    active.append(section)

            active = sorted(
                active,
                key=lambda item: (
                    self._safe_int(item.get("level"), default=0),
                    self._safe_int(item.get("order"), default=0),
                ),
            )

            current = active[-1] if active else {}

            page_context[str(page_raw.page_number)] = {
                "current_section_id": current.get("section_id", ""),
                "current_section_title": current.get("title", ""),
                "active_sections": active,
            }

        return page_context

    def _build_entity_network(
        self,
        entities: List[Dict[str, Any]],
        occurrences: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        nodes = []
        edges = []

        for entity in entities:
            nodes.append(
                {
                    "node_id": f"entity_{entity.get('entity_id', '')}",
                    "node_type": "entity",
                    "label": entity.get("text", ""),
                    "entity_type": entity.get("entity_type", ""),
                    "page_numbers": entity.get("page_numbers", []),
                    "occurrence_count": entity.get("occurrence_count", 0),
                    "confidence": entity.get("confidence", 0.5),
                }
            )

            for page_number in entity.get("page_numbers", []) or []:
                edges.append(
                    {
                        "edge_id": make_id("entity_edge"),
                        "source_id": f"entity_{entity.get('entity_id', '')}",
                        "target_id": f"page_{page_number}",
                        "edge_type": "entity_mentioned_on_page",
                        "page_number": page_number,
                        "weight": 0.65,
                        "confidence": entity.get("confidence", 0.5),
                    }
                )

        for link in links:
            page_numbers = link.get("page_numbers", []) or []

            for index in range(len(page_numbers) - 1):
                edges.append(
                    {
                        "edge_id": make_id("entity_edge"),
                        "source_id": f"page_{page_numbers[index]}",
                        "target_id": f"page_{page_numbers[index + 1]}",
                        "edge_type": "same_entity_cross_page",
                        "entity_id": link.get("entity_id", ""),
                        "entity_type": link.get("entity_type", ""),
                        "weight": 0.60,
                        "confidence": link.get("confidence", 0.6),
                    }
                )

        return {
            "nodes": nodes,
            "edges": self._deduplicate_edge_dicts(edges),
            "summary": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "linked_entity_count": len(links),
            },
        }

    def _group_entities_by_page(
        self,
        entities: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for entity in entities:
            for page_number in entity.get("page_numbers", []) or []:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(entity)

        return grouped

    def _group_occurrences_by_page(
        self,
        occurrences: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for occurrence in occurrences:
            page_number = occurrence.get("page_number")

            if page_number is None:
                continue

            page_key = str(page_number)
            grouped.setdefault(page_key, [])
            grouped[page_key].append(occurrence)

        return grouped

    def _group_links_by_page(
        self,
        links: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for link in links:
            for page_number in link.get("page_numbers", []) or []:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(link)

        return grouped

    def _group_entities_by_type(
        self,
        entities: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for entity in entities:
            entity_type = entity.get("entity_type", "unknown")
            grouped.setdefault(entity_type, [])
            grouped[entity_type].append(entity)

        return grouped

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        entities_by_page = result.get("entities_by_page", {})
        occurrences_by_page = result.get("entity_occurrences_by_page", {})
        links_by_page = result.get("entity_links_by_page", {})
        summary = result.get("entity_link_summary", {})

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("entity_linker", {})
            page_raw.metadata["entity_linker"] = {
                "processor": "EntityLinker",
                "entities_on_page": entities_by_page.get(page_key, []),
                "entity_occurrences_on_page": occurrences_by_page.get(page_key, []),
                "entity_links_on_page": links_by_page.get(page_key, []),
                "entity_count_on_page": len(entities_by_page.get(page_key, [])),
                "entity_occurrence_count_on_page": len(occurrences_by_page.get(page_key, [])),
                "entity_link_count_on_page": len(links_by_page.get(page_key, [])),
                "entity_link_summary": summary,
            }

    def _build_summary(
        self,
        entities: List[Dict[str, Any]],
        occurrences: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        linked_by_type: Dict[str, int] = {}
        by_page: Dict[str, int] = {}

        for entity in entities:
            entity_type = entity.get("entity_type", "unknown")
            by_type[entity_type] = by_type.get(entity_type, 0) + 1

        for link in links:
            entity_type = link.get("entity_type", "unknown")
            linked_by_type[entity_type] = linked_by_type.get(entity_type, 0) + 1

        for occurrence in occurrences:
            page_key = str(occurrence.get("page_number", "unknown"))
            by_page[page_key] = by_page.get(page_key, 0) + 1

        cross_page_entities = [
            entity for entity in entities
            if len(entity.get("page_numbers", []) or []) >= 2
        ]

        return {
            "has_entities": len(entities) > 0,
            "has_entity_links": len(links) > 0,
            "page_count": len(page_raws),
            "entity_count": len(entities),
            "entity_occurrence_count": len(occurrences),
            "entity_link_count": len(links),
            "cross_page_entity_count": len(cross_page_entities),
            "by_entity_type": by_type,
            "linked_by_entity_type": linked_by_type,
            "occurrence_by_page": by_page,
        }

    def _select_representative_occurrence(
        self,
        occurrences: List[EntityOccurrence],
    ) -> EntityOccurrence:
        sorted_items = sorted(
            occurrences,
            key=lambda item: (
                -len(item.text),
                item.page_number,
                item.start_char if item.start_char is not None else 999999,
            ),
        )

        return sorted_items[0]

    def _score_entity(
        self,
        entity_type: str,
        occurrence_count: int,
        page_count: int,
        base_confidence: float,
    ) -> float:
        score = base_confidence

        if occurrence_count >= 2:
            score += 0.05

        if occurrence_count >= 5:
            score += 0.05

        if page_count >= 2:
            score += 0.08

        if entity_type in [
            "legal_document",
            "decision",
            "decree",
            "circular",
            "law",
            "email",
            "url",
        ]:
            score += 0.05

        return round(max(0.0, min(score, 0.95)), 4)

    def _score_entity_link(
        self,
        entity: Entity,
    ) -> float:
        score = 0.50

        page_count = len(entity.page_numbers)
        occurrence_count = entity.occurrence_count

        score += min(0.20, page_count * 0.04)
        score += min(0.15, occurrence_count * 0.02)
        score += min(0.10, entity.confidence * 0.10)

        if entity.entity_type in [
            "legal_document",
            "decision",
            "decree",
            "circular",
            "law",
            "organization",
        ]:
            score += 0.05

        return round(max(0.0, min(score, 0.95)), 4)

    def _entity_id_from_key(
        self,
        entity_type: str,
        normalized_text: str,
    ) -> str:
        import hashlib

        key = f"{entity_type}|{normalized_text}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]

        return f"entity_{entity_type}_{digest}"

    def _valid_entity_text(
        self,
        text: str,
    ) -> bool:
        text = self._clean_text(text)

        if len(text) < self.config.min_entity_text_length:
            return False

        if len(text) > self.config.max_entity_text_length:
            return False

        if text in [".", ",", ";", ":", "-", "_"]:
            return False

        return True

    def _normalize_entity_text(
        self,
        text: str,
        entity_type: str,
    ) -> str:
        text = self._clean_text(text)

        if entity_type in ["email", "url", "code"]:
            return text.lower()

        if entity_type in [
            "legal_document",
            "decision",
            "decree",
            "circular",
            "law",
            "organization",
            "table_reference",
            "section_reference",
        ]:
            text = self._normalize_spaces_and_punctuation(text)
            return self._normalize_vietnamese_for_match(text)

        if entity_type == "date":
            return self._normalize_date_text(text)

        if entity_type in ["money", "percentage"]:
            return self._normalize_number_text(text)

        return self._normalize_vietnamese_for_match(text)

    def _normalize_spaces_and_punctuation(
        self,
        text: str,
    ) -> str:
        text = self._clean_text(text)
        text = re.sub(r"\s*\/\s*", "/", text)
        text = re.sub(r"\s*-\s*", "-", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _normalize_date_text(
        self,
        text: str,
    ) -> str:
        text = self._clean_text(text).lower()
        text = self._normalize_vietnamese_for_match(text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _normalize_number_text(
        self,
        text: str,
    ) -> str:
        text = self._clean_text(text).lower()
        text = text.replace("vnđ", "vnd")
        text = text.replace("đồng", "vnd")
        text = text.replace("đ", "vnd")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _normalize_vietnamese_for_match(
        self,
        text: str,
    ) -> str:
        text = self._clean_text(text).lower()

        replacements = {
            "à": "a", "á": "a", "ạ": "a", "ả": "a", "ã": "a",
            "â": "a", "ầ": "a", "ấ": "a", "ậ": "a", "ẩ": "a", "ẫ": "a",
            "ă": "a", "ằ": "a", "ắ": "a", "ặ": "a", "ẳ": "a", "ẵ": "a",
            "è": "e", "é": "e", "ẹ": "e", "ẻ": "e", "ẽ": "e",
            "ê": "e", "ề": "e", "ế": "e", "ệ": "e", "ể": "e", "ễ": "e",
            "ì": "i", "í": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
            "ò": "o", "ó": "o", "ọ": "o", "ỏ": "o", "õ": "o",
            "ô": "o", "ồ": "o", "ố": "o", "ộ": "o", "ổ": "o", "ỗ": "o",
            "ơ": "o", "ờ": "o", "ớ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
            "ù": "u", "ú": "u", "ụ": "u", "ủ": "u", "ũ": "u",
            "ư": "u", "ừ": "u", "ứ": "u", "ự": "u", "ử": "u", "ữ": "u",
            "ỳ": "y", "ý": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y",
            "đ": "d",
        }

        for src, dst in replacements.items():
            text = text.replace(src, dst)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _context_snippet(
        self,
        text: str,
        start: int,
        end: int,
    ) -> str:
        left = max(0, start - self.config.context_window_chars)
        right = min(len(text), end + self.config.context_window_chars)

        snippet = text[left:right]
        snippet = self._clean_text(snippet)

        return snippet

    def _page_text(
        self,
        page_raw: PageRaw,
    ) -> str:
        reading_meta = page_raw.metadata.get("reading_order_builder", {})
        text = reading_meta.get("reading_order_text", "")

        if text:
            return self._clean_text_block(text)

        return self._clean_text_block(page_raw.normalized_text or page_raw.raw_text or "")

    def _deduplicate_occurrences(
        self,
        occurrences: List[EntityOccurrence],
    ) -> List[EntityOccurrence]:
        seen = set()
        result: List[EntityOccurrence] = []

        for occurrence in occurrences:
            key = (
                occurrence.entity_type,
                occurrence.normalized_text,
                occurrence.page_number,
                occurrence.start_char,
                occurrence.end_char,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(occurrence)

        result = sorted(
            result,
            key=lambda item: (
                item.page_number,
                item.start_char if item.start_char is not None else 999999,
                item.entity_type,
            ),
        )

        return result

    def _deduplicate_edge_dicts(
        self,
        edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for edge in edges:
            key = (
                edge.get("source_id", ""),
                edge.get("target_id", ""),
                edge.get("edge_type", ""),
                edge.get("entity_id", ""),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(edge)

        return result

    def _clean_text_block(
        self,
        text: Any,
    ) -> str:
        if text is None:
            return ""

        text = str(text)
        text = text.replace("\u00a0", " ")
        text = text.replace("Ƣ", "Ư")
        text = text.replace("ƣ", "ư")
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

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


def link_entities(
    page_raws: List[PageRaw],
    document_structure_result: Optional[Dict[str, Any]] = None,
    section_link_result: Optional[Dict[str, Any]] = None,
    paragraph_continuation_result: Optional[Dict[str, Any]] = None,
    table_continuation_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    linker = EntityLinker()
    return linker.process(
        page_raws=page_raws,
        document_structure_result=document_structure_result,
        section_link_result=section_link_result,
        paragraph_continuation_result=paragraph_continuation_result,
        table_continuation_result=table_continuation_result,
    )
