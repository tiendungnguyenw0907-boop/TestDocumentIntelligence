"""
reference_linker.py

Production V1 - Colab Ready

Purpose
-------
Detect and link internal / external references across document pages.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- DocumentStructurePipeline
- SectionLinker
- EntityLinker

Output
------
Dictionary with:
- reference_links
- reference_links_by_page
- references_by_type
- reference_targets
- reference_network
- reference_link_summary

Reference types
---------------
- page_reference
- section_reference
- article_reference
- clause_reference
- table_reference
- figure_reference
- legal_document_reference
- entity_reference
- url_reference
- email_reference
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class ReferenceLinkerConfig:
    detect_page_references: bool = True
    detect_section_references: bool = True
    detect_article_clause_references: bool = True
    detect_table_references: bool = True
    detect_figure_references: bool = True
    detect_legal_document_references: bool = True
    detect_entity_references: bool = True
    detect_email_url_references: bool = True

    resolve_page_targets: bool = True
    resolve_section_targets: bool = True
    resolve_table_targets: bool = True
    resolve_figure_targets: bool = True
    resolve_entity_targets: bool = True

    attach_to_pages: bool = True
    include_context_snippet: bool = True
    context_window_chars: int = 120

    min_reference_text_length: int = 2
    max_reference_text_length: int = 250

    min_confidence: float = 0.35
    include_debug: bool = True


@dataclass
class ReferenceTarget:
    reference_target_id: str
    target_type: str
    target_key: str
    label: str

    page_number: Optional[int] = None
    page_numbers: Optional[List[int]] = None

    source_id: str = ""
    confidence: float = 0.6
    source: str = "reference_linker"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["page_numbers"] is None:
            data["page_numbers"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class ReferenceLink:
    reference_link_id: str
    reference_type: str
    text: str
    normalized_text: str

    from_page: int
    from_page_index: int

    target_type: str = ""
    target_key: str = ""
    target_id: str = ""
    target_page: Optional[int] = None
    target_pages: Optional[List[int]] = None

    start_char: Optional[int] = None
    end_char: Optional[int] = None

    section_id: str = ""
    section_title: str = ""

    link_type: str = "reference_link"
    confidence: float = 0.5
    source: str = "reference_linker"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["target_pages"] is None:
            data["target_pages"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class ReferenceLinker:
    def __init__(
        self,
        config: Optional[ReferenceLinkerConfig] = None,
    ):
        self.config = config or ReferenceLinkerConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Optional[Dict[str, Any]] = None,
        section_link_result: Optional[Dict[str, Any]] = None,
        entity_link_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        document_structure_result = document_structure_result or {}
        section_link_result = section_link_result or {}
        entity_link_result = entity_link_result or {}

        page_raws = sorted(
            page_raws,
            key=lambda page: page.page_number,
        )

        page_section_context = self._build_page_section_context(
            page_raws=page_raws,
            document_structure_result=document_structure_result,
            section_link_result=section_link_result,
        )

        reference_targets = self._build_reference_targets(
            page_raws=page_raws,
            document_structure_result=document_structure_result,
            section_link_result=section_link_result,
            entity_link_result=entity_link_result,
        )

        reference_links = self._detect_reference_links(
            page_raws=page_raws,
            page_section_context=page_section_context,
            reference_targets=reference_targets,
            entity_link_result=entity_link_result,
        )

        reference_links = self._deduplicate_reference_links(reference_links)

        reference_target_dicts = [
            target.to_dict() for target in reference_targets
        ]

        reference_link_dicts = [
            link.to_dict() for link in reference_links
        ]

        result = {
            "processor": "ReferenceLinker",
            "schema_version": "reference_linker_v1",
            "reference_links": reference_link_dicts,
            "reference_targets": reference_target_dicts,
            "reference_links_by_page": self._group_links_by_page(reference_link_dicts),
            "references_by_type": self._group_links_by_type(reference_link_dicts),
            "reference_targets_by_type": self._group_targets_by_type(reference_target_dicts),
            "reference_network": self._build_reference_network(
                reference_links=reference_link_dicts,
                reference_targets=reference_target_dicts,
            ),
            "reference_link_summary": self._build_summary(
                page_raws=page_raws,
                reference_links=reference_link_dicts,
                reference_targets=reference_target_dicts,
            ),
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                result=result,
            )

        return result

    def _detect_reference_links(
        self,
        page_raws: List[PageRaw],
        page_section_context: Dict[str, Dict[str, Any]],
        reference_targets: List[ReferenceTarget],
        entity_link_result: Dict[str, Any],
    ) -> List[ReferenceLink]:
        patterns = self._build_patterns()
        target_index = self._build_target_index(reference_targets)
        links: List[ReferenceLink] = []

        for page_raw in page_raws:
            text = self._page_text(page_raw)
            page_key = str(page_raw.page_number)
            section_context = page_section_context.get(page_key, {})
            seen_on_page = set()

            for reference_type, pattern_items in patterns.items():
                for pattern_item in pattern_items:
                    pattern = pattern_item["pattern"]
                    flags = pattern_item.get("flags", re.IGNORECASE)
                    base_confidence = pattern_item.get("confidence", 0.55)
                    pattern_name = pattern_item.get("name", reference_type)

                    for match in re.finditer(pattern, text, flags=flags):
                        raw_text = self._clean_text(match.group(0))

                        if not self._valid_reference_text(raw_text):
                            continue

                        normalized_text = self._normalize_reference_text(
                            text=raw_text,
                            reference_type=reference_type,
                        )

                        unique_key = (
                            reference_type,
                            normalized_text,
                            page_raw.page_number,
                            match.start(),
                            match.end(),
                        )

                        if unique_key in seen_on_page:
                            continue

                        seen_on_page.add(unique_key)

                        target = self._resolve_target(
                            reference_type=reference_type,
                            reference_text=raw_text,
                            normalized_text=normalized_text,
                            from_page=page_raw.page_number,
                            target_index=target_index,
                            entity_link_result=entity_link_result,
                        )

                        confidence = self._score_reference_link(
                            reference_type=reference_type,
                            base_confidence=base_confidence,
                            target=target,
                            from_page=page_raw.page_number,
                        )

                        if confidence < self.config.min_confidence:
                            continue

                        target_pages = target.get("target_pages", []) or []
                        target_page = target.get("target_page")

                        if target_page and target_page not in target_pages:
                            target_pages = [target_page] + target_pages

                        target_pages = sorted(
                            list(
                                dict.fromkeys(
                                    [
                                        self._safe_int(page, default=0)
                                        for page in target_pages
                                        if self._safe_int(page, default=0) > 0
                                    ]
                                )
                            )
                        )

                        links.append(
                            ReferenceLink(
                                reference_link_id=make_id("ref_link"),
                                reference_type=reference_type,
                                text=raw_text,
                                normalized_text=normalized_text,
                                from_page=page_raw.page_number,
                                from_page_index=page_raw.page_index,
                                target_type=target.get("target_type", ""),
                                target_key=target.get("target_key", ""),
                                target_id=target.get("target_id", ""),
                                target_page=target_page,
                                target_pages=target_pages,
                                start_char=match.start(),
                                end_char=match.end(),
                                section_id=section_context.get("current_section_id", ""),
                                section_title=section_context.get("current_section_title", ""),
                                link_type=self._infer_link_type(
                                    reference_type=reference_type,
                                    target=target,
                                ),
                                confidence=confidence,
                                source="reference_linker_regex",
                                metadata={
                                    "pattern_name": pattern_name,
                                    "context_snippet": self._context_snippet(
                                        text=text,
                                        start=match.start(),
                                        end=match.end(),
                                    ) if self.config.include_context_snippet else "",
                                    "target_resolution": target,
                                },
                            )
                        )

        links = sorted(
            links,
            key=lambda item: (
                item.from_page,
                item.start_char if item.start_char is not None else 999999,
                item.reference_type,
            ),
        )

        return links

    def _build_patterns(
        self,
    ) -> Dict[str, List[Dict[str, Any]]]:
        patterns: Dict[str, List[Dict[str, Any]]] = {}

        if self.config.detect_page_references:
            patterns["page_reference"] = [
                {
                    "name": "vietnamese_page_ref",
                    "pattern": r"\b(?:trang|Trang)\s+\d{1,5}\b",
                    "confidence": 0.80,
                },
                {
                    "name": "english_page_ref",
                    "pattern": r"\b(?:page|Page|p\.)\s*\d{1,5}\b",
                    "confidence": 0.75,
                },
            ]

        if self.config.detect_section_references:
            patterns["section_reference"] = [
                {
                    "name": "chapter_ref",
                    "pattern": r"\b(?:Chương|chương|CHƯƠNG)\s+[IVXLC\d]+(?:\.\d+)*\b",
                    "confidence": 0.78,
                },
                {
                    "name": "part_ref",
                    "pattern": r"\b(?:Phần|phần|PHẦN)\s+[IVXLC\d]+(?:\.\d+)*\b",
                    "confidence": 0.76,
                },
                {
                    "name": "section_ref",
                    "pattern": r"\b(?:Mục|mục|MỤC)\s+[IVXLC\d]+(?:\.\d+)*\b",
                    "confidence": 0.75,
                },
                {
                    "name": "numeric_section_ref",
                    "pattern": r"\b(?:mục|phần|chương|section)\s+\d+(?:\.\d+)*\b",
                    "confidence": 0.70,
                },
            ]

        if self.config.detect_article_clause_references:
            patterns["article_reference"] = [
                {
                    "name": "article_ref",
                    "pattern": r"\b(?:Điều|điều|Article)\s+\d+[A-Za-z]?\b",
                    "confidence": 0.78,
                }
            ]

            patterns["clause_reference"] = [
                {
                    "name": "clause_ref",
                    "pattern": r"\b(?:Khoản|khoản|Clause)\s+\d+[A-Za-z]?\b",
                    "confidence": 0.74,
                },
                {
                    "name": "point_ref",
                    "pattern": r"\b(?:điểm|Điểm)\s+[a-z]\b",
                    "confidence": 0.65,
                },
            ]

        if self.config.detect_table_references:
            patterns["table_reference"] = [
                {
                    "name": "vietnamese_table_ref",
                    "pattern": r"\b(?:Bảng|bảng|Bang|bang)\s+\d+(?:\.\d+)*\b",
                    "confidence": 0.82,
                },
                {
                    "name": "english_table_ref",
                    "pattern": r"\b(?:Table|table)\s+\d+(?:\.\d+)*\b",
                    "confidence": 0.78,
                },
            ]

        if self.config.detect_figure_references:
            patterns["figure_reference"] = [
                {
                    "name": "vietnamese_figure_ref",
                    "pattern": r"\b(?:Hình|hình|Hinh|hinh)\s+\d+(?:\.\d+)*\b",
                    "confidence": 0.80,
                },
                {
                    "name": "diagram_ref",
                    "pattern": r"\b(?:Sơ\s*đồ|sơ\s*đồ|So\s*do|so\s*do)\s+\d+(?:\.\d+)*\b",
                    "confidence": 0.78,
                },
                {
                    "name": "english_figure_ref",
                    "pattern": r"\b(?:Figure|figure|Fig\.)\s+\d+(?:\.\d+)*\b",
                    "confidence": 0.76,
                },
            ]

        if self.config.detect_legal_document_references:
            patterns["legal_document_reference"] = [
                {
                    "name": "legal_doc_ref",
                    "pattern": r"\b(?:Nghị\s*quyết|Nghị\s*định|Quyết\s*định|Thông\s*tư|Luật|Chỉ\s*thị|Công\s*văn)\s+(?:số\s*)?[0-9A-ZĐa-zđ\/\-.]+",
                    "confidence": 0.82,
                },
                {
                    "name": "legal_number_ref",
                    "pattern": r"\b[0-9]{1,5}\/(?:QĐ|NĐ|TT|CT|CV|NQ|CP|TTg|BCT|BTC|BTTTT|KTNN|UBND)[\-\/A-ZĐa-zđ0-9]*\b",
                    "confidence": 0.76,
                    "flags": 0,
                },
            ]

        if self.config.detect_email_url_references:
            patterns["email_reference"] = [
                {
                    "name": "email",
                    "pattern": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
                    "confidence": 0.90,
                    "flags": 0,
                }
            ]

            patterns["url_reference"] = [
                {
                    "name": "url",
                    "pattern": r"\b(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+",
                    "confidence": 0.88,
                    "flags": 0,
                }
            ]

        return patterns

    def _build_reference_targets(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Dict[str, Any],
        section_link_result: Dict[str, Any],
        entity_link_result: Dict[str, Any],
    ) -> List[ReferenceTarget]:
        targets: List[ReferenceTarget] = []

        targets.extend(
            self._build_page_targets(page_raws)
        )

        if self.config.resolve_section_targets:
            targets.extend(
                self._build_section_targets(
                    document_structure_result=document_structure_result,
                    section_link_result=section_link_result,
                )
            )

        if self.config.resolve_table_targets:
            targets.extend(
                self._build_table_targets(page_raws)
            )

        if self.config.resolve_figure_targets:
            targets.extend(
                self._build_figure_targets(page_raws)
            )

        if self.config.resolve_entity_targets:
            targets.extend(
                self._build_entity_targets(entity_link_result)
            )

        return self._deduplicate_targets(targets)

    def _build_page_targets(
        self,
        page_raws: List[PageRaw],
    ) -> List[ReferenceTarget]:
        targets: List[ReferenceTarget] = []

        for page_raw in page_raws:
            page_number = page_raw.page_number

            targets.append(
                ReferenceTarget(
                    reference_target_id=make_id("ref_target"),
                    target_type="page",
                    target_key=str(page_number),
                    label=f"Trang {page_number}",
                    page_number=page_number,
                    page_numbers=[page_number],
                    source_id=f"page_{page_number}",
                    confidence=1.0,
                    source="page_raws",
                    metadata={
                        "page_index": page_raw.page_index,
                        "width": page_raw.width,
                        "height": page_raw.height,
                    },
                )
            )

        return targets

    def _build_section_targets(
        self,
        document_structure_result: Dict[str, Any],
        section_link_result: Dict[str, Any],
    ) -> List[ReferenceTarget]:
        targets: List[ReferenceTarget] = []

        sections = document_structure_result.get("sections", []) or []

        if not sections:
            section_links = section_link_result.get("section_links", []) or []

            for link in section_links:
                sections.append(link)

        for section in sections:
            section_id = section.get("section_id", "") or section.get("id", "")

            if not section_id:
                continue

            title = self._clean_text(
                section.get("title")
                or section.get("heading")
                or ""
            )

            section_number = self._extract_section_number(
                title=title,
                section=section,
            )

            page_numbers = self._resolve_page_numbers(section)
            page_number = min(page_numbers) if page_numbers else None

            keys = [
                section_id,
                section_number,
                self._normalize_reference_text(title, "section_reference"),
            ]

            for key in keys:
                key = self._clean_text(key)

                if not key:
                    continue

                targets.append(
                    ReferenceTarget(
                        reference_target_id=make_id("ref_target"),
                        target_type="section",
                        target_key=self._normalize_reference_key(key),
                        label=title or section_id,
                        page_number=page_number,
                        page_numbers=page_numbers,
                        source_id=section_id,
                        confidence=0.75,
                        source="document_structure_or_section_linker",
                        metadata={
                            "section_id": section_id,
                            "section_number": section_number,
                            "title": title,
                            "level": section.get("level", 0),
                            "page_start": section.get("page_start"),
                            "page_end": section.get("page_end"),
                        },
                    )
                )

        return targets

    def _build_table_targets(
        self,
        page_raws: List[PageRaw],
    ) -> List[ReferenceTarget]:
        targets: List[ReferenceTarget] = []

        for page_raw in page_raws:
            caption_meta = page_raw.metadata.get("caption_detector", {}) or {}
            captions = caption_meta.get("caption_candidates", []) or caption_meta.get("captions_on_page", []) or []

            for caption in captions:
                caption_type = str(caption.get("caption_type", "")).lower()
                text = self._clean_text(caption.get("text", ""))

                if "table" not in caption_type and not re.match(r"^\s*(Bảng|Bang|Table)\s+\d+", text, flags=re.IGNORECASE):
                    continue

                table_key = self._extract_number_key(text)

                if not table_key:
                    continue

                targets.append(
                    ReferenceTarget(
                        reference_target_id=make_id("ref_target"),
                        target_type="table",
                        target_key=self._normalize_reference_key(f"bảng {table_key}"),
                        label=text,
                        page_number=page_raw.page_number,
                        page_numbers=[page_raw.page_number],
                        source_id=caption.get("caption_id", "") or caption.get("id", ""),
                        confidence=0.78,
                        source="caption_detector",
                        metadata={
                            "caption": caption,
                            "table_number": table_key,
                        },
                    )
                )

            table_meta_keys = [
                "table_semantic_recognizer",
                "table_understanding_pipeline",
                "table_structure_recognizer",
                "table_grid_builder",
            ]

            for meta_key in table_meta_keys:
                meta = page_raw.metadata.get(meta_key, {}) or {}

                for item_key in [
                    "table_semantics_on_page",
                    "table_structures_on_page",
                    "table_grids_on_page",
                    "table_semantics",
                    "table_structures",
                    "table_grids",
                ]:
                    tables = meta.get(item_key, []) or []

                    for table in tables:
                        title = self._clean_text(
                            table.get("title")
                            or table.get("caption")
                            or table.get("text_preview")
                            or ""
                        )

                        table_grid_id = table.get("table_grid_id", "")
                        table_semantic_id = table.get("table_semantic_id", "")
                        table_structure_id = table.get("table_structure_id", "")

                        source_id = table_semantic_id or table_structure_id or table_grid_id

                        if not source_id and not title:
                            continue

                        keys = [
                            source_id,
                            table_grid_id,
                            self._normalize_reference_text(title, "table_reference"),
                        ]

                        number_key = self._extract_number_key(title)

                        if number_key:
                            keys.append(f"bảng {number_key}")

                        for key in keys:
                            key = self._clean_text(key)

                            if not key:
                                continue

                            targets.append(
                                ReferenceTarget(
                                    reference_target_id=make_id("ref_target"),
                                    target_type="table",
                                    target_key=self._normalize_reference_key(key),
                                    label=title or source_id,
                                    page_number=page_raw.page_number,
                                    page_numbers=[page_raw.page_number],
                                    source_id=source_id,
                                    confidence=self._safe_float(table.get("confidence"), default=0.65),
                                    source=meta_key,
                                    metadata={
                                        "table_grid_id": table_grid_id,
                                        "table_semantic_id": table_semantic_id,
                                        "table_structure_id": table_structure_id,
                                        "title": title,
                                    },
                                )
                            )

        return targets

    def _build_figure_targets(
        self,
        page_raws: List[PageRaw],
    ) -> List[ReferenceTarget]:
        targets: List[ReferenceTarget] = []

        for page_raw in page_raws:
            caption_meta = page_raw.metadata.get("caption_detector", {}) or {}
            captions = caption_meta.get("caption_candidates", []) or caption_meta.get("captions_on_page", []) or []

            for caption in captions:
                caption_type = str(caption.get("caption_type", "")).lower()
                text = self._clean_text(caption.get("text", ""))

                is_figure_caption = (
                    "figure" in caption_type
                    or "image" in caption_type
                    or "chart" in caption_type
                    or bool(re.match(r"^\s*(Hình|Hinh|Figure|Fig\.|Sơ\s*đồ|So\s*do)\s+\d+", text, flags=re.IGNORECASE))
                )

                if not is_figure_caption:
                    continue

                figure_key = self._extract_number_key(text)

                if not figure_key:
                    continue

                normalized_keys = [
                    f"hình {figure_key}",
                    f"figure {figure_key}",
                ]

                for key in normalized_keys:
                    targets.append(
                        ReferenceTarget(
                            reference_target_id=make_id("ref_target"),
                            target_type="figure",
                            target_key=self._normalize_reference_key(key),
                            label=text,
                            page_number=page_raw.page_number,
                            page_numbers=[page_raw.page_number],
                            source_id=caption.get("caption_id", "") or caption.get("id", ""),
                            confidence=0.76,
                            source="caption_detector",
                            metadata={
                                "caption": caption,
                                "figure_number": figure_key,
                            },
                        )
                    )

        return targets

    def _build_entity_targets(
        self,
        entity_link_result: Dict[str, Any],
    ) -> List[ReferenceTarget]:
        targets: List[ReferenceTarget] = []

        entities = entity_link_result.get("entities", []) or []

        for entity in entities:
            entity_id = entity.get("entity_id", "")

            if not entity_id:
                continue

            text = self._clean_text(entity.get("text", ""))
            normalized = entity.get("normalized_text", "") or self._normalize_reference_key(text)
            page_numbers = entity.get("page_numbers", []) or []

            targets.append(
                ReferenceTarget(
                    reference_target_id=make_id("ref_target"),
                    target_type="entity",
                    target_key=self._normalize_reference_key(normalized),
                    label=text or entity_id,
                    page_number=min(page_numbers) if page_numbers else None,
                    page_numbers=page_numbers,
                    source_id=entity_id,
                    confidence=self._safe_float(entity.get("confidence"), default=0.60),
                    source="entity_linker",
                    metadata={
                        "entity_id": entity_id,
                        "entity_type": entity.get("entity_type", ""),
                        "occurrence_count": entity.get("occurrence_count", 0),
                    },
                )
            )

        return targets

    def _resolve_target(
        self,
        reference_type: str,
        reference_text: str,
        normalized_text: str,
        from_page: int,
        target_index: Dict[str, List[ReferenceTarget]],
        entity_link_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if reference_type == "page_reference" and self.config.resolve_page_targets:
            target_page = self._infer_page_number(reference_text)
            target_key = str(target_page) if target_page else ""

            target = self._find_target(
                target_type="page",
                target_key=target_key,
                target_index=target_index,
            )

            return self._target_to_resolution(target, target_key, target_page)

        if reference_type in ["section_reference", "article_reference", "clause_reference"] and self.config.resolve_section_targets:
            candidate_keys = self._candidate_section_keys(reference_text)

            for key in candidate_keys:
                target = self._find_target(
                    target_type="section",
                    target_key=key,
                    target_index=target_index,
                )

                if target:
                    return self._target_to_resolution(target, key, target.page_number)

            return {
                "target_type": "section",
                "target_key": self._normalize_reference_key(reference_text),
                "target_id": "",
                "target_page": None,
                "target_pages": [],
                "resolved": False,
            }

        if reference_type == "table_reference" and self.config.resolve_table_targets:
            candidate_keys = self._candidate_table_keys(reference_text)

            for key in candidate_keys:
                target = self._find_target(
                    target_type="table",
                    target_key=key,
                    target_index=target_index,
                )

                if target:
                    return self._target_to_resolution(target, key, target.page_number)

            return {
                "target_type": "table",
                "target_key": self._normalize_reference_key(reference_text),
                "target_id": "",
                "target_page": None,
                "target_pages": [],
                "resolved": False,
            }

        if reference_type == "figure_reference" and self.config.resolve_figure_targets:
            candidate_keys = self._candidate_figure_keys(reference_text)

            for key in candidate_keys:
                target = self._find_target(
                    target_type="figure",
                    target_key=key,
                    target_index=target_index,
                )

                if target:
                    return self._target_to_resolution(target, key, target.page_number)

            return {
                "target_type": "figure",
                "target_key": self._normalize_reference_key(reference_text),
                "target_id": "",
                "target_page": None,
                "target_pages": [],
                "resolved": False,
            }

        if reference_type in ["legal_document_reference", "email_reference", "url_reference"] and self.config.resolve_entity_targets:
            key = self._normalize_reference_key(reference_text)

            target = self._find_target(
                target_type="entity",
                target_key=key,
                target_index=target_index,
            )

            if target:
                return self._target_to_resolution(target, key, target.page_number)

        return {
            "target_type": "",
            "target_key": normalized_text,
            "target_id": "",
            "target_page": None,
            "target_pages": [],
            "resolved": False,
        }

    def _target_to_resolution(
        self,
        target: Optional[ReferenceTarget],
        target_key: str,
        target_page: Optional[int],
    ) -> Dict[str, Any]:
        if not target:
            return {
                "target_type": "",
                "target_key": target_key,
                "target_id": "",
                "target_page": target_page,
                "target_pages": [target_page] if target_page else [],
                "resolved": bool(target_page),
            }

        return {
            "target_type": target.target_type,
            "target_key": target.target_key,
            "target_id": target.source_id,
            "target_page": target.page_number,
            "target_pages": target.page_numbers or ([target.page_number] if target.page_number else []),
            "target_label": target.label,
            "target_confidence": target.confidence,
            "resolved": True,
        }

    def _build_target_index(
        self,
        targets: List[ReferenceTarget],
    ) -> Dict[str, List[ReferenceTarget]]:
        index: Dict[str, List[ReferenceTarget]] = {}

        for target in targets:
            key = f"{target.target_type}|{self._normalize_reference_key(target.target_key)}"
            index.setdefault(key, [])
            index[key].append(target)

        return index

    def _find_target(
        self,
        target_type: str,
        target_key: str,
        target_index: Dict[str, List[ReferenceTarget]],
    ) -> Optional[ReferenceTarget]:
        if not target_key:
            return None

        normalized_key = self._normalize_reference_key(target_key)
        index_key = f"{target_type}|{normalized_key}"

        targets = target_index.get(index_key, [])

        if not targets:
            return None

        targets = sorted(
            targets,
            key=lambda item: (
                -item.confidence,
                item.page_number if item.page_number is not None else 999999,
            ),
        )

        return targets[0]

    def _candidate_section_keys(
        self,
        reference_text: str,
    ) -> List[str]:
        text = self._normalize_reference_key(reference_text)
        keys = [text]

        number_key = self._extract_number_key(reference_text)

        if number_key:
            keys.extend(
                [
                    number_key,
                    f"muc {number_key}",
                    f"phan {number_key}",
                    f"chuong {number_key}",
                    f"dieu {number_key}",
                    f"khoan {number_key}",
                ]
            )

        return self._unique_clean(keys)

    def _candidate_table_keys(
        self,
        reference_text: str,
    ) -> List[str]:
        text = self._normalize_reference_key(reference_text)
        number_key = self._extract_number_key(reference_text)

        keys = [text]

        if number_key:
            keys.extend(
                [
                    f"bang {number_key}",
                    f"table {number_key}",
                    number_key,
                ]
            )

        return self._unique_clean(keys)

    def _candidate_figure_keys(
        self,
        reference_text: str,
    ) -> List[str]:
        text = self._normalize_reference_key(reference_text)
        number_key = self._extract_number_key(reference_text)

        keys = [text]

        if number_key:
            keys.extend(
                [
                    f"hinh {number_key}",
                    f"figure {number_key}",
                    f"so do {number_key}",
                    number_key,
                ]
            )

        return self._unique_clean(keys)

    def _infer_page_number(
        self,
        text: str,
    ) -> Optional[int]:
        match = re.search(r"\d{1,5}", text)

        if not match:
            return None

        try:
            return int(match.group(0))
        except Exception:
            return None

    def _extract_number_key(
        self,
        text: str,
    ) -> str:
        text = self._clean_text(text)

        match = re.search(r"\d+(?:\.\d+)*", text)

        if match:
            return match.group(0)

        match = re.search(r"\b[IVXLC]+\b", text, flags=re.IGNORECASE)

        if match:
            return match.group(0).upper()

        return ""

    def _extract_section_number(
        self,
        title: str,
        section: Dict[str, Any],
    ) -> str:
        for key in ["section_number", "number", "heading_number"]:
            value = self._clean_text(section.get(key, ""))

            if value:
                return value

        return self._extract_number_key(title)

    def _infer_link_type(
        self,
        reference_type: str,
        target: Dict[str, Any],
    ) -> str:
        resolved = target.get("resolved", False)

        if resolved:
            return f"{reference_type}_resolved"

        return f"{reference_type}_unresolved"

    def _score_reference_link(
        self,
        reference_type: str,
        base_confidence: float,
        target: Dict[str, Any],
        from_page: int,
    ) -> float:
        score = base_confidence

        if target.get("resolved"):
            score += 0.10

        if target.get("target_page") and target.get("target_page") != from_page:
            score += 0.05

        if reference_type in [
            "page_reference",
            "table_reference",
            "figure_reference",
            "legal_document_reference",
            "email_reference",
            "url_reference",
        ]:
            score += 0.03

        return round(max(0.0, min(score, 0.95)), 4)

    def _build_page_section_context(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Dict[str, Any],
        section_link_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        contexts: Dict[str, Dict[str, Any]] = {}

        for page_raw in page_raws:
            contexts[str(page_raw.page_number)] = {
                "current_section_id": "",
                "current_section_title": "",
                "active_sections": [],
            }

        page_sections = section_link_result.get("page_sections", {}) or {}

        if page_sections:
            for page_key, ctx in page_sections.items():
                current = ctx.get("current_section", {}) or {}

                contexts[str(page_key)] = {
                    "current_section_id": current.get("section_id", ctx.get("current_section_id", "")),
                    "current_section_title": current.get("title", ctx.get("current_section_title", "")),
                    "active_sections": ctx.get("active_sections", []) or [],
                }

            return contexts

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

                contexts[str(page_key)] = {
                    "current_section_id": current.get("section_id", ""),
                    "current_section_title": current.get("title", ""),
                    "active_sections": sorted_links,
                }

            return contexts

        sections = document_structure_result.get("sections", []) or []

        for page_raw in page_raws:
            active_sections = []

            for section in sections:
                page_numbers = self._resolve_page_numbers(section)

                if page_raw.page_number in page_numbers:
                    active_sections.append(section)

            active_sections = sorted(
                active_sections,
                key=lambda item: (
                    self._safe_int(item.get("level"), default=0),
                    self._safe_int(item.get("order"), default=0),
                ),
            )

            current = active_sections[-1] if active_sections else {}

            contexts[str(page_raw.page_number)] = {
                "current_section_id": current.get("section_id", ""),
                "current_section_title": current.get("title", ""),
                "active_sections": active_sections,
            }

        return contexts

    def _resolve_page_numbers(
        self,
        item: Dict[str, Any],
    ) -> List[int]:
        page_numbers = item.get("page_numbers", []) or item.get("content_page_numbers", []) or []

        resolved = [
            self._safe_int(page, default=0)
            for page in page_numbers
            if self._safe_int(page, default=0) > 0
        ]

        if resolved:
            return sorted(list(dict.fromkeys(resolved)))

        page_start = item.get("page_start")
        page_end = item.get("page_end")

        if page_start is not None and page_end is not None:
            try:
                return list(range(int(page_start), int(page_end) + 1))
            except Exception:
                pass

        page_number = self._safe_int(item.get("page_number"), default=0)

        if page_number > 0:
            return [page_number]

        return []

    def _group_links_by_page(
        self,
        links: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for link in links:
            page_numbers = [link.get("from_page")]

            for page in link.get("target_pages", []) or []:
                page_numbers.append(page)

            if link.get("target_page"):
                page_numbers.append(link.get("target_page"))

            for page_number in page_numbers:
                if not page_number:
                    continue

                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(link)

        return grouped

    def _group_links_by_type(
        self,
        links: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for link in links:
            reference_type = link.get("reference_type", "unknown")
            grouped.setdefault(reference_type, [])
            grouped[reference_type].append(link)

        return grouped

    def _group_targets_by_type(
        self,
        targets: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for target in targets:
            target_type = target.get("target_type", "unknown")
            grouped.setdefault(target_type, [])
            grouped[target_type].append(target)

        return grouped

    def _build_reference_network(
        self,
        reference_links: List[Dict[str, Any]],
        reference_targets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        nodes = []
        edges = []

        page_nodes = set()

        for link in reference_links:
            from_page = link.get("from_page")
            target_page = link.get("target_page")

            if from_page:
                page_nodes.add(from_page)

            if target_page:
                page_nodes.add(target_page)

            for page in link.get("target_pages", []) or []:
                page_nodes.add(page)

        for page_number in sorted(page_nodes):
            nodes.append(
                {
                    "node_id": f"page_{page_number}",
                    "node_type": "page",
                    "label": f"Page {page_number}",
                    "page_number": page_number,
                }
            )

        for target in reference_targets:
            target_id = target.get("source_id") or target.get("reference_target_id")

            if not target_id:
                continue

            nodes.append(
                {
                    "node_id": f"target_{target_id}",
                    "node_type": f"target_{target.get('target_type', 'unknown')}",
                    "label": target.get("label", ""),
                    "target_type": target.get("target_type", ""),
                    "page_numbers": target.get("page_numbers", []),
                    "confidence": target.get("confidence", 0.5),
                }
            )

        for link in reference_links:
            from_page = link.get("from_page")
            target_page = link.get("target_page")
            target_id = link.get("target_id", "")

            if from_page and target_page:
                edges.append(
                    {
                        "edge_id": make_id("ref_edge"),
                        "source_id": f"page_{from_page}",
                        "target_id": f"page_{target_page}",
                        "edge_type": "page_references_page",
                        "reference_type": link.get("reference_type", ""),
                        "confidence": link.get("confidence", 0.5),
                    }
                )

            if from_page and target_id:
                edges.append(
                    {
                        "edge_id": make_id("ref_edge"),
                        "source_id": f"page_{from_page}",
                        "target_id": f"target_{target_id}",
                        "edge_type": "page_references_target",
                        "reference_type": link.get("reference_type", ""),
                        "confidence": link.get("confidence", 0.5),
                    }
                )

        return {
            "nodes": self._deduplicate_node_dicts(nodes),
            "edges": self._deduplicate_edge_dicts(edges),
            "summary": {
                "node_count": len(self._deduplicate_node_dicts(nodes)),
                "edge_count": len(self._deduplicate_edge_dicts(edges)),
            },
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        links_by_page = result.get("reference_links_by_page", {})
        summary = result.get("reference_link_summary", {})

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("reference_linker", {})
            page_raw.metadata["reference_linker"] = {
                "processor": "ReferenceLinker",
                "reference_links_on_page": links_by_page.get(page_key, []),
                "reference_link_count_on_page": len(links_by_page.get(page_key, [])),
                "reference_link_summary": summary,
            }

    def _build_summary(
        self,
        page_raws: List[PageRaw],
        reference_links: List[Dict[str, Any]],
        reference_targets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        by_page: Dict[str, int] = {}
        resolved_count = 0

        for link in reference_links:
            reference_type = link.get("reference_type", "unknown")
            by_type[reference_type] = by_type.get(reference_type, 0) + 1

            page_key = str(link.get("from_page", "unknown"))
            by_page[page_key] = by_page.get(page_key, 0) + 1

            if link.get("target_id") or link.get("target_page"):
                resolved_count += 1

        return {
            "has_reference_links": len(reference_links) > 0,
            "page_count": len(page_raws),
            "reference_link_count": len(reference_links),
            "reference_target_count": len(reference_targets),
            "resolved_reference_count": resolved_count,
            "unresolved_reference_count": len(reference_links) - resolved_count,
            "resolution_ratio": round(resolved_count / max(len(reference_links), 1), 4),
            "by_reference_type": by_type,
            "by_page": by_page,
        }

    def _deduplicate_reference_links(
        self,
        links: List[ReferenceLink],
    ) -> List[ReferenceLink]:
        seen = set()
        result: List[ReferenceLink] = []

        for link in links:
            key = (
                link.reference_type,
                link.normalized_text,
                link.from_page,
                link.start_char,
                link.end_char,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(link)

        return result

    def _deduplicate_targets(
        self,
        targets: List[ReferenceTarget],
    ) -> List[ReferenceTarget]:
        seen = set()
        result: List[ReferenceTarget] = []

        for target in targets:
            key = (
                target.target_type,
                self._normalize_reference_key(target.target_key),
                target.source_id,
                target.page_number,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(target)

        return result

    def _deduplicate_node_dicts(
        self,
        nodes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for node in nodes:
            node_id = node.get("node_id", "")

            if not node_id or node_id in seen:
                continue

            seen.add(node_id)
            result.append(node)

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
                edge.get("reference_type", ""),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(edge)

        return result

    def _valid_reference_text(
        self,
        text: str,
    ) -> bool:
        text = self._clean_text(text)

        if len(text) < self.config.min_reference_text_length:
            return False

        if len(text) > self.config.max_reference_text_length:
            return False

        if text in [".", ",", ";", ":", "-", "_"]:
            return False

        return True

    def _normalize_reference_text(
        self,
        text: str,
        reference_type: str,
    ) -> str:
        text = self._clean_text(text)

        if reference_type in ["email_reference", "url_reference"]:
            return text.lower()

        text = self._normalize_reference_key(text)

        return text

    def _normalize_reference_key(
        self,
        text: Any,
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

        text = re.sub(r"\s*\/\s*", "/", text)
        text = re.sub(r"\s*-\s*", "-", text)
        text = re.sub(r"[^a-z0-9_\-\/\.%]+", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _unique_clean(
        self,
        values: List[str],
    ) -> List[str]:
        result = []
        seen = set()

        for value in values:
            cleaned = self._normalize_reference_key(value)

            if not cleaned or cleaned in seen:
                continue

            seen.add(cleaned)
            result.append(cleaned)

        return result

    def _context_snippet(
        self,
        text: str,
        start: int,
        end: int,
    ) -> str:
        left = max(0, start - self.config.context_window_chars)
        right = min(len(text), end + self.config.context_window_chars)

        return self._clean_text(text[left:right])

    def _page_text(
        self,
        page_raw: PageRaw,
    ) -> str:
        reading_meta = page_raw.metadata.get("reading_order_builder", {}) or {}
        text = reading_meta.get("reading_order_text", "")

        if text:
            return self._clean_text_block(text)

        return self._clean_text_block(page_raw.normalized_text or page_raw.raw_text or "")

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


def link_references(
    page_raws: List[PageRaw],
    document_structure_result: Optional[Dict[str, Any]] = None,
    section_link_result: Optional[Dict[str, Any]] = None,
    entity_link_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    linker = ReferenceLinker()
    return linker.process(
        page_raws=page_raws,
        document_structure_result=document_structure_result,
        section_link_result=section_link_result,
        entity_link_result=entity_link_result,
    )
