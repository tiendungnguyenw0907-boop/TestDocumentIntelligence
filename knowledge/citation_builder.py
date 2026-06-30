"""
citation_builder.py

Production V1 - Colab Ready

Purpose
-------
Build, normalize, group, and verify citations from evidence/chunks/pages.

Used by:
- KnowledgePipeline
- EvidenceBuilder
- RAGPipeline
- CitationVerifier

Input
-----
- page_raws
- evidence_result
- chunk_result
- document_structure_result
- table_understanding_result
- cross_page_context_result

Output
------
Dictionary with:
- citations
- citations_by_page
- citations_by_evidence
- citation_markers
- citation_summary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw
from document_ai.schemas.evidence_schema import (
    Evidence,
    Citation,
    EvidenceCollection,
    make_id,
    normalize_text,
    normalize_text_for_match,
)


@dataclass
class CitationBuilderConfig:
    build_from_evidence: bool = True
    build_from_chunks_if_no_evidence: bool = True
    build_from_tables_if_available: bool = True

    include_existing_citations: bool = True
    verify_citations: bool = True
    attach_to_pages: bool = True
    deduplicate_citations: bool = True

    citation_style: str = "vietnamese_page"
    marker_prefix: str = "tr."
    max_quote_chars: int = 500
    max_context_chars: int = 180

    min_quote_chars: int = 20
    min_confidence: float = 0.35

    prefer_evidence_quote: bool = True
    include_section_title: bool = True
    include_source_document: bool = True
    include_bbox: bool = True

    strict_page_verification: bool = False
    include_debug: bool = True


class CitationBuilder:
    def __init__(
        self,
        config: Optional[CitationBuilderConfig] = None,
    ):
        self.config = config or CitationBuilderConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        evidence_result: Optional[Dict[str, Any]] = None,
        chunk_result: Optional[Dict[str, Any]] = None,
        document_structure_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        cross_page_context_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        evidence_result = evidence_result or {}
        chunk_result = chunk_result or {}
        document_structure_result = document_structure_result or {}
        table_understanding_result = table_understanding_result or {}
        cross_page_context_result = cross_page_context_result or {}

        page_raws = sorted(
            page_raws or [],
            key=lambda item: item.page_number,
        )

        page_text_map = self._build_page_text_map(page_raws)

        evidence_items = self._collect_evidence(evidence_result)
        existing_citations = self._collect_existing_citations(evidence_result)

        citations: List[Citation] = []

        if self.config.include_existing_citations:
            citations.extend(existing_citations)

        if self.config.build_from_evidence and evidence_items:
            citations.extend(
                self._build_citations_from_evidence(
                    evidence_items=evidence_items,
                    page_text_map=page_text_map,
                )
            )

        if not evidence_items and self.config.build_from_chunks_if_no_evidence:
            citations.extend(
                self._build_citations_from_chunks(
                    chunk_result=chunk_result,
                    page_text_map=page_text_map,
                )
            )

        if self.config.build_from_tables_if_available:
            citations.extend(
                self._build_citations_from_tables(
                    table_understanding_result=table_understanding_result,
                    page_text_map=page_text_map,
                )
            )

        citations = [
            citation for citation in citations
            if citation.confidence >= self.config.min_confidence
        ]

        if self.config.verify_citations:
            citations = self._verify_citations(
                citations=citations,
                page_text_map=page_text_map,
            )

        if self.config.deduplicate_citations:
            citations = self._deduplicate_citations(citations)

        citations = self._sort_and_reindex_citations(citations)

        collection = EvidenceCollection(
            document_id=self._infer_document_id(
                page_raws=page_raws,
                evidence_items=evidence_items,
                chunk_result=chunk_result,
            ),
            source_document=self._infer_source_document(
                page_raws=page_raws,
                evidence_items=evidence_items,
            ),
            evidence=evidence_items,
            citations=citations,
            metadata={
                "processor": "CitationBuilder",
                "schema_version": "citation_builder_v1",
                "document_structure_available": bool(document_structure_result),
                "table_understanding_available": bool(table_understanding_result),
                "cross_page_context_available": bool(cross_page_context_result),
            },
        )

        result = collection.to_dict()
        result.update(
            {
                "processor": "CitationBuilder",
                "schema_version": "citation_builder_v1",
                "citations": [
                    citation.to_dict()
                    for citation in citations
                ],
                "citations_by_page": self._group_citations_by_page(citations),
                "citations_by_evidence": self._group_citations_by_evidence(citations),
                "citations_by_section": self._group_citations_by_section(citations),
                "citation_markers": self._build_citation_markers(citations),
                "citation_summary": self._build_summary(
                    citations=citations,
                    evidence_items=evidence_items,
                    page_raws=page_raws,
                ),
                "config": asdict(self.config),
            }
        )

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                citations=citations,
            )

        return result

    def _build_citations_from_evidence(
        self,
        evidence_items: List[Evidence],
        page_text_map: Dict[int, str],
    ) -> List[Citation]:
        citations: List[Citation] = []

        for evidence in evidence_items:
            quote = self._select_quote_from_evidence(evidence)

            if not quote:
                continue

            citation = Citation(
                citation_id=make_id("citation"),
                citation_type=self._infer_citation_type(evidence),
                evidence_id=evidence.evidence_id,
                evidence_ids=[evidence.evidence_id],
                document_id=evidence.document_id,
                source_document=evidence.source_document,
                page_number=evidence.page_number,
                page_numbers=evidence.page_numbers,
                page_start=evidence.page_start,
                page_end=evidence.page_end,
                section_id=evidence.section_id,
                section_title=evidence.section_title,
                chunk_id=evidence.chunk_id,
                bbox=evidence.bbox if self.config.include_bbox else [],
                quote=quote,
                citation_text=self._make_citation_text(
                    source_document=evidence.source_document,
                    section_title=evidence.section_title,
                    page_numbers=evidence.page_numbers,
                ),
                citation_marker=self._make_citation_marker(evidence.page_numbers),
                confidence=evidence.confidence,
                verified=False,
                verification_status="unverified",
                source="citation_builder_from_evidence",
                metadata={
                    "evidence_type": evidence.evidence_type,
                    "relevance_score": evidence.relevance_score,
                    "source_chunk_id": evidence.source_chunk_id,
                    "table_grid_id": evidence.table_grid_id,
                    "table_semantic_id": evidence.table_semantic_id,
                    "entity_id": evidence.entity_id,
                    "reference_id": evidence.reference_id,
                    "context_before": evidence.context_before,
                    "context_after": evidence.context_after,
                },
            )

            citations.append(citation)

        return citations

    def _build_citations_from_chunks(
        self,
        chunk_result: Dict[str, Any],
        page_text_map: Dict[int, str],
    ) -> List[Citation]:
        citations: List[Citation] = []

        chunks = self._collect_chunks(chunk_result)

        for chunk in chunks:
            text = normalize_text(chunk.get("text", ""))

            if len(text) < self.config.min_quote_chars:
                continue

            page_numbers = self._normalize_page_numbers(chunk.get("page_numbers", []))

            if not page_numbers and chunk.get("page_number"):
                page_numbers = [self._safe_int(chunk.get("page_number"), default=0)]
                page_numbers = [page for page in page_numbers if page > 0]

            quote = self._truncate_quote(text)

            citation = Citation(
                citation_id=make_id("citation"),
                citation_type="chunk_citation",
                evidence_id="",
                evidence_ids=[],
                document_id=chunk.get("document_id", ""),
                source_document=chunk.get("source_document", ""),
                page_number=min(page_numbers) if page_numbers else None,
                page_numbers=page_numbers,
                page_start=min(page_numbers) if page_numbers else None,
                page_end=max(page_numbers) if page_numbers else None,
                section_id=chunk.get("section_id", ""),
                section_title=chunk.get("section_title", ""),
                chunk_id=chunk.get("chunk_id", ""),
                bbox=chunk.get("bbox", []) if self.config.include_bbox else [],
                quote=quote,
                citation_text=self._make_citation_text(
                    source_document=chunk.get("source_document", ""),
                    section_title=chunk.get("section_title", ""),
                    page_numbers=page_numbers,
                ),
                citation_marker=self._make_citation_marker(page_numbers),
                confidence=self._safe_float(chunk.get("confidence"), default=0.60),
                verified=False,
                verification_status="unverified",
                source="citation_builder_from_chunk",
                metadata={
                    "chunk_type": chunk.get("chunk_type", ""),
                    "source": chunk.get("source", ""),
                },
            )

            citations.append(citation)

        return citations

    def _build_citations_from_tables(
        self,
        table_understanding_result: Dict[str, Any],
        page_text_map: Dict[int, str],
    ) -> List[Citation]:
        citations: List[Citation] = []

        table_items = []

        for key in [
            "table_semantics",
            "table_records",
            "multi_page_tables",
        ]:
            values = table_understanding_result.get(key, []) or []

            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        table_items.append((key, item))

        table_semantic_result = table_understanding_result.get("table_semantic_result", {}) or {}

        if isinstance(table_semantic_result, dict):
            for item in table_semantic_result.get("table_semantics", []) or []:
                if isinstance(item, dict):
                    table_items.append(("table_semantics", item))

        multi_page_result = table_understanding_result.get("multi_page_table_result", {}) or {}

        if isinstance(multi_page_result, dict):
            for item in multi_page_result.get("multi_page_tables", []) or []:
                if isinstance(item, dict):
                    table_items.append(("multi_page_tables", item))

        seen = set()

        for table_type, table in table_items:
            table_id = (
                table.get("table_semantic_id")
                or table.get("table_record_id")
                or table.get("multi_page_table_id")
                or table.get("table_grid_id")
                or ""
            )

            if not table_id:
                continue

            if table_id in seen:
                continue

            seen.add(table_id)

            page_numbers = self._normalize_page_numbers(
                table.get("page_numbers", [])
            )

            if not page_numbers and table.get("page_number"):
                page_numbers = [self._safe_int(table.get("page_number"), default=0)]
                page_numbers = [page for page in page_numbers if page > 0]

            if not page_numbers:
                continue

            quote = self._table_quote(table)

            if not quote:
                continue

            citation = Citation(
                citation_id=make_id("citation"),
                citation_type="table_citation",
                evidence_id="",
                evidence_ids=[],
                document_id=table.get("document_id", ""),
                source_document=table.get("source_document", ""),
                page_number=min(page_numbers),
                page_numbers=page_numbers,
                page_start=min(page_numbers),
                page_end=max(page_numbers),
                section_id=table.get("section_id", ""),
                section_title=table.get("section_title", ""),
                chunk_id="",
                bbox=table.get("bbox", []) if self.config.include_bbox else [],
                quote=quote,
                citation_text=self._make_citation_text(
                    source_document=table.get("source_document", ""),
                    section_title=table.get("title") or table.get("caption") or table.get("section_title", ""),
                    page_numbers=page_numbers,
                ),
                citation_marker=self._make_citation_marker(page_numbers),
                confidence=self._safe_float(table.get("confidence"), default=0.65),
                verified=False,
                verification_status="unverified",
                source="citation_builder_from_table",
                metadata={
                    "table_type": table_type,
                    "table_id": table_id,
                    "table_grid_id": table.get("table_grid_id", ""),
                    "table_structure_id": table.get("table_structure_id", ""),
                    "table_semantic_id": table.get("table_semantic_id", ""),
                    "multi_page_table_id": table.get("multi_page_table_id", ""),
                },
            )

            citations.append(citation)

        return citations

    def _verify_citations(
        self,
        citations: List[Citation],
        page_text_map: Dict[int, str],
    ) -> List[Citation]:
        verified: List[Citation] = []

        for citation in citations:
            status, score, details = self._verify_single_citation(
                citation=citation,
                page_text_map=page_text_map,
            )

            citation.verified = status in ["verified_exact", "verified_fuzzy", "verified_page_available"]
            citation.verification_status = status
            citation.confidence = round(
                max(citation.confidence, score) if citation.verified else min(citation.confidence, score),
                4,
            )

            citation.metadata.setdefault("verification", {})
            citation.metadata["verification"] = details

            if self.config.strict_page_verification and not citation.verified:
                continue

            verified.append(citation)

        return verified

    def _verify_single_citation(
        self,
        citation: Citation,
        page_text_map: Dict[int, str],
    ) -> Tuple[str, float, Dict[str, Any]]:
        quote = normalize_text(citation.quote)
        page_numbers = citation.page_numbers or []

        if not quote:
            return (
                "no_quote",
                0.35,
                {
                    "reason": "citation has no quote",
                    "page_numbers": page_numbers,
                },
            )

        if not page_numbers:
            return (
                "no_page_reference",
                0.40,
                {
                    "reason": "citation has no page_numbers",
                    "quote_preview": quote[:120],
                },
            )

        quote_match = normalize_text_for_match(quote)

        for page_number in page_numbers:
            page_text = page_text_map.get(page_number, "")
            page_match = normalize_text_for_match(page_text)

            if not page_text:
                continue

            if quote_match and quote_match in page_match:
                return (
                    "verified_exact",
                    0.92,
                    {
                        "page_number": page_number,
                        "match_type": "exact_normalized_quote_in_page",
                        "quote_preview": quote[:160],
                    },
                )

            fuzzy_score = self._fuzzy_overlap_score(quote_match, page_match)

            if fuzzy_score >= 0.55:
                return (
                    "verified_fuzzy",
                    round(0.65 + min(fuzzy_score, 0.30), 4),
                    {
                        "page_number": page_number,
                        "match_type": "token_overlap",
                        "fuzzy_score": round(fuzzy_score, 4),
                        "quote_preview": quote[:160],
                    },
                )

        has_any_page_text = any(page_text_map.get(page, "") for page in page_numbers)

        if has_any_page_text:
            return (
                "page_available_quote_not_found",
                0.45,
                {
                    "page_numbers": page_numbers,
                    "match_type": "page_available_but_quote_not_found",
                    "quote_preview": quote[:160],
                },
            )

        return (
            "verified_page_available",
            0.55,
            {
                "page_numbers": page_numbers,
                "match_type": "page_reference_only",
                "quote_preview": quote[:160],
            },
        )

    def _collect_evidence(
        self,
        evidence_result: Dict[str, Any],
    ) -> List[Evidence]:
        evidence_items = []

        candidate_lists = []

        for key in [
            "evidence",
            "evidence_items",
            "retrieved_evidence",
            "supporting_evidence",
        ]:
            values = evidence_result.get(key, []) or []

            if isinstance(values, list):
                candidate_lists.append(values)

        evidence_collection = evidence_result.get("evidence_collection", {}) or {}

        if isinstance(evidence_collection, dict):
            for key in ["evidence", "evidence_items"]:
                values = evidence_collection.get(key, []) or []

                if isinstance(values, list):
                    candidate_lists.append(values)

        for values in candidate_lists:
            for item in values:
                try:
                    if isinstance(item, Evidence):
                        evidence = item
                    elif isinstance(item, dict):
                        evidence = Evidence.from_dict(item)
                    else:
                        continue

                    evidence_items.append(evidence)
                except Exception:
                    continue

        return self._deduplicate_evidence(evidence_items)

    def _collect_existing_citations(
        self,
        evidence_result: Dict[str, Any],
    ) -> List[Citation]:
        citations = []

        candidate_lists = []

        for key in ["citations", "citation_items"]:
            values = evidence_result.get(key, []) or []

            if isinstance(values, list):
                candidate_lists.append(values)

        evidence_collection = evidence_result.get("evidence_collection", {}) or {}

        if isinstance(evidence_collection, dict):
            values = evidence_collection.get("citations", []) or []

            if isinstance(values, list):
                candidate_lists.append(values)

        for values in candidate_lists:
            for item in values:
                try:
                    if isinstance(item, Citation):
                        citation = item
                    elif isinstance(item, dict):
                        citation = Citation.from_dict(item)
                    else:
                        continue

                    citations.append(citation)
                except Exception:
                    continue

        return citations

    def _collect_chunks(
        self,
        chunk_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        chunks = []

        for key in [
            "chunks",
            "table_chunks",
            "child_chunks",
            "parent_chunks",
        ]:
            values = chunk_result.get(key, []) or []

            if isinstance(values, list):
                chunks.extend(
                    [
                        item for item in values
                        if isinstance(item, dict)
                    ]
                )

        chunk_collection = chunk_result.get("chunk_collection", {}) or {}

        if isinstance(chunk_collection, dict):
            for key in ["chunks", "table_chunks"]:
                values = chunk_collection.get(key, []) or []

                if isinstance(values, list):
                    chunks.extend(
                        [
                            item for item in values
                            if isinstance(item, dict)
                        ]
                    )

        return self._deduplicate_dicts(
            items=chunks,
            keys=["chunk_id", "content_hash"],
        )

    def _select_quote_from_evidence(
        self,
        evidence: Evidence,
    ) -> str:
        candidates = []

        if self.config.prefer_evidence_quote:
            candidates.extend(
                [
                    evidence.quote,
                    evidence.text,
                ]
            )
        else:
            candidates.extend(
                [
                    evidence.text,
                    evidence.quote,
                ]
            )

        for candidate in candidates:
            quote = normalize_text(candidate)

            if len(quote) >= self.config.min_quote_chars:
                return self._truncate_quote(quote)

        for candidate in candidates:
            quote = normalize_text(candidate)

            if quote:
                return self._truncate_quote(quote)

        return ""

    def _truncate_quote(
        self,
        text: str,
    ) -> str:
        text = normalize_text(text)

        if len(text) <= self.config.max_quote_chars:
            return text

        cut = text[: self.config.max_quote_chars]
        last_break = max(
            cut.rfind(". "),
            cut.rfind("; "),
            cut.rfind(", "),
            cut.rfind("\n"),
            cut.rfind(" "),
        )

        if last_break > self.config.max_quote_chars * 0.55:
            cut = cut[:last_break]

        return normalize_text(cut) + "..."

    def _table_quote(
        self,
        table: Dict[str, Any],
    ) -> str:
        parts = []

        for key in ["title", "caption"]:
            value = normalize_text(table.get(key, ""))

            if value:
                parts.append(value)

        headers = table.get("column_headers", []) or []

        if headers:
            header_text = " | ".join(
                [
                    normalize_text(item)
                    for item in headers
                    if normalize_text(item)
                ]
            )

            if header_text:
                parts.append(header_text)

        records = table.get("records", []) or []

        for record in records[:5]:
            if not isinstance(record, dict):
                continue

            raw_values = record.get("raw_values", {}) or record.get("values", {}) or {}

            if raw_values:
                line = " | ".join(
                    [
                        f"{key}: {normalize_text(value)}"
                        for key, value in raw_values.items()
                        if normalize_text(value)
                    ]
                )

                if line:
                    parts.append(line)

        if not parts and table.get("text"):
            parts.append(normalize_text(table.get("text", "")))

        return self._truncate_quote("\n".join(parts))

    def _make_citation_text(
        self,
        source_document: str = "",
        section_title: str = "",
        page_numbers: Optional[List[int]] = None,
    ) -> str:
        page_numbers = page_numbers or []

        parts = []

        if self.config.include_source_document and source_document:
            parts.append(normalize_text(source_document))

        if self.config.include_section_title and section_title:
            parts.append(normalize_text(section_title))

        if page_numbers:
            if len(page_numbers) == 1:
                parts.append(f"trang {page_numbers[0]}")
            else:
                parts.append(f"trang {page_numbers[0]}-{page_numbers[-1]}")

        if not parts:
            return "Nguồn trích dẫn"

        return ", ".join(parts)

    def _make_citation_marker(
        self,
        page_numbers: Optional[List[int]] = None,
    ) -> str:
        page_numbers = page_numbers or []

        if self.config.citation_style == "markdown_page":
            if len(page_numbers) == 1:
                return f"[p.{page_numbers[0]}]"

            if len(page_numbers) > 1:
                return f"[p.{page_numbers[0]}-{page_numbers[-1]}]"

            return "[source]"

        if len(page_numbers) == 1:
            return f"[{self.config.marker_prefix}{page_numbers[0]}]"

        if len(page_numbers) > 1:
            return f"[{self.config.marker_prefix}{page_numbers[0]}-{page_numbers[-1]}]"

        return "[nguồn]"

    def _infer_citation_type(
        self,
        evidence: Evidence,
    ) -> str:
        if evidence.table_grid_id or evidence.table_semantic_id:
            return "table_evidence_citation"

        if evidence.evidence_type:
            return f"{evidence.evidence_type}_citation"

        if evidence.page_numbers:
            return "page_citation"

        return "evidence_citation"

    def _build_page_text_map(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[int, str]:
        page_text_map: Dict[int, str] = {}

        for page_raw in page_raws:
            reading_meta = page_raw.metadata.get("reading_order_builder", {}) or {}
            reading_text = normalize_text(reading_meta.get("reading_order_text", ""))

            if reading_text:
                text = reading_text
            elif page_raw.normalized_text:
                text = page_raw.normalized_text
            else:
                text = page_raw.raw_text

            page_text_map[page_raw.page_number] = normalize_text(text)

        return page_text_map

    def _group_citations_by_page(
        self,
        citations: List[Citation],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for citation in citations:
            for page_number in citation.page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(citation.to_dict())

        return grouped

    def _group_citations_by_evidence(
        self,
        citations: List[Citation],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for citation in citations:
            evidence_ids = citation.evidence_ids or []

            if not evidence_ids:
                evidence_ids = ["no_evidence"]

            for evidence_id in evidence_ids:
                grouped.setdefault(evidence_id, [])
                grouped[evidence_id].append(citation.to_dict())

        return grouped

    def _group_citations_by_section(
        self,
        citations: List[Citation],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for citation in citations:
            section_key = citation.section_id or "no_section"
            grouped.setdefault(section_key, [])
            grouped[section_key].append(citation.to_dict())

        return grouped

    def _build_citation_markers(
        self,
        citations: List[Citation],
    ) -> Dict[str, Dict[str, Any]]:
        markers: Dict[str, Dict[str, Any]] = {}

        for citation in citations:
            marker = citation.citation_marker or self._make_citation_marker(citation.page_numbers)

            markers[marker] = {
                "citation_id": citation.citation_id,
                "citation_text": citation.citation_text,
                "page_numbers": citation.page_numbers,
                "section_id": citation.section_id,
                "section_title": citation.section_title,
                "source_document": citation.source_document,
                "verified": citation.verified,
                "verification_status": citation.verification_status,
                "confidence": citation.confidence,
            }

        return markers

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        citations: List[Citation],
    ) -> None:
        citations_by_page = self._group_citations_by_page(citations)

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("citation_builder", {})
            page_raw.metadata["citation_builder"] = {
                "processor": "CitationBuilder",
                "citations_on_page": citations_by_page.get(page_key, []),
                "citation_count_on_page": len(citations_by_page.get(page_key, [])),
            }

    def _build_summary(
        self,
        citations: List[Citation],
        evidence_items: List[Evidence],
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        by_page: Dict[str, int] = {}
        by_status: Dict[str, int] = {}

        verified_count = 0

        for citation in citations:
            by_type[citation.citation_type] = by_type.get(citation.citation_type, 0) + 1
            by_status[citation.verification_status] = by_status.get(citation.verification_status, 0) + 1

            if citation.verified:
                verified_count += 1

            for page_number in citation.page_numbers:
                page_key = str(page_number)
                by_page[page_key] = by_page.get(page_key, 0) + 1

        return {
            "has_citations": len(citations) > 0,
            "citation_count": len(citations),
            "evidence_count": len(evidence_items),
            "page_count": len(page_raws),
            "verified_citation_count": verified_count,
            "unverified_citation_count": len(citations) - verified_count,
            "verification_ratio": round(verified_count / max(len(citations), 1), 4),
            "by_citation_type": by_type,
            "by_verification_status": by_status,
            "by_page": by_page,
        }

    def _fuzzy_overlap_score(
        self,
        quote_text: str,
        page_text: str,
    ) -> float:
        quote_tokens = [
            token for token in quote_text.split()
            if len(token) >= 3
        ]

        page_tokens = set(
            token for token in page_text.split()
            if len(token) >= 3
        )

        if not quote_tokens or not page_tokens:
            return 0.0

        matched = sum(
            1 for token in quote_tokens
            if token in page_tokens
        )

        return matched / max(len(quote_tokens), 1)

    def _deduplicate_evidence(
        self,
        evidence_items: List[Evidence],
    ) -> List[Evidence]:
        seen = set()
        result = []

        for item in evidence_items:
            key = item.evidence_id or item.content_hash

            if not key:
                key = (
                    normalize_text_for_match(item.text),
                    tuple(item.page_numbers),
                    item.section_id,
                    item.chunk_id,
                )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _deduplicate_citations(
        self,
        citations: List[Citation],
    ) -> List[Citation]:
        seen = set()
        result = []

        sorted_items = sorted(
            citations,
            key=lambda item: (
                tuple(item.page_numbers),
                item.section_id,
                item.chunk_id,
                -item.confidence,
            ),
        )

        for citation in sorted_items:
            key = (
                tuple(citation.evidence_ids),
                tuple(citation.page_numbers),
                citation.section_id,
                citation.chunk_id,
                normalize_text_for_match(citation.quote)[:300],
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(citation)

        return result

    def _deduplicate_dicts(
        self,
        items: List[Dict[str, Any]],
        keys: List[str],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for item in items:
            key = ""

            for key_name in keys:
                if item.get(key_name):
                    key = str(item.get(key_name))
                    break

            if not key:
                key = str(
                    (
                        normalize_text_for_match(item.get("text", ""))[:300],
                        tuple(self._normalize_page_numbers(item.get("page_numbers", []))),
                    )
                )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _sort_and_reindex_citations(
        self,
        citations: List[Citation],
    ) -> List[Citation]:
        citations = sorted(
            citations,
            key=lambda item: (
                min(item.page_numbers) if item.page_numbers else 999999,
                item.section_id,
                item.chunk_id,
                item.citation_id,
            ),
        )

        marker_counts: Dict[str, int] = {}

        for index, citation in enumerate(citations):
            citation.metadata.setdefault("citation_order", index)

            base_marker = self._make_citation_marker(citation.page_numbers)
            marker_counts[base_marker] = marker_counts.get(base_marker, 0) + 1

            if marker_counts[base_marker] == 1:
                citation.citation_marker = base_marker
            else:
                citation.citation_marker = base_marker.replace(
                    "]",
                    f".{marker_counts[base_marker]}]"
                )

            if not citation.citation_text:
                citation.citation_text = self._make_citation_text(
                    source_document=citation.source_document,
                    section_title=citation.section_title,
                    page_numbers=citation.page_numbers,
                )

        return citations

    def _normalize_page_numbers(
        self,
        values: Any,
    ) -> List[int]:
        if values is None:
            return []

        if not isinstance(values, list):
            values = [values]

        result = []

        for value in values:
            page = self._safe_int(value, default=0)

            if page > 0:
                result.append(page)

        return sorted(list(dict.fromkeys(result)))

    def _infer_document_id(
        self,
        page_raws: List[PageRaw],
        evidence_items: List[Evidence],
        chunk_result: Dict[str, Any],
    ) -> str:
        for item in evidence_items:
            if item.document_id:
                return item.document_id

        if chunk_result.get("document_id"):
            return chunk_result.get("document_id", "")

        for page_raw in page_raws:
            if page_raw.document_id:
                return page_raw.document_id

        return ""

    def _infer_source_document(
        self,
        page_raws: List[PageRaw],
        evidence_items: List[Evidence],
    ) -> str:
        for item in evidence_items:
            if item.source_document:
                return item.source_document

        for page_raw in page_raws:
            if page_raw.source_document:
                return page_raw.source_document

        return ""

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


def build_citations(
    page_raws: List[PageRaw],
    evidence_result: Optional[Dict[str, Any]] = None,
    chunk_result: Optional[Dict[str, Any]] = None,
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    cross_page_context_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = CitationBuilder()
    return builder.process(
        page_raws=page_raws,
        evidence_result=evidence_result,
        chunk_result=chunk_result,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
        cross_page_context_result=cross_page_context_result,
    )
