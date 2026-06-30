"""
metadata_index_builder.py

Production V1 - Colab Ready

Purpose
-------
Build a pure-Python metadata index for fast filtering and faceted search
over pages, sections, tables, chunks, evidence, citations, and graph nodes.

Used by:
- KnowledgePipeline
- HybridRetriever
- Metadata-aware retrieval
- RAGPipeline

Input
-----
- page_raws
- metadata_enrichment_result
- knowledge_result
- chunk_result
- table_chunk_result
- evidence_result
- citation_result
- document_structure_result
- table_understanding_result
- graph_index_result
- bm25_index_result

Output
------
Dictionary with:
- metadata_index
- metadata_store
- field_indexes
- page_index
- section_index
- table_index
- chunk_index
- evidence_index
- citation_index
- metadata_index_summary
"""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from document_ai.schemas.page_raw_schema import PageRaw
from document_ai.schemas.page_raw_schema import (
    normalize_pdf_text,
    normalize_text_for_match,
    json_safe,
)


@dataclass
class MetadataIndexBuilderConfig:
    include_document_metadata: bool = True
    include_page_metadata: bool = True
    include_section_metadata: bool = True
    include_table_metadata: bool = True
    include_chunk_metadata: bool = True
    include_evidence_metadata: bool = True
    include_citation_metadata: bool = True
    include_graph_metadata: bool = True
    include_bm25_metadata: bool = True

    attach_to_pages: bool = True
    deduplicate_items: bool = True

    build_field_indexes: bool = True
    build_page_index: bool = True
    build_section_index: bool = True
    build_table_index: bool = True
    build_chunk_index: bool = True
    build_evidence_index: bool = True
    build_citation_index: bool = True
    build_type_index: bool = True
    build_keyword_index: bool = True
    build_quality_index: bool = True
    build_processing_hint_index: bool = True

    flatten_nested_metadata: bool = True
    max_flatten_depth: int = 3
    max_text_preview_chars: int = 600
    max_keyword_count_per_item: int = 50

    index_empty_values: bool = False
    normalize_index_values: bool = True
    include_debug: bool = True


class MetadataIndexBuilder:
    def __init__(
        self,
        config: Optional[MetadataIndexBuilderConfig] = None,
    ):
        self.config = config or MetadataIndexBuilderConfig()

    def process(
        self,
        page_raws: Optional[List[PageRaw]] = None,
        metadata_enrichment_result: Optional[Dict[str, Any]] = None,
        knowledge_result: Optional[Dict[str, Any]] = None,
        chunk_result: Optional[Dict[str, Any]] = None,
        table_chunk_result: Optional[Dict[str, Any]] = None,
        evidence_result: Optional[Dict[str, Any]] = None,
        citation_result: Optional[Dict[str, Any]] = None,
        document_structure_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        graph_index_result: Optional[Dict[str, Any]] = None,
        bm25_index_result: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        page_raws = sorted(
            page_raws or [],
            key=lambda item: item.page_number,
        )

        metadata_enrichment_result = metadata_enrichment_result or {}
        knowledge_result = knowledge_result or {}
        chunk_result = chunk_result or {}
        table_chunk_result = table_chunk_result or {}
        evidence_result = evidence_result or {}
        citation_result = citation_result or {}
        document_structure_result = document_structure_result or {}
        table_understanding_result = table_understanding_result or {}
        graph_index_result = graph_index_result or {}
        bm25_index_result = bm25_index_result or {}

        items = self._collect_metadata_items(
            page_raws=page_raws,
            metadata_enrichment_result=metadata_enrichment_result,
            knowledge_result=knowledge_result,
            chunk_result=chunk_result,
            table_chunk_result=table_chunk_result,
            evidence_result=evidence_result,
            citation_result=citation_result,
            document_structure_result=document_structure_result,
            table_understanding_result=table_understanding_result,
            graph_index_result=graph_index_result,
            bm25_index_result=bm25_index_result,
        )

        if self.config.deduplicate_items:
            items = self._deduplicate_items(items)

        metadata_store = self._build_metadata_store(items)

        field_indexes = {}
        page_index = {}
        section_index = {}
        table_index = {}
        chunk_index = {}
        evidence_index = {}
        citation_index = {}
        type_index = {}
        keyword_index = {}
        quality_index = {}
        processing_hint_index = {}

        if self.config.build_field_indexes:
            field_indexes = self._build_field_indexes(metadata_store)

        if self.config.build_page_index:
            page_index = self._build_page_index(metadata_store)

        if self.config.build_section_index:
            section_index = self._build_specific_index(metadata_store, "section_id")

        if self.config.build_table_index:
            table_index = self._build_specific_index(metadata_store, "table_id")

        if self.config.build_chunk_index:
            chunk_index = self._build_specific_index(metadata_store, "chunk_id")

        if self.config.build_evidence_index:
            evidence_index = self._build_specific_index(metadata_store, "evidence_id")

        if self.config.build_citation_index:
            citation_index = self._build_specific_index(metadata_store, "citation_id")

        if self.config.build_type_index:
            type_index = self._build_type_index(metadata_store)

        if self.config.build_keyword_index:
            keyword_index = self._build_keyword_index(metadata_store)

        if self.config.build_quality_index:
            quality_index = self._build_list_value_index(metadata_store, "quality_flags")

        if self.config.build_processing_hint_index:
            processing_hint_index = self._build_list_value_index(metadata_store, "processing_hints")

        metadata_index = {
            "index_type": "metadata",
            "schema_version": "metadata_index_builder_v1",
            "config": asdict(self.config),
            "item_count": len(metadata_store),
            "metadata_store": metadata_store,
            "field_indexes": field_indexes,
            "page_index": page_index,
            "section_index": section_index,
            "table_index": table_index,
            "chunk_index": chunk_index,
            "evidence_index": evidence_index,
            "citation_index": citation_index,
            "type_index": type_index,
            "keyword_index": keyword_index,
            "quality_index": quality_index,
            "processing_hint_index": processing_hint_index,
        }

        result = {
            "processor": "MetadataIndexBuilder",
            "schema_version": "metadata_index_builder_v1",
            "metadata_index": metadata_index,
            "metadata_store": metadata_store,
            "field_indexes": field_indexes,
            "page_index": page_index,
            "section_index": section_index,
            "table_index": table_index,
            "chunk_index": chunk_index,
            "evidence_index": evidence_index,
            "citation_index": citation_index,
            "type_index": type_index,
            "keyword_index": keyword_index,
            "quality_index": quality_index,
            "processing_hint_index": processing_hint_index,
            "metadata_index_summary": self._build_summary(
                metadata_store=metadata_store,
                field_indexes=field_indexes,
                page_index=page_index,
                type_index=type_index,
                keyword_index=keyword_index,
                quality_index=quality_index,
                processing_hint_index=processing_hint_index,
            ),
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                result=result,
            )

        return json_safe(result)

    def search(
        self,
        index_result: Dict[str, Any],
        filters: Optional[Dict[str, Any]] = None,
        query: str = "",
        item_types: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}
        item_types = item_types or []
        page_numbers = page_numbers or []

        metadata_index = index_result.get("metadata_index", index_result) or {}
        metadata_store = metadata_index.get("metadata_store", {}) or index_result.get("metadata_store", {}) or {}

        query_terms = self._tokenize(query)
        target_pages = set(self._normalize_page_numbers(page_numbers))

        results = []

        for item_id, item in metadata_store.items():
            if item_types and item.get("item_type") not in item_types:
                continue

            if target_pages:
                item_pages = set(self._normalize_page_numbers(item.get("page_numbers", [])))
                if not item_pages.intersection(target_pages):
                    continue

            if not self._passes_filters(item, filters):
                continue

            score = self._score_metadata_item(item, query_terms)

            if query_terms and score <= 0:
                continue

            results.append(
                {
                    "item_id": item_id,
                    "score": round(score, 6),
                    "item_type": item.get("item_type", ""),
                    "source_type": item.get("source_type", ""),
                    "title": item.get("title", ""),
                    "text_preview": item.get("text_preview", ""),
                    "page_numbers": item.get("page_numbers", []),
                    "section_id": item.get("section_id", ""),
                    "section_title": item.get("section_title", ""),
                    "table_id": item.get("table_id", ""),
                    "chunk_id": item.get("chunk_id", ""),
                    "evidence_id": item.get("evidence_id", ""),
                    "citation_id": item.get("citation_id", ""),
                    "quality_flags": item.get("quality_flags", []),
                    "processing_hints": item.get("processing_hints", []),
                    "metadata": item.get("metadata", {}),
                }
            )

        results = sorted(
            results,
            key=lambda item: (
                -item["score"],
                min(item["page_numbers"]) if item["page_numbers"] else 999999,
                item["item_type"],
                item["item_id"],
            ),
        )

        return results[:top_k]

    def facet(
        self,
        index_result: Dict[str, Any],
        field_name: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}

        metadata_index = index_result.get("metadata_index", index_result) or {}
        metadata_store = metadata_index.get("metadata_store", {}) or index_result.get("metadata_store", {}) or {}

        counts: Dict[str, int] = {}

        for item in metadata_store.values():
            if not self._passes_filters(item, filters):
                continue

            values = self._extract_field_values(item, field_name)

            for value in values:
                value_key = self._index_value(value)

                if not value_key and not self.config.index_empty_values:
                    continue

                counts[value_key] = counts.get(value_key, 0) + 1

        ranked = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]

        return [
            {
                "value": value,
                "count": count,
            }
            for value, count in ranked
        ]

    def _collect_metadata_items(
        self,
        page_raws: List[PageRaw],
        metadata_enrichment_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
        chunk_result: Dict[str, Any],
        table_chunk_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        citation_result: Dict[str, Any],
        document_structure_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        graph_index_result: Dict[str, Any],
        bm25_index_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []

        if self.config.include_document_metadata:
            document_metadata = metadata_enrichment_result.get("document_metadata", {}) or {}

            if document_metadata:
                items.append(
                    self._make_metadata_item(
                        item_id="document::root",
                        item_type="document",
                        source_type="document_metadata",
                        title=document_metadata.get("title", "") or document_metadata.get("file_name", "") or "Document",
                        text=document_metadata.get("title", "") or document_metadata.get("source_document", ""),
                        page_numbers=[],
                        metadata=document_metadata,
                    )
                )

        if self.config.include_page_metadata:
            items.extend(
                self._collect_page_metadata_items(
                    page_raws=page_raws,
                    metadata_enrichment_result=metadata_enrichment_result,
                )
            )

        if self.config.include_section_metadata:
            items.extend(
                self._collect_section_metadata_items(
                    metadata_enrichment_result=metadata_enrichment_result,
                    document_structure_result=document_structure_result,
                )
            )

        if self.config.include_table_metadata:
            items.extend(
                self._collect_table_metadata_items(
                    metadata_enrichment_result=metadata_enrichment_result,
                    table_understanding_result=table_understanding_result,
                )
            )

        if self.config.include_chunk_metadata:
            items.extend(
                self._collect_chunk_metadata_items(
                    metadata_enrichment_result=metadata_enrichment_result,
                    chunk_result=chunk_result,
                    table_chunk_result=table_chunk_result,
                    knowledge_result=knowledge_result,
                )
            )

        if self.config.include_evidence_metadata:
            items.extend(
                self._collect_evidence_metadata_items(
                    metadata_enrichment_result=metadata_enrichment_result,
                    evidence_result=evidence_result,
                    knowledge_result=knowledge_result,
                )
            )

        if self.config.include_citation_metadata:
            items.extend(
                self._collect_citation_metadata_items(
                    citation_result=citation_result,
                    evidence_result=evidence_result,
                    knowledge_result=knowledge_result,
                )
            )

        if self.config.include_graph_metadata:
            items.extend(
                self._collect_graph_metadata_items(
                    graph_index_result=graph_index_result,
                    knowledge_result=knowledge_result,
                )
            )

        if self.config.include_bm25_metadata:
            items.extend(
                self._collect_bm25_metadata_items(
                    bm25_index_result=bm25_index_result,
                )
            )

        return items

    def _collect_page_metadata_items(
        self,
        page_raws: List[PageRaw],
        metadata_enrichment_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []
        page_metadata = metadata_enrichment_result.get("page_metadata", {}) or {}

        used_pages = set()

        if isinstance(page_metadata, dict):
            for page_key, metadata in page_metadata.items():
                metadata = self._to_dict(metadata)
                page_number = self._safe_int(metadata.get("page_number") or page_key, default=0)

                if page_number <= 0:
                    continue

                used_pages.add(page_number)

                text = metadata.get("text_preview", "")

                items.append(
                    self._make_metadata_item(
                        item_id=f"page::{page_number}",
                        item_type="page",
                        source_type="page_metadata",
                        title=f"Trang {page_number}",
                        text=text,
                        page_numbers=[page_number],
                        metadata=metadata,
                    )
                )

        for page_raw in page_raws:
            if page_raw.page_number in used_pages:
                continue

            text = self._page_text(page_raw)

            metadata = {
                "page_number": page_raw.page_number,
                "page_index": page_raw.page_index,
                "document_id": page_raw.document_id,
                "source_document": page_raw.source_document,
                "width": page_raw.width,
                "height": page_raw.height,
                "rotation": page_raw.rotation,
                "page_kind": page_raw.page_kind,
                "text_preview": self._preview(text, self.config.max_text_preview_chars),
                "word_count": page_raw.word_count,
                "image_count": page_raw.image_count,
                "drawing_count": page_raw.drawing_count,
                "annotation_count": page_raw.annotation_count,
                "link_count": page_raw.link_count,
                "quality_flags": self._infer_page_quality_flags(page_raw, text),
            }

            items.append(
                self._make_metadata_item(
                    item_id=f"page::{page_raw.page_number}",
                    item_type="page",
                    source_type="page_raw",
                    title=f"Trang {page_raw.page_number}",
                    text=text,
                    page_numbers=[page_raw.page_number],
                    metadata=metadata,
                )
            )

        return items

    def _collect_section_metadata_items(
        self,
        metadata_enrichment_result: Dict[str, Any],
        document_structure_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []
        section_metadata = metadata_enrichment_result.get("section_metadata", {}) or {}

        used_sections = set()

        if isinstance(section_metadata, dict):
            for section_id, metadata in section_metadata.items():
                metadata = self._to_dict(metadata)
                section_id = metadata.get("section_id") or section_id

                if not section_id:
                    continue

                used_sections.add(section_id)

                page_numbers = self._resolve_page_numbers(metadata)

                items.append(
                    self._make_metadata_item(
                        item_id=f"section::{section_id}",
                        item_type="section",
                        source_type="section_metadata",
                        title=metadata.get("title", "") or section_id,
                        text=metadata.get("text_preview", "") or metadata.get("title", ""),
                        page_numbers=page_numbers,
                        section_id=section_id,
                        section_title=metadata.get("title", ""),
                        metadata=metadata,
                    )
                )

        sections = document_structure_result.get("sections", []) or []

        if isinstance(sections, list):
            for section in sections:
                section = self._to_dict(section)
                section_id = section.get("section_id") or section.get("id") or ""

                if not section_id or section_id in used_sections:
                    continue

                title = normalize_pdf_text(section.get("title") or section.get("heading") or section_id)
                page_numbers = self._resolve_page_numbers(section)

                items.append(
                    self._make_metadata_item(
                        item_id=f"section::{section_id}",
                        item_type="section",
                        source_type="document_structure",
                        title=title,
                        text=section.get("text_preview", "") or title,
                        page_numbers=page_numbers,
                        section_id=section_id,
                        section_title=title,
                        metadata=section,
                    )
                )

        return items

    def _collect_table_metadata_items(
        self,
        metadata_enrichment_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []
        table_metadata = metadata_enrichment_result.get("table_metadata", {}) or {}

        used_tables = set()

        if isinstance(table_metadata, dict):
            for table_id, metadata in table_metadata.items():
                metadata = self._to_dict(metadata)
                table_id = metadata.get("table_id") or table_id

                if not table_id:
                    continue

                used_tables.add(table_id)
                page_numbers = self._resolve_page_numbers(metadata)

                title = (
                    metadata.get("title")
                    or metadata.get("caption")
                    or table_id
                )

                items.append(
                    self._make_metadata_item(
                        item_id=f"table::{table_id}",
                        item_type="table",
                        source_type=metadata.get("table_kind", "table_metadata"),
                        title=title,
                        text=metadata.get("text_preview", "") or title,
                        page_numbers=page_numbers,
                        section_id=metadata.get("section_id", ""),
                        section_title=metadata.get("section_title", ""),
                        table_id=table_id,
                        metadata=metadata,
                    )
                )

        tables = self._collect_tables(table_understanding_result)

        for table in tables:
            table_id = self._table_id(table)

            if not table_id or table_id in used_tables:
                continue

            title = (
                normalize_pdf_text(table.get("title"))
                or normalize_pdf_text(table.get("caption"))
                or normalize_pdf_text(table.get("caption_text"))
                or table_id
            )

            page_numbers = self._resolve_page_numbers(table)

            if not page_numbers and table.get("page_number"):
                page_number = self._safe_int(table.get("page_number"), default=0)
                if page_number > 0:
                    page_numbers = [page_number]

            items.append(
                self._make_metadata_item(
                    item_id=f"table::{table_id}",
                    item_type="table",
                    source_type=self._table_kind(table),
                    title=title,
                    text=self._table_text(table),
                    page_numbers=page_numbers,
                    section_id=table.get("section_id", ""),
                    section_title=table.get("section_title", ""),
                    table_id=table_id,
                    metadata=table,
                )
            )

        return items

    def _collect_chunk_metadata_items(
        self,
        metadata_enrichment_result: Dict[str, Any],
        chunk_result: Dict[str, Any],
        table_chunk_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []
        chunk_metadata = metadata_enrichment_result.get("chunk_metadata", {}) or {}

        used_chunks = set()

        if isinstance(chunk_metadata, dict):
            for chunk_id, metadata in chunk_metadata.items():
                metadata = self._to_dict(metadata)
                chunk_id = metadata.get("chunk_id") or chunk_id

                if not chunk_id:
                    continue

                used_chunks.add(chunk_id)

                page_numbers = self._resolve_page_numbers(metadata)

                items.append(
                    self._make_metadata_item(
                        item_id=f"chunk::{chunk_id}",
                        item_type="chunk",
                        source_type=metadata.get("chunk_type", "chunk_metadata"),
                        title=metadata.get("section_title", "") or metadata.get("chunk_type", "") or chunk_id,
                        text=metadata.get("text_preview", ""),
                        page_numbers=page_numbers,
                        section_id=metadata.get("section_id", ""),
                        section_title=metadata.get("section_title", ""),
                        table_id=self._table_id(metadata),
                        chunk_id=chunk_id,
                        metadata=metadata,
                    )
                )

        chunks = []
        chunks.extend(self._collect_chunks(chunk_result))
        chunks.extend(self._collect_chunks(table_chunk_result))
        chunks.extend(self._collect_chunks(knowledge_result))

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")

            if not chunk_id or chunk_id in used_chunks:
                continue

            page_numbers = self._normalize_page_numbers(chunk.get("page_numbers", []))
            text = normalize_pdf_text(chunk.get("text", ""))

            items.append(
                self._make_metadata_item(
                    item_id=f"chunk::{chunk_id}",
                    item_type="chunk",
                    source_type=chunk.get("chunk_type", "chunk"),
                    title=chunk.get("section_title", "") or chunk.get("chunk_type", "") or chunk_id,
                    text=text,
                    page_numbers=page_numbers,
                    section_id=chunk.get("section_id", ""),
                    section_title=chunk.get("section_title", ""),
                    table_id=self._table_id(chunk),
                    chunk_id=chunk_id,
                    metadata=chunk,
                )
            )

        return items

    def _collect_evidence_metadata_items(
        self,
        metadata_enrichment_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []
        evidence_metadata = metadata_enrichment_result.get("evidence_metadata", {}) or {}

        used_evidence = set()

        if isinstance(evidence_metadata, dict):
            for evidence_id, metadata in evidence_metadata.items():
                metadata = self._to_dict(metadata)
                evidence_id = metadata.get("evidence_id") or evidence_id

                if not evidence_id:
                    continue

                used_evidence.add(evidence_id)

                page_numbers = self._resolve_page_numbers(metadata)

                items.append(
                    self._make_metadata_item(
                        item_id=f"evidence::{evidence_id}",
                        item_type="evidence",
                        source_type=metadata.get("evidence_type", "evidence_metadata"),
                        title=metadata.get("section_title", "") or metadata.get("evidence_type", "") or evidence_id,
                        text=metadata.get("text_preview", "") or metadata.get("quote_preview", ""),
                        page_numbers=page_numbers,
                        section_id=metadata.get("section_id", ""),
                        section_title=metadata.get("section_title", ""),
                        table_id=self._table_id(metadata),
                        chunk_id=metadata.get("chunk_id", ""),
                        evidence_id=evidence_id,
                        metadata=metadata,
                    )
                )

        evidence_items = []
        evidence_items.extend(self._collect_evidence(evidence_result))
        evidence_items.extend(self._collect_evidence(knowledge_result))

        for evidence in evidence_items:
            evidence_id = evidence.get("evidence_id", "")

            if not evidence_id or evidence_id in used_evidence:
                continue

            text = normalize_pdf_text(evidence.get("text") or evidence.get("quote") or "")
            page_numbers = self._resolve_page_numbers(evidence)

            items.append(
                self._make_metadata_item(
                    item_id=f"evidence::{evidence_id}",
                    item_type="evidence",
                    source_type=evidence.get("evidence_type", "evidence"),
                    title=evidence.get("section_title", "") or evidence.get("evidence_type", "") or evidence_id,
                    text=text,
                    page_numbers=page_numbers,
                    section_id=evidence.get("section_id", ""),
                    section_title=evidence.get("section_title", ""),
                    table_id=self._table_id(evidence),
                    chunk_id=evidence.get("chunk_id") or evidence.get("source_chunk_id", ""),
                    evidence_id=evidence_id,
                    metadata=evidence,
                )
            )

        return items

    def _collect_citation_metadata_items(
        self,
        citation_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        citations = []
        citations.extend(self._collect_citations(citation_result))
        citations.extend(self._collect_citations(evidence_result))
        citations.extend(self._collect_citations(knowledge_result))

        citations = self._deduplicate_dicts(citations, ["citation_id"])

        for citation in citations:
            citation_id = citation.get("citation_id", "")

            if not citation_id:
                continue

            text = normalize_pdf_text(
                citation.get("citation_text")
                or citation.get("quote")
                or citation.get("label")
                or ""
            )

            page_numbers = self._resolve_page_numbers(citation)

            items.append(
                self._make_metadata_item(
                    item_id=f"citation::{citation_id}",
                    item_type="citation",
                    source_type=citation.get("citation_type", "citation"),
                    title=citation.get("citation_marker", "") or citation_id,
                    text=text,
                    page_numbers=page_numbers,
                    section_id=citation.get("section_id", ""),
                    section_title=citation.get("section_title", ""),
                    chunk_id=citation.get("chunk_id", ""),
                    evidence_id=citation.get("evidence_id", ""),
                    citation_id=citation_id,
                    metadata=citation,
                )
            )

        return items

    def _collect_graph_metadata_items(
        self,
        graph_index_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        graph_sources = [graph_index_result, knowledge_result]

        for source in graph_sources:
            graph_index = source.get("graph_index", {}) or source.get("knowledge_graph", {}) or source.get("context_graph", {}) or {}

            node_store = graph_index.get("node_store", {}) or source.get("node_store", {}) or {}

            if isinstance(node_store, dict):
                for node_id, node in node_store.items():
                    node = self._to_dict(node)
                    node.setdefault("node_id", node_id)

                    page_numbers = self._resolve_page_numbers(node)

                    items.append(
                        self._make_metadata_item(
                            item_id=f"graph_node::{node.get('node_id', node_id)}",
                            item_type="graph_node",
                            source_type=node.get("node_type", "graph_node"),
                            title=node.get("label", "") or node.get("node_id", node_id),
                            text=node.get("text", "") or node.get("label", ""),
                            page_numbers=page_numbers,
                            section_id=node.get("section_id", "") or node.get("metadata", {}).get("section_id", ""),
                            table_id=node.get("table_id", "") or node.get("metadata", {}).get("table_id", ""),
                            chunk_id=node.get("chunk_id", "") or node.get("metadata", {}).get("chunk_id", ""),
                            evidence_id=node.get("evidence_id", "") or node.get("metadata", {}).get("evidence_id", ""),
                            metadata=node,
                        )
                    )

            nodes = graph_index.get("nodes", []) or source.get("nodes", []) or []

            if isinstance(nodes, list):
                for node in nodes:
                    node = self._to_dict(node)
                    node_id = node.get("node_id", "")

                    if not node_id:
                        continue

                    page_numbers = self._resolve_page_numbers(node)

                    items.append(
                        self._make_metadata_item(
                            item_id=f"graph_node::{node_id}",
                            item_type="graph_node",
                            source_type=node.get("node_type", "graph_node"),
                            title=node.get("label", "") or node_id,
                            text=node.get("text", "") or node.get("label", ""),
                            page_numbers=page_numbers,
                            section_id=node.get("metadata", {}).get("section_id", ""),
                            table_id=node.get("metadata", {}).get("table_id", ""),
                            chunk_id=node.get("metadata", {}).get("chunk_id", ""),
                            evidence_id=node.get("metadata", {}).get("evidence_id", ""),
                            metadata=node,
                        )
                    )

        return self._deduplicate_items(items)

    def _collect_bm25_metadata_items(
        self,
        bm25_index_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        bm25_index = bm25_index_result.get("bm25_index", bm25_index_result) or {}
        document_store = bm25_index.get("document_store", {}) or bm25_index_result.get("document_store", {}) or {}

        if not isinstance(document_store, dict):
            return items

        for doc_id, doc in document_store.items():
            doc = self._to_dict(doc)
            doc.setdefault("document_id", doc_id)

            items.append(
                self._make_metadata_item(
                    item_id=f"bm25_doc::{doc_id}",
                    item_type="bm25_document",
                    source_type=doc.get("source_type", "bm25_document"),
                    title=doc.get("title", "") or doc_id,
                    text=doc.get("text_preview", "") or doc.get("text", ""),
                    page_numbers=self._resolve_page_numbers(doc),
                    section_id=doc.get("section_id", ""),
                    section_title=doc.get("section_title", ""),
                    table_id=doc.get("table_id", ""),
                    chunk_id=doc.get("chunk_id", ""),
                    evidence_id=doc.get("evidence_id", ""),
                    metadata=doc,
                )
            )

        return items

    def _make_metadata_item(
        self,
        item_id: str,
        item_type: str,
        source_type: str,
        title: str = "",
        text: str = "",
        page_numbers: Optional[List[int]] = None,
        section_id: str = "",
        section_title: str = "",
        table_id: str = "",
        chunk_id: str = "",
        evidence_id: str = "",
        citation_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        metadata = metadata or {}
        page_numbers = self._normalize_page_numbers(page_numbers or [])

        text = normalize_pdf_text(text)
        title = normalize_pdf_text(title)

        if not item_id:
            item_id = self._stable_id(
                f"{item_type}|{source_type}|{title}|{text}|{page_numbers}",
                "metadata",
            )

        quality_flags = self._extract_list_field(metadata, "quality_flags")
        processing_hints = self._extract_list_field(metadata, "processing_hints")
        keywords = self._extract_keywords_from_metadata(metadata, text)

        flat_metadata = {}

        if self.config.flatten_nested_metadata:
            flat_metadata = self._flatten_dict(
                metadata,
                max_depth=self.config.max_flatten_depth,
            )

        return {
            "item_id": item_id,
            "item_type": item_type,
            "source_type": source_type,
            "title": title,
            "text_preview": self._preview(text or title, self.config.max_text_preview_chars),
            "normalized_text": normalize_text_for_match(f"{title}\n{text}"),
            "page_numbers": page_numbers,
            "page_start": min(page_numbers) if page_numbers else None,
            "page_end": max(page_numbers) if page_numbers else None,
            "section_id": section_id or metadata.get("section_id", ""),
            "section_title": section_title or metadata.get("section_title", ""),
            "table_id": table_id or self._table_id(metadata),
            "chunk_id": chunk_id or metadata.get("chunk_id", ""),
            "evidence_id": evidence_id or metadata.get("evidence_id", ""),
            "citation_id": citation_id or metadata.get("citation_id", ""),
            "quality_flags": quality_flags,
            "processing_hints": processing_hints,
            "keywords": keywords[: self.config.max_keyword_count_per_item],
            "content_hash": metadata.get("content_hash", "") or self._stable_hash(
                {
                    "item_type": item_type,
                    "source_type": source_type,
                    "title": title,
                    "text": text,
                    "page_numbers": page_numbers,
                }
            ),
            "metadata": metadata,
            "flat_metadata": flat_metadata,
        }

    def _build_metadata_store(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        store: Dict[str, Dict[str, Any]] = {}

        for item in items:
            item_id = item.get("item_id", "")

            if not item_id:
                continue

            store[item_id] = item

        return store

    def _build_field_indexes(
        self,
        metadata_store: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, List[str]]]:
        field_indexes: Dict[str, Dict[str, List[str]]] = {}

        default_fields = [
            "item_type",
            "source_type",
            "section_id",
            "section_title",
            "table_id",
            "chunk_id",
            "evidence_id",
            "citation_id",
            "page_start",
            "page_end",
        ]

        for item_id, item in metadata_store.items():
            for field_name in default_fields:
                values = self._extract_field_values(item, field_name)

                for value in values:
                    value_key = self._index_value(value)

                    if not value_key and not self.config.index_empty_values:
                        continue

                    field_indexes.setdefault(field_name, {})
                    field_indexes[field_name].setdefault(value_key, [])
                    field_indexes[field_name][value_key].append(item_id)

            flat_metadata = item.get("flat_metadata", {}) or {}

            if isinstance(flat_metadata, dict):
                for field_name, value in flat_metadata.items():
                    values = self._as_list(value)

                    for field_value in values:
                        value_key = self._index_value(field_value)

                        if not value_key and not self.config.index_empty_values:
                            continue

                        index_field_name = f"metadata.{field_name}"
                        field_indexes.setdefault(index_field_name, {})
                        field_indexes[index_field_name].setdefault(value_key, [])
                        field_indexes[index_field_name][value_key].append(item_id)

        return field_indexes

    def _build_page_index(
        self,
        metadata_store: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        page_index: Dict[str, List[str]] = {}

        for item_id, item in metadata_store.items():
            page_numbers = self._normalize_page_numbers(item.get("page_numbers", []))

            for page_number in page_numbers:
                page_key = str(page_number)
                page_index.setdefault(page_key, [])
                page_index[page_key].append(item_id)

        return page_index

    def _build_specific_index(
        self,
        metadata_store: Dict[str, Dict[str, Any]],
        field_name: str,
    ) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = {}

        for item_id, item in metadata_store.items():
            values = self._extract_field_values(item, field_name)

            for value in values:
                value_key = self._index_value(value)

                if not value_key and not self.config.index_empty_values:
                    continue

                index.setdefault(value_key, [])
                index[value_key].append(item_id)

        return index

    def _build_type_index(
        self,
        metadata_store: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        return self._build_specific_index(metadata_store, "item_type")

    def _build_keyword_index(
        self,
        metadata_store: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        keyword_index: Dict[str, List[Dict[str, Any]]] = {}

        for item_id, item in metadata_store.items():
            keywords = item.get("keywords", []) or []

            for keyword_item in keywords:
                if isinstance(keyword_item, dict):
                    keyword = keyword_item.get("keyword", "")
                    count = self._safe_int(keyword_item.get("count"), default=1)
                else:
                    keyword = str(keyword_item)
                    count = 1

                keyword = normalize_text_for_match(keyword)

                if not keyword:
                    continue

                keyword_index.setdefault(keyword, [])
                keyword_index[keyword].append(
                    {
                        "item_id": item_id,
                        "count": count,
                        "item_type": item.get("item_type", ""),
                        "page_numbers": item.get("page_numbers", []),
                    }
                )

        return keyword_index

    def _build_list_value_index(
        self,
        metadata_store: Dict[str, Dict[str, Any]],
        field_name: str,
    ) -> Dict[str, List[str]]:
        index: Dict[str, List[str]] = {}

        for item_id, item in metadata_store.items():
            values = item.get(field_name, []) or []

            if not isinstance(values, list):
                values = [values]

            for value in values:
                value_key = self._index_value(value)

                if not value_key and not self.config.index_empty_values:
                    continue

                index.setdefault(value_key, [])
                index[value_key].append(item_id)

        return index

    def _passes_filters(
        self,
        item: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:
        if not filters:
            return True

        for field_name, expected in filters.items():
            if expected is None:
                continue

            actual_values = self._extract_field_values(item, field_name)

            if not isinstance(expected, list):
                expected_values = [expected]
            else:
                expected_values = expected

            actual_set = set(self._index_value(value) for value in actual_values)
            expected_set = set(self._index_value(value) for value in expected_values)

            if not actual_set.intersection(expected_set):
                return False

        return True

    def _score_metadata_item(
        self,
        item: Dict[str, Any],
        query_terms: List[str],
    ) -> float:
        if not query_terms:
            return 1.0

        searchable_text = normalize_text_for_match(
            "\n".join(
                [
                    item.get("title", ""),
                    item.get("text_preview", ""),
                    item.get("section_title", ""),
                    item.get("item_type", ""),
                    item.get("source_type", ""),
                    " ".join(
                        [
                            kw.get("keyword", "") if isinstance(kw, dict) else str(kw)
                            for kw in item.get("keywords", []) or []
                        ]
                    ),
                ]
            )
        )

        if not searchable_text:
            return 0.0

        score = 0.0

        for term in query_terms:
            if term in searchable_text:
                score += 1.0 + searchable_text.count(term) * 0.15

        if item.get("item_type") in ["section", "table", "chunk", "evidence"]:
            score *= 1.1

        if item.get("quality_flags"):
            score *= 0.98

        return score

    def _extract_field_values(
        self,
        item: Dict[str, Any],
        field_name: str,
    ) -> List[Any]:
        if field_name in item:
            return self._as_list(item.get(field_name))

        if field_name.startswith("metadata."):
            path = field_name.replace("metadata.", "", 1)
            value = self._get_nested_value(item.get("metadata", {}) or {}, path)
            return self._as_list(value)

        if field_name.startswith("flat_metadata."):
            path = field_name.replace("flat_metadata.", "", 1)
            value = item.get("flat_metadata", {}).get(path)
            return self._as_list(value)

        metadata = item.get("metadata", {}) or {}

        if isinstance(metadata, dict) and field_name in metadata:
            return self._as_list(metadata.get(field_name))

        flat_metadata = item.get("flat_metadata", {}) or {}

        if isinstance(flat_metadata, dict) and field_name in flat_metadata:
            return self._as_list(flat_metadata.get(field_name))

        return []

    def _extract_list_field(
        self,
        metadata: Dict[str, Any],
        field_name: str,
    ) -> List[str]:
        values = metadata.get(field_name, []) or []

        if not isinstance(values, list):
            values = [values]

        result = []

        for value in values:
            value = normalize_pdf_text(value)

            if value and value not in result:
                result.append(value)

        return result

    def _extract_keywords_from_metadata(
        self,
        metadata: Dict[str, Any],
        text: str = "",
    ) -> List[Dict[str, Any]]:
        keywords = metadata.get("keywords", []) or []

        result = []

        if isinstance(keywords, list):
            for item in keywords:
                if isinstance(item, dict):
                    keyword = normalize_text_for_match(item.get("keyword", ""))
                    count = self._safe_int(item.get("count"), default=1)
                else:
                    keyword = normalize_text_for_match(item)
                    count = 1

                if keyword:
                    result.append(
                        {
                            "keyword": keyword,
                            "count": count,
                        }
                    )

        if result:
            return result[: self.config.max_keyword_count_per_item]

        return self._extract_keywords_from_text(text)

    def _extract_keywords_from_text(
        self,
        text: str,
        max_keywords: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        max_keywords = max_keywords or self.config.max_keyword_count_per_item

        text = normalize_text_for_match(text)

        if not text:
            return []

        stopwords = self._default_stopwords()
        tokens = [
            token for token in re.findall(r"[a-z0-9_]{3,}", text)
            if token not in stopwords and not token.isdigit()
        ]

        counts: Dict[str, int] = {}

        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        ranked = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:max_keywords]

        return [
            {
                "keyword": keyword,
                "count": count,
            }
            for keyword, count in ranked
        ]

    def _flatten_dict(
        self,
        data: Dict[str, Any],
        parent_key: str = "",
        depth: int = 0,
        max_depth: int = 3,
    ) -> Dict[str, Any]:
        if depth > max_depth:
            return {}

        flat: Dict[str, Any] = {}

        if not isinstance(data, dict):
            return flat

        for key, value in data.items():
            key = str(key)
            full_key = f"{parent_key}.{key}" if parent_key else key

            if isinstance(value, dict) and depth < max_depth:
                flat.update(
                    self._flatten_dict(
                        value,
                        parent_key=full_key,
                        depth=depth + 1,
                        max_depth=max_depth,
                    )
                )
            elif isinstance(value, list):
                if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
                    flat[full_key] = value
                else:
                    flat[full_key] = len(value)
            else:
                flat[full_key] = value

        return flat

    def _get_nested_value(
        self,
        data: Dict[str, Any],
        path: str,
    ) -> Any:
        current = data

        for part in path.split("."):
            if not isinstance(current, dict):
                return None

            if part not in current:
                return None

            current = current[part]

        return current

    def _as_list(
        self,
        value: Any,
    ) -> List[Any]:
        if value is None:
            return []

        if isinstance(value, list):
            return value

        return [value]

    def _index_value(
        self,
        value: Any,
    ) -> str:
        if value is None:
            return ""

        if isinstance(value, bool):
            return str(value).lower()

        if isinstance(value, (int, float)):
            return str(value)

        value = normalize_pdf_text(value)

        if self.config.normalize_index_values:
            return normalize_text_for_match(value)

        return value

    def _build_summary(
        self,
        metadata_store: Dict[str, Dict[str, Any]],
        field_indexes: Dict[str, Dict[str, List[str]]],
        page_index: Dict[str, List[str]],
        type_index: Dict[str, List[str]],
        keyword_index: Dict[str, List[Dict[str, Any]]],
        quality_index: Dict[str, List[str]],
        processing_hint_index: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        by_item_type = {
            key: len(value)
            for key, value in type_index.items()
        }

        by_page = {
            key: len(value)
            for key, value in page_index.items()
        }

        top_keywords = sorted(
            [
                {
                    "keyword": keyword,
                    "item_count": len(postings),
                    "total_count": sum(self._safe_int(item.get("count"), default=1) for item in postings),
                }
                for keyword, postings in keyword_index.items()
            ],
            key=lambda item: (-item["item_count"], -item["total_count"], item["keyword"]),
        )[:50]

        return {
            "has_metadata_index": len(metadata_store) > 0,
            "index_type": "metadata",
            "item_count": len(metadata_store),
            "field_index_count": len(field_indexes),
            "page_count_with_items": len(page_index),
            "item_type_count": len(type_index),
            "keyword_count": len(keyword_index),
            "quality_flag_count": len(quality_index),
            "processing_hint_count": len(processing_hint_index),
            "by_item_type": by_item_type,
            "by_page": by_page,
            "top_keywords": top_keywords,
            "quality_flags": {
                key: len(value)
                for key, value in quality_index.items()
            },
            "processing_hints": {
                key: len(value)
                for key, value in processing_hint_index.items()
            },
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        page_index = result.get("page_index", {}) or {}
        summary = result.get("metadata_index_summary", {}) or {}

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("metadata_index_builder", {})
            page_raw.metadata["metadata_index_builder"] = {
                "processor": "MetadataIndexBuilder",
                "metadata_item_ids_on_page": page_index.get(page_key, []),
                "metadata_item_count_on_page": len(page_index.get(page_key, [])),
                "index_summary": {
                    "item_count": summary.get("item_count", 0),
                    "field_index_count": summary.get("field_index_count", 0),
                    "keyword_count": summary.get("keyword_count", 0),
                },
            }

    def _deduplicate_items(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result_by_id: Dict[str, Dict[str, Any]] = {}

        for item in items:
            item_id = item.get("item_id", "")

            if not item_id:
                item_id = self._stable_id(
                    {
                        "item_type": item.get("item_type", ""),
                        "source_type": item.get("source_type", ""),
                        "title": item.get("title", ""),
                        "pages": item.get("page_numbers", []),
                        "text": item.get("text_preview", ""),
                    },
                    "metadata",
                )
                item["item_id"] = item_id

            if item_id not in result_by_id:
                result_by_id[item_id] = item
            else:
                result_by_id[item_id] = self._merge_items(result_by_id[item_id], item)

        return list(result_by_id.values())

    def _merge_items(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(existing)

        if len(incoming.get("text_preview", "")) > len(existing.get("text_preview", "")):
            merged["text_preview"] = incoming.get("text_preview", "")

        if len(incoming.get("title", "")) > len(existing.get("title", "")):
            merged["title"] = incoming.get("title", "")

        pages = self._normalize_page_numbers(existing.get("page_numbers", [])) + self._normalize_page_numbers(incoming.get("page_numbers", []))
        merged["page_numbers"] = sorted(list(dict.fromkeys(pages)))

        for list_field in ["quality_flags", "processing_hints", "keywords"]:
            merged[list_field] = self._merge_list_values(
                existing.get(list_field, []),
                incoming.get(list_field, []),
            )

        existing_metadata = existing.get("metadata", {}) or {}
        incoming_metadata = incoming.get("metadata", {}) or {}

        if isinstance(existing_metadata, dict) and isinstance(incoming_metadata, dict):
            merged["metadata"] = {
                **incoming_metadata,
                **existing_metadata,
            }

        existing_flat = existing.get("flat_metadata", {}) or {}
        incoming_flat = incoming.get("flat_metadata", {}) or {}

        if isinstance(existing_flat, dict) and isinstance(incoming_flat, dict):
            merged["flat_metadata"] = {
                **incoming_flat,
                **existing_flat,
            }

        return merged

    def _merge_list_values(
        self,
        a: Any,
        b: Any,
    ) -> List[Any]:
        values = []

        for source in [a, b]:
            if not isinstance(source, list):
                source = [source]

            for item in source:
                key = json.dumps(json_safe(item), ensure_ascii=False, sort_keys=True)

                if key not in [
                    json.dumps(json_safe(existing), ensure_ascii=False, sort_keys=True)
                    for existing in values
                ]:
                    values.append(item)

        return values

    def _collect_chunks(
        self,
        source: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        chunks = []

        for key in [
            "chunks",
            "parent_chunks",
            "child_chunks",
            "table_chunks",
            "table_summary_chunks",
            "table_record_chunks",
            "table_row_chunks",
            "multi_page_table_chunks",
        ]:
            values = source.get(key, []) or []

            if isinstance(values, list):
                chunks.extend([self._to_dict(item) for item in values])

        for sub_key in [
            "chunk_result",
            "chunk_collection",
            "parent_child_chunk_result",
            "table_chunk_result",
        ]:
            sub = source.get(sub_key, {}) or {}

            if not isinstance(sub, dict):
                continue

            for key in [
                "chunks",
                "parent_chunks",
                "child_chunks",
                "table_chunks",
                "table_summary_chunks",
                "table_record_chunks",
                "table_row_chunks",
                "multi_page_table_chunks",
            ]:
                values = sub.get(key, []) or []

                if isinstance(values, list):
                    chunks.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(chunks, ["chunk_id", "content_hash"])

    def _collect_evidence(
        self,
        source: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        evidence = []

        for key in ["evidence", "evidence_items", "retrieved_evidence", "supporting_evidence"]:
            values = source.get(key, []) or []

            if isinstance(values, list):
                evidence.extend([self._to_dict(item) for item in values])

        for sub_key in ["evidence_result", "evidence_collection"]:
            sub = source.get(sub_key, {}) or {}

            if not isinstance(sub, dict):
                continue

            values = sub.get("evidence", []) or sub.get("evidence_items", []) or []

            if isinstance(values, list):
                evidence.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(evidence, ["evidence_id", "content_hash"])

    def _collect_citations(
        self,
        source: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        citations = []

        for key in ["citations", "citation_items"]:
            values = source.get(key, []) or []

            if isinstance(values, list):
                citations.extend([self._to_dict(item) for item in values])

        for sub_key in ["citation_result", "evidence_collection"]:
            sub = source.get(sub_key, {}) or {}

            if not isinstance(sub, dict):
                continue

            values = sub.get("citations", []) or []

            if isinstance(values, list):
                citations.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(citations, ["citation_id"])

    def _collect_tables(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tables = []

        for key in [
            "table_semantics",
            "table_grids",
            "table_structures",
            "table_boundaries",
            "multi_page_tables",
        ]:
            values = table_understanding_result.get(key, []) or []

            if isinstance(values, list):
                tables.extend([self._to_dict(item) for item in values])

        for sub_key in [
            "table_semantic_result",
            "table_grid_result",
            "table_structure_result",
            "table_boundary_result",
            "multi_page_table_result",
        ]:
            sub = table_understanding_result.get(sub_key, {}) or {}

            if not isinstance(sub, dict):
                continue

            for key in [
                "table_semantics",
                "table_grids",
                "table_structures",
                "table_boundaries",
                "multi_page_tables",
            ]:
                values = sub.get(key, []) or []

                if isinstance(values, list):
                    tables.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(
            tables,
            [
                "table_semantic_id",
                "table_grid_id",
                "table_structure_id",
                "table_boundary_id",
                "multi_page_table_id",
            ],
        )

    def _table_id(
        self,
        item: Dict[str, Any],
    ) -> str:
        metadata = item.get("metadata", {}) or {}

        if not isinstance(metadata, dict):
            metadata = {}

        return (
            item.get("table_id")
            or item.get("table_semantic_id")
            or item.get("table_grid_id")
            or item.get("table_structure_id")
            or item.get("table_boundary_id")
            or item.get("multi_page_table_id")
            or metadata.get("table_id", "")
            or metadata.get("multi_page_table_id", "")
            or ""
        )

    def _table_kind(
        self,
        table: Dict[str, Any],
    ) -> str:
        if table.get("multi_page_table_id"):
            return "multi_page_table"

        if table.get("table_semantic_id"):
            return "table_semantic"

        if table.get("table_grid_id"):
            return "table_grid"

        if table.get("table_structure_id"):
            return "table_structure"

        if table.get("table_boundary_id"):
            return "table_boundary"

        return "table"

    def _table_text(
        self,
        table: Dict[str, Any],
    ) -> str:
        parts = []

        for key in ["title", "caption", "caption_text"]:
            value = normalize_pdf_text(table.get(key, ""))

            if value:
                parts.append(value)

        headers = table.get("column_headers", []) or []

        if headers:
            header_text = " | ".join(
                [
                    normalize_pdf_text(item)
                    for item in headers
                    if normalize_pdf_text(item)
                ]
            )

            if header_text:
                parts.append(header_text)

        if table.get("text"):
            parts.append(normalize_pdf_text(table.get("text", "")))

        records = table.get("records", []) or []

        for record in records[:8]:
            if not isinstance(record, dict):
                continue

            values = record.get("values", {}) or record.get("raw_values", {}) or {}

            if isinstance(values, dict):
                line = " | ".join(
                    [
                        f"{key}: {normalize_pdf_text(value)}"
                        for key, value in values.items()
                        if normalize_pdf_text(value)
                    ]
                )

                if line:
                    parts.append(line)

        return normalize_pdf_text("\n".join(parts))

    def _page_text(
        self,
        page_raw: PageRaw,
    ) -> str:
        reading_meta = page_raw.metadata.get("reading_order_builder", {}) or {}
        reading_text = normalize_pdf_text(reading_meta.get("reading_order_text", ""))

        if reading_text:
            return reading_text

        if page_raw.normalized_text:
            return page_raw.normalized_text

        if page_raw.raw_text:
            return page_raw.raw_text

        if page_raw.text_blocks:
            return normalize_pdf_text(
                "\n\n".join(
                    [
                        block.text
                        for block in page_raw.text_blocks
                        if getattr(block, "text", "")
                    ]
                )
            )

        if page_raw.text_lines:
            return normalize_pdf_text(
                "\n".join(
                    [
                        line.text
                        for line in page_raw.text_lines
                        if getattr(line, "text", "")
                    ]
                )
            )

        return ""

    def _infer_page_quality_flags(
        self,
        page_raw: PageRaw,
        text: str,
    ) -> List[str]:
        flags = []

        if getattr(page_raw, "is_blank", False):
            flags.append("blank_page")

        if not text and page_raw.image_count > 0:
            flags.append("possible_scanned_page")

        if len(text) < 50 and not getattr(page_raw, "is_blank", False):
            flags.append("low_text")

        if page_raw.image_count > 0:
            flags.append("has_images")

        if page_raw.drawing_count > 0:
            flags.append("has_drawings")

        if page_raw.annotation_count > 0:
            flags.append("has_annotations")

        if page_raw.link_count > 0:
            flags.append("has_links")

        return sorted(list(dict.fromkeys(flags)))

    def _resolve_page_numbers(
        self,
        item: Dict[str, Any],
    ) -> List[int]:
        page_numbers = (
            item.get("page_numbers")
            or item.get("content_page_numbers")
            or []
        )

        if page_numbers:
            return self._normalize_page_numbers(page_numbers)

        page_start = item.get("page_start")
        page_end = item.get("page_end")

        if page_start is not None and page_end is not None:
            page_start = self._safe_int(page_start, default=0)
            page_end = self._safe_int(page_end, default=0)

            if page_start > 0 and page_end >= page_start:
                return list(range(page_start, page_end + 1))

        page_number = self._safe_int(item.get("page_number"), default=0)

        if page_number > 0:
            return [page_number]

        return []

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

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:
        text = normalize_text_for_match(text)

        tokens = [
            token for token in re.findall(r"[a-z0-9_]+", text)
            if len(token) >= 2
        ]

        return tokens

    def _deduplicate_dicts(
        self,
        items: List[Dict[str, Any]],
        keys: List[str],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for item in items:
            item = self._to_dict(item)
            key = ""

            for key_name in keys:
                value = item.get(key_name)
                if value:
                    key = str(value)
                    break

            if not key:
                key = self._stable_hash(
                    {
                        "text": item.get("text") or item.get("title") or item.get("label") or "",
                        "pages": self._resolve_page_numbers(item),
                    }
                )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _preview(
        self,
        text: Any,
        max_chars: int = 500,
    ) -> str:
        text = normalize_pdf_text(text)

        if len(text) <= max_chars:
            return text

        cut = text[:max_chars]
        break_point = max(
            cut.rfind(". "),
            cut.rfind("; "),
            cut.rfind("\n"),
            cut.rfind(" "),
        )

        if break_point > max_chars * 0.60:
            cut = cut[:break_point]

        return cut.rstrip() + "..."

    def _stable_id(
        self,
        value: Any,
        prefix: str = "metadata",
    ) -> str:
        return f"{prefix}_{self._stable_hash(value)[:16]}"

    def _stable_hash(
        self,
        value: Any,
    ) -> str:
        try:
            text = json.dumps(
                json_safe(value),
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception:
            text = str(value)

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _to_dict(
        self,
        value: Any,
    ) -> Dict[str, Any]:
        if isinstance(value, dict):
            return dict(value)

        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            try:
                return dict(vars(value))
            except Exception:
                pass

        return {}

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

    def _default_stopwords(self) -> Set[str]:
        return {
            "va", "hoac", "cua", "cho", "cac", "mot", "nhung", "duoc", "trong",
            "ngoai", "theo", "voi", "den", "tu", "tai", "ve", "la", "co", "khong",
            "nay", "do", "khi", "sau", "truoc", "tren", "duoi", "vao", "ra", "de",
            "nham", "phuc", "vu", "can", "phai", "bao", "dam", "quy", "dinh",
            "noi", "dung", "thuc", "hien", "quan", "ly", "nha", "nuoc", "du",
            "lieu", "he", "thong", "chuc", "nang", "phan", "mem",
            "this", "that", "with", "from", "into", "about", "the", "and", "or",
            "for", "to", "of", "in", "on", "by", "is", "are", "be", "as", "at",
        }

    def save_index(
        self,
        index_result: Dict[str, Any],
        output_path: str,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                json_safe(index_result),
                f,
                ensure_ascii=False,
                indent=2,
            )

        return str(output_path)

    def load_index(
        self,
        input_path: str,
    ) -> Dict[str, Any]:
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)


def build_metadata_index(
    page_raws: Optional[List[PageRaw]] = None,
    metadata_enrichment_result: Optional[Dict[str, Any]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
    chunk_result: Optional[Dict[str, Any]] = None,
    table_chunk_result: Optional[Dict[str, Any]] = None,
    evidence_result: Optional[Dict[str, Any]] = None,
    citation_result: Optional[Dict[str, Any]] = None,
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    graph_index_result: Optional[Dict[str, Any]] = None,
    bm25_index_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = MetadataIndexBuilder()
    return builder.process(
        page_raws=page_raws,
        metadata_enrichment_result=metadata_enrichment_result,
        knowledge_result=knowledge_result,
        chunk_result=chunk_result,
        table_chunk_result=table_chunk_result,
        evidence_result=evidence_result,
        citation_result=citation_result,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
        graph_index_result=graph_index_result,
        bm25_index_result=bm25_index_result,
    )


def search_metadata_index(
    index_result: Dict[str, Any],
    filters: Optional[Dict[str, Any]] = None,
    query: str = "",
    item_types: Optional[List[str]] = None,
    page_numbers: Optional[List[int]] = None,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    builder = MetadataIndexBuilder()
    return builder.search(
        index_result=index_result,
        filters=filters,
        query=query,
        item_types=item_types,
        page_numbers=page_numbers,
        top_k=top_k,
    )


def facet_metadata_index(
    index_result: Dict[str, Any],
    field_name: str,
    filters: Optional[Dict[str, Any]] = None,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    builder = MetadataIndexBuilder()
    return builder.facet(
        index_result=index_result,
        field_name=field_name,
        filters=filters,
        top_k=top_k,
    )
