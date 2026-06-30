"""
metadata_enricher.py

Production V1 - Colab Ready

Purpose
-------
Enrich document/page/chunk/table/evidence metadata for downstream indexing,
knowledge graph, and RAG.

Used by:
- KnowledgePipeline
- ChunkBuilder
- EvidenceBuilder
- KnowledgeGraphBuilder
- Indexing
- RAGPipeline

Input
-----
- page_raws
- document_profile
- document_structure_result
- table_understanding_result
- cross_page_context_result
- chunk_result
- evidence_result

Output
------
Dictionary with:
- document_metadata
- page_metadata
- section_metadata
- table_metadata
- chunk_metadata
- evidence_metadata
- enrichment_summary
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw
from document_ai.schemas.page_raw_schema import (
    normalize_pdf_text,
    normalize_text_for_match,
    json_safe,
    make_id,
)


@dataclass
class MetadataEnricherConfig:
    enrich_document_metadata: bool = True
    enrich_page_metadata: bool = True
    enrich_section_metadata: bool = True
    enrich_table_metadata: bool = True
    enrich_chunk_metadata: bool = True
    enrich_evidence_metadata: bool = True
    enrich_context_metadata: bool = True

    attach_to_pages: bool = True

    infer_language: bool = True
    infer_document_domain: bool = True
    infer_document_category: bool = True
    infer_quality_flags: bool = True
    infer_processing_hints: bool = True

    build_page_text_stats: bool = True
    build_page_content_stats: bool = True
    build_document_text_stats: bool = True

    text_preview_chars: int = 700
    page_preview_chars: int = 400
    section_preview_chars: int = 500
    table_preview_chars: int = 500
    chunk_preview_chars: int = 400
    evidence_preview_chars: int = 400

    include_keyword_summary: bool = True
    max_keywords: int = 40

    include_debug: bool = True


class MetadataEnricher:
    def __init__(
        self,
        config: Optional[MetadataEnricherConfig] = None,
    ):
        self.config = config or MetadataEnricherConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        document_profile: Optional[Dict[str, Any]] = None,
        document_structure_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        cross_page_context_result: Optional[Dict[str, Any]] = None,
        chunk_result: Optional[Dict[str, Any]] = None,
        evidence_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        page_raws = sorted(
            page_raws or [],
            key=lambda item: item.page_number,
        )

        document_profile = self._to_dict(document_profile or {})
        document_structure_result = document_structure_result or {}
        table_understanding_result = table_understanding_result or {}
        cross_page_context_result = cross_page_context_result or {}
        chunk_result = chunk_result or {}
        evidence_result = evidence_result or {}

        page_text_map = self._build_page_text_map(page_raws)
        full_text = self._build_full_text(page_text_map)

        document_metadata = {}
        page_metadata = {}
        section_metadata = {}
        table_metadata = {}
        chunk_metadata = {}
        evidence_metadata = {}
        context_metadata = {}

        if self.config.enrich_document_metadata:
            document_metadata = self._build_document_metadata(
                page_raws=page_raws,
                document_profile=document_profile,
                document_structure_result=document_structure_result,
                table_understanding_result=table_understanding_result,
                cross_page_context_result=cross_page_context_result,
                chunk_result=chunk_result,
                evidence_result=evidence_result,
                full_text=full_text,
            )

        if self.config.enrich_page_metadata:
            page_metadata = self._build_page_metadata(
                page_raws=page_raws,
                page_text_map=page_text_map,
                document_structure_result=document_structure_result,
                table_understanding_result=table_understanding_result,
                cross_page_context_result=cross_page_context_result,
                chunk_result=chunk_result,
                evidence_result=evidence_result,
            )

        if self.config.enrich_section_metadata:
            section_metadata = self._build_section_metadata(
                document_structure_result=document_structure_result,
                page_text_map=page_text_map,
                chunk_result=chunk_result,
                evidence_result=evidence_result,
            )

        if self.config.enrich_table_metadata:
            table_metadata = self._build_table_metadata(
                table_understanding_result=table_understanding_result,
                page_text_map=page_text_map,
                chunk_result=chunk_result,
                evidence_result=evidence_result,
            )

        if self.config.enrich_chunk_metadata:
            chunk_metadata = self._build_chunk_metadata(
                chunk_result=chunk_result,
                page_text_map=page_text_map,
                document_metadata=document_metadata,
            )

        if self.config.enrich_evidence_metadata:
            evidence_metadata = self._build_evidence_metadata(
                evidence_result=evidence_result,
                page_text_map=page_text_map,
                document_metadata=document_metadata,
            )

        if self.config.enrich_context_metadata:
            context_metadata = self._build_context_metadata(
                cross_page_context_result=cross_page_context_result,
                page_text_map=page_text_map,
            )

        result = {
            "processor": "MetadataEnricher",
            "schema_version": "metadata_enricher_v1",
            "document_id": document_metadata.get("document_id", self._infer_document_id(page_raws)),
            "source_document": document_metadata.get("source_document", self._infer_source_document(page_raws)),
            "document_metadata": document_metadata,
            "page_metadata": page_metadata,
            "section_metadata": section_metadata,
            "table_metadata": table_metadata,
            "chunk_metadata": chunk_metadata,
            "evidence_metadata": evidence_metadata,
            "context_metadata": context_metadata,
            "metadata_summary": self._build_summary(
                document_metadata=document_metadata,
                page_metadata=page_metadata,
                section_metadata=section_metadata,
                table_metadata=table_metadata,
                chunk_metadata=chunk_metadata,
                evidence_metadata=evidence_metadata,
                context_metadata=context_metadata,
            ),
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                page_metadata=page_metadata,
                document_metadata=document_metadata,
            )

        return json_safe(result)

    def _build_document_metadata(
        self,
        page_raws: List[PageRaw],
        document_profile: Dict[str, Any],
        document_structure_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        cross_page_context_result: Dict[str, Any],
        chunk_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        full_text: str,
    ) -> Dict[str, Any]:
        document_id = (
            document_profile.get("document_id")
            or document_structure_result.get("document_id")
            or chunk_result.get("document_id")
            or evidence_result.get("document_id")
            or self._infer_document_id(page_raws)
        )

        source_document = (
            document_profile.get("source_path")
            or document_profile.get("source_document")
            or document_profile.get("file_name")
            or self._infer_source_document(page_raws)
        )

        profile_summary = (
            document_profile.get("document_profile_summary")
            or document_profile.get("summary")
            or {}
        )

        title = self._extract_title(document_structure_result, document_profile)
        sections = self._collect_list(document_structure_result, "sections")
        paragraphs = self._collect_list(document_structure_result, "paragraphs")
        chunks = self._collect_chunks(chunk_result)
        evidence_items = self._collect_evidence(evidence_result)
        tables = self._collect_tables(table_understanding_result)

        text_stats = self._text_stats(full_text) if self.config.build_document_text_stats else {}

        language = ""
        if self.config.infer_language:
            language = (
                document_profile.get("dominant_language")
                or profile_summary.get("dominant_language")
                or self._infer_language(full_text)
            )

        domain = self._infer_document_domain(full_text, title) if self.config.infer_document_domain else ""
        category = self._infer_document_category(full_text, title) if self.config.infer_document_category else ""

        quality_flags = []
        if self.config.infer_quality_flags:
            quality_flags = self._infer_document_quality_flags(
                page_raws=page_raws,
                document_profile=document_profile,
                full_text=full_text,
                tables=tables,
            )

        processing_hints = []
        if self.config.infer_processing_hints:
            processing_hints = self._infer_document_processing_hints(
                page_raws=page_raws,
                document_profile=document_profile,
                tables=tables,
                quality_flags=quality_flags,
            )

        keywords = []
        if self.config.include_keyword_summary:
            keywords = self._extract_keywords(
                full_text,
                max_keywords=self.config.max_keywords,
            )

        metadata = {
            "document_id": document_id,
            "source_document": source_document,
            "file_name": document_profile.get("file_name", Path(str(source_document)).name if source_document else ""),
            "file_extension": document_profile.get("file_extension", ""),
            "document_type": document_profile.get("document_type", ""),
            "mime_type": document_profile.get("mime_type", ""),
            "sha256": document_profile.get("sha256", ""),
            "content_hash": self._stable_hash(
                {
                    "document_id": document_id,
                    "source_document": source_document,
                    "text_hash": self._stable_hash(full_text[:100000]),
                    "page_count": len(page_raws),
                }
            ),
            "title": title,
            "language": language,
            "domain": domain,
            "category": category,
            "page_count": len(page_raws) or document_profile.get("page_count", 0),
            "section_count": len(sections),
            "paragraph_count": len(paragraphs),
            "table_count": len(tables),
            "chunk_count": len(chunks),
            "evidence_count": len(evidence_items),
            "has_text": bool(full_text),
            "has_tables": len(tables) > 0 or bool(document_profile.get("has_tables")),
            "has_images": any(page.image_count > 0 for page in page_raws),
            "has_drawings": any(page.drawing_count > 0 for page in page_raws),
            "has_annotations": any(page.annotation_count > 0 for page in page_raws),
            "has_links": any(page.link_count > 0 for page in page_raws),
            "need_ocr": bool(document_profile.get("need_ocr", False)),
            "pdf_type": document_profile.get("pdf_type", profile_summary.get("pdf_type", "")),
            "complexity_level": self._infer_complexity_level(
                page_raws=page_raws,
                table_count=len(tables),
                text_stats=text_stats,
                quality_flags=quality_flags,
            ),
            "processing_strategy": document_profile.get("processing_strategy", profile_summary.get("processing_strategy", "")),
            "text_stats": text_stats,
            "keywords": keywords,
            "quality_flags": quality_flags,
            "processing_hints": processing_hints,
            "source_counts": {
                "page_raw_count": len(page_raws),
                "section_count": len(sections),
                "paragraph_count": len(paragraphs),
                "table_count": len(tables),
                "chunk_count": len(chunks),
                "evidence_count": len(evidence_items),
                "context_node_count": self._context_node_count(cross_page_context_result),
                "context_edge_count": self._context_edge_count(cross_page_context_result),
            },
            "metadata_version": "metadata_enricher_v1",
        }

        if self.config.include_debug:
            metadata["debug"] = {
                "profile_keys": list(document_profile.keys()),
                "document_structure_keys": list(document_structure_result.keys()),
                "table_understanding_keys": list(table_understanding_result.keys()),
                "chunk_result_keys": list(chunk_result.keys()),
                "evidence_result_keys": list(evidence_result.keys()),
            }

        return metadata

    def _build_page_metadata(
        self,
        page_raws: List[PageRaw],
        page_text_map: Dict[int, str],
        document_structure_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        cross_page_context_result: Dict[str, Any],
        chunk_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        page_metadata: Dict[str, Dict[str, Any]] = {}

        sections_by_page = self._group_sections_by_page(
            self._collect_list(document_structure_result, "sections")
        )
        paragraphs_by_page = self._group_paragraphs_by_page(
            self._collect_list(document_structure_result, "paragraphs")
        )
        tables_by_page = self._group_tables_by_page(
            self._collect_tables(table_understanding_result)
        )
        chunks_by_page = self._group_chunks_by_page(
            self._collect_chunks(chunk_result)
        )
        evidence_by_page = self._group_evidence_by_page(
            self._collect_evidence(evidence_result)
        )
        context_by_page = self._group_context_by_page(cross_page_context_result)

        for page_raw in page_raws:
            page_number = page_raw.page_number
            page_key = str(page_number)
            text = page_text_map.get(page_number, "")

            page_sections = sections_by_page.get(page_key, [])
            page_paragraphs = paragraphs_by_page.get(page_key, [])
            page_tables = tables_by_page.get(page_key, [])
            page_chunks = chunks_by_page.get(page_key, [])
            page_evidence = evidence_by_page.get(page_key, [])
            page_context = context_by_page.get(page_key, {})

            text_stats = self._text_stats(text) if self.config.build_page_text_stats else {}

            content_stats = {}
            if self.config.build_page_content_stats:
                content_stats = {
                    "text_block_count": page_raw.text_block_count,
                    "text_line_count": page_raw.text_line_count,
                    "text_span_count": page_raw.text_span_count,
                    "word_count": page_raw.word_count,
                    "image_count": page_raw.image_count,
                    "drawing_count": page_raw.drawing_count,
                    "annotation_count": page_raw.annotation_count,
                    "link_count": page_raw.link_count,
                    "font_count": page_raw.font_count,
                    "section_count": len(page_sections),
                    "paragraph_count": len(page_paragraphs),
                    "table_count": len(page_tables),
                    "chunk_count": len(page_chunks),
                    "evidence_count": len(page_evidence),
                    "context_node_count": len(page_context.get("nodes", [])) if isinstance(page_context, dict) else 0,
                    "context_edge_count": len(page_context.get("edges", [])) if isinstance(page_context, dict) else 0,
                }

            quality_flags = self._infer_page_quality_flags(page_raw, text, page_tables)
            page_type = self._infer_page_type(page_raw, text, page_tables, page_sections)

            page_metadata[page_key] = {
                "page_number": page_number,
                "page_index": page_raw.page_index,
                "document_id": page_raw.document_id,
                "source_document": page_raw.source_document,
                "width": page_raw.width,
                "height": page_raw.height,
                "rotation": page_raw.rotation,
                "page_kind": page_raw.page_kind,
                "page_type": page_type,
                "content_hash": self._stable_hash(
                    {
                        "page_number": page_number,
                        "text": text,
                        "image_count": page_raw.image_count,
                        "drawing_count": page_raw.drawing_count,
                    }
                ),
                "text_preview": self._preview(text, self.config.page_preview_chars),
                "language": self._infer_language(text) if self.config.infer_language else "",
                "text_stats": text_stats,
                "content_stats": content_stats,
                "sections": [
                    self._compact_section(item)
                    for item in page_sections
                ],
                "tables": [
                    self._compact_table(item)
                    for item in page_tables
                ],
                "chunk_ids": [
                    item.get("chunk_id", "")
                    for item in page_chunks
                    if item.get("chunk_id")
                ],
                "evidence_ids": [
                    item.get("evidence_id", "")
                    for item in page_evidence
                    if item.get("evidence_id")
                ],
                "quality_flags": quality_flags,
                "processing_hints": self._infer_page_processing_hints(page_raw, quality_flags, page_tables),
                "metadata_version": "metadata_enricher_v1",
            }

        return page_metadata

    def _build_section_metadata(
        self,
        document_structure_result: Dict[str, Any],
        page_text_map: Dict[int, str],
        chunk_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        section_metadata: Dict[str, Dict[str, Any]] = {}

        sections = self._collect_list(document_structure_result, "sections")
        paragraphs = self._collect_list(document_structure_result, "paragraphs")
        chunks = self._collect_chunks(chunk_result)
        evidence_items = self._collect_evidence(evidence_result)

        paragraphs_by_section = self._group_paragraphs_by_section(paragraphs)
        chunks_by_section = self._group_chunks_by_section(chunks)
        evidence_by_section = self._group_evidence_by_section(evidence_items)

        for section in sections:
            section_id = section.get("section_id") or section.get("id") or ""
            if not section_id:
                continue

            title = normalize_pdf_text(section.get("title") or section.get("heading") or "")
            page_numbers = self._resolve_page_numbers(section)
            section_paragraphs = paragraphs_by_section.get(section_id, [])
            section_chunks = chunks_by_section.get(section_id, [])
            section_evidence = evidence_by_section.get(section_id, [])

            text_parts = []

            if title:
                text_parts.append(title)

            for paragraph in section_paragraphs:
                paragraph_text = normalize_pdf_text(paragraph.get("text", ""))
                if paragraph_text:
                    text_parts.append(paragraph_text)

            if not text_parts:
                text_preview = normalize_pdf_text(section.get("text_preview", ""))
                if text_preview:
                    text_parts.append(text_preview)

            if len(text_parts) <= 1:
                for page_number in page_numbers:
                    page_text = page_text_map.get(page_number, "")
                    if page_text:
                        text_parts.append(page_text)

            text = normalize_pdf_text("\n\n".join(text_parts))

            section_metadata[section_id] = {
                "section_id": section_id,
                "title": title,
                "normalized_title": normalize_text_for_match(title),
                "level": self._safe_int(section.get("level"), default=0),
                "order": self._safe_int(section.get("order"), default=0),
                "parent_id": section.get("parent_id", ""),
                "page_numbers": page_numbers,
                "page_start": min(page_numbers) if page_numbers else None,
                "page_end": max(page_numbers) if page_numbers else None,
                "text_preview": self._preview(text, self.config.section_preview_chars),
                "content_hash": self._stable_hash(
                    {
                        "section_id": section_id,
                        "title": title,
                        "text": text[:5000],
                        "page_numbers": page_numbers,
                    }
                ),
                "text_stats": self._text_stats(text),
                "paragraph_count": len(section_paragraphs),
                "chunk_count": len(section_chunks),
                "evidence_count": len(section_evidence),
                "chunk_ids": [
                    item.get("chunk_id", "")
                    for item in section_chunks
                    if item.get("chunk_id")
                ],
                "evidence_ids": [
                    item.get("evidence_id", "")
                    for item in section_evidence
                    if item.get("evidence_id")
                ],
                "keywords": self._extract_keywords(text, max_keywords=15),
                "metadata_version": "metadata_enricher_v1",
            }

        return section_metadata

    def _build_table_metadata(
        self,
        table_understanding_result: Dict[str, Any],
        page_text_map: Dict[int, str],
        chunk_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        table_metadata: Dict[str, Dict[str, Any]] = {}

        tables = self._collect_tables(table_understanding_result)
        chunks = self._collect_chunks(chunk_result)
        evidence_items = self._collect_evidence(evidence_result)

        chunks_by_table = self._group_chunks_by_table(chunks)
        evidence_by_table = self._group_evidence_by_table(evidence_items)

        for table in tables:
            table_id = self._table_id(table)
            if not table_id:
                continue

            page_numbers = self._resolve_page_numbers(table)

            if not page_numbers and table.get("page_number"):
                page_number = self._safe_int(table.get("page_number"), default=0)
                if page_number > 0:
                    page_numbers = [page_number]

            table_text = self._table_text(table)
            table_chunks = chunks_by_table.get(table_id, [])
            table_evidence = evidence_by_table.get(table_id, [])

            headers = table.get("column_headers", []) or []
            semantic_type = table.get("semantic_type", "")
            table_type = table.get("table_type", "")

            table_metadata[table_id] = {
                "table_id": table_id,
                "table_kind": self._table_kind(table),
                "page_numbers": page_numbers,
                "page_start": min(page_numbers) if page_numbers else None,
                "page_end": max(page_numbers) if page_numbers else None,
                "bbox": table.get("bbox", []) or [],
                "title": normalize_pdf_text(table.get("title", "")),
                "caption": normalize_pdf_text(table.get("caption") or table.get("caption_text") or ""),
                "semantic_type": semantic_type,
                "table_type": table_type,
                "row_count": self._safe_int(table.get("row_count") or table.get("total_row_count"), default=0),
                "col_count": self._safe_int(table.get("col_count"), default=0),
                "column_headers": [
                    normalize_pdf_text(item)
                    for item in headers
                    if normalize_pdf_text(item)
                ],
                "numeric_columns": table.get("numeric_columns", []) or [],
                "date_columns": table.get("date_columns", []) or [],
                "key_columns": table.get("key_columns", []) or [],
                "text_preview": self._preview(table_text, self.config.table_preview_chars),
                "content_hash": self._stable_hash(
                    {
                        "table_id": table_id,
                        "text": table_text,
                        "page_numbers": page_numbers,
                        "headers": headers,
                    }
                ),
                "text_stats": self._text_stats(table_text),
                "chunk_count": len(table_chunks),
                "evidence_count": len(table_evidence),
                "chunk_ids": [
                    item.get("chunk_id", "")
                    for item in table_chunks
                    if item.get("chunk_id")
                ],
                "evidence_ids": [
                    item.get("evidence_id", "")
                    for item in table_evidence
                    if item.get("evidence_id")
                ],
                "quality_flags": self._infer_table_quality_flags(table, table_text),
                "keywords": self._extract_keywords(table_text, max_keywords=15),
                "metadata_version": "metadata_enricher_v1",
            }

        return table_metadata

    def _build_chunk_metadata(
        self,
        chunk_result: Dict[str, Any],
        page_text_map: Dict[int, str],
        document_metadata: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        chunk_metadata: Dict[str, Dict[str, Any]] = {}

        chunks = self._collect_chunks(chunk_result)

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            if not chunk_id:
                continue

            text = normalize_pdf_text(chunk.get("text", ""))
            page_numbers = self._normalize_page_numbers(chunk.get("page_numbers", []))

            chunk_metadata[chunk_id] = {
                "chunk_id": chunk_id,
                "chunk_type": chunk.get("chunk_type", ""),
                "document_id": chunk.get("document_id") or document_metadata.get("document_id", ""),
                "source_document": chunk.get("source_document") or document_metadata.get("source_document", ""),
                "page_numbers": page_numbers,
                "page_start": min(page_numbers) if page_numbers else None,
                "page_end": max(page_numbers) if page_numbers else None,
                "section_id": chunk.get("section_id", ""),
                "section_title": chunk.get("section_title", ""),
                "section_level": chunk.get("section_level"),
                "paragraph_id": chunk.get("paragraph_id", ""),
                "table_grid_id": chunk.get("table_grid_id", ""),
                "table_structure_id": chunk.get("table_structure_id", ""),
                "table_semantic_id": chunk.get("table_semantic_id", ""),
                "table_boundary_id": chunk.get("table_boundary_id", ""),
                "text_preview": self._preview(text, self.config.chunk_preview_chars),
                "content_hash": chunk.get("content_hash") or self._stable_hash(text),
                "text_stats": self._text_stats(text),
                "keywords": self._extract_keywords(text, max_keywords=12),
                "quality_flags": self._infer_text_quality_flags(text),
                "source": chunk.get("source", ""),
                "confidence": self._safe_float(chunk.get("confidence"), default=0.70),
                "order": self._safe_int(chunk.get("order"), default=0),
                "metadata_version": "metadata_enricher_v1",
            }

        return chunk_metadata

    def _build_evidence_metadata(
        self,
        evidence_result: Dict[str, Any],
        page_text_map: Dict[int, str],
        document_metadata: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        evidence_metadata: Dict[str, Dict[str, Any]] = {}

        evidence_items = self._collect_evidence(evidence_result)

        for evidence in evidence_items:
            evidence_id = evidence.get("evidence_id", "")
            if not evidence_id:
                continue

            text = normalize_pdf_text(evidence.get("text") or evidence.get("quote") or "")
            page_numbers = self._resolve_page_numbers(evidence)

            evidence_metadata[evidence_id] = {
                "evidence_id": evidence_id,
                "evidence_type": evidence.get("evidence_type", ""),
                "document_id": evidence.get("document_id") or document_metadata.get("document_id", ""),
                "source_document": evidence.get("source_document") or document_metadata.get("source_document", ""),
                "page_numbers": page_numbers,
                "page_start": min(page_numbers) if page_numbers else None,
                "page_end": max(page_numbers) if page_numbers else None,
                "section_id": evidence.get("section_id", ""),
                "section_title": evidence.get("section_title", ""),
                "chunk_id": evidence.get("chunk_id") or evidence.get("source_chunk_id", ""),
                "table_grid_id": evidence.get("table_grid_id", ""),
                "table_structure_id": evidence.get("table_structure_id", ""),
                "table_semantic_id": evidence.get("table_semantic_id", ""),
                "table_boundary_id": evidence.get("table_boundary_id", ""),
                "text_preview": self._preview(text, self.config.evidence_preview_chars),
                "quote_preview": self._preview(evidence.get("quote", ""), self.config.evidence_preview_chars),
                "content_hash": evidence.get("content_hash") or self._stable_hash(text),
                "text_stats": self._text_stats(text),
                "keywords": self._extract_keywords(text, max_keywords=12),
                "quality_flags": self._infer_text_quality_flags(text),
                "relevance_score": self._safe_float(evidence.get("relevance_score"), default=0.0),
                "confidence": self._safe_float(evidence.get("confidence"), default=0.70),
                "weight": self._safe_float(evidence.get("weight"), default=1.0),
                "rank": self._safe_int(evidence.get("rank"), default=0),
                "source": evidence.get("source", ""),
                "metadata_version": "metadata_enricher_v1",
            }

        return evidence_metadata

    def _build_context_metadata(
        self,
        cross_page_context_result: Dict[str, Any],
        page_text_map: Dict[int, str],
    ) -> Dict[str, Any]:
        context_graph = cross_page_context_result.get("context_graph", {}) or {}

        nodes = context_graph.get("nodes", []) or cross_page_context_result.get("nodes", []) or []
        edges = context_graph.get("edges", []) or cross_page_context_result.get("edges", []) or []

        nodes_by_type: Dict[str, int] = {}
        edges_by_type: Dict[str, int] = {}
        nodes_by_page: Dict[str, int] = {}

        for node in nodes:
            if not isinstance(node, dict):
                continue

            node_type = node.get("node_type", "node")
            nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1

            page_numbers = self._resolve_page_numbers(node)
            for page_number in page_numbers:
                page_key = str(page_number)
                nodes_by_page[page_key] = nodes_by_page.get(page_key, 0) + 1

        for edge in edges:
            if not isinstance(edge, dict):
                continue

            edge_type = edge.get("edge_type", "related_to")
            edges_by_type[edge_type] = edges_by_type.get(edge_type, 0) + 1

        return {
            "has_context": bool(nodes or edges),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "nodes_by_page": nodes_by_page,
            "section_link_count": len(cross_page_context_result.get("section_links", []) or []),
            "paragraph_continuation_count": len(cross_page_context_result.get("paragraph_continuations", []) or []),
            "table_continuation_count": len(cross_page_context_result.get("table_continuations", []) or []),
            "entity_link_count": len(cross_page_context_result.get("entity_links", []) or []),
            "reference_link_count": len(cross_page_context_result.get("reference_links", []) or []),
            "metadata_version": "metadata_enricher_v1",
        }

    def _build_page_text_map(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[int, str]:
        page_text_map: Dict[int, str] = {}

        for page_raw in page_raws:
            reading_meta = page_raw.metadata.get("reading_order_builder", {}) or {}
            reading_text = normalize_pdf_text(reading_meta.get("reading_order_text", ""))

            if reading_text:
                text = reading_text
            elif page_raw.normalized_text:
                text = page_raw.normalized_text
            else:
                text = page_raw.raw_text

            page_text_map[page_raw.page_number] = normalize_pdf_text(text)

        return page_text_map

    def _build_full_text(
        self,
        page_text_map: Dict[int, str],
    ) -> str:
        parts = []

        for page_number in sorted(page_text_map):
            text = normalize_pdf_text(page_text_map.get(page_number, ""))
            if text:
                parts.append(text)

        return normalize_pdf_text("\n\n".join(parts))

    def _text_stats(
        self,
        text: str,
    ) -> Dict[str, Any]:
        text = normalize_pdf_text(text)
        words = re.findall(r"\S+", text)
        sentences = re.split(r"(?<=[\.\?\!])\s+", text) if text else []
        lines = [line for line in text.splitlines() if line.strip()]
        paragraphs = [item for item in re.split(r"\n\s*\n", text) if item.strip()]

        digit_count = len(re.findall(r"\d", text))
        uppercase_count = len(re.findall(r"[A-ZÀ-Ỵ]", text))
        vietnamese_count = len(re.findall(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", text.lower()))

        return {
            "char_count": len(text),
            "word_count": len(words),
            "sentence_count": len([item for item in sentences if item.strip()]),
            "line_count": len(lines),
            "paragraph_count": len(paragraphs),
            "digit_count": digit_count,
            "uppercase_count": uppercase_count,
            "vietnamese_char_count": vietnamese_count,
            "avg_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 2),
            "avg_sentence_words": round(len(words) / max(len(sentences), 1), 2),
            "numeric_density": round(digit_count / max(len(text), 1), 4),
            "uppercase_density": round(uppercase_count / max(len(text), 1), 4),
            "vietnamese_density": round(vietnamese_count / max(len(text), 1), 4),
        }

    def _extract_keywords(
        self,
        text: str,
        max_keywords: int = 30,
    ) -> List[Dict[str, Any]]:
        text = normalize_pdf_text(text)
        text_match = normalize_text_for_match(text)

        if not text_match:
            return []

        stopwords = {
            "va", "hoac", "cua", "cho", "cac", "mot", "nhung", "duoc", "trong", "ngoai",
            "theo", "voi", "den", "tu", "tai", "ve", "la", "co", "khong", "nay", "do",
            "khi", "sau", "truoc", "tren", "duoi", "vao", "ra", "de", "nham", "phuc",
            "vu", "can", "phai", "bao", "dam", "quy", "dinh", "noi", "dung",
            "thuc", "hien", "quan", "ly", "nha", "nuoc",
        }

        tokens = [
            token for token in re.findall(r"[a-z0-9_]{3,}", text_match)
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

    def _infer_language(
        self,
        text: str,
    ) -> str:
        text = normalize_pdf_text(text)

        if not text:
            return "unknown"

        vietnamese_chars = len(
            re.findall(
                r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
                text.lower(),
            )
        )

        common_vietnamese = len(
            re.findall(
                r"\b(của|và|các|được|trong|theo|quyết|định|thông|tư|nghị|định|báo|cáo|dữ|liệu)\b",
                text.lower(),
            )
        )

        if vietnamese_chars >= 10 or common_vietnamese >= 5:
            return "vi"

        ascii_letters = len(re.findall(r"[a-zA-Z]", text))

        if ascii_letters > 0:
            return "en_or_latin"

        return "unknown"

    def _infer_document_domain(
        self,
        text: str,
        title: str = "",
    ) -> str:
        source = normalize_text_for_match(f"{title}\n{text[:20000]}")

        domain_keywords = {
            "government": ["quyet dinh", "nghi dinh", "thong tu", "bo", "so", "ubnd", "chinh phu", "quan ly nha nuoc"],
            "audit": ["kiem toan", "ktnn", "bao cao kiem toan", "doan kiem toan", "ket luan kiem toan"],
            "healthcare": ["y te", "benh vien", "ho so benh an", "his", "emr", "lis", "pacs", "telehealth"],
            "industry_trade": ["cong thuong", "cong nghiep", "thuong mai", "san xuat cong nghiep"],
            "agriculture": ["nong nghiep", "truy xuat", "ma vung trong", "nong san", "sau rieng"],
            "finance": ["tai chinh", "ngan sach", "du toan", "von dau tu", "chi phi"],
            "technology": ["cong nghe thong tin", "chuyen doi so", "nen tang", "du lieu", "phan mem"],
        }

        scores = {}

        for domain, keywords in domain_keywords.items():
            score = 0

            for keyword in keywords:
                if keyword in source:
                    score += 1

            scores[domain] = score

        best_domain, best_score = max(scores.items(), key=lambda item: item[1])

        if best_score <= 0:
            return "general"

        return best_domain

    def _infer_document_category(
        self,
        text: str,
        title: str = "",
    ) -> str:
        source = normalize_text_for_match(f"{title}\n{text[:15000]}")

        if "quyet dinh" in source:
            return "decision"

        if "bao cao kinh te ky thuat" in source or "bao cao nghien cuu kha thi" in source:
            return "investment_report"

        if "de an" in source or "ke hoach" in source:
            return "plan_or_scheme"

        if "hop dong" in source:
            return "contract"

        if "bien ban" in source:
            return "minutes"

        if "bao cao" in source:
            return "report"

        if "to trinh" in source:
            return "submission"

        return "document"

    def _infer_complexity_level(
        self,
        page_raws: List[PageRaw],
        table_count: int,
        text_stats: Dict[str, Any],
        quality_flags: List[str],
    ) -> str:
        score = 0

        page_count = len(page_raws)

        if page_count >= 1000:
            score += 3
        elif page_count >= 300:
            score += 2
        elif page_count >= 80:
            score += 1

        if table_count >= 100:
            score += 3
        elif table_count >= 30:
            score += 2
        elif table_count >= 5:
            score += 1

        if text_stats.get("word_count", 0) >= 300000:
            score += 3
        elif text_stats.get("word_count", 0) >= 80000:
            score += 2
        elif text_stats.get("word_count", 0) >= 20000:
            score += 1

        if any(flag in quality_flags for flag in ["ocr_needed", "mixed_scanned_digital", "low_text_density"]):
            score += 1

        if score >= 6:
            return "very_high"

        if score >= 4:
            return "high"

        if score >= 2:
            return "medium"

        return "low"

    def _infer_document_quality_flags(
        self,
        page_raws: List[PageRaw],
        document_profile: Dict[str, Any],
        full_text: str,
        tables: List[Dict[str, Any]],
    ) -> List[str]:
        flags = []

        if not page_raws:
            flags.append("no_pages")

        if not full_text:
            flags.append("no_text")

        if document_profile.get("need_ocr"):
            flags.append("ocr_needed")

        text_pages = sum(1 for page in page_raws if page.has_text)
        image_pages = sum(1 for page in page_raws if page.has_images)

        if text_pages > 0 and image_pages > 0:
            flags.append("mixed_scanned_digital")

        if image_pages > text_pages and image_pages > 0:
            flags.append("image_heavy_document")

        if len(tables) > 0:
            flags.append("has_tables")

        if len(tables) >= 30:
            flags.append("table_heavy_document")

        if len(full_text) < max(len(page_raws), 1) * 100:
            flags.append("low_text_density")

        if any(page.annotation_count > 0 for page in page_raws):
            flags.append("has_annotations")

        if any(page.link_count > 0 for page in page_raws):
            flags.append("has_links")

        return sorted(list(dict.fromkeys(flags)))

    def _infer_document_processing_hints(
        self,
        page_raws: List[PageRaw],
        document_profile: Dict[str, Any],
        tables: List[Dict[str, Any]],
        quality_flags: List[str],
    ) -> List[str]:
        hints = []

        if "ocr_needed" in quality_flags:
            hints.append("run_selective_ocr_or_ocr_first")

        if "table_heavy_document" in quality_flags or len(tables) >= 10:
            hints.append("prioritize_table_understanding")

        if len(page_raws) >= 500:
            hints.append("use_batch_processing")

        if len(page_raws) >= 1000:
            hints.append("use_incremental_checkpointing")

        if "mixed_scanned_digital" in quality_flags:
            hints.append("separate_digital_and_scanned_pages")

        if document_profile.get("is_encrypted"):
            hints.append("handle_encrypted_pdf")

        if not hints:
            hints.append("standard_pipeline")

        return hints

    def _infer_page_type(
        self,
        page_raw: PageRaw,
        text: str,
        page_tables: List[Dict[str, Any]],
        page_sections: List[Dict[str, Any]],
    ) -> str:
        text_match = normalize_text_for_match(text)

        if page_raw.is_blank:
            return "blank_page"

        if page_tables:
            return "table_page"

        if "muc luc" in text_match or "table of contents" in text_match:
            return "toc_page"

        if page_sections and len(text) < 1200:
            return "section_start_page"

        if page_raw.image_count >= 2 and len(text) < 800:
            return "image_heavy_page"

        if page_raw.drawing_count >= 30:
            return "drawing_heavy_page"

        if len(text) > 2500:
            return "text_heavy_page"

        return "normal_page"

    def _infer_page_quality_flags(
        self,
        page_raw: PageRaw,
        text: str,
        page_tables: List[Dict[str, Any]],
    ) -> List[str]:
        flags = []

        if page_raw.is_blank:
            flags.append("blank_page")

        if not text and page_raw.image_count > 0:
            flags.append("possible_scanned_page")

        if len(text) < 50 and not page_raw.is_blank:
            flags.append("low_text")

        if page_raw.image_count > 0:
            flags.append("has_images")

        if page_raw.drawing_count > 0:
            flags.append("has_drawings")

        if page_tables:
            flags.append("has_tables")

        if page_raw.annotation_count > 0:
            flags.append("has_annotations")

        if page_raw.link_count > 0:
            flags.append("has_links")

        return sorted(list(dict.fromkeys(flags)))

    def _infer_page_processing_hints(
        self,
        page_raw: PageRaw,
        quality_flags: List[str],
        page_tables: List[Dict[str, Any]],
    ) -> List[str]:
        hints = []

        if "possible_scanned_page" in quality_flags:
            hints.append("run_ocr")

        if page_tables:
            hints.append("run_table_grid_and_structure_recognition")

        if page_raw.drawing_count >= 20:
            hints.append("analyze_drawing_lines")

        if page_raw.image_count >= 2:
            hints.append("analyze_figures_or_images")

        if not hints:
            hints.append("standard_page_processing")

        return hints

    def _infer_table_quality_flags(
        self,
        table: Dict[str, Any],
        table_text: str,
    ) -> List[str]:
        flags = []

        row_count = self._safe_int(table.get("row_count") or table.get("total_row_count"), default=0)
        col_count = self._safe_int(table.get("col_count"), default=0)
        headers = table.get("column_headers", []) or []

        if row_count == 0:
            flags.append("unknown_row_count")

        if col_count == 0:
            flags.append("unknown_col_count")

        if not headers:
            flags.append("missing_column_headers")

        if not table_text:
            flags.append("empty_table_text")

        if table.get("page_start") and table.get("page_end") and table.get("page_start") != table.get("page_end"):
            flags.append("multi_page_table")

        if row_count >= 100:
            flags.append("large_table")

        return sorted(list(dict.fromkeys(flags)))

    def _infer_text_quality_flags(
        self,
        text: str,
    ) -> List[str]:
        flags = []
        text = normalize_pdf_text(text)

        if not text:
            return ["empty_text"]

        if len(text) < 80:
            flags.append("short_text")

        if len(re.findall(r"\S+", text)) < 10:
            flags.append("few_words")

        if re.search(r"[�□]", text):
            flags.append("encoding_issue_possible")

        if len(re.findall(r"\d", text)) / max(len(text), 1) > 0.35:
            flags.append("numeric_heavy")

        if len(re.findall(r"[A-ZÀ-Ỵ]", text)) / max(len(text), 1) > 0.45:
            flags.append("uppercase_heavy")

        return sorted(list(dict.fromkeys(flags)))

    def _extract_title(
        self,
        document_structure_result: Dict[str, Any],
        document_profile: Dict[str, Any],
    ) -> str:
        title = document_structure_result.get("title", "")

        if isinstance(title, dict):
            title_text = normalize_pdf_text(title.get("text") or title.get("title") or "")
            if title_text:
                return title_text

        if isinstance(title, str) and normalize_pdf_text(title):
            return normalize_pdf_text(title)

        title_candidates = document_structure_result.get("title_candidates", []) or []

        if title_candidates:
            candidate = title_candidates[0]
            if isinstance(candidate, dict):
                return normalize_pdf_text(candidate.get("text", ""))

        return normalize_pdf_text(document_profile.get("title") or document_profile.get("file_name") or "")

    def _collect_chunks(
        self,
        chunk_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        chunks = []

        for key in ["chunks", "parent_chunks", "child_chunks", "table_chunks"]:
            values = chunk_result.get(key, []) or []

            if isinstance(values, list):
                chunks.extend([self._to_dict(item) for item in values])

        sub = chunk_result.get("chunk_collection", {}) or {}

        if isinstance(sub, dict):
            for key in ["chunks", "table_chunks"]:
                values = sub.get(key, []) or []

                if isinstance(values, list):
                    chunks.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(chunks, ["chunk_id", "content_hash"])

    def _collect_evidence(
        self,
        evidence_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        evidence = []

        for key in ["evidence", "evidence_items", "retrieved_evidence", "supporting_evidence"]:
            values = evidence_result.get(key, []) or []

            if isinstance(values, list):
                evidence.extend([self._to_dict(item) for item in values])

        sub = evidence_result.get("evidence_collection", {}) or {}

        if isinstance(sub, dict):
            values = sub.get("evidence", []) or sub.get("evidence_items", []) or []

            if isinstance(values, list):
                evidence.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(evidence, ["evidence_id", "content_hash"])

    def _collect_tables(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tables = []

        for key in ["table_semantics", "table_grids", "multi_page_tables"]:
            values = table_understanding_result.get(key, []) or []

            if isinstance(values, list):
                tables.extend([self._to_dict(item) for item in values])

        sub_keys = [
            "table_semantic_result",
            "table_grid_result",
            "multi_page_table_result",
        ]

        for sub_key in sub_keys:
            sub = table_understanding_result.get(sub_key, {}) or {}

            if not isinstance(sub, dict):
                continue

            for key in ["table_semantics", "table_grids", "multi_page_tables"]:
                values = sub.get(key, []) or []

                if isinstance(values, list):
                    tables.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(
            tables,
            ["table_semantic_id", "table_grid_id", "multi_page_table_id", "table_structure_id", "table_boundary_id"],
        )

    def _collect_list(
        self,
        source: Dict[str, Any],
        key: str,
    ) -> List[Dict[str, Any]]:
        values = source.get(key, []) or []

        if not isinstance(values, list):
            return []

        return [self._to_dict(item) for item in values]

    def _group_sections_by_page(
        self,
        sections: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for section in sections:
            for page_number in self._resolve_page_numbers(section):
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(section)

        return grouped

    def _group_paragraphs_by_page(
        self,
        paragraphs: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for paragraph in paragraphs:
            page_numbers = self._resolve_page_numbers(paragraph)

            if not page_numbers and paragraph.get("page_number"):
                page = self._safe_int(paragraph.get("page_number"), default=0)
                if page > 0:
                    page_numbers = [page]

            for page_number in page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(paragraph)

        return grouped

    def _group_tables_by_page(
        self,
        tables: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for table in tables:
            page_numbers = self._resolve_page_numbers(table)

            if not page_numbers and table.get("page_number"):
                page = self._safe_int(table.get("page_number"), default=0)
                if page > 0:
                    page_numbers = [page]

            for page_number in page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(table)

        return grouped

    def _group_chunks_by_page(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in chunks:
            for page_number in self._normalize_page_numbers(chunk.get("page_numbers", [])):
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(chunk)

        return grouped

    def _group_evidence_by_page(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for evidence in evidence_items:
            page_numbers = self._resolve_page_numbers(evidence)

            for page_number in page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(evidence)

        return grouped

    def _group_context_by_page(
        self,
        cross_page_context_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        context_graph = cross_page_context_result.get("context_graph", {}) or {}
        page_contexts = context_graph.get("page_contexts", {}) or cross_page_context_result.get("page_contexts", {}) or {}

        if isinstance(page_contexts, dict):
            return {
                str(key): value
                for key, value in page_contexts.items()
                if isinstance(value, dict)
            }

        return {}

    def _group_paragraphs_by_section(
        self,
        paragraphs: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for paragraph in paragraphs:
            section_id = paragraph.get("section_id", "")

            if not section_id:
                continue

            grouped.setdefault(section_id, [])
            grouped[section_id].append(paragraph)

        return grouped

    def _group_chunks_by_section(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in chunks:
            section_id = chunk.get("section_id", "") or "no_section"
            grouped.setdefault(section_id, [])
            grouped[section_id].append(chunk)

        return grouped

    def _group_evidence_by_section(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for evidence in evidence_items:
            section_id = evidence.get("section_id", "") or "no_section"
            grouped.setdefault(section_id, [])
            grouped[section_id].append(evidence)

        return grouped

    def _group_chunks_by_table(
        self,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in chunks:
            table_ids = [
                chunk.get("table_semantic_id", ""),
                chunk.get("table_grid_id", ""),
                chunk.get("table_structure_id", ""),
                chunk.get("table_boundary_id", ""),
            ]

            for table_id in table_ids:
                if not table_id:
                    continue

                grouped.setdefault(table_id, [])
                grouped[table_id].append(chunk)

        return grouped

    def _group_evidence_by_table(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for evidence in evidence_items:
            table_ids = [
                evidence.get("table_semantic_id", ""),
                evidence.get("table_grid_id", ""),
                evidence.get("table_structure_id", ""),
                evidence.get("table_boundary_id", ""),
            ]

            for table_id in table_ids:
                if not table_id:
                    continue

                grouped.setdefault(table_id, [])
                grouped[table_id].append(evidence)

        return grouped

    def _table_id(
        self,
        table: Dict[str, Any],
    ) -> str:
        return (
            table.get("table_semantic_id")
            or table.get("table_grid_id")
            or table.get("multi_page_table_id")
            or table.get("table_structure_id")
            or table.get("table_boundary_id")
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

            if value and value not in parts:
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
            parts.append(normalize_pdf_text(table.get("text")))

        records = table.get("records", []) or []

        for record in records[:5]:
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

        if not parts:
            table_id = self._table_id(table)
            if table_id:
                parts.append(table_id)

        return normalize_pdf_text("\n".join(parts))

    def _compact_section(
        self,
        section: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "section_id": section.get("section_id", "") or section.get("id", ""),
            "title": normalize_pdf_text(section.get("title", "")),
            "level": self._safe_int(section.get("level"), default=0),
            "page_numbers": self._resolve_page_numbers(section),
        }

    def _compact_table(
        self,
        table: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "table_id": self._table_id(table),
            "table_kind": self._table_kind(table),
            "title": normalize_pdf_text(table.get("title", "")),
            "caption": normalize_pdf_text(table.get("caption") or table.get("caption_text") or ""),
            "semantic_type": table.get("semantic_type", ""),
            "page_numbers": self._resolve_page_numbers(table),
            "row_count": self._safe_int(table.get("row_count") or table.get("total_row_count"), default=0),
            "col_count": self._safe_int(table.get("col_count"), default=0),
        }

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

    def _context_node_count(
        self,
        cross_page_context_result: Dict[str, Any],
    ) -> int:
        context_graph = cross_page_context_result.get("context_graph", {}) or {}
        nodes = context_graph.get("nodes", []) or cross_page_context_result.get("nodes", []) or []
        return len(nodes) if isinstance(nodes, list) else 0

    def _context_edge_count(
        self,
        cross_page_context_result: Dict[str, Any],
    ) -> int:
        context_graph = cross_page_context_result.get("context_graph", {}) or {}
        edges = context_graph.get("edges", []) or cross_page_context_result.get("edges", []) or []
        return len(edges) if isinstance(edges, list) else 0

    def _build_summary(
        self,
        document_metadata: Dict[str, Any],
        page_metadata: Dict[str, Any],
        section_metadata: Dict[str, Any],
        table_metadata: Dict[str, Any],
        chunk_metadata: Dict[str, Any],
        evidence_metadata: Dict[str, Any],
        context_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "has_metadata": True,
            "document_id": document_metadata.get("document_id", ""),
            "source_document": document_metadata.get("source_document", ""),
            "document_domain": document_metadata.get("domain", ""),
            "document_category": document_metadata.get("category", ""),
            "language": document_metadata.get("language", ""),
            "complexity_level": document_metadata.get("complexity_level", ""),
            "page_metadata_count": len(page_metadata),
            "section_metadata_count": len(section_metadata),
            "table_metadata_count": len(table_metadata),
            "chunk_metadata_count": len(chunk_metadata),
            "evidence_metadata_count": len(evidence_metadata),
            "context_node_count": context_metadata.get("node_count", 0),
            "context_edge_count": context_metadata.get("edge_count", 0),
            "quality_flags": document_metadata.get("quality_flags", []),
            "processing_hints": document_metadata.get("processing_hints", []),
            "metadata_version": "metadata_enricher_v1",
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        page_metadata: Dict[str, Dict[str, Any]],
        document_metadata: Dict[str, Any],
    ) -> None:
        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("metadata_enricher", {})
            page_raw.metadata["metadata_enricher"] = {
                "processor": "MetadataEnricher",
                "document_metadata": {
                    "document_id": document_metadata.get("document_id", ""),
                    "source_document": document_metadata.get("source_document", ""),
                    "title": document_metadata.get("title", ""),
                    "language": document_metadata.get("language", ""),
                    "domain": document_metadata.get("domain", ""),
                    "category": document_metadata.get("category", ""),
                    "complexity_level": document_metadata.get("complexity_level", ""),
                },
                "page_metadata": page_metadata.get(page_key, {}),
            }

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

    def _infer_document_id(
        self,
        page_raws: List[PageRaw],
    ) -> str:
        for page_raw in page_raws:
            if page_raw.document_id:
                return page_raw.document_id

        return ""

    def _infer_source_document(
        self,
        page_raws: List[PageRaw],
    ) -> str:
        for page_raw in page_raws:
            if page_raw.source_document:
                return page_raw.source_document

        return ""

    def _preview(
        self,
        text: Any,
        max_chars: int = 300,
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


def enrich_metadata(
    page_raws: List[PageRaw],
    document_profile: Optional[Dict[str, Any]] = None,
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    cross_page_context_result: Optional[Dict[str, Any]] = None,
    chunk_result: Optional[Dict[str, Any]] = None,
    evidence_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    enricher = MetadataEnricher()
    return enricher.process(
        page_raws=page_raws,
        document_profile=document_profile,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
        cross_page_context_result=cross_page_context_result,
        chunk_result=chunk_result,
        evidence_result=evidence_result,
    )
