"""
knowledge_pipeline.py

Production V1 - Colab Ready

Purpose
-------
Build knowledge layer from document understanding outputs.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- DocumentStructurePipeline
- TableUnderstandingPipeline
- CrossPageContextPipeline

Output
------
Dictionary with:
- metadata_enrichment
- chunks
- parent_child_chunks
- table_chunks
- evidence
- citations
- knowledge_graph
- knowledge_summary

Flow
----
DocumentStructurePipeline
    ↓
TableUnderstandingPipeline
    ↓
CrossPageContextPipeline
    ↓
KnowledgePipeline
        ├── MetadataEnricher
        ├── ChunkBuilder
        ├── ParentChildChunkBuilder
        ├── TableChunkBuilder
        ├── EvidenceBuilder
        ├── CitationBuilder
        └── KnowledgeGraphBuilder
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import traceback
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class KnowledgePipelineConfig:
    run_metadata_enricher: bool = True
    run_chunk_builder: bool = True
    run_parent_child_chunk_builder: bool = True
    run_table_chunk_builder: bool = True
    run_evidence_builder: bool = True
    run_citation_builder: bool = True
    run_knowledge_graph_builder: bool = True

    continue_on_error: bool = True
    attach_to_pages: bool = True

    chunk_size_chars: int = 1800
    chunk_overlap_chars: int = 250
    min_chunk_chars: int = 80
    max_chunk_chars: int = 3000

    include_page_chunks: bool = True
    include_section_chunks: bool = True
    include_paragraph_chunks: bool = True
    include_table_chunks: bool = True

    save_json: bool = False
    output_dir: str = "outputs/knowledge"

    include_debug: bool = True


class KnowledgePipeline:
    def __init__(
        self,
        config: Optional[KnowledgePipelineConfig] = None,
    ):
        self.config = config or KnowledgePipelineConfig()

        self.metadata_enricher = self._load_component(
            module_path="document_ai.knowledge.metadata_enricher",
            class_name="MetadataEnricher",
        )

        self.chunk_builder = self._load_component(
            module_path="document_ai.knowledge.chunk_builder",
            class_name="ChunkBuilder",
        )

        self.parent_child_chunk_builder = self._load_component(
            module_path="document_ai.knowledge.parent_child_chunk_builder",
            class_name="ParentChildChunkBuilder",
        )

        self.table_chunk_builder = self._load_component(
            module_path="document_ai.knowledge.table_chunk_builder",
            class_name="TableChunkBuilder",
        )

        self.evidence_builder = self._load_component(
            module_path="document_ai.knowledge.evidence_builder",
            class_name="EvidenceBuilder",
        )

        self.citation_builder = self._load_component(
            module_path="document_ai.knowledge.citation_builder",
            class_name="CitationBuilder",
        )

        self.knowledge_graph_builder = self._load_component(
            module_path="document_ai.knowledge.knowledge_graph_builder",
            class_name="KnowledgeGraphBuilder",
        )

    def process(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        cross_page_context_result: Optional[Dict[str, Any]] = None,
        page_documents: Optional[List[Any]] = None,
        document_profile: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        errors: List[Dict[str, Any]] = []

        metadata_result: Dict[str, Any] = {}
        chunk_result: Dict[str, Any] = {}
        parent_child_result: Dict[str, Any] = {}
        table_chunk_result: Dict[str, Any] = {}
        evidence_result: Dict[str, Any] = {}
        citation_result: Dict[str, Any] = {}
        knowledge_graph_result: Dict[str, Any] = {}

        if self.config.run_metadata_enricher:
            metadata_result = self._run_step(
                step_name="MetadataEnricher",
                component=self.metadata_enricher,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "document_structure_result": document_structure_result,
                    "table_understanding_result": table_understanding_result,
                    "cross_page_context_result": cross_page_context_result,
                },
                fallback_fn=lambda: self._fallback_metadata_enrichment(
                    page_raws=page_raws,
                    document_structure_result=document_structure_result,
                    table_understanding_result=table_understanding_result,
                    cross_page_context_result=cross_page_context_result,
                ),
                errors=errors,
            )

        if self.config.run_chunk_builder:
            chunk_result = self._run_step(
                step_name="ChunkBuilder",
                component=self.chunk_builder,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "document_structure_result": document_structure_result,
                    "metadata_result": metadata_result,
                    "metadata_enrichment_result": metadata_result,
                },
                fallback_fn=lambda: self._fallback_chunks(
                    page_raws=page_raws,
                    document_structure_result=document_structure_result,
                    metadata_result=metadata_result,
                ),
                errors=errors,
            )

        chunks = self._extract_chunks(chunk_result)

        if self.config.run_parent_child_chunk_builder:
            parent_child_result = self._run_step(
                step_name="ParentChildChunkBuilder",
                component=self.parent_child_chunk_builder,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "chunks": chunks,
                    "chunk_result": chunk_result,
                    "document_structure_result": document_structure_result,
                },
                fallback_fn=lambda: self._fallback_parent_child_chunks(
                    chunks=chunks,
                    document_structure_result=document_structure_result,
                ),
                errors=errors,
            )

        if self.config.run_table_chunk_builder:
            table_chunk_result = self._run_step(
                step_name="TableChunkBuilder",
                component=self.table_chunk_builder,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "table_understanding_result": table_understanding_result,
                    "document_structure_result": document_structure_result,
                    "metadata_result": metadata_result,
                },
                fallback_fn=lambda: self._fallback_table_chunks(
                    page_raws=page_raws,
                    table_understanding_result=table_understanding_result,
                ),
                errors=errors,
            )

        table_chunks = self._extract_chunks(table_chunk_result)
        all_chunks = self._deduplicate_chunks(chunks + table_chunks)

        if self.config.run_evidence_builder:
            evidence_result = self._run_step(
                step_name="EvidenceBuilder",
                component=self.evidence_builder,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "chunks": all_chunks,
                    "chunk_result": chunk_result,
                    "table_chunk_result": table_chunk_result,
                    "document_structure_result": document_structure_result,
                    "table_understanding_result": table_understanding_result,
                    "cross_page_context_result": cross_page_context_result,
                },
                fallback_fn=lambda: self._fallback_evidence(
                    chunks=all_chunks,
                    page_raws=page_raws,
                ),
                errors=errors,
            )

        evidence_items = self._extract_evidence(evidence_result)

        if self.config.run_citation_builder:
            citation_result = self._run_step(
                step_name="CitationBuilder",
                component=self.citation_builder,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "chunks": all_chunks,
                    "evidence": evidence_items,
                    "evidence_result": evidence_result,
                    "document_structure_result": document_structure_result,
                },
                fallback_fn=lambda: self._fallback_citations(
                    chunks=all_chunks,
                    evidence=evidence_items,
                ),
                errors=errors,
            )

        citations = self._extract_citations(citation_result)

        if self.config.run_knowledge_graph_builder:
            knowledge_graph_result = self._run_step(
                step_name="KnowledgeGraphBuilder",
                component=self.knowledge_graph_builder,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "chunks": all_chunks,
                    "evidence": evidence_items,
                    "citations": citations,
                    "document_structure_result": document_structure_result,
                    "table_understanding_result": table_understanding_result,
                    "cross_page_context_result": cross_page_context_result,
                    "metadata_result": metadata_result,
                },
                fallback_fn=lambda: self._fallback_knowledge_graph(
                    page_raws=page_raws,
                    chunks=all_chunks,
                    evidence=evidence_items,
                    citations=citations,
                    document_structure_result=document_structure_result,
                    table_understanding_result=table_understanding_result,
                    cross_page_context_result=cross_page_context_result,
                ),
                errors=errors,
            )

        result = {
            "processor": "KnowledgePipeline",
            "schema_version": "knowledge_pipeline_v1",
            "metadata_enrichment": metadata_result,
            "chunk_result": chunk_result,
            "parent_child_chunks": parent_child_result,
            "table_chunk_result": table_chunk_result,
            "chunks": all_chunks,
            "evidence_result": evidence_result,
            "evidence": evidence_items,
            "citation_result": citation_result,
            "citations": citations,
            "knowledge_graph": knowledge_graph_result,
            "knowledge_summary": self._build_summary(
                page_raws=page_raws,
                chunks=all_chunks,
                evidence=evidence_items,
                citations=citations,
                metadata_result=metadata_result,
                parent_child_result=parent_child_result,
                table_chunk_result=table_chunk_result,
                knowledge_graph_result=knowledge_graph_result,
                errors=errors,
            ),
            "errors": errors,
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                result=result,
            )

        if self.config.save_json:
            self.save_knowledge_result(
                result=result,
                page_raws=page_raws,
            )

        return result

    def _run_step(
        self,
        step_name: str,
        component: Any,
        kwargs: Dict[str, Any],
        fallback_fn: Any,
        errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if component is None:
            result = fallback_fn()

            if isinstance(result, dict):
                result.setdefault("processor", step_name)
                result.setdefault("mode", "fallback")
                result.setdefault("warning", f"{step_name} component not found, fallback output generated.")

            return result

        try:
            if hasattr(component, "process"):
                return self._safe_call(component.process, kwargs)

            if hasattr(component, "build"):
                return self._safe_call(component.build, kwargs)

            if callable(component):
                return self._safe_call(component, kwargs)

            raise RuntimeError(f"{step_name} component has no callable method.")

        except Exception as exc:
            error = {
                "step": step_name,
                "error": str(exc),
            }

            if self.config.include_debug:
                error["traceback"] = traceback.format_exc()

            errors.append(error)

            if not self.config.continue_on_error:
                raise

            result = fallback_fn()

            if isinstance(result, dict):
                result.setdefault("processor", step_name)
                result.setdefault("mode", "fallback_after_error")
                result.setdefault("error", str(exc))

            return result

    def _safe_call(
        self,
        fn: Any,
        kwargs: Dict[str, Any],
    ) -> Any:
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }

        try:
            signature = inspect.signature(fn)
            parameters = signature.parameters

            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )

            if accepts_kwargs:
                return fn(**kwargs)

            accepted_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in parameters
            }

            if accepted_kwargs:
                return fn(**accepted_kwargs)

            for key in ["page_raws", "pages", "chunks", "evidence"]:
                if key in kwargs:
                    return fn(kwargs[key])

            return fn()

        except ValueError:
            return fn(**kwargs)

    def _load_component(
        self,
        module_path: str,
        class_name: str,
    ) -> Any:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls()
        except Exception:
            return None

    def _fallback_metadata_enrichment(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Optional[Dict[str, Any]],
        table_understanding_result: Optional[Dict[str, Any]],
        cross_page_context_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        document_id = page_raws[0].document_id if page_raws else ""

        sections = []
        if document_structure_result:
            sections = document_structure_result.get("sections", []) or []

        table_count = 0
        if table_understanding_result:
            summary = table_understanding_result.get("table_understanding_summary", {}) or {}
            table_count = (
                summary.get("table_semantic_count")
                or summary.get("table_structure_count")
                or summary.get("table_grid_count")
                or 0
            )

        context_summary = {}
        if cross_page_context_result:
            context_summary = cross_page_context_result.get("cross_page_context_summary", {}) or {}

        page_metadata = []

        for page_raw in page_raws:
            page_metadata.append(
                {
                    "page_number": page_raw.page_number,
                    "page_index": page_raw.page_index,
                    "width": page_raw.width,
                    "height": page_raw.height,
                    "word_count": len(page_raw.words),
                    "text_line_count": len(page_raw.text_lines),
                    "image_count": len(page_raw.images),
                    "drawing_count": len(page_raw.drawings),
                    "has_text": bool(page_raw.normalized_text or page_raw.raw_text),
                    "metadata_keys": sorted(list(page_raw.metadata.keys())),
                }
            )

        return {
            "processor": "MetadataEnricher",
            "mode": "fallback",
            "document_metadata": {
                "document_id": document_id,
                "page_count": len(page_raws),
                "section_count": len(sections),
                "table_count": table_count,
                "context_graph_node_count": context_summary.get("context_graph_node_count", 0),
                "context_graph_edge_count": context_summary.get("context_graph_edge_count", 0),
            },
            "page_metadata": page_metadata,
            "metadata_summary": {
                "page_count": len(page_metadata),
                "has_document_structure": bool(document_structure_result),
                "has_table_understanding": bool(table_understanding_result),
                "has_cross_page_context": bool(cross_page_context_result),
            },
        }

    def _fallback_chunks(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Optional[Dict[str, Any]],
        metadata_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        chunks: List[Dict[str, Any]] = []

        if self.config.include_section_chunks and document_structure_result:
            section_chunks = self._build_section_chunks(
                document_structure_result=document_structure_result,
                page_raws=page_raws,
            )
            chunks.extend(section_chunks)

        if self.config.include_paragraph_chunks and document_structure_result:
            paragraph_chunks = self._build_paragraph_chunks(
                document_structure_result=document_structure_result,
            )
            chunks.extend(paragraph_chunks)

        if self.config.include_page_chunks:
            page_chunks = self._build_page_chunks(page_raws)
            chunks.extend(page_chunks)

        chunks = self._deduplicate_chunks(chunks)

        return {
            "processor": "ChunkBuilder",
            "mode": "fallback",
            "chunks": chunks,
            "chunk_summary": {
                "chunk_count": len(chunks),
                "by_type": self._count_by_key(chunks, "chunk_type"),
            },
        }

    def _build_section_chunks(
        self,
        document_structure_result: Dict[str, Any],
        page_raws: List[PageRaw],
    ) -> List[Dict[str, Any]]:
        sections = document_structure_result.get("sections", []) or []
        page_text_by_number = {
            page_raw.page_number: self._page_text(page_raw)
            for page_raw in page_raws
        }

        chunks: List[Dict[str, Any]] = []

        for section in sections:
            if section.get("section_type") == "root":
                continue

            section_id = section.get("section_id", "")
            title = self._clean_text(section.get("title", ""))
            page_numbers = section.get("content_page_numbers", []) or []

            if not page_numbers:
                page_start = section.get("page_start")
                page_end = section.get("page_end")

                if page_start is not None and page_end is not None:
                    page_numbers = list(range(int(page_start), int(page_end) + 1))

            text_parts = []

            for page_number in page_numbers:
                text = page_text_by_number.get(page_number, "")

                if text:
                    text_parts.append(text)

            text = "\n".join(text_parts).strip()

            if not text:
                text = self._clean_text(section.get("text_preview", ""))

            if not text:
                continue

            for part_index, text_part in enumerate(self._split_text(text)):
                chunks.append(
                    {
                        "chunk_id": make_id("chunk"),
                        "chunk_type": "section_chunk",
                        "text": text_part,
                        "normalized_text": self._normalize_text(text_part),
                        "page_numbers": page_numbers,
                        "page_start": min(page_numbers) if page_numbers else section.get("page_start"),
                        "page_end": max(page_numbers) if page_numbers else section.get("page_end"),
                        "section_id": section_id,
                        "section_title": title,
                        "section_level": section.get("level", 0),
                        "order": len(chunks),
                        "source": "fallback_section_chunks",
                        "metadata": {
                            "part_index": part_index,
                            "section_number": section.get("section_number", ""),
                            "parent_id": section.get("parent_id", ""),
                            "char_count": len(text_part),
                        },
                    }
                )

        return chunks

    def _build_paragraph_chunks(
        self,
        document_structure_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        paragraphs = document_structure_result.get("paragraphs", []) or []
        chunks: List[Dict[str, Any]] = []

        for paragraph in paragraphs:
            text = self._clean_text(paragraph.get("text", ""))

            if len(text) < self.config.min_chunk_chars:
                continue

            chunks.append(
                {
                    "chunk_id": make_id("chunk"),
                    "chunk_type": "paragraph_chunk",
                    "text": text[: self.config.max_chunk_chars],
                    "normalized_text": self._normalize_text(text[: self.config.max_chunk_chars]),
                    "page_numbers": [paragraph.get("page_number")] if paragraph.get("page_number") else [],
                    "page_start": paragraph.get("page_number"),
                    "page_end": paragraph.get("page_number"),
                    "section_id": paragraph.get("section_id", ""),
                    "paragraph_id": paragraph.get("paragraph_id", ""),
                    "order": len(chunks),
                    "source": "fallback_paragraph_chunks",
                    "metadata": {
                        "paragraph_type": paragraph.get("paragraph_type", ""),
                        "char_count": len(text),
                    },
                }
            )

        return chunks

    def _build_page_chunks(
        self,
        page_raws: List[PageRaw],
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            text = self._page_text(page_raw)

            if not text:
                continue

            for part_index, text_part in enumerate(self._split_text(text)):
                chunks.append(
                    {
                        "chunk_id": make_id("chunk"),
                        "chunk_type": "page_chunk",
                        "text": text_part,
                        "normalized_text": self._normalize_text(text_part),
                        "page_numbers": [page_raw.page_number],
                        "page_start": page_raw.page_number,
                        "page_end": page_raw.page_number,
                        "section_id": "",
                        "order": len(chunks),
                        "source": "fallback_page_chunks",
                        "metadata": {
                            "page_number": page_raw.page_number,
                            "page_index": page_raw.page_index,
                            "part_index": part_index,
                            "char_count": len(text_part),
                        },
                    }
                )

        return chunks

    def _fallback_parent_child_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_structure_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        parent_child_links = []

        section_parent_by_id = {}

        if document_structure_result:
            for section in document_structure_result.get("sections", []) or []:
                section_id = section.get("section_id", "")
                parent_id = section.get("parent_id", "")

                if section_id:
                    section_parent_by_id[section_id] = parent_id

        chunks_by_section: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in chunks:
            section_id = chunk.get("section_id", "")

            if section_id:
                chunks_by_section.setdefault(section_id, [])
                chunks_by_section[section_id].append(chunk)

        for section_id, section_chunks in chunks_by_section.items():
            parent_section_id = section_parent_by_id.get(section_id, "")

            if not parent_section_id:
                continue

            parent_chunks = chunks_by_section.get(parent_section_id, [])

            for child in section_chunks:
                for parent in parent_chunks[:1]:
                    parent_child_links.append(
                        {
                            "parent_child_id": make_id("parent_child"),
                            "parent_chunk_id": parent.get("chunk_id", ""),
                            "child_chunk_id": child.get("chunk_id", ""),
                            "parent_section_id": parent_section_id,
                            "child_section_id": section_id,
                            "relation_type": "section_parent_child",
                            "confidence": 0.75,
                            "source": "fallback_parent_child_chunks",
                        }
                    )

        return {
            "processor": "ParentChildChunkBuilder",
            "mode": "fallback",
            "parent_child_links": parent_child_links,
            "parent_child_summary": {
                "parent_child_link_count": len(parent_child_links),
            },
        }

    def _fallback_table_chunks(
        self,
        page_raws: List[PageRaw],
        table_understanding_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        table_chunks: List[Dict[str, Any]] = []

        table_semantics = []
        table_records = []
        table_matrices = {}

        if table_understanding_result:
            table_semantics = table_understanding_result.get("table_semantics", []) or []
            table_records = table_understanding_result.get("table_records", []) or []
            table_matrices = (
                table_understanding_result.get("table_matrices_with_spans")
                or table_understanding_result.get("table_matrices")
                or {}
            )

        if not table_semantics:
            for page_raw in page_raws:
                meta = page_raw.metadata.get("table_semantic_recognizer", {})
                table_semantics.extend(meta.get("table_semantics_on_page", []) or [])
                table_records.extend(meta.get("table_records_on_page", []) or [])

        records_by_table: Dict[str, List[Dict[str, Any]]] = {}

        for record in table_records:
            table_grid_id = record.get("table_grid_id", "") or "unknown_table"
            records_by_table.setdefault(table_grid_id, [])
            records_by_table[table_grid_id].append(record)

        for table in table_semantics:
            table_grid_id = table.get("table_grid_id", "")
            page_number = table.get("page_number")
            title = self._clean_text(table.get("title") or table.get("caption") or "")

            lines = []

            if title:
                lines.append(title)

            lines.append(f"Loại bảng: {table.get('semantic_type', table.get('table_type', 'table'))}")

            records = records_by_table.get(table_grid_id, [])

            for record in records[:20]:
                raw_values = record.get("raw_values", {}) or {}
                if raw_values:
                    line = " | ".join(
                        f"{key}: {value}"
                        for key, value in raw_values.items()
                        if str(value).strip()
                    )

                    if line:
                        lines.append(line)

            if len(lines) <= 2 and table_grid_id in table_matrices:
                matrix = table_matrices.get(table_grid_id, [])

                for row in matrix[:20]:
                    parts = []

                    for cell in row:
                        text = self._clean_text(
                            cell.get("normalized_text") or cell.get("text") or ""
                        )

                        if text:
                            parts.append(text)

                    if parts:
                        lines.append(" | ".join(parts))

            text = "\n".join(lines).strip()

            if not text:
                continue

            table_chunks.append(
                {
                    "chunk_id": make_id("tbl_chunk"),
                    "chunk_type": "table_chunk",
                    "text": text[: self.config.max_chunk_chars],
                    "normalized_text": self._normalize_text(text[: self.config.max_chunk_chars]),
                    "page_numbers": [page_number] if page_number else [],
                    "page_start": page_number,
                    "page_end": page_number,
                    "table_grid_id": table_grid_id,
                    "table_semantic_id": table.get("table_semantic_id", ""),
                    "table_structure_id": table.get("table_structure_id", ""),
                    "table_boundary_id": table.get("table_boundary_id", ""),
                    "source": "fallback_table_chunks",
                    "metadata": {
                        "title": title,
                        "table_type": table.get("table_type", ""),
                        "semantic_type": table.get("semantic_type", ""),
                        "record_count": len(records),
                    },
                }
            )

        return {
            "processor": "TableChunkBuilder",
            "mode": "fallback",
            "table_chunks": table_chunks,
            "chunks": table_chunks,
            "table_chunk_summary": {
                "table_chunk_count": len(table_chunks),
            },
        }

    def _fallback_evidence(
        self,
        chunks: List[Dict[str, Any]],
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        evidence_items = []

        for chunk in chunks:
            page_numbers = chunk.get("page_numbers", []) or []

            if not page_numbers and chunk.get("page_start"):
                page_numbers = [chunk.get("page_start")]

            evidence_items.append(
                {
                    "evidence_id": make_id("evidence"),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "evidence_type": chunk.get("chunk_type", "text"),
                    "text": chunk.get("text", ""),
                    "normalized_text": chunk.get("normalized_text", ""),
                    "page_numbers": page_numbers,
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "section_id": chunk.get("section_id", ""),
                    "table_grid_id": chunk.get("table_grid_id", ""),
                    "confidence": 0.70,
                    "source": "fallback_evidence",
                    "metadata": {
                        "chunk_type": chunk.get("chunk_type", ""),
                        "char_count": len(chunk.get("text", "")),
                    },
                }
            )

        return {
            "processor": "EvidenceBuilder",
            "mode": "fallback",
            "evidence": evidence_items,
            "evidence_summary": {
                "evidence_count": len(evidence_items),
                "by_type": self._count_by_key(evidence_items, "evidence_type"),
            },
        }

    def _fallback_citations(
        self,
        chunks: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        citations = []

        for item in evidence:
            page_numbers = item.get("page_numbers", []) or []
            page_text = ""

            if page_numbers:
                if len(page_numbers) == 1:
                    page_text = f"trang {page_numbers[0]}"
                else:
                    page_text = f"trang {min(page_numbers)}-{max(page_numbers)}"

            section_id = item.get("section_id", "")
            table_grid_id = item.get("table_grid_id", "")

            if table_grid_id:
                citation_text = f"Bảng {table_grid_id}, {page_text}".strip(", ")
            elif section_id:
                citation_text = f"Mục {section_id}, {page_text}".strip(", ")
            else:
                citation_text = page_text or "Không rõ vị trí"

            citations.append(
                {
                    "citation_id": make_id("citation"),
                    "evidence_id": item.get("evidence_id", ""),
                    "chunk_id": item.get("chunk_id", ""),
                    "citation_text": citation_text,
                    "page_numbers": page_numbers,
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "section_id": section_id,
                    "table_grid_id": table_grid_id,
                    "confidence": 0.70,
                    "source": "fallback_citations",
                }
            )

        return {
            "processor": "CitationBuilder",
            "mode": "fallback",
            "citations": citations,
            "citation_summary": {
                "citation_count": len(citations),
            },
        }

    def _fallback_knowledge_graph(
        self,
        page_raws: List[PageRaw],
        chunks: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        document_structure_result: Optional[Dict[str, Any]],
        table_understanding_result: Optional[Dict[str, Any]],
        cross_page_context_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        nodes = []
        edges = []

        for page_raw in page_raws:
            nodes.append(
                {
                    "node_id": f"page_{page_raw.page_number}",
                    "node_type": "page",
                    "label": f"Page {page_raw.page_number}",
                    "metadata": {
                        "page_number": page_raw.page_number,
                        "page_index": page_raw.page_index,
                        "word_count": len(page_raw.words),
                    },
                }
            )

        if document_structure_result:
            for section in document_structure_result.get("sections", []) or []:
                section_id = section.get("section_id", "")

                if not section_id:
                    continue

                node_id = f"section_{section_id}"

                nodes.append(
                    {
                        "node_id": node_id,
                        "node_type": "section",
                        "label": section.get("title", ""),
                        "metadata": section,
                    }
                )

                for page_number in section.get("content_page_numbers", []) or []:
                    edges.append(
                        {
                            "edge_id": make_id("kg_edge"),
                            "source_id": node_id,
                            "target_id": f"page_{page_number}",
                            "edge_type": "section_on_page",
                            "confidence": 0.75,
                        }
                    )

                parent_id = section.get("parent_id", "")

                if parent_id:
                    edges.append(
                        {
                            "edge_id": make_id("kg_edge"),
                            "source_id": f"section_{parent_id}",
                            "target_id": node_id,
                            "edge_type": "section_parent_of",
                            "confidence": 0.75,
                        }
                    )

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")

            if not chunk_id:
                continue

            node_id = f"chunk_{chunk_id}"

            nodes.append(
                {
                    "node_id": node_id,
                    "node_type": "chunk",
                    "label": chunk.get("chunk_type", "chunk"),
                    "metadata": {
                        "chunk_id": chunk_id,
                        "chunk_type": chunk.get("chunk_type", ""),
                        "page_numbers": chunk.get("page_numbers", []),
                        "section_id": chunk.get("section_id", ""),
                        "table_grid_id": chunk.get("table_grid_id", ""),
                    },
                }
            )

            for page_number in chunk.get("page_numbers", []) or []:
                edges.append(
                    {
                        "edge_id": make_id("kg_edge"),
                        "source_id": node_id,
                        "target_id": f"page_{page_number}",
                        "edge_type": "chunk_from_page",
                        "confidence": 0.75,
                    }
                )

            section_id = chunk.get("section_id", "")

            if section_id:
                edges.append(
                    {
                        "edge_id": make_id("kg_edge"),
                        "source_id": f"section_{section_id}",
                        "target_id": node_id,
                        "edge_type": "section_contains_chunk",
                        "confidence": 0.75,
                    }
                )

        if table_understanding_result:
            tables = table_understanding_result.get("table_semantics", []) or []

            for table in tables:
                table_id = table.get("table_semantic_id") or table.get("table_grid_id")

                if not table_id:
                    continue

                node_id = f"table_{table_id}"

                nodes.append(
                    {
                        "node_id": node_id,
                        "node_type": "table",
                        "label": table.get("title") or table.get("semantic_type", "table"),
                        "metadata": table,
                    }
                )

                page_number = table.get("page_number")

                if page_number:
                    edges.append(
                        {
                            "edge_id": make_id("kg_edge"),
                            "source_id": node_id,
                            "target_id": f"page_{page_number}",
                            "edge_type": "table_on_page",
                            "confidence": 0.75,
                        }
                    )

        if cross_page_context_result:
            context_graph = cross_page_context_result.get("context_graph", {}) or {}
            graph_nodes = context_graph.get("nodes", []) or []
            graph_edges = context_graph.get("edges", []) or []

            for node in graph_nodes:
                node_id = node.get("node_id", "")

                if not node_id:
                    continue

                nodes.append(
                    {
                        "node_id": f"context_{node_id}",
                        "node_type": f"context_{node.get('node_type', 'node')}",
                        "label": node.get("label", node_id),
                        "metadata": node,
                    }
                )

            for edge in graph_edges:
                edges.append(
                    {
                        "edge_id": make_id("kg_edge"),
                        "source_id": f"context_{edge.get('source_id', '')}",
                        "target_id": f"context_{edge.get('target_id', '')}",
                        "edge_type": f"context_{edge.get('edge_type', 'related_to')}",
                        "confidence": edge.get("confidence", 0.5),
                        "metadata": edge,
                    }
                )

        nodes = self._deduplicate_nodes(nodes)
        edges = self._deduplicate_edges(edges)

        return {
            "processor": "KnowledgeGraphBuilder",
            "mode": "fallback",
            "nodes": nodes,
            "edges": edges,
            "knowledge_graph_summary": {
                "has_knowledge_graph": len(nodes) > 0,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "by_node_type": self._count_by_key(nodes, "node_type"),
                "by_edge_type": self._count_by_key(edges, "edge_type"),
            },
        }

    def _extract_chunks(
        self,
        result: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(result, dict):
            return []

        for key in ["chunks", "document_chunks", "text_chunks", "table_chunks"]:
            value = result.get(key)

            if isinstance(value, list):
                return value

        return []

    def _extract_evidence(
        self,
        result: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(result, dict):
            return []

        for key in ["evidence", "evidence_items", "evidences"]:
            value = result.get(key)

            if isinstance(value, list):
                return value

        return []

    def _extract_citations(
        self,
        result: Any,
    ) -> List[Dict[str, Any]]:
        if not isinstance(result, dict):
            return []

        for key in ["citations", "citation_items"]:
            value = result.get(key)

            if isinstance(value, list):
                return value

        return []

    def _deduplicate_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")

            if not chunk_id:
                chunk_id = make_id("chunk")
                chunk["chunk_id"] = chunk_id

            if chunk_id in seen:
                continue

            seen.add(chunk_id)
            result.append(chunk)

        return result

    def _deduplicate_nodes(
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

    def _deduplicate_edges(
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
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(edge)

        return result

    def _split_text(
        self,
        text: str,
    ) -> List[str]:
        text = self._clean_text_block(text)

        if len(text) <= self.config.max_chunk_chars:
            return [text] if len(text) >= self.config.min_chunk_chars else []

        chunks = []
        start = 0
        length = len(text)

        while start < length:
            end = min(start + self.config.chunk_size_chars, length)
            part = text[start:end]

            if end < length:
                last_break = max(
                    part.rfind("\n"),
                    part.rfind(". "),
                    part.rfind("; "),
                )

                if last_break >= self.config.min_chunk_chars:
                    end = start + last_break + 1
                    part = text[start:end]

            part = part.strip()

            if len(part) >= self.config.min_chunk_chars:
                chunks.append(part)

            if end >= length:
                break

            start = max(end - self.config.chunk_overlap_chars, start + 1)

        return chunks

    def _page_text(
        self,
        page_raw: PageRaw,
    ) -> str:
        reading_meta = page_raw.metadata.get("reading_order_builder", {})
        text = reading_meta.get("reading_order_text", "")

        if text:
            return self._clean_text_block(text)

        return self._clean_text_block(page_raw.normalized_text or page_raw.raw_text or "")

    def _normalize_text(
        self,
        text: Any,
    ) -> str:
        text = self._clean_text_block(text)
        return text.lower()

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

    def _count_by_key(
        self,
        items: List[Dict[str, Any]],
        key: str,
    ) -> Dict[str, int]:
        result: Dict[str, int] = {}

        for item in items:
            value = item.get(key, "unknown") or "unknown"
            result[str(value)] = result.get(str(value), 0) + 1

        return result

    def _build_summary(
        self,
        page_raws: List[PageRaw],
        chunks: List[Dict[str, Any]],
        evidence: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        metadata_result: Dict[str, Any],
        parent_child_result: Dict[str, Any],
        table_chunk_result: Dict[str, Any],
        knowledge_graph_result: Dict[str, Any],
        errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        graph_summary = {}

        if isinstance(knowledge_graph_result, dict):
            graph_summary = knowledge_graph_result.get("knowledge_graph_summary", {}) or {}

        table_chunks = self._extract_chunks(table_chunk_result)

        return {
            "page_count": len(page_raws),
            "chunk_count": len(chunks),
            "table_chunk_count": len(table_chunks),
            "evidence_count": len(evidence),
            "citation_count": len(citations),
            "parent_child_link_count": len(parent_child_result.get("parent_child_links", []) or []) if isinstance(parent_child_result, dict) else 0,
            "knowledge_graph_node_count": graph_summary.get("node_count", 0),
            "knowledge_graph_edge_count": graph_summary.get("edge_count", 0),
            "by_chunk_type": self._count_by_key(chunks, "chunk_type"),
            "by_evidence_type": self._count_by_key(evidence, "evidence_type"),
            "has_metadata_enrichment": bool(metadata_result),
            "has_knowledge_graph": graph_summary.get("has_knowledge_graph", False),
            "error_count": len(errors),
            "has_errors": len(errors) > 0,
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        summary = result.get("knowledge_summary", {})
        chunks = result.get("chunks", []) or []
        evidence = result.get("evidence", []) or []
        citations = result.get("citations", []) or []

        chunks_by_page: Dict[str, List[Dict[str, Any]]] = {}
        evidence_by_page: Dict[str, List[Dict[str, Any]]] = {}
        citations_by_page: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in chunks:
            for page_number in chunk.get("page_numbers", []) or []:
                page_key = str(page_number)
                chunks_by_page.setdefault(page_key, [])
                chunks_by_page[page_key].append(chunk)

        for item in evidence:
            for page_number in item.get("page_numbers", []) or []:
                page_key = str(page_number)
                evidence_by_page.setdefault(page_key, [])
                evidence_by_page[page_key].append(item)

        for item in citations:
            for page_number in item.get("page_numbers", []) or []:
                page_key = str(page_number)
                citations_by_page.setdefault(page_key, [])
                citations_by_page[page_key].append(item)

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("knowledge_pipeline", {})
            page_raw.metadata["knowledge_pipeline"] = {
                "processor": "KnowledgePipeline",
                "chunks_on_page": chunks_by_page.get(page_key, []),
                "evidence_on_page": evidence_by_page.get(page_key, []),
                "citations_on_page": citations_by_page.get(page_key, []),
                "chunk_count_on_page": len(chunks_by_page.get(page_key, [])),
                "evidence_count_on_page": len(evidence_by_page.get(page_key, [])),
                "citation_count_on_page": len(citations_by_page.get(page_key, [])),
                "knowledge_summary": summary,
            }

    def save_knowledge_result(
        self,
        result: Dict[str, Any],
        page_raws: Optional[List[PageRaw]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        if output_path is None:
            document_id = "document"

            if page_raws:
                document_id = page_raws[0].document_id

            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{document_id}_knowledge.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        safe_result = self._json_safe(result)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                safe_result,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return str(output_path)

    def _json_safe(
        self,
        value: Any,
    ) -> Any:
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if is_dataclass(value):
            return self._json_safe(asdict(value))

        if isinstance(value, dict):
            return {
                str(key): self._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self._json_safe(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                self._json_safe(item)
                for item in value
            ]

        if hasattr(value, "to_dict"):
            try:
                return self._json_safe(value.to_dict())
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            try:
                return self._json_safe(vars(value))
            except Exception:
                pass

        return str(value)


def run_knowledge_pipeline(
    page_raws: List[PageRaw],
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    cross_page_context_result: Optional[Dict[str, Any]] = None,
    config: Optional[KnowledgePipelineConfig] = None,
) -> Dict[str, Any]:
    pipeline = KnowledgePipeline(config=config)
    return pipeline.process(
        page_raws=page_raws,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
        cross_page_context_result=cross_page_context_result,
    )


def build_knowledge(
    page_raws: List[PageRaw],
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    cross_page_context_result: Optional[Dict[str, Any]] = None,
    config: Optional[KnowledgePipelineConfig] = None,
) -> Dict[str, Any]:
    return run_knowledge_pipeline(
        page_raws=page_raws,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
        cross_page_context_result=cross_page_context_result,
        config=config,
    )
