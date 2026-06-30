"""
evidence_builder.py

Production V1 - Colab Ready

Purpose
-------
Build evidence items from chunks, tables, document structure, and cross-page context.

Used by:
- KnowledgePipeline
- CitationBuilder
- RAGPipeline
- CitationVerifier

Input
-----
- page_raws
- chunk_result
- table_understanding_result
- document_structure_result
- cross_page_context_result

Output
------
Dictionary with:
- evidence
- evidence_by_page
- evidence_by_section
- evidence_by_type
- evidence_summary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw
from document_ai.schemas.evidence_schema import (
    Evidence,
    EvidenceCollection,
    EvidenceRelation,
    make_id,
    normalize_text,
    normalize_text_for_match,
)


@dataclass
class EvidenceBuilderConfig:
    build_from_chunks: bool = True
    build_from_tables: bool = True
    build_from_sections: bool = True
    build_from_page_text: bool = True
    build_from_context_graph: bool = True

    attach_to_pages: bool = True
    deduplicate_evidence: bool = True

    min_evidence_chars: int = 60
    max_evidence_chars: int = 1200
    max_context_chars: int = 200

    max_page_evidence_per_page: int = 3
    max_section_evidence_per_section: int = 2
    max_table_records_per_table: int = 50

    include_quote: bool = True
    include_context_before_after: bool = True
    include_metadata: bool = True

    evidence_confidence_default: float = 0.70
    table_evidence_confidence: float = 0.72
    chunk_evidence_confidence: float = 0.74
    section_evidence_confidence: float = 0.68
    page_evidence_confidence: float = 0.60
    graph_evidence_confidence: float = 0.62

    include_debug: bool = True


class EvidenceBuilder:
    def __init__(
        self,
        config: Optional[EvidenceBuilderConfig] = None,
    ):
        self.config = config or EvidenceBuilderConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        chunk_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        document_structure_result: Optional[Dict[str, Any]] = None,
        cross_page_context_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        chunk_result = chunk_result or {}
        table_understanding_result = table_understanding_result or {}
        document_structure_result = document_structure_result or {}
        cross_page_context_result = cross_page_context_result or {}

        page_raws = sorted(
            page_raws or [],
            key=lambda item: item.page_number,
        )

        page_text_map = self._build_page_text_map(page_raws)
        evidence_items: List[Evidence] = []
        relations: List[EvidenceRelation] = []

        if self.config.build_from_chunks:
            evidence_items.extend(
                self._build_from_chunks(
                    chunk_result=chunk_result,
                    page_text_map=page_text_map,
                )
            )

        if self.config.build_from_tables:
            evidence_items.extend(
                self._build_from_tables(
                    table_understanding_result=table_understanding_result,
                    page_text_map=page_text_map,
                )
            )

        if self.config.build_from_sections:
            evidence_items.extend(
                self._build_from_sections(
                    document_structure_result=document_structure_result,
                    page_text_map=page_text_map,
                )
            )

        if self.config.build_from_page_text:
            evidence_items.extend(
                self._build_from_page_text(
                    page_raws=page_raws,
                    page_text_map=page_text_map,
                )
            )

        if self.config.build_from_context_graph:
            graph_evidence, graph_relations = self._build_from_context_graph(
                cross_page_context_result=cross_page_context_result,
                page_text_map=page_text_map,
            )
            evidence_items.extend(graph_evidence)
            relations.extend(graph_relations)

        evidence_items = [
            item for item in evidence_items
            if self._is_valid_evidence(item)
        ]

        if self.config.deduplicate_evidence:
            evidence_items = self._deduplicate_evidence(evidence_items)

        evidence_items = self._sort_and_rank_evidence(evidence_items)
        relations.extend(self._build_relations(evidence_items))

        collection = EvidenceCollection(
            document_id=self._infer_document_id(
                page_raws=page_raws,
                chunk_result=chunk_result,
                document_structure_result=document_structure_result,
            ),
            source_document=self._infer_source_document(page_raws),
            evidence=evidence_items,
            relations=relations,
            metadata={
                "processor": "EvidenceBuilder",
                "schema_version": "evidence_builder_v1",
                "chunk_result_available": bool(chunk_result),
                "table_understanding_available": bool(table_understanding_result),
                "document_structure_available": bool(document_structure_result),
                "cross_page_context_available": bool(cross_page_context_result),
            },
        )

        collection.build_citations_from_evidence()
        collection.deduplicate()

        result = collection.to_dict()
        result.update(
            {
                "processor": "EvidenceBuilder",
                "schema_version": "evidence_builder_v1",
                "evidence_items": [
                    item.to_dict()
                    for item in collection.evidence
                ],
                "evidence_relations": [
                    item.to_dict()
                    for item in collection.relations
                ],
                "evidence_builder_summary": self._build_summary(
                    evidence_items=collection.evidence,
                    relations=collection.relations,
                    page_raws=page_raws,
                ),
                "config": asdict(self.config),
            }
        )

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                evidence_items=collection.evidence,
            )

        return result

    def _build_from_chunks(
        self,
        chunk_result: Dict[str, Any],
        page_text_map: Dict[int, str],
    ) -> List[Evidence]:
        evidence_items: List[Evidence] = []
        chunks = self._collect_chunks(chunk_result)

        for chunk in chunks:
            text = normalize_text(chunk.get("text", ""))

            if not text:
                continue

            page_numbers = self._normalize_page_numbers(chunk.get("page_numbers", []))

            if not page_numbers and chunk.get("page_number"):
                page_number = self._safe_int(chunk.get("page_number"), default=0)
                if page_number > 0:
                    page_numbers = [page_number]

            quote = self._make_quote(text)

            if not quote:
                continue

            context_before, context_after = self._context_from_pages(
                quote=quote,
                page_numbers=page_numbers,
                page_text_map=page_text_map,
            )

            evidence_items.append(
                Evidence(
                    evidence_id=make_id("evidence"),
                    evidence_type=self._infer_chunk_evidence_type(chunk),
                    text=text,
                    document_id=chunk.get("document_id", ""),
                    source_document=chunk.get("source_document", ""),
                    chunk_id=chunk.get("chunk_id", ""),
                    source_chunk_id=chunk.get("chunk_id", ""),
                    page_number=min(page_numbers) if page_numbers else None,
                    page_numbers=page_numbers,
                    page_start=min(page_numbers) if page_numbers else None,
                    page_end=max(page_numbers) if page_numbers else None,
                    section_id=chunk.get("section_id", ""),
                    section_title=chunk.get("section_title", ""),
                    paragraph_id=chunk.get("paragraph_id", ""),
                    table_grid_id=chunk.get("table_grid_id", ""),
                    table_structure_id=chunk.get("table_structure_id", ""),
                    table_semantic_id=chunk.get("table_semantic_id", ""),
                    table_boundary_id=chunk.get("table_boundary_id", ""),
                    bbox=chunk.get("bbox", []) or [],
                    quote=quote,
                    context_before=context_before,
                    context_after=context_after,
                    relevance_score=self._safe_float(chunk.get("score"), default=0.0),
                    confidence=self._safe_float(
                        chunk.get("confidence"),
                        default=self.config.chunk_evidence_confidence,
                    ),
                    weight=1.0,
                    rank=0,
                    order=self._safe_int(chunk.get("order"), default=0),
                    source="evidence_builder_from_chunk",
                    extraction_method="chunk_to_evidence",
                    metadata={
                        "chunk_type": chunk.get("chunk_type", ""),
                        "content_hash": chunk.get("content_hash", ""),
                        "token_count": chunk.get("token_count", 0),
                        "word_count": chunk.get("word_count", 0),
                        "char_count": chunk.get("char_count", 0),
                        "chunk_source": chunk.get("source", ""),
                    },
                )
            )

        return evidence_items

    def _build_from_tables(
        self,
        table_understanding_result: Dict[str, Any],
        page_text_map: Dict[int, str],
    ) -> List[Evidence]:
        evidence_items: List[Evidence] = []

        table_semantics = self._collect_table_semantics(table_understanding_result)
        table_records = self._collect_table_records(table_understanding_result)
        multi_page_tables = self._collect_multi_page_tables(table_understanding_result)

        records_by_table = self._group_records_by_table(table_records)

        for table in table_semantics:
            table_id = table.get("table_semantic_id") or table.get("table_grid_id") or ""

            page_numbers = self._normalize_page_numbers(table.get("page_numbers", []))

            if not page_numbers and table.get("page_number"):
                page_number = self._safe_int(table.get("page_number"), default=0)
                if page_number > 0:
                    page_numbers = [page_number]

            table_related_records = records_by_table.get(table_id, [])[: self.config.max_table_records_per_table]

            text = self._table_text(
                table=table,
                records=table_related_records,
            )

            if not text:
                continue

            quote = self._make_quote(text)
            context_before, context_after = self._context_from_pages(
                quote=quote,
                page_numbers=page_numbers,
                page_text_map=page_text_map,
            )

            evidence_items.append(
                Evidence(
                    evidence_id=make_id("table_evidence"),
                    evidence_type="table_evidence",
                    text=text,
                    document_id=table.get("document_id", ""),
                    source_document=table.get("source_document", ""),
                    page_number=min(page_numbers) if page_numbers else None,
                    page_numbers=page_numbers,
                    page_start=min(page_numbers) if page_numbers else None,
                    page_end=max(page_numbers) if page_numbers else None,
                    section_id=table.get("section_id", ""),
                    section_title=table.get("section_title", ""),
                    table_grid_id=table.get("table_grid_id", ""),
                    table_structure_id=table.get("table_structure_id", ""),
                    table_semantic_id=table.get("table_semantic_id", ""),
                    table_boundary_id=table.get("table_boundary_id", ""),
                    bbox=table.get("bbox", []) or [],
                    quote=quote,
                    context_before=context_before,
                    context_after=context_after,
                    relevance_score=0.0,
                    confidence=self._safe_float(
                        table.get("confidence"),
                        default=self.config.table_evidence_confidence,
                    ),
                    weight=1.15,
                    source="evidence_builder_from_table_semantic",
                    extraction_method="table_semantic_to_evidence",
                    metadata={
                        "table_id": table_id,
                        "semantic_type": table.get("semantic_type", ""),
                        "table_type": table.get("table_type", ""),
                        "row_count": table.get("row_count", 0),
                        "col_count": table.get("col_count", 0),
                        "column_headers": table.get("column_headers", []),
                        "record_count": len(table_related_records),
                    },
                )
            )

        for record in table_records[:]:
            text = self._record_text(record)

            if not text:
                continue

            page_numbers = self._normalize_page_numbers(record.get("page_numbers", []))

            if not page_numbers and record.get("page_number"):
                page_number = self._safe_int(record.get("page_number"), default=0)
                if page_number > 0:
                    page_numbers = [page_number]

            evidence_items.append(
                Evidence(
                    evidence_id=make_id("record_evidence"),
                    evidence_type="table_record_evidence",
                    text=text,
                    document_id=record.get("document_id", ""),
                    source_document=record.get("source_document", ""),
                    page_number=min(page_numbers) if page_numbers else None,
                    page_numbers=page_numbers,
                    page_start=min(page_numbers) if page_numbers else None,
                    page_end=max(page_numbers) if page_numbers else None,
                    table_grid_id=record.get("table_grid_id", ""),
                    table_semantic_id=record.get("table_semantic_id", ""),
                    quote=self._make_quote(text),
                    confidence=self._safe_float(
                        record.get("confidence"),
                        default=self.config.table_evidence_confidence,
                    ),
                    weight=1.05,
                    source="evidence_builder_from_table_record",
                    extraction_method="table_record_to_evidence",
                    metadata={
                        "table_record_id": record.get("table_record_id", ""),
                        "row_index": record.get("row_index", 0),
                        "record_index": record.get("record_index", 0),
                        "cell_ids": record.get("cell_ids", []),
                    },
                )
            )

        for multi_table in multi_page_tables:
            text = self._multi_page_table_text(multi_table)

            if not text:
                continue

            page_numbers = self._normalize_page_numbers(multi_table.get("page_numbers", []))

            evidence_items.append(
                Evidence(
                    evidence_id=make_id("multi_table_evidence"),
                    evidence_type="multi_page_table_evidence",
                    text=text,
                    document_id=multi_table.get("document_id", ""),
                    source_document=multi_table.get("source_document", ""),
                    page_number=min(page_numbers) if page_numbers else None,
                    page_numbers=page_numbers,
                    page_start=min(page_numbers) if page_numbers else None,
                    page_end=max(page_numbers) if page_numbers else None,
                    quote=self._make_quote(text),
                    confidence=self._safe_float(
                        multi_table.get("confidence"),
                        default=self.config.table_evidence_confidence,
                    ),
                    weight=1.20,
                    source="evidence_builder_from_multi_page_table",
                    extraction_method="multi_page_table_to_evidence",
                    metadata={
                        "multi_page_table_id": multi_table.get("multi_page_table_id", ""),
                        "table_grid_ids": multi_table.get("table_grid_ids", []),
                        "table_semantic_ids": multi_table.get("table_semantic_ids", []),
                        "total_row_count": multi_table.get("total_row_count", 0),
                        "col_count": multi_table.get("col_count", 0),
                        "column_headers": multi_table.get("column_headers", []),
                    },
                )
            )

        return evidence_items

    def _build_from_sections(
        self,
        document_structure_result: Dict[str, Any],
        page_text_map: Dict[int, str],
    ) -> List[Evidence]:
        evidence_items: List[Evidence] = []

        sections = document_structure_result.get("sections", []) or []
        paragraphs = document_structure_result.get("paragraphs", []) or []
        paragraphs_by_section = self._group_paragraphs_by_section(paragraphs)

        for section in sections:
            if not isinstance(section, dict):
                section = self._to_dict(section)

            section_id = section.get("section_id", "") or section.get("id", "")
            title = normalize_text(section.get("title", "") or section.get("heading", ""))

            if not section_id and not title:
                continue

            section_paragraphs = paragraphs_by_section.get(section_id, [])
            page_numbers = self._resolve_page_numbers(section)

            parts = []

            if title:
                parts.append(title)

            for paragraph in section_paragraphs[: self.config.max_section_evidence_per_section]:
                text = normalize_text(paragraph.get("text", ""))

                if text:
                    parts.append(text)

            if len(parts) <= 1:
                text_preview = normalize_text(section.get("text_preview", ""))
                if text_preview:
                    parts.append(text_preview)

            text = normalize_text("\n\n".join(parts))

            if not text:
                continue

            quote = self._make_quote(text)
            context_before, context_after = self._context_from_pages(
                quote=quote,
                page_numbers=page_numbers,
                page_text_map=page_text_map,
            )

            evidence_items.append(
                Evidence(
                    evidence_id=make_id("section_evidence"),
                    evidence_type="section_evidence",
                    text=text,
                    document_id=document_structure_result.get("document_id", ""),
                    source_document=document_structure_result.get("source_document", ""),
                    page_number=min(page_numbers) if page_numbers else None,
                    page_numbers=page_numbers,
                    page_start=min(page_numbers) if page_numbers else None,
                    page_end=max(page_numbers) if page_numbers else None,
                    section_id=section_id,
                    section_title=title,
                    quote=quote,
                    context_before=context_before,
                    context_after=context_after,
                    confidence=self._safe_float(
                        section.get("confidence"),
                        default=self.config.section_evidence_confidence,
                    ),
                    weight=1.0,
                    source="evidence_builder_from_section",
                    extraction_method="section_to_evidence",
                    metadata={
                        "level": section.get("level", 0),
                        "order": section.get("order", 0),
                        "parent_id": section.get("parent_id", ""),
                        "paragraph_count": len(section_paragraphs),
                    },
                )
            )

        return evidence_items

    def _build_from_page_text(
        self,
        page_raws: List[PageRaw],
        page_text_map: Dict[int, str],
    ) -> List[Evidence]:
        evidence_items: List[Evidence] = []

        for page_raw in page_raws:
            text = normalize_text(page_text_map.get(page_raw.page_number, ""))

            if not text:
                continue

            pieces = self._split_page_text_for_evidence(text)

            for piece_index, piece in enumerate(pieces[: self.config.max_page_evidence_per_page]):
                if not piece:
                    continue

                quote = self._make_quote(piece)

                evidence_items.append(
                    Evidence(
                        evidence_id=make_id("page_evidence"),
                        evidence_type="page_text_evidence",
                        text=piece,
                        document_id=page_raw.document_id,
                        source_document=page_raw.source_document,
                        page_number=page_raw.page_number,
                        page_numbers=[page_raw.page_number],
                        page_start=page_raw.page_number,
                        page_end=page_raw.page_number,
                        quote=quote,
                        confidence=self.config.page_evidence_confidence,
                        weight=0.85,
                        rank=0,
                        order=piece_index,
                        source="evidence_builder_from_page_text",
                        extraction_method="page_text_to_evidence",
                        metadata={
                            "page_index": page_raw.page_index,
                            "piece_index": piece_index,
                            "piece_count": len(pieces),
                            "page_summary": page_raw.summary() if hasattr(page_raw, "summary") else {},
                        },
                    )
                )

        return evidence_items

    def _build_from_context_graph(
        self,
        cross_page_context_result: Dict[str, Any],
        page_text_map: Dict[int, str],
    ) -> Tuple[List[Evidence], List[EvidenceRelation]]:
        evidence_items: List[Evidence] = []
        relations: List[EvidenceRelation] = []

        context_graph = cross_page_context_result.get("context_graph", {}) or {}
        nodes = context_graph.get("nodes", []) or cross_page_context_result.get("nodes", []) or []
        edges = context_graph.get("edges", []) or cross_page_context_result.get("edges", []) or []

        for node in nodes:
            if not isinstance(node, dict):
                continue

            node_type = node.get("node_type", "")
            label = normalize_text(node.get("label", ""))
            text = normalize_text(node.get("text", "") or label)

            if node_type not in [
                "entity",
                "reference",
                "section",
                "table",
                "multi_page_table",
                "paragraph_continuation",
            ]:
                continue

            if not text:
                continue

            page_numbers = self._normalize_page_numbers(node.get("page_numbers", []))

            if not page_numbers and node.get("page_number"):
                page_number = self._safe_int(node.get("page_number"), default=0)
                if page_number > 0:
                    page_numbers = [page_number]

            evidence_items.append(
                Evidence(
                    evidence_id=make_id("graph_evidence"),
                    evidence_type=f"{node_type}_graph_evidence",
                    text=text,
                    page_number=min(page_numbers) if page_numbers else None,
                    page_numbers=page_numbers,
                    page_start=min(page_numbers) if page_numbers else None,
                    page_end=max(page_numbers) if page_numbers else None,
                    quote=self._make_quote(text),
                    confidence=self._safe_float(
                        node.get("confidence"),
                        default=self.config.graph_evidence_confidence,
                    ),
                    weight=self._safe_float(node.get("weight"), default=0.80),
                    source="evidence_builder_from_context_graph",
                    extraction_method="context_graph_node_to_evidence",
                    metadata={
                        "node_id": node.get("node_id", ""),
                        "node_type": node_type,
                        "source_id": node.get("source_id", ""),
                        "source_type": node.get("source_type", ""),
                        "graph_metadata": node.get("metadata", {}),
                    },
                )
            )

        evidence_by_source_node = {}

        for evidence in evidence_items:
            node_id = evidence.metadata.get("node_id", "")
            if node_id:
                evidence_by_source_node[node_id] = evidence.evidence_id

        for edge in edges:
            if not isinstance(edge, dict):
                continue

            source_node_id = edge.get("source_id", "")
            target_node_id = edge.get("target_id", "")

            source_evidence_id = evidence_by_source_node.get(source_node_id, "")
            target_evidence_id = evidence_by_source_node.get(target_node_id, "")

            if not source_evidence_id or not target_evidence_id:
                continue

            relations.append(
                EvidenceRelation(
                    evidence_relation_id=make_id("evidence_rel"),
                    source_evidence_id=source_evidence_id,
                    target_evidence_id=target_evidence_id,
                    relation_type=edge.get("edge_type", "related_to"),
                    source_page=edge.get("source_page"),
                    target_page=edge.get("target_page"),
                    confidence=self._safe_float(edge.get("confidence"), default=0.60),
                    weight=self._safe_float(edge.get("weight"), default=1.0),
                    source="evidence_builder_from_context_graph_edge",
                    metadata={
                        "edge_id": edge.get("edge_id", ""),
                        "edge_type": edge.get("edge_type", ""),
                        "graph_metadata": edge.get("metadata", {}),
                    },
                )
            )

        return evidence_items, relations

    def _build_relations(
        self,
        evidence_items: List[Evidence],
    ) -> List[EvidenceRelation]:
        relations: List[EvidenceRelation] = []

        by_section: Dict[str, List[Evidence]] = {}
        by_chunk: Dict[str, List[Evidence]] = {}
        by_table: Dict[str, List[Evidence]] = {}

        for item in evidence_items:
            if item.section_id:
                by_section.setdefault(item.section_id, [])
                by_section[item.section_id].append(item)

            if item.chunk_id:
                by_chunk.setdefault(item.chunk_id, [])
                by_chunk[item.chunk_id].append(item)

            table_key = item.table_semantic_id or item.table_grid_id

            if table_key:
                by_table.setdefault(table_key, [])
                by_table[table_key].append(item)

        for section_id, items in by_section.items():
            relations.extend(
                self._sequential_relations(
                    items=items,
                    relation_type="same_section",
                    source="evidence_builder_section_relation",
                )
            )

        for chunk_id, items in by_chunk.items():
            if len(items) >= 2:
                relations.extend(
                    self._pair_relations(
                        items=items,
                        relation_type="same_chunk",
                        source="evidence_builder_chunk_relation",
                    )
                )

        for table_id, items in by_table.items():
            if len(items) >= 2:
                relations.extend(
                    self._pair_relations(
                        items=items,
                        relation_type="same_table",
                        source="evidence_builder_table_relation",
                    )
                )

        return self._deduplicate_relations(relations)

    def _sequential_relations(
        self,
        items: List[Evidence],
        relation_type: str,
        source: str,
    ) -> List[EvidenceRelation]:
        items = sorted(
            items,
            key=lambda item: (
                min(item.page_numbers) if item.page_numbers else 999999,
                item.order,
                item.rank,
            ),
        )

        relations = []

        for index in range(len(items) - 1):
            current = items[index]
            next_item = items[index + 1]

            relations.append(
                EvidenceRelation(
                    evidence_relation_id=make_id("evidence_rel"),
                    source_evidence_id=current.evidence_id,
                    target_evidence_id=next_item.evidence_id,
                    relation_type=relation_type,
                    source_page=current.page_number,
                    target_page=next_item.page_number,
                    confidence=0.65,
                    weight=1.0,
                    source=source,
                    metadata={
                        "sequence_index": index,
                    },
                )
            )

        return relations

    def _pair_relations(
        self,
        items: List[Evidence],
        relation_type: str,
        source: str,
    ) -> List[EvidenceRelation]:
        relations = []

        items = sorted(
            items,
            key=lambda item: (
                min(item.page_numbers) if item.page_numbers else 999999,
                item.order,
                item.rank,
            ),
        )

        for index in range(len(items) - 1):
            current = items[index]
            next_item = items[index + 1]

            relations.append(
                EvidenceRelation(
                    evidence_relation_id=make_id("evidence_rel"),
                    source_evidence_id=current.evidence_id,
                    target_evidence_id=next_item.evidence_id,
                    relation_type=relation_type,
                    source_page=current.page_number,
                    target_page=next_item.page_number,
                    confidence=0.60,
                    weight=1.0,
                    source=source,
                    metadata={
                        "pair_index": index,
                    },
                )
            )

        return relations

    def _collect_chunks(
        self,
        chunk_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

        for key in [
            "chunks",
            "table_chunks",
            "parent_chunks",
            "child_chunks",
        ]:
            values = chunk_result.get(key, []) or []

            if isinstance(values, list):
                chunks.extend(
                    [
                        self._to_dict(item)
                        for item in values
                    ]
                )

        chunk_collection = chunk_result.get("chunk_collection", {}) or {}

        if isinstance(chunk_collection, dict):
            for key in ["chunks", "table_chunks"]:
                values = chunk_collection.get(key, []) or []

                if isinstance(values, list):
                    chunks.extend(
                        [
                            self._to_dict(item)
                            for item in values
                        ]
                    )

        return self._deduplicate_dicts(chunks, ["chunk_id", "content_hash"])

    def _collect_table_semantics(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        for key in ["table_semantics"]:
            values = table_understanding_result.get(key, []) or []
            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        sub = table_understanding_result.get("table_semantic_result", {}) or {}
        if isinstance(sub, dict):
            values = sub.get("table_semantics", []) or []
            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(items, ["table_semantic_id", "table_grid_id"])

    def _collect_table_records(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        for key in ["table_records"]:
            values = table_understanding_result.get(key, []) or []
            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        sub = table_understanding_result.get("table_semantic_result", {}) or {}
        if isinstance(sub, dict):
            values = sub.get("table_records", []) or []
            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(items, ["table_record_id"])

    def _collect_multi_page_tables(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        values = table_understanding_result.get("multi_page_tables", []) or []
        if isinstance(values, list):
            items.extend([self._to_dict(item) for item in values])

        sub = table_understanding_result.get("multi_page_table_result", {}) or {}
        if isinstance(sub, dict):
            values = sub.get("multi_page_tables", []) or []
            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(items, ["multi_page_table_id"])

    def _group_records_by_table(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for record in records:
            keys = [
                record.get("table_semantic_id", ""),
                record.get("table_grid_id", ""),
            ]

            for key in keys:
                if not key:
                    continue

                grouped.setdefault(key, [])
                grouped[key].append(record)

        return grouped

    def _group_paragraphs_by_section(
        self,
        paragraphs: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                paragraph = self._to_dict(paragraph)

            section_id = paragraph.get("section_id", "")

            if not section_id:
                continue

            grouped.setdefault(section_id, [])
            grouped[section_id].append(paragraph)

        return grouped

    def _table_text(
        self,
        table: Dict[str, Any],
        records: List[Dict[str, Any]],
    ) -> str:
        parts = []

        title = normalize_text(table.get("title", ""))
        caption = normalize_text(table.get("caption", ""))

        if title:
            parts.append(title)

        if caption and caption != title:
            parts.append(caption)

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

        for record in records:
            record_text = self._record_text(record)
            if record_text:
                parts.append(record_text)

        if not parts and table.get("text"):
            parts.append(normalize_text(table.get("text")))

        return normalize_text("\n".join(parts))

    def _record_text(
        self,
        record: Dict[str, Any],
    ) -> str:
        values = record.get("values", {}) or record.get("raw_values", {}) or {}

        if isinstance(values, dict) and values:
            parts = []

            for key, value in values.items():
                value_text = normalize_text(value)

                if value_text:
                    parts.append(f"{key}: {value_text}")

            return normalize_text(" | ".join(parts))

        text = normalize_text(record.get("text", ""))

        return text

    def _multi_page_table_text(
        self,
        table: Dict[str, Any],
    ) -> str:
        parts = []

        headers = table.get("column_headers", []) or []
        page_numbers = self._normalize_page_numbers(table.get("page_numbers", []))

        if page_numbers:
            parts.append(f"Bảng nhiều trang: trang {page_numbers[0]}-{page_numbers[-1]}")

        if headers:
            parts.append(
                " | ".join(
                    [
                        normalize_text(item)
                        for item in headers
                        if normalize_text(item)
                    ]
                )
            )

        total_row_count = table.get("total_row_count")
        col_count = table.get("col_count")

        if total_row_count or col_count:
            parts.append(f"Số dòng: {total_row_count or 0}; số cột: {col_count or 0}")

        semantic_type = normalize_text(table.get("semantic_type", ""))

        if semantic_type:
            parts.append(f"Loại bảng: {semantic_type}")

        return normalize_text("\n".join(parts))

    def _split_page_text_for_evidence(
        self,
        text: str,
    ) -> List[str]:
        text = normalize_text(text)

        if not text:
            return []

        paragraphs = [
            normalize_text(item)
            for item in re.split(r"\n\s*\n", text)
            if normalize_text(item)
        ]

        if not paragraphs:
            paragraphs = [
                normalize_text(item)
                for item in re.split(r"\n", text)
                if normalize_text(item)
            ]

        pieces = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.config.max_evidence_chars:
                if current:
                    pieces.append(current)
                    current = ""

                pieces.extend(self._hard_split(paragraph, self.config.max_evidence_chars))
                continue

            if not current:
                current = paragraph
                continue

            candidate = current + "\n\n" + paragraph

            if len(candidate) <= self.config.max_evidence_chars:
                current = candidate
            else:
                pieces.append(current)
                current = paragraph

        if current:
            pieces.append(current)

        return [
            item for item in pieces
            if len(normalize_text(item)) >= self.config.min_evidence_chars
        ]

    def _hard_split(
        self,
        text: str,
        max_chars: int,
    ) -> List[str]:
        result = []
        start = 0

        while start < len(text):
            end = min(start + max_chars, len(text))

            if end < len(text):
                break_point = max(
                    text.rfind(". ", start, end),
                    text.rfind("; ", start, end),
                    text.rfind("\n", start, end),
                    text.rfind(" ", start, end),
                )

                if break_point > start + int(max_chars * 0.55):
                    end = break_point + 1

            piece = normalize_text(text[start:end])

            if piece:
                result.append(piece)

            if end >= len(text):
                break

            start = end

        return result

    def _make_quote(
        self,
        text: str,
    ) -> str:
        text = normalize_text(text)

        if not self.config.include_quote:
            return ""

        if len(text) <= self.config.max_evidence_chars:
            return text

        cut = text[: self.config.max_evidence_chars]

        break_point = max(
            cut.rfind(". "),
            cut.rfind("; "),
            cut.rfind("\n"),
            cut.rfind(" "),
        )

        if break_point > self.config.max_evidence_chars * 0.55:
            cut = cut[:break_point]

        return normalize_text(cut) + "..."

    def _context_from_pages(
        self,
        quote: str,
        page_numbers: List[int],
        page_text_map: Dict[int, str],
    ) -> Tuple[str, str]:
        if not self.config.include_context_before_after:
            return "", ""

        quote_norm = normalize_text_for_match(quote)

        if not quote_norm or not page_numbers:
            return "", ""

        for page_number in page_numbers:
            page_text = normalize_text(page_text_map.get(page_number, ""))

            if not page_text:
                continue

            page_norm = normalize_text_for_match(page_text)

            index = page_norm.find(quote_norm[: min(len(quote_norm), 120)])

            if index < 0:
                continue

            raw_index = min(index, len(page_text))

            before = normalize_text(page_text[max(0, raw_index - self.config.max_context_chars):raw_index])
            after = normalize_text(page_text[raw_index + len(quote):raw_index + len(quote) + self.config.max_context_chars])

            return before, after

        return "", ""

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

    def _infer_chunk_evidence_type(
        self,
        chunk: Dict[str, Any],
    ) -> str:
        chunk_type = chunk.get("chunk_type", "")

        if chunk.get("table_grid_id") or chunk_type == "table_chunk":
            return "table_chunk_evidence"

        if chunk_type == "section_chunk":
            return "section_chunk_evidence"

        if chunk_type == "paragraph_chunk":
            return "paragraph_chunk_evidence"

        if chunk_type == "page_chunk":
            return "page_chunk_evidence"

        if chunk_type == "parent_chunk":
            return "parent_chunk_evidence"

        return "chunk_evidence"

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

    def _is_valid_evidence(
        self,
        evidence: Evidence,
    ) -> bool:
        text = normalize_text(evidence.text or evidence.quote)

        if not text:
            return False

        if len(text) < self.config.min_evidence_chars:
            word_count = len(re.findall(r"\w+", text))

            if word_count < 8:
                return False

        return True

    def _deduplicate_evidence(
        self,
        evidence_items: List[Evidence],
    ) -> List[Evidence]:
        seen = set()
        result = []

        for item in evidence_items:
            key = (
                item.evidence_type,
                normalize_text_for_match(item.text)[:500],
                tuple(item.page_numbers),
                item.section_id,
                item.chunk_id,
                item.table_grid_id,
                item.table_semantic_id,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _deduplicate_relations(
        self,
        relations: List[EvidenceRelation],
    ) -> List[EvidenceRelation]:
        seen = set()
        result = []

        for relation in relations:
            key = (
                relation.source_evidence_id,
                relation.target_evidence_id,
                relation.relation_type,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(relation)

        return result

    def _sort_and_rank_evidence(
        self,
        evidence_items: List[Evidence],
    ) -> List[Evidence]:
        evidence_items = sorted(
            evidence_items,
            key=lambda item: (
                min(item.page_numbers) if item.page_numbers else 999999,
                self._evidence_type_order(item.evidence_type),
                item.section_id,
                item.order,
                -item.confidence,
            ),
        )

        for index, item in enumerate(evidence_items):
            item.rank = index + 1

        return evidence_items

    def _evidence_type_order(
        self,
        evidence_type: str,
    ) -> int:
        order_map = {
            "section_chunk_evidence": 10,
            "paragraph_chunk_evidence": 20,
            "table_evidence": 30,
            "table_record_evidence": 31,
            "multi_page_table_evidence": 32,
            "page_chunk_evidence": 40,
            "page_text_evidence": 50,
            "section_evidence": 60,
        }

        return order_map.get(evidence_type, 99)

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        evidence_items: List[Evidence],
    ) -> None:
        evidence_by_page: Dict[str, List[Dict[str, Any]]] = {}

        for item in evidence_items:
            for page_number in item.page_numbers:
                page_key = str(page_number)
                evidence_by_page.setdefault(page_key, [])
                evidence_by_page[page_key].append(item.to_dict())

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("evidence_builder", {})
            page_raw.metadata["evidence_builder"] = {
                "processor": "EvidenceBuilder",
                "evidence_on_page": evidence_by_page.get(page_key, []),
                "evidence_count_on_page": len(evidence_by_page.get(page_key, [])),
            }

    def _build_summary(
        self,
        evidence_items: List[Evidence],
        relations: List[EvidenceRelation],
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        by_page: Dict[str, int] = {}
        by_section: Dict[str, int] = {}

        total_chars = 0
        total_words = 0

        for item in evidence_items:
            by_type[item.evidence_type] = by_type.get(item.evidence_type, 0) + 1

            total_chars += len(item.text)
            total_words += len(re.findall(r"\S+", item.text))

            for page_number in item.page_numbers:
                page_key = str(page_number)
                by_page[page_key] = by_page.get(page_key, 0) + 1

            section_key = item.section_id or "no_section"
            by_section[section_key] = by_section.get(section_key, 0) + 1

        by_relation_type: Dict[str, int] = {}

        for relation in relations:
            by_relation_type[relation.relation_type] = by_relation_type.get(relation.relation_type, 0) + 1

        return {
            "has_evidence": len(evidence_items) > 0,
            "evidence_count": len(evidence_items),
            "relation_count": len(relations),
            "page_count": len(page_raws),
            "page_count_with_evidence": len(by_page),
            "total_chars": total_chars,
            "total_words": total_words,
            "avg_chars_per_evidence": round(total_chars / max(len(evidence_items), 1), 2),
            "avg_words_per_evidence": round(total_words / max(len(evidence_items), 1), 2),
            "by_evidence_type": by_type,
            "by_relation_type": by_relation_type,
            "by_page": by_page,
            "by_section": by_section,
        }

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
            return dict(vars(value))

        return {}

    def _infer_document_id(
        self,
        page_raws: List[PageRaw],
        chunk_result: Dict[str, Any],
        document_structure_result: Dict[str, Any],
    ) -> str:
        if chunk_result.get("document_id"):
            return chunk_result.get("document_id", "")

        if document_structure_result.get("document_id"):
            return document_structure_result.get("document_id", "")

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


def build_evidence(
    page_raws: List[PageRaw],
    chunk_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    document_structure_result: Optional[Dict[str, Any]] = None,
    cross_page_context_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = EvidenceBuilder()
    return builder.process(
        page_raws=page_raws,
        chunk_result=chunk_result,
        table_understanding_result=table_understanding_result,
        document_structure_result=document_structure_result,
        cross_page_context_result=cross_page_context_result,
    )
