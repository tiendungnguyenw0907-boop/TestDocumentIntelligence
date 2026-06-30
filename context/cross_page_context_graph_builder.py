"""
cross_page_context_graph_builder.py

Production V1 - Colab Ready

Purpose
-------
Build a cross-page context graph for a document.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- DocumentStructurePipeline
- TableUnderstandingPipeline
- SectionLinker
- ParagraphContinuationDetector
- TableContinuationDetector
- EntityLinker
- ReferenceLinker

Output
------
Dictionary with:
- nodes
- edges
- adjacency
- nodes_by_type
- edges_by_type
- page_contexts
- context_graph_summary

Graph meaning
-------------
Nodes:
- page
- section
- paragraph_continuation
- table
- multi_page_table
- entity
- reference

Edges:
- section_appears_on_page
- section_parent_of
- paragraph_continues_to
- table_continues_to
- table_appears_on_page
- entity_mentioned_on_page
- reference_from_page
- reference_to_page
- page_next
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class CrossPageContextGraphBuilderConfig:
    include_page_nodes: bool = True
    include_page_sequence_edges: bool = True

    include_section_nodes: bool = True
    include_paragraph_continuation_nodes: bool = True
    include_table_nodes: bool = True
    include_entity_nodes: bool = True
    include_reference_nodes: bool = True

    include_table_understanding_tables: bool = True
    include_multi_page_tables: bool = True

    deduplicate_nodes: bool = True
    deduplicate_edges: bool = True

    attach_to_pages: bool = True
    include_debug: bool = True


@dataclass
class ContextGraphNode:
    node_id: str
    node_type: str
    label: str

    page_number: Optional[int] = None
    page_numbers: Optional[List[int]] = None

    source_id: str = ""
    source: str = "cross_page_context_graph_builder"
    confidence: float = 0.5
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["page_numbers"] is None:
            data["page_numbers"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class ContextGraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str

    source_page: Optional[int] = None
    target_page: Optional[int] = None

    weight: float = 1.0
    confidence: float = 0.5
    source: str = "cross_page_context_graph_builder"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class CrossPageContextGraphBuilder:
    def __init__(
        self,
        config: Optional[CrossPageContextGraphBuilderConfig] = None,
    ):
        self.config = config or CrossPageContextGraphBuilderConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        section_link_result: Optional[Dict[str, Any]] = None,
        paragraph_continuation_result: Optional[Dict[str, Any]] = None,
        table_continuation_result: Optional[Dict[str, Any]] = None,
        entity_link_result: Optional[Dict[str, Any]] = None,
        reference_link_result: Optional[Dict[str, Any]] = None,
        document_structure_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        section_link_result = section_link_result or {}
        paragraph_continuation_result = paragraph_continuation_result or {}
        table_continuation_result = table_continuation_result or {}
        entity_link_result = entity_link_result or {}
        reference_link_result = reference_link_result or {}
        document_structure_result = document_structure_result or {}
        table_understanding_result = table_understanding_result or {}

        nodes: List[ContextGraphNode] = []
        edges: List[ContextGraphEdge] = []

        if self.config.include_page_nodes:
            page_nodes, page_edges = self._build_page_nodes_and_edges(page_raws)
            nodes.extend(page_nodes)
            edges.extend(page_edges)

        if self.config.include_section_nodes:
            section_nodes, section_edges = self._build_section_graph(
                section_link_result=section_link_result,
                document_structure_result=document_structure_result,
            )
            nodes.extend(section_nodes)
            edges.extend(section_edges)

        if self.config.include_paragraph_continuation_nodes:
            paragraph_nodes, paragraph_edges = self._build_paragraph_continuation_graph(
                paragraph_continuation_result=paragraph_continuation_result,
            )
            nodes.extend(paragraph_nodes)
            edges.extend(paragraph_edges)

        if self.config.include_table_nodes:
            table_nodes, table_edges = self._build_table_graph(
                table_continuation_result=table_continuation_result,
                table_understanding_result=table_understanding_result,
            )
            nodes.extend(table_nodes)
            edges.extend(table_edges)

        if self.config.include_entity_nodes:
            entity_nodes, entity_edges = self._build_entity_graph(
                entity_link_result=entity_link_result,
            )
            nodes.extend(entity_nodes)
            edges.extend(entity_edges)

        if self.config.include_reference_nodes:
            reference_nodes, reference_edges = self._build_reference_graph(
                reference_link_result=reference_link_result,
            )
            nodes.extend(reference_nodes)
            edges.extend(reference_edges)

        if self.config.deduplicate_nodes:
            nodes = self._deduplicate_nodes(nodes)

        if self.config.deduplicate_edges:
            edges = self._deduplicate_edges(edges)

        node_dicts = [
            node.to_dict() for node in nodes
        ]

        edge_dicts = [
            edge.to_dict() for edge in edges
        ]

        adjacency = self._build_adjacency(
            nodes=node_dicts,
            edges=edge_dicts,
        )

        page_contexts = self._build_page_contexts(
            page_raws=page_raws,
            nodes=node_dicts,
            edges=edge_dicts,
        )

        result = {
            "processor": "CrossPageContextGraphBuilder",
            "schema_version": "context_graph_v1",
            "nodes": node_dicts,
            "edges": edge_dicts,
            "adjacency": adjacency,
            "nodes_by_type": self._group_nodes_by_type(node_dicts),
            "edges_by_type": self._group_edges_by_type(edge_dicts),
            "page_contexts": page_contexts,
            "context_graph_summary": self._build_summary(
                page_raws=page_raws,
                nodes=node_dicts,
                edges=edge_dicts,
                page_contexts=page_contexts,
            ),
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                result=result,
            )

        return result

    def _build_page_nodes_and_edges(
        self,
        page_raws: List[PageRaw],
    ) -> Tuple[List[ContextGraphNode], List[ContextGraphEdge]]:
        nodes: List[ContextGraphNode] = []
        edges: List[ContextGraphEdge] = []

        sorted_pages = sorted(
            page_raws,
            key=lambda page: page.page_number,
        )

        for page_raw in sorted_pages:
            nodes.append(
                ContextGraphNode(
                    node_id=self._page_node_id(page_raw.page_number),
                    node_type="page",
                    label=f"Page {page_raw.page_number}",
                    page_number=page_raw.page_number,
                    page_numbers=[page_raw.page_number],
                    source_id=f"page_{page_raw.page_number}",
                    confidence=1.0,
                    metadata={
                        "document_id": page_raw.document_id,
                        "page_index": page_raw.page_index,
                        "page_number": page_raw.page_number,
                        "width": page_raw.width,
                        "height": page_raw.height,
                        "rotation": page_raw.rotation,
                        "word_count": len(page_raw.words),
                        "text_line_count": len(page_raw.text_lines),
                        "image_count": len(page_raw.images),
                        "drawing_count": len(page_raw.drawings),
                        "metadata_keys": sorted(list(page_raw.metadata.keys())),
                    },
                )
            )

        if self.config.include_page_sequence_edges:
            for index in range(len(sorted_pages) - 1):
                current_page = sorted_pages[index].page_number
                next_page = sorted_pages[index + 1].page_number

                edges.append(
                    ContextGraphEdge(
                        edge_id=make_id("ctx_edge"),
                        source_id=self._page_node_id(current_page),
                        target_id=self._page_node_id(next_page),
                        edge_type="page_next",
                        source_page=current_page,
                        target_page=next_page,
                        weight=0.20,
                        confidence=1.0,
                        metadata={
                            "from_page": current_page,
                            "to_page": next_page,
                        },
                    )
                )

        return nodes, edges

    def _build_section_graph(
        self,
        section_link_result: Dict[str, Any],
        document_structure_result: Dict[str, Any],
    ) -> Tuple[List[ContextGraphNode], List[ContextGraphEdge]]:
        nodes: List[ContextGraphNode] = []
        edges: List[ContextGraphEdge] = []

        section_links = section_link_result.get("section_links", []) or []

        if not section_links:
            section_links = self._section_links_from_document_structure(
                document_structure_result
            )

        for section in section_links:
            section_id = section.get("section_id", "") or make_id("section_ref")
            node_id = self._section_node_id(section_id)
            title = self._clean_text(section.get("title", "")) or f"Section {section_id}"

            page_numbers = self._resolve_page_numbers(section)

            nodes.append(
                ContextGraphNode(
                    node_id=node_id,
                    node_type="section",
                    label=title,
                    page_numbers=page_numbers,
                    source_id=section_id,
                    confidence=self._safe_float(section.get("confidence", 0.75), default=0.75),
                    metadata={
                        "section_id": section_id,
                        "title": title,
                        "level": section.get("level", 0),
                        "page_start": section.get("page_start"),
                        "page_end": section.get("page_end"),
                        "parent_id": section.get("parent_id", ""),
                        "child_ids": section.get("child_ids", section.get("children", []) or []),
                        "link_type": section.get("link_type", "section_to_pages"),
                        "source": section.get("source", ""),
                    },
                )
            )

            for page_number in page_numbers:
                edges.append(
                    ContextGraphEdge(
                        edge_id=make_id("ctx_edge"),
                        source_id=node_id,
                        target_id=self._page_node_id(page_number),
                        edge_type="section_appears_on_page",
                        source_page=None,
                        target_page=page_number,
                        weight=0.85,
                        confidence=self._safe_float(section.get("confidence", 0.75), default=0.75),
                        metadata={
                            "section_id": section_id,
                            "page_number": page_number,
                        },
                    )
                )

            parent_id = section.get("parent_id", "")

            if parent_id:
                edges.append(
                    ContextGraphEdge(
                        edge_id=make_id("ctx_edge"),
                        source_id=self._section_node_id(parent_id),
                        target_id=node_id,
                        edge_type="section_parent_of",
                        weight=0.75,
                        confidence=0.80,
                        metadata={
                            "parent_section_id": parent_id,
                            "child_section_id": section_id,
                        },
                    )
                )

        return nodes, edges

    def _build_paragraph_continuation_graph(
        self,
        paragraph_continuation_result: Dict[str, Any],
    ) -> Tuple[List[ContextGraphNode], List[ContextGraphEdge]]:
        nodes: List[ContextGraphNode] = []
        edges: List[ContextGraphEdge] = []

        continuations = (
            paragraph_continuation_result.get("paragraph_continuations", [])
            or paragraph_continuation_result.get("continuations", [])
            or []
        )

        for item in continuations:
            continuation_id = (
                item.get("paragraph_continuation_id")
                or item.get("continuation_id")
                or make_id("para_cont")
            )

            from_page = self._safe_int(item.get("from_page"), default=0)
            to_page = self._safe_int(item.get("to_page"), default=0)

            if from_page <= 0 or to_page <= 0:
                continue

            node_id = self._paragraph_continuation_node_id(continuation_id)

            label = f"Paragraph continuation {from_page}→{to_page}"

            nodes.append(
                ContextGraphNode(
                    node_id=node_id,
                    node_type="paragraph_continuation",
                    label=label,
                    page_numbers=[from_page, to_page],
                    source_id=continuation_id,
                    confidence=self._safe_float(item.get("confidence", 0.6), default=0.6),
                    metadata={
                        "paragraph_continuation_id": continuation_id,
                        "from_page": from_page,
                        "to_page": to_page,
                        "from_text_tail": item.get("from_text_tail", ""),
                        "to_text_head": item.get("to_text_head", ""),
                        "continuation_type": item.get("continuation_type", "cross_page_paragraph"),
                        "source": item.get("source", ""),
                    },
                )
            )

            edges.append(
                ContextGraphEdge(
                    edge_id=make_id("ctx_edge"),
                    source_id=self._page_node_id(from_page),
                    target_id=self._page_node_id(to_page),
                    edge_type="paragraph_continues_to",
                    source_page=from_page,
                    target_page=to_page,
                    weight=0.80,
                    confidence=self._safe_float(item.get("confidence", 0.6), default=0.6),
                    metadata={
                        "paragraph_continuation_id": continuation_id,
                        "continuation_node_id": node_id,
                    },
                )
            )

            edges.append(
                ContextGraphEdge(
                    edge_id=make_id("ctx_edge"),
                    source_id=node_id,
                    target_id=self._page_node_id(from_page),
                    edge_type="continuation_starts_on_page",
                    target_page=from_page,
                    weight=0.55,
                    confidence=0.70,
                    metadata={
                        "paragraph_continuation_id": continuation_id,
                    },
                )
            )

            edges.append(
                ContextGraphEdge(
                    edge_id=make_id("ctx_edge"),
                    source_id=node_id,
                    target_id=self._page_node_id(to_page),
                    edge_type="continuation_ends_on_page",
                    target_page=to_page,
                    weight=0.55,
                    confidence=0.70,
                    metadata={
                        "paragraph_continuation_id": continuation_id,
                    },
                )
            )

        return nodes, edges

    def _build_table_graph(
        self,
        table_continuation_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
    ) -> Tuple[List[ContextGraphNode], List[ContextGraphEdge]]:
        nodes: List[ContextGraphNode] = []
        edges: List[ContextGraphEdge] = []

        if self.config.include_table_understanding_tables:
            table_nodes, table_edges = self._build_table_nodes_from_table_understanding(
                table_understanding_result
            )
            nodes.extend(table_nodes)
            edges.extend(table_edges)

        if self.config.include_multi_page_tables:
            multi_nodes, multi_edges = self._build_multi_page_table_graph(
                table_continuation_result=table_continuation_result,
                table_understanding_result=table_understanding_result,
            )
            nodes.extend(multi_nodes)
            edges.extend(multi_edges)

        continuation_edges = self._build_table_continuation_edges(
            table_continuation_result
        )
        edges.extend(continuation_edges)

        return nodes, edges

    def _build_table_nodes_from_table_understanding(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> Tuple[List[ContextGraphNode], List[ContextGraphEdge]]:
        nodes: List[ContextGraphNode] = []
        edges: List[ContextGraphEdge] = []

        tables = (
            table_understanding_result.get("table_semantics", [])
            or table_understanding_result.get("table_structures", [])
            or table_understanding_result.get("table_grids", [])
            or []
        )

        for table in tables:
            table_grid_id = table.get("table_grid_id", "")
            table_semantic_id = table.get("table_semantic_id", "")
            table_structure_id = table.get("table_structure_id", "")

            source_id = table_semantic_id or table_structure_id or table_grid_id

            if not source_id:
                source_id = make_id("table_ref")

            page_number = self._safe_int(table.get("page_number"), default=0)
            page_numbers = [page_number] if page_number > 0 else []

            node_id = self._table_node_id(source_id)

            label = (
                self._clean_text(table.get("title", ""))
                or self._clean_text(table.get("caption", ""))
                or self._clean_text(table.get("semantic_type", ""))
                or self._clean_text(table.get("table_type", ""))
                or f"Table {source_id}"
            )

            nodes.append(
                ContextGraphNode(
                    node_id=node_id,
                    node_type="table",
                    label=label,
                    page_number=page_number if page_number > 0 else None,
                    page_numbers=page_numbers,
                    source_id=source_id,
                    confidence=self._safe_float(table.get("confidence", 0.65), default=0.65),
                    metadata={
                        "table_grid_id": table_grid_id,
                        "table_semantic_id": table_semantic_id,
                        "table_structure_id": table_structure_id,
                        "table_boundary_id": table.get("table_boundary_id", ""),
                        "table_type": table.get("table_type", ""),
                        "semantic_type": table.get("semantic_type", ""),
                        "row_count": table.get("row_count", 0),
                        "col_count": table.get("col_count", 0),
                        "header_rows": table.get("header_rows", table.get("header_row_indices", []) or []),
                        "column_headers": table.get("column_headers", []) or [],
                        "bbox": table.get("bbox", []),
                    },
                )
            )

            if page_number > 0:
                edges.append(
                    ContextGraphEdge(
                        edge_id=make_id("ctx_edge"),
                        source_id=node_id,
                        target_id=self._page_node_id(page_number),
                        edge_type="table_appears_on_page",
                        target_page=page_number,
                        weight=0.85,
                        confidence=self._safe_float(table.get("confidence", 0.65), default=0.65),
                        metadata={
                            "table_grid_id": table_grid_id,
                            "table_semantic_id": table_semantic_id,
                            "table_structure_id": table_structure_id,
                        },
                    )
                )

        return nodes, edges

    def _build_multi_page_table_graph(
        self,
        table_continuation_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
    ) -> Tuple[List[ContextGraphNode], List[ContextGraphEdge]]:
        nodes: List[ContextGraphNode] = []
        edges: List[ContextGraphEdge] = []

        multi_page_tables = (
            table_continuation_result.get("multi_page_tables", [])
            or table_understanding_result.get("multi_page_tables", [])
            or []
        )

        segments = (
            table_continuation_result.get("multi_page_table_segments", [])
            or table_understanding_result.get("multi_page_table_segments", [])
            or []
        )

        segments_by_multi_id: Dict[str, List[Dict[str, Any]]] = {}

        for segment in segments:
            multi_id = segment.get("multi_page_table_id", "")

            if not multi_id:
                continue

            segments_by_multi_id.setdefault(multi_id, [])
            segments_by_multi_id[multi_id].append(segment)

        for table in multi_page_tables:
            multi_id = table.get("multi_page_table_id", "") or make_id("multi_tbl_ref")
            node_id = self._multi_page_table_node_id(multi_id)

            page_numbers = [
                self._safe_int(page, default=0)
                for page in table.get("page_numbers", []) or []
                if self._safe_int(page, default=0) > 0
            ]

            table_segments = segments_by_multi_id.get(multi_id, [])

            if not page_numbers and table_segments:
                page_numbers = [
                    self._safe_int(segment.get("page_number"), default=0)
                    for segment in table_segments
                    if self._safe_int(segment.get("page_number"), default=0) > 0
                ]

            page_numbers = sorted(list(dict.fromkeys(page_numbers)))

            label = f"Multi-page table {min(page_numbers)}-{max(page_numbers)}" if page_numbers else f"Multi-page table {multi_id}"

            nodes.append(
                ContextGraphNode(
                    node_id=node_id,
                    node_type="multi_page_table",
                    label=label,
                    page_numbers=page_numbers,
                    source_id=multi_id,
                    confidence=self._safe_float(table.get("confidence", 0.65), default=0.65),
                    metadata={
                        "multi_page_table_id": multi_id,
                        "page_start": table.get("page_start"),
                        "page_end": table.get("page_end"),
                        "table_grid_ids": table.get("table_grid_ids", []) or [],
                        "table_structure_ids": table.get("table_structure_ids", []) or [],
                        "table_semantic_ids": table.get("table_semantic_ids", []) or [],
                        "column_headers": table.get("column_headers", []) or [],
                        "total_row_count": table.get("total_row_count", 0),
                        "col_count": table.get("col_count", 0),
                        "table_type": table.get("table_type", ""),
                        "semantic_type": table.get("semantic_type", ""),
                    },
                )
            )

            for page_number in page_numbers:
                edges.append(
                    ContextGraphEdge(
                        edge_id=make_id("ctx_edge"),
                        source_id=node_id,
                        target_id=self._page_node_id(page_number),
                        edge_type="multi_page_table_appears_on_page",
                        target_page=page_number,
                        weight=0.90,
                        confidence=self._safe_float(table.get("confidence", 0.65), default=0.65),
                        metadata={
                            "multi_page_table_id": multi_id,
                            "page_number": page_number,
                        },
                    )
                )

            for segment in table_segments:
                table_grid_id = segment.get("table_grid_id", "")
                table_semantic_id = segment.get("table_semantic_id", "")
                table_structure_id = segment.get("table_structure_id", "")
                source_table_id = table_semantic_id or table_structure_id or table_grid_id

                if not source_table_id:
                    continue

                edges.append(
                    ContextGraphEdge(
                        edge_id=make_id("ctx_edge"),
                        source_id=node_id,
                        target_id=self._table_node_id(source_table_id),
                        edge_type="multi_page_table_has_segment",
                        target_page=self._safe_int(segment.get("page_number"), default=0),
                        weight=0.80,
                        confidence=self._safe_float(segment.get("continuation_score", 0.65), default=0.65),
                        metadata={
                            "multi_page_table_id": multi_id,
                            "segment_id": segment.get("segment_id", ""),
                            "table_grid_id": table_grid_id,
                            "segment_index": segment.get("segment_index", 0),
                            "segment_type": segment.get("segment_type", ""),
                        },
                    )
                )

        return nodes, edges

    def _build_table_continuation_edges(
        self,
        table_continuation_result: Dict[str, Any],
    ) -> List[ContextGraphEdge]:
        edges: List[ContextGraphEdge] = []

        continuations = table_continuation_result.get("table_continuations", []) or []

        for item in continuations:
            from_page = self._safe_int(item.get("from_page"), default=0)
            to_page = self._safe_int(item.get("to_page"), default=0)

            if from_page <= 0 or to_page <= 0:
                continue

            edges.append(
                ContextGraphEdge(
                    edge_id=make_id("ctx_edge"),
                    source_id=self._page_node_id(from_page),
                    target_id=self._page_node_id(to_page),
                    edge_type="table_continues_to",
                    source_page=from_page,
                    target_page=to_page,
                    weight=0.90,
                    confidence=self._safe_float(item.get("confidence", 0.65), default=0.65),
                    metadata={
                        "table_continuation_id": item.get("table_continuation_id", ""),
                        "multi_page_table_id": item.get("multi_page_table_id", ""),
                        "table_grid_ids": item.get("table_grid_ids", []) or [],
                        "continuation_type": item.get("continuation_type", "multi_page_table"),
                    },
                )
            )

        return edges

    def _build_entity_graph(
        self,
        entity_link_result: Dict[str, Any],
    ) -> Tuple[List[ContextGraphNode], List[ContextGraphEdge]]:
        nodes: List[ContextGraphNode] = []
        edges: List[ContextGraphEdge] = []

        entities = entity_link_result.get("entities", []) or []
        entity_links = entity_link_result.get("entity_links", []) or []

        linked_entity_ids = {
            item.get("entity_id", "")
            for item in entity_links
            if item.get("entity_id")
        }

        for entity in entities:
            entity_id = entity.get("entity_id", "") or make_id("entity_ref")
            node_id = self._entity_node_id(entity_id)

            page_numbers = [
                self._safe_int(page, default=0)
                for page in entity.get("page_numbers", []) or []
                if self._safe_int(page, default=0) > 0
            ]

            nodes.append(
                ContextGraphNode(
                    node_id=node_id,
                    node_type="entity",
                    label=self._clean_text(entity.get("text", "")) or entity_id,
                    page_numbers=page_numbers,
                    source_id=entity_id,
                    confidence=self._safe_float(entity.get("confidence", 0.55), default=0.55),
                    metadata={
                        "entity_id": entity_id,
                        "entity_type": entity.get("entity_type", ""),
                        "text": entity.get("text", ""),
                        "normalized_text": entity.get("normalized_text", ""),
                        "occurrence_count": entity.get("occurrence_count", 0),
                        "is_cross_page_entity": entity_id in linked_entity_ids or len(page_numbers) > 1,
                    },
                )
            )

            for page_number in page_numbers:
                edges.append(
                    ContextGraphEdge(
                        edge_id=make_id("ctx_edge"),
                        source_id=node_id,
                        target_id=self._page_node_id(page_number),
                        edge_type="entity_mentioned_on_page",
                        target_page=page_number,
                        weight=0.65,
                        confidence=self._safe_float(entity.get("confidence", 0.55), default=0.55),
                        metadata={
                            "entity_id": entity_id,
                            "entity_type": entity.get("entity_type", ""),
                            "page_number": page_number,
                        },
                    )
                )

        for link in entity_links:
            entity_id = link.get("entity_id", "")

            if not entity_id:
                continue

            page_numbers = [
                self._safe_int(page, default=0)
                for page in link.get("page_numbers", []) or []
                if self._safe_int(page, default=0) > 0
            ]

            for index in range(len(page_numbers) - 1):
                edges.append(
                    ContextGraphEdge(
                        edge_id=make_id("ctx_edge"),
                        source_id=self._page_node_id(page_numbers[index]),
                        target_id=self._page_node_id(page_numbers[index + 1]),
                        edge_type="same_entity_cross_page",
                        source_page=page_numbers[index],
                        target_page=page_numbers[index + 1],
                        weight=0.60,
                        confidence=self._safe_float(link.get("confidence", 0.65), default=0.65),
                        metadata={
                            "entity_link_id": link.get("entity_link_id", ""),
                            "entity_id": entity_id,
                            "entity_type": link.get("entity_type", ""),
                            "text": link.get("text", ""),
                        },
                    )
                )

        return nodes, edges

    def _build_reference_graph(
        self,
        reference_link_result: Dict[str, Any],
    ) -> Tuple[List[ContextGraphNode], List[ContextGraphEdge]]:
        nodes: List[ContextGraphNode] = []
        edges: List[ContextGraphEdge] = []

        references = reference_link_result.get("reference_links", []) or []

        for item in references:
            reference_id = item.get("reference_link_id", "") or make_id("ref_link")
            node_id = self._reference_node_id(reference_id)

            from_page = self._safe_int(item.get("from_page"), default=0)
            target_page = self._safe_int(item.get("target_page"), default=0)

            page_numbers = []

            if from_page > 0:
                page_numbers.append(from_page)

            if target_page > 0:
                page_numbers.append(target_page)

            page_numbers = sorted(list(dict.fromkeys(page_numbers)))

            label = self._clean_text(item.get("text", "")) or f"Reference {reference_id}"

            nodes.append(
                ContextGraphNode(
                    node_id=node_id,
                    node_type="reference",
                    label=label,
                    page_numbers=page_numbers,
                    source_id=reference_id,
                    confidence=self._safe_float(item.get("confidence", 0.50), default=0.50),
                    metadata={
                        "reference_link_id": reference_id,
                        "reference_type": item.get("reference_type", ""),
                        "text": item.get("text", ""),
                        "from_page": from_page,
                        "target_page": target_page,
                        "start_char": item.get("start_char"),
                        "end_char": item.get("end_char"),
                    },
                )
            )

            if from_page > 0:
                edges.append(
                    ContextGraphEdge(
                        edge_id=make_id("ctx_edge"),
                        source_id=self._page_node_id(from_page),
                        target_id=node_id,
                        edge_type="reference_from_page",
                        source_page=from_page,
                        weight=0.45,
                        confidence=self._safe_float(item.get("confidence", 0.50), default=0.50),
                        metadata={
                            "reference_link_id": reference_id,
                            "reference_type": item.get("reference_type", ""),
                        },
                    )
                )

            if target_page > 0:
                edges.append(
                    ContextGraphEdge(
                        edge_id=make_id("ctx_edge"),
                        source_id=node_id,
                        target_id=self._page_node_id(target_page),
                        edge_type="reference_to_page",
                        target_page=target_page,
                        weight=0.45,
                        confidence=self._safe_float(item.get("confidence", 0.50), default=0.50),
                        metadata={
                            "reference_link_id": reference_id,
                            "reference_type": item.get("reference_type", ""),
                        },
                    )
                )

            if from_page > 0 and target_page > 0:
                edges.append(
                    ContextGraphEdge(
                        edge_id=make_id("ctx_edge"),
                        source_id=self._page_node_id(from_page),
                        target_id=self._page_node_id(target_page),
                        edge_type="page_references_page",
                        source_page=from_page,
                        target_page=target_page,
                        weight=0.50,
                        confidence=self._safe_float(item.get("confidence", 0.50), default=0.50),
                        metadata={
                            "reference_link_id": reference_id,
                            "reference_type": item.get("reference_type", ""),
                            "text": item.get("text", ""),
                        },
                    )
                )

        return nodes, edges

    def _section_links_from_document_structure(
        self,
        document_structure_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        sections = document_structure_result.get("sections", []) or []
        links: List[Dict[str, Any]] = []

        for section in sections:
            section_id = section.get("section_id", "")

            if not section_id:
                continue

            page_numbers = section.get("content_page_numbers", []) or []

            page_start = section.get("page_start")
            page_end = section.get("page_end")

            if not page_numbers and page_start is not None and page_end is not None:
                try:
                    page_numbers = list(range(int(page_start), int(page_end) + 1))
                except Exception:
                    page_numbers = []

            links.append(
                {
                    "section_link_id": make_id("section_link"),
                    "section_id": section_id,
                    "title": section.get("title", ""),
                    "level": section.get("level", 0),
                    "page_start": page_start,
                    "page_end": page_end,
                    "page_numbers": page_numbers,
                    "parent_id": section.get("parent_id", ""),
                    "child_ids": section.get("children", []) or section.get("child_ids", []) or [],
                    "link_type": "section_to_pages",
                    "confidence": 0.70,
                    "source": "document_structure_result",
                }
            )

        return links

    def _build_adjacency(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        outgoing: Dict[str, List[Dict[str, Any]]] = {}
        incoming: Dict[str, List[Dict[str, Any]]] = {}

        for node in nodes:
            node_id = node.get("node_id", "")
            outgoing.setdefault(node_id, [])
            incoming.setdefault(node_id, [])

        for edge in edges:
            source_id = edge.get("source_id", "")
            target_id = edge.get("target_id", "")

            outgoing.setdefault(source_id, [])
            outgoing[source_id].append(edge)

            incoming.setdefault(target_id, [])
            incoming[target_id].append(edge)

        degree = {}

        node_ids = set(outgoing.keys()).union(set(incoming.keys()))

        for node_id in node_ids:
            degree[node_id] = {
                "in_degree": len(incoming.get(node_id, [])),
                "out_degree": len(outgoing.get(node_id, [])),
                "total_degree": len(incoming.get(node_id, [])) + len(outgoing.get(node_id, [])),
            }

        return {
            "outgoing": outgoing,
            "incoming": incoming,
            "degree": degree,
        }

    def _build_page_contexts(
        self,
        page_raws: List[PageRaw],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        page_contexts: Dict[str, Dict[str, Any]] = {}

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)
            page_node_id = self._page_node_id(page_raw.page_number)

            page_contexts[page_key] = {
                "page_number": page_raw.page_number,
                "page_node_id": page_node_id,
                "nodes": [],
                "edges": [],
                "neighbor_pages": [],
                "sections": [],
                "tables": [],
                "entities": [],
                "references": [],
                "paragraph_continuations": [],
            }

        for node in nodes:
            page_numbers = node.get("page_numbers", []) or []

            if node.get("page_number") and node.get("page_number") not in page_numbers:
                page_numbers.append(node.get("page_number"))

            for page_number in page_numbers:
                page_key = str(page_number)

                if page_key not in page_contexts:
                    continue

                page_contexts[page_key]["nodes"].append(node)

                node_type = node.get("node_type", "")

                if node_type == "section":
                    page_contexts[page_key]["sections"].append(node)
                elif node_type in ["table", "multi_page_table"]:
                    page_contexts[page_key]["tables"].append(node)
                elif node_type == "entity":
                    page_contexts[page_key]["entities"].append(node)
                elif node_type == "reference":
                    page_contexts[page_key]["references"].append(node)
                elif node_type == "paragraph_continuation":
                    page_contexts[page_key]["paragraph_continuations"].append(node)

        for edge in edges:
            source_page = edge.get("source_page")
            target_page = edge.get("target_page")

            page_numbers = []

            if source_page:
                page_numbers.append(source_page)

            if target_page:
                page_numbers.append(target_page)

            if not page_numbers:
                for node_key in ["source_id", "target_id"]:
                    node_id = edge.get(node_key, "")

                    if node_id.startswith("page_"):
                        page_number = self._safe_int(node_id.replace("page_", ""), default=0)

                        if page_number > 0:
                            page_numbers.append(page_number)

            for page_number in sorted(list(dict.fromkeys(page_numbers))):
                page_key = str(page_number)

                if page_key not in page_contexts:
                    continue

                page_contexts[page_key]["edges"].append(edge)

                neighbor = None

                if source_page == page_number and target_page:
                    neighbor = target_page
                elif target_page == page_number and source_page:
                    neighbor = source_page

                if neighbor and neighbor not in page_contexts[page_key]["neighbor_pages"]:
                    page_contexts[page_key]["neighbor_pages"].append(neighbor)

        for page_key in page_contexts:
            ctx = page_contexts[page_key]
            ctx["node_count"] = len(ctx["nodes"])
            ctx["edge_count"] = len(ctx["edges"])
            ctx["section_count"] = len(ctx["sections"])
            ctx["table_count"] = len(ctx["tables"])
            ctx["entity_count"] = len(ctx["entities"])
            ctx["reference_count"] = len(ctx["references"])
            ctx["paragraph_continuation_count"] = len(ctx["paragraph_continuations"])
            ctx["neighbor_pages"] = sorted(ctx["neighbor_pages"])

        return page_contexts

    def _group_nodes_by_type(
        self,
        nodes: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for node in nodes:
            node_type = node.get("node_type", "unknown")
            grouped.setdefault(node_type, [])
            grouped[node_type].append(node)

        return grouped

    def _group_edges_by_type(
        self,
        edges: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for edge in edges:
            edge_type = edge.get("edge_type", "unknown")
            grouped.setdefault(edge_type, [])
            grouped[edge_type].append(edge)

        return grouped

    def _deduplicate_nodes(
        self,
        nodes: List[ContextGraphNode],
    ) -> List[ContextGraphNode]:
        seen = set()
        result: List[ContextGraphNode] = []

        for node in nodes:
            if node.node_id in seen:
                continue

            seen.add(node.node_id)
            result.append(node)

        return result

    def _deduplicate_edges(
        self,
        edges: List[ContextGraphEdge],
    ) -> List[ContextGraphEdge]:
        seen = set()
        result: List[ContextGraphEdge] = []

        for edge in edges:
            key = (
                edge.source_id,
                edge.target_id,
                edge.edge_type,
                str(edge.metadata.get("section_id", "")) if edge.metadata else "",
                str(edge.metadata.get("entity_id", "")) if edge.metadata else "",
                str(edge.metadata.get("reference_link_id", "")) if edge.metadata else "",
                str(edge.metadata.get("multi_page_table_id", "")) if edge.metadata else "",
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(edge)

        return result

    def _build_summary(
        self,
        page_raws: List[PageRaw],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        page_contexts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        by_node_type: Dict[str, int] = {}
        by_edge_type: Dict[str, int] = {}

        for node in nodes:
            node_type = node.get("node_type", "unknown")
            by_node_type[node_type] = by_node_type.get(node_type, 0) + 1

        for edge in edges:
            edge_type = edge.get("edge_type", "unknown")
            by_edge_type[edge_type] = by_edge_type.get(edge_type, 0) + 1

        pages_with_context = [
            page_key for page_key, ctx in page_contexts.items()
            if ctx.get("node_count", 0) > 1 or ctx.get("edge_count", 0) > 0
        ]

        return {
            "has_context_graph": len(nodes) > 0,
            "page_count": len(page_raws),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "page_context_count": len(page_contexts),
            "page_count_with_context": len(pages_with_context),
            "section_node_count": by_node_type.get("section", 0),
            "table_node_count": by_node_type.get("table", 0),
            "multi_page_table_node_count": by_node_type.get("multi_page_table", 0),
            "entity_node_count": by_node_type.get("entity", 0),
            "reference_node_count": by_node_type.get("reference", 0),
            "paragraph_continuation_node_count": by_node_type.get("paragraph_continuation", 0),
            "by_node_type": by_node_type,
            "by_edge_type": by_edge_type,
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        page_contexts = result.get("page_contexts", {})
        summary = result.get("context_graph_summary", {})

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("cross_page_context_graph_builder", {})
            page_raw.metadata["cross_page_context_graph_builder"] = {
                "processor": "CrossPageContextGraphBuilder",
                "page_context": page_contexts.get(page_key, {}),
                "context_graph_summary": summary,
            }

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

    def _page_node_id(
        self,
        page_number: int,
    ) -> str:
        return f"page_{page_number}"

    def _section_node_id(
        self,
        section_id: str,
    ) -> str:
        return f"section_{section_id}"

    def _paragraph_continuation_node_id(
        self,
        continuation_id: str,
    ) -> str:
        return f"paragraph_continuation_{continuation_id}"

    def _table_node_id(
        self,
        table_id: str,
    ) -> str:
        return f"table_{table_id}"

    def _multi_page_table_node_id(
        self,
        multi_page_table_id: str,
    ) -> str:
        return f"multi_page_table_{multi_page_table_id}"

    def _entity_node_id(
        self,
        entity_id: str,
    ) -> str:
        return f"entity_{entity_id}"

    def _reference_node_id(
        self,
        reference_id: str,
    ) -> str:
        return f"reference_{reference_id}"

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


def build_cross_page_context_graph(
    page_raws: List[PageRaw],
    section_link_result: Optional[Dict[str, Any]] = None,
    paragraph_continuation_result: Optional[Dict[str, Any]] = None,
    table_continuation_result: Optional[Dict[str, Any]] = None,
    entity_link_result: Optional[Dict[str, Any]] = None,
    reference_link_result: Optional[Dict[str, Any]] = None,
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = CrossPageContextGraphBuilder()
    return builder.process(
        page_raws=page_raws,
        section_link_result=section_link_result,
        paragraph_continuation_result=paragraph_continuation_result,
        table_continuation_result=table_continuation_result,
        entity_link_result=entity_link_result,
        reference_link_result=reference_link_result,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
    )
