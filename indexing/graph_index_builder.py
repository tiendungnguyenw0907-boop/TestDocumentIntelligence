"""
graph_index_builder.py

Production V1 - Colab Ready

Purpose
-------
Build a pure-Python graph index from knowledge graph / context graph.

Used by:
- KnowledgePipeline
- GraphRetriever
- HybridRetriever
- RAGPipeline

Input
-----
- page_raws
- knowledge_result
- cross_page_context_result
- document_structure_result
- table_understanding_result
- chunk_result
- evidence_result

Output
------
Dictionary with:
- graph_index
- node_store
- edge_store
- adjacency
- reverse_adjacency
- undirected_adjacency
- nodes_by_type
- edges_by_type
- nodes_by_page
- edges_by_page
- graph_index_summary
"""

from __future__ import annotations

import json
import math
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from document_ai.schemas.page_raw_schema import PageRaw
from document_ai.schemas.page_raw_schema import (
    normalize_pdf_text,
    normalize_text_for_match,
    json_safe,
)


@dataclass
class GraphIndexBuilderConfig:
    include_knowledge_graph: bool = True
    include_cross_page_context_graph: bool = True
    include_document_structure: bool = True
    include_chunks: bool = True
    include_tables: bool = True
    include_evidence: bool = True
    include_pages: bool = True

    attach_to_pages: bool = True
    deduplicate_nodes: bool = True
    deduplicate_edges: bool = True

    build_adjacency: bool = True
    build_reverse_adjacency: bool = True
    build_undirected_adjacency: bool = True
    build_type_indexes: bool = True
    build_page_indexes: bool = True
    build_text_index: bool = True
    build_degree_stats: bool = True

    node_text_preview_chars: int = 800
    edge_text_preview_chars: int = 300
    max_neighbors_per_node: int = 1000

    min_token_length: int = 2
    max_token_length: int = 64
    remove_stopwords: bool = True

    include_edge_metadata: bool = True
    include_node_metadata: bool = True
    include_debug: bool = True


class GraphIndexBuilder:
    def __init__(
        self,
        config: Optional[GraphIndexBuilderConfig] = None,
    ):
        self.config = config or GraphIndexBuilderConfig()
        self.stopwords = self._default_stopwords()

    def process(
        self,
        page_raws: Optional[List[PageRaw]] = None,
        knowledge_result: Optional[Dict[str, Any]] = None,
        cross_page_context_result: Optional[Dict[str, Any]] = None,
        document_structure_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        chunk_result: Optional[Dict[str, Any]] = None,
        evidence_result: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        page_raws = sorted(
            page_raws or [],
            key=lambda item: item.page_number,
        )

        knowledge_result = knowledge_result or {}
        cross_page_context_result = cross_page_context_result or {}
        document_structure_result = document_structure_result or {}
        table_understanding_result = table_understanding_result or {}
        chunk_result = chunk_result or {}
        evidence_result = evidence_result or {}

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        if self.config.include_knowledge_graph:
            kg_nodes, kg_edges = self._collect_graph_from_result(
                source=knowledge_result,
                source_name="knowledge_graph",
            )
            nodes.extend(kg_nodes)
            edges.extend(kg_edges)

        if self.config.include_cross_page_context_graph:
            cg_nodes, cg_edges = self._collect_graph_from_result(
                source=cross_page_context_result,
                source_name="cross_page_context_graph",
            )
            nodes.extend(cg_nodes)
            edges.extend(cg_edges)

        synthetic_nodes, synthetic_edges = self._build_synthetic_graph(
            page_raws=page_raws,
            document_structure_result=document_structure_result,
            table_understanding_result=table_understanding_result,
            chunk_result=chunk_result,
            evidence_result=evidence_result,
        )

        nodes.extend(synthetic_nodes)
        edges.extend(synthetic_edges)

        if self.config.deduplicate_nodes:
            nodes = self._deduplicate_nodes(nodes)

        node_store = self._build_node_store(nodes)

        if self.config.deduplicate_edges:
            edges = self._deduplicate_edges(edges)

        edges = self._filter_edges_with_existing_nodes(
            edges=edges,
            node_store=node_store,
        )

        edge_store = self._build_edge_store(edges)

        adjacency = {}
        reverse_adjacency = {}
        undirected_adjacency = {}

        if self.config.build_adjacency:
            adjacency = self._build_adjacency(
                edge_store=edge_store,
                direction="forward",
            )

        if self.config.build_reverse_adjacency:
            reverse_adjacency = self._build_adjacency(
                edge_store=edge_store,
                direction="reverse",
            )

        if self.config.build_undirected_adjacency:
            undirected_adjacency = self._build_undirected_adjacency(edge_store)

        nodes_by_type = {}
        edges_by_type = {}
        nodes_by_page = {}
        edges_by_page = {}

        if self.config.build_type_indexes:
            nodes_by_type = self._build_nodes_by_type(node_store)
            edges_by_type = self._build_edges_by_type(edge_store)

        if self.config.build_page_indexes:
            nodes_by_page = self._build_nodes_by_page(node_store)
            edges_by_page = self._build_edges_by_page(edge_store)

        text_index = {}
        if self.config.build_text_index:
            text_index = self._build_text_index(node_store)

        degree_stats = {}
        if self.config.build_degree_stats:
            degree_stats = self._build_degree_stats(
                node_store=node_store,
                adjacency=adjacency,
                reverse_adjacency=reverse_adjacency,
                undirected_adjacency=undirected_adjacency,
            )

        graph_index = {
            "index_type": "graph",
            "schema_version": "graph_index_builder_v1",
            "config": asdict(self.config),
            "node_count": len(node_store),
            "edge_count": len(edge_store),
            "node_store": node_store,
            "edge_store": edge_store,
            "adjacency": adjacency,
            "reverse_adjacency": reverse_adjacency,
            "undirected_adjacency": undirected_adjacency,
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "nodes_by_page": nodes_by_page,
            "edges_by_page": edges_by_page,
            "text_index": text_index,
            "degree_stats": degree_stats,
        }

        result = {
            "processor": "GraphIndexBuilder",
            "schema_version": "graph_index_builder_v1",
            "graph_index": graph_index,
            "node_store": node_store,
            "edge_store": edge_store,
            "adjacency": adjacency,
            "reverse_adjacency": reverse_adjacency,
            "undirected_adjacency": undirected_adjacency,
            "nodes_by_type": nodes_by_type,
            "edges_by_type": edges_by_type,
            "nodes_by_page": nodes_by_page,
            "edges_by_page": edges_by_page,
            "text_index": text_index,
            "degree_stats": degree_stats,
            "graph_index_summary": self._build_summary(
                node_store=node_store,
                edge_store=edge_store,
                nodes_by_type=nodes_by_type,
                edges_by_type=edges_by_type,
                nodes_by_page=nodes_by_page,
                degree_stats=degree_stats,
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
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        expand_neighbors: bool = True,
        max_neighbors: int = 20,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}

        graph_index = index_result.get("graph_index", index_result) or {}
        node_store = graph_index.get("node_store", {}) or index_result.get("node_store", {}) or {}
        text_index = graph_index.get("text_index", {}) or index_result.get("text_index", {}) or {}
        undirected_adjacency = graph_index.get("undirected_adjacency", {}) or index_result.get("undirected_adjacency", {}) or {}
        edge_store = graph_index.get("edge_store", {}) or index_result.get("edge_store", {}) or {}

        query_tokens = self._tokenize(query)
        query_terms = list(dict.fromkeys(query_tokens))

        candidate_scores: Dict[str, float] = {}
        matched_terms: Dict[str, List[str]] = {}

        for term in query_terms:
            postings = text_index.get(term, []) or []

            for posting in postings:
                node_id = posting.get("node_id", "")

                if not node_id:
                    continue

                node = node_store.get(node_id, {})

                if not self._passes_filters(node, filters):
                    continue

                tf = posting.get("tf", 1)
                score = math.log(1 + tf)

                node_weight = self._safe_float(node.get("weight"), default=1.0)
                node_confidence = self._safe_float(node.get("confidence"), default=0.7)

                score = score * max(node_weight, 0.1) * max(node_confidence, 0.1)

                candidate_scores[node_id] = candidate_scores.get(node_id, 0.0) + score
                matched_terms.setdefault(node_id, [])

                if term not in matched_terms[node_id]:
                    matched_terms[node_id].append(term)

        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        results = []

        for rank, (node_id, score) in enumerate(ranked, start=1):
            node = node_store.get(node_id, {})

            neighbors = []

            if expand_neighbors:
                neighbors = self._expand_neighbors(
                    node_id=node_id,
                    node_store=node_store,
                    edge_store=edge_store,
                    undirected_adjacency=undirected_adjacency,
                    max_neighbors=max_neighbors,
                )

            results.append(
                {
                    "rank": rank,
                    "score": round(score, 6),
                    "node_id": node_id,
                    "node_type": node.get("node_type", ""),
                    "label": node.get("label", ""),
                    "text_preview": self._make_snippet(
                        text=node.get("text", ""),
                        query_terms=matched_terms.get(node_id, []),
                    ),
                    "matched_terms": matched_terms.get(node_id, []),
                    "page_number": node.get("page_number"),
                    "page_numbers": node.get("page_numbers", []),
                    "source_id": node.get("source_id", ""),
                    "source_type": node.get("source_type", ""),
                    "section_id": node.get("section_id", ""),
                    "chunk_id": node.get("chunk_id", ""),
                    "table_id": node.get("table_id", ""),
                    "evidence_id": node.get("evidence_id", ""),
                    "neighbors": neighbors,
                    "metadata": node.get("metadata", {}),
                }
            )

        return results

    def expand(
        self,
        index_result: Dict[str, Any],
        node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
        max_nodes: int = 100,
    ) -> Dict[str, Any]:
        graph_index = index_result.get("graph_index", index_result) or {}
        node_store = graph_index.get("node_store", {}) or {}
        edge_store = graph_index.get("edge_store", {}) or {}
        undirected_adjacency = graph_index.get("undirected_adjacency", {}) or {}

        edge_type_set = set(edge_types or [])

        visited = set()
        frontier = [(node_id, 0) for node_id in node_ids if node_id in node_store]

        expanded_nodes = {}
        expanded_edges = {}

        while frontier and len(expanded_nodes) < max_nodes:
            node_id, current_depth = frontier.pop(0)

            if node_id in visited:
                continue

            visited.add(node_id)

            if node_id in node_store:
                expanded_nodes[node_id] = node_store[node_id]

            if current_depth >= depth:
                continue

            neighbors = undirected_adjacency.get(node_id, []) or []

            for item in neighbors:
                edge_id = item.get("edge_id", "")
                neighbor_id = item.get("neighbor_id", "")

                edge = edge_store.get(edge_id, {})

                if edge_type_set and edge.get("edge_type") not in edge_type_set:
                    continue

                if edge_id:
                    expanded_edges[edge_id] = edge

                if neighbor_id and neighbor_id not in visited:
                    frontier.append((neighbor_id, current_depth + 1))

        return {
            "seed_node_ids": node_ids,
            "depth": depth,
            "node_count": len(expanded_nodes),
            "edge_count": len(expanded_edges),
            "nodes": list(expanded_nodes.values()),
            "edges": list(expanded_edges.values()),
        }

    def _collect_graph_from_result(
        self,
        source: Dict[str, Any],
        source_name: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        nodes = []
        edges = []

        graph_candidates = []

        if isinstance(source.get("knowledge_graph"), dict):
            graph_candidates.append(source.get("knowledge_graph", {}))

        if isinstance(source.get("context_graph"), dict):
            graph_candidates.append(source.get("context_graph", {}))

        if source.get("nodes") or source.get("edges"):
            graph_candidates.append(source)

        for graph in graph_candidates:
            for node in graph.get("nodes", []) or []:
                node = self._to_dict(node)

                if not node:
                    continue

                node.setdefault("source", source_name)
                nodes.append(node)

            for edge in graph.get("edges", []) or []:
                edge = self._to_dict(edge)

                if not edge:
                    continue

                edge.setdefault("source", source_name)
                edges.append(edge)

        return nodes, edges

    def _build_synthetic_graph(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        chunk_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        nodes = []
        edges = []

        page_nodes = {}
        section_nodes = {}
        chunk_nodes = {}
        table_nodes = {}
        evidence_nodes = {}

        if self.config.include_pages:
            page_nodes = self._build_page_nodes(page_raws)
            nodes.extend(page_nodes.values())

            sorted_pages = sorted(page_nodes.keys())

            for index in range(len(sorted_pages) - 1):
                current_page = sorted_pages[index]
                next_page = sorted_pages[index + 1]

                edges.append(
                    self._make_edge(
                        source_id=page_nodes[current_page]["node_id"],
                        target_id=page_nodes[next_page]["node_id"],
                        edge_type="page_next",
                        relation_label="next page",
                        source_page=current_page,
                        target_page=next_page,
                        page_numbers=[current_page, next_page],
                        source="graph_index_builder_synthetic_page",
                    )
                )

        if self.config.include_document_structure:
            section_nodes = self._build_section_nodes(document_structure_result)
            nodes.extend(section_nodes.values())

            for section_id, section_node in section_nodes.items():
                for page_number in section_node.get("page_numbers", []):
                    page_node = page_nodes.get(page_number)

                    if page_node:
                        edges.append(
                            self._make_edge(
                                source_id=page_node["node_id"],
                                target_id=section_node["node_id"],
                                edge_type="page_contains_section",
                                relation_label="contains section",
                                source_page=page_number,
                                target_page=page_number,
                                page_numbers=[page_number],
                                source="graph_index_builder_synthetic_section",
                            )
                        )

                parent_id = section_node.get("metadata", {}).get("parent_id", "")

                if parent_id and parent_id in section_nodes:
                    edges.append(
                        self._make_edge(
                            source_id=section_nodes[parent_id]["node_id"],
                            target_id=section_node["node_id"],
                            edge_type="section_parent_of",
                            relation_label="parent section",
                            page_numbers=section_node.get("page_numbers", []),
                            source="graph_index_builder_synthetic_section_hierarchy",
                        )
                    )

        if self.config.include_chunks:
            chunk_nodes = self._build_chunk_nodes(chunk_result)
            nodes.extend(chunk_nodes.values())

            for chunk_id, chunk_node in chunk_nodes.items():
                page_numbers = chunk_node.get("page_numbers", [])

                for page_number in page_numbers:
                    page_node = page_nodes.get(page_number)

                    if page_node:
                        edges.append(
                            self._make_edge(
                                source_id=page_node["node_id"],
                                target_id=chunk_node["node_id"],
                                edge_type="page_contains_chunk",
                                relation_label="contains chunk",
                                source_page=page_number,
                                target_page=page_number,
                                page_numbers=[page_number],
                                source="graph_index_builder_synthetic_chunk",
                            )
                        )

                section_id = chunk_node.get("metadata", {}).get("section_id", "")

                if section_id and section_id in section_nodes:
                    edges.append(
                        self._make_edge(
                            source_id=section_nodes[section_id]["node_id"],
                            target_id=chunk_node["node_id"],
                            edge_type="section_contains_chunk",
                            relation_label="contains chunk",
                            page_numbers=page_numbers,
                            source="graph_index_builder_synthetic_chunk_section",
                        )
                    )

                parent_chunk_id = chunk_node.get("metadata", {}).get("parent_chunk_id", "")
                if parent_chunk_id and parent_chunk_id in chunk_nodes:
                    edges.append(
                        self._make_edge(
                            source_id=chunk_nodes[parent_chunk_id]["node_id"],
                            target_id=chunk_node["node_id"],
                            edge_type="chunk_parent_of",
                            relation_label="parent chunk",
                            page_numbers=page_numbers,
                            source="graph_index_builder_synthetic_chunk_parent",
                        )
                    )

                next_chunk_id = chunk_node.get("metadata", {}).get("next_chunk_id", "")
                if next_chunk_id and next_chunk_id in chunk_nodes:
                    edges.append(
                        self._make_edge(
                            source_id=chunk_node["node_id"],
                            target_id=chunk_nodes[next_chunk_id]["node_id"],
                            edge_type="chunk_next",
                            relation_label="next chunk",
                            page_numbers=page_numbers,
                            source="graph_index_builder_synthetic_chunk_next",
                        )
                    )

        if self.config.include_tables:
            table_nodes = self._build_table_nodes(table_understanding_result)
            nodes.extend(table_nodes.values())

            for table_id, table_node in table_nodes.items():
                page_numbers = table_node.get("page_numbers", [])

                for page_number in page_numbers:
                    page_node = page_nodes.get(page_number)

                    if page_node:
                        edges.append(
                            self._make_edge(
                                source_id=page_node["node_id"],
                                target_id=table_node["node_id"],
                                edge_type="page_contains_table",
                                relation_label="contains table",
                                source_page=page_number,
                                target_page=page_number,
                                page_numbers=[page_number],
                                source="graph_index_builder_synthetic_table",
                            )
                        )

                section_id = table_node.get("metadata", {}).get("section_id", "")

                if section_id and section_id in section_nodes:
                    edges.append(
                        self._make_edge(
                            source_id=section_nodes[section_id]["node_id"],
                            target_id=table_node["node_id"],
                            edge_type="section_contains_table",
                            relation_label="contains table",
                            page_numbers=page_numbers,
                            source="graph_index_builder_synthetic_table_section",
                        )
                    )

        if self.config.include_evidence:
            evidence_nodes = self._build_evidence_nodes(evidence_result)
            nodes.extend(evidence_nodes.values())

            for evidence_id, evidence_node in evidence_nodes.items():
                page_numbers = evidence_node.get("page_numbers", [])

                for page_number in page_numbers:
                    page_node = page_nodes.get(page_number)

                    if page_node:
                        edges.append(
                            self._make_edge(
                                source_id=page_node["node_id"],
                                target_id=evidence_node["node_id"],
                                edge_type="page_has_evidence",
                                relation_label="has evidence",
                                source_page=page_number,
                                target_page=page_number,
                                page_numbers=[page_number],
                                source="graph_index_builder_synthetic_evidence",
                            )
                        )

                chunk_id = evidence_node.get("metadata", {}).get("chunk_id", "")

                if chunk_id and chunk_id in chunk_nodes:
                    edges.append(
                        self._make_edge(
                            source_id=chunk_nodes[chunk_id]["node_id"],
                            target_id=evidence_node["node_id"],
                            edge_type="chunk_supports_evidence",
                            relation_label="supports evidence",
                            page_numbers=page_numbers,
                            source="graph_index_builder_synthetic_evidence_chunk",
                        )
                    )

                table_id = evidence_node.get("metadata", {}).get("table_id", "")

                if table_id and table_id in table_nodes:
                    edges.append(
                        self._make_edge(
                            source_id=table_nodes[table_id]["node_id"],
                            target_id=evidence_node["node_id"],
                            edge_type="table_has_evidence",
                            relation_label="has evidence",
                            page_numbers=page_numbers,
                            source="graph_index_builder_synthetic_evidence_table",
                        )
                    )

        return nodes, edges

    def _build_page_nodes(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[int, Dict[str, Any]]:
        nodes = {}

        for page_raw in page_raws:
            text = self._page_text(page_raw)

            page_number = page_raw.page_number
            node_id = f"page_{page_number}"

            nodes[page_number] = {
                "node_id": node_id,
                "node_type": "page",
                "label": f"Trang {page_number}",
                "text": self._preview(text, self.config.node_text_preview_chars),
                "page_number": page_number,
                "page_numbers": [page_number],
                "source_id": node_id,
                "source_type": "page",
                "source": "graph_index_builder_page",
                "confidence": 1.0,
                "weight": 1.0,
                "metadata": {
                    "page_index": page_raw.page_index,
                    "page_kind": page_raw.page_kind,
                    "width": page_raw.width,
                    "height": page_raw.height,
                    "rotation": page_raw.rotation,
                    "word_count": page_raw.word_count,
                    "image_count": page_raw.image_count,
                    "drawing_count": page_raw.drawing_count,
                    "annotation_count": page_raw.annotation_count,
                    "link_count": page_raw.link_count,
                },
            }

        return nodes

    def _build_section_nodes(
        self,
        document_structure_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        nodes = {}
        sections = self._collect_list(document_structure_result, "sections")

        for section in sections:
            section_id = section.get("section_id") or section.get("id") or ""

            if not section_id:
                continue

            title = normalize_pdf_text(section.get("title") or section.get("heading") or section_id)
            page_numbers = self._resolve_page_numbers(section)

            nodes[section_id] = {
                "node_id": f"section_{section_id}",
                "node_type": "section",
                "label": title,
                "text": self._preview(section.get("text_preview") or title, self.config.node_text_preview_chars),
                "page_number": min(page_numbers) if page_numbers else None,
                "page_numbers": page_numbers,
                "source_id": section_id,
                "source_type": "section",
                "source": "graph_index_builder_section",
                "confidence": self._safe_float(section.get("confidence"), default=0.75),
                "weight": 1.20,
                "metadata": {
                    "section_id": section_id,
                    "title": title,
                    "level": self._safe_int(section.get("level"), default=0),
                    "order": self._safe_int(section.get("order"), default=0),
                    "parent_id": section.get("parent_id", ""),
                    "section_type": section.get("section_type", "section"),
                },
            }

        return nodes

    def _build_chunk_nodes(
        self,
        chunk_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        nodes = {}
        chunks = self._collect_chunks(chunk_result)

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")

            if not chunk_id:
                continue

            text = normalize_pdf_text(chunk.get("text", ""))

            if not text:
                continue

            page_numbers = self._normalize_page_numbers(chunk.get("page_numbers", []))
            table_id = self._table_id(chunk)

            nodes[chunk_id] = {
                "node_id": f"chunk_{chunk_id}",
                "node_type": "chunk",
                "label": self._preview(text, 120),
                "text": self._preview(text, self.config.node_text_preview_chars),
                "page_number": min(page_numbers) if page_numbers else None,
                "page_numbers": page_numbers,
                "source_id": chunk_id,
                "source_type": chunk.get("chunk_type", "chunk"),
                "source": "graph_index_builder_chunk",
                "confidence": self._safe_float(chunk.get("confidence"), default=0.72),
                "weight": 1.0,
                "metadata": {
                    "chunk_id": chunk_id,
                    "chunk_type": chunk.get("chunk_type", ""),
                    "section_id": chunk.get("section_id", ""),
                    "section_title": chunk.get("section_title", ""),
                    "paragraph_id": chunk.get("paragraph_id", ""),
                    "table_id": table_id,
                    "parent_chunk_id": chunk.get("parent_chunk_id", ""),
                    "child_chunk_ids": chunk.get("child_chunk_ids", []) or [],
                    "previous_chunk_id": chunk.get("previous_chunk_id") or chunk.get("prev_chunk_id", ""),
                    "next_chunk_id": chunk.get("next_chunk_id", ""),
                    "order": chunk.get("order", 0),
                    "content_hash": chunk.get("content_hash", ""),
                },
            }

        return nodes

    def _build_table_nodes(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        nodes = {}
        tables = self._collect_tables(table_understanding_result)

        for table in tables:
            table_id = self._table_id(table)

            if not table_id:
                continue

            text = self._table_text(table)
            page_numbers = self._resolve_page_numbers(table)

            if not page_numbers and table.get("page_number"):
                page_number = self._safe_int(table.get("page_number"), default=0)
                if page_number > 0:
                    page_numbers = [page_number]

            label = (
                normalize_pdf_text(table.get("title"))
                or normalize_pdf_text(table.get("caption"))
                or normalize_pdf_text(table.get("caption_text"))
                or table_id
            )

            nodes[table_id] = {
                "node_id": f"table_{table_id}",
                "node_type": "table",
                "label": label,
                "text": self._preview(text, self.config.node_text_preview_chars),
                "page_number": min(page_numbers) if page_numbers else None,
                "page_numbers": page_numbers,
                "source_id": table_id,
                "source_type": self._table_kind(table),
                "source": "graph_index_builder_table",
                "confidence": self._safe_float(table.get("confidence"), default=0.70),
                "weight": 1.15,
                "bbox": table.get("bbox", []) or [],
                "metadata": {
                    "table_id": table_id,
                    "section_id": table.get("section_id", ""),
                    "semantic_type": table.get("semantic_type", ""),
                    "table_type": table.get("table_type", ""),
                    "row_count": table.get("row_count", table.get("total_row_count", 0)),
                    "col_count": table.get("col_count", 0),
                    "column_headers": table.get("column_headers", []) or [],
                    "table_grid_id": table.get("table_grid_id", ""),
                    "table_structure_id": table.get("table_structure_id", ""),
                    "table_semantic_id": table.get("table_semantic_id", ""),
                    "table_boundary_id": table.get("table_boundary_id", ""),
                    "multi_page_table_id": table.get("multi_page_table_id", ""),
                },
            }

        return nodes

    def _build_evidence_nodes(
        self,
        evidence_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        nodes = {}
        evidence_items = self._collect_evidence(evidence_result)

        for evidence in evidence_items:
            evidence_id = evidence.get("evidence_id", "")

            if not evidence_id:
                continue

            text = normalize_pdf_text(evidence.get("text") or evidence.get("quote") or "")

            if not text:
                continue

            page_numbers = self._resolve_page_numbers(evidence)
            table_id = self._table_id(evidence)

            nodes[evidence_id] = {
                "node_id": f"evidence_{evidence_id}",
                "node_type": "evidence",
                "label": self._preview(text, 120),
                "text": self._preview(text, self.config.node_text_preview_chars),
                "page_number": min(page_numbers) if page_numbers else None,
                "page_numbers": page_numbers,
                "source_id": evidence_id,
                "source_type": evidence.get("evidence_type", "evidence"),
                "source": "graph_index_builder_evidence",
                "confidence": self._safe_float(evidence.get("confidence"), default=0.70),
                "weight": self._safe_float(evidence.get("weight"), default=1.0),
                "bbox": evidence.get("bbox", []) or [],
                "metadata": {
                    "evidence_id": evidence_id,
                    "evidence_type": evidence.get("evidence_type", ""),
                    "section_id": evidence.get("section_id", ""),
                    "section_title": evidence.get("section_title", ""),
                    "chunk_id": evidence.get("chunk_id") or evidence.get("source_chunk_id", ""),
                    "table_id": table_id,
                    "relevance_score": evidence.get("relevance_score", 0.0),
                    "rank": evidence.get("rank", 0),
                    "content_hash": evidence.get("content_hash", ""),
                },
            }

        return nodes

    def _build_node_store(
        self,
        nodes: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        node_store = {}

        for node in nodes:
            node = self._normalize_node(node)
            node_id = node.get("node_id", "")

            if not node_id:
                continue

            node_store[node_id] = node

        return node_store

    def _build_edge_store(
        self,
        edges: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        edge_store = {}

        for edge in edges:
            edge = self._normalize_edge(edge)
            edge_id = edge.get("edge_id", "")

            if not edge_id:
                continue

            edge_store[edge_id] = edge

        return edge_store

    def _normalize_node(
        self,
        node: Dict[str, Any],
    ) -> Dict[str, Any]:
        node = self._to_dict(node)

        node_id = (
            node.get("node_id")
            or node.get("id")
            or node.get("source_id")
            or self._stable_id(
                node.get("label", "") + node.get("text", ""),
                "node",
            )
        )

        node_type = normalize_pdf_text(node.get("node_type") or node.get("type") or "node")
        label = normalize_pdf_text(node.get("label") or node.get("title") or node_id)
        text = normalize_pdf_text(node.get("text") or label)

        page_numbers = self._resolve_page_numbers(node)

        if not page_numbers and node.get("page_number"):
            page_number = self._safe_int(node.get("page_number"), default=0)
            if page_number > 0:
                page_numbers = [page_number]

        metadata = node.get("metadata", {}) or {}

        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        normalized = {
            "node_id": str(node_id),
            "node_type": node_type,
            "label": label,
            "text": self._preview(text, self.config.node_text_preview_chars),
            "normalized_text": normalize_text_for_match(f"{label}\n{text}"),
            "page_number": min(page_numbers) if page_numbers else node.get("page_number"),
            "page_numbers": page_numbers,
            "page_start": min(page_numbers) if page_numbers else None,
            "page_end": max(page_numbers) if page_numbers else None,
            "source_id": node.get("source_id", "") or node_id,
            "source_type": node.get("source_type", "") or node_type,
            "source": node.get("source", "") or "graph_index_builder",
            "confidence": self._safe_float(node.get("confidence"), default=0.70),
            "weight": self._safe_float(node.get("weight"), default=1.0),
            "bbox": node.get("bbox", []) or [],
            "section_id": metadata.get("section_id", node.get("section_id", "")),
            "chunk_id": metadata.get("chunk_id", node.get("chunk_id", "")),
            "table_id": metadata.get("table_id", node.get("table_id", "")),
            "evidence_id": metadata.get("evidence_id", node.get("evidence_id", "")),
            "metadata": metadata if self.config.include_node_metadata else {},
        }

        return normalized

    def _normalize_edge(
        self,
        edge: Dict[str, Any],
    ) -> Dict[str, Any]:
        edge = self._to_dict(edge)

        source_id = (
            edge.get("source_id")
            or edge.get("from")
            or edge.get("from_node")
            or edge.get("source")
            or ""
        )

        target_id = (
            edge.get("target_id")
            or edge.get("to")
            or edge.get("to_node")
            or edge.get("target")
            or ""
        )

        edge_type = normalize_pdf_text(edge.get("edge_type") or edge.get("type") or "related_to")
        relation_label = normalize_pdf_text(edge.get("relation_label") or edge_type)

        edge_id = (
            edge.get("edge_id")
            or edge.get("id")
            or self._stable_id(f"{source_id}|{target_id}|{edge_type}", "edge")
        )

        page_numbers = self._resolve_page_numbers(edge)

        if not page_numbers:
            page_numbers = self._normalize_page_numbers(
                [
                    edge.get("source_page"),
                    edge.get("target_page"),
                ]
            )

        metadata = edge.get("metadata", {}) or {}

        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        text = normalize_pdf_text(
            edge.get("text")
            or f"{source_id} {relation_label} {target_id}"
        )

        return {
            "edge_id": str(edge_id),
            "source_id": str(source_id),
            "target_id": str(target_id),
            "edge_type": edge_type,
            "relation_label": relation_label,
            "text": self._preview(text, self.config.edge_text_preview_chars),
            "page_numbers": page_numbers,
            "source_page": edge.get("source_page"),
            "target_page": edge.get("target_page"),
            "confidence": self._safe_float(edge.get("confidence"), default=0.70),
            "weight": self._safe_float(edge.get("weight"), default=1.0),
            "directed": bool(edge.get("directed", True)),
            "source": edge.get("source", "") or "graph_index_builder",
            "metadata": metadata if self.config.include_edge_metadata else {},
        }

    def _filter_edges_with_existing_nodes(
        self,
        edges: List[Dict[str, Any]],
        node_store: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result = []

        for edge in edges:
            source_id = (
                edge.get("source_id")
                or edge.get("from")
                or edge.get("source")
                or ""
            )
            target_id = (
                edge.get("target_id")
                or edge.get("to")
                or edge.get("target")
                or ""
            )

            if source_id in node_store and target_id in node_store:
                result.append(edge)

        return result

    def _build_adjacency(
        self,
        edge_store: Dict[str, Dict[str, Any]],
        direction: str = "forward",
    ) -> Dict[str, List[Dict[str, Any]]]:
        adjacency: Dict[str, List[Dict[str, Any]]] = {}

        for edge_id, edge in edge_store.items():
            if direction == "reverse":
                from_node = edge.get("target_id", "")
                to_node = edge.get("source_id", "")
            else:
                from_node = edge.get("source_id", "")
                to_node = edge.get("target_id", "")

            if not from_node or not to_node:
                continue

            adjacency.setdefault(from_node, [])

            if len(adjacency[from_node]) >= self.config.max_neighbors_per_node:
                continue

            adjacency[from_node].append(
                {
                    "edge_id": edge_id,
                    "neighbor_id": to_node,
                    "edge_type": edge.get("edge_type", ""),
                    "relation_label": edge.get("relation_label", ""),
                    "weight": edge.get("weight", 1.0),
                    "confidence": edge.get("confidence", 0.70),
                    "direction": direction,
                }
            )

        return adjacency

    def _build_undirected_adjacency(
        self,
        edge_store: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        adjacency: Dict[str, List[Dict[str, Any]]] = {}

        for edge_id, edge in edge_store.items():
            source_id = edge.get("source_id", "")
            target_id = edge.get("target_id", "")

            if not source_id or not target_id:
                continue

            adjacency.setdefault(source_id, [])
            adjacency.setdefault(target_id, [])

            if len(adjacency[source_id]) < self.config.max_neighbors_per_node:
                adjacency[source_id].append(
                    {
                        "edge_id": edge_id,
                        "neighbor_id": target_id,
                        "edge_type": edge.get("edge_type", ""),
                        "relation_label": edge.get("relation_label", ""),
                        "weight": edge.get("weight", 1.0),
                        "confidence": edge.get("confidence", 0.70),
                        "direction": "out",
                    }
                )

            if len(adjacency[target_id]) < self.config.max_neighbors_per_node:
                adjacency[target_id].append(
                    {
                        "edge_id": edge_id,
                        "neighbor_id": source_id,
                        "edge_type": edge.get("edge_type", ""),
                        "relation_label": edge.get("relation_label", ""),
                        "weight": edge.get("weight", 1.0),
                        "confidence": edge.get("confidence", 0.70),
                        "direction": "in",
                    }
                )

        return adjacency

    def _build_nodes_by_type(
        self,
        node_store: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}

        for node_id, node in node_store.items():
            node_type = node.get("node_type", "node")
            grouped.setdefault(node_type, [])
            grouped[node_type].append(node_id)

        return grouped

    def _build_edges_by_type(
        self,
        edge_store: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}

        for edge_id, edge in edge_store.items():
            edge_type = edge.get("edge_type", "related_to")
            grouped.setdefault(edge_type, [])
            grouped[edge_type].append(edge_id)

        return grouped

    def _build_nodes_by_page(
        self,
        node_store: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}

        for node_id, node in node_store.items():
            page_numbers = self._normalize_page_numbers(node.get("page_numbers", []))

            for page_number in page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(node_id)

        return grouped

    def _build_edges_by_page(
        self,
        edge_store: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}

        for edge_id, edge in edge_store.items():
            page_numbers = self._normalize_page_numbers(edge.get("page_numbers", []))

            for page_number in page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(edge_id)

        return grouped

    def _build_text_index(
        self,
        node_store: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        text_index: Dict[str, List[Dict[str, Any]]] = {}

        for node_id, node in node_store.items():
            text = normalize_pdf_text(
                f"{node.get('label', '')}\n{node.get('text', '')}"
            )

            tokens = self._tokenize(text)
            freq = {}

            for token in tokens:
                freq[token] = freq.get(token, 0) + 1

            for term, tf in freq.items():
                text_index.setdefault(term, [])
                text_index[term].append(
                    {
                        "node_id": node_id,
                        "tf": tf,
                        "node_type": node.get("node_type", ""),
                        "page_numbers": node.get("page_numbers", []),
                    }
                )

        return text_index

    def _build_degree_stats(
        self,
        node_store: Dict[str, Dict[str, Any]],
        adjacency: Dict[str, List[Dict[str, Any]]],
        reverse_adjacency: Dict[str, List[Dict[str, Any]]],
        undirected_adjacency: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        degree_stats = {}

        for node_id in node_store:
            out_degree = len(adjacency.get(node_id, []) or [])
            in_degree = len(reverse_adjacency.get(node_id, []) or [])
            total_degree = len(undirected_adjacency.get(node_id, []) or [])

            degree_stats[node_id] = {
                "node_id": node_id,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": total_degree,
                "weighted_degree": round(
                    sum(
                        self._safe_float(item.get("weight"), default=1.0)
                        for item in undirected_adjacency.get(node_id, []) or []
                    ),
                    4,
                ),
            }

        return degree_stats

    def _expand_neighbors(
        self,
        node_id: str,
        node_store: Dict[str, Dict[str, Any]],
        edge_store: Dict[str, Dict[str, Any]],
        undirected_adjacency: Dict[str, List[Dict[str, Any]]],
        max_neighbors: int = 20,
    ) -> List[Dict[str, Any]]:
        neighbors = []

        for item in (undirected_adjacency.get(node_id, []) or [])[:max_neighbors]:
            neighbor_id = item.get("neighbor_id", "")
            edge_id = item.get("edge_id", "")

            neighbor = node_store.get(neighbor_id, {})
            edge = edge_store.get(edge_id, {})

            if not neighbor:
                continue

            neighbors.append(
                {
                    "node_id": neighbor_id,
                    "node_type": neighbor.get("node_type", ""),
                    "label": neighbor.get("label", ""),
                    "text_preview": self._preview(neighbor.get("text", ""), 250),
                    "page_numbers": neighbor.get("page_numbers", []),
                    "edge_id": edge_id,
                    "edge_type": edge.get("edge_type", item.get("edge_type", "")),
                    "relation_label": edge.get("relation_label", item.get("relation_label", "")),
                    "direction": item.get("direction", ""),
                    "weight": item.get("weight", 1.0),
                    "confidence": item.get("confidence", 0.70),
                }
            )

        return neighbors

    def _deduplicate_nodes(
        self,
        nodes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result_by_id = {}

        for node in nodes:
            node = self._normalize_node(node)
            node_id = node.get("node_id", "")

            if not node_id:
                continue

            if node_id not in result_by_id:
                result_by_id[node_id] = node
            else:
                result_by_id[node_id] = self._merge_nodes(result_by_id[node_id], node)

        return list(result_by_id.values())

    def _deduplicate_edges(
        self,
        edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result_by_key = {}

        for edge in edges:
            edge = self._normalize_edge(edge)

            key = (
                edge.get("source_id", ""),
                edge.get("target_id", ""),
                edge.get("edge_type", ""),
            )

            if key not in result_by_key:
                result_by_key[key] = edge
            else:
                existing = result_by_key[key]
                existing["confidence"] = max(
                    self._safe_float(existing.get("confidence"), default=0.0),
                    self._safe_float(edge.get("confidence"), default=0.0),
                )
                existing["weight"] = max(
                    self._safe_float(existing.get("weight"), default=0.0),
                    self._safe_float(edge.get("weight"), default=0.0),
                )

                existing_pages = self._normalize_page_numbers(existing.get("page_numbers", []))
                edge_pages = self._normalize_page_numbers(edge.get("page_numbers", []))
                existing["page_numbers"] = sorted(list(dict.fromkeys(existing_pages + edge_pages)))

        return list(result_by_key.values())

    def _merge_nodes(
        self,
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(existing)

        if len(incoming.get("text", "")) > len(existing.get("text", "")):
            merged["text"] = incoming.get("text", "")

        if len(incoming.get("label", "")) > len(existing.get("label", "")):
            merged["label"] = incoming.get("label", "")

        merged["confidence"] = max(
            self._safe_float(existing.get("confidence"), default=0.0),
            self._safe_float(incoming.get("confidence"), default=0.0),
        )

        merged["weight"] = max(
            self._safe_float(existing.get("weight"), default=0.0),
            self._safe_float(incoming.get("weight"), default=0.0),
        )

        pages = self._normalize_page_numbers(existing.get("page_numbers", [])) + self._normalize_page_numbers(incoming.get("page_numbers", []))
        merged["page_numbers"] = sorted(list(dict.fromkeys(pages)))

        existing_metadata = existing.get("metadata", {}) or {}
        incoming_metadata = incoming.get("metadata", {}) or {}

        if isinstance(existing_metadata, dict) and isinstance(incoming_metadata, dict):
            merged["metadata"] = {
                **incoming_metadata,
                **existing_metadata,
            }

        return merged

    def _build_summary(
        self,
        node_store: Dict[str, Dict[str, Any]],
        edge_store: Dict[str, Dict[str, Any]],
        nodes_by_type: Dict[str, List[str]],
        edges_by_type: Dict[str, List[str]],
        nodes_by_page: Dict[str, List[str]],
        degree_stats: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        top_degree_nodes = sorted(
            degree_stats.values(),
            key=lambda item: item.get("total_degree", 0),
            reverse=True,
        )[:20]

        isolated_count = sum(
            1 for item in degree_stats.values()
            if item.get("total_degree", 0) == 0
        )

        connected_count = sum(
            1 for item in degree_stats.values()
            if item.get("total_degree", 0) > 0
        )

        return {
            "has_graph_index": len(node_store) > 0,
            "index_type": "graph",
            "node_count": len(node_store),
            "edge_count": len(edge_store),
            "node_type_count": len(nodes_by_type),
            "edge_type_count": len(edges_by_type),
            "page_count_with_nodes": len(nodes_by_page),
            "connected_node_count": connected_count,
            "isolated_node_count": isolated_count,
            "by_node_type": {
                key: len(value)
                for key, value in nodes_by_type.items()
            },
            "by_edge_type": {
                key: len(value)
                for key, value in edges_by_type.items()
            },
            "by_page": {
                key: len(value)
                for key, value in nodes_by_page.items()
            },
            "top_degree_nodes": top_degree_nodes,
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        nodes_by_page = result.get("nodes_by_page", {}) or {}
        edges_by_page = result.get("edges_by_page", {}) or {}
        summary = result.get("graph_index_summary", {}) or {}

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("graph_index_builder", {})
            page_raw.metadata["graph_index_builder"] = {
                "processor": "GraphIndexBuilder",
                "node_ids_on_page": nodes_by_page.get(page_key, []),
                "edge_ids_on_page": edges_by_page.get(page_key, []),
                "node_count_on_page": len(nodes_by_page.get(page_key, [])),
                "edge_count_on_page": len(edges_by_page.get(page_key, [])),
                "index_summary": {
                    "node_count": summary.get("node_count", 0),
                    "edge_count": summary.get("edge_count", 0),
                    "node_type_count": summary.get("node_type_count", 0),
                    "edge_type_count": summary.get("edge_type_count", 0),
                },
            }

    def _collect_chunks(
        self,
        chunk_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        chunks = []

        for key in ["chunks", "parent_chunks", "child_chunks", "table_chunks"]:
            values = chunk_result.get(key, []) or []

            if isinstance(values, list):
                chunks.extend([self._to_dict(item) for item in values])

        for sub_key in ["chunk_result", "chunk_collection", "parent_child_chunk_result", "table_chunk_result"]:
            sub = chunk_result.get(sub_key, {}) or {}

            if not isinstance(sub, dict):
                continue

            for key in ["chunks", "parent_chunks", "child_chunks", "table_chunks"]:
                values = sub.get(key, []) or []

                if isinstance(values, list):
                    chunks.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(chunks, ["chunk_id", "content_hash"])

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

    def _collect_evidence(
        self,
        evidence_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        evidence = []

        for key in ["evidence", "evidence_items", "retrieved_evidence", "supporting_evidence"]:
            values = evidence_result.get(key, []) or []

            if isinstance(values, list):
                evidence.extend([self._to_dict(item) for item in values])

        for sub_key in ["evidence_result", "evidence_collection"]:
            sub = evidence_result.get(sub_key, {}) or {}

            if not isinstance(sub, dict):
                continue

            values = sub.get("evidence", []) or sub.get("evidence_items", []) or []

            if isinstance(values, list):
                evidence.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(evidence, ["evidence_id", "content_hash"])

    def _collect_list(
        self,
        source: Dict[str, Any],
        key: str,
    ) -> List[Dict[str, Any]]:
        values = source.get(key, []) or []

        if not isinstance(values, list):
            return []

        return [self._to_dict(item) for item in values]

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
                key = self._stable_id(
                    normalize_pdf_text(
                        item.get("text")
                        or item.get("label")
                        or item.get("title")
                        or item.get("caption")
                        or ""
                    ),
                    "item",
                )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

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

    def _make_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = "related_to",
        relation_label: str = "",
        source_page: Optional[int] = None,
        target_page: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
        confidence: float = 0.75,
        weight: float = 1.0,
        source: str = "graph_index_builder",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        page_numbers = self._normalize_page_numbers(page_numbers or [])

        if not page_numbers:
            page_numbers = self._normalize_page_numbers([source_page, target_page])

        return {
            "edge_id": self._stable_id(f"{source_id}|{target_id}|{edge_type}", "edge"),
            "source_id": source_id,
            "target_id": target_id,
            "edge_type": edge_type,
            "relation_label": relation_label or edge_type,
            "source_page": source_page,
            "target_page": target_page,
            "page_numbers": page_numbers,
            "confidence": confidence,
            "weight": weight,
            "directed": True,
            "source": source,
            "metadata": metadata or {},
        }

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

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:
        text = normalize_text_for_match(text)
        raw_tokens = re.findall(r"[a-z0-9_]+", text)

        tokens = []

        for token in raw_tokens:
            token = token.strip("_")

            if len(token) < self.config.min_token_length:
                continue

            if len(token) > self.config.max_token_length:
                continue

            if self.config.remove_stopwords and token in self.stopwords:
                continue

            tokens.append(token)

        return tokens

    def _passes_filters(
        self,
        node: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:
        if not filters:
            return True

        for key, expected in filters.items():
            if expected is None:
                continue

            if key == "page_numbers":
                actual_pages = set(self._normalize_page_numbers(node.get("page_numbers", [])))
                expected_pages = set(self._normalize_page_numbers(expected))

                if expected_pages and not actual_pages.intersection(expected_pages):
                    return False

                continue

            actual = node.get(key)

            if actual is None:
                actual = node.get("metadata", {}).get(key)

            if isinstance(expected, list):
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False

        return True

    def _make_snippet(
        self,
        text: str,
        query_terms: List[str],
        max_chars: int = 500,
    ) -> str:
        text = normalize_pdf_text(text)

        if not text:
            return ""

        if not query_terms:
            return self._preview(text, max_chars)

        normalized = normalize_text_for_match(text)
        best_index = -1

        for term in query_terms:
            index = normalized.find(term)

            if index >= 0:
                best_index = index
                break

        if best_index < 0:
            return self._preview(text, max_chars)

        start = max(0, best_index - max_chars // 3)
        end = min(len(text), start + max_chars)

        snippet = text[start:end]

        if start > 0:
            snippet = "..." + snippet

        if end < len(text):
            snippet = snippet + "..."

        return normalize_pdf_text(snippet)

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
        text: Any,
        prefix: str = "id",
    ) -> str:
        text = normalize_pdf_text(text)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{digest}"

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


def build_graph_index(
    page_raws: Optional[List[PageRaw]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
    cross_page_context_result: Optional[Dict[str, Any]] = None,
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    chunk_result: Optional[Dict[str, Any]] = None,
    evidence_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = GraphIndexBuilder()
    return builder.process(
        page_raws=page_raws,
        knowledge_result=knowledge_result,
        cross_page_context_result=cross_page_context_result,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
        chunk_result=chunk_result,
        evidence_result=evidence_result,
    )


def search_graph_index(
    index_result: Dict[str, Any],
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    expand_neighbors: bool = True,
    max_neighbors: int = 20,
) -> List[Dict[str, Any]]:
    builder = GraphIndexBuilder()
    return builder.search(
        index_result=index_result,
        query=query,
        top_k=top_k,
        filters=filters,
        expand_neighbors=expand_neighbors,
        max_neighbors=max_neighbors,
    )


def expand_graph_nodes(
    index_result: Dict[str, Any],
    node_ids: List[str],
    depth: int = 1,
    edge_types: Optional[List[str]] = None,
    max_nodes: int = 100,
) -> Dict[str, Any]:
    builder = GraphIndexBuilder()
    return builder.expand(
        index_result=index_result,
        node_ids=node_ids,
        depth=depth,
        edge_types=edge_types,
        max_nodes=max_nodes,
    )
