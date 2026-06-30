"""
context_expander.py

Production V1 - Colab Ready

Purpose
-------
Expand retrieved context for RAG by adding:
- neighboring chunks
- parent / child chunks
- same-section context
- same-page context
- related table/evidence/citation context
- graph neighbors
- metadata-filtered context

Used by:
- RAGPipeline
- HybridRetriever
- EvidenceAggregator
- PromptBuilder
- LLMReasoner

Input
-----
- retrieved_items
- page_raws
- chunk_result
- table_chunk_result
- evidence_result
- citation_result
- graph_index_result
- metadata_index_result
- bm25_index_result
- vector_index_result
- knowledge_result

Output
------
Dictionary with:
- expanded_context_items
- expansion_links
- context_by_source
- context_by_page
- context_by_section
- context_expansion_summary
"""

from __future__ import annotations

import json
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
class ContextExpanderConfig:
    expand_neighbors: bool = True
    expand_parent_child: bool = True
    expand_same_section: bool = True
    expand_same_page: bool = True
    expand_tables: bool = True
    expand_evidence: bool = True
    expand_citations: bool = True
    expand_graph: bool = True
    expand_metadata: bool = True

    include_original_items: bool = True
    deduplicate_context: bool = True
    attach_to_pages: bool = True

    max_expanded_items: int = 80
    max_neighbors_per_item: int = 8
    max_same_section_items: int = 8
    max_same_page_items: int = 8
    max_parent_child_items: int = 12
    max_table_items: int = 8
    max_evidence_items: int = 12
    max_citation_items: int = 8
    max_graph_depth: int = 1

    max_text_chars_per_item: int = 1800
    text_preview_chars: int = 700

    min_context_chars: int = 20

    score_original: float = 1.00
    score_neighbor: float = 0.72
    score_parent_child: float = 0.78
    score_same_section: float = 0.66
    score_same_page: float = 0.58
    score_table: float = 0.74
    score_evidence: float = 0.76
    score_citation: float = 0.62
    score_graph: float = 0.68
    score_metadata: float = 0.56

    include_debug: bool = True


class ContextExpander:
    def __init__(
        self,
        config: Optional[ContextExpanderConfig] = None,
    ):
        self.config = config or ContextExpanderConfig()

    def process(
        self,
        retrieved_items: Optional[List[Dict[str, Any]]] = None,
        page_raws: Optional[List[PageRaw]] = None,
        chunk_result: Optional[Dict[str, Any]] = None,
        table_chunk_result: Optional[Dict[str, Any]] = None,
        evidence_result: Optional[Dict[str, Any]] = None,
        citation_result: Optional[Dict[str, Any]] = None,
        graph_index_result: Optional[Dict[str, Any]] = None,
        metadata_index_result: Optional[Dict[str, Any]] = None,
        bm25_index_result: Optional[Dict[str, Any]] = None,
        vector_index_result: Optional[Dict[str, Any]] = None,
        knowledge_result: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        retrieved_items = [
            self._normalize_retrieved_item(item)
            for item in (retrieved_items or [])
            if isinstance(item, dict)
        ]

        page_raws = sorted(
            page_raws or [],
            key=lambda item: item.page_number,
        )

        chunk_result = chunk_result or {}
        table_chunk_result = table_chunk_result or {}
        evidence_result = evidence_result or {}
        citation_result = citation_result or {}
        graph_index_result = graph_index_result or {}
        metadata_index_result = metadata_index_result or {}
        bm25_index_result = bm25_index_result or {}
        vector_index_result = vector_index_result or {}
        knowledge_result = knowledge_result or {}

        page_text_map = self._build_page_text_map(page_raws)

        chunk_index = self._collect_chunk_index(
            chunk_result=chunk_result,
            table_chunk_result=table_chunk_result,
            knowledge_result=knowledge_result,
            bm25_index_result=bm25_index_result,
            vector_index_result=vector_index_result,
        )
        evidence_index = self._collect_evidence_index(
            evidence_result=evidence_result,
            knowledge_result=knowledge_result,
        )
        citation_index = self._collect_citation_index(
            citation_result=citation_result,
            evidence_result=evidence_result,
            knowledge_result=knowledge_result,
        )
        table_index = self._collect_table_index(
            table_chunk_result=table_chunk_result,
            knowledge_result=knowledge_result,
            metadata_index_result=metadata_index_result,
        )
        graph_index = self._collect_graph_index(graph_index_result)
        metadata_store = self._collect_metadata_store(metadata_index_result)

        chunk_neighbors = self._build_chunk_neighbor_index(chunk_index)
        chunks_by_section = self._group_items_by_field(chunk_index, "section_id")
        chunks_by_page = self._group_items_by_page(chunk_index)
        evidence_by_chunk = self._group_items_by_field(evidence_index, "chunk_id")
        evidence_by_section = self._group_items_by_field(evidence_index, "section_id")
        evidence_by_page = self._group_items_by_page(evidence_index)
        citations_by_evidence = self._group_citations_by_evidence(citation_index)
        tables_by_page = self._group_items_by_page(table_index)
        tables_by_section = self._group_items_by_field(table_index, "section_id")

        expanded_items: List[Dict[str, Any]] = []
        expansion_links: List[Dict[str, Any]] = []

        if self.config.include_original_items:
            for item in retrieved_items:
                expanded_item = self._make_context_item(
                    source_item=item,
                    expansion_type="original",
                    expansion_reason="retrieved_item",
                    score_multiplier=self.config.score_original,
                )
                expanded_items.append(expanded_item)

        for seed_item in retrieved_items:
            seed_context_id = self._context_id(seed_item)

            if self.config.expand_neighbors:
                items, links = self._expand_neighbors(
                    seed_item=seed_item,
                    chunk_index=chunk_index,
                    chunk_neighbors=chunk_neighbors,
                )
                expanded_items.extend(items)
                expansion_links.extend(self._links_from(seed_context_id, items, links))

            if self.config.expand_parent_child:
                items, links = self._expand_parent_child(
                    seed_item=seed_item,
                    chunk_index=chunk_index,
                )
                expanded_items.extend(items)
                expansion_links.extend(self._links_from(seed_context_id, items, links))

            if self.config.expand_same_section:
                items, links = self._expand_same_section(
                    seed_item=seed_item,
                    chunks_by_section=chunks_by_section,
                    evidence_by_section=evidence_by_section,
                    tables_by_section=tables_by_section,
                )
                expanded_items.extend(items)
                expansion_links.extend(self._links_from(seed_context_id, items, links))

            if self.config.expand_same_page:
                items, links = self._expand_same_page(
                    seed_item=seed_item,
                    chunks_by_page=chunks_by_page,
                    evidence_by_page=evidence_by_page,
                    tables_by_page=tables_by_page,
                    page_text_map=page_text_map,
                )
                expanded_items.extend(items)
                expansion_links.extend(self._links_from(seed_context_id, items, links))

            if self.config.expand_tables:
                items, links = self._expand_tables(
                    seed_item=seed_item,
                    table_index=table_index,
                    tables_by_page=tables_by_page,
                    tables_by_section=tables_by_section,
                )
                expanded_items.extend(items)
                expansion_links.extend(self._links_from(seed_context_id, items, links))

            if self.config.expand_evidence:
                items, links = self._expand_evidence(
                    seed_item=seed_item,
                    evidence_index=evidence_index,
                    evidence_by_chunk=evidence_by_chunk,
                    evidence_by_section=evidence_by_section,
                    evidence_by_page=evidence_by_page,
                )
                expanded_items.extend(items)
                expansion_links.extend(self._links_from(seed_context_id, items, links))

            if self.config.expand_citations:
                items, links = self._expand_citations(
                    seed_item=seed_item,
                    citation_index=citation_index,
                    citations_by_evidence=citations_by_evidence,
                )
                expanded_items.extend(items)
                expansion_links.extend(self._links_from(seed_context_id, items, links))

            if self.config.expand_graph:
                items, links = self._expand_graph(
                    seed_item=seed_item,
                    graph_index=graph_index,
                )
                expanded_items.extend(items)
                expansion_links.extend(self._links_from(seed_context_id, items, links))

            if self.config.expand_metadata:
                items, links = self._expand_metadata(
                    seed_item=seed_item,
                    metadata_store=metadata_store,
                )
                expanded_items.extend(items)
                expansion_links.extend(self._links_from(seed_context_id, items, links))

        expanded_items = [
            item for item in expanded_items
            if self._valid_context_item(item)
        ]

        if self.config.deduplicate_context:
            expanded_items = self._deduplicate_context_items(expanded_items)

        expanded_items = self._sort_context_items(expanded_items)
        expanded_items = expanded_items[: self.config.max_expanded_items]

        result = {
            "processor": "ContextExpander",
            "schema_version": "context_expander_v1",
            "expanded_context_items": expanded_items,
            "expansion_links": expansion_links,
            "context_by_source": self._group_context_by_source(expanded_items),
            "context_by_page": self._group_context_by_page(expanded_items),
            "context_by_section": self._group_context_by_section(expanded_items),
            "context_by_type": self._group_context_by_type(expanded_items),
            "context_text": self._build_combined_context_text(expanded_items),
            "context_expansion_summary": self._build_summary(
                retrieved_items=retrieved_items,
                expanded_items=expanded_items,
                expansion_links=expansion_links,
                page_raws=page_raws,
                chunk_index=chunk_index,
                evidence_index=evidence_index,
                citation_index=citation_index,
                table_index=table_index,
                graph_index=graph_index,
                metadata_store=metadata_store,
            ),
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                expansion_result=result,
            )

        return json_safe(result)

    def _expand_neighbors(
        self,
        seed_item: Dict[str, Any],
        chunk_index: Dict[str, Dict[str, Any]],
        chunk_neighbors: Dict[str, List[str]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        items = []
        links = []

        chunk_id = seed_item.get("chunk_id", "") or seed_item.get("source_id", "")

        if not chunk_id or chunk_id not in chunk_neighbors:
            return items, links

        neighbor_ids = chunk_neighbors.get(chunk_id, [])[: self.config.max_neighbors_per_item]

        for neighbor_id in neighbor_ids:
            neighbor = chunk_index.get(neighbor_id)

            if not neighbor:
                continue

            context_item = self._make_context_item(
                source_item=neighbor,
                expansion_type="neighbor",
                expansion_reason="previous_next_chunk",
                score_multiplier=self.config.score_neighbor,
                seed_item=seed_item,
            )
            items.append(context_item)
            links.append(
                self._make_expansion_link(
                    seed_item=seed_item,
                    target_item=context_item,
                    relation_type="neighbor_chunk",
                    score=self.config.score_neighbor,
                )
            )

        return items, links

    def _expand_parent_child(
        self,
        seed_item: Dict[str, Any],
        chunk_index: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        items = []
        links = []

        chunk_id = seed_item.get("chunk_id", "") or seed_item.get("source_id", "")

        if not chunk_id:
            return items, links

        seed_chunk = chunk_index.get(chunk_id, seed_item)

        related_ids = []

        parent_chunk_id = seed_chunk.get("parent_chunk_id", "") or seed_chunk.get("metadata", {}).get("parent_chunk_id", "")

        if parent_chunk_id:
            related_ids.append(parent_chunk_id)

        child_chunk_ids = seed_chunk.get("child_chunk_ids", []) or seed_chunk.get("metadata", {}).get("child_chunk_ids", []) or []

        for child_id in child_chunk_ids:
            related_ids.append(child_id)

        for candidate_id, candidate in chunk_index.items():
            if candidate.get("parent_chunk_id", "") == chunk_id:
                related_ids.append(candidate_id)

        related_ids = list(dict.fromkeys([item for item in related_ids if item]))

        for related_id in related_ids[: self.config.max_parent_child_items]:
            related = chunk_index.get(related_id)

            if not related:
                continue

            context_item = self._make_context_item(
                source_item=related,
                expansion_type="parent_child",
                expansion_reason="chunk_hierarchy",
                score_multiplier=self.config.score_parent_child,
                seed_item=seed_item,
            )
            items.append(context_item)
            links.append(
                self._make_expansion_link(
                    seed_item=seed_item,
                    target_item=context_item,
                    relation_type="parent_child_chunk",
                    score=self.config.score_parent_child,
                )
            )

        return items, links

    def _expand_same_section(
        self,
        seed_item: Dict[str, Any],
        chunks_by_section: Dict[str, List[Dict[str, Any]]],
        evidence_by_section: Dict[str, List[Dict[str, Any]]],
        tables_by_section: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        items = []
        links = []

        section_id = seed_item.get("section_id", "") or seed_item.get("metadata", {}).get("section_id", "")

        if not section_id:
            return items, links

        candidates = []
        candidates.extend(chunks_by_section.get(section_id, [])[: self.config.max_same_section_items])
        candidates.extend(evidence_by_section.get(section_id, [])[: self.config.max_sameSection_safe()])
        candidates.extend(tables_by_section.get(section_id, [])[: self.config.max_table_items])

        for candidate in candidates[: self.config.max_same_section_items]:
            if self._same_source(seed_item, candidate):
                continue

            context_item = self._make_context_item(
                source_item=candidate,
                expansion_type="same_section",
                expansion_reason="shared_section_id",
                score_multiplier=self.config.score_same_section,
                seed_item=seed_item,
            )
            items.append(context_item)
            links.append(
                self._make_expansion_link(
                    seed_item=seed_item,
                    target_item=context_item,
                    relation_type="same_section",
                    score=self.config.score_same_section,
                )
            )

        return items, links

    def _expand_same_page(
        self,
        seed_item: Dict[str, Any],
        chunks_by_page: Dict[str, List[Dict[str, Any]]],
        evidence_by_page: Dict[str, List[Dict[str, Any]]],
        tables_by_page: Dict[str, List[Dict[str, Any]]],
        page_text_map: Dict[int, str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        items = []
        links = []

        page_numbers = self._resolve_page_numbers(seed_item)

        for page_number in page_numbers:
            page_key = str(page_number)
            candidates = []
            candidates.extend(chunks_by_page.get(page_key, [])[: self.config.max_same_page_items])
            candidates.extend(evidence_by_page.get(page_key, [])[: self.config.max_evidence_items])
            candidates.extend(tables_by_page.get(page_key, [])[: self.config.max_table_items])

            page_text = page_text_map.get(page_number, "")

            if page_text:
                candidates.append(
                    {
                        "source_id": f"page_{page_number}",
                        "source_type": "page_text",
                        "item_type": "page",
                        "title": f"Trang {page_number}",
                        "text": page_text,
                        "page_numbers": [page_number],
                        "page_number": page_number,
                        "source": "page_raw",
                    }
                )

            for candidate in candidates[: self.config.max_same_page_items]:
                if self._same_source(seed_item, candidate):
                    continue

                context_item = self._make_context_item(
                    source_item=candidate,
                    expansion_type="same_page",
                    expansion_reason="shared_page_number",
                    score_multiplier=self.config.score_same_page,
                    seed_item=seed_item,
                )
                items.append(context_item)
                links.append(
                    self._make_expansion_link(
                        seed_item=seed_item,
                        target_item=context_item,
                        relation_type="same_page",
                        score=self.config.score_same_page,
                    )
                )

        return items, links

    def _expand_tables(
        self,
        seed_item: Dict[str, Any],
        table_index: Dict[str, Dict[str, Any]],
        tables_by_page: Dict[str, List[Dict[str, Any]]],
        tables_by_section: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        items = []
        links = []

        table_ids = self._table_ids_from_item(seed_item)

        candidates = []

        for table_id in table_ids:
            if table_id in table_index:
                candidates.append(table_index[table_id])

        for page_number in self._resolve_page_numbers(seed_item):
            candidates.extend(tables_by_page.get(str(page_number), [])[: self.config.max_table_items])

        section_id = seed_item.get("section_id", "") or seed_item.get("metadata", {}).get("section_id", "")

        if section_id:
            candidates.extend(tables_by_section.get(section_id, [])[: self.config.max_table_items])

        candidates = self._deduplicate_raw_items(candidates, ["table_id", "table_semantic_id", "table_grid_id", "source_id"])

        for candidate in candidates[: self.config.max_table_items]:
            if self._same_source(seed_item, candidate):
                continue

            context_item = self._make_context_item(
                source_item=candidate,
                expansion_type="table",
                expansion_reason="related_table",
                score_multiplier=self.config.score_table,
                seed_item=seed_item,
            )
            items.append(context_item)
            links.append(
                self._make_expansion_link(
                    seed_item=seed_item,
                    target_item=context_item,
                    relation_type="related_table",
                    score=self.config.score_table,
                )
            )

        return items, links

    def _expand_evidence(
        self,
        seed_item: Dict[str, Any],
        evidence_index: Dict[str, Dict[str, Any]],
        evidence_by_chunk: Dict[str, List[Dict[str, Any]]],
        evidence_by_section: Dict[str, List[Dict[str, Any]]],
        evidence_by_page: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        items = []
        links = []

        candidates = []

        evidence_id = seed_item.get("evidence_id", "") or seed_item.get("source_id", "")

        if evidence_id and evidence_id in evidence_index:
            candidates.append(evidence_index[evidence_id])

        chunk_id = seed_item.get("chunk_id", "") or seed_item.get("metadata", {}).get("chunk_id", "")

        if chunk_id:
            candidates.extend(evidence_by_chunk.get(chunk_id, [])[: self.config.max_evidence_items])

        section_id = seed_item.get("section_id", "") or seed_item.get("metadata", {}).get("section_id", "")

        if section_id:
            candidates.extend(evidence_by_section.get(section_id, [])[: self.config.max_evidence_items])

        for page_number in self._resolve_page_numbers(seed_item):
            candidates.extend(evidence_by_page.get(str(page_number), [])[: self.config.max_evidence_items])

        candidates = self._deduplicate_raw_items(candidates, ["evidence_id", "content_hash", "source_id"])

        for candidate in candidates[: self.config.max_evidence_items]:
            if self._same_source(seed_item, candidate):
                continue

            context_item = self._make_context_item(
                source_item=candidate,
                expansion_type="evidence",
                expansion_reason="related_evidence",
                score_multiplier=self.config.score_evidence,
                seed_item=seed_item,
            )
            items.append(context_item)
            links.append(
                self._make_expansion_link(
                    seed_item=seed_item,
                    target_item=context_item,
                    relation_type="related_evidence",
                    score=self.config.score_evidence,
                )
            )

        return items, links

    def _expand_citations(
        self,
        seed_item: Dict[str, Any],
        citation_index: Dict[str, Dict[str, Any]],
        citations_by_evidence: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        items = []
        links = []

        candidates = []

        citation_id = seed_item.get("citation_id", "")

        if citation_id and citation_id in citation_index:
            candidates.append(citation_index[citation_id])

        evidence_id = seed_item.get("evidence_id", "") or seed_item.get("metadata", {}).get("evidence_id", "")

        if evidence_id:
            candidates.extend(citations_by_evidence.get(evidence_id, [])[: self.config.max_citation_items])

        candidates = self._deduplicate_raw_items(candidates, ["citation_id", "citation_marker"])

        for candidate in candidates[: self.config.max_citation_items]:
            if self._same_source(seed_item, candidate):
                continue

            context_item = self._make_context_item(
                source_item=candidate,
                expansion_type="citation",
                expansion_reason="related_citation",
                score_multiplier=self.config.score_citation,
                seed_item=seed_item,
            )
            items.append(context_item)
            links.append(
                self._make_expansion_link(
                    seed_item=seed_item,
                    target_item=context_item,
                    relation_type="related_citation",
                    score=self.config.score_citation,
                )
            )

        return items, links

    def _expand_graph(
        self,
        seed_item: Dict[str, Any],
        graph_index: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        items = []
        links = []

        node_store = graph_index.get("node_store", {}) or {}
        edge_store = graph_index.get("edge_store", {}) or {}
        undirected_adjacency = graph_index.get("undirected_adjacency", {}) or {}

        seed_node_ids = self._candidate_graph_node_ids(seed_item)

        for seed_node_id in seed_node_ids:
            if seed_node_id not in node_store:
                continue

            neighbors = undirected_adjacency.get(seed_node_id, []) or []

            for neighbor_info in neighbors[: self.config.max_neighbors_per_item]:
                neighbor_id = neighbor_info.get("neighbor_id", "")
                edge_id = neighbor_info.get("edge_id", "")

                neighbor = node_store.get(neighbor_id)

                if not neighbor:
                    continue

                edge = edge_store.get(edge_id, {})

                context_item = self._make_context_item(
                    source_item=neighbor,
                    expansion_type="graph",
                    expansion_reason=edge.get("edge_type", "graph_neighbor"),
                    score_multiplier=self.config.score_graph,
                    seed_item=seed_item,
                )
                context_item["graph_edge"] = {
                    "edge_id": edge_id,
                    "edge_type": edge.get("edge_type", ""),
                    "relation_label": edge.get("relation_label", ""),
                    "direction": neighbor_info.get("direction", ""),
                    "weight": neighbor_info.get("weight", 1.0),
                    "confidence": neighbor_info.get("confidence", 0.0),
                }
                items.append(context_item)
                links.append(
                    self._make_expansion_link(
                        seed_item=seed_item,
                        target_item=context_item,
                        relation_type=edge.get("edge_type", "graph_neighbor"),
                        score=self.config.score_graph,
                        metadata={
                            "edge_id": edge_id,
                            "direction": neighbor_info.get("direction", ""),
                        },
                    )
                )

        return items, links

    def _expand_metadata(
        self,
        seed_item: Dict[str, Any],
        metadata_store: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        items = []
        links = []

        candidates = []

        seed_pages = set(self._resolve_page_numbers(seed_item))
        seed_section_id = seed_item.get("section_id", "") or seed_item.get("metadata", {}).get("section_id", "")
        seed_table_ids = set(self._table_ids_from_item(seed_item))
        seed_chunk_id = seed_item.get("chunk_id", "") or seed_item.get("metadata", {}).get("chunk_id", "")
        seed_evidence_id = seed_item.get("evidence_id", "") or seed_item.get("metadata", {}).get("evidence_id", "")

        for metadata_item in metadata_store.values():
            if seed_chunk_id and metadata_item.get("chunk_id") == seed_chunk_id:
                candidates.append(metadata_item)
                continue

            if seed_evidence_id and metadata_item.get("evidence_id") == seed_evidence_id:
                candidates.append(metadata_item)
                continue

            if seed_section_id and metadata_item.get("section_id") == seed_section_id:
                candidates.append(metadata_item)
                continue

            if seed_table_ids and metadata_item.get("table_id") in seed_table_ids:
                candidates.append(metadata_item)
                continue

            item_pages = set(self._resolve_page_numbers(metadata_item))
            if seed_pages and item_pages and seed_pages.intersection(item_pages):
                candidates.append(metadata_item)

        candidates = self._deduplicate_raw_items(candidates, ["item_id", "content_hash"])

        for candidate in candidates[: self.config.max_same_page_items]:
            if self._same_source(seed_item, candidate):
                continue

            context_item = self._make_context_item(
                source_item=candidate,
                expansion_type="metadata",
                expansion_reason="metadata_overlap",
                score_multiplier=self.config.score_metadata,
                seed_item=seed_item,
            )
            items.append(context_item)
            links.append(
                self._make_expansion_link(
                    seed_item=seed_item,
                    target_item=context_item,
                    relation_type="metadata_overlap",
                    score=self.config.score_metadata,
                )
            )

        return items, links

    def _normalize_retrieved_item(
        self,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        item = self._to_dict(item)

        text = normalize_pdf_text(
            item.get("text")
            or item.get("text_preview")
            or item.get("quote")
            or item.get("content")
            or item.get("label")
            or item.get("title")
            or ""
        )

        item["text"] = text
        item.setdefault("source_id", item.get("document_id") or item.get("chunk_id") or item.get("evidence_id") or item.get("node_id") or "")
        item.setdefault("source_type", item.get("chunk_type") or item.get("evidence_type") or item.get("node_type") or item.get("item_type") or "retrieved_item")
        item.setdefault("score", item.get("retrieval_score", item.get("rerank_score", 1.0)))
        item["page_numbers"] = self._resolve_page_numbers(item)

        return item

    def _make_context_item(
        self,
        source_item: Dict[str, Any],
        expansion_type: str,
        expansion_reason: str,
        score_multiplier: float,
        seed_item: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_item = self._to_dict(source_item)
        seed_item = seed_item or {}

        text = normalize_pdf_text(
            source_item.get("text")
            or source_item.get("text_preview")
            or source_item.get("quote")
            or source_item.get("citation_text")
            or source_item.get("label")
            or source_item.get("title")
            or ""
        )

        text = self._truncate_text(text, self.config.max_text_chars_per_item)

        base_score = self._safe_float(source_item.get("score"), default=1.0)
        confidence = self._safe_float(source_item.get("confidence"), default=0.70)

        context_id = self._context_id(source_item)

        metadata = source_item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        return {
            "context_id": context_id,
            "source_id": source_item.get("source_id", "")
            or source_item.get("document_id", "")
            or source_item.get("chunk_id", "")
            or source_item.get("evidence_id", "")
            or source_item.get("citation_id", "")
            or source_item.get("node_id", "")
            or context_id,
            "source_type": source_item.get("source_type", "")
            or source_item.get("chunk_type", "")
            or source_item.get("evidence_type", "")
            or source_item.get("node_type", "")
            or source_item.get("item_type", "")
            or "context",
            "source": source_item.get("source", ""),
            "expansion_type": expansion_type,
            "expansion_reason": expansion_reason,
            "seed_context_id": self._context_id(seed_item) if seed_item else "",
            "title": normalize_pdf_text(source_item.get("title") or source_item.get("section_title") or source_item.get("label") or ""),
            "text": text,
            "text_preview": self._preview(text, self.config.text_preview_chars),
            "normalized_text": normalize_text_for_match(text),
            "page_numbers": self._resolve_page_numbers(source_item),
            "page_start": source_item.get("page_start") or (min(self._resolve_page_numbers(source_item)) if self._resolve_page_numbers(source_item) else None),
            "page_end": source_item.get("page_end") or (max(self._resolve_page_numbers(source_item)) if self._resolve_page_numbers(source_item) else None),
            "section_id": source_item.get("section_id", "") or metadata.get("section_id", ""),
            "section_title": source_item.get("section_title", "") or metadata.get("section_title", ""),
            "chunk_id": source_item.get("chunk_id", "") or metadata.get("chunk_id", ""),
            "evidence_id": source_item.get("evidence_id", "") or metadata.get("evidence_id", ""),
            "citation_id": source_item.get("citation_id", "") or metadata.get("citation_id", ""),
            "table_id": self._table_ids_from_item(source_item)[0] if self._table_ids_from_item(source_item) else "",
            "node_id": source_item.get("node_id", "") or metadata.get("node_id", ""),
            "bbox": source_item.get("bbox", []) or [],
            "score": round(base_score * score_multiplier, 6),
            "base_score": base_score,
            "confidence": confidence,
            "weight": self._safe_float(source_item.get("weight"), default=1.0),
            "metadata": metadata if self.config.include_debug else {},
        }

    def _make_expansion_link(
        self,
        seed_item: Dict[str, Any],
        target_item: Dict[str, Any],
        relation_type: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        source_context_id = self._context_id(seed_item)
        target_context_id = target_item.get("context_id", self._context_id(target_item))

        return {
            "expansion_link_id": self._stable_id(
                f"{source_context_id}|{target_context_id}|{relation_type}",
                "expansion_link",
            ),
            "source_context_id": source_context_id,
            "target_context_id": target_context_id,
            "relation_type": relation_type,
            "score": score,
            "source_page_numbers": self._resolve_page_numbers(seed_item),
            "target_page_numbers": self._resolve_page_numbers(target_item),
            "metadata": metadata or {},
        }

    def _links_from(
        self,
        seed_context_id: str,
        items: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if links:
            return links

        generated = []

        for item in items:
            generated.append(
                {
                    "expansion_link_id": self._stable_id(
                        f"{seed_context_id}|{item.get('context_id', '')}|{item.get('expansion_type', '')}",
                        "expansion_link",
                    ),
                    "source_context_id": seed_context_id,
                    "target_context_id": item.get("context_id", ""),
                    "relation_type": item.get("expansion_type", ""),
                    "score": item.get("score", 0.0),
                    "source_page_numbers": [],
                    "target_page_numbers": item.get("page_numbers", []),
                    "metadata": {},
                }
            )

        return generated

    def _build_chunk_neighbor_index(
        self,
        chunk_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        neighbors: Dict[str, List[str]] = {}

        sorted_chunks = sorted(
            chunk_index.values(),
            key=lambda item: (
                min(self._resolve_page_numbers(item)) if self._resolve_page_numbers(item) else 999999,
                self._safe_int(item.get("order"), default=999999),
                item.get("chunk_id", ""),
            ),
        )

        chunk_ids = [item.get("chunk_id", "") for item in sorted_chunks if item.get("chunk_id", "")]

        for index, chunk_id in enumerate(chunk_ids):
            neighbors.setdefault(chunk_id, [])

            if index > 0:
                neighbors[chunk_id].append(chunk_ids[index - 1])

            if index < len(chunk_ids) - 1:
                neighbors[chunk_id].append(chunk_ids[index + 1])

        for chunk_id, chunk in chunk_index.items():
            for key in ["previous_chunk_id", "prev_chunk_id", "next_chunk_id"]:
                neighbor_id = chunk.get(key, "")

                if neighbor_id and neighbor_id in chunk_index:
                    neighbors.setdefault(chunk_id, [])

                    if neighbor_id not in neighbors[chunk_id]:
                        neighbors[chunk_id].append(neighbor_id)

        return neighbors

    def _collect_chunk_index(
        self,
        chunk_result: Dict[str, Any],
        table_chunk_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
        bm25_index_result: Dict[str, Any],
        vector_index_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        items = []

        for source in [chunk_result, table_chunk_result, knowledge_result]:
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
                    items.extend([self._to_dict(item) for item in values])

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
                        items.extend([self._to_dict(item) for item in values])

        for index_result in [bm25_index_result, vector_index_result]:
            index_obj = index_result.get("bm25_index", {}) or index_result.get("vector_index", {}) or index_result or {}
            document_store = index_obj.get("document_store", {}) or {}

            if isinstance(document_store, dict):
                for document in document_store.values():
                    document = self._to_dict(document)

                    if document.get("chunk_id"):
                        items.append(document)

        items = self._deduplicate_raw_items(items, ["chunk_id", "content_hash", "document_id"])

        index = {}

        for item in items:
            chunk_id = item.get("chunk_id", "")

            if not chunk_id:
                continue

            index[chunk_id] = item

        return index

    def _collect_evidence_index(
        self,
        evidence_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        items = []

        for source in [evidence_result, knowledge_result]:
            for key in ["evidence", "evidence_items", "retrieved_evidence", "supporting_evidence"]:
                values = source.get(key, []) or []

                if isinstance(values, list):
                    items.extend([self._to_dict(item) for item in values])

            for sub_key in ["evidence_result", "evidence_collection"]:
                sub = source.get(sub_key, {}) or {}

                if not isinstance(sub, dict):
                    continue

                values = sub.get("evidence", []) or sub.get("evidence_items", []) or []

                if isinstance(values, list):
                    items.extend([self._to_dict(item) for item in values])

        index = {}

        for item in self._deduplicate_raw_items(items, ["evidence_id", "content_hash"]):
            evidence_id = item.get("evidence_id", "")

            if evidence_id:
                index[evidence_id] = item

        return index

    def _collect_citation_index(
        self,
        citation_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        items = []

        for source in [citation_result, evidence_result, knowledge_result]:
            for key in ["citations", "citation_items", "verified_citations", "questionable_citations", "failed_citations"]:
                values = source.get(key, []) or []

                if isinstance(values, list):
                    items.extend([self._to_dict(item) for item in values])

            for sub_key in ["citation_result", "citation_collection", "evidence_collection"]:
                sub = source.get(sub_key, {}) or {}

                if not isinstance(sub, dict):
                    continue

                values = sub.get("citations", []) or sub.get("citation_items", []) or []

                if isinstance(values, list):
                    items.extend([self._to_dict(item) for item in values])

        index = {}

        for item in self._deduplicate_raw_items(items, ["citation_id", "citation_marker"]):
            citation_id = item.get("citation_id", "") or item.get("citation_marker", "")

            if citation_id:
                index[citation_id] = item

        return index

    def _collect_table_index(
        self,
        table_chunk_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        items = []

        for source in [table_chunk_result, knowledge_result]:
            for key in [
                "table_chunks",
                "table_summary_chunks",
                "table_record_chunks",
                "table_row_chunks",
                "multi_page_table_chunks",
                "table_semantics",
                "multi_page_tables",
            ]:
                values = source.get(key, []) or []

                if isinstance(values, list):
                    items.extend([self._to_dict(item) for item in values])

            for sub_key in ["table_chunk_result", "table_understanding_result", "table_semantic_result", "multi_page_table_result"]:
                sub = source.get(sub_key, {}) or {}

                if not isinstance(sub, dict):
                    continue

                for key in [
                    "table_chunks",
                    "table_summary_chunks",
                    "table_record_chunks",
                    "table_row_chunks",
                    "multi_page_table_chunks",
                    "table_semantics",
                    "multi_page_tables",
                ]:
                    values = sub.get(key, []) or []

                    if isinstance(values, list):
                        items.extend([self._to_dict(item) for item in values])

        metadata_index = metadata_index_result.get("metadata_index", metadata_index_result) or {}
        metadata_store = metadata_index.get("metadata_store", {}) or metadata_index_result.get("metadata_store", {}) or {}

        if isinstance(metadata_store, dict):
            for item in metadata_store.values():
                item = self._to_dict(item)

                if item.get("table_id"):
                    items.append(item)

        index = {}

        for item in self._deduplicate_raw_items(items, ["table_id", "table_semantic_id", "table_grid_id", "chunk_id"]):
            table_ids = self._table_ids_from_item(item)

            for table_id in table_ids:
                index[table_id] = item

        return index

    def _collect_graph_index(
        self,
        graph_index_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        graph_index = graph_index_result.get("graph_index", graph_index_result) or {}

        return {
            "node_store": graph_index.get("node_store", {}) or graph_index_result.get("node_store", {}) or {},
            "edge_store": graph_index.get("edge_store", {}) or graph_index_result.get("edge_store", {}) or {},
            "adjacency": graph_index.get("adjacency", {}) or graph_index_result.get("adjacency", {}) or {},
            "reverse_adjacency": graph_index.get("reverse_adjacency", {}) or graph_index_result.get("reverse_adjacency", {}) or {},
            "undirected_adjacency": graph_index.get("undirected_adjacency", {}) or graph_index_result.get("undirected_adjacency", {}) or {},
        }

    def _collect_metadata_store(
        self,
        metadata_index_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        metadata_index = metadata_index_result.get("metadata_index", metadata_index_result) or {}
        metadata_store = metadata_index.get("metadata_store", {}) or metadata_index_result.get("metadata_store", {}) or {}

        if not isinstance(metadata_store, dict):
            return {}

        return {
            str(item_id): self._to_dict(item)
            for item_id, item in metadata_store.items()
        }

    def _build_page_text_map(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[int, str]:
        page_text_map = {}

        for page_raw in page_raws:
            reading_meta = page_raw.metadata.get("reading_order_builder", {}) or {}
            reading_text = normalize_pdf_text(reading_meta.get("reading_order_text", ""))

            if reading_text:
                text = reading_text
            elif page_raw.normalized_text:
                text = page_raw.normalized_text
            elif page_raw.raw_text:
                text = page_raw.raw_text
            elif page_raw.text_blocks:
                text = "\n\n".join(
                    [
                        block.text
                        for block in page_raw.text_blocks
                        if getattr(block, "text", "")
                    ]
                )
            else:
                text = ""

            page_text_map[page_raw.page_number] = normalize_pdf_text(text)

        return page_text_map

    def _group_items_by_field(
        self,
        items: Dict[str, Dict[str, Any]],
        field_name: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for item in items.values():
            value = item.get(field_name, "") or item.get("metadata", {}).get(field_name, "")

            if not value:
                continue

            grouped.setdefault(str(value), [])
            grouped[str(value)].append(item)

        return grouped

    def _group_items_by_page(
        self,
        items: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for item in items.values():
            for page_number in self._resolve_page_numbers(item):
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(item)

        return grouped

    def _group_citations_by_evidence(
        self,
        citation_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for citation in citation_index.values():
            evidence_ids = citation.get("evidence_ids", []) or []

            if citation.get("evidence_id"):
                evidence_ids.append(citation.get("evidence_id"))

            for evidence_id in evidence_ids:
                if not evidence_id:
                    continue

                grouped.setdefault(evidence_id, [])
                grouped[evidence_id].append(citation)

        return grouped

    def _candidate_graph_node_ids(
        self,
        item: Dict[str, Any],
    ) -> List[str]:
        candidates = []

        for key in ["node_id", "source_id"]:
            if item.get(key):
                candidates.append(str(item.get(key)))

        for prefix, key in [
            ("chunk", "chunk_id"),
            ("evidence", "evidence_id"),
            ("table", "table_id"),
            ("section", "section_id"),
            ("page", "page_number"),
        ]:
            value = item.get(key) or item.get("metadata", {}).get(key, "")

            if value:
                candidates.append(f"{prefix}_{value}")

        for table_id in self._table_ids_from_item(item):
            candidates.append(f"table_{table_id}")

        return list(dict.fromkeys([item for item in candidates if item]))

    def _deduplicate_context_items(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result_by_key = {}

        for item in items:
            key = (
                item.get("source_id", ""),
                item.get("source_type", ""),
                item.get("chunk_id", ""),
                item.get("evidence_id", ""),
                item.get("citation_id", ""),
                item.get("table_id", ""),
                item.get("node_id", ""),
                normalize_text_for_match(item.get("text", ""))[:700],
                tuple(item.get("page_numbers", []) or []),
            )

            if key not in result_by_key:
                result_by_key[key] = item
            else:
                existing = result_by_key[key]

                if item.get("score", 0.0) > existing.get("score", 0.0):
                    merged = {
                        **existing,
                        **item,
                    }
                    merged["score"] = max(existing.get("score", 0.0), item.get("score", 0.0))
                    merged["expansion_type"] = existing.get("expansion_type", "") + "|" + item.get("expansion_type", "")
                    result_by_key[key] = merged

        return list(result_by_key.values())

    def _deduplicate_raw_items(
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
                value = item.get(key_name) or item.get("metadata", {}).get(key_name, "")

                if value:
                    key = str(value)
                    break

            if not key:
                key = self._stable_id(
                    {
                        "text": item.get("text") or item.get("text_preview") or item.get("quote") or item.get("label") or "",
                        "pages": self._resolve_page_numbers(item),
                    },
                    "raw",
                )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _sort_context_items(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return sorted(
            items,
            key=lambda item: (
                -self._safe_float(item.get("score"), default=0.0),
                self._expansion_type_order(item.get("expansion_type", "")),
                min(item.get("page_numbers", []) or [999999]),
                item.get("source_type", ""),
                item.get("source_id", ""),
            ),
        )

    def _expansion_type_order(
        self,
        expansion_type: str,
    ) -> int:
        order_map = {
            "original": 0,
            "parent_child": 1,
            "evidence": 2,
            "table": 3,
            "graph": 4,
            "neighbor": 5,
            "same_section": 6,
            "same_page": 7,
            "citation": 8,
            "metadata": 9,
        }

        for key, value in order_map.items():
            if key in expansion_type:
                return value

        return 99

    def _valid_context_item(
        self,
        item: Dict[str, Any],
    ) -> bool:
        text = normalize_pdf_text(item.get("text", ""))

        if not text:
            return False

        if len(text) < self.config.min_context_chars:
            word_count = len(re.findall(r"\w+", text))

            if word_count < 3:
                return False

        return True

    def _same_source(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
    ) -> bool:
        keys = ["source_id", "chunk_id", "evidence_id", "citation_id", "node_id"]

        for key in keys:
            av = a.get(key, "") or a.get("metadata", {}).get(key, "")
            bv = b.get(key, "") or b.get("metadata", {}).get(key, "")

            if av and bv and av == bv:
                return True

        return False

    def _context_id(
        self,
        item: Dict[str, Any],
    ) -> str:
        if not item:
            return ""

        for key in ["context_id", "source_id", "document_id", "chunk_id", "evidence_id", "citation_id", "node_id", "item_id"]:
            value = item.get(key)

            if value:
                return str(value)

        return self._stable_id(
            {
                "text": item.get("text") or item.get("text_preview") or item.get("quote") or "",
                "pages": self._resolve_page_numbers(item),
            },
            "context",
        )

    def _table_ids_from_item(
        self,
        item: Dict[str, Any],
    ) -> List[str]:
        metadata = item.get("metadata", {}) or {}

        if not isinstance(metadata, dict):
            metadata = {}

        ids = []

        for source in [item, metadata]:
            for key in [
                "table_id",
                "table_grid_id",
                "table_structure_id",
                "table_semantic_id",
                "table_boundary_id",
                "multi_page_table_id",
            ]:
                value = source.get(key)

                if value:
                    ids.append(str(value))

        return list(dict.fromkeys(ids))

    def _group_context_by_source(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for item in items:
            source_type = item.get("source_type", "unknown")
            grouped.setdefault(source_type, [])
            grouped[source_type].append(item)

        return grouped

    def _group_context_by_page(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for item in items:
            for page_number in item.get("page_numbers", []) or []:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(item)

        return grouped

    def _group_context_by_section(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for item in items:
            section_id = item.get("section_id", "") or "no_section"
            grouped.setdefault(section_id, [])
            grouped[section_id].append(item)

        return grouped

    def _group_context_by_type(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for item in items:
            expansion_type = item.get("expansion_type", "unknown")
            grouped.setdefault(expansion_type, [])
            grouped[expansion_type].append(item)

        return grouped

    def _build_combined_context_text(
        self,
        items: List[Dict[str, Any]],
    ) -> str:
        parts = []

        for index, item in enumerate(items, start=1):
            title = item.get("title", "") or item.get("source_type", "")
            pages = item.get("page_numbers", []) or []

            page_text = ""
            if pages:
                if len(pages) == 1:
                    page_text = f"trang {pages[0]}"
                else:
                    page_text = f"trang {pages[0]}-{pages[-1]}"

            header_parts = [f"[Context {index}]"]

            if title:
                header_parts.append(title)

            if page_text:
                header_parts.append(page_text)

            header_parts.append(f"type={item.get('expansion_type', '')}")

            parts.append(" | ".join(header_parts))
            parts.append(normalize_pdf_text(item.get("text", "")))

        return normalize_pdf_text("\n\n".join(parts))

    def _build_summary(
        self,
        retrieved_items: List[Dict[str, Any]],
        expanded_items: List[Dict[str, Any]],
        expansion_links: List[Dict[str, Any]],
        page_raws: List[PageRaw],
        chunk_index: Dict[str, Dict[str, Any]],
        evidence_index: Dict[str, Dict[str, Any]],
        citation_index: Dict[str, Dict[str, Any]],
        table_index: Dict[str, Dict[str, Any]],
        graph_index: Dict[str, Any],
        metadata_store: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        by_expansion_type = {}
        by_source_type = {}
        by_page = {}

        total_chars = 0
        total_words = 0

        for item in expanded_items:
            expansion_type = item.get("expansion_type", "unknown")
            source_type = item.get("source_type", "unknown")

            by_expansion_type[expansion_type] = by_expansion_type.get(expansion_type, 0) + 1
            by_source_type[source_type] = by_source_type.get(source_type, 0) + 1

            text = normalize_pdf_text(item.get("text", ""))
            total_chars += len(text)
            total_words += len(re.findall(r"\S+", text))

            for page_number in item.get("page_numbers", []) or []:
                page_key = str(page_number)
                by_page[page_key] = by_page.get(page_key, 0) + 1

        return {
            "has_expanded_context": len(expanded_items) > 0,
            "retrieved_item_count": len(retrieved_items),
            "expanded_context_count": len(expanded_items),
            "expansion_link_count": len(expansion_links),
            "page_count": len(page_raws),
            "chunk_index_count": len(chunk_index),
            "evidence_index_count": len(evidence_index),
            "citation_index_count": len(citation_index),
            "table_index_count": len(table_index),
            "graph_node_count": len(graph_index.get("node_store", {}) or {}),
            "graph_edge_count": len(graph_index.get("edge_store", {}) or {}),
            "metadata_item_count": len(metadata_store),
            "total_chars": total_chars,
            "total_words": total_words,
            "avg_chars_per_context": round(total_chars / max(len(expanded_items), 1), 2),
            "avg_words_per_context": round(total_words / max(len(expanded_items), 1), 2),
            "by_expansion_type": by_expansion_type,
            "by_source_type": by_source_type,
            "by_page": by_page,
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        expansion_result: Dict[str, Any],
    ) -> None:
        context_by_page = expansion_result.get("context_by_page", {}) or {}

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)
            page_context = context_by_page.get(page_key, [])

            page_raw.metadata.setdefault("context_expander", {})
            page_raw.metadata["context_expander"] = {
                "processor": "ContextExpander",
                "context_items_on_page": page_context,
                "context_item_count_on_page": len(page_context),
            }

    def _resolve_page_numbers(
        self,
        item: Dict[str, Any],
    ) -> List[int]:
        if not isinstance(item, dict):
            return []

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

        metadata = item.get("metadata", {}) or {}

        if isinstance(metadata, dict):
            return self._resolve_page_numbers(metadata)

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

    def _truncate_text(
        self,
        text: str,
        max_chars: int,
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

        return normalize_pdf_text(cut) + "..."

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
        prefix: str = "id",
    ) -> str:
        try:
            text = json.dumps(
                json_safe(value),
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception:
            text = str(value)

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

    def max_sameSection_safe(self) -> int:
        return max(1, min(self.config.max_same_section_items, self.config.max_evidence_items))

    def save_result(
        self,
        expansion_result: Dict[str, Any],
        output_path: str,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                json_safe(expansion_result),
                f,
                ensure_ascii=False,
                indent=2,
            )

        return str(output_path)

    def load_result(
        self,
        input_path: str,
    ) -> Dict[str, Any]:
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)


def expand_context(
    retrieved_items: Optional[List[Dict[str, Any]]] = None,
    page_raws: Optional[List[PageRaw]] = None,
    chunk_result: Optional[Dict[str, Any]] = None,
    table_chunk_result: Optional[Dict[str, Any]] = None,
    evidence_result: Optional[Dict[str, Any]] = None,
    citation_result: Optional[Dict[str, Any]] = None,
    graph_index_result: Optional[Dict[str, Any]] = None,
    metadata_index_result: Optional[Dict[str, Any]] = None,
    bm25_index_result: Optional[Dict[str, Any]] = None,
    vector_index_result: Optional[Dict[str, Any]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    expander = ContextExpander()
    return expander.process(
        retrieved_items=retrieved_items,
        page_raws=page_raws,
        chunk_result=chunk_result,
        table_chunk_result=table_chunk_result,
        evidence_result=evidence_result,
        citation_result=citation_result,
        graph_index_result=graph_index_result,
        metadata_index_result=metadata_index_result,
        bm25_index_result=bm25_index_result,
        vector_index_result=vector_index_result,
        knowledge_result=knowledge_result,
    )
