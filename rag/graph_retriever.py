"""
graph_retriever.py

Production V1 - Colab Ready

Purpose
-------
Retrieve and expand relevant context from graph index / knowledge graph.

Used by:
- RAGPipeline
- HybridRetriever
- ContextExpander
- EvidenceAggregator
- PromptBuilder

Input
-----
- query
- graph_index_result
- retrieved_items
- seed_node_ids
- filters
- node_types
- edge_types
- page_numbers
- metadata_index_result
- bm25_index_result
- vector_index_result

Output
------
Dictionary with:
- retrieved_graph_items
- seed_graph_items
- expanded_graph_items
- graph_paths
- graph_edges_used
- graph_context_text
- graph_retrieval_summary
"""

from __future__ import annotations

import json
import math
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from document_ai.schemas.page_raw_schema import (
    PageRaw,
    normalize_pdf_text,
    normalize_text_for_match,
    json_safe,
)


@dataclass
class GraphRetrieverConfig:
    top_k: int = 20
    max_seed_nodes: int = 20
    max_expanded_nodes: int = 80
    max_neighbors_per_node: int = 20
    max_depth: int = 2

    use_text_index: bool = True
    use_node_text_scan: bool = True
    use_seed_node_ids: bool = True
    use_retrieved_items_as_seeds: bool = True
    use_page_filter_as_seed: bool = True

    expand_neighbors: bool = True
    expand_forward_edges: bool = True
    expand_reverse_edges: bool = True
    expand_undirected_edges: bool = True

    include_seed_items: bool = True
    include_expanded_items: bool = True
    include_graph_edges: bool = True
    include_paths: bool = True
    include_context_text: bool = True

    deduplicate_items: bool = True

    min_query_token_len: int = 2
    min_score: float = 0.01

    score_query_match: float = 1.35
    score_seed_match: float = 1.20
    score_neighbor: float = 0.72
    score_path_decay: float = 0.82
    score_degree: float = 0.08
    score_confidence: float = 0.40
    score_weight: float = 0.30
    score_page_overlap: float = 0.35
    score_metadata_overlap: float = 0.20

    boost_page_node: float = 0.70
    boost_section_node: float = 1.15
    boost_chunk_node: float = 1.20
    boost_table_node: float = 1.18
    boost_evidence_node: float = 1.25
    boost_entity_node: float = 0.88
    boost_reference_node: float = 0.88

    max_context_chars: int = 12000
    max_text_chars_per_item: int = 1600
    text_preview_chars: int = 700

    include_debug: bool = True


class GraphRetriever:
    def __init__(
        self,
        config: Optional[GraphRetrieverConfig] = None,
    ):
        self.config = config or GraphRetrieverConfig()

    def process(
        self,
        query: str = "",
        graph_index_result: Optional[Dict[str, Any]] = None,
        retrieved_items: Optional[List[Dict[str, Any]]] = None,
        seed_node_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        metadata_index_result: Optional[Dict[str, Any]] = None,
        bm25_index_result: Optional[Dict[str, Any]] = None,
        vector_index_result: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        graph_index_result = graph_index_result or {}
        retrieved_items = retrieved_items or []
        seed_node_ids = seed_node_ids or []
        filters = filters or {}
        node_types = node_types or []
        edge_types = edge_types or []
        page_numbers = self._normalize_page_numbers(page_numbers or [])

        metadata_index_result = metadata_index_result or {}
        bm25_index_result = bm25_index_result or {}
        vector_index_result = vector_index_result or {}

        graph_index = self._unwrap_graph_index(graph_index_result)

        node_store = graph_index.get("node_store", {}) or {}
        edge_store = graph_index.get("edge_store", {}) or {}
        adjacency = graph_index.get("adjacency", {}) or {}
        reverse_adjacency = graph_index.get("reverse_adjacency", {}) or {}
        undirected_adjacency = graph_index.get("undirected_adjacency", {}) or {}
        text_index = graph_index.get("text_index", {}) or {}
        degree_stats = graph_index.get("degree_stats", {}) or {}
        nodes_by_page = graph_index.get("nodes_by_page", {}) or {}
        nodes_by_type = graph_index.get("nodes_by_type", {}) or {}

        metadata_store = self._collect_metadata_store(metadata_index_result)
        external_doc_store = self._collect_external_document_store(
            bm25_index_result=bm25_index_result,
            vector_index_result=vector_index_result,
        )

        seed_candidates = self._collect_seed_candidates(
            query=query,
            node_store=node_store,
            text_index=text_index,
            retrieved_items=retrieved_items,
            seed_node_ids=seed_node_ids,
            page_numbers=page_numbers,
            nodes_by_page=nodes_by_page,
            external_doc_store=external_doc_store,
        )

        seed_candidates = self._filter_candidate_node_ids(
            candidate_node_ids=seed_candidates,
            node_store=node_store,
            filters=filters,
            node_types=node_types,
            page_numbers=page_numbers,
        )

        seed_scored = self._score_seed_nodes(
            query=query,
            candidate_node_ids=seed_candidates,
            node_store=node_store,
            text_index=text_index,
            degree_stats=degree_stats,
            page_numbers=page_numbers,
            metadata_store=metadata_store,
        )

        seed_scored = [
            item for item in seed_scored
            if item.get("score", 0.0) >= self.config.min_score
        ]

        seed_scored = sorted(
            seed_scored,
            key=lambda item: item["score"],
            reverse=True,
        )[: self.config.max_seed_nodes]

        seed_node_ids_final = [
            item["node_id"]
            for item in seed_scored
        ]

        expanded_items = []
        graph_paths = []
        graph_edges_used = {}

        if self.config.expand_neighbors and seed_node_ids_final:
            expanded_items, graph_paths, graph_edges_used = self._expand_from_seeds(
                seed_node_ids=seed_node_ids_final,
                node_store=node_store,
                edge_store=edge_store,
                adjacency=adjacency,
                reverse_adjacency=reverse_adjacency,
                undirected_adjacency=undirected_adjacency,
                degree_stats=degree_stats,
                query=query,
                filters=filters,
                node_types=node_types,
                edge_types=edge_types,
                page_numbers=page_numbers,
                metadata_store=metadata_store,
            )

        seed_graph_items = [
            self._make_graph_item(
                node=node_store[item["node_id"]],
                score=item["score"],
                rank=index + 1,
                retrieval_type="seed",
                retrieval_reason=item.get("reason", "query_seed"),
                matched_terms=item.get("matched_terms", []),
                path=[],
                degree_stats=degree_stats.get(item["node_id"], {}),
            )
            for index, item in enumerate(seed_scored)
            if item["node_id"] in node_store
        ]

        all_items = []

        if self.config.include_seed_items:
            all_items.extend(seed_graph_items)

        if self.config.include_expanded_items:
            all_items.extend(expanded_items)

        if self.config.deduplicate_items:
            all_items = self._deduplicate_graph_items(all_items)

        all_items = self._sort_graph_items(all_items)

        all_items = all_items[: self.config.top_k]

        for rank, item in enumerate(all_items, start=1):
            item["rank"] = rank

        graph_context_text = ""

        if self.config.include_context_text:
            graph_context_text = self._build_context_text(
                query=query,
                graph_items=all_items,
                edge_store=edge_store,
            )

        result = {
            "processor": "GraphRetriever",
            "schema_version": "graph_retriever_v1",
            "query": query,
            "retrieved_graph_items": all_items,
            "seed_graph_items": seed_graph_items,
            "expanded_graph_items": expanded_items,
            "graph_paths": graph_paths if self.config.include_paths else [],
            "graph_edges_used": list(graph_edges_used.values()) if self.config.include_graph_edges else [],
            "graph_context_text": graph_context_text,
            "retrieved_by_page": self._group_items_by_page(all_items),
            "retrieved_by_type": self._group_items_by_type(all_items),
            "retrieved_by_section": self._group_items_by_section(all_items),
            "graph_retrieval_summary": self._build_summary(
                query=query,
                seed_candidates=seed_candidates,
                seed_graph_items=seed_graph_items,
                expanded_items=expanded_items,
                retrieved_graph_items=all_items,
                graph_paths=graph_paths,
                graph_edges_used=graph_edges_used,
                node_store=node_store,
                edge_store=edge_store,
                filters=filters,
                node_types=node_types,
                edge_types=edge_types,
                page_numbers=page_numbers,
            ),
            "config": asdict(self.config),
        }

        return json_safe(result)

    def search(
        self,
        graph_index_result: Dict[str, Any],
        query: str,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
        page_numbers: Optional[List[int]] = None,
        seed_node_ids: Optional[List[str]] = None,
        retrieved_items: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        old_top_k = self.config.top_k
        self.config.top_k = top_k

        result = self.process(
            query=query,
            graph_index_result=graph_index_result,
            retrieved_items=retrieved_items,
            seed_node_ids=seed_node_ids,
            filters=filters,
            node_types=node_types,
            edge_types=edge_types,
            page_numbers=page_numbers,
        )

        self.config.top_k = old_top_k

        return result.get("retrieved_graph_items", [])

    def expand(
        self,
        graph_index_result: Dict[str, Any],
        seed_node_ids: List[str],
        depth: int = 1,
        edge_types: Optional[List[str]] = None,
        node_types: Optional[List[str]] = None,
        max_nodes: int = 80,
    ) -> Dict[str, Any]:
        old_depth = self.config.max_depth
        old_max_nodes = self.config.max_expanded_nodes

        self.config.max_depth = depth
        self.config.max_expanded_nodes = max_nodes

        result = self.process(
            query="",
            graph_index_result=graph_index_result,
            seed_node_ids=seed_node_ids,
            edge_types=edge_types,
            node_types=node_types,
        )

        self.config.max_depth = old_depth
        self.config.max_expanded_nodes = old_max_nodes

        return {
            "seed_node_ids": seed_node_ids,
            "depth": depth,
            "expanded_graph_items": result.get("expanded_graph_items", []),
            "graph_paths": result.get("graph_paths", []),
            "graph_edges_used": result.get("graph_edges_used", []),
            "summary": result.get("graph_retrieval_summary", {}),
        }

    def _collect_seed_candidates(
        self,
        query: str,
        node_store: Dict[str, Dict[str, Any]],
        text_index: Dict[str, List[Dict[str, Any]]],
        retrieved_items: List[Dict[str, Any]],
        seed_node_ids: List[str],
        page_numbers: List[int],
        nodes_by_page: Dict[str, List[str]],
        external_doc_store: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        candidate_ids = []

        if self.config.use_seed_node_ids:
            for node_id in seed_node_ids:
                if node_id and node_id in node_store:
                    candidate_ids.append(node_id)

        if self.config.use_text_index and query and text_index:
            query_tokens = self._tokenize(query)

            for token in query_tokens:
                for posting in text_index.get(token, []) or []:
                    node_id = posting.get("node_id", "")

                    if node_id and node_id in node_store:
                        candidate_ids.append(node_id)

        if self.config.use_node_text_scan and query:
            scan_ids = self._scan_nodes_by_text(
                query=query,
                node_store=node_store,
            )
            candidate_ids.extend(scan_ids)

        if self.config.use_retrieved_items_as_seeds:
            mapped_ids = self._map_retrieved_items_to_node_ids(
                retrieved_items=retrieved_items,
                node_store=node_store,
                external_doc_store=external_doc_store,
            )
            candidate_ids.extend(mapped_ids)

        if self.config.use_page_filter_as_seed and page_numbers:
            for page_number in page_numbers:
                candidate_ids.extend(nodes_by_page.get(str(page_number), []) or [])

        return list(dict.fromkeys([item for item in candidate_ids if item]))

    def _scan_nodes_by_text(
        self,
        query: str,
        node_store: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        results = []

        for node_id, node in node_store.items():
            text = normalize_text_for_match(
                "\n".join(
                    [
                        node.get("label", ""),
                        node.get("text", ""),
                        node.get("source_type", ""),
                        node.get("node_type", ""),
                    ]
                )
            )

            if not text:
                continue

            matched = sum(1 for token in query_tokens if token in text)

            if matched > 0:
                results.append(
                    {
                        "node_id": node_id,
                        "matched": matched,
                    }
                )

        results = sorted(
            results,
            key=lambda item: item["matched"],
            reverse=True,
        )

        return [item["node_id"] for item in results[: max(self.config.max_seed_nodes * 3, 50)]]

    def _map_retrieved_items_to_node_ids(
        self,
        retrieved_items: List[Dict[str, Any]],
        node_store: Dict[str, Dict[str, Any]],
        external_doc_store: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        candidates = []

        node_ids = set(node_store.keys())

        for item in retrieved_items:
            item = self._to_dict(item)

            direct_node_id = item.get("node_id", "")

            if direct_node_id and direct_node_id in node_store:
                candidates.append(direct_node_id)

            source_id = item.get("source_id", "")

            if source_id and source_id in node_store:
                candidates.append(source_id)

            for prefix, key in [
                ("chunk", "chunk_id"),
                ("evidence", "evidence_id"),
                ("table", "table_id"),
                ("section", "section_id"),
                ("page", "page_number"),
            ]:
                value = item.get(key)

                if not value and isinstance(item.get("metadata"), dict):
                    value = item.get("metadata", {}).get(key)

                if value:
                    node_id = f"{prefix}_{value}"

                    if node_id in node_ids:
                        candidates.append(node_id)

            for table_id in self._table_ids_from_item(item):
                node_id = f"table_{table_id}"
                if node_id in node_ids:
                    candidates.append(node_id)

            document_id = item.get("document_id", "") or item.get("vector_id", "")

            if document_id and document_id in external_doc_store:
                external_doc = external_doc_store[document_id]

                for prefix, key in [
                    ("chunk", "chunk_id"),
                    ("evidence", "evidence_id"),
                    ("table", "table_id"),
                ]:
                    value = external_doc.get(key, "")

                    if value:
                        node_id = f"{prefix}_{value}"

                        if node_id in node_ids:
                            candidates.append(node_id)

        return list(dict.fromkeys([item for item in candidates if item]))

    def _score_seed_nodes(
        self,
        query: str,
        candidate_node_ids: List[str],
        node_store: Dict[str, Dict[str, Any]],
        text_index: Dict[str, List[Dict[str, Any]]],
        degree_stats: Dict[str, Dict[str, Any]],
        page_numbers: List[int],
        metadata_store: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        query_token_set = set(query_tokens)
        page_set = set(page_numbers)

        text_index_scores = self._text_index_scores(
            query_tokens=query_tokens,
            text_index=text_index,
        )

        scored = []

        for node_id in candidate_node_ids:
            node = node_store.get(node_id)

            if not node:
                continue

            node_text = normalize_text_for_match(
                "\n".join(
                    [
                        node.get("label", ""),
                        node.get("text", ""),
                        node.get("source_type", ""),
                        node.get("node_type", ""),
                    ]
                )
            )

            node_tokens = set(self._tokenize(node_text))

            query_score = 0.0
            matched_terms = []

            if query_token_set and node_tokens:
                matched_terms = sorted(list(query_token_set.intersection(node_tokens)))
                query_score = len(matched_terms) / max(len(query_token_set), 1)

            if node_id in text_index_scores:
                query_score = max(query_score, text_index_scores[node_id])

            confidence = self._safe_float(node.get("confidence"), default=0.70)
            weight = self._safe_float(node.get("weight"), default=1.0)
            node_type_boost = self._node_type_boost(node.get("node_type", ""))

            degree = degree_stats.get(node_id, {}).get("total_degree", 0)
            degree_score = math.log1p(degree)

            page_overlap_score = 0.0

            if page_set:
                node_pages = set(self._resolve_page_numbers(node))
                if node_pages:
                    page_overlap_score = len(page_set.intersection(node_pages)) / max(len(page_set), 1)

            metadata_score = self._metadata_overlap_score(node=node, metadata_store=metadata_store)

            score = (
                query_score * self.config.score_query_match
                + self.config.score_seed_match
                + confidence * self.config.score_confidence
                + weight * self.config.score_weight
                + degree_score * self.config.score_degree
                + page_overlap_score * self.config.score_page_overlap
                + metadata_score * self.config.score_metadata_overlap
            )

            score = score * node_type_boost

            scored.append(
                {
                    "node_id": node_id,
                    "score": round(score, 6),
                    "matched_terms": matched_terms,
                    "reason": self._seed_reason(
                        query_score=query_score,
                        page_overlap_score=page_overlap_score,
                        metadata_score=metadata_score,
                    ),
                }
            )

        return scored

    def _text_index_scores(
        self,
        query_tokens: List[str],
        text_index: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, float]:
        scores = {}

        if not query_tokens or not text_index:
            return scores

        unique_query_tokens = list(dict.fromkeys(query_tokens))

        for token in unique_query_tokens:
            postings = text_index.get(token, []) or []

            for posting in postings:
                node_id = posting.get("node_id", "")

                if not node_id:
                    continue

                tf = self._safe_float(posting.get("tf"), default=1.0)
                scores[node_id] = scores.get(node_id, 0.0) + math.log1p(tf)

        max_score = max(scores.values()) if scores else 1.0

        if max_score > 0:
            scores = {
                node_id: score / max_score
                for node_id, score in scores.items()
            }

        return scores

    def _expand_from_seeds(
        self,
        seed_node_ids: List[str],
        node_store: Dict[str, Dict[str, Any]],
        edge_store: Dict[str, Dict[str, Any]],
        adjacency: Dict[str, List[Dict[str, Any]]],
        reverse_adjacency: Dict[str, List[Dict[str, Any]]],
        undirected_adjacency: Dict[str, List[Dict[str, Any]]],
        degree_stats: Dict[str, Dict[str, Any]],
        query: str,
        filters: Dict[str, Any],
        node_types: List[str],
        edge_types: List[str],
        page_numbers: List[int],
        metadata_store: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        expanded_items = []
        graph_paths = []
        graph_edges_used = {}

        edge_type_set = set(edge_types or [])

        visited = set(seed_node_ids)
        queue = []

        for seed_node_id in seed_node_ids:
            queue.append(
                {
                    "node_id": seed_node_id,
                    "depth": 0,
                    "path": [seed_node_id],
                    "edge_path": [],
                    "score": self.config.score_seed_match,
                    "seed_node_id": seed_node_id,
                }
            )

        while queue and len(expanded_items) < self.config.max_expanded_nodes:
            current = queue.pop(0)
            current_node_id = current["node_id"]
            current_depth = current["depth"]

            if current_depth >= self.config.max_depth:
                continue

            neighbor_infos = self._neighbors_for_node(
                node_id=current_node_id,
                adjacency=adjacency,
                reverse_adjacency=reverse_adjacency,
                undirected_adjacency=undirected_adjacency,
            )

            for neighbor_info in neighbor_infos[: self.config.max_neighbors_per_node]:
                neighbor_id = neighbor_info.get("neighbor_id", "")
                edge_id = neighbor_info.get("edge_id", "")

                if not neighbor_id or neighbor_id not in node_store:
                    continue

                if neighbor_id in visited:
                    continue

                edge = edge_store.get(edge_id, {})

                if edge_type_set and edge.get("edge_type") not in edge_type_set and neighbor_info.get("edge_type") not in edge_type_set:
                    continue

                neighbor_node = node_store.get(neighbor_id, {})

                if not self._passes_filters(
                    node=neighbor_node,
                    filters=filters,
                    node_types=node_types,
                    page_numbers=page_numbers,
                ):
                    continue

                visited.add(neighbor_id)

                next_depth = current_depth + 1
                path = current["path"] + [neighbor_id]
                edge_path = current["edge_path"] + ([edge_id] if edge_id else [])

                neighbor_score = self._score_neighbor_node(
                    node=neighbor_node,
                    edge=edge,
                    neighbor_info=neighbor_info,
                    query=query,
                    depth=next_depth,
                    degree_stats=degree_stats.get(neighbor_id, {}),
                    metadata_store=metadata_store,
                )

                item = self._make_graph_item(
                    node=neighbor_node,
                    score=neighbor_score,
                    rank=0,
                    retrieval_type="expanded",
                    retrieval_reason=edge.get("edge_type", neighbor_info.get("edge_type", "graph_neighbor")),
                    matched_terms=self._matched_terms(query, neighbor_node),
                    path=path,
                    edge_path=edge_path,
                    degree_stats=degree_stats.get(neighbor_id, {}),
                    edge=edge,
                    seed_node_id=current["seed_node_id"],
                    depth=next_depth,
                )

                expanded_items.append(item)

                if edge_id and edge_id in edge_store:
                    graph_edges_used[edge_id] = edge_store[edge_id]

                graph_paths.append(
                    {
                        "path_id": self._stable_id(
                            {
                                "nodes": path,
                                "edges": edge_path,
                            },
                            "graph_path",
                        ),
                        "seed_node_id": current["seed_node_id"],
                        "target_node_id": neighbor_id,
                        "depth": next_depth,
                        "node_path": path,
                        "edge_path": edge_path,
                        "score": neighbor_score,
                        "relation_type": edge.get("edge_type", neighbor_info.get("edge_type", "")),
                    }
                )

                if next_depth < self.config.max_depth:
                    queue.append(
                        {
                            "node_id": neighbor_id,
                            "depth": next_depth,
                            "path": path,
                            "edge_path": edge_path,
                            "score": neighbor_score,
                            "seed_node_id": current["seed_node_id"],
                        }
                    )

                if len(expanded_items) >= self.config.max_expanded_nodes:
                    break

        return expanded_items, graph_paths, graph_edges_used

    def _neighbors_for_node(
        self,
        node_id: str,
        adjacency: Dict[str, List[Dict[str, Any]]],
        reverse_adjacency: Dict[str, List[Dict[str, Any]]],
        undirected_adjacency: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        neighbors = []

        if self.config.expand_undirected_edges and undirected_adjacency:
            neighbors.extend(undirected_adjacency.get(node_id, []) or [])
        else:
            if self.config.expand_forward_edges:
                neighbors.extend(adjacency.get(node_id, []) or [])

            if self.config.expand_reverse_edges:
                neighbors.extend(reverse_adjacency.get(node_id, []) or [])

        result = []
        seen = set()

        for item in neighbors:
            key = (
                item.get("neighbor_id", ""),
                item.get("edge_id", ""),
                item.get("direction", ""),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        result = sorted(
            result,
            key=lambda item: (
                -self._safe_float(item.get("weight"), default=1.0),
                -self._safe_float(item.get("confidence"), default=0.70),
                item.get("edge_type", ""),
                item.get("neighbor_id", ""),
            ),
        )

        return result

    def _score_neighbor_node(
        self,
        node: Dict[str, Any],
        edge: Dict[str, Any],
        neighbor_info: Dict[str, Any],
        query: str,
        depth: int,
        degree_stats: Dict[str, Any],
        metadata_store: Dict[str, Dict[str, Any]],
    ) -> float:
        query_overlap = self._query_overlap(query, node.get("text", "") + " " + node.get("label", ""))
        confidence = self._safe_float(node.get("confidence"), default=0.70)
        node_weight = self._safe_float(node.get("weight"), default=1.0)
        edge_weight = self._safe_float(edge.get("weight", neighbor_info.get("weight", 1.0)), default=1.0)
        edge_confidence = self._safe_float(edge.get("confidence", neighbor_info.get("confidence", 0.70)), default=0.70)
        degree = self._safe_float(degree_stats.get("total_degree"), default=0.0)
        degree_score = math.log1p(degree)
        metadata_score = self._metadata_overlap_score(node=node, metadata_store=metadata_store)

        base = (
            self.config.score_neighbor
            + query_overlap * self.config.score_query_match
            + confidence * self.config.score_confidence
            + node_weight * self.config.score_weight
            + edge_weight * 0.20
            + edge_confidence * 0.20
            + degree_score * self.config.score_degree
            + metadata_score * self.config.score_metadata_overlap
        )

        base = base * self._node_type_boost(node.get("node_type", ""))

        depth_decay = self.config.score_path_decay ** max(depth - 1, 0)

        return round(base * depth_decay, 6)

    def _make_graph_item(
        self,
        node: Dict[str, Any],
        score: float,
        rank: int,
        retrieval_type: str,
        retrieval_reason: str,
        matched_terms: Optional[List[str]] = None,
        path: Optional[List[str]] = None,
        edge_path: Optional[List[str]] = None,
        degree_stats: Optional[Dict[str, Any]] = None,
        edge: Optional[Dict[str, Any]] = None,
        seed_node_id: str = "",
        depth: int = 0,
    ) -> Dict[str, Any]:
        matched_terms = matched_terms or []
        path = path or []
        edge_path = edge_path or []
        degree_stats = degree_stats or {}
        edge = edge or {}

        text = normalize_pdf_text(node.get("text", "") or node.get("label", ""))
        text = self._truncate_text(text, self.config.max_text_chars_per_item)

        metadata = node.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        return {
            "rank": rank,
            "score": round(score, 6),
            "retrieval_type": retrieval_type,
            "retrieval_reason": retrieval_reason,
            "node_id": node.get("node_id", ""),
            "node_type": node.get("node_type", ""),
            "label": node.get("label", ""),
            "title": node.get("label", ""),
            "text": text,
            "text_preview": self._preview(text, self.config.text_preview_chars),
            "normalized_text": normalize_text_for_match(text),
            "matched_terms": matched_terms,
            "page_number": node.get("page_number"),
            "page_numbers": self._resolve_page_numbers(node),
            "page_start": node.get("page_start"),
            "page_end": node.get("page_end"),
            "source_id": node.get("source_id", ""),
            "source_type": node.get("source_type", ""),
            "source": node.get("source", ""),
            "section_id": node.get("section_id", "") or metadata.get("section_id", ""),
            "chunk_id": node.get("chunk_id", "") or metadata.get("chunk_id", ""),
            "table_id": node.get("table_id", "") or metadata.get("table_id", ""),
            "evidence_id": node.get("evidence_id", "") or metadata.get("evidence_id", ""),
            "bbox": node.get("bbox", []) or [],
            "confidence": self._safe_float(node.get("confidence"), default=0.70),
            "weight": self._safe_float(node.get("weight"), default=1.0),
            "degree_stats": degree_stats,
            "seed_node_id": seed_node_id,
            "depth": depth,
            "path": path,
            "edge_path": edge_path,
            "via_edge": {
                "edge_id": edge.get("edge_id", ""),
                "edge_type": edge.get("edge_type", ""),
                "relation_label": edge.get("relation_label", ""),
                "confidence": edge.get("confidence", 0.0),
                "weight": edge.get("weight", 1.0),
            } if edge else {},
            "metadata": metadata if self.config.include_debug else {},
        }

    def _filter_candidate_node_ids(
        self,
        candidate_node_ids: List[str],
        node_store: Dict[str, Dict[str, Any]],
        filters: Dict[str, Any],
        node_types: List[str],
        page_numbers: List[int],
    ) -> List[str]:
        result = []

        for node_id in candidate_node_ids:
            node = node_store.get(node_id)

            if not node:
                continue

            if self._passes_filters(
                node=node,
                filters=filters,
                node_types=node_types,
                page_numbers=page_numbers,
            ):
                result.append(node_id)

        return list(dict.fromkeys(result))

    def _passes_filters(
        self,
        node: Dict[str, Any],
        filters: Dict[str, Any],
        node_types: List[str],
        page_numbers: List[int],
    ) -> bool:
        if node_types and node.get("node_type", "") not in node_types:
            return False

        if page_numbers:
            target_pages = set(page_numbers)
            node_pages = set(self._resolve_page_numbers(node))

            if node_pages and not target_pages.intersection(node_pages):
                return False

            if not node_pages:
                return False

        if filters:
            metadata = node.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}

            for key, expected in filters.items():
                if expected is None:
                    continue

                if key == "page_numbers":
                    expected_pages = set(self._normalize_page_numbers(expected))
                    node_pages = set(self._resolve_page_numbers(node))

                    if expected_pages and not expected_pages.intersection(node_pages):
                        return False

                    continue

                actual = node.get(key)

                if actual is None or actual == "":
                    actual = metadata.get(key)

                if isinstance(expected, list):
                    if actual not in expected:
                        return False
                else:
                    if actual != expected:
                        return False

        return True

    def _unwrap_graph_index(
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
            "text_index": graph_index.get("text_index", {}) or graph_index_result.get("text_index", {}) or {},
            "degree_stats": graph_index.get("degree_stats", {}) or graph_index_result.get("degree_stats", {}) or {},
            "nodes_by_page": graph_index.get("nodes_by_page", {}) or graph_index_result.get("nodes_by_page", {}) or {},
            "nodes_by_type": graph_index.get("nodes_by_type", {}) or graph_index_result.get("nodes_by_type", {}) or {},
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

    def _collect_external_document_store(
        self,
        bm25_index_result: Dict[str, Any],
        vector_index_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        result = {}

        for index_result in [bm25_index_result, vector_index_result]:
            index_obj = index_result.get("bm25_index", {}) or index_result.get("vector_index", {}) or index_result or {}
            document_store = index_obj.get("document_store", {}) or {}

            if not isinstance(document_store, dict):
                continue

            for doc_id, doc in document_store.items():
                doc = self._to_dict(doc)
                doc.setdefault("document_id", doc_id)
                result[str(doc_id)] = doc

        return result

    def _metadata_overlap_score(
        self,
        node: Dict[str, Any],
        metadata_store: Dict[str, Dict[str, Any]],
    ) -> float:
        if not metadata_store:
            return 0.0

        metadata = node.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        keys = []

        for key in ["section_id", "chunk_id", "evidence_id", "table_id", "source_id"]:
            value = node.get(key) or metadata.get(key)

            if value:
                keys.append((key, str(value)))

        if not keys:
            return 0.0

        matched = 0

        for item in metadata_store.values():
            for key, value in keys:
                if str(item.get(key, "")) == value:
                    matched += 1
                    break

        return min(matched / 5.0, 1.0)

    def _query_overlap(
        self,
        query: str,
        text: str,
    ) -> float:
        query_tokens = self._tokenize(query)

        if not query_tokens:
            return 0.0

        text_tokens = set(self._tokenize(text))

        if not text_tokens:
            return 0.0

        matched = sum(1 for token in query_tokens if token in text_tokens)

        return matched / max(len(query_tokens), 1)

    def _matched_terms(
        self,
        query: str,
        node: Dict[str, Any],
    ) -> List[str]:
        query_tokens = set(self._tokenize(query))
        node_tokens = set(
            self._tokenize(
                "\n".join(
                    [
                        node.get("label", ""),
                        node.get("text", ""),
                        node.get("source_type", ""),
                        node.get("node_type", ""),
                    ]
                )
            )
        )

        return sorted(list(query_tokens.intersection(node_tokens)))

    def _node_type_boost(
        self,
        node_type: str,
    ) -> float:
        node_type = normalize_text_for_match(node_type)

        if "evidence" in node_type:
            return self.config.boost_evidence_node

        if "chunk" in node_type:
            return self.config.boost_chunk_node

        if "table" in node_type:
            return self.config.boost_table_node

        if "section" in node_type:
            return self.config.boost_section_node

        if "page" in node_type:
            return self.config.boost_page_node

        if "entity" in node_type:
            return self.config.boost_entity_node

        if "reference" in node_type:
            return self.config.boost_reference_node

        return 1.0

    def _seed_reason(
        self,
        query_score: float,
        page_overlap_score: float,
        metadata_score: float,
    ) -> str:
        reasons = []

        if query_score > 0:
            reasons.append("query_match")

        if page_overlap_score > 0:
            reasons.append("page_overlap")

        if metadata_score > 0:
            reasons.append("metadata_overlap")

        if not reasons:
            reasons.append("seed_reference")

        return "+".join(reasons)

    def _deduplicate_graph_items(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_node_id = {}

        for item in items:
            node_id = item.get("node_id", "")

            if not node_id:
                continue

            if node_id not in by_node_id:
                by_node_id[node_id] = item
            else:
                existing = by_node_id[node_id]

                if item.get("score", 0.0) > existing.get("score", 0.0):
                    merged = {
                        **existing,
                        **item,
                    }
                    merged["score"] = max(existing.get("score", 0.0), item.get("score", 0.0))
                    merged["retrieval_type"] = existing.get("retrieval_type", "") + "|" + item.get("retrieval_type", "")
                    by_node_id[node_id] = merged

        return list(by_node_id.values())

    def _sort_graph_items(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return sorted(
            items,
            key=lambda item: (
                -self._safe_float(item.get("score"), default=0.0),
                self._retrieval_type_order(item.get("retrieval_type", "")),
                min(item.get("page_numbers", []) or [999999]),
                item.get("node_type", ""),
                item.get("node_id", ""),
            ),
        )

    def _retrieval_type_order(
        self,
        retrieval_type: str,
    ) -> int:
        if "seed" in retrieval_type:
            return 0

        if "expanded" in retrieval_type:
            return 1

        return 9

    def _build_context_text(
        self,
        query: str,
        graph_items: List[Dict[str, Any]],
        edge_store: Dict[str, Dict[str, Any]],
    ) -> str:
        parts = []

        if query:
            parts.append(f"Truy vấn: {normalize_pdf_text(query)}")

        total_chars = 0

        for index, item in enumerate(graph_items, start=1):
            page_numbers = item.get("page_numbers", []) or []
            page_label = ""

            if page_numbers:
                if len(page_numbers) == 1:
                    page_label = f"trang {page_numbers[0]}"
                else:
                    page_label = f"trang {page_numbers[0]}-{page_numbers[-1]}"

            header_parts = [
                f"[Graph {index}]",
                f"type={item.get('node_type', '')}",
            ]

            if item.get("label"):
                header_parts.append(item.get("label", ""))

            if page_label:
                header_parts.append(page_label)

            if item.get("retrieval_reason"):
                header_parts.append(f"relation={item.get('retrieval_reason')}")

            text = normalize_pdf_text(item.get("text", ""))

            block = "\n".join(
                [
                    " | ".join([part for part in header_parts if part]),
                    text,
                ]
            )

            if total_chars + len(block) > self.config.max_context_chars:
                break

            parts.append(block)
            total_chars += len(block)

        return normalize_pdf_text("\n\n".join(parts))

    def _group_items_by_page(
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

    def _group_items_by_type(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for item in items:
            node_type = item.get("node_type", "unknown")
            grouped.setdefault(node_type, [])
            grouped[node_type].append(item)

        return grouped

    def _group_items_by_section(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for item in items:
            section_id = item.get("section_id", "") or "no_section"
            grouped.setdefault(section_id, [])
            grouped[section_id].append(item)

        return grouped

    def _build_summary(
        self,
        query: str,
        seed_candidates: List[str],
        seed_graph_items: List[Dict[str, Any]],
        expanded_items: List[Dict[str, Any]],
        retrieved_graph_items: List[Dict[str, Any]],
        graph_paths: List[Dict[str, Any]],
        graph_edges_used: Dict[str, Dict[str, Any]],
        node_store: Dict[str, Dict[str, Any]],
        edge_store: Dict[str, Dict[str, Any]],
        filters: Dict[str, Any],
        node_types: List[str],
        edge_types: List[str],
        page_numbers: List[int],
    ) -> Dict[str, Any]:
        by_node_type = {}
        by_retrieval_type = {}
        by_page = {}

        scores = []

        for item in retrieved_graph_items:
            node_type = item.get("node_type", "unknown")
            retrieval_type = item.get("retrieval_type", "unknown")

            by_node_type[node_type] = by_node_type.get(node_type, 0) + 1
            by_retrieval_type[retrieval_type] = by_retrieval_type.get(retrieval_type, 0) + 1
            scores.append(self._safe_float(item.get("score"), default=0.0))

            for page_number in item.get("page_numbers", []) or []:
                page_key = str(page_number)
                by_page[page_key] = by_page.get(page_key, 0) + 1

        return {
            "has_graph_results": len(retrieved_graph_items) > 0,
            "query": query,
            "node_count_in_index": len(node_store),
            "edge_count_in_index": len(edge_store),
            "seed_candidate_count": len(seed_candidates),
            "seed_graph_item_count": len(seed_graph_items),
            "expanded_graph_item_count": len(expanded_items),
            "retrieved_graph_item_count": len(retrieved_graph_items),
            "graph_path_count": len(graph_paths),
            "graph_edge_used_count": len(graph_edges_used),
            "avg_score": round(sum(scores) / max(len(scores), 1), 6),
            "max_score": round(max(scores), 6) if scores else 0.0,
            "min_score": round(min(scores), 6) if scores else 0.0,
            "by_node_type": by_node_type,
            "by_retrieval_type": by_retrieval_type,
            "by_page": by_page,
            "filters": filters,
            "node_types_filter": node_types,
            "edge_types_filter": edge_types,
            "page_numbers_filter": page_numbers,
            "max_depth": self.config.max_depth,
            "top_k": self.config.top_k,
        }

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

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:
        text = normalize_text_for_match(text)
        raw_tokens = re.findall(r"[a-z0-9_]+", text)

        tokens = []

        for token in raw_tokens:
            token = token.strip("_")

            if len(token) < self.config.min_query_token_len:
                continue

            tokens.append(token)

        return tokens

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

        if isinstance(metadata, dict) and metadata is not item:
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

    def save_result(
        self,
        retrieval_result: Dict[str, Any],
        output_path: str,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                json_safe(retrieval_result),
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


def retrieve_from_graph(
    query: str = "",
    graph_index_result: Optional[Dict[str, Any]] = None,
    retrieved_items: Optional[List[Dict[str, Any]]] = None,
    seed_node_ids: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    node_types: Optional[List[str]] = None,
    edge_types: Optional[List[str]] = None,
    page_numbers: Optional[List[int]] = None,
    metadata_index_result: Optional[Dict[str, Any]] = None,
    bm25_index_result: Optional[Dict[str, Any]] = None,
    vector_index_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    retriever = GraphRetriever()
    return retriever.process(
        query=query,
        graph_index_result=graph_index_result,
        retrieved_items=retrieved_items,
        seed_node_ids=seed_node_ids,
        filters=filters,
        node_types=node_types,
        edge_types=edge_types,
        page_numbers=page_numbers,
        metadata_index_result=metadata_index_result,
        bm25_index_result=bm25_index_result,
        vector_index_result=vector_index_result,
    )


def search_graph(
    graph_index_result: Dict[str, Any],
    query: str,
    top_k: int = 20,
    filters: Optional[Dict[str, Any]] = None,
    node_types: Optional[List[str]] = None,
    edge_types: Optional[List[str]] = None,
    page_numbers: Optional[List[int]] = None,
    seed_node_ids: Optional[List[str]] = None,
    retrieved_items: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    retriever = GraphRetriever()
    return retriever.search(
        graph_index_result=graph_index_result,
        query=query,
        top_k=top_k,
        filters=filters,
        node_types=node_types,
        edge_types=edge_types,
        page_numbers=page_numbers,
        seed_node_ids=seed_node_ids,
        retrieved_items=retrieved_items,
    )


def expand_graph_context(
    graph_index_result: Dict[str, Any],
    seed_node_ids: List[str],
    depth: int = 1,
    edge_types: Optional[List[str]] = None,
    node_types: Optional[List[str]] = None,
    max_nodes: int = 80,
) -> Dict[str, Any]:
    retriever = GraphRetriever()
    return retriever.expand(
        graph_index_result=graph_index_result,
        seed_node_ids=seed_node_ids,
        depth=depth,
        edge_types=edge_types,
        node_types=node_types,
        max_nodes=max_nodes,
    )
