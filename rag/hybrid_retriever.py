"""
hybrid_retriever.py

Production V1 - Colab Ready

Purpose
-------
Hybrid retrieval over:
- BM25 keyword index
- Vector semantic index
- Graph index
- Metadata index
- Table-aware index / table chunks

Used by:
- RAGPipeline
- ContextExpander
- EvidenceAggregator
- PromptBuilder
- LLMReasoner

Input
-----
- query
- bm25_index_result
- vector_index_result
- graph_index_result
- metadata_index_result
- table_chunk_result
- table_understanding_result
- filters
- page_numbers
- section_ids
- table_ids

Output
------
Dictionary with:
- retrieved_items
- bm25_results
- vector_results
- graph_results
- metadata_results
- table_results
- fused_results
- retrieval_summary
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
    normalize_pdf_text,
    normalize_text_for_match,
    json_safe,
)


@dataclass
class HybridRetrieverConfig:
    top_k: int = 20
    candidate_pool_size: int = 80

    use_bm25: bool = True
    use_vector: bool = True
    use_graph: bool = True
    use_metadata: bool = True
    use_table: bool = True

    bm25_weight: float = 1.00
    vector_weight: float = 1.15
    graph_weight: float = 0.85
    metadata_weight: float = 0.55
    table_weight: float = 1.10

    rrf_k: int = 60
    use_reciprocal_rank_fusion: bool = True
    use_score_normalization: bool = True

    boost_exact_phrase: float = 0.35
    boost_title_match: float = 0.25
    boost_page_filter_match: float = 0.20
    boost_section_filter_match: float = 0.20
    boost_table_filter_match: float = 0.25
    boost_table_result: float = 0.15
    boost_evidence_result: float = 0.18
    boost_chunk_result: float = 0.12

    min_score: float = 0.0
    deduplicate_results: bool = True

    max_text_chars: int = 1800
    text_preview_chars: int = 700

    include_source_scores: bool = True
    include_context_text: bool = True
    max_context_chars: int = 12000

    include_debug: bool = True


class HybridRetriever:
    def __init__(
        self,
        config: Optional[HybridRetrieverConfig] = None,
    ):
        self.config = config or HybridRetrieverConfig()

    def process(
        self,
        query: str,
        bm25_index_result: Optional[Dict[str, Any]] = None,
        vector_index_result: Optional[Dict[str, Any]] = None,
        graph_index_result: Optional[Dict[str, Any]] = None,
        metadata_index_result: Optional[Dict[str, Any]] = None,
        table_chunk_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        page_numbers: Optional[List[int]] = None,
        section_ids: Optional[List[str]] = None,
        table_ids: Optional[List[str]] = None,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        query = normalize_pdf_text(query)
        bm25_index_result = bm25_index_result or {}
        vector_index_result = vector_index_result or {}
        graph_index_result = graph_index_result or {}
        metadata_index_result = metadata_index_result or {}
        table_chunk_result = table_chunk_result or {}
        table_understanding_result = table_understanding_result or {}
        filters = filters or {}
        page_numbers = self._normalize_page_numbers(page_numbers or [])
        section_ids = [str(item) for item in (section_ids or []) if item]
        table_ids = [str(item) for item in (table_ids or []) if item]
        node_types = node_types or []
        edge_types = edge_types or []

        final_top_k = top_k or self.config.top_k
        candidate_k = max(self.config.candidate_pool_size, final_top_k * 4)

        merged_filters = self._merge_filters(
            filters=filters,
            page_numbers=page_numbers,
            section_ids=section_ids,
            table_ids=table_ids,
        )

        bm25_results = []
        vector_results = []
        graph_results = []
        metadata_results = []
        table_results = []

        if self.config.use_bm25:
            bm25_results = self._retrieve_bm25(
                query=query,
                bm25_index_result=bm25_index_result,
                filters=merged_filters,
                top_k=candidate_k,
            )

        if self.config.use_vector:
            vector_results = self._retrieve_vector(
                query=query,
                vector_index_result=vector_index_result,
                filters=merged_filters,
                top_k=candidate_k,
            )

        if self.config.use_graph:
            graph_results = self._retrieve_graph(
                query=query,
                graph_index_result=graph_index_result,
                filters=merged_filters,
                node_types=node_types,
                edge_types=edge_types,
                page_numbers=page_numbers,
                top_k=candidate_k,
            )

        if self.config.use_metadata:
            metadata_results = self._retrieve_metadata(
                query=query,
                metadata_index_result=metadata_index_result,
                filters=merged_filters,
                page_numbers=page_numbers,
                top_k=candidate_k,
            )

        if self.config.use_table:
            table_results = self._retrieve_table(
                query=query,
                table_chunk_result=table_chunk_result,
                table_understanding_result=table_understanding_result,
                metadata_index_result=metadata_index_result,
                filters=merged_filters,
                page_numbers=page_numbers,
                table_ids=table_ids,
                top_k=candidate_k,
            )

        normalized_sources = {
            "bm25": self._normalize_source_scores(bm25_results, "bm25"),
            "vector": self._normalize_source_scores(vector_results, "vector"),
            "graph": self._normalize_source_scores(graph_results, "graph"),
            "metadata": self._normalize_source_scores(metadata_results, "metadata"),
            "table": self._normalize_source_scores(table_results, "table"),
        }

        fused_results = self._fuse_results(
            query=query,
            source_results=normalized_sources,
            page_numbers=page_numbers,
            section_ids=section_ids,
            table_ids=table_ids,
        )

        if self.config.deduplicate_results:
            fused_results = self._deduplicate_fused_results(fused_results)

        fused_results = [
            item for item in fused_results
            if item.get("score", 0.0) >= self.config.min_score
        ]

        fused_results = sorted(
            fused_results,
            key=lambda item: (
                -self._safe_float(item.get("score"), default=0.0),
                min(item.get("page_numbers", []) or [999999]),
                item.get("source_type", ""),
                item.get("source_id", ""),
            ),
        )

        fused_results = fused_results[:final_top_k]

        for rank, item in enumerate(fused_results, start=1):
            item["rank"] = rank

        context_text = ""

        if self.config.include_context_text:
            context_text = self._build_context_text(
                query=query,
                results=fused_results,
            )

        result = {
            "processor": "HybridRetriever",
            "schema_version": "hybrid_retriever_v1",
            "query": query,
            "retrieved_items": fused_results,
            "fused_results": fused_results,
            "bm25_results": bm25_results,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "metadata_results": metadata_results,
            "table_results": table_results,
            "retrieval_context_text": context_text,
            "retrieved_by_page": self._group_by_page(fused_results),
            "retrieved_by_source": self._group_by_source(fused_results),
            "retrieved_by_section": self._group_by_section(fused_results),
            "retrieved_by_table": self._group_by_table(fused_results),
            "retrieval_summary": self._build_summary(
                query=query,
                bm25_results=bm25_results,
                vector_results=vector_results,
                graph_results=graph_results,
                metadata_results=metadata_results,
                table_results=table_results,
                fused_results=fused_results,
                filters=merged_filters,
                page_numbers=page_numbers,
                section_ids=section_ids,
                table_ids=table_ids,
            ),
            "config": asdict(self.config),
        }

        return json_safe(result)

    def search(
        self,
        query: str,
        bm25_index_result: Optional[Dict[str, Any]] = None,
        vector_index_result: Optional[Dict[str, Any]] = None,
        graph_index_result: Optional[Dict[str, Any]] = None,
        metadata_index_result: Optional[Dict[str, Any]] = None,
        table_chunk_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        page_numbers: Optional[List[int]] = None,
        section_ids: Optional[List[str]] = None,
        table_ids: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        result = self.process(
            query=query,
            bm25_index_result=bm25_index_result,
            vector_index_result=vector_index_result,
            graph_index_result=graph_index_result,
            metadata_index_result=metadata_index_result,
            table_chunk_result=table_chunk_result,
            table_understanding_result=table_understanding_result,
            filters=filters,
            page_numbers=page_numbers,
            section_ids=section_ids,
            table_ids=table_ids,
            top_k=top_k,
        )
        return result.get("retrieved_items", [])

    def _retrieve_bm25(
        self,
        query: str,
        bm25_index_result: Dict[str, Any],
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        try:
            from document_ai.indexing.bm25_index_builder import search_bm25_index

            results = search_bm25_index(
                index_result=bm25_index_result,
                query=query,
                top_k=top_k,
                filters=filters,
            )
        except Exception:
            results = self._fallback_bm25_search(
                index_result=bm25_index_result,
                query=query,
                top_k=top_k,
                filters=filters,
            )

        return [
            self._normalize_retrieved_item(
                item=item,
                retrieval_source="bm25",
                default_source_type="bm25_document",
            )
            for item in results
        ]

    def _retrieve_vector(
        self,
        query: str,
        vector_index_result: Dict[str, Any],
        filters: Dict[str, Any],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        try:
            from document_ai.indexing.vector_index_builder import search_vector_index

            results = search_vector_index(
                index_result=vector_index_result,
                query=query,
                top_k=top_k,
                filters=filters,
            )
        except Exception:
            results = self._fallback_vector_search(
                index_result=vector_index_result,
                query=query,
                top_k=top_k,
                filters=filters,
            )

        return [
            self._normalize_retrieved_item(
                item=item,
                retrieval_source="vector",
                default_source_type="vector_document",
            )
            for item in results
        ]

    def _retrieve_graph(
        self,
        query: str,
        graph_index_result: Dict[str, Any],
        filters: Dict[str, Any],
        node_types: List[str],
        edge_types: List[str],
        page_numbers: List[int],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        try:
            from document_ai.rag.graph_retriever import search_graph

            results = search_graph(
                graph_index_result=graph_index_result,
                query=query,
                top_k=top_k,
                filters=filters,
                node_types=node_types,
                edge_types=edge_types,
                page_numbers=page_numbers,
            )
        except Exception:
            results = self._fallback_graph_search(
                index_result=graph_index_result,
                query=query,
                top_k=top_k,
                filters=filters,
                page_numbers=page_numbers,
            )

        return [
            self._normalize_retrieved_item(
                item=item,
                retrieval_source="graph",
                default_source_type="graph_node",
            )
            for item in results
        ]

    def _retrieve_metadata(
        self,
        query: str,
        metadata_index_result: Dict[str, Any],
        filters: Dict[str, Any],
        page_numbers: List[int],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        try:
            from document_ai.indexing.metadata_index_builder import search_metadata_index

            results = search_metadata_index(
                index_result=metadata_index_result,
                filters=filters,
                query=query,
                page_numbers=page_numbers,
                top_k=top_k,
            )
        except Exception:
            results = self._fallback_metadata_search(
                index_result=metadata_index_result,
                query=query,
                top_k=top_k,
                filters=filters,
                page_numbers=page_numbers,
            )

        return [
            self._normalize_retrieved_item(
                item=item,
                retrieval_source="metadata",
                default_source_type="metadata_item",
            )
            for item in results
        ]

    def _retrieve_table(
        self,
        query: str,
        table_chunk_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
        filters: Dict[str, Any],
        page_numbers: List[int],
        table_ids: List[str],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        candidates = []

        for source in [table_chunk_result, table_understanding_result]:
            for key in [
                "table_chunks",
                "table_summary_chunks",
                "table_record_chunks",
                "table_row_chunks",
                "multi_page_table_chunks",
                "table_cell_context_chunks",
                "table_semantics",
                "multi_page_tables",
            ]:
                values = source.get(key, []) or []
                if isinstance(values, list):
                    candidates.extend([self._to_dict(item) for item in values])

            for sub_key in [
                "table_chunk_result",
                "table_semantic_result",
                "multi_page_table_result",
                "table_understanding_result",
            ]:
                sub = source.get(sub_key, {}) or {}
                if not isinstance(sub, dict):
                    continue

                for key in [
                    "table_chunks",
                    "table_summary_chunks",
                    "table_record_chunks",
                    "table_row_chunks",
                    "multi_page_table_chunks",
                    "table_cell_context_chunks",
                    "table_semantics",
                    "multi_page_tables",
                ]:
                    values = sub.get(key, []) or []
                    if isinstance(values, list):
                        candidates.extend([self._to_dict(item) for item in values])

        metadata_index = metadata_index_result.get("metadata_index", metadata_index_result) or {}
        metadata_store = metadata_index.get("metadata_store", {}) or {}

        if isinstance(metadata_store, dict):
            for item in metadata_store.values():
                item = self._to_dict(item)
                item_type = item.get("item_type", "")
                source_type = item.get("source_type", "")
                if "table" in f"{item_type} {source_type}".lower() or item.get("table_id"):
                    candidates.append(item)

        candidates = self._deduplicate_raw_items(
            candidates,
            ["chunk_id", "table_id", "table_semantic_id", "table_grid_id", "item_id"],
        )

        query_tokens = self._tokenize(query)
        scored = []

        table_id_set = set(table_ids or [])
        page_set = set(page_numbers or [])

        for item in candidates:
            if not self._passes_filters(item, filters):
                continue

            item_table_ids = set(self._table_ids_from_item(item))
            if table_id_set and not table_id_set.intersection(item_table_ids):
                continue

            item_pages = set(self._resolve_page_numbers(item))
            if page_set and item_pages and not page_set.intersection(item_pages):
                continue
            if page_set and not item_pages:
                continue

            text = normalize_pdf_text(
                item.get("text")
                or item.get("text_preview")
                or item.get("title")
                or item.get("caption")
                or item.get("caption_text")
                or item.get("label")
                or ""
            )

            if not text:
                headers = item.get("column_headers", []) or []
                if headers:
                    text = " | ".join([normalize_pdf_text(h) for h in headers if normalize_pdf_text(h)])

            if not text:
                continue

            text_norm = normalize_text_for_match(text)
            title_norm = normalize_text_for_match(item.get("title") or item.get("caption") or "")

            overlap = self._query_overlap_score(query_tokens, text_norm)
            title_overlap = self._query_overlap_score(query_tokens, title_norm)

            base_score = self._safe_float(item.get("score"), default=0.35)
            confidence = self._safe_float(item.get("confidence"), default=0.70)

            score = base_score + overlap * 1.35 + title_overlap * 0.50 + confidence * 0.35

            if table_id_set and table_id_set.intersection(item_table_ids):
                score += self.config.boost_table_filter_match

            if "table" in str(item.get("source_type", "")).lower() or "table" in str(item.get("chunk_type", "")).lower():
                score += self.config.boost_table_result

            scored.append(
                self._normalize_retrieved_item(
                    item={
                        **item,
                        "score": round(score, 6),
                        "text": text,
                        "matched_terms": self._matched_terms(query_tokens, text_norm),
                    },
                    retrieval_source="table",
                    default_source_type="table",
                )
            )

        scored = sorted(
            scored,
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )

        return scored[:top_k]

    def _fallback_bm25_search(
        self,
        index_result: Dict[str, Any],
        query: str,
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        bm25_index = index_result.get("bm25_index", index_result) or {}
        document_store = bm25_index.get("document_store", {}) or index_result.get("document_store", {}) or {}

        query_tokens = self._tokenize(query)
        results = []

        for doc_id, doc in document_store.items():
            doc = self._to_dict(doc)

            if not self._passes_filters(doc, filters):
                continue

            text = normalize_pdf_text(doc.get("text") or doc.get("text_preview") or "")
            text_norm = normalize_text_for_match(text)

            overlap = self._query_overlap_score(query_tokens, text_norm)
            phrase = 1.0 if normalize_text_for_match(query) in text_norm and query else 0.0

            score = overlap + phrase * self.config.boost_exact_phrase

            if score <= 0:
                continue

            results.append(
                {
                    **doc,
                    "document_id": doc.get("document_id", doc_id),
                    "score": round(score, 6),
                    "matched_terms": self._matched_terms(query_tokens, text_norm),
                }
            )

        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]

    def _fallback_vector_search(
        self,
        index_result: Dict[str, Any],
        query: str,
        top_k: int,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        vector_index = index_result.get("vector_index", index_result) or {}
        document_store = vector_index.get("document_store", {}) or index_result.get("document_store", {}) or {}

        query_tokens = self._tokenize(query)
        results = []

        for doc_id, doc in document_store.items():
            doc = self._to_dict(doc)

            if not self._passes_filters(doc, filters):
                continue

            text = normalize_pdf_text(doc.get("text") or doc.get("text_preview") or "")
            text_norm = normalize_text_for_match(text)

            overlap = self._query_overlap_score(query_tokens, text_norm)

            if overlap <= 0:
                continue

            results.append(
                {
                    **doc,
                    "document_id": doc.get("document_id", doc_id),
                    "score": round(overlap, 6),
                    "matched_terms": self._matched_terms(query_tokens, text_norm),
                }
            )

        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]

    def _fallback_graph_search(
        self,
        index_result: Dict[str, Any],
        query: str,
        top_k: int,
        filters: Dict[str, Any],
        page_numbers: List[int],
    ) -> List[Dict[str, Any]]:
        graph_index = index_result.get("graph_index", index_result) or {}
        node_store = graph_index.get("node_store", {}) or index_result.get("node_store", {}) or {}

        query_tokens = self._tokenize(query)
        page_set = set(page_numbers or [])
        results = []

        for node_id, node in node_store.items():
            node = self._to_dict(node)

            if not self._passes_filters(node, filters):
                continue

            node_pages = set(self._resolve_page_numbers(node))
            if page_set and node_pages and not page_set.intersection(node_pages):
                continue
            if page_set and not node_pages:
                continue

            text = normalize_pdf_text(node.get("text") or node.get("label") or "")
            text_norm = normalize_text_for_match(text)

            overlap = self._query_overlap_score(query_tokens, text_norm)

            if overlap <= 0:
                continue

            results.append(
                {
                    **node,
                    "node_id": node.get("node_id", node_id),
                    "score": round(overlap, 6),
                    "matched_terms": self._matched_terms(query_tokens, text_norm),
                }
            )

        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]

    def _fallback_metadata_search(
        self,
        index_result: Dict[str, Any],
        query: str,
        top_k: int,
        filters: Dict[str, Any],
        page_numbers: List[int],
    ) -> List[Dict[str, Any]]:
        metadata_index = index_result.get("metadata_index", index_result) or {}
        metadata_store = metadata_index.get("metadata_store", {}) or index_result.get("metadata_store", {}) or {}

        query_tokens = self._tokenize(query)
        page_set = set(page_numbers or [])
        results = []

        for item_id, item in metadata_store.items():
            item = self._to_dict(item)

            if not self._passes_filters(item, filters):
                continue

            item_pages = set(self._resolve_page_numbers(item))
            if page_set and item_pages and not page_set.intersection(item_pages):
                continue
            if page_set and not item_pages:
                continue

            text = normalize_pdf_text(
                "\n".join(
                    [
                        item.get("title", ""),
                        item.get("text_preview", ""),
                        item.get("section_title", ""),
                        item.get("item_type", ""),
                        item.get("source_type", ""),
                    ]
                )
            )
            text_norm = normalize_text_for_match(text)

            overlap = self._query_overlap_score(query_tokens, text_norm)

            if overlap <= 0 and query_tokens:
                continue

            results.append(
                {
                    **item,
                    "item_id": item.get("item_id", item_id),
                    "score": round(max(overlap, 0.1), 6),
                    "matched_terms": self._matched_terms(query_tokens, text_norm),
                }
            )

        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]

    def _normalize_source_scores(
        self,
        results: List[Dict[str, Any]],
        source_name: str,
    ) -> List[Dict[str, Any]]:
        if not results:
            return []

        scores = [
            self._safe_float(item.get("score"), default=0.0)
            for item in results
        ]

        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0

        normalized = []

        for rank, item in enumerate(results, start=1):
            item = self._to_dict(item)
            raw_score = self._safe_float(item.get("score"), default=0.0)

            if self.config.use_score_normalization and max_score > min_score:
                normalized_score = (raw_score - min_score) / max(max_score - min_score, 1e-9)
            elif max_score > 0:
                normalized_score = raw_score / max_score
            else:
                normalized_score = 0.0

            item["source_rank"] = rank
            item["raw_score"] = raw_score
            item["normalized_score"] = round(normalized_score, 6)
            item["retrieval_source"] = source_name
            normalized.append(item)

        return normalized

    def _fuse_results(
        self,
        query: str,
        source_results: Dict[str, List[Dict[str, Any]]],
        page_numbers: List[int],
        section_ids: List[str],
        table_ids: List[str],
    ) -> List[Dict[str, Any]]:
        weights = {
            "bm25": self.config.bm25_weight,
            "vector": self.config.vector_weight,
            "graph": self.config.graph_weight,
            "metadata": self.config.metadata_weight,
            "table": self.config.table_weight,
        }

        fused_by_key: Dict[str, Dict[str, Any]] = {}

        for source_name, results in source_results.items():
            source_weight = weights.get(source_name, 1.0)

            for item in results:
                key = self._result_key(item)

                if not key:
                    continue

                normalized_score = self._safe_float(item.get("normalized_score"), default=0.0)
                raw_score = self._safe_float(item.get("raw_score", item.get("score")), default=0.0)
                rank = self._safe_int(item.get("source_rank"), default=999999)

                if self.config.use_reciprocal_rank_fusion:
                    source_score = source_weight * (1.0 / (self.config.rrf_k + rank))
                    source_score += source_weight * normalized_score * 0.35
                else:
                    source_score = source_weight * normalized_score

                source_score += self._boost_item(
                    item=item,
                    query=query,
                    page_numbers=page_numbers,
                    section_ids=section_ids,
                    table_ids=table_ids,
                )

                if key not in fused_by_key:
                    fused_by_key[key] = self._make_fused_item(item)
                    fused_by_key[key]["score"] = 0.0
                    fused_by_key[key]["source_scores"] = {}
                    fused_by_key[key]["source_ranks"] = {}
                    fused_by_key[key]["matched_sources"] = []

                fused = fused_by_key[key]
                fused["score"] += source_score
                fused["source_scores"][source_name] = {
                    "source_score": round(source_score, 6),
                    "normalized_score": normalized_score,
                    "raw_score": raw_score,
                    "rank": rank,
                    "weight": source_weight,
                }
                fused["source_ranks"][source_name] = rank

                if source_name not in fused["matched_sources"]:
                    fused["matched_sources"].append(source_name)

                fused["matched_terms"] = self._merge_unique(
                    fused.get("matched_terms", []),
                    item.get("matched_terms", []),
                )

                fused = self._merge_richer_fields(fused, item)
                fused_by_key[key] = fused

        fused_results = list(fused_by_key.values())

        for item in fused_results:
            item["score"] = round(self._safe_float(item.get("score"), default=0.0), 6)
            item["matched_source_count"] = len(item.get("matched_sources", []))

            if item["matched_source_count"] >= 2:
                item["score"] = round(item["score"] * (1.0 + 0.08 * min(item["matched_source_count"] - 1, 4)), 6)

            if not self.config.include_source_scores:
                item.pop("source_scores", None)
                item.pop("source_ranks", None)

        return fused_results

    def _boost_item(
        self,
        item: Dict[str, Any],
        query: str,
        page_numbers: List[int],
        section_ids: List[str],
        table_ids: List[str],
    ) -> float:
        boost = 0.0

        query_norm = normalize_text_for_match(query)
        text_norm = normalize_text_for_match(item.get("text") or item.get("text_preview") or "")
        title_norm = normalize_text_for_match(item.get("title") or item.get("label") or item.get("section_title") or "")

        if query_norm and query_norm in text_norm:
            boost += self.config.boost_exact_phrase

        if query_norm and title_norm and query_norm in title_norm:
            boost += self.config.boost_title_match

        if page_numbers:
            item_pages = set(self._resolve_page_numbers(item))
            if item_pages and item_pages.intersection(set(page_numbers)):
                boost += self.config.boost_page_filter_match

        if section_ids:
            section_id = item.get("section_id", "") or item.get("metadata", {}).get("section_id", "")
            if section_id and section_id in section_ids:
                boost += self.config.boost_section_filter_match

        if table_ids:
            item_table_ids = set(self._table_ids_from_item(item))
            if item_table_ids and item_table_ids.intersection(set(table_ids)):
                boost += self.config.boost_table_filter_match

        joined_type = " ".join(
            [
                str(item.get("source_type", "")),
                str(item.get("chunk_type", "")),
                str(item.get("item_type", "")),
                str(item.get("node_type", "")),
                str(item.get("evidence_type", "")),
            ]
        ).lower()

        if "table" in joined_type:
            boost += self.config.boost_table_result

        if "evidence" in joined_type:
            boost += self.config.boost_evidence_result

        if "chunk" in joined_type:
            boost += self.config.boost_chunk_result

        return boost

    def _make_fused_item(
        self,
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        text = normalize_pdf_text(
            item.get("text")
            or item.get("text_preview")
            or item.get("quote")
            or item.get("content")
            or item.get("label")
            or item.get("title")
            or ""
        )
        text = self._truncate_text(text, self.config.max_text_chars)

        metadata = item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        return {
            "rank": 0,
            "score": self._safe_float(item.get("score"), default=0.0),
            "retrieval_source": item.get("retrieval_source", ""),
            "matched_sources": [item.get("retrieval_source", "")] if item.get("retrieval_source") else [],
            "source_id": item.get("source_id", "")
            or item.get("document_id", "")
            or item.get("vector_id", "")
            or item.get("node_id", "")
            or item.get("item_id", "")
            or item.get("chunk_id", "")
            or item.get("evidence_id", "")
            or item.get("citation_id", "")
            or self._stable_id(text, "result"),
            "source_type": item.get("source_type", "")
            or item.get("chunk_type", "")
            or item.get("evidence_type", "")
            or item.get("node_type", "")
            or item.get("item_type", "")
            or "retrieved_item",
            "title": normalize_pdf_text(item.get("title") or item.get("label") or item.get("section_title") or ""),
            "text": text,
            "text_preview": self._preview(text, self.config.text_preview_chars),
            "normalized_text": normalize_text_for_match(text),
            "matched_terms": item.get("matched_terms", []) or [],
            "page_numbers": self._resolve_page_numbers(item),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "section_id": item.get("section_id", "") or metadata.get("section_id", ""),
            "section_title": item.get("section_title", "") or metadata.get("section_title", ""),
            "chunk_id": item.get("chunk_id", "") or metadata.get("chunk_id", ""),
            "evidence_id": item.get("evidence_id", "") or metadata.get("evidence_id", ""),
            "citation_id": item.get("citation_id", "") or metadata.get("citation_id", ""),
            "table_id": self._table_ids_from_item(item)[0] if self._table_ids_from_item(item) else "",
            "node_id": item.get("node_id", "") or metadata.get("node_id", ""),
            "bbox": item.get("bbox", []) or [],
            "confidence": self._safe_float(item.get("confidence"), default=0.70),
            "metadata": metadata if self.config.include_debug else {},
        }

    def _merge_richer_fields(
        self,
        fused: Dict[str, Any],
        item: Dict[str, Any],
    ) -> Dict[str, Any]:
        text = normalize_pdf_text(
            item.get("text")
            or item.get("text_preview")
            or item.get("quote")
            or item.get("content")
            or item.get("label")
            or item.get("title")
            or ""
        )

        if len(text) > len(fused.get("text", "")):
            fused["text"] = self._truncate_text(text, self.config.max_text_chars)
            fused["text_preview"] = self._preview(fused["text"], self.config.text_preview_chars)
            fused["normalized_text"] = normalize_text_for_match(fused["text"])

        title = normalize_pdf_text(item.get("title") or item.get("label") or item.get("section_title") or "")
        if title and len(title) > len(fused.get("title", "")):
            fused["title"] = title

        pages = self._merge_unique(fused.get("page_numbers", []), self._resolve_page_numbers(item))
        fused["page_numbers"] = self._normalize_page_numbers(pages)

        for key in ["section_id", "section_title", "chunk_id", "evidence_id", "citation_id", "node_id"]:
            if not fused.get(key) and item.get(key):
                fused[key] = item.get(key)

        if not fused.get("table_id"):
            table_ids = self._table_ids_from_item(item)
            if table_ids:
                fused["table_id"] = table_ids[0]

        if not fused.get("bbox") and item.get("bbox"):
            fused["bbox"] = item.get("bbox", [])

        fused["confidence"] = max(
            self._safe_float(fused.get("confidence"), default=0.0),
            self._safe_float(item.get("confidence"), default=0.0),
        )

        return fused

    def _normalize_retrieved_item(
        self,
        item: Dict[str, Any],
        retrieval_source: str,
        default_source_type: str,
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

        metadata = item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        item["retrieval_source"] = retrieval_source
        item["source_type"] = (
            item.get("source_type")
            or item.get("chunk_type")
            or item.get("evidence_type")
            or item.get("node_type")
            or item.get("item_type")
            or default_source_type
        )
        item["source_id"] = (
            item.get("source_id")
            or item.get("document_id")
            or item.get("vector_id")
            or item.get("node_id")
            or item.get("item_id")
            or item.get("chunk_id")
            or item.get("evidence_id")
            or item.get("citation_id")
            or self._stable_id(text, retrieval_source)
        )
        item["text"] = text
        item["text_preview"] = item.get("text_preview") or self._preview(text, self.config.text_preview_chars)
        item["page_numbers"] = self._resolve_page_numbers(item)
        item["metadata"] = metadata

        return item

    def _deduplicate_fused_results(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result_by_key: Dict[str, Dict[str, Any]] = {}

        for item in results:
            key = self._result_key(item)

            if not key:
                continue

            if key not in result_by_key:
                result_by_key[key] = item
            else:
                existing = result_by_key[key]

                if item.get("score", 0.0) > existing.get("score", 0.0):
                    merged = {
                        **existing,
                        **item,
                    }
                else:
                    merged = {
                        **item,
                        **existing,
                    }

                merged["score"] = max(existing.get("score", 0.0), item.get("score", 0.0))
                merged["matched_sources"] = self._merge_unique(existing.get("matched_sources", []), item.get("matched_sources", []))
                merged["matched_terms"] = self._merge_unique(existing.get("matched_terms", []), item.get("matched_terms", []))

                if self.config.include_source_scores:
                    merged["source_scores"] = {
                        **existing.get("source_scores", {}),
                        **item.get("source_scores", {}),
                    }
                    merged["source_ranks"] = {
                        **existing.get("source_ranks", {}),
                        **item.get("source_ranks", {}),
                    }

                result_by_key[key] = merged

        return list(result_by_key.values())

    def _result_key(
        self,
        item: Dict[str, Any],
    ) -> str:
        for key in ["chunk_id", "evidence_id", "citation_id", "node_id", "item_id", "document_id", "vector_id", "source_id"]:
            value = item.get(key)
            if value:
                return f"{key}:{value}"

        table_ids = self._table_ids_from_item(item)
        if table_ids:
            return f"table_id:{table_ids[0]}"

        text_norm = normalize_text_for_match(item.get("text") or item.get("text_preview") or item.get("title") or "")
        pages = self._resolve_page_numbers(item)

        if text_norm:
            return f"text:{text_norm[:600]}|pages:{pages}"

        return ""

    def _merge_filters(
        self,
        filters: Dict[str, Any],
        page_numbers: List[int],
        section_ids: List[str],
        table_ids: List[str],
    ) -> Dict[str, Any]:
        merged = dict(filters or {})

        if page_numbers:
            merged.setdefault("page_numbers", page_numbers)

        if section_ids:
            merged.setdefault("section_id", section_ids)

        if table_ids:
            merged.setdefault("table_id", table_ids)

        return merged

    def _passes_filters(
        self,
        item: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:
        if not filters:
            return True

        metadata = item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        for key, expected in filters.items():
            if expected is None:
                continue

            if key == "page_numbers":
                expected_pages = set(self._normalize_page_numbers(expected))
                item_pages = set(self._resolve_page_numbers(item))

                if expected_pages and not item_pages.intersection(expected_pages):
                    return False

                continue

            if key == "table_id":
                actual_values = self._table_ids_from_item(item)
            else:
                actual = item.get(key)
                if actual is None or actual == "":
                    actual = metadata.get(key)
                actual_values = actual if isinstance(actual, list) else [actual]

            expected_values = expected if isinstance(expected, list) else [expected]
            actual_set = set(str(value) for value in actual_values if value not in [None, ""])
            expected_set = set(str(value) for value in expected_values if value not in [None, ""])

            if expected_set and not actual_set.intersection(expected_set):
                return False

        return True

    def _query_overlap_score(
        self,
        query_tokens: List[str],
        text_norm: str,
    ) -> float:
        if not query_tokens or not text_norm:
            return 0.0

        text_tokens = set(text_norm.split())

        if not text_tokens:
            return 0.0

        matched = sum(1 for token in query_tokens if token in text_tokens)
        return matched / max(len(query_tokens), 1)

    def _matched_terms(
        self,
        query_tokens: List[str],
        text_norm: str,
    ) -> List[str]:
        text_tokens = set(text_norm.split())
        return sorted(list(set(query_tokens).intersection(text_tokens)))

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:
        text = normalize_text_for_match(text)
        tokens = re.findall(r"[a-z0-9_]+", text)

        return [
            token for token in tokens
            if len(token) >= 2
        ]

    def _group_by_page(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in results:
            for page_number in item.get("page_numbers", []) or []:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(item)

        return grouped

    def _group_by_source(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in results:
            for source in item.get("matched_sources", []) or [item.get("retrieval_source", "unknown")]:
                source = source or "unknown"
                grouped.setdefault(source, [])
                grouped[source].append(item)

        return grouped

    def _group_by_section(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in results:
            section_id = item.get("section_id", "") or "no_section"
            grouped.setdefault(section_id, [])
            grouped[section_id].append(item)

        return grouped

    def _group_by_table(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in results:
            table_ids = self._table_ids_from_item(item)

            if not table_ids and item.get("table_id"):
                table_ids = [item.get("table_id")]

            if not table_ids:
                continue

            for table_id in table_ids:
                grouped.setdefault(str(table_id), [])
                grouped[str(table_id)].append(item)

        return grouped

    def _build_context_text(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> str:
        parts = []

        if query:
            parts.append(f"Truy vấn: {normalize_pdf_text(query)}")

        total_chars = 0

        for index, item in enumerate(results, start=1):
            page_numbers = item.get("page_numbers", []) or []
            page_label = ""

            if page_numbers:
                if len(page_numbers) == 1:
                    page_label = f"trang {page_numbers[0]}"
                else:
                    page_label = f"trang {page_numbers[0]}-{page_numbers[-1]}"

            header_parts = [
                f"[Result {index}]",
                f"score={item.get('score', 0)}",
                f"sources={','.join(item.get('matched_sources', []))}",
            ]

            if item.get("source_type"):
                header_parts.append(f"type={item.get('source_type')}")

            if item.get("title"):
                header_parts.append(item.get("title"))

            if page_label:
                header_parts.append(page_label)

            text = normalize_pdf_text(item.get("text", "") or item.get("text_preview", ""))

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

    def _build_summary(
        self,
        query: str,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        metadata_results: List[Dict[str, Any]],
        table_results: List[Dict[str, Any]],
        fused_results: List[Dict[str, Any]],
        filters: Dict[str, Any],
        page_numbers: List[int],
        section_ids: List[str],
        table_ids: List[str],
    ) -> Dict[str, Any]:
        by_source = {}
        by_type = {}
        by_page = {}

        scores = []

        for item in fused_results:
            scores.append(self._safe_float(item.get("score"), default=0.0))

            for source in item.get("matched_sources", []) or [item.get("retrieval_source", "unknown")]:
                source = source or "unknown"
                by_source[source] = by_source.get(source, 0) + 1

            source_type = item.get("source_type", "unknown")
            by_type[source_type] = by_type.get(source_type, 0) + 1

            for page_number in item.get("page_numbers", []) or []:
                page_key = str(page_number)
                by_page[page_key] = by_page.get(page_key, 0) + 1

        return {
            "has_results": len(fused_results) > 0,
            "query": query,
            "bm25_count": len(bm25_results),
            "vector_count": len(vector_results),
            "graph_count": len(graph_results),
            "metadata_count": len(metadata_results),
            "table_count": len(table_results),
            "fused_count": len(fused_results),
            "avg_score": round(sum(scores) / max(len(scores), 1), 6),
            "max_score": round(max(scores), 6) if scores else 0.0,
            "min_score": round(min(scores), 6) if scores else 0.0,
            "by_source": by_source,
            "by_source_type": by_type,
            "by_page": by_page,
            "filters": filters,
            "page_numbers_filter": page_numbers,
            "section_ids_filter": section_ids,
            "table_ids_filter": table_ids,
            "top_k": self.config.top_k,
            "candidate_pool_size": self.config.candidate_pool_size,
            "rrf_enabled": self.config.use_reciprocal_rank_fusion,
        }

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

            metadata = item.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}

            for key_name in keys:
                value = item.get(key_name) or metadata.get(key_name)

                if value:
                    key = f"{key_name}:{value}"
                    break

            if not key:
                text = normalize_text_for_match(
                    item.get("text")
                    or item.get("text_preview")
                    or item.get("title")
                    or item.get("caption")
                    or ""
                )
                key = f"text:{text[:500]}|pages:{self._resolve_page_numbers(item)}"

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

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

    def _merge_unique(
        self,
        a: Any,
        b: Any,
    ) -> List[Any]:
        if not isinstance(a, list):
            a = [a] if a not in [None, ""] else []

        if not isinstance(b, list):
            b = [b] if b not in [None, ""] else []

        result = []

        for value in a + b:
            if value in [None, ""]:
                continue

            if value not in result:
                result.append(value)

        return result

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


def retrieve_hybrid(
    query: str,
    bm25_index_result: Optional[Dict[str, Any]] = None,
    vector_index_result: Optional[Dict[str, Any]] = None,
    graph_index_result: Optional[Dict[str, Any]] = None,
    metadata_index_result: Optional[Dict[str, Any]] = None,
    table_chunk_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    page_numbers: Optional[List[int]] = None,
    section_ids: Optional[List[str]] = None,
    table_ids: Optional[List[str]] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    retriever = HybridRetriever()
    return retriever.process(
        query=query,
        bm25_index_result=bm25_index_result,
        vector_index_result=vector_index_result,
        graph_index_result=graph_index_result,
        metadata_index_result=metadata_index_result,
        table_chunk_result=table_chunk_result,
        table_understanding_result=table_understanding_result,
        filters=filters,
        page_numbers=page_numbers,
        section_ids=section_ids,
        table_ids=table_ids,
        top_k=top_k,
    )


def search_hybrid(
    query: str,
    bm25_index_result: Optional[Dict[str, Any]] = None,
    vector_index_result: Optional[Dict[str, Any]] = None,
    graph_index_result: Optional[Dict[str, Any]] = None,
    metadata_index_result: Optional[Dict[str, Any]] = None,
    table_chunk_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    page_numbers: Optional[List[int]] = None,
    section_ids: Optional[List[str]] = None,
    table_ids: Optional[List[str]] = None,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    retriever = HybridRetriever()
    return retriever.search(
        query=query,
        bm25_index_result=bm25_index_result,
        vector_index_result=vector_index_result,
        graph_index_result=graph_index_result,
        metadata_index_result=metadata_index_result,
        table_chunk_result=table_chunk_result,
        table_understanding_result=table_understanding_result,
        filters=filters,
        page_numbers=page_numbers,
        section_ids=section_ids,
        table_ids=table_ids,
        top_k=top_k,
    )
