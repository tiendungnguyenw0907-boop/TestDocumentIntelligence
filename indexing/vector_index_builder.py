"""
vector_index_builder.py

Production V1 - Colab Ready

Purpose
-------
Build a vector index for semantic retrieval over chunks, table chunks,
evidence, page text, and table text.

Design
------
- Works without external embedding services.
- Uses optional sentence-transformers if available.
- Falls back to deterministic hash-based embeddings if no model is available.

Used by:
- KnowledgePipeline
- HybridRetriever
- RAGPipeline
- Semantic Search / QA

Input
-----
- page_raws
- knowledge_result
- chunk_result
- table_chunk_result
- evidence_result
- metadata_enrichment_result
- table_understanding_result

Output
------
Dictionary with:
- vector_index
- vector_store
- document_store
- embedding_config
- vector_index_summary
"""

from __future__ import annotations

import json
import math
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw
from document_ai.schemas.page_raw_schema import (
    normalize_pdf_text,
    normalize_text_for_match,
    json_safe,
)


@dataclass
class VectorIndexBuilderConfig:
    include_chunks: bool = True
    include_table_chunks: bool = True
    include_evidence: bool = True
    include_page_text: bool = True
    include_tables: bool = True
    include_graph_nodes: bool = True

    attach_to_pages: bool = True
    deduplicate_documents: bool = True
    normalize_vectors: bool = True

    embedding_backend: str = "auto"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    fallback_embedding_dim: int = 384

    batch_size: int = 32
    max_text_chars: int = 4000
    text_preview_chars: int = 800

    min_text_chars: int = 10
    max_documents: int = 0

    include_full_text: bool = True
    include_debug: bool = True


class VectorIndexBuilder:
    def __init__(
        self,
        config: Optional[VectorIndexBuilderConfig] = None,
    ):
        self.config = config or VectorIndexBuilderConfig()
        self.embedding_model = None
        self.embedding_backend_used = "hash"

    def process(
        self,
        page_raws: Optional[List[PageRaw]] = None,
        knowledge_result: Optional[Dict[str, Any]] = None,
        chunk_result: Optional[Dict[str, Any]] = None,
        table_chunk_result: Optional[Dict[str, Any]] = None,
        evidence_result: Optional[Dict[str, Any]] = None,
        metadata_enrichment_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        graph_index_result: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        page_raws = sorted(
            page_raws or [],
            key=lambda item: item.page_number,
        )

        knowledge_result = knowledge_result or {}
        chunk_result = chunk_result or {}
        table_chunk_result = table_chunk_result or {}
        evidence_result = evidence_result or {}
        metadata_enrichment_result = metadata_enrichment_result or {}
        table_understanding_result = table_understanding_result or {}
        graph_index_result = graph_index_result or {}

        documents = self._collect_vector_documents(
            page_raws=page_raws,
            knowledge_result=knowledge_result,
            chunk_result=chunk_result,
            table_chunk_result=table_chunk_result,
            evidence_result=evidence_result,
            metadata_enrichment_result=metadata_enrichment_result,
            table_understanding_result=table_understanding_result,
            graph_index_result=graph_index_result,
        )

        if self.config.deduplicate_documents:
            documents = self._deduplicate_documents(documents)

        if self.config.max_documents and self.config.max_documents > 0:
            documents = documents[: self.config.max_documents]

        self._init_embedding_backend()

        vector_store: Dict[str, List[float]] = {}
        document_store: Dict[str, Dict[str, Any]] = {}

        texts = [
            self._prepare_text_for_embedding(document.get("text", ""))
            for document in documents
        ]

        embeddings = self._embed_texts(texts)

        for order, (document, vector) in enumerate(zip(documents, embeddings)):
            vector_id = document.get("vector_id") or document.get("document_id") or self._stable_id(
                document.get("text", ""),
                "vector",
            )

            vector = self._to_float_list(vector)

            if self.config.normalize_vectors:
                vector = self._normalize_vector(vector)

            if not vector:
                continue

            vector_store[vector_id] = vector

            text = normalize_pdf_text(document.get("text", ""))

            document_store[vector_id] = {
                "vector_id": vector_id,
                "document_id": document.get("document_id", vector_id),
                "source_id": document.get("source_id", ""),
                "source_type": document.get("source_type", ""),
                "source": document.get("source", ""),
                "title": document.get("title", ""),
                "text": text if self.config.include_full_text else "",
                "text_preview": self._preview(text, self.config.text_preview_chars),
                "normalized_text": normalize_text_for_match(text),
                "page_numbers": document.get("page_numbers", []),
                "page_start": document.get("page_start"),
                "page_end": document.get("page_end"),
                "section_id": document.get("section_id", ""),
                "section_title": document.get("section_title", ""),
                "chunk_id": document.get("chunk_id", ""),
                "evidence_id": document.get("evidence_id", ""),
                "citation_id": document.get("citation_id", ""),
                "table_id": document.get("table_id", ""),
                "node_id": document.get("node_id", ""),
                "bbox": document.get("bbox", []) or [],
                "order": order,
                "vector_dim": len(vector),
                "metadata": document.get("metadata", {}) or {},
            }

        embedding_config = {
            "embedding_backend_requested": self.config.embedding_backend,
            "embedding_backend_used": self.embedding_backend_used,
            "embedding_model_name": self.config.embedding_model_name,
            "embedding_dim": self._infer_vector_dim(vector_store),
            "normalize_vectors": self.config.normalize_vectors,
            "fallback_embedding_dim": self.config.fallback_embedding_dim,
        }

        vector_index = {
            "index_type": "vector",
            "schema_version": "vector_index_builder_v1",
            "config": asdict(self.config),
            "embedding_config": embedding_config,
            "vector_count": len(vector_store),
            "document_count": len(document_store),
            "vector_store": vector_store,
            "document_store": document_store,
        }

        result = {
            "processor": "VectorIndexBuilder",
            "schema_version": "vector_index_builder_v1",
            "vector_index": vector_index,
            "vector_store": vector_store,
            "document_store": document_store,
            "embedding_config": embedding_config,
            "vector_index_summary": self._build_summary(
                vector_store=vector_store,
                document_store=document_store,
                embedding_config=embedding_config,
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
    ) -> List[Dict[str, Any]]:
        filters = filters or {}

        vector_index = index_result.get("vector_index", index_result) or {}
        vector_store = vector_index.get("vector_store", {}) or index_result.get("vector_store", {}) or {}
        document_store = vector_index.get("document_store", {}) or index_result.get("document_store", {}) or {}
        embedding_config = vector_index.get("embedding_config", {}) or index_result.get("embedding_config", {}) or {}

        self.embedding_backend_used = embedding_config.get("embedding_backend_used", self.embedding_backend_used)
        self.config.embedding_model_name = embedding_config.get("embedding_model_name", self.config.embedding_model_name)

        if self.embedding_backend_used == "sentence_transformers" and self.embedding_model is None:
            self._init_embedding_backend(force_backend="sentence_transformers")

        query_vector = self._embed_texts([query])[0]
        query_vector = self._to_float_list(query_vector)

        if self.config.normalize_vectors:
            query_vector = self._normalize_vector(query_vector)

        scored = []

        for vector_id, vector in vector_store.items():
            document = document_store.get(vector_id, {})

            if not self._passes_filters(document, filters):
                continue

            score = self._cosine_similarity(query_vector, vector)

            if score <= 0:
                continue

            scored.append(
                {
                    "vector_id": vector_id,
                    "score": score,
                    "document": document,
                }
            )

        scored = sorted(
            scored,
            key=lambda item: item["score"],
            reverse=True,
        )[:top_k]

        results = []

        for rank, item in enumerate(scored, start=1):
            document = item["document"]

            results.append(
                {
                    "rank": rank,
                    "score": round(item["score"], 6),
                    "vector_id": item["vector_id"],
                    "document_id": document.get("document_id", ""),
                    "source_id": document.get("source_id", ""),
                    "source_type": document.get("source_type", ""),
                    "title": document.get("title", ""),
                    "text_preview": document.get("text_preview", ""),
                    "page_numbers": document.get("page_numbers", []),
                    "page_start": document.get("page_start"),
                    "page_end": document.get("page_end"),
                    "section_id": document.get("section_id", ""),
                    "section_title": document.get("section_title", ""),
                    "chunk_id": document.get("chunk_id", ""),
                    "evidence_id": document.get("evidence_id", ""),
                    "citation_id": document.get("citation_id", ""),
                    "table_id": document.get("table_id", ""),
                    "node_id": document.get("node_id", ""),
                    "metadata": document.get("metadata", {}),
                }
            )

        return results

    def _init_embedding_backend(
        self,
        force_backend: Optional[str] = None,
    ) -> None:
        backend = force_backend or self.config.embedding_backend

        if backend in ["auto", "sentence_transformers"]:
            try:
                from sentence_transformers import SentenceTransformer

                self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
                self.embedding_backend_used = "sentence_transformers"
                return
            except Exception:
                if backend == "sentence_transformers":
                    self.embedding_model = None
                    self.embedding_backend_used = "hash"
                    return

        self.embedding_model = None
        self.embedding_backend_used = "hash"

    def _embed_texts(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        if self.embedding_model is not None and self.embedding_backend_used == "sentence_transformers":
            try:
                vectors = self.embedding_model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=False,
                    normalize_embeddings=self.config.normalize_vectors,
                )

                return [
                    self._to_float_list(vector)
                    for vector in vectors
                ]
            except Exception:
                self.embedding_model = None
                self.embedding_backend_used = "hash"

        return [
            self._hash_embedding(text, dim=self.config.fallback_embedding_dim)
            for text in texts
        ]

    def _hash_embedding(
        self,
        text: str,
        dim: int = 384,
    ) -> List[float]:
        text = normalize_text_for_match(text)

        tokens = self._tokenize(text)

        if not tokens:
            return [0.0] * dim

        vector = [0.0] * dim

        for token in tokens:
            token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            index = token_hash % dim
            sign = 1.0 if ((token_hash >> 8) % 2 == 0) else -1.0

            weight = 1.0

            if token.isdigit():
                weight = 0.6
            elif len(token) >= 8:
                weight = 1.2

            vector[index] += sign * weight

            if len(token) >= 5:
                sub_hash = int(hashlib.sha1(token[:5].encode("utf-8")).hexdigest(), 16)
                sub_index = sub_hash % dim
                vector[sub_index] += sign * 0.35

        return self._normalize_vector(vector)

    def _collect_vector_documents(
        self,
        page_raws: List[PageRaw],
        knowledge_result: Dict[str, Any],
        chunk_result: Dict[str, Any],
        table_chunk_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        metadata_enrichment_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        graph_index_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []

        if self.config.include_chunks:
            for chunk in self._collect_chunks(chunk_result, knowledge_result):
                text = normalize_pdf_text(chunk.get("text", ""))

                if not self._valid_text(text):
                    continue

                chunk_id = chunk.get("chunk_id", "") or self._stable_id(text, "chunk")

                documents.append(
                    {
                        "vector_id": f"chunk::{chunk_id}",
                        "document_id": f"chunk::{chunk_id}",
                        "source_id": chunk_id,
                        "source_type": chunk.get("chunk_type", "chunk"),
                        "source": "chunk_result",
                        "title": chunk.get("section_title", "") or chunk.get("chunk_type", ""),
                        "text": text,
                        "page_numbers": self._normalize_page_numbers(chunk.get("page_numbers", [])),
                        "page_start": chunk.get("page_start"),
                        "page_end": chunk.get("page_end"),
                        "section_id": chunk.get("section_id", ""),
                        "section_title": chunk.get("section_title", ""),
                        "chunk_id": chunk_id,
                        "table_id": self._table_id(chunk),
                        "bbox": chunk.get("bbox", []) or [],
                        "metadata": {
                            "chunk_type": chunk.get("chunk_type", ""),
                            "confidence": chunk.get("confidence", 0.0),
                            "content_hash": chunk.get("content_hash", ""),
                            "source": chunk.get("source", ""),
                        },
                    }
                )

        if self.config.include_table_chunks:
            for chunk in self._collect_table_chunks(table_chunk_result, knowledge_result):
                text = normalize_pdf_text(chunk.get("text", ""))

                if not self._valid_text(text):
                    continue

                chunk_id = chunk.get("chunk_id", "") or self._stable_id(text, "table_chunk")

                documents.append(
                    {
                        "vector_id": f"table_chunk::{chunk_id}",
                        "document_id": f"table_chunk::{chunk_id}",
                        "source_id": chunk_id,
                        "source_type": chunk.get("chunk_type", "table_chunk"),
                        "source": "table_chunk_result",
                        "title": chunk.get("section_title", "") or chunk.get("metadata", {}).get("title", "") or "Table chunk",
                        "text": text,
                        "page_numbers": self._normalize_page_numbers(chunk.get("page_numbers", [])),
                        "page_start": chunk.get("page_start"),
                        "page_end": chunk.get("page_end"),
                        "section_id": chunk.get("section_id", ""),
                        "section_title": chunk.get("section_title", ""),
                        "chunk_id": chunk_id,
                        "table_id": self._table_id(chunk),
                        "bbox": chunk.get("bbox", []) or [],
                        "metadata": {
                            "chunk_type": chunk.get("chunk_type", ""),
                            "table_grid_id": chunk.get("table_grid_id", ""),
                            "table_structure_id": chunk.get("table_structure_id", ""),
                            "table_semantic_id": chunk.get("table_semantic_id", ""),
                            "table_boundary_id": chunk.get("table_boundary_id", ""),
                            "confidence": chunk.get("confidence", 0.0),
                            "source": chunk.get("source", ""),
                        },
                    }
                )

        if self.config.include_evidence:
            for evidence in self._collect_evidence(evidence_result, knowledge_result):
                text = normalize_pdf_text(evidence.get("text") or evidence.get("quote") or "")

                if not self._valid_text(text):
                    continue

                evidence_id = evidence.get("evidence_id", "") or self._stable_id(text, "evidence")

                documents.append(
                    {
                        "vector_id": f"evidence::{evidence_id}",
                        "document_id": f"evidence::{evidence_id}",
                        "source_id": evidence_id,
                        "source_type": evidence.get("evidence_type", "evidence"),
                        "source": "evidence_result",
                        "title": evidence.get("section_title", "") or evidence.get("evidence_type", ""),
                        "text": text,
                        "page_numbers": self._resolve_page_numbers(evidence),
                        "page_start": evidence.get("page_start"),
                        "page_end": evidence.get("page_end"),
                        "section_id": evidence.get("section_id", ""),
                        "section_title": evidence.get("section_title", ""),
                        "chunk_id": evidence.get("chunk_id") or evidence.get("source_chunk_id", ""),
                        "evidence_id": evidence_id,
                        "table_id": self._table_id(evidence),
                        "bbox": evidence.get("bbox", []) or [],
                        "metadata": {
                            "evidence_type": evidence.get("evidence_type", ""),
                            "relevance_score": evidence.get("relevance_score", 0.0),
                            "confidence": evidence.get("confidence", 0.0),
                            "weight": evidence.get("weight", 1.0),
                            "content_hash": evidence.get("content_hash", ""),
                            "source": evidence.get("source", ""),
                        },
                    }
                )

        if self.config.include_tables:
            for table in self._collect_tables(
                table_understanding_result=table_understanding_result,
                metadata_enrichment_result=metadata_enrichment_result,
                knowledge_result=knowledge_result,
            ):
                text = self._table_text(table)

                if not self._valid_text(text):
                    continue

                table_id = self._table_id(table) or self._stable_id(text, "table")

                documents.append(
                    {
                        "vector_id": f"table::{table_id}",
                        "document_id": f"table::{table_id}",
                        "source_id": table_id,
                        "source_type": self._table_kind(table),
                        "source": "table_understanding_or_metadata",
                        "title": table.get("title", "") or table.get("caption", "") or table_id,
                        "text": text,
                        "page_numbers": self._resolve_page_numbers(table),
                        "page_start": table.get("page_start"),
                        "page_end": table.get("page_end"),
                        "section_id": table.get("section_id", ""),
                        "section_title": table.get("section_title", ""),
                        "table_id": table_id,
                        "bbox": table.get("bbox", []) or [],
                        "metadata": {
                            "semantic_type": table.get("semantic_type", ""),
                            "table_type": table.get("table_type", ""),
                            "row_count": table.get("row_count", table.get("total_row_count", 0)),
                            "col_count": table.get("col_count", 0),
                            "column_headers": table.get("column_headers", []),
                        },
                    }
                )

        if self.config.include_graph_nodes:
            for node in self._collect_graph_nodes(graph_index_result, knowledge_result):
                text = normalize_pdf_text(node.get("text") or node.get("label") or "")

                if not self._valid_text(text):
                    continue

                node_id = node.get("node_id", "") or self._stable_id(text, "node")

                documents.append(
                    {
                        "vector_id": f"graph_node::{node_id}",
                        "document_id": f"graph_node::{node_id}",
                        "source_id": node_id,
                        "source_type": node.get("node_type", "graph_node"),
                        "source": "graph_index_or_knowledge",
                        "title": node.get("label", "") or node_id,
                        "text": text,
                        "page_numbers": self._resolve_page_numbers(node),
                        "page_start": node.get("page_start"),
                        "page_end": node.get("page_end"),
                        "section_id": node.get("section_id", "") or node.get("metadata", {}).get("section_id", ""),
                        "chunk_id": node.get("chunk_id", "") or node.get("metadata", {}).get("chunk_id", ""),
                        "evidence_id": node.get("evidence_id", "") or node.get("metadata", {}).get("evidence_id", ""),
                        "table_id": node.get("table_id", "") or node.get("metadata", {}).get("table_id", ""),
                        "node_id": node_id,
                        "bbox": node.get("bbox", []) or [],
                        "metadata": {
                            "node_type": node.get("node_type", ""),
                            "source_id": node.get("source_id", ""),
                            "source_type": node.get("source_type", ""),
                            "confidence": node.get("confidence", 0.0),
                            "weight": node.get("weight", 1.0),
                        },
                    }
                )

        if self.config.include_page_text:
            for page_raw in page_raws:
                text = self._page_text(page_raw)

                if not self._valid_text(text):
                    continue

                page_number = page_raw.page_number

                documents.append(
                    {
                        "vector_id": f"page::{page_number}",
                        "document_id": f"page::{page_number}",
                        "source_id": f"page_{page_number}",
                        "source_type": "page",
                        "source": "page_raw",
                        "title": f"Trang {page_number}",
                        "text": text,
                        "page_numbers": [page_number],
                        "page_start": page_number,
                        "page_end": page_number,
                        "bbox": [],
                        "metadata": {
                            "page_index": page_raw.page_index,
                            "page_kind": page_raw.page_kind,
                            "page_summary": page_raw.summary() if hasattr(page_raw, "summary") else {},
                        },
                    }
                )

        return documents

    def _collect_chunks(
        self,
        chunk_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        chunks = []

        for source in [chunk_result, knowledge_result]:
            for key in ["chunks", "parent_chunks", "child_chunks"]:
                values = source.get(key, []) or []

                if isinstance(values, list):
                    chunks.extend([self._to_dict(item) for item in values])

            for sub_key in ["chunk_result", "chunk_collection", "parent_child_chunk_result"]:
                sub = source.get(sub_key, {}) or {}

                if not isinstance(sub, dict):
                    continue

                for key in ["chunks", "parent_chunks", "child_chunks"]:
                    values = sub.get(key, []) or []

                    if isinstance(values, list):
                        chunks.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(chunks, ["chunk_id", "content_hash"])

    def _collect_table_chunks(
        self,
        table_chunk_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        chunks = []

        for source in [table_chunk_result, knowledge_result]:
            for key in [
                "table_chunks",
                "table_summary_chunks",
                "table_record_chunks",
                "table_row_chunks",
                "multi_page_table_chunks",
                "table_cell_context_chunks",
            ]:
                values = source.get(key, []) or []

                if isinstance(values, list):
                    chunks.extend([self._to_dict(item) for item in values])

            sub = source.get("table_chunk_result", {}) or {}

            if isinstance(sub, dict):
                for key in [
                    "table_chunks",
                    "table_summary_chunks",
                    "table_record_chunks",
                    "table_row_chunks",
                    "multi_page_table_chunks",
                    "table_cell_context_chunks",
                ]:
                    values = sub.get(key, []) or []

                    if isinstance(values, list):
                        chunks.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(chunks, ["chunk_id", "content_hash"])

    def _collect_evidence(
        self,
        evidence_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        evidence = []

        for source in [evidence_result, knowledge_result]:
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

    def _collect_tables(
        self,
        table_understanding_result: Dict[str, Any],
        metadata_enrichment_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tables = []

        table_metadata = metadata_enrichment_result.get("table_metadata", {}) or {}

        if isinstance(table_metadata, dict):
            for table_id, item in table_metadata.items():
                item = self._to_dict(item)
                item.setdefault("table_id", table_id)
                tables.append(item)

        for source in [table_understanding_result, knowledge_result]:
            for key in [
                "table_semantics",
                "table_grids",
                "table_structures",
                "table_boundaries",
                "multi_page_tables",
            ]:
                values = source.get(key, []) or []

                if isinstance(values, list):
                    tables.extend([self._to_dict(item) for item in values])

            for sub_key in [
                "table_understanding_result",
                "table_semantic_result",
                "table_grid_result",
                "table_structure_result",
                "table_boundary_result",
                "multi_page_table_result",
            ]:
                sub = source.get(sub_key, {}) or {}

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
            ["table_id", "table_semantic_id", "table_grid_id", "table_structure_id", "table_boundary_id", "multi_page_table_id"],
        )

    def _collect_graph_nodes(
        self,
        graph_index_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        nodes = []

        for source in [graph_index_result, knowledge_result]:
            graph_index = source.get("graph_index", {}) or source.get("knowledge_graph", {}) or source.get("context_graph", {}) or {}

            node_store = graph_index.get("node_store", {}) or source.get("node_store", {}) or {}

            if isinstance(node_store, dict):
                for node_id, node in node_store.items():
                    node = self._to_dict(node)
                    node.setdefault("node_id", node_id)
                    nodes.append(node)

            for key in ["nodes"]:
                values = graph_index.get(key, []) or source.get(key, []) or []

                if isinstance(values, list):
                    nodes.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(nodes, ["node_id", "source_id"])

    def _prepare_text_for_embedding(
        self,
        text: str,
    ) -> str:
        text = normalize_pdf_text(text)

        if len(text) <= self.config.max_text_chars:
            return text

        cut = text[: self.config.max_text_chars]
        break_point = max(
            cut.rfind(". "),
            cut.rfind("; "),
            cut.rfind("\n"),
            cut.rfind(" "),
        )

        if break_point > self.config.max_text_chars * 0.60:
            cut = cut[:break_point]

        return normalize_pdf_text(cut)

    def _valid_text(
        self,
        text: str,
    ) -> bool:
        text = normalize_pdf_text(text)

        if not text:
            return False

        if len(text) < self.config.min_text_chars:
            return False

        return True

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

        if table.get("text_preview"):
            parts.append(normalize_pdf_text(table.get("text_preview", "")))

        records = table.get("records", []) or []

        for record in records[:10]:
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

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:
        text = normalize_text_for_match(text)
        raw_tokens = re.findall(r"[a-z0-9_]+", text)

        tokens = []

        for token in raw_tokens:
            token = token.strip("_")

            if len(token) < 2:
                continue

            if len(token) > 64:
                continue

            tokens.append(token)

        return tokens

    def _normalize_vector(
        self,
        vector: List[float],
    ) -> List[float]:
        if not vector:
            return []

        norm = math.sqrt(sum(value * value for value in vector))

        if norm <= 0:
            return vector

        return [
            float(value / norm)
            for value in vector
        ]

    def _cosine_similarity(
        self,
        vector_a: List[float],
        vector_b: List[float],
    ) -> float:
        if not vector_a or not vector_b:
            return 0.0

        length = min(len(vector_a), len(vector_b))

        if length <= 0:
            return 0.0

        dot = sum(vector_a[index] * vector_b[index] for index in range(length))
        norm_a = math.sqrt(sum(vector_a[index] * vector_a[index] for index in range(length)))
        norm_b = math.sqrt(sum(vector_b[index] * vector_b[index] for index in range(length)))

        if norm_a <= 0 or norm_b <= 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def _to_float_list(
        self,
        vector: Any,
    ) -> List[float]:
        try:
            if hasattr(vector, "tolist"):
                vector = vector.tolist()

            return [
                float(item)
                for item in vector
            ]
        except Exception:
            return []

    def _passes_filters(
        self,
        document: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:
        if not filters:
            return True

        for key, expected in filters.items():
            if expected is None:
                continue

            if key == "page_numbers":
                actual_pages = set(self._normalize_page_numbers(document.get("page_numbers", [])))
                expected_pages = set(self._normalize_page_numbers(expected))

                if expected_pages and not actual_pages.intersection(expected_pages):
                    return False

                continue

            actual = document.get(key)

            if actual is None:
                actual = document.get("metadata", {}).get(key)

            if isinstance(expected, list):
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False

        return True

    def _deduplicate_documents(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for document in documents:
            key = (
                document.get("vector_id", ""),
                document.get("source_id", ""),
                document.get("source_type", ""),
                normalize_text_for_match(document.get("text", ""))[:1000],
                tuple(document.get("page_numbers", []) or []),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(document)

        return result

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
                        or item.get("title")
                        or item.get("label")
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

    def _infer_vector_dim(
        self,
        vector_store: Dict[str, List[float]],
    ) -> int:
        for vector in vector_store.values():
            return len(vector)

        return 0

    def _build_summary(
        self,
        vector_store: Dict[str, List[float]],
        document_store: Dict[str, Dict[str, Any]],
        embedding_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        by_source_type: Dict[str, int] = {}
        by_page: Dict[str, int] = {}

        for document in document_store.values():
            source_type = document.get("source_type", "unknown")
            by_source_type[source_type] = by_source_type.get(source_type, 0) + 1

            for page_number in document.get("page_numbers", []) or []:
                page_key = str(page_number)
                by_page[page_key] = by_page.get(page_key, 0) + 1

        vector_dims = [
            len(vector)
            for vector in vector_store.values()
            if vector
        ]

        return {
            "has_vector_index": len(vector_store) > 0,
            "index_type": "vector",
            "vector_count": len(vector_store),
            "document_count": len(document_store),
            "embedding_backend_used": embedding_config.get("embedding_backend_used", ""),
            "embedding_model_name": embedding_config.get("embedding_model_name", ""),
            "embedding_dim": embedding_config.get("embedding_dim", 0),
            "min_vector_dim": min(vector_dims) if vector_dims else 0,
            "max_vector_dim": max(vector_dims) if vector_dims else 0,
            "by_source_type": by_source_type,
            "by_page": by_page,
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        document_store = result.get("document_store", {}) or {}
        summary = result.get("vector_index_summary", {}) or {}

        docs_by_page: Dict[str, List[str]] = {}

        for vector_id, document in document_store.items():
            for page_number in document.get("page_numbers", []) or []:
                page_key = str(page_number)
                docs_by_page.setdefault(page_key, [])
                docs_by_page[page_key].append(vector_id)

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("vector_index_builder", {})
            page_raw.metadata["vector_index_builder"] = {
                "processor": "VectorIndexBuilder",
                "vector_ids_on_page": docs_by_page.get(page_key, []),
                "vector_count_on_page": len(docs_by_page.get(page_key, [])),
                "index_summary": {
                    "vector_count": summary.get("vector_count", 0),
                    "embedding_backend_used": summary.get("embedding_backend_used", ""),
                    "embedding_dim": summary.get("embedding_dim", 0),
                },
            }

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


def build_vector_index(
    page_raws: Optional[List[PageRaw]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
    chunk_result: Optional[Dict[str, Any]] = None,
    table_chunk_result: Optional[Dict[str, Any]] = None,
    evidence_result: Optional[Dict[str, Any]] = None,
    metadata_enrichment_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    graph_index_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = VectorIndexBuilder()
    return builder.process(
        page_raws=page_raws,
        knowledge_result=knowledge_result,
        chunk_result=chunk_result,
        table_chunk_result=table_chunk_result,
        evidence_result=evidence_result,
        metadata_enrichment_result=metadata_enrichment_result,
        table_understanding_result=table_understanding_result,
        graph_index_result=graph_index_result,
    )


def search_vector_index(
    index_result: Dict[str, Any],
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    builder = VectorIndexBuilder()
    return builder.search(
        index_result=index_result,
        query=query,
        top_k=top_k,
        filters=filters,
    )
