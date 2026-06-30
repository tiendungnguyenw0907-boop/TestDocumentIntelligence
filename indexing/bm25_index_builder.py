"""
bm25_index_builder.py

Production V1 - Colab Ready

Purpose
-------
Build a pure-Python BM25 keyword index for document chunks, table chunks,
evidence items, and page text.

Used by:
- KnowledgePipeline
- HybridRetriever
- RAGPipeline
- Search / QA

Input
-----
- page_raws
- knowledge_result
- chunk_result
- table_chunk_result
- evidence_result
- metadata_enrichment_result

Output
------
Dictionary with:
- bm25_index
- document_store
- inverted_index
- term_statistics
- bm25_summary
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
class BM25IndexBuilderConfig:
    include_chunks: bool = True
    include_table_chunks: bool = True
    include_evidence: bool = True
    include_page_text: bool = True
    include_tables: bool = True

    attach_to_pages: bool = True
    deduplicate_documents: bool = True

    k1: float = 1.5
    b: float = 0.75

    min_token_length: int = 2
    max_token_length: int = 64
    lowercase: bool = True
    remove_stopwords: bool = True
    keep_numbers: bool = True

    store_full_text: bool = True
    text_preview_chars: int = 800
    max_tokens_per_document: int = 20000

    include_positions: bool = True
    include_term_frequencies_in_store: bool = True

    include_debug: bool = True


class BM25IndexBuilder:
    def __init__(
        self,
        config: Optional[BM25IndexBuilderConfig] = None,
    ):
        self.config = config or BM25IndexBuilderConfig()
        self.stopwords = self._default_stopwords()

    def process(
        self,
        page_raws: Optional[List[PageRaw]] = None,
        knowledge_result: Optional[Dict[str, Any]] = None,
        chunk_result: Optional[Dict[str, Any]] = None,
        table_chunk_result: Optional[Dict[str, Any]] = None,
        evidence_result: Optional[Dict[str, Any]] = None,
        metadata_enrichment_result: Optional[Dict[str, Any]] = None,
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

        documents = self._collect_index_documents(
            page_raws=page_raws,
            knowledge_result=knowledge_result,
            chunk_result=chunk_result,
            table_chunk_result=table_chunk_result,
            evidence_result=evidence_result,
            metadata_enrichment_result=metadata_enrichment_result,
        )

        if self.config.deduplicate_documents:
            documents = self._deduplicate_documents(documents)

        document_store: Dict[str, Dict[str, Any]] = {}
        inverted_index: Dict[str, List[Dict[str, Any]]] = {}
        term_statistics: Dict[str, Dict[str, Any]] = {}

        total_doc_length = 0

        for order, document in enumerate(documents):
            document_id = document["document_id"]
            text = normalize_pdf_text(document.get("text", ""))

            tokens, positions = self._tokenize_with_positions(text)
            tokens = tokens[: self.config.max_tokens_per_document]

            term_freq = self._term_frequency(tokens)
            doc_length = len(tokens)
            total_doc_length += doc_length

            document_store[document_id] = {
                "document_id": document_id,
                "source_id": document.get("source_id", ""),
                "source_type": document.get("source_type", ""),
                "source": document.get("source", ""),
                "title": document.get("title", ""),
                "text": text if self.config.store_full_text else "",
                "text_preview": self._preview(text, self.config.text_preview_chars),
                "normalized_text": normalize_text_for_match(text),
                "page_numbers": document.get("page_numbers", []),
                "page_start": document.get("page_start"),
                "page_end": document.get("page_end"),
                "section_id": document.get("section_id", ""),
                "section_title": document.get("section_title", ""),
                "chunk_id": document.get("chunk_id", ""),
                "evidence_id": document.get("evidence_id", ""),
                "table_id": document.get("table_id", ""),
                "bbox": document.get("bbox", []) or [],
                "order": order,
                "doc_length": doc_length,
                "unique_term_count": len(term_freq),
                "term_frequencies": term_freq if self.config.include_term_frequencies_in_store else {},
                "metadata": document.get("metadata", {}) or {},
            }

            for term, tf in term_freq.items():
                posting = {
                    "document_id": document_id,
                    "tf": tf,
                    "doc_length": doc_length,
                }

                if self.config.include_positions:
                    posting["positions"] = positions.get(term, [])

                inverted_index.setdefault(term, [])
                inverted_index[term].append(posting)

        document_count = len(document_store)
        avg_doc_length = round(total_doc_length / max(document_count, 1), 4)

        for term, postings in inverted_index.items():
            df = len(postings)
            idf = self._idf(
                document_count=document_count,
                document_frequency=df,
            )

            term_statistics[term] = {
                "term": term,
                "document_frequency": df,
                "collection_frequency": sum(item["tf"] for item in postings),
                "idf": idf,
            }

        bm25_index = {
            "index_type": "bm25",
            "schema_version": "bm25_index_builder_v1",
            "config": asdict(self.config),
            "k1": self.config.k1,
            "b": self.config.b,
            "document_count": document_count,
            "avg_doc_length": avg_doc_length,
            "total_doc_length": total_doc_length,
            "vocabulary_size": len(inverted_index),
            "document_store": document_store,
            "inverted_index": inverted_index,
            "term_statistics": term_statistics,
        }

        result = {
            "processor": "BM25IndexBuilder",
            "schema_version": "bm25_index_builder_v1",
            "bm25_index": bm25_index,
            "document_store": document_store,
            "inverted_index": inverted_index,
            "term_statistics": term_statistics,
            "bm25_summary": self._build_summary(
                document_store=document_store,
                inverted_index=inverted_index,
                term_statistics=term_statistics,
                avg_doc_length=avg_doc_length,
                total_doc_length=total_doc_length,
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

        bm25_index = index_result.get("bm25_index", index_result) or {}

        document_store = bm25_index.get("document_store", {}) or index_result.get("document_store", {}) or {}
        inverted_index = bm25_index.get("inverted_index", {}) or index_result.get("inverted_index", {}) or {}
        term_statistics = bm25_index.get("term_statistics", {}) or index_result.get("term_statistics", {}) or {}

        document_count = bm25_index.get("document_count", len(document_store))
        avg_doc_length = bm25_index.get("avg_doc_length", 0.0) or 0.0
        k1 = bm25_index.get("k1", self.config.k1)
        b = bm25_index.get("b", self.config.b)

        query_tokens = self._tokenize(query)
        query_terms = list(dict.fromkeys(query_tokens))

        scores: Dict[str, float] = {}
        matched_terms: Dict[str, List[str]] = {}

        for term in query_terms:
            postings = inverted_index.get(term, []) or []

            if not postings:
                continue

            idf = term_statistics.get(term, {}).get(
                "idf",
                self._idf(document_count=document_count, document_frequency=len(postings)),
            )

            for posting in postings:
                document_id = posting.get("document_id", "")

                if not document_id:
                    continue

                document = document_store.get(document_id, {})

                if not self._passes_filters(document, filters):
                    continue

                tf = posting.get("tf", 0)
                doc_length = posting.get("doc_length", document.get("doc_length", 0))

                score = self._bm25_score(
                    tf=tf,
                    doc_length=doc_length,
                    avg_doc_length=avg_doc_length,
                    idf=idf,
                    k1=k1,
                    b=b,
                )

                scores[document_id] = scores.get(document_id, 0.0) + score
                matched_terms.setdefault(document_id, [])

                if term not in matched_terms[document_id]:
                    matched_terms[document_id].append(term)

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        results = []

        for rank, (document_id, score) in enumerate(ranked, start=1):
            document = document_store.get(document_id, {})
            text = document.get("text") or document.get("text_preview", "")

            results.append(
                {
                    "rank": rank,
                    "score": round(score, 6),
                    "document_id": document_id,
                    "source_id": document.get("source_id", ""),
                    "source_type": document.get("source_type", ""),
                    "title": document.get("title", ""),
                    "text_preview": self._make_snippet(
                        text=text,
                        query_terms=matched_terms.get(document_id, []),
                    ),
                    "matched_terms": matched_terms.get(document_id, []),
                    "page_numbers": document.get("page_numbers", []),
                    "page_start": document.get("page_start"),
                    "page_end": document.get("page_end"),
                    "section_id": document.get("section_id", ""),
                    "section_title": document.get("section_title", ""),
                    "chunk_id": document.get("chunk_id", ""),
                    "evidence_id": document.get("evidence_id", ""),
                    "table_id": document.get("table_id", ""),
                    "metadata": document.get("metadata", {}),
                }
            )

        return results

    def _collect_index_documents(
        self,
        page_raws: List[PageRaw],
        knowledge_result: Dict[str, Any],
        chunk_result: Dict[str, Any],
        table_chunk_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        metadata_enrichment_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []

        if self.config.include_chunks:
            for chunk in self._collect_chunks(chunk_result, knowledge_result):
                text = normalize_pdf_text(chunk.get("text", ""))

                if not text:
                    continue

                chunk_id = chunk.get("chunk_id", "") or self._stable_id(text, "chunk")

                documents.append(
                    {
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
                            "source": chunk.get("source", ""),
                            "content_hash": chunk.get("content_hash", ""),
                        },
                    }
                )

        if self.config.include_table_chunks:
            for chunk in self._collect_table_chunks(table_chunk_result, knowledge_result):
                text = normalize_pdf_text(chunk.get("text", ""))

                if not text:
                    continue

                chunk_id = chunk.get("chunk_id", "") or self._stable_id(text, "table_chunk")

                documents.append(
                    {
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

                if not text:
                    continue

                evidence_id = evidence.get("evidence_id", "") or self._stable_id(text, "evidence")

                documents.append(
                    {
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
                            "source": evidence.get("source", ""),
                            "content_hash": evidence.get("content_hash", ""),
                        },
                    }
                )

        if self.config.include_tables:
            for table in self._collect_tables(knowledge_result, metadata_enrichment_result):
                text = self._table_text(table)

                if not text:
                    continue

                table_id = self._table_id(table) or self._stable_id(text, "table")

                documents.append(
                    {
                        "document_id": f"table::{table_id}",
                        "source_id": table_id,
                        "source_type": "table",
                        "source": "table_metadata_or_knowledge",
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

        if self.config.include_page_text:
            for page_raw in page_raws:
                text = self._page_text(page_raw)

                if not text:
                    continue

                page_number = page_raw.page_number

                documents.append(
                    {
                        "document_id": f"page::{page_number}",
                        "source_id": f"page_{page_number}",
                        "source_type": "page",
                        "source": "page_raw",
                        "title": f"Trang {page_number}",
                        "text": text,
                        "page_numbers": [page_number],
                        "page_start": page_number,
                        "page_end": page_number,
                        "section_id": "",
                        "section_title": "",
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

            sub = source.get("chunk_result", {}) or source.get("chunk_collection", {}) or {}

            if isinstance(sub, dict):
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

            sub = source.get("evidence_result", {}) or source.get("evidence_collection", {}) or {}

            if isinstance(sub, dict):
                values = sub.get("evidence", []) or sub.get("evidence_items", []) or []

                if isinstance(values, list):
                    evidence.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(evidence, ["evidence_id", "content_hash"])

    def _collect_tables(
        self,
        knowledge_result: Dict[str, Any],
        metadata_enrichment_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        tables = []

        table_metadata = metadata_enrichment_result.get("table_metadata", {}) or {}

        if isinstance(table_metadata, dict):
            for table_id, item in table_metadata.items():
                item = self._to_dict(item)
                item.setdefault("table_id", table_id)
                tables.append(item)

        for source in [knowledge_result]:
            for key in ["table_semantics", "multi_page_tables"]:
                values = source.get(key, []) or []

                if isinstance(values, list):
                    tables.extend([self._to_dict(item) for item in values])

            table_understanding = source.get("table_understanding_result", {}) or {}

            if isinstance(table_understanding, dict):
                for key in ["table_semantics", "multi_page_tables"]:
                    values = table_understanding.get(key, []) or []

                    if isinstance(values, list):
                        tables.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(
            tables,
            ["table_id", "table_semantic_id", "table_grid_id", "multi_page_table_id"],
        )

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

        text_preview = normalize_pdf_text(table.get("text_preview", ""))

        if text_preview:
            parts.append(text_preview)

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
        tokens, _ = self._tokenize_with_positions(text)
        return tokens

    def _tokenize_with_positions(
        self,
        text: str,
    ) -> Tuple[List[str], Dict[str, List[int]]]:
        text = normalize_text_for_match(text)

        if self.config.lowercase:
            text = text.lower()

        raw_tokens = re.findall(r"[a-z0-9_]+", text)

        tokens = []
        positions: Dict[str, List[int]] = {}

        for raw_index, token in enumerate(raw_tokens):
            token = token.strip("_")

            if not token:
                continue

            if len(token) < self.config.min_token_length:
                continue

            if len(token) > self.config.max_token_length:
                continue

            if not self.config.keep_numbers and token.isdigit():
                continue

            if self.config.remove_stopwords and token in self.stopwords:
                continue

            position = len(tokens)
            tokens.append(token)
            positions.setdefault(token, [])
            positions[token].append(position)

        return tokens, positions

    def _term_frequency(
        self,
        tokens: List[str],
    ) -> Dict[str, int]:
        freq: Dict[str, int] = {}

        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

        return freq

    def _idf(
        self,
        document_count: int,
        document_frequency: int,
    ) -> float:
        if document_count <= 0 or document_frequency <= 0:
            return 0.0

        value = math.log(
            1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )

        return round(value, 8)

    def _bm25_score(
        self,
        tf: int,
        doc_length: int,
        avg_doc_length: float,
        idf: float,
        k1: float,
        b: float,
    ) -> float:
        if tf <= 0 or doc_length <= 0 or avg_doc_length <= 0:
            return 0.0

        denominator = tf + k1 * (1.0 - b + b * (doc_length / avg_doc_length))

        if denominator <= 0:
            return 0.0

        return idf * ((tf * (k1 + 1.0)) / denominator)

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

            actual = document.get(key)

            if key == "page_numbers":
                actual_pages = set(self._normalize_page_numbers(document.get("page_numbers", [])))
                expected_pages = set(self._normalize_page_numbers(expected))

                if expected_pages and not actual_pages.intersection(expected_pages):
                    return False

                continue

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
                document.get("document_id", ""),
                document.get("source_id", ""),
                normalize_text_for_match(document.get("text", ""))[:800],
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
            key = ""

            for key_name in keys:
                value = item.get(key_name)
                if value:
                    key = str(value)
                    break

            if not key:
                key = self._stable_id(
                    normalize_pdf_text(item.get("text") or item.get("title") or item.get("label") or ""),
                    "item",
                )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _group_top_terms(
        self,
        term_statistics: Dict[str, Dict[str, Any]],
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        ranked = sorted(
            term_statistics.values(),
            key=lambda item: (
                item.get("document_frequency", 0),
                item.get("collection_frequency", 0),
            ),
            reverse=True,
        )[:top_k]

        return ranked

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

    def _build_summary(
        self,
        document_store: Dict[str, Dict[str, Any]],
        inverted_index: Dict[str, List[Dict[str, Any]]],
        term_statistics: Dict[str, Dict[str, Any]],
        avg_doc_length: float,
        total_doc_length: int,
    ) -> Dict[str, Any]:
        by_source_type: Dict[str, int] = {}
        by_page: Dict[str, int] = {}

        for document in document_store.values():
            source_type = document.get("source_type", "unknown")
            by_source_type[source_type] = by_source_type.get(source_type, 0) + 1

            for page_number in document.get("page_numbers", []) or []:
                page_key = str(page_number)
                by_page[page_key] = by_page.get(page_key, 0) + 1

        return {
            "has_index": len(document_store) > 0,
            "index_type": "bm25",
            "document_count": len(document_store),
            "vocabulary_size": len(inverted_index),
            "total_doc_length": total_doc_length,
            "avg_doc_length": avg_doc_length,
            "by_source_type": by_source_type,
            "by_page": by_page,
            "top_terms": self._group_top_terms(term_statistics, top_k=50),
            "k1": self.config.k1,
            "b": self.config.b,
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        summary = result.get("bm25_summary", {}) or {}
        document_store = result.get("document_store", {}) or {}

        docs_by_page: Dict[str, List[str]] = {}

        for document_id, document in document_store.items():
            for page_number in document.get("page_numbers", []) or []:
                page_key = str(page_number)
                docs_by_page.setdefault(page_key, [])
                docs_by_page[page_key].append(document_id)

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("bm25_index_builder", {})
            page_raw.metadata["bm25_index_builder"] = {
                "processor": "BM25IndexBuilder",
                "document_ids_on_page": docs_by_page.get(page_key, []),
                "document_count_on_page": len(docs_by_page.get(page_key, [])),
                "index_summary": {
                    "document_count": summary.get("document_count", 0),
                    "vocabulary_size": summary.get("vocabulary_size", 0),
                    "avg_doc_length": summary.get("avg_doc_length", 0),
                },
            }

    def _table_id(
        self,
        item: Dict[str, Any],
    ) -> str:
        metadata = item.get("metadata", {}) or {}

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

    def _stable_id(
        self,
        text: Any,
        prefix: str = "doc",
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

    def _default_stopwords(self) -> Set[str]:
        return {
            "va", "hoac", "cua", "cho", "cac", "mot", "nhung", "duoc", "trong",
            "ngoai", "theo", "voi", "den", "tu", "tai", "ve", "la", "co", "khong",
            "nay", "do", "khi", "sau", "truoc", "tren", "duoi", "vao", "ra", "de",
            "nham", "phuc", "vu", "can", "phai", "bao", "dam", "quy", "dinh",
            "noi", "dung", "thuc", "hien", "quan", "ly", "nha", "nuoc", "du",
            "lieu", "he", "thong", "chuc", "nang", "phan", "mem", "duan", "du an",
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


def build_bm25_index(
    page_raws: Optional[List[PageRaw]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
    chunk_result: Optional[Dict[str, Any]] = None,
    table_chunk_result: Optional[Dict[str, Any]] = None,
    evidence_result: Optional[Dict[str, Any]] = None,
    metadata_enrichment_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = BM25IndexBuilder()
    return builder.process(
        page_raws=page_raws,
        knowledge_result=knowledge_result,
        chunk_result=chunk_result,
        table_chunk_result=table_chunk_result,
        evidence_result=evidence_result,
        metadata_enrichment_result=metadata_enrichment_result,
    )


def search_bm25_index(
    index_result: Dict[str, Any],
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    builder = BM25IndexBuilder()
    return builder.search(
        index_result=index_result,
        query=query,
        top_k=top_k,
        filters=filters,
    )
