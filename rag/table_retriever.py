"""
table_retriever.py

Production V1 - Colab Ready

Purpose
-------
Retrieve table-related context from:
- table chunks
- table semantic records
- table grids / structures / boundaries
- multi-page tables
- metadata index
- optional BM25 / vector index document stores

Used by:
- QueryRouter
- HybridRetriever
- ContextExpander
- EvidenceAggregator
- PromptBuilder

Input
-----
- query
- table_chunk_result
- table_understanding_result
- metadata_index_result
- bm25_index_result
- vector_index_result
- filters
- page_numbers
- table_ids

Output
------
Dictionary with:
- retrieved_table_items
- retrieved_items
- table_context_text
- table_results_by_page
- table_results_by_table
- table_retrieval_summary
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
class TableRetrieverConfig:
    top_k: int = 20
    candidate_pool_size: int = 120

    include_table_chunks: bool = True
    include_table_semantics: bool = True
    include_table_records: bool = True
    include_table_grids: bool = True
    include_table_structures: bool = True
    include_table_boundaries: bool = True
    include_multi_page_tables: bool = True
    include_metadata_tables: bool = True
    include_index_documents: bool = True

    use_query_overlap: bool = True
    use_numeric_overlap: bool = True
    use_header_match: bool = True
    use_caption_match: bool = True
    use_table_id_match: bool = True
    use_page_filter: bool = True
    use_section_filter: bool = True

    deduplicate_results: bool = True
    include_context_text: bool = True
    include_debug: bool = True

    min_score: float = 0.01

    score_text_overlap: float = 1.25
    score_header_overlap: float = 1.10
    score_caption_overlap: float = 0.85
    score_numeric_overlap: float = 0.65
    score_table_id_match: float = 1.50
    score_page_match: float = 0.45
    score_section_match: float = 0.35
    score_confidence: float = 0.35
    score_existing_rank: float = 0.25

    boost_table_chunk: float = 1.20
    boost_table_semantic: float = 1.18
    boost_table_record: float = 1.25
    boost_multi_page_table: float = 1.15
    boost_metadata_table: float = 0.90
    boost_grid_structure: float = 0.80
    boost_boundary: float = 0.60

    max_text_chars_per_item: int = 1800
    text_preview_chars: int = 700
    max_context_chars: int = 12000

    min_token_len: int = 2
    max_query_tokens: int = 40


class TableRetriever:
    def __init__(
        self,
        config: Optional[TableRetrieverConfig] = None,
    ):
        self.config = config or TableRetrieverConfig()

    def process(
        self,
        query: str = "",
        table_chunk_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        metadata_index_result: Optional[Dict[str, Any]] = None,
        bm25_index_result: Optional[Dict[str, Any]] = None,
        vector_index_result: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        page_numbers: Optional[List[int]] = None,
        section_ids: Optional[List[str]] = None,
        table_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        query = normalize_pdf_text(query)
        table_chunk_result = table_chunk_result or {}
        table_understanding_result = table_understanding_result or {}
        metadata_index_result = metadata_index_result or {}
        bm25_index_result = bm25_index_result or {}
        vector_index_result = vector_index_result or {}
        filters = filters or {}
        page_numbers = self._normalize_page_numbers(page_numbers or [])
        section_ids = [str(item) for item in (section_ids or []) if item]
        table_ids = [str(item) for item in (table_ids or []) if item]
        final_top_k = top_k or self.config.top_k

        merged_filters = self._merge_filters(
            filters=filters,
            page_numbers=page_numbers,
            section_ids=section_ids,
            table_ids=table_ids,
        )

        candidates = self._collect_table_candidates(
            table_chunk_result=table_chunk_result,
            table_understanding_result=table_understanding_result,
            metadata_index_result=metadata_index_result,
            bm25_index_result=bm25_index_result,
            vector_index_result=vector_index_result,
        )

        if self.config.deduplicate_results:
            candidates = self._deduplicate_raw_items(
                candidates,
                [
                    "chunk_id",
                    "table_id",
                    "table_semantic_id",
                    "table_grid_id",
                    "table_structure_id",
                    "table_boundary_id",
                    "multi_page_table_id",
                    "item_id",
                    "document_id",
                    "vector_id",
                ],
            )

        query_tokens = self._tokenize(query)
        query_numbers = self._extract_numbers(query)
        query_table_refs = self._extract_table_refs(query)

        if query_table_refs and not table_ids:
            table_ids = query_table_refs
            merged_filters.setdefault("table_id", table_ids)

        scored_items = []

        for candidate in candidates:
            candidate = self._normalize_table_candidate(candidate)

            if not self._passes_filters(candidate, merged_filters):
                continue

            score_details = self._score_candidate(
                candidate=candidate,
                query=query,
                query_tokens=query_tokens,
                query_numbers=query_numbers,
                table_ids=table_ids,
                page_numbers=page_numbers,
                section_ids=section_ids,
            )

            score = score_details.get("score", 0.0)

            if score < self.config.min_score:
                continue

            item = self._make_retrieved_item(
                candidate=candidate,
                score=score,
                score_details=score_details,
            )
            scored_items.append(item)

        scored_items = sorted(
            scored_items,
            key=lambda item: (
                -self._safe_float(item.get("score"), default=0.0),
                min(item.get("page_numbers", []) or [999999]),
                item.get("table_id", ""),
                item.get("source_type", ""),
            ),
        )

        scored_items = scored_items[:final_top_k]

        for rank, item in enumerate(scored_items, start=1):
            item["rank"] = rank

        table_context_text = ""

        if self.config.include_context_text:
            table_context_text = self._build_context_text(
                query=query,
                items=scored_items,
            )

        result = {
            "processor": "TableRetriever",
            "schema_version": "table_retriever_v1",
            "query": query,
            "retrieved_table_items": scored_items,
            "retrieved_items": scored_items,
            "table_context_text": table_context_text,
            "retrieval_context_text": table_context_text,
            "table_results_by_page": self._group_by_page(scored_items),
            "table_results_by_table": self._group_by_table(scored_items),
            "table_results_by_section": self._group_by_section(scored_items),
            "table_results_by_type": self._group_by_type(scored_items),
            "table_retrieval_summary": self._build_summary(
                query=query,
                candidates=candidates,
                retrieved_items=scored_items,
                filters=merged_filters,
                page_numbers=page_numbers,
                section_ids=section_ids,
                table_ids=table_ids,
                query_tokens=query_tokens,
                query_numbers=query_numbers,
            ),
            "retrieval_summary": {
                "has_results": len(scored_items) > 0,
                "route_name": "table",
                "query": query,
                "retrieved_count": len(scored_items),
                "candidate_count": len(candidates),
            },
            "config": asdict(self.config),
        }

        return json_safe(result)

    def search(
        self,
        query: str,
        table_chunk_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        metadata_index_result: Optional[Dict[str, Any]] = None,
        bm25_index_result: Optional[Dict[str, Any]] = None,
        vector_index_result: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        page_numbers: Optional[List[int]] = None,
        section_ids: Optional[List[str]] = None,
        table_ids: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        result = self.process(
            query=query,
            table_chunk_result=table_chunk_result,
            table_understanding_result=table_understanding_result,
            metadata_index_result=metadata_index_result,
            bm25_index_result=bm25_index_result,
            vector_index_result=vector_index_result,
            filters=filters,
            page_numbers=page_numbers,
            section_ids=section_ids,
            table_ids=table_ids,
            top_k=top_k,
        )

        return result.get("retrieved_table_items", [])

    def _collect_table_candidates(
        self,
        table_chunk_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
        bm25_index_result: Dict[str, Any],
        vector_index_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        if self.config.include_table_chunks:
            candidates.extend(
                self._collect_from_keys(
                    sources=[table_chunk_result],
                    keys=[
                        "table_chunks",
                        "table_summary_chunks",
                        "table_record_chunks",
                        "table_row_chunks",
                        "multi_page_table_chunks",
                        "table_cell_context_chunks",
                    ],
                    candidate_source="table_chunk_result",
                )
            )

        nested_table_chunk = table_chunk_result.get("table_chunk_result", {}) or {}
        if isinstance(nested_table_chunk, dict):
            candidates.extend(
                self._collect_from_keys(
                    sources=[nested_table_chunk],
                    keys=[
                        "table_chunks",
                        "table_summary_chunks",
                        "table_record_chunks",
                        "table_row_chunks",
                        "multi_page_table_chunks",
                        "table_cell_context_chunks",
                    ],
                    candidate_source="table_chunk_result_nested",
                )
            )

        if self.config.include_table_semantics:
            candidates.extend(
                self._collect_from_keys(
                    sources=[table_understanding_result],
                    keys=["table_semantics", "semantic_tables"],
                    candidate_source="table_semantic",
                )
            )

        if self.config.include_table_records:
            candidates.extend(
                self._collect_from_keys(
                    sources=[table_understanding_result],
                    keys=["records", "table_records", "semantic_records"],
                    candidate_source="table_record",
                )
            )

        if self.config.include_table_grids:
            candidates.extend(
                self._collect_from_keys(
                    sources=[table_understanding_result],
                    keys=["table_grids", "grids"],
                    candidate_source="table_grid",
                )
            )

        if self.config.include_table_structures:
            candidates.extend(
                self._collect_from_keys(
                    sources=[table_understanding_result],
                    keys=["table_structures", "structures"],
                    candidate_source="table_structure",
                )
            )

        if self.config.include_table_boundaries:
            candidates.extend(
                self._collect_from_keys(
                    sources=[table_understanding_result],
                    keys=["table_boundaries", "boundaries"],
                    candidate_source="table_boundary",
                )
            )

        if self.config.include_multi_page_tables:
            candidates.extend(
                self._collect_from_keys(
                    sources=[table_understanding_result],
                    keys=["multi_page_tables"],
                    candidate_source="multi_page_table",
                )
            )

        for sub_key in [
            "table_understanding_result",
            "table_semantic_result",
            "table_grid_result",
            "table_structure_result",
            "table_boundary_result",
            "multi_page_table_result",
        ]:
            sub = table_understanding_result.get(sub_key, {}) or {}

            if not isinstance(sub, dict):
                continue

            candidates.extend(
                self._collect_from_keys(
                    sources=[sub],
                    keys=[
                        "table_semantics",
                        "semantic_tables",
                        "records",
                        "table_records",
                        "semantic_records",
                        "table_grids",
                        "grids",
                        "table_structures",
                        "structures",
                        "table_boundaries",
                        "boundaries",
                        "multi_page_tables",
                    ],
                    candidate_source=f"nested_{sub_key}",
                )
            )

        if self.config.include_metadata_tables:
            candidates.extend(
                self._collect_metadata_tables(metadata_index_result=metadata_index_result)
            )

        if self.config.include_index_documents:
            candidates.extend(
                self._collect_index_table_documents(
                    bm25_index_result=bm25_index_result,
                    vector_index_result=vector_index_result,
                )
            )

        return candidates

    def _collect_from_keys(
        self,
        sources: List[Dict[str, Any]],
        keys: List[str],
        candidate_source: str,
    ) -> List[Dict[str, Any]]:
        result = []

        for source in sources:
            if not isinstance(source, dict):
                continue

            for key in keys:
                values = source.get(key, []) or []

                if isinstance(values, dict):
                    for item_id, item in values.items():
                        item = self._to_dict(item)
                        item.setdefault("item_id", item_id)
                        item.setdefault("_candidate_source", candidate_source)
                        item.setdefault("_source_key", key)
                        result.append(item)

                elif isinstance(values, list):
                    for item in values:
                        item = self._to_dict(item)
                        item.setdefault("_candidate_source", candidate_source)
                        item.setdefault("_source_key", key)
                        result.append(item)

        return result

    def _collect_metadata_tables(
        self,
        metadata_index_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        result = []

        metadata_index = metadata_index_result.get("metadata_index", metadata_index_result) or {}
        metadata_store = metadata_index.get("metadata_store", {}) or metadata_index_result.get("metadata_store", {}) or {}

        if not isinstance(metadata_store, dict):
            return result

        for item_id, item in metadata_store.items():
            item = self._to_dict(item)
            item.setdefault("item_id", item_id)

            joined_type = " ".join(
                [
                    str(item.get("item_type", "")),
                    str(item.get("source_type", "")),
                    str(item.get("metadata_type", "")),
                ]
            ).lower()

            if "table" in joined_type or self._table_ids_from_item(item):
                item.setdefault("_candidate_source", "metadata_table")
                item.setdefault("_source_key", "metadata_store")
                result.append(item)

        return result

    def _collect_index_table_documents(
        self,
        bm25_index_result: Dict[str, Any],
        vector_index_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        result = []

        for index_name, index_result in [
            ("bm25_document_store", bm25_index_result),
            ("vector_document_store", vector_index_result),
        ]:
            index_obj = index_result.get("bm25_index", {}) or index_result.get("vector_index", {}) or index_result or {}
            document_store = index_obj.get("document_store", {}) or {}

            if not isinstance(document_store, dict):
                continue

            for document_id, document in document_store.items():
                document = self._to_dict(document)
                document.setdefault("document_id", document_id)

                joined_type = " ".join(
                    [
                        str(document.get("source_type", "")),
                        str(document.get("chunk_type", "")),
                        str(document.get("item_type", "")),
                    ]
                ).lower()

                if "table" in joined_type or self._table_ids_from_item(document):
                    document.setdefault("_candidate_source", index_name)
                    document.setdefault("_source_key", "document_store")
                    result.append(document)

        return result

    def _normalize_table_candidate(
        self,
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidate = self._to_dict(candidate)

        metadata = candidate.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        table_ids = self._table_ids_from_item(candidate)
        table_id = table_ids[0] if table_ids else ""

        source_type = (
            candidate.get("source_type")
            or candidate.get("chunk_type")
            or candidate.get("table_type")
            or candidate.get("semantic_type")
            or candidate.get("item_type")
            or candidate.get("_candidate_source")
            or "table"
        )

        title = normalize_pdf_text(
            candidate.get("title")
            or candidate.get("caption")
            or candidate.get("caption_text")
            or candidate.get("section_title")
            or candidate.get("label")
            or table_id
            or source_type
        )

        text = self._candidate_text(candidate)
        headers = self._extract_headers(candidate)

        if headers and not text:
            text = " | ".join(headers)

        page_numbers = self._resolve_page_numbers(candidate)

        if not page_numbers:
            page_numbers = self._normalize_page_numbers(metadata.get("page_numbers", []))

        normalized = {
            **candidate,
            "table_id": table_id,
            "all_table_ids": table_ids,
            "source_type": source_type,
            "title": title,
            "text": text,
            "text_preview": self._preview(text, self.config.text_preview_chars),
            "normalized_text": normalize_text_for_match(text),
            "headers": headers,
            "header_text": " | ".join(headers),
            "caption_text": normalize_pdf_text(candidate.get("caption") or candidate.get("caption_text") or title),
            "page_numbers": page_numbers,
            "page_start": candidate.get("page_start") or (min(page_numbers) if page_numbers else None),
            "page_end": candidate.get("page_end") or (max(page_numbers) if page_numbers else None),
            "section_id": candidate.get("section_id", "") or metadata.get("section_id", ""),
            "section_title": candidate.get("section_title", "") or metadata.get("section_title", ""),
            "chunk_id": candidate.get("chunk_id", "") or metadata.get("chunk_id", ""),
            "evidence_id": candidate.get("evidence_id", "") or metadata.get("evidence_id", ""),
            "node_id": candidate.get("node_id", "") or metadata.get("node_id", ""),
            "bbox": candidate.get("bbox", []) or [],
            "row_count": self._safe_int(
                candidate.get("row_count", candidate.get("total_row_count", metadata.get("row_count", 0))),
                0,
            ),
            "col_count": self._safe_int(
                candidate.get("col_count", candidate.get("column_count", metadata.get("col_count", 0))),
                0,
            ),
            "confidence": self._safe_float(candidate.get("confidence"), 0.70),
            "metadata": metadata,
        }

        return normalized

    def _candidate_text(
        self,
        candidate: Dict[str, Any],
    ) -> str:
        parts = []

        for key in [
            "title",
            "caption",
            "caption_text",
            "section_title",
            "label",
            "text",
            "text_preview",
            "quote",
            "content",
            "description",
            "summary",
        ]:
            value = normalize_pdf_text(candidate.get(key, ""))
            if value:
                parts.append(value)

        headers = self._extract_headers(candidate)

        if headers:
            parts.append(" | ".join(headers))

        records = candidate.get("records", []) or candidate.get("table_records", []) or []

        if isinstance(records, list):
            for record in records[:12]:
                line = self._record_text(record)
                if line:
                    parts.append(line)

        rows = candidate.get("rows", []) or []

        if isinstance(rows, list):
            for row in rows[:12]:
                line = self._row_text(row)
                if line:
                    parts.append(line)

        cells = candidate.get("cells", []) or candidate.get("grid_cells", []) or []

        if isinstance(cells, list):
            cell_texts = []
            for cell in cells[:80]:
                cell = self._to_dict(cell)
                value = normalize_pdf_text(cell.get("text") or cell.get("value") or "")
                if value:
                    cell_texts.append(value)

            if cell_texts:
                parts.append(" | ".join(cell_texts))

        values = candidate.get("values", {}) or candidate.get("raw_values", {}) or {}

        if isinstance(values, dict):
            line = " | ".join(
                [
                    f"{normalize_pdf_text(key)}: {normalize_pdf_text(value)}"
                    for key, value in values.items()
                    if normalize_pdf_text(value)
                ]
            )
            if line:
                parts.append(line)

        text = normalize_pdf_text("\n".join(parts))
        return self._truncate_text(text, self.config.max_text_chars_per_item)

    def _record_text(
        self,
        record: Any,
    ) -> str:
        record = self._to_dict(record)

        if not record:
            return ""

        if record.get("text"):
            return normalize_pdf_text(record.get("text"))

        values = record.get("values", {}) or record.get("raw_values", {}) or {}

        if isinstance(values, dict):
            return normalize_pdf_text(
                " | ".join(
                    [
                        f"{normalize_pdf_text(key)}: {normalize_pdf_text(value)}"
                        for key, value in values.items()
                        if normalize_pdf_text(value)
                    ]
                )
            )

        return normalize_pdf_text(
            " | ".join(
                [
                    normalize_pdf_text(value)
                    for value in record.values()
                    if isinstance(value, (str, int, float))
                    and normalize_pdf_text(value)
                ]
            )
        )

    def _row_text(
        self,
        row: Any,
    ) -> str:
        if isinstance(row, list):
            return normalize_pdf_text(
                " | ".join(
                    [
                        normalize_pdf_text(value)
                        for value in row
                        if normalize_pdf_text(value)
                    ]
                )
            )

        row = self._to_dict(row)

        if row.get("text"):
            return normalize_pdf_text(row.get("text"))

        if row.get("cells") and isinstance(row.get("cells"), list):
            return normalize_pdf_text(
                " | ".join(
                    [
                        normalize_pdf_text(self._to_dict(cell).get("text") or self._to_dict(cell).get("value") or "")
                        for cell in row.get("cells", [])
                        if normalize_pdf_text(self._to_dict(cell).get("text") or self._to_dict(cell).get("value") or "")
                    ]
                )
            )

        return self._record_text(row)

    def _extract_headers(
        self,
        candidate: Dict[str, Any],
    ) -> List[str]:
        headers = []

        for key in [
            "headers",
            "column_headers",
            "header_cells",
            "columns",
            "column_names",
            "field_names",
        ]:
            values = candidate.get(key, []) or []

            if isinstance(values, dict):
                values = list(values.values())

            if not isinstance(values, list):
                values = [values]

            for item in values:
                if isinstance(item, dict):
                    text = normalize_pdf_text(
                        item.get("text")
                        or item.get("name")
                        or item.get("label")
                        or item.get("header")
                        or item.get("value")
                        or ""
                    )
                else:
                    text = normalize_pdf_text(item)

                if text and text not in headers:
                    headers.append(text)

        structure = candidate.get("structure", {}) or {}

        if isinstance(structure, dict):
            values = structure.get("column_headers", []) or structure.get("headers", []) or []

            if isinstance(values, list):
                for item in values:
                    text = normalize_pdf_text(item.get("text", "") if isinstance(item, dict) else item)
                    if text and text not in headers:
                        headers.append(text)

        return headers

    def _score_candidate(
        self,
        candidate: Dict[str, Any],
        query: str,
        query_tokens: List[str],
        query_numbers: List[str],
        table_ids: List[str],
        page_numbers: List[int],
        section_ids: List[str],
    ) -> Dict[str, Any]:
        text_norm = normalize_text_for_match(candidate.get("text", ""))
        header_norm = normalize_text_for_match(candidate.get("header_text", ""))
        caption_norm = normalize_text_for_match(candidate.get("caption_text", ""))
        title_norm = normalize_text_for_match(candidate.get("title", ""))

        text_overlap = self._query_overlap_score(query_tokens, text_norm)
        header_overlap = self._query_overlap_score(query_tokens, header_norm)
        caption_overlap = max(
            self._query_overlap_score(query_tokens, caption_norm),
            self._query_overlap_score(query_tokens, title_norm),
        )

        candidate_numbers = self._extract_numbers(candidate.get("text", ""))
        numeric_overlap = self._numeric_overlap_score(query_numbers, candidate_numbers)

        item_table_ids = set(candidate.get("all_table_ids", []) or self._table_ids_from_item(candidate))
        requested_table_ids = set(str(item) for item in table_ids if item)
        table_id_match = 0.0

        if requested_table_ids and item_table_ids:
            if requested_table_ids.intersection(item_table_ids):
                table_id_match = 1.0
            else:
                table_id_match = self._soft_table_ref_match(requested_table_ids, item_table_ids)

        item_pages = set(self._resolve_page_numbers(candidate))
        requested_pages = set(page_numbers or [])
        page_match = 0.0

        if requested_pages:
            if item_pages and item_pages.intersection(requested_pages):
                page_match = 1.0
            elif not item_pages:
                page_match = 0.0

        item_section = candidate.get("section_id", "") or candidate.get("metadata", {}).get("section_id", "")
        requested_sections = set(section_ids or [])
        section_match = 0.0

        if requested_sections and item_section and str(item_section) in requested_sections:
            section_match = 1.0

        base_score = self._safe_float(candidate.get("score"), 0.0)
        confidence = self._safe_float(candidate.get("confidence"), 0.70)
        source_boost = self._source_boost(candidate)

        score = 0.0
        score += text_overlap * self.config.score_text_overlap
        score += header_overlap * self.config.score_header_overlap
        score += caption_overlap * self.config.score_caption_overlap
        score += numeric_overlap * self.config.score_numeric_overlap
        score += table_id_match * self.config.score_table_id_match
        score += page_match * self.config.score_page_match
        score += section_match * self.config.score_section_match
        score += confidence * self.config.score_confidence
        score += base_score * self.config.score_existing_rank

        if query and normalize_text_for_match(query) in text_norm:
            score += 0.35

        if query and normalize_text_for_match(query) in caption_norm:
            score += 0.25

        if not query_tokens and (requested_pages or requested_table_ids or requested_sections):
            score += 0.50

        score = score * source_boost

        matched_terms = self._matched_terms(query_tokens, text_norm + " " + header_norm + " " + caption_norm)

        return {
            "score": round(score, 6),
            "text_overlap": round(text_overlap, 4),
            "header_overlap": round(header_overlap, 4),
            "caption_overlap": round(caption_overlap, 4),
            "numeric_overlap": round(numeric_overlap, 4),
            "table_id_match": round(table_id_match, 4),
            "page_match": round(page_match, 4),
            "section_match": round(section_match, 4),
            "source_boost": source_boost,
            "confidence": confidence,
            "base_score": base_score,
            "matched_terms": matched_terms,
            "matched_numbers": sorted(list(set(query_numbers).intersection(set(candidate_numbers)))),
        }

    def _make_retrieved_item(
        self,
        candidate: Dict[str, Any],
        score: float,
        score_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        metadata = candidate.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        source_id = (
            candidate.get("source_id")
            or candidate.get("document_id")
            or candidate.get("vector_id")
            or candidate.get("chunk_id")
            or candidate.get("table_id")
            or candidate.get("table_semantic_id")
            or candidate.get("table_grid_id")
            or candidate.get("table_structure_id")
            or candidate.get("table_boundary_id")
            or candidate.get("multi_page_table_id")
            or candidate.get("item_id")
            or self._stable_id(candidate.get("text", ""), "table")
        )

        table_id = candidate.get("table_id") or (candidate.get("all_table_ids", []) or [""])[0]

        return {
            "rank": 0,
            "score": round(score, 6),
            "retrieval_source": "table",
            "source_id": source_id,
            "source_type": candidate.get("source_type", "table"),
            "source": candidate.get("_candidate_source", ""),
            "title": candidate.get("title", ""),
            "text": candidate.get("text", ""),
            "text_preview": self._preview(candidate.get("text", ""), self.config.text_preview_chars),
            "normalized_text": normalize_text_for_match(candidate.get("text", "")),
            "page_numbers": candidate.get("page_numbers", []),
            "page_start": candidate.get("page_start"),
            "page_end": candidate.get("page_end"),
            "section_id": candidate.get("section_id", ""),
            "section_title": candidate.get("section_title", ""),
            "table_id": table_id,
            "all_table_ids": candidate.get("all_table_ids", []),
            "chunk_id": candidate.get("chunk_id", ""),
            "evidence_id": candidate.get("evidence_id", ""),
            "node_id": candidate.get("node_id", ""),
            "bbox": candidate.get("bbox", []) or [],
            "row_count": candidate.get("row_count", 0),
            "col_count": candidate.get("col_count", 0),
            "headers": candidate.get("headers", []),
            "caption_text": candidate.get("caption_text", ""),
            "matched_terms": score_details.get("matched_terms", []),
            "matched_numbers": score_details.get("matched_numbers", []),
            "score_details": score_details if self.config.include_debug else {},
            "confidence": candidate.get("confidence", 0.0),
            "metadata": metadata if self.config.include_debug else {},
        }

    def _source_boost(
        self,
        candidate: Dict[str, Any],
    ) -> float:
        source = " ".join(
            [
                str(candidate.get("_candidate_source", "")),
                str(candidate.get("_source_key", "")),
                str(candidate.get("source_type", "")),
                str(candidate.get("chunk_type", "")),
                str(candidate.get("item_type", "")),
            ]
        ).lower()

        if "record" in source:
            return self.config.boost_table_record

        if "chunk" in source:
            return self.config.boost_table_chunk

        if "semantic" in source:
            return self.config.boost_table_semantic

        if "multi_page" in source:
            return self.config.boost_multi_page_table

        if "metadata" in source:
            return self.config.boost_metadata_table

        if "grid" in source or "structure" in source:
            return self.config.boost_grid_structure

        if "boundary" in source:
            return self.config.boost_boundary

        return 1.0

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
            if expected in [None, ""]:
                continue

            if key in ["prefer_table", "require_evidence"]:
                continue

            if key == "page_numbers":
                expected_pages = set(self._normalize_page_numbers(expected))
                item_pages = set(self._resolve_page_numbers(item))

                if expected_pages and not item_pages.intersection(expected_pages):
                    return False

                continue

            if key == "table_id":
                actual_values = item.get("all_table_ids", []) or self._table_ids_from_item(item)
            else:
                actual = item.get(key)

                if actual in [None, ""]:
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

    def _numeric_overlap_score(
        self,
        query_numbers: List[str],
        candidate_numbers: List[str],
    ) -> float:
        if not query_numbers:
            return 0.0

        if not candidate_numbers:
            return 0.0

        query_set = set(query_numbers)
        candidate_set = set(candidate_numbers)

        return len(query_set.intersection(candidate_set)) / max(len(query_set), 1)

    def _soft_table_ref_match(
        self,
        requested: Set[str],
        actual: Set[str],
    ) -> float:
        if not requested or not actual:
            return 0.0

        best = 0.0

        requested_norm = {normalize_text_for_match(item) for item in requested}
        actual_norm = {normalize_text_for_match(item) for item in actual}

        for req in requested_norm:
            for act in actual_norm:
                if req == act:
                    best = max(best, 1.0)
                elif req and act and (req in act or act in req):
                    best = max(best, 0.65)

        return best

    def _matched_terms(
        self,
        query_tokens: List[str],
        text_norm: str,
    ) -> List[str]:
        if not query_tokens or not text_norm:
            return []

        text_tokens = set(text_norm.split())

        return sorted(list(set(query_tokens).intersection(text_tokens)))

    def _build_context_text(
        self,
        query: str,
        items: List[Dict[str, Any]],
    ) -> str:
        parts = []

        if query:
            parts.append(f"Truy vấn: {normalize_pdf_text(query)}")

        total_chars = 0

        for index, item in enumerate(items, start=1):
            pages = item.get("page_numbers", []) or []
            page_label = self._page_label(pages)

            header_parts = [
                f"[Table {index}]",
                f"score={item.get('score', 0)}",
                f"type={item.get('source_type', '')}",
            ]

            if item.get("table_id"):
                header_parts.append(f"table_id={item.get('table_id')}")

            if item.get("title"):
                header_parts.append(item.get("title"))

            if page_label:
                header_parts.append(page_label)

            if item.get("headers"):
                header_parts.append("headers=" + " | ".join(item.get("headers", [])[:8]))

            text = normalize_pdf_text(item.get("text") or item.get("text_preview") or "")

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

    def _group_by_page(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in items:
            for page_number in item.get("page_numbers", []) or []:
                key = str(page_number)
                grouped.setdefault(key, [])
                grouped[key].append(item)

        return grouped

    def _group_by_table(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in items:
            table_ids = item.get("all_table_ids", []) or self._table_ids_from_item(item)

            if not table_ids and item.get("table_id"):
                table_ids = [item.get("table_id")]

            for table_id in table_ids:
                if not table_id:
                    continue

                key = str(table_id)
                grouped.setdefault(key, [])
                grouped[key].append(item)

        return grouped

    def _group_by_section(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in items:
            section_id = item.get("section_id", "") or "no_section"
            grouped.setdefault(section_id, [])
            grouped[section_id].append(item)

        return grouped

    def _group_by_type(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in items:
            source_type = item.get("source_type", "") or "table"
            grouped.setdefault(source_type, [])
            grouped[source_type].append(item)

        return grouped

    def _build_summary(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        retrieved_items: List[Dict[str, Any]],
        filters: Dict[str, Any],
        page_numbers: List[int],
        section_ids: List[str],
        table_ids: List[str],
        query_tokens: List[str],
        query_numbers: List[str],
    ) -> Dict[str, Any]:
        by_type = {}
        by_page = {}
        by_table = {}

        scores = []

        for item in retrieved_items:
            source_type = item.get("source_type", "table")
            by_type[source_type] = by_type.get(source_type, 0) + 1

            scores.append(self._safe_float(item.get("score"), 0.0))

            for page in item.get("page_numbers", []) or []:
                page_key = str(page)
                by_page[page_key] = by_page.get(page_key, 0) + 1

            for table_id in item.get("all_table_ids", []) or [item.get("table_id", "")]:
                if not table_id:
                    continue

                table_key = str(table_id)
                by_table[table_key] = by_table.get(table_key, 0) + 1

        return {
            "has_results": len(retrieved_items) > 0,
            "query": query,
            "candidate_count": len(candidates),
            "retrieved_count": len(retrieved_items),
            "query_token_count": len(query_tokens),
            "query_numbers": query_numbers,
            "avg_score": round(sum(scores) / max(len(scores), 1), 6),
            "max_score": round(max(scores), 6) if scores else 0.0,
            "min_score": round(min(scores), 6) if scores else 0.0,
            "by_type": by_type,
            "by_page": by_page,
            "by_table": by_table,
            "filters": filters,
            "page_numbers_filter": page_numbers,
            "section_ids_filter": section_ids,
            "table_ids_filter": table_ids,
            "top_k": self.config.top_k,
            "candidate_pool_size": self.config.candidate_pool_size,
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
            metadata = item.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}

            key = ""

            for key_name in keys:
                value = item.get(key_name) or metadata.get(key_name)

                if value:
                    key = f"{key_name}:{value}"
                    break

            if not key:
                table_ids = self._table_ids_from_item(item)
                text = normalize_text_for_match(
                    item.get("text")
                    or item.get("text_preview")
                    or item.get("caption")
                    or item.get("title")
                    or ""
                )
                key = f"table_ids:{table_ids}|text:{text[:600]}|pages:{self._resolve_page_numbers(item)}"

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _extract_table_refs(
        self,
        query: str,
    ) -> List[str]:
        query_norm = normalize_text_for_match(query)
        refs = []

        patterns = [
            r"\bbang\s+(\d+(?:\.\d+)*)",
            r"\btable\s+(\d+(?:\.\d+)*)",
            r"\bbieu\s+(\d+(?:\.\d+)*)",
            r"\btable_id\s*[:=]\s*([a-z0-9_\-\.]+)",
        ]

        for pattern in patterns:
            for value in re.findall(pattern, query_norm, flags=re.IGNORECASE):
                value = normalize_pdf_text(value)
                if value:
                    refs.append(value)

        return list(dict.fromkeys(refs))

    def _extract_numbers(
        self,
        text: str,
    ) -> List[str]:
        text = normalize_pdf_text(text)

        numbers = re.findall(
            r"\b\d+(?:[.,]\d+)?(?:\s*%|\s*tỷ|\s*triệu|\s*nghìn|\s*ngàn|\s*usd|\s*vnd)?\b",
            text,
            flags=re.IGNORECASE,
        )

        result = []

        for number in numbers:
            number = normalize_text_for_match(number)
            number = number.replace(" ", "")
            if number:
                result.append(number)

        return list(dict.fromkeys(result))

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:
        text = normalize_text_for_match(text)
        raw_tokens = re.findall(r"[a-z0-9_]+", text)

        stopwords = self._stopwords()

        tokens = []

        for token in raw_tokens:
            token = token.strip("_")

            if len(token) < self.config.min_token_len:
                continue

            if token in stopwords:
                continue

            tokens.append(token)

        return list(dict.fromkeys(tokens))[: self.config.max_query_tokens]

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
                "table_semantic_id",
                "table_grid_id",
                "table_structure_id",
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
            page_start = self._safe_int(page_start, 0)
            page_end = self._safe_int(page_end, 0)

            if page_start > 0 and page_end >= page_start:
                return list(range(page_start, page_end + 1))

        page_number = self._safe_int(item.get("page_number"), 0)

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
            page = self._safe_int(value, 0)

            if page > 0:
                result.append(page)

        return sorted(list(dict.fromkeys(result)))

    def _page_label(
        self,
        page_numbers: List[int],
    ) -> str:
        page_numbers = self._normalize_page_numbers(page_numbers)

        if len(page_numbers) == 1:
            return f"trang {page_numbers[0]}"

        if len(page_numbers) > 1:
            return f"trang {page_numbers[0]}-{page_numbers[-1]}"

        return ""

    def _truncate_text(
        self,
        text: Any,
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

    def _stopwords(
        self,
    ) -> Set[str]:
        return {
            "va", "hoac", "cua", "cho", "cac", "mot", "nhung", "duoc", "trong",
            "ngoai", "theo", "voi", "den", "tu", "tai", "ve", "la", "co", "khong",
            "nay", "do", "khi", "sau", "truoc", "tren", "duoi", "vao", "ra", "de",
            "nham", "phuc", "vu", "can", "phai", "bao", "dam", "quy", "dinh",
            "noi", "dung", "thuc", "hien", "quan", "ly", "nha", "nuoc", "du",
            "lieu", "he", "thong", "chuc", "nang", "phan", "mem", "giup", "toi",
            "ban", "hay", "sinh", "file", "bang", "table", "cot", "dong",
            "what", "why", "how", "when", "where", "which", "please", "help",
            "the", "and", "or", "for", "to", "of", "in", "on", "by", "is", "are",
        }

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


def retrieve_from_table(
    query: str = "",
    table_chunk_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    metadata_index_result: Optional[Dict[str, Any]] = None,
    bm25_index_result: Optional[Dict[str, Any]] = None,
    vector_index_result: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    page_numbers: Optional[List[int]] = None,
    section_ids: Optional[List[str]] = None,
    table_ids: Optional[List[str]] = None,
    top_k: int = 20,
) -> Dict[str, Any]:
    retriever = TableRetriever()
    return retriever.process(
        query=query,
        table_chunk_result=table_chunk_result,
        table_understanding_result=table_understanding_result,
        metadata_index_result=metadata_index_result,
        bm25_index_result=bm25_index_result,
        vector_index_result=vector_index_result,
        filters=filters,
        page_numbers=page_numbers,
        section_ids=section_ids,
        table_ids=table_ids,
        top_k=top_k,
    )


def search_table(
    query: str,
    table_chunk_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    metadata_index_result: Optional[Dict[str, Any]] = None,
    bm25_index_result: Optional[Dict[str, Any]] = None,
    vector_index_result: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    page_numbers: Optional[List[int]] = None,
    section_ids: Optional[List[str]] = None,
    table_ids: Optional[List[str]] = None,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    retriever = TableRetriever()
    return retriever.search(
        query=query,
        table_chunk_result=table_chunk_result,
        table_understanding_result=table_understanding_result,
        metadata_index_result=metadata_index_result,
        bm25_index_result=bm25_index_result,
        vector_index_result=vector_index_result,
        filters=filters,
        page_numbers=page_numbers,
        section_ids=section_ids,
        table_ids=table_ids,
        top_k=top_k,
    )
