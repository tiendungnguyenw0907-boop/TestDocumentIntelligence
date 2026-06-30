"""
evidence_aggregator.py

Production V1 - Colab Ready

Purpose
-------
Aggregate, rank, deduplicate, and prepare evidence for RAG answer generation.

Used by:
- RAGPipeline
- ContextExpander
- PromptBuilder
- LLMReasoner
- CitationVerifier

Input
-----
- query
- retrieved_items
- expanded_context_result
- page_raws
- evidence_result
- citation_result
- chunk_result
- table_chunk_result
- graph_index_result
- metadata_index_result
- bm25_index_result
- vector_index_result

Output
------
Dictionary with:
- aggregated_evidence
- supporting_evidence
- evidence_context_text
- citations
- evidence_by_page
- evidence_by_section
- evidence_by_type
- evidence_aggregation_summary
"""

from __future__ import annotations

import json
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
class EvidenceAggregatorConfig:
    include_retrieved_items: bool = True
    include_expanded_context: bool = True
    include_existing_evidence: bool = True
    include_chunks: bool = True
    include_table_chunks: bool = True
    include_graph_nodes: bool = True
    include_metadata_items: bool = True
    include_page_text: bool = False

    deduplicate_evidence: bool = True
    attach_to_pages: bool = True

    max_evidence_items: int = 40
    max_supporting_evidence: int = 18
    max_context_chars: int = 12000
    max_text_chars_per_evidence: int = 1800
    text_preview_chars: int = 700
    quote_chars: int = 500

    min_text_chars: int = 20
    min_evidence_score: float = 0.05

    query_overlap_weight: float = 1.25
    retrieval_score_weight: float = 1.00
    confidence_weight: float = 0.75
    source_weight: float = 0.50
    citation_weight: float = 0.35
    recency_order_weight: float = 0.05

    boost_original_context: float = 1.20
    boost_evidence_source: float = 1.15
    boost_table_source: float = 1.10
    boost_chunk_source: float = 1.00
    boost_page_source: float = 0.75
    boost_graph_source: float = 0.80
    boost_metadata_source: float = 0.70

    build_citations: bool = True
    citation_marker_prefix: str = "tr."
    require_page_for_citation: bool = False

    include_debug: bool = True


class EvidenceAggregator:
    def __init__(
        self,
        config: Optional[EvidenceAggregatorConfig] = None,
    ):
        self.config = config or EvidenceAggregatorConfig()

    def process(
        self,
        query: str = "",
        retrieved_items: Optional[List[Dict[str, Any]]] = None,
        expanded_context_result: Optional[Dict[str, Any]] = None,
        page_raws: Optional[List[PageRaw]] = None,
        evidence_result: Optional[Dict[str, Any]] = None,
        citation_result: Optional[Dict[str, Any]] = None,
        chunk_result: Optional[Dict[str, Any]] = None,
        table_chunk_result: Optional[Dict[str, Any]] = None,
        graph_index_result: Optional[Dict[str, Any]] = None,
        metadata_index_result: Optional[Dict[str, Any]] = None,
        bm25_index_result: Optional[Dict[str, Any]] = None,
        vector_index_result: Optional[Dict[str, Any]] = None,
        knowledge_result: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        retrieved_items = [
            self._to_dict(item)
            for item in (retrieved_items or [])
            if isinstance(item, dict)
        ]

        expanded_context_result = expanded_context_result or {}
        page_raws = sorted(
            page_raws or [],
            key=lambda item: item.page_number,
        )
        evidence_result = evidence_result or {}
        citation_result = citation_result or {}
        chunk_result = chunk_result or {}
        table_chunk_result = table_chunk_result or {}
        graph_index_result = graph_index_result or {}
        metadata_index_result = metadata_index_result or {}
        bm25_index_result = bm25_index_result or {}
        vector_index_result = vector_index_result or {}
        knowledge_result = knowledge_result or {}

        page_text_map = self._build_page_text_map(page_raws)

        candidates = self._collect_candidates(
            retrieved_items=retrieved_items,
            expanded_context_result=expanded_context_result,
            page_raws=page_raws,
            evidence_result=evidence_result,
            citation_result=citation_result,
            chunk_result=chunk_result,
            table_chunk_result=table_chunk_result,
            graph_index_result=graph_index_result,
            metadata_index_result=metadata_index_result,
            bm25_index_result=bm25_index_result,
            vector_index_result=vector_index_result,
            knowledge_result=knowledge_result,
            page_text_map=page_text_map,
        )

        aggregated_evidence = []

        for candidate in candidates:
            evidence = self._candidate_to_evidence(
                candidate=candidate,
                query=query,
                page_text_map=page_text_map,
            )

            if not self._valid_evidence(evidence):
                continue

            evidence["evidence_score"] = self._score_evidence(
                evidence=evidence,
                query=query,
            )

            if evidence["evidence_score"] < self.config.min_evidence_score:
                continue

            aggregated_evidence.append(evidence)

        if self.config.deduplicate_evidence:
            aggregated_evidence = self._deduplicate_evidence(aggregated_evidence)

        aggregated_evidence = self._sort_evidence(aggregated_evidence)

        for rank, evidence in enumerate(aggregated_evidence, start=1):
            evidence["rank"] = rank
            evidence["is_supporting_evidence"] = rank <= self.config.max_supporting_evidence

        aggregated_evidence = aggregated_evidence[: self.config.max_evidence_items]
        supporting_evidence = aggregated_evidence[: self.config.max_supporting_evidence]

        citations = []

        if self.config.build_citations:
            citations = self._build_citations(
                evidence_items=supporting_evidence,
                citation_result=citation_result,
            )

        evidence_context_text = self._build_context_text(
            evidence_items=supporting_evidence,
            citations=citations,
            query=query,
        )

        result = {
            "processor": "EvidenceAggregator",
            "schema_version": "evidence_aggregator_v1",
            "query": query,
            "aggregated_evidence": aggregated_evidence,
            "supporting_evidence": supporting_evidence,
            "evidence_context_text": evidence_context_text,
            "citations": citations,
            "evidence_by_page": self._group_evidence_by_page(aggregated_evidence),
            "evidence_by_section": self._group_evidence_by_section(aggregated_evidence),
            "evidence_by_type": self._group_evidence_by_type(aggregated_evidence),
            "evidence_by_source_type": self._group_evidence_by_source_type(aggregated_evidence),
            "citation_markers": self._build_citation_markers(citations),
            "evidence_aggregation_summary": self._build_summary(
                query=query,
                candidates=candidates,
                aggregated_evidence=aggregated_evidence,
                supporting_evidence=supporting_evidence,
                citations=citations,
                page_raws=page_raws,
            ),
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                aggregation_result=result,
            )

        return json_safe(result)

    def _collect_candidates(
        self,
        retrieved_items: List[Dict[str, Any]],
        expanded_context_result: Dict[str, Any],
        page_raws: List[PageRaw],
        evidence_result: Dict[str, Any],
        citation_result: Dict[str, Any],
        chunk_result: Dict[str, Any],
        table_chunk_result: Dict[str, Any],
        graph_index_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
        bm25_index_result: Dict[str, Any],
        vector_index_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
        page_text_map: Dict[int, str],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        if self.config.include_retrieved_items:
            for item in retrieved_items:
                item = self._to_dict(item)
                item["_candidate_source"] = "retrieved_item"
                candidates.append(item)

        if self.config.include_expanded_context:
            expanded_items = expanded_context_result.get("expanded_context_items", []) or []

            if isinstance(expanded_items, list):
                for item in expanded_items:
                    item = self._to_dict(item)
                    item["_candidate_source"] = "expanded_context"
                    candidates.append(item)

        if self.config.include_existing_evidence:
            for item in self._collect_evidence(evidence_result, knowledge_result):
                item["_candidate_source"] = "existing_evidence"
                candidates.append(item)

        if self.config.include_chunks:
            for item in self._collect_chunks(chunk_result, knowledge_result):
                item["_candidate_source"] = "chunk"
                candidates.append(item)

        if self.config.include_table_chunks:
            for item in self._collect_table_chunks(table_chunk_result, knowledge_result):
                item["_candidate_source"] = "table_chunk"
                candidates.append(item)

        if self.config.include_graph_nodes:
            for item in self._collect_graph_nodes(graph_index_result, knowledge_result):
                item["_candidate_source"] = "graph_node"
                candidates.append(item)

        if self.config.include_metadata_items:
            for item in self._collect_metadata_items(metadata_index_result):
                item["_candidate_source"] = "metadata_item"
                candidates.append(item)

        if self.config.include_page_text:
            for page_raw in page_raws:
                text = page_text_map.get(page_raw.page_number, "")

                if not text:
                    continue

                candidates.append(
                    {
                        "_candidate_source": "page_text",
                        "source_id": f"page_{page_raw.page_number}",
                        "source_type": "page",
                        "title": f"Trang {page_raw.page_number}",
                        "text": text,
                        "page_number": page_raw.page_number,
                        "page_numbers": [page_raw.page_number],
                        "score": 0.30,
                        "confidence": 0.60,
                        "metadata": {
                            "page_index": page_raw.page_index,
                            "page_kind": page_raw.page_kind,
                        },
                    }
                )

        return candidates

    def _candidate_to_evidence(
        self,
        candidate: Dict[str, Any],
        query: str,
        page_text_map: Dict[int, str],
    ) -> Dict[str, Any]:
        candidate = self._to_dict(candidate)
        metadata = candidate.get("metadata", {}) or {}

        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        text = normalize_pdf_text(
            candidate.get("text")
            or candidate.get("text_preview")
            or candidate.get("quote")
            or candidate.get("content")
            or candidate.get("citation_text")
            or candidate.get("label")
            or candidate.get("title")
            or ""
        )

        text = self._truncate_text(text, self.config.max_text_chars_per_evidence)

        source_type = (
            candidate.get("source_type")
            or candidate.get("chunk_type")
            or candidate.get("evidence_type")
            or candidate.get("node_type")
            or candidate.get("item_type")
            or candidate.get("_candidate_source")
            or "evidence"
        )

        evidence_id = (
            candidate.get("evidence_id")
            or candidate.get("context_id")
            or candidate.get("chunk_id")
            or candidate.get("document_id")
            or candidate.get("node_id")
            or candidate.get("item_id")
            or candidate.get("source_id")
            or self._stable_id(text, "evidence")
        )

        page_numbers = self._resolve_page_numbers(candidate)

        if not page_numbers and candidate.get("page_number"):
            page_number = self._safe_int(candidate.get("page_number"), default=0)
            if page_number > 0:
                page_numbers = [page_number]

        quote = normalize_pdf_text(candidate.get("quote", ""))

        if not quote:
            quote = self._make_quote(text)

        context_before, context_after = self._find_page_context(
            quote=quote,
            page_numbers=page_numbers,
            page_text_map=page_text_map,
        )

        evidence = {
            "evidence_id": str(evidence_id),
            "evidence_type": self._infer_evidence_type(candidate),
            "source_id": candidate.get("source_id", "") or str(evidence_id),
            "source_type": source_type,
            "source": candidate.get("source", "") or candidate.get("_candidate_source", ""),
            "candidate_source": candidate.get("_candidate_source", ""),
            "title": normalize_pdf_text(candidate.get("title") or candidate.get("section_title") or candidate.get("label") or ""),
            "text": text,
            "quote": quote,
            "text_preview": self._preview(text, self.config.text_preview_chars),
            "normalized_text": normalize_text_for_match(text),
            "page_number": min(page_numbers) if page_numbers else None,
            "page_numbers": page_numbers,
            "page_start": candidate.get("page_start") or (min(page_numbers) if page_numbers else None),
            "page_end": candidate.get("page_end") or (max(page_numbers) if page_numbers else None),
            "section_id": candidate.get("section_id", "") or metadata.get("section_id", ""),
            "section_title": candidate.get("section_title", "") or metadata.get("section_title", ""),
            "chunk_id": candidate.get("chunk_id", "") or metadata.get("chunk_id", ""),
            "evidence_source_id": candidate.get("evidence_id", "") or metadata.get("evidence_id", ""),
            "citation_id": candidate.get("citation_id", "") or metadata.get("citation_id", ""),
            "table_id": self._table_ids_from_item(candidate)[0] if self._table_ids_from_item(candidate) else "",
            "node_id": candidate.get("node_id", "") or metadata.get("node_id", ""),
            "bbox": candidate.get("bbox", []) or [],
            "context_before": context_before,
            "context_after": context_after,
            "retrieval_score": self._safe_float(
                candidate.get("score", candidate.get("retrieval_score", candidate.get("rerank_score", 0.0))),
                default=0.0,
            ),
            "confidence": self._safe_float(candidate.get("confidence"), default=0.70),
            "weight": self._safe_float(candidate.get("weight"), default=1.0),
            "query_overlap": self._query_overlap(query, text),
            "has_citation": bool(candidate.get("citation_id") or candidate.get("citation_marker")),
            "citation_marker": candidate.get("citation_marker", ""),
            "citation_text": candidate.get("citation_text", ""),
            "content_hash": candidate.get("content_hash", "") or self._stable_hash(text),
            "metadata": metadata if self.config.include_debug else {},
        }

        return evidence

    def _score_evidence(
        self,
        evidence: Dict[str, Any],
        query: str,
    ) -> float:
        retrieval_score = self._safe_float(evidence.get("retrieval_score"), default=0.0)
        confidence = self._safe_float(evidence.get("confidence"), default=0.70)
        weight = self._safe_float(evidence.get("weight"), default=1.0)
        query_overlap = self._safe_float(evidence.get("query_overlap"), default=0.0)

        source_boost = self._source_boost(evidence)
        citation_boost = 1.0 + (self.config.citation_weight if evidence.get("has_citation") else 0.0)

        if retrieval_score <= 0:
            retrieval_score = 0.35

        score = (
            retrieval_score * self.config.retrieval_score_weight
            + query_overlap * self.config.query_overlap_weight
            + confidence * self.config.confidence_weight
            + source_boost * self.config.source_weight
        )

        score = score * max(weight, 0.1) * citation_boost

        return round(score, 6)

    def _source_boost(
        self,
        evidence: Dict[str, Any],
    ) -> float:
        evidence_type = evidence.get("evidence_type", "")
        source_type = evidence.get("source_type", "")
        candidate_source = evidence.get("candidate_source", "")

        joined = f"{evidence_type} {source_type} {candidate_source}".lower()

        if "original" in joined or candidate_source == "retrieved_item":
            return self.config.boost_original_context

        if "evidence" in joined:
            return self.config.boost_evidence_source

        if "table" in joined:
            return self.config.boost_table_source

        if "chunk" in joined:
            return self.config.boost_chunk_source

        if "page" in joined:
            return self.config.boost_page_source

        if "graph" in joined or "node" in joined:
            return self.config.boost_graph_source

        if "metadata" in joined:
            return self.config.boost_metadata_source

        return 1.0

    def _query_overlap(
        self,
        query: str,
        text: str,
    ) -> float:
        query_norm = normalize_text_for_match(query)
        text_norm = normalize_text_for_match(text)

        if not query_norm or not text_norm:
            return 0.0

        query_tokens = [
            token for token in query_norm.split()
            if len(token) >= 3
        ]

        if not query_tokens:
            return 0.0

        text_tokens = set(
            token for token in text_norm.split()
            if len(token) >= 3
        )

        matched = sum(1 for token in query_tokens if token in text_tokens)

        return round(matched / max(len(query_tokens), 1), 4)

    def _build_citations(
        self,
        evidence_items: List[Dict[str, Any]],
        citation_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        existing_citations = self._collect_citations(citation_result)
        existing_by_evidence = {}

        for citation in existing_citations:
            evidence_ids = citation.get("evidence_ids", []) or []

            if citation.get("evidence_id"):
                evidence_ids.append(citation.get("evidence_id"))

            for evidence_id in evidence_ids:
                if evidence_id:
                    existing_by_evidence.setdefault(evidence_id, [])
                    existing_by_evidence[evidence_id].append(citation)

        citations = []

        marker_counts = {}

        for evidence in evidence_items:
            evidence_id = evidence.get("evidence_id", "")

            linked = existing_by_evidence.get(evidence_id, [])

            if linked:
                for citation in linked:
                    citation = self._normalize_citation(citation, evidence=evidence)
                    citations.append(citation)
                continue

            page_numbers = self._resolve_page_numbers(evidence)

            if self.config.require_page_for_citation and not page_numbers:
                continue

            marker = self._make_citation_marker(page_numbers)
            marker_counts[marker] = marker_counts.get(marker, 0) + 1

            if marker_counts[marker] > 1:
                marker = marker.replace("]", f".{marker_counts[marker]}]")

            citation = {
                "citation_id": self._stable_id(
                    {
                        "evidence_id": evidence_id,
                        "page_numbers": page_numbers,
                        "quote": evidence.get("quote", ""),
                    },
                    "citation",
                ),
                "citation_type": f"{evidence.get('evidence_type', 'evidence')}_citation",
                "evidence_id": evidence_id,
                "evidence_ids": [evidence_id] if evidence_id else [],
                "chunk_id": evidence.get("chunk_id", ""),
                "table_id": evidence.get("table_id", ""),
                "node_id": evidence.get("node_id", ""),
                "page_number": min(page_numbers) if page_numbers else None,
                "page_numbers": page_numbers,
                "page_start": min(page_numbers) if page_numbers else None,
                "page_end": max(page_numbers) if page_numbers else None,
                "section_id": evidence.get("section_id", ""),
                "section_title": evidence.get("section_title", ""),
                "bbox": evidence.get("bbox", []) or [],
                "quote": evidence.get("quote", ""),
                "citation_marker": marker,
                "citation_text": self._make_citation_text(evidence, page_numbers),
                "confidence": evidence.get("confidence", 0.70),
                "verified": False,
                "verification_status": "unverified",
                "source": "evidence_aggregator_auto_citation",
                "metadata": {
                    "source_type": evidence.get("source_type", ""),
                    "evidence_score": evidence.get("evidence_score", 0.0),
                },
            }
            citations.append(citation)

        return self._deduplicate_citations(citations)

    def _normalize_citation(
        self,
        citation: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        citation = self._to_dict(citation)

        page_numbers = self._resolve_page_numbers(citation)

        if not page_numbers:
            page_numbers = self._resolve_page_numbers(evidence)

        if not citation.get("citation_marker"):
            citation["citation_marker"] = self._make_citation_marker(page_numbers)

        if not citation.get("citation_text"):
            citation["citation_text"] = self._make_citation_text(evidence, page_numbers)

        if not citation.get("quote"):
            citation["quote"] = evidence.get("quote", "")

        if not citation.get("evidence_id"):
            citation["evidence_id"] = evidence.get("evidence_id", "")

        evidence_ids = citation.get("evidence_ids", []) or []

        if citation.get("evidence_id") and citation.get("evidence_id") not in evidence_ids:
            evidence_ids.append(citation.get("evidence_id"))

        citation["evidence_ids"] = evidence_ids
        citation["page_numbers"] = page_numbers
        citation["page_number"] = min(page_numbers) if page_numbers else None
        citation["page_start"] = min(page_numbers) if page_numbers else None
        citation["page_end"] = max(page_numbers) if page_numbers else None

        if not citation.get("citation_id"):
            citation["citation_id"] = self._stable_id(
                {
                    "marker": citation.get("citation_marker", ""),
                    "evidence_id": citation.get("evidence_id", ""),
                    "quote": citation.get("quote", ""),
                },
                "citation",
            )

        return citation

    def _make_citation_marker(
        self,
        page_numbers: List[int],
    ) -> str:
        page_numbers = self._normalize_page_numbers(page_numbers)

        if len(page_numbers) == 1:
            return f"[{self.config.citation_marker_prefix}{page_numbers[0]}]"

        if len(page_numbers) > 1:
            return f"[{self.config.citation_marker_prefix}{page_numbers[0]}-{page_numbers[-1]}]"

        return "[nguồn]"

    def _make_citation_text(
        self,
        evidence: Dict[str, Any],
        page_numbers: List[int],
    ) -> str:
        parts = []

        title = normalize_pdf_text(
            evidence.get("title")
            or evidence.get("section_title")
            or evidence.get("source_type")
            or ""
        )

        if title:
            parts.append(title)

        if page_numbers:
            if len(page_numbers) == 1:
                parts.append(f"trang {page_numbers[0]}")
            else:
                parts.append(f"trang {page_numbers[0]}-{page_numbers[-1]}")

        if not parts:
            return "Nguồn trích dẫn"

        return ", ".join(parts)

    def _build_context_text(
        self,
        evidence_items: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        query: str,
    ) -> str:
        citation_by_evidence = {}

        for citation in citations:
            for evidence_id in citation.get("evidence_ids", []) or []:
                citation_by_evidence.setdefault(evidence_id, [])
                citation_by_evidence[evidence_id].append(citation)

        parts = []

        if query:
            parts.append(f"Câu hỏi / truy vấn: {normalize_pdf_text(query)}")

        total_chars = 0

        for index, evidence in enumerate(evidence_items, start=1):
            evidence_id = evidence.get("evidence_id", "")
            evidence_citations = citation_by_evidence.get(evidence_id, [])

            marker_text = ""
            if evidence_citations:
                marker_text = " ".join(
                    [
                        citation.get("citation_marker", "")
                        for citation in evidence_citations
                        if citation.get("citation_marker")
                    ]
                )

            header_parts = [f"[Evidence {index}]"]

            if evidence.get("source_type"):
                header_parts.append(f"type={evidence.get('source_type')}")

            if evidence.get("section_title"):
                header_parts.append(evidence.get("section_title"))

            page_numbers = evidence.get("page_numbers", []) or []

            if page_numbers:
                if len(page_numbers) == 1:
                    header_parts.append(f"trang {page_numbers[0]}")
                else:
                    header_parts.append(f"trang {page_numbers[0]}-{page_numbers[-1]}")

            if marker_text:
                header_parts.append(marker_text)

            text = normalize_pdf_text(evidence.get("quote") or evidence.get("text", ""))

            block = "\n".join(
                [
                    " | ".join(header_parts),
                    text,
                ]
            )

            if total_chars + len(block) > self.config.max_context_chars:
                break

            parts.append(block)
            total_chars += len(block)

        return normalize_pdf_text("\n\n".join(parts))

    def _find_page_context(
        self,
        quote: str,
        page_numbers: List[int],
        page_text_map: Dict[int, str],
    ) -> Tuple[str, str]:
        quote_norm = normalize_text_for_match(quote)

        if not quote_norm or not page_numbers:
            return "", ""

        for page_number in page_numbers:
            page_text = page_text_map.get(page_number, "")

            if not page_text:
                continue

            page_norm = normalize_text_for_match(page_text)
            index = page_norm.find(quote_norm[: min(len(quote_norm), 120)])

            if index < 0:
                continue

            raw_index = min(index, len(page_text))
            before = normalize_pdf_text(page_text[max(0, raw_index - 180):raw_index])
            after = normalize_pdf_text(page_text[raw_index + len(quote):raw_index + len(quote) + 180])

            return before, after

        return "", ""

    def _infer_evidence_type(
        self,
        candidate: Dict[str, Any],
    ) -> str:
        candidate_source = candidate.get("_candidate_source", "")
        source_type = candidate.get("source_type", "")
        item_type = candidate.get("item_type", "")
        chunk_type = candidate.get("chunk_type", "")
        evidence_type = candidate.get("evidence_type", "")
        node_type = candidate.get("node_type", "")

        if evidence_type:
            return evidence_type

        joined = " ".join([candidate_source, source_type, item_type, chunk_type, node_type]).lower()

        if "table" in joined:
            return "table_evidence"

        if "chunk" in joined:
            return "chunk_evidence"

        if "page" in joined:
            return "page_evidence"

        if "graph" in joined or "node" in joined:
            return "graph_evidence"

        if "metadata" in joined:
            return "metadata_evidence"

        if "citation" in joined:
            return "citation_evidence"

        return "aggregated_evidence"

    def _valid_evidence(
        self,
        evidence: Dict[str, Any],
    ) -> bool:
        text = normalize_pdf_text(evidence.get("text") or evidence.get("quote") or "")

        if not text:
            return False

        if len(text) < self.config.min_text_chars:
            word_count = len(re.findall(r"\w+", text))

            if word_count < 3:
                return False

        return True

    def _deduplicate_evidence(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result_by_key = {}

        for evidence in evidence_items:
            key = (
                evidence.get("evidence_id", ""),
                evidence.get("source_id", ""),
                evidence.get("chunk_id", ""),
                evidence.get("table_id", ""),
                evidence.get("node_id", ""),
                normalize_text_for_match(evidence.get("text", ""))[:700],
                tuple(evidence.get("page_numbers", []) or []),
            )

            if key not in result_by_key:
                result_by_key[key] = evidence
            else:
                existing = result_by_key[key]

                if evidence.get("evidence_score", 0.0) > existing.get("evidence_score", 0.0):
                    merged = {
                        **existing,
                        **evidence,
                    }
                    merged["evidence_score"] = max(existing.get("evidence_score", 0.0), evidence.get("evidence_score", 0.0))
                    result_by_key[key] = merged

        return list(result_by_key.values())

    def _deduplicate_citations(
        self,
        citations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for citation in citations:
            key = (
                citation.get("citation_id", ""),
                citation.get("citation_marker", ""),
                citation.get("evidence_id", ""),
                normalize_text_for_match(citation.get("quote", ""))[:300],
                tuple(citation.get("page_numbers", []) or []),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(citation)

        return result

    def _sort_evidence(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return sorted(
            evidence_items,
            key=lambda item: (
                -self._safe_float(item.get("evidence_score"), default=0.0),
                min(item.get("page_numbers", []) or [999999]),
                item.get("source_type", ""),
                item.get("source_id", ""),
            ),
        )

    def _group_evidence_by_page(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for evidence in evidence_items:
            for page_number in evidence.get("page_numbers", []) or []:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(evidence)

        return grouped

    def _group_evidence_by_section(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for evidence in evidence_items:
            section_id = evidence.get("section_id", "") or "no_section"
            grouped.setdefault(section_id, [])
            grouped[section_id].append(evidence)

        return grouped

    def _group_evidence_by_type(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for evidence in evidence_items:
            evidence_type = evidence.get("evidence_type", "unknown")
            grouped.setdefault(evidence_type, [])
            grouped[evidence_type].append(evidence)

        return grouped

    def _group_evidence_by_source_type(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for evidence in evidence_items:
            source_type = evidence.get("source_type", "unknown")
            grouped.setdefault(source_type, [])
            grouped[source_type].append(evidence)

        return grouped

    def _build_citation_markers(
        self,
        citations: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        markers = {}

        for citation in citations:
            marker = citation.get("citation_marker", "")

            if not marker:
                continue

            markers[marker] = {
                "citation_id": citation.get("citation_id", ""),
                "evidence_id": citation.get("evidence_id", ""),
                "evidence_ids": citation.get("evidence_ids", []),
                "citation_text": citation.get("citation_text", ""),
                "page_numbers": citation.get("page_numbers", []),
                "section_id": citation.get("section_id", ""),
                "section_title": citation.get("section_title", ""),
                "quote": citation.get("quote", ""),
            }

        return markers

    def _build_summary(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        aggregated_evidence: List[Dict[str, Any]],
        supporting_evidence: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        by_type = {}
        by_source = {}
        by_page = {}

        total_chars = 0
        total_words = 0
        scores = []

        for evidence in aggregated_evidence:
            evidence_type = evidence.get("evidence_type", "unknown")
            source_type = evidence.get("source_type", "unknown")

            by_type[evidence_type] = by_type.get(evidence_type, 0) + 1
            by_source[source_type] = by_source.get(source_type, 0) + 1

            text = normalize_pdf_text(evidence.get("text", ""))
            total_chars += len(text)
            total_words += len(re.findall(r"\S+", text))
            scores.append(self._safe_float(evidence.get("evidence_score"), default=0.0))

            for page_number in evidence.get("page_numbers", []) or []:
                page_key = str(page_number)
                by_page[page_key] = by_page.get(page_key, 0) + 1

        return {
            "has_evidence": len(aggregated_evidence) > 0,
            "query": query,
            "candidate_count": len(candidates),
            "aggregated_evidence_count": len(aggregated_evidence),
            "supporting_evidence_count": len(supporting_evidence),
            "citation_count": len(citations),
            "page_count": len(page_raws),
            "page_count_with_evidence": len(by_page),
            "total_chars": total_chars,
            "total_words": total_words,
            "avg_chars_per_evidence": round(total_chars / max(len(aggregated_evidence), 1), 2),
            "avg_words_per_evidence": round(total_words / max(len(aggregated_evidence), 1), 2),
            "avg_evidence_score": round(sum(scores) / max(len(scores), 1), 6),
            "max_evidence_score": round(max(scores), 6) if scores else 0.0,
            "min_evidence_score": round(min(scores), 6) if scores else 0.0,
            "by_evidence_type": by_type,
            "by_source_type": by_source,
            "by_page": by_page,
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        aggregation_result: Dict[str, Any],
    ) -> None:
        evidence_by_page = aggregation_result.get("evidence_by_page", {}) or {}

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)
            page_evidence = evidence_by_page.get(page_key, [])

            page_raw.metadata.setdefault("evidence_aggregator", {})
            page_raw.metadata["evidence_aggregator"] = {
                "processor": "EvidenceAggregator",
                "evidence_on_page": page_evidence,
                "evidence_count_on_page": len(page_evidence),
                "supporting_evidence_count_on_page": sum(
                    1 for item in page_evidence
                    if item.get("is_supporting_evidence")
                ),
            }

    def _collect_evidence(
        self,
        evidence_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        for source in [evidence_result, knowledge_result]:
            for key in ["evidence", "evidence_items", "retrieved_evidence", "supporting_evidence", "aggregated_evidence"]:
                values = source.get(key, []) or []

                if isinstance(values, list):
                    items.extend([self._to_dict(item) for item in values])

            for sub_key in ["evidence_result", "evidence_collection", "evidence_aggregation_result"]:
                sub = source.get(sub_key, {}) or {}

                if not isinstance(sub, dict):
                    continue

                for key in ["evidence", "evidence_items", "retrieved_evidence", "supporting_evidence", "aggregated_evidence"]:
                    values = sub.get(key, []) or []

                    if isinstance(values, list):
                        items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_raw_items(items, ["evidence_id", "content_hash"])

    def _collect_chunks(
        self,
        chunk_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        for source in [chunk_result, knowledge_result]:
            for key in ["chunks", "parent_chunks", "child_chunks"]:
                values = source.get(key, []) or []

                if isinstance(values, list):
                    items.extend([self._to_dict(item) for item in values])

            for sub_key in ["chunk_result", "chunk_collection", "parent_child_chunk_result"]:
                sub = source.get(sub_key, {}) or {}

                if not isinstance(sub, dict):
                    continue

                for key in ["chunks", "parent_chunks", "child_chunks"]:
                    values = sub.get(key, []) or []

                    if isinstance(values, list):
                        items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_raw_items(items, ["chunk_id", "content_hash"])

    def _collect_table_chunks(
        self,
        table_chunk_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

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
                    items.extend([self._to_dict(item) for item in values])

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
                        items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_raw_items(items, ["chunk_id", "content_hash"])

    def _collect_graph_nodes(
        self,
        graph_index_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        for source in [graph_index_result, knowledge_result]:
            graph_index = source.get("graph_index", {}) or source.get("knowledge_graph", {}) or source.get("context_graph", {}) or {}

            node_store = graph_index.get("node_store", {}) or source.get("node_store", {}) or {}

            if isinstance(node_store, dict):
                for node_id, node in node_store.items():
                    node = self._to_dict(node)
                    node.setdefault("node_id", node_id)
                    items.append(node)

            nodes = graph_index.get("nodes", []) or source.get("nodes", []) or []

            if isinstance(nodes, list):
                items.extend([self._to_dict(item) for item in nodes])

        return self._deduplicate_raw_items(items, ["node_id", "source_id"])

    def _collect_metadata_items(
        self,
        metadata_index_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        metadata_index = metadata_index_result.get("metadata_index", metadata_index_result) or {}
        metadata_store = metadata_index.get("metadata_store", {}) or metadata_index_result.get("metadata_store", {}) or {}

        if not isinstance(metadata_store, dict):
            return []

        items = []

        for item_id, item in metadata_store.items():
            item = self._to_dict(item)
            item.setdefault("item_id", item_id)
            items.append(item)

        return items

    def _collect_citations(
        self,
        citation_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        citations = []

        for key in ["citations", "citation_items", "verified_citations", "questionable_citations"]:
            values = citation_result.get(key, []) or []

            if isinstance(values, list):
                citations.extend([self._to_dict(item) for item in values])

        for sub_key in ["citation_result", "citation_collection", "evidence_collection"]:
            sub = citation_result.get(sub_key, {}) or {}

            if not isinstance(sub, dict):
                continue

            values = sub.get("citations", []) or sub.get("citation_items", []) or []

            if isinstance(values, list):
                citations.extend([self._to_dict(item) for item in values])

        return self._deduplicate_raw_items(citations, ["citation_id", "citation_marker"])

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
                value = item.get(key_name)

                if not value and isinstance(item.get("metadata"), dict):
                    value = item.get("metadata", {}).get(key_name)

                if value:
                    key = str(value)
                    break

            if not key:
                key = self._stable_hash(
                    {
                        "text": item.get("text") or item.get("text_preview") or item.get("quote") or item.get("label") or "",
                        "pages": self._resolve_page_numbers(item),
                    }
                )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

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

    def _make_quote(
        self,
        text: str,
    ) -> str:
        text = normalize_pdf_text(text)

        if len(text) <= self.config.quote_chars:
            return text

        cut = text[: self.config.quote_chars]
        break_point = max(
            cut.rfind(". "),
            cut.rfind("; "),
            cut.rfind("\n"),
            cut.rfind(" "),
        )

        if break_point > self.config.quote_chars * 0.60:
            cut = cut[:break_point]

        return normalize_pdf_text(cut) + "..."

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

    def _stable_hash(
        self,
        value: Any,
    ) -> str:
        try:
            text = json.dumps(
                json_safe(value),
                ensure_ascii=False,
                sort_keys=True,
            )
        except Exception:
            text = str(value)

        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _stable_id(
        self,
        value: Any,
        prefix: str = "id",
    ) -> str:
        return f"{prefix}_{self._stable_hash(value)[:16]}"

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
        aggregation_result: Dict[str, Any],
        output_path: str,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                json_safe(aggregation_result),
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


def aggregate_evidence(
    query: str = "",
    retrieved_items: Optional[List[Dict[str, Any]]] = None,
    expanded_context_result: Optional[Dict[str, Any]] = None,
    page_raws: Optional[List[PageRaw]] = None,
    evidence_result: Optional[Dict[str, Any]] = None,
    citation_result: Optional[Dict[str, Any]] = None,
    chunk_result: Optional[Dict[str, Any]] = None,
    table_chunk_result: Optional[Dict[str, Any]] = None,
    graph_index_result: Optional[Dict[str, Any]] = None,
    metadata_index_result: Optional[Dict[str, Any]] = None,
    bm25_index_result: Optional[Dict[str, Any]] = None,
    vector_index_result: Optional[Dict[str, Any]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    aggregator = EvidenceAggregator()
    return aggregator.process(
        query=query,
        retrieved_items=retrieved_items,
        expanded_context_result=expanded_context_result,
        page_raws=page_raws,
        evidence_result=evidence_result,
        citation_result=citation_result,
        chunk_result=chunk_result,
        table_chunk_result=table_chunk_result,
        graph_index_result=graph_index_result,
        metadata_index_result=metadata_index_result,
        bm25_index_result=bm25_index_result,
        vector_index_result=vector_index_result,
        knowledge_result=knowledge_result,
    )
