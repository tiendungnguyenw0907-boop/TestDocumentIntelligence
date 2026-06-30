"""
citation_verifier.py

Production V1 - Colab Ready

Purpose
-------
Verify citations used in RAG answers against page text, evidence, chunks,
tables, and indexes.

Used by:
- RAGPipeline
- LLMReasoner
- PromptBuilder
- EvidenceAggregator
- CitationBuilder

Input
-----
- page_raws
- answer_text
- citation_result
- evidence_result
- chunk_result
- table_understanding_result
- document_structure_result
- bm25_index_result
- vector_index_result
- graph_index_result
- metadata_index_result

Output
------
Dictionary with:
- verified_citations
- failed_citations
- questionable_citations
- answer_citation_markers
- missing_markers
- citation_verification_summary
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
class CitationVerifierConfig:
    verify_quote_in_page: bool = True
    verify_page_exists: bool = True
    verify_evidence_exists: bool = True
    verify_chunk_exists: bool = True
    verify_table_exists: bool = True
    verify_bbox: bool = True
    verify_answer_markers: bool = True

    attach_to_pages: bool = True

    strict_mode: bool = False
    allow_page_only_citation: bool = True
    allow_fuzzy_quote_match: bool = True

    exact_match_score: float = 0.95
    fuzzy_match_min_score: float = 0.55
    page_reference_score: float = 0.55
    source_id_match_score: float = 0.75

    min_verified_score: float = 0.65
    min_questionable_score: float = 0.45

    max_quote_chars_for_match: int = 800
    min_quote_chars_for_exact: int = 20
    snippet_chars: int = 280
    max_debug_text_chars: int = 300

    include_debug: bool = True


class CitationVerifier:
    def __init__(
        self,
        config: Optional[CitationVerifierConfig] = None,
    ):
        self.config = config or CitationVerifierConfig()

    def process(
        self,
        page_raws: Optional[List[PageRaw]] = None,
        answer_text: str = "",
        citation_result: Optional[Dict[str, Any]] = None,
        evidence_result: Optional[Dict[str, Any]] = None,
        chunk_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        document_structure_result: Optional[Dict[str, Any]] = None,
        bm25_index_result: Optional[Dict[str, Any]] = None,
        vector_index_result: Optional[Dict[str, Any]] = None,
        graph_index_result: Optional[Dict[str, Any]] = None,
        metadata_index_result: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        page_raws = sorted(
            page_raws or [],
            key=lambda item: item.page_number,
        )

        citation_result = citation_result or {}
        evidence_result = evidence_result or {}
        chunk_result = chunk_result or {}
        table_understanding_result = table_understanding_result or {}
        document_structure_result = document_structure_result or {}
        bm25_index_result = bm25_index_result or {}
        vector_index_result = vector_index_result or {}
        graph_index_result = graph_index_result or {}
        metadata_index_result = metadata_index_result or {}

        citations = self._collect_citations(
            citation_result=citation_result,
            evidence_result=evidence_result,
            kwargs=kwargs,
        )

        citations = self._deduplicate_citations(citations)

        page_text_map = self._build_page_text_map(page_raws)
        page_size_map = self._build_page_size_map(page_raws)

        evidence_index = self._collect_evidence_index(evidence_result=evidence_result)
        chunk_index = self._collect_chunk_index(
            chunk_result=chunk_result,
            bm25_index_result=bm25_index_result,
            vector_index_result=vector_index_result,
            metadata_index_result=metadata_index_result,
        )
        table_index = self._collect_table_index(
            table_understanding_result=table_understanding_result,
            metadata_index_result=metadata_index_result,
        )
        graph_node_index = self._collect_graph_node_index(graph_index_result=graph_index_result)

        verified_citations = []
        failed_citations = []
        questionable_citations = []

        for citation in citations:
            verification = self._verify_single_citation(
                citation=citation,
                page_text_map=page_text_map,
                page_size_map=page_size_map,
                evidence_index=evidence_index,
                chunk_index=chunk_index,
                table_index=table_index,
                graph_node_index=graph_node_index,
            )

            verified_citation = {
                **citation,
                "verified": verification["verified"],
                "verification_status": verification["verification_status"],
                "verification_score": verification["verification_score"],
                "verification_details": verification,
            }

            if verification["verified"]:
                verified_citations.append(verified_citation)
            elif verification["questionable"]:
                questionable_citations.append(verified_citation)
            else:
                failed_citations.append(verified_citation)

        answer_marker_result = {}

        if self.config.verify_answer_markers:
            answer_marker_result = self._verify_answer_markers(
                answer_text=answer_text,
                citations=verified_citations + questionable_citations + failed_citations,
            )

        result = {
            "processor": "CitationVerifier",
            "schema_version": "citation_verifier_v1",
            "verified_citations": verified_citations,
            "questionable_citations": questionable_citations,
            "failed_citations": failed_citations,
            "all_citations": verified_citations + questionable_citations + failed_citations,
            "citations_by_page": self._group_citations_by_page(
                verified_citations + questionable_citations + failed_citations
            ),
            "citations_by_status": self._group_citations_by_status(
                verified_citations + questionable_citations + failed_citations
            ),
            "answer_citation_markers": answer_marker_result.get("answer_citation_markers", []),
            "missing_markers": answer_marker_result.get("missing_markers", []),
            "unused_citation_markers": answer_marker_result.get("unused_citation_markers", []),
            "answer_marker_verification": answer_marker_result,
            "citation_verification_summary": self._build_summary(
                citations=citations,
                verified_citations=verified_citations,
                questionable_citations=questionable_citations,
                failed_citations=failed_citations,
                answer_marker_result=answer_marker_result,
                page_raws=page_raws,
                evidence_index=evidence_index,
                chunk_index=chunk_index,
                table_index=table_index,
            ),
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                verification_result=result,
            )

        return json_safe(result)

    def _verify_single_citation(
        self,
        citation: Dict[str, Any],
        page_text_map: Dict[int, str],
        page_size_map: Dict[int, Tuple[float, float]],
        evidence_index: Dict[str, Dict[str, Any]],
        chunk_index: Dict[str, Dict[str, Any]],
        table_index: Dict[str, Dict[str, Any]],
        graph_node_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        checks = []

        page_check = self._check_page_reference(
            citation=citation,
            page_text_map=page_text_map,
        )
        checks.append(page_check)

        if self.config.verify_quote_in_page:
            quote_check = self._check_quote_in_page(
                citation=citation,
                page_text_map=page_text_map,
            )
            checks.append(quote_check)
        else:
            quote_check = {
                "check": "quote_in_page",
                "status": "skipped",
                "score": 0.0,
                "passed": False,
            }

        if self.config.verify_evidence_exists:
            evidence_check = self._check_evidence_reference(
                citation=citation,
                evidence_index=evidence_index,
            )
            checks.append(evidence_check)

        if self.config.verify_chunk_exists:
            chunk_check = self._check_chunk_reference(
                citation=citation,
                chunk_index=chunk_index,
            )
            checks.append(chunk_check)

        if self.config.verify_table_exists:
            table_check = self._check_table_reference(
                citation=citation,
                table_index=table_index,
            )
            checks.append(table_check)

        if self.config.verify_bbox:
            bbox_check = self._check_bbox(
                citation=citation,
                page_size_map=page_size_map,
            )
            checks.append(bbox_check)

        source_check = self._check_source_consistency(
            citation=citation,
            evidence_index=evidence_index,
            chunk_index=chunk_index,
            table_index=table_index,
            graph_node_index=graph_node_index,
        )
        checks.append(source_check)

        score = self._combine_check_scores(checks)

        status = self._infer_verification_status(
            citation=citation,
            checks=checks,
            score=score,
        )

        verified = status in [
            "verified_exact_quote",
            "verified_fuzzy_quote",
            "verified_source_reference",
            "verified_page_reference",
        ]

        questionable = status in [
            "questionable_quote_not_found",
            "questionable_partial_reference",
            "questionable_weak_match",
        ]

        return {
            "citation_id": citation.get("citation_id", ""),
            "verified": verified,
            "questionable": questionable,
            "verification_status": status,
            "verification_score": round(score, 4),
            "checks": checks,
            "best_match": quote_check.get("best_match", {}),
            "page_numbers": self._resolve_page_numbers(citation),
            "quote_preview": self._preview(citation.get("quote", ""), self.config.max_debug_text_chars),
            "citation_marker": citation.get("citation_marker", ""),
            "citation_text": citation.get("citation_text", ""),
        }

    def _check_page_reference(
        self,
        citation: Dict[str, Any],
        page_text_map: Dict[int, str],
    ) -> Dict[str, Any]:
        page_numbers = self._resolve_page_numbers(citation)

        if not page_numbers:
            return {
                "check": "page_reference",
                "status": "no_page_numbers",
                "passed": False,
                "score": 0.0,
                "message": "Citation has no page reference.",
            }

        existing_pages = []
        missing_pages = []

        for page_number in page_numbers:
            if page_number in page_text_map:
                existing_pages.append(page_number)
            else:
                missing_pages.append(page_number)

        if not missing_pages:
            return {
                "check": "page_reference",
                "status": "page_exists",
                "passed": True,
                "score": self.config.page_reference_score,
                "existing_pages": existing_pages,
                "missing_pages": [],
            }

        if existing_pages:
            return {
                "check": "page_reference",
                "status": "partial_pages_exist",
                "passed": True,
                "score": 0.40,
                "existing_pages": existing_pages,
                "missing_pages": missing_pages,
            }

        return {
            "check": "page_reference",
            "status": "missing_pages",
            "passed": False,
            "score": 0.0,
            "existing_pages": [],
            "missing_pages": missing_pages,
        }

    def _check_quote_in_page(
        self,
        citation: Dict[str, Any],
        page_text_map: Dict[int, str],
    ) -> Dict[str, Any]:
        quote = normalize_pdf_text(citation.get("quote", ""))

        if not quote:
            quote = normalize_pdf_text(citation.get("text", ""))

        if not quote:
            return {
                "check": "quote_in_page",
                "status": "no_quote",
                "passed": False,
                "score": 0.0,
                "message": "Citation has no quote/text to verify.",
            }

        quote = quote[: self.config.max_quote_chars_for_match]
        page_numbers = self._resolve_page_numbers(citation)

        if not page_numbers:
            return {
                "check": "quote_in_page",
                "status": "no_page_for_quote",
                "passed": False,
                "score": 0.0,
                "quote_preview": self._preview(quote, self.config.max_debug_text_chars),
            }

        quote_norm = normalize_text_for_match(quote)

        if len(quote_norm) < self.config.min_quote_chars_for_exact:
            return {
                "check": "quote_in_page",
                "status": "quote_too_short",
                "passed": False,
                "score": 0.20,
                "quote_preview": self._preview(quote, self.config.max_debug_text_chars),
            }

        best_match = {
            "page_number": None,
            "match_type": "none",
            "score": 0.0,
            "snippet": "",
        }

        for page_number in page_numbers:
            page_text = normalize_pdf_text(page_text_map.get(page_number, ""))

            if not page_text:
                continue

            page_norm = normalize_text_for_match(page_text)

            if quote_norm and quote_norm in page_norm:
                snippet = self._snippet_around_match(
                    page_text=page_text,
                    query_text=quote,
                    max_chars=self.config.snippet_chars,
                )

                return {
                    "check": "quote_in_page",
                    "status": "exact_quote_found",
                    "passed": True,
                    "score": self.config.exact_match_score,
                    "best_match": {
                        "page_number": page_number,
                        "match_type": "exact",
                        "score": self.config.exact_match_score,
                        "snippet": snippet,
                    },
                }

            fuzzy_score = self._fuzzy_overlap_score(quote_norm, page_norm)

            if fuzzy_score > best_match["score"]:
                best_match = {
                    "page_number": page_number,
                    "match_type": "token_overlap",
                    "score": round(fuzzy_score, 4),
                    "snippet": self._best_fuzzy_snippet(
                        page_text=page_text,
                        quote_text=quote,
                        max_chars=self.config.snippet_chars,
                    ),
                }

        if self.config.allow_fuzzy_quote_match and best_match["score"] >= self.config.fuzzy_match_min_score:
            return {
                "check": "quote_in_page",
                "status": "fuzzy_quote_found",
                "passed": True,
                "score": round(0.60 + min(best_match["score"], 0.35), 4),
                "best_match": best_match,
            }

        return {
            "check": "quote_in_page",
            "status": "quote_not_found",
            "passed": False,
            "score": max(0.20, best_match["score"] * 0.60),
            "best_match": best_match,
            "quote_preview": self._preview(quote, self.config.max_debug_text_chars),
        }

    def _check_evidence_reference(
        self,
        citation: Dict[str, Any],
        evidence_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        evidence_ids = citation.get("evidence_ids", []) or []

        if citation.get("evidence_id"):
            evidence_ids.append(citation.get("evidence_id"))

        evidence_ids = [
            item for item in dict.fromkeys(evidence_ids)
            if item
        ]

        if not evidence_ids:
            return {
                "check": "evidence_reference",
                "status": "no_evidence_reference",
                "passed": False,
                "score": 0.0,
            }

        found = []
        missing = []

        for evidence_id in evidence_ids:
            if evidence_id in evidence_index:
                found.append(evidence_id)
            else:
                missing.append(evidence_id)

        if found and not missing:
            return {
                "check": "evidence_reference",
                "status": "evidence_found",
                "passed": True,
                "score": self.config.source_id_match_score,
                "found_evidence_ids": found,
                "missing_evidence_ids": [],
            }

        if found:
            return {
                "check": "evidence_reference",
                "status": "partial_evidence_found",
                "passed": True,
                "score": 0.50,
                "found_evidence_ids": found,
                "missing_evidence_ids": missing,
            }

        return {
            "check": "evidence_reference",
            "status": "evidence_not_found",
            "passed": False,
            "score": 0.0,
            "found_evidence_ids": [],
            "missing_evidence_ids": missing,
        }

    def _check_chunk_reference(
        self,
        citation: Dict[str, Any],
        chunk_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        chunk_ids = []

        for key in [
            "chunk_id",
            "source_chunk_id",
            "parent_chunk_id",
        ]:
            if citation.get(key):
                chunk_ids.append(citation.get(key))

        metadata = citation.get("metadata", {}) or {}

        if isinstance(metadata, dict):
            for key in ["chunk_id", "source_chunk_id", "parent_chunk_id"]:
                if metadata.get(key):
                    chunk_ids.append(metadata.get(key))

        chunk_ids = [
            item for item in dict.fromkeys(chunk_ids)
            if item
        ]

        if not chunk_ids:
            return {
                "check": "chunk_reference",
                "status": "no_chunk_reference",
                "passed": False,
                "score": 0.0,
            }

        found = []
        missing = []

        for chunk_id in chunk_ids:
            if chunk_id in chunk_index:
                found.append(chunk_id)
            else:
                missing.append(chunk_id)

        if found and not missing:
            return {
                "check": "chunk_reference",
                "status": "chunk_found",
                "passed": True,
                "score": self.config.source_id_match_score,
                "found_chunk_ids": found,
                "missing_chunk_ids": [],
            }

        if found:
            return {
                "check": "chunk_reference",
                "status": "partial_chunk_found",
                "passed": True,
                "score": 0.50,
                "found_chunk_ids": found,
                "missing_chunk_ids": missing,
            }

        return {
            "check": "chunk_reference",
            "status": "chunk_not_found",
            "passed": False,
            "score": 0.0,
            "found_chunk_ids": [],
            "missing_chunk_ids": missing,
        }

    def _check_table_reference(
        self,
        citation: Dict[str, Any],
        table_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        table_ids = []

        for key in [
            "table_id",
            "table_grid_id",
            "table_structure_id",
            "table_semantic_id",
            "table_boundary_id",
            "multi_page_table_id",
        ]:
            if citation.get(key):
                table_ids.append(citation.get(key))

        metadata = citation.get("metadata", {}) or {}

        if isinstance(metadata, dict):
            for key in [
                "table_id",
                "table_grid_id",
                "table_structure_id",
                "table_semantic_id",
                "table_boundary_id",
                "multi_page_table_id",
            ]:
                if metadata.get(key):
                    table_ids.append(metadata.get(key))

        table_ids = [
            item for item in dict.fromkeys(table_ids)
            if item
        ]

        if not table_ids:
            return {
                "check": "table_reference",
                "status": "no_table_reference",
                "passed": False,
                "score": 0.0,
            }

        found = []
        missing = []

        for table_id in table_ids:
            if table_id in table_index:
                found.append(table_id)
            else:
                missing.append(table_id)

        if found and not missing:
            return {
                "check": "table_reference",
                "status": "table_found",
                "passed": True,
                "score": self.config.source_id_match_score,
                "found_table_ids": found,
                "missing_table_ids": [],
            }

        if found:
            return {
                "check": "table_reference",
                "status": "partial_table_found",
                "passed": True,
                "score": 0.50,
                "found_table_ids": found,
                "missing_table_ids": missing,
            }

        return {
            "check": "table_reference",
            "status": "table_not_found",
            "passed": False,
            "score": 0.0,
            "found_table_ids": [],
            "missing_table_ids": missing,
        }

    def _check_bbox(
        self,
        citation: Dict[str, Any],
        page_size_map: Dict[int, Tuple[float, float]],
    ) -> Dict[str, Any]:
        bbox = citation.get("bbox", []) or []

        if not bbox:
            return {
                "check": "bbox",
                "status": "no_bbox",
                "passed": False,
                "score": 0.0,
            }

        if not isinstance(bbox, list) or len(bbox) != 4:
            return {
                "check": "bbox",
                "status": "invalid_bbox_format",
                "passed": False,
                "score": 0.0,
                "bbox": bbox,
            }

        try:
            x0, y0, x1, y1 = [float(value) for value in bbox]
        except Exception:
            return {
                "check": "bbox",
                "status": "invalid_bbox_values",
                "passed": False,
                "score": 0.0,
                "bbox": bbox,
            }

        if x1 <= x0 or y1 <= y0:
            return {
                "check": "bbox",
                "status": "invalid_bbox_geometry",
                "passed": False,
                "score": 0.0,
                "bbox": bbox,
            }

        page_numbers = self._resolve_page_numbers(citation)

        if not page_numbers:
            return {
                "check": "bbox",
                "status": "bbox_without_page",
                "passed": False,
                "score": 0.20,
                "bbox": bbox,
            }

        page_number = page_numbers[0]
        page_size = page_size_map.get(page_number)

        if not page_size:
            return {
                "check": "bbox",
                "status": "page_size_unknown",
                "passed": True,
                "score": 0.35,
                "bbox": bbox,
            }

        width, height = page_size

        tolerance = 10.0

        inside = (
            x0 >= -tolerance
            and y0 >= -tolerance
            and x1 <= width + tolerance
            and y1 <= height + tolerance
        )

        if inside:
            return {
                "check": "bbox",
                "status": "bbox_inside_page",
                "passed": True,
                "score": 0.60,
                "bbox": bbox,
                "page_number": page_number,
                "page_size": [width, height],
            }

        return {
            "check": "bbox",
            "status": "bbox_outside_page",
            "passed": False,
            "score": 0.20,
            "bbox": bbox,
            "page_number": page_number,
            "page_size": [width, height],
        }

    def _check_source_consistency(
        self,
        citation: Dict[str, Any],
        evidence_index: Dict[str, Dict[str, Any]],
        chunk_index: Dict[str, Dict[str, Any]],
        table_index: Dict[str, Dict[str, Any]],
        graph_node_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        citation_pages = set(self._resolve_page_numbers(citation))
        citation_quote = normalize_text_for_match(citation.get("quote", ""))

        candidate_sources = []

        for evidence_id in citation.get("evidence_ids", []) or []:
            if evidence_id in evidence_index:
                candidate_sources.append(("evidence", evidence_id, evidence_index[evidence_id]))

        if citation.get("evidence_id") in evidence_index:
            candidate_sources.append(("evidence", citation.get("evidence_id"), evidence_index[citation.get("evidence_id")]))

        for key in ["chunk_id", "source_chunk_id"]:
            chunk_id = citation.get(key)
            if chunk_id and chunk_id in chunk_index:
                candidate_sources.append(("chunk", chunk_id, chunk_index[chunk_id]))

        for key in [
            "table_id",
            "table_grid_id",
            "table_structure_id",
            "table_semantic_id",
            "table_boundary_id",
            "multi_page_table_id",
        ]:
            table_id = citation.get(key)
            if table_id and table_id in table_index:
                candidate_sources.append(("table", table_id, table_index[table_id]))

        metadata = citation.get("metadata", {}) or {}

        if isinstance(metadata, dict):
            node_id = metadata.get("node_id", "")
            if node_id and node_id in graph_node_index:
                candidate_sources.append(("graph_node", node_id, graph_node_index[node_id]))

        if not candidate_sources:
            return {
                "check": "source_consistency",
                "status": "no_source_to_compare",
                "passed": False,
                "score": 0.0,
            }

        best_score = 0.0
        best_source = {}

        for source_type, source_id, source in candidate_sources:
            source_pages = set(self._resolve_page_numbers(source))
            page_overlap = 0.0

            if citation_pages and source_pages:
                page_overlap = len(citation_pages.intersection(source_pages)) / max(len(citation_pages.union(source_pages)), 1)
            elif not citation_pages:
                page_overlap = 0.20

            source_text = normalize_text_for_match(
                source.get("text")
                or source.get("quote")
                or source.get("text_preview")
                or source.get("label")
                or ""
            )

            text_overlap = 0.0

            if citation_quote and source_text:
                text_overlap = self._fuzzy_overlap_score(citation_quote, source_text)

            score = max(page_overlap * 0.70, text_overlap)

            if score > best_score:
                best_score = score
                best_source = {
                    "source_type": source_type,
                    "source_id": source_id,
                    "page_overlap": round(page_overlap, 4),
                    "text_overlap": round(text_overlap, 4),
                }

        if best_score >= 0.50:
            return {
                "check": "source_consistency",
                "status": "source_consistent",
                "passed": True,
                "score": round(max(0.55, best_score), 4),
                "best_source": best_source,
            }

        return {
            "check": "source_consistency",
            "status": "source_weakly_consistent",
            "passed": False,
            "score": round(best_score, 4),
            "best_source": best_source,
        }

    def _combine_check_scores(
        self,
        checks: List[Dict[str, Any]],
    ) -> float:
        if not checks:
            return 0.0

        weighted_scores = []
        total_weight = 0.0

        weights = {
            "quote_in_page": 3.0,
            "page_reference": 1.5,
            "evidence_reference": 1.5,
            "chunk_reference": 1.2,
            "table_reference": 1.2,
            "source_consistency": 1.7,
            "bbox": 0.5,
        }

        for check in checks:
            check_name = check.get("check", "")
            status = check.get("status", "")

            if status.startswith("no_") and check_name in [
                "evidence_reference",
                "chunk_reference",
                "table_reference",
                "bbox",
            ]:
                continue

            weight = weights.get(check_name, 1.0)
            score = float(check.get("score", 0.0) or 0.0)

            weighted_scores.append(score * weight)
            total_weight += weight

        if total_weight <= 0:
            return 0.0

        return sum(weighted_scores) / total_weight

    def _infer_verification_status(
        self,
        citation: Dict[str, Any],
        checks: List[Dict[str, Any]],
        score: float,
    ) -> str:
        by_check = {
            check.get("check"): check
            for check in checks
        }

        quote_status = by_check.get("quote_in_page", {}).get("status", "")
        page_status = by_check.get("page_reference", {}).get("status", "")
        source_status = by_check.get("source_consistency", {}).get("status", "")

        has_source_reference = any(
            by_check.get(check_name, {}).get("passed")
            for check_name in [
                "evidence_reference",
                "chunk_reference",
                "table_reference",
            ]
        )

        if quote_status == "exact_quote_found":
            return "verified_exact_quote"

        if quote_status == "fuzzy_quote_found":
            return "verified_fuzzy_quote"

        if has_source_reference and source_status == "source_consistent" and score >= self.config.min_verified_score:
            return "verified_source_reference"

        if (
            self.config.allow_page_only_citation
            and page_status == "page_exists"
            and score >= self.config.min_verified_score
        ):
            return "verified_page_reference"

        if page_status in ["missing_pages", "no_page_numbers"]:
            return "failed_invalid_page_reference"

        if quote_status == "quote_not_found" and score >= self.config.min_questionable_score:
            return "questionable_quote_not_found"

        if has_source_reference and score >= self.config.min_questionable_score:
            return "questionable_partial_reference"

        if score >= self.config.min_questionable_score:
            return "questionable_weak_match"

        return "failed_unverified"

    def _verify_answer_markers(
        self,
        answer_text: str,
        citations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        answer_text = normalize_pdf_text(answer_text)

        answer_markers = self._extract_citation_markers_from_text(answer_text)

        citation_markers = []

        for citation in citations:
            marker = normalize_pdf_text(citation.get("citation_marker", ""))

            if marker:
                citation_markers.append(marker)

        citation_markers = list(dict.fromkeys(citation_markers))

        marker_set = set(citation_markers)
        answer_marker_set = set(answer_markers)

        missing_markers = [
            marker for marker in answer_markers
            if marker not in marker_set
        ]

        unused_citation_markers = [
            marker for marker in citation_markers
            if marker not in answer_marker_set
        ]

        marker_details = []

        citation_by_marker = {
            citation.get("citation_marker", ""): citation
            for citation in citations
            if citation.get("citation_marker", "")
        }

        for marker in answer_markers:
            citation = citation_by_marker.get(marker)

            marker_details.append(
                {
                    "marker": marker,
                    "found_in_citations": citation is not None,
                    "citation_id": citation.get("citation_id", "") if citation else "",
                    "verified": citation.get("verified", False) if citation else False,
                    "verification_status": citation.get("verification_status", "missing_marker") if citation else "missing_marker",
                    "verification_score": citation.get("verification_score", 0.0) if citation else 0.0,
                }
            )

        return {
            "answer_has_markers": len(answer_markers) > 0,
            "answer_citation_markers": answer_markers,
            "citation_markers": citation_markers,
            "missing_markers": missing_markers,
            "unused_citation_markers": unused_citation_markers,
            "marker_details": marker_details,
            "marker_count": len(answer_markers),
            "missing_marker_count": len(missing_markers),
            "unused_marker_count": len(unused_citation_markers),
            "all_answer_markers_resolved": len(missing_markers) == 0,
        }

    def _extract_citation_markers_from_text(
        self,
        text: str,
    ) -> List[str]:
        markers = []

        patterns = [
            r"\[tr\.\s*\d+(?:\s*-\s*\d+)?(?:\.\d+)?\]",
            r"\[p\.\s*\d+(?:\s*-\s*\d+)?(?:\.\d+)?\]",
            r"\[page\s*\d+(?:\s*-\s*\d+)?(?:\.\d+)?\]",
            r"\[\d+(?:\s*-\s*\d+)?(?:\.\d+)?\]",
            r"【[^】]{1,80}】",
        ]

        for pattern in patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                marker = normalize_pdf_text(match)

                if marker and marker not in markers:
                    markers.append(marker)

        return markers

    def _collect_citations(
        self,
        citation_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        kwargs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        citations = []

        sources = [
            citation_result,
            evidence_result,
            kwargs.get("knowledge_result", {}) or {},
            kwargs.get("rag_result", {}) or {},
        ]

        for source in sources:
            if not isinstance(source, dict):
                continue

            for key in [
                "citations",
                "citation_items",
                "verified_citations",
                "failed_citations",
                "questionable_citations",
            ]:
                values = source.get(key, []) or []

                if isinstance(values, list):
                    citations.extend([self._to_dict(item) for item in values])

            for sub_key in [
                "citation_result",
                "evidence_collection",
                "citation_collection",
            ]:
                sub = source.get(sub_key, {}) or {}

                if not isinstance(sub, dict):
                    continue

                for key in [
                    "citations",
                    "citation_items",
                    "verified_citations",
                    "failed_citations",
                    "questionable_citations",
                ]:
                    values = sub.get(key, []) or []

                    if isinstance(values, list):
                        citations.extend([self._to_dict(item) for item in values])

        return citations

    def _collect_evidence_index(
        self,
        evidence_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        items = []

        for key in ["evidence", "evidence_items", "retrieved_evidence", "supporting_evidence"]:
            values = evidence_result.get(key, []) or []

            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        for sub_key in ["evidence_result", "evidence_collection"]:
            sub = evidence_result.get(sub_key, {}) or {}

            if not isinstance(sub, dict):
                continue

            values = sub.get("evidence", []) or sub.get("evidence_items", []) or []

            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        index = {}

        for item in items:
            evidence_id = item.get("evidence_id", "")

            if evidence_id:
                index[evidence_id] = item

        return index

    def _collect_chunk_index(
        self,
        chunk_result: Dict[str, Any],
        bm25_index_result: Dict[str, Any],
        vector_index_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        items = []

        for source in [chunk_result]:
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

        metadata_index = metadata_index_result.get("metadata_index", metadata_index_result) or {}
        metadata_store = metadata_index.get("metadata_store", {}) or metadata_index_result.get("metadata_store", {}) or {}

        if isinstance(metadata_store, dict):
            for item in metadata_store.values():
                item = self._to_dict(item)
                if item.get("chunk_id"):
                    items.append(item)

        index = {}

        for item in items:
            chunk_id = item.get("chunk_id", "")

            if chunk_id:
                index[chunk_id] = item

        return index

    def _collect_table_index(
        self,
        table_understanding_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        items = []

        for key in [
            "table_semantics",
            "table_grids",
            "table_structures",
            "table_boundaries",
            "multi_page_tables",
        ]:
            values = table_understanding_result.get(key, []) or []

            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

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
                    items.extend([self._to_dict(item) for item in values])

        metadata_index = metadata_index_result.get("metadata_index", metadata_index_result) or {}
        metadata_store = metadata_index.get("metadata_store", {}) or metadata_index_result.get("metadata_store", {}) or {}

        if isinstance(metadata_store, dict):
            for item in metadata_store.values():
                item = self._to_dict(item)
                if item.get("table_id"):
                    items.append(item)

        index = {}

        for item in items:
            for table_id in self._table_ids_from_item(item):
                if table_id:
                    index[table_id] = item

        return index

    def _collect_graph_node_index(
        self,
        graph_index_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        graph_index = graph_index_result.get("graph_index", graph_index_result) or {}
        node_store = graph_index.get("node_store", {}) or graph_index_result.get("node_store", {}) or {}

        if not isinstance(node_store, dict):
            return {}

        return {
            str(node_id): self._to_dict(node)
            for node_id, node in node_store.items()
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

    def _build_page_size_map(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[int, Tuple[float, float]]:
        page_size_map = {}

        for page_raw in page_raws:
            try:
                page_size_map[page_raw.page_number] = (
                    float(page_raw.width),
                    float(page_raw.height),
                )
            except Exception:
                continue

        return page_size_map

    def _group_citations_by_page(
        self,
        citations: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for citation in citations:
            for page_number in self._resolve_page_numbers(citation):
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(citation)

        return grouped

    def _group_citations_by_status(
        self,
        citations: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for citation in citations:
            status = citation.get("verification_status", "unknown")
            grouped.setdefault(status, [])
            grouped[status].append(citation)

        return grouped

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        verification_result: Dict[str, Any],
    ) -> None:
        citations_by_page = verification_result.get("citations_by_page", {}) or {}

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)
            page_citations = citations_by_page.get(page_key, [])

            page_raw.metadata.setdefault("citation_verifier", {})
            page_raw.metadata["citation_verifier"] = {
                "processor": "CitationVerifier",
                "citations_on_page": page_citations,
                "citation_count_on_page": len(page_citations),
                "verified_count_on_page": sum(1 for item in page_citations if item.get("verified")),
                "failed_count_on_page": sum(1 for item in page_citations if item.get("verification_status", "").startswith("failed")),
            }

    def _build_summary(
        self,
        citations: List[Dict[str, Any]],
        verified_citations: List[Dict[str, Any]],
        questionable_citations: List[Dict[str, Any]],
        failed_citations: List[Dict[str, Any]],
        answer_marker_result: Dict[str, Any],
        page_raws: List[PageRaw],
        evidence_index: Dict[str, Dict[str, Any]],
        chunk_index: Dict[str, Dict[str, Any]],
        table_index: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        all_verified_output = verified_citations + questionable_citations + failed_citations

        status_counts = {}

        for citation in all_verified_output:
            status = citation.get("verification_status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        scores = [
            self._safe_float(item.get("verification_score"), default=0.0)
            for item in all_verified_output
        ]

        verified_ratio = len(verified_citations) / max(len(all_verified_output), 1)
        questionable_ratio = len(questionable_citations) / max(len(all_verified_output), 1)
        failed_ratio = len(failed_citations) / max(len(all_verified_output), 1)

        return {
            "has_citations": len(citations) > 0,
            "citation_count": len(citations),
            "verified_citation_count": len(verified_citations),
            "questionable_citation_count": len(questionable_citations),
            "failed_citation_count": len(failed_citations),
            "verified_ratio": round(verified_ratio, 4),
            "questionable_ratio": round(questionable_ratio, 4),
            "failed_ratio": round(failed_ratio, 4),
            "avg_verification_score": round(sum(scores) / max(len(scores), 1), 4),
            "min_verification_score": round(min(scores), 4) if scores else 0.0,
            "max_verification_score": round(max(scores), 4) if scores else 0.0,
            "status_counts": status_counts,
            "page_count": len(page_raws),
            "evidence_index_count": len(evidence_index),
            "chunk_index_count": len(chunk_index),
            "table_index_count": len(table_index),
            "answer_marker_summary": {
                "answer_has_markers": answer_marker_result.get("answer_has_markers", False),
                "marker_count": answer_marker_result.get("marker_count", 0),
                "missing_marker_count": answer_marker_result.get("missing_marker_count", 0),
                "unused_marker_count": answer_marker_result.get("unused_marker_count", 0),
                "all_answer_markers_resolved": answer_marker_result.get("all_answer_markers_resolved", True),
            },
            "strict_mode": self.config.strict_mode,
        }

    def _snippet_around_match(
        self,
        page_text: str,
        query_text: str,
        max_chars: int = 300,
    ) -> str:
        page_text = normalize_pdf_text(page_text)
        query_norm = normalize_text_for_match(query_text)
        page_norm = normalize_text_for_match(page_text)

        index = page_norm.find(query_norm[: min(len(query_norm), 160)])

        if index < 0:
            return self._preview(page_text, max_chars)

        start = max(0, index - max_chars // 3)
        end = min(len(page_text), start + max_chars)

        snippet = page_text[start:end]

        if start > 0:
            snippet = "..." + snippet

        if end < len(page_text):
            snippet += "..."

        return normalize_pdf_text(snippet)

    def _best_fuzzy_snippet(
        self,
        page_text: str,
        quote_text: str,
        max_chars: int = 300,
    ) -> str:
        page_text = normalize_pdf_text(page_text)
        quote_tokens = [
            token for token in normalize_text_for_match(quote_text).split()
            if len(token) >= 4
        ]

        if not quote_tokens:
            return self._preview(page_text, max_chars)

        page_norm = normalize_text_for_match(page_text)

        best_index = -1

        for token in quote_tokens[:20]:
            index = page_norm.find(token)

            if index >= 0:
                best_index = index
                break

        if best_index < 0:
            return self._preview(page_text, max_chars)

        start = max(0, best_index - max_chars // 3)
        end = min(len(page_text), start + max_chars)

        snippet = page_text[start:end]

        if start > 0:
            snippet = "..." + snippet

        if end < len(page_text):
            snippet += "..."

        return normalize_pdf_text(snippet)

    def _fuzzy_overlap_score(
        self,
        quote_text: str,
        source_text: str,
    ) -> float:
        quote_tokens = [
            token for token in quote_text.split()
            if len(token) >= 3
        ]

        source_tokens = set(
            token for token in source_text.split()
            if len(token) >= 3
        )

        if not quote_tokens or not source_tokens:
            return 0.0

        matched = sum(
            1 for token in quote_tokens
            if token in source_tokens
        )

        return matched / max(len(quote_tokens), 1)

    def _deduplicate_citations(
        self,
        citations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for citation in citations:
            citation = self._to_dict(citation)

            key = (
                citation.get("citation_id", ""),
                citation.get("citation_marker", ""),
                normalize_text_for_match(citation.get("quote", ""))[:500],
                tuple(self._resolve_page_numbers(citation)),
            )

            if not any(key):
                key = (
                    self._stable_id(citation, "citation"),
                )

            if key in seen:
                continue

            seen.add(key)
            result.append(citation)

        return result

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

        marker_pages = self._pages_from_marker(item.get("citation_marker", ""))

        if marker_pages:
            return marker_pages

        return []

    def _pages_from_marker(
        self,
        marker: str,
    ) -> List[int]:
        marker = normalize_pdf_text(marker)

        if not marker:
            return []

        match = re.search(r"(\d+)\s*-\s*(\d+)", marker)

        if match:
            start = self._safe_int(match.group(1), default=0)
            end = self._safe_int(match.group(2), default=0)

            if start > 0 and end >= start:
                return list(range(start, end + 1))

        match = re.search(r"(\d+)", marker)

        if match:
            page = self._safe_int(match.group(1), default=0)
            if page > 0:
                return [page]

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

    def _preview(
        self,
        text: Any,
        max_chars: int = 300,
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
        verification_result: Dict[str, Any],
        output_path: str,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                json_safe(verification_result),
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


def verify_citations(
    page_raws: Optional[List[PageRaw]] = None,
    answer_text: str = "",
    citation_result: Optional[Dict[str, Any]] = None,
    evidence_result: Optional[Dict[str, Any]] = None,
    chunk_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    document_structure_result: Optional[Dict[str, Any]] = None,
    bm25_index_result: Optional[Dict[str, Any]] = None,
    vector_index_result: Optional[Dict[str, Any]] = None,
    graph_index_result: Optional[Dict[str, Any]] = None,
    metadata_index_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    verifier = CitationVerifier()
    return verifier.process(
        page_raws=page_raws,
        answer_text=answer_text,
        citation_result=citation_result,
        evidence_result=evidence_result,
        chunk_result=chunk_result,
        table_understanding_result=table_understanding_result,
        document_structure_result=document_structure_result,
        bm25_index_result=bm25_index_result,
        vector_index_result=vector_index_result,
        graph_index_result=graph_index_result,
        metadata_index_result=metadata_index_result,
    )


def verify_answer_citations(
    answer_text: str,
    citation_result: Dict[str, Any],
    page_raws: Optional[List[PageRaw]] = None,
    evidence_result: Optional[Dict[str, Any]] = None,
    chunk_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    verifier = CitationVerifier()
    return verifier.process(
        page_raws=page_raws,
        answer_text=answer_text,
        citation_result=citation_result,
        evidence_result=evidence_result,
        chunk_result=chunk_result,
        table_understanding_result=table_understanding_result,
    )
