"""
query_router.py

Production V1 - Colab Ready

Purpose
-------
Route classified query to the right retrieval / RAG flow.

Used by:
- RAGPipeline
- HybridRetriever
- GraphRetriever
- TableRetriever
- PromptBuilder
- LLMReasoner

Input
-----
- query
- query_classification_result
- index results: bm25/vector/graph/metadata/table
- filters / routing options

Output
------
Dictionary with:
- route_name
- route_plan
- retrieval_result
- route_outputs
- routing_summary
"""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import (
    normalize_pdf_text,
    normalize_text_for_match,
    json_safe,
)


@dataclass
class QueryRouterConfig:
    default_route: str = "hybrid"
    allow_classifier_fallback: bool = True

    enable_hybrid_route: bool = True
    enable_bm25_route: bool = True
    enable_vector_route: bool = True
    enable_graph_route: bool = True
    enable_metadata_route: bool = True
    enable_table_route: bool = True

    enable_context_expansion: bool = True
    enable_evidence_aggregation: bool = True
    enable_prompt_building: bool = False
    enable_llm_reasoning: bool = False
    enable_citation_verification: bool = False

    top_k: int = 20
    candidate_pool_size: int = 80

    table_top_k: int = 20
    graph_top_k: int = 20
    metadata_top_k: int = 30

    force_hybrid_for_complex_queries: bool = True
    high_complexity_threshold: str = "high"

    include_route_debug: bool = True
    include_context_text: bool = True


class QueryRouter:
    def __init__(
        self,
        config: Optional[QueryRouterConfig] = None,
    ):
        self.config = config or QueryRouterConfig()

    def process(
        self,
        query: str,
        query_classification_result: Optional[Dict[str, Any]] = None,
        bm25_index_result: Optional[Dict[str, Any]] = None,
        vector_index_result: Optional[Dict[str, Any]] = None,
        graph_index_result: Optional[Dict[str, Any]] = None,
        metadata_index_result: Optional[Dict[str, Any]] = None,
        table_chunk_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        page_raws: Optional[List[Any]] = None,
        evidence_result: Optional[Dict[str, Any]] = None,
        citation_result: Optional[Dict[str, Any]] = None,
        knowledge_result: Optional[Dict[str, Any]] = None,
        filters: Optional[Dict[str, Any]] = None,
        page_numbers: Optional[List[int]] = None,
        section_ids: Optional[List[str]] = None,
        table_ids: Optional[List[str]] = None,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
        route_options: Optional[Dict[str, Any]] = None,
        llm_fn: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        llm_model: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        query = normalize_pdf_text(query)

        bm25_index_result = bm25_index_result or {}
        vector_index_result = vector_index_result or {}
        graph_index_result = graph_index_result or {}
        metadata_index_result = metadata_index_result or {}
        table_chunk_result = table_chunk_result or {}
        table_understanding_result = table_understanding_result or {}
        page_raws = page_raws or []
        evidence_result = evidence_result or {}
        citation_result = citation_result or {}
        knowledge_result = knowledge_result or {}
        filters = filters or {}
        route_options = route_options or {}
        node_types = node_types or []
        edge_types = edge_types or []

        page_numbers = self._normalize_page_numbers(page_numbers or [])
        section_ids = [str(item) for item in (section_ids or []) if item]
        table_ids = [str(item) for item in (table_ids or []) if item]

        if not query_classification_result and self.config.allow_classifier_fallback:
            query_classification_result = self._classify_query(query)

        query_classification_result = query_classification_result or {}

        merged_filters = self._merge_filters(
            explicit_filters=filters,
            classification_result=query_classification_result,
            page_numbers=page_numbers,
            section_ids=section_ids,
            table_ids=table_ids,
        )

        route_plan = self._build_route_plan(
            query=query,
            classification_result=query_classification_result,
            filters=merged_filters,
            page_numbers=page_numbers,
            section_ids=section_ids,
            table_ids=table_ids,
            node_types=node_types,
            edge_types=edge_types,
            route_options=route_options,
        )

        retrieval_result = self._execute_retrieval_route(
            query=query,
            route_plan=route_plan,
            bm25_index_result=bm25_index_result,
            vector_index_result=vector_index_result,
            graph_index_result=graph_index_result,
            metadata_index_result=metadata_index_result,
            table_chunk_result=table_chunk_result,
            table_understanding_result=table_understanding_result,
            filters=merged_filters,
            page_numbers=route_plan.get("page_numbers", []),
            section_ids=route_plan.get("section_ids", []),
            table_ids=route_plan.get("table_ids", []),
            node_types=route_plan.get("node_types", []),
            edge_types=route_plan.get("edge_types", []),
        )

        route_outputs = {
            "retrieval_result": retrieval_result,
        }

        expanded_context_result = {}
        evidence_aggregation_result = {}
        prompt_result = {}
        llm_reasoner_result = {}
        citation_verification_result = {}

        if route_plan.get("use_context_expansion"):
            expanded_context_result = self._execute_context_expansion(
                retrieved_items=retrieval_result.get("retrieved_items", []),
                page_raws=page_raws,
                graph_index_result=graph_index_result,
                metadata_index_result=metadata_index_result,
                bm25_index_result=bm25_index_result,
                vector_index_result=vector_index_result,
                table_chunk_result=table_chunk_result,
                evidence_result=evidence_result,
                citation_result=citation_result,
                knowledge_result=knowledge_result,
            )
            route_outputs["expanded_context_result"] = expanded_context_result

        if route_plan.get("use_evidence_aggregation"):
            evidence_aggregation_result = self._execute_evidence_aggregation(
                query=query,
                retrieved_items=retrieval_result.get("retrieved_items", []),
                expanded_context_result=expanded_context_result,
                page_raws=page_raws,
                evidence_result=evidence_result,
                citation_result=citation_result,
                graph_index_result=graph_index_result,
                metadata_index_result=metadata_index_result,
                bm25_index_result=bm25_index_result,
                vector_index_result=vector_index_result,
                table_chunk_result=table_chunk_result,
                knowledge_result=knowledge_result,
            )
            route_outputs["evidence_aggregation_result"] = evidence_aggregation_result

        if route_plan.get("use_prompt_building"):
            prompt_result = self._execute_prompt_building(
                query=query,
                retrieval_result=retrieval_result,
                expanded_context_result=expanded_context_result,
                evidence_aggregation_result=evidence_aggregation_result,
                citation_verification_result=citation_verification_result,
                query_classification_result=query_classification_result,
            )
            route_outputs["prompt_result"] = prompt_result

        if route_plan.get("use_llm_reasoning"):
            llm_reasoner_result = self._execute_llm_reasoning(
                query=query,
                prompt_result=prompt_result,
                evidence_aggregation_result=evidence_aggregation_result,
                retrieval_result=retrieval_result,
                expanded_context_result=expanded_context_result,
                citation_verification_result=citation_verification_result,
                llm_fn=llm_fn,
                llm_client=llm_client,
                llm_model=llm_model,
            )
            route_outputs["llm_reasoner_result"] = llm_reasoner_result

        if route_plan.get("use_citation_verification"):
            answer_text = llm_reasoner_result.get("answer_text", "")

            citation_verification_result = self._execute_citation_verification(
                page_raws=page_raws,
                answer_text=answer_text,
                citation_result=evidence_aggregation_result,
                evidence_result=evidence_aggregation_result,
                chunk_result=knowledge_result,
                table_understanding_result=table_understanding_result,
                bm25_index_result=bm25_index_result,
                vector_index_result=vector_index_result,
                graph_index_result=graph_index_result,
                metadata_index_result=metadata_index_result,
            )
            route_outputs["citation_verification_result"] = citation_verification_result

        result = {
            "processor": "QueryRouter",
            "schema_version": "query_router_v1",
            "query": query,
            "route_name": route_plan.get("route_name", self.config.default_route),
            "route_plan": route_plan,
            "query_classification_result": query_classification_result,
            "retrieval_result": retrieval_result,
            "route_outputs": route_outputs,
            "expanded_context_result": expanded_context_result,
            "evidence_aggregation_result": evidence_aggregation_result,
            "prompt_result": prompt_result,
            "llm_reasoner_result": llm_reasoner_result,
            "citation_verification_result": citation_verification_result,
            "routing_summary": self._build_summary(
                query=query,
                route_plan=route_plan,
                classification_result=query_classification_result,
                retrieval_result=retrieval_result,
                expanded_context_result=expanded_context_result,
                evidence_aggregation_result=evidence_aggregation_result,
                prompt_result=prompt_result,
                llm_reasoner_result=llm_reasoner_result,
                citation_verification_result=citation_verification_result,
            ),
            "config": asdict(self.config),
        }

        return json_safe(result)

    def _classify_query(
        self,
        query: str,
    ) -> Dict[str, Any]:
        try:
            from document_ai.rag.query_classifier import classify_query

            return classify_query(query=query)
        except Exception:
            return self._fallback_classification(query)

    def _fallback_classification(
        self,
        query: str,
    ) -> Dict[str, Any]:
        normalized = normalize_text_for_match(query)

        query_type = "factual"
        query_intent = "factual"
        route = "hybrid"

        if re.search(r"\bbang\b|\btable\b|\bcot\b|\bdong\b", normalized):
            query_type = "table"
            query_intent = "table_query"
            route = "table"

        elif re.search(r"\blien quan\b|\bmoi quan he\b|\bket noi\b|\blink\b|\brelationship\b", normalized):
            query_type = "graph"
            query_intent = "graph_query"
            route = "graph"

        elif re.search(r"\bmetadata\b|\bthong tin tai lieu\b|\bloai tai lieu\b", normalized):
            query_type = "metadata"
            query_intent = "metadata_query"
            route = "metadata"

        elif re.search(r"\btom tat\b|\btong hop\b|\bsummary\b", normalized):
            query_type = "summary"
            query_intent = "summary"

        elif re.search(r"\bso sanh\b|\bcompare\b|\bdifference\b", normalized):
            query_type = "comparison"
            query_intent = "comparison"

        page_numbers = self._extract_page_numbers(normalized)

        return {
            "processor": "QueryClassifierFallback",
            "query": query,
            "normalized_query": normalized,
            "query_type": query_type,
            "query_intent": query_intent,
            "query_language": "vi",
            "query_scope": "page" if page_numbers else "open",
            "query_filters": {
                "page_numbers": page_numbers,
            } if page_numbers else {},
            "retrieval_strategy": {
                "strategy_name": route,
                "use_bm25": route in ["hybrid", "bm25"],
                "use_vector": route in ["hybrid", "vector"],
                "use_graph": route in ["hybrid", "graph"],
                "use_metadata": route in ["hybrid", "metadata"],
                "use_table": route in ["hybrid", "table"],
                "use_context_expansion": True,
                "use_evidence_aggregation": True,
                "use_citation_verification": True,
                "top_k": self.config.top_k,
                "candidate_pool_size": self.config.candidate_pool_size,
                "reason": ["fallback_classification"],
            },
            "routing_hints": {
                "preferred_retriever": route,
                "page_numbers": page_numbers,
                "top_k": self.config.top_k,
                "candidate_pool_size": self.config.candidate_pool_size,
            },
            "prompt_hints": {
                "language": "vi",
                "task_mode": "grounded_qa",
                "answer_style": "formal",
                "answer_format": "structured",
                "require_citations": True,
            },
            "query_complexity": {
                "level": "medium",
                "score": 0.8,
                "reasons": ["fallback"],
            },
        }

    def _build_route_plan(
        self,
        query: str,
        classification_result: Dict[str, Any],
        filters: Dict[str, Any],
        page_numbers: List[int],
        section_ids: List[str],
        table_ids: List[str],
        node_types: List[str],
        edge_types: List[str],
        route_options: Dict[str, Any],
    ) -> Dict[str, Any]:
        routing_hints = classification_result.get("routing_hints", {}) or {}
        retrieval_strategy = classification_result.get("retrieval_strategy", {}) or {}
        prompt_hints = classification_result.get("prompt_hints", {}) or {}
        query_filters = classification_result.get("query_filters", {}) or {}
        complexity = classification_result.get("query_complexity", {}) or {}

        route_name = (
            route_options.get("route_name")
            or routing_hints.get("preferred_retriever")
            or retrieval_strategy.get("strategy_name")
            or self.config.default_route
        )

        if (
            self.config.force_hybrid_for_complex_queries
            and complexity.get("level") == self.config.high_complexity_threshold
        ):
            route_name = "hybrid"

        route_name = self._normalize_route_name(route_name)

        page_numbers = self._normalize_page_numbers(
            page_numbers
            or routing_hints.get("page_numbers", [])
            or query_filters.get("page_numbers", [])
        )

        if not section_ids:
            section_ids = [
                str(item)
                for item in query_filters.get("section_refs", [])
                if item
            ]

        if not table_ids:
            table_ids = [
                str(item)
                for item in query_filters.get("table_refs", [])
                if item
            ]

        route_plan = {
            "route_name": route_name,
            "query_type": classification_result.get("query_type", "factual"),
            "query_intent": classification_result.get("query_intent", "factual"),
            "query_scope": classification_result.get("query_scope", "open"),
            "answer_format": classification_result.get("answer_format", "structured"),
            "expected_answer_type": classification_result.get("expected_answer_type", "grounded_answer"),
            "filters": filters,
            "page_numbers": page_numbers,
            "section_ids": section_ids,
            "table_ids": table_ids,
            "node_types": node_types,
            "edge_types": edge_types,
            "top_k": route_options.get("top_k") or routing_hints.get("top_k") or retrieval_strategy.get("top_k") or self.config.top_k,
            "candidate_pool_size": (
                route_options.get("candidate_pool_size")
                or routing_hints.get("candidate_pool_size")
                or retrieval_strategy.get("candidate_pool_size")
                or self.config.candidate_pool_size
            ),
            "use_bm25": bool(retrieval_strategy.get("use_bm25", route_name in ["hybrid", "bm25"])),
            "use_vector": bool(retrieval_strategy.get("use_vector", route_name in ["hybrid", "vector"])),
            "use_graph": bool(retrieval_strategy.get("use_graph", route_name in ["hybrid", "graph"])),
            "use_metadata": bool(retrieval_strategy.get("use_metadata", route_name in ["hybrid", "metadata"])),
            "use_table": bool(retrieval_strategy.get("use_table", route_name in ["hybrid", "table"])),
            "use_context_expansion": bool(
                route_options.get(
                    "use_context_expansion",
                    retrieval_strategy.get("use_context_expansion", self.config.enable_context_expansion),
                )
            ),
            "use_evidence_aggregation": bool(
                route_options.get(
                    "use_evidence_aggregation",
                    retrieval_strategy.get("use_evidence_aggregation", self.config.enable_evidence_aggregation),
                )
            ),
            "use_prompt_building": bool(route_options.get("use_prompt_building", self.config.enable_prompt_building)),
            "use_llm_reasoning": bool(route_options.get("use_llm_reasoning", self.config.enable_llm_reasoning)),
            "use_citation_verification": bool(
                route_options.get(
                    "use_citation_verification",
                    retrieval_strategy.get("use_citation_verification", self.config.enable_citation_verification),
                )
            ),
            "prompt_hints": prompt_hints,
            "routing_reason": self._route_reason(
                route_name=route_name,
                classification_result=classification_result,
                retrieval_strategy=retrieval_strategy,
                complexity=complexity,
            ),
        }

        route_plan = self._apply_route_capabilities(route_plan)

        return route_plan

    def _apply_route_capabilities(
        self,
        route_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        route_name = route_plan.get("route_name", self.config.default_route)

        if route_name == "hybrid":
            route_plan["use_bm25"] = self.config.enable_bm25_route and route_plan.get("use_bm25", True)
            route_plan["use_vector"] = self.config.enable_vector_route and route_plan.get("use_vector", True)
            route_plan["use_graph"] = self.config.enable_graph_route and route_plan.get("use_graph", True)
            route_plan["use_metadata"] = self.config.enable_metadata_route and route_plan.get("use_metadata", True)
            route_plan["use_table"] = self.config.enable_table_route and route_plan.get("use_table", True)

        elif route_name == "bm25":
            route_plan["use_bm25"] = self.config.enable_bm25_route
            route_plan["use_vector"] = False
            route_plan["use_graph"] = False
            route_plan["use_metadata"] = False
            route_plan["use_table"] = False

        elif route_name == "vector":
            route_plan["use_bm25"] = False
            route_plan["use_vector"] = self.config.enable_vector_route
            route_plan["use_graph"] = False
            route_plan["use_metadata"] = False
            route_plan["use_table"] = False

        elif route_name == "graph":
            route_plan["use_bm25"] = False
            route_plan["use_vector"] = False
            route_plan["use_graph"] = self.config.enable_graph_route
            route_plan["use_metadata"] = False
            route_plan["use_table"] = False

        elif route_name == "metadata":
            route_plan["use_bm25"] = False
            route_plan["use_vector"] = False
            route_plan["use_graph"] = False
            route_plan["use_metadata"] = self.config.enable_metadata_route
            route_plan["use_table"] = False

        elif route_name == "table":
            route_plan["use_bm25"] = False
            route_plan["use_vector"] = False
            route_plan["use_graph"] = False
            route_plan["use_metadata"] = True
            route_plan["use_table"] = self.config.enable_table_route

        return route_plan

    def _execute_retrieval_route(
        self,
        query: str,
        route_plan: Dict[str, Any],
        bm25_index_result: Dict[str, Any],
        vector_index_result: Dict[str, Any],
        graph_index_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
        table_chunk_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        filters: Dict[str, Any],
        page_numbers: List[int],
        section_ids: List[str],
        table_ids: List[str],
        node_types: List[str],
        edge_types: List[str],
    ) -> Dict[str, Any]:
        route_name = route_plan.get("route_name", self.config.default_route)
        top_k = self._safe_int(route_plan.get("top_k"), self.config.top_k)

        if route_name == "hybrid":
            return self._run_hybrid_retriever(
                query=query,
                route_plan=route_plan,
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

        if route_name == "bm25":
            return self._run_bm25_retriever(
                query=query,
                bm25_index_result=bm25_index_result,
                filters=filters,
                top_k=top_k,
            )

        if route_name == "vector":
            return self._run_vector_retriever(
                query=query,
                vector_index_result=vector_index_result,
                filters=filters,
                top_k=top_k,
            )

        if route_name == "graph":
            return self._run_graph_retriever(
                query=query,
                graph_index_result=graph_index_result,
                filters=filters,
                node_types=node_types,
                edge_types=edge_types,
                page_numbers=page_numbers,
                top_k=top_k,
            )

        if route_name == "metadata":
            return self._run_metadata_retriever(
                query=query,
                metadata_index_result=metadata_index_result,
                filters=filters,
                page_numbers=page_numbers,
                top_k=top_k,
            )

        if route_name == "table":
            return self._run_table_retriever(
                query=query,
                table_chunk_result=table_chunk_result,
                table_understanding_result=table_understanding_result,
                metadata_index_result=metadata_index_result,
                filters=filters,
                page_numbers=page_numbers,
                table_ids=table_ids,
                top_k=top_k,
            )

        return self._run_hybrid_retriever(
            query=query,
            route_plan=route_plan,
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

    def _run_hybrid_retriever(
        self,
        query: str,
        route_plan: Dict[str, Any],
        bm25_index_result: Dict[str, Any],
        vector_index_result: Dict[str, Any],
        graph_index_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
        table_chunk_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        filters: Dict[str, Any],
        page_numbers: List[int],
        section_ids: List[str],
        table_ids: List[str],
        top_k: int,
    ) -> Dict[str, Any]:
        try:
            from document_ai.rag.hybrid_retriever import HybridRetriever, HybridRetrieverConfig

            cfg = HybridRetrieverConfig(
                top_k=top_k,
                candidate_pool_size=self._safe_int(
                    route_plan.get("candidate_pool_size"),
                    self.config.candidate_pool_size,
                ),
                use_bm25=route_plan.get("use_bm25", True),
                use_vector=route_plan.get("use_vector", True),
                use_graph=route_plan.get("use_graph", True),
                use_metadata=route_plan.get("use_metadata", True),
                use_table=route_plan.get("use_table", True),
            )

            retriever = HybridRetriever(config=cfg)

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

        except Exception as exc:
            return self._empty_retrieval_result(
                route_name="hybrid",
                query=query,
                error=str(exc),
            )

    def _run_bm25_retriever(
        self,
        query: str,
        bm25_index_result: Dict[str, Any],
        filters: Dict[str, Any],
        top_k: int,
    ) -> Dict[str, Any]:
        try:
            from document_ai.indexing.bm25_index_builder import search_bm25_index

            items = search_bm25_index(
                index_result=bm25_index_result,
                query=query,
                top_k=top_k,
                filters=filters,
            )

            items = [
                self._normalize_item(item, retrieval_source="bm25")
                for item in items
            ]

            return self._simple_retrieval_result(
                route_name="bm25",
                query=query,
                items=items,
            )

        except Exception as exc:
            return self._empty_retrieval_result(
                route_name="bm25",
                query=query,
                error=str(exc),
            )

    def _run_vector_retriever(
        self,
        query: str,
        vector_index_result: Dict[str, Any],
        filters: Dict[str, Any],
        top_k: int,
    ) -> Dict[str, Any]:
        try:
            from document_ai.indexing.vector_index_builder import search_vector_index

            items = search_vector_index(
                index_result=vector_index_result,
                query=query,
                top_k=top_k,
                filters=filters,
            )

            items = [
                self._normalize_item(item, retrieval_source="vector")
                for item in items
            ]

            return self._simple_retrieval_result(
                route_name="vector",
                query=query,
                items=items,
            )

        except Exception as exc:
            return self._empty_retrieval_result(
                route_name="vector",
                query=query,
                error=str(exc),
            )

    def _run_graph_retriever(
        self,
        query: str,
        graph_index_result: Dict[str, Any],
        filters: Dict[str, Any],
        node_types: List[str],
        edge_types: List[str],
        page_numbers: List[int],
        top_k: int,
    ) -> Dict[str, Any]:
        try:
            from document_ai.rag.graph_retriever import retrieve_from_graph

            result = retrieve_from_graph(
                query=query,
                graph_index_result=graph_index_result,
                filters=filters,
                node_types=node_types,
                edge_types=edge_types,
                page_numbers=page_numbers,
            )

            items = result.get("retrieved_graph_items", [])

            result["retrieved_items"] = [
                self._normalize_item(item, retrieval_source="graph")
                for item in items
            ]
            result["route_name"] = "graph"

            return result

        except Exception as exc:
            return self._empty_retrieval_result(
                route_name="graph",
                query=query,
                error=str(exc),
            )

    def _run_metadata_retriever(
        self,
        query: str,
        metadata_index_result: Dict[str, Any],
        filters: Dict[str, Any],
        page_numbers: List[int],
        top_k: int,
    ) -> Dict[str, Any]:
        try:
            from document_ai.indexing.metadata_index_builder import search_metadata_index

            items = search_metadata_index(
                index_result=metadata_index_result,
                query=query,
                filters=filters,
                page_numbers=page_numbers,
                top_k=top_k,
            )

            items = [
                self._normalize_item(item, retrieval_source="metadata")
                for item in items
            ]

            return self._simple_retrieval_result(
                route_name="metadata",
                query=query,
                items=items,
            )

        except Exception as exc:
            return self._empty_retrieval_result(
                route_name="metadata",
                query=query,
                error=str(exc),
            )

    def _run_table_retriever(
        self,
        query: str,
        table_chunk_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
        filters: Dict[str, Any],
        page_numbers: List[int],
        table_ids: List[str],
        top_k: int,
    ) -> Dict[str, Any]:
        try:
            from document_ai.rag.table_retriever import retrieve_from_table

            result = retrieve_from_table(
                query=query,
                table_chunk_result=table_chunk_result,
                table_understanding_result=table_understanding_result,
                metadata_index_result=metadata_index_result,
                filters=filters,
                page_numbers=page_numbers,
                table_ids=table_ids,
                top_k=top_k,
            )

            items = result.get("retrieved_table_items", []) or result.get("retrieved_items", [])
            result["retrieved_items"] = [
                self._normalize_item(item, retrieval_source="table")
                for item in items
            ]
            result["route_name"] = "table"

            return result

        except Exception:
            return self._fallback_table_retriever(
                query=query,
                table_chunk_result=table_chunk_result,
                table_understanding_result=table_understanding_result,
                metadata_index_result=metadata_index_result,
                filters=filters,
                page_numbers=page_numbers,
                table_ids=table_ids,
                top_k=top_k,
            )

    def _fallback_table_retriever(
        self,
        query: str,
        table_chunk_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
        filters: Dict[str, Any],
        page_numbers: List[int],
        table_ids: List[str],
        top_k: int,
    ) -> Dict[str, Any]:
        candidates = []

        for source in [table_chunk_result, table_understanding_result]:
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
                    candidates.extend([self._to_dict(item) for item in values])

            for sub_key in [
                "table_chunk_result",
                "table_semantic_result",
                "table_understanding_result",
                "multi_page_table_result",
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
                joined_type = f"{item.get('item_type', '')} {item.get('source_type', '')}".lower()
                if "table" in joined_type or item.get("table_id"):
                    candidates.append(item)

        query_tokens = self._tokenize(query)
        page_set = set(page_numbers or [])
        table_id_set = set(table_ids or [])

        scored = []

        for item in candidates:
            if not self._passes_filters(item, filters):
                continue

            item_pages = set(self._resolve_page_numbers(item))
            if page_set and item_pages and not page_set.intersection(item_pages):
                continue
            if page_set and not item_pages:
                continue

            item_table_ids = set(self._table_ids_from_item(item))
            if table_id_set and not table_id_set.intersection(item_table_ids):
                continue

            text = normalize_pdf_text(
                item.get("text")
                or item.get("text_preview")
                or item.get("caption")
                or item.get("caption_text")
                or item.get("title")
                or ""
            )

            if not text and item.get("column_headers"):
                text = " | ".join(
                    [
                        normalize_pdf_text(header)
                        for header in item.get("column_headers", [])
                        if normalize_pdf_text(header)
                    ]
                )

            if not text:
                continue

            text_norm = normalize_text_for_match(text)
            matched = [token for token in query_tokens if token in text_norm]
            score = len(matched) / max(len(query_tokens), 1) if query_tokens else 0.5
            score += self._safe_float(item.get("score"), 0.0)

            scored.append(
                self._normalize_item(
                    {
                        **item,
                        "score": round(score, 6),
                        "text": text,
                        "matched_terms": matched,
                    },
                    retrieval_source="table",
                )
            )

        scored = sorted(
            scored,
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )[:top_k]

        return self._simple_retrieval_result(
            route_name="table",
            query=query,
            items=scored,
        )

    def _execute_context_expansion(
        self,
        retrieved_items: List[Dict[str, Any]],
        page_raws: List[Any],
        graph_index_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
        bm25_index_result: Dict[str, Any],
        vector_index_result: Dict[str, Any],
        table_chunk_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        citation_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            from document_ai.rag.context_expander import expand_context

            return expand_context(
                retrieved_items=retrieved_items,
                page_raws=page_raws,
                table_chunk_result=table_chunk_result,
                evidence_result=evidence_result,
                citation_result=citation_result,
                graph_index_result=graph_index_result,
                metadata_index_result=metadata_index_result,
                bm25_index_result=bm25_index_result,
                vector_index_result=vector_index_result,
                knowledge_result=knowledge_result,
            )
        except Exception as exc:
            return {
                "processor": "ContextExpander",
                "error": str(exc),
                "expanded_context_items": [],
                "context_expansion_summary": {
                    "has_expanded_context": False,
                    "error": str(exc),
                },
            }

    def _execute_evidence_aggregation(
        self,
        query: str,
        retrieved_items: List[Dict[str, Any]],
        expanded_context_result: Dict[str, Any],
        page_raws: List[Any],
        evidence_result: Dict[str, Any],
        citation_result: Dict[str, Any],
        graph_index_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
        bm25_index_result: Dict[str, Any],
        vector_index_result: Dict[str, Any],
        table_chunk_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            from document_ai.rag.evidence_aggregator import aggregate_evidence

            return aggregate_evidence(
                query=query,
                retrieved_items=retrieved_items,
                expanded_context_result=expanded_context_result,
                page_raws=page_raws,
                evidence_result=evidence_result,
                citation_result=citation_result,
                graph_index_result=graph_index_result,
                metadata_index_result=metadata_index_result,
                bm25_index_result=bm25_index_result,
                vector_index_result=vector_index_result,
                table_chunk_result=table_chunk_result,
                knowledge_result=knowledge_result,
            )
        except Exception as exc:
            return {
                "processor": "EvidenceAggregator",
                "error": str(exc),
                "aggregated_evidence": [],
                "supporting_evidence": [],
                "citations": [],
                "evidence_aggregation_summary": {
                    "has_evidence": False,
                    "error": str(exc),
                },
            }

    def _execute_prompt_building(
        self,
        query: str,
        retrieval_result: Dict[str, Any],
        expanded_context_result: Dict[str, Any],
        evidence_aggregation_result: Dict[str, Any],
        citation_verification_result: Dict[str, Any],
        query_classification_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            from document_ai.rag.prompt_builder import build_prompt

            prompt_hints = query_classification_result.get("prompt_hints", {}) or {}

            return build_prompt(
                query=query,
                retrieval_result=retrieval_result,
                expanded_context_result=expanded_context_result,
                evidence_aggregation_result=evidence_aggregation_result,
                citation_verification_result=citation_verification_result,
                prompt_options=prompt_hints,
            )
        except Exception as exc:
            return {
                "processor": "PromptBuilder",
                "error": str(exc),
                "prompt_text": "",
                "prompt_summary": {
                    "has_prompt": False,
                    "error": str(exc),
                },
            }

    def _execute_llm_reasoning(
        self,
        query: str,
        prompt_result: Dict[str, Any],
        evidence_aggregation_result: Dict[str, Any],
        retrieval_result: Dict[str, Any],
        expanded_context_result: Dict[str, Any],
        citation_verification_result: Dict[str, Any],
        llm_fn: Optional[Any],
        llm_client: Optional[Any],
        llm_model: str,
    ) -> Dict[str, Any]:
        try:
            from document_ai.rag.llm_reasoner import reason_with_llm

            return reason_with_llm(
                query=query,
                prompt_result=prompt_result,
                evidence_aggregation_result=evidence_aggregation_result,
                retrieval_result=retrieval_result,
                expanded_context_result=expanded_context_result,
                citation_verification_result=citation_verification_result,
                llm_fn=llm_fn,
                llm_client=llm_client,
                llm_model=llm_model,
            )
        except Exception as exc:
            return {
                "processor": "LLMReasoner",
                "error": str(exc),
                "answer_text": "",
                "answer_status": "reasoner_error",
                "llm_reasoning_summary": {
                    "safe_to_use": False,
                    "error": str(exc),
                },
            }

    def _execute_citation_verification(
        self,
        page_raws: List[Any],
        answer_text: str,
        citation_result: Dict[str, Any],
        evidence_result: Dict[str, Any],
        chunk_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        bm25_index_result: Dict[str, Any],
        vector_index_result: Dict[str, Any],
        graph_index_result: Dict[str, Any],
        metadata_index_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            from document_ai.rag.citation_verifier import verify_citations

            return verify_citations(
                page_raws=page_raws,
                answer_text=answer_text,
                citation_result=citation_result,
                evidence_result=evidence_result,
                chunk_result=chunk_result,
                table_understanding_result=table_understanding_result,
                bm25_index_result=bm25_index_result,
                vector_index_result=vector_index_result,
                graph_index_result=graph_index_result,
                metadata_index_result=metadata_index_result,
            )
        except Exception as exc:
            return {
                "processor": "CitationVerifier",
                "error": str(exc),
                "verified_citations": [],
                "questionable_citations": [],
                "failed_citations": [],
                "citation_verification_summary": {
                    "has_citations": False,
                    "error": str(exc),
                },
            }

    def _simple_retrieval_result(
        self,
        route_name: str,
        query: str,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        for rank, item in enumerate(items, start=1):
            item["rank"] = rank

        return {
            "processor": "QueryRouter",
            "schema_version": "query_router_retrieval_v1",
            "route_name": route_name,
            "query": query,
            "retrieved_items": items,
            "retrieval_context_text": self._build_context_text(query, items) if self.config.include_context_text else "",
            "retrieved_by_page": self._group_by_page(items),
            "retrieved_by_source": self._group_by_source(items),
            "retrieval_summary": {
                "has_results": len(items) > 0,
                "route_name": route_name,
                "query": query,
                "retrieved_count": len(items),
            },
        }

    def _empty_retrieval_result(
        self,
        route_name: str,
        query: str,
        error: str = "",
    ) -> Dict[str, Any]:
        return {
            "processor": "QueryRouter",
            "schema_version": "query_router_retrieval_v1",
            "route_name": route_name,
            "query": query,
            "retrieved_items": [],
            "retrieval_context_text": "",
            "retrieved_by_page": {},
            "retrieved_by_source": {},
            "error": error,
            "retrieval_summary": {
                "has_results": False,
                "route_name": route_name,
                "query": query,
                "retrieved_count": 0,
                "error": error,
            },
        }

    def _normalize_item(
        self,
        item: Dict[str, Any],
        retrieval_source: str,
    ) -> Dict[str, Any]:
        item = self._to_dict(item)

        document = item.get("document", {})
        if isinstance(document, dict):
            merged = {
                **document,
                **item,
            }
            item = merged

        metadata = item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        text = normalize_pdf_text(
            item.get("text")
            or item.get("text_preview")
            or item.get("quote")
            or item.get("content")
            or item.get("label")
            or item.get("title")
            or ""
        )

        source_id = (
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

        return {
            **item,
            "retrieval_source": retrieval_source,
            "source_id": source_id,
            "source_type": (
                item.get("source_type")
                or item.get("chunk_type")
                or item.get("evidence_type")
                or item.get("node_type")
                or item.get("item_type")
                or retrieval_source
            ),
            "title": normalize_pdf_text(item.get("title") or item.get("label") or item.get("section_title") or ""),
            "text": text,
            "text_preview": item.get("text_preview") or self._preview(text, 700),
            "page_numbers": self._resolve_page_numbers(item),
            "section_id": item.get("section_id", "") or metadata.get("section_id", ""),
            "section_title": item.get("section_title", "") or metadata.get("section_title", ""),
            "table_id": item.get("table_id", "") or metadata.get("table_id", ""),
            "chunk_id": item.get("chunk_id", "") or metadata.get("chunk_id", ""),
            "evidence_id": item.get("evidence_id", "") or metadata.get("evidence_id", ""),
            "citation_id": item.get("citation_id", "") or metadata.get("citation_id", ""),
            "node_id": item.get("node_id", "") or metadata.get("node_id", ""),
            "score": self._safe_float(item.get("score"), 0.0),
            "metadata": metadata,
        }

    def _merge_filters(
        self,
        explicit_filters: Dict[str, Any],
        classification_result: Dict[str, Any],
        page_numbers: List[int],
        section_ids: List[str],
        table_ids: List[str],
    ) -> Dict[str, Any]:
        merged = dict(explicit_filters or {})

        query_filters = classification_result.get("query_filters", {}) or {}

        if query_filters.get("page_numbers") and not page_numbers:
            page_numbers = self._normalize_page_numbers(query_filters.get("page_numbers", []))

        if page_numbers:
            merged.setdefault("page_numbers", page_numbers)

        if section_ids:
            merged.setdefault("section_id", section_ids)

        if table_ids:
            merged.setdefault("table_id", table_ids)

        if query_filters.get("prefer_table"):
            merged.setdefault("prefer_table", True)

        if query_filters.get("require_evidence"):
            merged.setdefault("require_evidence", True)

        if query_filters.get("quoted_phrases"):
            merged.setdefault("quoted_phrases", query_filters.get("quoted_phrases"))

        if query_filters.get("years"):
            merged.setdefault("years", query_filters.get("years"))

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
            if key in ["prefer_table", "require_evidence", "quoted_phrases", "years"]:
                continue

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

    def _route_reason(
        self,
        route_name: str,
        classification_result: Dict[str, Any],
        retrieval_strategy: Dict[str, Any],
        complexity: Dict[str, Any],
    ) -> List[str]:
        reasons = []

        reasons.extend(retrieval_strategy.get("reason", []) or [])

        query_type = classification_result.get("query_type", "")
        query_intent = classification_result.get("query_intent", "")

        if query_type:
            reasons.append(f"query_type={query_type}")

        if query_intent:
            reasons.append(f"query_intent={query_intent}")

        if complexity.get("level"):
            reasons.append(f"complexity={complexity.get('level')}")

        reasons.append(f"route={route_name}")

        return list(dict.fromkeys([item for item in reasons if item]))

    def _normalize_route_name(
        self,
        route_name: str,
    ) -> str:
        route_name = normalize_text_for_match(route_name)

        route_aliases = {
            "keyword": "bm25",
            "semantic": "vector",
            "vector_search": "vector",
            "graph_search": "graph",
            "metadata_search": "metadata",
            "table_search": "table",
            "hybrid_search": "hybrid",
        }

        route_name = route_aliases.get(route_name, route_name)

        allowed = {"hybrid", "bm25", "vector", "graph", "metadata", "table"}

        if route_name not in allowed:
            return self.config.default_route

        return route_name

    def _build_context_text(
        self,
        query: str,
        items: List[Dict[str, Any]],
    ) -> str:
        parts = []

        if query:
            parts.append(f"Truy vấn: {normalize_pdf_text(query)}")

        total_chars = 0
        max_chars = 12000

        for index, item in enumerate(items, start=1):
            page_numbers = item.get("page_numbers", []) or []
            page_label = ""

            if page_numbers:
                if len(page_numbers) == 1:
                    page_label = f"trang {page_numbers[0]}"
                else:
                    page_label = f"trang {page_numbers[0]}-{page_numbers[-1]}"

            header_parts = [
                f"[RouteResult {index}]",
                f"source={item.get('retrieval_source', '')}",
                f"type={item.get('source_type', '')}",
                f"score={item.get('score', 0)}",
            ]

            if item.get("title"):
                header_parts.append(item.get("title"))

            if page_label:
                header_parts.append(page_label)

            text = normalize_pdf_text(item.get("text") or item.get("text_preview") or "")

            block = "\n".join(
                [
                    " | ".join([part for part in header_parts if part]),
                    text,
                ]
            )

            if total_chars + len(block) > max_chars:
                break

            parts.append(block)
            total_chars += len(block)

        return normalize_pdf_text("\n\n".join(parts))

    def _group_by_page(
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

    def _group_by_source(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = {}

        for item in items:
            source = item.get("retrieval_source", "unknown") or "unknown"
            grouped.setdefault(source, [])
            grouped[source].append(item)

        return grouped

    def _build_summary(
        self,
        query: str,
        route_plan: Dict[str, Any],
        classification_result: Dict[str, Any],
        retrieval_result: Dict[str, Any],
        expanded_context_result: Dict[str, Any],
        evidence_aggregation_result: Dict[str, Any],
        prompt_result: Dict[str, Any],
        llm_reasoner_result: Dict[str, Any],
        citation_verification_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        retrieved_items = retrieval_result.get("retrieved_items", []) or []
        expanded_items = expanded_context_result.get("expanded_context_items", []) or []
        evidence_items = evidence_aggregation_result.get("aggregated_evidence", []) or []
        citations = evidence_aggregation_result.get("citations", []) or []
        verified_citations = citation_verification_result.get("verified_citations", []) or []

        return {
            "has_query": bool(query),
            "route_name": route_plan.get("route_name", ""),
            "query_type": route_plan.get("query_type", ""),
            "query_intent": route_plan.get("query_intent", ""),
            "retrieved_count": len(retrieved_items),
            "expanded_context_count": len(expanded_items),
            "evidence_count": len(evidence_items),
            "citation_count": len(citations),
            "verified_citation_count": len(verified_citations),
            "has_prompt": bool(prompt_result.get("prompt_text", "")),
            "has_answer": bool(llm_reasoner_result.get("answer_text", "")),
            "answer_status": llm_reasoner_result.get("answer_status", ""),
            "answer_confidence": llm_reasoner_result.get("answer_confidence", 0.0),
            "route_uses": {
                "bm25": route_plan.get("use_bm25", False),
                "vector": route_plan.get("use_vector", False),
                "graph": route_plan.get("use_graph", False),
                "metadata": route_plan.get("use_metadata", False),
                "table": route_plan.get("use_table", False),
                "context_expansion": route_plan.get("use_context_expansion", False),
                "evidence_aggregation": route_plan.get("use_evidence_aggregation", False),
                "prompt_building": route_plan.get("use_prompt_building", False),
                "llm_reasoning": route_plan.get("use_llm_reasoning", False),
                "citation_verification": route_plan.get("use_citation_verification", False),
            },
            "routing_reason": route_plan.get("routing_reason", []),
            "classification_summary": classification_result.get("query_classification_summary", {}),
        }

    def _extract_page_numbers(
        self,
        normalized_query: str,
    ) -> List[int]:
        pages = []

        patterns = [
            r"\btrang\s*(\d+)\s*-\s*(\d+)",
            r"\bpage\s*(\d+)\s*-\s*(\d+)",
            r"\btr\.\s*(\d+)\s*-\s*(\d+)",
            r"\bp\.\s*(\d+)\s*-\s*(\d+)",
        ]

        for pattern in patterns:
            for start, end in re.findall(pattern, normalized_query, flags=re.IGNORECASE):
                start_i = self._safe_int(start, 0)
                end_i = self._safe_int(end, 0)

                if start_i > 0 and end_i >= start_i:
                    pages.extend(list(range(start_i, end_i + 1)))

        single_patterns = [
            r"\btrang\s*(\d+)",
            r"\bpage\s*(\d+)",
            r"\btr\.\s*(\d+)",
            r"\bp\.\s*(\d+)",
        ]

        for pattern in single_patterns:
            for value in re.findall(pattern, normalized_query, flags=re.IGNORECASE):
                page = self._safe_int(value, 0)

                if page > 0:
                    pages.append(page)

        return sorted(list(dict.fromkeys(pages)))

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

        page_numbers = item.get("page_numbers") or item.get("content_page_numbers") or []

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

    def _preview(
        self,
        text: Any,
        max_chars: int = 700,
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
        route_result: Dict[str, Any],
        output_path: str,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                json_safe(route_result),
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


def route_query(
    query: str,
    query_classification_result: Optional[Dict[str, Any]] = None,
    bm25_index_result: Optional[Dict[str, Any]] = None,
    vector_index_result: Optional[Dict[str, Any]] = None,
    graph_index_result: Optional[Dict[str, Any]] = None,
    metadata_index_result: Optional[Dict[str, Any]] = None,
    table_chunk_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    page_raws: Optional[List[Any]] = None,
    evidence_result: Optional[Dict[str, Any]] = None,
    citation_result: Optional[Dict[str, Any]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    page_numbers: Optional[List[int]] = None,
    section_ids: Optional[List[str]] = None,
    table_ids: Optional[List[str]] = None,
    node_types: Optional[List[str]] = None,
    edge_types: Optional[List[str]] = None,
    route_options: Optional[Dict[str, Any]] = None,
    llm_fn: Optional[Any] = None,
    llm_client: Optional[Any] = None,
    llm_model: str = "",
) -> Dict[str, Any]:
    router = QueryRouter()
    return router.process(
        query=query,
        query_classification_result=query_classification_result,
        bm25_index_result=bm25_index_result,
        vector_index_result=vector_index_result,
        graph_index_result=graph_index_result,
        metadata_index_result=metadata_index_result,
        table_chunk_result=table_chunk_result,
        table_understanding_result=table_understanding_result,
        page_raws=page_raws,
        evidence_result=evidence_result,
        citation_result=citation_result,
        knowledge_result=knowledge_result,
        filters=filters,
        page_numbers=page_numbers,
        section_ids=section_ids,
        table_ids=table_ids,
        node_types=node_types,
        edge_types=edge_types,
        route_options=route_options,
        llm_fn=llm_fn,
        llm_client=llm_client,
        llm_model=llm_model,
    )


def plan_query_route(
    query: str,
    query_classification_result: Optional[Dict[str, Any]] = None,
    route_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    router = QueryRouter()
    classification = query_classification_result or router._classify_query(query)
    filters = router._merge_filters(
        explicit_filters={},
        classification_result=classification,
        page_numbers=[],
        section_ids=[],
        table_ids=[],
    )
    return router._build_route_plan(
        query=query,
        classification_result=classification,
        filters=filters,
        page_numbers=[],
        section_ids=[],
        table_ids=[],
        node_types=[],
        edge_types=[],
        route_options=route_options or {},
    )
