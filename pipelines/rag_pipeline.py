"""
rag_pipeline.py

Production V1 - Colab Ready

Purpose
-------
Run retrieval augmented generation over processed document knowledge.

Input
-----
- query: user question
- page_raws
- knowledge_result
- indexing_result
- document_structure_result
- table_understanding_result
- cross_page_context_result

Output
------
Dictionary with:
- query_classification
- query_route
- retrieval_result
- graph_retrieval_result
- table_retrieval_result
- context_expansion_result
- evidence_aggregation_result
- prompt_result
- answer_result
- citation_verification_result
- rag_summary

Flow
----
KnowledgePipeline
    ↓
RAGPipeline
        ├── QueryClassifier
        ├── QueryRouter
        ├── HybridRetriever
        ├── GraphRetriever
        ├── TableRetriever
        ├── ContextExpander
        ├── EvidenceAggregator
        ├── PromptBuilder
        ├── LLMReasoner
        └── CitationVerifier
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import re
import traceback
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class RAGPipelineConfig:
    run_query_classifier: bool = True
    run_query_router: bool = True
    run_hybrid_retriever: bool = True
    run_graph_retriever: bool = True
    run_table_retriever: bool = True
    run_context_expander: bool = True
    run_evidence_aggregator: bool = True
    run_prompt_builder: bool = True
    run_llm_reasoner: bool = True
    run_citation_verifier: bool = True

    continue_on_error: bool = True

    top_k_chunks: int = 8
    top_k_evidence: int = 8
    top_k_tables: int = 5
    top_k_graph: int = 10

    min_retrieval_score: float = 0.05
    context_window_pages: int = 1
    max_context_chars: int = 12000
    max_answer_chars: int = 6000

    use_table_retrieval_for_table_query: bool = True
    use_graph_retrieval_for_context_query: bool = True

    save_json: bool = False
    output_dir: str = "outputs/rag"

    include_debug: bool = True


class RAGPipeline:
    def __init__(
        self,
        config: Optional[RAGPipelineConfig] = None,
    ):
        self.config = config or RAGPipelineConfig()

        self.query_classifier = self._load_component(
            module_path="document_ai.rag.query_classifier",
            class_name="QueryClassifier",
        )

        self.query_router = self._load_component(
            module_path="document_ai.rag.query_router",
            class_name="QueryRouter",
        )

        self.hybrid_retriever = self._load_component(
            module_path="document_ai.rag.hybrid_retriever",
            class_name="HybridRetriever",
        )

        self.graph_retriever = self._load_component(
            module_path="document_ai.rag.graph_retriever",
            class_name="GraphRetriever",
        )

        self.table_retriever = self._load_component(
            module_path="document_ai.rag.table_retriever",
            class_name="TableRetriever",
        )

        self.context_expander = self._load_component(
            module_path="document_ai.rag.context_expander",
            class_name="ContextExpander",
        )

        self.evidence_aggregator = self._load_component(
            module_path="document_ai.rag.evidence_aggregator",
            class_name="EvidenceAggregator",
        )

        self.prompt_builder = self._load_component(
            module_path="document_ai.rag.prompt_builder",
            class_name="PromptBuilder",
        )

        self.llm_reasoner = self._load_component(
            module_path="document_ai.rag.llm_reasoner",
            class_name="LLMReasoner",
        )

        self.citation_verifier = self._load_component(
            module_path="document_ai.rag.citation_verifier",
            class_name="CitationVerifier",
        )

    def process(
        self,
        query: str,
        page_raws: Optional[List[PageRaw]] = None,
        knowledge_result: Optional[Dict[str, Any]] = None,
        indexing_result: Optional[Dict[str, Any]] = None,
        document_structure_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        cross_page_context_result: Optional[Dict[str, Any]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        page_raws = page_raws or []
        knowledge_result = knowledge_result or {}
        indexing_result = indexing_result or {}
        document_structure_result = document_structure_result or {}
        table_understanding_result = table_understanding_result or {}
        cross_page_context_result = cross_page_context_result or {}
        extra_context = extra_context or {}

        errors: List[Dict[str, Any]] = []

        query_classification: Dict[str, Any] = {}
        query_route: Dict[str, Any] = {}
        retrieval_result: Dict[str, Any] = {}
        graph_retrieval_result: Dict[str, Any] = {}
        table_retrieval_result: Dict[str, Any] = {}
        context_expansion_result: Dict[str, Any] = {}
        evidence_aggregation_result: Dict[str, Any] = {}
        prompt_result: Dict[str, Any] = {}
        answer_result: Dict[str, Any] = {}
        citation_verification_result: Dict[str, Any] = {}

        if self.config.run_query_classifier:
            query_classification = self._run_step(
                step_name="QueryClassifier",
                component=self.query_classifier,
                kwargs={
                    "query": query,
                    "knowledge_result": knowledge_result,
                    "table_understanding_result": table_understanding_result,
                    "extra_context": extra_context,
                },
                fallback_fn=lambda: self._fallback_query_classification(query),
                errors=errors,
            )

        if self.config.run_query_router:
            query_route = self._run_step(
                step_name="QueryRouter",
                component=self.query_router,
                kwargs={
                    "query": query,
                    "query_classification": query_classification,
                    "knowledge_result": knowledge_result,
                    "indexing_result": indexing_result,
                    "extra_context": extra_context,
                },
                fallback_fn=lambda: self._fallback_query_route(
                    query=query,
                    query_classification=query_classification,
                ),
                errors=errors,
            )

        if self.config.run_hybrid_retriever:
            retrieval_result = self._run_step(
                step_name="HybridRetriever",
                component=self.hybrid_retriever,
                kwargs={
                    "query": query,
                    "query_classification": query_classification,
                    "query_route": query_route,
                    "knowledge_result": knowledge_result,
                    "indexing_result": indexing_result,
                    "page_raws": page_raws,
                    "top_k": self.config.top_k_chunks,
                },
                fallback_fn=lambda: self._fallback_hybrid_retrieval(
                    query=query,
                    knowledge_result=knowledge_result,
                    page_raws=page_raws,
                    top_k=self.config.top_k_chunks,
                ),
                errors=errors,
            )

        if self.config.run_graph_retriever:
            graph_retrieval_result = self._run_step(
                step_name="GraphRetriever",
                component=self.graph_retriever,
                kwargs={
                    "query": query,
                    "query_classification": query_classification,
                    "query_route": query_route,
                    "knowledge_result": knowledge_result,
                    "cross_page_context_result": cross_page_context_result,
                    "top_k": self.config.top_k_graph,
                },
                fallback_fn=lambda: self._fallback_graph_retrieval(
                    query=query,
                    knowledge_result=knowledge_result,
                    cross_page_context_result=cross_page_context_result,
                    top_k=self.config.top_k_graph,
                ),
                errors=errors,
            )

        if self.config.run_table_retriever:
            table_retrieval_result = self._run_step(
                step_name="TableRetriever",
                component=self.table_retriever,
                kwargs={
                    "query": query,
                    "query_classification": query_classification,
                    "query_route": query_route,
                    "knowledge_result": knowledge_result,
                    "table_understanding_result": table_understanding_result,
                    "top_k": self.config.top_k_tables,
                },
                fallback_fn=lambda: self._fallback_table_retrieval(
                    query=query,
                    knowledge_result=knowledge_result,
                    table_understanding_result=table_understanding_result,
                    top_k=self.config.top_k_tables,
                ),
                errors=errors,
            )

        if self.config.run_context_expander:
            context_expansion_result = self._run_step(
                step_name="ContextExpander",
                component=self.context_expander,
                kwargs={
                    "query": query,
                    "retrieval_result": retrieval_result,
                    "graph_retrieval_result": graph_retrieval_result,
                    "table_retrieval_result": table_retrieval_result,
                    "knowledge_result": knowledge_result,
                    "document_structure_result": document_structure_result,
                    "cross_page_context_result": cross_page_context_result,
                    "page_raws": page_raws,
                },
                fallback_fn=lambda: self._fallback_context_expansion(
                    query=query,
                    retrieval_result=retrieval_result,
                    graph_retrieval_result=graph_retrieval_result,
                    table_retrieval_result=table_retrieval_result,
                    knowledge_result=knowledge_result,
                    page_raws=page_raws,
                ),
                errors=errors,
            )

        if self.config.run_evidence_aggregator:
            evidence_aggregation_result = self._run_step(
                step_name="EvidenceAggregator",
                component=self.evidence_aggregator,
                kwargs={
                    "query": query,
                    "retrieval_result": retrieval_result,
                    "graph_retrieval_result": graph_retrieval_result,
                    "table_retrieval_result": table_retrieval_result,
                    "context_expansion_result": context_expansion_result,
                    "knowledge_result": knowledge_result,
                    "top_k": self.config.top_k_evidence,
                },
                fallback_fn=lambda: self._fallback_evidence_aggregation(
                    query=query,
                    retrieval_result=retrieval_result,
                    graph_retrieval_result=graph_retrieval_result,
                    table_retrieval_result=table_retrieval_result,
                    context_expansion_result=context_expansion_result,
                    top_k=self.config.top_k_evidence,
                ),
                errors=errors,
            )

        if self.config.run_prompt_builder:
            prompt_result = self._run_step(
                step_name="PromptBuilder",
                component=self.prompt_builder,
                kwargs={
                    "query": query,
                    "query_classification": query_classification,
                    "query_route": query_route,
                    "evidence_aggregation_result": evidence_aggregation_result,
                    "context_expansion_result": context_expansion_result,
                    "extra_context": extra_context,
                },
                fallback_fn=lambda: self._fallback_prompt_builder(
                    query=query,
                    evidence_aggregation_result=evidence_aggregation_result,
                ),
                errors=errors,
            )

        if self.config.run_llm_reasoner:
            answer_result = self._run_step(
                step_name="LLMReasoner",
                component=self.llm_reasoner,
                kwargs={
                    "query": query,
                    "prompt_result": prompt_result,
                    "evidence_aggregation_result": evidence_aggregation_result,
                    "context_expansion_result": context_expansion_result,
                    "query_classification": query_classification,
                },
                fallback_fn=lambda: self._fallback_answer(
                    query=query,
                    evidence_aggregation_result=evidence_aggregation_result,
                    prompt_result=prompt_result,
                ),
                errors=errors,
            )

        if self.config.run_citation_verifier:
            citation_verification_result = self._run_step(
                step_name="CitationVerifier",
                component=self.citation_verifier,
                kwargs={
                    "query": query,
                    "answer_result": answer_result,
                    "evidence_aggregation_result": evidence_aggregation_result,
                    "knowledge_result": knowledge_result,
                },
                fallback_fn=lambda: self._fallback_citation_verification(
                    answer_result=answer_result,
                    evidence_aggregation_result=evidence_aggregation_result,
                ),
                errors=errors,
            )

        result = {
            "processor": "RAGPipeline",
            "schema_version": "rag_pipeline_v1",
            "query": query,
            "query_classification": query_classification,
            "query_route": query_route,
            "retrieval_result": retrieval_result,
            "graph_retrieval_result": graph_retrieval_result,
            "table_retrieval_result": table_retrieval_result,
            "context_expansion_result": context_expansion_result,
            "evidence_aggregation_result": evidence_aggregation_result,
            "prompt_result": prompt_result,
            "answer_result": answer_result,
            "citation_verification_result": citation_verification_result,
            "answer": answer_result.get("answer", ""),
            "citations": citation_verification_result.get("verified_citations", [])
            or answer_result.get("citations", [])
            or evidence_aggregation_result.get("citations", []),
            "rag_summary": self._build_summary(
                query_classification=query_classification,
                query_route=query_route,
                retrieval_result=retrieval_result,
                graph_retrieval_result=graph_retrieval_result,
                table_retrieval_result=table_retrieval_result,
                context_expansion_result=context_expansion_result,
                evidence_aggregation_result=evidence_aggregation_result,
                answer_result=answer_result,
                citation_verification_result=citation_verification_result,
                errors=errors,
            ),
            "errors": errors,
            "config": asdict(self.config),
        }

        if self.config.save_json:
            self.save_rag_result(result)

        return result

    def _run_step(
        self,
        step_name: str,
        component: Any,
        kwargs: Dict[str, Any],
        fallback_fn: Any,
        errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if component is None:
            result = fallback_fn()

            if isinstance(result, dict):
                result.setdefault("processor", step_name)
                result.setdefault("mode", "fallback")
                result.setdefault("warning", f"{step_name} component not found, fallback output generated.")

            return result

        try:
            if hasattr(component, "process"):
                return self._safe_call(component.process, kwargs)

            if hasattr(component, "run"):
                return self._safe_call(component.run, kwargs)

            if hasattr(component, "retrieve"):
                return self._safe_call(component.retrieve, kwargs)

            if hasattr(component, "build"):
                return self._safe_call(component.build, kwargs)

            if hasattr(component, "answer"):
                return self._safe_call(component.answer, kwargs)

            if callable(component):
                return self._safe_call(component, kwargs)

            raise RuntimeError(f"{step_name} component has no callable method.")

        except Exception as exc:
            error = {
                "step": step_name,
                "error": str(exc),
            }

            if self.config.include_debug:
                error["traceback"] = traceback.format_exc()

            errors.append(error)

            if not self.config.continue_on_error:
                raise

            result = fallback_fn()

            if isinstance(result, dict):
                result.setdefault("processor", step_name)
                result.setdefault("mode", "fallback_after_error")
                result.setdefault("error", str(exc))

            return result

    def _safe_call(
        self,
        fn: Any,
        kwargs: Dict[str, Any],
    ) -> Any:
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }

        try:
            signature = inspect.signature(fn)
            parameters = signature.parameters

            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )

            if accepts_kwargs:
                return fn(**kwargs)

            accepted_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in parameters
            }

            if accepted_kwargs:
                return fn(**accepted_kwargs)

            for key in ["query", "retrieval_result", "evidence_aggregation_result"]:
                if key in kwargs:
                    return fn(kwargs[key])

            return fn()

        except ValueError:
            return fn(**kwargs)

    def _load_component(
        self,
        module_path: str,
        class_name: str,
    ) -> Any:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            return cls()
        except Exception:
            return None

    def _fallback_query_classification(
        self,
        query: str,
    ) -> Dict[str, Any]:
        normalized = self._normalize_match_text(query)

        query_type = "general_question"
        intent = "answer"
        answer_style = "short_extractive"

        if any(token in normalized for token in ["bang", "cot", "dong", "table", "bieu"]):
            query_type = "table_question"
            answer_style = "table_aware"

        elif any(token in normalized for token in ["bao nhieu", "so luong", "tong", "dem", "count", "total"]):
            query_type = "count_or_metric_question"
            answer_style = "metric_focused"

        elif any(token in normalized for token in ["la gi", "dinh nghia", "khai niem", "what is"]):
            query_type = "definition_question"
            answer_style = "definition"

        elif any(token in normalized for token in ["can cu", "quyet dinh", "nghi dinh", "thong tu", "luat", "phap ly"]):
            query_type = "legal_reference_question"
            answer_style = "citation_focused"

        elif any(token in normalized for token in ["so sanh", "khac nhau", "compare"]):
            query_type = "comparison_question"
            answer_style = "comparison"

        elif any(token in normalized for token in ["tom tat", "tong hop", "summary", "summarize"]):
            query_type = "summary_question"
            answer_style = "summary"

        elif any(token in normalized for token in ["vi sao", "nguyen nhan", "tai sao", "why"]):
            query_type = "reasoning_question"
            answer_style = "reasoned_extractive"

        if any(token in normalized for token in ["tim", "liet ke", "find", "list"]):
            intent = "search_or_list"

        return {
            "processor": "QueryClassifier",
            "mode": "fallback",
            "query": query,
            "normalized_query": normalized,
            "query_type": query_type,
            "intent": intent,
            "answer_style": answer_style,
            "requires_table_retrieval": query_type in ["table_question", "count_or_metric_question"],
            "requires_graph_retrieval": query_type in ["legal_reference_question", "comparison_question", "reasoning_question"],
            "requires_citations": True,
            "confidence": 0.65,
        }

    def _fallback_query_route(
        self,
        query: str,
        query_classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        query_type = query_classification.get("query_type", "general_question")

        routes = ["hybrid_retrieval"]

        if query_classification.get("requires_table_retrieval"):
            routes.append("table_retrieval")

        if query_classification.get("requires_graph_retrieval"):
            routes.append("graph_retrieval")

        if query_type in ["summary_question", "comparison_question"]:
            routes.append("context_expansion")

        return {
            "processor": "QueryRouter",
            "mode": "fallback",
            "query": query,
            "routes": routes,
            "primary_route": routes[0],
            "retrieval_strategy": "hybrid_keyword_semantic_fallback",
            "top_k_chunks": self.config.top_k_chunks,
            "top_k_tables": self.config.top_k_tables,
            "top_k_graph": self.config.top_k_graph,
            "confidence": 0.70,
        }

    def _fallback_hybrid_retrieval(
        self,
        query: str,
        knowledge_result: Dict[str, Any],
        page_raws: List[PageRaw],
        top_k: int,
    ) -> Dict[str, Any]:
        chunks = self._collect_chunks(
            knowledge_result=knowledge_result,
            page_raws=page_raws,
        )

        scored_items = []

        for chunk in chunks:
            text = chunk.get("normalized_text") or chunk.get("text") or ""
            score, details = self._score_text(query, text)

            if score < self.config.min_retrieval_score:
                continue

            scored_items.append(
                {
                    "retrieval_id": make_id("retrieval"),
                    "chunk_id": chunk.get("chunk_id", ""),
                    "source_id": chunk.get("chunk_id", ""),
                    "source_type": chunk.get("chunk_type", "chunk"),
                    "text": chunk.get("text", ""),
                    "normalized_text": chunk.get("normalized_text", ""),
                    "page_numbers": chunk.get("page_numbers", []),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "section_id": chunk.get("section_id", ""),
                    "table_grid_id": chunk.get("table_grid_id", ""),
                    "score": score,
                    "rank": 0,
                    "source": "fallback_hybrid_retrieval",
                    "metadata": {
                        "details": details,
                        "chunk_metadata": chunk.get("metadata", {}),
                    },
                }
            )

        scored_items = sorted(
            scored_items,
            key=lambda item: item["score"],
            reverse=True,
        )[:top_k]

        for index, item in enumerate(scored_items):
            item["rank"] = index + 1

        return {
            "processor": "HybridRetriever",
            "mode": "fallback",
            "query": query,
            "retrieved_items": scored_items,
            "retrieval_summary": {
                "retrieved_count": len(scored_items),
                "candidate_count": len(chunks),
                "top_score": scored_items[0]["score"] if scored_items else 0.0,
            },
        }

    def _fallback_graph_retrieval(
        self,
        query: str,
        knowledge_result: Dict[str, Any],
        cross_page_context_result: Dict[str, Any],
        top_k: int,
    ) -> Dict[str, Any]:
        nodes, edges = self._collect_graph(
            knowledge_result=knowledge_result,
            cross_page_context_result=cross_page_context_result,
        )

        scored_nodes = []

        for node in nodes:
            text = " ".join(
                [
                    str(node.get("label", "")),
                    json.dumps(node.get("metadata", {}), ensure_ascii=False)[:2000],
                ]
            )

            score, details = self._score_text(query, text)

            if score < self.config.min_retrieval_score:
                continue

            scored_nodes.append(
                {
                    "graph_retrieval_id": make_id("graph_ret"),
                    "node_id": node.get("node_id", ""),
                    "node_type": node.get("node_type", ""),
                    "label": node.get("label", ""),
                    "score": score,
                    "rank": 0,
                    "source": "fallback_graph_retrieval",
                    "metadata": {
                        "details": details,
                        "node_metadata": node.get("metadata", {}),
                    },
                }
            )

        scored_nodes = sorted(
            scored_nodes,
            key=lambda item: item["score"],
            reverse=True,
        )[:top_k]

        for index, item in enumerate(scored_nodes):
            item["rank"] = index + 1

        related_edges = []

        node_ids = {
            item["node_id"]
            for item in scored_nodes
            if item.get("node_id")
        }

        for edge in edges:
            if edge.get("source_id") in node_ids or edge.get("target_id") in node_ids:
                related_edges.append(edge)

        return {
            "processor": "GraphRetriever",
            "mode": "fallback",
            "query": query,
            "retrieved_nodes": scored_nodes,
            "related_edges": related_edges[:top_k * 2],
            "graph_retrieval_summary": {
                "retrieved_node_count": len(scored_nodes),
                "related_edge_count": len(related_edges[:top_k * 2]),
                "candidate_node_count": len(nodes),
                "candidate_edge_count": len(edges),
            },
        }

    def _fallback_table_retrieval(
        self,
        query: str,
        knowledge_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        top_k: int,
    ) -> Dict[str, Any]:
        table_items = self._collect_table_items(
            knowledge_result=knowledge_result,
            table_understanding_result=table_understanding_result,
        )

        scored_items = []

        for item in table_items:
            text = item.get("text", "")

            if not text:
                text = json.dumps(item.get("raw", {}), ensure_ascii=False)

            score, details = self._score_text(query, text)

            if score < self.config.min_retrieval_score:
                continue

            scored_items.append(
                {
                    "table_retrieval_id": make_id("table_ret"),
                    "table_grid_id": item.get("table_grid_id", ""),
                    "table_semantic_id": item.get("table_semantic_id", ""),
                    "source_type": item.get("source_type", "table"),
                    "text": text[:3000],
                    "page_numbers": item.get("page_numbers", []),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "score": score,
                    "rank": 0,
                    "source": "fallback_table_retrieval",
                    "metadata": {
                        "details": details,
                        "raw": item.get("raw", {}),
                    },
                }
            )

        scored_items = sorted(
            scored_items,
            key=lambda item: item["score"],
            reverse=True,
        )[:top_k]

        for index, item in enumerate(scored_items):
            item["rank"] = index + 1

        return {
            "processor": "TableRetriever",
            "mode": "fallback",
            "query": query,
            "retrieved_tables": scored_items,
            "table_retrieval_summary": {
                "retrieved_table_count": len(scored_items),
                "candidate_table_item_count": len(table_items),
                "top_score": scored_items[0]["score"] if scored_items else 0.0,
            },
        }

    def _fallback_context_expansion(
        self,
        query: str,
        retrieval_result: Dict[str, Any],
        graph_retrieval_result: Dict[str, Any],
        table_retrieval_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        retrieved_items = retrieval_result.get("retrieved_items", []) or []
        retrieved_tables = table_retrieval_result.get("retrieved_tables", []) or []
        retrieved_nodes = graph_retrieval_result.get("retrieved_nodes", []) or []

        expanded_contexts = []

        all_chunks = self._collect_chunks(
            knowledge_result=knowledge_result,
            page_raws=page_raws,
        )

        chunks_by_page: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in all_chunks:
            for page_number in chunk.get("page_numbers", []) or []:
                page_key = str(page_number)
                chunks_by_page.setdefault(page_key, [])
                chunks_by_page[page_key].append(chunk)

        seen_context_keys = set()

        for item in retrieved_items:
            context_key = item.get("chunk_id") or item.get("source_id") or item.get("retrieval_id")

            if context_key and context_key not in seen_context_keys:
                seen_context_keys.add(context_key)
                expanded_contexts.append(
                    {
                        "context_id": make_id("context"),
                        "context_type": "retrieved_chunk",
                        "text": item.get("text", ""),
                        "page_numbers": item.get("page_numbers", []),
                        "source_id": context_key,
                        "score": item.get("score", 0.0),
                        "source": "fallback_context_expansion",
                        "metadata": item.get("metadata", {}),
                    }
                )

            for page_number in item.get("page_numbers", []) or []:
                neighbor_pages = range(
                    int(page_number) - self.config.context_window_pages,
                    int(page_number) + self.config.context_window_pages + 1,
                )

                for neighbor_page in neighbor_pages:
                    for neighbor_chunk in chunks_by_page.get(str(neighbor_page), [])[:2]:
                        neighbor_key = neighbor_chunk.get("chunk_id", "")

                        if not neighbor_key or neighbor_key in seen_context_keys:
                            continue

                        seen_context_keys.add(neighbor_key)

                        expanded_contexts.append(
                            {
                                "context_id": make_id("context"),
                                "context_type": "neighbor_chunk",
                                "text": neighbor_chunk.get("text", ""),
                                "page_numbers": neighbor_chunk.get("page_numbers", []),
                                "source_id": neighbor_key,
                                "score": max(item.get("score", 0.0) - 0.05, 0.0),
                                "source": "fallback_context_expansion",
                                "metadata": {
                                    "expanded_from": context_key,
                                    "neighbor_page": neighbor_page,
                                },
                            }
                        )

        for table in retrieved_tables:
            table_key = table.get("table_grid_id") or table.get("table_retrieval_id")

            if table_key and table_key not in seen_context_keys:
                seen_context_keys.add(table_key)
                expanded_contexts.append(
                    {
                        "context_id": make_id("context"),
                        "context_type": "retrieved_table",
                        "text": table.get("text", ""),
                        "page_numbers": table.get("page_numbers", []),
                        "source_id": table_key,
                        "score": table.get("score", 0.0),
                        "source": "fallback_context_expansion",
                        "metadata": table.get("metadata", {}),
                    }
                )

        for node in retrieved_nodes:
            node_key = node.get("node_id") or node.get("graph_retrieval_id")

            if node_key and node_key not in seen_context_keys:
                seen_context_keys.add(node_key)
                expanded_contexts.append(
                    {
                        "context_id": make_id("context"),
                        "context_type": "graph_node",
                        "text": node.get("label", ""),
                        "page_numbers": [],
                        "source_id": node_key,
                        "score": node.get("score", 0.0),
                        "source": "fallback_context_expansion",
                        "metadata": node.get("metadata", {}),
                    }
                )

        expanded_contexts = self._limit_context_chars(expanded_contexts)

        return {
            "processor": "ContextExpander",
            "mode": "fallback",
            "query": query,
            "expanded_contexts": expanded_contexts,
            "context_expansion_summary": {
                "expanded_context_count": len(expanded_contexts),
                "total_context_chars": sum(len(item.get("text", "")) for item in expanded_contexts),
            },
        }

    def _fallback_evidence_aggregation(
        self,
        query: str,
        retrieval_result: Dict[str, Any],
        graph_retrieval_result: Dict[str, Any],
        table_retrieval_result: Dict[str, Any],
        context_expansion_result: Dict[str, Any],
        top_k: int,
    ) -> Dict[str, Any]:
        candidates = []

        for item in retrieval_result.get("retrieved_items", []) or []:
            candidates.append(
                self._to_evidence_candidate(
                    item=item,
                    source_type="chunk",
                    text_key="text",
                    score_key="score",
                )
            )

        for item in table_retrieval_result.get("retrieved_tables", []) or []:
            candidates.append(
                self._to_evidence_candidate(
                    item=item,
                    source_type="table",
                    text_key="text",
                    score_key="score",
                )
            )

        for item in context_expansion_result.get("expanded_contexts", []) or []:
            candidates.append(
                self._to_evidence_candidate(
                    item=item,
                    source_type=item.get("context_type", "context"),
                    text_key="text",
                    score_key="score",
                )
            )

        candidates = [
            item for item in candidates
            if self._clean_text(item.get("text", ""))
        ]

        candidates = self._deduplicate_evidence(candidates)

        candidates = sorted(
            candidates,
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )[:top_k]

        evidence = []

        for index, item in enumerate(candidates):
            evidence_id = item.get("evidence_id") or make_id("rag_evidence")
            page_numbers = item.get("page_numbers", []) or []

            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "rank": index + 1,
                    "source_type": item.get("source_type", ""),
                    "source_id": item.get("source_id", ""),
                    "text": item.get("text", ""),
                    "page_numbers": page_numbers,
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                    "score": item.get("score", 0.0),
                    "citation_text": self._make_citation_text(item),
                    "metadata": item.get("metadata", {}),
                }
            )

        citations = [
            {
                "citation_id": make_id("rag_citation"),
                "evidence_id": item.get("evidence_id", ""),
                "citation_text": item.get("citation_text", ""),
                "page_numbers": item.get("page_numbers", []),
                "source_id": item.get("source_id", ""),
                "confidence": min(0.95, max(0.5, item.get("score", 0.0))),
            }
            for item in evidence
        ]

        return {
            "processor": "EvidenceAggregator",
            "mode": "fallback",
            "query": query,
            "evidence": evidence,
            "citations": citations,
            "evidence_aggregation_summary": {
                "candidate_count": len(candidates),
                "evidence_count": len(evidence),
                "citation_count": len(citations),
                "top_score": evidence[0]["score"] if evidence else 0.0,
            },
        }

    def _fallback_prompt_builder(
        self,
        query: str,
        evidence_aggregation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        evidence = evidence_aggregation_result.get("evidence", []) or []

        context_blocks = []

        for item in evidence:
            citation = item.get("citation_text", "")
            text = item.get("text", "")

            context_blocks.append(
                f"[{item.get('rank')}] {citation}\n{text}"
            )

        context_text = "\n\n".join(context_blocks)
        context_text = context_text[: self.config.max_context_chars]

        system_prompt = (
            "Bạn là trợ lý phân tích tài liệu. "
            "Chỉ trả lời dựa trên bằng chứng được cung cấp. "
            "Nếu không đủ bằng chứng, hãy nói rõ là chưa đủ dữ liệu."
        )

        user_prompt = (
            f"Câu hỏi:\n{query}\n\n"
            f"Bằng chứng:\n{context_text}\n\n"
            "Yêu cầu trả lời:\n"
            "- Trả lời ngắn gọn, rõ ý.\n"
            "- Dẫn nguồn theo trang/bảng nếu có.\n"
            "- Không bịa thông tin ngoài bằng chứng."
        )

        return {
            "processor": "PromptBuilder",
            "mode": "fallback",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context_text": context_text,
            "prompt_summary": {
                "evidence_count": len(evidence),
                "context_chars": len(context_text),
            },
        }

    def _fallback_answer(
        self,
        query: str,
        evidence_aggregation_result: Dict[str, Any],
        prompt_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        evidence = evidence_aggregation_result.get("evidence", []) or []

        if not evidence:
            return {
                "processor": "LLMReasoner",
                "mode": "fallback",
                "query": query,
                "answer": "Chưa đủ bằng chứng trong tài liệu đã xử lý để trả lời câu hỏi này.",
                "citations": [],
                "confidence": 0.20,
                "answer_type": "insufficient_evidence",
            }

        query_tokens = set(self._tokenize(query))
        answer_parts = []

        for item in evidence[: min(5, len(evidence))]:
            text = self._clean_text_block(item.get("text", ""))
            citation = item.get("citation_text", "")

            snippet = self._select_relevant_snippet(
                query_tokens=query_tokens,
                text=text,
                max_chars=700,
            )

            if not snippet:
                continue

            if citation:
                answer_parts.append(f"- {snippet} ({citation})")
            else:
                answer_parts.append(f"- {snippet}")

        if not answer_parts:
            first = evidence[0]
            answer_parts.append(
                f"- {self._clean_text_block(first.get('text', ''))[:700]} ({first.get('citation_text', '')})"
            )

        answer = "\n".join(answer_parts)
        answer = answer[: self.config.max_answer_chars]

        citations = [
            {
                "citation_text": item.get("citation_text", ""),
                "evidence_id": item.get("evidence_id", ""),
                "page_numbers": item.get("page_numbers", []),
                "source_id": item.get("source_id", ""),
            }
            for item in evidence
            if item.get("citation_text")
        ]

        confidence = min(
            0.90,
            0.40 + (len(evidence) * 0.05) + min(evidence[0].get("score", 0.0), 0.25),
        )

        return {
            "processor": "LLMReasoner",
            "mode": "fallback_extractive",
            "query": query,
            "answer": answer,
            "citations": citations,
            "confidence": round(confidence, 4),
            "answer_type": "extractive_answer",
            "metadata": {
                "evidence_count": len(evidence),
                "prompt_context_chars": len(prompt_result.get("context_text", "")),
            },
        }

    def _fallback_citation_verification(
        self,
        answer_result: Dict[str, Any],
        evidence_aggregation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        citations = answer_result.get("citations", []) or evidence_aggregation_result.get("citations", []) or []
        evidence = evidence_aggregation_result.get("evidence", []) or []

        evidence_ids = {
            item.get("evidence_id", "")
            for item in evidence
            if item.get("evidence_id")
        }

        verified = []
        unverified = []

        for citation in citations:
            evidence_id = citation.get("evidence_id", "")

            if evidence_id and evidence_id in evidence_ids:
                item = dict(citation)
                item["verified"] = True
                item["verification_status"] = "verified_by_evidence_id"
                item["confidence"] = max(citation.get("confidence", 0.7), 0.7)
                verified.append(item)
            elif citation.get("page_numbers"):
                item = dict(citation)
                item["verified"] = True
                item["verification_status"] = "verified_by_page_reference"
                item["confidence"] = max(citation.get("confidence", 0.6), 0.6)
                verified.append(item)
            else:
                item = dict(citation)
                item["verified"] = False
                item["verification_status"] = "missing_evidence_reference"
                item["confidence"] = citation.get("confidence", 0.3)
                unverified.append(item)

        return {
            "processor": "CitationVerifier",
            "mode": "fallback",
            "verified_citations": verified,
            "unverified_citations": unverified,
            "citation_verification_summary": {
                "citation_count": len(citations),
                "verified_count": len(verified),
                "unverified_count": len(unverified),
                "all_verified": len(unverified) == 0,
            },
        }

    def _collect_chunks(
        self,
        knowledge_result: Dict[str, Any],
        page_raws: List[PageRaw],
    ) -> List[Dict[str, Any]]:
        chunks = []

        if isinstance(knowledge_result, dict):
            chunks.extend(knowledge_result.get("chunks", []) or [])

            chunk_result = knowledge_result.get("chunk_result", {}) or {}
            chunks.extend(chunk_result.get("chunks", []) or [])

            table_chunk_result = knowledge_result.get("table_chunk_result", {}) or {}
            chunks.extend(table_chunk_result.get("chunks", []) or [])
            chunks.extend(table_chunk_result.get("table_chunks", []) or [])

        if chunks:
            return self._deduplicate_by_key(chunks, "chunk_id")

        for page_raw in page_raws:
            text = self._page_text(page_raw)

            if not text:
                continue

            chunks.append(
                {
                    "chunk_id": make_id("page_chunk"),
                    "chunk_type": "page_chunk",
                    "text": text,
                    "normalized_text": self._normalize_text(text),
                    "page_numbers": [page_raw.page_number],
                    "page_start": page_raw.page_number,
                    "page_end": page_raw.page_number,
                    "source": "rag_pipeline_page_fallback",
                    "metadata": {
                        "page_number": page_raw.page_number,
                    },
                }
            )

        return chunks

    def _collect_table_items(
        self,
        knowledge_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        table_chunk_result = knowledge_result.get("table_chunk_result", {}) or {}

        for chunk in table_chunk_result.get("table_chunks", []) or table_chunk_result.get("chunks", []) or []:
            items.append(
                {
                    "source_type": "table_chunk",
                    "table_grid_id": chunk.get("table_grid_id", ""),
                    "table_semantic_id": chunk.get("table_semantic_id", ""),
                    "text": chunk.get("text", ""),
                    "page_numbers": chunk.get("page_numbers", []),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "raw": chunk,
                }
            )

        for table in table_understanding_result.get("table_semantics", []) or []:
            text = " ".join(
                [
                    table.get("title", ""),
                    table.get("caption", ""),
                    table.get("semantic_type", ""),
                    table.get("text_preview", ""),
                ]
            )

            items.append(
                {
                    "source_type": "table_semantic",
                    "table_grid_id": table.get("table_grid_id", ""),
                    "table_semantic_id": table.get("table_semantic_id", ""),
                    "text": text,
                    "page_numbers": [table.get("page_number")] if table.get("page_number") else [],
                    "page_start": table.get("page_number"),
                    "page_end": table.get("page_number"),
                    "raw": table,
                }
            )

        for record in table_understanding_result.get("table_records", []) or []:
            raw_values = record.get("raw_values", {}) or {}
            text = " | ".join(
                f"{key}: {value}"
                for key, value in raw_values.items()
                if str(value).strip()
            )

            items.append(
                {
                    "source_type": "table_record",
                    "table_grid_id": record.get("table_grid_id", ""),
                    "table_semantic_id": "",
                    "text": text,
                    "page_numbers": [record.get("page_number")] if record.get("page_number") else [],
                    "page_start": record.get("page_number"),
                    "page_end": record.get("page_number"),
                    "raw": record,
                }
            )

        return items

    def _collect_graph(
        self,
        knowledge_result: Dict[str, Any],
        cross_page_context_result: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        nodes = []
        edges = []

        knowledge_graph = knowledge_result.get("knowledge_graph", {}) or {}

        nodes.extend(knowledge_graph.get("nodes", []) or [])
        edges.extend(knowledge_graph.get("edges", []) or [])

        context_graph = cross_page_context_result.get("context_graph", {}) or {}

        nodes.extend(context_graph.get("nodes", []) or [])
        edges.extend(context_graph.get("edges", []) or [])

        return (
            self._deduplicate_by_key(nodes, "node_id"),
            self._deduplicate_edges(edges),
        )

    def _to_evidence_candidate(
        self,
        item: Dict[str, Any],
        source_type: str,
        text_key: str,
        score_key: str,
    ) -> Dict[str, Any]:
        source_id = (
            item.get("source_id")
            or item.get("chunk_id")
            or item.get("table_grid_id")
            or item.get("node_id")
            or item.get("retrieval_id")
            or item.get("table_retrieval_id")
            or item.get("context_id")
            or ""
        )

        return {
            "evidence_id": item.get("evidence_id") or make_id("rag_evidence"),
            "source_type": source_type,
            "source_id": source_id,
            "text": item.get(text_key, ""),
            "page_numbers": item.get("page_numbers", []),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "score": item.get(score_key, 0.0),
            "metadata": item.get("metadata", {}),
        }

    def _score_text(
        self,
        query: str,
        text: str,
    ) -> Tuple[float, Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        text_tokens = self._tokenize(text)

        if not query_tokens or not text_tokens:
            return 0.0, {
                "token_overlap": 0,
                "query_token_count": len(query_tokens),
                "text_token_count": len(text_tokens),
            }

        query_set = set(query_tokens)
        text_set = set(text_tokens)

        overlap = query_set.intersection(text_set)

        overlap_score = len(overlap) / max(len(query_set), 1)

        phrase_score = 0.0
        normalized_query = self._normalize_match_text(query)
        normalized_text = self._normalize_match_text(text)

        if normalized_query and normalized_query in normalized_text:
            phrase_score = 0.35

        bm25_like = 0.0
        text_token_counts: Dict[str, int] = {}

        for token in text_tokens:
            text_token_counts[token] = text_token_counts.get(token, 0) + 1

        for token in query_set:
            tf = text_token_counts.get(token, 0)

            if tf > 0:
                bm25_like += (tf * 2.0) / (tf + 1.2)

        bm25_like = bm25_like / max(len(query_set), 1)

        score = (0.60 * overlap_score) + (0.25 * min(bm25_like, 1.0)) + phrase_score
        score = round(max(0.0, min(score, 1.0)), 4)

        return score, {
            "token_overlap": len(overlap),
            "overlap_tokens": sorted(list(overlap))[:20],
            "query_token_count": len(query_set),
            "text_token_count": len(text_set),
            "overlap_score": round(overlap_score, 4),
            "bm25_like": round(bm25_like, 4),
            "phrase_score": round(phrase_score, 4),
        }

    def _limit_context_chars(
        self,
        contexts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        contexts = sorted(
            contexts,
            key=lambda item: item.get("score", 0.0),
            reverse=True,
        )

        selected = []
        total_chars = 0

        for context in contexts:
            text = context.get("text", "")
            next_len = len(text)

            if total_chars + next_len > self.config.max_context_chars and selected:
                break

            selected.append(context)
            total_chars += next_len

        return selected

    def _deduplicate_evidence(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for item in items:
            key = item.get("source_id") or self._normalize_match_text(item.get("text", ""))[:120]

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _deduplicate_by_key(
        self,
        items: List[Dict[str, Any]],
        key: str,
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for item in items:
            item_key = item.get(key, "")

            if item_key and item_key in seen:
                continue

            if item_key:
                seen.add(item_key)

            result.append(item)

        return result

    def _deduplicate_edges(
        self,
        edges: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for edge in edges:
            key = (
                edge.get("source_id", ""),
                edge.get("target_id", ""),
                edge.get("edge_type", ""),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(edge)

        return result

    def _make_citation_text(
        self,
        item: Dict[str, Any],
    ) -> str:
        page_numbers = item.get("page_numbers", []) or []

        if page_numbers:
            clean_pages = [
                int(page)
                for page in page_numbers
                if str(page).isdigit()
            ]

            if len(clean_pages) == 1:
                return f"trang {clean_pages[0]}"

            if len(clean_pages) > 1:
                return f"trang {min(clean_pages)}-{max(clean_pages)}"

        if item.get("table_grid_id"):
            return f"bảng {item.get('table_grid_id')}"

        return item.get("source_id", "") or "nguồn nội bộ"

    def _select_relevant_snippet(
        self,
        query_tokens: set,
        text: str,
        max_chars: int = 700,
    ) -> str:
        text = self._clean_text_block(text)

        if len(text) <= max_chars:
            return text

        sentences = self._split_sentences(text)

        if not sentences:
            return text[:max_chars].strip()

        scored = []

        for sentence in sentences:
            tokens = set(self._tokenize(sentence))
            score = len(tokens.intersection(query_tokens))

            scored.append(
                {
                    "sentence": sentence,
                    "score": score,
                }
            )

        scored = sorted(
            scored,
            key=lambda item: item["score"],
            reverse=True,
        )

        selected_sentences = []
        total_chars = 0

        for item in scored:
            sentence = item["sentence"]

            if not sentence:
                continue

            if total_chars + len(sentence) > max_chars and selected_sentences:
                break

            selected_sentences.append(sentence)
            total_chars += len(sentence)

            if len(selected_sentences) >= 3:
                break

        if selected_sentences:
            return " ".join(selected_sentences).strip()

        return text[:max_chars].strip()

    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:
        text = self._clean_text_block(text)

        parts = re.split(r"(?<=[\.\?\!])\s+", text)

        if len(parts) <= 1:
            parts = text.split("\n")

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    def _page_text(
        self,
        page_raw: PageRaw,
    ) -> str:
        reading_meta = page_raw.metadata.get("reading_order_builder", {})
        text = reading_meta.get("reading_order_text", "")

        if text:
            return self._clean_text_block(text)

        return self._clean_text_block(page_raw.normalized_text or page_raw.raw_text or "")

    def _tokenize(
        self,
        text: Any,
    ) -> List[str]:
        text = self._normalize_match_text(text)

        if not text:
            return []

        tokens = re.findall(r"[a-z0-9_]+", text)

        stopwords = {
            "la", "gi", "va", "cua", "cho", "trong", "tren", "duoi",
            "nhung", "cac", "mot", "nhieu", "co", "khong", "duoc",
            "the", "thi", "nay", "do", "voi", "tu", "den", "ve",
            "theo", "can", "hoi", "noi", "dung", "hay", "giup",
            "toi", "ban", "em", "anh", "chi",
            "what", "is", "the", "a", "an", "of", "for", "to", "and",
        }

        return [
            token for token in tokens
            if token not in stopwords and len(token) >= 2
        ]

    def _normalize_text(
        self,
        text: Any,
    ) -> str:
        return self._clean_text_block(text).lower()

    def _normalize_match_text(
        self,
        text: Any,
    ) -> str:
        text = self._clean_text_block(text).lower()

        replacements = {
            "à": "a", "á": "a", "ạ": "a", "ả": "a", "ã": "a",
            "â": "a", "ầ": "a", "ấ": "a", "ậ": "a", "ẩ": "a", "ẫ": "a",
            "ă": "a", "ằ": "a", "ắ": "a", "ặ": "a", "ẳ": "a", "ẵ": "a",
            "è": "e", "é": "e", "ẹ": "e", "ẻ": "e", "ẽ": "e",
            "ê": "e", "ề": "e", "ế": "e", "ệ": "e", "ể": "e", "ễ": "e",
            "ì": "i", "í": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
            "ò": "o", "ó": "o", "ọ": "o", "ỏ": "o", "õ": "o",
            "ô": "o", "ồ": "o", "ố": "o", "ộ": "o", "ổ": "o", "ỗ": "o",
            "ơ": "o", "ờ": "o", "ớ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
            "ù": "u", "ú": "u", "ụ": "u", "ủ": "u", "ũ": "u",
            "ư": "u", "ừ": "u", "ứ": "u", "ự": "u", "ử": "u", "ữ": "u",
            "ỳ": "y", "ý": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y",
            "đ": "d",
        }

        for src, dst in replacements.items():
            text = text.replace(src, dst)

        text = re.sub(r"[^a-z0-9_%]+", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _clean_text_block(
        self,
        text: Any,
    ) -> str:
        if text is None:
            return ""

        text = str(text)
        text = text.replace("\u00a0", " ")
        text = text.replace("Ƣ", "Ư")
        text = text.replace("ƣ", "ư")
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _clean_text(
        self,
        text: Any,
    ) -> str:
        if text is None:
            return ""

        text = str(text)
        text = text.replace("\u00a0", " ")
        text = text.replace("Ƣ", "Ư")
        text = text.replace("ƣ", "ư")
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def _build_summary(
        self,
        query_classification: Dict[str, Any],
        query_route: Dict[str, Any],
        retrieval_result: Dict[str, Any],
        graph_retrieval_result: Dict[str, Any],
        table_retrieval_result: Dict[str, Any],
        context_expansion_result: Dict[str, Any],
        evidence_aggregation_result: Dict[str, Any],
        answer_result: Dict[str, Any],
        citation_verification_result: Dict[str, Any],
        errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "query_type": query_classification.get("query_type", ""),
            "intent": query_classification.get("intent", ""),
            "routes": query_route.get("routes", []),
            "retrieved_chunk_count": len(retrieval_result.get("retrieved_items", []) or []),
            "retrieved_graph_node_count": len(graph_retrieval_result.get("retrieved_nodes", []) or []),
            "retrieved_table_count": len(table_retrieval_result.get("retrieved_tables", []) or []),
            "expanded_context_count": len(context_expansion_result.get("expanded_contexts", []) or []),
            "evidence_count": len(evidence_aggregation_result.get("evidence", []) or []),
            "citation_count": len(evidence_aggregation_result.get("citations", []) or []),
            "verified_citation_count": len(citation_verification_result.get("verified_citations", []) or []),
            "answer_type": answer_result.get("answer_type", ""),
            "answer_confidence": answer_result.get("confidence", 0.0),
            "error_count": len(errors),
            "has_errors": len(errors) > 0,
        }

    def save_rag_result(
        self,
        result: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        if output_path is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "rag_result.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        safe_result = self._json_safe(result)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                safe_result,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return str(output_path)

    def _json_safe(
        self,
        value: Any,
    ) -> Any:
        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        if is_dataclass(value):
            return self._json_safe(asdict(value))

        if isinstance(value, dict):
            return {
                str(key): self._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [
                self._json_safe(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return [
                self._json_safe(item)
                for item in value
            ]

        if hasattr(value, "to_dict"):
            try:
                return self._json_safe(value.to_dict())
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            try:
                return self._json_safe(vars(value))
            except Exception:
                pass

        return str(value)


def run_rag_pipeline(
    query: str,
    page_raws: Optional[List[PageRaw]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
    indexing_result: Optional[Dict[str, Any]] = None,
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    cross_page_context_result: Optional[Dict[str, Any]] = None,
    extra_context: Optional[Dict[str, Any]] = None,
    config: Optional[RAGPipelineConfig] = None,
) -> Dict[str, Any]:
    pipeline = RAGPipeline(config=config)
    return pipeline.process(
        query=query,
        page_raws=page_raws,
        knowledge_result=knowledge_result,
        indexing_result=indexing_result,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
        cross_page_context_result=cross_page_context_result,
        extra_context=extra_context,
    )


def answer_question(
    query: str,
    page_raws: Optional[List[PageRaw]] = None,
    knowledge_result: Optional[Dict[str, Any]] = None,
    indexing_result: Optional[Dict[str, Any]] = None,
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    cross_page_context_result: Optional[Dict[str, Any]] = None,
    extra_context: Optional[Dict[str, Any]] = None,
    config: Optional[RAGPipelineConfig] = None,
) -> Dict[str, Any]:
    return run_rag_pipeline(
        query=query,
        page_raws=page_raws,
        knowledge_result=knowledge_result,
        indexing_result=indexing_result,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
        cross_page_context_result=cross_page_context_result,
        extra_context=extra_context,
        config=config,
    )
