"""
query_classifier.py

Production V1 - Colab Ready

Purpose
-------
Classify user query for RAG routing.

Used by:
- RAGPipeline
- QueryRouter
- HybridRetriever
- GraphRetriever
- TableRetriever
- PromptBuilder

Input
-----
- query

Output
------
Dictionary with:
- query_type
- query_intent
- query_language
- query_scope
- query_entities
- query_filters
- retrieval_strategy
- routing_hints
- prompt_hints
- query_classification_summary
"""

from __future__ import annotations

import json
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
class QueryClassifierConfig:
    default_language: str = "vi"

    enable_language_detection: bool = True
    enable_intent_detection: bool = True
    enable_entity_extraction: bool = True
    enable_filter_extraction: bool = True
    enable_routing_hints: bool = True
    enable_prompt_hints: bool = True
    enable_complexity_estimation: bool = True

    min_token_len: int = 2
    max_entity_len: int = 80
    max_entities: int = 30
    max_keywords: int = 30

    prefer_hybrid_default: bool = True
    require_citations_default: bool = True
    include_debug: bool = True


class QueryClassifier:
    def __init__(
        self,
        config: Optional[QueryClassifierConfig] = None,
    ):
        self.config = config or QueryClassifierConfig()
        self.stopwords_vi = self._stopwords_vi()
        self.stopwords_en = self._stopwords_en()

    def process(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        query = normalize_pdf_text(query)
        context = context or {}

        normalized_query = normalize_text_for_match(query)
        tokens = self._tokenize(query)
        keywords = self._extract_keywords(query)

        query_language = self._detect_language(query) if self.config.enable_language_detection else self.config.default_language

        intent_scores = self._detect_intents(query=query, normalized_query=normalized_query, tokens=tokens)
        query_intent = self._select_primary_intent(intent_scores)
        query_type = self._infer_query_type(query=query, normalized_query=normalized_query, intent_scores=intent_scores)

        query_scope = self._infer_scope(query=query, normalized_query=normalized_query)
        answer_format = self._infer_answer_format(query=query, normalized_query=normalized_query, query_intent=query_intent)
        expected_answer_type = self._infer_expected_answer_type(query_intent=query_intent, query_type=query_type)

        entities = self._extract_entities(query=query, tokens=tokens) if self.config.enable_entity_extraction else []
        filters = self._extract_filters(query=query, normalized_query=normalized_query) if self.config.enable_filter_extraction else {}

        complexity = self._estimate_complexity(
            query=query,
            tokens=tokens,
            entities=entities,
            filters=filters,
            intent_scores=intent_scores,
        )

        retrieval_strategy = self._build_retrieval_strategy(
            query=query,
            query_type=query_type,
            query_intent=query_intent,
            normalized_query=normalized_query,
            filters=filters,
            complexity=complexity,
        )

        routing_hints = self._build_routing_hints(
            query_type=query_type,
            query_intent=query_intent,
            query_scope=query_scope,
            filters=filters,
            retrieval_strategy=retrieval_strategy,
            complexity=complexity,
        )

        prompt_hints = self._build_prompt_hints(
            query_type=query_type,
            query_intent=query_intent,
            answer_format=answer_format,
            expected_answer_type=expected_answer_type,
            query_language=query_language,
            complexity=complexity,
        )

        result = {
            "processor": "QueryClassifier",
            "schema_version": "query_classifier_v1",
            "query": query,
            "normalized_query": normalized_query,
            "query_language": query_language,
            "query_type": query_type,
            "query_intent": query_intent,
            "intent_scores": intent_scores,
            "query_scope": query_scope,
            "answer_format": answer_format,
            "expected_answer_type": expected_answer_type,
            "query_tokens": tokens,
            "query_keywords": keywords,
            "query_entities": entities,
            "query_filters": filters,
            "retrieval_strategy": retrieval_strategy,
            "routing_hints": routing_hints,
            "prompt_hints": prompt_hints,
            "query_complexity": complexity,
            "query_classification_summary": self._build_summary(
                query=query,
                query_language=query_language,
                query_type=query_type,
                query_intent=query_intent,
                query_scope=query_scope,
                answer_format=answer_format,
                entities=entities,
                filters=filters,
                retrieval_strategy=retrieval_strategy,
                complexity=complexity,
            ),
            "config": asdict(self.config),
        }

        return json_safe(result)

    def _detect_language(
        self,
        query: str,
    ) -> str:
        query_norm = query.lower()

        vietnamese_marks = re.findall(
            r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]",
            query_norm,
        )

        vi_words = [
            "là", "gì", "như", "thế", "nào", "tại", "sao", "vì", "sao",
            "liệt", "kê", "tóm", "tắt", "so", "sánh", "trích", "dẫn",
            "bảng", "trang", "mục", "nội", "dung", "tài", "liệu",
        ]

        en_words = [
            "what", "why", "how", "when", "where", "which", "summarize",
            "compare", "list", "extract", "table", "page", "section",
            "document", "evidence",
        ]

        vi_score = len(vietnamese_marks) * 2
        vi_score += sum(1 for word in vi_words if re.search(rf"\b{re.escape(word)}\b", query_norm))

        en_score = sum(1 for word in en_words if re.search(rf"\b{re.escape(word)}\b", query_norm))

        if vi_score > en_score:
            return "vi"

        if en_score > vi_score:
            return "en"

        return self.config.default_language

    def _detect_intents(
        self,
        query: str,
        normalized_query: str,
        tokens: List[str],
    ) -> Dict[str, float]:
        patterns = {
            "definition": [
                r"\bla gi\b",
                r"\bla gi vay\b",
                r"\bkhai niem\b",
                r"\bdinh nghia\b",
                r"\bwhat is\b",
                r"\bdefine\b",
                r"\bdefinition\b",
            ],
            "summary": [
                r"\btom tat\b",
                r"\btong hop\b",
                r"\bnoi dung chinh\b",
                r"\by chinh\b",
                r"\bkhai quat\b",
                r"\bsummarize\b",
                r"\bsummary\b",
                r"\boverview\b",
            ],
            "extraction": [
                r"\btrich xuat\b",
                r"\blay ra\b",
                r"\btim giup\b",
                r"\bneu ro\b",
                r"\bextract\b",
                r"\bfind\b",
                r"\bidentify\b",
            ],
            "list": [
                r"\bliet ke\b",
                r"\bdanh sach\b",
                r"\bcac noi dung\b",
                r"\bnhung noi dung\b",
                r"\bgom nhung\b",
                r"\bbao gom\b",
                r"\blist\b",
                r"\benumerate\b",
            ],
            "comparison": [
                r"\bso sanh\b",
                r"\bkhac nhau\b",
                r"\bgiong nhau\b",
                r"\bdiem khac\b",
                r"\bdiem giong\b",
                r"\bcompare\b",
                r"\bdifference\b",
                r"\bsimilar\b",
            ],
            "reasoning": [
                r"\btai sao\b",
                r"\bvi sao\b",
                r"\bnguyen nhan\b",
                r"\bly do\b",
                r"\bphan tich\b",
                r"\bdanh gia\b",
                r"\bwhy\b",
                r"\breason\b",
                r"\banalyze\b",
                r"\bevaluate\b",
            ],
            "procedure": [
                r"\blam the nao\b",
                r"\bcach\b",
                r"\bquy trinh\b",
                r"\bcac buoc\b",
                r"\bhuong dan\b",
                r"\bhow to\b",
                r"\bprocedure\b",
                r"\bsteps\b",
                r"\bguide\b",
            ],
            "yes_no": [
                r"\bco phai\b",
                r"\bco dung\b",
                r"\bco xung dot\b",
                r"\bco can\b",
                r"\bco nen\b",
                r"\bis it\b",
                r"\bdoes\b",
                r"\bdo\b",
                r"\bshould\b",
                r"\bcan\b",
            ],
            "table_query": [
                r"\bbang\b",
                r"\bcot\b",
                r"\bdong\b",
                r"\bo nao\b",
                r"\btable\b",
                r"\bcolumn\b",
                r"\brow\b",
                r"\bcell\b",
            ],
            "counting": [
                r"\bbao nhieu\b",
                r"\bso luong\b",
                r"\bdem\b",
                r"\btong so\b",
                r"\bcount\b",
                r"\bhow many\b",
                r"\bnumber of\b",
                r"\btotal\b",
            ],
            "citation": [
                r"\btrich dan\b",
                r"\bdan chung\b",
                r"\bcan cu\b",
                r"\bbang chung\b",
                r"\bcitation\b",
                r"\bevidence\b",
                r"\bsource\b",
                r"\bproof\b",
            ],
            "page_lookup": [
                r"\btrang\b",
                r"\bpage\b",
                r"\btr\.\s*\d+",
                r"\bp\.\s*\d+",
            ],
            "graph_query": [
                r"\blien quan\b",
                r"\bmoi quan he\b",
                r"\bket noi\b",
                r"\blien ket\b",
                r"\btham chieu\b",
                r"\brelationship\b",
                r"\bconnected\b",
                r"\blink\b",
                r"\breference\b",
            ],
            "metadata_query": [
                r"\bmetadata\b",
                r"\bthong tin tai lieu\b",
                r"\bloai tai lieu\b",
                r"\bten file\b",
                r"\bfile name\b",
                r"\bdocument type\b",
            ],
        }

        scores = {}

        for intent, intent_patterns in patterns.items():
            score = 0.0

            for pattern in intent_patterns:
                matches = re.findall(pattern, normalized_query, flags=re.IGNORECASE)
                score += len(matches)

            scores[intent] = round(score, 4)

        if not any(score > 0 for score in scores.values()):
            scores["factual"] = 1.0
        else:
            scores["factual"] = 0.3

        if "?" in query:
            scores["question"] = 1.0

        if len(tokens) <= 5:
            scores["short_query"] = 0.6

        return scores

    def _select_primary_intent(
        self,
        intent_scores: Dict[str, float],
    ) -> str:
        if not intent_scores:
            return "factual"

        priority = [
            "table_query",
            "citation",
            "comparison",
            "summary",
            "extraction",
            "list",
            "counting",
            "definition",
            "procedure",
            "reasoning",
            "yes_no",
            "graph_query",
            "metadata_query",
            "page_lookup",
            "factual",
        ]

        best_intent = "factual"
        best_score = -1.0

        for intent in priority:
            score = intent_scores.get(intent, 0.0)

            if score > best_score:
                best_intent = intent
                best_score = score

        if best_score <= 0:
            return "factual"

        return best_intent

    def _infer_query_type(
        self,
        query: str,
        normalized_query: str,
        intent_scores: Dict[str, float],
    ) -> str:
        if intent_scores.get("table_query", 0) > 0:
            return "table"

        if intent_scores.get("citation", 0) > 0:
            return "evidence"

        if intent_scores.get("graph_query", 0) > 0:
            return "graph"

        if intent_scores.get("metadata_query", 0) > 0:
            return "metadata"

        if intent_scores.get("comparison", 0) > 0:
            return "comparison"

        if intent_scores.get("summary", 0) > 0:
            return "summary"

        if intent_scores.get("counting", 0) > 0:
            return "count"

        if intent_scores.get("yes_no", 0) > 0:
            return "yes_no"

        if intent_scores.get("procedure", 0) > 0:
            return "procedure"

        if intent_scores.get("definition", 0) > 0:
            return "definition"

        if intent_scores.get("reasoning", 0) > 0:
            return "analysis"

        if self._has_numeric_or_page_reference(normalized_query):
            return "lookup"

        return "factual"

    def _infer_scope(
        self,
        query: str,
        normalized_query: str,
    ) -> str:
        if re.search(r"\btoan bo\b|\bca tai lieu\b|\ball document\b|\bwhole document\b", normalized_query):
            return "document"

        if re.search(r"\btrang\s*\d+|\bpage\s*\d+|\btr\.\s*\d+|\bp\.\s*\d+", normalized_query):
            return "page"

        if re.search(r"\bmuc\b|\bphan\b|\bchuong\b|\bsection\b|\bchapter\b", normalized_query):
            return "section"

        if re.search(r"\bbang\b|\btable\b", normalized_query):
            return "table"

        if re.search(r"\bdoan\b|\bparagraph\b|\bchunk\b", normalized_query):
            return "chunk"

        return "open"

    def _infer_answer_format(
        self,
        query: str,
        normalized_query: str,
        query_intent: str,
    ) -> str:
        if re.search(r"\bbang markdown\b|\bdang bang\b|\btable format\b|\bmarkdown table\b", normalized_query):
            return "table"

        if query_intent in ["list", "procedure", "comparison"]:
            return "bullets"

        if query_intent == "summary":
            return "structured"

        if query_intent == "yes_no":
            return "direct_then_explain"

        if re.search(r"\bngan gon\b|\btom gon\b|\bbrief\b|\bconcise\b", normalized_query):
            return "concise"

        return "structured"

    def _infer_expected_answer_type(
        self,
        query_intent: str,
        query_type: str,
    ) -> str:
        mapping = {
            "definition": "definition_answer",
            "summary": "summary_answer",
            "extraction": "extraction_answer",
            "list": "list_answer",
            "comparison": "comparison_answer",
            "reasoning": "analysis_answer",
            "procedure": "procedure_answer",
            "yes_no": "yes_no_answer",
            "table_query": "table_answer",
            "counting": "count_answer",
            "citation": "evidence_answer",
            "graph_query": "relationship_answer",
            "metadata_query": "metadata_answer",
        }

        if query_intent in mapping:
            return mapping[query_intent]

        type_mapping = {
            "table": "table_answer",
            "graph": "relationship_answer",
            "metadata": "metadata_answer",
            "evidence": "evidence_answer",
            "summary": "summary_answer",
            "comparison": "comparison_answer",
            "count": "count_answer",
            "yes_no": "yes_no_answer",
            "procedure": "procedure_answer",
            "definition": "definition_answer",
            "analysis": "analysis_answer",
        }

        return type_mapping.get(query_type, "grounded_answer")

    def _extract_entities(
        self,
        query: str,
        tokens: List[str],
    ) -> List[Dict[str, Any]]:
        entities = []

        quoted = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', query)

        for match in quoted:
            value = next((item for item in match if item), "")
            value = normalize_pdf_text(value)

            if value:
                entities.append(
                    {
                        "text": value,
                        "entity_type": "quoted_phrase",
                        "confidence": 0.95,
                        "source": "quote_pattern",
                    }
                )

        code_like = re.findall(r"\b[A-ZĐ]{2,}[A-Z0-9_\-\.]*\b", query)

        for value in code_like:
            value = normalize_pdf_text(value)

            if len(value) > self.config.max_entity_len:
                continue

            entities.append(
                {
                    "text": value,
                    "entity_type": self._infer_code_entity_type(value),
                    "confidence": 0.80,
                    "source": "uppercase_pattern",
                }
            )

        title_like = re.findall(
            r"\b(?:[A-ZÀ-ỴĐ][\wÀ-ỹĐđ\-]+(?:\s+|$)){2,6}",
            query,
        )

        for value in title_like:
            value = normalize_pdf_text(value)

            if not value:
                continue

            if len(value) > self.config.max_entity_len:
                continue

            if value.lower() in ["copy nguyên", "colab"]:
                continue

            entities.append(
                {
                    "text": value,
                    "entity_type": "proper_noun_phrase",
                    "confidence": 0.60,
                    "source": "capitalized_phrase",
                }
            )

        numeric_entities = re.findall(
            r"\b\d+(?:[.,]\d+)?(?:\s*%|\s*tỷ|\s*triệu|\s*nghìn|\s*trang|\s*page)?\b",
            query,
            flags=re.IGNORECASE,
        )

        for value in numeric_entities:
            value = normalize_pdf_text(value)

            entities.append(
                {
                    "text": value,
                    "entity_type": "number",
                    "confidence": 0.70,
                    "source": "number_pattern",
                }
            )

        entities = self._deduplicate_entities(entities)

        return entities[: self.config.max_entities]

    def _extract_filters(
        self,
        query: str,
        normalized_query: str,
    ) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}

        page_numbers = self._extract_page_numbers(query, normalized_query)

        if page_numbers:
            filters["page_numbers"] = page_numbers

        section_refs = self._extract_section_refs(query, normalized_query)

        if section_refs:
            filters["section_refs"] = section_refs

        table_refs = self._extract_table_refs(query, normalized_query)

        if table_refs:
            filters["table_refs"] = table_refs

        quoted_phrases = self._extract_quoted_phrases(query)

        if quoted_phrases:
            filters["quoted_phrases"] = quoted_phrases

        years = self._extract_years(query)

        if years:
            filters["years"] = years

        date_refs = self._extract_date_refs(query)

        if date_refs:
            filters["date_refs"] = date_refs

        if re.search(r"\btrong bang\b|\bin table\b", normalized_query):
            filters["prefer_table"] = True

        if re.search(r"\btrich dan\b|\bcitation\b|\bevidence\b|\bdan chung\b|\bcan cu\b", normalized_query):
            filters["require_evidence"] = True

        return filters

    def _build_retrieval_strategy(
        self,
        query: str,
        query_type: str,
        query_intent: str,
        normalized_query: str,
        filters: Dict[str, Any],
        complexity: Dict[str, Any],
    ) -> Dict[str, Any]:
        strategy = {
            "strategy_name": "hybrid",
            "use_bm25": True,
            "use_vector": True,
            "use_graph": False,
            "use_metadata": False,
            "use_table": False,
            "use_context_expansion": True,
            "use_evidence_aggregation": True,
            "use_citation_verification": True,
            "top_k": 20,
            "candidate_pool_size": 80,
            "rerank": True,
            "reason": [],
        }

        if query_type in ["table"] or query_intent == "table_query" or filters.get("prefer_table"):
            strategy["use_table"] = True
            strategy["use_bm25"] = True
            strategy["use_vector"] = True
            strategy["reason"].append("table_query_detected")

        if query_type in ["graph"] or query_intent == "graph_query":
            strategy["use_graph"] = True
            strategy["use_context_expansion"] = True
            strategy["reason"].append("relationship_query_detected")

        if query_type in ["metadata"] or query_intent == "metadata_query":
            strategy["use_metadata"] = True
            strategy["use_bm25"] = False
            strategy["use_vector"] = False
            strategy["reason"].append("metadata_query_detected")

        if query_type in ["evidence"] or query_intent == "citation" or filters.get("require_evidence"):
            strategy["use_evidence_aggregation"] = True
            strategy["use_citation_verification"] = True
            strategy["reason"].append("evidence_required")

        if query_type in ["summary", "comparison", "analysis", "procedure"]:
            strategy["use_context_expansion"] = True
            strategy["candidate_pool_size"] = 120
            strategy["top_k"] = 30
            strategy["reason"].append("broad_context_needed")

        if query_type in ["count", "lookup"]:
            strategy["use_bm25"] = True
            strategy["use_metadata"] = True
            strategy["top_k"] = 30
            strategy["reason"].append("lookup_or_count_query")

        if filters.get("page_numbers") or filters.get("section_refs"):
            strategy["use_metadata"] = True
            strategy["candidate_pool_size"] = 60
            strategy["reason"].append("explicit_filter_detected")

        if complexity.get("level") == "high":
            strategy["use_graph"] = True
            strategy["use_metadata"] = True
            strategy["use_context_expansion"] = True
            strategy["candidate_pool_size"] = 150
            strategy["top_k"] = 40
            strategy["reason"].append("high_complexity_query")

        if not strategy["reason"]:
            strategy["reason"].append("default_hybrid_strategy")

        return strategy

    def _build_routing_hints(
        self,
        query_type: str,
        query_intent: str,
        query_scope: str,
        filters: Dict[str, Any],
        retrieval_strategy: Dict[str, Any],
        complexity: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "preferred_retriever": self._preferred_retriever(query_type, query_intent, retrieval_strategy),
            "use_bm25": retrieval_strategy.get("use_bm25", True),
            "use_vector": retrieval_strategy.get("use_vector", True),
            "use_graph": retrieval_strategy.get("use_graph", False),
            "use_metadata": retrieval_strategy.get("use_metadata", False),
            "use_table": retrieval_strategy.get("use_table", False),
            "need_context_expansion": retrieval_strategy.get("use_context_expansion", True),
            "need_evidence_aggregation": retrieval_strategy.get("use_evidence_aggregation", True),
            "need_citation_verification": retrieval_strategy.get("use_citation_verification", True),
            "top_k": retrieval_strategy.get("top_k", 20),
            "candidate_pool_size": retrieval_strategy.get("candidate_pool_size", 80),
            "page_numbers": filters.get("page_numbers", []),
            "section_refs": filters.get("section_refs", []),
            "table_refs": filters.get("table_refs", []),
            "query_scope": query_scope,
            "complexity_level": complexity.get("level", "medium"),
        }

    def _build_prompt_hints(
        self,
        query_type: str,
        query_intent: str,
        answer_format: str,
        expected_answer_type: str,
        query_language: str,
        complexity: Dict[str, Any],
    ) -> Dict[str, Any]:
        task_mode = "grounded_qa"

        if query_type == "summary":
            task_mode = "summarization"
        elif query_type == "comparison":
            task_mode = "comparison"
        elif query_type in ["table", "count", "lookup"]:
            task_mode = "extraction"
        elif query_type == "analysis":
            task_mode = "analysis"

        answer_style = "formal"

        if answer_format == "concise":
            answer_style = "concise"
        elif query_type in ["table", "metadata"]:
            answer_style = "technical"

        return {
            "language": query_language,
            "task_mode": task_mode,
            "answer_style": answer_style,
            "answer_format": answer_format,
            "expected_answer_type": expected_answer_type,
            "require_citations": self.config.require_citations_default,
            "allow_insufficient_evidence_answer": True,
            "forbid_external_knowledge": True,
            "include_limitations": complexity.get("level") in ["medium", "high"],
        }

    def _estimate_complexity(
        self,
        query: str,
        tokens: List[str],
        entities: List[Dict[str, Any]],
        filters: Dict[str, Any],
        intent_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        score = 0.0
        reasons = []

        if len(tokens) > 18:
            score += 1.0
            reasons.append("long_query")

        if len(entities) >= 3:
            score += 0.8
            reasons.append("multiple_entities")

        if len(filters) >= 2:
            score += 0.7
            reasons.append("multiple_filters")

        if intent_scores.get("comparison", 0) > 0:
            score += 0.8
            reasons.append("comparison")

        if intent_scores.get("reasoning", 0) > 0:
            score += 0.7
            reasons.append("reasoning")

        if intent_scores.get("summary", 0) > 0:
            score += 0.6
            reasons.append("summary")

        if intent_scores.get("table_query", 0) > 0:
            score += 0.6
            reasons.append("table")

        if re.search(r"\bva\b|\band\b|\bhoac\b|\bor\b", normalize_text_for_match(query)):
            score += 0.4
            reasons.append("compound_query")

        if score >= 2.0:
            level = "high"
        elif score >= 0.8:
            level = "medium"
        else:
            level = "low"

        return {
            "level": level,
            "score": round(score, 4),
            "reasons": reasons,
            "token_count": len(tokens),
            "entity_count": len(entities),
            "filter_count": len(filters),
        }

    def _preferred_retriever(
        self,
        query_type: str,
        query_intent: str,
        retrieval_strategy: Dict[str, Any],
    ) -> str:
        if retrieval_strategy.get("strategy_name") == "metadata":
            return "metadata"

        if query_type == "table" or query_intent == "table_query":
            return "table"

        if query_type == "graph" or query_intent == "graph_query":
            return "graph"

        if query_type == "metadata" or query_intent == "metadata_query":
            return "metadata"

        if query_type in ["summary", "analysis", "comparison"]:
            return "hybrid"

        return "hybrid"

    def _extract_page_numbers(
        self,
        query: str,
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

    def _extract_section_refs(
        self,
        query: str,
        normalized_query: str,
    ) -> List[str]:
        refs = []

        patterns = [
            r"\bmuc\s+([ivx\d]+(?:\.\d+)*)",
            r"\bphan\s+([ivx\d]+(?:\.\d+)*)",
            r"\bchuong\s+([ivx\d]+(?:\.\d+)*)",
            r"\bsection\s+([a-z0-9]+(?:\.\d+)*)",
            r"\bchapter\s+([a-z0-9]+(?:\.\d+)*)",
        ]

        for pattern in patterns:
            for value in re.findall(pattern, normalized_query, flags=re.IGNORECASE):
                value = normalize_pdf_text(value)
                if value:
                    refs.append(value)

        return list(dict.fromkeys(refs))

    def _extract_table_refs(
        self,
        query: str,
        normalized_query: str,
    ) -> List[str]:
        refs = []

        patterns = [
            r"\bbang\s+(\d+(?:\.\d+)*)",
            r"\btable\s+(\d+(?:\.\d+)*)",
            r"\bbieu\s+(\d+(?:\.\d+)*)",
        ]

        for pattern in patterns:
            for value in re.findall(pattern, normalized_query, flags=re.IGNORECASE):
                value = normalize_pdf_text(value)
                if value:
                    refs.append(value)

        return list(dict.fromkeys(refs))

    def _extract_quoted_phrases(
        self,
        query: str,
    ) -> List[str]:
        phrases = []

        for match in re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', query):
            value = next((item for item in match if item), "")
            value = normalize_pdf_text(value)
            if value:
                phrases.append(value)

        return list(dict.fromkeys(phrases))

    def _extract_years(
        self,
        query: str,
    ) -> List[int]:
        years = []

        for value in re.findall(r"\b(19\d{2}|20\d{2}|21\d{2})\b", query):
            year = self._safe_int(value, 0)
            if year > 0:
                years.append(year)

        return sorted(list(dict.fromkeys(years)))

    def _extract_date_refs(
        self,
        query: str,
    ) -> List[str]:
        refs = []

        patterns = [
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
            r"\b\d{4}-\d{1,2}-\d{1,2}\b",
            r"\bthang\s*\d{1,2}/\d{4}\b",
            r"\bquy\s*[ivx\d]+/\d{4}\b",
        ]

        for pattern in patterns:
            refs.extend(re.findall(pattern, query, flags=re.IGNORECASE))

        return list(dict.fromkeys([normalize_pdf_text(item) for item in refs if normalize_pdf_text(item)]))

    def _extract_keywords(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        language = self._detect_language(query)
        stopwords = self.stopwords_vi if language == "vi" else self.stopwords_en

        tokens = self._tokenize(query)

        counts: Dict[str, int] = {}

        for token in tokens:
            if token in stopwords:
                continue

            if token.isdigit():
                continue

            counts[token] = counts.get(token, 0) + 1

        ranked = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[: self.config.max_keywords]

        return [
            {
                "keyword": keyword,
                "count": count,
            }
            for keyword, count in ranked
        ]

    def _has_numeric_or_page_reference(
        self,
        normalized_query: str,
    ) -> bool:
        return bool(
            re.search(r"\b\d+\b", normalized_query)
            or re.search(r"\btrang\b|\bpage\b|\btr\.", normalized_query)
        )

    def _infer_code_entity_type(
        self,
        value: str,
    ) -> str:
        if re.match(r"^QĐ|QD|NĐ|ND|TT|NQ|CV|VB", value, flags=re.IGNORECASE):
            return "legal_document_code"

        if re.match(r"^[A-Z]{2,}[A-Z0-9_\-]*$", value):
            return "abbreviation"

        return "code"

    def _deduplicate_entities(
        self,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        by_text = {}

        for entity in entities:
            text = normalize_pdf_text(entity.get("text", ""))

            if not text:
                continue

            key = normalize_text_for_match(text)

            if key not in by_text:
                entity["text"] = text
                by_text[key] = entity
            else:
                existing = by_text[key]
                if entity.get("confidence", 0) > existing.get("confidence", 0):
                    by_text[key] = entity

        return list(by_text.values())

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:
        text = normalize_text_for_match(text)

        tokens = re.findall(r"[a-z0-9_]+", text)

        return [
            token.strip("_")
            for token in tokens
            if len(token.strip("_")) >= self.config.min_token_len
        ]

    def _build_summary(
        self,
        query: str,
        query_language: str,
        query_type: str,
        query_intent: str,
        query_scope: str,
        answer_format: str,
        entities: List[Dict[str, Any]],
        filters: Dict[str, Any],
        retrieval_strategy: Dict[str, Any],
        complexity: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "has_query": bool(query),
            "query_language": query_language,
            "query_type": query_type,
            "query_intent": query_intent,
            "query_scope": query_scope,
            "answer_format": answer_format,
            "entity_count": len(entities),
            "filter_count": len(filters),
            "complexity_level": complexity.get("level", "medium"),
            "complexity_score": complexity.get("score", 0.0),
            "preferred_retriever": self._preferred_retriever(
                query_type=query_type,
                query_intent=query_intent,
                retrieval_strategy=retrieval_strategy,
            ),
            "use_bm25": retrieval_strategy.get("use_bm25", False),
            "use_vector": retrieval_strategy.get("use_vector", False),
            "use_graph": retrieval_strategy.get("use_graph", False),
            "use_metadata": retrieval_strategy.get("use_metadata", False),
            "use_table": retrieval_strategy.get("use_table", False),
            "need_citations": retrieval_strategy.get("use_citation_verification", False),
            "strategy_reason": retrieval_strategy.get("reason", []),
        }

    def _stopwords_vi(self) -> Set[str]:
        return {
            "va", "hoac", "cua", "cho", "cac", "mot", "nhung", "duoc", "trong",
            "ngoai", "theo", "voi", "den", "tu", "tai", "ve", "la", "co", "khong",
            "nay", "do", "khi", "sau", "truoc", "tren", "duoi", "vao", "ra", "de",
            "nham", "phuc", "vu", "can", "phai", "bao", "dam", "quy", "dinh",
            "noi", "dung", "thuc", "hien", "quan", "ly", "nha", "nuoc", "du",
            "lieu", "he", "thong", "chuc", "nang", "phan", "mem", "giup", "toi",
            "ban", "hay", "sinh", "file",
        }

    def _stopwords_en(self) -> Set[str]:
        return {
            "this", "that", "with", "from", "into", "about", "the", "and", "or",
            "for", "to", "of", "in", "on", "by", "is", "are", "be", "as", "at",
            "what", "why", "how", "when", "where", "which", "please", "help",
        }

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

    def save_result(
        self,
        classification_result: Dict[str, Any],
        output_path: str,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                json_safe(classification_result),
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


def classify_query(
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    classifier = QueryClassifier()
    return classifier.process(
        query=query,
        context=context,
    )


def get_query_routing_hints(
    query: str,
) -> Dict[str, Any]:
    classifier = QueryClassifier()
    result = classifier.process(query=query)
    return result.get("routing_hints", {})


def get_query_prompt_hints(
    query: str,
) -> Dict[str, Any]:
    classifier = QueryClassifier()
    result = classifier.process(query=query)
    return result.get("prompt_hints", {})
