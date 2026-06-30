"""
llm_reasoner.py

Production V1 - Colab Ready

Purpose
-------
Generate grounded answers from retrieved context and evidence.

Design
------
- Does not require external LLM by default.
- Can use a user-provided llm_fn / llm_client when available.
- Falls back to extractive reasoning based on evidence.
- Keeps citations attached to evidence.
- Avoids unsupported claims when evidence is weak.

Used by:
- RAGPipeline
- PromptBuilder
- EvidenceAggregator
- CitationVerifier

Input
-----
- query
- prompt_result
- evidence_aggregation_result
- retrieval_result
- expanded_context_result
- citation_verification_result
- llm_fn / llm_client optional

Output
------
Dictionary with:
- answer_text
- answer_status
- answer_type
- answer_confidence
- used_evidence
- used_citations
- unsupported_claims
- reasoning_summary
- llm_reasoning_summary
"""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple

from document_ai.schemas.page_raw_schema import (
    normalize_pdf_text,
    normalize_text_for_match,
    json_safe,
)


@dataclass
class LLMReasonerConfig:
    use_external_llm: bool = True
    fallback_to_extractive: bool = True

    answer_language: str = "vi"
    answer_style: str = "formal"
    answer_mode: str = "grounded"
    answer_format: str = "paragraph"

    max_answer_chars: int = 6000
    max_context_chars: int = 16000
    max_evidence_items: int = 18
    max_citations: int = 12

    min_evidence_items: int = 1
    min_confidence_for_direct_answer: float = 0.35

    require_citations: bool = True
    include_citation_markers: bool = True
    include_evidence_quotes: bool = False
    include_reasoning_summary: bool = True
    include_limitations: bool = True
    include_debug: bool = True

    hallucination_guard: bool = True
    unsupported_claim_threshold: float = 0.35
    max_sentence_per_evidence: int = 3

    extractive_sentence_count: int = 6
    extractive_min_sentence_chars: int = 25
    extractive_max_sentence_chars: int = 500

    llm_temperature: float = 0.1
    llm_max_tokens: int = 1600

    citation_marker_fallback: str = "[nguồn]"


class LLMReasoner:
    def __init__(
        self,
        config: Optional[LLMReasonerConfig] = None,
    ):
        self.config = config or LLMReasonerConfig()

    def process(
        self,
        query: str = "",
        prompt_result: Optional[Dict[str, Any]] = None,
        evidence_aggregation_result: Optional[Dict[str, Any]] = None,
        retrieval_result: Optional[Dict[str, Any]] = None,
        expanded_context_result: Optional[Dict[str, Any]] = None,
        citation_verification_result: Optional[Dict[str, Any]] = None,
        llm_fn: Optional[Callable[..., Any]] = None,
        llm_client: Optional[Any] = None,
        llm_model: str = "",
        generation_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        query = normalize_pdf_text(query)
        prompt_result = prompt_result or {}
        evidence_aggregation_result = evidence_aggregation_result or {}
        retrieval_result = retrieval_result or {}
        expanded_context_result = expanded_context_result or {}
        citation_verification_result = citation_verification_result or {}
        generation_config = generation_config or {}

        evidence_items = self._collect_evidence_items(
            evidence_aggregation_result=evidence_aggregation_result,
            retrieval_result=retrieval_result,
            expanded_context_result=expanded_context_result,
        )

        citations = self._collect_citations(
            evidence_aggregation_result=evidence_aggregation_result,
            citation_verification_result=citation_verification_result,
        )

        verified_citation_map = self._build_verified_citation_map(citation_verification_result)

        evidence_items = self._attach_citations_to_evidence(
            evidence_items=evidence_items,
            citations=citations,
            verified_citation_map=verified_citation_map,
        )

        evidence_items = self._rank_evidence(evidence_items)
        evidence_items = evidence_items[: self.config.max_evidence_items]

        prompt_text = self._resolve_prompt_text(
            query=query,
            prompt_result=prompt_result,
            evidence_items=evidence_items,
            citations=citations,
            generation_config=generation_config,
        )

        answer_text = ""
        answer_source = "none"
        llm_raw_response = None
        llm_error = ""

        if self.config.use_external_llm and (llm_fn is not None or llm_client is not None):
            try:
                llm_raw_response = self._call_external_llm(
                    prompt_text=prompt_text,
                    llm_fn=llm_fn,
                    llm_client=llm_client,
                    llm_model=llm_model,
                    generation_config=generation_config,
                )
                answer_text = self._extract_text_from_llm_response(llm_raw_response)
                answer_source = "external_llm"
            except Exception as exc:
                llm_error = str(exc)
                answer_text = ""
                answer_source = "external_llm_error"

        if not answer_text and self.config.fallback_to_extractive:
            answer_text = self._build_extractive_answer(
                query=query,
                evidence_items=evidence_items,
                citations=citations,
            )
            answer_source = "extractive_fallback"

        if not answer_text:
            answer_text = self._build_no_evidence_answer(query)
            answer_source = "no_answer"

        answer_text = self._postprocess_answer(
            answer_text=answer_text,
            evidence_items=evidence_items,
            citations=citations,
        )

        used_citations = self._extract_used_citations(
            answer_text=answer_text,
            citations=citations,
            evidence_items=evidence_items,
        )

        used_evidence = self._extract_used_evidence(
            answer_text=answer_text,
            evidence_items=evidence_items,
            used_citations=used_citations,
        )

        unsupported_claims = []

        if self.config.hallucination_guard:
            unsupported_claims = self._detect_unsupported_claims(
                answer_text=answer_text,
                evidence_items=evidence_items,
            )

        answer_confidence = self._compute_answer_confidence(
            answer_text=answer_text,
            evidence_items=evidence_items,
            used_evidence=used_evidence,
            used_citations=used_citations,
            unsupported_claims=unsupported_claims,
            answer_source=answer_source,
        )

        answer_status = self._infer_answer_status(
            evidence_items=evidence_items,
            answer_text=answer_text,
            answer_confidence=answer_confidence,
            unsupported_claims=unsupported_claims,
            llm_error=llm_error,
        )

        answer_type = self._infer_answer_type(
            query=query,
            answer_text=answer_text,
            evidence_items=evidence_items,
        )

        result = {
            "processor": "LLMReasoner",
            "schema_version": "llm_reasoner_v1",
            "query": query,
            "answer_text": answer_text,
            "answer_status": answer_status,
            "answer_type": answer_type,
            "answer_source": answer_source,
            "answer_confidence": answer_confidence,
            "used_evidence": used_evidence,
            "used_citations": used_citations,
            "unsupported_claims": unsupported_claims,
            "prompt_text": prompt_text if self.config.include_debug else "",
            "llm_raw_response": self._safe_llm_response(llm_raw_response) if self.config.include_debug else None,
            "llm_error": llm_error,
            "reasoning_summary": self._build_reasoning_summary(
                query=query,
                answer_text=answer_text,
                evidence_items=evidence_items,
                used_evidence=used_evidence,
                used_citations=used_citations,
                unsupported_claims=unsupported_claims,
                answer_source=answer_source,
                answer_confidence=answer_confidence,
            ),
            "llm_reasoning_summary": self._build_llm_reasoning_summary(
                answer_status=answer_status,
                answer_type=answer_type,
                answer_source=answer_source,
                answer_confidence=answer_confidence,
                evidence_items=evidence_items,
                used_evidence=used_evidence,
                used_citations=used_citations,
                unsupported_claims=unsupported_claims,
                llm_error=llm_error,
            ),
            "config": asdict(self.config),
        }

        return json_safe(result)

    def _call_external_llm(
        self,
        prompt_text: str,
        llm_fn: Optional[Callable[..., Any]],
        llm_client: Optional[Any],
        llm_model: str,
        generation_config: Dict[str, Any],
    ) -> Any:
        temperature = generation_config.get("temperature", self.config.llm_temperature)
        max_tokens = generation_config.get("max_tokens", self.config.llm_max_tokens)

        if llm_fn is not None:
            try:
                return llm_fn(
                    prompt=prompt_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=llm_model,
                )
            except TypeError:
                try:
                    return llm_fn(prompt_text)
                except TypeError:
                    return llm_fn(
                        prompt_text,
                        temperature,
                        max_tokens,
                    )

        if llm_client is None:
            return None

        if hasattr(llm_client, "generate"):
            try:
                return llm_client.generate(
                    prompt=prompt_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=llm_model,
                )
            except TypeError:
                return llm_client.generate(prompt_text)

        if hasattr(llm_client, "invoke"):
            return llm_client.invoke(prompt_text)

        if hasattr(llm_client, "complete"):
            try:
                return llm_client.complete(
                    prompt=prompt_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=llm_model,
                )
            except TypeError:
                return llm_client.complete(prompt_text)

        if callable(llm_client):
            return llm_client(prompt_text)

        return None

    def _extract_text_from_llm_response(
        self,
        response: Any,
    ) -> str:
        if response is None:
            return ""

        if isinstance(response, str):
            return normalize_pdf_text(response)

        if isinstance(response, dict):
            for key in [
                "answer",
                "answer_text",
                "text",
                "content",
                "response",
                "output",
                "generated_text",
            ]:
                if response.get(key):
                    return normalize_pdf_text(response.get(key))

            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]

                if isinstance(first, dict):
                    if first.get("text"):
                        return normalize_pdf_text(first.get("text"))

                    message = first.get("message", {})
                    if isinstance(message, dict) and message.get("content"):
                        return normalize_pdf_text(message.get("content"))

            return normalize_pdf_text(json.dumps(json_safe(response), ensure_ascii=False))

        if hasattr(response, "content"):
            return normalize_pdf_text(getattr(response, "content"))

        if hasattr(response, "text"):
            return normalize_pdf_text(getattr(response, "text"))

        return normalize_pdf_text(str(response))

    def _resolve_prompt_text(
        self,
        query: str,
        prompt_result: Dict[str, Any],
        evidence_items: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        generation_config: Dict[str, Any],
    ) -> str:
        for key in [
            "prompt_text",
            "final_prompt",
            "rag_prompt",
            "llm_prompt",
            "prompt",
        ]:
            value = normalize_pdf_text(prompt_result.get(key, ""))

            if value:
                return self._truncate_context(value, self.config.max_context_chars)

        return self._build_prompt_from_evidence(
            query=query,
            evidence_items=evidence_items,
            citations=citations,
            generation_config=generation_config,
        )

    def _build_prompt_from_evidence(
        self,
        query: str,
        evidence_items: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        generation_config: Dict[str, Any],
    ) -> str:
        parts = []

        parts.append("Bạn là hệ thống trả lời câu hỏi dựa trên bằng chứng trong tài liệu.")
        parts.append("Chỉ sử dụng thông tin có trong phần BẰNG CHỨNG. Không suy diễn ngoài tài liệu.")
        parts.append("Nếu bằng chứng không đủ, hãy nêu rõ là chưa đủ căn cứ trong tài liệu.")
        parts.append("Khi trả lời, gắn mã trích dẫn ở cuối câu hoặc cuối đoạn nếu có.")
        parts.append("")
        parts.append(f"CÂU HỎI: {normalize_pdf_text(query)}")
        parts.append("")
        parts.append("BẰNG CHỨNG:")

        citation_by_evidence = self._citations_by_evidence(citations)

        total_chars = 0

        for index, evidence in enumerate(evidence_items[: self.config.max_evidence_items], start=1):
            evidence_id = evidence.get("evidence_id", "")
            evidence_citations = citation_by_evidence.get(evidence_id, [])

            marker = ""
            if evidence_citations:
                marker = " ".join(
                    citation.get("citation_marker", "")
                    for citation in evidence_citations
                    if citation.get("citation_marker")
                )

            if not marker:
                marker = evidence.get("citation_marker", "")

            page_numbers = evidence.get("page_numbers", []) or []
            page_text = ""

            if page_numbers:
                if len(page_numbers) == 1:
                    page_text = f"trang {page_numbers[0]}"
                else:
                    page_text = f"trang {page_numbers[0]}-{page_numbers[-1]}"

            header = f"[E{index}]"

            if page_text:
                header += f" {page_text}"

            if marker:
                header += f" {marker}"

            text = normalize_pdf_text(evidence.get("quote") or evidence.get("text") or "")

            block = f"{header}\n{text}"

            if total_chars + len(block) > self.config.max_context_chars:
                break

            parts.append(block)
            parts.append("")
            total_chars += len(block)

        parts.append("YÊU CẦU TRẢ LỜI:")
        parts.append("- Trả lời trực tiếp vào câu hỏi.")
        parts.append("- Không đưa thông tin không có trong bằng chứng.")
        parts.append("- Nếu cần liệt kê, trình bày ngắn gọn, rõ ý.")
        parts.append("- Giữ nguyên mã trích dẫn dạng [tr.x] hoặc [nguồn] nếu có.")

        return normalize_pdf_text("\n".join(parts))

    def _build_extractive_answer(
        self,
        query: str,
        evidence_items: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> str:
        if not evidence_items:
            return self._build_no_evidence_answer(query)

        citation_by_evidence = self._citations_by_evidence(citations)
        query_tokens = self._tokenize(query)

        selected_sentences = []

        for evidence in evidence_items[: self.config.max_evidence_items]:
            text = normalize_pdf_text(evidence.get("quote") or evidence.get("text") or "")
            sentences = self._split_sentences(text)

            scored_sentences = []

            for sentence in sentences:
                if not self._valid_sentence(sentence):
                    continue

                score = self._sentence_score(
                    sentence=sentence,
                    query_tokens=query_tokens,
                    evidence=evidence,
                )

                scored_sentences.append(
                    {
                        "sentence": sentence,
                        "score": score,
                    }
                )

            scored_sentences = sorted(
                scored_sentences,
                key=lambda item: item["score"],
                reverse=True,
            )[: self.config.max_sentence_per_evidence]

            marker = self._marker_for_evidence(
                evidence=evidence,
                citation_by_evidence=citation_by_evidence,
            )

            for item in scored_sentences:
                sentence = normalize_pdf_text(item["sentence"])

                if marker and self.config.include_citation_markers:
                    sentence = self._append_marker(sentence, marker)

                selected_sentences.append(
                    {
                        "sentence": sentence,
                        "score": item["score"] * self._safe_float(evidence.get("evidence_score"), default=1.0),
                        "evidence_id": evidence.get("evidence_id", ""),
                    }
                )

        selected_sentences = sorted(
            selected_sentences,
            key=lambda item: item["score"],
            reverse=True,
        )

        selected_sentences = self._deduplicate_sentences(selected_sentences)
        selected_sentences = selected_sentences[: self.config.extractive_sentence_count]

        if not selected_sentences:
            return self._build_no_evidence_answer(query)

        answer_parts = []

        if self._looks_like_summary_query(query):
            answer_parts.append("Tóm tắt theo nội dung tài liệu:")

            for item in selected_sentences:
                answer_parts.append(f"- {item['sentence']}")

        elif self._looks_like_list_query(query):
            answer_parts.append("Theo nội dung tài liệu, có thể xác định các ý chính sau:")

            for item in selected_sentences:
                answer_parts.append(f"- {item['sentence']}")

        else:
            answer_parts.extend([item["sentence"] for item in selected_sentences])

        if self.config.include_limitations:
            weak_evidence = len(evidence_items) < self.config.min_evidence_items

            if weak_evidence:
                answer_parts.append("Lưu ý: bằng chứng trích xuất hiện còn ít, cần đối chiếu thêm tài liệu gốc để kết luận đầy đủ.")

        return self._truncate_answer(normalize_pdf_text("\n".join(answer_parts)))

    def _postprocess_answer(
        self,
        answer_text: str,
        evidence_items: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> str:
        answer_text = normalize_pdf_text(answer_text)

        if not answer_text:
            return ""

        answer_text = self._remove_prompt_echo(answer_text)
        answer_text = self._truncate_answer(answer_text)

        if self.config.require_citations and self.config.include_citation_markers:
            answer_text = self._ensure_answer_has_citation(
                answer_text=answer_text,
                evidence_items=evidence_items,
                citations=citations,
            )

        return answer_text

    def _ensure_answer_has_citation(
        self,
        answer_text: str,
        evidence_items: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
    ) -> str:
        markers = self._extract_markers(answer_text)

        if markers:
            return answer_text

        marker = ""

        for citation in citations:
            if citation.get("citation_marker"):
                marker = citation.get("citation_marker")
                break

        if not marker:
            for evidence in evidence_items:
                if evidence.get("citation_marker"):
                    marker = evidence.get("citation_marker")
                    break

        if not marker:
            marker = self.config.citation_marker_fallback

        if answer_text.endswith((".", "!", "?", "。")):
            return answer_text + f" {marker}"

        return answer_text + f". {marker}"

    def _remove_prompt_echo(
        self,
        answer_text: str,
    ) -> str:
        patterns = [
            r"^CÂU HỎI\s*:.*?\n",
            r"^BẰNG CHỨNG\s*:.*?\n",
            r"^YÊU CẦU TRẢ LỜI\s*:.*?\n",
        ]

        cleaned = answer_text

        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

        markers = [
            "TRẢ LỜI:",
            "Câu trả lời:",
            "Answer:",
        ]

        for marker in markers:
            index = cleaned.lower().find(marker.lower())

            if 0 <= index <= 100:
                cleaned = cleaned[index + len(marker):]

        return normalize_pdf_text(cleaned)

    def _collect_evidence_items(
        self,
        evidence_aggregation_result: Dict[str, Any],
        retrieval_result: Dict[str, Any],
        expanded_context_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        for key in ["supporting_evidence", "aggregated_evidence", "evidence_items", "evidence"]:
            values = evidence_aggregation_result.get(key, []) or []

            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        for key in ["retrieved_items", "fused_results"]:
            values = retrieval_result.get(key, []) or []

            if isinstance(values, list):
                items.extend([self._retrieved_item_to_evidence(item) for item in values])

        values = expanded_context_result.get("expanded_context_items", []) or []

        if isinstance(values, list):
            items.extend([self._retrieved_item_to_evidence(item) for item in values])

        return self._deduplicate_evidence(items)

    def _collect_citations(
        self,
        evidence_aggregation_result: Dict[str, Any],
        citation_verification_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        citations = []

        for source in [evidence_aggregation_result, citation_verification_result]:
            for key in [
                "citations",
                "used_citations",
                "verified_citations",
                "questionable_citations",
                "all_citations",
            ]:
                values = source.get(key, []) or []

                if isinstance(values, list):
                    citations.extend([self._to_dict(item) for item in values])

        return self._deduplicate_citations(citations)

    def _retrieved_item_to_evidence(
        self,
        item: Dict[str, Any],
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

        evidence_id = (
            item.get("evidence_id")
            or item.get("context_id")
            or item.get("source_id")
            or item.get("document_id")
            or item.get("chunk_id")
            or item.get("node_id")
            or self._stable_id(text, "evidence")
        )

        return {
            "evidence_id": evidence_id,
            "evidence_type": item.get("evidence_type") or item.get("source_type") or item.get("retrieval_source") or "retrieved_evidence",
            "source_id": item.get("source_id", ""),
            "source_type": item.get("source_type", ""),
            "title": item.get("title", "") or item.get("label", ""),
            "text": text,
            "quote": item.get("quote", "") or self._make_quote(text),
            "page_numbers": self._resolve_page_numbers(item),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "section_id": item.get("section_id", ""),
            "section_title": item.get("section_title", ""),
            "chunk_id": item.get("chunk_id", ""),
            "table_id": item.get("table_id", ""),
            "node_id": item.get("node_id", ""),
            "citation_id": item.get("citation_id", ""),
            "citation_marker": item.get("citation_marker", ""),
            "evidence_score": self._safe_float(item.get("evidence_score", item.get("score")), default=0.35),
            "confidence": self._safe_float(item.get("confidence"), default=0.65),
            "metadata": item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {},
        }

    def _attach_citations_to_evidence(
        self,
        evidence_items: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        verified_citation_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        citation_by_evidence = self._citations_by_evidence(citations)

        result = []

        for evidence in evidence_items:
            evidence = self._to_dict(evidence)
            evidence_id = evidence.get("evidence_id", "")
            linked = citation_by_evidence.get(evidence_id, [])

            if linked:
                evidence["citations"] = linked

                markers = [
                    citation.get("citation_marker", "")
                    for citation in linked
                    if citation.get("citation_marker")
                ]

                if markers and not evidence.get("citation_marker"):
                    evidence["citation_marker"] = markers[0]

                verified_scores = []

                for citation in linked:
                    citation_id = citation.get("citation_id", "")
                    verified = verified_citation_map.get(citation_id, {})
                    if verified:
                        verified_scores.append(self._safe_float(verified.get("verification_score"), default=0.0))

                if verified_scores:
                    evidence["citation_verification_score"] = max(verified_scores)

            result.append(evidence)

        return result

    def _build_verified_citation_map(
        self,
        citation_verification_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        result = {}

        for key in ["verified_citations", "questionable_citations", "failed_citations", "all_citations"]:
            values = citation_verification_result.get(key, []) or []

            if not isinstance(values, list):
                continue

            for citation in values:
                citation = self._to_dict(citation)
                citation_id = citation.get("citation_id", "")

                if citation_id:
                    result[citation_id] = citation

        return result

    def _citations_by_evidence(
        self,
        citations: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        result = {}

        for citation in citations:
            evidence_ids = citation.get("evidence_ids", []) or []

            if citation.get("evidence_id"):
                evidence_ids.append(citation.get("evidence_id"))

            for evidence_id in evidence_ids:
                if not evidence_id:
                    continue

                result.setdefault(evidence_id, [])
                result[evidence_id].append(citation)

        return result

    def _rank_evidence(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ranked = sorted(
            evidence_items,
            key=lambda item: (
                -self._safe_float(item.get("evidence_score"), default=0.0),
                -self._safe_float(item.get("citation_verification_score"), default=0.0),
                min(self._resolve_page_numbers(item) or [999999]),
                item.get("source_type", ""),
                item.get("evidence_id", ""),
            ),
        )

        for rank, item in enumerate(ranked, start=1):
            item["rank"] = rank

        return ranked

    def _extract_used_citations(
        self,
        answer_text: str,
        citations: List[Dict[str, Any]],
        evidence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        markers_in_answer = set(self._extract_markers(answer_text))
        used = []

        for citation in citations:
            marker = citation.get("citation_marker", "")

            if marker and marker in markers_in_answer:
                used.append(citation)

        if used:
            return self._deduplicate_citations(used)[: self.config.max_citations]

        for evidence in evidence_items[: self.config.max_citations]:
            for citation in evidence.get("citations", []) or []:
                used.append(citation)

        return self._deduplicate_citations(used)[: self.config.max_citations]

    def _extract_used_evidence(
        self,
        answer_text: str,
        evidence_items: List[Dict[str, Any]],
        used_citations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        used_evidence_ids = set()

        for citation in used_citations:
            for evidence_id in citation.get("evidence_ids", []) or []:
                if evidence_id:
                    used_evidence_ids.add(evidence_id)

            if citation.get("evidence_id"):
                used_evidence_ids.add(citation.get("evidence_id"))

        used = []

        answer_norm = normalize_text_for_match(answer_text)

        for evidence in evidence_items:
            evidence_id = evidence.get("evidence_id", "")

            if evidence_id and evidence_id in used_evidence_ids:
                used.append(evidence)
                continue

            quote_norm = normalize_text_for_match(evidence.get("quote", ""))

            if quote_norm and quote_norm[:120] in answer_norm:
                used.append(evidence)
                continue

            overlap = self._text_overlap_score(answer_text, evidence.get("text", "") or evidence.get("quote", ""))

            if overlap >= 0.20:
                used.append(evidence)

        if not used:
            used = evidence_items[: min(3, len(evidence_items))]

        return self._compact_evidence(used)

    def _detect_unsupported_claims(
        self,
        answer_text: str,
        evidence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        evidence_text = normalize_text_for_match(
            "\n".join(
                [
                    evidence.get("text", "") or evidence.get("quote", "")
                    for evidence in evidence_items
                ]
            )
        )

        if not evidence_text:
            return []

        unsupported = []

        for sentence in self._split_sentences(answer_text):
            if not self._valid_sentence(sentence):
                continue

            sentence_clean = self._remove_citation_markers(sentence)
            sentence_norm = normalize_text_for_match(sentence_clean)

            if not sentence_norm:
                continue

            overlap = self._fuzzy_overlap_score(sentence_norm, evidence_text)

            if overlap < self.config.unsupported_claim_threshold:
                unsupported.append(
                    {
                        "claim": sentence_clean,
                        "support_score": round(overlap, 4),
                        "status": "weakly_supported",
                    }
                )

        return unsupported

    def _compute_answer_confidence(
        self,
        answer_text: str,
        evidence_items: List[Dict[str, Any]],
        used_evidence: List[Dict[str, Any]],
        used_citations: List[Dict[str, Any]],
        unsupported_claims: List[Dict[str, Any]],
        answer_source: str,
    ) -> float:
        if not answer_text:
            return 0.0

        if not evidence_items:
            return 0.15

        evidence_scores = [
            self._safe_float(item.get("evidence_score"), default=0.0)
            for item in used_evidence or evidence_items[:3]
        ]

        confidence_scores = [
            self._safe_float(item.get("confidence"), default=0.65)
            for item in used_evidence or evidence_items[:3]
        ]

        citation_bonus = min(len(used_citations) / max(len(used_evidence), 1), 1.0) * 0.15
        evidence_bonus = min(len(used_evidence) / max(self.config.min_evidence_items, 1), 1.0) * 0.15

        base = 0.0

        if evidence_scores:
            base += min(sum(evidence_scores) / max(len(evidence_scores), 1), 1.5) * 0.35

        if confidence_scores:
            base += (sum(confidence_scores) / max(len(confidence_scores), 1)) * 0.25

        base += citation_bonus
        base += evidence_bonus

        if answer_source == "external_llm":
            base += 0.08
        elif answer_source == "extractive_fallback":
            base += 0.04

        penalty = min(len(unsupported_claims) * 0.10, 0.35)

        confidence = max(0.0, min(1.0, base - penalty))

        return round(confidence, 4)

    def _infer_answer_status(
        self,
        evidence_items: List[Dict[str, Any]],
        answer_text: str,
        answer_confidence: float,
        unsupported_claims: List[Dict[str, Any]],
        llm_error: str,
    ) -> str:
        if llm_error:
            if answer_text and evidence_items:
                return "answered_with_fallback_after_llm_error"
            return "failed_llm_error"

        if not evidence_items:
            return "insufficient_evidence"

        if not answer_text:
            return "no_answer_generated"

        if unsupported_claims and answer_confidence < self.config.min_confidence_for_direct_answer:
            return "answer_has_weak_support"

        if answer_confidence >= 0.70:
            return "answered_high_confidence"

        if answer_confidence >= self.config.min_confidence_for_direct_answer:
            return "answered_medium_confidence"

        return "answered_low_confidence"

    def _infer_answer_type(
        self,
        query: str,
        answer_text: str,
        evidence_items: List[Dict[str, Any]],
    ) -> str:
        query_norm = normalize_text_for_match(query)

        if not evidence_items:
            return "no_evidence_answer"

        if any(token in query_norm for token in ["liet ke", "danh sach", "nhung gi", "cac", "gồm", "gom"]):
            return "list_answer"

        if any(token in query_norm for token in ["tom tat", "tóm tắt", "tong hop", "tổng hợp"]):
            return "summary_answer"

        if any(token in query_norm for token in ["so sanh", "khac nhau", "giống nhau", "giong nhau"]):
            return "comparison_answer"

        if any(token in query_norm for token in ["vi sao", "tai sao", "nguyen nhan", "lý do", "ly do"]):
            return "explanation_answer"

        if len(answer_text) < 600:
            return "direct_answer"

        return "grounded_answer"

    def _build_no_evidence_answer(
        self,
        query: str,
    ) -> str:
        if query:
            return "Chưa đủ căn cứ trong tài liệu được cung cấp để trả lời chắc chắn cho câu hỏi này."

        return "Chưa có câu hỏi hoặc bằng chứng phù hợp để tạo câu trả lời."

    def _build_reasoning_summary(
        self,
        query: str,
        answer_text: str,
        evidence_items: List[Dict[str, Any]],
        used_evidence: List[Dict[str, Any]],
        used_citations: List[Dict[str, Any]],
        unsupported_claims: List[Dict[str, Any]],
        answer_source: str,
        answer_confidence: float,
    ) -> Dict[str, Any]:
        if not self.config.include_reasoning_summary:
            return {}

        return {
            "method": answer_source,
            "query": query,
            "evidence_considered": len(evidence_items),
            "evidence_used": len(used_evidence),
            "citations_used": len(used_citations),
            "unsupported_claim_count": len(unsupported_claims),
            "answer_confidence": answer_confidence,
            "grounding_status": "grounded" if len(unsupported_claims) == 0 else "partially_grounded",
            "summary": self._reasoning_summary_text(
                evidence_items=evidence_items,
                used_evidence=used_evidence,
                used_citations=used_citations,
                unsupported_claims=unsupported_claims,
                answer_confidence=answer_confidence,
            ),
        }

    def _build_llm_reasoning_summary(
        self,
        answer_status: str,
        answer_type: str,
        answer_source: str,
        answer_confidence: float,
        evidence_items: List[Dict[str, Any]],
        used_evidence: List[Dict[str, Any]],
        used_citations: List[Dict[str, Any]],
        unsupported_claims: List[Dict[str, Any]],
        llm_error: str,
    ) -> Dict[str, Any]:
        return {
            "answer_status": answer_status,
            "answer_type": answer_type,
            "answer_source": answer_source,
            "answer_confidence": answer_confidence,
            "input_evidence_count": len(evidence_items),
            "used_evidence_count": len(used_evidence),
            "used_citation_count": len(used_citations),
            "unsupported_claim_count": len(unsupported_claims),
            "llm_error": llm_error,
            "safe_to_use": answer_status in [
                "answered_high_confidence",
                "answered_medium_confidence",
                "answered_with_fallback_after_llm_error",
            ],
        }

    def _reasoning_summary_text(
        self,
        evidence_items: List[Dict[str, Any]],
        used_evidence: List[Dict[str, Any]],
        used_citations: List[Dict[str, Any]],
        unsupported_claims: List[Dict[str, Any]],
        answer_confidence: float,
    ) -> str:
        parts = []

        parts.append(f"Đã xem xét {len(evidence_items)} bằng chứng và sử dụng {len(used_evidence)} bằng chứng chính.")

        if used_citations:
            parts.append(f"Câu trả lời có {len(used_citations)} trích dẫn đi kèm.")

        if unsupported_claims:
            parts.append(f"Có {len(unsupported_claims)} câu/ý cần kiểm tra thêm vì mức khớp bằng chứng thấp.")

        parts.append(f"Độ tin cậy tổng hợp: {answer_confidence:.2f}.")

        return " ".join(parts)

    def _sentence_score(
        self,
        sentence: str,
        query_tokens: List[str],
        evidence: Dict[str, Any],
    ) -> float:
        sentence_norm = normalize_text_for_match(sentence)
        sentence_tokens = set(sentence_norm.split())

        if query_tokens:
            overlap = sum(1 for token in query_tokens if token in sentence_tokens) / max(len(query_tokens), 1)
        else:
            overlap = 0.5

        evidence_score = self._safe_float(evidence.get("evidence_score"), default=0.35)
        confidence = self._safe_float(evidence.get("confidence"), default=0.65)

        length_bonus = min(len(sentence) / 180.0, 1.0) * 0.15

        return overlap * 1.2 + evidence_score * 0.7 + confidence * 0.3 + length_bonus

    def _split_sentences(
        self,
        text: str,
    ) -> List[str]:
        text = normalize_pdf_text(text)

        if not text:
            return []

        parts = re.split(r"(?<=[\.\?\!。])\s+|\n+", text)

        result = []

        for part in parts:
            part = normalize_pdf_text(part)

            if not part:
                continue

            if len(part) > self.config.extractive_max_sentence_chars:
                subparts = re.split(r";\s+|,\s+(?=[A-ZÀ-Ỵa-zà-ỵ])", part)
                result.extend([normalize_pdf_text(item) for item in subparts if normalize_pdf_text(item)])
            else:
                result.append(part)

        return result

    def _valid_sentence(
        self,
        sentence: str,
    ) -> bool:
        sentence = normalize_pdf_text(sentence)

        if len(sentence) < self.config.extractive_min_sentence_chars:
            return False

        if len(sentence) > self.config.extractive_max_sentence_chars:
            return False

        if len(re.findall(r"\w+", sentence)) < 4:
            return False

        return True

    def _deduplicate_sentences(
        self,
        sentence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for item in sentence_items:
            key = normalize_text_for_match(self._remove_citation_markers(item.get("sentence", "")))[:300]

            if not key or key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _deduplicate_evidence(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for evidence in evidence_items:
            evidence = self._to_dict(evidence)

            text = normalize_pdf_text(evidence.get("text") or evidence.get("quote") or "")

            if not text:
                continue

            key = (
                evidence.get("evidence_id", ""),
                evidence.get("chunk_id", ""),
                evidence.get("table_id", ""),
                evidence.get("node_id", ""),
                normalize_text_for_match(text)[:700],
                tuple(self._resolve_page_numbers(evidence)),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(evidence)

        return result

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
                citation.get("evidence_id", ""),
                normalize_text_for_match(citation.get("quote", ""))[:300],
                tuple(self._resolve_page_numbers(citation)),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(citation)

        return result

    def _compact_evidence(
        self,
        evidence_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result = []

        for evidence in evidence_items:
            result.append(
                {
                    "evidence_id": evidence.get("evidence_id", ""),
                    "evidence_type": evidence.get("evidence_type", ""),
                    "source_type": evidence.get("source_type", ""),
                    "title": evidence.get("title", ""),
                    "quote": self._preview(evidence.get("quote") or evidence.get("text", ""), 500),
                    "page_numbers": self._resolve_page_numbers(evidence),
                    "section_id": evidence.get("section_id", ""),
                    "section_title": evidence.get("section_title", ""),
                    "chunk_id": evidence.get("chunk_id", ""),
                    "table_id": evidence.get("table_id", ""),
                    "node_id": evidence.get("node_id", ""),
                    "evidence_score": evidence.get("evidence_score", 0.0),
                    "confidence": evidence.get("confidence", 0.0),
                    "citation_marker": evidence.get("citation_marker", ""),
                }
            )

        return result

    def _marker_for_evidence(
        self,
        evidence: Dict[str, Any],
        citation_by_evidence: Dict[str, List[Dict[str, Any]]],
    ) -> str:
        evidence_id = evidence.get("evidence_id", "")
        citations = citation_by_evidence.get(evidence_id, []) or evidence.get("citations", []) or []

        for citation in citations:
            if citation.get("citation_marker"):
                return citation.get("citation_marker")

        if evidence.get("citation_marker"):
            return evidence.get("citation_marker")

        page_numbers = self._resolve_page_numbers(evidence)

        if len(page_numbers) == 1:
            return f"[tr.{page_numbers[0]}]"

        if len(page_numbers) > 1:
            return f"[tr.{page_numbers[0]}-{page_numbers[-1]}]"

        return self.config.citation_marker_fallback

    def _append_marker(
        self,
        sentence: str,
        marker: str,
    ) -> str:
        sentence = normalize_pdf_text(sentence)
        marker = normalize_pdf_text(marker)

        if not marker:
            return sentence

        if marker in sentence:
            return sentence

        if sentence.endswith((".", "!", "?")):
            return f"{sentence} {marker}"

        return f"{sentence}. {marker}"

    def _extract_markers(
        self,
        text: str,
    ) -> List[str]:
        text = normalize_pdf_text(text)

        patterns = [
            r"\[tr\.\s*\d+(?:\s*-\s*\d+)?(?:\.\d+)?\]",
            r"\[p\.\s*\d+(?:\s*-\s*\d+)?(?:\.\d+)?\]",
            r"\[page\s*\d+(?:\s*-\s*\d+)?(?:\.\d+)?\]",
            r"\[\d+(?:\s*-\s*\d+)?(?:\.\d+)?\]",
            r"\[nguồn\]",
            r"\[nguon\]",
            r"【[^】]{1,80}】",
        ]

        markers = []

        for pattern in patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                marker = normalize_pdf_text(match)
                if marker and marker not in markers:
                    markers.append(marker)

        return markers

    def _remove_citation_markers(
        self,
        text: str,
    ) -> str:
        text = normalize_pdf_text(text)

        text = re.sub(r"\[tr\.\s*\d+(?:\s*-\s*\d+)?(?:\.\d+)?\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[p\.\s*\d+(?:\s*-\s*\d+)?(?:\.\d+)?\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[page\s*\d+(?:\s*-\s*\d+)?(?:\.\d+)?\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\d+(?:\s*-\s*\d+)?(?:\.\d+)?\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[nguồn\]|\[nguon\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"【[^】]{1,80}】", "", text)

        return normalize_pdf_text(text)

    def _looks_like_summary_query(
        self,
        query: str,
    ) -> bool:
        query_norm = normalize_text_for_match(query)
        return any(token in query_norm for token in ["tom tat", "tong hop", "khai quat", "summary"])

    def _looks_like_list_query(
        self,
        query: str,
    ) -> bool:
        query_norm = normalize_text_for_match(query)
        return any(token in query_norm for token in ["liet ke", "danh sach", "cac noi dung", "nhung noi dung", "gồm", "gom"])

    def _text_overlap_score(
        self,
        text_a: str,
        text_b: str,
    ) -> float:
        a_tokens = set(
            token for token in normalize_text_for_match(text_a).split()
            if len(token) >= 3
        )
        b_tokens = set(
            token for token in normalize_text_for_match(text_b).split()
            if len(token) >= 3
        )

        if not a_tokens or not b_tokens:
            return 0.0

        return len(a_tokens.intersection(b_tokens)) / max(len(a_tokens), 1)

    def _fuzzy_overlap_score(
        self,
        claim_text: str,
        evidence_text: str,
    ) -> float:
        claim_tokens = [
            token for token in claim_text.split()
            if len(token) >= 3
        ]

        evidence_tokens = set(
            token for token in evidence_text.split()
            if len(token) >= 3
        )

        if not claim_tokens or not evidence_tokens:
            return 0.0

        matched = sum(1 for token in claim_tokens if token in evidence_tokens)

        return matched / max(len(claim_tokens), 1)

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:
        text = normalize_text_for_match(text)

        return [
            token for token in re.findall(r"[a-z0-9_]+", text)
            if len(token) >= 2
        ]

    def _make_quote(
        self,
        text: str,
    ) -> str:
        text = normalize_pdf_text(text)

        if len(text) <= 500:
            return text

        cut = text[:500]
        break_point = max(
            cut.rfind(". "),
            cut.rfind("; "),
            cut.rfind("\n"),
            cut.rfind(" "),
        )

        if break_point > 300:
            cut = cut[:break_point]

        return normalize_pdf_text(cut) + "..."

    def _truncate_context(
        self,
        text: str,
        max_chars: int,
    ) -> str:
        text = normalize_pdf_text(text)

        if len(text) <= max_chars:
            return text

        return normalize_pdf_text(text[:max_chars]) + "..."

    def _truncate_answer(
        self,
        text: str,
    ) -> str:
        text = normalize_pdf_text(text)

        if len(text) <= self.config.max_answer_chars:
            return text

        cut = text[: self.config.max_answer_chars]
        break_point = max(
            cut.rfind("\n\n"),
            cut.rfind(". "),
            cut.rfind("; "),
            cut.rfind("\n"),
            cut.rfind(" "),
        )

        if break_point > self.config.max_answer_chars * 0.60:
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

    def _safe_llm_response(
        self,
        response: Any,
    ) -> Any:
        if response is None:
            return None

        if isinstance(response, (str, int, float, bool, list, dict)):
            return json_safe(response)

        if hasattr(response, "to_dict"):
            try:
                return json_safe(response.to_dict())
            except Exception:
                pass

        if hasattr(response, "__dict__"):
            try:
                return json_safe(vars(response))
            except Exception:
                pass

        return str(response)

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
        reasoner_result: Dict[str, Any],
        output_path: str,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                json_safe(reasoner_result),
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


def reason_with_llm(
    query: str = "",
    prompt_result: Optional[Dict[str, Any]] = None,
    evidence_aggregation_result: Optional[Dict[str, Any]] = None,
    retrieval_result: Optional[Dict[str, Any]] = None,
    expanded_context_result: Optional[Dict[str, Any]] = None,
    citation_verification_result: Optional[Dict[str, Any]] = None,
    llm_fn: Optional[Callable[..., Any]] = None,
    llm_client: Optional[Any] = None,
    llm_model: str = "",
    generation_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reasoner = LLMReasoner()
    return reasoner.process(
        query=query,
        prompt_result=prompt_result,
        evidence_aggregation_result=evidence_aggregation_result,
        retrieval_result=retrieval_result,
        expanded_context_result=expanded_context_result,
        citation_verification_result=citation_verification_result,
        llm_fn=llm_fn,
        llm_client=llm_client,
        llm_model=llm_model,
        generation_config=generation_config,
    )


def generate_grounded_answer(
    query: str,
    evidence_aggregation_result: Optional[Dict[str, Any]] = None,
    retrieval_result: Optional[Dict[str, Any]] = None,
    expanded_context_result: Optional[Dict[str, Any]] = None,
    citation_verification_result: Optional[Dict[str, Any]] = None,
    llm_fn: Optional[Callable[..., Any]] = None,
    llm_client: Optional[Any] = None,
    llm_model: str = "",
) -> Dict[str, Any]:
    reasoner = LLMReasoner()
    return reasoner.process(
        query=query,
        evidence_aggregation_result=evidence_aggregation_result,
        retrieval_result=retrieval_result,
        expanded_context_result=expanded_context_result,
        citation_verification_result=citation_verification_result,
        llm_fn=llm_fn,
        llm_client=llm_client,
        llm_model=llm_model,
    )
