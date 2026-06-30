"""
prompt_builder.py

Production V1 - Colab Ready

Purpose
-------
Build grounded RAG prompts from:
- user query
- hybrid retrieval results
- expanded context
- aggregated evidence
- citations / citation verification
- metadata / document profile

Used by:
- RAGPipeline
- LLMReasoner
- EvidenceAggregator
- CitationVerifier

Input
-----
- query
- retrieval_result
- expanded_context_result
- evidence_aggregation_result
- citation_verification_result
- metadata_enrichment_result
- document_profile
- prompt_options

Output
------
Dictionary with:
- prompt_text
- system_prompt
- user_prompt
- context_blocks
- evidence_blocks
- citation_blocks
- prompt_summary
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
class PromptBuilderConfig:
    language: str = "vi"
    role_name: str = "Trợ lý phân tích tài liệu"
    task_mode: str = "grounded_qa"
    answer_style: str = "formal"
    answer_format: str = "structured"

    require_grounding: bool = True
    require_citations: bool = True
    allow_insufficient_evidence_answer: bool = True
    forbid_external_knowledge: bool = True

    include_system_prompt: bool = True
    include_query: bool = True
    include_document_context: bool = True
    include_retrieval_context: bool = True
    include_expanded_context: bool = True
    include_evidence: bool = True
    include_citations: bool = True
    include_verified_citation_status: bool = True
    include_metadata: bool = True
    include_output_instructions: bool = True
    include_safety_instructions: bool = True

    max_prompt_chars: int = 24000
    max_context_chars: int = 16000
    max_retrieved_items: int = 12
    max_expanded_context_items: int = 16
    max_evidence_items: int = 18
    max_citation_items: int = 18
    max_metadata_chars: int = 2500

    max_text_chars_per_block: int = 1600
    max_quote_chars_per_evidence: int = 700
    text_preview_chars: int = 700

    preserve_page_references: bool = True
    preserve_section_references: bool = True
    preserve_table_references: bool = True

    citation_marker_fallback: str = "[nguồn]"
    context_block_prefix: str = "CTX"
    evidence_block_prefix: str = "E"
    citation_block_prefix: str = "C"

    include_debug: bool = True


class PromptBuilder:
    def __init__(
        self,
        config: Optional[PromptBuilderConfig] = None,
    ):
        self.config = config or PromptBuilderConfig()

    def process(
        self,
        query: str = "",
        retrieval_result: Optional[Dict[str, Any]] = None,
        expanded_context_result: Optional[Dict[str, Any]] = None,
        evidence_aggregation_result: Optional[Dict[str, Any]] = None,
        citation_verification_result: Optional[Dict[str, Any]] = None,
        metadata_enrichment_result: Optional[Dict[str, Any]] = None,
        document_profile: Optional[Dict[str, Any]] = None,
        prompt_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        query = normalize_pdf_text(query)
        retrieval_result = retrieval_result or {}
        expanded_context_result = expanded_context_result or {}
        evidence_aggregation_result = evidence_aggregation_result or {}
        citation_verification_result = citation_verification_result or {}
        metadata_enrichment_result = metadata_enrichment_result or {}
        document_profile = document_profile or {}
        prompt_options = prompt_options or {}

        effective_config = self._effective_config(prompt_options)

        retrieved_items = self._collect_retrieved_items(retrieval_result)
        expanded_context_items = self._collect_expanded_context_items(expanded_context_result)
        evidence_items = self._collect_evidence_items(evidence_aggregation_result)
        citation_items = self._collect_citation_items(
            evidence_aggregation_result=evidence_aggregation_result,
            citation_verification_result=citation_verification_result,
        )

        verified_citation_map = self._build_verified_citation_map(citation_verification_result)

        context_blocks = self._build_context_blocks(
            retrieved_items=retrieved_items,
            expanded_context_items=expanded_context_items,
            effective_config=effective_config,
        )

        evidence_blocks = self._build_evidence_blocks(
            evidence_items=evidence_items,
            citation_items=citation_items,
            verified_citation_map=verified_citation_map,
            effective_config=effective_config,
        )

        citation_blocks = self._build_citation_blocks(
            citation_items=citation_items,
            verified_citation_map=verified_citation_map,
            effective_config=effective_config,
        )

        metadata_block = self._build_metadata_block(
            metadata_enrichment_result=metadata_enrichment_result,
            document_profile=document_profile,
            effective_config=effective_config,
        )

        system_prompt = self._build_system_prompt(effective_config)
        user_prompt = self._build_user_prompt(
            query=query,
            context_blocks=context_blocks,
            evidence_blocks=evidence_blocks,
            citation_blocks=citation_blocks,
            metadata_block=metadata_block,
            effective_config=effective_config,
        )

        prompt_text = self._combine_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            effective_config=effective_config,
        )

        prompt_text = self._truncate_prompt(prompt_text, effective_config.max_prompt_chars)

        result = {
            "processor": "PromptBuilder",
            "schema_version": "prompt_builder_v1",
            "query": query,
            "prompt_text": prompt_text,
            "final_prompt": prompt_text,
            "rag_prompt": prompt_text,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context_blocks": context_blocks,
            "evidence_blocks": evidence_blocks,
            "citation_blocks": citation_blocks,
            "metadata_block": metadata_block,
            "prompt_variables": {
                "query": query,
                "context_count": len(context_blocks),
                "evidence_count": len(evidence_blocks),
                "citation_count": len(citation_blocks),
                "language": effective_config.language,
                "task_mode": effective_config.task_mode,
                "answer_style": effective_config.answer_style,
                "answer_format": effective_config.answer_format,
            },
            "prompt_summary": self._build_summary(
                query=query,
                prompt_text=prompt_text,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                context_blocks=context_blocks,
                evidence_blocks=evidence_blocks,
                citation_blocks=citation_blocks,
                metadata_block=metadata_block,
                retrieved_items=retrieved_items,
                expanded_context_items=expanded_context_items,
                evidence_items=evidence_items,
                citation_items=citation_items,
            ),
            "config": asdict(effective_config),
        }

        return json_safe(result)

    def _effective_config(
        self,
        prompt_options: Dict[str, Any],
    ) -> PromptBuilderConfig:
        base = asdict(self.config)

        for key, value in (prompt_options or {}).items():
            if key in base and value is not None:
                base[key] = value

        return PromptBuilderConfig(**base)

    def _build_system_prompt(
        self,
        config: PromptBuilderConfig,
    ) -> str:
        if not config.include_system_prompt:
            return ""

        parts = []

        parts.append(f"Bạn là {config.role_name}.")
        parts.append("Nhiệm vụ của bạn là trả lời câu hỏi dựa trên bằng chứng được cung cấp từ tài liệu.")

        if config.forbid_external_knowledge:
            parts.append("Không sử dụng kiến thức bên ngoài, không tự bổ sung thông tin nếu không có trong bằng chứng.")

        if config.require_grounding:
            parts.append("Mọi nhận định quan trọng phải bám sát nội dung trong phần BẰNG CHỨNG hoặc NGỮ CẢNH.")

        if config.require_citations:
            parts.append("Khi sử dụng thông tin từ bằng chứng, giữ mã trích dẫn kèm theo như [tr.1], [tr.2-3] hoặc [nguồn].")

        if config.allow_insufficient_evidence_answer:
            parts.append("Nếu bằng chứng chưa đủ để kết luận, hãy nêu rõ 'chưa đủ căn cứ trong tài liệu' thay vì suy đoán.")

        if config.include_safety_instructions:
            parts.append("Không tạo trích dẫn giả, không khẳng định điều không xuất hiện trong tài liệu, không che giấu hạn chế của nguồn.")

        if config.answer_style == "formal":
            parts.append("Văn phong: trang trọng, rõ ràng, phù hợp tài liệu nghiệp vụ/báo cáo.")
        elif config.answer_style == "concise":
            parts.append("Văn phong: ngắn gọn, đi thẳng vào câu trả lời.")
        elif config.answer_style == "technical":
            parts.append("Văn phong: kỹ thuật, có cấu trúc, ưu tiên thuật ngữ chính xác.")
        else:
            parts.append(f"Văn phong: {config.answer_style}.")

        return normalize_pdf_text("\n".join(parts))

    def _build_user_prompt(
        self,
        query: str,
        context_blocks: List[Dict[str, Any]],
        evidence_blocks: List[Dict[str, Any]],
        citation_blocks: List[Dict[str, Any]],
        metadata_block: Dict[str, Any],
        effective_config: PromptBuilderConfig,
    ) -> str:
        parts = []

        if effective_config.include_query:
            parts.append("## CÂU HỎI")
            parts.append(query if query else "Chưa có câu hỏi cụ thể.")
            parts.append("")

        if effective_config.include_metadata and metadata_block.get("text"):
            parts.append("## THÔNG TIN TÀI LIỆU")
            parts.append(metadata_block["text"])
            parts.append("")

        if effective_config.include_evidence and evidence_blocks:
            parts.append("## BẰNG CHỨNG ƯU TIÊN")
            for block in evidence_blocks:
                parts.append(self._render_evidence_block(block))
            parts.append("")

        if effective_config.include_retrieval_context and context_blocks:
            parts.append("## NGỮ CẢNH TRUY XUẤT")
            current_chars = 0

            for block in context_blocks:
                rendered = self._render_context_block(block)

                if current_chars + len(rendered) > effective_config.max_context_chars:
                    break

                parts.append(rendered)
                current_chars += len(rendered)

            parts.append("")

        if effective_config.include_citations and citation_blocks:
            parts.append("## DANH MỤC TRÍCH DẪN")
            for block in citation_blocks:
                parts.append(self._render_citation_block(block))
            parts.append("")

        if effective_config.include_output_instructions:
            parts.append("## YÊU CẦU TRẢ LỜI")
            parts.extend(self._output_instructions(effective_config))
            parts.append("")

        return normalize_pdf_text("\n".join(parts))

    def _combine_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        effective_config: PromptBuilderConfig,
    ) -> str:
        if system_prompt:
            return normalize_pdf_text(
                "\n\n".join(
                    [
                        "### SYSTEM",
                        system_prompt,
                        "### USER",
                        user_prompt,
                    ]
                )
            )

        return normalize_pdf_text(user_prompt)

    def _output_instructions(
        self,
        config: PromptBuilderConfig,
    ) -> List[str]:
        instructions = []

        if config.answer_format == "structured":
            instructions.append("- Trả lời có cấu trúc: nêu kết luận trước, sau đó giải thích các ý chính.")
        elif config.answer_format == "bullets":
            instructions.append("- Trả lời bằng các gạch đầu dòng ngắn gọn.")
        elif config.answer_format == "table":
            instructions.append("- Nếu phù hợp, trình bày dưới dạng bảng markdown.")
        elif config.answer_format == "paragraph":
            instructions.append("- Trả lời dưới dạng đoạn văn rõ ràng.")
        else:
            instructions.append(f"- Định dạng trả lời: {config.answer_format}.")

        if config.require_citations:
            instructions.append("- Gắn mã trích dẫn ngay sau câu/ý sử dụng bằng chứng.")
            instructions.append("- Không tự tạo mã trích dẫn nếu không có trong phần bằng chứng; khi thiếu thì dùng [nguồn] hoặc nêu chưa có trích dẫn cụ thể.")

        if config.require_grounding:
            instructions.append("- Không đưa nhận định vượt quá nội dung bằng chứng.")

        if config.allow_insufficient_evidence_answer:
            instructions.append("- Nếu bằng chứng không đủ, nêu rõ phần nào đã có căn cứ và phần nào cần kiểm tra thêm.")

        if config.task_mode == "summarization":
            instructions.append("- Ưu tiên tóm tắt các ý chính, không diễn giải dài.")
        elif config.task_mode == "comparison":
            instructions.append("- Nếu có nhiều đối tượng, so sánh theo tiêu chí rõ ràng.")
        elif config.task_mode == "extraction":
            instructions.append("- Chỉ trích xuất thông tin có trong tài liệu, giữ nguyên số liệu/tên riêng nếu có.")
        elif config.task_mode == "analysis":
            instructions.append("- Có thể phân tích nhưng mọi phân tích phải chỉ ra căn cứ từ bằng chứng.")

        return instructions

    def _build_context_blocks(
        self,
        retrieved_items: List[Dict[str, Any]],
        expanded_context_items: List[Dict[str, Any]],
        effective_config: PromptBuilderConfig,
    ) -> List[Dict[str, Any]]:
        blocks = []

        if effective_config.include_retrieval_context:
            for item in retrieved_items[: effective_config.max_retrieved_items]:
                block = self._item_to_context_block(
                    item=item,
                    block_type="retrieved",
                    prefix=effective_config.context_block_prefix,
                    index=len(blocks) + 1,
                    max_text_chars=effective_config.max_text_chars_per_block,
                )
                blocks.append(block)

        if effective_config.include_expanded_context:
            for item in expanded_context_items[: effective_config.max_expanded_context_items]:
                block = self._item_to_context_block(
                    item=item,
                    block_type=item.get("expansion_type", "expanded"),
                    prefix=effective_config.context_block_prefix,
                    index=len(blocks) + 1,
                    max_text_chars=effective_config.max_text_chars_per_block,
                )
                blocks.append(block)

        blocks = self._deduplicate_blocks(blocks)
        blocks = self._sort_blocks(blocks)

        return blocks

    def _build_evidence_blocks(
        self,
        evidence_items: List[Dict[str, Any]],
        citation_items: List[Dict[str, Any]],
        verified_citation_map: Dict[str, Dict[str, Any]],
        effective_config: PromptBuilderConfig,
    ) -> List[Dict[str, Any]]:
        citation_by_evidence = self._citations_by_evidence(citation_items)

        blocks = []

        evidence_items = sorted(
            evidence_items,
            key=lambda item: (
                -self._safe_float(item.get("evidence_score"), default=0.0),
                -self._safe_float(item.get("confidence"), default=0.0),
                min(self._resolve_page_numbers(item) or [999999]),
            ),
        )

        for index, evidence in enumerate(evidence_items[: effective_config.max_evidence_items], start=1):
            evidence_id = evidence.get("evidence_id", "")
            citations = citation_by_evidence.get(evidence_id, []) or evidence.get("citations", []) or []
            citation_markers = []

            for citation in citations:
                marker = citation.get("citation_marker", "")
                if marker and marker not in citation_markers:
                    citation_markers.append(marker)

            if not citation_markers and evidence.get("citation_marker"):
                citation_markers.append(evidence.get("citation_marker"))

            citation_statuses = []

            for citation in citations:
                citation_id = citation.get("citation_id", "")
                verified = verified_citation_map.get(citation_id, {})
                if verified:
                    citation_statuses.append(
                        {
                            "citation_id": citation_id,
                            "marker": citation.get("citation_marker", ""),
                            "status": verified.get("verification_status", ""),
                            "score": verified.get("verification_score", 0.0),
                        }
                    )

            text = normalize_pdf_text(evidence.get("quote") or evidence.get("text") or "")
            text = self._truncate_text(text, effective_config.max_quote_chars_per_evidence)

            pages = self._resolve_page_numbers(evidence)

            block = {
                "block_id": f"{effective_config.evidence_block_prefix}{index}",
                "block_type": "evidence",
                "evidence_id": evidence_id,
                "title": normalize_pdf_text(evidence.get("title") or evidence.get("section_title") or evidence.get("source_type") or ""),
                "text": text,
                "page_numbers": pages,
                "page_label": self._page_label(pages),
                "section_id": evidence.get("section_id", ""),
                "section_title": evidence.get("section_title", ""),
                "table_id": evidence.get("table_id", ""),
                "chunk_id": evidence.get("chunk_id", ""),
                "source_type": evidence.get("source_type", ""),
                "evidence_type": evidence.get("evidence_type", ""),
                "score": self._safe_float(evidence.get("evidence_score"), default=0.0),
                "confidence": self._safe_float(evidence.get("confidence"), default=0.0),
                "citation_markers": citation_markers,
                "citation_statuses": citation_statuses,
                "metadata": evidence.get("metadata", {}) if effective_config.include_debug else {},
            }
            blocks.append(block)

        return blocks

    def _build_citation_blocks(
        self,
        citation_items: List[Dict[str, Any]],
        verified_citation_map: Dict[str, Dict[str, Any]],
        effective_config: PromptBuilderConfig,
    ) -> List[Dict[str, Any]]:
        blocks = []

        citation_items = self._deduplicate_citations(citation_items)

        for index, citation in enumerate(citation_items[: effective_config.max_citation_items], start=1):
            citation_id = citation.get("citation_id", "")
            verified = verified_citation_map.get(citation_id, {})

            pages = self._resolve_page_numbers(citation)

            block = {
                "block_id": f"{effective_config.citation_block_prefix}{index}",
                "block_type": "citation",
                "citation_id": citation_id,
                "citation_marker": citation.get("citation_marker", "") or self._marker_from_pages(pages, effective_config),
                "citation_text": normalize_pdf_text(citation.get("citation_text", "")),
                "quote": self._truncate_text(
                    citation.get("quote", "") or citation.get("text", ""),
                    effective_config.max_quote_chars_per_evidence,
                ),
                "page_numbers": pages,
                "page_label": self._page_label(pages),
                "evidence_id": citation.get("evidence_id", ""),
                "evidence_ids": citation.get("evidence_ids", []) or [],
                "section_id": citation.get("section_id", ""),
                "section_title": citation.get("section_title", ""),
                "table_id": citation.get("table_id", ""),
                "chunk_id": citation.get("chunk_id", ""),
                "verified": verified.get("verified", citation.get("verified", False)),
                "verification_status": verified.get("verification_status", citation.get("verification_status", "unverified")),
                "verification_score": verified.get("verification_score", citation.get("verification_score", 0.0)),
            }
            blocks.append(block)

        return blocks

    def _build_metadata_block(
        self,
        metadata_enrichment_result: Dict[str, Any],
        document_profile: Dict[str, Any],
        effective_config: PromptBuilderConfig,
    ) -> Dict[str, Any]:
        if not effective_config.include_metadata:
            return {}

        parts = []

        document_metadata = metadata_enrichment_result.get("document_metadata", {}) or {}

        if document_metadata:
            title = document_metadata.get("title") or document_metadata.get("file_name") or document_metadata.get("source_document") or ""
            if title:
                parts.append(f"Tên/tài liệu: {normalize_pdf_text(title)}")

            page_count = document_metadata.get("page_count") or document_metadata.get("total_pages")
            if page_count:
                parts.append(f"Số trang: {page_count}")

            document_type = document_metadata.get("document_type") or document_metadata.get("category")
            if document_type:
                parts.append(f"Loại tài liệu: {normalize_pdf_text(document_type)}")

        if document_profile:
            profile_title = document_profile.get("file_name") or document_profile.get("source_document") or document_profile.get("document_id") or ""
            if profile_title and not any(profile_title in part for part in parts):
                parts.append(f"Hồ sơ tài liệu: {normalize_pdf_text(profile_title)}")

            for key, label in [
                ("page_count", "Số trang"),
                ("pdf_type", "Loại PDF"),
                ("processing_strategy", "Chiến lược xử lý"),
                ("complexity", "Độ phức tạp"),
            ]:
                value = document_profile.get(key)
                if value:
                    parts.append(f"{label}: {normalize_pdf_text(value)}")

        metadata_summary = metadata_enrichment_result.get("metadata_summary", {}) or {}

        if metadata_summary:
            for key, label in [
                ("page_count", "Số trang có metadata"),
                ("section_count", "Số mục/phần"),
                ("table_count", "Số bảng"),
                ("chunk_count", "Số đoạn/chunk"),
                ("evidence_count", "Số bằng chứng"),
            ]:
                value = metadata_summary.get(key)
                if value not in [None, ""]:
                    parts.append(f"{label}: {value}")

        text = normalize_pdf_text("\n".join(parts))
        text = self._truncate_text(text, effective_config.max_metadata_chars)

        return {
            "block_type": "metadata",
            "text": text,
            "document_metadata": document_metadata if effective_config.include_debug else {},
            "document_profile": document_profile if effective_config.include_debug else {},
            "metadata_summary": metadata_summary if effective_config.include_debug else {},
        }

    def _item_to_context_block(
        self,
        item: Dict[str, Any],
        block_type: str,
        prefix: str,
        index: int,
        max_text_chars: int,
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

        text = self._truncate_text(text, max_text_chars)

        metadata = item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {"raw_metadata": metadata}

        pages = self._resolve_page_numbers(item)

        return {
            "block_id": f"{prefix}{index}",
            "block_type": block_type,
            "source_id": item.get("source_id", "")
            or item.get("document_id", "")
            or item.get("context_id", "")
            or item.get("chunk_id", "")
            or item.get("evidence_id", "")
            or item.get("node_id", "")
            or item.get("item_id", "")
            or self._stable_id(text, "context"),
            "source_type": item.get("source_type", "")
            or item.get("chunk_type", "")
            or item.get("evidence_type", "")
            or item.get("node_type", "")
            or item.get("item_type", "")
            or block_type,
            "title": normalize_pdf_text(item.get("title") or item.get("section_title") or item.get("label") or ""),
            "text": text,
            "page_numbers": pages,
            "page_label": self._page_label(pages),
            "section_id": item.get("section_id", "") or metadata.get("section_id", ""),
            "section_title": item.get("section_title", "") or metadata.get("section_title", ""),
            "table_id": item.get("table_id", "") or metadata.get("table_id", ""),
            "chunk_id": item.get("chunk_id", "") or metadata.get("chunk_id", ""),
            "evidence_id": item.get("evidence_id", "") or metadata.get("evidence_id", ""),
            "node_id": item.get("node_id", "") or metadata.get("node_id", ""),
            "score": self._safe_float(item.get("score"), default=0.0),
            "confidence": self._safe_float(item.get("confidence"), default=0.0),
            "matched_sources": item.get("matched_sources", []) or [],
            "matched_terms": item.get("matched_terms", []) or [],
            "metadata": metadata if self.config.include_debug else {},
        }

    def _render_context_block(
        self,
        block: Dict[str, Any],
    ) -> str:
        header_parts = [
            f"[{block.get('block_id', '')}]",
            f"type={block.get('source_type', '')}",
        ]

        if block.get("title"):
            header_parts.append(block.get("title"))

        if block.get("page_label"):
            header_parts.append(block.get("page_label"))

        if block.get("section_title"):
            header_parts.append(f"mục={block.get('section_title')}")

        if block.get("score"):
            header_parts.append(f"score={block.get('score')}")

        return normalize_pdf_text(
            "\n".join(
                [
                    " | ".join([part for part in header_parts if part]),
                    block.get("text", ""),
                ]
            )
        )

    def _render_evidence_block(
        self,
        block: Dict[str, Any],
    ) -> str:
        marker_text = " ".join(block.get("citation_markers", []) or [])

        header_parts = [
            f"[{block.get('block_id', '')}]",
            f"type={block.get('evidence_type') or block.get('source_type', '')}",
        ]

        if block.get("title"):
            header_parts.append(block.get("title"))

        if block.get("page_label"):
            header_parts.append(block.get("page_label"))

        if marker_text:
            header_parts.append(marker_text)

        if block.get("score"):
            header_parts.append(f"score={block.get('score')}")

        if block.get("citation_statuses"):
            status_text = "; ".join(
                [
                    f"{item.get('marker', '')}:{item.get('status', '')}"
                    for item in block.get("citation_statuses", [])
                    if item.get("marker") or item.get("status")
                ]
            )
            if status_text:
                header_parts.append(f"citation_status={status_text}")

        return normalize_pdf_text(
            "\n".join(
                [
                    " | ".join([part for part in header_parts if part]),
                    block.get("text", ""),
                ]
            )
        )

    def _render_citation_block(
        self,
        block: Dict[str, Any],
    ) -> str:
        header_parts = [
            f"[{block.get('block_id', '')}]",
            block.get("citation_marker", ""),
        ]

        if block.get("page_label"):
            header_parts.append(block.get("page_label"))

        if block.get("verification_status"):
            header_parts.append(f"status={block.get('verification_status')}")

        text_parts = [
            " | ".join([part for part in header_parts if part]),
        ]

        if block.get("citation_text"):
            text_parts.append(block.get("citation_text"))

        if block.get("quote"):
            text_parts.append(f"Trích đoạn: {block.get('quote')}")

        return normalize_pdf_text("\n".join(text_parts))

    def _collect_retrieved_items(
        self,
        retrieval_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        for key in [
            "retrieved_items",
            "fused_results",
            "bm25_results",
            "vector_results",
            "graph_results",
            "metadata_results",
            "table_results",
        ]:
            values = retrieval_result.get(key, []) or []

            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_raw_items(items)

    def _collect_expanded_context_items(
        self,
        expanded_context_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        values = expanded_context_result.get("expanded_context_items", []) or []

        if not isinstance(values, list):
            return []

        return self._deduplicate_raw_items([self._to_dict(item) for item in values])

    def _collect_evidence_items(
        self,
        evidence_aggregation_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        for key in [
            "supporting_evidence",
            "aggregated_evidence",
            "evidence_items",
            "evidence",
        ]:
            values = evidence_aggregation_result.get(key, []) or []

            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_evidence(items)

    def _collect_citation_items(
        self,
        evidence_aggregation_result: Dict[str, Any],
        citation_verification_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        for source in [evidence_aggregation_result, citation_verification_result]:
            for key in [
                "citations",
                "citation_items",
                "verified_citations",
                "questionable_citations",
                "failed_citations",
                "all_citations",
            ]:
                values = source.get(key, []) or []

                if isinstance(values, list):
                    items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_citations(items)

    def _build_verified_citation_map(
        self,
        citation_verification_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        result = {}

        for key in [
            "verified_citations",
            "questionable_citations",
            "failed_citations",
            "all_citations",
        ]:
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

    def _deduplicate_blocks(
        self,
        blocks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for block in blocks:
            key = (
                block.get("source_id", ""),
                block.get("source_type", ""),
                block.get("chunk_id", ""),
                block.get("evidence_id", ""),
                block.get("node_id", ""),
                normalize_text_for_match(block.get("text", ""))[:600],
                tuple(block.get("page_numbers", []) or []),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(block)

        for index, block in enumerate(result, start=1):
            prefix = re.sub(r"\d+$", "", block.get("block_id", "")) or "CTX"
            block["block_id"] = f"{prefix}{index}"

        return result

    def _sort_blocks(
        self,
        blocks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return sorted(
            blocks,
            key=lambda item: (
                -self._safe_float(item.get("score"), default=0.0),
                min(item.get("page_numbers", []) or [999999]),
                item.get("block_type", ""),
                item.get("source_type", ""),
            ),
        )

    def _deduplicate_raw_items(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for item in items:
            item = self._to_dict(item)

            key = (
                item.get("source_id", ""),
                item.get("document_id", ""),
                item.get("chunk_id", ""),
                item.get("evidence_id", ""),
                item.get("node_id", ""),
                item.get("item_id", ""),
                normalize_text_for_match(item.get("text") or item.get("text_preview") or item.get("quote") or "")[:700],
                tuple(self._resolve_page_numbers(item)),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _deduplicate_evidence(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for item in items:
            item = self._to_dict(item)

            text = normalize_pdf_text(item.get("text") or item.get("quote") or "")
            if not text:
                continue

            key = (
                item.get("evidence_id", ""),
                item.get("chunk_id", ""),
                item.get("table_id", ""),
                item.get("node_id", ""),
                normalize_text_for_match(text)[:700],
                tuple(self._resolve_page_numbers(item)),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _deduplicate_citations(
        self,
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for item in items:
            item = self._to_dict(item)

            key = (
                item.get("citation_id", ""),
                item.get("citation_marker", ""),
                item.get("evidence_id", ""),
                normalize_text_for_match(item.get("quote") or item.get("citation_text") or "")[:400],
                tuple(self._resolve_page_numbers(item)),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _build_summary(
        self,
        query: str,
        prompt_text: str,
        system_prompt: str,
        user_prompt: str,
        context_blocks: List[Dict[str, Any]],
        evidence_blocks: List[Dict[str, Any]],
        citation_blocks: List[Dict[str, Any]],
        metadata_block: Dict[str, Any],
        retrieved_items: List[Dict[str, Any]],
        expanded_context_items: List[Dict[str, Any]],
        evidence_items: List[Dict[str, Any]],
        citation_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        by_context_type = {}
        by_evidence_type = {}
        by_page = {}

        for block in context_blocks:
            block_type = block.get("block_type", "unknown")
            by_context_type[block_type] = by_context_type.get(block_type, 0) + 1

            for page in block.get("page_numbers", []) or []:
                page_key = str(page)
                by_page[page_key] = by_page.get(page_key, 0) + 1

        for block in evidence_blocks:
            evidence_type = block.get("evidence_type", "unknown")
            by_evidence_type[evidence_type] = by_evidence_type.get(evidence_type, 0) + 1

            for page in block.get("page_numbers", []) or []:
                page_key = str(page)
                by_page[page_key] = by_page.get(page_key, 0) + 1

        return {
            "has_prompt": bool(prompt_text),
            "query": query,
            "prompt_chars": len(prompt_text),
            "system_prompt_chars": len(system_prompt),
            "user_prompt_chars": len(user_prompt),
            "context_block_count": len(context_blocks),
            "evidence_block_count": len(evidence_blocks),
            "citation_block_count": len(citation_blocks),
            "metadata_chars": len(metadata_block.get("text", "")) if metadata_block else 0,
            "retrieved_item_count": len(retrieved_items),
            "expanded_context_item_count": len(expanded_context_items),
            "input_evidence_count": len(evidence_items),
            "input_citation_count": len(citation_items),
            "by_context_type": by_context_type,
            "by_evidence_type": by_evidence_type,
            "by_page": by_page,
            "requires_citations": self.config.require_citations,
            "requires_grounding": self.config.require_grounding,
        }

    def _marker_from_pages(
        self,
        page_numbers: List[int],
        config: PromptBuilderConfig,
    ) -> str:
        page_numbers = self._normalize_page_numbers(page_numbers)

        if len(page_numbers) == 1:
            return f"[tr.{page_numbers[0]}]"

        if len(page_numbers) > 1:
            return f"[tr.{page_numbers[0]}-{page_numbers[-1]}]"

        return config.citation_marker_fallback

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

    def _truncate_prompt(
        self,
        text: str,
        max_chars: int,
    ) -> str:
        text = normalize_pdf_text(text)

        if len(text) <= max_chars:
            return text

        head_budget = int(max_chars * 0.35)
        tail_budget = max_chars - head_budget - 120

        head = text[:head_budget]
        tail = text[-tail_budget:] if tail_budget > 0 else ""

        return normalize_pdf_text(
            head
            + "\n\n...[Nội dung prompt đã được rút gọn do vượt giới hạn ký tự]...\n\n"
            + tail
        )

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

    def save_result(
        self,
        prompt_result: Dict[str, Any],
        output_path: str,
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                json_safe(prompt_result),
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


def build_prompt(
    query: str = "",
    retrieval_result: Optional[Dict[str, Any]] = None,
    expanded_context_result: Optional[Dict[str, Any]] = None,
    evidence_aggregation_result: Optional[Dict[str, Any]] = None,
    citation_verification_result: Optional[Dict[str, Any]] = None,
    metadata_enrichment_result: Optional[Dict[str, Any]] = None,
    document_profile: Optional[Dict[str, Any]] = None,
    prompt_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = PromptBuilder()
    return builder.process(
        query=query,
        retrieval_result=retrieval_result,
        expanded_context_result=expanded_context_result,
        evidence_aggregation_result=evidence_aggregation_result,
        citation_verification_result=citation_verification_result,
        metadata_enrichment_result=metadata_enrichment_result,
        document_profile=document_profile,
        prompt_options=prompt_options,
    )


def build_rag_prompt(
    query: str,
    retrieval_result: Optional[Dict[str, Any]] = None,
    evidence_aggregation_result: Optional[Dict[str, Any]] = None,
    expanded_context_result: Optional[Dict[str, Any]] = None,
    citation_verification_result: Optional[Dict[str, Any]] = None,
    prompt_options: Optional[Dict[str, Any]] = None,
) -> str:
    builder = PromptBuilder()
    result = builder.process(
        query=query,
        retrieval_result=retrieval_result,
        evidence_aggregation_result=evidence_aggregation_result,
        expanded_context_result=expanded_context_result,
        citation_verification_result=citation_verification_result,
        prompt_options=prompt_options,
    )
    return result.get("prompt_text", "")
