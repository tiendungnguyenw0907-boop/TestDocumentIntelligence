"""
paragraph_continuation_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect paragraphs that continue across page boundaries.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- DocumentStructurePipeline
- SectionLinker

Output
------
Dictionary with:
- paragraph_continuations
- paragraph_continuations_by_page
- page_continuation_contexts
- paragraph_continuation_summary

Flow
----
PageUnderstandingPipeline
    ↓
DocumentStructurePipeline
    ↓
SectionLinker
    ↓
ParagraphContinuationDetector
    ↓
CrossPageContextGraphBuilder
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class ParagraphContinuationDetectorConfig:
    max_page_gap: int = 1

    use_document_structure_paragraphs: bool = True
    use_page_metadata_paragraphs: bool = True
    use_reading_order_fallback: bool = True
    use_text_lines_fallback: bool = True

    attach_to_pages: bool = True

    min_confidence: float = 0.35
    high_confidence_threshold: float = 0.70

    tail_max_chars: int = 500
    head_max_chars: int = 500
    min_tail_chars: int = 8
    min_head_chars: int = 3

    max_tail_lines: int = 4
    max_head_lines: int = 4

    ignore_header_footer: bool = True
    ignore_page_numbers: bool = True
    ignore_headings: bool = True
    ignore_captions: bool = True
    ignore_tables: bool = True

    require_same_section_when_available: bool = False
    section_match_bonus: float = 0.15

    include_debug: bool = True


@dataclass
class ParagraphEndpoint:
    endpoint_id: str
    page_number: int
    page_index: int

    text: str
    normalized_text: str

    source_type: str
    source_id: str = ""

    section_id: str = ""
    section_title: str = ""

    bbox: Optional[List[float]] = None
    line_count: int = 1
    char_count: int = 0

    is_heading_like: bool = False
    is_caption_like: bool = False
    is_table_like: bool = False
    is_page_number_like: bool = False

    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["bbox"] is None:
            data["bbox"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class ParagraphContinuation:
    paragraph_continuation_id: str

    from_page: int
    to_page: int

    from_page_index: int
    to_page_index: int

    from_endpoint_id: str
    to_endpoint_id: str

    from_text_tail: str
    to_text_head: str
    merged_preview: str

    continuation_type: str
    confidence: float

    same_section: bool = False
    section_id: str = ""
    section_title: str = ""

    source: str = "paragraph_continuation_detector"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class ParagraphContinuationDetector:
    def __init__(
        self,
        config: Optional[ParagraphContinuationDetectorConfig] = None,
    ):
        self.config = config or ParagraphContinuationDetectorConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Optional[Dict[str, Any]] = None,
        section_link_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        document_structure_result = document_structure_result or {}
        section_link_result = section_link_result or {}

        page_raws = sorted(
            page_raws,
            key=lambda page: page.page_number,
        )

        page_section_context = self._build_page_section_context(
            page_raws=page_raws,
            document_structure_result=document_structure_result,
            section_link_result=section_link_result,
        )

        endpoints_by_page = self._build_page_endpoints(
            page_raws=page_raws,
            document_structure_result=document_structure_result,
            page_section_context=page_section_context,
        )

        continuations = self._detect_continuations(
            page_raws=page_raws,
            endpoints_by_page=endpoints_by_page,
        )

        continuation_dicts = [
            item.to_dict() for item in continuations
        ]

        result = {
            "processor": "ParagraphContinuationDetector",
            "schema_version": "paragraph_continuation_detector_v1",
            "paragraph_continuations": continuation_dicts,
            "paragraph_continuations_by_page": self._group_continuations_by_page(continuation_dicts),
            "page_endpoints": {
                str(page_number): {
                    "tail_candidates": [
                        item.to_dict() for item in data.get("tail_candidates", [])
                    ],
                    "head_candidates": [
                        item.to_dict() for item in data.get("head_candidates", [])
                    ],
                }
                for page_number, data in endpoints_by_page.items()
            },
            "page_continuation_contexts": self._build_page_contexts(
                page_raws=page_raws,
                continuations=continuation_dicts,
            ),
            "paragraph_continuation_summary": self._build_summary(
                page_raws=page_raws,
                continuations=continuation_dicts,
            ),
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                result=result,
            )

        return result

    def _build_page_endpoints(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Dict[str, Any],
        page_section_context: Dict[str, Dict[str, Any]],
    ) -> Dict[int, Dict[str, List[ParagraphEndpoint]]]:
        endpoints_by_page: Dict[int, Dict[str, List[ParagraphEndpoint]]] = {}

        paragraph_index = self._build_paragraph_index(document_structure_result)

        for page_raw in page_raws:
            page_number = page_raw.page_number
            section_context = page_section_context.get(str(page_number), {})

            paragraphs = paragraph_index.get(page_number, [])

            if not paragraphs:
                paragraphs = self._collect_page_paragraphs_from_metadata(page_raw)

            if not paragraphs and self.config.use_reading_order_fallback:
                paragraphs = self._collect_page_paragraphs_from_reading_order(page_raw)

            if not paragraphs and self.config.use_text_lines_fallback:
                paragraphs = self._collect_page_paragraphs_from_text_lines(page_raw)

            paragraphs = self._filter_paragraph_like_items(paragraphs)

            tail_candidates = self._make_tail_candidates(
                page_raw=page_raw,
                paragraphs=paragraphs,
                section_context=section_context,
            )

            head_candidates = self._make_head_candidates(
                page_raw=page_raw,
                paragraphs=paragraphs,
                section_context=section_context,
            )

            endpoints_by_page[page_number] = {
                "tail_candidates": tail_candidates,
                "head_candidates": head_candidates,
            }

        return endpoints_by_page

    def _detect_continuations(
        self,
        page_raws: List[PageRaw],
        endpoints_by_page: Dict[int, Dict[str, List[ParagraphEndpoint]]],
    ) -> List[ParagraphContinuation]:
        continuations: List[ParagraphContinuation] = []

        pages = sorted(
            [
                page.page_number for page in page_raws
            ]
        )

        for index, current_page in enumerate(pages):
            current_data = endpoints_by_page.get(current_page, {})
            tail_candidates = current_data.get("tail_candidates", [])

            if not tail_candidates:
                continue

            for next_page in pages[index + 1:]:
                page_gap = next_page - current_page

                if page_gap <= 0:
                    continue

                if page_gap > self.config.max_page_gap:
                    break

                next_data = endpoints_by_page.get(next_page, {})
                head_candidates = next_data.get("head_candidates", [])

                if not head_candidates:
                    continue

                best = self._find_best_continuation_pair(
                    from_page=current_page,
                    to_page=next_page,
                    tail_candidates=tail_candidates,
                    head_candidates=head_candidates,
                )

                if not best:
                    continue

                tail, head, score, details = best

                if score < self.config.min_confidence:
                    continue

                continuation = self._make_continuation(
                    tail=tail,
                    head=head,
                    score=score,
                    details=details,
                )

                continuations.append(continuation)

        continuations = self._deduplicate_continuations(continuations)

        continuations = sorted(
            continuations,
            key=lambda item: (
                item.from_page,
                item.to_page,
                -item.confidence,
            ),
        )

        return continuations

    def _find_best_continuation_pair(
        self,
        from_page: int,
        to_page: int,
        tail_candidates: List[ParagraphEndpoint],
        head_candidates: List[ParagraphEndpoint],
    ) -> Optional[Tuple[ParagraphEndpoint, ParagraphEndpoint, float, Dict[str, Any]]]:
        best_tail = None
        best_head = None
        best_score = 0.0
        best_details: Dict[str, Any] = {}

        for tail in tail_candidates:
            for head in head_candidates:
                score, details = self._score_continuation_pair(
                    tail=tail,
                    head=head,
                    page_gap=to_page - from_page,
                )

                if score > best_score:
                    best_score = score
                    best_tail = tail
                    best_head = head
                    best_details = details

        if best_tail is None or best_head is None:
            return None

        return best_tail, best_head, best_score, best_details

    def _score_continuation_pair(
        self,
        tail: ParagraphEndpoint,
        head: ParagraphEndpoint,
        page_gap: int,
    ) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        reasons: List[str] = []

        tail_text = tail.text.strip()
        head_text = head.text.strip()

        if not tail_text or not head_text:
            return 0.0, {
                "reasons": ["empty_tail_or_head"],
            }

        if len(tail_text) < self.config.min_tail_chars:
            return 0.0, {
                "reasons": ["tail_too_short"],
            }

        if len(head_text) < self.config.min_head_chars:
            return 0.0, {
                "reasons": ["head_too_short"],
            }

        if self._should_ignore_endpoint(tail):
            return 0.0, {
                "reasons": ["tail_ignored"],
            }

        if self._should_ignore_endpoint(head):
            return 0.0, {
                "reasons": ["head_ignored"],
            }

        same_section = bool(
            tail.section_id
            and head.section_id
            and tail.section_id == head.section_id
        )

        different_section = bool(
            tail.section_id
            and head.section_id
            and tail.section_id != head.section_id
        )

        if self.config.require_same_section_when_available and different_section:
            return 0.0, {
                "reasons": ["different_section"],
                "tail_section_id": tail.section_id,
                "head_section_id": head.section_id,
            }

        if page_gap == 1:
            score += 0.08
            reasons.append("adjacent_pages")

        if same_section:
            score += self.config.section_match_bonus
            reasons.append("same_section")

        if different_section:
            score -= 0.08
            reasons.append("different_section_penalty")

        punctuation_score, punctuation_reason = self._tail_punctuation_score(tail_text)

        score += punctuation_score

        if punctuation_reason:
            reasons.append(punctuation_reason)

        head_start_score, head_start_reason = self._head_start_score(head_text)

        score += head_start_score

        if head_start_reason:
            reasons.append(head_start_reason)

        lexical_score, lexical_details = self._lexical_continuation_score(
            tail_text=tail_text,
            head_text=head_text,
        )

        score += lexical_score

        if lexical_score > 0:
            reasons.append("lexical_continuity")

        geometry_score, geometry_details = self._geometry_score(
            tail=tail,
            head=head,
        )

        score += geometry_score

        if geometry_score > 0:
            reasons.append("geometry_continuity")

        if tail_text.endswith("-") or tail_text.endswith("–"):
            score += 0.20
            reasons.append("hyphenated_word_break")

        if self._looks_like_sentence_completion(tail_text, head_text):
            score += 0.12
            reasons.append("sentence_completion")

        if self._tail_is_complete_sentence(tail_text) and self._head_starts_like_new_paragraph(head_text):
            score -= 0.20
            reasons.append("complete_sentence_new_paragraph_penalty")

        if self._head_starts_like_heading(head_text):
            score -= 0.25
            reasons.append("heading_start_penalty")

        if self._tail_or_head_table_like(tail_text, head_text):
            score -= 0.15
            reasons.append("table_like_penalty")

        score = round(max(0.0, min(score, 0.95)), 4)

        details = {
            "reasons": reasons,
            "same_section": same_section,
            "different_section": different_section,
            "page_gap": page_gap,
            "punctuation_score": round(punctuation_score, 4),
            "head_start_score": round(head_start_score, 4),
            "lexical_score": round(lexical_score, 4),
            "geometry_score": round(geometry_score, 4),
            "lexical_details": lexical_details,
            "geometry_details": geometry_details,
            "tail_source_type": tail.source_type,
            "head_source_type": head.source_type,
        }

        return score, details

    def _make_continuation(
        self,
        tail: ParagraphEndpoint,
        head: ParagraphEndpoint,
        score: float,
        details: Dict[str, Any],
    ) -> ParagraphContinuation:
        merged_preview = self._merge_preview(
            tail_text=tail.text,
            head_text=head.text,
        )

        continuation_type = "cross_page_paragraph"

        if tail.text.rstrip().endswith(("-", "–")):
            continuation_type = "hyphenated_cross_page_paragraph"
        elif details.get("same_section"):
            continuation_type = "same_section_cross_page_paragraph"

        same_section = bool(details.get("same_section", False))

        section_id = tail.section_id if same_section else ""
        section_title = tail.section_title if same_section else ""

        return ParagraphContinuation(
            paragraph_continuation_id=make_id("para_cont"),
            from_page=tail.page_number,
            to_page=head.page_number,
            from_page_index=tail.page_index,
            to_page_index=head.page_index,
            from_endpoint_id=tail.endpoint_id,
            to_endpoint_id=head.endpoint_id,
            from_text_tail=tail.text,
            to_text_head=head.text,
            merged_preview=merged_preview,
            continuation_type=continuation_type,
            confidence=score,
            same_section=same_section,
            section_id=section_id,
            section_title=section_title,
            source="paragraph_continuation_detector",
            metadata={
                "details": details,
                "tail": tail.to_dict(),
                "head": head.to_dict(),
                "quality": "high" if score >= self.config.high_confidence_threshold else "medium",
            },
        )

    def _make_tail_candidates(
        self,
        page_raw: PageRaw,
        paragraphs: List[Dict[str, Any]],
        section_context: Dict[str, Any],
    ) -> List[ParagraphEndpoint]:
        candidates: List[ParagraphEndpoint] = []

        sorted_paragraphs = sorted(
            paragraphs,
            key=lambda item: (
                self._bbox_y0(item.get("bbox", [])),
                self._safe_int(item.get("order"), default=0),
            ),
        )

        for item in reversed(sorted_paragraphs[-10:]):
            text = self._clean_text_block(
                item.get("text")
                or item.get("normalized_text")
                or item.get("content")
                or ""
            )

            if not text:
                continue

            text = self._tail_text(text)

            endpoint = self._make_endpoint(
                page_raw=page_raw,
                item=item,
                text=text,
                endpoint_kind="tail",
                section_context=section_context,
            )

            candidates.append(endpoint)

            if len(candidates) >= self.config.max_tail_lines:
                break

        return candidates

    def _make_head_candidates(
        self,
        page_raw: PageRaw,
        paragraphs: List[Dict[str, Any]],
        section_context: Dict[str, Any],
    ) -> List[ParagraphEndpoint]:
        candidates: List[ParagraphEndpoint] = []

        sorted_paragraphs = sorted(
            paragraphs,
            key=lambda item: (
                self._bbox_y0(item.get("bbox", [])),
                self._safe_int(item.get("order"), default=0),
            ),
        )

        for item in sorted_paragraphs[:10]:
            text = self._clean_text_block(
                item.get("text")
                or item.get("normalized_text")
                or item.get("content")
                or ""
            )

            if not text:
                continue

            text = self._head_text(text)

            endpoint = self._make_endpoint(
                page_raw=page_raw,
                item=item,
                text=text,
                endpoint_kind="head",
                section_context=section_context,
            )

            candidates.append(endpoint)

            if len(candidates) >= self.config.max_head_lines:
                break

        return candidates

    def _make_endpoint(
        self,
        page_raw: PageRaw,
        item: Dict[str, Any],
        text: str,
        endpoint_kind: str,
        section_context: Dict[str, Any],
    ) -> ParagraphEndpoint:
        paragraph_type = (
            item.get("paragraph_type")
            or item.get("type")
            or item.get("object_type")
            or ""
        )

        source_type = (
            item.get("source_type")
            or item.get("source")
            or paragraph_type
            or "paragraph"
        )

        source_id = (
            item.get("paragraph_id")
            or item.get("line_id")
            or item.get("block_id")
            or item.get("id")
            or ""
        )

        section_id = (
            item.get("section_id")
            or section_context.get("current_section_id", "")
            or ""
        )

        section_title = (
            item.get("section_title")
            or section_context.get("current_section_title", "")
            or ""
        )

        bbox = item.get("bbox", []) or []

        line_count = max(1, len(text.splitlines()))

        normalized_text = self._normalize_text(text)

        return ParagraphEndpoint(
            endpoint_id=make_id(f"para_{endpoint_kind}"),
            page_number=page_raw.page_number,
            page_index=page_raw.page_index,
            text=text,
            normalized_text=normalized_text,
            source_type=source_type,
            source_id=source_id,
            section_id=section_id,
            section_title=section_title,
            bbox=self._normalize_bbox(bbox),
            line_count=line_count,
            char_count=len(text),
            is_heading_like=self._is_heading_like_item(item, text),
            is_caption_like=self._is_caption_like_item(item, text),
            is_table_like=self._is_table_like_item(item, text),
            is_page_number_like=self._is_page_number_like(text),
            metadata={
                "endpoint_kind": endpoint_kind,
                "paragraph_type": paragraph_type,
                "raw_keys": sorted(list(item.keys())),
            },
        )

    def _build_paragraph_index(
        self,
        document_structure_result: Dict[str, Any],
    ) -> Dict[int, List[Dict[str, Any]]]:
        paragraph_index: Dict[int, List[Dict[str, Any]]] = {}

        if not self.config.use_document_structure_paragraphs:
            return paragraph_index

        paragraphs = document_structure_result.get("paragraphs", []) or []

        if not paragraphs:
            paragraph_result = document_structure_result.get("paragraph_result", {}) or {}
            paragraphs = paragraph_result.get("paragraphs", []) or []

        for index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                continue

            page_number = self._safe_int(paragraph.get("page_number"), default=0)

            if page_number <= 0:
                page_numbers = paragraph.get("page_numbers", []) or []

                if page_numbers:
                    page_number = self._safe_int(page_numbers[0], default=0)

            if page_number <= 0:
                continue

            item = dict(paragraph)
            item.setdefault("order", index)
            item.setdefault("source_type", "document_structure_paragraph")

            paragraph_index.setdefault(page_number, [])
            paragraph_index[page_number].append(item)

        return paragraph_index

    def _collect_page_paragraphs_from_metadata(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        if not self.config.use_page_metadata_paragraphs:
            return []

        collected: List[Dict[str, Any]] = []

        for meta_key in [
            "paragraph_builder",
            "document_structure_pipeline",
        ]:
            meta = page_raw.metadata.get(meta_key, {}) or {}

            for item_key in [
                "paragraphs_on_page",
                "paragraphs",
            ]:
                items = meta.get(item_key, []) or []

                for index, item in enumerate(items):
                    if not isinstance(item, dict):
                        continue

                    normalized = dict(item)
                    normalized.setdefault("page_number", page_raw.page_number)
                    normalized.setdefault("page_index", page_raw.page_index)
                    normalized.setdefault("order", index)
                    normalized.setdefault("source_type", meta_key)
                    collected.append(normalized)

        return collected

    def _collect_page_paragraphs_from_reading_order(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        collected: List[Dict[str, Any]] = []

        reading_meta = page_raw.metadata.get("reading_order_builder", {}) or {}
        items = reading_meta.get("reading_order_items", []) or []

        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            text = self._clean_text_block(item.get("text", ""))

            if not text:
                continue

            collected.append(
                {
                    "paragraph_id": item.get("item_id", "") or item.get("id", "") or make_id("ro_para"),
                    "text": text,
                    "bbox": item.get("bbox", []),
                    "page_number": page_raw.page_number,
                    "page_index": page_raw.page_index,
                    "order": item.get("order", index),
                    "paragraph_type": item.get("item_type", "reading_order_item"),
                    "source_type": "reading_order",
                    "metadata": item,
                }
            )

        return collected

    def _collect_page_paragraphs_from_text_lines(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        lines = []

        for index, line in enumerate(page_raw.text_lines):
            text = self._clean_text(
                self._get_attr_or_key(line, "text", "")
            )

            if not text:
                continue

            bbox = self._get_attr_or_key(line, "bbox", []) or []

            lines.append(
                {
                    "line_id": self._get_attr_or_key(line, "line_id", "") or make_id("line"),
                    "text": text,
                    "bbox": bbox,
                    "order": index,
                    "source_type": "text_line",
                }
            )

        if not lines:
            raw_lines = [
                self._clean_text(line)
                for line in (page_raw.normalized_text or page_raw.raw_text or "").splitlines()
                if self._clean_text(line)
            ]

            for index, text in enumerate(raw_lines):
                lines.append(
                    {
                        "line_id": make_id("raw_line"),
                        "text": text,
                        "bbox": [],
                        "order": index,
                        "source_type": "raw_text_line",
                    }
                )

        merged = self._merge_lines_to_paragraphs(
            lines=lines,
            page_raw=page_raw,
        )

        return merged

    def _merge_lines_to_paragraphs(
        self,
        lines: List[Dict[str, Any]],
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        if not lines:
            return []

        paragraphs: List[Dict[str, Any]] = []
        current: List[Dict[str, Any]] = []

        for line in lines:
            text = self._clean_text(line.get("text", ""))

            if not text:
                continue

            if self._is_page_number_like(text):
                continue

            if not current:
                current.append(line)
                continue

            previous_text = self._clean_text(current[-1].get("text", ""))

            should_continue = self._line_should_continue_paragraph(
                previous_text=previous_text,
                current_text=text,
                previous_bbox=current[-1].get("bbox", []),
                current_bbox=line.get("bbox", []),
            )

            if should_continue:
                current.append(line)
            else:
                paragraphs.append(
                    self._paragraph_from_lines(
                        lines=current,
                        page_raw=page_raw,
                        order=len(paragraphs),
                    )
                )
                current = [line]

        if current:
            paragraphs.append(
                self._paragraph_from_lines(
                    lines=current,
                    page_raw=page_raw,
                    order=len(paragraphs),
                )
            )

        return paragraphs

    def _paragraph_from_lines(
        self,
        lines: List[Dict[str, Any]],
        page_raw: PageRaw,
        order: int,
    ) -> Dict[str, Any]:
        text = " ".join(
            [
                self._clean_text(line.get("text", ""))
                for line in lines
                if self._clean_text(line.get("text", ""))
            ]
        )

        bboxes = [
            line.get("bbox", [])
            for line in lines
            if line.get("bbox") and len(line.get("bbox", [])) == 4
        ]

        bbox = self._merge_bboxes(bboxes)

        return {
            "paragraph_id": make_id("line_para"),
            "text": text,
            "bbox": bbox,
            "page_number": page_raw.page_number,
            "page_index": page_raw.page_index,
            "order": order,
            "paragraph_type": "line_merged_paragraph",
            "source_type": "text_lines_merged",
            "metadata": {
                "line_count": len(lines),
                "line_ids": [
                    line.get("line_id", "") for line in lines
                ],
            },
        }

    def _filter_paragraph_like_items(
        self,
        paragraphs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        result = []

        for item in paragraphs:
            text = self._clean_text_block(
                item.get("text")
                or item.get("normalized_text")
                or item.get("content")
                or ""
            )

            if not text:
                continue

            if self._is_page_number_like(text) and self.config.ignore_page_numbers:
                continue

            if self._is_header_footer_item(item) and self.config.ignore_header_footer:
                continue

            item = dict(item)
            item["text"] = text
            result.append(item)

        return result

    def _tail_punctuation_score(
        self,
        text: str,
    ) -> Tuple[float, str]:
        stripped = text.rstrip()

        if not stripped:
            return 0.0, ""

        if stripped.endswith(("-", "–")):
            return 0.22, "tail_hyphen"

        if stripped.endswith((",", ";", ":", "(", "[", "{")):
            return 0.22, "tail_open_punctuation"

        if stripped.endswith((".", "!", "?", ".”", "!”", "?”", '"')):
            return -0.15, "tail_terminal_punctuation"

        if re.search(r"\b(và|hoặc|của|với|theo|tại|trong|để|do|khi|nếu|gồm|bao gồm|là)\s*$", stripped, flags=re.IGNORECASE):
            return 0.25, "tail_ends_with_connector"

        return 0.18, "tail_no_terminal_punctuation"

    def _head_start_score(
        self,
        text: str,
    ) -> Tuple[float, str]:
        stripped = text.lstrip()

        if not stripped:
            return 0.0, ""

        first_char = stripped[:1]

        if first_char.islower():
            return 0.22, "head_starts_lowercase"

        if stripped.startswith((",", ";", ":", ")", "]")):
            return 0.18, "head_starts_with_continuation_punctuation"

        if re.match(r"^(và|hoặc|của|với|theo|tại|trong|để|do|khi|nếu|gồm|bao gồm|là)\b", stripped, flags=re.IGNORECASE):
            return 0.20, "head_starts_with_connector"

        if re.match(r"^\d+[\.\)]\s+", stripped):
            return -0.10, "head_starts_numbered_item"

        if self._head_starts_like_heading(stripped):
            return -0.20, "head_starts_heading_like"

        return 0.05, "head_neutral"

    def _lexical_continuation_score(
        self,
        tail_text: str,
        head_text: str,
    ) -> Tuple[float, Dict[str, Any]]:
        tail_tokens = self._tokenize(tail_text)
        head_tokens = self._tokenize(head_text)

        if not tail_tokens or not head_tokens:
            return 0.0, {
                "overlap": 0,
                "tail_tokens": len(tail_tokens),
                "head_tokens": len(head_tokens),
            }

        tail_set = set(tail_tokens)
        head_set = set(head_tokens)

        overlap = tail_set.intersection(head_set)
        overlap_ratio = len(overlap) / max(min(len(tail_set), len(head_set)), 1)

        score = 0.0

        if overlap_ratio >= 0.15:
            score += min(0.10, overlap_ratio * 0.20)

        tail_last = tail_tokens[-1]
        head_first = head_tokens[0]

        if tail_last and head_first and tail_last == head_first:
            score -= 0.05

        if self._looks_like_broken_word(tail_text, head_text):
            score += 0.18

        return round(score, 4), {
            "overlap": len(overlap),
            "overlap_ratio": round(overlap_ratio, 4),
            "overlap_tokens": sorted(list(overlap))[:10],
            "tail_token_count": len(tail_tokens),
            "head_token_count": len(head_tokens),
        }

    def _geometry_score(
        self,
        tail: ParagraphEndpoint,
        head: ParagraphEndpoint,
    ) -> Tuple[float, Dict[str, Any]]:
        tail_bbox = tail.bbox or []
        head_bbox = head.bbox or []

        if len(tail_bbox) != 4 or len(head_bbox) != 4:
            return 0.0, {
                "has_geometry": False,
            }

        score = 0.0

        x0_diff = abs(float(tail_bbox[0]) - float(head_bbox[0]))
        width_tail = max(float(tail_bbox[2]) - float(tail_bbox[0]), 1.0)
        width_head = max(float(head_bbox[2]) - float(head_bbox[0]), 1.0)

        width_similarity = min(width_tail, width_head) / max(width_tail, width_head)

        if x0_diff <= 30:
            score += 0.07

        if width_similarity >= 0.70:
            score += 0.06

        return round(score, 4), {
            "has_geometry": True,
            "x0_diff": round(x0_diff, 4),
            "width_similarity": round(width_similarity, 4),
        }

    def _looks_like_sentence_completion(
        self,
        tail_text: str,
        head_text: str,
    ) -> bool:
        tail = tail_text.rstrip()
        head = head_text.lstrip()

        if not tail or not head:
            return False

        if tail.endswith((",", ";", ":", "-", "–")):
            return True

        if re.search(r"\b(và|hoặc|của|với|theo|tại|trong|để|do|khi|nếu|gồm|bao gồm|là)\s*$", tail, flags=re.IGNORECASE):
            return True

        if head[:1].islower():
            return True

        return False

    def _tail_is_complete_sentence(
        self,
        text: str,
    ) -> bool:
        text = text.rstrip()

        if not text:
            return False

        return bool(re.search(r"[\.\!\?][”\"]?$", text))

    def _head_starts_like_new_paragraph(
        self,
        text: str,
    ) -> bool:
        text = text.lstrip()

        if not text:
            return False

        if re.match(r"^(\d+(\.\d+)*[\.\)]|[a-zA-Z][\.\)]|[-•–])\s+", text):
            return True

        if text[:1].isupper():
            return True

        return False

    def _head_starts_like_heading(
        self,
        text: str,
    ) -> bool:
        text = self._clean_text(text)

        patterns = [
            r"^(CHƯƠNG|Chương)\s+[IVXLC\d]+",
            r"^(PHẦN|Phần)\s+[IVXLC\d]+",
            r"^[IVXLC]+\.\s+",
            r"^\d+(\.\d+)*[\.\)]\s+[A-ZĐĂÂÊÔƠƯ]",
            r"^[A-ZĐĂÂÊÔƠƯ][A-ZĐĂÂÊÔƠƯ\s,\-:]{8,}$",
        ]

        return any(re.search(pattern, text) for pattern in patterns)

    def _tail_or_head_table_like(
        self,
        tail_text: str,
        head_text: str,
    ) -> bool:
        return self._is_table_like_text(tail_text) or self._is_table_like_text(head_text)

    def _line_should_continue_paragraph(
        self,
        previous_text: str,
        current_text: str,
        previous_bbox: List[float],
        current_bbox: List[float],
    ) -> bool:
        if not previous_text or not current_text:
            return False

        if self._head_starts_like_heading(current_text):
            return False

        if previous_text.endswith((".", "!", "?")) and current_text[:1].isupper():
            return False

        if previous_text.endswith((",", ";", ":", "-", "–")):
            return True

        if current_text[:1].islower():
            return True

        if len(previous_bbox) == 4 and len(current_bbox) == 4:
            x_diff = abs(float(previous_bbox[0]) - float(current_bbox[0]))

            if x_diff <= 25 and not self._head_starts_like_heading(current_text):
                return True

        return False

    def _should_ignore_endpoint(
        self,
        endpoint: ParagraphEndpoint,
    ) -> bool:
        if self.config.ignore_page_numbers and endpoint.is_page_number_like:
            return True

        if self.config.ignore_headings and endpoint.is_heading_like:
            return True

        if self.config.ignore_captions and endpoint.is_caption_like:
            return True

        if self.config.ignore_tables and endpoint.is_table_like:
            return True

        return False

    def _is_heading_like_item(
        self,
        item: Dict[str, Any],
        text: str,
    ) -> bool:
        paragraph_type = str(
            item.get("paragraph_type")
            or item.get("type")
            or item.get("object_type")
            or ""
        ).lower()

        if "heading" in paragraph_type or "title" in paragraph_type:
            return True

        return self._head_starts_like_heading(text)

    def _is_caption_like_item(
        self,
        item: Dict[str, Any],
        text: str,
    ) -> bool:
        paragraph_type = str(
            item.get("paragraph_type")
            or item.get("type")
            or item.get("object_type")
            or ""
        ).lower()

        if "caption" in paragraph_type:
            return True

        return bool(
            re.match(
                r"^\s*(Bảng|Bang|Table|Hình|Hinh|Figure|Fig\.|Sơ đồ|So do)\s+\d+",
                text,
                flags=re.IGNORECASE,
            )
        )

    def _is_table_like_item(
        self,
        item: Dict[str, Any],
        text: str,
    ) -> bool:
        paragraph_type = str(
            item.get("paragraph_type")
            or item.get("type")
            or item.get("object_type")
            or ""
        ).lower()

        if "table" in paragraph_type:
            return True

        return self._is_table_like_text(text)

    def _is_table_like_text(
        self,
        text: str,
    ) -> bool:
        if not text:
            return False

        if "|" in text:
            return True

        if "\t" in text:
            return True

        numeric_tokens = re.findall(r"\b\d+(?:[,.]\d+)?%?\b", text)
        all_tokens = re.findall(r"\S+", text)

        if len(all_tokens) >= 5 and len(numeric_tokens) / max(len(all_tokens), 1) >= 0.50:
            return True

        if re.search(r"\s{3,}", text) and len(all_tokens) >= 4:
            return True

        return False

    def _is_header_footer_item(
        self,
        item: Dict[str, Any],
    ) -> bool:
        item_type = str(
            item.get("region_type")
            or item.get("paragraph_type")
            or item.get("type")
            or ""
        ).lower()

        if "header" in item_type or "footer" in item_type:
            return True

        metadata = item.get("metadata", {}) or {}

        for key in ["is_header", "is_footer", "header_footer_role"]:
            value = metadata.get(key)

            if value:
                return True

        return False

    def _is_page_number_like(
        self,
        text: str,
    ) -> bool:
        text = self._clean_text(text)

        if not text:
            return False

        if re.fullmatch(r"\d{1,4}", text):
            return True

        if re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", text):
            return True

        if re.fullmatch(r"(trang|page)\s+\d{1,4}", text, flags=re.IGNORECASE):
            return True

        return False

    def _looks_like_broken_word(
        self,
        tail_text: str,
        head_text: str,
    ) -> bool:
        tail = tail_text.rstrip()
        head = head_text.lstrip()

        if tail.endswith(("-", "–")) and head[:1].islower():
            return True

        tail_last = re.findall(r"[A-Za-zÀ-ỹĐđ]+$", tail)
        head_first = re.findall(r"^[A-Za-zÀ-ỹĐđ]+", head)

        if not tail_last or not head_first:
            return False

        if len(tail_last[0]) <= 3 and head[:1].islower():
            return True

        return False

    def _tail_text(
        self,
        text: str,
    ) -> str:
        text = self._clean_text_block(text)

        if len(text) <= self.config.tail_max_chars:
            return text

        return text[-self.config.tail_max_chars:].strip()

    def _head_text(
        self,
        text: str,
    ) -> str:
        text = self._clean_text_block(text)

        if len(text) <= self.config.head_max_chars:
            return text

        return text[: self.config.head_max_chars].strip()

    def _merge_preview(
        self,
        tail_text: str,
        head_text: str,
    ) -> str:
        tail = self._clean_text_block(tail_text)
        head = self._clean_text_block(head_text)

        if tail.endswith(("-", "–")):
            return (tail[:-1] + head).strip()

        if tail.endswith(("(", "[", "{")):
            return (tail + head).strip()

        if head.startswith((",", ";", ":", ")", "]")):
            return (tail + head).strip()

        return (tail + " " + head).strip()

    def _build_page_section_context(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Dict[str, Any],
        section_link_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        contexts: Dict[str, Dict[str, Any]] = {}

        for page_raw in page_raws:
            contexts[str(page_raw.page_number)] = {
                "current_section_id": "",
                "current_section_title": "",
                "active_sections": [],
            }

        page_sections = section_link_result.get("page_sections", {}) or {}

        if page_sections:
            for page_key, ctx in page_sections.items():
                current = ctx.get("current_section", {}) or {}

                contexts[str(page_key)] = {
                    "current_section_id": current.get("section_id", ctx.get("current_section_id", "")),
                    "current_section_title": current.get("title", ctx.get("current_section_title", "")),
                    "active_sections": ctx.get("active_sections", []) or [],
                }

            return contexts

        links_by_page = section_link_result.get("section_links_by_page", {}) or {}

        if links_by_page:
            for page_key, links in links_by_page.items():
                links = links or []

                sorted_links = sorted(
                    links,
                    key=lambda item: (
                        self._safe_int(item.get("level"), default=0),
                        self._safe_int(item.get("order"), default=0),
                    ),
                )

                current = sorted_links[-1] if sorted_links else {}

                contexts[str(page_key)] = {
                    "current_section_id": current.get("section_id", ""),
                    "current_section_title": current.get("title", ""),
                    "active_sections": sorted_links,
                }

            return contexts

        sections = document_structure_result.get("sections", []) or []

        for page_raw in page_raws:
            active_sections = []

            for section in sections:
                page_numbers = section.get("page_numbers", []) or section.get("content_page_numbers", []) or []

                if not page_numbers:
                    page_start = section.get("page_start")
                    page_end = section.get("page_end")

                    if page_start is not None and page_end is not None:
                        try:
                            page_numbers = list(range(int(page_start), int(page_end) + 1))
                        except Exception:
                            page_numbers = []

                if page_raw.page_number in page_numbers:
                    active_sections.append(section)

            active_sections = sorted(
                active_sections,
                key=lambda item: (
                    self._safe_int(item.get("level"), default=0),
                    self._safe_int(item.get("order"), default=0),
                ),
            )

            current = active_sections[-1] if active_sections else {}

            contexts[str(page_raw.page_number)] = {
                "current_section_id": current.get("section_id", ""),
                "current_section_title": current.get("title", ""),
                "active_sections": active_sections,
            }

        return contexts

    def _group_continuations_by_page(
        self,
        continuations: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in continuations:
            for key in ["from_page", "to_page"]:
                page_number = item.get(key)

                if page_number is None:
                    continue

                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(item)

        return grouped

    def _build_page_contexts(
        self,
        page_raws: List[PageRaw],
        continuations: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        by_page = self._group_continuations_by_page(continuations)
        contexts: Dict[str, Dict[str, Any]] = {}

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)
            items = by_page.get(page_key, [])

            incoming = [
                item for item in items
                if item.get("to_page") == page_raw.page_number
            ]

            outgoing = [
                item for item in items
                if item.get("from_page") == page_raw.page_number
            ]

            contexts[page_key] = {
                "page_number": page_raw.page_number,
                "incoming_continuations": incoming,
                "outgoing_continuations": outgoing,
                "incoming_count": len(incoming),
                "outgoing_count": len(outgoing),
                "has_incoming": len(incoming) > 0,
                "has_outgoing": len(outgoing) > 0,
                "neighbor_pages": sorted(
                    list(
                        dict.fromkeys(
                            [
                                item.get("from_page")
                                for item in incoming
                            ]
                            + [
                                item.get("to_page")
                                for item in outgoing
                            ]
                        )
                    )
                ),
            }

        return contexts

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        by_page = result.get("paragraph_continuations_by_page", {})
        page_contexts = result.get("page_continuation_contexts", {})
        summary = result.get("paragraph_continuation_summary", {})

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("paragraph_continuation_detector", {})
            page_raw.metadata["paragraph_continuation_detector"] = {
                "processor": "ParagraphContinuationDetector",
                "paragraph_continuations_on_page": by_page.get(page_key, []),
                "page_continuation_context": page_contexts.get(page_key, {}),
                "paragraph_continuation_count_on_page": len(by_page.get(page_key, [])),
                "paragraph_continuation_summary": summary,
            }

    def _build_summary(
        self,
        page_raws: List[PageRaw],
        continuations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        by_page = self._group_continuations_by_page(continuations)

        by_type: Dict[str, int] = {}

        high_confidence_count = 0

        for item in continuations:
            continuation_type = item.get("continuation_type", "unknown")
            by_type[continuation_type] = by_type.get(continuation_type, 0) + 1

            if self._safe_float(item.get("confidence"), default=0.0) >= self.config.high_confidence_threshold:
                high_confidence_count += 1

        return {
            "has_paragraph_continuations": len(continuations) > 0,
            "page_count": len(page_raws),
            "paragraph_continuation_count": len(continuations),
            "high_confidence_count": high_confidence_count,
            "page_count_with_continuations": len(by_page),
            "by_type": by_type,
            "by_page": {
                page_key: len(items)
                for page_key, items in by_page.items()
            },
        }

    def _deduplicate_continuations(
        self,
        continuations: List[ParagraphContinuation],
    ) -> List[ParagraphContinuation]:
        seen = set()
        result: List[ParagraphContinuation] = []

        sorted_items = sorted(
            continuations,
            key=lambda item: (
                item.from_page,
                item.to_page,
                -item.confidence,
            ),
        )

        for item in sorted_items:
            key = (
                item.from_page,
                item.to_page,
                item.from_endpoint_id,
                item.to_endpoint_id,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _tokenize(
        self,
        text: str,
    ) -> List[str]:
        text = self._normalize_text(text)
        tokens = re.findall(r"[a-zA-ZÀ-ỹĐđ0-9_]+", text)

        stopwords = {
            "và", "hoặc", "của", "cho", "trong", "trên", "dưới",
            "các", "những", "một", "được", "không", "theo", "với",
            "từ", "đến", "là", "có", "này", "đó", "để",
            "and", "or", "of", "the", "a", "an", "to", "for", "in",
        }

        return [
            token for token in tokens
            if token.lower() not in stopwords and len(token) >= 2
        ]

    def _normalize_text(
        self,
        text: Any,
    ) -> str:
        return self._clean_text_block(text).lower()

    def _normalize_bbox(
        self,
        bbox: Any,
    ) -> List[float]:
        if not bbox or len(bbox) != 4:
            return []

        try:
            return [
                round(float(bbox[0]), 4),
                round(float(bbox[1]), 4),
                round(float(bbox[2]), 4),
                round(float(bbox[3]), 4),
            ]
        except Exception:
            return []

    def _merge_bboxes(
        self,
        bboxes: List[List[float]],
    ) -> List[float]:
        normalized = [
            self._normalize_bbox(bbox)
            for bbox in bboxes
        ]

        normalized = [
            bbox for bbox in normalized
            if len(bbox) == 4
        ]

        if not normalized:
            return []

        return [
            min(bbox[0] for bbox in normalized),
            min(bbox[1] for bbox in normalized),
            max(bbox[2] for bbox in normalized),
            max(bbox[3] for bbox in normalized),
        ]

    def _bbox_y0(
        self,
        bbox: Any,
    ) -> float:
        bbox = self._normalize_bbox(bbox)

        if len(bbox) != 4:
            return 999999.0

        return float(bbox[1])

    def _get_attr_or_key(
        self,
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        if obj is None:
            return default

        if isinstance(obj, dict):
            return obj.get(key, default)

        return getattr(obj, key, default)

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


def detect_paragraph_continuations(
    page_raws: List[PageRaw],
    document_structure_result: Optional[Dict[str, Any]] = None,
    section_link_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    detector = ParagraphContinuationDetector()
    return detector.process(
        page_raws=page_raws,
        document_structure_result=document_structure_result,
        section_link_result=section_link_result,
    )
