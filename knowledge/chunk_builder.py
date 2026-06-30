"""
chunk_builder.py

Production V1 - Colab Ready

Purpose
-------
Build text chunks from page_raws and document_structure_result.

Used by:
- KnowledgePipeline
- ParentChildChunkBuilder
- EvidenceBuilder
- Indexing
- RAGPipeline

Input
-----
- page_raws: List[PageRaw]
- document_structure_result: Dict

Output
------
Dictionary with:
- chunks
- chunks_by_page
- chunks_by_section
- chunks_by_type
- chunk_summary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw
from document_ai.schemas.chunk_schema import (
    Chunk,
    ChunkCollection,
    make_id,
    normalize_text,
    normalize_text_for_match,
)


@dataclass
class ChunkBuilderConfig:
    build_page_chunks: bool = True
    build_section_chunks: bool = True
    build_paragraph_chunks: bool = True

    prefer_document_structure: bool = True
    attach_to_pages: bool = True
    deduplicate_chunks: bool = True

    min_chunk_chars: int = 80
    max_chunk_chars: int = 1800
    overlap_chars: int = 180

    max_page_chunk_chars: int = 2200
    max_section_chunk_chars: int = 2600
    max_paragraph_chunk_chars: int = 1800

    split_long_chunks: bool = True
    preserve_sentence_boundary: bool = True

    include_page_context: bool = True
    include_section_context: bool = True
    include_metadata: bool = True

    chunk_prefix_page: str = "page_chunk"
    chunk_prefix_section: str = "section_chunk"
    chunk_prefix_paragraph: str = "paragraph_chunk"

    include_debug: bool = True


class ChunkBuilder:
    def __init__(
        self,
        config: Optional[ChunkBuilderConfig] = None,
    ):
        self.config = config or ChunkBuilderConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        cross_page_context_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        document_structure_result = document_structure_result or {}
        table_understanding_result = table_understanding_result or {}
        cross_page_context_result = cross_page_context_result or {}

        page_raws = sorted(
            page_raws or [],
            key=lambda item: item.page_number,
        )

        page_text_map = self._build_page_text_map(page_raws)

        sections = self._collect_sections(document_structure_result)
        paragraphs = self._collect_paragraphs(document_structure_result)

        chunks: List[Chunk] = []

        if self.config.build_section_chunks:
            chunks.extend(
                self._build_section_chunks(
                    sections=sections,
                    paragraphs=paragraphs,
                    page_text_map=page_text_map,
                    document_structure_result=document_structure_result,
                )
            )

        if self.config.build_paragraph_chunks:
            chunks.extend(
                self._build_paragraph_chunks(
                    paragraphs=paragraphs,
                    sections=sections,
                )
            )

        if self.config.build_page_chunks:
            chunks.extend(
                self._build_page_chunks(
                    page_raws=page_raws,
                    page_text_map=page_text_map,
                    sections=sections,
                    paragraphs=paragraphs,
                )
            )

        chunks = self._sort_chunks(chunks)

        if self.config.split_long_chunks:
            chunks = self._split_long_chunks(chunks)

        chunks = self._reorder_chunks(chunks)
        chunks = self._link_sequential_chunks(chunks)

        if self.config.deduplicate_chunks:
            chunks = self._deduplicate_chunks(chunks)

        collection = ChunkCollection(
            document_id=self._infer_document_id(page_raws, document_structure_result),
            source_document=self._infer_source_document(page_raws),
            chunks=chunks,
            metadata={
                "processor": "ChunkBuilder",
                "schema_version": "chunk_builder_v1",
                "table_understanding_available": bool(table_understanding_result),
                "cross_page_context_available": bool(cross_page_context_result),
            },
        )

        result = collection.to_dict()
        result.update(
            {
                "processor": "ChunkBuilder",
                "schema_version": "chunk_builder_v1",
                "chunk_builder_summary": self._build_summary(
                    chunks=chunks,
                    page_raws=page_raws,
                    sections=sections,
                    paragraphs=paragraphs,
                ),
                "config": asdict(self.config),
            }
        )

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                chunks=chunks,
            )

        return result

    def _build_page_chunks(
        self,
        page_raws: List[PageRaw],
        page_text_map: Dict[int, str],
        sections: List[Dict[str, Any]],
        paragraphs: List[Dict[str, Any]],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []

        sections_by_page = self._group_sections_by_page(sections)
        paragraphs_by_page = self._group_paragraphs_by_page(paragraphs)

        for page_raw in page_raws:
            page_number = page_raw.page_number
            text = normalize_text(page_text_map.get(page_number, ""))

            if not self._valid_text(text):
                continue

            page_sections = sections_by_page.get(str(page_number), [])
            page_paragraphs = paragraphs_by_page.get(str(page_number), [])

            section_id = ""
            section_title = ""
            section_level = None

            if page_sections:
                selected_section = sorted(
                    page_sections,
                    key=lambda item: (
                        self._safe_int(item.get("level"), default=0),
                        self._safe_int(item.get("order"), default=0),
                    ),
                )[-1]

                section_id = selected_section.get("section_id", "")
                section_title = selected_section.get("title", "")
                section_level = self._safe_int(selected_section.get("level"), default=0)

            chunk_text = text

            if len(chunk_text) > self.config.max_page_chunk_chars:
                split_texts = self._split_text(
                    text=chunk_text,
                    max_chars=self.config.max_page_chunk_chars,
                    overlap_chars=self.config.overlap_chars,
                )
            else:
                split_texts = [chunk_text]

            for split_index, split_text in enumerate(split_texts):
                if not self._valid_text(split_text):
                    continue

                chunks.append(
                    Chunk(
                        chunk_id=make_id(self.config.chunk_prefix_page),
                        chunk_type="page_chunk",
                        text=split_text,
                        document_id=page_raw.document_id,
                        source_document=page_raw.source_document,
                        page_numbers=[page_number],
                        page_start=page_number,
                        page_end=page_number,
                        section_id=section_id,
                        section_title=section_title,
                        section_level=section_level,
                        order=0,
                        source="chunk_builder_page",
                        confidence=0.70,
                        metadata={
                            "page_number": page_number,
                            "page_index": page_raw.page_index,
                            "split_index": split_index,
                            "split_count": len(split_texts),
                            "paragraph_count_on_page": len(page_paragraphs),
                            "section_count_on_page": len(page_sections),
                            "page_summary": page_raw.summary() if hasattr(page_raw, "summary") else {},
                        },
                    )
                )

        return chunks

    def _build_section_chunks(
        self,
        sections: List[Dict[str, Any]],
        paragraphs: List[Dict[str, Any]],
        page_text_map: Dict[int, str],
        document_structure_result: Dict[str, Any],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []

        if not sections:
            return chunks

        paragraphs_by_section = self._group_paragraphs_by_section(paragraphs)

        for section in sections:
            section_id = section.get("section_id", "") or section.get("id", "")
            section_title = normalize_text(section.get("title", "") or section.get("heading", ""))

            if not section_id and not section_title:
                continue

            page_numbers = self._resolve_page_numbers(section)

            section_paragraphs = paragraphs_by_section.get(section_id, [])

            section_text = self._build_section_text(
                section=section,
                section_paragraphs=section_paragraphs,
                page_text_map=page_text_map,
            )

            if not self._valid_text(section_text):
                continue

            max_chars = self.config.max_section_chunk_chars

            if len(section_text) > max_chars:
                split_texts = self._split_text(
                    text=section_text,
                    max_chars=max_chars,
                    overlap_chars=self.config.overlap_chars,
                )
            else:
                split_texts = [section_text]

            for split_index, split_text in enumerate(split_texts):
                if not self._valid_text(split_text):
                    continue

                chunks.append(
                    Chunk(
                        chunk_id=make_id(self.config.chunk_prefix_section),
                        chunk_type="section_chunk",
                        text=split_text,
                        document_id=document_structure_result.get("document_id", ""),
                        source_document=document_structure_result.get("source_document", ""),
                        page_numbers=page_numbers,
                        page_start=min(page_numbers) if page_numbers else None,
                        page_end=max(page_numbers) if page_numbers else None,
                        section_id=section_id,
                        section_title=section_title,
                        section_level=self._safe_int(section.get("level"), default=0),
                        order=self._safe_int(section.get("order"), default=0),
                        source="chunk_builder_section",
                        confidence=self._safe_float(section.get("confidence"), default=0.72),
                        metadata={
                            "section": self._compact_section(section),
                            "split_index": split_index,
                            "split_count": len(split_texts),
                            "paragraph_count": len(section_paragraphs),
                        },
                    )
                )

        return chunks

    def _build_paragraph_chunks(
        self,
        paragraphs: List[Dict[str, Any]],
        sections: List[Dict[str, Any]],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []

        if not paragraphs:
            return chunks

        section_by_id = {
            item.get("section_id", ""): item
            for item in sections
            if item.get("section_id")
        }

        for paragraph in paragraphs:
            paragraph_id = paragraph.get("paragraph_id", "") or paragraph.get("id", "")
            text = normalize_text(paragraph.get("text", ""))

            if not self._valid_text(text):
                continue

            section_id = paragraph.get("section_id", "")
            section = section_by_id.get(section_id, {})

            section_title = (
                paragraph.get("section_title")
                or section.get("title")
                or ""
            )

            page_numbers = self._resolve_page_numbers(paragraph)

            if not page_numbers and paragraph.get("page_number"):
                page_numbers = [self._safe_int(paragraph.get("page_number"), default=0)]
                page_numbers = [page for page in page_numbers if page > 0]

            max_chars = self.config.max_paragraph_chunk_chars

            if len(text) > max_chars:
                split_texts = self._split_text(
                    text=text,
                    max_chars=max_chars,
                    overlap_chars=self.config.overlap_chars,
                )
            else:
                split_texts = [text]

            for split_index, split_text in enumerate(split_texts):
                if not self._valid_text(split_text):
                    continue

                chunks.append(
                    Chunk(
                        chunk_id=make_id(self.config.chunk_prefix_paragraph),
                        chunk_type="paragraph_chunk",
                        text=split_text,
                        document_id=paragraph.get("document_id", ""),
                        source_document=paragraph.get("source_document", ""),
                        page_numbers=page_numbers,
                        page_start=min(page_numbers) if page_numbers else None,
                        page_end=max(page_numbers) if page_numbers else None,
                        section_id=section_id,
                        section_title=normalize_text(section_title),
                        section_level=self._safe_int(section.get("level"), default=None),
                        paragraph_id=paragraph_id,
                        order=self._safe_int(paragraph.get("global_order") or paragraph.get("order"), default=0),
                        source="chunk_builder_paragraph",
                        confidence=self._safe_float(paragraph.get("confidence"), default=0.74),
                        metadata={
                            "paragraph_type": paragraph.get("paragraph_type", "paragraph"),
                            "split_index": split_index,
                            "split_count": len(split_texts),
                            "paragraph": self._compact_paragraph(paragraph),
                        },
                    )
                )

        return chunks

    def _build_section_text(
        self,
        section: Dict[str, Any],
        section_paragraphs: List[Dict[str, Any]],
        page_text_map: Dict[int, str],
    ) -> str:
        title = normalize_text(section.get("title", "") or section.get("heading", ""))
        parts = []

        if title:
            parts.append(title)

        if section_paragraphs:
            section_paragraphs = sorted(
                section_paragraphs,
                key=lambda item: (
                    self._safe_int(item.get("page_number"), default=999999),
                    self._safe_int(item.get("global_order") or item.get("order"), default=999999),
                ),
            )

            for paragraph in section_paragraphs:
                text = normalize_text(paragraph.get("text", ""))

                if text:
                    parts.append(text)

            return "\n\n".join(parts)

        text_preview = normalize_text(section.get("text_preview", ""))

        if text_preview:
            parts.append(text_preview)
            return "\n\n".join(parts)

        page_numbers = self._resolve_page_numbers(section)

        for page_number in page_numbers:
            page_text = normalize_text(page_text_map.get(page_number, ""))

            if page_text:
                parts.append(page_text)

        return "\n\n".join(parts)

    def _split_long_chunks(
        self,
        chunks: List[Chunk],
    ) -> List[Chunk]:
        result: List[Chunk] = []

        for chunk in chunks:
            max_chars = self._max_chars_for_chunk(chunk)

            if len(chunk.text) <= max_chars:
                result.append(chunk)
                continue

            split_texts = self._split_text(
                text=chunk.text,
                max_chars=max_chars,
                overlap_chars=self.config.overlap_chars,
            )

            if len(split_texts) <= 1:
                result.append(chunk)
                continue

            child_ids = []

            for split_index, split_text in enumerate(split_texts):
                if not self._valid_text(split_text):
                    continue

                split_chunk = Chunk(
                    chunk_id=make_id(chunk.chunk_type.replace("_chunk", "_split")),
                    chunk_type=chunk.chunk_type,
                    text=split_text,
                    document_id=chunk.document_id,
                    source_document=chunk.source_document,
                    page_numbers=chunk.page_numbers,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_id=chunk.section_id,
                    section_title=chunk.section_title,
                    section_level=chunk.section_level,
                    paragraph_id=chunk.paragraph_id,
                    table_grid_id=chunk.table_grid_id,
                    table_structure_id=chunk.table_structure_id,
                    table_semantic_id=chunk.table_semantic_id,
                    table_boundary_id=chunk.table_boundary_id,
                    bbox=chunk.bbox,
                    order=chunk.order,
                    parent_chunk_id=chunk.chunk_id,
                    source=f"{chunk.source}_split",
                    confidence=chunk.confidence,
                    metadata={
                        **(chunk.metadata or {}),
                        "original_chunk_id": chunk.chunk_id,
                        "split_index": split_index,
                        "split_count": len(split_texts),
                    },
                )

                child_ids.append(split_chunk.chunk_id)
                result.append(split_chunk)

            parent_chunk = Chunk(
                chunk_id=chunk.chunk_id,
                chunk_type="parent_chunk",
                text=chunk.preview(max_chars=800),
                document_id=chunk.document_id,
                source_document=chunk.source_document,
                page_numbers=chunk.page_numbers,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_id=chunk.section_id,
                section_title=chunk.section_title,
                section_level=chunk.section_level,
                paragraph_id=chunk.paragraph_id,
                bbox=chunk.bbox,
                order=chunk.order,
                child_chunk_ids=child_ids,
                source=f"{chunk.source}_parent",
                confidence=chunk.confidence,
                metadata={
                    **(chunk.metadata or {}),
                    "is_parent_for_split_chunks": True,
                    "child_count": len(child_ids),
                    "original_char_count": len(chunk.text),
                },
            )

            result.append(parent_chunk)

        return result

    def _split_text(
        self,
        text: str,
        max_chars: int,
        overlap_chars: int = 0,
    ) -> List[str]:
        text = normalize_text(text)

        if len(text) <= max_chars:
            return [text]

        if self.config.preserve_sentence_boundary:
            units = self._split_into_sentences_or_paragraphs(text)
        else:
            units = [text[index:index + max_chars] for index in range(0, len(text), max_chars)]

        chunks = []
        current = ""

        for unit in units:
            unit = normalize_text(unit)

            if not unit:
                continue

            if len(unit) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""

                chunks.extend(
                    self._hard_split_text(
                        text=unit,
                        max_chars=max_chars,
                        overlap_chars=overlap_chars,
                    )
                )
                continue

            if not current:
                current = unit
                continue

            candidate = current + "\n\n" + unit

            if len(candidate) <= max_chars:
                current = candidate
            else:
                chunks.append(current)

                if overlap_chars > 0:
                    overlap = self._tail_text(current, overlap_chars)
                    current = normalize_text(overlap + "\n\n" + unit) if overlap else unit
                else:
                    current = unit

        if current:
            chunks.append(current)

        chunks = [
            normalize_text(item)
            for item in chunks
            if normalize_text(item)
        ]

        return chunks

    def _split_into_sentences_or_paragraphs(
        self,
        text: str,
    ) -> List[str]:
        paragraphs = [
            normalize_text(item)
            for item in re.split(r"\n\s*\n", text)
            if normalize_text(item)
        ]

        units = []

        for paragraph in paragraphs:
            if len(paragraph) <= max(self.config.max_chunk_chars // 2, 400):
                units.append(paragraph)
                continue

            sentences = re.split(
                r"(?<=[\.\?\!\:;])\s+(?=[A-ZÀ-Ỵ0-9])",
                paragraph,
            )

            if len(sentences) <= 1:
                units.append(paragraph)
            else:
                units.extend(
                    [
                        normalize_text(sentence)
                        for sentence in sentences
                        if normalize_text(sentence)
                    ]
                )

        return units

    def _hard_split_text(
        self,
        text: str,
        max_chars: int,
        overlap_chars: int = 0,
    ) -> List[str]:
        result = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + max_chars, text_length)

            if end < text_length:
                break_points = [
                    text.rfind("\n", start, end),
                    text.rfind(". ", start, end),
                    text.rfind("; ", start, end),
                    text.rfind(", ", start, end),
                    text.rfind(" ", start, end),
                ]

                break_point = max(break_points)

                if break_point > start + int(max_chars * 0.50):
                    end = break_point + 1

            piece = normalize_text(text[start:end])

            if piece:
                result.append(piece)

            if end >= text_length:
                break

            start = max(end - overlap_chars, start + 1)

        return result

    def _tail_text(
        self,
        text: str,
        max_chars: int,
    ) -> str:
        text = normalize_text(text)

        if len(text) <= max_chars:
            return text

        tail = text[-max_chars:]
        first_space = tail.find(" ")

        if first_space > 0:
            tail = tail[first_space + 1:]

        return normalize_text(tail)

    def _build_page_text_map(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[int, str]:
        page_text_map: Dict[int, str] = {}

        for page_raw in page_raws:
            text = ""

            reading_meta = page_raw.metadata.get("reading_order_builder", {}) or {}
            reading_text = normalize_text(reading_meta.get("reading_order_text", ""))

            if reading_text:
                text = reading_text
            elif page_raw.normalized_text:
                text = page_raw.normalized_text
            elif page_raw.raw_text:
                text = page_raw.raw_text
            elif page_raw.text_blocks:
                text = "\n\n".join(
                    [
                        normalize_text(block.text)
                        for block in page_raw.text_blocks
                        if normalize_text(block.text)
                    ]
                )
            elif page_raw.text_lines:
                text = "\n".join(
                    [
                        normalize_text(line.text)
                        for line in page_raw.text_lines
                        if normalize_text(line.text)
                    ]
                )

            page_text_map[page_raw.page_number] = normalize_text(text)

        return page_text_map

    def _collect_sections(
        self,
        document_structure_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        sections = document_structure_result.get("sections", []) or []

        normalized = []

        for index, section in enumerate(sections):
            if not isinstance(section, dict):
                if hasattr(section, "to_dict"):
                    section = section.to_dict()
                elif hasattr(section, "__dict__"):
                    section = vars(section)
                else:
                    continue

            item = dict(section)
            item.setdefault("order", index)
            item.setdefault("section_id", item.get("id", "") or make_id("section"))
            normalized.append(item)

        return normalized

    def _collect_paragraphs(
        self,
        document_structure_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        paragraphs = document_structure_result.get("paragraphs", []) or []

        normalized = []

        for index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, dict):
                if hasattr(paragraph, "to_dict"):
                    paragraph = paragraph.to_dict()
                elif hasattr(paragraph, "__dict__"):
                    paragraph = vars(paragraph)
                else:
                    continue

            item = dict(paragraph)
            item.setdefault("global_order", index)
            item.setdefault("paragraph_id", item.get("id", "") or make_id("paragraph"))
            normalized.append(item)

        return normalized

    def _group_sections_by_page(
        self,
        sections: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for section in sections:
            page_numbers = self._resolve_page_numbers(section)

            for page_number in page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(section)

        return grouped

    def _group_paragraphs_by_page(
        self,
        paragraphs: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for paragraph in paragraphs:
            page_numbers = self._resolve_page_numbers(paragraph)

            if not page_numbers and paragraph.get("page_number"):
                page_numbers = [self._safe_int(paragraph.get("page_number"), default=0)]

            for page_number in page_numbers:
                if page_number <= 0:
                    continue

                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(paragraph)

        return grouped

    def _group_paragraphs_by_section(
        self,
        paragraphs: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for paragraph in paragraphs:
            section_id = paragraph.get("section_id", "")

            if not section_id:
                continue

            grouped.setdefault(section_id, [])
            grouped[section_id].append(paragraph)

        return grouped

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
            result = []

            for page in page_numbers:
                page_number = self._safe_int(page, default=0)

                if page_number > 0:
                    result.append(page_number)

            return sorted(list(dict.fromkeys(result)))

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

    def _sort_chunks(
        self,
        chunks: List[Chunk],
    ) -> List[Chunk]:
        return sorted(
            chunks,
            key=lambda chunk: (
                min(chunk.page_numbers) if chunk.page_numbers else 999999,
                self._chunk_type_order(chunk.chunk_type),
                chunk.section_level if chunk.section_level is not None else 999999,
                chunk.order,
                chunk.chunk_id,
            ),
        )

    def _reorder_chunks(
        self,
        chunks: List[Chunk],
    ) -> List[Chunk]:
        for index, chunk in enumerate(chunks):
            chunk.order = index

        return chunks

    def _link_sequential_chunks(
        self,
        chunks: List[Chunk],
    ) -> List[Chunk]:
        for index, chunk in enumerate(chunks):
            if index > 0:
                chunk.previous_chunk_id = chunks[index - 1].chunk_id

            if index < len(chunks) - 1:
                chunk.next_chunk_id = chunks[index + 1].chunk_id

        return chunks

    def _deduplicate_chunks(
        self,
        chunks: List[Chunk],
    ) -> List[Chunk]:
        seen = set()
        result = []

        for chunk in chunks:
            key = (
                chunk.chunk_type,
                normalize_text_for_match(chunk.text),
                tuple(chunk.page_numbers),
                chunk.section_id,
                chunk.paragraph_id,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(chunk)

        return result

    def _chunk_type_order(
        self,
        chunk_type: str,
    ) -> int:
        order_map = {
            "section_chunk": 10,
            "paragraph_chunk": 20,
            "page_chunk": 30,
            "parent_chunk": 40,
            "table_chunk": 50,
        }

        return order_map.get(chunk_type, 99)

    def _max_chars_for_chunk(
        self,
        chunk: Chunk,
    ) -> int:
        if chunk.chunk_type == "page_chunk":
            return self.config.max_page_chunk_chars

        if chunk.chunk_type == "section_chunk":
            return self.config.max_section_chunk_chars

        if chunk.chunk_type == "paragraph_chunk":
            return self.config.max_paragraph_chunk_chars

        return self.config.max_chunk_chars

    def _valid_text(
        self,
        text: str,
    ) -> bool:
        text = normalize_text(text)

        if not text:
            return False

        if len(text) < self.config.min_chunk_chars:
            if len(re.findall(r"\w+", text)) < 8:
                return False

        return True

    def _compact_section(
        self,
        section: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "section_id": section.get("section_id", ""),
            "title": section.get("title", ""),
            "level": section.get("level", 0),
            "order": section.get("order", 0),
            "page_start": section.get("page_start"),
            "page_end": section.get("page_end"),
            "page_numbers": self._resolve_page_numbers(section),
            "parent_id": section.get("parent_id", ""),
        }

    def _compact_paragraph(
        self,
        paragraph: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "paragraph_id": paragraph.get("paragraph_id", ""),
            "section_id": paragraph.get("section_id", ""),
            "page_number": paragraph.get("page_number"),
            "page_numbers": self._resolve_page_numbers(paragraph),
            "order": paragraph.get("order", 0),
            "global_order": paragraph.get("global_order", 0),
            "paragraph_type": paragraph.get("paragraph_type", "paragraph"),
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        chunks: List[Chunk],
    ) -> None:
        chunks_by_page: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in chunks:
            for page_number in chunk.page_numbers:
                page_key = str(page_number)
                chunks_by_page.setdefault(page_key, [])
                chunks_by_page[page_key].append(chunk.to_dict())

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("chunk_builder", {})
            page_raw.metadata["chunk_builder"] = {
                "processor": "ChunkBuilder",
                "chunks_on_page": chunks_by_page.get(page_key, []),
                "chunk_count_on_page": len(chunks_by_page.get(page_key, [])),
            }

    def _build_summary(
        self,
        chunks: List[Chunk],
        page_raws: List[PageRaw],
        sections: List[Dict[str, Any]],
        paragraphs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        by_page: Dict[str, int] = {}
        by_section: Dict[str, int] = {}

        total_chars = 0
        total_words = 0

        for chunk in chunks:
            by_type[chunk.chunk_type] = by_type.get(chunk.chunk_type, 0) + 1

            total_chars += chunk.char_count
            total_words += chunk.word_count

            for page_number in chunk.page_numbers:
                page_key = str(page_number)
                by_page[page_key] = by_page.get(page_key, 0) + 1

            section_key = chunk.section_id or "no_section"
            by_section[section_key] = by_section.get(section_key, 0) + 1

        return {
            "has_chunks": len(chunks) > 0,
            "chunk_count": len(chunks),
            "page_count": len(page_raws),
            "section_count": len(sections),
            "paragraph_count": len(paragraphs),
            "total_chars": total_chars,
            "total_words": total_words,
            "avg_chars_per_chunk": round(total_chars / max(len(chunks), 1), 2),
            "avg_words_per_chunk": round(total_words / max(len(chunks), 1), 2),
            "by_chunk_type": by_type,
            "by_page": by_page,
            "by_section": by_section,
        }

    def _infer_document_id(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Dict[str, Any],
    ) -> str:
        if document_structure_result.get("document_id"):
            return document_structure_result.get("document_id", "")

        for page_raw in page_raws:
            if page_raw.document_id:
                return page_raw.document_id

        return ""

    def _infer_source_document(
        self,
        page_raws: List[PageRaw],
    ) -> str:
        for page_raw in page_raws:
            if page_raw.source_document:
                return page_raw.source_document

        return ""

    def _safe_int(
        self,
        value: Any,
        default: Optional[int] = 0,
    ) -> Optional[int]:
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


def build_chunks(
    page_raws: List[PageRaw],
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    cross_page_context_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = ChunkBuilder()
    return builder.process(
        page_raws=page_raws,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
        cross_page_context_result=cross_page_context_result,
    )
