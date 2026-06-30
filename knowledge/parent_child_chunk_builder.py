"""
parent_child_chunk_builder.py

Production V1 - Colab Ready

Purpose
-------
Build parent-child relationships between chunks for hierarchical retrieval.

Used by:
- KnowledgePipeline
- EvidenceBuilder
- KnowledgeGraphBuilder
- RAGPipeline

Input
-----
- chunk_result
- document_structure_result
- table_understanding_result

Output
------
Dictionary with:
- parent_chunks
- child_chunks
- parent_child_links
- chunk_relations
- chunk_hierarchy
- parent_child_summary
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.chunk_schema import (
    Chunk,
    ParentChildChunk,
    ChunkRelation,
    ChunkCollection,
    make_id,
    normalize_text,
    normalize_text_for_match,
)


@dataclass
class ParentChildChunkBuilderConfig:
    build_section_parent_chunks: bool = True
    build_page_parent_chunks: bool = True
    build_table_parent_chunks: bool = True
    build_existing_parent_links: bool = True

    attach_parent_to_children: bool = True
    attach_children_to_parent: bool = True
    create_chunk_relations: bool = True
    deduplicate_chunks: bool = True

    min_children_per_parent: int = 1
    max_parent_text_chars: int = 2500
    max_child_preview_chars: int = 400

    include_parent_summary_text: bool = True
    include_child_preview_text: bool = True
    include_debug: bool = True


class ParentChildChunkBuilder:
    def __init__(
        self,
        config: Optional[ParentChildChunkBuilderConfig] = None,
    ):
        self.config = config or ParentChildChunkBuilderConfig()

    def process(
        self,
        chunk_result: Dict[str, Any],
        document_structure_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        chunk_result = chunk_result or {}
        document_structure_result = document_structure_result or {}
        table_understanding_result = table_understanding_result or {}

        chunks = self._collect_chunks(chunk_result)

        if self.config.deduplicate_chunks:
            chunks = self._deduplicate_chunks(chunks)

        chunk_by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
            if chunk.chunk_id
        }

        parent_chunks: List[Chunk] = []
        child_chunks: List[Chunk] = []
        parent_child_links: List[ParentChildChunk] = []
        chunk_relations: List[ChunkRelation] = []

        if self.config.build_existing_parent_links:
            existing_parents, existing_children, existing_links, existing_relations = self._build_from_existing_links(
                chunks=chunks,
                chunk_by_id=chunk_by_id,
            )
            parent_chunks.extend(existing_parents)
            child_chunks.extend(existing_children)
            parent_child_links.extend(existing_links)
            chunk_relations.extend(existing_relations)

        if self.config.build_section_parent_chunks:
            section_parents, section_children, section_links, section_relations = self._build_section_parent_chunks(
                chunks=chunks,
                document_structure_result=document_structure_result,
            )
            parent_chunks.extend(section_parents)
            child_chunks.extend(section_children)
            parent_child_links.extend(section_links)
            chunk_relations.extend(section_relations)

        if self.config.build_page_parent_chunks:
            page_parents, page_children, page_links, page_relations = self._build_page_parent_chunks(
                chunks=chunks,
            )
            parent_chunks.extend(page_parents)
            child_chunks.extend(page_children)
            parent_child_links.extend(page_links)
            chunk_relations.extend(page_relations)

        if self.config.build_table_parent_chunks:
            table_parents, table_children, table_links, table_relations = self._build_table_parent_chunks(
                chunks=chunks,
                table_understanding_result=table_understanding_result,
            )
            parent_chunks.extend(table_parents)
            child_chunks.extend(table_children)
            parent_child_links.extend(table_links)
            chunk_relations.extend(table_relations)

        parent_chunks = self._deduplicate_chunks(parent_chunks)
        child_chunks = self._deduplicate_chunks(child_chunks)
        parent_child_links = self._deduplicate_parent_child_links(parent_child_links)
        chunk_relations = self._deduplicate_chunk_relations(chunk_relations)

        all_chunks = self._merge_all_chunks(
            original_chunks=chunks,
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
        )

        self._apply_parent_child_links_to_chunks(
            chunks=all_chunks,
            links=parent_child_links,
        )

        hierarchy = self._build_hierarchy(
            chunks=all_chunks,
            links=parent_child_links,
            relations=chunk_relations,
        )

        collection = ChunkCollection(
            document_id=chunk_result.get("document_id", self._infer_document_id(all_chunks)),
            source_document=chunk_result.get("source_document", self._infer_source_document(all_chunks)),
            chunks=all_chunks,
            parent_child_chunks=parent_child_links,
            chunk_relations=chunk_relations,
            metadata={
                "processor": "ParentChildChunkBuilder",
                "schema_version": "parent_child_chunk_builder_v1",
                "document_structure_available": bool(document_structure_result),
                "table_understanding_available": bool(table_understanding_result),
            },
        )

        result = collection.to_dict()
        result.update(
            {
                "processor": "ParentChildChunkBuilder",
                "schema_version": "parent_child_chunk_builder_v1",
                "parent_chunks": [chunk.to_dict() for chunk in parent_chunks],
                "child_chunks": [chunk.to_dict() for chunk in child_chunks],
                "parent_child_links": [link.to_dict() for link in parent_child_links],
                "chunk_relations": [relation.to_dict() for relation in chunk_relations],
                "chunk_hierarchy": hierarchy,
                "parent_child_summary": self._build_summary(
                    all_chunks=all_chunks,
                    parent_chunks=parent_chunks,
                    child_chunks=child_chunks,
                    parent_child_links=parent_child_links,
                    chunk_relations=chunk_relations,
                ),
                "config": asdict(self.config),
            }
        )

        return result

    def _build_from_existing_links(
        self,
        chunks: List[Chunk],
        chunk_by_id: Dict[str, Chunk],
    ) -> Tuple[List[Chunk], List[Chunk], List[ParentChildChunk], List[ChunkRelation]]:
        parent_chunks = []
        child_chunks = []
        links = []
        relations = []

        for chunk in chunks:
            parent_id = getattr(chunk, "parent_chunk_id", "") or ""

            if parent_id and parent_id in chunk_by_id:
                parent = chunk_by_id[parent_id]

                parent_chunks.append(parent)
                child_chunks.append(chunk)

                links.append(
                    self._make_parent_child_link(
                        parent_chunk=parent,
                        child_chunks=[chunk],
                        relation_type="existing_parent_child",
                        source="parent_child_chunk_builder_existing_parent",
                    )
                )

                relations.append(
                    self._make_chunk_relation(
                        source_chunk_id=parent.chunk_id,
                        target_chunk_id=chunk.chunk_id,
                        relation_type="parent_of",
                        source="parent_child_chunk_builder_existing_parent",
                        confidence=0.90,
                    )
                )

            child_ids = getattr(chunk, "child_chunk_ids", []) or []

            for child_id in child_ids:
                if child_id not in chunk_by_id:
                    continue

                child = chunk_by_id[child_id]

                parent_chunks.append(chunk)
                child_chunks.append(child)

                links.append(
                    self._make_parent_child_link(
                        parent_chunk=chunk,
                        child_chunks=[child],
                        relation_type="existing_child_reference",
                        source="parent_child_chunk_builder_existing_child",
                    )
                )

                relations.append(
                    self._make_chunk_relation(
                        source_chunk_id=chunk.chunk_id,
                        target_chunk_id=child.chunk_id,
                        relation_type="parent_of",
                        source="parent_child_chunk_builder_existing_child",
                        confidence=0.90,
                    )
                )

        return parent_chunks, child_chunks, links, relations

    def _build_section_parent_chunks(
        self,
        chunks: List[Chunk],
        document_structure_result: Dict[str, Any],
    ) -> Tuple[List[Chunk], List[Chunk], List[ParentChildChunk], List[ChunkRelation]]:
        parent_chunks = []
        child_chunks = []
        links = []
        relations = []

        grouped = self._group_chunks_by_section(chunks)
        sections_by_id = self._collect_sections_by_id(document_structure_result)

        for section_id, section_chunks in grouped.items():
            if not section_id or section_id == "no_section":
                continue

            section_chunks = self._sort_chunks(section_chunks)

            if len(section_chunks) < self.config.min_children_per_parent:
                continue

            section = sections_by_id.get(section_id, {})
            section_title = (
                section.get("title")
                or section_chunks[0].section_title
                or section_id
            )

            page_numbers = self._merge_chunk_pages(section_chunks)
            parent_text = self._build_parent_text(
                title=section_title,
                child_chunks=section_chunks,
                parent_type="section_parent_chunk",
            )

            parent_chunk = Chunk(
                chunk_id=make_id("section_parent_chunk"),
                chunk_type="section_parent_chunk",
                text=parent_text,
                document_id=self._infer_document_id(section_chunks),
                source_document=self._infer_source_document(section_chunks),
                page_numbers=page_numbers,
                page_start=min(page_numbers) if page_numbers else None,
                page_end=max(page_numbers) if page_numbers else None,
                section_id=section_id,
                section_title=normalize_text(section_title),
                section_level=self._safe_int(
                    section.get("level"),
                    default=section_chunks[0].section_level if section_chunks else None,
                ),
                order=min([chunk.order for chunk in section_chunks] or [0]),
                child_chunk_ids=[chunk.chunk_id for chunk in section_chunks],
                source="parent_child_chunk_builder_section",
                confidence=0.78,
                metadata={
                    "parent_type": "section",
                    "section_id": section_id,
                    "section_title": section_title,
                    "child_count": len(section_chunks),
                    "child_chunk_types": self._count_by_attr(section_chunks, "chunk_type"),
                    "source_section": self._compact_section(section),
                },
            )

            parent_chunks.append(parent_chunk)
            child_chunks.extend(section_chunks)

            links.append(
                self._make_parent_child_link(
                    parent_chunk=parent_chunk,
                    child_chunks=section_chunks,
                    relation_type="section_parent_of_chunks",
                    source="parent_child_chunk_builder_section",
                )
            )

            for child in section_chunks:
                relations.append(
                    self._make_chunk_relation(
                        source_chunk_id=parent_chunk.chunk_id,
                        target_chunk_id=child.chunk_id,
                        relation_type="parent_of",
                        source="parent_child_chunk_builder_section",
                        confidence=0.82,
                        metadata={
                            "parent_type": "section",
                            "section_id": section_id,
                        },
                    )
                )

        return parent_chunks, child_chunks, links, relations

    def _build_page_parent_chunks(
        self,
        chunks: List[Chunk],
    ) -> Tuple[List[Chunk], List[Chunk], List[ParentChildChunk], List[ChunkRelation]]:
        parent_chunks = []
        child_chunks = []
        links = []
        relations = []

        grouped = self._group_chunks_by_page(chunks)

        for page_key, page_chunks in grouped.items():
            page_number = self._safe_int(page_key, default=0)

            if page_number <= 0:
                continue

            page_chunks = self._sort_chunks(page_chunks)

            if len(page_chunks) < self.config.min_children_per_parent:
                continue

            title = f"Trang {page_number}"
            parent_text = self._build_parent_text(
                title=title,
                child_chunks=page_chunks,
                parent_type="page_parent_chunk",
            )

            page_numbers = [page_number]

            parent_chunk = Chunk(
                chunk_id=make_id("page_parent_chunk"),
                chunk_type="page_parent_chunk",
                text=parent_text,
                document_id=self._infer_document_id(page_chunks),
                source_document=self._infer_source_document(page_chunks),
                page_numbers=page_numbers,
                page_start=page_number,
                page_end=page_number,
                section_id=self._dominant_section_id(page_chunks),
                section_title=self._dominant_section_title(page_chunks),
                section_level=self._dominant_section_level(page_chunks),
                order=min([chunk.order for chunk in page_chunks] or [0]),
                child_chunk_ids=[chunk.chunk_id for chunk in page_chunks],
                source="parent_child_chunk_builder_page",
                confidence=0.72,
                metadata={
                    "parent_type": "page",
                    "page_number": page_number,
                    "child_count": len(page_chunks),
                    "child_chunk_types": self._count_by_attr(page_chunks, "chunk_type"),
                },
            )

            parent_chunks.append(parent_chunk)
            child_chunks.extend(page_chunks)

            links.append(
                self._make_parent_child_link(
                    parent_chunk=parent_chunk,
                    child_chunks=page_chunks,
                    relation_type="page_parent_of_chunks",
                    source="parent_child_chunk_builder_page",
                )
            )

            for child in page_chunks:
                relations.append(
                    self._make_chunk_relation(
                        source_chunk_id=parent_chunk.chunk_id,
                        target_chunk_id=child.chunk_id,
                        relation_type="parent_of",
                        source="parent_child_chunk_builder_page",
                        confidence=0.76,
                        metadata={
                            "parent_type": "page",
                            "page_number": page_number,
                        },
                    )
                )

        return parent_chunks, child_chunks, links, relations

    def _build_table_parent_chunks(
        self,
        chunks: List[Chunk],
        table_understanding_result: Dict[str, Any],
    ) -> Tuple[List[Chunk], List[Chunk], List[ParentChildChunk], List[ChunkRelation]]:
        parent_chunks = []
        child_chunks = []
        links = []
        relations = []

        grouped = self._group_chunks_by_table(chunks)
        tables_by_id = self._collect_tables_by_id(table_understanding_result)

        for table_id, table_chunks in grouped.items():
            if not table_id:
                continue

            table_chunks = self._sort_chunks(table_chunks)

            if len(table_chunks) < self.config.min_children_per_parent:
                continue

            table = tables_by_id.get(table_id, {})
            title = (
                table.get("title")
                or table.get("caption")
                or table.get("caption_text")
                or f"Bảng {table_id}"
            )

            page_numbers = self._merge_chunk_pages(table_chunks)
            parent_text = self._build_parent_text(
                title=title,
                child_chunks=table_chunks,
                parent_type="table_parent_chunk",
            )

            parent_chunk = Chunk(
                chunk_id=make_id("table_parent_chunk"),
                chunk_type="table_parent_chunk",
                text=parent_text,
                document_id=self._infer_document_id(table_chunks),
                source_document=self._infer_source_document(table_chunks),
                page_numbers=page_numbers,
                page_start=min(page_numbers) if page_numbers else None,
                page_end=max(page_numbers) if page_numbers else None,
                section_id=self._dominant_section_id(table_chunks),
                section_title=self._dominant_section_title(table_chunks),
                section_level=self._dominant_section_level(table_chunks),
                table_grid_id=table.get("table_grid_id", ""),
                table_structure_id=table.get("table_structure_id", ""),
                table_semantic_id=table.get("table_semantic_id", ""),
                table_boundary_id=table.get("table_boundary_id", ""),
                order=min([chunk.order for chunk in table_chunks] or [0]),
                child_chunk_ids=[chunk.chunk_id for chunk in table_chunks],
                source="parent_child_chunk_builder_table",
                confidence=0.78,
                metadata={
                    "parent_type": "table",
                    "table_id": table_id,
                    "table_title": title,
                    "child_count": len(table_chunks),
                    "semantic_type": table.get("semantic_type", ""),
                    "row_count": table.get("row_count", table.get("total_row_count", 0)),
                    "col_count": table.get("col_count", 0),
                    "column_headers": table.get("column_headers", []),
                },
            )

            parent_chunks.append(parent_chunk)
            child_chunks.extend(table_chunks)

            links.append(
                self._make_parent_child_link(
                    parent_chunk=parent_chunk,
                    child_chunks=table_chunks,
                    relation_type="table_parent_of_chunks",
                    source="parent_child_chunk_builder_table",
                )
            )

            for child in table_chunks:
                relations.append(
                    self._make_chunk_relation(
                        source_chunk_id=parent_chunk.chunk_id,
                        target_chunk_id=child.chunk_id,
                        relation_type="parent_of",
                        source="parent_child_chunk_builder_table",
                        confidence=0.82,
                        metadata={
                            "parent_type": "table",
                            "table_id": table_id,
                        },
                    )
                )

        return parent_chunks, child_chunks, links, relations

    def _make_parent_child_link(
        self,
        parent_chunk: Chunk,
        child_chunks: List[Chunk],
        relation_type: str = "parent_of",
        source: str = "parent_child_chunk_builder",
    ) -> ParentChildChunk:
        child_chunk_ids = [
            child.chunk_id
            for child in child_chunks
            if child.chunk_id
        ]

        page_numbers = sorted(
            list(
                dict.fromkeys(
                    parent_chunk.page_numbers
                    + [
                        page
                        for child in child_chunks
                        for page in child.page_numbers
                    ]
                )
            )
        )

        return ParentChildChunk(
            parent_chunk_id=parent_chunk.chunk_id,
            child_chunk_ids=child_chunk_ids,
            relation_type=relation_type,
            document_id=parent_chunk.document_id,
            page_numbers=page_numbers,
            section_id=parent_chunk.section_id,
            table_grid_id=parent_chunk.table_grid_id,
            table_structure_id=parent_chunk.table_structure_id,
            table_semantic_id=parent_chunk.table_semantic_id,
            table_boundary_id=parent_chunk.table_boundary_id,
            confidence=parent_chunk.confidence,
            source=source,
            metadata={
                "parent_chunk_type": parent_chunk.chunk_type,
                "child_count": len(child_chunk_ids),
                "child_chunk_types": self._count_by_attr(child_chunks, "chunk_type"),
            },
        )

    def _make_chunk_relation(
        self,
        source_chunk_id: str,
        target_chunk_id: str,
        relation_type: str = "related_to",
        source: str = "parent_child_chunk_builder",
        confidence: float = 0.70,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChunkRelation:
        return ChunkRelation(
            source_chunk_id=source_chunk_id,
            target_chunk_id=target_chunk_id,
            relation_type=relation_type,
            confidence=confidence,
            source=source,
            metadata=metadata or {},
        )

    def _build_parent_text(
        self,
        title: str,
        child_chunks: List[Chunk],
        parent_type: str,
    ) -> str:
        title = normalize_text(title)

        parts = []

        if title:
            parts.append(title)

        if self.config.include_parent_summary_text:
            summary_line = self._make_summary_line(
                parent_type=parent_type,
                child_chunks=child_chunks,
            )

            if summary_line:
                parts.append(summary_line)

        if self.config.include_child_preview_text:
            for index, child in enumerate(child_chunks):
                preview = self._preview(child.text, self.config.max_child_preview_chars)

                if preview:
                    child_label = child.section_title or child.chunk_type or f"chunk {index + 1}"
                    parts.append(f"- {child_label}: {preview}")

        text = normalize_text("\n".join(parts))

        if len(text) <= self.config.max_parent_text_chars:
            return text

        cut = text[: self.config.max_parent_text_chars]
        break_point = max(
            cut.rfind("\n"),
            cut.rfind(". "),
            cut.rfind("; "),
            cut.rfind(" "),
        )

        if break_point > self.config.max_parent_text_chars * 0.60:
            cut = cut[:break_point]

        return normalize_text(cut) + "..."

    def _make_summary_line(
        self,
        parent_type: str,
        child_chunks: List[Chunk],
    ) -> str:
        page_numbers = self._merge_chunk_pages(child_chunks)
        child_types = self._count_by_attr(child_chunks, "chunk_type")
        total_words = sum(child.word_count for child in child_chunks)
        total_chars = sum(child.char_count for child in child_chunks)

        page_text = ""

        if page_numbers:
            if len(page_numbers) == 1:
                page_text = f"trang {page_numbers[0]}"
            else:
                page_text = f"trang {page_numbers[0]}-{page_numbers[-1]}"

        type_text = ", ".join(
            [
                f"{key}: {value}"
                for key, value in child_types.items()
            ]
        )

        parts = [
            f"Nhóm {len(child_chunks)} chunk con",
        ]

        if page_text:
            parts.append(page_text)

        if type_text:
            parts.append(type_text)

        parts.append(f"{total_words} từ")
        parts.append(f"{total_chars} ký tự")

        return "; ".join(parts)

    def _apply_parent_child_links_to_chunks(
        self,
        chunks: List[Chunk],
        links: List[ParentChildChunk],
    ) -> None:
        chunk_by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
            if chunk.chunk_id
        }

        for link in links:
            parent = chunk_by_id.get(link.parent_chunk_id)

            if parent and self.config.attach_children_to_parent:
                existing = list(parent.child_chunk_ids or [])

                for child_id in link.child_chunk_ids:
                    if child_id not in existing:
                        existing.append(child_id)

                parent.child_chunk_ids = existing

            if self.config.attach_parent_to_children:
                for child_id in link.child_chunk_ids:
                    child = chunk_by_id.get(child_id)

                    if child and not child.parent_chunk_id:
                        child.parent_chunk_id = link.parent_chunk_id

    def _build_hierarchy(
        self,
        chunks: List[Chunk],
        links: List[ParentChildChunk],
        relations: List[ChunkRelation],
    ) -> Dict[str, Any]:
        chunk_by_id = {
            chunk.chunk_id: chunk
            for chunk in chunks
            if chunk.chunk_id
        }

        children_by_parent: Dict[str, List[str]] = {}
        parent_by_child: Dict[str, str] = {}

        for link in links:
            children_by_parent.setdefault(link.parent_chunk_id, [])

            for child_id in link.child_chunk_ids:
                if child_id not in children_by_parent[link.parent_chunk_id]:
                    children_by_parent[link.parent_chunk_id].append(child_id)

                parent_by_child[child_id] = link.parent_chunk_id

        roots = [
            chunk_id
            for chunk_id in chunk_by_id
            if chunk_id not in parent_by_child
        ]

        tree_nodes = []

        for root_id in roots:
            tree_nodes.append(
                self._build_tree_node(
                    chunk_id=root_id,
                    chunk_by_id=chunk_by_id,
                    children_by_parent=children_by_parent,
                    depth=0,
                    visited=set(),
                )
            )

        return {
            "root_chunk_ids": roots,
            "parent_chunk_ids": list(children_by_parent.keys()),
            "child_chunk_ids": list(parent_by_child.keys()),
            "children_by_parent": children_by_parent,
            "parent_by_child": parent_by_child,
            "tree": tree_nodes,
            "relation_count": len(relations),
        }

    def _build_tree_node(
        self,
        chunk_id: str,
        chunk_by_id: Dict[str, Chunk],
        children_by_parent: Dict[str, List[str]],
        depth: int = 0,
        visited: Optional[set] = None,
    ) -> Dict[str, Any]:
        visited = visited or set()

        if chunk_id in visited:
            return {
                "chunk_id": chunk_id,
                "cycle_detected": True,
            }

        visited.add(chunk_id)

        chunk = chunk_by_id.get(chunk_id)

        if not chunk:
            return {
                "chunk_id": chunk_id,
                "missing": True,
            }

        child_ids = children_by_parent.get(chunk_id, [])

        return {
            "chunk_id": chunk.chunk_id,
            "chunk_type": chunk.chunk_type,
            "label": self._preview(chunk.text, 120),
            "section_id": chunk.section_id,
            "section_title": chunk.section_title,
            "page_numbers": chunk.page_numbers,
            "depth": depth,
            "child_count": len(child_ids),
            "children": [
                self._build_tree_node(
                    chunk_id=child_id,
                    chunk_by_id=chunk_by_id,
                    children_by_parent=children_by_parent,
                    depth=depth + 1,
                    visited=set(visited),
                )
                for child_id in child_ids
            ],
        }

    def _merge_all_chunks(
        self,
        original_chunks: List[Chunk],
        parent_chunks: List[Chunk],
        child_chunks: List[Chunk],
    ) -> List[Chunk]:
        chunk_by_id: Dict[str, Chunk] = {}

        for chunk in original_chunks + child_chunks + parent_chunks:
            if not chunk.chunk_id:
                continue

            if chunk.chunk_id not in chunk_by_id:
                chunk_by_id[chunk.chunk_id] = chunk

        return self._sort_chunks(list(chunk_by_id.values()))

    def _collect_chunks(
        self,
        chunk_result: Dict[str, Any],
    ) -> List[Chunk]:
        candidates = []

        for key in [
            "chunks",
            "parent_chunks",
            "child_chunks",
            "table_chunks",
        ]:
            values = chunk_result.get(key, []) or []

            if isinstance(values, list):
                candidates.extend(values)

        collection = chunk_result.get("chunk_collection", {}) or {}

        if isinstance(collection, dict):
            for key in ["chunks", "parent_chunks", "child_chunks", "table_chunks"]:
                values = collection.get(key, []) or []

                if isinstance(values, list):
                    candidates.extend(values)

        chunks = []

        for item in candidates:
            try:
                if isinstance(item, Chunk):
                    chunks.append(item)
                elif isinstance(item, dict):
                    chunks.append(Chunk.from_dict(item))
            except Exception:
                continue

        return chunks

    def _collect_sections_by_id(
        self,
        document_structure_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        sections = document_structure_result.get("sections", []) or []
        result = {}

        for item in sections:
            item = self._to_dict(item)
            section_id = item.get("section_id") or item.get("id") or ""

            if section_id:
                result[section_id] = item

        return result

    def _collect_tables_by_id(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        tables = []

        for key in [
            "table_semantics",
            "table_grids",
            "multi_page_tables",
            "table_structures",
            "table_boundaries",
        ]:
            values = table_understanding_result.get(key, []) or []

            if isinstance(values, list):
                tables.extend(values)

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
                "multi_page_tables",
                "table_structures",
                "table_boundaries",
            ]:
                values = sub.get(key, []) or []

                if isinstance(values, list):
                    tables.extend(values)

        result = {}

        for item in tables:
            item = self._to_dict(item)
            table_ids = [
                item.get("table_semantic_id", ""),
                item.get("table_grid_id", ""),
                item.get("table_structure_id", ""),
                item.get("table_boundary_id", ""),
                item.get("multi_page_table_id", ""),
            ]

            for table_id in table_ids:
                if table_id:
                    result[table_id] = item

        return result

    def _group_chunks_by_section(
        self,
        chunks: List[Chunk],
    ) -> Dict[str, List[Chunk]]:
        grouped: Dict[str, List[Chunk]] = {}

        for chunk in chunks:
            if chunk.chunk_type in [
                "section_parent_chunk",
                "page_parent_chunk",
                "table_parent_chunk",
                "parent_chunk",
            ]:
                continue

            section_id = chunk.section_id or "no_section"

            grouped.setdefault(section_id, [])
            grouped[section_id].append(chunk)

        return grouped

    def _group_chunks_by_page(
        self,
        chunks: List[Chunk],
    ) -> Dict[str, List[Chunk]]:
        grouped: Dict[str, List[Chunk]] = {}

        for chunk in chunks:
            if chunk.chunk_type in [
                "section_parent_chunk",
                "page_parent_chunk",
                "table_parent_chunk",
                "parent_chunk",
            ]:
                continue

            for page_number in chunk.page_numbers or []:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(chunk)

        return grouped

    def _group_chunks_by_table(
        self,
        chunks: List[Chunk],
    ) -> Dict[str, List[Chunk]]:
        grouped: Dict[str, List[Chunk]] = {}

        for chunk in chunks:
            if chunk.chunk_type in [
                "section_parent_chunk",
                "page_parent_chunk",
                "table_parent_chunk",
                "parent_chunk",
            ]:
                continue

            table_ids = [
                chunk.table_semantic_id,
                chunk.table_grid_id,
                chunk.table_structure_id,
                chunk.table_boundary_id,
            ]

            for table_id in table_ids:
                if not table_id:
                    continue

                grouped.setdefault(table_id, [])
                grouped[table_id].append(chunk)

        return grouped

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

    def _chunk_type_order(
        self,
        chunk_type: str,
    ) -> int:
        order_map = {
            "section_parent_chunk": 5,
            "page_parent_chunk": 6,
            "table_parent_chunk": 7,
            "parent_chunk": 8,
            "section_chunk": 10,
            "paragraph_chunk": 20,
            "page_chunk": 30,
            "table_chunk": 40,
        }

        return order_map.get(chunk_type, 99)

    def _deduplicate_chunks(
        self,
        chunks: List[Chunk],
    ) -> List[Chunk]:
        seen = set()
        result = []

        for chunk in chunks:
            key = chunk.chunk_id or (
                chunk.chunk_type,
                normalize_text_for_match(chunk.text)[:500],
                tuple(chunk.page_numbers),
                chunk.section_id,
                chunk.paragraph_id,
            )

            if str(key) in seen:
                continue

            seen.add(str(key))
            result.append(chunk)

        return result

    def _deduplicate_parent_child_links(
        self,
        links: List[ParentChildChunk],
    ) -> List[ParentChildChunk]:
        seen = set()
        result = []

        for link in links:
            key = (
                link.parent_chunk_id,
                tuple(sorted(link.child_chunk_ids)),
                link.relation_type,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(link)

        return result

    def _deduplicate_chunk_relations(
        self,
        relations: List[ChunkRelation],
    ) -> List[ChunkRelation]:
        seen = set()
        result = []

        for relation in relations:
            key = (
                relation.source_chunk_id,
                relation.target_chunk_id,
                relation.relation_type,
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(relation)

        return result

    def _merge_chunk_pages(
        self,
        chunks: List[Chunk],
    ) -> List[int]:
        pages = []

        for chunk in chunks:
            for page in chunk.page_numbers or []:
                try:
                    page = int(page)
                    if page > 0 and page not in pages:
                        pages.append(page)
                except Exception:
                    continue

        return sorted(pages)

    def _dominant_section_id(
        self,
        chunks: List[Chunk],
    ) -> str:
        return self._dominant_attr(chunks, "section_id") or ""

    def _dominant_section_title(
        self,
        chunks: List[Chunk],
    ) -> str:
        return self._dominant_attr(chunks, "section_title") or ""

    def _dominant_section_level(
        self,
        chunks: List[Chunk],
    ) -> Optional[int]:
        levels = [
            chunk.section_level
            for chunk in chunks
            if chunk.section_level is not None
        ]

        if not levels:
            return None

        return max(set(levels), key=levels.count)

    def _dominant_attr(
        self,
        chunks: List[Chunk],
        attr_name: str,
    ) -> str:
        values = [
            str(getattr(chunk, attr_name, "") or "")
            for chunk in chunks
            if getattr(chunk, attr_name, "")
        ]

        if not values:
            return ""

        return max(set(values), key=values.count)

    def _count_by_attr(
        self,
        chunks: List[Chunk],
        attr_name: str,
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}

        for chunk in chunks:
            value = str(getattr(chunk, attr_name, "") or "unknown")
            counts[value] = counts.get(value, 0) + 1

        return counts

    def _compact_section(
        self,
        section: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "section_id": section.get("section_id", "") or section.get("id", ""),
            "title": section.get("title", ""),
            "level": section.get("level", 0),
            "order": section.get("order", 0),
            "page_start": section.get("page_start"),
            "page_end": section.get("page_end"),
            "page_numbers": section.get("page_numbers", []),
            "parent_id": section.get("parent_id", ""),
        }

    def _build_summary(
        self,
        all_chunks: List[Chunk],
        parent_chunks: List[Chunk],
        child_chunks: List[Chunk],
        parent_child_links: List[ParentChildChunk],
        chunk_relations: List[ChunkRelation],
    ) -> Dict[str, Any]:
        by_parent_type: Dict[str, int] = {}
        by_child_type: Dict[str, int] = {}
        by_relation_type: Dict[str, int] = {}

        for chunk in parent_chunks:
            by_parent_type[chunk.chunk_type] = by_parent_type.get(chunk.chunk_type, 0) + 1

        for chunk in child_chunks:
            by_child_type[chunk.chunk_type] = by_child_type.get(chunk.chunk_type, 0) + 1

        for relation in chunk_relations:
            by_relation_type[relation.relation_type] = by_relation_type.get(relation.relation_type, 0) + 1

        total_child_refs = sum(len(link.child_chunk_ids) for link in parent_child_links)

        return {
            "has_parent_child_chunks": len(parent_child_links) > 0,
            "total_chunk_count": len(all_chunks),
            "parent_chunk_count": len(parent_chunks),
            "child_chunk_count": len(child_chunks),
            "parent_child_link_count": len(parent_child_links),
            "chunk_relation_count": len(chunk_relations),
            "total_child_references": total_child_refs,
            "avg_children_per_parent": round(total_child_refs / max(len(parent_child_links), 1), 2),
            "by_parent_chunk_type": by_parent_type,
            "by_child_chunk_type": by_child_type,
            "by_relation_type": by_relation_type,
        }

    def _infer_document_id(
        self,
        chunks: List[Chunk],
    ) -> str:
        for chunk in chunks:
            if chunk.document_id:
                return chunk.document_id

        return ""

    def _infer_source_document(
        self,
        chunks: List[Chunk],
    ) -> str:
        for chunk in chunks:
            if chunk.source_document:
                return chunk.source_document

        return ""

    def _preview(
        self,
        text: Any,
        max_chars: int = 300,
    ) -> str:
        text = normalize_text(text)

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


def build_parent_child_chunks(
    chunk_result: Dict[str, Any],
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = ParentChildChunkBuilder()
    return builder.process(
        chunk_result=chunk_result,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
    )
