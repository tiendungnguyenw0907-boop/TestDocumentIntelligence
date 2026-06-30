"""
document_tree_builder.py

Production V1 - Colab Ready

Purpose
-------
Build final document structure tree from title, TOC, headings, sections,
paragraphs, and lists.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- TitleDetector
- TOCDetector
- HeadingDetector
- SectionBuilder
- ParagraphBuilder
- ListDetector

Output
------
Dictionary with:
- document_tree
- nodes
- root_node
- pages
- sections
- paragraphs
- lists
- document_summary
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class DocumentTreeBuilderConfig:
    attach_to_pages: bool = True

    include_paragraphs: bool = True
    include_lists: bool = True
    include_toc: bool = True
    include_headings: bool = True
    include_page_summaries: bool = True
    include_table_summary: bool = True
    include_figure_summary: bool = True
    include_caption_summary: bool = True

    include_text_preview: bool = True
    text_preview_chars: int = 1200

    include_empty_sections: bool = True

    root_title_fallback: str = "Document"

    save_json: bool = False
    output_dir: str = "outputs/document_structure"


@dataclass
class DocumentTreeNode:
    node_id: str
    node_type: str
    title: str
    level: int
    order: int

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    parent_id: str = ""
    section_id: str = ""
    heading_id: str = ""
    section_number: str = ""

    children: Optional[List[str]] = None
    paragraph_ids: Optional[List[str]] = None
    list_ids: Optional[List[str]] = None

    text_preview: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["children"] is None:
            data["children"] = []

        if data["paragraph_ids"] is None:
            data["paragraph_ids"] = []

        if data["list_ids"] is None:
            data["list_ids"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class DocumentTreeBuilder:
    def __init__(
        self,
        config: Optional[DocumentTreeBuilderConfig] = None,
    ):
        self.config = config or DocumentTreeBuilderConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        title_result: Optional[Dict[str, Any]] = None,
        toc_result: Optional[Dict[str, Any]] = None,
        heading_result: Optional[Dict[str, Any]] = None,
        section_result: Optional[Dict[str, Any]] = None,
        paragraph_result: Optional[Dict[str, Any]] = None,
        list_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        title_result = title_result or self._collect_title_result(page_raws)
        toc_result = toc_result or self._collect_toc_result(page_raws)
        heading_result = heading_result or self._collect_heading_result(page_raws)
        section_result = section_result or self._collect_section_result(page_raws)
        paragraph_result = paragraph_result or self._collect_paragraph_result(page_raws)
        list_result = list_result or self._collect_list_result(page_raws)

        sections = self._collect_sections(section_result)
        paragraphs = self._collect_paragraphs(paragraph_result)
        lists = self._collect_lists(list_result)

        root_title = self._resolve_root_title(
            title_result=title_result,
            section_result=section_result,
        )

        nodes = self._build_nodes_from_sections(
            sections=sections,
            root_title=root_title,
            page_raws=page_raws,
        )

        nodes = self._attach_paragraphs_to_nodes(
            nodes=nodes,
            paragraphs=paragraphs,
        )

        nodes = self._attach_lists_to_nodes(
            nodes=nodes,
            lists=lists,
        )

        nodes = self._attach_text_preview_to_nodes(
            nodes=nodes,
            paragraphs=paragraphs,
        )

        document_tree = self._build_tree(nodes)
        page_summaries = self._build_page_summaries(page_raws)

        result = {
            "schema_version": "document_tree_v1",
            "processor": "DocumentTreeBuilder",
            "document_tree": document_tree,
            "root_node": document_tree,
            "nodes": [
                node.to_dict() for node in nodes
            ],
            "pages": page_summaries,
            "sections": sections,
            "paragraphs": paragraphs if self.config.include_paragraphs else [],
            "lists": lists if self.config.include_lists else [],
            "toc": toc_result if self.config.include_toc else {},
            "headings": heading_result if self.config.include_headings else {},
            "document_summary": self._build_document_summary(
                page_raws=page_raws,
                nodes=nodes,
                sections=sections,
                paragraphs=paragraphs,
                lists=lists,
                toc_result=toc_result,
                heading_result=heading_result,
                title_result=title_result,
            ),
            "config": {
                "include_paragraphs": self.config.include_paragraphs,
                "include_lists": self.config.include_lists,
                "include_toc": self.config.include_toc,
                "include_headings": self.config.include_headings,
                "include_page_summaries": self.config.include_page_summaries,
                "include_table_summary": self.config.include_table_summary,
                "include_figure_summary": self.config.include_figure_summary,
                "include_caption_summary": self.config.include_caption_summary,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                result=result,
            )

        if self.config.save_json:
            self.save_document_tree(
                result=result,
                page_raws=page_raws,
            )

        return result

    def _build_nodes_from_sections(
        self,
        sections: List[Dict[str, Any]],
        root_title: str,
        page_raws: List[PageRaw],
    ) -> List[DocumentTreeNode]:
        nodes: List[DocumentTreeNode] = []

        page_numbers = [
            page_raw.page_number for page_raw in page_raws
        ]

        first_page = min(page_numbers) if page_numbers else None
        last_page = max(page_numbers) if page_numbers else None

        if not sections:
            root = DocumentTreeNode(
                node_id=make_id("node"),
                node_type="root",
                title=root_title,
                level=0,
                order=0,
                page_start=first_page,
                page_end=last_page,
                parent_id="",
                section_id="",
                heading_id="",
                section_number="",
                children=[],
                paragraph_ids=[],
                list_ids=[],
                metadata={
                    "created_from": "fallback_root",
                },
            )

            return [root]

        section_id_to_node_id: Dict[str, str] = {}

        for index, section in enumerate(sections):
            section_id = section.get("section_id", "") or make_id("section_ref")
            node_id = make_id("node")

            section_id_to_node_id[section_id] = node_id

            section_type = section.get("section_type", "section")
            node_type = "root" if section_type == "root" else "section"

            node = DocumentTreeNode(
                node_id=node_id,
                node_type=node_type,
                title=section.get("title", "") or root_title,
                level=self._safe_int(section.get("level", 0), default=0),
                order=self._safe_int(section.get("order", index), default=index),
                page_start=section.get("page_start"),
                page_end=section.get("page_end"),
                parent_id="",
                section_id=section_id,
                heading_id=section.get("heading_id", ""),
                section_number=section.get("section_number", ""),
                children=[],
                paragraph_ids=[],
                list_ids=[],
                text_preview=section.get("text_preview", ""),
                metadata={
                    "source_section": section,
                    "section_type": section_type,
                },
            )

            nodes.append(node)

        for node in nodes:
            source_section = node.metadata.get("source_section", {})
            parent_section_id = source_section.get("parent_id", "")

            if parent_section_id and parent_section_id in section_id_to_node_id:
                node.parent_id = section_id_to_node_id[parent_section_id]

        by_id = {
            node.node_id: node for node in nodes
        }

        for node in nodes:
            if node.parent_id:
                parent = by_id.get(node.parent_id)

                if parent:
                    if parent.children is None:
                        parent.children = []

                    if node.node_id not in parent.children:
                        parent.children.append(node.node_id)

        roots = [
            node for node in nodes
            if not node.parent_id
        ]

        if not roots and nodes:
            nodes[0].parent_id = ""
            roots = [nodes[0]]

        if len(roots) > 1:
            virtual_root = DocumentTreeNode(
                node_id=make_id("node"),
                node_type="root",
                title=root_title,
                level=0,
                order=0,
                page_start=first_page,
                page_end=last_page,
                parent_id="",
                section_id="",
                heading_id="",
                section_number="",
                children=[
                    node.node_id for node in roots
                ],
                paragraph_ids=[],
                list_ids=[],
                metadata={
                    "created_from": "virtual_root",
                    "root_count": len(roots),
                },
            )

            for node in roots:
                node.parent_id = virtual_root.node_id

            nodes.insert(0, virtual_root)

        return nodes

    def _attach_paragraphs_to_nodes(
        self,
        nodes: List[DocumentTreeNode],
        paragraphs: List[Dict[str, Any]],
    ) -> List[DocumentTreeNode]:
        if not self.config.include_paragraphs:
            return nodes

        section_id_to_node = {
            node.section_id: node for node in nodes
            if node.section_id
        }

        root_node = self._get_root_node(nodes)

        for paragraph in paragraphs:
            paragraph_id = paragraph.get("paragraph_id", "")

            if not paragraph_id:
                continue

            section_id = paragraph.get("section_id", "")
            node = section_id_to_node.get(section_id)

            if node is None:
                node = root_node

            if node is None:
                continue

            if node.paragraph_ids is None:
                node.paragraph_ids = []

            if paragraph_id not in node.paragraph_ids:
                node.paragraph_ids.append(paragraph_id)

        return nodes

    def _attach_lists_to_nodes(
        self,
        nodes: List[DocumentTreeNode],
        lists: List[Dict[str, Any]],
    ) -> List[DocumentTreeNode]:
        if not self.config.include_lists:
            return nodes

        section_id_to_node = {
            node.section_id: node for node in nodes
            if node.section_id
        }

        root_node = self._get_root_node(nodes)

        for doc_list in lists:
            list_id = doc_list.get("list_id", "")

            if not list_id:
                continue

            section_id = doc_list.get("section_id", "")
            node = section_id_to_node.get(section_id)

            if node is None:
                node = root_node

            if node is None:
                continue

            if node.list_ids is None:
                node.list_ids = []

            if list_id not in node.list_ids:
                node.list_ids.append(list_id)

        return nodes

    def _attach_text_preview_to_nodes(
        self,
        nodes: List[DocumentTreeNode],
        paragraphs: List[Dict[str, Any]],
    ) -> List[DocumentTreeNode]:
        if not self.config.include_text_preview:
            return nodes

        paragraph_by_id = {
            paragraph.get("paragraph_id", ""): paragraph
            for paragraph in paragraphs
            if paragraph.get("paragraph_id")
        }

        for node in nodes:
            if node.text_preview:
                node.text_preview = node.text_preview[: self.config.text_preview_chars]
                continue

            parts: List[str] = []

            for paragraph_id in node.paragraph_ids or []:
                paragraph = paragraph_by_id.get(paragraph_id)

                if not paragraph:
                    continue

                text = paragraph.get("text", "")

                if text:
                    parts.append(text)

                joined = "\n".join(parts)

                if len(joined) >= self.config.text_preview_chars:
                    break

            node.text_preview = "\n".join(parts).strip()[: self.config.text_preview_chars]

        return nodes

    def _build_tree(
        self,
        nodes: List[DocumentTreeNode],
    ) -> Dict[str, Any]:
        if not nodes:
            return {}

        by_id = {
            node.node_id: node for node in nodes
        }

        def build_node(node: DocumentTreeNode) -> Dict[str, Any]:
            child_nodes = []

            for child_id in node.children or []:
                child = by_id.get(child_id)

                if child:
                    child_nodes.append(build_node(child))

            data = node.to_dict()
            data["children"] = child_nodes

            return data

        root = self._get_root_node(nodes)

        if root:
            return build_node(root)

        return build_node(nodes[0])

    def _build_page_summaries(
        self,
        page_raws: List[PageRaw],
    ) -> List[Dict[str, Any]]:
        if not self.config.include_page_summaries:
            return []

        pages: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            layout_meta = page_raw.metadata.get("page_layout_profiler", {})
            layout_summary = layout_meta.get("layout_summary", {})

            section_meta = page_raw.metadata.get("section_builder", {})
            paragraph_meta = page_raw.metadata.get("paragraph_builder", {})
            list_meta = page_raw.metadata.get("list_detector", {})

            table_meta = page_raw.metadata.get("table_boundary_detector", {})
            table_refiner_meta = page_raw.metadata.get("table_boundary_refiner", {})
            table_grid_meta = page_raw.metadata.get("table_grid_builder", {})
            table_structure_meta = page_raw.metadata.get("table_structure_recognizer", {})

            figure_meta = page_raw.metadata.get("figure_detector", {})
            caption_meta = page_raw.metadata.get("caption_detector", {})

            page_summary = {
                "page_number": page_raw.page_number,
                "page_index": page_raw.page_index,
                "width": page_raw.width,
                "height": page_raw.height,
                "word_count": len(page_raw.words),
                "text_line_count": len(page_raw.text_lines),
                "text_block_count": len(page_raw.text_blocks),
                "image_count": len(page_raw.images),
                "drawing_count": len(page_raw.drawings),
                "layout_summary": layout_summary,
                "section_count_on_page": section_meta.get("section_count_on_page", 0),
                "paragraph_count_on_page": paragraph_meta.get("paragraph_count_on_page", 0),
                "list_item_count_on_page": list_meta.get("list_item_count_on_page", 0),
                "list_count_on_page": list_meta.get("list_count_on_page", 0),
            }

            if self.config.include_table_summary:
                page_summary["table_summary"] = table_meta.get("summary", {})
                page_summary["table_candidate_count"] = table_meta.get("table_candidate_count", 0)
                page_summary["table_boundary_count"] = table_refiner_meta.get("table_boundary_count_on_page", 0)
                page_summary["table_grid_count"] = table_grid_meta.get("table_grid_count_on_page", 0)
                page_summary["table_structure_count"] = table_structure_meta.get("table_structure_count_on_page", 0)

            if self.config.include_figure_summary:
                page_summary["figure_summary"] = figure_meta.get("summary", {})
                page_summary["figure_candidate_count"] = figure_meta.get("figure_candidate_count", 0)

            if self.config.include_caption_summary:
                page_summary["caption_summary"] = caption_meta.get("summary", {})
                page_summary["caption_candidate_count"] = caption_meta.get("caption_candidate_count", 0)

            pages.append(page_summary)

        return pages

    def _build_document_summary(
        self,
        page_raws: List[PageRaw],
        nodes: List[DocumentTreeNode],
        sections: List[Dict[str, Any]],
        paragraphs: List[Dict[str, Any]],
        lists: List[Dict[str, Any]],
        toc_result: Optional[Dict[str, Any]],
        heading_result: Optional[Dict[str, Any]],
        title_result: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        title_text = self._resolve_root_title(
            title_result=title_result,
            section_result={
                "sections": sections,
            },
        )

        total_words = 0
        total_chars = 0

        for paragraph in paragraphs:
            total_words += self._safe_int(paragraph.get("word_count", 0), default=0)
            total_chars += self._safe_int(paragraph.get("char_count", 0), default=0)

        table_candidate_count = 0
        table_boundary_count = 0
        table_grid_count = 0
        table_structure_count = 0
        figure_candidate_count = 0
        caption_candidate_count = 0

        for page_raw in page_raws:
            table_meta = page_raw.metadata.get("table_boundary_detector", {})
            table_refiner_meta = page_raw.metadata.get("table_boundary_refiner", {})
            table_grid_meta = page_raw.metadata.get("table_grid_builder", {})
            table_structure_meta = page_raw.metadata.get("table_structure_recognizer", {})

            figure_meta = page_raw.metadata.get("figure_detector", {})
            caption_meta = page_raw.metadata.get("caption_detector", {})

            table_candidate_count += self._safe_int(
                table_meta.get("table_candidate_count", 0),
                default=0,
            )

            table_boundary_count += self._safe_int(
                table_refiner_meta.get("table_boundary_count_on_page", 0),
                default=0,
            )

            table_grid_count += self._safe_int(
                table_grid_meta.get("table_grid_count_on_page", 0),
                default=0,
            )

            table_structure_count += self._safe_int(
                table_structure_meta.get("table_structure_count_on_page", 0),
                default=0,
            )

            figure_candidate_count += self._safe_int(
                figure_meta.get("figure_candidate_count", 0),
                default=0,
            )

            caption_candidate_count += self._safe_int(
                caption_meta.get("caption_candidate_count", 0),
                default=0,
            )

        toc_summary = {}
        if toc_result:
            toc_summary = toc_result.get("toc_summary", {})

        heading_summary = {}
        if heading_result:
            heading_summary = heading_result.get("heading_summary", {})

        root_node = self._get_root_node(nodes)

        return {
            "title": title_text,
            "page_count": len(page_raws),
            "node_count": len(nodes),
            "section_count": len(sections),
            "paragraph_count": len(paragraphs),
            "list_count": len(lists),
            "total_words": total_words,
            "total_chars": total_chars,
            "table_candidate_count": table_candidate_count,
            "table_boundary_count": table_boundary_count,
            "table_grid_count": table_grid_count,
            "table_structure_count": table_structure_count,
            "figure_candidate_count": figure_candidate_count,
            "caption_candidate_count": caption_candidate_count,
            "has_toc": bool(toc_summary.get("has_toc", False)),
            "toc_entry_count": toc_summary.get("toc_entry_count", 0),
            "heading_count": heading_summary.get("heading_count", 0),
            "max_section_level": max([node.level for node in nodes], default=0),
            "root_node_id": root_node.node_id if root_node else "",
        }

    def _collect_title_result(
        self,
        page_raws: List[PageRaw],
    ) -> Optional[Dict[str, Any]]:
        for page_raw in page_raws[:3]:
            title_meta = page_raw.metadata.get("title_detector", {})

            if title_meta:
                return title_meta

        return None

    def _collect_toc_result(
        self,
        page_raws: List[PageRaw],
    ) -> Optional[Dict[str, Any]]:
        toc_pages = []
        toc_entries = []
        toc_summary = {}

        for page_raw in page_raws:
            toc_meta = page_raw.metadata.get("toc_detector", {})

            if not toc_meta:
                continue

            toc_summary = toc_meta.get("toc_summary", toc_summary)

            if toc_meta.get("is_toc_page"):
                toc_pages.append(
                    {
                        "page_number": page_raw.page_number,
                        "page_index": page_raw.page_index,
                    }
                )

            for entry in toc_meta.get("toc_entries_on_page", []):
                toc_entries.append(entry)

        if not toc_pages and not toc_entries:
            return None

        return {
            "processor": "TOCDetector",
            "toc_pages": toc_pages,
            "toc_entries": toc_entries,
            "toc_summary": toc_summary,
        }

    def _collect_heading_result(
        self,
        page_raws: List[PageRaw],
    ) -> Optional[Dict[str, Any]]:
        headings = []
        heading_summary = {}

        for page_raw in page_raws:
            heading_meta = page_raw.metadata.get("heading_detector", {})

            if not heading_meta:
                continue

            heading_summary = heading_meta.get("heading_summary", heading_summary)

            for heading in heading_meta.get("heading_candidates_on_page", []):
                headings.append(heading)

        if not headings:
            return None

        return {
            "processor": "HeadingDetector",
            "heading_candidates": headings,
            "heading_summary": heading_summary,
        }

    def _collect_section_result(
        self,
        page_raws: List[PageRaw],
    ) -> Optional[Dict[str, Any]]:
        sections = []
        section_summary = {}
        seen = set()

        for page_raw in page_raws:
            section_meta = page_raw.metadata.get("section_builder", {})

            if not section_meta:
                continue

            section_summary = section_meta.get("section_summary", section_summary)

            for section in section_meta.get("sections_on_page", []):
                section_id = section.get("section_id", "")

                if not section_id:
                    continue

                if section_id in seen:
                    continue

                seen.add(section_id)
                sections.append(section)

        if not sections:
            return None

        sections = sorted(
            sections,
            key=lambda item: (
                self._safe_int(item.get("level", 0), default=0),
                self._safe_int(item.get("order", 0), default=0),
            ),
        )

        return {
            "processor": "SectionBuilder",
            "sections": sections,
            "section_summary": section_summary,
        }

    def _collect_paragraph_result(
        self,
        page_raws: List[PageRaw],
    ) -> Optional[Dict[str, Any]]:
        paragraphs = []
        paragraph_summary = {}
        seen = set()

        for page_raw in page_raws:
            paragraph_meta = page_raw.metadata.get("paragraph_builder", {})

            if not paragraph_meta:
                continue

            paragraph_summary = paragraph_meta.get("paragraph_summary", paragraph_summary)

            for paragraph in paragraph_meta.get("paragraphs_on_page", []):
                paragraph_id = paragraph.get("paragraph_id", "")

                if paragraph_id and paragraph_id in seen:
                    continue

                if paragraph_id:
                    seen.add(paragraph_id)

                paragraphs.append(paragraph)

        if not paragraphs:
            return None

        return {
            "processor": "ParagraphBuilder",
            "paragraphs": paragraphs,
            "paragraph_summary": paragraph_summary,
        }

    def _collect_list_result(
        self,
        page_raws: List[PageRaw],
    ) -> Optional[Dict[str, Any]]:
        list_items = []
        lists = []
        list_summary = {}
        seen_items = set()
        seen_lists = set()

        for page_raw in page_raws:
            list_meta = page_raw.metadata.get("list_detector", {})

            if not list_meta:
                continue

            list_summary = list_meta.get("list_summary", list_summary)

            for item in list_meta.get("list_items_on_page", []):
                item_id = item.get("list_item_id", "")

                if item_id and item_id in seen_items:
                    continue

                if item_id:
                    seen_items.add(item_id)

                list_items.append(item)

            for doc_list in list_meta.get("lists_on_page", []):
                list_id = doc_list.get("list_id", "")

                if list_id and list_id in seen_lists:
                    continue

                if list_id:
                    seen_lists.add(list_id)

                lists.append(doc_list)

        if not list_items and not lists:
            return None

        return {
            "processor": "ListDetector",
            "list_items": list_items,
            "lists": lists,
            "list_summary": list_summary,
        }

    def _collect_sections(
        self,
        section_result: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not section_result:
            return []

        sections = section_result.get("sections", [])

        if not self.config.include_empty_sections:
            sections = [
                section for section in sections
                if section.get("content_page_numbers")
                or section.get("section_type") == "root"
            ]

        return sections

    def _collect_paragraphs(
        self,
        paragraph_result: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not paragraph_result:
            return []

        return paragraph_result.get("paragraphs", [])

    def _collect_lists(
        self,
        list_result: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not list_result:
            return []

        return list_result.get("lists", [])

    def _resolve_root_title(
        self,
        title_result: Optional[Dict[str, Any]],
        section_result: Optional[Dict[str, Any]],
    ) -> str:
        if title_result:
            selected = title_result.get("selected_title")

            if isinstance(selected, dict):
                title = selected.get("title_text", "")

                if title:
                    return self._clean_text(title)

        if section_result:
            root = section_result.get("root_section")

            if isinstance(root, dict):
                title = root.get("title", "")

                if title:
                    return self._clean_text(title)

            sections = section_result.get("sections", [])

            for section in sections:
                if section.get("section_type") == "root":
                    title = section.get("title", "")

                    if title:
                        return self._clean_text(title)

        return self.config.root_title_fallback

    def _get_root_node(
        self,
        nodes: List[DocumentTreeNode],
    ) -> Optional[DocumentTreeNode]:
        for node in nodes:
            if node.node_type == "root" and not node.parent_id:
                return node

        for node in nodes:
            if not node.parent_id:
                return node

        return nodes[0] if nodes else None

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        summary = result.get("document_summary", {})

        for page_raw in page_raws:
            page_raw.metadata.setdefault("document_tree_builder", {})
            page_raw.metadata["document_tree_builder"] = {
                "processor": "DocumentTreeBuilder",
                "document_summary": summary,
                "has_document_tree": bool(result.get("document_tree")),
            }

    def save_document_tree(
        self,
        result: Dict[str, Any],
        page_raws: Optional[List[PageRaw]] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        if output_path is None:
            document_id = "document"

            if page_raws:
                document_id = page_raws[0].document_id

            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{document_id}_document_tree.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                result,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return str(output_path)

    def _clean_text(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

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


def build_document_tree(
    page_raws: List[PageRaw],
    title_result: Optional[Dict[str, Any]] = None,
    toc_result: Optional[Dict[str, Any]] = None,
    heading_result: Optional[Dict[str, Any]] = None,
    section_result: Optional[Dict[str, Any]] = None,
    paragraph_result: Optional[Dict[str, Any]] = None,
    list_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = DocumentTreeBuilder()
    return builder.process(
        page_raws=page_raws,
        title_result=title_result,
        toc_result=toc_result,
        heading_result=heading_result,
        section_result=section_result,
        paragraph_result=paragraph_result,
        list_result=list_result,
    )
