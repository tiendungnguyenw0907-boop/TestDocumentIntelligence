"""
cross_page_context_pipeline.py

Production V1 - Colab Ready

Purpose
-------
Run cross-page context understanding for a document.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- DocumentStructurePipeline
- TableUnderstandingPipeline

Output
------
Dictionary with:
- section_links
- paragraph_continuations
- table_continuations
- entity_links
- reference_links
- context_graph
- cross_page_context_summary

Flow
----
DocumentStructurePipeline
    ↓
TableUnderstandingPipeline
    ↓
CrossPageContextPipeline
        ├── SectionLinker
        ├── ParagraphContinuationDetector
        ├── TableContinuationDetector
        ├── EntityLinker
        ├── ReferenceLinker
        └── CrossPageContextGraphBuilder
"""

from __future__ import annotations

import importlib
import json
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from document_ai.schemas.page_raw_schema import PageRaw


@dataclass
class CrossPageContextPipelineConfig:
    run_section_linker: bool = True
    run_paragraph_continuation_detector: bool = True
    run_table_continuation_detector: bool = True
    run_entity_linker: bool = True
    run_reference_linker: bool = True
    run_context_graph_builder: bool = True

    continue_on_error: bool = True
    attach_to_pages: bool = True

    save_json: bool = False
    output_dir: str = "outputs/cross_page_context"

    include_debug: bool = True


class CrossPageContextPipeline:
    def __init__(
        self,
        config: Optional[CrossPageContextPipelineConfig] = None,
    ):
        self.config = config or CrossPageContextPipelineConfig()

        self.section_linker = self._load_component(
            module_path="document_ai.context.section_linker",
            class_name="SectionLinker",
        )

        self.paragraph_continuation_detector = self._load_component(
            module_path="document_ai.context.paragraph_continuation_detector",
            class_name="ParagraphContinuationDetector",
        )

        self.table_continuation_detector = self._load_component(
            module_path="document_ai.context.table_continuation_detector",
            class_name="TableContinuationDetector",
        )

        self.entity_linker = self._load_component(
            module_path="document_ai.context.entity_linker",
            class_name="EntityLinker",
        )

        self.reference_linker = self._load_component(
            module_path="document_ai.context.reference_linker",
            class_name="ReferenceLinker",
        )

        self.context_graph_builder = self._load_component(
            module_path="document_ai.context.cross_page_context_graph_builder",
            class_name="CrossPageContextGraphBuilder",
        )

    def process(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        page_documents: Optional[List[Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        errors: List[Dict[str, Any]] = []

        section_result: Dict[str, Any] = {}
        paragraph_continuation_result: Dict[str, Any] = {}
        table_continuation_result: Dict[str, Any] = {}
        entity_link_result: Dict[str, Any] = {}
        reference_link_result: Dict[str, Any] = {}
        context_graph_result: Dict[str, Any] = {}

        if self.config.run_section_linker:
            section_result = self._run_step(
                step_name="SectionLinker",
                component=self.section_linker,
                method_kwargs={
                    "page_raws": page_raws,
                    "document_structure_result": document_structure_result,
                },
                fallback_fn=lambda: self._fallback_section_links(
                    page_raws=page_raws,
                    document_structure_result=document_structure_result,
                ),
                errors=errors,
            )

        if self.config.run_paragraph_continuation_detector:
            paragraph_continuation_result = self._run_step(
                step_name="ParagraphContinuationDetector",
                component=self.paragraph_continuation_detector,
                method_kwargs={
                    "page_raws": page_raws,
                    "document_structure_result": document_structure_result,
                    "section_link_result": section_result,
                },
                fallback_fn=lambda: self._fallback_paragraph_continuations(
                    page_raws=page_raws,
                    document_structure_result=document_structure_result,
                ),
                errors=errors,
            )

        if self.config.run_table_continuation_detector:
            table_continuation_result = self._run_step(
                step_name="TableContinuationDetector",
                component=self.table_continuation_detector,
                method_kwargs={
                    "page_raws": page_raws,
                    "table_understanding_result": table_understanding_result,
                },
                fallback_fn=lambda: self._fallback_table_continuations(
                    page_raws=page_raws,
                    table_understanding_result=table_understanding_result,
                ),
                errors=errors,
            )

        if self.config.run_entity_linker:
            entity_link_result = self._run_step(
                step_name="EntityLinker",
                component=self.entity_linker,
                method_kwargs={
                    "page_raws": page_raws,
                    "document_structure_result": document_structure_result,
                    "section_link_result": section_result,
                    "paragraph_continuation_result": paragraph_continuation_result,
                    "table_continuation_result": table_continuation_result,
                },
                fallback_fn=lambda: self._fallback_entity_links(
                    page_raws=page_raws,
                ),
                errors=errors,
            )

        if self.config.run_reference_linker:
            reference_link_result = self._run_step(
                step_name="ReferenceLinker",
                component=self.reference_linker,
                method_kwargs={
                    "page_raws": page_raws,
                    "document_structure_result": document_structure_result,
                    "section_link_result": section_result,
                    "entity_link_result": entity_link_result,
                },
                fallback_fn=lambda: self._fallback_reference_links(
                    page_raws=page_raws,
                ),
                errors=errors,
            )

        if self.config.run_context_graph_builder:
            context_graph_result = self._run_step(
                step_name="CrossPageContextGraphBuilder",
                component=self.context_graph_builder,
                method_kwargs={
                    "page_raws": page_raws,
                    "section_link_result": section_result,
                    "paragraph_continuation_result": paragraph_continuation_result,
                    "table_continuation_result": table_continuation_result,
                    "entity_link_result": entity_link_result,
                    "reference_link_result": reference_link_result,
                    "document_structure_result": document_structure_result,
                    "table_understanding_result": table_understanding_result,
                },
                fallback_fn=lambda: self._fallback_context_graph(
                    page_raws=page_raws,
                    section_link_result=section_result,
                    paragraph_continuation_result=paragraph_continuation_result,
                    table_continuation_result=table_continuation_result,
                    entity_link_result=entity_link_result,
                    reference_link_result=reference_link_result,
                ),
                errors=errors,
            )

        result = {
            "processor": "CrossPageContextPipeline",
            "schema_version": "cross_page_context_pipeline_v1",
            "section_links": section_result,
            "paragraph_continuations": paragraph_continuation_result,
            "table_continuations": table_continuation_result,
            "entity_links": entity_link_result,
            "reference_links": reference_link_result,
            "context_graph": context_graph_result,
            "cross_page_context_summary": self._build_summary(
                page_raws=page_raws,
                section_result=section_result,
                paragraph_continuation_result=paragraph_continuation_result,
                table_continuation_result=table_continuation_result,
                entity_link_result=entity_link_result,
                reference_link_result=reference_link_result,
                context_graph_result=context_graph_result,
                errors=errors,
            ),
            "errors": errors,
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                result=result,
            )

        if self.config.save_json:
            self.save_context_result(
                result=result,
                page_raws=page_raws,
            )

        return result

    def _run_step(
        self,
        step_name: str,
        component: Any,
        method_kwargs: Dict[str, Any],
        fallback_fn: Any,
        errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if component is None:
            result = fallback_fn()
            result.setdefault("processor", step_name)
            result.setdefault("mode", "fallback")
            result.setdefault("warning", f"{step_name} component not found, fallback output generated.")
            return result

        try:
            return self._call_component(
                component=component,
                method_kwargs=method_kwargs,
            )

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
            result.setdefault("processor", step_name)
            result.setdefault("mode", "fallback_after_error")
            result.setdefault("warning", f"{step_name} failed, fallback output generated.")
            result.setdefault("error", str(exc))

            return result

    def _call_component(
        self,
        component: Any,
        method_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        if hasattr(component, "process"):
            return self._safe_call(
                fn=component.process,
                kwargs=method_kwargs,
            )

        if callable(component):
            return self._safe_call(
                fn=component,
                kwargs=method_kwargs,
            )

        return {}

    def _safe_call(
        self,
        fn: Any,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            return fn(**kwargs)
        except TypeError:
            reduced_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key == "page_raws"
            }

            return fn(**reduced_kwargs)

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

    def _fallback_section_links(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sections = []

        if document_structure_result:
            sections = document_structure_result.get("sections", []) or []

        if not sections:
            sections = self._collect_sections_from_pages(page_raws)

        links = []

        for section in sections:
            section_id = section.get("section_id", "")
            page_start = section.get("page_start")
            page_end = section.get("page_end")
            content_page_numbers = section.get("content_page_numbers", []) or []

            if not content_page_numbers and page_start is not None and page_end is not None:
                content_page_numbers = list(range(int(page_start), int(page_end) + 1))

            links.append(
                {
                    "section_link_id": self._make_id("section_link"),
                    "section_id": section_id,
                    "title": section.get("title", ""),
                    "level": section.get("level", 0),
                    "page_start": page_start,
                    "page_end": page_end,
                    "page_numbers": content_page_numbers,
                    "parent_id": section.get("parent_id", ""),
                    "child_ids": section.get("children", []) or [],
                    "link_type": "section_to_pages",
                    "confidence": 0.75 if content_page_numbers else 0.45,
                    "source": "fallback_section_links",
                }
            )

        return {
            "processor": "SectionLinker",
            "mode": "fallback",
            "section_links": links,
            "section_links_by_page": self._group_links_by_page(
                links=links,
                page_key_name="page_numbers",
            ),
            "section_link_summary": {
                "has_section_links": len(links) > 0,
                "section_link_count": len(links),
                "page_count_with_section_links": len(
                    self._group_links_by_page(
                        links=links,
                        page_key_name="page_numbers",
                    )
                ),
            },
        }

    def _fallback_paragraph_continuations(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        continuations = []

        page_texts = [
            {
                "page_number": page_raw.page_number,
                "text": self._page_text(page_raw),
            }
            for page_raw in page_raws
        ]

        for index in range(len(page_texts) - 1):
            current = page_texts[index]
            next_item = page_texts[index + 1]

            current_tail = self._last_non_empty_line(current["text"])
            next_head = self._first_non_empty_line(next_item["text"])

            if not current_tail or not next_head:
                continue

            score = 0.0

            if not current_tail.endswith((".", ":", ";", "!", "?", "”", '"')):
                score += 0.35

            if next_head and next_head[:1].islower():
                score += 0.30

            if len(current_tail) < 120:
                score += 0.10

            if score >= 0.35:
                continuations.append(
                    {
                        "paragraph_continuation_id": self._make_id("para_cont"),
                        "from_page": current["page_number"],
                        "to_page": next_item["page_number"],
                        "from_text_tail": current_tail,
                        "to_text_head": next_head,
                        "continuation_type": "cross_page_paragraph",
                        "confidence": round(min(score, 0.90), 4),
                        "source": "fallback_paragraph_continuations",
                    }
                )

        return {
            "processor": "ParagraphContinuationDetector",
            "mode": "fallback",
            "paragraph_continuations": continuations,
            "paragraph_continuations_by_page": self._group_pair_links_by_page(
                links=continuations,
                from_key="from_page",
                to_key="to_page",
            ),
            "paragraph_continuation_summary": {
                "has_paragraph_continuations": len(continuations) > 0,
                "paragraph_continuation_count": len(continuations),
            },
        }

    def _fallback_table_continuations(
        self,
        page_raws: List[PageRaw],
        table_understanding_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        multi_page_tables = []

        if table_understanding_result:
            multi_page_tables = table_understanding_result.get("multi_page_tables", []) or []

        if not multi_page_tables:
            for page_raw in page_raws:
                meta = page_raw.metadata.get("multi_page_table_detector", {})
                for item in meta.get("multi_page_tables_on_page", []) or []:
                    multi_page_tables.append(item)

        seen = set()
        unique_tables = []

        for item in multi_page_tables:
            table_id = item.get("multi_page_table_id", "")

            if table_id and table_id in seen:
                continue

            if table_id:
                seen.add(table_id)

            unique_tables.append(item)

        continuations = []

        for table in unique_tables:
            page_numbers = table.get("page_numbers", []) or []

            for index in range(len(page_numbers) - 1):
                continuations.append(
                    {
                        "table_continuation_id": self._make_id("table_cont"),
                        "multi_page_table_id": table.get("multi_page_table_id", ""),
                        "from_page": page_numbers[index],
                        "to_page": page_numbers[index + 1],
                        "table_grid_ids": table.get("table_grid_ids", []) or [],
                        "continuation_type": "multi_page_table",
                        "confidence": table.get("confidence", 0.65),
                        "source": "fallback_table_continuations",
                    }
                )

        return {
            "processor": "TableContinuationDetector",
            "mode": "fallback",
            "table_continuations": continuations,
            "table_continuations_by_page": self._group_pair_links_by_page(
                links=continuations,
                from_key="from_page",
                to_key="to_page",
            ),
            "table_continuation_summary": {
                "has_table_continuations": len(continuations) > 0,
                "table_continuation_count": len(continuations),
                "multi_page_table_count": len(unique_tables),
            },
        }

    def _fallback_entity_links(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        entities = []
        entity_occurrences = []

        patterns = [
            ("legal_document", r"\b(?:Nghị quyết|Nghị định|Quyết định|Thông tư|Luật)\s+số\s+[\w\/\-.]+"),
            ("date", r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b"),
            ("money", r"\b\d[\d\.,]*\s*(?:VNĐ|VND|đồng|tỷ|triệu)\b"),
            ("percentage", r"\b\d+(?:[,.]\d+)?\s*%\b"),
        ]

        entity_key_to_id: Dict[str, str] = {}

        for page_raw in page_raws:
            text = self._page_text(page_raw)

            for entity_type, pattern in patterns:
                for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                    value = self._clean_text(match.group(0))
                    key = f"{entity_type}|{value.lower()}"

                    if key not in entity_key_to_id:
                        entity_id = self._make_id("entity")
                        entity_key_to_id[key] = entity_id

                        entities.append(
                            {
                                "entity_id": entity_id,
                                "entity_type": entity_type,
                                "text": value,
                                "normalized_text": value.lower(),
                                "page_numbers": [],
                                "occurrence_count": 0,
                                "confidence": 0.55,
                                "source": "fallback_entity_links",
                            }
                        )

                    entity_id = entity_key_to_id[key]

                    entity_occurrences.append(
                        {
                            "entity_occurrence_id": self._make_id("entity_occ"),
                            "entity_id": entity_id,
                            "entity_type": entity_type,
                            "text": value,
                            "page_number": page_raw.page_number,
                            "start_char": match.start(),
                            "end_char": match.end(),
                            "confidence": 0.55,
                        }
                    )

        entity_by_id = {
            item["entity_id"]: item
            for item in entities
        }

        for occurrence in entity_occurrences:
            entity = entity_by_id.get(occurrence["entity_id"])

            if not entity:
                continue

            if occurrence["page_number"] not in entity["page_numbers"]:
                entity["page_numbers"].append(occurrence["page_number"])

            entity["occurrence_count"] += 1

        links = []

        for entity in entities:
            page_numbers = sorted(entity["page_numbers"])

            if len(page_numbers) <= 1:
                continue

            links.append(
                {
                    "entity_link_id": self._make_id("entity_link"),
                    "entity_id": entity["entity_id"],
                    "entity_type": entity["entity_type"],
                    "text": entity["text"],
                    "page_numbers": page_numbers,
                    "link_type": "same_entity_cross_page",
                    "confidence": 0.65,
                    "source": "fallback_entity_links",
                }
            )

        return {
            "processor": "EntityLinker",
            "mode": "fallback",
            "entities": entities,
            "entity_occurrences": entity_occurrences,
            "entity_links": links,
            "entity_links_by_page": self._group_links_by_page(
                links=links,
                page_key_name="page_numbers",
            ),
            "entity_link_summary": {
                "has_entity_links": len(links) > 0,
                "entity_count": len(entities),
                "entity_occurrence_count": len(entity_occurrences),
                "entity_link_count": len(links),
            },
        }

    def _fallback_reference_links(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        references = []

        patterns = [
            ("section_reference", r"\b(?:mục|phần|chương|điều|khoản)\s+[IVXLC\d\.]+", re.IGNORECASE),
            ("page_reference", r"\btrang\s+\d+\b", re.IGNORECASE),
            ("table_reference", r"\b(?:bảng|bang)\s+\d+(?:\.\d+)?\b", re.IGNORECASE),
            ("figure_reference", r"\b(?:hình|hinh|sơ đồ|so do)\s+\d+(?:\.\d+)?\b", re.IGNORECASE),
        ]

        for page_raw in page_raws:
            text = self._page_text(page_raw)

            for reference_type, pattern, flags in patterns:
                for match in re.finditer(pattern, text, flags=flags):
                    references.append(
                        {
                            "reference_link_id": self._make_id("ref_link"),
                            "reference_type": reference_type,
                            "text": self._clean_text(match.group(0)),
                            "from_page": page_raw.page_number,
                            "target_page": self._infer_target_page(match.group(0)),
                            "start_char": match.start(),
                            "end_char": match.end(),
                            "confidence": 0.50,
                            "source": "fallback_reference_links",
                        }
                    )

        return {
            "processor": "ReferenceLinker",
            "mode": "fallback",
            "reference_links": references,
            "reference_links_by_page": self._group_pair_links_by_page(
                links=references,
                from_key="from_page",
                to_key="target_page",
            ),
            "reference_link_summary": {
                "has_reference_links": len(references) > 0,
                "reference_link_count": len(references),
            },
        }

    def _fallback_context_graph(
        self,
        page_raws: List[PageRaw],
        section_link_result: Dict[str, Any],
        paragraph_continuation_result: Dict[str, Any],
        table_continuation_result: Dict[str, Any],
        entity_link_result: Dict[str, Any],
        reference_link_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        nodes = []
        edges = []

        for page_raw in page_raws:
            nodes.append(
                {
                    "node_id": f"page_{page_raw.page_number}",
                    "node_type": "page",
                    "page_number": page_raw.page_number,
                    "label": f"Page {page_raw.page_number}",
                    "metadata": {
                        "page_index": page_raw.page_index,
                        "width": page_raw.width,
                        "height": page_raw.height,
                    },
                }
            )

        for link in section_link_result.get("section_links", []) or []:
            section_node_id = f"section_{link.get('section_id') or self._make_id('section_ref')}"

            nodes.append(
                {
                    "node_id": section_node_id,
                    "node_type": "section",
                    "label": link.get("title", ""),
                    "metadata": link,
                }
            )

            for page_number in link.get("page_numbers", []) or []:
                edges.append(
                    {
                        "edge_id": self._make_id("edge"),
                        "source_id": section_node_id,
                        "target_id": f"page_{page_number}",
                        "edge_type": "section_appears_on_page",
                        "confidence": link.get("confidence", 0.5),
                    }
                )

        for link in paragraph_continuation_result.get("paragraph_continuations", []) or []:
            edges.append(
                {
                    "edge_id": self._make_id("edge"),
                    "source_id": f"page_{link.get('from_page')}",
                    "target_id": f"page_{link.get('to_page')}",
                    "edge_type": "paragraph_continues_to",
                    "confidence": link.get("confidence", 0.5),
                    "metadata": link,
                }
            )

        for link in table_continuation_result.get("table_continuations", []) or []:
            edges.append(
                {
                    "edge_id": self._make_id("edge"),
                    "source_id": f"page_{link.get('from_page')}",
                    "target_id": f"page_{link.get('to_page')}",
                    "edge_type": "table_continues_to",
                    "confidence": link.get("confidence", 0.5),
                    "metadata": link,
                }
            )

        for entity in entity_link_result.get("entities", []) or []:
            entity_node_id = f"entity_{entity.get('entity_id')}"

            nodes.append(
                {
                    "node_id": entity_node_id,
                    "node_type": "entity",
                    "label": entity.get("text", ""),
                    "metadata": entity,
                }
            )

            for page_number in entity.get("page_numbers", []) or []:
                edges.append(
                    {
                        "edge_id": self._make_id("edge"),
                        "source_id": entity_node_id,
                        "target_id": f"page_{page_number}",
                        "edge_type": "entity_mentioned_on_page",
                        "confidence": entity.get("confidence", 0.5),
                    }
                )

        for ref in reference_link_result.get("reference_links", []) or []:
            from_page = ref.get("from_page")
            target_page = ref.get("target_page")

            if from_page and target_page:
                edges.append(
                    {
                        "edge_id": self._make_id("edge"),
                        "source_id": f"page_{from_page}",
                        "target_id": f"page_{target_page}",
                        "edge_type": "page_references_page",
                        "confidence": ref.get("confidence", 0.5),
                        "metadata": ref,
                    }
                )

        return {
            "processor": "CrossPageContextGraphBuilder",
            "mode": "fallback",
            "nodes": nodes,
            "edges": edges,
            "context_graph_summary": {
                "has_context_graph": len(nodes) > 0,
                "node_count": len(nodes),
                "edge_count": len(edges),
            },
        }

    def _build_summary(
        self,
        page_raws: List[PageRaw],
        section_result: Dict[str, Any],
        paragraph_continuation_result: Dict[str, Any],
        table_continuation_result: Dict[str, Any],
        entity_link_result: Dict[str, Any],
        reference_link_result: Dict[str, Any],
        context_graph_result: Dict[str, Any],
        errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        graph_summary = context_graph_result.get("context_graph_summary", {})

        return {
            "page_count": len(page_raws),
            "section_link_count": self._count_items(section_result, ["section_links"]),
            "paragraph_continuation_count": self._count_items(paragraph_continuation_result, ["paragraph_continuations"]),
            "table_continuation_count": self._count_items(table_continuation_result, ["table_continuations"]),
            "entity_count": self._count_items(entity_link_result, ["entities"]),
            "entity_link_count": self._count_items(entity_link_result, ["entity_links"]),
            "reference_link_count": self._count_items(reference_link_result, ["reference_links"]),
            "context_graph_node_count": graph_summary.get("node_count", self._count_items(context_graph_result, ["nodes"])),
            "context_graph_edge_count": graph_summary.get("edge_count", self._count_items(context_graph_result, ["edges"])),
            "error_count": len(errors),
            "has_errors": len(errors) > 0,
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        summary = result.get("cross_page_context_summary", {})

        section_by_page = result.get("section_links", {}).get("section_links_by_page", {})
        paragraph_by_page = result.get("paragraph_continuations", {}).get("paragraph_continuations_by_page", {})
        table_by_page = result.get("table_continuations", {}).get("table_continuations_by_page", {})
        entity_by_page = result.get("entity_links", {}).get("entity_links_by_page", {})
        reference_by_page = result.get("reference_links", {}).get("reference_links_by_page", {})

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("cross_page_context_pipeline", {})
            page_raw.metadata["cross_page_context_pipeline"] = {
                "processor": "CrossPageContextPipeline",
                "section_links_on_page": section_by_page.get(page_key, []),
                "paragraph_continuations_on_page": paragraph_by_page.get(page_key, []),
                "table_continuations_on_page": table_by_page.get(page_key, []),
                "entity_links_on_page": entity_by_page.get(page_key, []),
                "reference_links_on_page": reference_by_page.get(page_key, []),
                "cross_page_context_summary": summary,
            }

    def save_context_result(
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
            output_path = output_dir / f"{document_id}_cross_page_context.json"

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

    def _collect_sections_from_pages(
        self,
        page_raws: List[PageRaw],
    ) -> List[Dict[str, Any]]:
        sections = []
        seen = set()

        for page_raw in page_raws:
            meta = page_raw.metadata.get("section_builder", {})
            for section in meta.get("sections_on_page", []) or []:
                section_id = section.get("section_id", "")

                if section_id and section_id in seen:
                    continue

                if section_id:
                    seen.add(section_id)

                sections.append(section)

        return sections

    def _group_links_by_page(
        self,
        links: List[Dict[str, Any]],
        page_key_name: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for link in links:
            page_numbers = link.get(page_key_name, []) or []

            for page_number in page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(link)

        return grouped

    def _group_pair_links_by_page(
        self,
        links: List[Dict[str, Any]],
        from_key: str,
        to_key: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for link in links:
            for key in [from_key, to_key]:
                page_number = link.get(key)

                if not page_number:
                    continue

                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(link)

        return grouped

    def _page_text(
        self,
        page_raw: PageRaw,
    ) -> str:
        reading_meta = page_raw.metadata.get("reading_order_builder", {})
        text = reading_meta.get("reading_order_text", "")

        if text:
            return text

        return page_raw.normalized_text or page_raw.raw_text or ""

    def _first_non_empty_line(
        self,
        text: str,
    ) -> str:
        for line in text.splitlines():
            line = self._clean_text(line)

            if line:
                return line

        return ""

    def _last_non_empty_line(
        self,
        text: str,
    ) -> str:
        for line in reversed(text.splitlines()):
            line = self._clean_text(line)

            if line:
                return line

        return ""

    def _infer_target_page(
        self,
        text: str,
    ) -> Optional[int]:
        match = re.search(r"\d+", text)

        if not match:
            return None

        try:
            return int(match.group(0))
        except Exception:
            return None

    def _count_items(
        self,
        result: Dict[str, Any],
        keys: List[str],
    ) -> int:
        for key in keys:
            value = result.get(key)

            if isinstance(value, list):
                return len(value)

        return 0

    def _make_id(
        self,
        prefix: str,
    ) -> str:
        try:
            from document_ai.schemas.page_raw_schema import make_id

            return make_id(prefix)
        except Exception:
            import uuid

            return f"{prefix}_{uuid.uuid4().hex[:12]}"

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
        text = " ".join(text.split())

        return text.strip()


def run_cross_page_context_pipeline(
    page_raws: List[PageRaw],
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pipeline = CrossPageContextPipeline()
    return pipeline.process(
        page_raws=page_raws,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
    )


def build_cross_page_context(
    page_raws: List[PageRaw],
    document_structure_result: Optional[Dict[str, Any]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return run_cross_page_context_pipeline(
        page_raws=page_raws,
        document_structure_result=document_structure_result,
        table_understanding_result=table_understanding_result,
    )
