"""
document_pipeline.py

Production V1 - Colab Ready

Purpose
-------
Master pipeline for end-to-end document processing.

Input
-----
document_path: PDF / DOCX / TXT / Image

Output
------
Dictionary with:
- loaded_document
- document_profile
- page_extraction
- page_understanding
- document_structure
- table_understanding
- cross_page_context
- knowledge
- indexing
- rag
- document_pipeline_summary

Flow
----
DocumentLoader
    ↓
DocumentProfiler
    ↓
PageExtractionPipeline
    ↓
PageUnderstandingPipeline
    ↓
DocumentStructurePipeline
    ↓
TableUnderstandingPipeline
    ↓
CrossPageContextPipeline
    ↓
KnowledgePipeline
    ↓
Indexing
    ↓
RAG
"""

from __future__ import annotations

import importlib
import inspect
import json
import traceback
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class DocumentPipelineConfig:
    run_loader: bool = True
    run_profiler: bool = True

    run_page_extraction: bool = True
    run_page_understanding: bool = True
    run_document_structure: bool = True
    run_table_understanding: bool = True
    run_cross_page_context: bool = True

    run_knowledge: bool = False
    run_indexing: bool = False
    run_rag: bool = False

    continue_on_error: bool = True
    attach_summary_to_pages: bool = True

    save_json: bool = False
    save_intermediate_json: bool = False
    output_dir: str = "outputs/document_pipeline"

    include_page_raws_in_result: bool = True
    include_page_records_in_result: bool = False
    include_debug: bool = True


class DocumentPipeline:
    def __init__(
        self,
        config: Optional[DocumentPipelineConfig] = None,
    ):
        self.config = config or DocumentPipelineConfig()

        self.loader = self._load_component(
            module_path="document_ai.ingestion.document_loader",
            class_name="DocumentLoader",
        )

        self.profiler = self._load_component(
            module_path="document_ai.ingestion.document_profiler",
            class_name="DocumentProfiler",
        )

        self.page_extraction_pipeline = self._load_component(
            module_path="document_ai.pipelines.page_extraction_pipeline",
            class_name="PageExtractionPipeline",
        )

        self.page_understanding_pipeline = self._load_component(
            module_path="document_ai.pipelines.page_understanding_pipeline",
            class_name="PageUnderstandingPipeline",
        )

        self.document_structure_pipeline = self._load_component(
            module_path="document_ai.pipelines.document_structure_pipeline",
            class_name="DocumentStructurePipeline",
        )

        self.table_understanding_pipeline = self._load_component(
            module_path="document_ai.pipelines.table_understanding_pipeline",
            class_name="TableUnderstandingPipeline",
        )

        self.cross_page_context_pipeline = self._load_component(
            module_path="document_ai.pipelines.cross_page_context_pipeline",
            class_name="CrossPageContextPipeline",
        )

        self.knowledge_pipeline = self._load_component(
            module_path="document_ai.pipelines.knowledge_pipeline",
            class_name="KnowledgePipeline",
        )

        self.rag_pipeline = self._load_component(
            module_path="document_ai.pipelines.rag_pipeline",
            class_name="RAGPipeline",
        )

    def process(
        self,
        document_path: Union[str, Path],
        query: Optional[str] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        document_path = str(document_path)
        extra_context = extra_context or {}

        errors: List[Dict[str, Any]] = []

        loaded_document: Any = None
        document_profile: Any = None

        page_extraction_result: Dict[str, Any] = {}
        page_understanding_result: Dict[str, Any] = {}
        document_structure_result: Dict[str, Any] = {}
        table_understanding_result: Dict[str, Any] = {}
        cross_page_context_result: Dict[str, Any] = {}
        knowledge_result: Dict[str, Any] = {}
        indexing_result: Dict[str, Any] = {}
        rag_result: Dict[str, Any] = {}

        page_raws: List[Any] = []

        if self.config.run_loader:
            loaded_document = self._run_step(
                step_name="DocumentLoader",
                component=self.loader,
                call_plan=[
                    {
                        "method_names": ["load", "process"],
                        "kwargs": {
                            "document_path": document_path,
                            "file_path": document_path,
                            "path": document_path,
                        },
                    }
                ],
                fallback_fn=lambda: self._fallback_loaded_document(document_path),
                errors=errors,
            )

        if self.config.run_profiler:
            document_profile = self._run_step(
                step_name="DocumentProfiler",
                component=self.profiler,
                call_plan=[
                    {
                        "method_names": ["process", "profile"],
                        "kwargs": {
                            "loaded_document": loaded_document,
                            "document": loaded_document,
                            "document_path": document_path,
                            "file_path": document_path,
                            "path": document_path,
                        },
                    }
                ],
                fallback_fn=lambda: self._fallback_document_profile(
                    document_path=document_path,
                    loaded_document=loaded_document,
                ),
                errors=errors,
            )

        if self.config.run_page_extraction:
            page_extraction_result = self._run_step(
                step_name="PageExtractionPipeline",
                component=self.page_extraction_pipeline,
                call_plan=[
                    {
                        "method_names": ["process_document", "process_file", "process"],
                        "kwargs": {
                            "document_path": document_path,
                            "file_path": document_path,
                            "path": document_path,
                            "loaded_document": loaded_document,
                            "document": loaded_document,
                            "document_profile": document_profile,
                            "profile": document_profile,
                            "max_pages": max_pages,
                            "page_numbers": page_numbers,
                        },
                    }
                ],
                fallback_fn=lambda: {
                    "processor": "PageExtractionPipeline",
                    "mode": "fallback",
                    "page_raws": [],
                    "warning": "PageExtractionPipeline component not found or failed.",
                },
                errors=errors,
            )

            page_extraction_result = self._normalize_page_stage_result(
                result=page_extraction_result,
                stage_name="page_extraction",
            )

            page_raws = self._extract_page_raws(page_extraction_result)

        if self.config.run_page_understanding and page_raws:
            page_understanding_result = self._run_step(
                step_name="PageUnderstandingPipeline",
                component=self.page_understanding_pipeline,
                call_plan=[
                    {
                        "method_names": ["process", "understand_document"],
                        "kwargs": {
                            "page_raws": page_raws,
                            "pages": page_raws,
                            "document_path": document_path,
                            "document_profile": document_profile,
                            "profile": document_profile,
                        },
                    },
                    {
                        "method_names": ["process_document"],
                        "kwargs": {
                            "document_path": document_path,
                            "file_path": document_path,
                            "path": document_path,
                            "max_pages": max_pages,
                            "page_numbers": page_numbers,
                        },
                    },
                ],
                fallback_fn=lambda: {
                    "processor": "PageUnderstandingPipeline",
                    "mode": "fallback",
                    "page_raws": page_raws,
                    "warning": "PageUnderstandingPipeline component not found or failed.",
                },
                errors=errors,
            )

            page_understanding_result = self._normalize_page_stage_result(
                result=page_understanding_result,
                stage_name="page_understanding",
            )

            updated_page_raws = self._extract_page_raws(page_understanding_result)

            if updated_page_raws:
                page_raws = updated_page_raws

        if self.config.run_document_structure and page_raws:
            document_structure_result = self._run_step(
                step_name="DocumentStructurePipeline",
                component=self.document_structure_pipeline,
                call_plan=[
                    {
                        "method_names": ["process", "build"],
                        "kwargs": {
                            "page_raws": page_raws,
                            "pages": page_raws,
                            "document_path": document_path,
                            "document_profile": document_profile,
                            "profile": document_profile,
                        },
                    },
                    {
                        "method_names": ["process_document"],
                        "kwargs": {
                            "document_path": document_path,
                            "file_path": document_path,
                            "path": document_path,
                            "max_pages": max_pages,
                            "page_numbers": page_numbers,
                        },
                    },
                ],
                fallback_fn=lambda: self._fallback_document_structure(page_raws),
                errors=errors,
            )

        if self.config.run_table_understanding and page_raws:
            table_understanding_result = self._run_step(
                step_name="TableUnderstandingPipeline",
                component=self.table_understanding_pipeline,
                call_plan=[
                    {
                        "method_names": ["process", "build"],
                        "kwargs": {
                            "page_raws": page_raws,
                            "pages": page_raws,
                            "document_structure_result": document_structure_result,
                            "document_structure": document_structure_result,
                        },
                    }
                ],
                fallback_fn=lambda: self._fallback_table_understanding(page_raws),
                errors=errors,
            )

        if self.config.run_cross_page_context and page_raws:
            cross_page_context_result = self._run_step(
                step_name="CrossPageContextPipeline",
                component=self.cross_page_context_pipeline,
                call_plan=[
                    {
                        "method_names": ["process", "build"],
                        "kwargs": {
                            "page_raws": page_raws,
                            "pages": page_raws,
                            "document_structure_result": document_structure_result,
                            "table_understanding_result": table_understanding_result,
                        },
                    }
                ],
                fallback_fn=lambda: self._fallback_cross_page_context(page_raws),
                errors=errors,
            )

        if self.config.run_knowledge and page_raws:
            knowledge_result = self._run_step(
                step_name="KnowledgePipeline",
                component=self.knowledge_pipeline,
                call_plan=[
                    {
                        "method_names": ["process", "build"],
                        "kwargs": {
                            "page_raws": page_raws,
                            "pages": page_raws,
                            "document_structure_result": document_structure_result,
                            "table_understanding_result": table_understanding_result,
                            "cross_page_context_result": cross_page_context_result,
                        },
                    }
                ],
                fallback_fn=lambda: {
                    "processor": "KnowledgePipeline",
                    "mode": "fallback",
                    "chunks": [],
                    "evidence": [],
                    "knowledge_graph": {},
                    "warning": "KnowledgePipeline component not found or disabled.",
                },
                errors=errors,
            )

        if self.config.run_indexing:
            indexing_result = self._fallback_indexing(
                knowledge_result=knowledge_result,
            )

        if self.config.run_rag:
            rag_result = self._run_step(
                step_name="RAGPipeline",
                component=self.rag_pipeline,
                call_plan=[
                    {
                        "method_names": ["process", "run", "answer"],
                        "kwargs": {
                            "query": query,
                            "page_raws": page_raws,
                            "document_structure_result": document_structure_result,
                            "table_understanding_result": table_understanding_result,
                            "cross_page_context_result": cross_page_context_result,
                            "knowledge_result": knowledge_result,
                            "indexing_result": indexing_result,
                            "extra_context": extra_context,
                        },
                    }
                ],
                fallback_fn=lambda: {
                    "processor": "RAGPipeline",
                    "mode": "fallback",
                    "query": query,
                    "answer": "",
                    "citations": [],
                    "warning": "RAGPipeline component not found or disabled.",
                },
                errors=errors,
            )

        summary = self._build_summary(
            document_path=document_path,
            loaded_document=loaded_document,
            document_profile=document_profile,
            page_raws=page_raws,
            page_extraction_result=page_extraction_result,
            page_understanding_result=page_understanding_result,
            document_structure_result=document_structure_result,
            table_understanding_result=table_understanding_result,
            cross_page_context_result=cross_page_context_result,
            knowledge_result=knowledge_result,
            indexing_result=indexing_result,
            rag_result=rag_result,
            errors=errors,
        )

        result = {
            "processor": "DocumentPipeline",
            "schema_version": "document_pipeline_v1",
            "document_path": document_path,
            "loaded_document": self._json_safe(loaded_document),
            "document_profile": self._json_safe(document_profile),
            "page_extraction": self._stage_result_for_output(page_extraction_result),
            "page_understanding": self._stage_result_for_output(page_understanding_result),
            "document_structure": self._json_safe(document_structure_result),
            "table_understanding": self._json_safe(table_understanding_result),
            "cross_page_context": self._json_safe(cross_page_context_result),
            "knowledge": self._json_safe(knowledge_result),
            "indexing": self._json_safe(indexing_result),
            "rag": self._json_safe(rag_result),
            "document_pipeline_summary": summary,
            "errors": errors,
            "config": asdict(self.config),
        }

        if self.config.include_page_raws_in_result:
            result["page_raws"] = self._json_safe(page_raws)

        if self.config.attach_summary_to_pages:
            self._attach_summary_to_pages(
                page_raws=page_raws,
                summary=summary,
            )

        if self.config.save_json:
            self.save_result(
                result=result,
                document_path=document_path,
            )

        return result

    def _run_step(
        self,
        step_name: str,
        component: Any,
        call_plan: List[Dict[str, Any]],
        fallback_fn: Any,
        errors: List[Dict[str, Any]],
    ) -> Any:
        if component is None:
            result = fallback_fn()

            if isinstance(result, dict):
                result.setdefault("processor", step_name)
                result.setdefault("mode", "fallback")
                result.setdefault("warning", f"{step_name} component not found.")

            return result

        for plan in call_plan:
            method_names = plan.get("method_names", [])
            kwargs = plan.get("kwargs", {})

            for method_name in method_names:
                if not hasattr(component, method_name):
                    continue

                method = getattr(component, method_name)

                try:
                    return self._safe_call(
                        fn=method,
                        kwargs=kwargs,
                    )

                except Exception as exc:
                    last_error = exc

                    continue

        try:
            if callable(component):
                return self._safe_call(
                    fn=component,
                    kwargs=call_plan[0].get("kwargs", {}) if call_plan else {},
                )

        except Exception as exc:
            last_error = exc

        error = {
            "step": step_name,
            "error": str(last_error) if "last_error" in locals() else "No callable method found.",
        }

        if self.config.include_debug:
            error["traceback"] = traceback.format_exc()

        errors.append(error)

        if not self.config.continue_on_error:
            raise RuntimeError(error["error"])

        result = fallback_fn()

        if isinstance(result, dict):
            result.setdefault("processor", step_name)
            result.setdefault("mode", "fallback_after_error")
            result.setdefault("error", error["error"])

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
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in parameters.values()
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

            positional_values = []

            for candidate_key in [
                "document_path",
                "file_path",
                "path",
                "page_raws",
                "pages",
                "loaded_document",
                "document",
            ]:
                if candidate_key in kwargs:
                    positional_values.append(kwargs[candidate_key])
                    break

            if positional_values:
                return fn(*positional_values)

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

    def _normalize_page_stage_result(
        self,
        result: Any,
        stage_name: str,
    ) -> Dict[str, Any]:
        if result is None:
            return {
                "processor": stage_name,
                "page_raws": [],
            }

        if isinstance(result, dict):
            if "page_raws" not in result:
                page_raws = self._extract_page_raws(result)

                if page_raws:
                    result["page_raws"] = page_raws

            return result

        if isinstance(result, list):
            return {
                "processor": stage_name,
                "page_raws": result,
                "page_count": len(result),
            }

        return {
            "processor": stage_name,
            "result": result,
            "page_raws": [],
        }

    def _extract_page_raws(
        self,
        result: Any,
    ) -> List[Any]:
        if result is None:
            return []

        if isinstance(result, list):
            return result

        if not isinstance(result, dict):
            return []

        candidate_keys = [
            "page_raws",
            "pages",
            "page_documents",
            "page_results",
            "processed_pages",
        ]

        for key in candidate_keys:
            value = result.get(key)

            if isinstance(value, list):
                return value

        if "page_extraction" in result:
            nested = self._extract_page_raws(result["page_extraction"])

            if nested:
                return nested

        if "page_understanding" in result:
            nested = self._extract_page_raws(result["page_understanding"])

            if nested:
                return nested

        return []

    def _stage_result_for_output(
        self,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return self._json_safe(result)

        output = {}

        for key, value in result.items():
            if key == "page_raws" and not self.config.include_page_raws_in_result:
                output["page_count"] = len(value) if isinstance(value, list) else 0
                continue

            output[key] = self._json_safe(value)

        return output

    def _fallback_loaded_document(
        self,
        document_path: str,
    ) -> Dict[str, Any]:
        path = Path(document_path)

        return {
            "processor": "DocumentLoader",
            "mode": "fallback",
            "source_path": document_path,
            "file_name": path.name,
            "file_extension": path.suffix.lower(),
            "exists": path.exists(),
            "file_size_bytes": path.stat().st_size if path.exists() else 0,
        }

    def _fallback_document_profile(
        self,
        document_path: str,
        loaded_document: Any,
    ) -> Dict[str, Any]:
        path = Path(document_path)

        return {
            "processor": "DocumentProfiler",
            "mode": "fallback",
            "source_path": document_path,
            "file_name": path.name,
            "file_extension": path.suffix.lower(),
            "document_type": self._infer_document_type(path),
            "page_count": self._get_attr_or_key(loaded_document, "page_count", 0),
            "need_ocr": False,
            "processing_strategy": "standard_pipeline",
        }

    def _fallback_document_structure(
        self,
        page_raws: List[Any],
    ) -> Dict[str, Any]:
        page_numbers = [
            self._get_attr_or_key(page_raw, "page_number", index + 1)
            for index, page_raw in enumerate(page_raws)
        ]

        root_section = {
            "section_id": "fallback_root_section",
            "title": "Document",
            "level": 0,
            "order": 0,
            "page_start": min(page_numbers) if page_numbers else None,
            "page_end": max(page_numbers) if page_numbers else None,
            "content_page_numbers": page_numbers,
            "section_type": "root",
            "children": [],
        }

        return {
            "processor": "DocumentStructurePipeline",
            "mode": "fallback",
            "root_section": root_section,
            "sections": [root_section],
            "section_tree": root_section,
            "paragraphs": [],
            "lists": [],
            "document_summary": {
                "page_count": len(page_raws),
                "section_count": 1,
                "paragraph_count": 0,
                "list_count": 0,
            },
        }

    def _fallback_table_understanding(
        self,
        page_raws: List[Any],
    ) -> Dict[str, Any]:
        table_boundaries = []
        table_grids = []
        table_structures = []
        table_cells = []
        table_headers = []
        table_spans = []
        table_semantics = []
        multi_page_tables = []

        for page_raw in page_raws:
            metadata = self._get_attr_or_key(page_raw, "metadata", {}) or {}

            table_refiner = metadata.get("table_boundary_refiner", {})
            table_boundaries.extend(table_refiner.get("table_boundaries_on_page", []) or [])

            grid_meta = metadata.get("table_grid_builder", {})
            table_grids.extend(grid_meta.get("table_grids_on_page", []) or [])

            structure_meta = metadata.get("table_structure_recognizer", {})
            table_structures.extend(structure_meta.get("table_structures_on_page", []) or [])

            cell_meta = metadata.get("table_cell_extractor", {})
            table_cells.extend(cell_meta.get("table_cells_on_page", []) or [])

            header_meta = metadata.get("table_header_detector", {})
            table_headers.extend(header_meta.get("table_headers_on_page", []) or [])

            span_meta = metadata.get("table_span_detector", {})
            table_spans.extend(span_meta.get("table_spans_on_page", []) or [])

            semantic_meta = metadata.get("table_semantic_recognizer", {})
            table_semantics.extend(semantic_meta.get("table_semantics_on_page", []) or [])

            multi_page_meta = metadata.get("multi_page_table_detector", {})
            multi_page_tables.extend(multi_page_meta.get("multi_page_tables_on_page", []) or [])

        return {
            "processor": "TableUnderstandingPipeline",
            "mode": "fallback_from_page_metadata",
            "table_boundaries": table_boundaries,
            "table_grids": table_grids,
            "table_structures": table_structures,
            "table_cells": table_cells,
            "table_headers": table_headers,
            "table_spans": table_spans,
            "table_semantics": table_semantics,
            "multi_page_tables": multi_page_tables,
            "table_understanding_summary": {
                "table_boundary_count": len(table_boundaries),
                "table_grid_count": len(table_grids),
                "table_structure_count": len(table_structures),
                "table_cell_count": len(table_cells),
                "table_header_count": len(table_headers),
                "table_span_count": len(table_spans),
                "table_semantic_count": len(table_semantics),
                "multi_page_table_count": len(multi_page_tables),
            },
        }

    def _fallback_cross_page_context(
        self,
        page_raws: List[Any],
    ) -> Dict[str, Any]:
        return {
            "processor": "CrossPageContextPipeline",
            "mode": "fallback",
            "section_links": {},
            "paragraph_continuations": {},
            "table_continuations": {},
            "entity_links": {},
            "reference_links": {},
            "context_graph": {
                "nodes": [
                    {
                        "node_id": f"page_{self._get_attr_or_key(page_raw, 'page_number', index + 1)}",
                        "node_type": "page",
                        "page_number": self._get_attr_or_key(page_raw, "page_number", index + 1),
                    }
                    for index, page_raw in enumerate(page_raws)
                ],
                "edges": [],
            },
            "cross_page_context_summary": {
                "page_count": len(page_raws),
                "context_graph_node_count": len(page_raws),
                "context_graph_edge_count": 0,
            },
        }

    def _fallback_indexing(
        self,
        knowledge_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        chunks = knowledge_result.get("chunks", []) if isinstance(knowledge_result, dict) else []

        return {
            "processor": "Indexing",
            "mode": "fallback",
            "bm25_index": {},
            "vector_index": {},
            "graph_index": {},
            "metadata_index": {},
            "indexing_summary": {
                "chunk_count": len(chunks),
                "has_index": False,
            },
        }

    def _build_summary(
        self,
        document_path: str,
        loaded_document: Any,
        document_profile: Any,
        page_raws: List[Any],
        page_extraction_result: Dict[str, Any],
        page_understanding_result: Dict[str, Any],
        document_structure_result: Dict[str, Any],
        table_understanding_result: Dict[str, Any],
        cross_page_context_result: Dict[str, Any],
        knowledge_result: Dict[str, Any],
        indexing_result: Dict[str, Any],
        rag_result: Dict[str, Any],
        errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        table_summary = {}

        if isinstance(table_understanding_result, dict):
            table_summary = (
                table_understanding_result.get("table_understanding_summary")
                or table_understanding_result.get("table_summary")
                or {}
            )

        context_summary = {}

        if isinstance(cross_page_context_result, dict):
            context_summary = cross_page_context_result.get("cross_page_context_summary", {}) or {}

        knowledge_summary = {}

        if isinstance(knowledge_result, dict):
            knowledge_summary = knowledge_result.get("knowledge_summary", {}) or {}

        return {
            "document_path": document_path,
            "file_name": Path(document_path).name,
            "document_type": self._infer_document_type(Path(document_path)),
            "loaded": loaded_document is not None,
            "profiled": document_profile is not None,
            "page_count": len(page_raws),
            "has_page_extraction": bool(page_extraction_result),
            "has_page_understanding": bool(page_understanding_result),
            "has_document_structure": bool(document_structure_result),
            "has_table_understanding": bool(table_understanding_result),
            "has_cross_page_context": bool(cross_page_context_result),
            "has_knowledge": bool(knowledge_result),
            "has_indexing": bool(indexing_result),
            "has_rag": bool(rag_result),
            "table_boundary_count": table_summary.get("table_boundary_count", 0),
            "table_grid_count": table_summary.get("table_grid_count", 0),
            "table_structure_count": table_summary.get("table_structure_count", 0),
            "table_cell_count": table_summary.get("table_cell_count", 0),
            "table_semantic_count": table_summary.get("table_semantic_count", 0),
            "multi_page_table_count": table_summary.get("multi_page_table_count", 0),
            "context_graph_node_count": context_summary.get("context_graph_node_count", 0),
            "context_graph_edge_count": context_summary.get("context_graph_edge_count", 0),
            "chunk_count": knowledge_summary.get("chunk_count", 0),
            "evidence_count": knowledge_summary.get("evidence_count", 0),
            "error_count": len(errors),
            "has_errors": len(errors) > 0,
        }

    def _attach_summary_to_pages(
        self,
        page_raws: List[Any],
        summary: Dict[str, Any],
    ) -> None:
        for page_raw in page_raws:
            try:
                if hasattr(page_raw, "metadata"):
                    page_raw.metadata.setdefault("document_pipeline", {})
                    page_raw.metadata["document_pipeline"] = {
                        "processor": "DocumentPipeline",
                        "document_pipeline_summary": summary,
                    }
                elif isinstance(page_raw, dict):
                    page_raw.setdefault("metadata", {})
                    page_raw["metadata"]["document_pipeline"] = {
                        "processor": "DocumentPipeline",
                        "document_pipeline_summary": summary,
                    }
            except Exception:
                continue

    def save_result(
        self,
        result: Dict[str, Any],
        document_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        if output_path is None:
            document_name = Path(document_path).stem
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{document_name}_document_pipeline.json"

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

    def _infer_document_type(
        self,
        path: Path,
    ) -> str:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return "pdf"

        if suffix in [".docx", ".doc"]:
            return "word"

        if suffix in [".txt", ".md"]:
            return "text"

        if suffix in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]:
            return "image"

        return "unknown"


def run_document_pipeline(
    document_path: Union[str, Path],
    query: Optional[str] = None,
    max_pages: Optional[int] = None,
    page_numbers: Optional[List[int]] = None,
    config: Optional[DocumentPipelineConfig] = None,
) -> Dict[str, Any]:
    pipeline = DocumentPipeline(config=config)
    return pipeline.process(
        document_path=document_path,
        query=query,
        max_pages=max_pages,
        page_numbers=page_numbers,
    )


def process_document(
    document_path: Union[str, Path],
    query: Optional[str] = None,
    max_pages: Optional[int] = None,
    page_numbers: Optional[List[int]] = None,
    config: Optional[DocumentPipelineConfig] = None,
) -> Dict[str, Any]:
    return run_document_pipeline(
        document_path=document_path,
        query=query,
        max_pages=max_pages,
        page_numbers=page_numbers,
        config=config,
    )
