"""
table_understanding_pipeline.py

Production V1 - Colab Ready

Purpose
-------
Run full table understanding pipeline for document pages.

Input
-----
List[PageRaw] after:
- PageExtractionPipeline
- PageUnderstandingPipeline

Output
------
Dictionary with:
- table_boundary_result
- table_grid_result
- table_structure_result
- table_cell_result
- table_header_result
- table_span_result
- table_semantic_result
- multi_page_table_result
- table_understanding_summary

Flow
----
PageUnderstandingPipeline
    ↓
TableBoundaryDetector
    ↓
TableGridBuilder
    ↓
TableStructureRecognizer
    ↓
TableCellExtractor
    ↓
TableHeaderDetector
    ↓
TableSpanDetector
    ↓
TableSemanticRecognizer
    ↓
MultiPageTableDetector
"""

from __future__ import annotations

import importlib
import inspect
import json
import traceback
from dataclasses import dataclass, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from document_ai.schemas.page_raw_schema import PageRaw


@dataclass
class TableUnderstandingPipelineConfig:
    run_table_boundary_detector: bool = True
    run_table_grid_builder: bool = True
    run_table_structure_recognizer: bool = True
    run_table_cell_extractor: bool = True
    run_table_header_detector: bool = True
    run_table_span_detector: bool = True
    run_table_semantic_recognizer: bool = True
    run_multi_page_table_detector: bool = True

    continue_on_error: bool = True
    attach_to_pages: bool = True

    save_json: bool = False
    output_dir: str = "outputs/table_understanding"

    include_debug: bool = True


class TableUnderstandingPipeline:
    def __init__(
        self,
        config: Optional[TableUnderstandingPipelineConfig] = None,
    ):
        self.config = config or TableUnderstandingPipelineConfig()

        self.table_boundary_detector = self._load_component(
            module_path="document_ai.table.table_boundary_detector",
            class_name="TableBoundaryDetector",
        )

        self.table_grid_builder = self._load_component(
            module_path="document_ai.table.table_grid_builder",
            class_name="TableGridBuilder",
        )

        self.table_structure_recognizer = self._load_component(
            module_path="document_ai.table.table_structure_recognizer",
            class_name="TableStructureRecognizer",
        )

        self.table_cell_extractor = self._load_component(
            module_path="document_ai.table.table_cell_extractor",
            class_name="TableCellExtractor",
        )

        self.table_header_detector = self._load_component(
            module_path="document_ai.table.table_header_detector",
            class_name="TableHeaderDetector",
        )

        self.table_span_detector = self._load_component(
            module_path="document_ai.table.table_span_detector",
            class_name="TableSpanDetector",
        )

        self.table_semantic_recognizer = self._load_component(
            module_path="document_ai.table.table_semantic_recognizer",
            class_name="TableSemanticRecognizer",
        )

        self.multi_page_table_detector = self._load_component(
            module_path="document_ai.table.multi_page_table_detector",
            class_name="MultiPageTableDetector",
        )

    def process(
        self,
        page_raws: List[PageRaw],
        document_structure_result: Optional[Dict[str, Any]] = None,
        page_documents: Optional[List[Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        errors: List[Dict[str, Any]] = []

        table_boundary_result: Dict[str, Any] = {}
        table_grid_result: Dict[str, Any] = {}
        table_structure_result: Dict[str, Any] = {}
        table_cell_result: Dict[str, Any] = {}
        table_header_result: Dict[str, Any] = {}
        table_span_result: Dict[str, Any] = {}
        table_semantic_result: Dict[str, Any] = {}
        multi_page_table_result: Dict[str, Any] = {}

        if self.config.run_table_boundary_detector:
            table_boundary_result = self._run_step(
                step_name="TableBoundaryDetector",
                component=self.table_boundary_detector,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                },
                fallback_fn=lambda: self._fallback_table_boundaries(page_raws),
                errors=errors,
            )

        if self.config.run_table_grid_builder:
            table_grid_result = self._run_step(
                step_name="TableGridBuilder",
                component=self.table_grid_builder,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "table_boundary_result": table_boundary_result,
                    "table_boundary_detector_result": table_boundary_result,
                },
                fallback_fn=lambda: self._fallback_table_grids(page_raws),
                errors=errors,
            )

        if self.config.run_table_structure_recognizer:
            table_structure_result = self._run_step(
                step_name="TableStructureRecognizer",
                component=self.table_structure_recognizer,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "table_grid_result": table_grid_result,
                    "table_boundary_result": table_boundary_result,
                },
                fallback_fn=lambda: self._fallback_table_structures(page_raws),
                errors=errors,
            )

        if self.config.run_table_cell_extractor:
            table_cell_result = self._run_step(
                step_name="TableCellExtractor",
                component=self.table_cell_extractor,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "table_grid_result": table_grid_result,
                    "table_structure_result": table_structure_result,
                },
                fallback_fn=lambda: self._fallback_table_cells(page_raws),
                errors=errors,
            )

        if self.config.run_table_header_detector:
            table_header_result = self._run_step(
                step_name="TableHeaderDetector",
                component=self.table_header_detector,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "table_cell_result": table_cell_result,
                    "table_structure_result": table_structure_result,
                },
                fallback_fn=lambda: self._fallback_table_headers(page_raws),
                errors=errors,
            )

        if self.config.run_table_span_detector:
            table_span_result = self._run_step(
                step_name="TableSpanDetector",
                component=self.table_span_detector,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "table_cell_result": table_cell_result,
                    "table_header_result": table_header_result,
                    "table_grid_result": table_grid_result,
                },
                fallback_fn=lambda: self._fallback_table_spans(page_raws),
                errors=errors,
            )

        if self.config.run_table_semantic_recognizer:
            table_semantic_result = self._run_step(
                step_name="TableSemanticRecognizer",
                component=self.table_semantic_recognizer,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "table_cell_result": table_cell_result,
                    "table_header_result": table_header_result,
                    "table_span_result": table_span_result,
                    "table_structure_result": table_structure_result,
                },
                fallback_fn=lambda: self._fallback_table_semantics(page_raws),
                errors=errors,
            )

        if self.config.run_multi_page_table_detector:
            multi_page_table_result = self._run_step(
                step_name="MultiPageTableDetector",
                component=self.multi_page_table_detector,
                kwargs={
                    "page_raws": page_raws,
                    "pages": page_raws,
                    "table_semantic_result": table_semantic_result,
                    "table_structure_result": table_structure_result,
                    "table_grid_result": table_grid_result,
                },
                fallback_fn=lambda: self._fallback_multi_page_tables(page_raws),
                errors=errors,
            )

        result = {
            "processor": "TableUnderstandingPipeline",
            "schema_version": "table_understanding_pipeline_v1",

            "table_boundary_result": table_boundary_result,
            "table_grid_result": table_grid_result,
            "table_structure_result": table_structure_result,
            "table_cell_result": table_cell_result,
            "table_header_result": table_header_result,
            "table_span_result": table_span_result,
            "table_semantic_result": table_semantic_result,
            "multi_page_table_result": multi_page_table_result,

            "table_boundaries": self._extract_list(
                table_boundary_result,
                ["table_boundaries"],
            ),
            "table_grids": self._extract_list(
                table_grid_result,
                ["table_grids"],
            ),
            "table_structures": self._extract_list(
                table_structure_result,
                ["table_structures"],
            ),
            "table_cells": self._extract_list(
                table_cell_result,
                ["table_cells"],
            ),
            "table_headers": self._extract_list(
                table_header_result,
                ["table_headers"],
            ),
            "table_header_cells": self._extract_list(
                table_header_result,
                ["table_header_cells"],
            ),
            "table_spans": self._extract_list(
                table_span_result,
                ["table_spans"],
            ),
            "table_cells_with_spans": self._extract_list(
                table_span_result,
                ["table_cells_with_spans"],
            ),
            "table_semantics": self._extract_list(
                table_semantic_result,
                ["table_semantics"],
            ),
            "table_columns": self._extract_list(
                table_semantic_result,
                ["table_columns"],
            ),
            "table_rows": self._extract_list(
                table_semantic_result,
                ["table_rows"],
            ),
            "table_records": self._extract_list(
                table_semantic_result,
                ["table_records"],
            ),
            "multi_page_tables": self._extract_list(
                multi_page_table_result,
                ["multi_page_tables"],
            ),
            "multi_page_table_segments": self._extract_list(
                multi_page_table_result,
                ["multi_page_table_segments"],
            ),

            "table_matrices": self._extract_dict(
                table_cell_result,
                ["table_matrices"],
            ),
            "table_matrices_with_spans": self._extract_dict(
                table_span_result,
                ["table_matrices_with_spans"],
            ),

            "table_understanding_summary": self._build_summary(
                page_raws=page_raws,
                table_boundary_result=table_boundary_result,
                table_grid_result=table_grid_result,
                table_structure_result=table_structure_result,
                table_cell_result=table_cell_result,
                table_header_result=table_header_result,
                table_span_result=table_span_result,
                table_semantic_result=table_semantic_result,
                multi_page_table_result=multi_page_table_result,
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
            self.save_table_understanding_result(
                result=result,
                page_raws=page_raws,
            )

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
            result.setdefault("processor", step_name)
            result.setdefault("mode", "fallback")
            result.setdefault("warning", f"{step_name} component not found, fallback output generated.")
            return result

        try:
            if hasattr(component, "process"):
                return self._safe_call(component.process, kwargs)

            if hasattr(component, "build"):
                return self._safe_call(component.build, kwargs)

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

            if "page_raws" in kwargs:
                return fn(kwargs["page_raws"])

            if "pages" in kwargs:
                return fn(kwargs["pages"])

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

    def _fallback_table_boundaries(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        items = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_boundary_refiner", {})
            page_items = meta.get("table_boundaries_on_page", []) or []

            if not page_items:
                rough_meta = page_raw.metadata.get("table_boundary_detector", {})
                page_items = rough_meta.get("table_candidates", []) or []

            for item in page_items:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                items.append(normalized)

        return {
            "processor": "TableBoundaryDetector",
            "mode": "fallback_from_page_metadata",
            "table_boundaries": items,
            "table_boundaries_by_page": self._group_by_page(items),
            "table_boundary_summary": {
                "has_table_boundaries": len(items) > 0,
                "table_boundary_count": len(items),
                "page_count_with_tables": len(self._group_by_page(items)),
            },
        }

    def _fallback_table_grids(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        items = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_grid_builder", {})
            page_items = meta.get("table_grids_on_page", []) or []

            for item in page_items:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                items.append(normalized)

        return {
            "processor": "TableGridBuilder",
            "mode": "fallback_from_page_metadata",
            "table_grids": items,
            "table_grids_by_page": self._group_by_page(items),
            "table_grid_summary": {
                "has_table_grids": len(items) > 0,
                "table_grid_count": len(items),
                "page_count_with_grids": len(self._group_by_page(items)),
            },
        }

    def _fallback_table_structures(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        items = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_structure_recognizer", {})
            page_items = meta.get("table_structures_on_page", []) or []

            for item in page_items:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                items.append(normalized)

        return {
            "processor": "TableStructureRecognizer",
            "mode": "fallback_from_page_metadata",
            "table_structures": items,
            "table_structures_by_page": self._group_by_page(items),
            "table_structure_summary": {
                "has_table_structures": len(items) > 0,
                "table_structure_count": len(items),
                "page_count_with_structures": len(self._group_by_page(items)),
            },
        }

    def _fallback_table_cells(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        items = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_cell_extractor", {})
            page_items = meta.get("table_cells_on_page", []) or []

            for item in page_items:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                items.append(normalized)

        return {
            "processor": "TableCellExtractor",
            "mode": "fallback_from_page_metadata",
            "table_cells": items,
            "table_cells_by_page": self._group_by_page(items),
            "table_cells_by_table": self._group_by_table(items),
            "table_matrices": self._build_matrix_from_cells(items),
            "table_cell_summary": {
                "has_table_cells": len(items) > 0,
                "table_cell_count": len(items),
                "page_count_with_cells": len(self._group_by_page(items)),
                "table_count_with_cells": len(self._group_by_table(items)),
            },
        }

    def _fallback_table_headers(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        headers = []
        header_cells = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_header_detector", {})
            page_headers = meta.get("table_headers_on_page", []) or []
            page_header_cells = meta.get("table_header_cells_on_page", []) or []

            for item in page_headers:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                headers.append(normalized)

            for item in page_header_cells:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                header_cells.append(normalized)

        return {
            "processor": "TableHeaderDetector",
            "mode": "fallback_from_page_metadata",
            "table_headers": headers,
            "table_header_cells": header_cells,
            "table_headers_by_page": self._group_by_page(headers),
            "table_headers_by_table": self._group_by_table(headers),
            "table_header_summary": {
                "has_table_headers": len(headers) > 0,
                "table_header_count": len(headers),
                "table_header_cell_count": len(header_cells),
            },
        }

    def _fallback_table_spans(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        spans = []
        cells_with_spans = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_span_detector", {})
            page_spans = meta.get("table_spans_on_page", []) or []
            page_cells = meta.get("table_cells_with_spans_on_page", []) or []

            for item in page_spans:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                spans.append(normalized)

            for item in page_cells:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                cells_with_spans.append(normalized)

        return {
            "processor": "TableSpanDetector",
            "mode": "fallback_from_page_metadata",
            "table_spans": spans,
            "table_cells_with_spans": cells_with_spans,
            "table_matrices_with_spans": self._build_matrix_from_cells(cells_with_spans),
            "table_spans_by_page": self._group_by_page(spans),
            "table_spans_by_table": self._group_by_table(spans),
            "table_span_summary": {
                "has_table_spans": len(spans) > 0,
                "table_span_count": len(spans),
                "table_cell_with_span_count": len(cells_with_spans),
            },
        }

    def _fallback_table_semantics(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        semantics = []
        columns = []
        rows = []
        records = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_semantic_recognizer", {})

            for item in meta.get("table_semantics_on_page", []) or []:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                semantics.append(normalized)

            for item in meta.get("table_columns_on_page", []) or []:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                columns.append(normalized)

            for item in meta.get("table_rows_on_page", []) or []:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                rows.append(normalized)

            for item in meta.get("table_records_on_page", []) or []:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                records.append(normalized)

        return {
            "processor": "TableSemanticRecognizer",
            "mode": "fallback_from_page_metadata",
            "table_semantics": semantics,
            "table_columns": columns,
            "table_rows": rows,
            "table_records": records,
            "table_semantics_by_page": self._group_by_page(semantics),
            "table_semantics_by_table": self._group_by_table(semantics),
            "table_semantic_summary": {
                "has_table_semantics": len(semantics) > 0,
                "table_semantic_count": len(semantics),
                "table_column_count": len(columns),
                "table_row_count": len(rows),
                "table_record_count": len(records),
            },
        }

    def _fallback_multi_page_tables(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        tables = []
        segments = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("multi_page_table_detector", {})

            for item in meta.get("multi_page_tables_on_page", []) or []:
                normalized = dict(item)
                tables.append(normalized)

            for item in meta.get("multi_page_table_segments_on_page", []) or []:
                normalized = dict(item)
                normalized.setdefault("page_number", page_raw.page_number)
                normalized.setdefault("page_index", page_raw.page_index)
                segments.append(normalized)

        tables = self._deduplicate_by_id(tables, "multi_page_table_id")
        segments = self._deduplicate_by_id(segments, "segment_id")

        return {
            "processor": "MultiPageTableDetector",
            "mode": "fallback_from_page_metadata",
            "multi_page_tables": tables,
            "multi_page_table_segments": segments,
            "multi_page_tables_by_page": self._group_multi_page_tables_by_page(
                tables=tables,
                segments=segments,
            ),
            "multi_page_segments_by_page": self._group_by_page(segments),
            "multi_page_table_summary": {
                "has_multi_page_tables": len(tables) > 0,
                "multi_page_table_count": len(tables),
                "multi_page_table_segment_count": len(segments),
            },
        }

    def _build_summary(
        self,
        page_raws: List[PageRaw],
        table_boundary_result: Dict[str, Any],
        table_grid_result: Dict[str, Any],
        table_structure_result: Dict[str, Any],
        table_cell_result: Dict[str, Any],
        table_header_result: Dict[str, Any],
        table_span_result: Dict[str, Any],
        table_semantic_result: Dict[str, Any],
        multi_page_table_result: Dict[str, Any],
        errors: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        table_boundaries = self._extract_list(table_boundary_result, ["table_boundaries"])
        table_grids = self._extract_list(table_grid_result, ["table_grids"])
        table_structures = self._extract_list(table_structure_result, ["table_structures"])
        table_cells = self._extract_list(table_cell_result, ["table_cells"])
        table_headers = self._extract_list(table_header_result, ["table_headers"])
        table_header_cells = self._extract_list(table_header_result, ["table_header_cells"])
        table_spans = self._extract_list(table_span_result, ["table_spans"])
        table_semantics = self._extract_list(table_semantic_result, ["table_semantics"])
        table_columns = self._extract_list(table_semantic_result, ["table_columns"])
        table_rows = self._extract_list(table_semantic_result, ["table_rows"])
        table_records = self._extract_list(table_semantic_result, ["table_records"])
        multi_page_tables = self._extract_list(multi_page_table_result, ["multi_page_tables"])
        multi_page_segments = self._extract_list(multi_page_table_result, ["multi_page_table_segments"])

        page_numbers_with_tables = set()

        for collection in [
            table_boundaries,
            table_grids,
            table_structures,
            table_semantics,
        ]:
            for item in collection:
                page_number = item.get("page_number")

                if page_number is not None:
                    page_numbers_with_tables.add(str(page_number))

        return {
            "page_count": len(page_raws),
            "page_count_with_tables": len(page_numbers_with_tables),

            "table_boundary_count": len(table_boundaries),
            "table_grid_count": len(table_grids),
            "table_structure_count": len(table_structures),
            "table_cell_count": len(table_cells),
            "table_header_count": len(table_headers),
            "table_header_cell_count": len(table_header_cells),
            "table_span_count": len(table_spans),
            "table_semantic_count": len(table_semantics),
            "table_column_count": len(table_columns),
            "table_row_count": len(table_rows),
            "table_record_count": len(table_records),
            "multi_page_table_count": len(multi_page_tables),
            "multi_page_table_segment_count": len(multi_page_segments),

            "has_tables": len(table_boundaries) > 0 or len(table_grids) > 0 or len(table_structures) > 0,
            "has_table_cells": len(table_cells) > 0,
            "has_table_semantics": len(table_semantics) > 0,
            "has_multi_page_tables": len(multi_page_tables) > 0,

            "by_page": {
                "boundaries": self._count_by_page(table_boundaries),
                "grids": self._count_by_page(table_grids),
                "structures": self._count_by_page(table_structures),
                "cells": self._count_by_page(table_cells),
                "semantics": self._count_by_page(table_semantics),
                "multi_page_tables": self._count_by_page(multi_page_segments),
            },

            "error_count": len(errors),
            "has_errors": len(errors) > 0,
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        summary = result.get("table_understanding_summary", {})

        table_boundaries_by_page = self._group_by_page(result.get("table_boundaries", []))
        table_grids_by_page = self._group_by_page(result.get("table_grids", []))
        table_structures_by_page = self._group_by_page(result.get("table_structures", []))
        table_cells_by_page = self._group_by_page(result.get("table_cells", []))
        table_headers_by_page = self._group_by_page(result.get("table_headers", []))
        table_spans_by_page = self._group_by_page(result.get("table_spans", []))
        table_semantics_by_page = self._group_by_page(result.get("table_semantics", []))
        multi_page_segments_by_page = self._group_by_page(result.get("multi_page_table_segments", []))

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("table_understanding_pipeline", {})
            page_raw.metadata["table_understanding_pipeline"] = {
                "processor": "TableUnderstandingPipeline",

                "table_boundaries_on_page": table_boundaries_by_page.get(page_key, []),
                "table_grids_on_page": table_grids_by_page.get(page_key, []),
                "table_structures_on_page": table_structures_by_page.get(page_key, []),
                "table_cells_on_page": table_cells_by_page.get(page_key, []),
                "table_headers_on_page": table_headers_by_page.get(page_key, []),
                "table_spans_on_page": table_spans_by_page.get(page_key, []),
                "table_semantics_on_page": table_semantics_by_page.get(page_key, []),
                "multi_page_table_segments_on_page": multi_page_segments_by_page.get(page_key, []),

                "table_boundary_count_on_page": len(table_boundaries_by_page.get(page_key, [])),
                "table_grid_count_on_page": len(table_grids_by_page.get(page_key, [])),
                "table_structure_count_on_page": len(table_structures_by_page.get(page_key, [])),
                "table_cell_count_on_page": len(table_cells_by_page.get(page_key, [])),
                "table_header_count_on_page": len(table_headers_by_page.get(page_key, [])),
                "table_span_count_on_page": len(table_spans_by_page.get(page_key, [])),
                "table_semantic_count_on_page": len(table_semantics_by_page.get(page_key, [])),
                "multi_page_table_segment_count_on_page": len(multi_page_segments_by_page.get(page_key, [])),

                "table_understanding_summary": summary,
            }

    def save_table_understanding_result(
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
            output_path = output_dir / f"{document_id}_table_understanding.json"

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

    def _extract_list(
        self,
        result: Any,
        keys: List[str],
    ) -> List[Dict[str, Any]]:
        if not isinstance(result, dict):
            return []

        for key in keys:
            value = result.get(key)

            if isinstance(value, list):
                return value

        return []

    def _extract_dict(
        self,
        result: Any,
        keys: List[str],
    ) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {}

        for key in keys:
            value = result.get(key)

            if isinstance(value, dict):
                return value

        return {}

    def _group_by_page(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in items:
            page_number = item.get("page_number")

            if page_number is None:
                continue

            page_key = str(page_number)
            grouped.setdefault(page_key, [])
            grouped[page_key].append(item)

        return grouped

    def _group_by_table(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in items:
            table_key = (
                item.get("table_grid_id")
                or item.get("table_structure_id")
                or item.get("table_boundary_id")
                or item.get("table_semantic_id")
                or "unknown_table"
            )

            grouped.setdefault(table_key, [])
            grouped[table_key].append(item)

        return grouped

    def _count_by_page(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        grouped = self._group_by_page(items)

        return {
            page_key: len(page_items)
            for page_key, page_items in grouped.items()
        }

    def _build_matrix_from_cells(
        self,
        cells: List[Dict[str, Any]],
    ) -> Dict[str, List[List[Dict[str, Any]]]]:
        cells_by_table = self._group_by_table(cells)
        matrices: Dict[str, List[List[Dict[str, Any]]]] = {}

        for table_key, table_cells in cells_by_table.items():
            if not table_cells:
                continue

            row_count = max(
                [
                    self._safe_int(cell.get("row_index"), default=-1)
                    for cell in table_cells
                ],
                default=-1,
            ) + 1

            col_count = max(
                [
                    self._safe_int(cell.get("col_index"), default=-1)
                    for cell in table_cells
                ],
                default=-1,
            ) + 1

            if row_count <= 0 or col_count <= 0:
                continue

            matrix = []

            for row_index in range(row_count):
                row = []

                for col_index in range(col_count):
                    row.append(
                        {
                            "row_index": row_index,
                            "col_index": col_index,
                            "table_cell_id": "",
                            "text": "",
                            "normalized_text": "",
                            "is_empty": True,
                        }
                    )

                matrix.append(row)

            for cell in table_cells:
                row_index = self._safe_int(cell.get("row_index"), default=-1)
                col_index = self._safe_int(cell.get("col_index"), default=-1)

                if row_index < 0 or row_index >= row_count:
                    continue

                if col_index < 0 or col_index >= col_count:
                    continue

                matrix[row_index][col_index] = cell

            matrices[table_key] = matrix

        return matrices

    def _group_multi_page_tables_by_page(
        self,
        tables: List[Dict[str, Any]],
        segments: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        table_by_id = {
            item.get("multi_page_table_id", ""): item
            for item in tables
            if item.get("multi_page_table_id")
        }

        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for segment in segments:
            page_number = segment.get("page_number")
            table_id = segment.get("multi_page_table_id", "")

            if page_number is None:
                continue

            table = table_by_id.get(table_id)

            if not table:
                continue

            page_key = str(page_number)
            grouped.setdefault(page_key, [])

            if table not in grouped[page_key]:
                grouped[page_key].append(table)

        return grouped

    def _deduplicate_by_id(
        self,
        items: List[Dict[str, Any]],
        id_key: str,
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for item in items:
            item_id = item.get(id_key, "")

            if item_id and item_id in seen:
                continue

            if item_id:
                seen.add(item_id)

            result.append(item)

        return result

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


def run_table_understanding_pipeline(
    page_raws: List[PageRaw],
    document_structure_result: Optional[Dict[str, Any]] = None,
    config: Optional[TableUnderstandingPipelineConfig] = None,
) -> Dict[str, Any]:
    pipeline = TableUnderstandingPipeline(config=config)
    return pipeline.process(
        page_raws=page_raws,
        document_structure_result=document_structure_result,
    )


def understand_tables(
    page_raws: List[PageRaw],
    document_structure_result: Optional[Dict[str, Any]] = None,
    config: Optional[TableUnderstandingPipelineConfig] = None,
) -> Dict[str, Any]:
    return run_table_understanding_pipeline(
        page_raws=page_raws,
        document_structure_result=document_structure_result,
        config=config,
    )
