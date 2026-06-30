"""
table_chunk_builder.py

Production V1 - Colab Ready

Purpose
-------
Build table-aware chunks from table understanding results.

Used by:
- KnowledgePipeline
- EvidenceBuilder
- KnowledgeGraphBuilder
- TableRetriever
- RAGPipeline

Input
-----
- page_raws
- table_understanding_result
- document_structure_result
- chunk_result
- metadata_enrichment_result

Output
------
Dictionary with:
- table_chunks
- table_record_chunks
- table_row_chunks
- table_summary_chunks
- table_chunks_by_page
- table_chunks_by_table
- table_chunk_summary
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
class TableChunkBuilderConfig:
    build_table_summary_chunks: bool = True
    build_table_record_chunks: bool = True
    build_table_row_chunks: bool = True
    build_table_cell_context_chunks: bool = False
    build_multi_page_table_chunks: bool = True

    attach_to_pages: bool = True
    deduplicate_chunks: bool = True

    max_table_summary_chars: int = 2200
    max_record_chunk_chars: int = 1600
    max_row_chunk_chars: int = 1400
    max_records_per_table_summary: int = 20
    max_rows_per_table_chunk: int = 30

    min_chunk_chars: int = 20

    include_caption: bool = True
    include_column_headers: bool = True
    include_table_metadata: bool = True
    include_cell_coordinates: bool = True
    include_debug: bool = True

    table_chunk_prefix: str = "table_chunk"
    table_record_chunk_prefix: str = "table_record_chunk"
    table_row_chunk_prefix: str = "table_row_chunk"
    multi_page_table_chunk_prefix: str = "multi_page_table_chunk"

    @property
    def build_cell_context_chunks(self) -> bool:
        """Backward-compatible alias for older code."""
        return self.build_table_cell_context_chunks


class TableChunkBuilder:
    def __init__(
        self,
        config: Optional[TableChunkBuilderConfig] = None,
    ):
        self.config = config or TableChunkBuilderConfig()

    def process(
        self,
        page_raws: Optional[List[PageRaw]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        document_structure_result: Optional[Dict[str, Any]] = None,
        chunk_result: Optional[Dict[str, Any]] = None,
        metadata_enrichment_result: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        page_raws = sorted(
            page_raws or [],
            key=lambda item: item.page_number,
        )

        table_understanding_result = table_understanding_result or {}
        document_structure_result = document_structure_result or {}
        chunk_result = chunk_result or {}
        metadata_enrichment_result = metadata_enrichment_result or {}

        table_semantics = self._collect_table_semantics(table_understanding_result)
        table_records = self._collect_table_records(table_understanding_result)
        table_cells = self._collect_table_cells(table_understanding_result)
        table_grids = self._collect_table_grids(table_understanding_result)
        table_structures = self._collect_table_structures(table_understanding_result)
        table_boundaries = self._collect_table_boundaries(table_understanding_result)
        multi_page_tables = self._collect_multi_page_tables(table_understanding_result)

        records_by_table = self._group_records_by_table(table_records)
        cells_by_table = self._group_cells_by_table(table_cells)
        grids_by_id = self._index_by_ids(
            table_grids,
            ["table_grid_id"],
        )
        structures_by_id = self._index_by_ids(
            table_structures,
            ["table_structure_id", "table_grid_id"],
        )
        boundaries_by_id = self._index_by_ids(
            table_boundaries,
            ["table_boundary_id", "table_grid_id"],
        )
        section_by_id = self._collect_sections_by_id(document_structure_result)
        page_text_map = self._build_page_text_map(page_raws)
        table_metadata = self._collect_table_metadata(metadata_enrichment_result)

        table_chunks: List[Chunk] = []
        table_summary_chunks: List[Chunk] = []
        table_record_chunks: List[Chunk] = []
        table_row_chunks: List[Chunk] = []
        multi_page_table_chunks: List[Chunk] = []
        cell_context_chunks: List[Chunk] = []

        if self.config.build_table_summary_chunks:
            table_summary_chunks = self._build_table_summary_chunks(
                table_semantics=table_semantics,
                records_by_table=records_by_table,
                cells_by_table=cells_by_table,
                grids_by_id=grids_by_id,
                structures_by_id=structures_by_id,
                boundaries_by_id=boundaries_by_id,
                section_by_id=section_by_id,
                table_metadata=table_metadata,
            )
            table_chunks.extend(table_summary_chunks)

        if self.config.build_table_record_chunks:
            table_record_chunks = self._build_table_record_chunks(
                table_records=table_records,
                table_semantics=table_semantics,
                section_by_id=section_by_id,
            )
            table_chunks.extend(table_record_chunks)

        if self.config.build_table_row_chunks:
            table_row_chunks = self._build_table_row_chunks(
                table_cells=table_cells,
                table_semantics=table_semantics,
                grids_by_id=grids_by_id,
                section_by_id=section_by_id,
            )
            table_chunks.extend(table_row_chunks)

        if self.config.build_table_cell_context_chunks:
            cell_context_chunks = self._build_cell_context_chunks(
                table_cells=table_cells,
                table_semantics=table_semantics,
            )
            table_chunks.extend(cell_context_chunks)

        if self.config.build_multi_page_table_chunks:
            multi_page_table_chunks = self._build_multi_page_table_chunks(
                multi_page_tables=multi_page_tables,
                table_semantics=table_semantics,
                records_by_table=records_by_table,
                table_metadata=table_metadata,
            )
            table_chunks.extend(multi_page_table_chunks)

        table_chunks = [
            chunk for chunk in table_chunks
            if self._valid_chunk_text(chunk.text)
        ]

        table_chunks = self._sort_chunks(table_chunks)

        if self.config.deduplicate_chunks:
            table_chunks = self._deduplicate_chunks(table_chunks)

        table_chunks = self._link_sequential_chunks(table_chunks)

        collection = ChunkCollection(
            document_id=self._infer_document_id(
                page_raws=page_raws,
                table_chunks=table_chunks,
                table_understanding_result=table_understanding_result,
            ),
            source_document=self._infer_source_document(
                page_raws=page_raws,
                table_chunks=table_chunks,
            ),
            chunks=table_chunks,
            metadata={
                "processor": "TableChunkBuilder",
                "schema_version": "table_chunk_builder_v1",
                "table_semantic_count": len(table_semantics),
                "table_record_count": len(table_records),
                "table_cell_count": len(table_cells),
                "table_grid_count": len(table_grids),
                "multi_page_table_count": len(multi_page_tables),
            },
        )

        result = collection.to_dict()
        result.update(
            {
                "processor": "TableChunkBuilder",
                "schema_version": "table_chunk_builder_v1",
                "table_chunks": [
                    chunk.to_dict()
                    for chunk in table_chunks
                ],
                "table_summary_chunks": [
                    chunk.to_dict()
                    for chunk in table_chunks
                    if chunk.chunk_type == "table_summary_chunk"
                ],
                "table_record_chunks": [
                    chunk.to_dict()
                    for chunk in table_chunks
                    if chunk.chunk_type == "table_record_chunk"
                ],
                "table_row_chunks": [
                    chunk.to_dict()
                    for chunk in table_chunks
                    if chunk.chunk_type == "table_row_chunk"
                ],
                "multi_page_table_chunks": [
                    chunk.to_dict()
                    for chunk in table_chunks
                    if chunk.chunk_type == "multi_page_table_chunk"
                ],
                "table_cell_context_chunks": [
                    chunk.to_dict()
                    for chunk in table_chunks
                    if chunk.chunk_type == "table_cell_context_chunk"
                ],
                "table_chunks_by_page": self._group_chunks_by_page(table_chunks),
                "table_chunks_by_table": self._group_chunks_by_table(table_chunks),
                "table_chunks_by_type": self._group_chunks_by_type(table_chunks),
                "table_chunk_summary": self._build_summary(
                    table_chunks=table_chunks,
                    table_semantics=table_semantics,
                    table_records=table_records,
                    table_cells=table_cells,
                    multi_page_tables=multi_page_tables,
                ),
                "config": asdict(self.config),
            }
        )

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                table_chunks=table_chunks,
            )

        return result

    def _build_table_summary_chunks(
        self,
        table_semantics: List[Dict[str, Any]],
        records_by_table: Dict[str, List[Dict[str, Any]]],
        cells_by_table: Dict[str, List[Dict[str, Any]]],
        grids_by_id: Dict[str, Dict[str, Any]],
        structures_by_id: Dict[str, Dict[str, Any]],
        boundaries_by_id: Dict[str, Dict[str, Any]],
        section_by_id: Dict[str, Dict[str, Any]],
        table_metadata: Dict[str, Dict[str, Any]],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []

        for table in table_semantics:
            table_id = self._table_id(table)

            if not table_id:
                continue

            records = records_by_table.get(table_id, [])
            cells = cells_by_table.get(table_id, [])

            table_grid_id = table.get("table_grid_id", "")
            table_structure_id = table.get("table_structure_id", "")
            table_boundary_id = table.get("table_boundary_id", "")

            grid = grids_by_id.get(table_grid_id, {})
            structure = structures_by_id.get(table_structure_id) or structures_by_id.get(table_grid_id, {}) or {}
            boundary = boundaries_by_id.get(table_boundary_id) or boundaries_by_id.get(table_grid_id, {}) or {}

            section_id = table.get("section_id", "")
            section = section_by_id.get(section_id, {})
            metadata = table_metadata.get(table_id, {})

            page_numbers = self._resolve_page_numbers(table)

            if not page_numbers and table.get("page_number"):
                page_number = self._safe_int(table.get("page_number"), default=0)
                if page_number > 0:
                    page_numbers = [page_number]

            text = self._table_summary_text(
                table=table,
                records=records,
                cells=cells,
                grid=grid,
                structure=structure,
                boundary=boundary,
                metadata=metadata,
            )

            text = self._truncate_text(text, self.config.max_table_summary_chars)

            if not self._valid_chunk_text(text):
                continue

            chunk = Chunk(
                chunk_id=make_id(self.config.table_chunk_prefix),
                chunk_type="table_summary_chunk",
                text=text,
                document_id=table.get("document_id", ""),
                source_document=table.get("source_document", ""),
                page_numbers=page_numbers,
                page_start=min(page_numbers) if page_numbers else None,
                page_end=max(page_numbers) if page_numbers else None,
                section_id=section_id,
                section_title=table.get("section_title", "") or section.get("title", ""),
                section_level=section.get("level"),
                table_grid_id=table_grid_id,
                table_structure_id=table_structure_id,
                table_semantic_id=table.get("table_semantic_id", ""),
                table_boundary_id=table_boundary_id,
                bbox=table.get("bbox", []) or boundary.get("bbox", []),
                order=self._safe_int(table.get("order"), default=0),
                source="table_chunk_builder_summary",
                confidence=self._safe_float(table.get("confidence"), default=0.72),
                metadata={
                    "table_id": table_id,
                    "table_kind": "table_semantic",
                    "semantic_type": table.get("semantic_type", ""),
                    "table_type": table.get("table_type", ""),
                    "row_count": table.get("row_count", 0),
                    "col_count": table.get("col_count", 0),
                    "record_count": len(records),
                    "cell_count": len(cells),
                    "column_headers": table.get("column_headers", []),
                    "title": table.get("title", ""),
                    "caption": table.get("caption", "") or table.get("caption_text", ""),
                    "metadata_enriched": metadata,
                },
            )

            chunks.append(chunk)

        return chunks

    def _build_table_record_chunks(
        self,
        table_records: List[Dict[str, Any]],
        table_semantics: List[Dict[str, Any]],
        section_by_id: Dict[str, Dict[str, Any]],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        table_by_id = self._index_tables(table_semantics)

        for record in table_records:
            record_id = record.get("table_record_id", "") or make_id("record")
            table_id = (
                record.get("table_semantic_id")
                or record.get("table_grid_id")
                or ""
            )

            table = table_by_id.get(table_id, {})
            section_id = record.get("section_id", "") or table.get("section_id", "")
            section = section_by_id.get(section_id, {})

            page_numbers = self._resolve_page_numbers(record)

            if not page_numbers and record.get("page_number"):
                page_number = self._safe_int(record.get("page_number"), default=0)
                if page_number > 0:
                    page_numbers = [page_number]

            text = self._record_text(
                record=record,
                table=table,
            )

            text = self._truncate_text(text, self.config.max_record_chunk_chars)

            if not self._valid_chunk_text(text):
                continue

            chunk = Chunk(
                chunk_id=make_id(self.config.table_record_chunk_prefix),
                chunk_type="table_record_chunk",
                text=text,
                document_id=record.get("document_id", "") or table.get("document_id", ""),
                source_document=record.get("source_document", "") or table.get("source_document", ""),
                page_numbers=page_numbers,
                page_start=min(page_numbers) if page_numbers else None,
                page_end=max(page_numbers) if page_numbers else None,
                section_id=section_id,
                section_title=record.get("section_title", "") or table.get("section_title", "") or section.get("title", ""),
                section_level=section.get("level"),
                table_grid_id=record.get("table_grid_id", "") or table.get("table_grid_id", ""),
                table_structure_id=record.get("table_structure_id", "") or table.get("table_structure_id", ""),
                table_semantic_id=record.get("table_semantic_id", "") or table.get("table_semantic_id", ""),
                table_boundary_id=record.get("table_boundary_id", "") or table.get("table_boundary_id", ""),
                order=self._safe_int(record.get("record_index") or record.get("row_index"), default=0),
                source="table_chunk_builder_record",
                confidence=self._safe_float(record.get("confidence"), default=0.72),
                metadata={
                    "table_id": table_id,
                    "table_record_id": record_id,
                    "row_index": record.get("row_index", 0),
                    "record_index": record.get("record_index", 0),
                    "cell_ids": record.get("cell_ids", []) or [],
                    "values": record.get("values", {}) or {},
                    "raw_values": record.get("raw_values", {}) or {},
                    "semantic_type": table.get("semantic_type", ""),
                    "column_headers": table.get("column_headers", []),
                },
            )

            chunks.append(chunk)

        return chunks

    def _build_table_row_chunks(
        self,
        table_cells: List[Dict[str, Any]],
        table_semantics: List[Dict[str, Any]],
        grids_by_id: Dict[str, Dict[str, Any]],
        section_by_id: Dict[str, Dict[str, Any]],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        table_by_id = self._index_tables(table_semantics)

        grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

        for cell in table_cells:
            table_id = (
                cell.get("table_semantic_id")
                or cell.get("table_grid_id")
                or cell.get("table_structure_id")
                or ""
            )

            if not table_id:
                continue

            row_index = self._safe_int(cell.get("row_index"), default=0)
            grouped.setdefault((table_id, row_index), [])
            grouped[(table_id, row_index)].append(cell)

        for (table_id, row_index), cells in grouped.items():
            cells = sorted(
                cells,
                key=lambda item: self._safe_int(item.get("col_index"), default=0),
            )

            table = table_by_id.get(table_id, {})
            grid = grids_by_id.get(table.get("table_grid_id", "")) or grids_by_id.get(table_id, {}) or {}

            section_id = table.get("section_id", "")
            section = section_by_id.get(section_id, {})

            page_numbers = []

            for cell in cells:
                page_numbers.extend(self._resolve_page_numbers(cell))

                if cell.get("page_number"):
                    page_number = self._safe_int(cell.get("page_number"), default=0)
                    if page_number > 0:
                        page_numbers.append(page_number)

            if not page_numbers:
                page_numbers = self._resolve_page_numbers(table)

            page_numbers = sorted(list(dict.fromkeys(page_numbers)))

            text = self._row_text(
                row_index=row_index,
                cells=cells,
                table=table,
                grid=grid,
            )

            text = self._truncate_text(text, self.config.max_row_chunk_chars)

            if not self._valid_chunk_text(text):
                continue

            bbox = self._merge_bboxes(
                [
                    cell.get("bbox", [])
                    for cell in cells
                    if cell.get("bbox")
                ]
            )

            chunk = Chunk(
                chunk_id=make_id(self.config.table_row_chunk_prefix),
                chunk_type="table_row_chunk",
                text=text,
                document_id=table.get("document_id", ""),
                source_document=table.get("source_document", ""),
                page_numbers=page_numbers,
                page_start=min(page_numbers) if page_numbers else None,
                page_end=max(page_numbers) if page_numbers else None,
                section_id=section_id,
                section_title=table.get("section_title", "") or section.get("title", ""),
                section_level=section.get("level"),
                table_grid_id=table.get("table_grid_id", "") or table_id,
                table_structure_id=table.get("table_structure_id", ""),
                table_semantic_id=table.get("table_semantic_id", ""),
                table_boundary_id=table.get("table_boundary_id", ""),
                bbox=bbox,
                order=row_index,
                source="table_chunk_builder_row",
                confidence=self._average_confidence(cells, default=0.68),
                metadata={
                    "table_id": table_id,
                    "row_index": row_index,
                    "cell_count": len(cells),
                    "cell_ids": [
                        cell.get("table_cell_id") or cell.get("cell_id") or ""
                        for cell in cells
                        if cell.get("table_cell_id") or cell.get("cell_id")
                    ],
                    "semantic_type": table.get("semantic_type", ""),
                    "column_headers": table.get("column_headers", []),
                    "is_header_row": self._is_header_row(row_index, cells, table),
                },
            )

            chunks.append(chunk)

        return chunks

    def _build_cell_context_chunks(
        self,
        table_cells: List[Dict[str, Any]],
        table_semantics: List[Dict[str, Any]],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        table_by_id = self._index_tables(table_semantics)

        for cell in table_cells:
            text = normalize_text(cell.get("text", ""))

            if not text:
                continue

            table_id = (
                cell.get("table_semantic_id")
                or cell.get("table_grid_id")
                or cell.get("table_structure_id")
                or ""
            )

            table = table_by_id.get(table_id, {})
            headers = table.get("column_headers", []) or []
            col_index = self._safe_int(cell.get("col_index"), default=0)
            row_index = self._safe_int(cell.get("row_index"), default=0)

            header = ""
            if 0 <= col_index < len(headers):
                header = normalize_text(headers[col_index])

            chunk_text = self._cell_context_text(
                cell=cell,
                table=table,
                header=header,
            )

            if not self._valid_chunk_text(chunk_text):
                continue

            page_numbers = self._resolve_page_numbers(cell)
            if not page_numbers and cell.get("page_number"):
                page_number = self._safe_int(cell.get("page_number"), default=0)
                if page_number > 0:
                    page_numbers = [page_number]

            chunk = Chunk(
                chunk_id=make_id("table_cell_context_chunk"),
                chunk_type="table_cell_context_chunk",
                text=chunk_text,
                document_id=table.get("document_id", ""),
                source_document=table.get("source_document", ""),
                page_numbers=page_numbers,
                page_start=min(page_numbers) if page_numbers else None,
                page_end=max(page_numbers) if page_numbers else None,
                section_id=table.get("section_id", ""),
                section_title=table.get("section_title", ""),
                table_grid_id=cell.get("table_grid_id", "") or table.get("table_grid_id", ""),
                table_structure_id=cell.get("table_structure_id", "") or table.get("table_structure_id", ""),
                table_semantic_id=cell.get("table_semantic_id", "") or table.get("table_semantic_id", ""),
                table_boundary_id=cell.get("table_boundary_id", "") or table.get("table_boundary_id", ""),
                bbox=cell.get("bbox", []) or [],
                order=row_index * 1000 + col_index,
                source="table_chunk_builder_cell_context",
                confidence=self._safe_float(cell.get("confidence"), default=0.65),
                metadata={
                    "table_id": table_id,
                    "cell_id": cell.get("table_cell_id") or cell.get("cell_id") or "",
                    "row_index": row_index,
                    "col_index": col_index,
                    "column_header": header,
                    "role": cell.get("role", ""),
                    "data_type": cell.get("data_type", ""),
                },
            )

            chunks.append(chunk)

        return chunks

    def _build_multi_page_table_chunks(
        self,
        multi_page_tables: List[Dict[str, Any]],
        table_semantics: List[Dict[str, Any]],
        records_by_table: Dict[str, List[Dict[str, Any]]],
        table_metadata: Dict[str, Dict[str, Any]],
    ) -> List[Chunk]:
        chunks: List[Chunk] = []
        table_by_id = self._index_tables(table_semantics)

        for multi_table in multi_page_tables:
            multi_table_id = multi_table.get("multi_page_table_id", "") or make_id("multi_page_table")
            page_numbers = self._resolve_page_numbers(multi_table)

            table_ids = []
            table_ids.extend(multi_table.get("table_semantic_ids", []) or [])
            table_ids.extend(multi_table.get("table_grid_ids", []) or [])
            table_ids.extend(multi_table.get("table_structure_ids", []) or [])
            table_ids.extend(multi_table.get("table_boundary_ids", []) or [])
            table_ids = [
                str(item)
                for item in table_ids
                if item
            ]

            related_tables = [
                table_by_id[table_id]
                for table_id in table_ids
                if table_id in table_by_id
            ]

            related_records = []

            for table_id in table_ids:
                related_records.extend(records_by_table.get(table_id, []))

            metadata = table_metadata.get(multi_table_id, {})

            text = self._multi_page_table_text(
                multi_table=multi_table,
                related_tables=related_tables,
                related_records=related_records,
                metadata=metadata,
            )

            text = self._truncate_text(text, self.config.max_table_summary_chars)

            if not self._valid_chunk_text(text):
                continue

            chunk = Chunk(
                chunk_id=make_id(self.config.multi_page_table_chunk_prefix),
                chunk_type="multi_page_table_chunk",
                text=text,
                document_id=multi_table.get("document_id", ""),
                source_document=multi_table.get("source_document", ""),
                page_numbers=page_numbers,
                page_start=min(page_numbers) if page_numbers else None,
                page_end=max(page_numbers) if page_numbers else None,
                table_grid_id=(multi_table.get("table_grid_ids", []) or [""])[0] if multi_table.get("table_grid_ids") else "",
                table_structure_id=(multi_table.get("table_structure_ids", []) or [""])[0] if multi_table.get("table_structure_ids") else "",
                table_semantic_id=(multi_table.get("table_semantic_ids", []) or [""])[0] if multi_table.get("table_semantic_ids") else "",
                table_boundary_id=(multi_table.get("table_boundary_ids", []) or [""])[0] if multi_table.get("table_boundary_ids") else "",
                order=self._safe_int(multi_table.get("order"), default=0),
                source="table_chunk_builder_multi_page",
                confidence=self._safe_float(multi_table.get("confidence"), default=0.76),
                metadata={
                    "multi_page_table_id": multi_table_id,
                    "table_ids": table_ids,
                    "segment_count": len(multi_table.get("segments", []) or []),
                    "total_row_count": multi_table.get("total_row_count", 0),
                    "col_count": multi_table.get("col_count", 0),
                    "column_headers": multi_table.get("column_headers", []),
                    "semantic_type": multi_table.get("semantic_type", ""),
                    "table_type": multi_table.get("table_type", ""),
                    "related_table_count": len(related_tables),
                    "related_record_count": len(related_records),
                    "metadata_enriched": metadata,
                },
            )

            chunks.append(chunk)

        return chunks

    def _table_summary_text(
        self,
        table: Dict[str, Any],
        records: List[Dict[str, Any]],
        cells: List[Dict[str, Any]],
        grid: Dict[str, Any],
        structure: Dict[str, Any],
        boundary: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> str:
        parts = []

        title = normalize_text(table.get("title", ""))
        caption = normalize_text(table.get("caption") or table.get("caption_text") or "")

        if self.config.include_caption:
            if title:
                parts.append(f"Tiêu đề bảng: {title}")

            if caption and caption != title:
                parts.append(f"Chú thích bảng: {caption}")

        semantic_type = normalize_text(table.get("semantic_type", ""))
        table_type = normalize_text(table.get("table_type", ""))

        if semantic_type or table_type:
            parts.append(
                "Loại bảng: "
                + "; ".join(
                    [
                        item
                        for item in [semantic_type, table_type]
                        if item
                    ]
                )
            )

        headers = table.get("column_headers", []) or []

        if self.config.include_column_headers and headers:
            header_text = " | ".join(
                [
                    normalize_text(item)
                    for item in headers
                    if normalize_text(item)
                ]
            )

            if header_text:
                parts.append(f"Cột dữ liệu: {header_text}")

        row_count = (
            table.get("row_count")
            or grid.get("row_count")
            or structure.get("row_count")
            or 0
        )
        col_count = (
            table.get("col_count")
            or grid.get("col_count")
            or structure.get("col_count")
            or 0
        )

        if row_count or col_count:
            parts.append(f"Quy mô bảng: {row_count or 0} dòng, {col_count or 0} cột")

        if records:
            parts.append("Một số dòng dữ liệu:")

            for record in records[: self.config.max_records_per_table_summary]:
                record_text = self._record_values_text(record)
                if record_text:
                    parts.append(f"- {record_text}")

        elif cells:
            row_texts = self._sample_rows_from_cells(cells, headers=headers)

            if row_texts:
                parts.append("Một số dòng dữ liệu:")
                parts.extend([f"- {item}" for item in row_texts[: self.config.max_records_per_table_summary]])

        if self.config.include_table_metadata:
            quality_flags = metadata.get("quality_flags", []) or []

            if quality_flags:
                parts.append("Cờ chất lượng bảng: " + ", ".join(quality_flags))

        return normalize_text("\n".join(parts))

    def _record_text(
        self,
        record: Dict[str, Any],
        table: Dict[str, Any],
    ) -> str:
        parts = []

        title = normalize_text(table.get("title") or table.get("caption") or "")
        if title:
            parts.append(f"Bảng: {title}")

        row_index = record.get("row_index")
        record_index = record.get("record_index")

        if row_index is not None or record_index is not None:
            parts.append(
                f"Dòng dữ liệu: row_index={row_index if row_index is not None else ''}; record_index={record_index if record_index is not None else ''}"
            )

        value_text = self._record_values_text(record)

        if value_text:
            parts.append(value_text)

        return normalize_text("\n".join(parts))

    def _row_text(
        self,
        row_index: int,
        cells: List[Dict[str, Any]],
        table: Dict[str, Any],
        grid: Dict[str, Any],
    ) -> str:
        parts = []

        title = normalize_text(table.get("title") or table.get("caption") or table.get("caption_text") or "")
        if title:
            parts.append(f"Bảng: {title}")

        parts.append(f"Dòng {row_index}")

        headers = table.get("column_headers", []) or []
        cell_parts = []

        for cell in cells:
            col_index = self._safe_int(cell.get("col_index"), default=0)
            cell_text = normalize_text(cell.get("text", ""))

            if not cell_text:
                continue

            if 0 <= col_index < len(headers):
                header = normalize_text(headers[col_index])
            else:
                header = f"Cột {col_index}"

            if self.config.include_cell_coordinates:
                cell_parts.append(f"{header} [r{row_index}, c{col_index}]: {cell_text}")
            else:
                cell_parts.append(f"{header}: {cell_text}")

        if cell_parts:
            parts.append(" | ".join(cell_parts))

        return normalize_text("\n".join(parts))

    def _cell_context_text(
        self,
        cell: Dict[str, Any],
        table: Dict[str, Any],
        header: str = "",
    ) -> str:
        parts = []

        title = normalize_text(table.get("title") or table.get("caption") or "")
        if title:
            parts.append(f"Bảng: {title}")

        row_index = self._safe_int(cell.get("row_index"), default=0)
        col_index = self._safe_int(cell.get("col_index"), default=0)
        text = normalize_text(cell.get("text", ""))

        if header:
            parts.append(f"Cột: {header}")

        parts.append(f"Ô [r{row_index}, c{col_index}]: {text}")

        role = normalize_text(cell.get("role", ""))
        data_type = normalize_text(cell.get("data_type", ""))

        if role or data_type:
            parts.append(
                "Ngữ nghĩa ô: "
                + "; ".join(
                    [
                        item
                        for item in [role, data_type]
                        if item
                    ]
                )
            )

        return normalize_text("\n".join(parts))

    def _multi_page_table_text(
        self,
        multi_table: Dict[str, Any],
        related_tables: List[Dict[str, Any]],
        related_records: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> str:
        parts = []

        page_numbers = self._resolve_page_numbers(multi_table)

        if page_numbers:
            if len(page_numbers) == 1:
                parts.append(f"Bảng trên trang {page_numbers[0]}")
            else:
                parts.append(f"Bảng nhiều trang từ trang {page_numbers[0]} đến trang {page_numbers[-1]}")

        headers = multi_table.get("column_headers", []) or []

        if not headers:
            for table in related_tables:
                headers = table.get("column_headers", []) or []
                if headers:
                    break

        if headers:
            parts.append(
                "Cột dữ liệu: "
                + " | ".join(
                    [
                        normalize_text(item)
                        for item in headers
                        if normalize_text(item)
                    ]
                )
            )

        total_row_count = multi_table.get("total_row_count", 0)
        col_count = multi_table.get("col_count", 0)

        if total_row_count or col_count:
            parts.append(f"Quy mô bảng nhiều trang: {total_row_count or 0} dòng, {col_count or 0} cột")

        semantic_type = normalize_text(multi_table.get("semantic_type", ""))
        table_type = normalize_text(multi_table.get("table_type", ""))

        if semantic_type or table_type:
            parts.append(
                "Loại bảng: "
                + "; ".join(
                    [
                        item
                        for item in [semantic_type, table_type]
                        if item
                    ]
                )
            )

        if related_tables:
            parts.append(f"Số đoạn bảng liên quan: {len(related_tables)}")

        if related_records:
            parts.append("Một số dòng dữ liệu:")

            for record in related_records[: self.config.max_records_per_table_summary]:
                record_text = self._record_values_text(record)
                if record_text:
                    parts.append(f"- {record_text}")

        quality_flags = metadata.get("quality_flags", []) or []

        if quality_flags:
            parts.append("Cờ chất lượng bảng: " + ", ".join(quality_flags))

        return normalize_text("\n".join(parts))

    def _record_values_text(
        self,
        record: Dict[str, Any],
    ) -> str:
        values = record.get("values", {}) or record.get("raw_values", {}) or {}

        if isinstance(values, dict) and values:
            parts = []

            for key, value in values.items():
                value_text = normalize_text(value)

                if not value_text:
                    continue

                key_text = normalize_text(key)

                if key_text:
                    parts.append(f"{key_text}: {value_text}")
                else:
                    parts.append(value_text)

            return normalize_text(" | ".join(parts))

        text = normalize_text(record.get("text", ""))

        if text:
            return text

        return ""

    def _sample_rows_from_cells(
        self,
        cells: List[Dict[str, Any]],
        headers: Optional[List[str]] = None,
    ) -> List[str]:
        headers = headers or []
        grouped: Dict[int, List[Dict[str, Any]]] = {}

        for cell in cells:
            row_index = self._safe_int(cell.get("row_index"), default=0)
            grouped.setdefault(row_index, [])
            grouped[row_index].append(cell)

        rows = []

        for row_index in sorted(grouped.keys())[: self.config.max_rows_per_table_chunk]:
            row_cells = sorted(
                grouped[row_index],
                key=lambda item: self._safe_int(item.get("col_index"), default=0),
            )

            parts = []

            for cell in row_cells:
                text = normalize_text(cell.get("text", ""))

                if not text:
                    continue

                col_index = self._safe_int(cell.get("col_index"), default=0)

                if 0 <= col_index < len(headers):
                    header = normalize_text(headers[col_index])
                    parts.append(f"{header}: {text}")
                else:
                    parts.append(text)

            row_text = normalize_text(" | ".join(parts))

            if row_text:
                rows.append(row_text)

        return rows

    def _collect_table_semantics(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        values = table_understanding_result.get("table_semantics", []) or []
        if isinstance(values, list):
            items.extend([self._to_dict(item) for item in values])

        sub = table_understanding_result.get("table_semantic_result", {}) or {}
        if isinstance(sub, dict):
            values = sub.get("table_semantics", []) or []
            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(items, ["table_semantic_id", "table_grid_id"])

    def _collect_table_records(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        values = table_understanding_result.get("table_records", []) or []
        if isinstance(values, list):
            items.extend([self._to_dict(item) for item in values])

        sub = table_understanding_result.get("table_semantic_result", {}) or {}
        if isinstance(sub, dict):
            values = sub.get("table_records", []) or []
            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(items, ["table_record_id"])

    def _collect_table_cells(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        for key in ["table_cells", "cells", "extracted_cells"]:
            values = table_understanding_result.get(key, []) or []
            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        sub_keys = [
            "table_cell_result",
            "table_span_result",
            "table_header_result",
        ]

        for sub_key in sub_keys:
            sub = table_understanding_result.get(sub_key, {}) or {}
            if not isinstance(sub, dict):
                continue

            for key in ["table_cells", "cells", "table_cells_with_spans", "header_cells"]:
                values = sub.get(key, []) or []
                if isinstance(values, list):
                    items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(items, ["table_cell_id", "cell_id", "grid_cell_id"])

    def _collect_table_grids(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        values = table_understanding_result.get("table_grids", []) or []
        if isinstance(values, list):
            items.extend([self._to_dict(item) for item in values])

        sub = table_understanding_result.get("table_grid_result", {}) or {}
        if isinstance(sub, dict):
            values = sub.get("table_grids", []) or []
            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(items, ["table_grid_id"])

    def _collect_table_structures(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        values = table_understanding_result.get("table_structures", []) or []
        if isinstance(values, list):
            items.extend([self._to_dict(item) for item in values])

        sub = table_understanding_result.get("table_structure_result", {}) or {}
        if isinstance(sub, dict):
            values = sub.get("table_structures", []) or []
            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(items, ["table_structure_id", "table_grid_id"])

    def _collect_table_boundaries(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        values = table_understanding_result.get("table_boundaries", []) or []
        if isinstance(values, list):
            items.extend([self._to_dict(item) for item in values])

        sub = table_understanding_result.get("table_boundary_result", {}) or {}
        if isinstance(sub, dict):
            values = sub.get("table_boundaries", []) or []
            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(items, ["table_boundary_id", "table_grid_id"])

    def _collect_multi_page_tables(
        self,
        table_understanding_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        items = []

        values = table_understanding_result.get("multi_page_tables", []) or []
        if isinstance(values, list):
            items.extend([self._to_dict(item) for item in values])

        sub = table_understanding_result.get("multi_page_table_result", {}) or {}
        if isinstance(sub, dict):
            values = sub.get("multi_page_tables", []) or []
            if isinstance(values, list):
                items.extend([self._to_dict(item) for item in values])

        return self._deduplicate_dicts(items, ["multi_page_table_id"])

    def _collect_sections_by_id(
        self,
        document_structure_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        result = {}

        values = document_structure_result.get("sections", []) or []

        if not isinstance(values, list):
            return result

        for item in values:
            item = self._to_dict(item)
            section_id = item.get("section_id") or item.get("id") or ""

            if section_id:
                result[section_id] = item

        return result

    def _collect_table_metadata(
        self,
        metadata_enrichment_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        table_metadata = metadata_enrichment_result.get("table_metadata", {}) or {}

        if not isinstance(table_metadata, dict):
            return {}

        return {
            str(key): self._to_dict(value)
            for key, value in table_metadata.items()
        }

    def _build_page_text_map(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[int, str]:
        page_text_map: Dict[int, str] = {}

        for page_raw in page_raws:
            reading_meta = page_raw.metadata.get("reading_order_builder", {}) or {}
            reading_text = normalize_text(reading_meta.get("reading_order_text", ""))

            if reading_text:
                text = reading_text
            elif page_raw.normalized_text:
                text = page_raw.normalized_text
            else:
                text = page_raw.raw_text

            page_text_map[page_raw.page_number] = normalize_text(text)

        return page_text_map

    def _group_records_by_table(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for record in records:
            table_ids = [
                record.get("table_semantic_id", ""),
                record.get("table_grid_id", ""),
                record.get("table_structure_id", ""),
            ]

            for table_id in table_ids:
                if not table_id:
                    continue

                grouped.setdefault(table_id, [])
                grouped[table_id].append(record)

        return grouped

    def _group_cells_by_table(
        self,
        cells: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for cell in cells:
            table_ids = [
                cell.get("table_semantic_id", ""),
                cell.get("table_grid_id", ""),
                cell.get("table_structure_id", ""),
            ]

            for table_id in table_ids:
                if not table_id:
                    continue

                grouped.setdefault(table_id, [])
                grouped[table_id].append(cell)

        return grouped

    def _index_tables(
        self,
        table_semantics: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        return self._index_by_ids(
            table_semantics,
            [
                "table_semantic_id",
                "table_grid_id",
                "table_structure_id",
                "table_boundary_id",
            ],
        )

    def _index_by_ids(
        self,
        items: List[Dict[str, Any]],
        id_keys: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        indexed: Dict[str, Dict[str, Any]] = {}

        for item in items:
            for key in id_keys:
                value = item.get(key, "")

                if value:
                    indexed[str(value)] = item

        return indexed

    def _group_chunks_by_page(
        self,
        chunks: List[Chunk],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in chunks:
            for page_number in chunk.page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(chunk.to_dict())

        return grouped

    def _group_chunks_by_table(
        self,
        chunks: List[Chunk],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in chunks:
            table_ids = [
                chunk.table_semantic_id,
                chunk.table_grid_id,
                chunk.table_structure_id,
                chunk.table_boundary_id,
                chunk.metadata.get("multi_page_table_id", "") if isinstance(chunk.metadata, dict) else "",
                chunk.metadata.get("table_id", "") if isinstance(chunk.metadata, dict) else "",
            ]

            for table_id in table_ids:
                if not table_id:
                    continue

                grouped.setdefault(str(table_id), [])
                grouped[str(table_id)].append(chunk.to_dict())

        return grouped

    def _group_chunks_by_type(
        self,
        chunks: List[Chunk],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in chunks:
            grouped.setdefault(chunk.chunk_type, [])
            grouped[chunk.chunk_type].append(chunk.to_dict())

        return grouped

    def _sort_chunks(
        self,
        chunks: List[Chunk],
    ) -> List[Chunk]:
        chunks = sorted(
            chunks,
            key=lambda chunk: (
                min(chunk.page_numbers) if chunk.page_numbers else 999999,
                self._chunk_type_order(chunk.chunk_type),
                chunk.table_semantic_id or chunk.table_grid_id or "",
                chunk.order,
                chunk.chunk_id,
            ),
        )

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
                normalize_text_for_match(chunk.text)[:600],
                tuple(chunk.page_numbers),
                chunk.table_semantic_id,
                chunk.table_grid_id,
                chunk.metadata.get("table_record_id", "") if isinstance(chunk.metadata, dict) else "",
                chunk.metadata.get("row_index", "") if isinstance(chunk.metadata, dict) else "",
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(chunk)

        return result

    def _deduplicate_dicts(
        self,
        items: List[Dict[str, Any]],
        keys: List[str],
    ) -> List[Dict[str, Any]]:
        seen = set()
        result = []

        for item in items:
            key = ""

            for key_name in keys:
                value = item.get(key_name)
                if value:
                    key = str(value)
                    break

            if not key:
                key = str(
                    (
                        normalize_text_for_match(item.get("text", ""))[:300],
                        tuple(self._resolve_page_numbers(item)),
                        item.get("row_index", ""),
                        item.get("col_index", ""),
                    )
                )

            if key in seen:
                continue

            seen.add(key)
            result.append(item)

        return result

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        table_chunks: List[Chunk],
    ) -> None:
        chunks_by_page = self._group_chunks_by_page(table_chunks)

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("table_chunk_builder", {})
            page_raw.metadata["table_chunk_builder"] = {
                "processor": "TableChunkBuilder",
                "table_chunks_on_page": chunks_by_page.get(page_key, []),
                "table_chunk_count_on_page": len(chunks_by_page.get(page_key, [])),
            }

    def _build_summary(
        self,
        table_chunks: List[Chunk],
        table_semantics: List[Dict[str, Any]],
        table_records: List[Dict[str, Any]],
        table_cells: List[Dict[str, Any]],
        multi_page_tables: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        by_page: Dict[str, int] = {}
        by_table: Dict[str, int] = {}

        total_chars = 0
        total_words = 0

        for chunk in table_chunks:
            by_type[chunk.chunk_type] = by_type.get(chunk.chunk_type, 0) + 1
            total_chars += chunk.char_count
            total_words += chunk.word_count

            for page_number in chunk.page_numbers:
                page_key = str(page_number)
                by_page[page_key] = by_page.get(page_key, 0) + 1

            table_id = (
                chunk.table_semantic_id
                or chunk.table_grid_id
                or chunk.table_structure_id
                or chunk.table_boundary_id
                or chunk.metadata.get("multi_page_table_id", "")
                or "unknown_table"
            )

            by_table[table_id] = by_table.get(table_id, 0) + 1

        return {
            "has_table_chunks": len(table_chunks) > 0,
            "table_chunk_count": len(table_chunks),
            "table_semantic_count": len(table_semantics),
            "table_record_count": len(table_records),
            "table_cell_count": len(table_cells),
            "multi_page_table_count": len(multi_page_tables),
            "page_count_with_table_chunks": len(by_page),
            "table_count_with_chunks": len(by_table),
            "total_chars": total_chars,
            "total_words": total_words,
            "avg_chars_per_chunk": round(total_chars / max(len(table_chunks), 1), 2),
            "avg_words_per_chunk": round(total_words / max(len(table_chunks), 1), 2),
            "by_chunk_type": by_type,
            "by_page": by_page,
            "by_table": by_table,
        }

    def _table_id(
        self,
        table: Dict[str, Any],
    ) -> str:
        return (
            table.get("table_semantic_id")
            or table.get("table_grid_id")
            or table.get("table_structure_id")
            or table.get("table_boundary_id")
            or table.get("multi_page_table_id")
            or ""
        )

    def _chunk_type_order(
        self,
        chunk_type: str,
    ) -> int:
        order_map = {
            "table_summary_chunk": 10,
            "multi_page_table_chunk": 15,
            "table_record_chunk": 20,
            "table_row_chunk": 30,
            "table_cell_context_chunk": 40,
        }

        return order_map.get(chunk_type, 99)

    def _is_header_row(
        self,
        row_index: int,
        cells: List[Dict[str, Any]],
        table: Dict[str, Any],
    ) -> bool:
        header_rows = table.get("header_rows", []) or table.get("header_row_indices", []) or []

        if row_index in header_rows:
            return True

        if row_index == 0:
            header_votes = sum(
                1 for cell in cells
                if cell.get("is_header") or cell.get("role") in ["header", "column_header"]
            )

            return header_votes >= max(1, len(cells) // 2)

        return False

    def _average_confidence(
        self,
        items: List[Dict[str, Any]],
        default: float = 0.70,
    ) -> float:
        values = []

        for item in items:
            if item.get("confidence") is None:
                continue

            values.append(self._safe_float(item.get("confidence"), default=default))

        if not values:
            return default

        return round(sum(values) / len(values), 4)

    def _valid_chunk_text(
        self,
        text: str,
    ) -> bool:
        text = normalize_text(text)

        if not text:
            return False

        if len(text) < self.config.min_chunk_chars:
            word_count = len(re.findall(r"\w+", text))

            if word_count < 3:
                return False

        return True

    def _truncate_text(
        self,
        text: str,
        max_chars: int,
    ) -> str:
        text = normalize_text(text)

        if len(text) <= max_chars:
            return text

        cut = text[:max_chars]
        break_point = max(
            cut.rfind("\n"),
            cut.rfind(". "),
            cut.rfind("; "),
            cut.rfind(" | "),
            cut.rfind(" "),
        )

        if break_point > max_chars * 0.60:
            cut = cut[:break_point]

        return normalize_text(cut) + "..."

    def _merge_bboxes(
        self,
        bboxes: List[List[float]],
    ) -> List[float]:
        valid = []

        for bbox in bboxes:
            if not bbox or len(bbox) != 4:
                continue

            try:
                valid.append(
                    [
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ]
                )
            except Exception:
                continue

        if not valid:
            return []

        return [
            min(bbox[0] for bbox in valid),
            min(bbox[1] for bbox in valid),
            max(bbox[2] for bbox in valid),
            max(bbox[3] for bbox in valid),
        ]

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

    def _infer_document_id(
        self,
        page_raws: List[PageRaw],
        table_chunks: List[Chunk],
        table_understanding_result: Dict[str, Any],
    ) -> str:
        if table_understanding_result.get("document_id"):
            return table_understanding_result.get("document_id", "")

        for chunk in table_chunks:
            if chunk.document_id:
                return chunk.document_id

        for page_raw in page_raws:
            if page_raw.document_id:
                return page_raw.document_id

        return ""

    def _infer_source_document(
        self,
        page_raws: List[PageRaw],
        table_chunks: List[Chunk],
    ) -> str:
        for chunk in table_chunks:
            if chunk.source_document:
                return chunk.source_document

        for page_raw in page_raws:
            if page_raw.source_document:
                return page_raw.source_document

        return ""

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


def build_table_chunks(
    page_raws: Optional[List[PageRaw]] = None,
    table_understanding_result: Optional[Dict[str, Any]] = None,
    document_structure_result: Optional[Dict[str, Any]] = None,
    chunk_result: Optional[Dict[str, Any]] = None,
    metadata_enrichment_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    builder = TableChunkBuilder()
    return builder.process(
        page_raws=page_raws,
        table_understanding_result=table_understanding_result,
        document_structure_result=document_structure_result,
        chunk_result=chunk_result,
        metadata_enrichment_result=metadata_enrichment_result,
    )
