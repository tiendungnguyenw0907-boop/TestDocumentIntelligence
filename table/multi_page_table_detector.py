"""
multi_page_table_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect and link tables that continue across multiple pages.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- TableBoundaryDetector
- TableGridBuilder
- TableStructureRecognizer
- TableCellExtractor
- TableHeaderDetector
- TableSpanDetector
- TableSemanticRecognizer

Output
------
Dictionary with:
- multi_page_tables
- multi_page_table_segments
- multi_page_tables_by_page
- multi_page_table_summary

Flow
----
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

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class MultiPageTableDetectorConfig:
    max_page_gap: int = 1

    min_link_confidence: float = 0.45
    min_header_similarity: float = 0.40
    min_column_count_similarity: float = 0.70
    min_bbox_width_similarity: float = 0.60
    min_horizontal_overlap_ratio: float = 0.50

    use_semantic_tables: bool = True
    use_structure_tables: bool = True
    use_grid_tables: bool = True

    allow_same_header_continuation: bool = True
    allow_no_header_continuation: bool = True
    allow_caption_continuation_hint: bool = True

    attach_to_pages: bool = True
    include_debug: bool = True


@dataclass
class MultiPageTableSegment:
    segment_id: str
    multi_page_table_id: str

    table_grid_id: str
    table_structure_id: str
    table_boundary_id: str
    table_semantic_id: str

    page_number: int
    page_index: int
    segment_index: int

    bbox: List[float]
    row_count: int
    col_count: int

    header_rows: List[int]
    column_headers: List[str]

    segment_type: str = "middle"
    continuation_score: float = 0.5
    source: str = "multi_page_table_detector"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class MultiPageTable:
    multi_page_table_id: str

    page_start: int
    page_end: int
    page_numbers: List[int]

    segment_ids: List[str]
    table_grid_ids: List[str]
    table_structure_ids: List[str]
    table_semantic_ids: List[str]

    total_row_count: int
    col_count: int
    column_headers: List[str]

    table_type: str = "multi_page_table"
    semantic_type: str = "unknown_table"
    confidence: float = 0.5
    source: str = "multi_page_table_detector"
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class MultiPageTableDetector:
    def __init__(
        self,
        config: Optional[MultiPageTableDetectorConfig] = None,
    ):
        self.config = config or MultiPageTableDetectorConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        table_semantic_result: Optional[Dict[str, Any]] = None,
        table_structure_result: Optional[Dict[str, Any]] = None,
        table_grid_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        table_candidates = self._collect_table_candidates(
            page_raws=page_raws,
            table_semantic_result=table_semantic_result,
            table_structure_result=table_structure_result,
            table_grid_result=table_grid_result,
        )

        table_candidates = self._sort_table_candidates(table_candidates)

        chains = self._build_table_chains(table_candidates)

        multi_page_tables: List[MultiPageTable] = []
        all_segments: List[MultiPageTableSegment] = []

        for chain in chains:
            if len(chain) < 2:
                continue

            multi_page_table, segments = self._build_multi_page_table(chain)

            if multi_page_table:
                multi_page_tables.append(multi_page_table)
                all_segments.extend(segments)

        multi_page_tables = self._sort_multi_page_tables(multi_page_tables)
        all_segments = self._sort_segments(all_segments)

        result = {
            "processor": "MultiPageTableDetector",
            "multi_page_tables": [
                item.to_dict() for item in multi_page_tables
            ],
            "multi_page_table_segments": [
                item.to_dict() for item in all_segments
            ],
            "multi_page_tables_by_page": self._group_tables_by_page(
                multi_page_tables=multi_page_tables,
                segments=all_segments,
            ),
            "multi_page_segments_by_page": self._group_segments_by_page(all_segments),
            "multi_page_table_summary": self._build_summary(
                multi_page_tables=multi_page_tables,
                segments=all_segments,
            ),
            "config": {
                "max_page_gap": self.config.max_page_gap,
                "min_link_confidence": self.config.min_link_confidence,
                "min_header_similarity": self.config.min_header_similarity,
                "min_column_count_similarity": self.config.min_column_count_similarity,
                "min_bbox_width_similarity": self.config.min_bbox_width_similarity,
                "min_horizontal_overlap_ratio": self.config.min_horizontal_overlap_ratio,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                multi_page_tables=multi_page_tables,
                segments=all_segments,
                result=result,
            )

        return result

    def _collect_table_candidates(
        self,
        page_raws: List[PageRaw],
        table_semantic_result: Optional[Dict[str, Any]] = None,
        table_structure_result: Optional[Dict[str, Any]] = None,
        table_grid_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        semantic_tables = self._collect_semantic_tables(
            page_raws=page_raws,
            table_semantic_result=table_semantic_result,
        )

        structure_tables = self._collect_structure_tables(
            page_raws=page_raws,
            table_structure_result=table_structure_result,
        )

        grid_tables = self._collect_grid_tables(
            page_raws=page_raws,
            table_grid_result=table_grid_result,
        )

        structure_by_grid_id = {
            item.get("table_grid_id", ""): item
            for item in structure_tables
            if item.get("table_grid_id")
        }

        grid_by_id = {
            item.get("table_grid_id", ""): item
            for item in grid_tables
            if item.get("table_grid_id")
        }

        used_grid_ids = set()

        if self.config.use_semantic_tables:
            for semantic in semantic_tables:
                table_grid_id = semantic.get("table_grid_id", "")
                structure = structure_by_grid_id.get(table_grid_id, {})
                grid = grid_by_id.get(table_grid_id, {})

                candidates.append(
                    self._normalize_candidate(
                        source_table=semantic,
                        structure=structure,
                        grid=grid,
                        source="table_semantic_recognizer",
                    )
                )

                if table_grid_id:
                    used_grid_ids.add(table_grid_id)

        if self.config.use_structure_tables:
            for structure in structure_tables:
                table_grid_id = structure.get("table_grid_id", "")

                if table_grid_id in used_grid_ids:
                    continue

                grid = grid_by_id.get(table_grid_id, {})

                candidates.append(
                    self._normalize_candidate(
                        source_table={},
                        structure=structure,
                        grid=grid,
                        source="table_structure_recognizer",
                    )
                )

                if table_grid_id:
                    used_grid_ids.add(table_grid_id)

        if self.config.use_grid_tables:
            for grid in grid_tables:
                table_grid_id = grid.get("table_grid_id", "")

                if table_grid_id in used_grid_ids:
                    continue

                candidates.append(
                    self._normalize_candidate(
                        source_table={},
                        structure={},
                        grid=grid,
                        source="table_grid_builder",
                    )
                )

                if table_grid_id:
                    used_grid_ids.add(table_grid_id)

        candidates = [
            item for item in candidates
            if item.get("page_number") is not None
            and item.get("bbox")
        ]

        return candidates

    def _normalize_candidate(
        self,
        source_table: Dict[str, Any],
        structure: Dict[str, Any],
        grid: Dict[str, Any],
        source: str,
    ) -> Dict[str, Any]:
        table_grid_id = (
            source_table.get("table_grid_id")
            or structure.get("table_grid_id")
            or grid.get("table_grid_id")
            or ""
        )

        bbox = (
            source_table.get("bbox")
            or structure.get("bbox")
            or grid.get("bbox")
            or []
        )

        page_number = self._safe_int(
            source_table.get("page_number")
            or structure.get("page_number")
            or grid.get("page_number"),
            default=0,
        )

        page_index = self._safe_int(
            source_table.get("page_index")
            or structure.get("page_index")
            or grid.get("page_index"),
            default=max(page_number - 1, 0),
        )

        row_count = self._safe_int(
            source_table.get("row_count")
            or structure.get("row_count")
            or grid.get("row_count"),
            default=0,
        )

        col_count = self._safe_int(
            source_table.get("col_count")
            or structure.get("col_count")
            or grid.get("col_count"),
            default=0,
        )

        column_headers = (
            source_table.get("column_headers")
            or structure.get("column_headers")
            or []
        )

        if not column_headers:
            metadata = source_table.get("metadata", {}) or {}
            column_headers = metadata.get("column_headers", []) or []

        header_rows = (
            source_table.get("header_rows")
            or source_table.get("header_row_indices")
            or structure.get("header_row_indices")
            or []
        )

        caption = (
            source_table.get("caption")
            or source_table.get("title")
            or ""
        )

        table_type = (
            source_table.get("table_type")
            or structure.get("table_type")
            or "table"
        )

        semantic_type = (
            source_table.get("semantic_type")
            or "unknown_table"
        )

        confidence = self._safe_float(
            source_table.get("confidence")
            or structure.get("confidence")
            or grid.get("confidence")
            or 0.5,
            default=0.5,
        )

        return {
            "table_grid_id": table_grid_id,
            "table_structure_id": (
                source_table.get("table_structure_id")
                or structure.get("table_structure_id")
                or ""
            ),
            "table_boundary_id": (
                source_table.get("table_boundary_id")
                or structure.get("table_boundary_id")
                or grid.get("table_boundary_id")
                or ""
            ),
            "table_semantic_id": source_table.get("table_semantic_id", ""),
            "page_number": page_number,
            "page_index": page_index,
            "bbox": self._normalize_bbox(bbox),
            "row_count": row_count,
            "col_count": col_count,
            "header_rows": [
                self._safe_int(item, default=-1)
                for item in header_rows
                if self._safe_int(item, default=-1) >= 0
            ],
            "column_headers": [
                self._clean_text(item)
                for item in column_headers
            ],
            "caption": self._clean_text(caption),
            "table_type": table_type,
            "semantic_type": semantic_type,
            "confidence": confidence,
            "source": source,
            "metadata": {
                "source_table": source_table,
                "structure": structure,
                "grid": grid,
            },
        }

    def _build_table_chains(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        chains: List[List[Dict[str, Any]]] = []
        used = set()

        for index, candidate in enumerate(candidates):
            candidate_key = self._candidate_key(candidate)

            if candidate_key in used:
                continue

            chain = [candidate]
            used.add(candidate_key)

            current = candidate

            while True:
                next_candidate, score, details = self._find_best_next_candidate(
                    current=current,
                    candidates=candidates,
                    used=used,
                )

                if not next_candidate:
                    break

                if score < self.config.min_link_confidence:
                    break

                next_key = self._candidate_key(next_candidate)
                next_candidate = dict(next_candidate)
                next_candidate.setdefault("metadata", {})
                next_candidate["metadata"]["multi_page_link_score"] = score
                next_candidate["metadata"]["multi_page_link_details"] = details

                chain.append(next_candidate)
                used.add(next_key)
                current = next_candidate

            chains.append(chain)

        return chains

    def _find_best_next_candidate(
        self,
        current: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        used: set,
    ) -> Tuple[Optional[Dict[str, Any]], float, Dict[str, Any]]:
        best_candidate = None
        best_score = 0.0
        best_details: Dict[str, Any] = {}

        current_page = self._safe_int(current.get("page_number"), default=0)

        for candidate in candidates:
            candidate_key = self._candidate_key(candidate)

            if candidate_key in used:
                continue

            candidate_page = self._safe_int(candidate.get("page_number"), default=0)
            page_gap = candidate_page - current_page

            if page_gap <= 0:
                continue

            if page_gap > self.config.max_page_gap:
                continue

            score, details = self._score_table_continuation(
                current=current,
                next_table=candidate,
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_details = details

        return best_candidate, best_score, best_details

    def _score_table_continuation(
        self,
        current: Dict[str, Any],
        next_table: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        score = 0.0

        current_cols = self._safe_int(current.get("col_count"), default=0)
        next_cols = self._safe_int(next_table.get("col_count"), default=0)

        col_similarity = self._column_count_similarity(current_cols, next_cols)

        if col_similarity >= self.config.min_column_count_similarity:
            score += 0.25 * col_similarity

        header_similarity = self._header_similarity(
            current.get("column_headers", []),
            next_table.get("column_headers", []),
        )

        if header_similarity >= self.config.min_header_similarity:
            score += 0.25 * header_similarity

        bbox_width_similarity = self._bbox_width_similarity(
            current.get("bbox", []),
            next_table.get("bbox", []),
        )

        if bbox_width_similarity >= self.config.min_bbox_width_similarity:
            score += 0.15 * bbox_width_similarity

        horizontal_overlap = self._horizontal_overlap_ratio(
            current.get("bbox", []),
            next_table.get("bbox", []),
        )

        if horizontal_overlap >= self.config.min_horizontal_overlap_ratio:
            score += 0.15 * horizontal_overlap

        caption_score = self._caption_continuation_score(
            current.get("caption", ""),
            next_table.get("caption", ""),
        )

        score += caption_score

        no_header_bonus = 0.0

        current_has_header = len(current.get("header_rows", [])) > 0
        next_has_header = len(next_table.get("header_rows", [])) > 0

        if self.config.allow_no_header_continuation and current_has_header and not next_has_header:
            no_header_bonus = 0.10
            score += no_header_bonus

        same_type_bonus = 0.0

        if current.get("semantic_type") and current.get("semantic_type") == next_table.get("semantic_type"):
            same_type_bonus += 0.05

        if current.get("table_type") and current.get("table_type") == next_table.get("table_type"):
            same_type_bonus += 0.05

        score += same_type_bonus

        page_gap = self._safe_int(next_table.get("page_number"), default=0) - self._safe_int(current.get("page_number"), default=0)

        if page_gap == 1:
            score += 0.05

        score = round(max(0.0, min(score, 0.95)), 4)

        details = {
            "column_count_similarity": round(col_similarity, 4),
            "header_similarity": round(header_similarity, 4),
            "bbox_width_similarity": round(bbox_width_similarity, 4),
            "horizontal_overlap_ratio": round(horizontal_overlap, 4),
            "caption_score": round(caption_score, 4),
            "no_header_bonus": round(no_header_bonus, 4),
            "same_type_bonus": round(same_type_bonus, 4),
            "page_gap": page_gap,
        }

        return score, details

    def _build_multi_page_table(
        self,
        chain: List[Dict[str, Any]],
    ) -> Tuple[Optional[MultiPageTable], List[MultiPageTableSegment]]:
        if len(chain) < 2:
            return None, []

        multi_page_table_id = make_id("multi_tbl")

        segments: List[MultiPageTableSegment] = []

        for segment_index, item in enumerate(chain):
            if segment_index == 0:
                segment_type = "start"
                continuation_score = 1.0
            elif segment_index == len(chain) - 1:
                segment_type = "end"
                continuation_score = self._safe_float(
                    item.get("metadata", {}).get("multi_page_link_score", 0.5),
                    default=0.5,
                )
            else:
                segment_type = "middle"
                continuation_score = self._safe_float(
                    item.get("metadata", {}).get("multi_page_link_score", 0.5),
                    default=0.5,
                )

            segments.append(
                MultiPageTableSegment(
                    segment_id=make_id("multi_tbl_seg"),
                    multi_page_table_id=multi_page_table_id,
                    table_grid_id=item.get("table_grid_id", ""),
                    table_structure_id=item.get("table_structure_id", ""),
                    table_boundary_id=item.get("table_boundary_id", ""),
                    table_semantic_id=item.get("table_semantic_id", ""),
                    page_number=self._safe_int(item.get("page_number"), default=0),
                    page_index=self._safe_int(item.get("page_index"), default=0),
                    segment_index=segment_index,
                    bbox=item.get("bbox", []),
                    row_count=self._safe_int(item.get("row_count"), default=0),
                    col_count=self._safe_int(item.get("col_count"), default=0),
                    header_rows=item.get("header_rows", []),
                    column_headers=item.get("column_headers", []),
                    segment_type=segment_type,
                    continuation_score=round(continuation_score, 4),
                    source="multi_page_table_detector",
                    metadata={
                        "source": item.get("source", ""),
                        "caption": item.get("caption", ""),
                        "table_type": item.get("table_type", ""),
                        "semantic_type": item.get("semantic_type", ""),
                        "candidate_confidence": item.get("confidence", 0.5),
                        "link_details": item.get("metadata", {}).get("multi_page_link_details", {}),
                    },
                )
            )

        page_numbers = [
            self._safe_int(item.get("page_number"), default=0)
            for item in chain
        ]

        col_counts = [
            self._safe_int(item.get("col_count"), default=0)
            for item in chain
            if self._safe_int(item.get("col_count"), default=0) > 0
        ]

        col_count = max(set(col_counts), key=col_counts.count) if col_counts else 0

        column_headers = self._select_best_column_headers(chain)

        total_row_count = sum(
            max(0, self._safe_int(item.get("row_count"), default=0))
            for item in chain
        )

        confidence = self._score_multi_page_table(
            chain=chain,
            segments=segments,
        )

        first = chain[0]

        multi_page_table = MultiPageTable(
            multi_page_table_id=multi_page_table_id,
            page_start=min(page_numbers),
            page_end=max(page_numbers),
            page_numbers=sorted(page_numbers),
            segment_ids=[
                segment.segment_id for segment in segments
            ],
            table_grid_ids=[
                item.get("table_grid_id", "") for item in chain
                if item.get("table_grid_id")
            ],
            table_structure_ids=[
                item.get("table_structure_id", "") for item in chain
                if item.get("table_structure_id")
            ],
            table_semantic_ids=[
                item.get("table_semantic_id", "") for item in chain
                if item.get("table_semantic_id")
            ],
            total_row_count=total_row_count,
            col_count=col_count,
            column_headers=column_headers,
            table_type=first.get("table_type", "multi_page_table"),
            semantic_type=first.get("semantic_type", "unknown_table"),
            confidence=confidence,
            source="multi_page_table_detector",
            metadata={
                "segment_count": len(segments),
                "page_span": max(page_numbers) - min(page_numbers) + 1,
                "captions": [
                    item.get("caption", "") for item in chain
                    if item.get("caption")
                ],
                "link_scores": [
                    segment.continuation_score for segment in segments
                ],
            },
        )

        return multi_page_table, segments

    def _score_multi_page_table(
        self,
        chain: List[Dict[str, Any]],
        segments: List[MultiPageTableSegment],
    ) -> float:
        if not chain or not segments:
            return 0.0

        score = 0.45

        if len(chain) >= 2:
            score += 0.15

        link_scores = [
            segment.continuation_score
            for segment in segments[1:]
        ]

        if link_scores:
            score += min(0.20, (sum(link_scores) / len(link_scores)) * 0.20)

        header_similarity_values = []

        for index in range(len(chain) - 1):
            header_similarity_values.append(
                self._header_similarity(
                    chain[index].get("column_headers", []),
                    chain[index + 1].get("column_headers", []),
                )
            )

        if header_similarity_values:
            score += min(0.10, (sum(header_similarity_values) / len(header_similarity_values)) * 0.10)

        confidence_values = [
            self._safe_float(item.get("confidence"), default=0.5)
            for item in chain
        ]

        if confidence_values:
            score += min(0.10, (sum(confidence_values) / len(confidence_values)) * 0.10)

        return round(max(0.0, min(score, 0.95)), 4)

    def _select_best_column_headers(
        self,
        chain: List[Dict[str, Any]],
    ) -> List[str]:
        best_headers: List[str] = []
        best_score = -1

        for item in chain:
            headers = item.get("column_headers", []) or []
            non_empty_count = len(
                [
                    header for header in headers
                    if self._clean_text(header)
                ]
            )

            if non_empty_count > best_score:
                best_score = non_empty_count
                best_headers = headers

        return [
            self._clean_text(header)
            for header in best_headers
        ]

    def _column_count_similarity(
        self,
        a: int,
        b: int,
    ) -> float:
        if a <= 0 or b <= 0:
            return 0.0

        return min(a, b) / max(a, b)

    def _header_similarity(
        self,
        headers_a: List[str],
        headers_b: List[str],
    ) -> float:
        headers_a = [
            self._normalize_match_text(item)
            for item in headers_a
            if self._clean_text(item)
        ]

        headers_b = [
            self._normalize_match_text(item)
            for item in headers_b
            if self._clean_text(item)
        ]

        if not headers_a or not headers_b:
            return 0.0

        tokens_a = set()
        tokens_b = set()

        for header in headers_a:
            tokens_a.update(header.split())

        for header in headers_b:
            tokens_b.update(header.split())

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))

        if union <= 0:
            return 0.0

        return intersection / union

    def _bbox_width_similarity(
        self,
        bbox_a: List[float],
        bbox_b: List[float],
    ) -> float:
        if len(bbox_a) != 4 or len(bbox_b) != 4:
            return 0.0

        width_a = max(float(bbox_a[2]) - float(bbox_a[0]), 0.0)
        width_b = max(float(bbox_b[2]) - float(bbox_b[0]), 0.0)

        if width_a <= 0 or width_b <= 0:
            return 0.0

        return min(width_a, width_b) / max(width_a, width_b)

    def _horizontal_overlap_ratio(
        self,
        bbox_a: List[float],
        bbox_b: List[float],
    ) -> float:
        if len(bbox_a) != 4 or len(bbox_b) != 4:
            return 0.0

        x0 = max(float(bbox_a[0]), float(bbox_b[0]))
        x1 = min(float(bbox_a[2]), float(bbox_b[2]))
        overlap = max(x1 - x0, 0.0)

        width_a = max(float(bbox_a[2]) - float(bbox_a[0]), 0.0)
        width_b = max(float(bbox_b[2]) - float(bbox_b[0]), 0.0)

        smaller = min(width_a, width_b)

        if smaller <= 0:
            return 0.0

        return overlap / smaller

    def _caption_continuation_score(
        self,
        caption_a: str,
        caption_b: str,
    ) -> float:
        if not self.config.allow_caption_continuation_hint:
            return 0.0

        caption_a_norm = self._normalize_match_text(caption_a)
        caption_b_norm = self._normalize_match_text(caption_b)

        continuation_keywords = [
            "tiep",
            "tiep theo",
            "continued",
            "cont",
            "continuation",
            "trang sau",
        ]

        if any(keyword in caption_b_norm for keyword in continuation_keywords):
            return 0.15

        if caption_a_norm and caption_b_norm:
            if caption_a_norm == caption_b_norm:
                return 0.10

            tokens_a = set(caption_a_norm.split())
            tokens_b = set(caption_b_norm.split())

            if tokens_a and tokens_b:
                overlap = len(tokens_a.intersection(tokens_b)) / max(len(tokens_a.union(tokens_b)), 1)

                if overlap >= 0.50:
                    return 0.08

        return 0.0

    def _collect_semantic_tables(
        self,
        page_raws: List[PageRaw],
        table_semantic_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if table_semantic_result:
            tables = table_semantic_result.get("table_semantics", [])

            if tables:
                return tables

        collected: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_semantic_recognizer", {})
            items = meta.get("table_semantics_on_page", [])

            for item in items:
                collected.append(item)

        return collected

    def _collect_structure_tables(
        self,
        page_raws: List[PageRaw],
        table_structure_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if table_structure_result:
            tables = table_structure_result.get("table_structures", [])

            if tables:
                return tables

        collected: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_structure_recognizer", {})
            items = meta.get("table_structures_on_page", [])

            for item in items:
                collected.append(item)

        return collected

    def _collect_grid_tables(
        self,
        page_raws: List[PageRaw],
        table_grid_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if table_grid_result:
            tables = table_grid_result.get("table_grids", [])

            if tables:
                return tables

        collected: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            meta = page_raw.metadata.get("table_grid_builder", {})
            items = meta.get("table_grids_on_page", [])

            for item in items:
                collected.append(item)

        return collected

    def _group_tables_by_page(
        self,
        multi_page_tables: List[MultiPageTable],
        segments: List[MultiPageTableSegment],
    ) -> Dict[str, List[Dict[str, Any]]]:
        table_by_id = {
            table.multi_page_table_id: table.to_dict()
            for table in multi_page_tables
        }

        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for segment in segments:
            page_key = str(segment.page_number)
            grouped.setdefault(page_key, [])

            table = table_by_id.get(segment.multi_page_table_id)

            if table and table not in grouped[page_key]:
                grouped[page_key].append(table)

        return grouped

    def _group_segments_by_page(
        self,
        segments: List[MultiPageTableSegment],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for segment in segments:
            page_key = str(segment.page_number)
            grouped.setdefault(page_key, [])
            grouped[page_key].append(segment.to_dict())

        return grouped

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        multi_page_tables: List[MultiPageTable],
        segments: List[MultiPageTableSegment],
        result: Dict[str, Any],
    ) -> None:
        tables_by_page = self._group_tables_by_page(
            multi_page_tables=multi_page_tables,
            segments=segments,
        )

        segments_by_page = self._group_segments_by_page(segments)

        for page_raw in page_raws:
            page_key = str(page_raw.page_number)

            page_raw.metadata.setdefault("multi_page_table_detector", {})
            page_raw.metadata["multi_page_table_detector"] = {
                "processor": "MultiPageTableDetector",
                "multi_page_tables_on_page": tables_by_page.get(page_key, []),
                "multi_page_table_segments_on_page": segments_by_page.get(page_key, []),
                "multi_page_table_count_on_page": len(tables_by_page.get(page_key, [])),
                "multi_page_table_segment_count_on_page": len(segments_by_page.get(page_key, [])),
                "multi_page_table_summary": result.get("multi_page_table_summary", {}),
            }

    def _build_summary(
        self,
        multi_page_tables: List[MultiPageTable],
        segments: List[MultiPageTableSegment],
    ) -> Dict[str, Any]:
        by_page: Dict[str, int] = {}
        by_span_length: Dict[str, int] = {}

        for segment in segments:
            page_key = str(segment.page_number)
            by_page[page_key] = by_page.get(page_key, 0) + 1

        for table in multi_page_tables:
            span_length = len(table.page_numbers)
            key = str(span_length)
            by_span_length[key] = by_span_length.get(key, 0) + 1

        return {
            "has_multi_page_tables": len(multi_page_tables) > 0,
            "multi_page_table_count": len(multi_page_tables),
            "multi_page_table_segment_count": len(segments),
            "page_count_with_multi_page_tables": len(by_page),
            "by_page": by_page,
            "by_span_length": by_span_length,
        }

    def _sort_table_candidates(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return sorted(
            candidates,
            key=lambda item: (
                self._safe_int(item.get("page_number"), default=0),
                item.get("bbox", [0, 999999, 0, 0])[1]
                if item.get("bbox")
                else 999999,
                item.get("bbox", [999999, 0, 0, 0])[0]
                if item.get("bbox")
                else 999999,
            ),
        )

    def _sort_multi_page_tables(
        self,
        tables: List[MultiPageTable],
    ) -> List[MultiPageTable]:
        return sorted(
            tables,
            key=lambda item: (
                item.page_start,
                item.page_end,
                item.multi_page_table_id,
            ),
        )

    def _sort_segments(
        self,
        segments: List[MultiPageTableSegment],
    ) -> List[MultiPageTableSegment]:
        return sorted(
            segments,
            key=lambda item: (
                item.page_number,
                item.segment_index,
                item.multi_page_table_id,
            ),
        )

    def _candidate_key(
        self,
        item: Dict[str, Any],
    ) -> str:
        if item.get("table_grid_id"):
            return f"grid:{item.get('table_grid_id')}"

        if item.get("table_structure_id"):
            return f"structure:{item.get('table_structure_id')}"

        if item.get("table_semantic_id"):
            return f"semantic:{item.get('table_semantic_id')}"

        bbox = item.get("bbox", [])

        return f"page:{item.get('page_number')}|bbox:{bbox}"

    def _normalize_bbox(
        self,
        bbox: Any,
    ) -> List[float]:
        if not bbox or len(bbox) != 4:
            return []

        return [
            round(float(bbox[0]), 4),
            round(float(bbox[1]), 4),
            round(float(bbox[2]), 4),
            round(float(bbox[3]), 4),
        ]

    def _normalize_match_text(
        self,
        text: Any,
    ) -> str:
        text = self._clean_text(text).lower()

        replacements = {
            "à": "a", "á": "a", "ạ": "a", "ả": "a", "ã": "a",
            "â": "a", "ầ": "a", "ấ": "a", "ậ": "a", "ẩ": "a", "ẫ": "a",
            "ă": "a", "ằ": "a", "ắ": "a", "ặ": "a", "ẳ": "a", "ẵ": "a",
            "è": "e", "é": "e", "ẹ": "e", "ẻ": "e", "ẽ": "e",
            "ê": "e", "ề": "e", "ế": "e", "ệ": "e", "ể": "e", "ễ": "e",
            "ì": "i", "í": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
            "ò": "o", "ó": "o", "ọ": "o", "ỏ": "o", "õ": "o",
            "ô": "o", "ồ": "o", "ố": "o", "ộ": "o", "ổ": "o", "ỗ": "o",
            "ơ": "o", "ờ": "o", "ớ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
            "ù": "u", "ú": "u", "ụ": "u", "ủ": "u", "ũ": "u",
            "ư": "u", "ừ": "u", "ứ": "u", "ự": "u", "ử": "u", "ữ": "u",
            "ỳ": "y", "ý": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y",
            "đ": "d",
        }

        for src, dst in replacements.items():
            text = text.replace(src, dst)

        text = re.sub(r"[^a-z0-9%]+", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

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


def detect_multi_page_tables(
    page_raws: List[PageRaw],
    table_semantic_result: Optional[Dict[str, Any]] = None,
    table_structure_result: Optional[Dict[str, Any]] = None,
    table_grid_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    detector = MultiPageTableDetector()
    return detector.process(
        page_raws=page_raws,
        table_semantic_result=table_semantic_result,
        table_structure_result=table_structure_result,
        table_grid_result=table_grid_result,
    )
