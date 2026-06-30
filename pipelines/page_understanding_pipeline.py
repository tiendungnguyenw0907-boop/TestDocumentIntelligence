"""
page_understanding_pipeline.py

Production V1 - Colab Ready

Purpose
-------
Run page-understanding modules after physical extraction.

Input
-----
PageRaw from PageExtractionPipeline.

Output
------
PageRaw with metadata from:
- ObjectNormalizer
- ObjectMerger
- ReadingOrderBuilder
- RegionDetector
- HeaderFooterDetector
- TableBoundaryDetector
- FigureDetector
- CaptionDetector
- PageLayoutProfiler

Flow
----
PageRaw
    ↓
ObjectNormalizer
    ↓
ObjectMerger
    ↓
ReadingOrderBuilder
    ↓
RegionDetector
    ↓
HeaderFooterDetector
    ↓
TableBoundaryDetector
    ↓
FigureDetector
    ↓
CaptionDetector
    ↓
PageLayoutProfiler
    ↓
PageRaw enriched with page_understanding metadata
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from document_ai.schemas.page_raw_schema import PageRaw
from document_ai.pipelines.page_extraction_pipeline import PageExtractionPipeline

from document_ai.page_understanding.object_normalizer import ObjectNormalizer
from document_ai.page_understanding.object_merger import ObjectMerger
from document_ai.page_understanding.reading_order_builder import ReadingOrderBuilder
from document_ai.page_understanding.region_detector import RegionDetector
from document_ai.page_understanding.header_footer_detector import HeaderFooterDetector
from document_ai.page_understanding.table_boundary_detector import TableBoundaryDetector
from document_ai.page_understanding.figure_detector import FigureDetector
from document_ai.page_understanding.caption_detector import CaptionDetector
from document_ai.page_understanding.page_layout_profiler import PageLayoutProfiler


@dataclass
class PageUnderstandingPipelineConfig:
    enable_object_normalizer: bool = True
    enable_object_merger: bool = True
    enable_reading_order_builder: bool = True
    enable_region_detector: bool = True
    enable_header_footer_detector: bool = True
    enable_table_boundary_detector: bool = True
    enable_figure_detector: bool = True
    enable_caption_detector: bool = True
    enable_page_layout_profiler: bool = True

    continue_on_error: bool = True

    save_page_understanding_json: bool = False
    output_dir: str = "outputs/page_understanding"


class PageUnderstandingPipeline:
    def __init__(
        self,
        config: Optional[PageUnderstandingPipelineConfig] = None,
        extraction_pipeline: Optional[PageExtractionPipeline] = None,
        object_normalizer: Optional[ObjectNormalizer] = None,
        object_merger: Optional[ObjectMerger] = None,
        reading_order_builder: Optional[ReadingOrderBuilder] = None,
        region_detector: Optional[RegionDetector] = None,
        header_footer_detector: Optional[HeaderFooterDetector] = None,
        table_boundary_detector: Optional[TableBoundaryDetector] = None,
        figure_detector: Optional[FigureDetector] = None,
        caption_detector: Optional[CaptionDetector] = None,
        page_layout_profiler: Optional[PageLayoutProfiler] = None,
    ):
        self.config = config or PageUnderstandingPipelineConfig()

        self.extraction_pipeline = extraction_pipeline or PageExtractionPipeline()

        self.object_normalizer = object_normalizer or ObjectNormalizer()
        self.object_merger = object_merger or ObjectMerger()
        self.reading_order_builder = reading_order_builder or ReadingOrderBuilder()
        self.region_detector = region_detector or RegionDetector()
        self.header_footer_detector = header_footer_detector or HeaderFooterDetector()
        self.table_boundary_detector = table_boundary_detector or TableBoundaryDetector()
        self.figure_detector = figure_detector or FigureDetector()
        self.caption_detector = caption_detector or CaptionDetector()
        self.page_layout_profiler = page_layout_profiler or PageLayoutProfiler()

    def process(
        self,
        page_raw: PageRaw,
    ) -> PageRaw:
        errors: List[str] = []

        page_raw = self._run_step(
            page_raw=page_raw,
            enabled=self.config.enable_object_normalizer,
            processor_name="ObjectNormalizer",
            processor=self.object_normalizer,
            errors=errors,
        )

        page_raw = self._run_step(
            page_raw=page_raw,
            enabled=self.config.enable_object_merger,
            processor_name="ObjectMerger",
            processor=self.object_merger,
            errors=errors,
        )

        page_raw = self._run_step(
            page_raw=page_raw,
            enabled=self.config.enable_reading_order_builder,
            processor_name="ReadingOrderBuilder",
            processor=self.reading_order_builder,
            errors=errors,
        )

        page_raw = self._run_step(
            page_raw=page_raw,
            enabled=self.config.enable_region_detector,
            processor_name="RegionDetector",
            processor=self.region_detector,
            errors=errors,
        )

        page_raw = self._run_step(
            page_raw=page_raw,
            enabled=self.config.enable_header_footer_detector,
            processor_name="HeaderFooterDetector",
            processor=self.header_footer_detector,
            errors=errors,
        )

        page_raw = self._run_step(
            page_raw=page_raw,
            enabled=self.config.enable_table_boundary_detector,
            processor_name="TableBoundaryDetector",
            processor=self.table_boundary_detector,
            errors=errors,
        )

        page_raw = self._run_step(
            page_raw=page_raw,
            enabled=self.config.enable_figure_detector,
            processor_name="FigureDetector",
            processor=self.figure_detector,
            errors=errors,
        )

        page_raw = self._run_step(
            page_raw=page_raw,
            enabled=self.config.enable_caption_detector,
            processor_name="CaptionDetector",
            processor=self.caption_detector,
            errors=errors,
        )

        page_raw = self._run_step(
            page_raw=page_raw,
            enabled=self.config.enable_page_layout_profiler,
            processor_name="PageLayoutProfiler",
            processor=self.page_layout_profiler,
            errors=errors,
        )

        if errors:
            page_raw.warnings.extend(errors)

        page_raw.metadata.setdefault("page_understanding_pipeline", {})
        page_raw.metadata["page_understanding_pipeline"] = {
            "processor": "PageUnderstandingPipeline",
            "enabled_modules": self._enabled_modules(),
            "completed": True,
            "error_count": len(errors),
            "errors": errors,
            "summary": self._build_page_summary(page_raw),
        }

        if self.config.save_page_understanding_json:
            self.save_page_understanding(page_raw)

        return page_raw

    def process_document(
        self,
        document_path: Union[str, Path],
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> List[PageRaw]:
        results: List[PageRaw] = []

        page_raws = self.extraction_pipeline.process_document(
            document_path=document_path,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
            page_numbers=page_numbers,
        )

        for page_raw in page_raws:
            results.append(self.process(page_raw))

        return results

    def iter_document(
        self,
        document_path: Union[str, Path],
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> Iterator[PageRaw]:
        for page_raw in self.extraction_pipeline.iter_document(
            document_path=document_path,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
            page_numbers=page_numbers,
        ):
            yield self.process(page_raw)

    def process_page_raws(
        self,
        page_raws: List[PageRaw],
    ) -> List[PageRaw]:
        return [self.process(page_raw) for page_raw in page_raws]

    def save_page_understanding(
        self,
        page_raw: PageRaw,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        out_dir = Path(output_dir or self.config.output_dir)
        doc_dir = out_dir / page_raw.document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        file_path = doc_dir / f"page_{page_raw.page_number:05d}_understanding.json"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(page_raw.to_json(ensure_ascii=False, indent=2))

        return str(file_path)

    def save_document_json(
        self,
        page_raws: List[PageRaw],
        output_path: Union[str, Path],
    ) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "schema_version": "page_understanding_document_v1",
            "page_count": len(page_raws),
            "pages": [page_raw.to_dict() for page_raw in page_raws],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return str(output_path)

    def _run_step(
        self,
        page_raw: PageRaw,
        enabled: bool,
        processor_name: str,
        processor: Any,
        errors: List[str],
    ) -> PageRaw:
        if not enabled:
            return page_raw

        try:
            return processor.process(page_raw)
        except Exception as exc:
            message = f"{processor_name} failed: {exc}"

            if not self.config.continue_on_error:
                raise RuntimeError(message) from exc

            errors.append(message)
            return page_raw

    def _enabled_modules(
        self,
    ) -> List[str]:
        modules: List[str] = []

        if self.config.enable_object_normalizer:
            modules.append("ObjectNormalizer")

        if self.config.enable_object_merger:
            modules.append("ObjectMerger")

        if self.config.enable_reading_order_builder:
            modules.append("ReadingOrderBuilder")

        if self.config.enable_region_detector:
            modules.append("RegionDetector")

        if self.config.enable_header_footer_detector:
            modules.append("HeaderFooterDetector")

        if self.config.enable_table_boundary_detector:
            modules.append("TableBoundaryDetector")

        if self.config.enable_figure_detector:
            modules.append("FigureDetector")

        if self.config.enable_caption_detector:
            modules.append("CaptionDetector")

        if self.config.enable_page_layout_profiler:
            modules.append("PageLayoutProfiler")

        return modules

    def _build_page_summary(
        self,
        page_raw: PageRaw,
    ) -> Dict[str, Any]:
        layout_meta = page_raw.metadata.get("page_layout_profiler", {})
        layout_summary = layout_meta.get("layout_summary", {})

        region_meta = page_raw.metadata.get("region_detector", {})
        table_meta = page_raw.metadata.get("table_boundary_detector", {})
        figure_meta = page_raw.metadata.get("figure_detector", {})
        caption_meta = page_raw.metadata.get("caption_detector", {})
        header_footer_meta = page_raw.metadata.get("header_footer_detector", {})

        return {
            "page_number": page_raw.page_number,
            "word_count": len(page_raw.words),
            "text_line_count": len(page_raw.text_lines),
            "image_count": len(page_raw.images),
            "drawing_count": len(page_raw.drawings),
            "region_count": region_meta.get("region_count", 0),
            "table_candidate_count": table_meta.get("table_candidate_count", 0),
            "figure_candidate_count": figure_meta.get("figure_candidate_count", 0),
            "caption_candidate_count": caption_meta.get("caption_candidate_count", 0),
            "header_footer_summary": header_footer_meta.get("summary", {}),
            "layout_summary": layout_summary,
        }


def understand_page(
    page_raw: PageRaw,
) -> PageRaw:
    pipeline = PageUnderstandingPipeline()
    return pipeline.process(page_raw)


def understand_document(
    document_path: Union[str, Path],
    max_pages: Optional[int] = None,
) -> List[PageRaw]:
    pipeline = PageUnderstandingPipeline()
    return pipeline.process_document(
        document_path=document_path,
        max_pages=max_pages,
    )
