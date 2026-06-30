"""
page_extraction_pipeline.py

Production V1 - Colab Ready

Purpose
-------
Run all physical extraction modules for each page.

Input
-----
- PageRecord from page_iterator.py
or
- document_path

Output
------
- PageRaw for one page
- List[PageRaw] for document

Flow
----
PageRecord
    ↓
TextExtractor
    ↓
ImageExtractor
    ↓
DrawingExtractor
    ↓
AnnotationExtractor
    ↓
LinkExtractor
    ↓
FontExtractor
    ↓
PageRaw

Important
---------
This pipeline only performs physical extraction.

It does not perform:
- object normalization
- object merging
- reading order
- region detection
- table boundary detection
- header/footer detection
- figure/caption detection

Those belong to:
- page_understanding_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Union

from document_ai.ingestion.page_iterator import PageIterator, PageIteratorConfig
from document_ai.schemas.page_raw_schema import PageRaw

from document_ai.extraction.text_extractor import (
    TextExtractor,
    TextExtractorConfig,
)
from document_ai.extraction.image_extractor import (
    ImageExtractor,
    ImageExtractorConfig,
)
from document_ai.extraction.drawing_extractor import (
    DrawingExtractor,
    DrawingExtractorConfig,
)
from document_ai.extraction.annotation_extractor import (
    AnnotationExtractor,
    AnnotationExtractorConfig,
)
from document_ai.extraction.link_extractor import (
    LinkExtractor,
    LinkExtractorConfig,
)
from document_ai.extraction.font_extractor import (
    FontExtractor,
    FontExtractorConfig,
)


@dataclass
class PageExtractionPipelineConfig:
    """
    Configuration for PageExtractionPipeline.
    """

    enable_text_extraction: bool = True
    enable_image_extraction: bool = True
    enable_drawing_extraction: bool = True
    enable_annotation_extraction: bool = True
    enable_link_extraction: bool = True
    enable_font_extraction: bool = True

    save_page_raw_json: bool = False
    output_dir: str = "outputs/page_raw"

    continue_on_error: bool = True

    include_text_preview_in_iterator: bool = False
    render_pdf_page_image: bool = False


class PageExtractionPipeline:
    """
    Physical page extraction pipeline.
    """

    def __init__(
        self,
        config: Optional[PageExtractionPipelineConfig] = None,
        page_iterator: Optional[PageIterator] = None,
        text_extractor: Optional[TextExtractor] = None,
        image_extractor: Optional[ImageExtractor] = None,
        drawing_extractor: Optional[DrawingExtractor] = None,
        annotation_extractor: Optional[AnnotationExtractor] = None,
        link_extractor: Optional[LinkExtractor] = None,
        font_extractor: Optional[FontExtractor] = None,
    ):
        self.config = config or PageExtractionPipelineConfig()

        self.page_iterator = page_iterator or PageIterator(
            PageIteratorConfig(
                include_text_preview=self.config.include_text_preview_in_iterator,
                render_pdf_page_image=self.config.render_pdf_page_image,
            )
        )

        self.text_extractor = text_extractor or TextExtractor()
        self.image_extractor = image_extractor or ImageExtractor()
        self.drawing_extractor = drawing_extractor or DrawingExtractor()
        self.annotation_extractor = annotation_extractor or AnnotationExtractor()
        self.link_extractor = link_extractor or LinkExtractor()
        self.font_extractor = font_extractor or FontExtractor()

    def process_page(
        self,
        page_record: Any,
    ) -> PageRaw:
        """
        Process one PageRecord into PageRaw.
        """

        page_raw: Optional[PageRaw] = None
        errors: List[str] = []

        if self.config.enable_text_extraction:
            try:
                page_raw = self.text_extractor.process(page_record)
            except Exception as exc:
                if not self.config.continue_on_error:
                    raise
                errors.append(f"TextExtractor error: {exc}")

        if page_raw is None:
            page_raw = self._create_minimal_page_raw(page_record)

        if self.config.enable_image_extraction:
            try:
                page_raw = self.image_extractor.process(page_record, page_raw)
            except Exception as exc:
                if not self.config.continue_on_error:
                    raise
                errors.append(f"ImageExtractor error: {exc}")

        if self.config.enable_drawing_extraction:
            try:
                page_raw = self.drawing_extractor.process(page_record, page_raw)
            except Exception as exc:
                if not self.config.continue_on_error:
                    raise
                errors.append(f"DrawingExtractor error: {exc}")

        if self.config.enable_annotation_extraction:
            try:
                page_raw = self.annotation_extractor.process(page_record, page_raw)
            except Exception as exc:
                if not self.config.continue_on_error:
                    raise
                errors.append(f"AnnotationExtractor error: {exc}")

        if self.config.enable_link_extraction:
            try:
                page_raw = self.link_extractor.process(page_record, page_raw)
            except Exception as exc:
                if not self.config.continue_on_error:
                    raise
                errors.append(f"LinkExtractor error: {exc}")

        if self.config.enable_font_extraction:
            try:
                page_raw = self.font_extractor.process(page_record, page_raw)
            except Exception as exc:
                if not self.config.continue_on_error:
                    raise
                errors.append(f"FontExtractor error: {exc}")

        if errors:
            page_raw.warnings.extend(errors)

        page_raw.metadata.setdefault("page_extraction_pipeline", {})
        page_raw.metadata["page_extraction_pipeline"] = {
            "pipeline": "PageExtractionPipeline",
            "schema_version": page_raw.schema_version,
            "enabled_extractors": self._enabled_extractors(),
            "text_block_count": len(page_raw.text_blocks),
            "text_line_count": len(page_raw.text_lines),
            "text_span_count": len(page_raw.text_spans),
            "word_count": len(page_raw.words),
            "image_count": len(page_raw.images),
            "drawing_count": len(page_raw.drawings),
            "annotation_count": len(page_raw.annotations),
            "link_count": len(page_raw.links),
            "font_count": len(page_raw.fonts),
        }

        if self.config.save_page_raw_json:
            self.save_page_raw(page_raw)

        return page_raw

    def process_document(
        self,
        document_path: Union[str, Path],
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> List[PageRaw]:
        """
        Process a document and return list of PageRaw.

        Important
        ---------
        PDF page objects are valid only inside iterator loop.
        Therefore all extractors are executed inside this method.
        """

        results: List[PageRaw] = []

        for page_record in self.page_iterator.iter_pages(
            document_path=document_path,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
            page_numbers=page_numbers,
        ):
            page_raw = self.process_page(page_record)
            results.append(page_raw)

        return results

    def iter_document(
        self,
        document_path: Union[str, Path],
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> Iterator[PageRaw]:
        """
        Stream PageRaw one by one.

        This is better for large PDFs.
        """

        for page_record in self.page_iterator.iter_pages(
            document_path=document_path,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
            page_numbers=page_numbers,
        ):
            yield self.process_page(page_record)

    def process_batches(
        self,
        document_path: Union[str, Path],
        batch_size: int = 10,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> Iterator[List[PageRaw]]:
        """
        Process document in batches.
        """

        batch: List[PageRaw] = []

        for page_raw in self.iter_document(
            document_path=document_path,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
            page_numbers=page_numbers,
        ):
            batch.append(page_raw)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def save_page_raw(
        self,
        page_raw: PageRaw,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Save one PageRaw to JSON file.
        """

        out_dir = Path(output_dir or self.config.output_dir)
        doc_dir = out_dir / page_raw.document_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        file_path = doc_dir / f"page_{page_raw.page_number:05d}_raw.json"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(page_raw.to_json(ensure_ascii=False, indent=2))

        return str(file_path)

    def save_document_page_raws(
        self,
        page_raws: List[PageRaw],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        """
        Save multiple PageRaw objects.
        """

        paths = []

        for page_raw in page_raws:
            paths.append(
                self.save_page_raw(
                    page_raw=page_raw,
                    output_dir=output_dir,
                )
            )

        return paths

    def to_document_json(
        self,
        page_raws: List[PageRaw],
        ensure_ascii: bool = False,
        indent: int = 2,
    ) -> str:
        """
        Convert list of PageRaw objects to JSON.
        """

        data = {
            "schema_version": "document_page_raw_v1",
            "page_count": len(page_raws),
            "pages": [p.to_dict() for p in page_raws],
        }

        return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)

    def save_document_json(
        self,
        page_raws: List[PageRaw],
        output_path: Union[str, Path],
    ) -> str:
        """
        Save all PageRaw results into one JSON file.
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                self.to_document_json(
                    page_raws=page_raws,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        return str(output_path)

    def _enabled_extractors(self) -> List[str]:
        enabled = []

        if self.config.enable_text_extraction:
            enabled.append("TextExtractor")

        if self.config.enable_image_extraction:
            enabled.append("ImageExtractor")

        if self.config.enable_drawing_extraction:
            enabled.append("DrawingExtractor")

        if self.config.enable_annotation_extraction:
            enabled.append("AnnotationExtractor")

        if self.config.enable_link_extraction:
            enabled.append("LinkExtractor")

        if self.config.enable_font_extraction:
            enabled.append("FontExtractor")

        return enabled

    def _create_minimal_page_raw(
        self,
        page_record: Any,
    ) -> PageRaw:
        """
        Fallback PageRaw if TextExtractor fails or is disabled.
        """

        raw_text = getattr(page_record, "text_content", "") or ""

        return PageRaw(
	        document_id=getattr(page_record, "document_id", ""),
	        source_document=getattr(page_record, "source_path", "")
	        or getattr(page_record, "source_document", "")
	        or getattr(page_record, "file_name", ""),
	        page_number=getattr(page_record, "page_number", 1),
	        page_index=getattr(page_record, "page_index", 0),
	        width=getattr(page_record, "width", 0.0),
	        height=getattr(page_record, "height", 0.0),
	        rotation=getattr(page_record, "rotation", 0),
	        raw_text=raw_text,
	        normalized_text=raw_text,
	        page_kind=getattr(page_record, "page_kind", "pdf_page"),
	        extraction_method="page_extraction_pipeline",
	        extraction_status="partial",
	        metadata={
	            "created_by": "PageExtractionPipeline",
	            "minimal_page_raw": True,
	            "source_path": getattr(page_record, "source_path", ""),
	            "file_name": getattr(page_record, "file_name", ""),
	            "document_type": getattr(page_record, "document_type", ""),
	            "page_kind": getattr(page_record, "page_kind", "unknown"),
	        },
	        warnings=[
	            "Created minimal PageRaw because TextExtractor failed or was disabled."
	        ],
	    )


def extract_page_raws(
    document_path: Union[str, Path],
    max_pages: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Colab helper function.

    Returns JSON-safe list of PageRaw dictionaries.
    """

    pipeline = PageExtractionPipeline()
    page_raws = pipeline.process_document(
        document_path=document_path,
        max_pages=max_pages,
    )

    return [p.to_dict() for p in page_raws]
