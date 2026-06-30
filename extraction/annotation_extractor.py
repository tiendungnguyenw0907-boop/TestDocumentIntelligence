"""
annotation_extractor.py

Production V1 - Colab Ready

Purpose
-------
Extract PDF annotation objects from a PageRecord and attach them to PageRaw.

Input
-----
- PageRecord from page_iterator.py
- Optional existing PageRaw from previous extractors

Output
------
PageRaw with annotations populated.

Supported page types
--------------------
- PDF page: uses PyMuPDF page.annots()
- Image page: no PDF annotations
- Virtual TXT/DOCX page: no PDF annotations

Important
---------
This module performs physical annotation extraction only.

It does not perform:
- semantic interpretation
- comment summarization
- citation linking
- reference extraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import (
    PageRaw,
    AnnotationRaw,
    make_id,
    normalize_bbox,
    normalize_pdf_text,
)


try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


@dataclass
class AnnotationExtractorConfig:
    """
    Configuration for AnnotationExtractor.
    """

    include_raw_metadata: bool = False
    include_empty_annotations: bool = True
    extract_popup: bool = True


class AnnotationExtractor:
    """
    Extract PDF annotations from one page.
    """

    def __init__(self, config: Optional[AnnotationExtractorConfig] = None):
        self.config = config or AnnotationExtractorConfig()

    def process(
        self,
        page_record: Any,
        page_raw: Optional[PageRaw] = None,
    ) -> PageRaw:
        """
        Extract annotations from PageRecord and return PageRaw.
        """

        if page_raw is None:
            page_raw = self._create_empty_page_raw(page_record)

        page_kind = getattr(page_record, "page_kind", "unknown")

        if page_kind == "pdf_page":
            return self._process_pdf_page(page_record, page_raw)

        if page_kind in {"image_page", "virtual_text_page", "virtual_docx_page"}:
            page_raw.metadata.setdefault("annotation_extractor", {})
            page_raw.metadata["annotation_extractor"] = {
                "extractor": "AnnotationExtractor",
                "page_kind": page_kind,
                "annotation_count_added": 0,
                "note": "This page kind has no PDF annotation layer.",
            }
            return page_raw

        page_raw.warnings.append(
            f"Unsupported page_kind for AnnotationExtractor: {page_kind}"
        )
        return page_raw

    def _process_pdf_page(
        self,
        page_record: Any,
        page_raw: PageRaw,
    ) -> PageRaw:
        page = getattr(page_record, "page_object", None)

        if page is None:
            page_raw.warnings.append(
                "PDF page_object is None in AnnotationExtractor."
            )
            return page_raw

        annotations: List[AnnotationRaw] = []
        warnings: List[str] = []

        try:
            raw_annots = list(page.annots() or [])
        except Exception as exc:
            page_raw.warnings.append(f"Failed to extract annotations: {exc}")
            return page_raw

        for annot_index, annot in enumerate(raw_annots):
            try:
                annot_obj = self._parse_pdf_annotation(
                    annot=annot,
                    page_number=page_record.page_number,
                    annot_index=annot_index,
                )

                if annot_obj is None:
                    continue

                annotations.append(annot_obj)

            except Exception as exc:
                warnings.append(
                    f"Failed to parse annotation {annot_index}: {exc}"
                )

        page_raw.annotations.extend(annotations)

        page_raw.metadata.setdefault("annotation_extractor", {})
        page_raw.metadata["annotation_extractor"] = {
            "extractor": "AnnotationExtractor",
            "page_kind": page_record.page_kind,
            "annotation_count_added": len(annotations),
            "total_annotation_count": len(page_raw.annotations),
            "raw_annotation_count": len(raw_annots),
        }

        page_raw.warnings.extend(warnings)

        return page_raw

    def _parse_pdf_annotation(
        self,
        annot: Any,
        page_number: int,
        annot_index: int,
    ) -> Optional[AnnotationRaw]:
        """
        Parse one PyMuPDF annotation.
        """

        bbox = self._rect_to_bbox(getattr(annot, "rect", None))

        annot_type = self._get_annotation_type(annot)
        info = self._safe_info(getattr(annot, "info", None))

        content = (
            info.get("content")
            or info.get("title")
            or info.get("subject")
            or ""
        )

        if not self.config.include_empty_annotations and not content:
            return None

        metadata: Dict[str, Any] = {
            "source": "pdf_annotation",
            "annotation_index": annot_index,
            "xref": self._safe_metadata_value(getattr(annot, "xref", None)),
            "type": annot_type,
            "info": info,
            "flags": self._safe_metadata_value(getattr(annot, "flags", None)),
            "colors": self._extract_colors(annot),
            "vertices": self._extract_vertices(annot),
        }

        if self.config.extract_popup:
            popup = self._extract_popup(annot)
            if popup:
                metadata["popup"] = popup

        if self.config.include_raw_metadata:
            metadata["raw_annotation"] = self._safe_annotation_raw(annot)

        return AnnotationRaw(
            annotation_id=make_id("annot"),
            page_number=page_number,
            bbox=bbox,
            annot_type=annot_type,
            content=content,
            metadata=metadata,
        )

    def _create_empty_page_raw(self, page_record: Any) -> PageRaw:
        raw_text = getattr(page_record, "text_content", "") or ""
        normalized_text = normalize_pdf_text(raw_text)

        return PageRaw(
            document_id=page_record.document_id,
            source_path=page_record.source_path,
            file_name=page_record.file_name,
            document_type=page_record.document_type,
            page_number=page_record.page_number,
            page_index=page_record.page_index,
            width=page_record.width,
            height=page_record.height,
            rotation=page_record.rotation,
            raw_text=raw_text,
            normalized_text=normalized_text,
            metadata={
                "created_by": "AnnotationExtractor",
                "page_kind": getattr(page_record, "page_kind", "unknown"),
            },
        )

    def _get_annotation_type(
        self,
        annot: Any,
    ) -> Optional[str]:
        try:
            annot_type = annot.type

            if isinstance(annot_type, (list, tuple)):
                if len(annot_type) >= 2:
                    return str(annot_type[1])
                if len(annot_type) == 1:
                    return str(annot_type[0])

            return str(annot_type)

        except Exception:
            return None

    def _safe_info(
        self,
        info: Any,
    ) -> Dict[str, Any]:
        if not info:
            return {}

        if isinstance(info, dict):
            return {
                str(k): self._safe_metadata_value(v)
                for k, v in info.items()
            }

        return {"value": self._safe_metadata_value(info)}

    def _extract_colors(
        self,
        annot: Any,
    ) -> Dict[str, Any]:
        try:
            colors = annot.colors
            return self._safe_metadata_value(colors)
        except Exception:
            return {}

    def _extract_vertices(
        self,
        annot: Any,
    ) -> Optional[Any]:
        try:
            vertices = getattr(annot, "vertices", None)

            if vertices is None:
                return None

            return self._safe_metadata_value(vertices)

        except Exception:
            return None

    def _extract_popup(
        self,
        annot: Any,
    ) -> Optional[Dict[str, Any]]:
        try:
            popup = getattr(annot, "popup_rect", None)

            if popup is None:
                return None

            return {
                "popup_rect": self._rect_to_bbox(popup),
            }

        except Exception:
            return None

    def _safe_annotation_raw(
        self,
        annot: Any,
    ) -> Dict[str, Any]:
        raw: Dict[str, Any] = {}

        for attr in [
            "xref",
            "type",
            "rect",
            "flags",
            "colors",
            "vertices",
            "info",
        ]:
            try:
                value = getattr(annot, attr, None)
                raw[attr] = self._safe_metadata_value(value)
            except Exception as exc:
                raw[attr] = f"<error:{exc}>"

        return raw

    def _rect_to_bbox(
        self,
        rect: Any,
    ) -> Optional[List[float]]:
        if rect is None:
            return None

        try:
            return [
                float(rect.x0),
                float(rect.y0),
                float(rect.x1),
                float(rect.y1),
            ]
        except Exception:
            pass

        try:
            return normalize_bbox(rect)
        except Exception:
            return None

    def _safe_metadata_value(
        self,
        value: Any,
    ) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, bytes):
            return f"<binary:{len(value)} bytes>"

        if isinstance(value, bytearray):
            return f"<binary:{len(value)} bytes>"

        if isinstance(value, list):
            return [self._safe_metadata_value(x) for x in value]

        if isinstance(value, tuple):
            return [self._safe_metadata_value(x) for x in value]

        if isinstance(value, dict):
            return {
                str(k): self._safe_metadata_value(v)
                for k, v in value.items()
            }

        if hasattr(value, "x") and hasattr(value, "y"):
            try:
                return {
                    "x": float(value.x),
                    "y": float(value.y),
                }
            except Exception:
                return str(value)

        if all(hasattr(value, attr) for attr in ["x0", "y0", "x1", "y1"]):
            try:
                return [
                    float(value.x0),
                    float(value.y0),
                    float(value.x1),
                    float(value.y1),
                ]
            except Exception:
                return str(value)

        return str(value)
