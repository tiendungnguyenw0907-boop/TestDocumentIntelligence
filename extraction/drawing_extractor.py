"""
drawing_extractor.py

Production V1 - Colab Ready

Purpose
-------
Extract vector drawing objects from a PageRecord and attach them to PageRaw.

Input
-----
- PageRecord from page_iterator.py
- Optional existing PageRaw from previous extractors

Output
------
PageRaw with drawings populated.

Supported page types
--------------------
- PDF page: uses PyMuPDF page.get_drawings()
- Image page: no vector drawings
- Virtual TXT/DOCX page: no vector drawings

Important
---------
This module performs physical drawing extraction only.

It does not perform:
- table boundary detection
- figure detection
- chart detection
- layout understanding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import (
    PageRaw,
    DrawingRaw,
    make_id,
    normalize_bbox,
    normalize_pdf_text,
)


try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


@dataclass
class DrawingExtractorConfig:
    """
    Configuration for DrawingExtractor.
    """

    include_items: bool = True
    include_raw_metadata: bool = False

    min_width: Optional[float] = None
    min_height: Optional[float] = None

    # Keep very thin lines because they are useful for table detection.
    keep_thin_lines: bool = True


class DrawingExtractor:
    """
    Extract physical vector drawings from one page.
    """

    def __init__(self, config: Optional[DrawingExtractorConfig] = None):
        self.config = config or DrawingExtractorConfig()

    def process(
        self,
        page_record: Any,
        page_raw: Optional[PageRaw] = None,
    ) -> PageRaw:
        """
        Extract drawings from PageRecord and return PageRaw.
        """

        if page_raw is None:
            page_raw = self._create_empty_page_raw(page_record)

        page_kind = getattr(page_record, "page_kind", "unknown")

        if page_kind == "pdf_page":
            return self._process_pdf_page(page_record, page_raw)

        if page_kind in {"image_page", "virtual_text_page", "virtual_docx_page"}:
            page_raw.metadata.setdefault("drawing_extractor", {})
            page_raw.metadata["drawing_extractor"] = {
                "extractor": "DrawingExtractor",
                "page_kind": page_kind,
                "drawing_count_added": 0,
                "note": "This page kind has no vector drawing layer.",
            }
            return page_raw

        page_raw.warnings.append(
            f"Unsupported page_kind for DrawingExtractor: {page_kind}"
        )
        return page_raw

    def _process_pdf_page(
        self,
        page_record: Any,
        page_raw: PageRaw,
    ) -> PageRaw:
        page = getattr(page_record, "page_object", None)

        if page is None:
            page_raw.warnings.append("PDF page_object is None in DrawingExtractor.")
            return page_raw

        warnings: List[str] = []
        drawings: List[DrawingRaw] = []

        try:
            raw_drawings = page.get_drawings() or []
        except Exception as exc:
            page_raw.warnings.append(f"Failed to extract drawings: {exc}")
            return page_raw

        for drawing_index, drawing in enumerate(raw_drawings):
            try:
                drawing_obj = self._parse_pdf_drawing(
                    drawing=drawing,
                    page_number=page_record.page_number,
                    drawing_index=drawing_index,
                )

                if drawing_obj is None:
                    continue

                drawings.append(drawing_obj)

            except Exception as exc:
                warnings.append(
                    f"Failed to parse drawing {drawing_index}: {exc}"
                )

        page_raw.drawings.extend(drawings)

        page_raw.metadata.setdefault("drawing_extractor", {})
        page_raw.metadata["drawing_extractor"] = {
            "extractor": "DrawingExtractor",
            "page_kind": page_record.page_kind,
            "drawing_count_added": len(drawings),
            "total_drawing_count": len(page_raw.drawings),
            "raw_drawing_count": len(raw_drawings),
        }

        page_raw.warnings.extend(warnings)

        return page_raw

    def _parse_pdf_drawing(
        self,
        drawing: Dict[str, Any],
        page_number: int,
        drawing_index: int,
    ) -> Optional[DrawingRaw]:
        bbox = self._extract_drawing_bbox(drawing)

        if not self._passes_size_filter(bbox):
            return None

        drawing_type = drawing.get("type")

        stroke = self._safe_metadata_value(drawing.get("color"))
        fill = self._safe_metadata_value(drawing.get("fill"))
        width = self._safe_float(drawing.get("width"))

        items = []
        if self.config.include_items:
            items = self._normalize_drawing_items(drawing.get("items", []))

        metadata: Dict[str, Any] = {
            "source": "pdf_drawing",
            "drawing_index": drawing_index,
            "line_cap": self._safe_metadata_value(drawing.get("lineCap")),
            "line_join": self._safe_metadata_value(drawing.get("lineJoin")),
            "dashes": self._safe_metadata_value(drawing.get("dashes")),
            "close_path": drawing.get("closePath"),
            "even_odd": drawing.get("even_odd"),
            "fill_opacity": drawing.get("fill_opacity"),
            "stroke_opacity": drawing.get("stroke_opacity"),
            "is_horizontal_line": self._is_horizontal_line(bbox),
            "is_vertical_line": self._is_vertical_line(bbox),
            "is_thin_line": self._is_thin_line(bbox),
            "likely_table_line": self._is_likely_table_line(bbox),
        }

        if self.config.include_raw_metadata:
            metadata["raw_drawing"] = self._safe_metadata_value(drawing)

        return DrawingRaw(
            drawing_id=make_id("draw"),
            page_number=page_number,
            bbox=bbox,
            drawing_type=str(drawing_type) if drawing_type is not None else None,
            stroke=stroke,
            fill=fill,
            width=width,
            items=items,
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
                "created_by": "DrawingExtractor",
                "page_kind": getattr(page_record, "page_kind", "unknown"),
            },
        )

    def _extract_drawing_bbox(
        self,
        drawing: Dict[str, Any],
    ) -> Optional[List[float]]:
        bbox = drawing.get("bbox")

        if bbox is not None:
            normalized = normalize_bbox(bbox)
            if normalized:
                return normalized

        rect = drawing.get("rect")

        if rect is not None:
            return self._rect_to_bbox(rect)

        items = drawing.get("items", []) or []
        return self._infer_bbox_from_items(items)

    def _rect_to_bbox(self, rect: Any) -> Optional[List[float]]:
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
            if len(rect) == 4:
                return [
                    float(rect[0]),
                    float(rect[1]),
                    float(rect[2]),
                    float(rect[3]),
                ]
        except Exception:
            pass

        return None

    def _infer_bbox_from_items(
        self,
        items: List[Any],
    ) -> Optional[List[float]]:
        points: List[List[float]] = []

        for item in items:
            points.extend(self._extract_points_from_item(item))

        if not points:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        return [
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        ]

    def _extract_points_from_item(
        self,
        item: Any,
    ) -> List[List[float]]:
        points: List[List[float]] = []

        if item is None:
            return points

        if isinstance(item, (list, tuple)):
            for part in item:
                points.extend(self._extract_points_from_item(part))
            return points

        if hasattr(item, "x") and hasattr(item, "y"):
            try:
                points.append([float(item.x), float(item.y)])
                return points
            except Exception:
                return points

        if all(hasattr(item, attr) for attr in ["x0", "y0", "x1", "y1"]):
            try:
                points.append([float(item.x0), float(item.y0)])
                points.append([float(item.x1), float(item.y1)])
                return points
            except Exception:
                return points

        return points

    def _normalize_drawing_items(
        self,
        items: List[Any],
    ) -> List[Any]:
        return [self._safe_metadata_value(item) for item in items]

    def _passes_size_filter(
        self,
        bbox: Optional[List[float]],
    ) -> bool:
        if not bbox:
            return True

        x0, y0, x1, y1 = bbox

        width = max(x1 - x0, 0.0)
        height = max(y1 - y0, 0.0)

        if self.config.keep_thin_lines:
            if width <= 1.0 or height <= 1.0:
                return True

        if self.config.min_width is not None and width < self.config.min_width:
            return False

        if self.config.min_height is not None and height < self.config.min_height:
            return False

        return True

    def _is_horizontal_line(
        self,
        bbox: Optional[List[float]],
    ) -> bool:
        if not bbox:
            return False

        x0, y0, x1, y1 = bbox

        width = abs(x1 - x0)
        height = abs(y1 - y0)

        return width > 10 and height <= 2

    def _is_vertical_line(
        self,
        bbox: Optional[List[float]],
    ) -> bool:
        if not bbox:
            return False

        x0, y0, x1, y1 = bbox

        width = abs(x1 - x0)
        height = abs(y1 - y0)

        return height > 10 and width <= 2

    def _is_thin_line(
        self,
        bbox: Optional[List[float]],
    ) -> bool:
        if not bbox:
            return False

        x0, y0, x1, y1 = bbox

        width = abs(x1 - x0)
        height = abs(y1 - y0)

        return width <= 2 or height <= 2

    def _is_likely_table_line(
        self,
        bbox: Optional[List[float]],
    ) -> bool:
        return self._is_horizontal_line(bbox) or self._is_vertical_line(bbox)

    def _safe_float(
        self,
        value: Any,
    ) -> Optional[float]:
        try:
            if value is None:
                return None

            return float(value)
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
