"""
object_normalizer.py

Production V1 - Colab Ready

Purpose
-------
Normalize physical extraction objects before layout understanding.

Input
-----
PageRaw from page_extraction_pipeline.py

Output
------
PageRaw with normalized:
- bbox
- text
- metadata
- object geometry
- relative coordinates

Flow position
-------------
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
PageDocument

Important
---------
This module does not merge objects.
It only cleans and standardizes objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import (
    PageRaw,
    normalize_pdf_text,
)


@dataclass
class ObjectNormalizerConfig:
    """
    Configuration for ObjectNormalizer.
    """

    normalize_text: bool = True
    normalize_bbox: bool = True
    clamp_bbox_to_page: bool = True
    add_geometry_metadata: bool = True
    add_relative_bbox: bool = True
    remove_empty_text_objects: bool = False

    min_bbox_width: float = 0.0
    min_bbox_height: float = 0.0


class ObjectNormalizer:
    """
    Normalize extracted page objects.

    This class operates on PageRaw and updates object fields in-place.
    It returns the same PageRaw object for pipeline compatibility.
    """

    def __init__(self, config: Optional[ObjectNormalizerConfig] = None):
        self.config = config or ObjectNormalizerConfig()

    def process(self, page_raw: PageRaw) -> PageRaw:
        """
        Normalize all physical objects inside PageRaw.
        """

        warnings: List[str] = []

        try:
            self._normalize_page_text(page_raw)
        except Exception as exc:
            warnings.append(f"Failed to normalize page text: {exc}")

        object_groups = [
            ("text_blocks", page_raw.text_blocks),
            ("text_lines", page_raw.text_lines),
            ("text_spans", page_raw.text_spans),
            ("words", page_raw.words),
            ("images", page_raw.images),
            ("drawings", page_raw.drawings),
            ("annotations", page_raw.annotations),
            ("links", page_raw.links),
        ]

        for group_name, objects in object_groups:
            try:
                self._normalize_object_group(
                    page_raw=page_raw,
                    group_name=group_name,
                    objects=objects,
                )
            except Exception as exc:
                warnings.append(f"Failed to normalize {group_name}: {exc}")

        if self.config.remove_empty_text_objects:
            self._remove_empty_text_objects(page_raw)

        page_raw.metadata.setdefault("object_normalizer", {})
        page_raw.metadata["object_normalizer"] = {
            "processor": "ObjectNormalizer",
            "normalized": True,
            "config": {
                "normalize_text": self.config.normalize_text,
                "normalize_bbox": self.config.normalize_bbox,
                "clamp_bbox_to_page": self.config.clamp_bbox_to_page,
                "add_geometry_metadata": self.config.add_geometry_metadata,
                "add_relative_bbox": self.config.add_relative_bbox,
                "remove_empty_text_objects": self.config.remove_empty_text_objects,
            },
            "counts": {
                "text_blocks": len(page_raw.text_blocks),
                "text_lines": len(page_raw.text_lines),
                "text_spans": len(page_raw.text_spans),
                "words": len(page_raw.words),
                "images": len(page_raw.images),
                "drawings": len(page_raw.drawings),
                "annotations": len(page_raw.annotations),
                "links": len(page_raw.links),
            },
        }

        if warnings:
            page_raw.warnings.extend(warnings)

        return page_raw

    def _normalize_page_text(self, page_raw: PageRaw) -> None:
        if not self.config.normalize_text:
            return

        page_raw.raw_text = page_raw.raw_text or ""
        page_raw.normalized_text = normalize_pdf_text(page_raw.raw_text)

    def _normalize_object_group(
        self,
        page_raw: PageRaw,
        group_name: str,
        objects: List[Any],
    ) -> None:
        for index, obj in enumerate(objects):
            self._normalize_single_object(
                page_raw=page_raw,
                obj=obj,
                group_name=group_name,
                object_index=index,
            )

    def _normalize_single_object(
        self,
        page_raw: PageRaw,
        obj: Any,
        group_name: str,
        object_index: int,
    ) -> None:
        if self.config.normalize_text:
            self._normalize_object_text(obj)

        if self.config.normalize_bbox:
            bbox = getattr(obj, "bbox", None)
            normalized_bbox = self._normalize_bbox(
                bbox=bbox,
                page_width=page_raw.width,
                page_height=page_raw.height,
            )

            setattr(obj, "bbox", normalized_bbox)

        if self.config.add_geometry_metadata:
            self._add_geometry_metadata(
                page_raw=page_raw,
                obj=obj,
                group_name=group_name,
                object_index=object_index,
            )

    def _normalize_object_text(self, obj: Any) -> None:
        if hasattr(obj, "text"):
            text = getattr(obj, "text", "") or ""
            setattr(obj, "text", text)

        if hasattr(obj, "normalized_text"):
            source_text = getattr(obj, "text", None)

            if source_text is None:
                source_text = getattr(obj, "normalized_text", "") or ""

            setattr(obj, "normalized_text", normalize_pdf_text(source_text))

    def _normalize_bbox(
        self,
        bbox: Any,
        page_width: Optional[float],
        page_height: Optional[float],
    ) -> Optional[List[float]]:
        if bbox is None:
            return None

        try:
            if len(bbox) != 4:
                return None

            x0, y0, x1, y1 = [float(v) for v in bbox]

            # Reorder invalid bbox
            left = min(x0, x1)
            right = max(x0, x1)
            top = min(y0, y1)
            bottom = max(y0, y1)

            if self.config.clamp_bbox_to_page:
                if page_width is not None:
                    left = max(0.0, min(left, float(page_width)))
                    right = max(0.0, min(right, float(page_width)))

                if page_height is not None:
                    top = max(0.0, min(top, float(page_height)))
                    bottom = max(0.0, min(bottom, float(page_height)))

            width = right - left
            height = bottom - top

            if width < self.config.min_bbox_width:
                return None

            if height < self.config.min_bbox_height:
                return None

            return [
                round(left, 4),
                round(top, 4),
                round(right, 4),
                round(bottom, 4),
            ]

        except Exception:
            return None

    def _add_geometry_metadata(
        self,
        page_raw: PageRaw,
        obj: Any,
        group_name: str,
        object_index: int,
    ) -> None:
        if not hasattr(obj, "metadata"):
            return

        if obj.metadata is None:
            obj.metadata = {}

        bbox = getattr(obj, "bbox", None)

        geometry = self._compute_geometry(
            bbox=bbox,
            page_width=page_raw.width,
            page_height=page_raw.height,
        )

        obj.metadata.setdefault("normalization", {})
        obj.metadata["normalization"].update(
            {
                "group_name": group_name,
                "object_index": object_index,
                "has_bbox": bbox is not None,
                "geometry": geometry,
            }
        )

    def _compute_geometry(
        self,
        bbox: Optional[List[float]],
        page_width: Optional[float],
        page_height: Optional[float],
    ) -> Dict[str, Any]:
        if not bbox:
            return {
                "bbox": None,
                "width": None,
                "height": None,
                "area": None,
                "center": None,
                "relative_bbox": None,
                "page_zone": None,
            }

        x0, y0, x1, y1 = bbox

        width = max(x1 - x0, 0.0)
        height = max(y1 - y0, 0.0)
        area = width * height

        center_x = x0 + width / 2
        center_y = y0 + height / 2

        relative_bbox = None

        if (
            self.config.add_relative_bbox
            and page_width
            and page_height
            and page_width > 0
            and page_height > 0
        ):
            relative_bbox = [
                round(x0 / page_width, 6),
                round(y0 / page_height, 6),
                round(x1 / page_width, 6),
                round(y1 / page_height, 6),
            ]

        return {
            "bbox": bbox,
            "width": round(width, 4),
            "height": round(height, 4),
            "area": round(area, 4),
            "center": [
                round(center_x, 4),
                round(center_y, 4),
            ],
            "relative_bbox": relative_bbox,
            "page_zone": self._detect_page_zone(
                center_y=center_y,
                page_height=page_height,
            ),
        }

    def _detect_page_zone(
        self,
        center_y: float,
        page_height: Optional[float],
    ) -> Optional[str]:
        if not page_height or page_height <= 0:
            return None

        ratio = center_y / page_height

        if ratio <= 0.12:
            return "header_zone"

        if ratio >= 0.88:
            return "footer_zone"

        return "body_zone"

    def _remove_empty_text_objects(self, page_raw: PageRaw) -> None:
        page_raw.text_spans = [
            obj
            for obj in page_raw.text_spans
            if getattr(obj, "normalized_text", "").strip()
        ]

        page_raw.words = [
            obj
            for obj in page_raw.words
            if getattr(obj, "normalized_text", "").strip()
        ]

        page_raw.text_lines = [
            obj
            for obj in page_raw.text_lines
            if getattr(obj, "normalized_text", "").strip()
        ]

        page_raw.text_blocks = [
            obj
            for obj in page_raw.text_blocks
            if getattr(obj, "normalized_text", "").strip()
        ]


def normalize_page_raw(page_raw: PageRaw) -> PageRaw:
    """
    Colab helper function.
    """

    normalizer = ObjectNormalizer()
    return normalizer.process(page_raw)
