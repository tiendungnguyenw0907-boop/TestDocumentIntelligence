"""
image_extractor.py

Production V1 - Colab Ready

Purpose
-------
Extract image objects from a PageRecord and attach them to PageRaw.

Input
-----
- PageRecord from page_iterator.py
- Optional existing PageRaw from text_extractor.py

Output
------
PageRaw with images populated.

Supported page types
--------------------
- PDF page:
    - Extract image blocks from page.get_text("dict")
    - Extract image xref metadata from page.get_images(full=True)
    - Optionally save extracted images to disk

- Image page:
    - Treat the image itself as one page image

Important
---------
This module performs physical image extraction only.
It does not detect figures, captions, charts, diagrams, or semantic meaning.
Those belong to:
- figure_detector.py
- caption_detector.py
- page_understanding_pipeline.py
"""

from __future__ import annotations

import os
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None


from document_ai.schemas.page_raw_schema import (
    PageRaw,
    ImageRaw,
    make_id,
    normalize_bbox,
    normalize_pdf_text,
)


# ============================================================
# Config
# ============================================================

@dataclass
class ImageExtractorConfig:
    """
    Configuration for ImageExtractor.
    """

    extract_pdf_image_blocks: bool = True
    extract_pdf_xref_images: bool = True

    save_extracted_images: bool = False
    output_dir: str = "outputs/extracted_images"

    include_image_bytes_base64: bool = False

    min_image_width: int = 1
    min_image_height: int = 1

    include_raw_metadata: bool = False


# ============================================================
# Image Extractor
# ============================================================

class ImageExtractor:
    """
    Extract physical image objects from a page.

    The extractor can be used in two ways:

    1. Standalone:
        page_raw = image_extractor.process(page_record)

    2. Append to existing PageRaw:
        page_raw = text_extractor.process(page_record)
        page_raw = image_extractor.process(page_record, page_raw)
    """

    def __init__(self, config: Optional[ImageExtractorConfig] = None):
        self.config = config or ImageExtractorConfig()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def process(
        self,
        page_record: Any,
        page_raw: Optional[PageRaw] = None,
    ) -> PageRaw:
        """
        Extract images from PageRecord and return PageRaw.
        """

        if page_raw is None:
            page_raw = self._create_empty_page_raw(page_record)

        page_kind = getattr(page_record, "page_kind", "unknown")

        if page_kind == "pdf_page":
            return self._process_pdf_page(page_record, page_raw)

        if page_kind == "image_page":
            return self._process_image_page(page_record, page_raw)

        if page_kind in {"virtual_text_page", "virtual_docx_page"}:
            page_raw.metadata.setdefault("image_extractor", {})
            page_raw.metadata["image_extractor"] = {
                "image_count": 0,
                "note": "Virtual text/docx page has no image object."
            }
            return page_raw

        page_raw.warnings.append(
            f"Unsupported page_kind for ImageExtractor: {page_kind}"
        )
        return page_raw

    # --------------------------------------------------------
    # PDF page
    # --------------------------------------------------------

    def _process_pdf_page(
        self,
        page_record: Any,
        page_raw: PageRaw,
    ) -> PageRaw:
        page = getattr(page_record, "page_object", None)

        if page is None:
            page_raw.warnings.append("PDF page_object is None in ImageExtractor.")
            return page_raw

        images: List[ImageRaw] = []
        warnings: List[str] = []

        image_blocks: List[ImageRaw] = []
        xref_images: List[ImageRaw] = []

        if self.config.extract_pdf_image_blocks:
            try:
                image_blocks = self._extract_pdf_image_blocks(
                    page=page,
                    page_record=page_record,
                )
            except Exception as exc:
                warnings.append(f"Failed to extract PDF image blocks: {exc}")

        if self.config.extract_pdf_xref_images:
            try:
                xref_images = self._extract_pdf_xref_images(
                    page=page,
                    page_record=page_record,
                )
            except Exception as exc:
                warnings.append(f"Failed to extract PDF xref images: {exc}")

        # Prefer image blocks because they contain displayed bbox.
        # Xref images are still useful when image blocks are missing.
        if image_blocks:
            images.extend(image_blocks)

            # Keep xref metadata in page_raw metadata instead of duplicating
            page_raw.metadata.setdefault("image_xref_summary", [])
            page_raw.metadata["image_xref_summary"] = [
                img.to_dict() for img in xref_images
            ]

        else:
            images.extend(xref_images)

        page_raw.images.extend(images)

        page_raw.metadata.setdefault("image_extractor", {})
        page_raw.metadata["image_extractor"] = {
            "extractor": "ImageExtractor",
            "page_kind": page_record.page_kind,
            "image_block_count": len(image_blocks),
            "xref_image_count": len(xref_images),
            "image_count_added": len(images),
            "total_image_count": len(page_raw.images),
            "save_extracted_images": self.config.save_extracted_images,
        }

        page_raw.warnings.extend(warnings)

        return page_raw

    def _extract_pdf_image_blocks(
        self,
        page: Any,
        page_record: Any,
    ) -> List[ImageRaw]:
        """
        Extract displayed image blocks from page.get_text("dict").

        These usually contain bbox, width, height and image bytes.
        """

        result: List[ImageRaw] = []

        text_dict = page.get_text("dict") or {}
        blocks = text_dict.get("blocks", []) or []

        for block_no, block in enumerate(blocks):
            if block.get("type") != 1:
                continue

            bbox = normalize_bbox(block.get("bbox"))

            width = self._safe_int(block.get("width"))
            height = self._safe_int(block.get("height"))

            if not self._passes_size_filter(width, height):
                continue

            ext = block.get("ext")
            colorspace = block.get("colorspace")

            image_bytes = block.get("image")

            saved_path = None
            image_base64 = None

            if image_bytes:
                if self.config.save_extracted_images:
                    saved_path = self._save_image_bytes(
                        image_bytes=image_bytes,
                        document_id=page_record.document_id,
                        page_number=page_record.page_number,
                        image_name=f"block_{block_no}",
                        ext=ext or "png",
                    )

                if self.config.include_image_bytes_base64:
                    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            metadata: Dict[str, Any] = {
                "source": "pdf_image_block",
                "block_no": block_no,
                "saved_path": saved_path,
                "image_base64": image_base64,
                "xres": block.get("xres"),
                "yres": block.get("yres"),
                "bpc": block.get("bpc"),
                "transform": block.get("transform"),
            }

            if self.config.include_raw_metadata:
                metadata["raw_block"] = self._strip_binary_from_dict(block)

            image_obj = ImageRaw(
                image_id=make_id("img"),
                page_number=page_record.page_number,
                bbox=bbox,
                width=width,
                height=height,
                colorspace=str(colorspace) if colorspace is not None else None,
                xref=None,
                ext=ext,
                metadata=metadata,
            )

            result.append(image_obj)

        return result

    def _extract_pdf_xref_images(
        self,
        page: Any,
        page_record: Any,
    ) -> List[ImageRaw]:
        """
        Extract image metadata from page.get_images(full=True).

        PyMuPDF tuple format commonly:
        (xref, smask, width, height, bpc, colorspace, alt_colorspace, name, filter, referencer)
        """

        result: List[ImageRaw] = []

        raw_images = page.get_images(full=True) or []

        for index, item in enumerate(raw_images):
            try:
                xref = item[0]
                width = self._safe_int(item[2])
                height = self._safe_int(item[3])
                bpc = item[4] if len(item) > 4 else None
                colorspace = item[5] if len(item) > 5 else None
                alt_colorspace = item[6] if len(item) > 6 else None
                name = item[7] if len(item) > 7 else None
                image_filter = item[8] if len(item) > 8 else None

                if not self._passes_size_filter(width, height):
                    continue

                ext = None
                saved_path = None
                image_base64 = None

                extracted = None

                if self.config.save_extracted_images or self.config.include_image_bytes_base64:
                    try:
                        doc = page.parent
                        extracted = doc.extract_image(xref)
                        ext = extracted.get("ext")
                        image_bytes = extracted.get("image")

                        if image_bytes and self.config.save_extracted_images:
                            saved_path = self._save_image_bytes(
                                image_bytes=image_bytes,
                                document_id=page_record.document_id,
                                page_number=page_record.page_number,
                                image_name=f"xref_{xref}",
                                ext=ext or "png",
                            )

                        if image_bytes and self.config.include_image_bytes_base64:
                            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

                    except Exception as exc:
                        extracted = {"error": str(exc)}

                metadata: Dict[str, Any] = {
                    "source": "pdf_xref_image",
                    "image_index": index,
                    "smask": item[1] if len(item) > 1 else None,
                    "bpc": bpc,
                    "alt_colorspace": alt_colorspace,
                    "name": name,
                    "filter": image_filter,
                    "saved_path": saved_path,
                    "image_base64": image_base64,
                }

                if extracted and self.config.include_raw_metadata:
                    metadata["extracted_metadata"] = self._strip_binary_from_dict(extracted)

                if self.config.include_raw_metadata:
                    metadata["raw_tuple"] = [str(x) for x in item]

                image_obj = ImageRaw(
                    image_id=make_id("img"),
                    page_number=page_record.page_number,
                    bbox=None,
                    width=width,
                    height=height,
                    colorspace=str(colorspace) if colorspace is not None else None,
                    xref=int(xref) if xref is not None else None,
                    ext=ext,
                    metadata=metadata,
                )

                result.append(image_obj)

            except Exception:
                continue

        return result

    # --------------------------------------------------------
    # Image file page
    # --------------------------------------------------------

    def _process_image_page(
        self,
        page_record: Any,
        page_raw: PageRaw,
    ) -> PageRaw:
        image_obj = getattr(page_record, "image_object", None)

        width = self._safe_int(getattr(page_record, "width", None))
        height = self._safe_int(getattr(page_record, "height", None))

        bbox = None
        if width is not None and height is not None:
            bbox = [0.0, 0.0, float(width), float(height)]

        metadata: Dict[str, Any] = {
            "source": "image_page",
            "page_kind": page_record.page_kind,
        }

        if image_obj is not None:
            metadata.update(
                {
                    "image_mode": getattr(image_obj, "mode", None),
                    "image_format": getattr(image_obj, "format", None),
                }
            )

        image_raw = ImageRaw(
            image_id=make_id("img"),
            page_number=page_record.page_number,
            bbox=bbox,
            width=width,
            height=height,
            colorspace=getattr(image_obj, "mode", None) if image_obj is not None else None,
            xref=None,
            ext=self._extension_from_file_name(page_record.file_name),
            metadata=metadata,
        )

        page_raw.images.append(image_raw)

        page_raw.metadata.setdefault("image_extractor", {})
        page_raw.metadata["image_extractor"] = {
            "extractor": "ImageExtractor",
            "page_kind": page_record.page_kind,
            "image_count_added": 1,
            "total_image_count": len(page_raw.images),
        }

        return page_raw

    # --------------------------------------------------------
    # PageRaw helper
    # --------------------------------------------------------

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
                "created_by": "ImageExtractor",
                "page_kind": getattr(page_record, "page_kind", "unknown"),
            },
        )

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    def _passes_size_filter(
        self,
        width: Optional[int],
        height: Optional[int],
    ) -> bool:
        if width is None or height is None:
            return True

        return (
            width >= self.config.min_image_width
            and height >= self.config.min_image_height
        )

    def _save_image_bytes(
        self,
        image_bytes: bytes,
        document_id: str,
        page_number: int,
        image_name: str,
        ext: str = "png",
    ) -> str:
        out_dir = Path(self.config.output_dir) / document_id / f"page_{page_number}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ext = ext.replace(".", "").lower() if ext else "png"
        path = out_dir / f"{image_name}.{ext}"

        with open(path, "wb") as f:
            f.write(image_bytes)

        return str(path)

    def _strip_binary_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = {}

        for key, value in data.items():
            if isinstance(value, (bytes, bytearray)):
                cleaned[key] = f"<binary:{len(value)} bytes>"
            else:
                try:
                    json_safe_value = str(value) if not self._is_json_safe(value) else value
                    cleaned[key] = json_safe_value
                except Exception:
                    cleaned[key] = str(type(value))

        return cleaned

    def _is_json_safe(self, value: Any) -> bool:
        return value is None or isinstance(value, (str, int, float, bool, list, dict))

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except Exception:
            return None

    def _extension_from_file_name(self, file_name: str) -> Optional[str]:
        if not file_name:
            return None

        suffix = Path(file_name).suffix.lower().replace(".", "")
        return suffix or None
