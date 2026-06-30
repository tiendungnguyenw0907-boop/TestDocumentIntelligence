"""
link_extractor.py

Production V1 - Colab Ready

Purpose
-------
Extract hyperlink / internal link objects from a PageRecord and attach them to PageRaw.

Input
-----
- PageRecord from page_iterator.py
- Optional existing PageRaw from previous extractors

Output
------
PageRaw with links populated.

Supported page types
--------------------
- PDF page: uses PyMuPDF page.get_links()
- Image page: no PDF links
- Virtual TXT/DOCX page: no PDF links

Important
---------
This module performs physical link extraction only.

It does not perform:
- reference linking
- citation linking
- section linking
- semantic relation extraction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import (
    PageRaw,
    LinkRaw,
    make_id,
    normalize_bbox,
    normalize_pdf_text,
)


try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None


@dataclass
class LinkExtractorConfig:
    """
    Configuration for LinkExtractor.
    """

    include_raw_metadata: bool = False
    include_empty_links: bool = False


class LinkExtractor:
    """
    Extract PDF links from one page.
    """

    def __init__(self, config: Optional[LinkExtractorConfig] = None):
        self.config = config or LinkExtractorConfig()

    def process(
        self,
        page_record: Any,
        page_raw: Optional[PageRaw] = None,
    ) -> PageRaw:
        """
        Extract links from PageRecord and return PageRaw.
        """

        if page_raw is None:
            page_raw = self._create_empty_page_raw(page_record)

        page_kind = getattr(page_record, "page_kind", "unknown")

        if page_kind == "pdf_page":
            return self._process_pdf_page(page_record, page_raw)

        if page_kind in {"image_page", "virtual_text_page", "virtual_docx_page"}:
            page_raw.metadata.setdefault("link_extractor", {})
            page_raw.metadata["link_extractor"] = {
                "extractor": "LinkExtractor",
                "page_kind": page_kind,
                "link_count_added": 0,
                "note": "This page kind has no PDF link layer.",
            }
            return page_raw

        page_raw.warnings.append(
            f"Unsupported page_kind for LinkExtractor: {page_kind}"
        )
        return page_raw

    def _process_pdf_page(
        self,
        page_record: Any,
        page_raw: PageRaw,
    ) -> PageRaw:
        page = getattr(page_record, "page_object", None)

        if page is None:
            page_raw.warnings.append("PDF page_object is None in LinkExtractor.")
            return page_raw

        links: List[LinkRaw] = []
        warnings: List[str] = []

        try:
            raw_links = page.get_links() or []
        except Exception as exc:
            page_raw.warnings.append(f"Failed to extract links: {exc}")
            return page_raw

        for link_index, link in enumerate(raw_links):
            try:
                link_obj = self._parse_pdf_link(
                    link=link,
                    page_number=page_record.page_number,
                    link_index=link_index,
                )

                if link_obj is None:
                    continue

                links.append(link_obj)

            except Exception as exc:
                warnings.append(f"Failed to parse link {link_index}: {exc}")

        page_raw.links.extend(links)

        page_raw.metadata.setdefault("link_extractor", {})
        page_raw.metadata["link_extractor"] = {
            "extractor": "LinkExtractor",
            "page_kind": page_record.page_kind,
            "link_count_added": len(links),
            "total_link_count": len(page_raw.links),
            "raw_link_count": len(raw_links),
        }

        page_raw.warnings.extend(warnings)

        return page_raw

    def _parse_pdf_link(
        self,
        link: Dict[str, Any],
        page_number: int,
        link_index: int,
    ) -> Optional[LinkRaw]:
        """
        Parse one PyMuPDF link dictionary.

        Common PyMuPDF fields:
        - kind
        - from
        - uri
        - page
        - to
        - xref
        - file
        - id
        """

        bbox = self._rect_to_bbox(link.get("from"))

        kind = link.get("kind")
        link_type = self._map_link_kind(kind)

        uri = link.get("uri")
        target_page = link.get("page")

        if target_page is not None:
            try:
                # PyMuPDF page index is often 0-based.
                target_page = int(target_page) + 1
            except Exception:
                target_page = None

        has_target = bool(uri) or target_page is not None or link.get("file")

        if not self.config.include_empty_links and not has_target:
            return None

        metadata: Dict[str, Any] = {
            "source": "pdf_link",
            "link_index": link_index,
            "kind": kind,
            "kind_name": link_type,
            "xref": self._safe_metadata_value(link.get("xref")),
            "file": self._safe_metadata_value(link.get("file")),
            "id": self._safe_metadata_value(link.get("id")),
            "to": self._safe_metadata_value(link.get("to")),
        }

        if self.config.include_raw_metadata:
            metadata["raw_link"] = self._safe_metadata_value(link)

        return LinkRaw(
            link_id=make_id("link"),
            page_number=page_number,
            bbox=bbox,
            uri=uri,
            target_page=target_page,
            link_type=link_type,
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
                "created_by": "LinkExtractor",
                "page_kind": getattr(page_record, "page_kind", "unknown"),
            },
        )

    def _map_link_kind(
        self,
        kind: Any,
    ) -> Optional[str]:
        """
        Convert PyMuPDF link kind to readable name.
        """

        if kind is None:
            return None

        try:
            kind_int = int(kind)
        except Exception:
            return str(kind)

        mapping = {
            1: "goto",
            2: "uri",
            3: "launch",
            4: "named",
            5: "goto_remote",
        }

        return mapping.get(kind_int, f"unknown_{kind_int}")

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
