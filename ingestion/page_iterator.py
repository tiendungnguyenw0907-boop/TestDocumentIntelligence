"""
page_iterator.py

Production V1 - Colab Ready

Purpose
-------
Iterate document pages in a safe and consistent way for downstream pipelines.

This module is responsible for:
- Opening document pages
- Iterating pages by range
- Iterating pages by batch
- Returning PageRecord objects
- Supporting PDF pages and image pages
- Creating virtual pages for TXT / DOCX when needed

Important
---------
For PDF, the returned fitz.Page object is only valid inside the iteration loop.
Do not store fitz.Page objects for later use. Extract what you need inside the loop.

Typical usage
-------------
from document_ai.ingestion.page_iterator import PageIterator

iterator = PageIterator()

for page_record in iterator.iter_pages("sample.pdf"):
    page = page_record.page_object
    print(page_record.page_number, page_record.width, page_record.height)

Batch usage
-----------
for batch in iterator.iter_batches("sample.pdf", batch_size=10):
    for page_record in batch:
        print(page_record.page_number)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union


# ============================================================
# Local imports
# ============================================================

try:
    from document_ai.ingestion.document_loader import (
        DocumentLoader,
        DocumentLoaderConfig,
        LoadedDocument,
        MissingDependencyError,
        DocumentLoaderError,
        UnsupportedDocumentTypeError,
    )
except Exception:
    DocumentLoader = None
    DocumentLoaderConfig = None
    LoadedDocument = None

    class MissingDependencyError(Exception):
        pass

    class DocumentLoaderError(Exception):
        pass

    class UnsupportedDocumentTypeError(Exception):
        pass


# ============================================================
# Optional dependencies
# ============================================================

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image, ImageSequence
except Exception:
    Image = None
    ImageSequence = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None


# ============================================================
# Config
# ============================================================

@dataclass
class PageIteratorConfig:
    """
    Configuration for PageIterator.
    """

    # For virtual pages from TXT/DOCX
    text_virtual_page_char_limit: int = 4000
    docx_paragraphs_per_virtual_page: int = 40

    # For PDF preview text
    include_text_preview: bool = False
    text_preview_chars: int = 1000

    # For image rendering preview from PDF
    render_pdf_page_image: bool = False
    render_dpi: int = 150

    # Memory safety
    allow_encrypted_pdf: bool = False


# ============================================================
# Data class
# ============================================================

@dataclass
class PageRecord:
    """
    Page-level record passed to downstream extraction pipelines.

    Raw objects:
    - page_object: fitz.Page for PDF
    - image_object: PIL.Image for image files or rendered pages
    - text_content: text for virtual TXT/DOCX pages

    JSON-safe fields can be obtained using to_dict().
    """

    document_id: str
    source_path: str
    file_name: str
    document_type: str

    page_number: int
    page_index: int

    width: Optional[float] = None
    height: Optional[float] = None
    rotation: int = 0

    page_kind: str = "unknown"
    page_label: Optional[str] = None

    page_object: Optional[Any] = None
    image_object: Optional[Any] = None
    text_content: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(
        self,
        include_text_content: bool = False,
        include_raw_object_info: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert PageRecord to JSON-safe dictionary.
        Raw objects are not serialized.
        """

        data = {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "file_name": self.file_name,
            "document_type": self.document_type,
            "page_number": self.page_number,
            "page_index": self.page_index,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "page_kind": self.page_kind,
            "page_label": self.page_label,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }

        if include_text_content:
            data["text_content"] = self.text_content

        if include_raw_object_info:
            data["has_page_object"] = self.page_object is not None
            data["has_image_object"] = self.image_object is not None
            data["has_text_content"] = self.text_content is not None

        return data

    def to_json(
        self,
        ensure_ascii: bool = False,
        indent: int = 2,
        include_text_content: bool = False,
    ) -> str:
        return json.dumps(
            self.to_dict(include_text_content=include_text_content),
            ensure_ascii=ensure_ascii,
            indent=indent,
        )


# ============================================================
# Page Iterator
# ============================================================

class PageIterator:
    """
    Iterate pages from PDF, image, TXT, and DOCX.

    This class should not perform extraction logic.
    It only returns PageRecord objects for downstream processing.
    """

    def __init__(
        self,
        config: Optional[PageIteratorConfig] = None,
        loader: Optional[Any] = None,
    ):
        self.config = config or PageIteratorConfig()

        if loader is not None:
            self.loader = loader
        else:
            if DocumentLoader is None:
                raise MissingDependencyError(
                    "DocumentLoader is not available. "
                    "Make sure document_ai/ingestion/document_loader.py exists."
                )

            self.loader = DocumentLoader(
                DocumentLoaderConfig(
                    allow_encrypted_pdf=self.config.allow_encrypted_pdf
                )
            )

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def iter_pages(
        self,
        document_path: Union[str, Path],
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
        loaded_document: Optional[Any] = None,
    ) -> Iterator[PageRecord]:
        """
        Iterate pages from a document.

        Parameters
        ----------
        document_path:
            Path to input document.

        start_page:
            1-based start page.

        end_page:
            1-based inclusive end page.

        max_pages:
            Maximum number of yielded pages.

        page_numbers:
            Optional exact 1-based page numbers to iterate.
            If provided, start_page/end_page are ignored.

        loaded_document:
            Optional LoadedDocument from DocumentLoader.
            If not provided, loader.load() will be called.

        Yields
        ------
        PageRecord
        """

        loaded_doc = loaded_document or self.loader.load(document_path)

        if loaded_doc.document_type == "pdf":
            yield from self.iter_pdf_pages(
                document_path=document_path,
                loaded_document=loaded_doc,
                start_page=start_page,
                end_page=end_page,
                max_pages=max_pages,
                page_numbers=page_numbers,
            )
            return

        if loaded_doc.document_type == "image":
            yield from self.iter_image_pages(
                document_path=document_path,
                loaded_document=loaded_doc,
                start_page=start_page,
                end_page=end_page,
                max_pages=max_pages,
                page_numbers=page_numbers,
            )
            return

        if loaded_doc.document_type == "text":
            yield from self.iter_text_virtual_pages(
                document_path=document_path,
                loaded_document=loaded_doc,
                start_page=start_page,
                end_page=end_page,
                max_pages=max_pages,
                page_numbers=page_numbers,
            )
            return

        if loaded_doc.document_type == "docx":
            yield from self.iter_docx_virtual_pages(
                document_path=document_path,
                loaded_document=loaded_doc,
                start_page=start_page,
                end_page=end_page,
                max_pages=max_pages,
                page_numbers=page_numbers,
            )
            return

        raise UnsupportedDocumentTypeError(
            f"Unsupported document type for page iteration: "
            f"{loaded_doc.document_type}"
        )

    def iter_batches(
        self,
        document_path: Union[str, Path],
        batch_size: int = 10,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
        loaded_document: Optional[Any] = None,
    ) -> Iterator[List[PageRecord]]:
        """
        Iterate pages in batches.

        Useful for large PDFs and async/parallel downstream processing.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        batch: List[PageRecord] = []

        for page_record in self.iter_pages(
            document_path=document_path,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
            page_numbers=page_numbers,
            loaded_document=loaded_document,
        ):
            batch.append(page_record)

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def get_page_count(
        self,
        document_path: Union[str, Path],
        loaded_document: Optional[Any] = None,
    ) -> Optional[int]:
        """
        Return page count from LoadedDocument.
        """

        loaded_doc = loaded_document or self.loader.load(document_path)
        return loaded_doc.page_count

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    def iter_pdf_pages(
        self,
        document_path: Union[str, Path],
        loaded_document: Optional[Any] = None,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> Iterator[PageRecord]:
        """
        Iterate PDF pages.

        The fitz.Page object is only valid inside the loop.
        """

        if fitz is None:
            raise MissingDependencyError(
                "PyMuPDF is not installed. Run: pip install pymupdf"
            )

        loaded_doc = loaded_document or self.loader.load(document_path)
        path = Path(document_path)

        with fitz.open(str(path)) as pdf:
            if pdf.is_encrypted and not self.config.allow_encrypted_pdf:
                raise DocumentLoaderError(
                    "PDF is encrypted. Set allow_encrypted_pdf=True if needed."
                )

            page_indices = self._resolve_page_indices(
                total_pages=pdf.page_count,
                start_page=start_page,
                end_page=end_page,
                max_pages=max_pages,
                page_numbers=page_numbers,
            )

            for page_index in page_indices:
                page = pdf.load_page(page_index)
                rect = page.rect

                metadata: Dict[str, Any] = {
                    "pdf_page_index": page_index,
                    "pdf_page_number": page_index + 1,
                    "mediabox": self._rect_to_list(page.mediabox),
                    "cropbox": self._rect_to_list(page.cropbox),
                }

                text_preview = None

                if self.config.include_text_preview:
                    try:
                        text = page.get_text("text") or ""
                        text_preview = text[: self.config.text_preview_chars]
                        metadata["text_preview_char_count"] = len(text_preview)
                    except Exception as exc:
                        metadata["text_preview_error"] = str(exc)

                image_object = None

                if self.config.render_pdf_page_image:
                    try:
                        image_object = self._render_pdf_page_to_image(
                            page=page,
                            dpi=self.config.render_dpi,
                        )
                        metadata["render_dpi"] = self.config.render_dpi
                    except Exception as exc:
                        metadata["render_error"] = str(exc)

                yield PageRecord(
                    document_id=loaded_doc.document_id,
                    source_path=loaded_doc.source_path,
                    file_name=loaded_doc.file_name,
                    document_type=loaded_doc.document_type,
                    page_number=page_index + 1,
                    page_index=page_index,
                    width=float(rect.width),
                    height=float(rect.height),
                    rotation=int(page.rotation),
                    page_kind="pdf_page",
                    page_label=self._get_pdf_page_label(pdf, page_index),
                    page_object=page,
                    image_object=image_object,
                    text_content=text_preview,
                    metadata=metadata,
                )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    def iter_image_pages(
        self,
        document_path: Union[str, Path],
        loaded_document: Optional[Any] = None,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> Iterator[PageRecord]:
        """
        Iterate image pages.

        Single-frame image has 1 page.
        Multi-frame TIFF can have multiple pages.
        """

        if Image is None or ImageSequence is None:
            raise MissingDependencyError(
                "Pillow is not installed. Run: pip install pillow"
            )

        loaded_doc = loaded_document or self.loader.load(document_path)
        path = Path(document_path)

        with Image.open(str(path)) as img:
            frame_count = getattr(img, "n_frames", 1)

            page_indices = self._resolve_page_indices(
                total_pages=frame_count,
                start_page=start_page,
                end_page=end_page,
                max_pages=max_pages,
                page_numbers=page_numbers,
            )

            for page_index in page_indices:
                img.seek(page_index)
                frame = img.copy()
                width, height = frame.size

                yield PageRecord(
                    document_id=loaded_doc.document_id,
                    source_path=loaded_doc.source_path,
                    file_name=loaded_doc.file_name,
                    document_type=loaded_doc.document_type,
                    page_number=page_index + 1,
                    page_index=page_index,
                    width=float(width),
                    height=float(height),
                    rotation=0,
                    page_kind="image_page",
                    page_label=f"Image page {page_index + 1}",
                    page_object=None,
                    image_object=frame,
                    text_content=None,
                    metadata={
                        "image_format": img.format,
                        "image_mode": frame.mode,
                        "frame_count": frame_count,
                        "dpi": img.info.get("dpi"),
                    },
                )

    # --------------------------------------------------------
    # TXT virtual pages
    # --------------------------------------------------------

    def iter_text_virtual_pages(
        self,
        document_path: Union[str, Path],
        loaded_document: Optional[Any] = None,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> Iterator[PageRecord]:
        """
        Create virtual pages from TXT / MD files.
        """

        loaded_doc = loaded_document or self.loader.load(document_path)
        path = Path(document_path)

        text = self._read_text_file(path)
        chunks = self._split_text_by_char_limit(
            text=text,
            char_limit=self.config.text_virtual_page_char_limit,
        )

        total_pages = len(chunks)

        page_indices = self._resolve_page_indices(
            total_pages=total_pages,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
            page_numbers=page_numbers,
        )

        for page_index in page_indices:
            chunk = chunks[page_index]

            yield PageRecord(
                document_id=loaded_doc.document_id,
                source_path=loaded_doc.source_path,
                file_name=loaded_doc.file_name,
                document_type=loaded_doc.document_type,
                page_number=page_index + 1,
                page_index=page_index,
                width=None,
                height=None,
                rotation=0,
                page_kind="virtual_text_page",
                page_label=f"Virtual text page {page_index + 1}",
                page_object=None,
                image_object=None,
                text_content=chunk,
                metadata={
                    "char_count": len(chunk),
                    "word_count": len(chunk.split()),
                    "virtual_page_char_limit": self.config.text_virtual_page_char_limit,
                    "total_virtual_pages": total_pages,
                },
            )

    # --------------------------------------------------------
    # DOCX virtual pages
    # --------------------------------------------------------

    def iter_docx_virtual_pages(
        self,
        document_path: Union[str, Path],
        loaded_document: Optional[Any] = None,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> Iterator[PageRecord]:
        """
        Create virtual pages from DOCX paragraphs.
        """

        if DocxDocument is None:
            raise MissingDependencyError(
                "python-docx is not installed. Run: pip install python-docx"
            )

        loaded_doc = loaded_document or self.loader.load(document_path)
        doc = DocxDocument(str(document_path))

        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        chunks = self._split_list(
            paragraphs,
            self.config.docx_paragraphs_per_virtual_page,
        )

        total_pages = len(chunks)

        page_indices = self._resolve_page_indices(
            total_pages=total_pages,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
            page_numbers=page_numbers,
        )

        for page_index in page_indices:
            paragraph_chunk = chunks[page_index]
            text = "\n".join(paragraph_chunk)

            yield PageRecord(
                document_id=loaded_doc.document_id,
                source_path=loaded_doc.source_path,
                file_name=loaded_doc.file_name,
                document_type=loaded_doc.document_type,
                page_number=page_index + 1,
                page_index=page_index,
                width=None,
                height=None,
                rotation=0,
                page_kind="virtual_docx_page",
                page_label=f"Virtual DOCX page {page_index + 1}",
                page_object=None,
                image_object=None,
                text_content=text,
                metadata={
                    "paragraph_count": len(paragraph_chunk),
                    "char_count": len(text),
                    "word_count": len(text.split()),
                    "paragraphs_per_virtual_page": self.config.docx_paragraphs_per_virtual_page,
                    "total_virtual_pages": total_pages,
                },
            )

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _resolve_page_indices(
        self,
        total_pages: int,
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> List[int]:
        """
        Resolve selected pages to zero-based page indices.

        start_page / end_page / page_numbers are 1-based.
        end_page is inclusive.
        """

        if total_pages <= 0:
            return []

        if page_numbers:
            indices = []

            for page_no in page_numbers:
                if 1 <= page_no <= total_pages:
                    indices.append(page_no - 1)

            indices = sorted(set(indices))

        else:
            start = max(start_page, 1)
            end = total_pages if end_page is None else min(end_page, total_pages)

            if start > end:
                return []

            indices = list(range(start - 1, end))

        if max_pages is not None:
            indices = indices[:max_pages]

        return indices

    def _rect_to_list(self, rect: Any) -> Optional[List[float]]:
        try:
            return [
                float(rect.x0),
                float(rect.y0),
                float(rect.x1),
                float(rect.y1),
            ]
        except Exception:
            return None

    def _get_pdf_page_label(self, pdf: Any, page_index: int) -> Optional[str]:
        """
        Return PDF page label if available.
        PyMuPDF support may vary by version.
        """

        try:
            label = pdf.get_page_labels()
            if not label:
                return None
        except Exception:
            return None

        return None

    def _render_pdf_page_to_image(self, page: Any, dpi: int = 150) -> Any:
        """
        Render PDF page to PIL Image.

        Used only when config.render_pdf_page_image=True.
        """

        if Image is None:
            raise MissingDependencyError(
                "Pillow is not installed. Run: pip install pillow"
            )

        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)

        pix = page.get_pixmap(matrix=matrix, alpha=False)

        img = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )

        return img

    def _read_text_file(self, path: Path) -> str:
        encodings = ["utf-8", "utf-8-sig", "cp1258", "latin-1"]

        for enc in encodings:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _split_text_by_char_limit(
        self,
        text: str,
        char_limit: int,
    ) -> List[str]:
        """
        Split text into virtual pages by char limit.
        Try to split at line boundary when possible.
        """

        if not text:
            return [""]

        if char_limit <= 0:
            return [text]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        lines = text.splitlines()

        for line in lines:
            line_len = len(line) + 1

            if current and current_len + line_len > char_limit:
                chunks.append("\n".join(current))
                current = []
                current_len = 0

            current.append(line)
            current_len += line_len

        if current:
            chunks.append("\n".join(current))

        return chunks or [""]

    def _split_list(
        self,
        items: List[Any],
        chunk_size: int,
    ) -> List[List[Any]]:
        if not items:
            return [[]]

        if chunk_size <= 0:
            return [items]

        return [
            items[i : i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]


# ============================================================
# Colab helper functions
# ============================================================

def iter_document_pages(
    document_path: Union[str, Path],
    start_page: int = 1,
    end_page: Optional[int] = None,
    max_pages: Optional[int] = None,
) -> Iterator[PageRecord]:
    """
    Convenience helper for notebooks.
    """

    iterator = PageIterator()

    yield from iterator.iter_pages(
        document_path=document_path,
        start_page=start_page,
        end_page=end_page,
        max_pages=max_pages,
    )


def preview_pages(
    document_path: Union[str, Path],
    max_pages: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return JSON-safe preview of first pages.
    """

    iterator = PageIterator(
        PageIteratorConfig(
            include_text_preview=True,
            render_pdf_page_image=False,
        )
    )

    results = []

    for page_record in iterator.iter_pages(
        document_path=document_path,
        max_pages=max_pages,
    ):
        results.append(
            page_record.to_dict(
                include_text_content=True,
                include_raw_object_info=True,
            )
        )

    return results
