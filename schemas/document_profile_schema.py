"""
document_profile_schema.py

Production V1 - Colab Ready

Purpose
-------
Schema definitions for document profiling.

Used by:
- DocumentLoader
- DocumentProfiler
- PageIterator
- PageExtractionPipeline
- DocumentPipeline

Main objects
------------
- PageProfile
- DocumentProfile
- DocumentProfileSummary
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, asdict, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


SCHEMA_VERSION = "document_profile_schema_v1"


def make_id(prefix: str = "doc_profile") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("Ƣ", "Ư")
    text = text.replace("ƣ", "ư")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_path(path: Any) -> str:
    if path is None:
        return ""

    try:
        return str(Path(path))
    except Exception:
        return str(path)


def normalize_page_numbers(values: Any) -> List[int]:
    if values is None:
        return []

    if not isinstance(values, list):
        values = [values]

    result = []

    for value in values:
        try:
            page_number = int(value)

            if page_number > 0:
                result.append(page_number)
        except Exception:
            continue

    return sorted(list(dict.fromkeys(result)))


def clamp_float(
    value: Any,
    default: float = 0.0,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> float:
    try:
        value = float(value)
    except Exception:
        value = default

    return round(max(min_value, min(value, max_value)), 4)


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        if value is None:
            return default

        return int(value)
    except Exception:
        return default


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return json_safe(asdict(value))

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            json_safe(item)
            for item in value
        ]

    if hasattr(value, "to_dict"):
        try:
            return json_safe(value.to_dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return json_safe(vars(value))
        except Exception:
            pass

    return str(value)


@dataclass
class PageProfile:
    page_number: int
    page_index: int = 0

    width: float = 0.0
    height: float = 0.0
    rotation: int = 0

    has_text: bool = False
    has_images: bool = False
    has_drawings: bool = False
    has_annotations: bool = False
    has_links: bool = False

    text_length: int = 0
    word_count: int = 0
    line_count: int = 0

    image_count: int = 0
    drawing_count: int = 0
    annotation_count: int = 0
    link_count: int = 0
    font_count: int = 0

    estimated_table_count: int = 0
    is_table_candidate: bool = False
    is_ocr_candidate: bool = False
    is_blank: bool = False

    page_kind: str = "unknown_page"
    layout_type: str = "unknown_layout"
    complexity_level: str = "low"

    dominant_language: str = ""
    text_preview: str = ""

    confidence: float = 0.70
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))

        try:
            self.width = float(self.width or 0.0)
        except Exception:
            self.width = 0.0

        try:
            self.height = float(self.height or 0.0)
        except Exception:
            self.height = 0.0

        self.rotation = safe_int(self.rotation, default=0)

        self.text_preview = normalize_text(self.text_preview)[:500]
        self.confidence = clamp_float(self.confidence, default=0.70)

        self.text_length = safe_int(self.text_length, default=len(self.text_preview))
        self.word_count = safe_int(self.word_count, default=0)
        self.line_count = safe_int(self.line_count, default=0)

        self.image_count = safe_int(self.image_count, default=0)
        self.drawing_count = safe_int(self.drawing_count, default=0)
        self.annotation_count = safe_int(self.annotation_count, default=0)
        self.link_count = safe_int(self.link_count, default=0)
        self.font_count = safe_int(self.font_count, default=0)
        self.estimated_table_count = safe_int(self.estimated_table_count, default=0)

        if self.text_length > 0 or self.word_count > 0:
            self.has_text = True

        if self.image_count > 0:
            self.has_images = True

        if self.drawing_count > 0:
            self.has_drawings = True

        if self.annotation_count > 0:
            self.has_annotations = True

        if self.link_count > 0:
            self.has_links = True

        self.is_blank = not self.has_text and not self.has_images and not self.has_drawings

        if not self.page_kind or self.page_kind == "unknown_page":
            self.page_kind = self.infer_page_kind()

        if not self.layout_type or self.layout_type == "unknown_layout":
            self.layout_type = self.infer_layout_type()

        if not self.complexity_level:
            self.complexity_level = self.infer_complexity_level()

    def infer_page_kind(self) -> str:
        if self.is_blank:
            return "blank_page"

        if self.has_text and not self.has_images:
            return "digital_text_page"

        if self.has_images and not self.has_text:
            return "image_or_scanned_page"

        if self.has_text and self.has_images:
            return "hybrid_page"

        if self.has_drawings:
            return "drawing_page"

        return "unknown_page"

    def infer_layout_type(self) -> str:
        if self.is_blank:
            return "blank_layout"

        if self.is_table_candidate or self.estimated_table_count > 0:
            return "table_layout"

        if self.image_count >= 2 and self.text_length < 500:
            return "image_heavy_layout"

        if self.has_text and self.text_length > 2000:
            return "text_heavy_layout"

        if self.has_text and self.has_images:
            return "mixed_layout"

        return "normal_layout"

    def infer_complexity_level(self) -> str:
        score = 0

        if self.text_length > 2500:
            score += 1

        if self.image_count >= 3:
            score += 1

        if self.drawing_count >= 20:
            score += 1

        if self.estimated_table_count >= 1 or self.is_table_candidate:
            score += 1

        if self.has_annotations or self.has_links:
            score += 1

        if score >= 4:
            return "high"

        if score >= 2:
            return "medium"

        return "low"

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageProfile":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)

    def validate(self) -> Dict[str, Any]:
        errors = []
        warnings = []

        if self.page_number <= 0:
            errors.append("page_number must be greater than 0")

        if self.width < 0 or self.height < 0:
            errors.append("width and height must be non-negative")

        if self.text_length == 0 and self.has_text:
            warnings.append("has_text is true but text_length is 0")

        if self.image_count == 0 and self.has_images:
            warnings.append("has_images is true but image_count is 0")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


@dataclass
class DocumentProfileSummary:
    document_id: str = ""

    document_type: str = "unknown"
    pdf_type: str = "unknown"

    page_count: int = 0
    profiled_page_count: int = 0

    has_text_layer: bool = False
    has_image_layer: bool = False
    has_tables: bool = False
    has_annotations: bool = False
    has_links: bool = False

    need_ocr: bool = False
    ocr_page_count: int = 0
    table_candidate_page_count: int = 0
    image_page_count: int = 0
    blank_page_count: int = 0

    total_text_length: int = 0
    total_word_count: int = 0
    total_image_count: int = 0
    total_drawing_count: int = 0
    total_annotation_count: int = 0
    total_link_count: int = 0

    complexity_level: str = "low"
    processing_strategy: str = "standard_extraction"

    dominant_language: str = ""
    confidence: float = 0.70

    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_count = safe_int(self.page_count, default=0)
        self.profiled_page_count = safe_int(self.profiled_page_count, default=0)
        self.ocr_page_count = safe_int(self.ocr_page_count, default=0)
        self.table_candidate_page_count = safe_int(self.table_candidate_page_count, default=0)
        self.image_page_count = safe_int(self.image_page_count, default=0)
        self.blank_page_count = safe_int(self.blank_page_count, default=0)

        self.total_text_length = safe_int(self.total_text_length, default=0)
        self.total_word_count = safe_int(self.total_word_count, default=0)
        self.total_image_count = safe_int(self.total_image_count, default=0)
        self.total_drawing_count = safe_int(self.total_drawing_count, default=0)
        self.total_annotation_count = safe_int(self.total_annotation_count, default=0)
        self.total_link_count = safe_int(self.total_link_count, default=0)

        self.confidence = clamp_float(self.confidence, default=0.70)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentProfileSummary":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)


@dataclass
class DocumentProfile:
    document_id: str = field(default_factory=lambda: make_id("doc"))
    source_path: str = ""
    file_name: str = ""
    file_extension: str = ""

    document_type: str = "unknown"
    mime_type: str = ""

    file_size_bytes: int = 0
    sha256: str = ""

    page_count: int = 0
    profiled_page_count: int = 0

    is_encrypted: bool = False
    is_password_required: bool = False
    is_readable: bool = True

    has_text_layer: bool = False
    has_image_layer: bool = False
    has_tables: bool = False
    has_annotations: bool = False
    has_links: bool = False

    need_ocr: bool = False
    ocr_page_numbers: List[int] = field(default_factory=list)
    table_candidate_pages: List[int] = field(default_factory=list)
    image_pages: List[int] = field(default_factory=list)
    blank_pages: List[int] = field(default_factory=list)

    pdf_type: str = "unknown"
    processing_strategy: str = "standard_extraction"
    complexity_level: str = "low"

    dominant_language: str = ""
    language_hints: List[str] = field(default_factory=list)

    page_profiles: List[PageProfile] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    created_by: str = "DocumentProfiler"
    confidence: float = 0.70
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.source_path = normalize_path(self.source_path)
        self.file_name = normalize_text(self.file_name)
        self.file_extension = normalize_text(self.file_extension).lower()

        if self.file_extension and not self.file_extension.startswith("."):
            self.file_extension = "." + self.file_extension

        self.file_size_bytes = safe_int(self.file_size_bytes, default=0)
        self.page_count = safe_int(self.page_count, default=0)

        normalized_profiles = []

        for item in self.page_profiles:
            if isinstance(item, PageProfile):
                normalized_profiles.append(item)
            elif isinstance(item, dict):
                normalized_profiles.append(PageProfile.from_dict(item))

        self.page_profiles = normalized_profiles

        if self.page_profiles:
            self.profiled_page_count = len(self.page_profiles)
        else:
            self.profiled_page_count = safe_int(self.profiled_page_count, default=0)

        self.ocr_page_numbers = normalize_page_numbers(self.ocr_page_numbers)
        self.table_candidate_pages = normalize_page_numbers(self.table_candidate_pages)
        self.image_pages = normalize_page_numbers(self.image_pages)
        self.blank_pages = normalize_page_numbers(self.blank_pages)

        self.confidence = clamp_float(self.confidence, default=0.70)

        self._refresh_flags_from_pages()

    def _refresh_flags_from_pages(self) -> None:
        if not self.page_profiles:
            return

        self.has_text_layer = any(page.has_text for page in self.page_profiles)
        self.has_image_layer = any(page.has_images for page in self.page_profiles)
        self.has_tables = any(page.is_table_candidate or page.estimated_table_count > 0 for page in self.page_profiles)
        self.has_annotations = any(page.has_annotations for page in self.page_profiles)
        self.has_links = any(page.has_links for page in self.page_profiles)

        if not self.ocr_page_numbers:
            self.ocr_page_numbers = [
                page.page_number
                for page in self.page_profiles
                if page.is_ocr_candidate
            ]

        if not self.table_candidate_pages:
            self.table_candidate_pages = [
                page.page_number
                for page in self.page_profiles
                if page.is_table_candidate or page.estimated_table_count > 0
            ]

        if not self.image_pages:
            self.image_pages = [
                page.page_number
                for page in self.page_profiles
                if page.has_images
            ]

        if not self.blank_pages:
            self.blank_pages = [
                page.page_number
                for page in self.page_profiles
                if page.is_blank
            ]

        self.need_ocr = len(self.ocr_page_numbers) > 0 or (
            self.has_image_layer and not self.has_text_layer
        )

        if self.pdf_type == "unknown":
            self.pdf_type = self.infer_pdf_type()

        if not self.processing_strategy or self.processing_strategy == "standard_extraction":
            self.processing_strategy = self.infer_processing_strategy()

        if not self.complexity_level or self.complexity_level == "low":
            self.complexity_level = self.infer_complexity_level()

    def infer_pdf_type(self) -> str:
        if self.document_type != "pdf" and self.file_extension != ".pdf":
            return "not_pdf"

        if self.has_text_layer and not self.has_image_layer:
            return "digital_pdf"

        if self.has_image_layer and not self.has_text_layer:
            return "scanned_pdf"

        if self.has_text_layer and self.has_image_layer:
            if self.need_ocr:
                return "mixed_pdf"

            return "hybrid_pdf"

        return "blank_or_unknown_pdf"

    def infer_processing_strategy(self) -> str:
        if not self.is_readable:
            return "unreadable_document"

        if self.is_encrypted or self.is_password_required:
            return "password_or_encrypted_document"

        if self.need_ocr and self.has_text_layer:
            return "hybrid_extraction_with_selective_ocr"

        if self.need_ocr:
            return "ocr_first_extraction"

        if self.has_tables:
            return "layout_and_table_extraction"

        if self.has_text_layer:
            return "digital_text_extraction"

        if self.has_image_layer:
            return "image_extraction"

        return "standard_extraction"

    def infer_complexity_level(self) -> str:
        score = 0

        if self.page_count >= 500:
            score += 2
        elif self.page_count >= 100:
            score += 1

        if len(self.table_candidate_pages) >= 20:
            score += 2
        elif len(self.table_candidate_pages) >= 5:
            score += 1

        if len(self.image_pages) >= 50:
            score += 1

        if self.need_ocr:
            score += 1

        if self.has_annotations or self.has_links:
            score += 1

        page_complexities = [
            page.complexity_level for page in self.page_profiles
        ]

        if page_complexities.count("high") >= 5:
            score += 2
        elif page_complexities.count("medium") >= 10:
            score += 1

        if score >= 5:
            return "high"

        if score >= 2:
            return "medium"

        return "low"

    def summary(self) -> DocumentProfileSummary:
        total_text_length = sum(page.text_length for page in self.page_profiles)
        total_word_count = sum(page.word_count for page in self.page_profiles)
        total_image_count = sum(page.image_count for page in self.page_profiles)
        total_drawing_count = sum(page.drawing_count for page in self.page_profiles)
        total_annotation_count = sum(page.annotation_count for page in self.page_profiles)
        total_link_count = sum(page.link_count for page in self.page_profiles)

        return DocumentProfileSummary(
            document_id=self.document_id,
            document_type=self.document_type,
            pdf_type=self.pdf_type,
            page_count=self.page_count,
            profiled_page_count=self.profiled_page_count,
            has_text_layer=self.has_text_layer,
            has_image_layer=self.has_image_layer,
            has_tables=self.has_tables,
            has_annotations=self.has_annotations,
            has_links=self.has_links,
            need_ocr=self.need_ocr,
            ocr_page_count=len(self.ocr_page_numbers),
            table_candidate_page_count=len(self.table_candidate_pages),
            image_page_count=len(self.image_pages),
            blank_page_count=len(self.blank_pages),
            total_text_length=total_text_length,
            total_word_count=total_word_count,
            total_image_count=total_image_count,
            total_drawing_count=total_drawing_count,
            total_annotation_count=total_annotation_count,
            total_link_count=total_link_count,
            complexity_level=self.complexity_level,
            processing_strategy=self.processing_strategy,
            dominant_language=self.dominant_language,
            confidence=self.confidence,
            warnings=self.warnings,
            metadata={
                "language_hints": self.language_hints,
                "file_name": self.file_name,
                "file_extension": self.file_extension,
                "file_size_bytes": self.file_size_bytes,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_path": self.source_path,
            "file_name": self.file_name,
            "file_extension": self.file_extension,
            "document_type": self.document_type,
            "mime_type": self.mime_type,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "page_count": self.page_count,
            "profiled_page_count": self.profiled_page_count,
            "is_encrypted": self.is_encrypted,
            "is_password_required": self.is_password_required,
            "is_readable": self.is_readable,
            "has_text_layer": self.has_text_layer,
            "has_image_layer": self.has_image_layer,
            "has_tables": self.has_tables,
            "has_annotations": self.has_annotations,
            "has_links": self.has_links,
            "need_ocr": self.need_ocr,
            "ocr_page_numbers": self.ocr_page_numbers,
            "table_candidate_pages": self.table_candidate_pages,
            "image_pages": self.image_pages,
            "blank_pages": self.blank_pages,
            "pdf_type": self.pdf_type,
            "processing_strategy": self.processing_strategy,
            "complexity_level": self.complexity_level,
            "dominant_language": self.dominant_language,
            "language_hints": self.language_hints,
            "page_profiles": [
                page.to_dict() for page in self.page_profiles
            ],
            "document_profile_summary": self.summary().to_dict(),
            "metadata": json_safe(self.metadata),
            "warnings": self.warnings,
            "errors": self.errors,
            "created_by": self.created_by,
            "confidence": self.confidence,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentProfile":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        if "page_profiles" in data:
            data["page_profiles"] = [
                PageProfile.from_dict(item) if isinstance(item, dict) else item
                for item in data.get("page_profiles", []) or []
            ]

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)

    def validate(self) -> Dict[str, Any]:
        errors = []
        warnings = []

        if not self.document_id:
            errors.append("document_id is required")

        if not self.document_type:
            warnings.append("document_type is empty")

        if self.page_count < 0:
            errors.append("page_count must be non-negative")

        if self.file_size_bytes < 0:
            errors.append("file_size_bytes must be non-negative")

        if self.profiled_page_count > self.page_count and self.page_count > 0:
            warnings.append("profiled_page_count is greater than page_count")

        for page in self.page_profiles:
            result = page.validate()

            for error in result.get("errors", []):
                errors.append(f"page:{page.page_number}: {error}")

            for warning in result.get("warnings", []):
                warnings.append(f"page:{page.page_number}: {warning}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def save_json(self, output_path: Union[str, Path]) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                self.to_dict(),
                f,
                ensure_ascii=False,
                indent=2,
            )

        return str(output_path)

    @classmethod
    def load_json(cls, input_path: Union[str, Path]) -> "DocumentProfile":
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)


def make_page_profile(
    page_number: int,
    page_index: Optional[int] = None,
    width: float = 0.0,
    height: float = 0.0,
    text_preview: str = "",
    has_text: bool = False,
    has_images: bool = False,
    image_count: int = 0,
    drawing_count: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> PageProfile:
    return PageProfile(
        page_number=page_number,
        page_index=max(page_number - 1, 0) if page_index is None else page_index,
        width=width,
        height=height,
        text_preview=text_preview,
        has_text=has_text,
        has_images=has_images,
        image_count=image_count,
        drawing_count=drawing_count,
        metadata=metadata or {},
    )


def make_document_profile(
    document_id: Optional[str] = None,
    source_path: str = "",
    file_name: str = "",
    document_type: str = "unknown",
    page_count: int = 0,
    page_profiles: Optional[List[Union[PageProfile, Dict[str, Any]]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DocumentProfile:
    return DocumentProfile(
        document_id=document_id or make_id("doc"),
        source_path=source_path,
        file_name=file_name,
        document_type=document_type,
        page_count=page_count,
        page_profiles=page_profiles or [],
        metadata=metadata or {},
    )


def document_profile_from_dict(data: Dict[str, Any]) -> DocumentProfile:
    return DocumentProfile.from_dict(data)


def document_profile_to_dict(
    profile: Union[DocumentProfile, Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(profile, dict):
        return json_safe(profile)

    return profile.to_dict()


def save_document_profile_json(
    profile: Union[DocumentProfile, Dict[str, Any]],
    output_path: Union[str, Path],
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(profile, DocumentProfile):
        data = profile.to_dict()
    else:
        data = json_safe(profile)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return str(output_path)


def load_document_profile_json(
    input_path: Union[str, Path],
) -> DocumentProfile:
    return DocumentProfile.load_json(input_path)
