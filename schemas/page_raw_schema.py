"""
page_raw_schema.py

Production V1 - Colab Ready

Purpose
-------
Schema definitions for raw page extraction output.

Used by:
- TextExtractor
- ImageExtractor
- DrawingExtractor
- AnnotationExtractor
- LinkExtractor
- FontExtractor
- PageExtractionPipeline

Main objects
------------
- TextSpanRaw
- TextLineRaw
- TextBlockRaw
- WordRaw
- ImageRaw
- DrawingRaw
- AnnotationRaw
- LinkRaw
- FontRaw
- PageRaw
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, asdict, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


SCHEMA_VERSION = "page_raw_schema_v1"


def make_id(prefix: str = "raw") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def normalize_bbox(bbox: Any) -> List[float]:
    if not bbox or len(bbox) != 4:
        return []

    try:
        x0, y0, x1, y1 = bbox

        return [
            round(float(x0), 4),
            round(float(y0), 4),
            round(float(x1), 4),
            round(float(y1), 4),
        ]
    except Exception:
        return []


def bbox_area(bbox: Any) -> float:
    bbox = normalize_bbox(bbox)

    if len(bbox) != 4:
        return 0.0

    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])

    return round(width * height, 4)


def merge_bboxes(bboxes: List[List[float]]) -> List[float]:
    valid = [
        normalize_bbox(bbox)
        for bbox in bboxes
    ]

    valid = [
        bbox for bbox in valid
        if len(bbox) == 4
    ]

    if not valid:
        return []

    return [
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    ]


def bbox_overlap_ratio(
    bbox_a: Any,
    bbox_b: Any,
) -> float:
    a = normalize_bbox(bbox_a)
    b = normalize_bbox(bbox_b)

    if len(a) != 4 or len(b) != 4:
        return 0.0

    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])

    inter_w = max(0.0, x1 - x0)
    inter_h = max(0.0, y1 - y0)
    inter_area = inter_w * inter_h

    smaller_area = min(bbox_area(a), bbox_area(b))

    if smaller_area <= 0:
        return 0.0

    return round(inter_area / smaller_area, 4)


def normalize_pdf_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text)

    replacements = {
        "\u00a0": " ",
        "Ƣ": "Ư",
        "ƣ": "ư",
        "§": "đ",
        "\r\n": "\n",
        "\r": "\n",
    }

    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_text_for_match(text: Any) -> str:
    text = normalize_pdf_text(text).lower()

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

    text = re.sub(r"[^a-z0-9_\-\/\.%]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default

        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        return float(value)
    except Exception:
        return default


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
class TextSpanRaw:
    span_id: str = field(default_factory=lambda: make_id("span"))

    text: str = ""
    normalized_text: str = ""

    bbox: List[float] = field(default_factory=list)

    font: str = ""
    font_size: float = 0.0
    font_flags: int = 0
    color: Any = None

    ascender: Optional[float] = None
    descender: Optional[float] = None
    origin: List[float] = field(default_factory=list)

    block_index: int = 0
    line_index: int = 0
    span_index: int = 0

    page_number: int = 1
    page_index: int = 0

    source: str = "text_extractor"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.text = normalize_pdf_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_pdf_text(self.text)
        else:
            self.normalized_text = normalize_pdf_text(self.normalized_text)

        self.bbox = normalize_bbox(self.bbox)
        self.font = normalize_pdf_text(self.font)
        self.font_size = safe_float(self.font_size, default=0.0)
        self.font_flags = safe_int(self.font_flags, default=0)

        self.block_index = safe_int(self.block_index, default=0)
        self.line_index = safe_int(self.line_index, default=0)
        self.span_index = safe_int(self.span_index, default=0)
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextSpanRaw":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class TextLineRaw:
    line_id: str = field(default_factory=lambda: make_id("line"))

    text: str = ""
    normalized_text: str = ""

    bbox: List[float] = field(default_factory=list)
    spans: List[TextSpanRaw] = field(default_factory=list)

    block_index: int = 0
    line_index: int = 0

    page_number: int = 1
    page_index: int = 0

    direction: List[float] = field(default_factory=list)
    writing_mode: int = 0

    source: str = "text_extractor"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.spans = [
            TextSpanRaw.from_dict(item) if isinstance(item, dict) else item
            for item in self.spans
        ]

        if not self.text and self.spans:
            self.text = "".join([span.text for span in self.spans])

        self.text = normalize_pdf_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_pdf_text(self.text)
        else:
            self.normalized_text = normalize_pdf_text(self.normalized_text)

        if not self.bbox and self.spans:
            self.bbox = merge_bboxes([span.bbox for span in self.spans])
        else:
            self.bbox = normalize_bbox(self.bbox)

        self.block_index = safe_int(self.block_index, default=0)
        self.line_index = safe_int(self.line_index, default=0)
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.writing_mode = safe_int(self.writing_mode, default=0)

    @property
    def span_count(self) -> int:
        return len(self.spans)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["span_count"] = self.span_count
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextLineRaw":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class TextBlockRaw:
    block_id: str = field(default_factory=lambda: make_id("block"))

    text: str = ""
    normalized_text: str = ""

    bbox: List[float] = field(default_factory=list)

    block_type: str = "text"
    block_index: int = 0

    lines: List[TextLineRaw] = field(default_factory=list)

    page_number: int = 1
    page_index: int = 0

    source: str = "text_extractor"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.lines = [
            TextLineRaw.from_dict(item) if isinstance(item, dict) else item
            for item in self.lines
        ]

        if not self.text and self.lines:
            self.text = "\n".join([line.text for line in self.lines if line.text])

        self.text = normalize_pdf_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_pdf_text(self.text)
        else:
            self.normalized_text = normalize_pdf_text(self.normalized_text)

        if not self.bbox and self.lines:
            self.bbox = merge_bboxes([line.bbox for line in self.lines])
        else:
            self.bbox = normalize_bbox(self.bbox)

        self.block_type = normalize_pdf_text(self.block_type) or "text"
        self.block_index = safe_int(self.block_index, default=0)
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def span_count(self) -> int:
        return sum(line.span_count for line in self.lines)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["line_count"] = self.line_count
        data["span_count"] = self.span_count
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextBlockRaw":
        data = dict(data or {})

        aliases = {
            "type": "block_type",
            "number": "block_index",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class WordRaw:
    word_id: str = field(default_factory=lambda: make_id("word"))

    text: str = ""
    normalized_text: str = ""

    bbox: List[float] = field(default_factory=list)

    block_index: int = 0
    line_index: int = 0
    word_index: int = 0

    page_number: int = 1
    page_index: int = 0

    confidence: float = 1.0

    source: str = "text_extractor"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.text = normalize_pdf_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_pdf_text(self.text)
        else:
            self.normalized_text = normalize_pdf_text(self.normalized_text)

        self.bbox = normalize_bbox(self.bbox)

        self.block_index = safe_int(self.block_index, default=0)
        self.line_index = safe_int(self.line_index, default=0)
        self.word_index = safe_int(self.word_index, default=0)
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.confidence = clamp_float(self.confidence, default=1.0)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WordRaw":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class ImageRaw:
    image_id: str = field(default_factory=lambda: make_id("image"))

    bbox: List[float] = field(default_factory=list)

    width: int = 0
    height: int = 0
    colorspace: str = ""
    bits_per_component: int = 0

    xref: Optional[int] = None
    image_ext: str = ""
    image_name: str = ""
    image_path: str = ""

    page_number: int = 1
    page_index: int = 0
    block_index: Optional[int] = None

    source: str = "image_extractor"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.bbox = normalize_bbox(self.bbox)
        self.width = safe_int(self.width, default=0)
        self.height = safe_int(self.height, default=0)
        self.bits_per_component = safe_int(self.bits_per_component, default=0)

        if self.xref is not None:
            self.xref = safe_int(self.xref, default=0)

        self.image_ext = normalize_pdf_text(self.image_ext).lower()
        self.image_name = normalize_pdf_text(self.image_name)
        self.image_path = str(self.image_path or "")

        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))

        if self.block_index is not None:
            self.block_index = safe_int(self.block_index, default=0)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageRaw":
        data = dict(data or {})

        aliases = {
            "ext": "image_ext",
            "name": "image_name",
            "path": "image_path",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class DrawingRaw:
    drawing_id: str = field(default_factory=lambda: make_id("drawing"))

    bbox: List[float] = field(default_factory=list)

    drawing_type: str = ""
    items: List[Any] = field(default_factory=list)

    color: Any = None
    fill: Any = None
    width: float = 0.0
    dashes: Any = None
    line_cap: Any = None
    line_join: Any = None

    page_number: int = 1
    page_index: int = 0
    drawing_index: int = 0

    source: str = "drawing_extractor"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.bbox = normalize_bbox(self.bbox)
        self.drawing_type = normalize_pdf_text(self.drawing_type)
        self.width = safe_float(self.width, default=0.0)
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.drawing_index = safe_int(self.drawing_index, default=0)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DrawingRaw":
        data = dict(data or {})

        aliases = {
            "type": "drawing_type",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class AnnotationRaw:
    annotation_id: str = field(default_factory=lambda: make_id("annot"))

    bbox: List[float] = field(default_factory=list)

    annotation_type: str = ""
    content: str = ""

    xref: Optional[int] = None
    flags: int = 0

    page_number: int = 1
    page_index: int = 0
    annotation_index: int = 0

    source: str = "annotation_extractor"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.bbox = normalize_bbox(self.bbox)
        self.annotation_type = normalize_pdf_text(self.annotation_type)
        self.content = normalize_pdf_text(self.content)

        if self.xref is not None:
            self.xref = safe_int(self.xref, default=0)

        self.flags = safe_int(self.flags, default=0)
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.annotation_index = safe_int(self.annotation_index, default=0)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationRaw":
        data = dict(data or {})

        aliases = {
            "type": "annotation_type",
            "text": "content",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class LinkRaw:
    link_id: str = field(default_factory=lambda: make_id("link"))

    bbox: List[float] = field(default_factory=list)

    link_type: str = ""
    uri: str = ""
    target_page: Optional[int] = None
    target_point: List[float] = field(default_factory=list)

    page_number: int = 1
    page_index: int = 0
    link_index: int = 0

    source: str = "link_extractor"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.bbox = normalize_bbox(self.bbox)
        self.link_type = normalize_pdf_text(self.link_type)
        self.uri = str(self.uri or "")

        if self.target_page is not None:
            self.target_page = safe_int(self.target_page, default=0)

        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.link_index = safe_int(self.link_index, default=0)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinkRaw":
        data = dict(data or {})

        aliases = {
            "kind": "link_type",
            "page": "target_page",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class FontRaw:
    font_id: str = field(default_factory=lambda: make_id("font"))

    font_name: str = ""
    font_size: float = 0.0
    font_flags: int = 0
    color: Any = None

    span_count: int = 0
    text_length: int = 0

    page_number: int = 1
    page_index: int = 0

    is_bold: bool = False
    is_italic: bool = False
    is_monospace: bool = False
    is_serif: bool = False

    source: str = "font_extractor"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.font_name = normalize_pdf_text(self.font_name)
        self.font_size = safe_float(self.font_size, default=0.0)
        self.font_flags = safe_int(self.font_flags, default=0)
        self.span_count = safe_int(self.span_count, default=0)
        self.text_length = safe_int(self.text_length, default=0)
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FontRaw":
        data = dict(data or {})

        aliases = {
            "font": "font_name",
            "size": "font_size",
            "flags": "font_flags",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class PageRaw:
    document_id: str = ""
    source_document: str = ""

    page_number: int = 1
    page_index: int = 0

    width: float = 0.0
    height: float = 0.0
    rotation: int = 0

    raw_text: str = ""
    normalized_text: str = ""

    text_blocks: List[TextBlockRaw] = field(default_factory=list)
    text_lines: List[TextLineRaw] = field(default_factory=list)
    text_spans: List[TextSpanRaw] = field(default_factory=list)
    words: List[WordRaw] = field(default_factory=list)

    images: List[ImageRaw] = field(default_factory=list)
    drawings: List[DrawingRaw] = field(default_factory=list)
    annotations: List[AnnotationRaw] = field(default_factory=list)
    links: List[LinkRaw] = field(default_factory=list)
    fonts: List[FontRaw] = field(default_factory=list)

    page_kind: str = "pdf_page"

    extraction_method: str = "page_extraction_pipeline"
    extraction_status: str = "success"

    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    processor: str = "PageRaw"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.width = safe_float(self.width, default=0.0)
        self.height = safe_float(self.height, default=0.0)
        self.rotation = safe_int(self.rotation, default=0)

        self.raw_text = normalize_pdf_text(self.raw_text)

        if not self.normalized_text:
            self.normalized_text = normalize_pdf_text(self.raw_text)
        else:
            self.normalized_text = normalize_pdf_text(self.normalized_text)

        self.text_blocks = [
            TextBlockRaw.from_dict(item) if isinstance(item, dict) else item
            for item in self.text_blocks
        ]

        self.text_lines = [
            TextLineRaw.from_dict(item) if isinstance(item, dict) else item
            for item in self.text_lines
        ]

        self.text_spans = [
            TextSpanRaw.from_dict(item) if isinstance(item, dict) else item
            for item in self.text_spans
        ]

        self.words = [
            WordRaw.from_dict(item) if isinstance(item, dict) else item
            for item in self.words
        ]

        self.images = [
            ImageRaw.from_dict(item) if isinstance(item, dict) else item
            for item in self.images
        ]

        self.drawings = [
            DrawingRaw.from_dict(item) if isinstance(item, dict) else item
            for item in self.drawings
        ]

        self.annotations = [
            AnnotationRaw.from_dict(item) if isinstance(item, dict) else item
            for item in self.annotations
        ]

        self.links = [
            LinkRaw.from_dict(item) if isinstance(item, dict) else item
            for item in self.links
        ]

        self.fonts = [
            FontRaw.from_dict(item) if isinstance(item, dict) else item
            for item in self.fonts
        ]

        if not self.text_lines and self.text_blocks:
            for block in self.text_blocks:
                self.text_lines.extend(block.lines)

        if not self.text_spans and self.text_lines:
            for line in self.text_lines:
                self.text_spans.extend(line.spans)

        if not self.raw_text:
            self.raw_text = self._build_text_from_blocks_or_lines()
            self.normalized_text = normalize_pdf_text(self.raw_text)

        self.page_kind = normalize_pdf_text(self.page_kind) or "pdf_page"
        self.extraction_method = normalize_pdf_text(self.extraction_method)
        self.extraction_status = normalize_pdf_text(self.extraction_status) or "success"

    @property
    def page_area(self) -> float:
        return round(max(self.width, 0.0) * max(self.height, 0.0), 4)

    @property
    def text_block_count(self) -> int:
        return len(self.text_blocks)

    @property
    def text_line_count(self) -> int:
        return len(self.text_lines)

    @property
    def text_span_count(self) -> int:
        return len(self.text_spans)

    @property
    def word_count(self) -> int:
        if self.words:
            return len(self.words)

        return len(re.findall(r"\S+", self.normalized_text or self.raw_text or ""))

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def drawing_count(self) -> int:
        return len(self.drawings)

    @property
    def annotation_count(self) -> int:
        return len(self.annotations)

    @property
    def link_count(self) -> int:
        return len(self.links)

    @property
    def font_count(self) -> int:
        return len(self.fonts)

    @property
    def has_text(self) -> bool:
        return bool(self.normalized_text or self.text_blocks or self.text_lines or self.words)

    @property
    def has_images(self) -> bool:
        return bool(self.images)

    @property
    def has_drawings(self) -> bool:
        return bool(self.drawings)

    @property
    def is_blank(self) -> bool:
        return not self.has_text and not self.has_images and not self.has_drawings

    def _build_text_from_blocks_or_lines(self) -> str:
        if self.text_blocks:
            return "\n".join(
                [
                    block.text
                    for block in self.text_blocks
                    if block.text
                ]
            )

        if self.text_lines:
            return "\n".join(
                [
                    line.text
                    for line in self.text_lines
                    if line.text
                ]
            )

        if self.words:
            return " ".join(
                [
                    word.text
                    for word in self.words
                    if word.text
                ]
            )

        return ""

    def summary(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_document": self.source_document,
            "page_number": self.page_number,
            "page_index": self.page_index,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "page_area": self.page_area,
            "page_kind": self.page_kind,
            "text_length": len(self.normalized_text or self.raw_text or ""),
            "word_count": self.word_count,
            "text_block_count": self.text_block_count,
            "text_line_count": self.text_line_count,
            "text_span_count": self.text_span_count,
            "image_count": self.image_count,
            "drawing_count": self.drawing_count,
            "annotation_count": self.annotation_count,
            "link_count": self.link_count,
            "font_count": self.font_count,
            "has_text": self.has_text,
            "has_images": self.has_images,
            "has_drawings": self.has_drawings,
            "is_blank": self.is_blank,
            "extraction_method": self.extraction_method,
            "extraction_status": self.extraction_status,
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processor": self.processor,
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "source_document": self.source_document,
            "page_number": self.page_number,
            "page_index": self.page_index,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "text_blocks": [item.to_dict() for item in self.text_blocks],
            "text_lines": [item.to_dict() for item in self.text_lines],
            "text_spans": [item.to_dict() for item in self.text_spans],
            "words": [item.to_dict() for item in self.words],
            "images": [item.to_dict() for item in self.images],
            "drawings": [item.to_dict() for item in self.drawings],
            "annotations": [item.to_dict() for item in self.annotations],
            "links": [item.to_dict() for item in self.links],
            "fonts": [item.to_dict() for item in self.fonts],
            "page_kind": self.page_kind,
            "extraction_method": self.extraction_method,
            "extraction_status": self.extraction_status,
            "page_raw_summary": self.summary(),
            "metadata": json_safe(self.metadata),
            "warnings": self.warnings,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageRaw":
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

        if self.extraction_status not in ["success", "partial", "failed", "skipped"]:
            warnings.append(f"unknown extraction_status: {self.extraction_status}")

        if self.is_blank:
            warnings.append("page has no text, images, or drawings")

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
    def load_json(cls, input_path: Union[str, Path]) -> "PageRaw":
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)


def make_page_raw(
    document_id: str = "",
    source_document: str = "",
    page_number: int = 1,
    page_index: Optional[int] = None,
    width: float = 0.0,
    height: float = 0.0,
    rotation: int = 0,
    raw_text: str = "",
    normalized_text: str = "",
    page_kind: str = "pdf_page",
    metadata: Optional[Dict[str, Any]] = None,
) -> PageRaw:
    return PageRaw(
        document_id=document_id,
        source_document=source_document,
        page_number=page_number,
        page_index=max(page_number - 1, 0) if page_index is None else page_index,
        width=width,
        height=height,
        rotation=rotation,
        raw_text=raw_text,
        normalized_text=normalized_text,
        page_kind=page_kind,
        metadata=metadata or {},
    )


def page_raw_from_dict(data: Dict[str, Any]) -> PageRaw:
    return PageRaw.from_dict(data)


def page_raw_to_dict(page_raw: Union[PageRaw, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(page_raw, dict):
        return json_safe(page_raw)

    return page_raw.to_dict()


def save_page_raw_json(
    page_raw: Union[PageRaw, Dict[str, Any]],
    output_path: Union[str, Path],
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(page_raw, PageRaw):
        data = page_raw.to_dict()
    else:
        data = json_safe(page_raw)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return str(output_path)


def load_page_raw_json(input_path: Union[str, Path]) -> PageRaw:
    return PageRaw.load_json(input_path)


PageRawSchema = PageRaw
TextSpanRawSchema = TextSpanRaw
TextLineRawSchema = TextLineRaw
TextBlockRawSchema = TextBlockRaw
WordRawSchema = WordRaw
ImageRawSchema = ImageRaw
DrawingRawSchema = DrawingRaw
AnnotationRawSchema = AnnotationRaw
LinkRawSchema = LinkRaw
FontRawSchema = FontRaw


# =============================================================================
# Backward compatibility layer
# =============================================================================
# This layer keeps the canonical schema above, but accepts older constructor
# arguments used by existing extraction modules, for example:
# - PageRaw(source_path=..., file_name=..., document_type=...)
# - TextSpanRaw(size=..., flags=..., block_no=..., line_no=..., span_no=...)
# - WordRaw(block_no=..., line_no=..., word_no=...)
# - ImageRaw(ext=...)
# - AnnotationRaw(annot_type=...)
# - DrawingRaw(stroke=...)
# - FontRaw(size=..., count=...)
# It also exposes read/write alias properties such as span.size, font.count,
# image.ext, page_raw.source_path so older code continues to run.


def _compat_dataclass_fields(cls) -> set:
    return set(getattr(cls, "__dataclass_fields__", {}).keys())


def _compat_merge_metadata(metadata: Any, extra: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(metadata, dict):
        merged = dict(metadata)
    elif metadata in [None, ""]:
        merged = {}
    else:
        merged = {"raw_metadata": metadata}

    for key, value in (extra or {}).items():
        if value in [None, ""]:
            continue
        if key not in merged:
            merged[key] = value

    return merged


def _compat_wrap_init(cls, alias_map: Dict[str, str], metadata_only: Optional[List[str]] = None) -> None:
    if getattr(cls, "_compat_init_wrapped", False):
        return

    original_init = cls.__init__
    fields = _compat_dataclass_fields(cls)
    metadata_only = metadata_only or []

    def __init__(self, *args, **kwargs):
        alias_metadata: Dict[str, Any] = {}

        # Translate known aliases to canonical fields.
        for old_key, new_key in list(alias_map.items()):
            if old_key not in kwargs:
                continue

            value = kwargs.pop(old_key)
            alias_metadata[old_key] = value

            if new_key and new_key in fields and new_key not in kwargs:
                kwargs[new_key] = value

        # Store metadata-only aliases.
        for key in list(metadata_only):
            if key in kwargs:
                alias_metadata[key] = kwargs.pop(key)

        # Preserve unknown kwargs in metadata instead of crashing.
        unknown = {
            key: kwargs.pop(key)
            for key in list(kwargs.keys())
            if key not in fields
        }

        if "metadata" in fields and (alias_metadata or unknown):
            metadata = _compat_merge_metadata(kwargs.get("metadata"), alias_metadata)
            if unknown:
                metadata.setdefault("unknown_constructor_kwargs", {}).update(unknown)
            kwargs["metadata"] = metadata

        original_init(self, *args, **kwargs)

    cls.__init__ = __init__
    cls._compat_init_wrapped = True


def _compat_alias_property(canonical_name: str, metadata_key: Optional[str] = None):
    def getter(self):
        if hasattr(self, canonical_name):
            value = getattr(self, canonical_name)
            if value not in [None, ""]:
                return value

        if metadata_key:
            metadata = getattr(self, "metadata", {}) or {}
            if isinstance(metadata, dict):
                return metadata.get(metadata_key, "")

        return ""

    def setter(self, value):
        if hasattr(self, canonical_name):
            setattr(self, canonical_name, value)
        elif metadata_key:
            metadata = getattr(self, "metadata", None)
            if not isinstance(metadata, dict):
                metadata = {}
                setattr(self, "metadata", metadata)
            metadata[metadata_key] = value

    return property(getter, setter)


def _compat_metadata_property(metadata_key: str, default: Any = ""):
    def getter(self):
        metadata = getattr(self, "metadata", {}) or {}
        if isinstance(metadata, dict):
            return metadata.get(metadata_key, default)
        return default

    def setter(self, value):
        metadata = getattr(self, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(self, "metadata", metadata)
        metadata[metadata_key] = value

    return property(getter, setter)


def _compat_to_json(self, ensure_ascii: bool = False, indent: int = 2) -> str:
    if hasattr(self, "to_dict"):
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)
    return json.dumps(json_safe(self), ensure_ascii=ensure_ascii, indent=indent)


def _install_page_raw_compatibility() -> None:
    _compat_wrap_init(
        TextSpanRaw,
        alias_map={
            "size": "font_size",
            "flags": "font_flags",
            "block_no": "block_index",
            "line_no": "line_index",
            "span_no": "span_index",
        },
        metadata_only=["is_bold", "is_italic", "is_upper", "is_heading_candidate"],
    )

    _compat_wrap_init(
        TextLineRaw,
        alias_map={
            "block_no": "block_index",
            "line_no": "line_index",
            "wmode": "writing_mode",
            "dir": "direction",
        },
    )

    _compat_wrap_init(
        TextBlockRaw,
        alias_map={
            "type": "block_type",
            "number": "block_index",
            "block_no": "block_index",
        },
    )

    _compat_wrap_init(
        WordRaw,
        alias_map={
            "block_no": "block_index",
            "line_no": "line_index",
            "word_no": "word_index",
        },
    )

    _compat_wrap_init(
        ImageRaw,
        alias_map={
            "ext": "image_ext",
            "name": "image_name",
            "path": "image_path",
        },
    )

    _compat_wrap_init(
        DrawingRaw,
        alias_map={
            "type": "drawing_type",
            "stroke": "color",
        },
    )

    _compat_wrap_init(
        AnnotationRaw,
        alias_map={
            "annot_type": "annotation_type",
            "type": "annotation_type",
            "text": "content",
        },
    )

    _compat_wrap_init(
        LinkRaw,
        alias_map={
            "kind": "link_type",
            "page": "target_page",
            "from": "bbox",
        },
    )

    _compat_wrap_init(
        FontRaw,
        alias_map={
            "font": "font_name",
            "name": "font_name",
            "size": "font_size",
            "flags": "font_flags",
            "count": "span_count",
        },
    )

    _compat_wrap_init(
        PageRaw,
        alias_map={
            "source_path": "source_document",
            "path": "source_document",
            "file_path": "source_document",
        },
        metadata_only=["file_name", "document_type"],
    )

    # Constructor aliases are handled above; these runtime properties support
    # older code that reads or writes attributes directly.
    TextSpanRaw.size = _compat_alias_property("font_size")
    TextSpanRaw.flags = _compat_alias_property("font_flags")
    TextSpanRaw.block_no = _compat_alias_property("block_index")
    TextSpanRaw.line_no = _compat_alias_property("line_index")
    TextSpanRaw.span_no = _compat_alias_property("span_index")
    TextSpanRaw.is_bold = _compat_metadata_property("is_bold", False)
    TextSpanRaw.is_italic = _compat_metadata_property("is_italic", False)
    TextSpanRaw.is_upper = _compat_metadata_property("is_upper", False)

    TextLineRaw.block_no = _compat_alias_property("block_index")
    TextLineRaw.line_no = _compat_alias_property("line_index")

    TextBlockRaw.block_no = _compat_alias_property("block_index")
    TextBlockRaw.type = _compat_alias_property("block_type")

    WordRaw.block_no = _compat_alias_property("block_index")
    WordRaw.line_no = _compat_alias_property("line_index")
    WordRaw.word_no = _compat_alias_property("word_index")

    ImageRaw.ext = _compat_alias_property("image_ext")
    ImageRaw.name = _compat_alias_property("image_name")
    ImageRaw.path = _compat_alias_property("image_path")

    DrawingRaw.type = _compat_alias_property("drawing_type")
    DrawingRaw.stroke = _compat_alias_property("color")

    AnnotationRaw.annot_type = _compat_alias_property("annotation_type")
    AnnotationRaw.type = _compat_alias_property("annotation_type")
    AnnotationRaw.text = _compat_alias_property("content")

    LinkRaw.kind = _compat_alias_property("link_type")
    LinkRaw.page = _compat_alias_property("target_page")

    FontRaw.font = _compat_alias_property("font_name")
    FontRaw.size = _compat_alias_property("font_size")
    FontRaw.flags = _compat_alias_property("font_flags")
    FontRaw.count = _compat_alias_property("span_count")

    PageRaw.source_path = _compat_alias_property("source_document", "source_path")
    PageRaw.file_path = _compat_alias_property("source_document", "source_path")
    PageRaw.path = _compat_alias_property("source_document", "source_path")
    PageRaw.file_name = _compat_metadata_property("file_name", "")
    PageRaw.document_type = _compat_metadata_property("document_type", "")

    for cls in [
        TextSpanRaw,
        TextLineRaw,
        TextBlockRaw,
        WordRaw,
        ImageRaw,
        DrawingRaw,
        AnnotationRaw,
        LinkRaw,
        FontRaw,
        PageRaw,
    ]:
        if not hasattr(cls, "to_json"):
            cls.to_json = _compat_to_json


_install_page_raw_compatibility()
