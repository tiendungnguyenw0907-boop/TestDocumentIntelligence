"""
page_document_schema.py

Production V1 - Colab Ready

Purpose
-------
Schema definitions for page-level understanding output.

Used by:
- PageUnderstandingPipeline
- ObjectNormalizer
- ObjectMerger
- ReadingOrderBuilder
- RegionDetector
- HeaderFooterDetector
- TableBoundaryDetector
- FigureDetector
- CaptionDetector
- PageLayoutProfiler

Main objects
------------
- PageObject
- ReadingOrderItem
- PageRegion
- HeaderFooterCandidate
- TableCandidate
- FigureCandidate
- CaptionCandidate
- PageLayoutProfile
- PageDocument
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, asdict, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


SCHEMA_VERSION = "page_document_schema_v1"


def make_id(prefix: str = "page_doc") -> str:
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


def normalize_text_for_match(text: Any) -> str:
    text = normalize_text(text).lower()

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


def normalize_page_number(value: Any, default: int = 1) -> int:
    try:
        page_number = int(value)

        if page_number > 0:
            return page_number

        return default
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
class PageObject:
    object_id: str = field(default_factory=lambda: make_id("obj"))
    object_type: str = "object"

    text: str = ""
    normalized_text: str = ""

    bbox: List[float] = field(default_factory=list)

    page_number: int = 1
    page_index: int = 0

    source_id: str = ""
    source_type: str = ""
    source: str = "page_document_schema"

    reading_order: Optional[int] = None

    region_id: str = ""
    region_type: str = ""

    column_index: Optional[int] = None
    line_index: Optional[int] = None
    block_index: Optional[int] = None

    confidence: float = 0.70

    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text)
        else:
            self.normalized_text = normalize_text_for_match(self.normalized_text)

        self.bbox = normalize_bbox(self.bbox)
        self.page_number = normalize_page_number(self.page_number)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))

        if self.reading_order is not None:
            self.reading_order = safe_int(self.reading_order, default=0)

        if self.column_index is not None:
            self.column_index = safe_int(self.column_index, default=0)

        if self.line_index is not None:
            self.line_index = safe_int(self.line_index, default=0)

        if self.block_index is not None:
            self.block_index = safe_int(self.block_index, default=0)

        self.object_type = normalize_text(self.object_type) or "object"
        self.source_type = normalize_text(self.source_type)
        self.region_type = normalize_text(self.region_type)
        self.confidence = clamp_float(self.confidence, default=0.70)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageObject":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)


@dataclass
class ReadingOrderItem:
    item_id: str = field(default_factory=lambda: make_id("ro_item"))
    item_type: str = "text"

    text: str = ""
    normalized_text: str = ""

    bbox: List[float] = field(default_factory=list)

    page_number: int = 1
    page_index: int = 0

    order: int = 0
    column_index: int = 0

    source_object_id: str = ""
    source_object_ids: List[str] = field(default_factory=list)

    confidence: float = 0.70
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text)
        else:
            self.normalized_text = normalize_text_for_match(self.normalized_text)

        self.bbox = normalize_bbox(self.bbox)
        self.page_number = normalize_page_number(self.page_number)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.order = safe_int(self.order, default=0)
        self.column_index = safe_int(self.column_index, default=0)
        self.item_type = normalize_text(self.item_type) or "text"

        if not self.source_object_ids and self.source_object_id:
            self.source_object_ids = [self.source_object_id]

        self.source_object_ids = [
            str(item) for item in self.source_object_ids if item
        ]

        self.confidence = clamp_float(self.confidence, default=0.70)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReadingOrderItem":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)


@dataclass
class PageRegion:
    region_id: str = field(default_factory=lambda: make_id("region"))
    region_type: str = "body"

    label: str = ""

    bbox: List[float] = field(default_factory=list)

    page_number: int = 1
    page_index: int = 0

    object_ids: List[str] = field(default_factory=list)

    text_preview: str = ""
    confidence: float = 0.70

    source: str = "page_document_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.region_type = normalize_text(self.region_type) or "body"
        self.label = normalize_text(self.label) or self.region_type
        self.bbox = normalize_bbox(self.bbox)
        self.page_number = normalize_page_number(self.page_number)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.object_ids = [str(item) for item in self.object_ids if item]
        self.text_preview = normalize_text(self.text_preview)[:1000]
        self.confidence = clamp_float(self.confidence, default=0.70)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageRegion":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)


@dataclass
class HeaderFooterCandidate:
    candidate_id: str = field(default_factory=lambda: make_id("hf"))
    candidate_type: str = "header_footer"

    role: str = ""
    text: str = ""
    normalized_text: str = ""

    bbox: List[float] = field(default_factory=list)

    page_number: int = 1
    page_index: int = 0

    is_header: bool = False
    is_footer: bool = False
    is_page_number: bool = False

    confidence: float = 0.60
    source: str = "page_document_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.role = normalize_text(self.role)

        if not self.role:
            if self.is_header:
                self.role = "header"
            elif self.is_footer:
                self.role = "footer"
            else:
                self.role = self.candidate_type

        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text)
        else:
            self.normalized_text = normalize_text_for_match(self.normalized_text)

        self.bbox = normalize_bbox(self.bbox)
        self.page_number = normalize_page_number(self.page_number)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.confidence = clamp_float(self.confidence, default=0.60)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HeaderFooterCandidate":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)


@dataclass
class TableCandidate:
    table_candidate_id: str = field(default_factory=lambda: make_id("table_cand"))

    bbox: List[float] = field(default_factory=list)

    page_number: int = 1
    page_index: int = 0

    row_count_estimate: int = 0
    col_count_estimate: int = 0

    detection_method: str = ""
    candidate_type: str = "table_candidate"

    caption_id: str = ""
    caption_text: str = ""

    confidence: float = 0.60
    source: str = "page_document_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.bbox = normalize_bbox(self.bbox)
        self.page_number = normalize_page_number(self.page_number)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.row_count_estimate = safe_int(self.row_count_estimate, default=0)
        self.col_count_estimate = safe_int(self.col_count_estimate, default=0)
        self.detection_method = normalize_text(self.detection_method)
        self.candidate_type = normalize_text(self.candidate_type) or "table_candidate"
        self.caption_text = normalize_text(self.caption_text)
        self.confidence = clamp_float(self.confidence, default=0.60)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableCandidate":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        aliases = {
            "table_boundary_id": "table_candidate_id",
            "table_id": "table_candidate_id",
            "row_count": "row_count_estimate",
            "col_count": "col_count_estimate",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)


@dataclass
class FigureCandidate:
    figure_candidate_id: str = field(default_factory=lambda: make_id("figure"))

    bbox: List[float] = field(default_factory=list)

    page_number: int = 1
    page_index: int = 0

    figure_type: str = "figure"

    caption_id: str = ""
    caption_text: str = ""

    image_ids: List[str] = field(default_factory=list)
    drawing_ids: List[str] = field(default_factory=list)

    confidence: float = 0.60
    source: str = "page_document_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.bbox = normalize_bbox(self.bbox)
        self.page_number = normalize_page_number(self.page_number)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.figure_type = normalize_text(self.figure_type) or "figure"
        self.caption_text = normalize_text(self.caption_text)
        self.image_ids = [str(item) for item in self.image_ids if item]
        self.drawing_ids = [str(item) for item in self.drawing_ids if item]
        self.confidence = clamp_float(self.confidence, default=0.60)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FigureCandidate":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        aliases = {
            "figure_id": "figure_candidate_id",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)


@dataclass
class CaptionCandidate:
    caption_id: str = field(default_factory=lambda: make_id("caption"))

    text: str = ""
    normalized_text: str = ""

    bbox: List[float] = field(default_factory=list)

    page_number: int = 1
    page_index: int = 0

    caption_type: str = "caption"
    target_type: str = ""
    target_id: str = ""

    target_bbox: List[float] = field(default_factory=list)

    confidence: float = 0.60
    source: str = "page_document_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text)
        else:
            self.normalized_text = normalize_text_for_match(self.normalized_text)

        self.bbox = normalize_bbox(self.bbox)
        self.target_bbox = normalize_bbox(self.target_bbox)
        self.page_number = normalize_page_number(self.page_number)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.caption_type = normalize_text(self.caption_type) or "caption"
        self.target_type = normalize_text(self.target_type)
        self.confidence = clamp_float(self.confidence, default=0.60)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaptionCandidate":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)


@dataclass
class PageLayoutProfile:
    profile_id: str = field(default_factory=lambda: make_id("layout_profile"))

    page_number: int = 1
    page_index: int = 0

    width: float = 0.0
    height: float = 0.0
    rotation: int = 0

    orientation: str = "portrait"
    layout_type: str = "normal_layout"
    page_kind: str = "digital_text_page"
    complexity_level: str = "low"

    column_count: int = 1

    text_area_ratio: float = 0.0
    image_area_ratio: float = 0.0
    table_area_ratio: float = 0.0
    drawing_area_ratio: float = 0.0

    object_count: int = 0
    text_object_count: int = 0
    image_object_count: int = 0
    drawing_object_count: int = 0
    table_candidate_count: int = 0
    figure_candidate_count: int = 0

    processing_strategy: str = "standard_page_understanding"

    quality_flags: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    confidence: float = 0.70
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = normalize_page_number(self.page_number)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.width = safe_float(self.width, default=0.0)
        self.height = safe_float(self.height, default=0.0)
        self.rotation = safe_int(self.rotation, default=0)

        if self.width > self.height:
            self.orientation = "landscape"
        elif self.height > 0:
            self.orientation = "portrait"

        self.column_count = max(1, safe_int(self.column_count, default=1))
        self.object_count = safe_int(self.object_count, default=0)
        self.text_object_count = safe_int(self.text_object_count, default=0)
        self.image_object_count = safe_int(self.image_object_count, default=0)
        self.drawing_object_count = safe_int(self.drawing_object_count, default=0)
        self.table_candidate_count = safe_int(self.table_candidate_count, default=0)
        self.figure_candidate_count = safe_int(self.figure_candidate_count, default=0)

        self.text_area_ratio = clamp_float(self.text_area_ratio, default=0.0)
        self.image_area_ratio = clamp_float(self.image_area_ratio, default=0.0)
        self.table_area_ratio = clamp_float(self.table_area_ratio, default=0.0)
        self.drawing_area_ratio = clamp_float(self.drawing_area_ratio, default=0.0)

        self.layout_type = normalize_text(self.layout_type) or "normal_layout"
        self.page_kind = normalize_text(self.page_kind) or "digital_text_page"
        self.complexity_level = normalize_text(self.complexity_level) or "low"
        self.processing_strategy = normalize_text(self.processing_strategy) or "standard_page_understanding"
        self.confidence = clamp_float(self.confidence, default=0.70)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageLayoutProfile":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)


@dataclass
class PageDocument:
    document_id: str = ""
    source_document: str = ""

    page_number: int = 1
    page_index: int = 0

    width: float = 0.0
    height: float = 0.0
    rotation: int = 0

    raw_text: str = ""
    normalized_text: str = ""

    objects: List[PageObject] = field(default_factory=list)
    merged_objects: List[PageObject] = field(default_factory=list)

    reading_order_items: List[ReadingOrderItem] = field(default_factory=list)
    reading_order_text: str = ""

    regions: List[PageRegion] = field(default_factory=list)
    header_footer_candidates: List[HeaderFooterCandidate] = field(default_factory=list)

    table_candidates: List[TableCandidate] = field(default_factory=list)
    figure_candidates: List[FigureCandidate] = field(default_factory=list)
    caption_candidates: List[CaptionCandidate] = field(default_factory=list)

    layout_profile: Optional[PageLayoutProfile] = None

    page_summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    processor: str = "PageDocument"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = normalize_page_number(self.page_number)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.width = safe_float(self.width, default=0.0)
        self.height = safe_float(self.height, default=0.0)
        self.rotation = safe_int(self.rotation, default=0)

        self.raw_text = normalize_text(self.raw_text)

        if not self.normalized_text:
            self.normalized_text = normalize_text(self.raw_text)
        else:
            self.normalized_text = normalize_text(self.normalized_text)

        self.reading_order_text = normalize_text(self.reading_order_text)

        self.objects = [
            PageObject.from_dict(item) if isinstance(item, dict) else item
            for item in self.objects
        ]

        self.merged_objects = [
            PageObject.from_dict(item) if isinstance(item, dict) else item
            for item in self.merged_objects
        ]

        self.reading_order_items = [
            ReadingOrderItem.from_dict(item) if isinstance(item, dict) else item
            for item in self.reading_order_items
        ]

        self.regions = [
            PageRegion.from_dict(item) if isinstance(item, dict) else item
            for item in self.regions
        ]

        self.header_footer_candidates = [
            HeaderFooterCandidate.from_dict(item) if isinstance(item, dict) else item
            for item in self.header_footer_candidates
        ]

        self.table_candidates = [
            TableCandidate.from_dict(item) if isinstance(item, dict) else item
            for item in self.table_candidates
        ]

        self.figure_candidates = [
            FigureCandidate.from_dict(item) if isinstance(item, dict) else item
            for item in self.figure_candidates
        ]

        self.caption_candidates = [
            CaptionCandidate.from_dict(item) if isinstance(item, dict) else item
            for item in self.caption_candidates
        ]

        if isinstance(self.layout_profile, dict):
            self.layout_profile = PageLayoutProfile.from_dict(self.layout_profile)

        if self.layout_profile is None:
            self.layout_profile = self.infer_layout_profile()

        if not self.page_summary:
            self.page_summary = self.summary()

    def infer_layout_profile(self) -> PageLayoutProfile:
        page_area = max(self.width * self.height, 1.0)

        text_bboxes = [
            item.bbox for item in self.objects + self.merged_objects
            if item.object_type in ["text", "text_line", "text_block", "paragraph", "merged_paragraph"]
        ]

        image_bboxes = [
            item.bbox for item in self.objects + self.merged_objects
            if item.object_type in ["image", "figure_image"]
        ]

        drawing_bboxes = [
            item.bbox for item in self.objects + self.merged_objects
            if item.object_type in ["drawing", "line", "shape"]
        ]

        table_bboxes = [
            item.bbox for item in self.table_candidates
        ]

        text_area = sum(bbox_area(bbox) for bbox in text_bboxes)
        image_area = sum(bbox_area(bbox) for bbox in image_bboxes)
        drawing_area = sum(bbox_area(bbox) for bbox in drawing_bboxes)
        table_area = sum(bbox_area(bbox) for bbox in table_bboxes)

        object_count = len(self.objects) + len(self.merged_objects)

        layout_type = "normal_layout"

        if self.table_candidates:
            layout_type = "table_layout"
        elif image_area / page_area > 0.40:
            layout_type = "image_heavy_layout"
        elif len(self.regions) >= 5:
            layout_type = "multi_region_layout"

        complexity_score = 0

        if object_count > 100:
            complexity_score += 1

        if len(self.table_candidates) > 0:
            complexity_score += 1

        if len(self.figure_candidates) > 1:
            complexity_score += 1

        if len(self.regions) > 5:
            complexity_score += 1

        if complexity_score >= 3:
            complexity_level = "high"
        elif complexity_score >= 1:
            complexity_level = "medium"
        else:
            complexity_level = "low"

        return PageLayoutProfile(
            page_number=self.page_number,
            page_index=self.page_index,
            width=self.width,
            height=self.height,
            rotation=self.rotation,
            layout_type=layout_type,
            page_kind="page_document",
            complexity_level=complexity_level,
            column_count=self._infer_column_count(),
            text_area_ratio=round(text_area / page_area, 4),
            image_area_ratio=round(image_area / page_area, 4),
            table_area_ratio=round(table_area / page_area, 4),
            drawing_area_ratio=round(drawing_area / page_area, 4),
            object_count=object_count,
            text_object_count=len(text_bboxes),
            image_object_count=len(image_bboxes),
            drawing_object_count=len(drawing_bboxes),
            table_candidate_count=len(self.table_candidates),
            figure_candidate_count=len(self.figure_candidates),
            processing_strategy="page_understanding",
            confidence=0.65,
            metadata={
                "inferred": True,
            },
        )

    def _infer_column_count(self) -> int:
        items = self.reading_order_items or []

        if not items:
            return 1

        column_indexes = {
            item.column_index
            for item in items
            if item.column_index is not None
        }

        if not column_indexes:
            return 1

        return max(1, len(column_indexes))

    def summary(self) -> Dict[str, Any]:
        by_object_type: Dict[str, int] = {}
        by_region_type: Dict[str, int] = {}

        for item in self.objects + self.merged_objects:
            by_object_type[item.object_type] = by_object_type.get(item.object_type, 0) + 1

        for region in self.regions:
            by_region_type[region.region_type] = by_region_type.get(region.region_type, 0) + 1

        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "page_index": self.page_index,
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "text_length": len(self.normalized_text or self.raw_text or ""),
            "object_count": len(self.objects),
            "merged_object_count": len(self.merged_objects),
            "reading_order_item_count": len(self.reading_order_items),
            "region_count": len(self.regions),
            "header_footer_candidate_count": len(self.header_footer_candidates),
            "table_candidate_count": len(self.table_candidates),
            "figure_candidate_count": len(self.figure_candidates),
            "caption_candidate_count": len(self.caption_candidates),
            "by_object_type": by_object_type,
            "by_region_type": by_region_type,
            "layout_type": self.layout_profile.layout_type if self.layout_profile else "",
            "complexity_level": self.layout_profile.complexity_level if self.layout_profile else "",
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
        }

    def objects_by_type(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in self.objects:
            grouped.setdefault(item.object_type, [])
            grouped[item.object_type].append(item.to_dict())

        return grouped

    def regions_by_type(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in self.regions:
            grouped.setdefault(item.region_type, [])
            grouped[item.region_type].append(item.to_dict())

        return grouped

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
            "objects": [item.to_dict() for item in self.objects],
            "merged_objects": [item.to_dict() for item in self.merged_objects],
            "reading_order_items": [item.to_dict() for item in self.reading_order_items],
            "reading_order_text": self.reading_order_text,
            "regions": [item.to_dict() for item in self.regions],
            "header_footer_candidates": [item.to_dict() for item in self.header_footer_candidates],
            "table_candidates": [item.to_dict() for item in self.table_candidates],
            "figure_candidates": [item.to_dict() for item in self.figure_candidates],
            "caption_candidates": [item.to_dict() for item in self.caption_candidates],
            "layout_profile": self.layout_profile.to_dict() if self.layout_profile else {},
            "objects_by_type": self.objects_by_type(),
            "regions_by_type": self.regions_by_type(),
            "page_summary": self.summary(),
            "metadata": json_safe(self.metadata),
            "warnings": self.warnings,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageDocument":
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

        if not self.objects and not self.normalized_text:
            warnings.append("page has no objects and no text")

        for item in self.table_candidates:
            if not item.bbox:
                warnings.append(f"table_candidate:{item.table_candidate_id}: missing bbox")

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
    def load_json(cls, input_path: Union[str, Path]) -> "PageDocument":
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)


def make_page_document(
    document_id: str = "",
    page_number: int = 1,
    page_index: Optional[int] = None,
    width: float = 0.0,
    height: float = 0.0,
    raw_text: str = "",
    normalized_text: str = "",
    objects: Optional[List[Union[PageObject, Dict[str, Any]]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PageDocument:
    return PageDocument(
        document_id=document_id,
        page_number=page_number,
        page_index=max(page_number - 1, 0) if page_index is None else page_index,
        width=width,
        height=height,
        raw_text=raw_text,
        normalized_text=normalized_text,
        objects=objects or [],
        metadata=metadata or {},
    )


def page_document_from_dict(data: Dict[str, Any]) -> PageDocument:
    return PageDocument.from_dict(data)


def page_document_to_dict(
    page_document: Union[PageDocument, Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(page_document, dict):
        return json_safe(page_document)

    return page_document.to_dict()


def save_page_document_json(
    page_document: Union[PageDocument, Dict[str, Any]],
    output_path: Union[str, Path],
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(page_document, PageDocument):
        data = page_document.to_dict()
    else:
        data = json_safe(page_document)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return str(output_path)


def load_page_document_json(
    input_path: Union[str, Path],
) -> PageDocument:
    return PageDocument.load_json(input_path)


PageDocumentSchema = PageDocument
PageObjectSchema = PageObject
ReadingOrderItemSchema = ReadingOrderItem
PageRegionSchema = PageRegion
