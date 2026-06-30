"""
table_schema.py

Production V1 - Colab Ready

Purpose
-------
Schema definitions for table understanding.

Used by:
- TableBoundaryDetector
- TableGridBuilder
- TableStructureRecognizer
- TableCellExtractor
- TableHeaderDetector
- TableSpanDetector
- TableSemanticRecognizer
- MultiPageTableDetector
- TableUnderstandingPipeline

Main objects
------------
- TableBoundary
- TableGridCell
- TableGrid
- TableCell
- TableHeader
- TableSpan
- TableRecord
- TableSemantic
- MultiPageTableSegment
- MultiPageTable
- TableCollection
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, asdict, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


SCHEMA_VERSION = "table_schema_v1"


def make_id(prefix: str = "table") -> str:
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
        return [
            round(float(bbox[0]), 4),
            round(float(bbox[1]), 4),
            round(float(bbox[2]), 4),
            round(float(bbox[3]), 4),
        ]
    except Exception:
        return []


def bbox_area(bbox: Any) -> float:
    bbox = normalize_bbox(bbox)

    if len(bbox) != 4:
        return 0.0

    return round(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]), 4)


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
class TableBoundary:
    table_boundary_id: str = field(default_factory=lambda: make_id("boundary"))

    page_number: int = 1
    page_index: int = 0

    bbox: List[float] = field(default_factory=list)

    detection_method: str = ""
    boundary_type: str = "table_boundary"

    row_count_estimate: int = 0
    col_count_estimate: int = 0

    caption_id: str = ""
    caption_text: str = ""

    confidence: float = 0.60
    source: str = "table_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.bbox = normalize_bbox(self.bbox)
        self.detection_method = normalize_text(self.detection_method)
        self.boundary_type = normalize_text(self.boundary_type) or "table_boundary"
        self.row_count_estimate = safe_int(self.row_count_estimate, default=0)
        self.col_count_estimate = safe_int(self.col_count_estimate, default=0)
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
    def from_dict(cls, data: Dict[str, Any]) -> "TableBoundary":
        data = dict(data or {})

        aliases = {
            "table_id": "table_boundary_id",
            "id": "table_boundary_id",
            "row_count": "row_count_estimate",
            "col_count": "col_count_estimate",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class TableGridCell:
    cell_id: str = field(default_factory=lambda: make_id("grid_cell"))

    table_grid_id: str = ""

    page_number: int = 1
    page_index: int = 0

    row_index: int = 0
    col_index: int = 0

    row_span: int = 1
    col_span: int = 1

    bbox: List[float] = field(default_factory=list)

    text: str = ""
    normalized_text: str = ""

    is_header: bool = False
    is_stub: bool = False
    is_empty: bool = False

    confidence: float = 0.70
    source: str = "table_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.row_index = safe_int(self.row_index, default=0)
        self.col_index = safe_int(self.col_index, default=0)
        self.row_span = max(1, safe_int(self.row_span, default=1))
        self.col_span = max(1, safe_int(self.col_span, default=1))
        self.bbox = normalize_bbox(self.bbox)

        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text)
        else:
            self.normalized_text = normalize_text_for_match(self.normalized_text)

        self.is_empty = self.is_empty or not bool(self.text)
        self.confidence = clamp_float(self.confidence, default=0.70)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableGridCell":
        data = dict(data or {})

        aliases = {
            "grid_cell_id": "cell_id",
            "table_cell_id": "cell_id",
            "row": "row_index",
            "col": "col_index",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class TableGrid:
    table_grid_id: str = field(default_factory=lambda: make_id("grid"))

    table_boundary_id: str = ""

    page_number: int = 1
    page_index: int = 0

    bbox: List[float] = field(default_factory=list)

    row_count: int = 0
    col_count: int = 0

    rows: List[List[str]] = field(default_factory=list)
    cells: List[TableGridCell] = field(default_factory=list)

    x_positions: List[float] = field(default_factory=list)
    y_positions: List[float] = field(default_factory=list)

    grid_type: str = "detected_grid"
    detection_method: str = ""

    confidence: float = 0.65
    source: str = "table_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.bbox = normalize_bbox(self.bbox)
        self.row_count = safe_int(self.row_count, default=0)
        self.col_count = safe_int(self.col_count, default=0)

        self.cells = [
            TableGridCell.from_dict(item) if isinstance(item, dict) else item
            for item in self.cells
        ]

        if self.cells:
            if self.row_count <= 0:
                self.row_count = max(cell.row_index for cell in self.cells) + 1

            if self.col_count <= 0:
                self.col_count = max(cell.col_index for cell in self.cells) + 1

        self.x_positions = [safe_float(item, default=0.0) for item in self.x_positions]
        self.y_positions = [safe_float(item, default=0.0) for item in self.y_positions]
        self.grid_type = normalize_text(self.grid_type) or "detected_grid"
        self.detection_method = normalize_text(self.detection_method)
        self.confidence = clamp_float(self.confidence, default=0.65)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def matrix(self) -> List[List[Dict[str, Any]]]:
        matrix = [
            [
                {}
                for _ in range(max(self.col_count, 0))
            ]
            for _ in range(max(self.row_count, 0))
        ]

        for cell in self.cells:
            if 0 <= cell.row_index < self.row_count and 0 <= cell.col_index < self.col_count:
                matrix[cell.row_index][cell.col_index] = cell.to_dict()

        return matrix

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        data["matrix"] = self.matrix()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableGrid":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class TableCell:
    table_cell_id: str = field(default_factory=lambda: make_id("cell"))

    table_grid_id: str = ""
    table_structure_id: str = ""
    table_semantic_id: str = ""

    page_number: int = 1
    page_index: int = 0

    row_index: int = 0
    col_index: int = 0

    row_span: int = 1
    col_span: int = 1

    bbox: List[float] = field(default_factory=list)

    text: str = ""
    normalized_text: str = ""

    role: str = "body"
    data_type: str = "text"

    is_header: bool = False
    is_stub: bool = False
    is_numeric: bool = False
    is_empty: bool = False

    confidence: float = 0.70
    source: str = "table_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.row_index = safe_int(self.row_index, default=0)
        self.col_index = safe_int(self.col_index, default=0)
        self.row_span = max(1, safe_int(self.row_span, default=1))
        self.col_span = max(1, safe_int(self.col_span, default=1))
        self.bbox = normalize_bbox(self.bbox)

        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text)
        else:
            self.normalized_text = normalize_text_for_match(self.normalized_text)

        self.role = normalize_text(self.role) or "body"
        self.data_type = normalize_text(self.data_type) or self.infer_data_type()

        self.is_empty = self.is_empty or not bool(self.text)
        self.is_numeric = self.is_numeric or self.data_type in ["number", "money", "percentage"]

        if self.role in ["header", "column_header", "row_header"]:
            self.is_header = True

        if self.role in ["stub", "row_header"]:
            self.is_stub = True

        self.confidence = clamp_float(self.confidence, default=0.70)

    def infer_data_type(self) -> str:
        text = self.text.strip()

        if not text:
            return "empty"

        if re.fullmatch(r"\d+(?:[,.]\d+)?\s*%", text):
            return "percentage"

        if re.fullmatch(r"\d[\d\.,]*\s*(?:VNĐ|VND|đồng|đ|tỷ|triệu)?", text, flags=re.IGNORECASE):
            return "number"

        if re.search(r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b", text):
            return "date"

        return "text"

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableCell":
        data = dict(data or {})

        aliases = {
            "cell_id": "table_cell_id",
            "grid_cell_id": "table_cell_id",
            "row": "row_index",
            "col": "col_index",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class TableHeader:
    table_header_id: str = field(default_factory=lambda: make_id("header"))

    table_grid_id: str = ""
    table_structure_id: str = ""
    table_semantic_id: str = ""

    page_number: int = 1
    page_index: int = 0

    header_row_indices: List[int] = field(default_factory=list)
    header_col_indices: List[int] = field(default_factory=list)

    column_headers: List[str] = field(default_factory=list)
    row_headers: List[str] = field(default_factory=list)

    header_cells: List[TableCell] = field(default_factory=list)

    confidence: float = 0.70
    source: str = "table_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))

        self.header_row_indices = sorted(list(dict.fromkeys([
            safe_int(item, default=-1)
            for item in self.header_row_indices
            if safe_int(item, default=-1) >= 0
        ])))

        self.header_col_indices = sorted(list(dict.fromkeys([
            safe_int(item, default=-1)
            for item in self.header_col_indices
            if safe_int(item, default=-1) >= 0
        ])))

        self.column_headers = [
            normalize_text(item)
            for item in self.column_headers
            if normalize_text(item)
        ]

        self.row_headers = [
            normalize_text(item)
            for item in self.row_headers
            if normalize_text(item)
        ]

        self.header_cells = [
            TableCell.from_dict(item) if isinstance(item, dict) else item
            for item in self.header_cells
        ]

        self.confidence = clamp_float(self.confidence, default=0.70)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableHeader":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class TableSpan:
    table_span_id: str = field(default_factory=lambda: make_id("span"))

    table_grid_id: str = ""
    table_cell_id: str = ""

    page_number: int = 1
    page_index: int = 0

    row_index: int = 0
    col_index: int = 0
    row_span: int = 1
    col_span: int = 1

    span_type: str = "cell_span"

    covered_cell_ids: List[str] = field(default_factory=list)
    covered_positions: List[Dict[str, int]] = field(default_factory=list)

    confidence: float = 0.65
    source: str = "table_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.row_index = safe_int(self.row_index, default=0)
        self.col_index = safe_int(self.col_index, default=0)
        self.row_span = max(1, safe_int(self.row_span, default=1))
        self.col_span = max(1, safe_int(self.col_span, default=1))
        self.span_type = normalize_text(self.span_type) or "cell_span"
        self.covered_cell_ids = [str(item) for item in self.covered_cell_ids if item]
        self.confidence = clamp_float(self.confidence, default=0.65)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableSpan":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class TableRecord:
    table_record_id: str = field(default_factory=lambda: make_id("record"))

    table_grid_id: str = ""
    table_semantic_id: str = ""

    page_number: int = 1
    page_index: int = 0

    row_index: int = 0
    record_index: int = 0

    values: Dict[str, Any] = field(default_factory=dict)
    raw_values: Dict[str, Any] = field(default_factory=dict)

    cell_ids: List[str] = field(default_factory=list)

    confidence: float = 0.70
    source: str = "table_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.row_index = safe_int(self.row_index, default=0)
        self.record_index = safe_int(self.record_index, default=0)
        self.cell_ids = [str(item) for item in self.cell_ids if item]

        if not self.raw_values and self.values:
            self.raw_values = dict(self.values)

        if not self.values and self.raw_values:
            self.values = dict(self.raw_values)

        self.confidence = clamp_float(self.confidence, default=0.70)

    def text(self) -> str:
        parts = []

        for key, value in self.values.items():
            value_text = normalize_text(value)

            if value_text:
                parts.append(f"{key}: {value_text}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["text"] = self.text()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableRecord":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class TableSemantic:
    table_semantic_id: str = field(default_factory=lambda: make_id("semantic"))

    table_grid_id: str = ""
    table_structure_id: str = ""
    table_boundary_id: str = ""

    page_number: int = 1
    page_index: int = 0

    bbox: List[float] = field(default_factory=list)

    title: str = ""
    caption: str = ""

    semantic_type: str = "general_data"
    table_type: str = "table"

    row_count: int = 0
    col_count: int = 0

    column_headers: List[str] = field(default_factory=list)
    numeric_columns: List[int] = field(default_factory=list)
    date_columns: List[int] = field(default_factory=list)
    key_columns: List[int] = field(default_factory=list)

    records: List[TableRecord] = field(default_factory=list)

    confidence: float = 0.70
    source: str = "table_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.bbox = normalize_bbox(self.bbox)
        self.title = normalize_text(self.title)
        self.caption = normalize_text(self.caption)
        self.semantic_type = normalize_text(self.semantic_type) or "general_data"
        self.table_type = normalize_text(self.table_type) or "table"
        self.row_count = safe_int(self.row_count, default=0)
        self.col_count = safe_int(self.col_count, default=0)

        self.column_headers = [
            normalize_text(item)
            for item in self.column_headers
            if normalize_text(item)
        ]

        self.numeric_columns = sorted(list(dict.fromkeys([
            safe_int(item, default=-1)
            for item in self.numeric_columns
            if safe_int(item, default=-1) >= 0
        ])))

        self.date_columns = sorted(list(dict.fromkeys([
            safe_int(item, default=-1)
            for item in self.date_columns
            if safe_int(item, default=-1) >= 0
        ])))

        self.key_columns = sorted(list(dict.fromkeys([
            safe_int(item, default=-1)
            for item in self.key_columns
            if safe_int(item, default=-1) >= 0
        ])))

        self.records = [
            TableRecord.from_dict(item) if isinstance(item, dict) else item
            for item in self.records
        ]

        self.confidence = clamp_float(self.confidence, default=0.70)

    @property
    def area(self) -> float:
        return bbox_area(self.bbox)

    def text(self) -> str:
        lines = []

        if self.title:
            lines.append(self.title)

        if self.caption and self.caption != self.title:
            lines.append(self.caption)

        if self.column_headers:
            lines.append(" | ".join(self.column_headers))

        for record in self.records:
            record_text = record.text()

            if record_text:
                lines.append(record_text)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["area"] = self.area
        data["text"] = self.text()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableSemantic":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class MultiPageTableSegment:
    segment_id: str = field(default_factory=lambda: make_id("multi_seg"))

    multi_page_table_id: str = ""

    table_grid_id: str = ""
    table_structure_id: str = ""
    table_semantic_id: str = ""
    table_boundary_id: str = ""

    page_number: int = 1
    page_index: int = 0

    segment_index: int = 0
    segment_type: str = "middle"

    bbox: List[float] = field(default_factory=list)

    row_count: int = 0
    col_count: int = 0

    header_rows: List[int] = field(default_factory=list)
    column_headers: List[str] = field(default_factory=list)

    continuation_score: float = 0.60

    source: str = "table_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.page_number = safe_int(self.page_number, default=1)
        self.page_index = safe_int(self.page_index, default=max(self.page_number - 1, 0))
        self.segment_index = safe_int(self.segment_index, default=0)
        self.segment_type = normalize_text(self.segment_type) or "middle"
        self.bbox = normalize_bbox(self.bbox)
        self.row_count = safe_int(self.row_count, default=0)
        self.col_count = safe_int(self.col_count, default=0)

        self.header_rows = sorted(list(dict.fromkeys([
            safe_int(item, default=-1)
            for item in self.header_rows
            if safe_int(item, default=-1) >= 0
        ])))

        self.column_headers = [
            normalize_text(item)
            for item in self.column_headers
            if normalize_text(item)
        ]

        self.continuation_score = clamp_float(self.continuation_score, default=0.60)

    def table_id(self) -> str:
        return (
            self.table_semantic_id
            or self.table_structure_id
            or self.table_grid_id
            or self.table_boundary_id
            or ""
        )

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["table_id"] = self.table_id()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiPageTableSegment":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class MultiPageTable:
    multi_page_table_id: str = field(default_factory=lambda: make_id("multi_table"))

    page_start: Optional[int] = None
    page_end: Optional[int] = None
    page_numbers: List[int] = field(default_factory=list)

    table_grid_ids: List[str] = field(default_factory=list)
    table_structure_ids: List[str] = field(default_factory=list)
    table_semantic_ids: List[str] = field(default_factory=list)
    table_boundary_ids: List[str] = field(default_factory=list)

    segments: List[MultiPageTableSegment] = field(default_factory=list)

    total_row_count: int = 0
    col_count: int = 0

    column_headers: List[str] = field(default_factory=list)

    table_type: str = "table"
    semantic_type: str = "general_data"

    confidence: float = 0.70
    source: str = "table_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.segments = [
            MultiPageTableSegment.from_dict(item) if isinstance(item, dict) else item
            for item in self.segments
        ]

        self.page_numbers = normalize_page_numbers(self.page_numbers)

        if not self.page_numbers and self.segments:
            self.page_numbers = normalize_page_numbers([
                segment.page_number
                for segment in self.segments
            ])

        if self.page_numbers:
            if self.page_start is None:
                self.page_start = min(self.page_numbers)

            if self.page_end is None:
                self.page_end = max(self.page_numbers)

        self.table_grid_ids = self._unique_strings(self.table_grid_ids)
        self.table_structure_ids = self._unique_strings(self.table_structure_ids)
        self.table_semantic_ids = self._unique_strings(self.table_semantic_ids)
        self.table_boundary_ids = self._unique_strings(self.table_boundary_ids)

        if self.segments:
            self.table_grid_ids = self._unique_strings(self.table_grid_ids + [s.table_grid_id for s in self.segments])
            self.table_structure_ids = self._unique_strings(self.table_structure_ids + [s.table_structure_id for s in self.segments])
            self.table_semantic_ids = self._unique_strings(self.table_semantic_ids + [s.table_semantic_id for s in self.segments])
            self.table_boundary_ids = self._unique_strings(self.table_boundary_ids + [s.table_boundary_id for s in self.segments])

        self.total_row_count = safe_int(self.total_row_count, default=0)

        if self.total_row_count <= 0 and self.segments:
            self.total_row_count = sum(safe_int(segment.row_count, default=0) for segment in self.segments)

        self.col_count = safe_int(self.col_count, default=0)

        if self.col_count <= 0 and self.segments:
            col_counts = [
                segment.col_count
                for segment in self.segments
                if segment.col_count > 0
            ]

            if col_counts:
                self.col_count = max(set(col_counts), key=col_counts.count)

        self.column_headers = [
            normalize_text(item)
            for item in self.column_headers
            if normalize_text(item)
        ]

        if not self.column_headers and self.segments:
            for segment in self.segments:
                if segment.column_headers:
                    self.column_headers = segment.column_headers
                    break

        self.table_type = normalize_text(self.table_type) or "table"
        self.semantic_type = normalize_text(self.semantic_type) or "general_data"
        self.confidence = clamp_float(self.confidence, default=0.70)

    def _unique_strings(self, values: List[Any]) -> List[str]:
        result = []

        for value in values or []:
            value = str(value or "").strip()

            if value and value not in result:
                result.append(value)

        return result

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiPageTable":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class TableCollection:
    document_id: str = ""
    source_document: str = ""

    table_boundaries: List[TableBoundary] = field(default_factory=list)
    table_grids: List[TableGrid] = field(default_factory=list)
    table_cells: List[TableCell] = field(default_factory=list)
    table_headers: List[TableHeader] = field(default_factory=list)
    table_spans: List[TableSpan] = field(default_factory=list)
    table_records: List[TableRecord] = field(default_factory=list)
    table_semantics: List[TableSemantic] = field(default_factory=list)
    multi_page_tables: List[MultiPageTable] = field(default_factory=list)
    multi_page_table_segments: List[MultiPageTableSegment] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.table_boundaries = [
            TableBoundary.from_dict(item) if isinstance(item, dict) else item
            for item in self.table_boundaries
        ]

        self.table_grids = [
            TableGrid.from_dict(item) if isinstance(item, dict) else item
            for item in self.table_grids
        ]

        self.table_cells = [
            TableCell.from_dict(item) if isinstance(item, dict) else item
            for item in self.table_cells
        ]

        self.table_headers = [
            TableHeader.from_dict(item) if isinstance(item, dict) else item
            for item in self.table_headers
        ]

        self.table_spans = [
            TableSpan.from_dict(item) if isinstance(item, dict) else item
            for item in self.table_spans
        ]

        self.table_records = [
            TableRecord.from_dict(item) if isinstance(item, dict) else item
            for item in self.table_records
        ]

        self.table_semantics = [
            TableSemantic.from_dict(item) if isinstance(item, dict) else item
            for item in self.table_semantics
        ]

        self.multi_page_tables = [
            MultiPageTable.from_dict(item) if isinstance(item, dict) else item
            for item in self.multi_page_tables
        ]

        self.multi_page_table_segments = [
            MultiPageTableSegment.from_dict(item) if isinstance(item, dict) else item
            for item in self.multi_page_table_segments
        ]

    def tables_by_page(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in self.table_semantics:
            page_key = str(item.page_number)
            grouped.setdefault(page_key, [])
            grouped[page_key].append(item.to_dict())

        for item in self.table_grids:
            page_key = str(item.page_number)
            grouped.setdefault(page_key, [])

            if not any(existing.get("table_grid_id") == item.table_grid_id for existing in grouped[page_key]):
                grouped[page_key].append(item.to_dict())

        return grouped

    def cells_by_table(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for cell in self.table_cells:
            table_key = cell.table_grid_id or cell.table_structure_id or cell.table_semantic_id or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(cell.to_dict())

        return grouped

    def records_by_table(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for record in self.table_records:
            table_key = record.table_semantic_id or record.table_grid_id or "unknown_table"
            grouped.setdefault(table_key, [])
            grouped[table_key].append(record.to_dict())

        return grouped

    def summary(self) -> Dict[str, Any]:
        pages = set()

        for item in self.table_boundaries:
            pages.add(item.page_number)

        for item in self.table_grids:
            pages.add(item.page_number)

        for item in self.table_semantics:
            pages.add(item.page_number)

        semantic_types: Dict[str, int] = {}

        for item in self.table_semantics:
            semantic_types[item.semantic_type] = semantic_types.get(item.semantic_type, 0) + 1

        return {
            "document_id": self.document_id,
            "source_document": self.source_document,
            "table_boundary_count": len(self.table_boundaries),
            "table_grid_count": len(self.table_grids),
            "table_cell_count": len(self.table_cells),
            "table_header_count": len(self.table_headers),
            "table_span_count": len(self.table_spans),
            "table_record_count": len(self.table_records),
            "table_semantic_count": len(self.table_semantics),
            "multi_page_table_count": len(self.multi_page_tables),
            "multi_page_table_segment_count": len(self.multi_page_table_segments),
            "page_count_with_tables": len(pages),
            "by_semantic_type": semantic_types,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_document": self.source_document,
            "table_boundaries": [item.to_dict() for item in self.table_boundaries],
            "table_grids": [item.to_dict() for item in self.table_grids],
            "table_cells": [item.to_dict() for item in self.table_cells],
            "table_headers": [item.to_dict() for item in self.table_headers],
            "table_spans": [item.to_dict() for item in self.table_spans],
            "table_records": [item.to_dict() for item in self.table_records],
            "table_semantics": [item.to_dict() for item in self.table_semantics],
            "multi_page_tables": [item.to_dict() for item in self.multi_page_tables],
            "multi_page_table_segments": [item.to_dict() for item in self.multi_page_table_segments],
            "tables_by_page": self.tables_by_page(),
            "cells_by_table": self.cells_by_table(),
            "records_by_table": self.records_by_table(),
            "table_summary": self.summary(),
            "metadata": json_safe(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableCollection":
        data = dict(data or {})

        return cls(
            document_id=data.get("document_id", ""),
            source_document=data.get("source_document", ""),
            table_boundaries=data.get("table_boundaries", []) or [],
            table_grids=data.get("table_grids", []) or [],
            table_cells=data.get("table_cells", []) or [],
            table_headers=data.get("table_headers", []) or [],
            table_spans=data.get("table_spans", []) or [],
            table_records=data.get("table_records", []) or [],
            table_semantics=data.get("table_semantics", []) or [],
            multi_page_tables=data.get("multi_page_tables", []) or [],
            multi_page_table_segments=data.get("multi_page_table_segments", []) or [],
            metadata=data.get("metadata", {}) or {},
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

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
    def load_json(cls, input_path: Union[str, Path]) -> "TableCollection":
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)


def make_table_boundary(
    page_number: int,
    bbox: List[float],
    row_count_estimate: int = 0,
    col_count_estimate: int = 0,
    confidence: float = 0.60,
    metadata: Optional[Dict[str, Any]] = None,
) -> TableBoundary:
    return TableBoundary(
        page_number=page_number,
        bbox=bbox,
        row_count_estimate=row_count_estimate,
        col_count_estimate=col_count_estimate,
        confidence=confidence,
        metadata=metadata or {},
    )


def make_table_cell(
    table_grid_id: str,
    row_index: int,
    col_index: int,
    text: str = "",
    bbox: Optional[List[float]] = None,
    role: str = "body",
    metadata: Optional[Dict[str, Any]] = None,
) -> TableCell:
    return TableCell(
        table_grid_id=table_grid_id,
        row_index=row_index,
        col_index=col_index,
        text=text,
        bbox=bbox or [],
        role=role,
        metadata=metadata or {},
    )


def table_collection_from_dict(data: Dict[str, Any]) -> TableCollection:
    return TableCollection.from_dict(data)


def table_collection_to_dict(
    collection: Union[TableCollection, Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(collection, dict):
        return json_safe(collection)

    return collection.to_dict()


def save_table_json(
    table_data: Union[TableCollection, Dict[str, Any]],
    output_path: Union[str, Path],
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(table_data, TableCollection):
        data = table_data.to_dict()
    else:
        data = json_safe(table_data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return str(output_path)


def load_table_json(input_path: Union[str, Path]) -> TableCollection:
    return TableCollection.load_json(input_path)


TableBoundarySchema = TableBoundary
TableGridCellSchema = TableGridCell
TableGridSchema = TableGrid
TableCellSchema = TableCell
TableHeaderSchema = TableHeader
TableSpanSchema = TableSpan
TableRecordSchema = TableRecord
TableSemanticSchema = TableSemantic
MultiPageTableSegmentSchema = MultiPageTableSegment
MultiPageTableSchema = MultiPageTable
TableCollectionSchema = TableCollection
