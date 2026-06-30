"""
document_structure_schema.py

Production V1 - Colab Ready

Purpose
-------
Schema definitions for document structure understanding.

Used by:
- TitleDetector
- TOCDetector
- HeadingDetector
- SectionBuilder
- ParagraphBuilder
- ListDetector
- DocumentTreeBuilder
- DocumentStructurePipeline

Main objects
------------
- DocumentTitle
- TOCEntry
- DocumentHeading
- DocumentSection
- DocumentParagraph
- DocumentListItem
- DocumentList
- DocumentTreeNode
- DocumentStructure
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, asdict, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


SCHEMA_VERSION = "document_structure_schema_v1"


def make_id(prefix: str = "doc_struct") -> str:
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
class DocumentTitle:
    title_id: str = field(default_factory=lambda: make_id("title"))
    text: str = ""
    normalized_text: str = ""

    page_number: Optional[int] = None
    page_index: Optional[int] = None
    bbox: List[float] = field(default_factory=list)

    confidence: float = 0.70
    source: str = "document_structure_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text)
        else:
            self.normalized_text = normalize_text_for_match(self.normalized_text)

        self.bbox = normalize_bbox(self.bbox)
        self.confidence = clamp_float(self.confidence, default=0.70)

        if self.page_number is not None:
            self.page_number = safe_int(self.page_number, default=0) or None

        if self.page_index is not None:
            self.page_index = safe_int(self.page_index, default=0)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentTitle":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class TOCEntry:
    toc_entry_id: str = field(default_factory=lambda: make_id("toc_entry"))

    title: str = ""
    normalized_title: str = ""

    level: int = 1
    order: int = 0

    page_number: Optional[int] = None
    target_page: Optional[int] = None
    source_page: Optional[int] = None

    prefix: str = ""
    page_label: str = ""

    bbox: List[float] = field(default_factory=list)

    confidence: float = 0.60
    source: str = "document_structure_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.title = normalize_text(self.title)

        if not self.normalized_title:
            self.normalized_title = normalize_text_for_match(self.title)
        else:
            self.normalized_title = normalize_text_for_match(self.normalized_title)

        self.level = max(0, safe_int(self.level, default=1))
        self.order = safe_int(self.order, default=0)

        if self.page_number is not None:
            self.page_number = safe_int(self.page_number, default=0) or None

        if self.target_page is not None:
            self.target_page = safe_int(self.target_page, default=0) or None

        if self.source_page is not None:
            self.source_page = safe_int(self.source_page, default=0) or None

        self.prefix = normalize_text(self.prefix)
        self.page_label = normalize_text(self.page_label)
        self.bbox = normalize_bbox(self.bbox)
        self.confidence = clamp_float(self.confidence, default=0.60)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TOCEntry":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class DocumentHeading:
    heading_id: str = field(default_factory=lambda: make_id("heading"))

    text: str = ""
    normalized_text: str = ""

    level: int = 1
    order: int = 0

    page_number: Optional[int] = None
    page_index: Optional[int] = None
    bbox: List[float] = field(default_factory=list)

    heading_type: str = "heading"
    numbering: str = ""

    section_id: str = ""
    parent_heading_id: str = ""

    confidence: float = 0.65
    source: str = "document_structure_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text)
        else:
            self.normalized_text = normalize_text_for_match(self.normalized_text)

        self.level = max(0, safe_int(self.level, default=1))
        self.order = safe_int(self.order, default=0)

        if self.page_number is not None:
            self.page_number = safe_int(self.page_number, default=0) or None

        if self.page_index is not None:
            self.page_index = safe_int(self.page_index, default=0)

        self.bbox = normalize_bbox(self.bbox)
        self.heading_type = normalize_text(self.heading_type) or "heading"
        self.numbering = normalize_text(self.numbering)
        self.confidence = clamp_float(self.confidence, default=0.65)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentHeading":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class DocumentSection:
    section_id: str = field(default_factory=lambda: make_id("section"))

    title: str = ""
    normalized_title: str = ""

    level: int = 0
    order: int = 0

    page_start: Optional[int] = None
    page_end: Optional[int] = None
    page_numbers: List[int] = field(default_factory=list)
    content_page_numbers: List[int] = field(default_factory=list)

    parent_id: str = ""
    child_ids: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    heading_id: str = ""
    toc_entry_id: str = ""

    section_type: str = "section"
    section_number: str = ""

    text_preview: str = ""
    bbox: List[float] = field(default_factory=list)

    paragraph_ids: List[str] = field(default_factory=list)
    list_ids: List[str] = field(default_factory=list)
    table_ids: List[str] = field(default_factory=list)

    confidence: float = 0.70
    source: str = "document_structure_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.title = normalize_text(self.title)

        if not self.normalized_title:
            self.normalized_title = normalize_text_for_match(self.title)
        else:
            self.normalized_title = normalize_text_for_match(self.normalized_title)

        self.level = max(0, safe_int(self.level, default=0))
        self.order = safe_int(self.order, default=0)

        if self.page_start is not None:
            self.page_start = safe_int(self.page_start, default=0) or None

        if self.page_end is not None:
            self.page_end = safe_int(self.page_end, default=0) or None

        self.page_numbers = normalize_page_numbers(self.page_numbers)
        self.content_page_numbers = normalize_page_numbers(self.content_page_numbers)

        if not self.content_page_numbers and self.page_numbers:
            self.content_page_numbers = self.page_numbers[:]

        if not self.page_numbers and self.content_page_numbers:
            self.page_numbers = self.content_page_numbers[:]

        if not self.page_numbers and self.page_start is not None and self.page_end is not None:
            if self.page_end >= self.page_start:
                self.page_numbers = list(range(self.page_start, self.page_end + 1))
                self.content_page_numbers = self.page_numbers[:]

        if self.page_numbers:
            if self.page_start is None:
                self.page_start = min(self.page_numbers)

            if self.page_end is None:
                self.page_end = max(self.page_numbers)

        self.child_ids = [str(item) for item in self.child_ids if item]
        self.children = [str(item) for item in self.children if item]

        if not self.children and self.child_ids:
            self.children = self.child_ids[:]

        if not self.child_ids and self.children:
            self.child_ids = self.children[:]

        self.text_preview = normalize_text(self.text_preview)[:1200]
        self.bbox = normalize_bbox(self.bbox)
        self.section_type = normalize_text(self.section_type) or "section"
        self.section_number = normalize_text(self.section_number)
        self.confidence = clamp_float(self.confidence, default=0.70)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentSection":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)

    def validate(self) -> Dict[str, Any]:
        errors = []
        warnings = []

        if not self.section_id:
            errors.append("section_id is required")

        if not self.title:
            warnings.append("section title is empty")

        if self.page_start is not None and self.page_end is not None:
            if self.page_end < self.page_start:
                errors.append("page_end must be greater than or equal to page_start")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


@dataclass
class DocumentParagraph:
    paragraph_id: str = field(default_factory=lambda: make_id("paragraph"))

    text: str = ""
    normalized_text: str = ""

    page_number: Optional[int] = None
    page_index: Optional[int] = None
    page_numbers: List[int] = field(default_factory=list)

    section_id: str = ""
    section_title: str = ""

    order: int = 0
    global_order: int = 0

    paragraph_type: str = "paragraph"

    bbox: List[float] = field(default_factory=list)

    word_count: int = 0
    char_count: int = 0

    source_object_ids: List[str] = field(default_factory=list)

    confidence: float = 0.70
    source: str = "document_structure_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text)
        else:
            self.normalized_text = normalize_text_for_match(self.normalized_text)

        if self.page_number is not None:
            self.page_number = safe_int(self.page_number, default=0) or None

        if self.page_index is not None:
            self.page_index = safe_int(self.page_index, default=0)

        self.page_numbers = normalize_page_numbers(self.page_numbers)

        if self.page_number and self.page_number not in self.page_numbers:
            self.page_numbers.append(self.page_number)
            self.page_numbers = sorted(list(dict.fromkeys(self.page_numbers)))

        if self.page_number is None and self.page_numbers:
            self.page_number = self.page_numbers[0]

        self.order = safe_int(self.order, default=0)
        self.global_order = safe_int(self.global_order, default=self.order)

        self.paragraph_type = normalize_text(self.paragraph_type) or "paragraph"
        self.bbox = normalize_bbox(self.bbox)

        self.char_count = len(self.text)
        self.word_count = len(re.findall(r"\S+", self.text))

        self.source_object_ids = [str(item) for item in self.source_object_ids if item]
        self.confidence = clamp_float(self.confidence, default=0.70)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentParagraph":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)

    def validate(self) -> Dict[str, Any]:
        errors = []
        warnings = []

        if not self.paragraph_id:
            errors.append("paragraph_id is required")

        if not self.text:
            warnings.append("paragraph text is empty")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


@dataclass
class DocumentListItem:
    list_item_id: str = field(default_factory=lambda: make_id("list_item"))

    text: str = ""
    normalized_text: str = ""

    marker: str = ""
    marker_type: str = ""
    level: int = 0
    order: int = 0

    page_number: Optional[int] = None
    page_index: Optional[int] = None
    section_id: str = ""

    parent_item_id: str = ""
    child_item_ids: List[str] = field(default_factory=list)

    bbox: List[float] = field(default_factory=list)

    confidence: float = 0.65
    source: str = "document_structure_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text)
        else:
            self.normalized_text = normalize_text_for_match(self.normalized_text)

        self.marker = normalize_text(self.marker)
        self.marker_type = normalize_text(self.marker_type)
        self.level = max(0, safe_int(self.level, default=0))
        self.order = safe_int(self.order, default=0)

        if self.page_number is not None:
            self.page_number = safe_int(self.page_number, default=0) or None

        if self.page_index is not None:
            self.page_index = safe_int(self.page_index, default=0)

        self.child_item_ids = [str(item) for item in self.child_item_ids if item]
        self.bbox = normalize_bbox(self.bbox)
        self.confidence = clamp_float(self.confidence, default=0.65)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentListItem":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class DocumentList:
    list_id: str = field(default_factory=lambda: make_id("list"))

    list_type: str = "list"
    section_id: str = ""

    page_numbers: List[int] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    item_ids: List[str] = field(default_factory=list)
    items: List[DocumentListItem] = field(default_factory=list)

    order: int = 0
    confidence: float = 0.65
    source: str = "document_structure_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.list_type = normalize_text(self.list_type) or "list"
        self.page_numbers = normalize_page_numbers(self.page_numbers)

        normalized_items = []

        for item in self.items:
            if isinstance(item, DocumentListItem):
                normalized_items.append(item)
            elif isinstance(item, dict):
                normalized_items.append(DocumentListItem.from_dict(item))

        self.items = normalized_items

        if not self.item_ids and self.items:
            self.item_ids = [item.list_item_id for item in self.items]

        self.item_ids = [str(item) for item in self.item_ids if item]

        if self.items and not self.page_numbers:
            self.page_numbers = normalize_page_numbers(
                [
                    item.page_number
                    for item in self.items
                    if item.page_number
                ]
            )

        if self.page_numbers:
            if self.page_start is None:
                self.page_start = min(self.page_numbers)

            if self.page_end is None:
                self.page_end = max(self.page_numbers)

        self.order = safe_int(self.order, default=0)
        self.confidence = clamp_float(self.confidence, default=0.65)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentList":
        data = dict(data or {})

        if "items" in data:
            data["items"] = [
                DocumentListItem.from_dict(item) if isinstance(item, dict) else item
                for item in data.get("items", []) or []
            ]

        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}

        return cls(**clean)


@dataclass
class DocumentTreeNode:
    node_id: str = field(default_factory=lambda: make_id("tree_node"))

    node_type: str = "node"
    title: str = ""
    text: str = ""

    section_id: str = ""
    paragraph_id: str = ""
    list_id: str = ""

    parent_id: str = ""
    child_ids: List[str] = field(default_factory=list)
    children: List[Dict[str, Any]] = field(default_factory=list)

    level: int = 0
    order: int = 0

    page_numbers: List[int] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.node_type = normalize_text(self.node_type) or "node"
        self.title = normalize_text(self.title)
        self.text = normalize_text(self.text)

        self.level = max(0, safe_int(self.level, default=0))
        self.order = safe_int(self.order, default=0)

        self.page_numbers = normalize_page_numbers(self.page_numbers)

        if self.page_numbers:
            if self.page_start is None:
                self.page_start = min(self.page_numbers)

            if self.page_end is None:
                self.page_end = max(self.page_numbers)

        self.child_ids = [str(item) for item in self.child_ids if item]

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentTreeNode":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {key: value for key, value in data.items() if key in allowed}
        return cls(**clean)


@dataclass
class DocumentStructure:
    document_id: str = ""

    title: Optional[DocumentTitle] = None
    toc_entries: List[TOCEntry] = field(default_factory=list)
    headings: List[DocumentHeading] = field(default_factory=list)
    sections: List[DocumentSection] = field(default_factory=list)
    paragraphs: List[DocumentParagraph] = field(default_factory=list)
    lists: List[DocumentList] = field(default_factory=list)
    list_items: List[DocumentListItem] = field(default_factory=list)

    root_node: Optional[DocumentTreeNode] = None
    nodes: List[DocumentTreeNode] = field(default_factory=list)

    pages: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    processor: str = "DocumentStructure"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.title, dict):
            self.title = DocumentTitle.from_dict(self.title)

        self.toc_entries = [
            TOCEntry.from_dict(item) if isinstance(item, dict) else item
            for item in self.toc_entries
        ]

        self.headings = [
            DocumentHeading.from_dict(item) if isinstance(item, dict) else item
            for item in self.headings
        ]

        self.sections = [
            DocumentSection.from_dict(item) if isinstance(item, dict) else item
            for item in self.sections
        ]

        self.paragraphs = [
            DocumentParagraph.from_dict(item) if isinstance(item, dict) else item
            for item in self.paragraphs
        ]

        self.lists = [
            DocumentList.from_dict(item) if isinstance(item, dict) else item
            for item in self.lists
        ]

        self.list_items = [
            DocumentListItem.from_dict(item) if isinstance(item, dict) else item
            for item in self.list_items
        ]

        if isinstance(self.root_node, dict):
            self.root_node = DocumentTreeNode.from_dict(self.root_node)

        self.nodes = [
            DocumentTreeNode.from_dict(item) if isinstance(item, dict) else item
            for item in self.nodes
        ]

    def summary(self) -> Dict[str, Any]:
        page_numbers = set()

        for section in self.sections:
            for page_number in section.page_numbers:
                page_numbers.add(page_number)

        for paragraph in self.paragraphs:
            for page_number in paragraph.page_numbers:
                page_numbers.add(page_number)

        by_section_level: Dict[str, int] = {}

        for section in self.sections:
            level_key = str(section.level)
            by_section_level[level_key] = by_section_level.get(level_key, 0) + 1

        by_paragraph_type: Dict[str, int] = {}

        for paragraph in self.paragraphs:
            paragraph_type = paragraph.paragraph_type or "paragraph"
            by_paragraph_type[paragraph_type] = by_paragraph_type.get(paragraph_type, 0) + 1

        return {
            "document_id": self.document_id,
            "has_title": self.title is not None and bool(self.title.text),
            "toc_entry_count": len(self.toc_entries),
            "heading_count": len(self.headings),
            "section_count": len(self.sections),
            "paragraph_count": len(self.paragraphs),
            "list_count": len(self.lists),
            "list_item_count": len(self.list_items),
            "tree_node_count": len(self.nodes),
            "page_count_with_structure": len(page_numbers),
            "by_section_level": by_section_level,
            "by_paragraph_type": by_paragraph_type,
            "warning_count": len(self.warnings),
            "error_count": len(self.errors),
        }

    def sections_by_page(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for section in self.sections:
            for page_number in section.page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(section.to_dict())

        return grouped

    def paragraphs_by_page(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for paragraph in self.paragraphs:
            for page_number in paragraph.page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(paragraph.to_dict())

        return grouped

    def lists_by_page(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for doc_list in self.lists:
            for page_number in doc_list.page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(doc_list.to_dict())

        return grouped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processor": self.processor,
            "schema_version": self.schema_version,
            "document_id": self.document_id,
            "title": self.title.to_dict() if self.title else {},
            "toc_entries": [item.to_dict() for item in self.toc_entries],
            "headings": [item.to_dict() for item in self.headings],
            "sections": [item.to_dict() for item in self.sections],
            "paragraphs": [item.to_dict() for item in self.paragraphs],
            "lists": [item.to_dict() for item in self.lists],
            "list_items": [item.to_dict() for item in self.list_items],
            "root_node": self.root_node.to_dict() if self.root_node else {},
            "nodes": [item.to_dict() for item in self.nodes],
            "pages": json_safe(self.pages),
            "sections_by_page": self.sections_by_page(),
            "paragraphs_by_page": self.paragraphs_by_page(),
            "lists_by_page": self.lists_by_page(),
            "document_structure_summary": self.summary(),
            "metadata": json_safe(self.metadata),
            "warnings": self.warnings,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentStructure":
        data = dict(data or {})

        title_data = data.get("title") or None

        if isinstance(title_data, dict) and not title_data:
            title_data = None

        return cls(
            document_id=data.get("document_id", ""),
            title=DocumentTitle.from_dict(title_data) if isinstance(title_data, dict) else title_data,
            toc_entries=data.get("toc_entries", []) or data.get("toc", []) or [],
            headings=data.get("headings", []) or [],
            sections=data.get("sections", []) or [],
            paragraphs=data.get("paragraphs", []) or [],
            lists=data.get("lists", []) or [],
            list_items=data.get("list_items", []) or [],
            root_node=data.get("root_node") or None,
            nodes=data.get("nodes", []) or [],
            pages=data.get("pages", []) or [],
            metadata=data.get("metadata", {}) or {},
            warnings=data.get("warnings", []) or [],
            errors=data.get("errors", []) or [],
            processor=data.get("processor", "DocumentStructure"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

    def validate(self) -> Dict[str, Any]:
        errors = []
        warnings = []

        section_ids = {
            section.section_id
            for section in self.sections
            if section.section_id
        }

        paragraph_ids = {
            paragraph.paragraph_id
            for paragraph in self.paragraphs
            if paragraph.paragraph_id
        }

        for section in self.sections:
            result = section.validate()

            for error in result.get("errors", []):
                errors.append(f"section:{section.section_id}: {error}")

            for warning in result.get("warnings", []):
                warnings.append(f"section:{section.section_id}: {warning}")

            if section.parent_id and section.parent_id not in section_ids:
                warnings.append(f"section:{section.section_id}: parent_id not found: {section.parent_id}")

        for paragraph in self.paragraphs:
            result = paragraph.validate()

            for error in result.get("errors", []):
                errors.append(f"paragraph:{paragraph.paragraph_id}: {error}")

            for warning in result.get("warnings", []):
                warnings.append(f"paragraph:{paragraph.paragraph_id}: {warning}")

            if paragraph.section_id and paragraph.section_id not in section_ids:
                warnings.append(f"paragraph:{paragraph.paragraph_id}: section_id not found: {paragraph.section_id}")

        for doc_list in self.lists:
            for item_id in doc_list.item_ids:
                if item_id and item_id not in {item.list_item_id for item in self.list_items}:
                    warnings.append(f"list:{doc_list.list_id}: item_id not found in list_items: {item_id}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "section_id_count": len(section_ids),
            "paragraph_id_count": len(paragraph_ids),
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
    def load_json(cls, input_path: Union[str, Path]) -> "DocumentStructure":
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)


def make_document_structure(
    document_id: str = "",
    title: Optional[Union[DocumentTitle, Dict[str, Any]]] = None,
    toc_entries: Optional[List[Union[TOCEntry, Dict[str, Any]]]] = None,
    headings: Optional[List[Union[DocumentHeading, Dict[str, Any]]]] = None,
    sections: Optional[List[Union[DocumentSection, Dict[str, Any]]]] = None,
    paragraphs: Optional[List[Union[DocumentParagraph, Dict[str, Any]]]] = None,
    lists: Optional[List[Union[DocumentList, Dict[str, Any]]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DocumentStructure:
    return DocumentStructure(
        document_id=document_id,
        title=title,
        toc_entries=toc_entries or [],
        headings=headings or [],
        sections=sections or [],
        paragraphs=paragraphs or [],
        lists=lists or [],
        metadata=metadata or {},
    )


def document_structure_from_dict(data: Dict[str, Any]) -> DocumentStructure:
    return DocumentStructure.from_dict(data)


def document_structure_to_dict(
    structure: Union[DocumentStructure, Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(structure, dict):
        return json_safe(structure)

    return structure.to_dict()


def save_document_structure_json(
    structure: Union[DocumentStructure, Dict[str, Any]],
    output_path: Union[str, Path],
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(structure, DocumentStructure):
        data = structure.to_dict()
    else:
        data = json_safe(structure)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return str(output_path)


def load_document_structure_json(
    input_path: Union[str, Path],
) -> DocumentStructure:
    return DocumentStructure.load_json(input_path)
