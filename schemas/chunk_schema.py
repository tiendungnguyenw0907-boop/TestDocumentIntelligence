"""
chunk_schema.py

Production V1 - Colab Ready

Purpose
-------
Schema definitions for document chunks used in Knowledge Pipeline, Indexing, and RAG.

Chunk types
-----------
- page_chunk
- section_chunk
- paragraph_chunk
- table_chunk
- parent_chunk
- child_chunk
- evidence_chunk
- mixed_chunk

Main classes
------------
- Chunk
- ChunkRelation
- ParentChildChunk
- TableChunk
- ChunkCollection
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, asdict, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


SCHEMA_VERSION = "chunk_schema_v1"


def make_id(prefix: str = "chunk") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def stable_hash(text: Any) -> str:
    value = "" if text is None else str(text)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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

    text = re.sub(r"[^a-z0-9_%]+", " ", text)
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
class Chunk:
    chunk_id: str = field(default_factory=lambda: make_id("chunk"))
    chunk_type: str = "text_chunk"

    text: str = ""
    normalized_text: str = ""

    document_id: str = ""
    source_document: str = ""

    page_numbers: List[int] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    section_id: str = ""
    section_title: str = ""
    section_level: Optional[int] = None

    paragraph_id: str = ""

    table_grid_id: str = ""
    table_structure_id: str = ""
    table_semantic_id: str = ""
    table_boundary_id: str = ""

    bbox: List[float] = field(default_factory=list)

    order: int = 0
    parent_chunk_id: str = ""
    child_chunk_ids: List[str] = field(default_factory=list)

    previous_chunk_id: str = ""
    next_chunk_id: str = ""

    source: str = "chunk_schema"
    confidence: float = 0.70

    token_count: int = 0
    char_count: int = 0
    word_count: int = 0

    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text)
        else:
            self.normalized_text = normalize_text(self.normalized_text)

        self.bbox = normalize_bbox(self.bbox)

        self.page_numbers = self._normalize_page_numbers(self.page_numbers)

        if self.page_numbers:
            if self.page_start is None:
                self.page_start = min(self.page_numbers)

            if self.page_end is None:
                self.page_end = max(self.page_numbers)

        if self.page_start is not None and self.page_end is not None and not self.page_numbers:
            try:
                self.page_numbers = list(range(int(self.page_start), int(self.page_end) + 1))
            except Exception:
                self.page_numbers = []

        self.char_count = len(self.text)
        self.word_count = len(re.findall(r"\S+", self.text))
        self.token_count = self.word_count

        if not self.content_hash:
            self.content_hash = stable_hash(
                {
                    "chunk_type": self.chunk_type,
                    "text": self.text,
                    "page_numbers": self.page_numbers,
                    "section_id": self.section_id,
                    "table_grid_id": self.table_grid_id,
                }
            )

    def _normalize_page_numbers(self, values: List[Any]) -> List[int]:
        result = []

        for value in values or []:
            try:
                page = int(value)

                if page > 0:
                    result.append(page)
            except Exception:
                continue

        return sorted(list(dict.fromkeys(result)))

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }
        return cls(**clean)

    def preview(self, max_chars: int = 300) -> str:
        text = normalize_text(self.text)

        if len(text) <= max_chars:
            return text

        return text[:max_chars].rstrip() + "..."

    def is_table_chunk(self) -> bool:
        return self.chunk_type == "table_chunk" or bool(self.table_grid_id)

    def is_section_chunk(self) -> bool:
        return self.chunk_type == "section_chunk" or bool(self.section_id)

    def is_page_chunk(self) -> bool:
        return self.chunk_type == "page_chunk"

    def validate(self) -> Dict[str, Any]:
        errors = []
        warnings = []

        if not self.chunk_id:
            errors.append("chunk_id is required")

        if not self.text:
            warnings.append("text is empty")

        if not self.chunk_type:
            errors.append("chunk_type is required")

        if not self.page_numbers and self.page_start is None:
            warnings.append("page_numbers/page_start is missing")

        if self.page_start is not None and self.page_end is not None:
            if self.page_end < self.page_start:
                errors.append("page_end must be greater than or equal to page_start")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


@dataclass
class TableChunk(Chunk):
    chunk_type: str = "table_chunk"

    table_title: str = ""
    table_caption: str = ""
    table_type: str = ""
    semantic_type: str = ""

    row_count: int = 0
    col_count: int = 0

    column_headers: List[str] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)
    matrix: List[List[Dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chunk_type = "table_chunk"
        super().__post_init__()

        self.column_headers = [
            normalize_text(item)
            for item in self.column_headers
            if normalize_text(item)
        ]

    @classmethod
    def from_table(
        cls,
        table: Dict[str, Any],
        records: Optional[List[Dict[str, Any]]] = None,
        matrix: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> "TableChunk":
        records = records or []
        matrix = matrix or []

        table_grid_id = table.get("table_grid_id", "")
        table_semantic_id = table.get("table_semantic_id", "")
        table_structure_id = table.get("table_structure_id", "")
        table_boundary_id = table.get("table_boundary_id", "")

        page_number = table.get("page_number")

        page_numbers = [page_number] if page_number else []

        title = normalize_text(
            table.get("title")
            or table.get("caption")
            or table.get("table_title")
            or ""
        )

        lines = []

        if title:
            lines.append(title)

        column_headers = table.get("column_headers", []) or []

        if column_headers:
            lines.append(" | ".join([normalize_text(item) for item in column_headers]))

        for record in records[:50]:
            raw_values = record.get("raw_values", {}) or record.get("values", {}) or {}

            if raw_values:
                line = " | ".join(
                    f"{key}: {value}"
                    for key, value in raw_values.items()
                    if normalize_text(value)
                )

                if line:
                    lines.append(line)

        if not lines and matrix:
            for row in matrix[:50]:
                parts = []

                for cell in row:
                    if isinstance(cell, dict):
                        cell_text = normalize_text(
                            cell.get("text")
                            or cell.get("normalized_text")
                            or ""
                        )
                    else:
                        cell_text = normalize_text(cell)

                    if cell_text:
                        parts.append(cell_text)

                if parts:
                    lines.append(" | ".join(parts))

        text = "\n".join(lines)

        return cls(
            chunk_id=make_id("tbl_chunk"),
            text=text,
            page_numbers=page_numbers,
            page_start=page_number,
            page_end=page_number,
            table_grid_id=table_grid_id,
            table_structure_id=table_structure_id,
            table_semantic_id=table_semantic_id,
            table_boundary_id=table_boundary_id,
            bbox=table.get("bbox", []) or [],
            table_title=title,
            table_caption=normalize_text(table.get("caption", "")),
            table_type=table.get("table_type", ""),
            semantic_type=table.get("semantic_type", ""),
            row_count=int(table.get("row_count", 0) or 0),
            col_count=int(table.get("col_count", 0) or 0),
            column_headers=column_headers,
            records=records,
            matrix=matrix,
            confidence=float(table.get("confidence", 0.70) or 0.70),
            source="table_chunk_from_table",
            metadata={
                "raw_table": table,
                "record_count": len(records),
                "matrix_row_count": len(matrix),
            },
        )


@dataclass
class ParentChildChunk:
    parent_child_id: str = field(default_factory=lambda: make_id("parent_child"))

    parent_chunk_id: str = ""
    child_chunk_id: str = ""

    parent_section_id: str = ""
    child_section_id: str = ""

    relation_type: str = "parent_child"
    weight: float = 1.0
    confidence: float = 0.70

    source: str = "chunk_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParentChildChunk":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }
        return cls(**clean)


@dataclass
class ChunkRelation:
    chunk_relation_id: str = field(default_factory=lambda: make_id("chunk_rel"))

    source_chunk_id: str = ""
    target_chunk_id: str = ""
    relation_type: str = "related_to"

    source_page: Optional[int] = None
    target_page: Optional[int] = None

    weight: float = 1.0
    confidence: float = 0.60

    source: str = "chunk_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRelation":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }
        return cls(**clean)


@dataclass
class ChunkCollection:
    document_id: str = ""
    source_document: str = ""

    chunks: List[Chunk] = field(default_factory=list)
    table_chunks: List[TableChunk] = field(default_factory=list)
    parent_child_links: List[ParentChildChunk] = field(default_factory=list)
    relations: List[ChunkRelation] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def add_chunk(self, chunk: Union[Chunk, Dict[str, Any]]) -> Chunk:
        if isinstance(chunk, dict):
            chunk = Chunk.from_dict(chunk)

        self.chunks.append(chunk)
        return chunk

    def add_table_chunk(self, chunk: Union[TableChunk, Dict[str, Any]]) -> TableChunk:
        if isinstance(chunk, dict):
            chunk = TableChunk.from_dict(chunk)

        self.table_chunks.append(chunk)
        return chunk

    def all_chunks(self) -> List[Chunk]:
        return self.chunks + self.table_chunks

    def deduplicate(self) -> None:
        seen = set()
        unique_chunks = []

        for chunk in self.chunks:
            key = chunk.chunk_id or chunk.content_hash

            if key in seen:
                continue

            seen.add(key)
            unique_chunks.append(chunk)

        self.chunks = unique_chunks

        seen_table = set()
        unique_table_chunks = []

        for chunk in self.table_chunks:
            key = chunk.chunk_id or chunk.content_hash

            if key in seen_table:
                continue

            seen_table.add(key)
            unique_table_chunks.append(chunk)

        self.table_chunks = unique_table_chunks

    def by_page(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in self.all_chunks():
            for page_number in chunk.page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(chunk.to_dict())

        return grouped

    def by_section(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in self.all_chunks():
            section_id = chunk.section_id or "no_section"
            grouped.setdefault(section_id, [])
            grouped[section_id].append(chunk.to_dict())

        return grouped

    def by_type(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for chunk in self.all_chunks():
            grouped.setdefault(chunk.chunk_type, [])
            grouped[chunk.chunk_type].append(chunk.to_dict())

        return grouped

    def summary(self) -> Dict[str, Any]:
        chunks = self.all_chunks()

        by_type: Dict[str, int] = {}

        for chunk in chunks:
            by_type[chunk.chunk_type] = by_type.get(chunk.chunk_type, 0) + 1

        pages = set()

        for chunk in chunks:
            for page_number in chunk.page_numbers:
                pages.add(page_number)

        return {
            "document_id": self.document_id,
            "source_document": self.source_document,
            "chunk_count": len(chunks),
            "text_chunk_count": len(self.chunks),
            "table_chunk_count": len(self.table_chunks),
            "parent_child_link_count": len(self.parent_child_links),
            "relation_count": len(self.relations),
            "page_count_with_chunks": len(pages),
            "by_chunk_type": by_type,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_document": self.source_document,
            "chunks": [
                chunk.to_dict() for chunk in self.chunks
            ],
            "table_chunks": [
                chunk.to_dict() for chunk in self.table_chunks
            ],
            "parent_child_links": [
                link.to_dict() for link in self.parent_child_links
            ],
            "relations": [
                relation.to_dict() for relation in self.relations
            ],
            "chunks_by_page": self.by_page(),
            "chunks_by_section": self.by_section(),
            "chunks_by_type": self.by_type(),
            "chunk_summary": self.summary(),
            "metadata": json_safe(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkCollection":
        data = dict(data or {})

        collection = cls(
            document_id=data.get("document_id", ""),
            source_document=data.get("source_document", ""),
            metadata=data.get("metadata", {}) or {},
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

        for item in data.get("chunks", []) or []:
            collection.chunks.append(Chunk.from_dict(item))

        for item in data.get("table_chunks", []) or []:
            collection.table_chunks.append(TableChunk.from_dict(item))

        for item in data.get("parent_child_links", []) or []:
            collection.parent_child_links.append(ParentChildChunk.from_dict(item))

        for item in data.get("relations", []) or []:
            collection.relations.append(ChunkRelation.from_dict(item))

        return collection

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
    def load_json(cls, input_path: Union[str, Path]) -> "ChunkCollection":
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)


def make_chunk(
    text: str,
    chunk_type: str = "text_chunk",
    page_numbers: Optional[List[int]] = None,
    section_id: str = "",
    section_title: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Chunk:
    return Chunk(
        chunk_type=chunk_type,
        text=text,
        page_numbers=page_numbers or [],
        section_id=section_id,
        section_title=section_title,
        metadata=metadata or {},
    )


def chunk_from_dict(data: Dict[str, Any]) -> Chunk:
    return Chunk.from_dict(data)


def chunk_to_dict(chunk: Union[Chunk, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(chunk, dict):
        return json_safe(chunk)

    return chunk.to_dict()


def save_chunks_json(
    chunks: List[Union[Chunk, Dict[str, Any]]],
    output_path: Union[str, Path],
    document_id: str = "",
    source_document: str = "",
) -> str:
    collection = ChunkCollection(
        document_id=document_id,
        source_document=source_document,
    )

    for chunk in chunks:
        if isinstance(chunk, Chunk):
            collection.chunks.append(chunk)
        elif isinstance(chunk, dict):
            collection.chunks.append(Chunk.from_dict(chunk))

    return collection.save_json(output_path)


# =============================================================================
# Backward compatibility layer
# =============================================================================
# Keeps canonical chunk schema while accepting older constructor arguments used
# by knowledge builders:
# - ChunkCollection(parent_child_chunks=..., chunk_relations=...)
# - ParentChildChunk(child_chunk_ids=..., document_id=..., page_numbers=...)
# Also exposes alias properties: collection.parent_child_chunks,
# collection.chunk_relations, link.child_chunk_ids.


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

        for old_key, new_key in list(alias_map.items()):
            if old_key not in kwargs:
                continue

            value = kwargs.pop(old_key)
            alias_metadata[old_key] = value

            if new_key and new_key in fields and new_key not in kwargs:
                if new_key == "page_numbers" and not isinstance(value, list):
                    kwargs[new_key] = [value]
                elif new_key == "child_chunk_id" and isinstance(value, list):
                    kwargs[new_key] = str(value[0]) if value else ""
                else:
                    kwargs[new_key] = value

        for key in list(metadata_only):
            if key in kwargs:
                alias_metadata[key] = kwargs.pop(key)

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


def _compat_to_json(self, ensure_ascii: bool = False, indent: int = 2) -> str:
    if hasattr(self, "to_dict"):
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)
    return json.dumps(json_safe(self), ensure_ascii=ensure_ascii, indent=indent)


def _metadata_list_property(metadata_key: str):
    def getter(self):
        metadata = getattr(self, "metadata", {}) or {}
        values = []

        if isinstance(metadata, dict):
            raw = metadata.get(metadata_key, [])
            if isinstance(raw, list):
                values.extend([str(item) for item in raw if item not in [None, ""]])
            elif raw not in [None, ""]:
                values.append(str(raw))

        scalar = getattr(self, "child_chunk_id", "")
        if metadata_key == "child_chunk_ids" and scalar:
            values.append(str(scalar))

        return list(dict.fromkeys(values))

    def setter(self, value):
        if value is None:
            values = []
        elif isinstance(value, list):
            values = [str(item) for item in value if item not in [None, ""]]
        else:
            values = [str(value)]

        metadata = getattr(self, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(self, "metadata", metadata)
        metadata[metadata_key] = values

        if metadata_key == "child_chunk_ids" and values:
            setattr(self, "child_chunk_id", values[0])

    return property(getter, setter)


def _collection_alias_property(canonical_name: str):
    def getter(self):
        return getattr(self, canonical_name)

    def setter(self, value):
        setattr(self, canonical_name, value or [])

    return property(getter, setter)


def _normalize_page_numbers_compat(values: Any) -> List[int]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]
    result = []
    for value in values:
        try:
            page = int(value)
            if page > 0:
                result.append(page)
        except Exception:
            continue
    return sorted(list(dict.fromkeys(result)))


def _install_chunk_compatibility() -> None:
    _compat_wrap_init(
        Chunk,
        alias_map={
            "source_path": "source_document",
            "file_path": "source_document",
            "page_number": "page_numbers",
            "table_id": "table_grid_id",
        },
        metadata_only=["file_name", "document_type"],
    )

    _compat_wrap_init(
        TableChunk,
        alias_map={
            "source_path": "source_document",
            "file_path": "source_document",
            "page_number": "page_numbers",
            "table_id": "table_grid_id",
            "headers": "column_headers",
            "columns": "column_headers",
        },
        metadata_only=["file_name", "document_type"],
    )

    _compat_wrap_init(
        ParentChildChunk,
        alias_map={
            "child_chunk_ids": "child_chunk_id",
            "section_id": "parent_section_id",
        },
        metadata_only=[
            "document_id",
            "page_numbers",
            "table_grid_id",
            "table_structure_id",
            "table_semantic_id",
            "table_boundary_id",
        ],
    )

    _compat_wrap_init(
        ChunkRelation,
        alias_map={
            "source_id": "source_chunk_id",
            "target_id": "target_chunk_id",
        },
    )

    _compat_wrap_init(
        ChunkCollection,
        alias_map={
            "parent_child_chunks": "parent_child_links",
            "chunk_relations": "relations",
        },
    )

    ParentChildChunk.child_chunk_ids = _metadata_list_property("child_chunk_ids")

    def _pc_to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["child_chunk_ids"] = self.child_chunk_ids
        metadata = data.get("metadata", {}) or {}
        if isinstance(metadata, dict):
            for key in [
                "document_id",
                "page_numbers",
                "table_grid_id",
                "table_structure_id",
                "table_semantic_id",
                "table_boundary_id",
            ]:
                if key in metadata and key not in data:
                    data[key] = metadata.get(key)
        return data

    ParentChildChunk.to_dict = _pc_to_dict

    ChunkCollection.parent_child_chunks = _collection_alias_property("parent_child_links")
    ChunkCollection.chunk_relations = _collection_alias_property("relations")

    original_collection_to_dict = ChunkCollection.to_dict

    def _collection_to_dict(self) -> Dict[str, Any]:
        data = original_collection_to_dict(self)
        data["parent_child_chunks"] = [link.to_dict() for link in self.parent_child_links]
        data["chunk_relations"] = [relation.to_dict() for relation in self.relations]
        return data

    ChunkCollection.to_dict = _collection_to_dict

    @classmethod
    def _collection_from_dict(cls, data: Dict[str, Any]) -> "ChunkCollection":
        data = dict(data or {})
        collection = cls(
            document_id=data.get("document_id", ""),
            source_document=data.get("source_document", "") or data.get("source_path", ""),
            metadata=data.get("metadata", {}) or {},
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

        for item in data.get("chunks", []) or []:
            item = dict(item or {})
            chunk_type = item.get("chunk_type", "")
            if chunk_type == "table_chunk":
                collection.table_chunks.append(TableChunk.from_dict(item))
            else:
                collection.chunks.append(Chunk.from_dict(item))

        for item in data.get("table_chunks", []) or []:
            collection.table_chunks.append(TableChunk.from_dict(item))

        links = data.get("parent_child_links", []) or data.get("parent_child_chunks", []) or []
        for item in links:
            collection.parent_child_links.append(ParentChildChunk.from_dict(item))

        relations = data.get("relations", []) or data.get("chunk_relations", []) or []
        for item in relations:
            collection.relations.append(ChunkRelation.from_dict(item))

        return collection

    ChunkCollection.from_dict = _collection_from_dict

    def _chunk_init_page_numbers_fix(self):
        if isinstance(getattr(self, "page_numbers", None), int):
            self.page_numbers = [self.page_numbers]
        elif isinstance(getattr(self, "page_numbers", None), str):
            self.page_numbers = _normalize_page_numbers_compat(self.page_numbers)

    # Add JSON helpers.
    for cls in [Chunk, TableChunk, ParentChildChunk, ChunkRelation, ChunkCollection]:
        if not hasattr(cls, "to_json"):
            cls.to_json = _compat_to_json


_install_chunk_compatibility()
