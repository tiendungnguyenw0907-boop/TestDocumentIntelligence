"""
evidence_schema.py

Production V1 - Colab Ready

Purpose
-------
Schema definitions for evidence and citations used in Knowledge Pipeline and RAG.

Used by:
- EvidenceBuilder
- CitationBuilder
- RAGPipeline
- CitationVerifier

Main objects
------------
- Evidence
- Citation
- EvidenceRelation
- EvidenceCollection
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, asdict, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


SCHEMA_VERSION = "evidence_schema_v1"


def make_id(prefix: str = "evidence") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def stable_hash(value: Any) -> str:
    text = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
class Evidence:
    evidence_id: str = field(default_factory=lambda: make_id("evidence"))
    evidence_type: str = "text_evidence"

    text: str = ""
    normalized_text: str = ""

    document_id: str = ""
    source_document: str = ""

    chunk_id: str = ""
    source_chunk_id: str = ""

    page_number: Optional[int] = None
    page_numbers: List[int] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    section_id: str = ""
    section_title: str = ""

    paragraph_id: str = ""

    table_grid_id: str = ""
    table_structure_id: str = ""
    table_semantic_id: str = ""
    table_boundary_id: str = ""

    entity_id: str = ""
    reference_id: str = ""

    bbox: List[float] = field(default_factory=list)

    quote: str = ""
    context_before: str = ""
    context_after: str = ""

    relevance_score: float = 0.0
    confidence: float = 0.70
    weight: float = 1.0

    rank: int = 0
    order: int = 0

    source: str = "evidence_schema"
    extraction_method: str = ""

    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text)
        else:
            self.normalized_text = normalize_text_for_match(self.normalized_text)

        self.quote = normalize_text(self.quote)

        if not self.quote and self.text:
            self.quote = self.text

        self.context_before = normalize_text(self.context_before)
        self.context_after = normalize_text(self.context_after)

        self.page_numbers = normalize_page_numbers(self.page_numbers)

        if self.page_number is not None:
            page_number = safe_int(self.page_number, default=0)

            if page_number > 0 and page_number not in self.page_numbers:
                self.page_numbers.append(page_number)
                self.page_numbers = sorted(list(dict.fromkeys(self.page_numbers)))

            self.page_number = page_number or None

        if self.page_number is None and self.page_numbers:
            self.page_number = self.page_numbers[0]

        if self.page_numbers:
            if self.page_start is None:
                self.page_start = min(self.page_numbers)

            if self.page_end is None:
                self.page_end = max(self.page_numbers)

        if self.page_start is not None and self.page_end is not None and not self.page_numbers:
            if self.page_end >= self.page_start:
                self.page_numbers = list(range(int(self.page_start), int(self.page_end) + 1))

        self.bbox = normalize_bbox(self.bbox)

        self.relevance_score = clamp_float(self.relevance_score, default=0.0)
        self.confidence = clamp_float(self.confidence, default=0.70)
        self.weight = clamp_float(self.weight, default=1.0, min_value=0.0, max_value=999999.0)

        self.rank = safe_int(self.rank, default=0)
        self.order = safe_int(self.order, default=0)

        if not self.source_chunk_id and self.chunk_id:
            self.source_chunk_id = self.chunk_id

        if not self.chunk_id and self.source_chunk_id:
            self.chunk_id = self.source_chunk_id

        if not self.content_hash:
            self.content_hash = stable_hash(
                {
                    "text": self.text,
                    "page_numbers": self.page_numbers,
                    "section_id": self.section_id,
                    "chunk_id": self.chunk_id,
                    "table_grid_id": self.table_grid_id,
                }
            )

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Evidence":
        data = dict(data or {})

        aliases = {
            "id": "evidence_id",
            "type": "evidence_type",
            "content": "text",
            "snippet": "text",
            "page": "page_number",
            "pages": "page_numbers",
            "score": "relevance_score",
            "source_id": "source_chunk_id",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

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

    def citation_label(self) -> str:
        if self.page_numbers:
            if len(self.page_numbers) == 1:
                return f"trang {self.page_numbers[0]}"

            return f"trang {self.page_numbers[0]}-{self.page_numbers[-1]}"

        if self.section_title:
            return self.section_title

        if self.source_document:
            return self.source_document

        return self.evidence_id

    def validate(self) -> Dict[str, Any]:
        errors = []
        warnings = []

        if not self.evidence_id:
            errors.append("evidence_id is required")

        if not self.evidence_type:
            errors.append("evidence_type is required")

        if not self.text:
            warnings.append("text is empty")

        if not self.page_numbers and self.page_number is None:
            warnings.append("page reference is missing")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


@dataclass
class Citation:
    citation_id: str = field(default_factory=lambda: make_id("citation"))

    citation_type: str = "page_citation"
    label: str = ""

    evidence_id: str = ""
    evidence_ids: List[str] = field(default_factory=list)

    document_id: str = ""
    source_document: str = ""

    page_number: Optional[int] = None
    page_numbers: List[int] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    section_id: str = ""
    section_title: str = ""

    chunk_id: str = ""

    bbox: List[float] = field(default_factory=list)

    quote: str = ""
    citation_text: str = ""
    citation_marker: str = ""

    confidence: float = 0.70
    verified: bool = False
    verification_status: str = "unverified"

    source: str = "evidence_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.label = normalize_text(self.label)
        self.quote = normalize_text(self.quote)
        self.citation_text = normalize_text(self.citation_text)
        self.citation_marker = normalize_text(self.citation_marker)

        self.page_numbers = normalize_page_numbers(self.page_numbers)

        if self.page_number is not None:
            page_number = safe_int(self.page_number, default=0)

            if page_number > 0 and page_number not in self.page_numbers:
                self.page_numbers.append(page_number)
                self.page_numbers = sorted(list(dict.fromkeys(self.page_numbers)))

            self.page_number = page_number or None

        if self.page_number is None and self.page_numbers:
            self.page_number = self.page_numbers[0]

        if self.page_numbers:
            if self.page_start is None:
                self.page_start = min(self.page_numbers)

            if self.page_end is None:
                self.page_end = max(self.page_numbers)

        if not self.evidence_ids and self.evidence_id:
            self.evidence_ids = [self.evidence_id]

        if not self.evidence_id and self.evidence_ids:
            self.evidence_id = self.evidence_ids[0]

        self.evidence_ids = [
            str(item) for item in self.evidence_ids if item
        ]

        self.bbox = normalize_bbox(self.bbox)
        self.confidence = clamp_float(self.confidence, default=0.70)

        if not self.label:
            self.label = self._make_default_label()

        if not self.citation_text:
            self.citation_text = self._make_citation_text()

        if not self.citation_marker:
            self.citation_marker = self._make_citation_marker()

    def _make_default_label(self) -> str:
        if self.page_numbers:
            if len(self.page_numbers) == 1:
                return f"Trang {self.page_numbers[0]}"

            return f"Trang {self.page_numbers[0]}-{self.page_numbers[-1]}"

        if self.section_title:
            return self.section_title

        if self.source_document:
            return self.source_document

        return self.citation_id

    def _make_citation_text(self) -> str:
        parts = []

        if self.source_document:
            parts.append(self.source_document)

        if self.section_title:
            parts.append(self.section_title)

        if self.page_numbers:
            if len(self.page_numbers) == 1:
                parts.append(f"trang {self.page_numbers[0]}")
            else:
                parts.append(f"trang {self.page_numbers[0]}-{self.page_numbers[-1]}")

        if not parts:
            return self.label

        return ", ".join(parts)

    def _make_citation_marker(self) -> str:
        if self.page_numbers:
            if len(self.page_numbers) == 1:
                return f"[tr.{self.page_numbers[0]}]"

            return f"[tr.{self.page_numbers[0]}-{self.page_numbers[-1]}]"

        if self.section_id:
            return f"[{self.section_id}]"

        return f"[{self.citation_id}]"

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Citation":
        data = dict(data or {})

        aliases = {
            "id": "citation_id",
            "type": "citation_type",
            "page": "page_number",
            "pages": "page_numbers",
            "text": "citation_text",
            "marker": "citation_marker",
        }

        for old_key, new_key in aliases.items():
            if old_key in data and new_key not in data:
                data[new_key] = data[old_key]

        allowed = set(cls.__dataclass_fields__.keys())

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)

    @classmethod
    def from_evidence(
        cls,
        evidence: Union[Evidence, Dict[str, Any]],
        citation_type: str = "evidence_citation",
    ) -> "Citation":
        if isinstance(evidence, dict):
            evidence = Evidence.from_dict(evidence)

        return cls(
            citation_type=citation_type,
            evidence_id=evidence.evidence_id,
            evidence_ids=[evidence.evidence_id],
            document_id=evidence.document_id,
            source_document=evidence.source_document,
            page_number=evidence.page_number,
            page_numbers=evidence.page_numbers,
            page_start=evidence.page_start,
            page_end=evidence.page_end,
            section_id=evidence.section_id,
            section_title=evidence.section_title,
            chunk_id=evidence.chunk_id,
            bbox=evidence.bbox,
            quote=evidence.quote or evidence.text,
            confidence=evidence.confidence,
            source="citation_from_evidence",
            metadata={
                "evidence_type": evidence.evidence_type,
                "source_chunk_id": evidence.source_chunk_id,
            },
        )

    def validate(self) -> Dict[str, Any]:
        errors = []
        warnings = []

        if not self.citation_id:
            errors.append("citation_id is required")

        if not self.citation_type:
            errors.append("citation_type is required")

        if not self.evidence_ids:
            warnings.append("citation has no evidence_ids")

        if not self.page_numbers and not self.section_id and not self.source_document:
            warnings.append("citation has weak source reference")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


@dataclass
class EvidenceRelation:
    evidence_relation_id: str = field(default_factory=lambda: make_id("evidence_rel"))

    source_evidence_id: str = ""
    target_evidence_id: str = ""
    relation_type: str = "related_to"

    source_page: Optional[int] = None
    target_page: Optional[int] = None

    confidence: float = 0.60
    weight: float = 1.0

    source: str = "evidence_schema"
    metadata: Dict[str, Any] = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.confidence = clamp_float(self.confidence, default=0.60)
        self.weight = clamp_float(self.weight, default=1.0, min_value=0.0, max_value=999999.0)

        if self.source_page is not None:
            self.source_page = safe_int(self.source_page, default=0) or None

        if self.target_page is not None:
            self.target_page = safe_int(self.target_page, default=0) or None

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRelation":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)


@dataclass
class EvidenceCollection:
    document_id: str = ""
    source_document: str = ""

    evidence: List[Evidence] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    relations: List[EvidenceRelation] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.evidence = [
            Evidence.from_dict(item) if isinstance(item, dict) else item
            for item in self.evidence
        ]

        self.citations = [
            Citation.from_dict(item) if isinstance(item, dict) else item
            for item in self.citations
        ]

        self.relations = [
            EvidenceRelation.from_dict(item) if isinstance(item, dict) else item
            for item in self.relations
        ]

    def add_evidence(
        self,
        item: Union[Evidence, Dict[str, Any]],
    ) -> Evidence:
        if isinstance(item, dict):
            item = Evidence.from_dict(item)

        if self.document_id and not item.document_id:
            item.document_id = self.document_id

        if self.source_document and not item.source_document:
            item.source_document = self.source_document

        self.evidence.append(item)
        return item

    def add_citation(
        self,
        item: Union[Citation, Dict[str, Any]],
    ) -> Citation:
        if isinstance(item, dict):
            item = Citation.from_dict(item)

        if self.document_id and not item.document_id:
            item.document_id = self.document_id

        if self.source_document and not item.source_document:
            item.source_document = self.source_document

        self.citations.append(item)
        return item

    def add_relation(
        self,
        item: Union[EvidenceRelation, Dict[str, Any]],
    ) -> EvidenceRelation:
        if isinstance(item, dict):
            item = EvidenceRelation.from_dict(item)

        self.relations.append(item)
        return item

    def build_citations_from_evidence(self) -> List[Citation]:
        existing_evidence_ids = set()

        for citation in self.citations:
            for evidence_id in citation.evidence_ids:
                existing_evidence_ids.add(evidence_id)

        for item in self.evidence:
            if item.evidence_id in existing_evidence_ids:
                continue

            self.citations.append(
                Citation.from_evidence(item)
            )

        return self.citations

    def deduplicate(self) -> None:
        seen = set()
        unique_evidence = []

        for item in self.evidence:
            key = item.evidence_id or item.content_hash

            if key in seen:
                continue

            seen.add(key)
            unique_evidence.append(item)

        self.evidence = unique_evidence

        seen_citations = set()
        unique_citations = []

        for citation in self.citations:
            key = (
                citation.citation_id,
                tuple(citation.evidence_ids),
                tuple(citation.page_numbers),
                citation.section_id,
            )

            if key in seen_citations:
                continue

            seen_citations.add(key)
            unique_citations.append(citation)

        self.citations = unique_citations

        seen_relations = set()
        unique_relations = []

        for relation in self.relations:
            key = (
                relation.source_evidence_id,
                relation.target_evidence_id,
                relation.relation_type,
            )

            if key in seen_relations:
                continue

            seen_relations.add(key)
            unique_relations.append(relation)

        self.relations = unique_relations

    def evidence_by_page(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in self.evidence:
            for page_number in item.page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(item.to_dict())

        return grouped

    def evidence_by_section(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in self.evidence:
            section_key = item.section_id or "no_section"
            grouped.setdefault(section_key, [])
            grouped[section_key].append(item.to_dict())

        return grouped

    def evidence_by_type(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for item in self.evidence:
            grouped.setdefault(item.evidence_type, [])
            grouped[item.evidence_type].append(item.to_dict())

        return grouped

    def citations_by_page(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for citation in self.citations:
            for page_number in citation.page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(citation.to_dict())

        return grouped

    def summary(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        by_page: Dict[str, int] = {}

        for item in self.evidence:
            by_type[item.evidence_type] = by_type.get(item.evidence_type, 0) + 1

            for page_number in item.page_numbers:
                page_key = str(page_number)
                by_page[page_key] = by_page.get(page_key, 0) + 1

        verified_citation_count = sum(
            1 for citation in self.citations
            if citation.verified
        )

        return {
            "document_id": self.document_id,
            "source_document": self.source_document,
            "evidence_count": len(self.evidence),
            "citation_count": len(self.citations),
            "relation_count": len(self.relations),
            "verified_citation_count": verified_citation_count,
            "unverified_citation_count": len(self.citations) - verified_citation_count,
            "page_count_with_evidence": len(by_page),
            "by_evidence_type": by_type,
            "by_page": by_page,
        }

    def validate(self) -> Dict[str, Any]:
        errors = []
        warnings = []

        evidence_ids = {
            item.evidence_id
            for item in self.evidence
            if item.evidence_id
        }

        for item in self.evidence:
            result = item.validate()

            for error in result.get("errors", []):
                errors.append(f"evidence:{item.evidence_id}: {error}")

            for warning in result.get("warnings", []):
                warnings.append(f"evidence:{item.evidence_id}: {warning}")

        for citation in self.citations:
            result = citation.validate()

            for error in result.get("errors", []):
                errors.append(f"citation:{citation.citation_id}: {error}")

            for warning in result.get("warnings", []):
                warnings.append(f"citation:{citation.citation_id}: {warning}")

            for evidence_id in citation.evidence_ids:
                if evidence_id and evidence_id not in evidence_ids:
                    warnings.append(
                        f"citation:{citation.citation_id}: evidence_id not found: {evidence_id}"
                    )

        for relation in self.relations:
            if relation.source_evidence_id and relation.source_evidence_id not in evidence_ids:
                warnings.append(
                    f"relation:{relation.evidence_relation_id}: source_evidence_id not found"
                )

            if relation.target_evidence_id and relation.target_evidence_id not in evidence_ids:
                warnings.append(
                    f"relation:{relation.evidence_relation_id}: target_evidence_id not found"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_document": self.source_document,
            "evidence": [
                item.to_dict() for item in self.evidence
            ],
            "citations": [
                item.to_dict() for item in self.citations
            ],
            "relations": [
                item.to_dict() for item in self.relations
            ],
            "evidence_by_page": self.evidence_by_page(),
            "evidence_by_section": self.evidence_by_section(),
            "evidence_by_type": self.evidence_by_type(),
            "citations_by_page": self.citations_by_page(),
            "evidence_summary": self.summary(),
            "metadata": json_safe(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceCollection":
        data = dict(data or {})

        return cls(
            document_id=data.get("document_id", ""),
            source_document=data.get("source_document", ""),
            evidence=data.get("evidence", []) or data.get("evidence_items", []) or [],
            citations=data.get("citations", []) or [],
            relations=data.get("relations", []) or data.get("evidence_relations", []) or [],
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
    def load_json(cls, input_path: Union[str, Path]) -> "EvidenceCollection":
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)


def make_evidence(
    text: str,
    evidence_type: str = "text_evidence",
    page_numbers: Optional[List[int]] = None,
    chunk_id: str = "",
    section_id: str = "",
    section_title: str = "",
    confidence: float = 0.70,
    relevance_score: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Evidence:
    return Evidence(
        evidence_type=evidence_type,
        text=text,
        page_numbers=page_numbers or [],
        chunk_id=chunk_id,
        source_chunk_id=chunk_id,
        section_id=section_id,
        section_title=section_title,
        confidence=confidence,
        relevance_score=relevance_score,
        metadata=metadata or {},
    )


def make_citation(
    evidence: Union[Evidence, Dict[str, Any]],
    citation_type: str = "evidence_citation",
) -> Citation:
    return Citation.from_evidence(
        evidence=evidence,
        citation_type=citation_type,
    )


def evidence_from_dict(data: Dict[str, Any]) -> Evidence:
    return Evidence.from_dict(data)


def evidence_to_dict(evidence: Union[Evidence, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(evidence, dict):
        return json_safe(evidence)

    return evidence.to_dict()


def citation_from_dict(data: Dict[str, Any]) -> Citation:
    return Citation.from_dict(data)


def citation_to_dict(citation: Union[Citation, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(citation, dict):
        return json_safe(citation)

    return citation.to_dict()


def evidence_collection_from_dict(data: Dict[str, Any]) -> EvidenceCollection:
    return EvidenceCollection.from_dict(data)


def evidence_collection_to_dict(
    collection: Union[EvidenceCollection, Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(collection, dict):
        return json_safe(collection)

    return collection.to_dict()


def save_evidence_json(
    evidence: Union[EvidenceCollection, Dict[str, Any], List[Union[Evidence, Dict[str, Any]]]],
    output_path: Union[str, Path],
    document_id: str = "",
    source_document: str = "",
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(evidence, EvidenceCollection):
        data = evidence.to_dict()
    elif isinstance(evidence, list):
        collection = EvidenceCollection(
            document_id=document_id,
            source_document=source_document,
        )

        for item in evidence:
            collection.add_evidence(item)

        collection.build_citations_from_evidence()
        data = collection.to_dict()
    else:
        data = json_safe(evidence)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return str(output_path)


def load_evidence_json(
    input_path: Union[str, Path],
) -> EvidenceCollection:
    return EvidenceCollection.load_json(input_path)


EvidenceItem = Evidence
EvidenceSchema = Evidence
CitationSchema = Citation
EvidenceCollectionSchema = EvidenceCollection
