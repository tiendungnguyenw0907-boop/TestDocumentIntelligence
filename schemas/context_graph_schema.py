"""
context_graph_schema.py

Production V1 - Colab Ready

Purpose
-------
Schema definitions for cross-page context graph.

Used by:
- CrossPageContextPipeline
- CrossPageContextGraphBuilder
- KnowledgeGraphBuilder
- GraphRetriever
- RAGPipeline

Main objects
------------
- ContextGraphNode
- ContextGraphEdge
- PageContext
- ContextGraph
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, asdict, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


SCHEMA_VERSION = "context_graph_schema_v1"


def make_id(prefix: str = "ctx") -> str:
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


def normalize_page_numbers(values: Any) -> List[int]:
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
class ContextGraphNode:
    node_id: str = field(default_factory=lambda: make_id("ctx_node"))
    node_type: str = "node"
    label: str = ""

    document_id: str = ""

    page_number: Optional[int] = None
    page_numbers: List[int] = field(default_factory=list)

    source_id: str = ""
    source_type: str = ""
    source: str = "context_graph_schema"

    confidence: float = 0.50
    weight: float = 1.0

    bbox: List[float] = field(default_factory=list)

    text: str = ""
    normalized_text: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.label = normalize_text(self.label)
        self.text = normalize_text(self.text)

        if not self.normalized_text:
            self.normalized_text = normalize_text_for_match(self.text or self.label)
        else:
            self.normalized_text = normalize_text_for_match(self.normalized_text)

        self.page_numbers = normalize_page_numbers(self.page_numbers)

        if self.page_number is not None:
            try:
                page_number = int(self.page_number)

                if page_number > 0 and page_number not in self.page_numbers:
                    self.page_numbers.append(page_number)
                    self.page_numbers = sorted(list(dict.fromkeys(self.page_numbers)))
            except Exception:
                self.page_number = None

        if self.page_number is None and self.page_numbers:
            self.page_number = self.page_numbers[0]

        self.bbox = normalize_bbox(self.bbox)
        self.confidence = self._clamp_float(self.confidence)
        self.weight = self._clamp_float(self.weight, min_value=0.0, max_value=999999.0)

    def _clamp_float(
        self,
        value: Any,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> float:
        try:
            value = float(value)
        except Exception:
            value = min_value

        return round(max(min_value, min(value, max_value)), 4)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextGraphNode":
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

        if not self.node_id:
            errors.append("node_id is required")

        if not self.node_type:
            errors.append("node_type is required")

        if not self.label:
            warnings.append("label is empty")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


@dataclass
class ContextGraphEdge:
    edge_id: str = field(default_factory=lambda: make_id("ctx_edge"))

    source_id: str = ""
    target_id: str = ""
    edge_type: str = "related_to"

    document_id: str = ""

    source_page: Optional[int] = None
    target_page: Optional[int] = None
    page_numbers: List[int] = field(default_factory=list)

    relation_label: str = ""

    source: str = "context_graph_schema"
    confidence: float = 0.50
    weight: float = 1.0
    directed: bool = True

    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.relation_label = normalize_text(self.relation_label)
        self.page_numbers = normalize_page_numbers(self.page_numbers)

        for page in [self.source_page, self.target_page]:
            if page is None:
                continue

            try:
                page = int(page)

                if page > 0 and page not in self.page_numbers:
                    self.page_numbers.append(page)
            except Exception:
                continue

        self.page_numbers = sorted(list(dict.fromkeys(self.page_numbers)))
        self.confidence = self._clamp_float(self.confidence)
        self.weight = self._clamp_float(self.weight, min_value=0.0, max_value=999999.0)

    def _clamp_float(
        self,
        value: Any,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ) -> float:
        try:
            value = float(value)
        except Exception:
            value = min_value

        return round(max(min_value, min(value, max_value)), 4)

    def to_dict(self) -> Dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextGraphEdge":
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

        if not self.edge_id:
            errors.append("edge_id is required")

        if not self.source_id:
            errors.append("source_id is required")

        if not self.target_id:
            errors.append("target_id is required")

        if not self.edge_type:
            errors.append("edge_type is required")

        if self.source_id == self.target_id:
            warnings.append("self-loop edge detected")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }


@dataclass
class PageContext:
    page_number: int
    page_node_id: str = ""

    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)

    neighbor_pages: List[int] = field(default_factory=list)

    sections: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    references: List[Dict[str, Any]] = field(default_factory=list)
    paragraph_continuations: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.page_node_id:
            self.page_node_id = f"page_{self.page_number}"

        self.neighbor_pages = normalize_page_numbers(self.neighbor_pages)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> Dict[str, Any]:
        data = json_safe(asdict(self))
        data["node_count"] = self.node_count
        data["edge_count"] = self.edge_count
        data["section_count"] = len(self.sections)
        data["table_count"] = len(self.tables)
        data["entity_count"] = len(self.entities)
        data["reference_count"] = len(self.references)
        data["paragraph_continuation_count"] = len(self.paragraph_continuations)

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageContext":
        data = dict(data or {})
        allowed = set(cls.__dataclass_fields__.keys())

        clean = {
            key: value
            for key, value in data.items()
            if key in allowed
        }

        return cls(**clean)


@dataclass
class ContextGraph:
    document_id: str = ""

    nodes: List[ContextGraphNode] = field(default_factory=list)
    edges: List[ContextGraphEdge] = field(default_factory=list)
    page_contexts: Dict[str, PageContext] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def add_node(
        self,
        node: Union[ContextGraphNode, Dict[str, Any]],
    ) -> ContextGraphNode:
        if isinstance(node, dict):
            node = ContextGraphNode.from_dict(node)

        if self.document_id and not node.document_id:
            node.document_id = self.document_id

        self.nodes.append(node)
        return node

    def add_edge(
        self,
        edge: Union[ContextGraphEdge, Dict[str, Any]],
    ) -> ContextGraphEdge:
        if isinstance(edge, dict):
            edge = ContextGraphEdge.from_dict(edge)

        if self.document_id and not edge.document_id:
            edge.document_id = self.document_id

        self.edges.append(edge)
        return edge

    def node_dicts(self) -> List[Dict[str, Any]]:
        return [
            node.to_dict() for node in self.nodes
        ]

    def edge_dicts(self) -> List[Dict[str, Any]]:
        return [
            edge.to_dict() for edge in self.edges
        ]

    def deduplicate(self) -> None:
        self.nodes = self._deduplicate_nodes(self.nodes)
        self.edges = self._deduplicate_edges(self.edges)

    def _deduplicate_nodes(
        self,
        nodes: List[ContextGraphNode],
    ) -> List[ContextGraphNode]:
        seen = set()
        result = []

        for node in nodes:
            key = node.node_id

            if not key:
                continue

            if key in seen:
                continue

            seen.add(key)
            result.append(node)

        return result

    def _deduplicate_edges(
        self,
        edges: List[ContextGraphEdge],
    ) -> List[ContextGraphEdge]:
        seen = set()
        result = []

        for edge in edges:
            key = (
                edge.source_id,
                edge.target_id,
                edge.edge_type,
                str(edge.source_page),
                str(edge.target_page),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(edge)

        return result

    def nodes_by_type(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for node in self.nodes:
            grouped.setdefault(node.node_type, [])
            grouped[node.node_type].append(node.to_dict())

        return grouped

    def edges_by_type(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for edge in self.edges:
            grouped.setdefault(edge.edge_type, [])
            grouped[edge.edge_type].append(edge.to_dict())

        return grouped

    def nodes_by_page(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for node in self.nodes:
            for page_number in node.page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(node.to_dict())

        return grouped

    def edges_by_page(self) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for edge in self.edges:
            page_numbers = edge.page_numbers[:]

            if edge.source_page:
                page_numbers.append(edge.source_page)

            if edge.target_page:
                page_numbers.append(edge.target_page)

            page_numbers = normalize_page_numbers(page_numbers)

            for page_number in page_numbers:
                page_key = str(page_number)
                grouped.setdefault(page_key, [])
                grouped[page_key].append(edge.to_dict())

        return grouped

    def build_adjacency(self) -> Dict[str, Any]:
        outgoing: Dict[str, List[Dict[str, Any]]] = {}
        incoming: Dict[str, List[Dict[str, Any]]] = {}

        for node in self.nodes:
            outgoing.setdefault(node.node_id, [])
            incoming.setdefault(node.node_id, [])

        for edge in self.edges:
            edge_data = edge.to_dict()

            outgoing.setdefault(edge.source_id, [])
            outgoing[edge.source_id].append(edge_data)

            incoming.setdefault(edge.target_id, [])
            incoming[edge.target_id].append(edge_data)

            if not edge.directed:
                outgoing.setdefault(edge.target_id, [])
                outgoing[edge.target_id].append(edge_data)

                incoming.setdefault(edge.source_id, [])
                incoming[edge.source_id].append(edge_data)

        degree = {}

        node_ids = set(outgoing.keys()).union(set(incoming.keys()))

        for node_id in node_ids:
            degree[node_id] = {
                "in_degree": len(incoming.get(node_id, [])),
                "out_degree": len(outgoing.get(node_id, [])),
                "total_degree": len(incoming.get(node_id, [])) + len(outgoing.get(node_id, [])),
            }

        return {
            "outgoing": outgoing,
            "incoming": incoming,
            "degree": degree,
        }

    def build_page_contexts(self) -> Dict[str, PageContext]:
        nodes_by_page = self.nodes_by_page()
        edges_by_page = self.edges_by_page()

        page_numbers = set()

        for page_key in nodes_by_page:
            try:
                page_numbers.add(int(page_key))
            except Exception:
                continue

        for page_key in edges_by_page:
            try:
                page_numbers.add(int(page_key))
            except Exception:
                continue

        page_contexts: Dict[str, PageContext] = {}

        for page_number in sorted(page_numbers):
            page_key = str(page_number)
            nodes = nodes_by_page.get(page_key, [])
            edges = edges_by_page.get(page_key, [])

            sections = [
                node for node in nodes
                if node.get("node_type") == "section"
            ]

            tables = [
                node for node in nodes
                if node.get("node_type") in ["table", "multi_page_table"]
            ]

            entities = [
                node for node in nodes
                if node.get("node_type") == "entity"
            ]

            references = [
                node for node in nodes
                if node.get("node_type") == "reference"
            ]

            paragraph_continuations = [
                node for node in nodes
                if node.get("node_type") == "paragraph_continuation"
            ]

            neighbor_pages = []

            for edge in edges:
                source_page = edge.get("source_page")
                target_page = edge.get("target_page")

                if source_page == page_number and target_page:
                    neighbor_pages.append(target_page)

                if target_page == page_number and source_page:
                    neighbor_pages.append(source_page)

            page_contexts[page_key] = PageContext(
                page_number=page_number,
                page_node_id=f"page_{page_number}",
                nodes=nodes,
                edges=edges,
                neighbor_pages=normalize_page_numbers(neighbor_pages),
                sections=sections,
                tables=tables,
                entities=entities,
                references=references,
                paragraph_continuations=paragraph_continuations,
            )

        self.page_contexts = page_contexts
        return page_contexts

    def summary(self) -> Dict[str, Any]:
        by_node_type: Dict[str, int] = {}
        by_edge_type: Dict[str, int] = {}

        for node in self.nodes:
            by_node_type[node.node_type] = by_node_type.get(node.node_type, 0) + 1

        for edge in self.edges:
            by_edge_type[edge.edge_type] = by_edge_type.get(edge.edge_type, 0) + 1

        page_numbers = set()

        for node in self.nodes:
            for page_number in node.page_numbers:
                page_numbers.add(page_number)

        for edge in self.edges:
            for page_number in edge.page_numbers:
                page_numbers.add(page_number)

        return {
            "document_id": self.document_id,
            "has_context_graph": len(self.nodes) > 0,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "page_context_count": len(self.page_contexts),
            "page_count_with_graph_items": len(page_numbers),
            "by_node_type": by_node_type,
            "by_edge_type": by_edge_type,
        }

    def validate(self) -> Dict[str, Any]:
        errors = []
        warnings = []

        node_ids = {
            node.node_id
            for node in self.nodes
            if node.node_id
        }

        for node in self.nodes:
            result = node.validate()

            for error in result.get("errors", []):
                errors.append(f"node:{node.node_id}: {error}")

            for warning in result.get("warnings", []):
                warnings.append(f"node:{node.node_id}: {warning}")

        for edge in self.edges:
            result = edge.validate()

            for error in result.get("errors", []):
                errors.append(f"edge:{edge.edge_id}: {error}")

            for warning in result.get("warnings", []):
                warnings.append(f"edge:{edge.edge_id}: {warning}")

            if edge.source_id and edge.source_id not in node_ids:
                warnings.append(f"edge:{edge.edge_id}: source_id not found in nodes: {edge.source_id}")

            if edge.target_id and edge.target_id not in node_ids:
                warnings.append(f"edge:{edge.edge_id}: target_id not found in nodes: {edge.target_id}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def to_dict(self) -> Dict[str, Any]:
        if not self.page_contexts:
            self.build_page_contexts()

        return {
            "document_id": self.document_id,
            "nodes": self.node_dicts(),
            "edges": self.edge_dicts(),
            "adjacency": self.build_adjacency(),
            "nodes_by_type": self.nodes_by_type(),
            "edges_by_type": self.edges_by_type(),
            "nodes_by_page": self.nodes_by_page(),
            "edges_by_page": self.edges_by_page(),
            "page_contexts": {
                page_key: page_context.to_dict()
                for page_key, page_context in self.page_contexts.items()
            },
            "context_graph_summary": self.summary(),
            "metadata": json_safe(self.metadata),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextGraph":
        data = dict(data or {})

        graph = cls(
            document_id=data.get("document_id", ""),
            metadata=data.get("metadata", {}) or {},
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )

        for item in data.get("nodes", []) or []:
            graph.nodes.append(ContextGraphNode.from_dict(item))

        for item in data.get("edges", []) or []:
            graph.edges.append(ContextGraphEdge.from_dict(item))

        page_contexts = data.get("page_contexts", {}) or {}

        for page_key, page_context in page_contexts.items():
            if isinstance(page_context, dict):
                graph.page_contexts[str(page_key)] = PageContext.from_dict(page_context)

        return graph

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
    def load_json(cls, input_path: Union[str, Path]) -> "ContextGraph":
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)


def make_page_node(
    page_number: int,
    document_id: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> ContextGraphNode:
    return ContextGraphNode(
        node_id=f"page_{page_number}",
        node_type="page",
        label=f"Page {page_number}",
        document_id=document_id,
        page_number=page_number,
        page_numbers=[page_number],
        source_id=f"page_{page_number}",
        source_type="page",
        confidence=1.0,
        metadata=metadata or {},
    )


def make_section_node(
    section_id: str,
    title: str,
    page_numbers: Optional[List[int]] = None,
    document_id: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> ContextGraphNode:
    return ContextGraphNode(
        node_id=f"section_{section_id}",
        node_type="section",
        label=title,
        document_id=document_id,
        page_numbers=page_numbers or [],
        source_id=section_id,
        source_type="section",
        confidence=0.75,
        metadata=metadata or {},
    )


def make_edge(
    source_id: str,
    target_id: str,
    edge_type: str = "related_to",
    source_page: Optional[int] = None,
    target_page: Optional[int] = None,
    confidence: float = 0.60,
    weight: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> ContextGraphEdge:
    return ContextGraphEdge(
        source_id=source_id,
        target_id=target_id,
        edge_type=edge_type,
        source_page=source_page,
        target_page=target_page,
        confidence=confidence,
        weight=weight,
        metadata=metadata or {},
    )


def context_graph_from_dict(data: Dict[str, Any]) -> ContextGraph:
    return ContextGraph.from_dict(data)


def context_graph_to_dict(graph: Union[ContextGraph, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(graph, dict):
        return json_safe(graph)

    return graph.to_dict()


def save_context_graph_json(
    graph: Union[ContextGraph, Dict[str, Any]],
    output_path: Union[str, Path],
) -> str:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(graph, ContextGraph):
        data = graph.to_dict()
    else:
        data = json_safe(graph)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    return str(output_path)


def load_context_graph_json(
    input_path: Union[str, Path],
) -> ContextGraph:
    return ContextGraph.load_json(input_path)
