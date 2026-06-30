"""
knowledge_graph_builder.py

Production V1 - Colab Ready

Purpose
-------
Build a lightweight, JSON-safe knowledge graph from document structure,
chunks, evidence, tables, and cross-page context.

This implementation intentionally avoids external dependencies. It focuses on
runtime compatibility for the Document AI pipeline and provides a useful graph
for GraphIndexBuilder / GraphRetriever.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from document_ai.schemas.page_raw_schema import PageRaw, normalize_pdf_text, normalize_text_for_match, json_safe
except Exception:  # pragma: no cover
    PageRaw = Any  # type: ignore

    def normalize_pdf_text(text: Any) -> str:
        return "" if text is None else str(text).strip()

    def normalize_text_for_match(text: Any) -> str:
        return normalize_pdf_text(text).lower()

    def json_safe(value: Any) -> Any:
        return value


@dataclass
class KnowledgeGraphBuilderConfig:
    include_pages: bool = True
    include_sections: bool = True
    include_paragraphs: bool = True
    include_chunks: bool = True
    include_table_chunks: bool = True
    include_evidence: bool = True
    include_tables: bool = True
    include_cross_page_context: bool = True
    include_keyword_nodes: bool = True
    attach_to_pages: bool = True
    max_keywords_per_chunk: int = 8
    max_text_preview_chars: int = 500
    min_keyword_len: int = 3
    include_debug: bool = True


class KnowledgeGraphBuilder:
    def __init__(self, config: Optional[KnowledgeGraphBuilderConfig] = None, *args, **kwargs):
        self.config = config or KnowledgeGraphBuilderConfig()

    def process(
        self,
        page_raws: Optional[List[Any]] = None,
        document_structure_result: Optional[Dict[str, Any]] = None,
        table_understanding_result: Optional[Dict[str, Any]] = None,
        cross_page_context_result: Optional[Dict[str, Any]] = None,
        chunk_result: Optional[Dict[str, Any]] = None,
        table_chunk_result: Optional[Dict[str, Any]] = None,
        evidence_result: Optional[Dict[str, Any]] = None,
        metadata_enrichment_result: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        page_raws = page_raws or []
        document_structure_result = document_structure_result or {}
        table_understanding_result = table_understanding_result or {}
        cross_page_context_result = cross_page_context_result or {}
        chunk_result = chunk_result or {}
        table_chunk_result = table_chunk_result or {}
        evidence_result = evidence_result or {}
        metadata_enrichment_result = metadata_enrichment_result or {}

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        node_ids = set()
        edge_ids = set()

        document_id = self._infer_document_id(page_raws, chunk_result, document_structure_result)
        source_document = self._infer_source_document(page_raws, metadata_enrichment_result)

        doc_node_id = f"document:{document_id or self._stable_id(source_document or 'document')}"
        self._add_node(
            nodes,
            node_ids,
            {
                "node_id": doc_node_id,
                "node_type": "document",
                "label": document_structure_result.get("title") or source_document or document_id or "Document",
                "text": document_structure_result.get("title") or source_document or "",
                "document_id": document_id,
                "source_document": source_document,
                "page_numbers": [getattr(p, "page_number", 0) for p in page_raws if getattr(p, "page_number", 0)],
                "metadata": {"source": "knowledge_graph_builder"},
            },
        )

        if self.config.include_pages:
            self._add_pages(page_raws, doc_node_id, nodes, edges, node_ids, edge_ids)

        section_nodes = {}
        if self.config.include_sections:
            section_nodes = self._add_sections(document_structure_result, doc_node_id, nodes, edges, node_ids, edge_ids)

        if self.config.include_paragraphs:
            self._add_paragraphs(document_structure_result, doc_node_id, section_nodes, nodes, edges, node_ids, edge_ids)

        if self.config.include_chunks:
            self._add_chunks(chunk_result, doc_node_id, section_nodes, nodes, edges, node_ids, edge_ids, source_name="chunk")

        if self.config.include_table_chunks:
            self._add_chunks(table_chunk_result, doc_node_id, section_nodes, nodes, edges, node_ids, edge_ids, source_name="table_chunk")

        if self.config.include_evidence:
            self._add_evidence(evidence_result, doc_node_id, nodes, edges, node_ids, edge_ids)

        if self.config.include_tables:
            self._add_tables(table_understanding_result, doc_node_id, nodes, edges, node_ids, edge_ids)

        if self.config.include_cross_page_context:
            self._add_context_graph(cross_page_context_result, nodes, edges, node_ids, edge_ids)

        if self.config.include_keyword_nodes:
            self._add_keyword_nodes(nodes, edges, node_ids, edge_ids)

        result = {
            "processor": "KnowledgeGraphBuilder",
            "schema_version": "knowledge_graph_v1",
            "document_id": document_id,
            "source_document": source_document,
            "nodes": nodes,
            "edges": edges,
            "knowledge_graph": {
                "nodes": nodes,
                "edges": edges,
            },
            "node_count": len(nodes),
            "edge_count": len(edges),
            "knowledge_graph_summary": self._build_summary(nodes, edges),
            "config": asdict(self.config),
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(page_raws, result)

        return json_safe(result)

    def _add_pages(self, page_raws, doc_node_id, nodes, edges, node_ids, edge_ids):
        prev_id = ""
        for page in page_raws:
            page_number = self._safe_int(getattr(page, "page_number", 0), 0)
            if page_number <= 0:
                continue
            node_id = f"page:{page_number}"
            text = normalize_pdf_text(getattr(page, "raw_text", "") or getattr(page, "normalized_text", ""))
            self._add_node(nodes, node_ids, {
                "node_id": node_id,
                "node_type": "page",
                "label": f"Page {page_number}",
                "text": self._preview(text),
                "page_number": page_number,
                "page_numbers": [page_number],
                "metadata": {
                    "text_length": len(text),
                    "word_count": getattr(page, "word_count", 0),
                    "image_count": getattr(page, "image_count", 0),
                    "drawing_count": getattr(page, "drawing_count", 0),
                },
            })
            self._add_edge(edges, edge_ids, doc_node_id, node_id, "HAS_PAGE", 1.0)
            if prev_id:
                self._add_edge(edges, edge_ids, prev_id, node_id, "NEXT_PAGE", 1.0)
            prev_id = node_id

    def _add_sections(self, ds, doc_node_id, nodes, edges, node_ids, edge_ids):
        section_nodes = {}
        sections = self._unwrap_list(ds.get("sections", []), "sections")
        prev_id = ""
        for i, section in enumerate(sections):
            section = self._to_dict(section)
            sid = str(section.get("section_id") or section.get("id") or f"section_{i+1}")
            node_id = f"section:{sid}"
            title = normalize_pdf_text(section.get("title") or section.get("heading") or f"Section {i+1}")
            pages = self._resolve_page_numbers(section)
            self._add_node(nodes, node_ids, {
                "node_id": node_id,
                "node_type": "section",
                "label": title,
                "title": title,
                "text": self._preview(section.get("text") or title),
                "section_id": sid,
                "section_title": title,
                "level": self._safe_int(section.get("level"), 0),
                "page_numbers": pages,
                "metadata": section,
            })
            section_nodes[sid] = node_id
            self._add_edge(edges, edge_ids, doc_node_id, node_id, "HAS_SECTION", 1.0)
            parent_id = section.get("parent_section_id") or section.get("parent_id")
            if parent_id and str(parent_id) in section_nodes:
                self._add_edge(edges, edge_ids, section_nodes[str(parent_id)], node_id, "HAS_SUBSECTION", 1.0)
            if prev_id:
                self._add_edge(edges, edge_ids, prev_id, node_id, "NEXT_SECTION", 1.0)
            prev_id = node_id
        return section_nodes

    def _add_paragraphs(self, ds, doc_node_id, section_nodes, nodes, edges, node_ids, edge_ids):
        paragraphs = self._unwrap_list(ds.get("paragraphs", []), "paragraphs")
        for i, paragraph in enumerate(paragraphs):
            paragraph = self._to_dict(paragraph)
            pid = str(paragraph.get("paragraph_id") or paragraph.get("id") or f"paragraph_{i+1}")
            node_id = f"paragraph:{pid}"
            text = normalize_pdf_text(paragraph.get("text") or paragraph.get("content") or "")
            pages = self._resolve_page_numbers(paragraph)
            self._add_node(nodes, node_ids, {
                "node_id": node_id,
                "node_type": "paragraph",
                "label": self._preview(text, 80) or pid,
                "text": self._preview(text),
                "paragraph_id": pid,
                "section_id": paragraph.get("section_id", ""),
                "page_numbers": pages,
                "metadata": paragraph,
            })
            sid = str(paragraph.get("section_id") or "")
            if sid and sid in section_nodes:
                self._add_edge(edges, edge_ids, section_nodes[sid], node_id, "HAS_PARAGRAPH", 0.9)
            else:
                self._add_edge(edges, edge_ids, doc_node_id, node_id, "HAS_PARAGRAPH", 0.5)

    def _add_chunks(self, chunk_result, doc_node_id, section_nodes, nodes, edges, node_ids, edge_ids, source_name="chunk"):
        chunks = []
        for key in ["chunks", "table_chunks", "parent_chunks", "child_chunks", "table_summary_chunks", "table_record_chunks", "table_row_chunks", "multi_page_table_chunks"]:
            chunks.extend(self._unwrap_list(chunk_result.get(key, []), key))
        collection = chunk_result.get("chunk_collection", {}) if isinstance(chunk_result, dict) else {}
        if isinstance(collection, dict):
            chunks.extend(self._unwrap_list(collection.get("chunks", []), "chunks"))
        seen = set()
        for i, chunk in enumerate(chunks):
            chunk = self._to_dict(chunk)
            cid = str(chunk.get("chunk_id") or chunk.get("id") or f"{source_name}_{i+1}")
            if cid in seen:
                continue
            seen.add(cid)
            node_id = f"chunk:{cid}"
            text = normalize_pdf_text(chunk.get("text") or chunk.get("content") or "")
            pages = self._resolve_page_numbers(chunk)
            chunk_type = chunk.get("chunk_type") or source_name
            self._add_node(nodes, node_ids, {
                "node_id": node_id,
                "node_type": "chunk",
                "label": chunk.get("title") or chunk.get("section_title") or self._preview(text, 80) or cid,
                "text": self._preview(text),
                "chunk_id": cid,
                "chunk_type": chunk_type,
                "section_id": chunk.get("section_id", ""),
                "table_id": chunk.get("table_id") or chunk.get("table_grid_id") or chunk.get("table_semantic_id") or "",
                "page_numbers": pages,
                "metadata": chunk,
            })
            sid = str(chunk.get("section_id") or "")
            if sid and sid in section_nodes:
                self._add_edge(edges, edge_ids, section_nodes[sid], node_id, "HAS_CHUNK", 0.85)
            else:
                self._add_edge(edges, edge_ids, doc_node_id, node_id, "HAS_CHUNK", 0.55)

    def _add_evidence(self, evidence_result, doc_node_id, nodes, edges, node_ids, edge_ids):
        evidences = []
        for key in ["evidence", "evidences", "evidence_items", "supporting_evidence", "aggregated_evidence"]:
            evidences.extend(self._unwrap_list(evidence_result.get(key, []), key))
        for i, evidence in enumerate(evidences):
            evidence = self._to_dict(evidence)
            eid = str(evidence.get("evidence_id") or evidence.get("id") or f"evidence_{i+1}")
            node_id = f"evidence:{eid}"
            text = normalize_pdf_text(evidence.get("text") or evidence.get("quote") or evidence.get("content") or "")
            self._add_node(nodes, node_ids, {
                "node_id": node_id,
                "node_type": "evidence",
                "label": self._preview(text, 80) or eid,
                "text": self._preview(text),
                "evidence_id": eid,
                "page_numbers": self._resolve_page_numbers(evidence),
                "metadata": evidence,
            })
            self._add_edge(edges, edge_ids, doc_node_id, node_id, "HAS_EVIDENCE", 0.7)

    def _add_tables(self, table_result, doc_node_id, nodes, edges, node_ids, edge_ids):
        tables = []
        for key in ["table_semantics", "semantic_tables", "table_grids", "table_structures", "table_boundaries", "multi_page_tables"]:
            tables.extend(self._unwrap_list(table_result.get(key, []), key))
        for i, table in enumerate(tables):
            table = self._to_dict(table)
            tid = str(table.get("table_id") or table.get("table_semantic_id") or table.get("table_grid_id") or table.get("table_structure_id") or table.get("table_boundary_id") or table.get("multi_page_table_id") or f"table_{i+1}")
            node_id = f"table:{tid}"
            title = normalize_pdf_text(table.get("title") or table.get("caption") or table.get("caption_text") or tid)
            headers = table.get("column_headers") or table.get("headers") or []
            text = " | ".join([title] + [normalize_pdf_text(h) for h in headers if normalize_pdf_text(h)])
            self._add_node(nodes, node_ids, {
                "node_id": node_id,
                "node_type": "table",
                "label": title,
                "title": title,
                "text": self._preview(text),
                "table_id": tid,
                "page_numbers": self._resolve_page_numbers(table),
                "metadata": table,
            })
            self._add_edge(edges, edge_ids, doc_node_id, node_id, "HAS_TABLE", 0.8)

    def _add_context_graph(self, context_result, nodes, edges, node_ids, edge_ids):
        graph = context_result.get("context_graph") or context_result.get("graph") or {}
        if not isinstance(graph, dict):
            return
        for node in self._unwrap_list(graph.get("nodes", []), "nodes"):
            node = self._to_dict(node)
            nid = str(node.get("node_id") or node.get("id") or "")
            if not nid:
                continue
            node.setdefault("node_id", nid)
            node.setdefault("node_type", node.get("type") or "context")
            node.setdefault("label", node.get("label") or node.get("title") or nid)
            node.setdefault("text", self._preview(node.get("text") or node.get("label") or ""))
            self._add_node(nodes, node_ids, node)
        for edge in self._unwrap_list(graph.get("edges", []), "edges"):
            edge = self._to_dict(edge)
            src = edge.get("source") or edge.get("source_id") or edge.get("from")
            tgt = edge.get("target") or edge.get("target_id") or edge.get("to")
            if src and tgt:
                self._add_edge(edges, edge_ids, str(src), str(tgt), edge.get("edge_type") or edge.get("type") or "RELATED_TO", self._safe_float(edge.get("weight"), 0.5), edge)

    def _add_keyword_nodes(self, nodes, edges, node_ids, edge_ids):
        original_nodes = list(nodes)
        for node in original_nodes:
            if node.get("node_type") not in {"chunk", "section", "paragraph", "table", "evidence"}:
                continue
            text = node.get("text", "") or node.get("label", "")
            for kw in self._keywords(text)[: self.config.max_keywords_per_chunk]:
                kw_id = f"keyword:{kw}"
                self._add_node(nodes, node_ids, {"node_id": kw_id, "node_type": "keyword", "label": kw, "text": kw, "metadata": {}})
                self._add_edge(edges, edge_ids, node.get("node_id"), kw_id, "MENTIONS_KEYWORD", 0.25)

    def _add_node(self, nodes, node_ids, node):
        node = self._to_dict(node)
        node_id = str(node.get("node_id") or node.get("id") or self._stable_id(node))
        if node_id in node_ids:
            return
        node["node_id"] = node_id
        node.setdefault("id", node_id)
        node.setdefault("node_type", "unknown")
        node.setdefault("label", node_id)
        node.setdefault("text", "")
        node.setdefault("metadata", {})
        node_ids.add(node_id)
        nodes.append(node)

    def _add_edge(self, edges, edge_ids, source, target, edge_type="RELATED_TO", weight=0.5, metadata=None):
        if not source or not target:
            return
        edge_id = f"edge:{source}->{target}:{edge_type}"
        if edge_id in edge_ids:
            return
        edge_ids.add(edge_id)
        edges.append({
            "edge_id": edge_id,
            "source": source,
            "target": target,
            "source_id": source,
            "target_id": target,
            "edge_type": edge_type,
            "type": edge_type,
            "weight": self._safe_float(weight, 0.5),
            "metadata": metadata or {},
        })

    def _unwrap_list(self, value, key="items"):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            if isinstance(value.get(key), list):
                return value.get(key)
            for k in ["items", "data", "results", "chunks", "nodes", "edges", "sections", "paragraphs"]:
                if isinstance(value.get(k), list):
                    return value.get(k)
        return []

    def _resolve_page_numbers(self, item):
        item = self._to_dict(item)
        nums = item.get("page_numbers") or item.get("content_page_numbers") or []
        if nums:
            return self._normalize_page_numbers(nums)
        start = self._safe_int(item.get("page_start"), 0)
        end = self._safe_int(item.get("page_end"), 0)
        if start > 0 and end >= start:
            return list(range(start, end + 1))
        page = self._safe_int(item.get("page_number"), 0)
        return [page] if page > 0 else []

    def _normalize_page_numbers(self, values):
        if not isinstance(values, list):
            values = [values]
        out = []
        for v in values:
            i = self._safe_int(v, 0)
            if i > 0:
                out.append(i)
        return sorted(list(dict.fromkeys(out)))

    def _keywords(self, text):
        stop = {"the", "and", "for", "with", "that", "this", "cua", "cac", "cho", "trong", "duoc", "khong", "bang", "muc", "noi", "dung"}
        tokens = re.findall(r"[a-z0-9_]+", normalize_text_for_match(text))
        seen = []
        for t in tokens:
            if len(t) >= self.config.min_keyword_len and t not in stop and t not in seen:
                seen.append(t)
        return seen

    def _build_summary(self, nodes, edges):
        by_type = {}
        for n in nodes:
            t = n.get("node_type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1
        by_edge = {}
        for e in edges:
            t = e.get("edge_type", "RELATED_TO")
            by_edge[t] = by_edge.get(t, 0) + 1
        return {"node_count": len(nodes), "edge_count": len(edges), "nodes_by_type": by_type, "edges_by_type": by_edge, "has_graph": bool(nodes)}

    def _attach_to_pages(self, page_raws, result):
        summary = result.get("knowledge_graph_summary", {})
        for p in page_raws or []:
            try:
                p.metadata.setdefault("knowledge_graph_builder", {})
                p.metadata["knowledge_graph_builder"] = summary
            except Exception:
                pass

    def _infer_document_id(self, page_raws, chunk_result, ds):
        if page_raws:
            return getattr(page_raws[0], "document_id", "") or "document"
        return chunk_result.get("document_id") or ds.get("document_id") or "document"

    def _infer_source_document(self, page_raws, metadata):
        if page_raws:
            return getattr(page_raws[0], "source_document", "") or getattr(page_raws[0], "source_path", "") or ""
        doc = metadata.get("document_metadata", {}) if isinstance(metadata, dict) else {}
        return doc.get("source_document", "") or doc.get("file_name", "")

    def _preview(self, text, max_chars=None):
        max_chars = max_chars or self.config.max_text_preview_chars
        text = normalize_pdf_text(text)
        return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."

    def _to_dict(self, value):
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "to_dict"):
            try:
                return value.to_dict()
            except Exception:
                pass
        if hasattr(value, "__dict__"):
            try:
                return dict(vars(value))
            except Exception:
                pass
        return {}

    def _safe_int(self, value, default=0):
        try:
            return int(value) if value is not None else default
        except Exception:
            return default

    def _safe_float(self, value, default=0.0):
        try:
            return float(value) if value is not None else default
        except Exception:
            return default

    def _stable_id(self, value):
        try:
            text = json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def save_result(self, result: Dict[str, Any], output_path: str) -> str:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_safe(result), f, ensure_ascii=False, indent=2)
        return str(output_path)
