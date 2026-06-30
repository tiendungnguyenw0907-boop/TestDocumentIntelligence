"""
document_structure_pipeline.py

Production V1 - Colab Ready

Purpose
-------
Run document-level structure understanding after page understanding.

Input
-----
List[PageRaw] from PageUnderstandingPipeline.

Output
------
document_structure dictionary with:
- title
- toc
- headings
- sections
- paragraphs
- lists
- document_tree

Flow
----
PageUnderstandingPipeline
    ↓
TitleDetector
    ↓
TOCDetector
    ↓
HeadingDetector
    ↓
SectionBuilder
    ↓
ParagraphBuilder
    ↓
ListDetector
    ↓
DocumentTreeBuilder
    ↓
document_structure.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from document_ai.schemas.page_raw_schema import PageRaw
from document_ai.pipelines.page_understanding_pipeline import PageUnderstandingPipeline

from document_ai.structure.title_detector import TitleDetector
from document_ai.structure.toc_detector import TOCDetector
from document_ai.structure.heading_detector import HeadingDetector
from document_ai.structure.section_builder import SectionBuilder
from document_ai.structure.paragraph_builder import ParagraphBuilder
from document_ai.structure.list_detector import ListDetector
from document_ai.structure.document_tree_builder import DocumentTreeBuilder


@dataclass
class DocumentStructurePipelineConfig:
    enable_title_detector: bool = True
    enable_toc_detector: bool = True
    enable_heading_detector: bool = True
    enable_section_builder: bool = True
    enable_paragraph_builder: bool = True
    enable_list_detector: bool = True
    enable_document_tree_builder: bool = True

    continue_on_error: bool = True

    save_json: bool = False
    output_dir: str = "outputs/document_structure"
    output_file_name: str = "document_structure.json"

    attach_result_to_pages: bool = True


class DocumentStructurePipeline:
    def __init__(
        self,
        config: Optional[DocumentStructurePipelineConfig] = None,
        page_understanding_pipeline: Optional[PageUnderstandingPipeline] = None,
        title_detector: Optional[TitleDetector] = None,
        toc_detector: Optional[TOCDetector] = None,
        heading_detector: Optional[HeadingDetector] = None,
        section_builder: Optional[SectionBuilder] = None,
        paragraph_builder: Optional[ParagraphBuilder] = None,
        list_detector: Optional[ListDetector] = None,
        document_tree_builder: Optional[DocumentTreeBuilder] = None,
    ):
        self.config = config or DocumentStructurePipelineConfig()

        self.page_understanding_pipeline = (
            page_understanding_pipeline or PageUnderstandingPipeline()
        )

        self.title_detector = title_detector or TitleDetector()
        self.toc_detector = toc_detector or TOCDetector()
        self.heading_detector = heading_detector or HeadingDetector()
        self.section_builder = section_builder or SectionBuilder()
        self.paragraph_builder = paragraph_builder or ParagraphBuilder()
        self.list_detector = list_detector or ListDetector()
        self.document_tree_builder = document_tree_builder or DocumentTreeBuilder()

    def process(
        self,
        page_raws: List[PageRaw],
    ) -> Dict[str, Any]:
        errors: List[str] = []

        title_result: Optional[Dict[str, Any]] = None
        toc_result: Optional[Dict[str, Any]] = None
        heading_result: Optional[Dict[str, Any]] = None
        section_result: Optional[Dict[str, Any]] = None
        paragraph_result: Optional[Dict[str, Any]] = None
        list_result: Optional[Dict[str, Any]] = None
        document_tree_result: Optional[Dict[str, Any]] = None

        if self.config.enable_title_detector:
            title_result = self._run_document_step(
                processor_name="TitleDetector",
                processor=self.title_detector,
                page_raws=page_raws,
                errors=errors,
            )

        if self.config.enable_toc_detector:
            toc_result = self._run_document_step(
                processor_name="TOCDetector",
                processor=self.toc_detector,
                page_raws=page_raws,
                errors=errors,
            )

        if self.config.enable_heading_detector:
            heading_result = self._run_document_step(
                processor_name="HeadingDetector",
                processor=self.heading_detector,
                page_raws=page_raws,
                errors=errors,
            )

        if self.config.enable_section_builder:
            section_result = self._run_section_builder(
                page_raws=page_raws,
                heading_result=heading_result,
                title_result=title_result,
                toc_result=toc_result,
                errors=errors,
            )

        if self.config.enable_paragraph_builder:
            paragraph_result = self._run_paragraph_builder(
                page_raws=page_raws,
                section_result=section_result,
                errors=errors,
            )

        if self.config.enable_list_detector:
            list_result = self._run_list_detector(
                page_raws=page_raws,
                paragraph_result=paragraph_result,
                errors=errors,
            )

        if self.config.enable_document_tree_builder:
            document_tree_result = self._run_document_tree_builder(
                page_raws=page_raws,
                title_result=title_result,
                toc_result=toc_result,
                heading_result=heading_result,
                section_result=section_result,
                paragraph_result=paragraph_result,
                list_result=list_result,
                errors=errors,
            )

        # ------------------------------------------------------------------
        # Compatibility note
        # ------------------------------------------------------------------
        # Downstream modules (context/knowledge/indexing/RAG) expect these
        # top-level keys to be flat lists:
        #   sections, paragraphs, headings, toc, lists
        # Older detector outputs are still preserved under *_result keys so no
        # diagnostic information is lost.
        result = {
            "schema_version": "document_structure_v1",
            "processor": "DocumentStructurePipeline",
            "completed": True,
            "error_count": len(errors),
            "errors": errors,
            "enabled_modules": self._enabled_modules(),
            "document_id": page_raws[0].document_id if page_raws else "",
            "page_count": len(page_raws),

            # Flat canonical outputs used by downstream pipelines.
            "title": self._extract_title_text(title_result),
            "toc": self._unwrap_list(toc_result, ["toc_entries", "toc"]),
            "headings": self._unwrap_list(heading_result, ["heading_candidates", "headings"]),
            "sections": self._unwrap_list(section_result, ["sections"]),
            "paragraphs": self._unwrap_list(paragraph_result, ["paragraphs"]),
            "lists": self._unwrap_list(list_result, ["lists"]),
            "document_tree": document_tree_result or {},

            # Raw module outputs kept for debugging/backward compatibility.
            "title_result": title_result or {},
            "toc_result": toc_result or {},
            "heading_result": heading_result or {},
            "section_result": section_result or {},
            "paragraph_result": paragraph_result or {},
            "list_result": list_result or {},
            "document_tree_result": document_tree_result or {},

            # Backward-compatible aliases.
            "toc_entries": self._unwrap_list(toc_result, ["toc_entries", "toc"]),
            "heading_candidates": self._unwrap_list(heading_result, ["heading_candidates", "headings"]),
            "document_sections": self._unwrap_list(section_result, ["sections"]),
            "document_paragraphs": self._unwrap_list(paragraph_result, ["paragraphs"]),
            "document_lists": self._unwrap_list(list_result, ["lists"]),

            "summary": self._build_summary(
                page_raws=page_raws,
                title_result=title_result,
                toc_result=toc_result,
                heading_result=heading_result,
                section_result=section_result,
                paragraph_result=paragraph_result,
                list_result=list_result,
                document_tree_result=document_tree_result,
                errors=errors,
            ),
        }

        if self.config.attach_result_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                result=result,
            )

        if self.config.save_json:
            self.save_document_structure(result)

        return result

    def process_document(
        self,
        document_path: Union[str, Path],
        start_page: int = 1,
        end_page: Optional[int] = None,
        max_pages: Optional[int] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        page_raws = self.page_understanding_pipeline.process_document(
            document_path=document_path,
            start_page=start_page,
            end_page=end_page,
            max_pages=max_pages,
            page_numbers=page_numbers,
        )

        return self.process(page_raws)

    def save_document_structure(
        self,
        result: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        if output_path is None:
            document_id = result.get("document_id") or "document"
            output_dir = Path(self.config.output_dir) / document_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / self.config.output_file_name

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                result,
                f,
                ensure_ascii=False,
                indent=2,
            )

        return str(output_path)

    def _run_document_step(
        self,
        processor_name: str,
        processor: Any,
        page_raws: List[PageRaw],
        errors: List[str],
    ) -> Optional[Dict[str, Any]]:
        try:
            return processor.process(page_raws)
        except Exception as exc:
            message = f"{processor_name} failed: {exc}"

            if not self.config.continue_on_error:
                raise RuntimeError(message) from exc

            errors.append(message)
            return None

    def _run_section_builder(
        self,
        page_raws: List[PageRaw],
        heading_result: Optional[Dict[str, Any]],
        title_result: Optional[Dict[str, Any]],
        toc_result: Optional[Dict[str, Any]],
        errors: List[str],
    ) -> Optional[Dict[str, Any]]:
        try:
            return self.section_builder.process(
                page_raws=page_raws,
                heading_result=heading_result,
                title_result=title_result,
                toc_result=toc_result,
            )
        except Exception as exc:
            message = f"SectionBuilder failed: {exc}"

            if not self.config.continue_on_error:
                raise RuntimeError(message) from exc

            errors.append(message)
            return None

    def _run_paragraph_builder(
        self,
        page_raws: List[PageRaw],
        section_result: Optional[Dict[str, Any]],
        errors: List[str],
    ) -> Optional[Dict[str, Any]]:
        try:
            return self.paragraph_builder.process(
                page_raws=page_raws,
                section_result=section_result,
            )
        except Exception as exc:
            message = f"ParagraphBuilder failed: {exc}"

            if not self.config.continue_on_error:
                raise RuntimeError(message) from exc

            errors.append(message)
            return None

    def _run_list_detector(
        self,
        page_raws: List[PageRaw],
        paragraph_result: Optional[Dict[str, Any]],
        errors: List[str],
    ) -> Optional[Dict[str, Any]]:
        try:
            return self.list_detector.process(
                page_raws=page_raws,
                paragraph_result=paragraph_result,
            )
        except Exception as exc:
            message = f"ListDetector failed: {exc}"

            if not self.config.continue_on_error:
                raise RuntimeError(message) from exc

            errors.append(message)
            return None

    def _run_document_tree_builder(
        self,
        page_raws: List[PageRaw],
        title_result: Optional[Dict[str, Any]],
        toc_result: Optional[Dict[str, Any]],
        heading_result: Optional[Dict[str, Any]],
        section_result: Optional[Dict[str, Any]],
        paragraph_result: Optional[Dict[str, Any]],
        list_result: Optional[Dict[str, Any]],
        errors: List[str],
    ) -> Optional[Dict[str, Any]]:
        try:
            return self.document_tree_builder.process(
                page_raws=page_raws,
                title_result=title_result,
                toc_result=toc_result,
                heading_result=heading_result,
                section_result=section_result,
                paragraph_result=paragraph_result,
                list_result=list_result,
            )
        except Exception as exc:
            message = f"DocumentTreeBuilder failed: {exc}"

            if not self.config.continue_on_error:
                raise RuntimeError(message) from exc

            errors.append(message)
            return None

    def _unwrap_list(
        self,
        result: Optional[Dict[str, Any]],
        candidate_keys: List[str],
    ) -> List[Any]:
        """
        Return a flat list from detector result dictionaries.

        This keeps DocumentStructurePipeline compatible with downstream modules
        that expect document_structure_result["sections"] etc. to be lists.
        """
        if result is None:
            return []

        if isinstance(result, list):
            return result

        if not isinstance(result, dict):
            return []

        for key in candidate_keys:
            value = result.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                # Some modules may return {"items": [...]} or nested canonical key.
                for nested_key in candidate_keys + ["items", "data", "results"]:
                    nested_value = value.get(nested_key)
                    if isinstance(nested_value, list):
                        return nested_value

        return []

    def _extract_title_text(
        self,
        title_result: Optional[Dict[str, Any]],
    ) -> str:
        """
        Extract a plain title string while keeping raw title_result separately.
        """
        if not title_result:
            return ""

        if isinstance(title_result, str):
            return title_result

        if not isinstance(title_result, dict):
            return ""

        summary = title_result.get("title_summary", {}) or {}
        selected = title_result.get("selected_title", {}) or {}

        for value in [
            summary.get("selected_title_text"),
            selected.get("title_text") if isinstance(selected, dict) else "",
            selected.get("text") if isinstance(selected, dict) else "",
            title_result.get("title"),
            title_result.get("text"),
        ]:
            if value:
                return str(value)

        candidates = title_result.get("title_candidates", []) or []
        if candidates:
            first = candidates[0]
            if isinstance(first, dict):
                return str(first.get("title_text") or first.get("text") or "")
            return str(first)

        return ""

    def _enabled_modules(
        self,
    ) -> List[str]:
        modules: List[str] = []

        if self.config.enable_title_detector:
            modules.append("TitleDetector")

        if self.config.enable_toc_detector:
            modules.append("TOCDetector")

        if self.config.enable_heading_detector:
            modules.append("HeadingDetector")

        if self.config.enable_section_builder:
            modules.append("SectionBuilder")

        if self.config.enable_paragraph_builder:
            modules.append("ParagraphBuilder")

        if self.config.enable_list_detector:
            modules.append("ListDetector")

        if self.config.enable_document_tree_builder:
            modules.append("DocumentTreeBuilder")

        return modules

    def _build_summary(
        self,
        page_raws: List[PageRaw],
        title_result: Optional[Dict[str, Any]],
        toc_result: Optional[Dict[str, Any]],
        heading_result: Optional[Dict[str, Any]],
        section_result: Optional[Dict[str, Any]],
        paragraph_result: Optional[Dict[str, Any]],
        list_result: Optional[Dict[str, Any]],
        document_tree_result: Optional[Dict[str, Any]],
        errors: List[str],
    ) -> Dict[str, Any]:
        title_text = ""

        if title_result:
            title_summary = title_result.get("title_summary", {})
            title_text = title_summary.get("selected_title_text", "")

        toc_summary = toc_result.get("toc_summary", {}) if toc_result else {}
        heading_summary = heading_result.get("heading_summary", {}) if heading_result else {}
        section_summary = section_result.get("section_summary", {}) if section_result else {}
        paragraph_summary = paragraph_result.get("paragraph_summary", {}) if paragraph_result else {}
        list_summary = list_result.get("list_summary", {}) if list_result else {}

        document_summary = {}

        if document_tree_result:
            document_summary = document_tree_result.get("document_summary", {})

        return {
            "document_id": page_raws[0].document_id if page_raws else "",
            "page_count": len(page_raws),
            "title": title_text,
            "has_toc": bool(toc_summary.get("has_toc", False)),
            "toc_entry_count": toc_summary.get("toc_entry_count", 0),
            "heading_count": heading_summary.get("heading_count", 0),
            "section_count": section_summary.get("section_count", 0),
            "paragraph_count": paragraph_summary.get("paragraph_count", 0),
            "list_count": list_summary.get("list_count", 0),
            "total_words": paragraph_summary.get(
                "total_words",
                document_summary.get("total_words", 0),
            ),
            "document_tree_node_count": document_summary.get("node_count", 0),
            "completed_with_errors": len(errors) > 0,
            "error_count": len(errors),
        }

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        result: Dict[str, Any],
    ) -> None:
        summary = result.get("summary", {})

        for page_raw in page_raws:
            page_raw.metadata.setdefault("document_structure_pipeline", {})
            page_raw.metadata["document_structure_pipeline"] = {
                "processor": "DocumentStructurePipeline",
                "document_id": result.get("document_id", ""),
                "summary": summary,
                "completed": True,
                "error_count": result.get("error_count", 0),
            }


def build_document_structure(
    page_raws: List[PageRaw],
) -> Dict[str, Any]:
    pipeline = DocumentStructurePipeline()
    return pipeline.process(page_raws)


def build_document_structure_from_file(
    document_path: Union[str, Path],
    max_pages: Optional[int] = None,
) -> Dict[str, Any]:
    pipeline = DocumentStructurePipeline()
    return pipeline.process_document(
        document_path=document_path,
        max_pages=max_pages,
    )
