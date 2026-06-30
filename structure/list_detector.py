"""
list_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect list items and group them into document lists.

Input
-----
List[PageRaw] after:
- PageUnderstandingPipeline
- TitleDetector
- TOCDetector
- HeadingDetector
- SectionBuilder
- ParagraphBuilder

Output
------
Dictionary with:
- list_items
- lists
- lists_by_section
- list_summary

Flow
----
ParagraphBuilder
    ↓
ListDetector
    ↓
DocumentTreeBuilder
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class ListDetectorConfig:
    min_item_text_chars: int = 1
    max_item_text_chars: int = 3000

    attach_to_pages: bool = True

    use_paragraphs_first: bool = True
    use_text_lines_fallback: bool = True

    detect_bullets: bool = True
    detect_numbered: bool = True
    detect_letters: bool = True
    detect_roman: bool = True
    detect_dash_items: bool = True

    group_consecutive_items: bool = True
    max_vertical_gap: float = 35.0
    max_indent_difference: float = 30.0

    include_text_preview: bool = True
    text_preview_chars: int = 500


@dataclass
class DocumentListItem:
    list_item_id: str
    page_number: int
    page_index: int
    order: int
    raw_text: str
    item_text: str
    marker: str
    marker_type: str
    level: int

    bbox: Optional[List[float]] = None
    section_id: str = ""
    section_title: str = ""
    paragraph_id: str = ""
    list_id: str = ""
    confidence: float = 0.5
    source: str = "unknown"
    source_object_id: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


@dataclass
class DocumentList:
    list_id: str
    list_type: str
    page_start: int
    page_end: int
    order: int

    section_id: str = ""
    section_title: str = ""
    item_count: int = 0
    levels: Optional[List[int]] = None
    item_ids: Optional[List[str]] = None
    text_preview: str = ""
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["levels"] is None:
            data["levels"] = []

        if data["item_ids"] is None:
            data["item_ids"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class ListDetector:
    def __init__(
        self,
        config: Optional[ListDetectorConfig] = None,
    ):
        self.config = config or ListDetectorConfig()

    def process(
        self,
        page_raws: List[PageRaw],
        paragraph_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        objects = self._collect_candidate_objects(
            page_raws=page_raws,
            paragraph_result=paragraph_result,
        )

        list_items = self._detect_list_items(objects)

        list_items = self._sort_items(list_items)

        for order, item in enumerate(list_items, start=1):
            item.order = order

        lists = self._group_items_into_lists(list_items)

        lists_by_section = self._group_lists_by_section(lists)

        result = {
            "processor": "ListDetector",
            "list_items": [
                item.to_dict() for item in list_items
            ],
            "lists": [
                item.to_dict() for item in lists
            ],
            "lists_by_section": lists_by_section,
            "list_summary": self._build_summary(
                list_items=list_items,
                lists=lists,
            ),
            "config": {
                "detect_bullets": self.config.detect_bullets,
                "detect_numbered": self.config.detect_numbered,
                "detect_letters": self.config.detect_letters,
                "detect_roman": self.config.detect_roman,
                "detect_dash_items": self.config.detect_dash_items,
                "group_consecutive_items": self.config.group_consecutive_items,
            },
        }

        if self.config.attach_to_pages:
            self._attach_to_pages(
                page_raws=page_raws,
                list_items=list_items,
                lists=lists,
                result=result,
            )

        return result

    def _collect_candidate_objects(
        self,
        page_raws: List[PageRaw],
        paragraph_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.config.use_paragraphs_first:
            objects = self._collect_from_paragraphs(
                page_raws=page_raws,
                paragraph_result=paragraph_result,
            )

            if objects:
                return objects

        if self.config.use_text_lines_fallback:
            return self._collect_from_text_lines(page_raws)

        return []

    def _collect_from_paragraphs(
        self,
        page_raws: List[PageRaw],
        paragraph_result: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        paragraph_items: List[Dict[str, Any]] = []

        if paragraph_result:
            paragraphs = paragraph_result.get("paragraphs", [])

            for index, paragraph in enumerate(paragraphs):
                paragraph_items.append(
                    {
                        "object_id": paragraph.get("paragraph_id", ""),
                        "text": paragraph.get("text", ""),
                        "bbox": paragraph.get("bbox"),
                        "page_number": paragraph.get("page_number"),
                        "page_index": paragraph.get("page_index"),
                        "section_id": paragraph.get("section_id", ""),
                        "section_title": paragraph.get("section_title", ""),
                        "paragraph_id": paragraph.get("paragraph_id", ""),
                        "source": "paragraph_builder.paragraphs",
                        "source_index": index,
                        "metadata": paragraph.get("metadata", {}),
                    }
                )

            if paragraph_items:
                return paragraph_items

        for page_raw in page_raws:
            paragraph_meta = page_raw.metadata.get("paragraph_builder", {})
            paragraphs = paragraph_meta.get("paragraphs_on_page", [])

            for index, paragraph in enumerate(paragraphs):
                paragraph_items.append(
                    {
                        "object_id": paragraph.get("paragraph_id", ""),
                        "text": paragraph.get("text", ""),
                        "bbox": paragraph.get("bbox"),
                        "page_number": paragraph.get("page_number", page_raw.page_number),
                        "page_index": paragraph.get("page_index", page_raw.page_index),
                        "section_id": paragraph.get("section_id", ""),
                        "section_title": paragraph.get("section_title", ""),
                        "paragraph_id": paragraph.get("paragraph_id", ""),
                        "source": "paragraph_builder.paragraphs_on_page",
                        "source_index": index,
                        "metadata": paragraph.get("metadata", {}),
                    }
                )

        return paragraph_items

    def _collect_from_text_lines(
        self,
        page_raws: List[PageRaw],
    ) -> List[Dict[str, Any]]:
        objects: List[Dict[str, Any]] = []

        for page_raw in page_raws:
            for index, line in enumerate(page_raw.text_lines):
                text = line.normalized_text or line.text or ""

                if not text.strip():
                    continue

                objects.append(
                    {
                        "object_id": line.line_id,
                        "text": text,
                        "bbox": line.bbox,
                        "page_number": page_raw.page_number,
                        "page_index": page_raw.page_index,
                        "section_id": "",
                        "section_title": "",
                        "paragraph_id": "",
                        "source": "page_raw.text_lines",
                        "source_index": index,
                        "metadata": line.metadata or {},
                    }
                )

        return objects

    def _detect_list_items(
        self,
        objects: List[Dict[str, Any]],
    ) -> List[DocumentListItem]:
        items: List[DocumentListItem] = []

        for obj in objects:
            raw_text = self._clean_text(obj.get("text", ""))

            if not raw_text:
                continue

            parsed = self._parse_list_item(raw_text)

            if parsed is None:
                continue

            item_text = parsed.get("item_text", "")

            if not self._is_valid_item_text(item_text):
                continue

            bbox = obj.get("bbox")

            level = parsed.get("level", 1)

            if bbox:
                level = max(level, self._infer_level_from_indent(bbox))

            items.append(
                DocumentListItem(
                    list_item_id=make_id("listitem"),
                    page_number=self._safe_int(obj.get("page_number"), default=1),
                    page_index=self._safe_int(obj.get("page_index"), default=0),
                    order=0,
                    raw_text=raw_text,
                    item_text=item_text,
                    marker=parsed.get("marker", ""),
                    marker_type=parsed.get("marker_type", "unknown"),
                    level=level,
                    bbox=bbox,
                    section_id=obj.get("section_id", ""),
                    section_title=obj.get("section_title", ""),
                    paragraph_id=obj.get("paragraph_id", ""),
                    confidence=parsed.get("confidence", 0.5),
                    source=obj.get("source", "unknown"),
                    source_object_id=obj.get("object_id", ""),
                    metadata={
                        "source_index": obj.get("source_index"),
                        "object_metadata": obj.get("metadata", {}),
                        "text_preview": self._text_preview(item_text),
                    },
                )
            )

        return items

    def _parse_list_item(
        self,
        text: str,
    ) -> Optional[Dict[str, Any]]:
        clean = self._clean_text(text)

        if not clean:
            return None

        patterns: List[Tuple[str, str]] = []

        if self.config.detect_bullets:
            patterns.extend(
                [
                    ("bullet", r"^(?P<marker>[•●○▪▫■□◦])\s+(?P<text>.+)$"),
                ]
            )

        if self.config.detect_dash_items:
            patterns.extend(
                [
                    ("dash", r"^(?P<marker>[-–—])\s+(?P<text>.+)$"),
                ]
            )

        if self.config.detect_numbered:
            patterns.extend(
                [
                    ("numbered", r"^(?P<marker>\d+(?:\.\d+)*[\.\)])\s+(?P<text>.+)$"),
                    ("numbered", r"^(?P<marker>\(\d+\))\s+(?P<text>.+)$"),
                ]
            )

        if self.config.detect_letters:
            patterns.extend(
                [
                    ("letter", r"^(?P<marker>[a-zA-Z][\.\)])\s+(?P<text>.+)$"),
                    ("letter", r"^(?P<marker>\([a-zA-Z]\))\s+(?P<text>.+)$"),
                ]
            )

        if self.config.detect_roman:
            patterns.extend(
                [
                    ("roman", r"^(?P<marker>[IVXLCDM]+[\.\)])\s+(?P<text>.+)$"),
                    ("roman", r"^(?P<marker>\([IVXLCDM]+\))\s+(?P<text>.+)$"),
                ]
            )

        for marker_type, pattern in patterns:
            match = re.match(pattern, clean)

            if not match:
                continue

            marker = match.group("marker").strip()
            item_text = match.group("text").strip()

            level = self._infer_level_from_marker(
                marker=marker,
                marker_type=marker_type,
            )

            confidence = self._score_list_item(
                marker=marker,
                marker_type=marker_type,
                item_text=item_text,
            )

            return {
                "marker": marker,
                "marker_type": marker_type,
                "item_text": item_text,
                "level": level,
                "confidence": confidence,
            }

        return None

    def _group_items_into_lists(
        self,
        items: List[DocumentListItem],
    ) -> List[DocumentList]:
        if not self.config.group_consecutive_items:
            return self._one_item_per_list(items)

        if not items:
            return []

        lists: List[DocumentList] = []
        current_items: List[DocumentListItem] = []

        for item in items:
            if not current_items:
                current_items = [item]
                continue

            previous = current_items[-1]

            if self._should_continue_list(previous, item):
                current_items.append(item)
            else:
                lists.append(
                    self._make_list_from_items(
                        items=current_items,
                        order=len(lists) + 1,
                    )
                )
                current_items = [item]

        if current_items:
            lists.append(
                self._make_list_from_items(
                    items=current_items,
                    order=len(lists) + 1,
                )
            )

        return lists

    def _one_item_per_list(
        self,
        items: List[DocumentListItem],
    ) -> List[DocumentList]:
        lists: List[DocumentList] = []

        for index, item in enumerate(items, start=1):
            lists.append(
                self._make_list_from_items(
                    items=[item],
                    order=index,
                )
            )

        return lists

    def _should_continue_list(
        self,
        previous: DocumentListItem,
        current: DocumentListItem,
    ) -> bool:
        if previous.section_id != current.section_id:
            return False

        if current.page_number < previous.page_number:
            return False

        if current.marker_type != previous.marker_type:
            compatible = {
                ("numbered", "letter"),
                ("letter", "numbered"),
                ("numbered", "roman"),
                ("roman", "numbered"),
                ("bullet", "dash"),
                ("dash", "bullet"),
            }

            if (previous.marker_type, current.marker_type) not in compatible:
                return False

        if current.page_number == previous.page_number:
            gap = self._vertical_gap(previous.bbox, current.bbox)

            if gap is not None and gap > self.config.max_vertical_gap:
                return False

            indent_diff = self._indent_difference(previous.bbox, current.bbox)

            if indent_diff is not None and indent_diff > self.config.max_indent_difference:
                if abs(previous.level - current.level) > 1:
                    return False

        return True

    def _make_list_from_items(
        self,
        items: List[DocumentListItem],
        order: int,
    ) -> DocumentList:
        list_id = make_id("list")

        for item in items:
            item.list_id = list_id

        page_numbers = [
            item.page_number for item in items
        ]

        marker_types = [
            item.marker_type for item in items
        ]

        list_type = self._infer_list_type(marker_types)

        levels = sorted(
            {
                item.level for item in items
            }
        )

        text_preview = "\n".join(
            [
                f"{item.marker} {item.item_text}"
                for item in items
            ]
        )

        if self.config.include_text_preview:
            text_preview = text_preview[: self.config.text_preview_chars]
        else:
            text_preview = ""

        first = items[0]

        return DocumentList(
            list_id=list_id,
            list_type=list_type,
            page_start=min(page_numbers),
            page_end=max(page_numbers),
            order=order,
            section_id=first.section_id,
            section_title=first.section_title,
            item_count=len(items),
            levels=levels,
            item_ids=[
                item.list_item_id for item in items
            ],
            text_preview=text_preview,
            metadata={
                "marker_types": sorted(set(marker_types)),
                "source_pages": sorted(set(page_numbers)),
                "first_item_text": first.item_text,
                "last_item_text": items[-1].item_text,
            },
        )

    def _infer_list_type(
        self,
        marker_types: List[str],
    ) -> str:
        unique_types = set(marker_types)

        if unique_types == {"bullet"}:
            return "bullet_list"

        if unique_types == {"dash"}:
            return "dash_list"

        if unique_types == {"numbered"}:
            return "numbered_list"

        if unique_types == {"letter"}:
            return "letter_list"

        if unique_types == {"roman"}:
            return "roman_list"

        if unique_types.intersection({"numbered", "letter", "roman"}):
            return "ordered_mixed_list"

        if unique_types.intersection({"bullet", "dash"}):
            return "unordered_mixed_list"

        return "mixed_list"

    def _group_lists_by_section(
        self,
        lists: List[DocumentList],
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}

        for doc_list in lists:
            section_id = doc_list.section_id or "unassigned"
            grouped.setdefault(section_id, [])
            grouped[section_id].append(doc_list.to_dict())

        return grouped

    def _attach_to_pages(
        self,
        page_raws: List[PageRaw],
        list_items: List[DocumentListItem],
        lists: List[DocumentList],
        result: Dict[str, Any],
    ) -> None:
        items_by_page: Dict[int, List[Dict[str, Any]]] = {}
        lists_by_page: Dict[int, List[Dict[str, Any]]] = {}

        for item in list_items:
            items_by_page.setdefault(item.page_number, [])
            items_by_page[item.page_number].append(item.to_dict())

        for doc_list in lists:
            for page_number in range(doc_list.page_start, doc_list.page_end + 1):
                lists_by_page.setdefault(page_number, [])
                lists_by_page[page_number].append(doc_list.to_dict())

        for page_raw in page_raws:
            page_raw.metadata.setdefault("list_detector", {})
            page_raw.metadata["list_detector"] = {
                "processor": "ListDetector",
                "list_items_on_page": items_by_page.get(page_raw.page_number, []),
                "lists_on_page": lists_by_page.get(page_raw.page_number, []),
                "list_item_count_on_page": len(items_by_page.get(page_raw.page_number, [])),
                "list_count_on_page": len(lists_by_page.get(page_raw.page_number, [])),
                "list_summary": result.get("list_summary", {}),
            }

    def _build_summary(
        self,
        list_items: List[DocumentListItem],
        lists: List[DocumentList],
    ) -> Dict[str, Any]:
        by_marker_type: Dict[str, int] = {}
        by_list_type: Dict[str, int] = {}
        by_level: Dict[str, int] = {}

        for item in list_items:
            by_marker_type[item.marker_type] = by_marker_type.get(item.marker_type, 0) + 1
            level_key = str(item.level)
            by_level[level_key] = by_level.get(level_key, 0) + 1

        for doc_list in lists:
            by_list_type[doc_list.list_type] = by_list_type.get(doc_list.list_type, 0) + 1

        return {
            "has_lists": len(lists) > 0,
            "list_count": len(lists),
            "list_item_count": len(list_items),
            "by_marker_type": by_marker_type,
            "by_list_type": by_list_type,
            "by_level": by_level,
            "max_level": max([item.level for item in list_items], default=0),
        }

    def _infer_level_from_marker(
        self,
        marker: str,
        marker_type: str,
    ) -> int:
        marker_clean = marker.strip("()").strip(".").strip(")")

        if marker_type == "numbered":
            if "." in marker_clean:
                return max(1, marker_clean.count(".") + 1)

            return 1

        if marker_type == "letter":
            return 2

        if marker_type == "roman":
            return 1

        if marker_type in {"bullet", "dash"}:
            return 1

        return 1

    def _infer_level_from_indent(
        self,
        bbox: Optional[List[float]],
    ) -> int:
        if not bbox:
            return 1

        x0 = float(bbox[0])

        if x0 < 50:
            return 1

        if x0 < 90:
            return 2

        if x0 < 130:
            return 3

        if x0 < 170:
            return 4

        return 5

    def _score_list_item(
        self,
        marker: str,
        marker_type: str,
        item_text: str,
    ) -> float:
        score = 0.60

        if marker_type in {"numbered", "bullet"}:
            score += 0.15

        if marker_type in {"letter", "roman", "dash"}:
            score += 0.10

        if 3 <= len(item_text) <= 300:
            score += 0.10

        if len(item_text.split()) >= 2:
            score += 0.05

        if item_text.endswith((".", ";", ":")):
            score += 0.03

        return round(max(0.0, min(score, 0.95)), 4)

    def _is_valid_item_text(
        self,
        text: str,
    ) -> bool:
        if text is None:
            return False

        clean = text.strip()

        if len(clean) < self.config.min_item_text_chars:
            return False

        if len(clean) > self.config.max_item_text_chars:
            return False

        return True

    def _vertical_gap(
        self,
        bbox_a: Optional[List[float]],
        bbox_b: Optional[List[float]],
    ) -> Optional[float]:
        if not bbox_a or not bbox_b:
            return None

        return max(float(bbox_b[1]) - float(bbox_a[3]), 0.0)

    def _indent_difference(
        self,
        bbox_a: Optional[List[float]],
        bbox_b: Optional[List[float]],
    ) -> Optional[float]:
        if not bbox_a or not bbox_b:
            return None

        return abs(float(bbox_a[0]) - float(bbox_b[0]))

    def _sort_items(
        self,
        items: List[DocumentListItem],
    ) -> List[DocumentListItem]:
        return sorted(
            items,
            key=lambda item: (
                item.page_number,
                item.bbox[1] if item.bbox else 999999,
                item.bbox[0] if item.bbox else 999999,
            ),
        )

    def _clean_text(
        self,
        text: str,
    ) -> str:
        if not text:
            return ""

        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _text_preview(
        self,
        text: str,
    ) -> str:
        if not self.config.include_text_preview:
            return ""

        return text[: self.config.text_preview_chars]

    def _safe_int(
        self,
        value: Any,
        default: int = 0,
    ) -> int:
        try:
            if value is None:
                return default

            return int(value)
        except Exception:
            return default


def detect_lists(
    page_raws: List[PageRaw],
    paragraph_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    detector = ListDetector()
    return detector.process(
        page_raws=page_raws,
        paragraph_result=paragraph_result,
    )
