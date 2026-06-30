"""
figure_detector.py

Production V1 - Colab Ready

Purpose
-------
Detect figure candidates from images, drawings, and visual regions.

Input
-----
PageRaw after:
- PageExtractionPipeline
- ObjectNormalizer
- ObjectMerger
- ReadingOrderBuilder
- RegionDetector
- HeaderFooterDetector
- TableBoundaryDetector

Output
------
PageRaw with metadata:
- figure_candidates
- figure_summary

Important
---------
This module detects figure-like visual regions only.

It does not:
- classify chart type
- OCR text inside figure
- understand figure semantics
- build final document structure
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from document_ai.schemas.page_raw_schema import PageRaw, make_id


@dataclass
class FigureDetectorConfig:
    """
    Configuration for FigureDetector.
    """

    use_image_objects: bool = True
    use_drawing_objects: bool = True
    use_region_detector: bool = True

    min_figure_width: float = 40.0
    min_figure_height: float = 40.0
    min_figure_area_ratio: float = 0.005

    drawing_cluster_gap: float = 12.0
    min_drawings_per_figure: int = 3

    exclude_table_candidates: bool = True
    table_overlap_threshold: float = 0.40

    attach_nearby_caption: bool = True
    caption_search_margin: float = 40.0
    caption_max_distance: float = 60.0

    include_text_preview: bool = True
    text_preview_chars: int = 300


@dataclass
class FigureCandidate:
    """
    One figure candidate.
    """

    figure_id: str
    page_number: int
    bbox: Optional[List[float]]
    detection_method: str

    confidence: float = 0.5
    source: str = "unknown"
    source_object_ids: Optional[List[str]] = None
    caption_text: str = ""
    caption_bbox: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        if data["source_object_ids"] is None:
            data["source_object_ids"] = []

        if data["metadata"] is None:
            data["metadata"] = {}

        return data


class FigureDetector:
    """
    Detect figure candidates from one PageRaw.
    """

    def __init__(
        self,
        config: Optional[FigureDetectorConfig] = None,
    ):
        self.config = config or FigureDetectorConfig()

    def process(
        self,
        page_raw: PageRaw,
    ) -> PageRaw:
        """
        Detect figure candidates and attach result to page_raw.metadata.
        """

        warnings: List[str] = []
        candidates: List[FigureCandidate] = []

        try:
            if self.config.use_image_objects:
                candidates.extend(self._detect_from_images(page_raw))
        except Exception as exc:
            warnings.append(f"Image-based figure detection failed: {exc}")

        try:
            if self.config.use_drawing_objects:
                candidates.extend(self._detect_from_drawings(page_raw))
        except Exception as exc:
            warnings.append(f"Drawing-based figure detection failed: {exc}")

        try:
            if self.config.use_region_detector:
                candidates.extend(self._detect_from_regions(page_raw))
        except Exception as exc:
            warnings.append(f"Region-based figure detection failed: {exc}")

        candidates = self._filter_candidates(
            page_raw=page_raw,
            candidates=candidates,
        )

        if self.config.exclude_table_candidates:
            candidates = self._exclude_table_overlaps(
                page_raw=page_raw,
                candidates=candidates,
            )

        candidates = self._merge_overlapping_figures(candidates)

        if self.config.attach_nearby_caption:
            candidates = [
                self._attach_caption(
                    page_raw=page_raw,
                    candidate=candidate,
                )
                for candidate in candidates
            ]

        candidates = self._sort_candidates(candidates)

        page_raw.metadata.setdefault("figure_detector", {})
        page_raw.metadata["figure_detector"] = {
            "processor": "FigureDetector",
            "figure_candidate_count": len(candidates),
            "figure_candidates": [
                candidate.to_dict() for candidate in candidates
            ],
            "summary": self._build_summary(candidates),
            "config": {
                "use_image_objects": self.config.use_image_objects,
                "use_drawing_objects": self.config.use_drawing_objects,
                "use_region_detector": self.config.use_region_detector,
                "exclude_table_candidates": self.config.exclude_table_candidates,
                "attach_nearby_caption": self.config.attach_nearby_caption,
            },
        }

        if warnings:
            page_raw.warnings.extend(warnings)

        return page_raw

    def _detect_from_images(
        self,
        page_raw: PageRaw,
    ) -> List[FigureCandidate]:
        """
        Detect figures from extracted image objects.
        """

        candidates: List[FigureCandidate] = []

        for index, image in enumerate(page_raw.images):
            bbox = image.bbox

            if not bbox:
                continue

            candidates.append(
                FigureCandidate(
                    figure_id=make_id("figure"),
                    page_number=page_raw.page_number,
                    bbox=bbox,
                    detection_method="image_object",
                    confidence=0.82,
                    source="page_raw.images",
                    source_object_ids=[image.image_id],
                    metadata={
                        "source_index": index,
                        "image_width": image.width,
                        "image_height": image.height,
                        "image_ext": image.ext,
                        "colorspace": image.colorspace,
                    },
                )
            )

        return candidates

    def _detect_from_drawings(
        self,
        page_raw: PageRaw,
    ) -> List[FigureCandidate]:
        """
        Detect figure-like clusters from drawings.

        This is useful for charts, diagrams, boxes, lines, and vector graphics.
        """

        drawings = [
            drawing for drawing in page_raw.drawings
            if drawing.bbox
        ]

        if len(drawings) < self.config.min_drawings_per_figure:
            return []

        clusters = self._cluster_drawings(drawings)

        candidates: List[FigureCandidate] = []

        for cluster_index, cluster in enumerate(clusters):
            if len(cluster) < self.config.min_drawings_per_figure:
                continue

            bbox = self._merge_bboxes([drawing.bbox for drawing in cluster])

            if not bbox:
                continue

            candidates.append(
                FigureCandidate(
                    figure_id=make_id("figure"),
                    page_number=page_raw.page_number,
                    bbox=bbox,
                    detection_method="drawing_cluster",
                    confidence=0.62,
                    source="page_raw.drawings",
                    source_object_ids=[drawing.drawing_id for drawing in cluster],
                    metadata={
                        "cluster_index": cluster_index,
                        "drawing_count": len(cluster),
                        "method": "visual_drawing_cluster",
                    },
                )
            )

        return candidates

    def _detect_from_regions(
        self,
        page_raw: PageRaw,
    ) -> List[FigureCandidate]:
        """
        Detect figures from RegionDetector image/drawing regions.
        """

        candidates: List[FigureCandidate] = []

        region_meta = page_raw.metadata.get("region_detector", {})
        regions = region_meta.get("detected_regions", [])

        for index, region in enumerate(regions):
            region_type = region.get("region_type", "")

            if region_type not in {
                "image_region",
                "drawing_region",
                "drawing_line_group_region",
            }:
                continue

            bbox = region.get("bbox")

            if not bbox:
                continue

            candidates.append(
                FigureCandidate(
                    figure_id=make_id("figure"),
                    page_number=page_raw.page_number,
                    bbox=bbox,
                    detection_method="region_detector",
                    confidence=float(region.get("confidence", 0.60)),
                    source="region_detector.detected_regions",
                    source_object_ids=region.get("source_object_ids", []),
                    metadata={
                        "source_index": index,
                        "region_id": region.get("region_id"),
                        "region_type": region_type,
                        "region_metadata": region.get("metadata", {}),
                    },
                )
            )

        return candidates

    def _cluster_drawings(
        self,
        drawings: List[Any],
    ) -> List[List[Any]]:
        """
        Cluster drawings by bbox proximity.
        """

        sorted_drawings = sorted(
            drawings,
            key=lambda drawing: (
                drawing.bbox[1],
                drawing.bbox[0],
            ),
        )

        clusters: List[List[Any]] = []

        for drawing in sorted_drawings:
            placed = False

            for cluster in clusters:
                cluster_bbox = self._merge_bboxes(
                    [item.bbox for item in cluster]
                )

                if self._bboxes_near(
                    drawing.bbox,
                    cluster_bbox,
                    self.config.drawing_cluster_gap,
                ):
                    cluster.append(drawing)
                    placed = True
                    break

            if not placed:
                clusters.append([drawing])

        return clusters

    def _attach_caption(
        self,
        page_raw: PageRaw,
        candidate: FigureCandidate,
    ) -> FigureCandidate:
        """
        Attach nearby caption text to figure candidate.
        """

        if not candidate.bbox:
            return candidate

        caption_objects = self._collect_caption_like_text_objects(page_raw)

        if not caption_objects:
            return candidate

        best_caption = None
        best_distance = None

        for obj in caption_objects:
            bbox = obj.get("bbox")

            if not bbox:
                continue

            distance = self._caption_distance(
                figure_bbox=candidate.bbox,
                caption_bbox=bbox,
            )

            if distance is None:
                continue

            if distance > self.config.caption_max_distance:
                continue

            if best_distance is None or distance < best_distance:
                best_caption = obj
                best_distance = distance

        if best_caption:
            candidate.caption_text = self._preview_text(
                best_caption.get("text") or ""
            )
            candidate.caption_bbox = best_caption.get("bbox")

            if candidate.metadata is None:
                candidate.metadata = {}

            candidate.metadata["caption"] = {
                "source_object_id": best_caption.get("object_id"),
                "source": best_caption.get("source"),
                "distance": best_distance,
            }

            candidate.confidence = min(candidate.confidence + 0.08, 0.95)

        return candidate

    def _collect_caption_like_text_objects(
        self,
        page_raw: PageRaw,
    ) -> List[Dict[str, Any]]:
        """
        Collect caption-like text objects from reading order or text lines.
        """

        result: List[Dict[str, Any]] = []

        reading_meta = page_raw.metadata.get("reading_order_builder", {})
        reading_items = reading_meta.get("reading_order_items", [])

        for index, item in enumerate(reading_items):
            text = item.get("text") or ""

            if not self._looks_like_caption(text):
                continue

            result.append(
                {
                    "object_id": item.get("item_id") or f"reading_item_{index}",
                    "text": text,
                    "bbox": item.get("bbox"),
                    "source": "reading_order_builder.reading_order_items",
                    "source_index": index,
                }
            )

        if result:
            return result

        for index, line in enumerate(page_raw.text_lines):
            text = line.normalized_text or line.text or ""

            if not self._looks_like_caption(text):
                continue

            result.append(
                {
                    "object_id": line.line_id,
                    "text": text,
                    "bbox": line.bbox,
                    "source": "page_raw.text_lines",
                    "source_index": index,
                }
            )

        return result

    def _looks_like_caption(
        self,
        text: str,
    ) -> bool:
        """
        Detect Vietnamese/English figure caption patterns.
        """

        if not text:
            return False

        lower = text.strip().lower()

        if not lower:
            return False

        patterns = [
            "hình ",
            "hinh ",
            "biểu đồ",
            "bieu do",
            "sơ đồ",
            "so do",
            "ảnh ",
            "anh ",
            "figure ",
            "fig.",
            "chart ",
            "diagram ",
        ]

        return any(lower.startswith(pattern) for pattern in patterns)

    def _caption_distance(
        self,
        figure_bbox: List[float],
        caption_bbox: List[float],
    ) -> Optional[float]:
        """
        Compute vertical distance between figure and caption.

        Caption may be below or above the figure.
        """

        figure_x0, figure_y0, figure_x1, figure_y1 = figure_bbox
        caption_x0, caption_y0, caption_x1, caption_y1 = caption_bbox

        figure_center_x = (figure_x0 + figure_x1) / 2.0
        caption_center_x = (caption_x0 + caption_x1) / 2.0

        horizontal_margin = self.config.caption_search_margin
        figure_width = max(figure_x1 - figure_x0, 1.0)

        if abs(figure_center_x - caption_center_x) > figure_width / 2.0 + horizontal_margin:
            return None

        if caption_y0 >= figure_y1:
            return caption_y0 - figure_y1

        if figure_y0 >= caption_y1:
            return figure_y0 - caption_y1

        return 0.0

    def _filter_candidates(
        self,
        page_raw: PageRaw,
        candidates: List[FigureCandidate],
    ) -> List[FigureCandidate]:
        """
        Remove invalid or too-small candidates.
        """

        valid: List[FigureCandidate] = []

        page_area = float(page_raw.width or 0) * float(page_raw.height or 0)

        for candidate in candidates:
            bbox = candidate.bbox

            if not bbox:
                continue

            width = self._bbox_width(bbox)
            height = self._bbox_height(bbox)
            area = width * height

            if width < self.config.min_figure_width:
                continue

            if height < self.config.min_figure_height:
                continue

            if page_area > 0:
                area_ratio = area / page_area

                if area_ratio < self.config.min_figure_area_ratio:
                    continue

            valid.append(candidate)

        return valid

    def _exclude_table_overlaps(
        self,
        page_raw: PageRaw,
        candidates: List[FigureCandidate],
    ) -> List[FigureCandidate]:
        """
        Exclude figure candidates that overlap with detected tables.
        """

        table_bboxes = self._collect_table_bboxes(page_raw)

        if not table_bboxes:
            return candidates

        result: List[FigureCandidate] = []

        for candidate in candidates:
            is_table_overlap = False

            for table_bbox in table_bboxes:
                overlap = self._bbox_overlap_ratio(candidate.bbox, table_bbox)

                if overlap >= self.config.table_overlap_threshold:
                    is_table_overlap = True
                    break

            if not is_table_overlap:
                result.append(candidate)

        return result

    def _collect_table_bboxes(
        self,
        page_raw: PageRaw,
    ) -> List[List[float]]:
        """
        Collect table candidate bboxes from TableBoundaryDetector.
        """

        table_meta = page_raw.metadata.get("table_boundary_detector", {})
        table_candidates = table_meta.get("table_candidates", [])

        bboxes: List[List[float]] = []

        for table in table_candidates:
            bbox = table.get("bbox")

            if bbox and len(bbox) == 4:
                bboxes.append(bbox)

        return bboxes

    def _merge_overlapping_figures(
        self,
        candidates: List[FigureCandidate],
    ) -> List[FigureCandidate]:
        """
        Merge duplicated figure candidates.
        """

        if not candidates:
            return []

        sorted_candidates = self._sort_candidates(candidates)
        groups: List[List[FigureCandidate]] = []

        for candidate in sorted_candidates:
            placed = False

            for group in groups:
                group_bbox = self._merge_bboxes(
                    [item.bbox for item in group]
                )

                if self._bbox_overlap_ratio(candidate.bbox, group_bbox) >= 0.50:
                    group.append(candidate)
                    placed = True
                    break

            if not placed:
                groups.append([candidate])

        merged: List[FigureCandidate] = []

        for group_index, group in enumerate(groups):
            if len(group) == 1:
                merged.append(group[0])
                continue

            bbox = self._merge_bboxes([item.bbox for item in group])
            source_ids: List[str] = []
            methods: List[str] = []
            confidence = 0.0

            for item in group:
                source_ids.extend(item.source_object_ids or [])
                methods.append(item.detection_method)
                confidence = max(confidence, item.confidence)

            if len(set(methods)) > 1:
                confidence = min(confidence + 0.06, 0.95)

            merged.append(
                FigureCandidate(
                    figure_id=make_id("figure"),
                    page_number=group[0].page_number,
                    bbox=bbox,
                    detection_method="merged",
                    confidence=round(confidence, 4),
                    source="figure_detector.merge_overlapping_figures",
                    source_object_ids=list(dict.fromkeys(source_ids)),
                    metadata={
                        "group_index": group_index,
                        "merged_candidate_count": len(group),
                        "detection_methods": sorted(set(methods)),
                    },
                )
            )

        return merged

    def _build_summary(
        self,
        candidates: List[FigureCandidate],
    ) -> Dict[str, Any]:
        by_method: Dict[str, int] = {}

        for candidate in candidates:
            by_method[candidate.detection_method] = (
                by_method.get(candidate.detection_method, 0) + 1
            )

        return {
            "has_figure_candidates": len(candidates) > 0,
            "figure_candidate_count": len(candidates),
            "by_detection_method": by_method,
            "with_caption_count": len(
                [candidate for candidate in candidates if candidate.caption_text]
            ),
        }

    def _sort_candidates(
        self,
        candidates: List[FigureCandidate],
    ) -> List[FigureCandidate]:
        return sorted(
            candidates,
            key=lambda item: (
                item.bbox[1] if item.bbox else 0,
                item.bbox[0] if item.bbox else 0,
            ),
        )

    def _bboxes_near(
        self,
        bbox_a: Optional[List[float]],
        bbox_b: Optional[List[float]],
        margin: float,
    ) -> bool:
        if not bbox_a or not bbox_b:
            return False

        a = self._pad_bbox(bbox_a, margin)
        b = self._pad_bbox(bbox_b, margin)

        return not (
            a[2] < b[0]
            or a[0] > b[2]
            or a[3] < b[1]
            or a[1] > b[3]
        )

    def _bbox_overlap_ratio(
        self,
        bbox_a: Optional[List[float]],
        bbox_b: Optional[List[float]],
    ) -> float:
        if not bbox_a or not bbox_b:
            return 0.0

        x0 = max(bbox_a[0], bbox_b[0])
        y0 = max(bbox_a[1], bbox_b[1])
        x1 = min(bbox_a[2], bbox_b[2])
        y1 = min(bbox_a[3], bbox_b[3])

        inter_width = max(x1 - x0, 0.0)
        inter_height = max(y1 - y0, 0.0)
        intersection = inter_width * inter_height

        if intersection <= 0:
            return 0.0

        area_a = self._bbox_area(bbox_a)
        area_b = self._bbox_area(bbox_b)

        smaller_area = min(area_a, area_b)

        if smaller_area <= 0:
            return 0.0

        return intersection / smaller_area

    def _merge_bboxes(
        self,
        bboxes: List[Optional[List[float]]],
    ) -> Optional[List[float]]:
        valid = [
            bbox for bbox in bboxes
            if bbox and len(bbox) == 4
        ]

        if not valid:
            return None

        return [
            min(float(bbox[0]) for bbox in valid),
            min(float(bbox[1]) for bbox in valid),
            max(float(bbox[2]) for bbox in valid),
            max(float(bbox[3]) for bbox in valid),
        ]

    def _pad_bbox(
        self,
        bbox: Optional[List[float]],
        padding: float,
    ) -> Optional[List[float]]:
        if not bbox:
            return None

        return [
            float(bbox[0]) - padding,
            float(bbox[1]) - padding,
            float(bbox[2]) + padding,
            float(bbox[3]) + padding,
        ]

    def _bbox_width(
        self,
        bbox: Optional[List[float]],
    ) -> float:
        if not bbox:
            return 0.0

        return max(float(bbox[2]) - float(bbox[0]), 0.0)

    def _bbox_height(
        self,
        bbox: Optional[List[float]],
    ) -> float:
        if not bbox:
            return 0.0

        return max(float(bbox[3]) - float(bbox[1]), 0.0)

    def _bbox_area(
        self,
        bbox: Optional[List[float]],
    ) -> float:
        if not bbox:
            return 0.0

        return self._bbox_width(bbox) * self._bbox_height(bbox)

    def _preview_text(
        self,
        text: str,
    ) -> str:
        if not self.config.include_text_preview:
            return ""

        if not text:
            return ""

        return text[: self.config.text_preview_chars]


def detect_figures(
    page_raw: PageRaw,
) -> PageRaw:
    """
    Colab helper function.
    """

    detector = FigureDetector()
    return detector.process(page_raw)
