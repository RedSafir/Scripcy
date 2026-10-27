"""Module for deterministic classification of scientific visual assets (SIMPLE vs COMPLEX).

Classifies images, diagrams, plots, tables, and formulas using:
1. PyMuPDF vector path primitives (curves/beziers, dashes, scatter points, closed boxes).
2. Collinear monotonic numeric sequences (axis tick labels).
3. Pure PIL / lightweight raster heuristics (Sobel/edge density, color palette, aspect ratio).
"""

from __future__ import annotations

from enum import Enum
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFilter

NUMERIC_REGEX = re.compile(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?%?$")
MATH_SYMBOLS = set("∑∫∏√αβγδεθλμπσφψω≤≥±≠≈∈∉⊂⊆∇∂×÷")

# High-confidence semantic keyword indicators
COMPLEX_SEMANTIC_KEYWORDS = re.compile(
    r"\b(plot|plots|curve|curves|scatter|heatmap|roc|auc|distribution|regression|"
    r"confusion\s+matrix|benchmark|graph|graphs|chart|charts|histogram|correlation|"
    r"trend|accuracy|akurasi|persentase|versus|\bvs\b|table|tabel|matrix|formula|equation)\b",
    re.IGNORECASE,
)

SIMPLE_SEMANTIC_KEYWORDS = re.compile(
    r"\b(flowchart|architecture|overview|pipeline|workflow|schematic|taxonomy|"
    r"diagram\s+alir|alur\s+kerja|bagan|framework|tree|hierarchy|struktur)\b",
    re.IGNORECASE,
)


class VisualType(str, Enum):
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"


class VisualClassificationResult:
    """Detailed outcome of visual classification."""

    def __init__(
        self,
        visual_type: VisualType,
        score_complex: int,
        score_simple: int,
        reasons: list[str],
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.visual_type = visual_type
        self.score_complex = score_complex
        self.score_simple = score_simple
        self.reasons = reasons
        self.metrics = metrics or {}

    @property
    def is_complex(self) -> bool:
        return self.visual_type == VisualType.COMPLEX

    @property
    def is_simple(self) -> bool:
        return self.visual_type == VisualType.SIMPLE

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual_type": self.visual_type.value,
            "is_complex": self.is_complex,
            "score_complex": self.score_complex,
            "score_simple": self.score_simple,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


class VisualClassifier:
    """High-speed deterministic classifier for academic visual assets."""

    @classmethod
    def check_monotonic_labels(cls, words: list[tuple[float, float, float, float, str, Any, Any, Any]]) -> bool:
        """Detect if at least 3 numbers are collinear along an axis and strictly monotonic."""
        num_boxes: list[tuple[float, float, float]] = []
        for w in words:
            text = str(w[4]).strip().replace("%", "").replace(",", "")
            try:
                val = float(text)
                cx = (w[0] + w[2]) / 2.0
                cy = (w[1] + w[3]) / 2.0
                num_boxes.append((cx, cy, val))
            except (ValueError, TypeError):
                continue

        if len(num_boxes) < 3:
            return False

        # 1. Check Vertical Alignment (Y-axis tick labels: similar X, varying Y)
        for i in range(len(num_boxes)):
            aligned = [num_boxes[i]]
            for j in range(len(num_boxes)):
                if i != j and abs(num_boxes[i][0] - num_boxes[j][0]) <= 14.0:
                    aligned.append(num_boxes[j])
            if len(aligned) >= 3:
                aligned.sort(key=lambda item: item[1])
                vals = [item[2] for item in aligned]
                is_inc = all(x < y for x, y in zip(vals, vals[1:]))
                is_dec = all(x > y for x, y in zip(vals, vals[1:]))
                if is_inc or is_dec:
                    return True

        # 2. Check Horizontal Alignment (X-axis tick labels: similar Y, varying X)
        for i in range(len(num_boxes)):
            aligned = [num_boxes[i]]
            for j in range(len(num_boxes)):
                if i != j and abs(num_boxes[i][1] - num_boxes[j][1]) <= 10.0:
                    aligned.append(num_boxes[j])
            if len(aligned) >= 3:
                aligned.sort(key=lambda item: item[0])
                vals = [item[2] for item in aligned]
                is_inc = all(x < y for x, y in zip(vals, vals[1:]))
                is_dec = all(x > y for x, y in zip(vals, vals[1:]))
                if is_inc or is_dec:
                    return True

        return False

    @classmethod
    def classify_vector_bbox(
        cls,
        page: Any,
        bbox: Any,
        caption: str = "",
    ) -> VisualClassificationResult:
        """Inspect PyMuPDF vector streams and text placements within a bounding box (< 2ms)."""
        reasons: list[str] = []
        score_complex = 0
        score_simple = 0

        # Semantic keywords in caption
        if caption:
            if COMPLEX_SEMANTIC_KEYWORDS.search(caption):
                score_complex += 4
                reasons.append("caption_indicates_data_chart_or_plot")
            if SIMPLE_SEMANTIC_KEYWORDS.search(caption):
                score_simple += 3
                reasons.append("caption_indicates_flowchart_or_architecture")

        n_lines = 0
        n_curves = 0
        n_rects = 0
        n_dashes = 0
        small_markers = 0

        try:
            drawings = page.get_drawings()
        except Exception:
            drawings = []

        clip_drawings = []
        for d in drawings:
            d_rect = d.get("rect")
            if d_rect and bbox.intersects(d_rect):
                clip_drawings.append(d)

        for d in clip_drawings:
            if d.get("dashes"):
                n_dashes += 1
            for item in d.get("items", []):
                cmd = item[0]
                if cmd == "l":
                    n_lines += 1
                elif cmd == "c":
                    n_curves += 1
                elif cmd == "re":
                    n_rects += 1
                    r = item[1]
                    # Small rect marker (scatter point or legend marker)
                    if getattr(r, "width", 99) <= 8 and getattr(r, "height", 99) <= 8:
                        small_markers += 1

        total_primitives = n_lines + n_curves + n_rects

        # Text & numbers inspection
        try:
            words = page.get_text("words", clip=bbox)
        except Exception:
            words = []

        n_words = len(words)
        num_count = sum(1 for w in words if NUMERIC_REGEX.match(str(w[4])))
        math_count = sum(1 for w in words if any(c in MATH_SYMBOLS for c in str(w[4])))
        numeric_ratio = num_count / max(1, n_words)

        # Monotonic check
        has_monotonic_seq = cls.check_monotonic_labels(words)
        if has_monotonic_seq:
            score_complex += 6
            reasons.append("monotonic_axis_tick_labels")

        # Mathematical equation inspection
        has_fraction_or_script = False
        if n_words > 0:
            font_sizes = [abs(w[3] - w[1]) for w in words if abs(w[3] - w[1]) > 0]
            if font_sizes and (max(font_sizes) - min(font_sizes)) / max(font_sizes) > 0.35 and math_count > 0:
                has_fraction_or_script = True

        if has_fraction_or_script or math_count >= 2:
            score_complex += 5
            reasons.append("mathematical_symbols_and_scripts")

        if numeric_ratio > 0.35 and n_words >= 5:
            score_complex += 4
            reasons.append(f"high_numeric_density({numeric_ratio:.2f})")

        total_curve_line = n_lines + n_curves
        curve_ratio = n_curves / max(1, total_curve_line)
        if n_curves >= 8 and curve_ratio > 0.15:
            score_complex += 5
            reasons.append(f"high_bezier_curve_ratio({curve_ratio:.2f})")

        if small_markers >= 12:
            score_complex += 4
            reasons.append("scatter_markers_detected")

        if n_dashes >= 3:
            score_complex += 3
            reasons.append("dashed_coordinate_grid_lines")

        # Flowchart indicators
        if n_rects >= 3 and n_curves <= 3 and numeric_ratio < 0.20:
            score_simple += 5
            reasons.append("rectilinear_flowchart_boxes")

        if total_primitives <= 6 and n_words <= 4 and not has_monotonic_seq:
            score_simple += 3
            reasons.append("sparse_conceptual_icon_or_diagram")

        v_type = VisualType.COMPLEX if (score_complex >= score_simple and score_complex >= 4) else VisualType.SIMPLE
        return VisualClassificationResult(
            visual_type=v_type,
            score_complex=score_complex,
            score_simple=score_simple,
            reasons=reasons,
            metrics={
                "lines": n_lines,
                "curves": n_curves,
                "rects": n_rects,
                "small_markers": small_markers,
                "dashes": n_dashes,
                "words": n_words,
                "numeric_ratio": round(numeric_ratio, 2),
                "math_symbols": math_count,
                "has_monotonic_seq": has_monotonic_seq,
            },
        )

    @classmethod
    def classify_raster(
        cls,
        image_path_or_bytes: Path | bytes,
        caption: str = "",
    ) -> VisualClassificationResult:
        """Inspect raster image using PIL edge/contrast/color heuristics (< 15ms)."""
        reasons: list[str] = []
        score_complex = 0
        score_simple = 0

        # Semantic keywords in caption
        if caption:
            if COMPLEX_SEMANTIC_KEYWORDS.search(caption):
                score_complex += 4
                reasons.append("caption_indicates_data_chart_or_plot")
            if SIMPLE_SEMANTIC_KEYWORDS.search(caption):
                score_simple += 3
                reasons.append("caption_indicates_flowchart_or_architecture")

        try:
            if isinstance(image_path_or_bytes, (str, Path)):
                img = Image.open(image_path_or_bytes)
            else:
                import io
                img = Image.open(io.BytesIO(image_path_or_bytes))
        except Exception as e:
            return VisualClassificationResult(
                visual_type=VisualType.SIMPLE,
                score_complex=0,
                score_simple=1,
                reasons=[f"raster_open_failed: {e}"],
            )

        with img:
            w, h = img.size
            ratio = w / max(1, h)

            # Convert to grayscale for edge analysis
            gray = img.convert("L")
            # Downsample if image is huge to guarantee <15ms
            if max(w, h) > 800:
                gray.thumbnail((800, 800))

            # 1. Edge analysis via FIND_EDGES
            edges = gray.filter(ImageFilter.FIND_EDGES)
            if hasattr(edges, "get_flattened_data"):
                edge_data = list(edges.get_flattened_data())
            else:
                edge_data = list(edges.getdata())
            total_px = len(edge_data)
            edge_px = sum(1 for p in edge_data if p > 60)
            edge_density = edge_px / max(1, total_px)

            # 2. Color count (Entropy & Palette)
            # Flowcharts have limited palette; heatmaps/charts have diverse shades
            sample = img.convert("RGB").resize((100, 100))
            if hasattr(sample, "get_flattened_data"):
                raw_sample = list(sample.get_flattened_data())
                unique_colors = len(set(raw_sample))
            else:
                unique_colors = len(set(sample.getdata()))

            # 3. Variance & standard deviation
            stat_mean = sum(edge_data) / max(1, total_px)
            stat_var = sum((p - stat_mean) ** 2 for p in edge_data) / max(1, total_px)
            stat_std = math.sqrt(stat_var)

        # Decision metrics
        if edge_density > 0.16 or (edge_density > 0.12 and stat_std > 30):
            score_complex += 4
            reasons.append(f"high_edge_density({edge_density:.2f}, std={stat_std:.1f})")

        if unique_colors > 450:
            score_complex += 3
            reasons.append(f"rich_color_palette({unique_colors}_colors)")

        if edge_density < 0.08 and unique_colors < 150:
            score_simple += 4
            reasons.append(f"low_edge_density_solid_palette({edge_density:.2f}, {unique_colors}_colors)")

        # Aspect ratio checks (very wide or tall images are often tables or flow sequences)
        if ratio > 3.0 or ratio < 0.3:
            score_complex += 2
            reasons.append(f"extreme_aspect_ratio({ratio:.2f})")

        v_type = VisualType.COMPLEX if (score_complex >= score_simple and score_complex >= 4) else VisualType.SIMPLE
        return VisualClassificationResult(
            visual_type=v_type,
            score_complex=score_complex,
            score_simple=score_simple,
            reasons=reasons,
            metrics={
                "width": w,
                "height": h,
                "aspect_ratio": round(ratio, 2),
                "edge_density": round(edge_density, 3),
                "unique_colors": unique_colors,
                "edge_std": round(stat_std, 1),
            },
        )
