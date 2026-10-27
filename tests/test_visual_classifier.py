"""Unit tests for VisualClassifier (PyMuPDF vector inspection & PIL raster heuristics)."""

from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from scripcy.core.classifier import VisualClassificationResult, VisualClassifier, VisualType


class TestVisualClassifier:
    def test_check_monotonic_labels_vertical(self):
        # Y-axis ticks with X around 50, Y increasing, values increasing
        words = [
            (49.0, 100.0, 52.0, 110.0, "0", 0, 0, 0),
            (50.0, 150.0, 53.0, 160.0, "20", 0, 1, 0),
            (51.0, 200.0, 54.0, 210.0, "40", 0, 2, 0),
            (49.5, 250.0, 52.5, 260.0, "60", 0, 3, 0),
        ]
        assert VisualClassifier.check_monotonic_labels(words) is True

    def test_check_monotonic_labels_horizontal(self):
        # X-axis ticks with Y around 300, X increasing, values increasing
        words = [
            (100.0, 299.0, 115.0, 305.0, "10", 0, 0, 0),
            (200.0, 301.0, 215.0, 307.0, "20", 0, 0, 1),
            (300.0, 300.0, 315.0, 306.0, "30", 0, 0, 2),
        ]
        assert VisualClassifier.check_monotonic_labels(words) is True

    def test_check_monotonic_labels_non_monotonic_or_text(self):
        words = [
            (50.0, 100.0, 60.0, 110.0, "Input", 0, 0, 0),
            (50.0, 150.0, 60.0, 160.0, "Encoder", 0, 1, 0),
            (50.0, 200.0, 60.0, 210.0, "Decoder", 0, 2, 0),
        ]
        assert VisualClassifier.check_monotonic_labels(words) is False

    def test_classify_raster_simple_flowchart(self, tmp_path: Path):
        # Create a simple flowchart image: clean white canvas with 2 solid rectangles
        img = Image.new("RGB", (300, 300), color="white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 120, 100], fill="lightblue", outline="black")
        draw.rectangle([180, 50, 250, 100], fill="lightgreen", outline="black")
        draw.line([120, 75, 180, 75], fill="black", width=2)

        img_path = tmp_path / "flowchart.png"
        img.save(img_path)

        res = VisualClassifier.classify_raster(img_path, caption="Figure 1: Proposed architecture pipeline")
        assert res.visual_type == VisualType.SIMPLE
        assert res.is_simple is True

    def test_classify_raster_complex_data_plot(self, tmp_path: Path):
        # Create a complex data plot: axes, grid, and multi-line sinusoidal curves
        img = Image.new("RGB", (500, 400), color="white")
        draw = ImageDraw.Draw(img)
        # Sumbu X & Y
        draw.line([50, 350, 450, 350], fill="black", width=2)
        draw.line([50, 50, 50, 350], fill="black", width=2)

        # Banyak garis kurva & grid
        import math
        for x in range(50, 450, 2):
            y1 = int(200 + 80 * math.sin(x * 0.05))
            y2 = int(200 + 60 * math.cos(x * 0.03))
            draw.point((x, y1), fill="red")
            draw.point((x, y2), fill="blue")

        # Grid lines
        for y_g in range(80, 350, 30):
            draw.line([50, y_g, 450, y_g], fill="gray", width=1)

        img_path = tmp_path / "data_plot.png"
        img.save(img_path)

        res = VisualClassifier.classify_raster(img_path, caption="Figure 4: Training loss curve and convergence plot")
        assert res.visual_type == VisualType.COMPLEX
        assert res.is_complex is True

    def test_classify_vector_mock_flowchart(self):
        class MockRect:
            def __init__(self, x0, y0, x1, y1):
                self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
                self.width, self.height = x1 - x0, y1 - y0

            def intersects(self, other):
                return True

        class MockPage:
            def get_drawings(self):
                return [
                    {"items": [("re", MockRect(50, 50, 150, 100)), ("re", MockRect(200, 50, 300, 100))], "rect": MockRect(0, 0, 500, 500)},
                    {"items": [("l", (150, 75), (200, 75))], "rect": MockRect(0, 0, 500, 500)},
                ]

            def get_text(self, kind, clip=None):
                return [
                    (60, 60, 100, 75, "Step", 0, 0, 0),
                    (105, 60, 120, 75, "One", 0, 0, 1),
                    (210, 60, 250, 75, "Output", 0, 0, 2),
                ]

        res = VisualClassifier.classify_vector_bbox(MockPage(), MockRect(0, 0, 500, 500), caption="Overview framework diagram")
        assert res.visual_type == VisualType.SIMPLE

    def test_classify_vector_mock_complex_curve(self):
        class MockRect:
            def __init__(self, x0, y0, x1, y1):
                self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
                self.width, self.height = x1 - x0, y1 - y0

            def intersects(self, other):
                return True

        class MockPage:
            def get_drawings(self):
                # 15 bezier curves (high bezier ratio) and dashed grid lines
                curve_items = [("c", (0, 0), (1, 1), (2, 2), (3, 3)) for _ in range(15)]
                return [
                    {"items": curve_items, "dashes": "3 3", "rect": MockRect(0, 0, 500, 500)},
                ]

            def get_text(self, kind, clip=None):
                # Monotonic labels
                return [
                    (49.0, 100.0, 52.0, 110.0, "0.1", 0, 0, 0),
                    (50.0, 150.0, 53.0, 160.0, "0.5", 0, 1, 0),
                    (51.0, 200.0, 54.0, 210.0, "0.9", 0, 2, 0),
                ]

        res = VisualClassifier.classify_vector_bbox(MockPage(), MockRect(0, 0, 500, 500), caption="ROC AUC performance curve")
        assert res.visual_type == VisualType.COMPLEX
