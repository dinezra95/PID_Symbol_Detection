"""Stage 3: Pipe Detection Pipeline.

Detects pipe lines, junctions, and connectivity from P&ID drawings
using classical computer vision.

Pipeline:
    1. Downscale to working resolution
    2. MSER text detection → mask text regions
    3. Mask stage1 symbol detections
    4. Binarize → remove text+symbols → morphological line extraction
    5. Hough Transform → line segments → merge collinear → bridge gaps
    6. Build connectivity graph (junctions + edges)
    7. OCR text reading + label-to-pipe matching (pytesseract)
    8. Output structured graph (JSON) + visualizations
"""

from pipeline.base import BasePipeline
from typing import Dict, Optional
from pathlib import Path
import cv2
import logging

from utils.pipe_detection.data_structures import Point, LineSegment
from utils.pipe_detection.preprocessor import PipePreprocessor
from utils.pipe_detection.line_detector import LineDetector
from utils.pipe_detection.graph_builder import GraphBuilder
from utils.pipe_detection.label_matcher import LabelMatcher
from utils.pipe_detection.visualizer import PipeVisualizer
from utils.helpers import get_files

logger = logging.getLogger(__name__)


class Stage3PipeDetectionPipeline(BasePipeline):
    """Stage 3: Classical CV pipe detection from P&ID drawings."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        super().__init__(config_path)

    def validate(self) -> bool:
        return True

    def run(
        self,
        input_dir: Optional[str] = None,
        labels_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        pipe_config = self.config_manager.config.get("pipe_detection", {})

        src = Path(input_dir) if input_dir else self.get_data_paths().get("raw_images_dir")
        dst = Path(output_dir) if output_dir else Path("results/stage_3/pipe_detection")
        lbl_dir = Path(labels_dir) if labels_dir else None

        if not src or not src.exists():
            raise FileNotFoundError(f"Input directory not found: {src}")

        image_files = get_files(src, [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"])
        if not image_files:
            raise FileNotFoundError(f"No images found in {src}")

        logger.info(f"Processing {len(image_files)} images from {src}")
        dst.mkdir(parents=True, exist_ok=True)

        for img_path in image_files:
            self._process_single_image(img_path, lbl_dir, dst, pipe_config)

    def _process_single_image(
        self,
        img_path: Path,
        labels_dir: Optional[Path],
        output_dir: Path,
        pipe_config: Dict,
    ):
        img_name = img_path.stem
        logger.info(f"\n{'='*60}\nProcessing: {img_name}\n{'='*60}")

        image = cv2.imread(str(img_path))
        if image is None:
            logger.error(f"Failed to load image: {img_path}")
            return

        orig_h, orig_w = image.shape[:2]
        img_output = output_dir / img_name
        img_output.mkdir(parents=True, exist_ok=True)

        label_path = self._find_label_file(img_path, labels_dir)

        preproc_cfg = pipe_config.get("preprocessing", {})
        preprocessor = PipePreprocessor(preproc_cfg)
        line_detector = LineDetector(pipe_config.get("line_detection", {}))
        graph_builder = GraphBuilder(pipe_config.get("graph", {}))
        label_matcher = LabelMatcher(pipe_config.get("ocr", {}))
        visualizer = PipeVisualizer(str(img_output))

        working_width = preproc_cfg.get("working_width", None)
        if working_width is None or working_width >= orig_w:
            scale = 1.0
        else:
            scale = working_width / orig_w
        inv_scale = 1.0 / scale

        # --- Step 1: Load symbol bboxes ---
        bboxes_orig = []
        if label_path:
            bboxes_orig = preprocessor.load_symbol_bboxes(str(label_path), orig_w, orig_h)
        else:
            logger.warning(f"No label file for {img_name} — running without symbol masking")

        # --- Step 2-4: Preprocess (text detect + mask + binarize + morph) ---
        pipe_mask, symbol_mask, text_mask, working_img, text_bboxes = \
            preprocessor.preprocess(image, bboxes_orig)

        visualizer.save_mask(pipe_mask, "01_pipe_mask")
        visualizer.save_mask(text_mask, "02_text_mask")

        # Draw text detection overlay
        text_vis = working_img.copy()
        for x1, y1, x2, y2 in text_bboxes:
            cv2.rectangle(text_vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite(str(img_output / "03_text_detection.png"), text_vis)

        # --- Step 5: Line detection (at working resolution) ---
        bboxes_small = preprocessor._scale_bboxes(bboxes_orig, scale)
        segments_small = line_detector.detect_and_refine(pipe_mask, bboxes_small)

        if not segments_small:
            logger.warning(f"No pipe segments detected in {img_name}")
            return

        # Scale segments to original resolution
        segments_orig = [
            LineSegment(
                Point(s.start.x * inv_scale, s.start.y * inv_scale),
                Point(s.end.x * inv_scale, s.end.y * inv_scale),
            )
            for s in segments_small
        ]
        visualizer.draw_lines_on_image(image, segments_orig, "04_detected_lines", thickness=3)

        # --- Step 6: Build graph (at working resolution) ---
        graph = graph_builder.build_graph(segments_small)
        graph = graph_builder.simplify_graph(graph)

        # Scale graph to original resolution
        self._scale_graph(graph, inv_scale)

        # --- Step 7: OCR label matching ---
        text_regions = label_matcher.extract_text_regions(image)
        if text_regions:
            pipe_labels = label_matcher.filter_pipe_labels(text_regions)
            label_matcher.match_labels_to_pipes(pipe_labels, graph)

        # --- Step 7b: Filter — keep only components with diameter labels ---
        visualizer.draw_pipe_graph(image, graph, "05_pipe_graph_unfiltered")
        graph = graph_builder.filter_to_labeled_components(graph)

        # --- Step 8: Output ---
        visualizer.draw_pipe_graph(image, graph, "06_pipe_graph_filtered")

        json_path = str(img_output / "pipe_graph.json")
        graph.save_json(json_path)
        logger.info(f"Saved: {json_path}")
        logger.info(graph.summary())

    @staticmethod
    def _scale_graph(graph, factor: float):
        for j in graph.junctions.values():
            j.position = Point(j.position.x * factor, j.position.y * factor)
        for p in graph.pipe_segments.values():
            for seg in p.segments:
                seg.start = Point(seg.start.x * factor, seg.start.y * factor)
                seg.end = Point(seg.end.x * factor, seg.end.y * factor)
        for nid in list(graph.graph.nodes):
            if "pos" in graph.graph.nodes[nid]:
                old = graph.graph.nodes[nid]["pos"]
                graph.graph.nodes[nid]["pos"] = (int(old[0] * factor), int(old[1] * factor))

    def _find_label_file(self, img_path: Path, labels_dir: Optional[Path]) -> Optional[Path]:
        stem = img_path.stem
        if labels_dir:
            for candidate in [labels_dir / f"{stem}.txt", labels_dir / "labels" / f"{stem}.txt"]:
                if candidate.exists():
                    return candidate
        for candidate in [img_path.with_suffix(".txt"),
                          img_path.parent / f"{stem}.txt",
                          img_path.parent.parent / f"{stem}.txt"]:
            if candidate.exists():
                return candidate
        return None
