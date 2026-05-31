import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from .text_detector import TextDetector
import logging

logger = logging.getLogger(__name__)


class PipePreprocessor:
    """Prepares P&ID image for pipe detection.

    Pipeline:
        1. Downscale to working resolution (~3000px width)
        2. Detect text regions (MSER) and mask them
        3. Mask detected symbols (from YOLO stage1)
        4. Binarize → remove text+symbol regions
        5. Morphological line extraction (H+V kernels)
        6. Cleanup + border exclusion

    All processing happens at working resolution; results are
    upscaled back to original resolution at the end.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.working_width = config.get("working_width", None)

        # Binarization
        self.binary_threshold = config.get("binary_threshold", 200)

        # Morphological line extraction
        self.line_kernel_length = config.get("line_kernel_length", 15)
        self.morph_close_iterations = config.get("morph_close_iterations", 2)

        # Masking
        self.symbol_dilate_px = config.get("symbol_dilate_px", 3)

        # Border exclusion
        self.exclude_bottom_pct = config.get("exclude_bottom_pct", 0.08)
        self.exclude_left_pct = config.get("exclude_left_pct", 0.12)

        # Text detection config
        self.text_config = config.get("text_detection", {})

    def load_symbol_bboxes(
        self, label_path: str, img_width: int, img_height: int
    ) -> List[Tuple[int, int, int, int]]:
        """Load YOLO-format bounding boxes → pixel coords (x1, y1, x2, y2)."""
        bboxes = []
        path = Path(label_path)
        if not path.exists():
            logger.warning(f"Label file not found: {label_path}")
            return bboxes
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = (float(p) for p in parts[:5])
                x1 = int((cx - w / 2) * img_width)
                y1 = int((cy - h / 2) * img_height)
                x2 = int((cx + w / 2) * img_width)
                y2 = int((cy + h / 2) * img_height)
                bboxes.append((x1, y1, x2, y2))
        logger.info(f"Loaded {len(bboxes)} symbol bounding boxes")
        return bboxes

    def _compute_scale(self, orig_width: int) -> float:
        if self.working_width is None or self.working_width >= orig_width:
            return 1.0
        return self.working_width / orig_width

    def _scale_bboxes(
        self, bboxes: List[Tuple[int, int, int, int]], scale: float
    ) -> List[Tuple[int, int, int, int]]:
        return [
            (int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale))
            for x1, y1, x2, y2 in bboxes
        ]

    def _create_mask(
        self, shape: Tuple[int, int], bboxes: List[Tuple[int, int, int, int]], dilate: int = 0
    ) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for x1, y1, x2, y2 in bboxes:
            mask[max(0, y1 - dilate):min(h, y2 + dilate),
                 max(0, x1 - dilate):min(w, x2 + dilate)] = 255
        return mask

    def preprocess(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
        """Full preprocessing pipeline.

        Returns (all at working resolution):
            pipe_mask: binary mask of detected pipe pixels
            symbol_mask: binary mask of symbol regions
            text_mask: binary mask of text regions
            working_image: downscaled image
            text_bboxes: list of (x1,y1,x2,y2) text bounding boxes at working res
        """
        orig_h, orig_w = image.shape[:2]
        scale = self._compute_scale(orig_w)
        sw = int(orig_w * scale)
        sh = int(orig_h * scale)
        logger.info(f"Preprocessing {orig_w}x{orig_h} → {sw}x{sh} (scale={scale:.3f})")

        if scale < 1.0:
            small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
        else:
            small = image.copy()

        # --- Text detection (MSER) ---
        text_det = TextDetector(self.text_config)
        text_mask, text_bboxes = text_det.detect_and_mask(small)

        # --- Symbol mask ---
        bboxes_small = self._scale_bboxes(bboxes, scale)
        symbol_mask = self._create_mask((sh, sw), bboxes_small, dilate=self.symbol_dilate_px)

        # --- Combined exclusion mask ---
        exclusion_mask = cv2.bitwise_or(text_mask, symbol_mask)

        # --- Binarize (dark features → white on black) ---
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.binary_threshold, 255, cv2.THRESH_BINARY_INV)

        # Remove text + symbol regions
        pipe_raw = cv2.bitwise_and(binary, cv2.bitwise_not(exclusion_mask))

        # --- Border exclusion ---
        if self.exclude_bottom_pct > 0:
            pipe_raw[int(sh * (1 - self.exclude_bottom_pct)):, :] = 0
        if self.exclude_left_pct > 0:
            pipe_raw[:, :int(sw * self.exclude_left_pct)] = 0

        # --- Morphological line extraction ---
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        pipe_raw = cv2.morphologyEx(pipe_raw, cv2.MORPH_CLOSE, k_close,
                                     iterations=self.morph_close_iterations)

        klen = self.line_kernel_length
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, klen))
        h_lines = cv2.morphologyEx(pipe_raw, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(pipe_raw, cv2.MORPH_OPEN, v_kernel)
        pipe_mask = cv2.bitwise_or(h_lines, v_lines)

        # Dilate slightly for junction connectivity
        pipe_mask = cv2.dilate(pipe_mask, k_close, iterations=1)

        px = np.count_nonzero(pipe_mask)
        logger.info(
            f"Pipe mask: {px} px ({px / pipe_mask.size:.4f}), "
            f"text regions: {len(text_bboxes)}, symbols masked: {len(bboxes_small)}"
        )

        return pipe_mask, symbol_mask, text_mask, small, text_bboxes
