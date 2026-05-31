import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class TextDetector:
    """Detect text regions in P&ID drawings using MSER + optional pytesseract.

    MSER (Maximally Stable Extremal Regions) detects character-like regions
    without needing any external OCR engine. Characters are then clustered
    into word/line bounding boxes that can be used as masks.

    Optionally, pytesseract can be used to also read the text content.
    """

    def __init__(self, config: Dict = None):
        config = config or {}
        # MSER params
        self.mser_delta = config.get("mser_delta", 5)
        self.mser_min_area = config.get("mser_min_area", 10)
        self.mser_max_area = config.get("mser_max_area", 2000)
        self.mser_max_variation = config.get("mser_max_variation", 0.25)

        # Character filtering
        self.char_min_aspect = config.get("char_min_aspect", 0.15)
        self.char_max_aspect = config.get("char_max_aspect", 5.0)
        self.char_max_width = config.get("char_max_width", 80)
        self.char_max_height = config.get("char_max_height", 80)

        # Clustering: group nearby characters into text lines
        self.cluster_x_gap = config.get("cluster_x_gap", 15)
        self.cluster_y_gap = config.get("cluster_y_gap", 8)
        self.min_chars_per_group = config.get("min_chars_per_group", 2)

        # Mask dilation around text regions
        self.text_dilate_px = config.get("text_dilate_px", 3)

    def detect_text_regions_mser(
        self, image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Detect text character regions using MSER and cluster into text bboxes.

        Returns list of (x1, y1, x2, y2) bounding boxes of text regions.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        mser = cv2.MSER_create()
        mser.setDelta(self.mser_delta)
        mser.setMinArea(self.mser_min_area)
        mser.setMaxArea(self.mser_max_area)
        mser.setMaxVariation(self.mser_max_variation)

        regions, _ = mser.detectRegions(gray)

        char_bboxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            if w == 0 or h == 0:
                continue
            aspect = w / h
            if (self.char_min_aspect <= aspect <= self.char_max_aspect
                    and w <= self.char_max_width
                    and h <= self.char_max_height):
                char_bboxes.append((x, y, x + w, y + h))

        logger.info(f"MSER detected {len(char_bboxes)} character-like regions from {len(regions)} raw regions")

        # Remove duplicates (MSER often finds overlapping regions for same character)
        char_bboxes = self._deduplicate_bboxes(char_bboxes)

        # Cluster characters into text lines/words
        text_bboxes = self._cluster_characters(char_bboxes)

        logger.info(f"Clustered into {len(text_bboxes)} text regions")
        return text_bboxes

    def _deduplicate_bboxes(
        self, bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Remove overlapping bboxes, keeping the larger one."""
        if not bboxes:
            return []

        bboxes = sorted(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        keep = []
        for box in bboxes:
            is_dup = False
            for kept in keep:
                if self._iou(box, kept) > 0.5:
                    is_dup = True
                    break
            if not is_dup:
                keep.append(box)
        return keep

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

    def _cluster_characters(
        self, char_bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Tuple[int, int, int, int]]:
        """Cluster nearby character bboxes into text line bounding boxes.

        Uses a simple greedy approach: sort by y then x, merge characters
        that are close in both x and y into groups.
        """
        if not char_bboxes:
            return []

        # Sort by y-center, then x-center
        chars = sorted(char_bboxes, key=lambda b: ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2))

        groups: List[List[Tuple[int, int, int, int]]] = []
        used = [False] * len(chars)

        for i, box in enumerate(chars):
            if used[i]:
                continue
            group = [box]
            used[i] = True
            cy_i = (box[1] + box[3]) / 2

            for j in range(i + 1, len(chars)):
                if used[j]:
                    continue
                other = chars[j]
                cy_j = (other[1] + other[3]) / 2

                # Stop scanning if too far below (sorted by y)
                if cy_j - cy_i > self.cluster_y_gap * 3:
                    break

                # Check if close enough in y
                if abs(cy_j - cy_i) > self.cluster_y_gap:
                    continue

                # Check if close enough in x to any member of the group
                for member in group:
                    mx2 = member[2]
                    mx1 = member[0]
                    ox1 = other[0]
                    ox2 = other[2]
                    x_gap = max(ox1 - mx2, mx1 - ox2)
                    if x_gap <= self.cluster_x_gap:
                        group.append(other)
                        used[j] = True
                        break

            groups.append(group)

        # Convert groups to bounding boxes
        text_bboxes = []
        for group in groups:
            if len(group) < self.min_chars_per_group:
                continue
            x1 = min(b[0] for b in group)
            y1 = min(b[1] for b in group)
            x2 = max(b[2] for b in group)
            y2 = max(b[3] for b in group)
            text_bboxes.append((x1, y1, x2, y2))

        return text_bboxes

    def create_text_mask(
        self,
        img_shape: Tuple[int, ...],
        text_bboxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """Create binary mask where detected text regions are white (255)."""
        h, w = img_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        d = self.text_dilate_px
        for x1, y1, x2, y2 in text_bboxes:
            mask[max(0, y1 - d):min(h, y2 + d), max(0, x1 - d):min(w, x2 + d)] = 255
        return mask

    def detect_and_mask(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """Detect text regions and return mask + bounding boxes.

        Returns:
            text_mask: binary mask (255 = text region)
            text_bboxes: list of (x1, y1, x2, y2)
        """
        text_bboxes = self.detect_text_regions_mser(image)
        text_mask = self.create_text_mask(image.shape, text_bboxes)
        return text_mask, text_bboxes
