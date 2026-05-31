import cv2
import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from .data_structures import Point, LineSegment, PipeSegment, TextRegion, PipeGraph
import logging

logger = logging.getLogger(__name__)


class LabelMatcher:
    """Extract text labels from P&ID drawing and match them to pipe segments."""

    def __init__(self, config: Dict):
        self.config = config
        self.max_label_distance = config.get("max_label_distance", 80)
        self.ocr_backend = config.get("ocr_backend", "pytesseract")
        self.ocr_config = config.get("ocr_config", "--oem 3 --psm 6")
        self.pipe_label_pattern = config.get(
            "pipe_label_pattern",
            r'(?:Ø|O|D)?[\s]*[\d/\."]+[\s]*(?:mm|in|"|\')?[\s]*[A-Za-z_]*'
        )
        self._ocr_engine = None

    def _get_ocr_engine(self):
        if self._ocr_engine is not None:
            return self._ocr_engine

        if self.ocr_backend == "pytesseract":
            try:
                import pytesseract
                self._ocr_engine = pytesseract
                return self._ocr_engine
            except ImportError:
                logger.warning(
                    "pytesseract not installed. Install with: pip install pytesseract\n"
                    "Also install Tesseract OCR: brew install tesseract"
                )
        elif self.ocr_backend == "easyocr":
            try:
                import easyocr
                self._ocr_engine = easyocr.Reader(["en"])
                return self._ocr_engine
            except ImportError:
                logger.warning("easyocr not installed. Install with: pip install easyocr")

        return None

    def extract_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """Run OCR on the image and return detected text regions."""
        engine = self._get_ocr_engine()

        if engine is None:
            logger.warning("No OCR engine available — skipping text extraction")
            return []

        if self.ocr_backend == "pytesseract":
            return self._extract_pytesseract(image, engine)
        elif self.ocr_backend == "easyocr":
            return self._extract_easyocr(image, engine)

        return []

    def _extract_pytesseract(self, image: np.ndarray, pytesseract) -> List[TextRegion]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        data = pytesseract.image_to_data(gray, config=self.ocr_config, output_type=pytesseract.Output.DICT)

        regions = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not text or conf < 30:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            regions.append(TextRegion(bbox=(x, y, w, h), text=text, confidence=conf / 100.0))

        logger.info(f"pytesseract extracted {len(regions)} text regions")
        return regions

    def _extract_easyocr(self, image: np.ndarray, reader) -> List[TextRegion]:
        results = reader.readtext(image)
        regions = []
        for bbox_pts, text, conf in results:
            if conf < 0.3 or not text.strip():
                continue
            pts = np.array(bbox_pts)
            x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
            w = int(pts[:, 0].max()) - x
            h = int(pts[:, 1].max()) - y
            regions.append(TextRegion(bbox=(x, y, w, h), text=text.strip(), confidence=conf))

        logger.info(f"easyocr extracted {len(regions)} text regions")
        return regions

    def filter_pipe_labels(self, text_regions: List[TextRegion]) -> List[TextRegion]:
        """Filter text regions that look like pipe diameter labels.

        Matches:
            - Ø/O/@/D + number: @3", O75, D110
            - Number + inch mark: 8", 1/4", 21"
            - Fraction patterns: 1/4, 3/8 (common pipe sizes)
            - Standalone small numbers near pipes (interpreted as inches)
        Excludes:
            - Instrument tags: PI-2, TI-3, T-2, F-01, P-01, PR-01
            - Flow/power: 150GPM, 160kW, 180gpm
            - Reference numbers: 2A.1, 3A.2, phone numbers
        """
        exclude_pattern = re.compile(
            r'^(?:PI|TI|PR|PT|FI|FT|LI|LT|P|T|F|V|HV|CV|SV|BV)[-_]?\d|'  # instrument tags
            r'\d+\s*[gk][pPwW]|'          # flow/power: 150GPM, 160kW
            r'\d+[A-Z]\.\d|'              # reference: 2A.1, 3A.2
            r'^\d{3,}-|'                   # phone-like: 052-xxx
            r'www\.|\.co\.|@.*\.',         # URLs/emails
            re.IGNORECASE
        )

        diameter_pattern = re.compile(
            r'[Ø@][\s]*\d+|'              # Ø3, @3, @90X2
            r'[OD]\d{2,3}\b|'             # O75, D110 (O/D + 2-3 digits)
            r'\b\d{1,2}/\d{1,2}\b\s*["\']?|'  # 1/4", 3/8 (small numerator/denominator)
            r'\d+\s*"|'                    # 8", 21"
            r'PVCPN\d+',                   # PVCPN16 (pipe spec)
            re.IGNORECASE
        )

        pipe_labels = []
        for tr in text_regions:
            text = tr.text.strip()
            if not text:
                continue
            if exclude_pattern.search(text):
                continue
            if text.startswith('0') and len(text) > 1:
                continue
            if diameter_pattern.search(text):
                pipe_labels.append(tr)

        logger.info(f"Filtered {len(pipe_labels)} pipe-label candidates from {len(text_regions)} text regions")
        return pipe_labels

    def match_labels_to_pipes(
        self, text_regions: List[TextRegion], graph: PipeGraph
    ) -> PipeGraph:
        """Associate each text region with the nearest pipe segment."""
        for tr in text_regions:
            graph.text_regions.append(tr)

            best_pipe_id = None
            best_dist = float("inf")

            for pid, pipe in graph.pipe_segments.items():
                for seg in pipe.segments:
                    dist = seg.perpendicular_distance(tr.center)
                    if dist < best_dist:
                        best_dist = dist
                        best_pipe_id = pid

            if best_pipe_id is not None and best_dist <= self.max_label_distance:
                pipe = graph.pipe_segments[best_pipe_id]
                parsed = self.parse_pipe_label(tr.text)
                if parsed.get("diameter"):
                    pipe.diameter = parsed["diameter"]
                if parsed.get("pipe_type"):
                    pipe.pipe_type = parsed["pipe_type"]
                if pipe.label:
                    pipe.label = f"{pipe.label}; {tr.text}"
                else:
                    pipe.label = tr.text

        labeled_count = sum(1 for p in graph.pipe_segments.values() if p.label)
        logger.info(f"Matched labels: {labeled_count}/{len(graph.pipe_segments)} pipes have labels")
        return graph

    def parse_pipe_label(self, text: str) -> Dict[str, Optional[str]]:
        """Parse a pipe label string into structured fields."""
        result: Dict[str, Optional[str]] = {"diameter": None, "pipe_type": None, "material": None}

        diameter_match = re.search(
            r'(?:Ø|O|D)?\s*([\d/]+(?:\.\d+)?)\s*(?:"|\binch\b|\bmm\b|\')?',
            text, re.IGNORECASE
        )
        if diameter_match:
            result["diameter"] = diameter_match.group(0).strip()

        gas_patterns = [
            (r'\bN[_2]?\b.*\bX[_e]?\b', "Ne_Xe"),
            (r'\bHe\b', "He"),
            (r'\bN2\b', "N2"),
            (r'\bF[_2]?\b', "F2"),
            (r'\bPCW\b', "PCW"),
            (r'\bCDA\b', "CDA"),
            (r'\bDIW?\b', "DIW"),
        ]
        for pattern, ptype in gas_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result["pipe_type"] = ptype
                break

        return result
