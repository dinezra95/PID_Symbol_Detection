from ultralytics import YOLO
from pathlib import Path
import logging
from typing import List, Optional, Union
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction, predict
import torch
from utils.helpers import resolve_image_paths
from utils.bbox_utils import coco_array_to_yolo_file

class YOLOPredictor:
    def __init__(self, model_path: Path, ):
        """
        Initialize the YOLO predictor.

        Args:
            model_path: Path to the YOLO model
        """
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)

    def perform_standard_inference(self):        
        pass

    def perform_sliced_inference(self, 
                                 src: Union[str, Path, List[str], List[Path]], 
                                 conf: float = 0.5,
                                 slice_height: int = 1024,
                                 slice_width: int = 1024,
                                 save_txt: bool = True,
                                 save_conf: bool = False,
                                 overlap_height_ratio: float = 0.1,
                                 overlap_width_ratio: float = 0.1,
                                 output_dir: Optional[Path] = None,
                                 ) -> None:

        image_paths = resolve_image_paths(src)
        # print('image_paths: ', image_paths)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # print('device: ', device)
        conf = conf
        detection_model = AutoDetectionModel.from_pretrained(
            model_type = 'ultralytics',
            model_path = self.model_path,
            confidence_threshold=conf,
            device = device)
        
        for image_path in image_paths:
            result = get_sliced_prediction(str(image_path), detection_model, slice_height, slice_width, overlap_height_ratio, overlap_width_ratio)
            result.export_visuals(export_dir=f"{str(output_dir)}/{str(Path(image_path).stem)}", hide_labels=True, hide_conf=True)
            if save_txt:
                coco_array_to_yolo_file(result.to_coco_predictions(), result, output_dir, Path(image_path), save_conf)

