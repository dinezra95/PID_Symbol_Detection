from pipeline.base import BasePipeline
from typing import Union, Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from utils.helpers import get_files, resolve_image_paths
from utils.yolo_predictor import YOLOPredictor

class Stage1InferencePipeline(BasePipeline):
    """Stage 1 Evaluation Pipeline."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize stage 1 evaluation pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        self.yolo_predictor = None
        self.logger = logging.getLogger(__name__)
        
    def validate(self) -> bool:
        """Validate stage 1 evaluation pipeline inputs.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        paths = self.get_data_paths()
        # Check if raw data directories exist
        if not paths['raw_images_dir'].exists():
            raise FileNotFoundError("Raw image directory not found")
        
    def run(self, input_dir=None, output_dir=None, model_path=None) -> None:
        """Run stage 1 evaluation pipeline."""

        self.validate()
        self.config = self.get_stage1_inference_config()

        src = Path(input_dir) if input_dir else self.get_data_paths()['raw_images_dir']
        dst = Path(output_dir) if output_dir else self.get_data_paths()['class_agnostic_results_dir']

        # Initialize YOLOPredictor
        self.model_path = Path(model_path) if model_path else get_files(self.get_model_paths()['stage1_class_agnostic_weights_dir'], [".pt"])[0]
        self.logger.info(f"Loading YOLO model from {self.model_path}")
        self.yolo_predictor = YOLOPredictor(self.model_path)

        # Perform inference
        self.yolo_predictor.perform_sliced_inference(
            src=src,
            conf=self.config['conf'],
            slice_height=self.config['slice_height'],
            slice_width=self.config['slice_width'],
            save_txt=self.config['save_txt'],
            save_conf=self.config['save_conf'],
            overlap_height_ratio=self.config['overlap_height_ratio'],
            overlap_width_ratio=self.config['overlap_width_ratio'],
            output_dir=dst
        )
        self.logger.info(f"Stage 1 inference completed. Results saved to {dst}")
       
        
    