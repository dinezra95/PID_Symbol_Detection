from pipeline.base import BasePipeline
from typing import Union, Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from utils.few_shot_predictor import FewShotClassPredictor
from utils.helpers import get_files, resolve_image_paths
import torch
import torchvision.transforms as v2

class Stage2FewShotInferencePipeline(BasePipeline):
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize stage 2 few-shot inference pipeline.

        Args:
            config_path: Path to configuration file
        """
        super().__init__(config_path)
        self.few_shot_predictor = None
        self.logger = logging.getLogger(__name__)

    def validate(self) -> bool:
        """Validate stage 2 few-shot inference pipeline inputs.

        Returns:
            bool: True if validation passes, False otherwise
        """
        paths = self.get_data_paths()
        model_paths = self.get_model_paths()

        # Check if stage 1 model dir exists
        if not model_paths['stage1_class_agnostic_weights_dir'].exists():
            raise FileNotFoundError("Stage 1 model directory not found")

        # Check if raw data directories exist
        if not paths['raw_images_dir'].exists() or not paths['raw_class_aware_labels_dir'].exists():
            raise FileNotFoundError("Raw data directories not found")
        
        # Check if few-shot model directory exists
        if not len(get_files(model_paths['stage2_few_shot_weights_dir'], [".pth"])) > 0:
            raise FileNotFoundError("Few-shot model directory not found or empty")
        return True
    
    def run(self, input_dir=None, labels_dir=None, output_dir=None) -> None:
        """Run stage 2 few-shot inference pipeline."""
        self.validate()
        self.config = self.get_stage2_inference_config()
        model_paths = self.get_model_paths()
        paths = self.get_data_paths()

        src = Path(input_dir) if input_dir else paths['raw_images_dir']
        lbls = Path(labels_dir) if labels_dir else paths['class_agnostic_results_dir']
        dst = Path(output_dir) if output_dir else paths['few_shot_results_dir']

        # Initialize FewShotClassPredictor
        self.model_path = model_paths['stage2_few_shot_weights_dir'] / 'best_fewshot_model.pth'
        self.logger.info(f"Loading Few-Shot model from {self.model_path}")
        apn_transform = v2.Compose([
            v2.Resize(size=(128, 128), antialias=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.few_shot_predictor = FewShotClassPredictor(self.model_path, transform=apn_transform)

        # Compute class prototypes
        self.logger.info("Computing class prototypes")
        self.class_prototypes = self.few_shot_predictor.compute_class_prototypes(paths['few_shot_dir'])
        self.logger.info("Class prototypes computed successfully")

        # Perform inference
        self.logger.info("Performing few-shot inference")
        self.few_shot_predictor.predict_class_from_prototypes(
            img_src=src,
            labels_dir=lbls,
            class_prototypes_dict=self.class_prototypes,
            output_dir=dst,
            distance_metric=self.config['distance_metric'],
        )

        self.logger.info("Stage 2 few-shot inference completed successfully.")
        self.logger.info(f"Results saved to {dst}")

       
