import torch
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Any, Optional, Literal, Union, Callable
from torchvision.io import decode_image
from utils.few_shots_triplets_generator import decode_image_from_path
from utils.helpers import get_image_files, resolve_image_paths
from utils.bbox_utils import BBoxUtils
import cv2


class FewShotClassPredictor:
    def __init__(self, model_path: Union[Path, str], transform: Optional[Callable] = None) -> None:
        """
        Initialize the Few-Shot Class Predictor.

        Args:
            model_path: Path to the pre-trained model
            transform: Optional transform to be applied to the image
        """
        self.fewshot_model = torch.load(model_path, map_location="cpu")
        self.fewshot_model.eval()
        self.embedding_model = self.fewshot_model.embedding_net
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model.to(self.device)
        self.embedding_model.eval()
        self.transform = transform
        self.logger = logging.getLogger(__name__)

    def _compute_embedding_from_image_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute the image embedding for a given image.

        Args:
            image: Image tensor as a batch

        Returns:
            Image embedding as a tensor
        """
        # Compute the embedding using the model
        with torch.no_grad():
            embedding = self.embedding_model(image)
        return embedding


    def _compute_embedding_from_image_path(self, image_path: str) -> torch.Tensor:
        """
        Compute the image embedding for a given image.

        Args:
            image_path: Path to the image

        Returns:
            Image embedding as a tensor
        """
        # Load the image and preprocess it
        image = decode_image_from_path(image_path) # C,H,W and uint8
        image = image.float()/255.0
        if self.transform:
            image = self.transform(image)
        else:
            raise ValueError("Transform function is not defined. Please provide a transform function.")
        
        image = image.unsqueeze(0).to(self.device)  # Add batch dimension and move to device
        # Compute the embedding using the model
        embedding = self.embedding_model(image)
        return embedding
    
    def _compute_average_embedding_from_image_paths(self, image_paths: List[str]) -> torch.Tensor:
        """
        Compute the average embedding for a list of images.

        Args:
            image_paths: List of paths to the images

        Returns:
            Average embedding as a tensor
        """
        embeddings = [self._compute_embedding_from_image_path(path) for path in image_paths]
        average_embedding = torch.mean(torch.stack(embeddings), dim=0)
        return average_embedding.detach()
    
    def compute_class_prototypes(self, few_shot_dir: Path) -> Dict[int, torch.Tensor]:
        """
        Computes the class prototypes for each class in the few-shot directory.

        Args:
            few_shot_dir: Path to the few-shot directory containing images for each class.

        Returns:
            A dictionary where keys are class IDs (integers) and values are the corresponding class prototype embeddings (torch.Tensor).
        """
        few_shot_dir = Path(few_shot_dir)
        if not few_shot_dir.exists() or not few_shot_dir.is_dir():
            raise ValueError(f"Invalid few-shot directory: {few_shot_dir}")
        
        class_prototypes_dict = {}
        for class_folder in few_shot_dir.iterdir():
            if class_folder.is_dir():
                class_label = int(class_folder.name)
                image_paths = get_image_files(class_folder)
                if image_paths:
                    class_prototype = self._compute_average_embedding_from_image_paths(image_paths)
                    class_prototypes_dict[class_label] = class_prototype
                else:
                    self.logger.warning(f"No images found in {class_folder}. Skipping this class.")
        self.logger.info(f"Computed class prototypes for {len(class_prototypes_dict)} classes.")
        return class_prototypes_dict
    
    def predict_class_from_prototypes(self, img_src: Union[str, Path, List[str], List[Path]], labels_dir: Union[str, Path], class_prototypes_dict: Dict[int, torch.Tensor], output_dir: Path, distance_metric: Literal["euclidean", "cosine"] = "euclidean"):
        """
        Predicts the class of objects in an image based on pre-computed class prototypes.

        Args:
            img_src: Path to the image or list of images or directory.
            labels_dir: Directory containing the YOLO format label files (.txt).
            class_prototypes_dict: A dictionary where keys are class IDs (integers)
                                   and values are the corresponding class prototype embeddings (torch.Tensor).
        """
        
        image_paths = resolve_image_paths(img_src)
        if not image_paths:
            self.logger.error(f"No images found in {img_src}. Check the extensions and the directory.")
            return
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            self.logger.info(f"Processing image: {image_path}")
            # Check if the image exists
            if not Path(image_path).exists():
                self.logger.error(f"Image not found: {image_path}")
                continue

            # Read the corresponding label file

            label_path = Path(labels_dir) / (Path(image_path).stem + ".txt")
            output_path = Path(output_dir) / (Path(image_path).stem + ".txt")
            image = cv2.imread(str(image_path))
            image_height, image_width = image.shape[0], image.shape[1]

            try:
                predicted_labels = []
                yolo_preds = BBoxUtils.get_bboxes_array_from_file(label_path)
                for line in yolo_preds:
                    class_id, xc, yc, w, h = line[:5]
                    class_id = int(class_id)
                    x_min = int((xc - w / 2) * image_width)
                    y_min = int((yc - h / 2) * image_height)
                    x_max = int((xc + w / 2) * image_width)
                    y_max = int((yc + h / 2) * image_height)

                    # Crop the object from the image
                    cropped_image = BBoxUtils.crop_image(image, x_min, y_min, x_max, y_max)
                    cropped_image = torch.from_numpy(cropped_image).permute(2, 0, 1)  # H,W,C -> C,H,W
                    cropped_image= cropped_image.float()/255.0  # Normalize to [0,1]
                    if cropped_image is None:
                        self.logger.warning(f"Cropped image is None for object in {image_path} at coordinates (x1:{x_min}, y1:{y_min}, x2:{x_max}, y2:{y_max}). Skipping prediction for this object.")
                        predicted_labels.append(f"{-1} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                        continue
                    # Apply transformation if defined
                    if self.transform:
                        cropped_image = self.transform(cropped_image)
                    else:
                        raise ValueError("Transform function is not defined. Please provide a transform function.")
                    
                    # Add batch dimension and move to device
                    cropped_image = cropped_image.unsqueeze(0).to(self.device)

                    # Compute embedding for the cropped image
                    object_embedding = self.embedding_model(cropped_image)
                    
                    if distance_metric == "euclidean":
                        # Find the closest class prototype (k=1)
                        min_distance = float('inf')
                        predicted_class = -1
                        for cls_id, prototype in class_prototypes_dict.items():
                            distance = torch.cdist(object_embedding, prototype.unsqueeze(0)).item()
                            if distance < min_distance:
                                min_distance = distance
                                predicted_class = cls_id
                    elif distance_metric == "cosine":
                        max_similarity = -1.0
                        predicted_class = -1
                        for cls_id, prototype in class_prototypes_dict.items():
                            similarity = torch.nn.functional.cosine_similarity(object_embedding, prototype.unsqueeze(0)).item()
                            if similarity > max_similarity:
                                max_similarity = similarity
                                predicted_class = cls_id

                    predicted_labels.append(f"{predicted_class} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

                # Write the predicted labels to the output file
                with open(output_path, 'w') as f:
                    for label in predicted_labels:
                        f.write(label + '\n')
                self.logger.info(f"Predicted labels written to: {output_path}")

            except FileNotFoundError:
                self.logger.error(f"Label file not found at: {label_path}")
                continue

        return predicted_labels

    
# if __name__ == "__main__":
#     model_path = Path(r"C:\Users\mgupta70.ASURITE\Dropbox (ASU)\ASU\PhD\Courses\Github_projects\PID-Final-Cursor\models\symbol_classification\stage2\few_shot\best_fewshot_model.pth")
#     output_dir = Path(r"C:\Users\mgupta70.ASURITE\Dropbox (ASU)\ASU\PhD\Courses\Github_projects\PID-Final-Cursor\results\label_transferred")
#     fewshot_root_dir = Path(r"C:\Users\mgupta70.ASURITE\Dropbox (ASU)\ASU\PhD\Courses\Github_projects\PID-Final-Cursor\data\processed\stage_2\few_shot")
#     fewshot_model = torch.load(model_path)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     fewshot_model.to(device)
#     fewshot_model.eval()
#     embedding_net = fewshot_model.embedding_net  # Extract the embedding network
#     embedding_net.to(device)
#     embedding_net.eval()

#     class_protoypes_dict = {}
#     for class_folder in fewshot_root_dir.iterdir():
#         if class_folder.is_dir():
#             class_label = class_folder.name
#             images = get_files(class_folder, extensions=[".jpg", ".png"])
#             if images:
#                 class_protoypes_dict[class_label] = images
                