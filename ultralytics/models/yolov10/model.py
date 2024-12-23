from ultralytics.engine.model import Model
from ultralytics.nn.tasks import YOLOv10DetectionModel
from .val import YOLOv10DetectionValidator
from .predict import YOLOv10DetectionPredictor
from .train import YOLOv10DetectionTrainer
import torch
from huggingface_hub import PyTorchModelHubMixin
from .card import card_template_text

# Usage: # Assuming 'dataset' contains annotations with bounding boxes
# yolo = YOLOv10(model="yolov10n.pt", dataset=training_dataset, num_anchors=9)
# # Training and evaluation process remains the same

class YOLOv10(Model, PyTorchModelHubMixin, model_card_template=card_template_text):

    def __init__(self, model="yolov10n.pt", task=None, verbose=False, 
                 names=None, dataset=None, num_anchors=9):
        super().__init__(model=model, task=task, verbose=verbose)
        if names is not None:
            setattr(self.model, 'names', names)
        if dataset:
            self.anchors = self._generate_anchors(dataset, num_anchors)
            self._apply_anchors_to_model()

    def _generate_anchors(self, dataset, num_anchors):
        """
        Generate optimized anchor boxes using K-means clustering.
        Args:
            dataset: The dataset containing bounding box annotations.
            num_anchors: The number of anchor boxes to generate.
        Returns:
            List of optimized anchor box dimensions.
        """
        from sklearn.cluster import KMeans
        import numpy as np

        # Extract all ground truth boxes (width, height) from dataset
        boxes = []
        for image in dataset:
            for annotation in image['annotations']:
                width = annotation['xmax'] - annotation['xmin']
                height = annotation['ymax'] - annotation['ymin']
                boxes.append([width, height])

        boxes = np.array(boxes)
        kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(boxes)
        return kmeans.cluster_centers_.tolist()

    def _apply_anchors_to_model(self):
        """
        Apply the generated anchors to the YOLO model structure.
        """
        if not hasattr(self.model, 'model') or not self.anchors:
            raise ValueError("Model structure or anchors are missing.")

        detect_layer = next(
            (layer for layer in self.model.model if hasattr(layer, "anchors")), None
        )
        if detect_layer:
            detect_layer.anchors = torch.tensor(self.anchors).float()

    def push_to_hub(self, repo_name, **kwargs):
        config = kwargs.get('config', {})
        config['names'] = self.names
        config['model'] = self.model.yaml['yaml_file']
        config['task'] = self.task
        kwargs['config'] = config
        super().push_to_hub(repo_name, **kwargs)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10DetectionModel,
                "trainer": YOLOv10DetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
        }