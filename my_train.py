# coding:utf-8
from ultralytics import YOLOv10
import os
import yaml
import json

# 模型配置文件
model_yaml_path = "ultralytics/cfg/models/v10/yolov10n.yaml"
# 数据集配置文件
data_yaml_path = 'ultralytics/cfg/datasets/tobacco.yaml'
# 预训练模型
pre_model_name = 'yolov10n.pt'


def load_dataset(data_yaml_path):
    """
    Load the dataset from the YAML configuration file and extract bounding boxes.

    Args:
        data_yaml_path (str): Path to the YAML file containing dataset configuration.

    Returns:
        list: A list of dictionaries, each containing the image path and its corresponding bounding box annotations.
              Example:
              [
                  {
                      "image": "path/to/image1.jpg",
                      "annotations": [
                          {"xmin": 10, "ymin": 20, "xmax": 50, "ymax": 60},
                          {"xmin": 30, "ymin": 40, "xmax": 70, "ymax": 80}
                      ]
                  },
                  ...
              ]
    """
    # Load dataset configuration from YAML file
    with open(data_yaml_path, 'r') as f:
        dataset_config = yaml.safe_load(f)

    dataset = []

    # Paths to the train, validation, and test datasets
    for split in ["train", "val", "test"]:
        if split in dataset_config:
            split_dir = dataset_config[split]  # Path to the split directory
            annotations_path = os.path.join(split_dir, "annotations.json")  # Example annotation file path

            # Check if the annotations file exists
            if not os.path.exists(annotations_path):
                raise FileNotFoundError(f"Annotations file not found at: {annotations_path}")

            # Load annotations from JSON file
            with open(annotations_path, 'r') as af:
                annotations = json.load(af)

            # Parse annotations for each image
            for img_data in annotations:
                image_path = os.path.join(split_dir, img_data["image"])  # Full path to the image
                bboxes = img_data.get("bboxes", [])  # List of bounding boxes

                # Prepare bounding boxes
                bbox_annotations = []
                for bbox in bboxes:
                    bbox_annotations.append({
                        "xmin": bbox[0],
                        "ymin": bbox[1],
                        "xmax": bbox[2],
                        "ymax": bbox[3]
                    })

                # Append to dataset list
                dataset.append({
                    "image": image_path,
                    "annotations": bbox_annotations
                })

    return dataset

if __name__ == '__main__':
    # 加载数据集
    dataset = load_dataset(data_yaml_path)

    # 加载预训练模型
    model = YOLOv10(model_yaml_path).load('runs/detect/train_v103/weights/best.pt')

    # 动态生成 Anchor Boxes 并训练模型
    results = model.train(
        dataset=dataset,  # 传入数据集用于 Anchor Box 生成
        data=data_yaml_path,  # 数据集配置文件路径
        epochs=100,  # 训练轮数
        batch=8,  # 批量大小
        num_anchors=9,  # Anchor Boxes 数量
        name='train_v10_dynamic_anchors'  # 训练任务名称
    )