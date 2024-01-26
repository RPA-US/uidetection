import json

import cv2
from tqdm import tqdm

from utils import *


def predict(directory, output_dir):
    images = dict()
    for file in os.listdir(directory):
        if file.endswith(".png") or file.endswith(".jpg"):
            images[file] = f"{directory}/{file}"

    detections = dict()

    if not os.path.exists(output_dir + "/yolo"):
        os.makedirs(output_dir + "/yolo")
    if not os.path.exists(output_dir + "/yolo/detections"):
        os.makedirs(output_dir + "/yolo/detections")

    for img_name, img_path in tqdm(images.items(), desc="Running YOLO predictions"):
        detections[img_name] = dict()
        image_pil = cv2.imread(img_path)

        # Elements level preditions
        shapes = yolo_prediction("Models/base/yolov8s-seg.pt", image_pil, "seg", 0, 0.5)
        detections[img_name]["shapes"] = shapes
        detections[img_name]["imageWidth"] = image_pil.shape[1]
        detections[img_name]["imageHeight"] = image_pil.shape[0]

        # Save detections
        json.dump(
            detections[img_name],
            open(output_dir + f"/yolo/detections/detections_{img_name}.json", "w"),
        )

    return detections
