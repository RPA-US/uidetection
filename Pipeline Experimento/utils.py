import json
import os

import cv2
import numpy as np
import utils
from shapely.geometry import Polygon


def load_dataset(dataset_path):
    dataset_labels = dict()
    for f in os.listdir(dataset_path):
        if f.endswith(".json"):
            labeled_json = json.load(open(f"{dataset_path}/{f}"))

            # Order the shapes by area (from largest to smallest) for readability
            labeled_json["shapes"].sort(key=lambda x: Polygon(x["points"]).area, reverse=True)

            # We add these ids to track the elements troughout all the metrics
            for i, shape in enumerate(labeled_json["shapes"]):
                shape["id"] = i

            img_name = f.split(".")[0] + ".png"
            dataset_labels[img_name] = labeled_json

    return dataset_labels


def coco_to_labelme(coco_anns, type="bbox", id_start=0):
    res = []
    for i, ann in enumerate(coco_anns):
        if type == "bbox":
            x, y, w, h = ann["bbox"]
            points = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
            ]

            if Polygon(points).is_valid == False:
                continue

            res.append(
                {
                    "label": ann["category_name"],
                    "points": points,
                }
            )

        elif type == "seg":
            points = np.array(ann["segmentation"][0]).reshape(-1, 2)

            if Polygon(points).is_valid == False:
                continue

            res.append(
                {
                    "label": ann["category_name"],
                    "points": points.tolist(),
                }
            )
        else:
            raise ValueError("Invalid type. Valid types are 'bbox' and 'seg'")

        for i, shape in enumerate(res):
            shape["id"] = i + id_start
    
    return res



def json_inference_to_labelme(anns, type="bbox", id_start=0):
    res = []
    for i, ann in enumerate(anns):
        if type == "bbox":
            box = ann["box"]

            points = [
                [box["x1"], box["y1"]],
                [box["x2"], box["y1"]],
                [box["x2"], box["y2"]],
                [box["x1"], box["y2"]],
            ]

            if Polygon(points).is_valid == False:
                continue

            res.append(
                {
                    "label": ann["name"],
                    "points": points,
                }
            )

        elif type == "seg":
            x_points = ann["segments"]["x"]
            y_points = ann["segments"]["y"]
            points = np.array([x_points, y_points]).T

            if Polygon(points).is_valid == False:
                continue

            res.append(
                {
                    "label": ann["name"],
                    "points": points.tolist(),
                }
            )
        else:
            raise ValueError("Invalid type. Valid types are 'bbox' and 'seg'")
        
        for i, shape in enumerate(res):
            shape["id"] = i + id_start

    return res