import json
import os
from charset_normalizer import detect

import numpy as np
import torch
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from shapely.geometry import Polygon
from ultralytics import YOLO

def resize_detections(detections, new_width, new_height):
    # Loop over all the bounding boxes
    for shape in detections["shapes"]:
        for point in shape["points"]:
            # Update the point to match new dimensions
            point[0] = point[0] * new_width / detections["imageWidth"]
            point[1] = point[1] * new_height / detections["imageHeight"]

    # Update the size
    detections["imageWidth"] = new_width
    detections["imageHeight"] = new_height

    return detections

def load_dataset(dataset_path):
    dataset_labels = dict()
    for f in os.listdir(dataset_path):
        if f.endswith(".json"):
            labeled_json = json.load(open(f"{dataset_path}/{f}"))

            #Convert rectangles into 4 coordinates shapes and keep coordinates as positive numeric values
            for shape in labeled_json["shapes"]:
                if shape["shape_type"] == "rectangle" and len(shape["points"]) == 2:
                    x, y, x2, y2 = [coor for coords in shape["points"] for coor in coords]
                    shape["points"] = [[x, y], [x2, y], [x2, y2], [x, y2]]
                    shape["shape_type"] = "polygon"
                for point in shape["points"]:
                    point[0] = max(0, point[0])
                    point[1] = max(0, point[1])

            # Order the shapes by area (from largest to smallest) for readability
            labeled_json["shapes"].sort(
                key=lambda x: Polygon(x["points"]).area, reverse=True
            )

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

            try:
                if Polygon(points).is_valid == False:
                    continue
            except:
                continue

            res.append(
                {
                    "label": ann["category_name"],
                    "points": points,
                }
            )

        elif type == "seg":
            points = np.array(ann["segmentation"][0]).reshape(-1, 2)

            try:
                if Polygon(points).is_valid == False:
                    continue
            except:
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


def json_inference_to_labelme(anns, type="bbox", id_start=0, remove_holes=False):
    """
    Convert JSON inference annotations to LabelMe format.

    Args:
        anns (list): List of JSON inference annotations.
        type (str, optional): Type of annotation. Valid types are 'bbox' and 'seg'. Defaults to 'bbox'.
        id_start (int, optional): Starting ID for the annotations. Defaults to 0.
        remove_holes (bool, optional): Whether to remove holes in polygons. Defaults to False.

    Returns:
        list: List of annotations in LabelMe format.
    """
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

            try:
                if Polygon(points).is_valid == False:
                    if remove_holes:
                        points = Polygon(points).buffer(0).exterior.coords[:-1]
                    continue
            except:
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

            try:
                if Polygon(points).is_valid == False:
                    if remove_holes:
                        points = Polygon(points).buffer(0).exterior.coords[:-1]
                    continue
            except:
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


def uied_to_labelme(compos):
    res = []

    for compo in compos:
        (x1, y1, x2, y2) = compo.put_bbox()
        points = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
        ]

        try:
            if Polygon(points).is_valid == False:
                continue
        except:
            continue

        res.append(
            {
                "label": compo.category,
                "points": points,
            }
        )

    for i, shape in enumerate(res):
        shape["id"] = i

    return res


def yolo_prediction(model_path, image_pil, type, id_start, thress):
    model = YOLO(model_path)

    result = json.loads(model(image_pil, conf=thress, verbose=False)[0].tojson())
    shapes = json_inference_to_labelme(result, type=type, id_start=id_start)

    # Unload model from memory
    del model
    torch.cuda.empty_cache()

    return shapes


def sahi_predictions(
    model_path, image_pil, slice_width, slice_height, overlap, type, id_start, thress
):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=thress,
    )

    result = get_sliced_prediction(
        image_pil,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=True,
        verbose=0,
    )
    anns = result.to_coco_annotations()
    shapes = coco_to_labelme(anns, type=type, id_start=id_start)

    # Unload model from memory
    del detection_model
    torch.cuda.empty_cache()

    return shapes


def save_detections(detections_path, img_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for f in os.listdir(detections_path):
        if f.endswith(".json"):
            with open(f"{detections_path}/{f}") as json_file:
                data = json.load(json_file)
            
            detected_shapes = data["shapes"]

            img_name = "_".join(f.split("_")[1:])
            img_name = img_name.split(".")[0] + ".png"
            img_path = f"{img_dir}/{img_name}"

            tint_colors = {}
            for i in range(len(detected_shapes)):
                if detected_shapes[i]["label"] not in tint_colors:
                    tint_colors[detected_shapes[i]["label"]] = (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    )

            img = cv2.imread(img_path)
            for i in range(len(detected_shapes)):
                # Show both polygons (labeled and detected)
                cv2.polylines(
                    img,
                    np.int32([detected_shapes[i]["points"]]),
                    True,
                    tint_colors[detected_shapes[i]["label"]],
                    2,
                )

            cv2.imwrite(f"{output_dir}/{img_name}", img)