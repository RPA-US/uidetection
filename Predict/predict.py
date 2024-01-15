from tkinter import TOP
from tkinter.tix import TEXT
import numpy as np
from pyparsing import C
import torch
import json
import cv2
from shapely.geometry import Polygon

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
import copy

from hierarchy_constructor import *
from utils import *

ELEMENTS_MODEL = "Models/trained/Yolov8n-seg - Elements/best.pt"
CONTAINER_MODEL = "Models/trained/CustomSAM - Container/best.pt"
TEXT_MODEL = "Models/trained/Yolov8s - Text/best.pt"
APPLEVEL_MODEL = "Models/trained/Yolov8s-seg - AppLevel/best.pt"
TOP_MODEL = "Models/trained/Yolov8s-seg - Top/best.pt"

def predict(image_path):
    image_pil = cv2.imread(image_path)

    detections = dict()

    # Elements level preditions
    elements_shapes = sahi_predictions(ELEMENTS_MODEL, image_pil, 240, 240, 0.3, "bbox", 0)
    detections["shapes"] = elements_shapes
    detections["imageWidth"] = image_pil.shape[1]
    detections["imageHeight"] = image_pil.shape[0]

    # Text level predictions
    text_shapes = sahi_predictions(TEXT_MODEL, image_pil, 240, 240, 0.3, "bbox", len(detections["shapes"]))
    detections["shapes"].extend(text_shapes)

    # Container Level predictions
    container_shapes = yolo_prediction(CONTAINER_MODEL, image_pil, "bbox", len(detections["shapes"]))
    detections["shapes"].extend(container_shapes)

    # Application level predictions
    applevel_shapes = yolo_prediction(APPLEVEL_MODEL, image_pil, "seg", len(detections["shapes"]))
    detections["shapes"].extend(applevel_shapes)

    # Top level predictions
    toplevel_shapes = yolo_prediction(TOP_MODEL, image_pil, "seg", len(detections["shapes"]))
    detections["shapes"].extend(toplevel_shapes)

    # Order shapes by area
    detections["shapes"].sort(key=lambda x: Polygon(x["points"]).area, reverse=True)

    # Image crops from shapes
    recortes = []

    for i, shape in enumerate(detections["shapes"]):
        x1, y1 = np.min(shape["points"], axis=0)
        x2, y2 = np.max(shape["points"], axis=0)
        recortes.append(image_pil[int(y1):int(y2), int(x1):int(x2)])

    # SOM from shapes
    som = labels_to_soms(copy.deepcopy(detections))

    return recortes, detections["shapes"], som

   
def yolo_prediction(model_path, image_pil, type, id_start):
    model = YOLO(model_path)

    result = json.loads(model(image_pil, conf=0.4)[0].tojson())
    shapes = json_inference_to_labelme(
        result, type=type, id_start=id_start
    )

    # Unload model from memory
    del model
    torch.cuda.empty_cache()

    return shapes

def sahi_predictions(model_path, image_pil, slice_width, slice_height, overlap, type, id_start):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=0.4,
    )

    result = get_sliced_prediction(
        image_pil,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        perform_standard_pred=True,
    )
    anns = result.to_coco_annotations()
    shapes = coco_to_labelme(
        anns, type=type, id_start=id_start
    )

    # Unload model from memory
    del detection_model
    torch.cuda.empty_cache() 

    return shapes


if __name__ == '__main__':
    image_path = "Comparison Experiment/data/image.png"

    recortes, detections, som = predict(image_path)
    # save json
    with open("Comparison Experiment/data/detections.json", "w") as f:
        json.dump(detections, f, indent=4)
    # save som
    with open("Comparison Experiment/data/som.json", "w") as f:
        json.dump(som, f, indent=4)
    # save recortes
    for i, recorte in enumerate(recortes):
        cv2.imwrite(f"Comparison Experiment/data/recorte{i}.jpg", recorte)