from tkinter import TOP
from tkinter.tix import TEXT
import numpy as np
from pyparsing import C
import torch
import json
import cv2
import os
from shapely.geometry import Polygon
import sys
# Sahi imports
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
import copy
# Custom imports
from hierarchy_constructor import *
from utils import *

ELEMENTS_MODEL = "Models/elements.pt"
TEXT_MODEL = "Models/text.pt"
CONTAINER_MODEL = "Models/container.pt"
APPLEVEL_MODEL = "Models/applevel.pt"
TOP_MODEL = "Models/toplevel.pt"

def predict(image_path):
    image_pil = cv2.imread(image_path)
    image_pil = cv2.resize(image_pil, (640, 360))
    cv2.imwrite(image_path, image_pil)

    detections = dict()

    # Elements level preditions
    elements_shapes = sahi_predictions(
                ELEMENTS_MODEL, image_pil, 240, 240, 0.3, "bbox", 0, 0.4
    )
    detections["shapes"] = elements_shapes
    detections["imageWidth"] = image_pil.shape[1]
    detections["imageHeight"] = image_pil.shape[0]

    # Text level predictions
    text_shapes = sahi_predictions(
        TEXT_MODEL, image_pil, 240, 240, 0.3, "bbox", len(detections["shapes"]), 0.2
    )
    detections["shapes"].extend(text_shapes)

    # Container Level predictions
    container_shapes = yolo_prediction(
        CONTAINER_MODEL, image_pil, "bbox", len(detections["shapes"]), 0.7
    )
    detections["shapes"].extend(container_shapes)

    # Application level predictions
    applevel_shapes = yolo_prediction(
        APPLEVEL_MODEL, image_pil, "seg", len(detections["shapes"]), 0.4
    )
    detections["shapes"].extend(applevel_shapes)

    # Top level predictions
    toplevel_shapes = yolo_prediction(
        TOP_MODEL, image_pil, "seg", len(detections["shapes"]), 0.6
    )
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

if __name__ == '__main__':
    
    root_path = "Comparison Experiment/data/"
    print("Comenzando prediccion...")
    
    # Verifica si se ha pasado un argumento para image_path
    if len(sys.argv) > 1:
        image_path = root_path + sys.argv[1]
    else:
        print("No se ha proporcionado un path para la imagen. Usando un valor por defecto.")
        image_path = root_path + "image.png"  # Valor por defecto

    recortes, detections, som = predict(image_path)
    # save json
    with open(root_path + "detections.json", "w") as f:
        json.dump(detections, f, indent=4)
    # save som
    with open(root_path + "som.json", "w") as f:
        json.dump(som, f, indent=4)
        

    if not os.path.exists(root_path + "recortes/"):
        os.makedirs(root_path + "recortes/")

    
    save_bordered_images(image_path, detections, root_path + "recortes/")