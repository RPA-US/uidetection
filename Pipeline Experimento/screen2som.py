import json
import cv2

import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics import YOLO
from utils import *

ELEMENTS_MODEL = "../Models/trained/Yolov8n-seg - Elements/best.pt"
TEXT_MODEL = "../Models/trained/Yolov8s - Text/best.pt"
CONTAINER_MODEL = "../Models/trained/CustomSAM - Container/best.pt"
APPLEVEL_MODEL = "../Models/trained/Yolov8s-seg - AppLevel/best.pt"
TOP_MODEL = "../Models/trained/Yolov8s-seg - Top/best.pt"

def predict(directory, output_dir):

    images = dict()
    for file in os.listdir(directory):
        if file.endswith(".png") or file.endswith(".jpg"):
            images[file] = f"{directory}/{file}"

    detections = dict()

    if not os.path.exists(output_dir + "/screen2som"):
        os.makedirs(output_dir + "/screen2som")
    if not os.path.exists(output_dir + "/screen2som/detections"):
        os.makedirs(output_dir + "/screen2som/detections")

    for img_name, img_path in images.items():
        detections[img_name] = dict()
        image_pil = cv2.imread(img_path)

        # Elements level preditions
        elements_shapes = sahi_predictions(ELEMENTS_MODEL, image_pil, 240, 240, 0.3, "bbox", 0)
        detections[img_name]["shapes"] = elements_shapes
        detections[img_name]["imageWidth"] = image_pil.shape[1]
        detections[img_name]["imageHeight"] = image_pil.shape[0]

        # Text level predictions
        text_shapes = sahi_predictions(TEXT_MODEL, image_pil, 240, 240, 0.3, "bbox", len(detections[img_name]["shapes"]))
        detections[img_name]["shapes"].extend(text_shapes)

        # Container Level predictions
        container_shapes = yolo_prediction(CONTAINER_MODEL, image_pil, "bbox", len(detections[img_name]["shapes"]))
        detections[img_name]["shapes"].extend(container_shapes)

        # Application level predictions
        applevel_shapes = yolo_prediction(APPLEVEL_MODEL, image_pil, "seg", len(detections[img_name]["shapes"]))
        detections[img_name]["shapes"].extend(applevel_shapes)

        # Top level predictions
        toplevel_shapes = yolo_prediction(TOP_MODEL, image_pil, "seg", len(detections[img_name]["shapes"]))
        detections[img_name]["shapes"].extend(toplevel_shapes)

        # Save detections
        json.dump(detections[img_name], open(output_dir + f"/screen2som/detections/detections_{img_name}.json", "w"))
    
    return detections

   
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