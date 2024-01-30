import json

import cv2
from tqdm import tqdm

from utils import *

ELEMENTS_MODEL = "Models/trained/Yolov8n-seg - Elements/best.pt"
TEXT_MODEL = "Models/trained/Yolov8s - Text/best.pt"
CONTAINER_MODEL = "Models/trained/CustomSAM - Container/best.pt"
APPLEVEL_MODEL = "Models/trained/Yolov8s-seg - AppLevel/best.pt"
TOP_MODEL = "Models/trained/Yolov8s-seg - Top/best.pt"

CLUSTER_ELEMENTS_MODEL = "Models/cluster/elements-cluster.pt"
CLUSTER_CONTAINER_MODEL = "Models/cluster/container-cluster.pt"
CLUSTER_TEXT_MODEL = "Models/cluster/text-cluster.pt"
CLUSTER_APPLEVEL_MODEL = "Models/cluster/applevel-cluster.pt"
CLUSTER_TOP_MODEL = "Models/cluster/toplevel-cluster.pt"


def predict(directory, output_dir, optimized=False):
    images = dict()
    for file in os.listdir(directory):
        if file.endswith(".png") or file.endswith(".jpg"):
            images[file] = f"{directory}/{file}"

    detections = dict()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + "/detections"):
        os.makedirs(output_dir + "/detections")

    for img_name, img_path in tqdm(images.items(), desc="Running Screen2SOM predictions"):
        detections[img_name] = dict()
        image_pil = cv2.imread(img_path)

        if optimized:
            # Elements level preditions
            elements_shapes = sahi_predictions(
                CLUSTER_ELEMENTS_MODEL, image_pil, 240, 240, 0.3, "bbox", 0, 0.4
            )
            detections[img_name]["shapes"] = elements_shapes
            detections[img_name]["imageWidth"] = image_pil.shape[1]
            detections[img_name]["imageHeight"] = image_pil.shape[0]

            # Text level predictions
            text_shapes = sahi_predictions(
                CLUSTER_TEXT_MODEL, image_pil, 240, 240, 0.3, "bbox", len(detections[img_name]["shapes"]), 0.2
            )
            detections[img_name]["shapes"].extend(text_shapes)
            
            # Application level predictions
            applevel_shapes = yolo_prediction(
                CLUSTER_APPLEVEL_MODEL, image_pil, "seg", len(detections[img_name]["shapes"]), 0.3
            )
        else:
            elements_shapes = sahi_predictions(
                ELEMENTS_MODEL, image_pil, 240, 240, 0.3, "bbox", 0, 0.4
            )
            detections[img_name]["shapes"] = elements_shapes
            detections[img_name]["imageWidth"] = image_pil.shape[1]
            detections[img_name]["imageHeight"] = image_pil.shape[0]

            text_shapes = sahi_predictions(
                TEXT_MODEL, image_pil, 240, 240, 0.3, "bbox", len(detections[img_name]["shapes"]), 0.4
            )
            detections[img_name]["shapes"].extend(text_shapes)

            # Application level predictions
            applevel_shapes = yolo_prediction(
                APPLEVEL_MODEL, image_pil, "seg", len(detections[img_name]["shapes"]), 0.4
            )

        detections[img_name]["shapes"].extend(applevel_shapes)

        # Container Level predictions
        container_shapes = yolo_prediction(
            CONTAINER_MODEL, image_pil, "bbox", len(detections[img_name]["shapes"]), 0.7
        )
        detections[img_name]["shapes"].extend(container_shapes)


        # Top level predictions
        toplevel_shapes = yolo_prediction(
            TOP_MODEL, image_pil, "seg", len(detections[img_name]["shapes"]), 0.4
        )
        detections[img_name]["shapes"].extend(toplevel_shapes)

        # Save detections
        json.dump(
            detections[img_name],
            open(
                output_dir + f"/detections/detections_{img_name}.json", "w"
            ),
        )

    return detections
