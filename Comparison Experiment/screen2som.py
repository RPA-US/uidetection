import json

import cv2
from tqdm import tqdm

from utils import *

ELEMENTS_MODEL = "Models/elements.pt"
TEXT_MODEL = "Models/text.pt"
CONTAINER_MODEL = "Models/container.pt"
APPLEVEL_MODEL = "Models/applevel.pt"
TOP_MODEL = "Models/toplevel.pt"

CLUSTER_ELEMENTS_MODEL = "Models/elements-cluster.pt"
CLUSTER_CONTAINER_MODEL = "Models/container-cluster.pt"
CLUSTER_TEXT_MODEL = "Models/text-cluster.pt"
CLUSTER_APPLEVEL_MODEL = "Models/applevel-cluster.pt"
CLUSTER_TOP_MODEL = "Models/toplevel-cluster.pt"


def predict(directory, output_dir, optimized=False, sahi=True):
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
        image_pil_og = cv2.imread(img_path)
        image_pil = cv2.resize(image_pil, (640, 360))

        if optimized:
            # Elements level preditions
            elements_shapes = sahi_predictions(
                CLUSTER_ELEMENTS_MODEL, image_pil, 240, 240, 0.3, "bbox", 0, 0.4
            ) if sahi else yolo_prediction(
                CLUSTER_ELEMENTS_MODEL, image_pil, "seg", 0, 0.4
            )
            detections[img_name]["shapes"] = elements_shapes
            detections[img_name]["imageWidth"] = image_pil.shape[1]
            detections[img_name]["imageHeight"] = image_pil.shape[0]

            # Text level predictions
            text_shapes = sahi_predictions(
                CLUSTER_TEXT_MODEL, image_pil, 240, 240, 0.3, "bbox", len(detections[img_name]["shapes"]), 0.2
            ) if sahi else yolo_prediction(
                CLUSTER_TEXT_MODEL, image_pil, "bbox", len(detections[img_name]["shapes"]), 0.2
            )
            detections[img_name]["shapes"].extend(text_shapes)
        else:
            elements_shapes = sahi_predictions(
                ELEMENTS_MODEL, image_pil, 240, 240, 0.3, "bbox", 0, 0.4
            ) if sahi else yolo_prediction(
                ELEMENTS_MODEL, image_pil, "seg", 0, 0.4
            )
            detections[img_name]["shapes"] = elements_shapes
            detections[img_name]["imageWidth"] = image_pil.shape[1]
            detections[img_name]["imageHeight"] = image_pil.shape[0]

            text_shapes = sahi_predictions(
                TEXT_MODEL, image_pil, 240, 240, 0.3, "bbox", len(detections[img_name]["shapes"]), 0.4
            ) if sahi else yolo_prediction(
                TEXT_MODEL, image_pil, "bbox", len(detections[img_name]["shapes"]), 0.4
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

        detections = resize_detections(detections, image_pil_og.shape[1], image_pil_og.shape[0])

        # Save detections
        json.dump(
            detections[img_name],
            open(
                output_dir + f"/detections/detections_{img_name}.json", "w"
            ),
        )

    return detections
