import json

import cv2
from ultralytics import SAM
from tqdm import tqdm

from utils import *


def predict(directory, output_dir):
    images = dict()
    for file in os.listdir(directory):
        if file.endswith(".png") or file.endswith(".jpg"):
            images[file] = f"{directory}/{file}"

    detections = dict()

    if not os.path.exists(output_dir + "/mobileSAM"):
        os.makedirs(output_dir + "/mobileSAM")
    if not os.path.exists(output_dir + "/mobileSAM/detections"):
        os.makedirs(output_dir + "/mobileSAM/detections")

    for img_name, img_path in tqdm(images.items(), desc="Running MobileSAM predictions"):
        detections[img_name] = dict()
        image_pil = cv2.imread(img_path)

        shapes = sam_prediction("Models/mobile_sam.pt", image_pil, "seg", 0)
        detections[img_name]["shapes"] = shapes
        detections[img_name]["imageWidth"] = image_pil.shape[1]
        detections[img_name]["imageHeight"] = image_pil.shape[0]

        # Save detections
        json.dump(
            detections[img_name],
            open(output_dir + f"/mobileSAM/detections/detections_{img_name}.json", "w"),
        )

    return detections


def sam_prediction(model_path, image_pil, type="bbox", id_start=0):
    model = SAM(model_path)

    result = json.loads(model(image_pil, conf=0.4, verbose=False)[0].tojson())
    shapes = json_inference_to_labelme(result, type=type, id_start=id_start, remove_holes=True)

    # Unload model from memory
    del model
    torch.cuda.empty_cache()

    return shapes
