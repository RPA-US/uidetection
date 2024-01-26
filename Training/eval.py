from pyexpat import model
from tkinter import Y
from ultralytics import YOLO, SAM
import torch

ELEMENTS_MODEL = "../Models/trained/Yolov8n-seg - Elements/best.pt"
CONTAINER_MODEL = "../Models/trained/CustomSAM - Container/best.pt"
TEXT_MODEL = "../Models/trained/Yolov8s - Text/best.pt"
APPLEVEL_MODEL = "../Models/trained/Yolov8s-seg - AppLevel/best.pt"
TOP_MODEL = "../Models/trained/Yolov8s-seg - Top/best.pt"

CLUSTER_ELEMENTS_MODEL = "../Models/cluster/elements-cluster.pt"
CLUSTER_CONTAINER_MODEL = "../Models/cluster/container-cluster.pt"
CLUSTER_TEXT_MODEL = "../Models/cluster/text-cluster.pt"
CLUSTER_APPLEVEL_MODEL = "../Models/cluster/applevel-cluster.pt"
CLUSTER_TOP_MODEL = "../Models/cluster/toplevel-cluster.pt"

ELEMENTS_DATASET = "../YOLO_Datasets/ElementLevel_train/data.yaml"
CONTAINER_DATASET = "../YOLO_Datasets/ContainerLevel_train/data.yaml"
TEXT_DATASET = "../YOLO_Datasets/TextLevel_train/data.yaml"
APPLEVEL_DATASET = "../YOLO_Datasets/ApplicationLevel_train/data.yaml"
TOP_DATASET = "../YOLO_Datasets/TopLevel_train/data.yaml"

datasets = {
    ELEMENTS_MODEL: ELEMENTS_DATASET,
    CONTAINER_MODEL: CONTAINER_DATASET,
    TEXT_MODEL: TEXT_DATASET,
    APPLEVEL_MODEL: APPLEVEL_DATASET,
    TOP_MODEL: TOP_DATASET,
    CLUSTER_ELEMENTS_MODEL: ELEMENTS_DATASET,
    CLUSTER_CONTAINER_MODEL: CONTAINER_DATASET,
    CLUSTER_TEXT_MODEL: TEXT_DATASET,
    CLUSTER_APPLEVEL_MODEL: APPLEVEL_DATASET,
    CLUSTER_TOP_MODEL: TOP_DATASET
}

if __name__ == "__main__":
    for path in [CONTAINER_MODEL, CLUSTER_CONTAINER_MODEL]:
        model = YOLO(path)
        model.val(data=datasets[path], workers=1)

        # Clean cache
        del model
        torch.cuda.empty_cache()