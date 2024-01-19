from ultralytics import YOLO, SAM
import torch

root_path = "/home/amrojas/uidetection/YOLO_Datasets/"

YOLO_PATH= root_path + "ElementLevel/"
YOLO_PATH_TRAIN_EX= root_path + "ElementLevel_train/" # Data exclusively for training, not validating

train_epochs=1
container_level_iterations = 1

# check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using {} device'.format(device))


uidetection_abstraction_levels = {
    "ApplicationLevel_train/": {
        "args": {
            "workers": 1,
            "epochs": train_epochs,
            "optimizer": 'AdamW',
            "plots": False,
            "save": True,
            "hsv_h": 0.0, 
            "hsv_s": 0.0, 
            "hsv_v": 0.0
        },
        "model": 'YOLO("yolov8m-seg")'
    },
    "ContainerLevel_train/": {
        "args": {
            "workers": 1,
            "epochs": train_epochs,
            "patience": 15,
            "optimizer": 'AdamW',
            "plots": False,
            "save": True,
            "hsv_h": 0.0, 
            "hsv_s": 0.0, 
            "hsv_v": 0.0, 
            "fliplr": 0.0
        },
        "model": 'SAM("mobile_sam.pt")'
    },
    "ElementLevel_train/": {
        "args": {
            "workers": 1,
            "epochs": train_epochs,
            "optimizer": 'AdamW',
            "plots": False,
            "save": True,
            "hsv_h": 0.0, 
            "hsv_s": 0.0, 
            "hsv_v": 0.0, 
            "translate": 0.0, 
            "scale": 0.0, 
            "fliplr": 0.0, 
            "mosaic": 0.0,
        },
        "model": 'YOLO("yolov8l-seg")'
    },
    "TextLevel_train/": {
        "args": {
            "workers": 1,
            "epochs": train_epochs,
            "optimizer": 'AdamW',
            "plots": False,
            "save": True,
            "hsv_h": 0.0, 
            "hsv_s": 0.0, 
            "hsv_v": 0.0, 
            "translate": 0.0, 
            "scale": 0.0, 
            "fliplr": 0.0, 
            "mosaic": 0.0,
        },
        "model": 'YOLO("yolov8m")'
    },
    "TopLevel_train/": {
        "args": {
            "workers": 1,
            "epochs": train_epochs,
            "optimizer": 'AdamW',
            "plots": False,
            "save": True,
            "hsv_h": 0.0, 
            "hsv_s": 0.0, 
            "hsv_v": 0.0
        },
        "model": 'YOLO("yolov8m-seg")'
    }
}

everything = {
    "Dataset_train/": {
        "args": {
            "workers": 1,
            "epochs": train_epochs,
            "optimizer": 'AdamW',
            "plots": False,
            "save": True,
            "hsv_h": 0.0, 
            "hsv_s": 0.0, 
            "hsv_v": 0.0, 
            "translate": 0.0, 
            "scale": 0.0, 
            "fliplr": 0.0, 
            "mosaic": 0.0,
        },
        "model": 'YOLO("yolov8n-seg")'
    }
}

for abstraction_level, value in uidetection_abstraction_levels.items():
    # Tune hyperparameters on dataset for 30 epochs
    model = eval(value["model"])
    args = value["args"]
    if abstraction_level == "ContainerLevel_train/":
        model.tune(data=root_path + abstraction_level + "data.yaml", **args, iterations=container_level_iterations)   
    else:
        model.train(data=root_path + abstraction_level + "data.yaml", **args)
    # model.tune(data=root_path + abstraction_level + "data.yaml", iterations=10, **args)