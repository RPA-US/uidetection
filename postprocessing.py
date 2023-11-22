import os
import json

from shapely.geometry import Polygon

prefix = "Datasets\Dataset"

for file in os.listdir(prefix):
    if file.endswith(".json"):
        labeled_json = json.load(open(f"{prefix}/{file}"))

        # Transform 2 point rectangles into 4 point rectangles
        for i in range(len(labeled_json["shapes"])):
            if labeled_json["shapes"][i]["label"] == "Form":
                labeled_json["shapes"][i]["label"] = "Container"
            if labeled_json["shapes"][i]["label"] == "CheckBoxChecked":
                labeled_json["shapes"][i]["label"] = "CheckboxChecked"
            if labeled_json["shapes"][i]["label"] == "icon":
                labeled_json["shapes"][i]["label"] = "Icon"
            if labeled_json["shapes"][i]["label"] == "BroswerURLInput":
                labeled_json["shapes"][i]["label"] = "BrowserURLInput"
            if labeled_json["shapes"][i]["shape_type"] == "rectangle":
                labeled_json["shapes"][i]["shape_type"] = "polygon" 
                labeled_json["shapes"][i]["points"] = [
                    labeled_json["shapes"][i]["points"][0],
                    [
                        labeled_json["shapes"][i]["points"][1][0],
                        labeled_json["shapes"][i]["points"][0][1],
                    ],
                    labeled_json["shapes"][i]["points"][1],
                    [
                        labeled_json["shapes"][i]["points"][0][0],
                        labeled_json["shapes"][i]["points"][1][1],
                    ],
                ]
            if len(labeled_json["shapes"][i]["points"]) < 4:
                print(file)
                print(labeled_json["shapes"][i])

        # Order the shapes by area (from largest to smallest) for readability
        labeled_json["shapes"].sort(key=lambda x: Polygon(x["points"]).area, reverse=True)

        # Copy to another json all shapes that are nor "ItemList"
        unlabeled_json = {"shapes": []}
        for shape in labeled_json["shapes"]:
            if shape["label"] != "ItemList":
                unlabeled_json["shapes"].append(shape)

        labeled_json["shapes"] = unlabeled_json["shapes"]


        json.dump(
            labeled_json,
            open(f"{prefix}/{file}", "w"),
            indent=2,
        )

