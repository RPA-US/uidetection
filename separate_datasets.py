import os
import json

from shutil import copyfile
from shapely.geometry import Polygon

prefix = "Datasets/Dataset"

if not os.path.exists(f"{prefix}_TopLevel"):
    os.mkdir(f"{prefix}_TopLevel")

if not os.path.exists(f"{prefix}_ApplicationLevel"):
    os.mkdir(f"{prefix}_ApplicationLevel")

if not os.path.exists(f"{prefix}_ContainerLevel"):
    os.mkdir(f"{prefix}_ContainerLevel")

if not os.path.exists(f"{prefix}_ElementLevel"):
    os.mkdir(f"{prefix}_ElementLevel")

if not os.path.exists(f"{prefix}_TextLevel"):
    os.mkdir(f"{prefix}_TextLevel")

for file in os.listdir(prefix):
    if file.endswith(".json"):
        labeled_json = json.load(open(f"{prefix}/{file}"))

        top_level_json = labeled_json.copy()
        top_level_json["shapes"] = list(
            filter(
                lambda x: x["label"] in ["Application", "Taskbar", "Dock"],
                top_level_json["shapes"],
            )
        )

        application_level_json = labeled_json.copy()
        application_level_json["shapes"] = list(
            filter(
                lambda x: x["label"]
                in ["Header", "BrowserToolbar", "Toolbar", "Scrollbar"],
                application_level_json["shapes"],
            )
        )

        container_level_json = labeled_json.copy()
        container_level_json["shapes"] = list(
            filter(
                lambda x: x["label"]
                in [
                    "TabActive",
                    "TabInactive",
                    "Sidebar",
                    "Navbar",
                    "Container",
                    "Image",
                    "BrowserURLInput",
                ],
                container_level_json["shapes"],
            )
        )

        element_level_json = labeled_json.copy()
        element_level_json["shapes"] = list(
            filter(
                lambda x: x["label"]
                in [
                    "WebIcon",
                    "Icon",
                    "Switch",
                    "BtnSq",
                    "BtnPill",
                    "BtnCirc",
                    "CheckboxChecked",
                    "CheckboxUnchecked",
                    "RadiobtnSelected",
                    "RadiobtnUnselected",
                    "TextInput",
                    "Dropdown",
                    "Link",
                ],
                element_level_json["shapes"],
            )
        )

        text_level_json = labeled_json.copy()
        text_level_json["shapes"] = list(filter(lambda x: x["label"] == "Text", text_level_json["shapes"]))

        json.dump(
            top_level_json,
            open(f"{prefix}_TopLevel/{file}", "w"),
            indent=2,
        )

        json.dump(
            application_level_json,
            open(f"{prefix}_ApplicationLevel/{file}", "w"),
            indent=2,
        )

        json.dump(
            container_level_json,
            open(f"{prefix}_ContainerLevel/{file}", "w"),
            indent=2,
        )

        json.dump(
            element_level_json,
            open(f"{prefix}_ElementLevel/{file}", "w"),
            indent=2,
        )

        json.dump(
            text_level_json,
            open(f"{prefix}_TextLevel/{file}", "w"),
            indent=2,
        )

    else:
        copyfile(f"{prefix}/{file}", f"{prefix}_TopLevel/{file}")
        copyfile(f"{prefix}/{file}", f"{prefix}_ApplicationLevel/{file}")
        copyfile(f"{prefix}/{file}", f"{prefix}_ContainerLevel/{file}")
        copyfile(f"{prefix}/{file}", f"{prefix}_ElementLevel/{file}")
        copyfile(f"{prefix}/{file}", f"{prefix}_TextLevel/{file}")
