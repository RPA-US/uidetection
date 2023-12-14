from os.path import join as pjoin
import cv2
import os
import numpy as np
import uied_utils.lib_ip.ip_region_proposal as ip
from tqdm import tqdm
from utils import *


def resize_height_by_longest_edge(img_path, resize_length=800):
    org = cv2.imread(img_path)
    height, width = org.shape[:2]
    if height > width:
        return resize_length
    else:
        return int(resize_length * (height / width))


def color_tips():
    color_map = {
        "Text": (0, 0, 255),
        "Compo": (0, 255, 0),
        "Block": (0, 255, 255),
        "Text Content": (255, 0, 255),
    }
    board = np.zeros((200, 200, 3), dtype=np.uint8)

    board[:50, :, :] = (0, 0, 255)
    board[50:100, :, :] = (0, 255, 0)
    board[100:150, :, :] = (255, 0, 255)
    board[150:200, :, :] = (0, 255, 255)
    cv2.putText(board, "Text", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(
        board, "Non-text Compo", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
    )
    cv2.putText(
        board,
        "Compo's Text Content",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        2,
    )
    cv2.putText(board, "Block", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.imshow("colors", board)


def batch_component_detection(
    directory,
    output_dir,
    key_params={'min-grad': 3, 'ffl-block': 5, 'min-ele-area': 25, 'merge-contained-ele': False, 'max-word-inline-gap': 4, 'max-line-gap': 4}):    
    """
    ele:min-grad: gradient threshold to produce binary map
    ele:ffl-block: fill-flood threshold
    ele:min-ele-area: minimum area for selected elements
    ele:merge-contained-ele: if True, merge elements contained in others
    text:max-word-inline-gap: words with smaller distance than the gap are counted as a line
    text:max-line-gap: lines with smaller distance than the gap are counted as a paragraph

    Tips:
    1. Larger *min-grad* produces fine-grained binary-map while prone to over-segment element to small pieces
    2. Smaller *min-ele-area* leaves tiny elements while prone to produce noises
    3. If not *merge-contained-ele*, the elements inside others will be recognized, while prone to produce noises
    4. The *max-word-inline-gap* and *max-line-gap* should be dependent on the input image size and resolution

    mobile: {'min-grad':4, 'ffl-block':5, 'min-ele-area':50, 'max-word-inline-gap':6, 'max-line-gap':1}
    web   : {'min-grad':3, 'ffl-block':5, 'min-ele-area':25, 'max-word-inline-gap':4, 'max-line-gap':4}
    """

    if directory[-1] != "/":
        directory = directory + "/"

    images = dict()
    for file in os.listdir(directory):
        if file.endswith(".png") or file.endswith(".jpg"):
            images[file] = f"{directory}/{file}"

    detections = dict()

    if not os.path.exists(output_dir + "/UIED"):
        os.makedirs(output_dir + "/UIED")
    if not os.path.exists(output_dir + "/UIED/detections"):
        os.makedirs(output_dir + "/UIED/detections")

    for img_name, img_path in tqdm(images.items(), desc="Running UIED predictions"):
        classifier = None
        compos = ip.compo_detection(
            img_path,
            key_params,
            classifier=classifier,
            show=False,
        )

        image_pil = cv2.imread(img_path)
        detections[img_name] = dict()
        detections[img_name]["imageWidth"] = image_pil.shape[1]
        detections[img_name]["imageHeight"] = image_pil.shape[0]
        detections[img_name]["shapes"] = uied_to_labelme(compos)

        # Save detections
        json.dump(
            detections[img_name],
            open(output_dir + f"/UIED/detections/detections_{img_name}.json", "w"),
        )

    return detections
